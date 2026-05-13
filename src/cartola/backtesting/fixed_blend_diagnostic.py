from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from cartola.backtesting.budgeting import advance_budget, initial_budget_state
from cartola.backtesting.config import BacktestConfig
from cartola.backtesting.optimizer import optimize_squad
from cartola.backtesting.scoring_contract import actual_scores_with_captain

_WEIGHT_SUM_TOLERANCE = 1e-9
_SORT_COLUMNS = ("rodada", "id_atleta", "id_clube", "posicao")
_BASE_COLUMNS = (
    "rodada",
    "id_atleta",
    "id_clube",
    "posicao",
    "apelido",
    "nome_clube",
    "preco_pre_rodada",
    "pontuacao",
    "entrou_em_campo",
    "variacao",
)


class FixedBlendDiagnosticError(ValueError):
    """Raised when fixed-blend diagnostic inputs or artifacts are invalid."""


@dataclass(frozen=True)
class BlendComponent:
    model_id: str
    weight: float


@dataclass(frozen=True)
class BlendSpec:
    name: str
    components: tuple[BlendComponent, ...]


@dataclass(frozen=True)
class FixedBlendDecision:
    blend_name: str
    status: str
    reason: str


@dataclass(frozen=True)
class BlendReplayResult:
    round_rows: pd.DataFrame
    selected_player_rows: pd.DataFrame
    invalid_rows: pd.DataFrame


def parse_blend_specs(raw_specs: tuple[str, ...]) -> tuple[BlendSpec, ...]:
    if not raw_specs:
        raise FixedBlendDiagnosticError("At least one blend spec is required")

    specs: list[BlendSpec] = []
    seen_names: set[str] = set()
    for raw_spec in raw_specs:
        name, separator, raw_components = raw_spec.partition("=")
        name = name.strip()
        if not separator or not name or not raw_components.strip():
            raise FixedBlendDiagnosticError(f"Invalid blend spec: {raw_spec!r}")
        if name in seen_names:
            raise FixedBlendDiagnosticError(f"Duplicate blend name: {name}")
        seen_names.add(name)

        components = _parse_components(raw_components, raw_spec=raw_spec)
        specs.append(BlendSpec(name=name, components=components))

    return tuple(specs)


def child_path_for(
    experiment_path: Path,
    *,
    season: int,
    model_id: str,
    feature_pack: str,
) -> Path:
    return (
        experiment_path
        / "runs"
        / f"season={season}"
        / f"model={model_id}"
        / f"feature_pack={feature_pack}"
        / "player_predictions.csv"
    )


def score_column_for(model_id: str) -> str:
    return f"{model_id}_score"


def load_blend_candidate_frame(
    *,
    experiment_path: Path,
    season: int,
    feature_pack: str,
    blend_spec: BlendSpec,
) -> pd.DataFrame:
    base_frame: pd.DataFrame | None = None
    blend_scores: pd.Series | None = None

    for component in blend_spec.components:
        model_score_column = score_column_for(component.model_id)
        path = child_path_for(
            experiment_path,
            season=season,
            model_id=component.model_id,
            feature_pack=feature_pack,
        )
        frame = _read_component_frame(path, score_column=model_score_column)
        component_column = _component_score_column(component.model_id)
        frame[component_column] = _finite_numeric_column(frame, model_score_column)

        if base_frame is None:
            base_frame = frame.drop(columns=[model_score_column]).copy()
            blend_scores = pd.Series(0.0, index=base_frame.index, dtype=float)
        else:
            _assert_candidate_identity_matches(base_frame, frame, model_id=component.model_id)
            base_frame[component_column] = frame[component_column].to_numpy()

        if blend_scores is None:
            raise FixedBlendDiagnosticError("Internal blend score initialization failed")
        blend_scores = blend_scores + (frame[component_column] * component.weight)

    if base_frame is None or blend_scores is None:
        raise FixedBlendDiagnosticError(f"Blend {blend_spec.name!r} has no components")

    base_frame[_blend_score_column(blend_spec.name)] = blend_scores
    return base_frame


def run_blend_replay_for_season(
    *,
    experiment_path: Path,
    season: int,
    feature_pack: str,
    blend_specs: tuple[BlendSpec, ...],
    config: BacktestConfig,
) -> BlendReplayResult:
    round_rows: list[dict[str, object]] = []
    selected_frames: list[pd.DataFrame] = []
    invalid_rows: list[dict[str, object]] = []

    for blend_spec in blend_specs:
        candidates = load_blend_candidate_frame(
            experiment_path=experiment_path,
            season=season,
            feature_pack=feature_pack,
            blend_spec=blend_spec,
        )
        blend_score_column = _blend_score_column(blend_spec.name)
        budget_state = initial_budget_state(config.budget)

        rounds = sorted(int(round_number) for round_number in candidates["rodada"].unique())
        for round_number in rounds:
            if round_number < config.start_round:
                continue
            round_candidates = candidates.loc[candidates["rodada"].eq(round_number)].copy()
            round_candidates["predicted_points"] = round_candidates[blend_score_column]
            result = optimize_squad(
                round_candidates,
                score_column="predicted_points",
                config=config,
                budget=budget_state.current_budget,
            )
            actual_scores = _actual_scores_for_selected(
                result.selected,
                blend_name=blend_spec.name,
                round_number=round_number,
                solver_status=result.status,
            )
            budget_update = advance_budget(budget_state, result.selected, budget_used=result.budget_used)
            budget_state = budget_update.next_state

            if result.status != "Optimal":
                invalid_rows.append(
                    {
                        "season": season,
                        "rodada": round_number,
                        "blend_name": blend_spec.name,
                        "solver_status": result.status,
                        "reason": result.infeasibility_reason,
                    }
                )

            round_rows.append(
                {
                    "season": season,
                    "rodada": round_number,
                    "blend_name": blend_spec.name,
                    "solver_status": result.status,
                    "formation": result.formation_name,
                    "selected_count": result.selected_count,
                    "budget_used": result.budget_used,
                    "budget_before_round": budget_update.budget_before_round,
                    "budget_after_round": budget_update.budget_after_round,
                    "budget_delta": budget_update.budget_delta,
                    "budget_remaining": budget_update.budget_remaining,
                    "budget_peak": budget_update.budget_peak,
                    "budget_drawdown": budget_update.budget_drawdown,
                    "predicted_points": result.predicted_points_with_captain,
                    "predicted_points_base": result.predicted_points_base,
                    "captain_bonus_predicted": result.captain_bonus_predicted,
                    "predicted_points_with_captain": result.predicted_points_with_captain,
                    "actual_points": actual_scores["actual_points_with_captain"],
                    "actual_points_base": actual_scores["actual_points_base"],
                    "captain_bonus_actual": actual_scores["captain_bonus_actual"],
                    "actual_points_with_captain": actual_scores["actual_points_with_captain"],
                    "captain_id": result.captain_id,
                    "captain_name": result.captain_name,
                }
            )

            if not result.selected.empty:
                selected = result.selected.copy()
                selected["season"] = season
                selected["rodada"] = round_number
                selected["blend_name"] = blend_spec.name
                selected_frames.append(selected)

    return BlendReplayResult(
        round_rows=pd.DataFrame(round_rows),
        selected_player_rows=_concat_or_empty(selected_frames),
        invalid_rows=pd.DataFrame(invalid_rows),
    )


def _parse_components(raw_components: str, *, raw_spec: str) -> tuple[BlendComponent, ...]:
    components: list[BlendComponent] = []
    seen_model_ids: set[str] = set()
    for raw_component in raw_components.split(","):
        model_id, separator, raw_weight = raw_component.partition(":")
        model_id = model_id.strip()
        raw_weight = raw_weight.strip()
        if not separator or not model_id or not raw_weight:
            raise FixedBlendDiagnosticError(f"Invalid blend component in spec: {raw_spec!r}")
        if model_id in seen_model_ids:
            raise FixedBlendDiagnosticError(f"Duplicate component model ID: {model_id}")
        seen_model_ids.add(model_id)

        try:
            weight = float(raw_weight)
        except ValueError as exc:
            raise FixedBlendDiagnosticError(f"Invalid component weight for model {model_id}: {raw_weight!r}") from exc
        if not np.isfinite(weight) or weight < 0.0:
            raise FixedBlendDiagnosticError(f"Component weight must be finite and non-negative: {model_id}")
        components.append(BlendComponent(model_id=model_id, weight=weight))

    if len(components) < 2:
        raise FixedBlendDiagnosticError("A blend must contain at least two components")
    weight_sum = sum(component.weight for component in components)
    if abs(weight_sum - 1.0) > _WEIGHT_SUM_TOLERANCE:
        raise FixedBlendDiagnosticError(f"Blend weights must sum to 1.0, got {weight_sum}")
    return tuple(components)


def _read_component_frame(path: Path, *, score_column: str) -> pd.DataFrame:
    if not path.exists():
        raise FixedBlendDiagnosticError(f"Missing component artifact: {path}")
    frame = pd.read_csv(path)
    required_columns = {*_BASE_COLUMNS, score_column}
    missing_columns = sorted(required_columns - set(frame.columns))
    if missing_columns:
        raise FixedBlendDiagnosticError(f"Missing required columns in {path}: {', '.join(missing_columns)}")
    sorted_frame = frame.sort_values(list(_SORT_COLUMNS), kind="mergesort").reset_index(drop=True)
    return sorted_frame


def _assert_candidate_identity_matches(base_frame: pd.DataFrame, frame: pd.DataFrame, *, model_id: str) -> None:
    try:
        assert_frame_equal(
            base_frame.loc[:, list(_BASE_COLUMNS)],
            frame.loc[:, list(_BASE_COLUMNS)],
            check_dtype=False,
            check_like=False,
        )
    except AssertionError as exc:
        raise FixedBlendDiagnosticError(f"Component candidate identity mismatch for model {model_id}") from exc


def _finite_numeric_column(frame: pd.DataFrame, column: str) -> pd.Series:
    values = pd.to_numeric(frame[column], errors="coerce")
    if values.isna().any() or not np.isfinite(values).all():
        raise FixedBlendDiagnosticError(f"Column must contain finite numeric values: {column}")
    return values.astype(float)


def _actual_scores_for_selected(
    selected: pd.DataFrame,
    *,
    blend_name: str,
    round_number: int,
    solver_status: str,
) -> dict[str, float]:
    if solver_status != "Optimal" or selected.empty:
        return {
            "actual_points_base": 0.0,
            "captain_bonus_actual": 0.0,
            "actual_points_with_captain": 0.0,
        }
    try:
        scores = actual_scores_with_captain(selected, actual_column="pontuacao")
    except ValueError as exc:
        raise FixedBlendDiagnosticError(
            f"Failed to score selected actuals for blend={blend_name!r} round={round_number}"
        ) from exc
    return scores


def _component_score_column(model_id: str) -> str:
    return f"m006_component_{model_id}_score"


def _blend_score_column(blend_name: str) -> str:
    return f"m006_blend_{blend_name}_score"


def _concat_or_empty(frames: list[pd.DataFrame]) -> pd.DataFrame:
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)
