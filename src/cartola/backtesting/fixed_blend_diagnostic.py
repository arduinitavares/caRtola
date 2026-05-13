from __future__ import annotations

import html
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from cartola.backtesting.budgeting import advance_budget, initial_budget_state
from cartola.backtesting.config import BacktestConfig
from cartola.backtesting.experiment_metrics import calibration_slope_intercept, top_k_rows_by_round
from cartola.backtesting.optimizer import optimize_squad
from cartola.backtesting.policy_simulation import PolicySimulationError, reproduce_no_policy_round
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
_DISASTER_POINTS_THRESHOLD = 45.0
_TOP_K_CANDIDATES = 50
_INVALID_ROW_COLUMNS = (
    "season",
    "rodada",
    "blend_name",
    "source",
    "solver_status",
    "reason",
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


def decide_blend_candidate(
    *,
    blend_name: str,
    source_valid: bool,
    aggregate_delta: float,
    improved_seasons: int,
    worst_season_delta: float,
    season_2025_delta: float,
    final_budget_delta: float,
    min_budget_delta: float,
    max_drawdown_delta: float,
    selected_calibration_slope: float,
    top50_spearman_delta: float,
    disaster_rounds_under45_delta: int,
    worst_2_round_delta: float,
    top_two_concentration: float,
    non_optimal_delta: int,
) -> FixedBlendDecision:
    if not source_valid:
        return FixedBlendDecision(blend_name=blend_name, status="invalid", reason="source reproduction failed.")
    if non_optimal_delta > 0:
        return FixedBlendDecision(
            blend_name=blend_name,
            status="rejected",
            reason="blend introduced non-optimal solver rounds versus control.",
        )
    if disaster_rounds_under45_delta > 1:
        return FixedBlendDecision(
            blend_name=blend_name,
            status="rejected",
            reason="blend increased disaster rounds under 45 by more than one.",
        )

    if (
        aggregate_delta >= 85.0
        and improved_seasons >= 3
        and worst_season_delta >= -25.0
        and season_2025_delta >= -15.0
        and final_budget_delta >= -10.0
        and min_budget_delta >= -10.0
        and max_drawdown_delta <= 10.0
        and 0.75 <= selected_calibration_slope <= 1.25
        and top50_spearman_delta >= -0.03
        and disaster_rounds_under45_delta <= 0
        and worst_2_round_delta >= -10.0
        and top_two_concentration <= 0.50
    ):
        return FixedBlendDecision(
            blend_name=blend_name,
            status="candidate_blend",
            reason="passes fixed_blend_v1 candidate gates.",
        )

    if (
        aggregate_delta >= 40.0
        and improved_seasons >= 3
        and worst_season_delta >= -35.0
        and season_2025_delta >= -25.0
        and final_budget_delta >= -15.0
        and top_two_concentration <= 0.65
    ):
        return FixedBlendDecision(
            blend_name=blend_name,
            status="weak_positive_research_lead",
            reason="passes fixed_blend_v1 weak-positive research gates.",
        )

    if -20.0 <= aggregate_delta < 40.0 and final_budget_delta >= -20.0 and max_drawdown_delta <= 20.0:
        return FixedBlendDecision(
            blend_name=blend_name,
            status="inconclusive",
            reason="small aggregate movement within inconclusive band.",
        )

    return FixedBlendDecision(blend_name=blend_name, status="rejected", reason="failed fixed_blend_v1 gates.")


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


def child_dir_for(
    experiment_path: Path,
    *,
    season: int,
    model_id: str,
    feature_pack: str,
) -> Path:
    return experiment_path / "runs" / f"season={season}" / f"model={model_id}" / f"feature_pack={feature_pack}"


def child_path_for(
    experiment_path: Path,
    *,
    season: int,
    model_id: str,
    feature_pack: str,
) -> Path:
    return child_dir_for(
        experiment_path,
        season=season,
        model_id=model_id,
        feature_pack=feature_pack,
    ) / "player_predictions.csv"


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
        invalid_rows=pd.DataFrame(invalid_rows, columns=pd.Index(_INVALID_ROW_COLUMNS)),
    )


def build_blend_per_season_summary(
    *,
    experiment_path: Path,
    blend_round_results: pd.DataFrame,
    seasons: tuple[int, ...],
    control_model: str,
    feature_pack: str,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for season in seasons:
        control_rounds = _read_control_round_results(
            experiment_path=experiment_path,
            season=season,
            control_model=control_model,
            feature_pack=feature_pack,
        )
        control_metrics = _season_round_metrics(control_rounds)
        season_blends = blend_round_results.loc[blend_round_results["season"].astype(int).eq(int(season))]
        for blend_name, blend_rounds in season_blends.groupby("blend_name", sort=True):
            blend_metrics = _season_round_metrics(blend_rounds)
            rows.append(
                {
                    "blend_name": str(blend_name),
                    "season": int(season),
                    "control_actual_points": control_metrics["actual_points"],
                    "blend_actual_points": blend_metrics["actual_points"],
                    "actual_points_delta": blend_metrics["actual_points"] - control_metrics["actual_points"],
                    "control_final_budget": control_metrics["final_budget"],
                    "blend_final_budget": blend_metrics["final_budget"],
                    "final_budget_delta": blend_metrics["final_budget"] - control_metrics["final_budget"],
                    "control_min_budget": control_metrics["min_budget"],
                    "blend_min_budget": blend_metrics["min_budget"],
                    "min_budget_delta": blend_metrics["min_budget"] - control_metrics["min_budget"],
                    "control_max_drawdown": control_metrics["max_drawdown"],
                    "blend_max_drawdown": blend_metrics["max_drawdown"],
                    "max_drawdown_delta": blend_metrics["max_drawdown"] - control_metrics["max_drawdown"],
                    "control_non_optimal_rounds": control_metrics["non_optimal_rounds"],
                    "blend_non_optimal_rounds": blend_metrics["non_optimal_rounds"],
                    "non_optimal_delta": blend_metrics["non_optimal_rounds"] - control_metrics["non_optimal_rounds"],
                    "control_disaster_rounds_under45": control_metrics["disaster_rounds_under45"],
                    "blend_disaster_rounds_under45": blend_metrics["disaster_rounds_under45"],
                    "disaster_rounds_under45_delta": (
                        blend_metrics["disaster_rounds_under45"] - control_metrics["disaster_rounds_under45"]
                    ),
                    "control_worst_2_round_total": control_metrics["worst_2_round_total"],
                    "blend_worst_2_round_total": blend_metrics["worst_2_round_total"],
                    "worst_2_round_delta": (
                        blend_metrics["worst_2_round_total"] - control_metrics["worst_2_round_total"]
                    ),
                }
            )
    return pd.DataFrame(rows).sort_values(["blend_name", "season"], kind="mergesort").reset_index(drop=True)


def build_blend_ranked_summary(
    per_season_summary: pd.DataFrame,
    selected_players: pd.DataFrame,
    *,
    source_valid: bool,
    top50_spearman_delta: pd.DataFrame | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    top50_lookup = _top50_delta_lookup(top50_spearman_delta)
    for blend_name, summary in per_season_summary.groupby("blend_name", sort=True):
        selected = selected_players.loc[selected_players["blend_name"].astype(str).eq(str(blend_name))]
        calibration = calibration_slope_intercept(
            pd.to_numeric(selected.get("predicted_points", pd.Series(dtype=float)), errors="coerce"),
            pd.to_numeric(selected.get("pontuacao", pd.Series(dtype=float)), errors="coerce"),
        )
        selected_calibration_slope = _neutral_if_missing(calibration.get("calibration_slope"), neutral=1.0)
        aggregate_delta = float(summary["actual_points_delta"].sum())
        season_2025_rows = summary.loc[summary["season"].astype(int).eq(2025), "actual_points_delta"]
        season_2025_delta = float(season_2025_rows.iloc[0]) if not season_2025_rows.empty else 0.0
        top_two_concentration = _top_two_concentration(selected)
        row = {
            "blend_name": str(blend_name),
            "source_valid": bool(source_valid),
            "control_actual_points": float(summary["control_actual_points"].sum()),
            "blend_actual_points": float(summary["blend_actual_points"].sum()),
            "aggregate_delta": aggregate_delta,
            "improved_seasons": int((summary["actual_points_delta"] > 0).sum()),
            "worst_season_delta": float(summary["actual_points_delta"].min()),
            "season_2025_delta": season_2025_delta,
            "final_budget_delta": float(summary["final_budget_delta"].sum()),
            "min_budget_delta": float(summary["min_budget_delta"].sum()),
            "max_drawdown_delta": float(summary["max_drawdown_delta"].sum()),
            "non_optimal_delta": int(summary["non_optimal_delta"].sum()),
            "disaster_rounds_under45_delta": int(summary["disaster_rounds_under45_delta"].sum()),
            "worst_2_round_delta": float(summary["worst_2_round_delta"].sum()),
            "top_two_concentration": top_two_concentration,
            "selected_calibration_slope": selected_calibration_slope,
            "selected_calibration_intercept": _none_if_missing(calibration.get("calibration_intercept")),
            "selected_calibration_warning": calibration.get("warning"),
            "top50_spearman_delta": top50_lookup.get(str(blend_name), 0.0),
        }
        decision = decide_blend_candidate(
            blend_name=str(blend_name),
            source_valid=bool(row["source_valid"]),
            aggregate_delta=float(row["aggregate_delta"]),
            improved_seasons=int(row["improved_seasons"]),
            worst_season_delta=float(row["worst_season_delta"]),
            season_2025_delta=float(row["season_2025_delta"]),
            final_budget_delta=float(row["final_budget_delta"]),
            min_budget_delta=float(row["min_budget_delta"]),
            max_drawdown_delta=float(row["max_drawdown_delta"]),
            selected_calibration_slope=float(row["selected_calibration_slope"]),
            top50_spearman_delta=float(row["top50_spearman_delta"]),
            disaster_rounds_under45_delta=int(row["disaster_rounds_under45_delta"]),
            worst_2_round_delta=float(row["worst_2_round_delta"]),
            top_two_concentration=float(row["top_two_concentration"]),
            non_optimal_delta=int(row["non_optimal_delta"]),
        )
        row["decision_status"] = decision.status
        row["decision_reason"] = decision.reason
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["decision_status", "aggregate_delta"], ascending=[True, False]).reset_index(drop=True)


def build_blend_complementarity(
    *,
    experiment_path: Path,
    seasons: tuple[int, ...],
    feature_pack: str,
    model_a: str,
    model_b: str,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    joined_frames: list[pd.DataFrame] = []
    for season in seasons:
        joined = _joined_model_predictions(
            experiment_path=experiment_path,
            season=season,
            feature_pack=feature_pack,
            model_a=model_a,
            model_b=model_b,
        )
        joined_frames.append(joined.assign(season=int(season)))
        rows.append(_complementarity_row(joined, scope="season", season=str(season), model_a=model_a, model_b=model_b))
    if joined_frames:
        overall = pd.concat(joined_frames, ignore_index=True)
        rows.append(_complementarity_row(overall, scope="overall", season="all", model_a=model_a, model_b=model_b))
    return pd.DataFrame(rows)


def run_fixed_blend_diagnostic(
    *,
    experiment_path: Path,
    seasons: tuple[int, ...],
    feature_pack: str,
    control_model: str,
    blend_specs: tuple[BlendSpec, ...],
    initial_budget: float,
    current_year: int,
    output_root: Path,
) -> Path:
    output_path = output_root / f"fixed_blend_started_at={_timestamp_id()}"
    source_valid, reproduction_invalid_rows = _validate_control_reproduction(
        experiment_path=experiment_path,
        seasons=seasons,
        control_model=control_model,
        feature_pack=feature_pack,
    )
    replay_results = [
        run_blend_replay_for_season(
            experiment_path=experiment_path,
            season=season,
            feature_pack=feature_pack,
            blend_specs=blend_specs,
            config=BacktestConfig(season=season, start_round=_start_round_for_control(experiment_path, season, control_model, feature_pack), budget=initial_budget),
        )
        for season in seasons
    ]
    blend_round_results = _concat_or_empty([result.round_rows for result in replay_results])
    blend_selected_players = _concat_or_empty([result.selected_player_rows for result in replay_results])
    replay_invalid_rows = _concat_or_empty([result.invalid_rows for result in replay_results])
    invalid_rows = _concat_or_empty([reproduction_invalid_rows, replay_invalid_rows])

    per_season_summary = build_blend_per_season_summary(
        experiment_path=experiment_path,
        blend_round_results=blend_round_results,
        seasons=seasons,
        control_model=control_model,
        feature_pack=feature_pack,
    )
    top50_delta = _compute_top50_spearman_deltas(
        experiment_path=experiment_path,
        seasons=seasons,
        feature_pack=feature_pack,
        control_model=control_model,
        blend_specs=blend_specs,
    )
    ranked_summary = build_blend_ranked_summary(
        per_season_summary,
        blend_selected_players,
        source_valid=source_valid,
        top50_spearman_delta=top50_delta,
    )
    complementarity = _build_all_complementarity(
        experiment_path=experiment_path,
        seasons=seasons,
        feature_pack=feature_pack,
        blend_specs=blend_specs,
    )
    manifest = _fixed_blend_manifest(
        experiment_path=experiment_path,
        seasons=seasons,
        feature_pack=feature_pack,
        control_model=control_model,
        blend_specs=blend_specs,
        initial_budget=initial_budget,
        current_year=current_year,
        source_valid=source_valid,
    )
    decision_payload = _decision_payload(ranked_summary, source_valid=source_valid)
    _write_fixed_blend_artifacts(
        output_path=output_path,
        manifest=manifest,
        complementarity=complementarity,
        round_results=blend_round_results,
        selected_players=blend_selected_players,
        per_season_summary=per_season_summary,
        ranked_summary=ranked_summary,
        decision_payload=decision_payload,
        invalid_rows=invalid_rows,
    )
    return output_path


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


def _read_control_round_results(
    *,
    experiment_path: Path,
    season: int,
    control_model: str,
    feature_pack: str,
) -> pd.DataFrame:
    path = child_dir_for(
        experiment_path,
        season=season,
        model_id=control_model,
        feature_pack=feature_pack,
    ) / "round_results.csv"
    if not path.exists():
        raise FixedBlendDiagnosticError(f"Missing control round_results.csv: {path}")
    frame = pd.read_csv(path)
    required = {
        "rodada",
        "strategy",
        "solver_status",
        "budget_after_round",
        "budget_before_round",
        "actual_points_with_captain",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise FixedBlendDiagnosticError(f"Missing required columns in {path}: {', '.join(missing)}")
    control = frame.loc[frame["strategy"].astype(str).eq(control_model)].copy()
    if control.empty:
        raise FixedBlendDiagnosticError(f"Missing control strategy rows in {path}: {control_model}")
    control["season"] = int(season)
    return control.sort_values("rodada", kind="mergesort").reset_index(drop=True)


def _season_round_metrics(rounds: pd.DataFrame) -> dict[str, float | int]:
    sorted_rounds = rounds.sort_values("rodada", kind="mergesort").reset_index(drop=True)
    actual_points = pd.to_numeric(sorted_rounds["actual_points_with_captain"], errors="coerce").astype(float)
    budget_after = pd.to_numeric(sorted_rounds["budget_after_round"], errors="coerce").astype(float)
    budget_before = pd.to_numeric(sorted_rounds["budget_before_round"], errors="coerce").astype(float)
    drawdown = (
        pd.to_numeric(sorted_rounds["budget_drawdown"], errors="coerce").astype(float)
        if "budget_drawdown" in sorted_rounds.columns
        else (budget_after.cummax() - budget_after)
    )
    solver_status = sorted_rounds["solver_status"].astype(str)
    return {
        "actual_points": float(actual_points.sum()),
        "final_budget": float(budget_after.iloc[-1]),
        "min_budget": float(pd.concat([budget_before, budget_after], ignore_index=True).min()),
        "max_drawdown": float(drawdown.max()),
        "non_optimal_rounds": int((solver_status != "Optimal").sum()),
        "disaster_rounds_under45": int((actual_points < _DISASTER_POINTS_THRESHOLD).sum()),
        "worst_2_round_total": _worst_rolling_two_total(actual_points),
    }


def _worst_rolling_two_total(actual_points: pd.Series) -> float:
    if actual_points.empty:
        return 0.0
    if len(actual_points) == 1:
        return float(actual_points.iloc[0])
    return float(actual_points.rolling(window=2).sum().dropna().min())


def _top50_delta_lookup(top50_spearman_delta: pd.DataFrame | None) -> dict[str, float]:
    if top50_spearman_delta is None or top50_spearman_delta.empty:
        return {}
    required = {"blend_name", "top50_spearman_delta"}
    if not required.issubset(top50_spearman_delta.columns):
        return {}
    return {
        str(row["blend_name"]): float(row["top50_spearman_delta"])
        for row in top50_spearman_delta.to_dict(orient="records")
    }


def _neutral_if_missing(value: object, *, neutral: float) -> float:
    if value is None or pd.isna(value):
        return neutral
    numeric = float(value)
    if not np.isfinite(numeric):
        return neutral
    return numeric


def _none_if_missing(value: object) -> float | None:
    if value is None or pd.isna(value):
        return None
    numeric = float(value)
    if not np.isfinite(numeric):
        return None
    return numeric


def _top_two_concentration(selected: pd.DataFrame) -> float:
    if selected.empty or "id_atleta" not in selected.columns:
        return 0.0
    counts = selected["id_atleta"].astype(str).value_counts()
    if counts.empty:
        return 0.0
    return float(counts.head(2).sum() / counts.sum())


def _joined_model_predictions(
    *,
    experiment_path: Path,
    season: int,
    feature_pack: str,
    model_a: str,
    model_b: str,
) -> pd.DataFrame:
    score_a = score_column_for(model_a)
    score_b = score_column_for(model_b)
    frame_a = _read_component_frame(
        child_path_for(experiment_path, season=season, model_id=model_a, feature_pack=feature_pack),
        score_column=score_a,
    )
    frame_b = _read_component_frame(
        child_path_for(experiment_path, season=season, model_id=model_b, feature_pack=feature_pack),
        score_column=score_b,
    )
    _assert_candidate_identity_matches(frame_a.drop(columns=[score_a]), frame_b, model_id=model_b)
    return pd.DataFrame(
        {
            "rodada": frame_a["rodada"].astype(int),
            "actual": pd.to_numeric(frame_a["pontuacao"], errors="coerce").astype(float),
            "model_a_prediction": _finite_numeric_column(frame_a, score_a),
            "model_b_prediction": _finite_numeric_column(frame_b, score_b),
        }
    )


def _complementarity_row(
    frame: pd.DataFrame,
    *,
    scope: str,
    season: str,
    model_a: str,
    model_b: str,
) -> dict[str, object]:
    pred_a = frame["model_a_prediction"].astype(float)
    pred_b = frame["model_b_prediction"].astype(float)
    actual = frame["actual"].astype(float)
    residual_a = actual - pred_a
    residual_b = actual - pred_b
    return {
        "scope": scope,
        "season": season,
        "model_a": model_a,
        "model_b": model_b,
        "row_count": int(len(frame)),
        "prediction_correlation": _finite_corr(pred_a, pred_b),
        "residual_correlation": _finite_corr(residual_a, residual_b),
        "mean_abs_pred_diff": float((pred_a - pred_b).abs().mean()),
    }


def _finite_corr(left: pd.Series, right: pd.Series, *, method: str = "pearson") -> float:
    if left.nunique(dropna=True) < 2 or right.nunique(dropna=True) < 2:
        return 0.0
    value = left.corr(right, method=method)
    if value is None or pd.isna(value) or not np.isfinite(value):
        return 0.0
    return float(value)


def _validate_control_reproduction(
    *,
    experiment_path: Path,
    seasons: tuple[int, ...],
    control_model: str,
    feature_pack: str,
) -> tuple[bool, pd.DataFrame]:
    invalid_rows: list[dict[str, object]] = []
    for season in seasons:
        child_path = child_dir_for(
            experiment_path,
            season=season,
            model_id=control_model,
            feature_pack=feature_pack,
        )
        try:
            control_rounds = _read_control_round_results(
                experiment_path=experiment_path,
                season=season,
                control_model=control_model,
                feature_pack=feature_pack,
            )
            round_numbers = sorted(control_rounds["rodada"].astype(int).unique().tolist())
            for round_number in round_numbers:
                result = reproduce_no_policy_round(child_path, round_number=round_number)
                if result.status != "ok":
                    invalid_rows.append(
                        {
                            "season": int(season),
                            "rodada": int(round_number),
                            "blend_name": None,
                            "source": "control_reproduction",
                            "solver_status": result.status,
                            "reason": result.failure_reason,
                        }
                    )
        except (PolicySimulationError, FixedBlendDiagnosticError, ValueError) as exc:
            invalid_rows.append(
                {
                    "season": int(season),
                    "rodada": None,
                    "blend_name": None,
                    "source": "control_reproduction",
                    "solver_status": "error",
                    "reason": str(exc),
                }
            )
    frame = pd.DataFrame(invalid_rows, columns=pd.Index(_INVALID_ROW_COLUMNS))
    return frame.empty, frame


def _start_round_for_control(experiment_path: Path, season: int, control_model: str, feature_pack: str) -> int:
    control_rounds = _read_control_round_results(
        experiment_path=experiment_path,
        season=season,
        control_model=control_model,
        feature_pack=feature_pack,
    )
    return int(control_rounds["rodada"].astype(int).min())


def _compute_top50_spearman_deltas(
    *,
    experiment_path: Path,
    seasons: tuple[int, ...],
    feature_pack: str,
    control_model: str,
    blend_specs: tuple[BlendSpec, ...],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    control_score = score_column_for(control_model)
    for blend_spec in blend_specs:
        control_values: list[float] = []
        blend_values: list[float] = []
        for season in seasons:
            control = _read_component_frame(
                child_path_for(
                    experiment_path,
                    season=season,
                    model_id=control_model,
                    feature_pack=feature_pack,
                ),
                score_column=control_score,
            )
            blend = load_blend_candidate_frame(
                experiment_path=experiment_path,
                season=season,
                feature_pack=feature_pack,
                blend_spec=blend_spec,
            )
            blend_score = _blend_score_column(blend_spec.name)
            for round_number in sorted(control["rodada"].astype(int).unique().tolist()):
                control_round = control.loc[control["rodada"].astype(int).eq(round_number)]
                blend_round = blend.loc[blend["rodada"].astype(int).eq(round_number)]
                control_top = top_k_rows_by_round(control_round, score_column=control_score, k=_TOP_K_CANDIDATES)
                blend_top = top_k_rows_by_round(blend_round, score_column=blend_score, k=_TOP_K_CANDIDATES)
                control_values.append(_spearman_or_zero(control_top, score_column=control_score))
                blend_values.append(_spearman_or_zero(blend_top, score_column=blend_score))
        control_mean = float(np.mean(control_values)) if control_values else 0.0
        blend_mean = float(np.mean(blend_values)) if blend_values else 0.0
        rows.append(
            {
                "blend_name": blend_spec.name,
                "control_top50_spearman": control_mean,
                "blend_top50_spearman": blend_mean,
                "top50_spearman_delta": blend_mean - control_mean,
            }
        )
    return pd.DataFrame(rows)


def _spearman_or_zero(frame: pd.DataFrame, *, score_column: str) -> float:
    score = pd.to_numeric(frame[score_column], errors="coerce")
    actual = pd.to_numeric(frame["pontuacao"], errors="coerce")
    return _finite_corr(score, actual, method="spearman")


def _build_all_complementarity(
    *,
    experiment_path: Path,
    seasons: tuple[int, ...],
    feature_pack: str,
    blend_specs: tuple[BlendSpec, ...],
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    pairs: set[tuple[str, str]] = set()
    for blend_spec in blend_specs:
        for left, right in combinations([component.model_id for component in blend_spec.components], 2):
            pairs.add(tuple(sorted((left, right))))
    for model_a, model_b in sorted(pairs):
        frames.append(
            build_blend_complementarity(
                experiment_path=experiment_path,
                seasons=seasons,
                feature_pack=feature_pack,
                model_a=model_a,
                model_b=model_b,
            )
        )
    return _concat_or_empty(frames)


def _fixed_blend_manifest(
    *,
    experiment_path: Path,
    seasons: tuple[int, ...],
    feature_pack: str,
    control_model: str,
    blend_specs: tuple[BlendSpec, ...],
    initial_budget: float,
    current_year: int,
    source_valid: bool,
) -> dict[str, object]:
    return {
        "hypothesis_id": "M006",
        "design_revision": "fixed_blend_v1",
        "source_experiment_path": str(experiment_path),
        "seasons": [int(season) for season in seasons],
        "feature_pack": feature_pack,
        "control_model": control_model,
        "initial_budget": float(initial_budget),
        "budget_policy": "moving",
        "current_year": int(current_year),
        "source_valid": bool(source_valid),
        "blend_specs": [
            {
                "name": blend_spec.name,
                "components": [
                    {"model_id": component.model_id, "weight": component.weight}
                    for component in blend_spec.components
                ],
            }
            for blend_spec in blend_specs
        ],
    }


def _decision_payload(ranked_summary: pd.DataFrame, *, source_valid: bool) -> dict[str, object]:
    decisions = ranked_summary[
        ["blend_name", "decision_status", "decision_reason", "aggregate_delta"]
    ].to_dict(orient="records")
    candidate_count = int((ranked_summary["decision_status"].astype(str) == "candidate_blend").sum())
    return {
        "source_valid": bool(source_valid),
        "candidate_count": candidate_count,
        "decisions": decisions,
    }


def _write_fixed_blend_artifacts(
    *,
    output_path: Path,
    manifest: dict[str, object],
    complementarity: pd.DataFrame,
    round_results: pd.DataFrame,
    selected_players: pd.DataFrame,
    per_season_summary: pd.DataFrame,
    ranked_summary: pd.DataFrame,
    decision_payload: dict[str, object],
    invalid_rows: pd.DataFrame,
) -> None:
    output_path.mkdir(parents=True, exist_ok=False)
    _write_json(output_path / "fixed_blend_manifest.json", manifest)
    complementarity.to_csv(output_path / "blend_complementarity.csv", index=False)
    round_results.to_csv(output_path / "blend_round_results.csv", index=False)
    selected_players.to_csv(output_path / "blend_selected_players.csv", index=False)
    per_season_summary.to_csv(output_path / "blend_per_season_summary.csv", index=False)
    ranked_summary.to_csv(output_path / "blend_ranked_summary.csv", index=False)
    _write_json(output_path / "blend_decision.json", decision_payload)
    invalid_rows.to_csv(output_path / "invalid_rows.csv", index=False)
    (output_path / "fixed_blend_report.html").write_text(
        _fixed_blend_report_html(
            manifest=manifest,
            ranked_summary=ranked_summary,
            per_season_summary=per_season_summary,
            complementarity=complementarity,
            invalid_rows=invalid_rows,
            decision_payload=decision_payload,
        ),
        encoding="utf-8",
    )


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def _fixed_blend_report_html(
    *,
    manifest: dict[str, object],
    ranked_summary: pd.DataFrame,
    per_season_summary: pd.DataFrame,
    complementarity: pd.DataFrame,
    invalid_rows: pd.DataFrame,
    decision_payload: dict[str, object],
) -> str:
    sections = [
        "<h1>M006 Fixed Blend Diagnostic</h1>",
        _json_section("Decision", decision_payload),
        _table_section("Ranked Summary", ranked_summary),
        _table_section("Per-Season Summary", per_season_summary),
        _table_section("Complementarity", complementarity),
        _table_section("Invalid Rows", invalid_rows),
        _json_section("Manifest", manifest),
    ]
    return "<!doctype html><html><head><meta charset='utf-8'><title>M006 Fixed Blend Diagnostic</title></head><body>" + "\n".join(sections) + "</body></html>"


def _json_section(title: str, payload: dict[str, object]) -> str:
    return f"<h2>{html.escape(title)}</h2><pre>{html.escape(json.dumps(payload, indent=2, sort_keys=True, default=str))}</pre>"


def _table_section(title: str, frame: pd.DataFrame) -> str:
    return f"<h2>{html.escape(title)}</h2>{frame.to_html(index=False, escape=True)}"


def _timestamp_id() -> str:
    return datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%S%fZ")


def _component_score_column(model_id: str) -> str:
    return f"m006_component_{model_id}_score"


def _blend_score_column(blend_name: str) -> str:
    return f"m006_blend_{blend_name}_score"


def _concat_or_empty(frames: list[pd.DataFrame]) -> pd.DataFrame:
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)
