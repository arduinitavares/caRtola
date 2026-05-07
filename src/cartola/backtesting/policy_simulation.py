from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

import pandas as pd

from cartola.backtesting.config import BacktestConfig
from cartola.backtesting.optimizer import optimize_squad
from cartola.backtesting.optimizer_policies import NO_POLICY, normalize_policy_candidates
from cartola.backtesting.scoring_contract import SCORING_CONTRACT_VERSION, actual_scores_with_captain

_SOURCE_ARTIFACTS: tuple[str, ...] = ("player_predictions.csv", "round_results.csv", "selected_players.csv")
_TOLERANCE = 1e-6

_PLAYER_PREDICTION_COLUMNS: tuple[str, ...] = (
    "rodada",
    "id_atleta",
    "apelido",
    "posicao",
    "id_clube",
    "nome_clube",
    "preco_pre_rodada",
    "pontuacao",
    "entrou_em_campo",
    "variacao",
)
_ROUND_RESULT_COLUMNS: tuple[str, ...] = (
    "rodada",
    "strategy",
    "solver_status",
    "formation",
    "budget_before_round",
    "budget_after_round",
    "budget_delta",
    "budget_used",
    "actual_points_with_captain",
    "predicted_points_with_captain",
    "captain_id",
)
_SELECTED_PLAYER_COLUMNS: tuple[str, ...] = (
    "rodada",
    "strategy",
    "id_atleta",
    "id_clube",
    "posicao",
    "preco_pre_rodada",
    "pontuacao",
    "entrou_em_campo",
    "variacao",
    "is_captain",
)


class PolicySimulationError(ValueError):
    pass


@dataclass(frozen=True)
class PolicySourceContext:
    child_path: Path
    season: int
    model_id: str
    feature_pack: str
    fixture_mode: str
    matchup_context_mode: str
    budget_policy: str
    scoring_contract_version: str
    score_column: str
    strategy: str


@dataclass(frozen=True)
class NoPolicyReproductionResult:
    status: Literal["ok", "mismatch"]
    selected_ids_match: bool
    captain_id_match: bool
    formation_match: bool
    budget_used_delta: float
    predicted_points_delta: float
    actual_points_delta: float
    failure_reason: str | None


def load_policy_source_context(child_path: Path) -> PolicySourceContext:
    resolved_child_path = Path(child_path)
    metadata = _read_metadata(resolved_child_path)
    _validate_metadata(metadata)
    _validate_artifact_files(resolved_child_path)

    model_id = _metadata_text(metadata, "primary_model_id") or _metadata_text(metadata, "model_id")
    if model_id is None:
        model_id = _path_value(resolved_child_path, "model")
    if model_id is None:
        raise PolicySimulationError("Cannot determine model_id from source metadata or child path.")

    feature_pack = _metadata_text(metadata, "feature_pack")
    if feature_pack is None:
        feature_pack = _path_value(resolved_child_path, "feature_pack")
    if feature_pack is None:
        raise PolicySimulationError("Cannot determine feature_pack from source metadata or child path.")

    strategy = _primary_strategy_from_metadata(metadata, model_id=model_id)
    score_column = _score_column_from_metadata(metadata, model_id=model_id, strategy=strategy)

    prediction_columns = _read_csv_columns(resolved_child_path / "player_predictions.csv")
    _validate_columns(
        resolved_child_path / "player_predictions.csv",
        prediction_columns,
        (*_PLAYER_PREDICTION_COLUMNS, score_column),
    )
    _validate_columns(
        resolved_child_path / "round_results.csv",
        _read_csv_columns(resolved_child_path / "round_results.csv"),
        _ROUND_RESULT_COLUMNS,
    )
    _validate_columns(
        resolved_child_path / "selected_players.csv",
        _read_csv_columns(resolved_child_path / "selected_players.csv"),
        _SELECTED_PLAYER_COLUMNS,
    )

    return PolicySourceContext(
        child_path=resolved_child_path,
        season=_metadata_int(metadata, "season"),
        model_id=model_id,
        feature_pack=feature_pack,
        fixture_mode=_required_metadata_text(metadata, "fixture_mode"),
        matchup_context_mode=str(metadata.get("matchup_context_mode", "none")),
        budget_policy="moving",
        scoring_contract_version=SCORING_CONTRACT_VERSION,
        score_column=score_column,
        strategy=strategy,
    )


def reproduce_no_policy_round(child_path: Path, round_number: int) -> NoPolicyReproductionResult:
    context = load_policy_source_context(child_path)
    player_predictions = _read_csv(context.child_path / "player_predictions.csv")
    round_results = _read_csv(context.child_path / "round_results.csv")
    selected_players = _read_csv(context.child_path / "selected_players.csv")

    source_round = _source_round_result(round_results, context=context, round_number=round_number)
    source_selected = _source_selected_players(selected_players, context=context, round_number=round_number)
    candidates = _round_candidates(player_predictions, context=context, round_number=round_number)
    normalized_candidates = normalize_policy_candidates(candidates, score_column=context.score_column)
    result = optimize_squad(
        normalized_candidates,
        score_column=context.score_column,
        config=BacktestConfig(
            season=context.season,
            start_round=round_number,
            budget=float(source_round["budget_before_round"]),
            fixture_mode="none",
            matchup_context_mode="none",
        ),
        budget=float(source_round["budget_before_round"]),
        policy=NO_POLICY,
    )
    actual_scores = _actual_scores_for_selected(result.selected, round_number=round_number, strategy=context.strategy)

    selected_ids_match = _selected_ids(result.selected) == _selected_ids(source_selected)
    captain_id_match = _optional_int(source_round["captain_id"]) == result.captain_id
    formation_match = str(source_round["formation"]) == result.formation_name
    budget_used_delta = result.budget_used - float(source_round["budget_used"])
    predicted_points_delta = result.predicted_points_with_captain - float(source_round["predicted_points_with_captain"])
    actual_points_delta = actual_scores["actual_points_with_captain"] - float(source_round["actual_points_with_captain"])

    mismatch_reasons = _mismatch_reasons(
        selected_ids_match=selected_ids_match,
        captain_id_match=captain_id_match,
        formation_match=formation_match,
        budget_used_delta=budget_used_delta,
        predicted_points_delta=predicted_points_delta,
        actual_points_delta=actual_points_delta,
    )
    status: Literal["ok", "mismatch"] = "ok" if not mismatch_reasons else "mismatch"
    return NoPolicyReproductionResult(
        status=status,
        selected_ids_match=selected_ids_match,
        captain_id_match=captain_id_match,
        formation_match=formation_match,
        budget_used_delta=budget_used_delta,
        predicted_points_delta=predicted_points_delta,
        actual_points_delta=actual_points_delta,
        failure_reason=None if not mismatch_reasons else ", ".join(mismatch_reasons),
    )


def _read_metadata(child_path: Path) -> dict[str, object]:
    metadata_path = child_path / "run_metadata.json"
    if not metadata_path.exists():
        raise PolicySimulationError(f"Missing run metadata: {metadata_path}")
    try:
        metadata: Any = json.loads(metadata_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise PolicySimulationError(f"Invalid run metadata JSON: {metadata_path}") from exc
    if not isinstance(metadata, dict):
        raise PolicySimulationError(f"run_metadata.json must contain an object: {metadata_path}")
    return metadata


def _validate_metadata(metadata: dict[str, object]) -> None:
    if metadata.get("budget_policy") != "moving":
        raise PolicySimulationError("Policy simulation requires budget_policy=moving source artifacts.")
    if metadata.get("scoring_contract_version") != SCORING_CONTRACT_VERSION:
        raise PolicySimulationError(
            f"Policy simulation requires scoring_contract_version={SCORING_CONTRACT_VERSION}."
        )


def _validate_artifact_files(child_path: Path) -> None:
    for artifact_name in _SOURCE_ARTIFACTS:
        artifact_path = child_path / artifact_name
        if not artifact_path.exists():
            raise PolicySimulationError(f"Missing source artifact: {artifact_path}")


def _read_csv_columns(csv_path: Path) -> set[str]:
    return set(pd.read_csv(csv_path, nrows=0).columns)


def _validate_columns(csv_path: Path, actual_columns: set[str], required_columns: tuple[str, ...]) -> None:
    missing = [column for column in required_columns if column not in actual_columns]
    if missing:
        raise PolicySimulationError(f"Missing required columns in {csv_path.name}: {', '.join(missing)}")


def _metadata_text(metadata: dict[str, object], key: str) -> str | None:
    value = metadata.get(key)
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return text


def _required_metadata_text(metadata: dict[str, object], key: str) -> str:
    value = _metadata_text(metadata, key)
    if value is None:
        raise PolicySimulationError(f"Missing required source metadata field: {key}")
    return value


def _metadata_int(metadata: dict[str, object], key: str) -> int:
    if key not in metadata:
        raise PolicySimulationError(f"Missing required source metadata field: {key}")
    try:
        return int(cast(Any, metadata[key]))
    except (TypeError, ValueError) as exc:
        raise PolicySimulationError(f"Source metadata field must be an integer: {key}") from exc


def _path_value(path: Path, key: str) -> str | None:
    prefix = f"{key}="
    for part in reversed(path.parts):
        if part.startswith(prefix):
            value = part[len(prefix) :].strip()
            return value or None
    return None


def _primary_strategy_from_metadata(metadata: dict[str, object], *, model_id: str) -> str:
    strategy_roles = metadata.get("strategy_roles")
    if isinstance(strategy_roles, dict):
        primary_strategies = [
            str(strategy)
            for strategy, role in strategy_roles.items()
            if str(role) == "primary_model"
        ]
        if len(primary_strategies) > 1:
            raise PolicySimulationError(
                "Cannot determine primary model strategy from source metadata: multiple primary_model roles."
            )
        if len(primary_strategies) == 1:
            return primary_strategies[0]

    strategy = _metadata_text(metadata, "strategy")
    return model_id if strategy is None else strategy


def _score_column_from_metadata(metadata: dict[str, object], *, model_id: str, strategy: str) -> str:
    score_column = _metadata_text(metadata, "score_column")
    if score_column is not None:
        return score_column
    if strategy == "baseline":
        return "baseline_score"
    if strategy == "price":
        return "price_score"
    if strategy:
        return f"{strategy}_score"
    if model_id:
        return f"{model_id}_score"
    raise PolicySimulationError("Cannot determine primary model score column from source metadata.")


def _read_csv(csv_path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(csv_path)
    except pd.errors.EmptyDataError as exc:
        raise PolicySimulationError(f"Source artifact is empty: {csv_path}") from exc


def _source_round_result(
    round_results: pd.DataFrame,
    *,
    context: PolicySourceContext,
    round_number: int,
) -> pd.Series:
    rows = round_results.loc[
        round_results["rodada"].astype(int).eq(int(round_number))
        & round_results["strategy"].astype(str).eq(context.strategy)
    ]
    if rows.empty:
        raise PolicySimulationError(
            f"Missing round_results row for round={round_number} strategy={context.strategy!r}."
        )
    if len(rows) > 1:
        raise PolicySimulationError(
            f"Multiple round_results rows for round={round_number} strategy={context.strategy!r}."
        )
    return rows.iloc[0]


def _source_selected_players(
    selected_players: pd.DataFrame,
    *,
    context: PolicySourceContext,
    round_number: int,
) -> pd.DataFrame:
    return selected_players.loc[
        selected_players["rodada"].astype(int).eq(int(round_number))
        & selected_players["strategy"].astype(str).eq(context.strategy)
    ].copy()


def _round_candidates(
    player_predictions: pd.DataFrame,
    *,
    context: PolicySourceContext,
    round_number: int,
) -> pd.DataFrame:
    candidates = player_predictions.loc[player_predictions["rodada"].astype(int).eq(int(round_number))].copy()
    if candidates.empty:
        raise PolicySimulationError(f"Missing player_predictions rows for round={round_number}.")
    if context.score_column not in candidates.columns:
        raise PolicySimulationError(f"Missing score column in player_predictions.csv: {context.score_column}")
    return candidates


def _actual_scores_for_selected(
    selected: pd.DataFrame,
    *,
    round_number: int,
    strategy: str,
) -> dict[str, float]:
    if selected.empty:
        return {
            "actual_points_base": 0.0,
            "captain_bonus_actual": 0.0,
            "actual_points_with_captain": 0.0,
        }
    try:
        return actual_scores_with_captain(selected, actual_column="pontuacao")
    except ValueError as exc:
        raise PolicySimulationError(
            f"Failed to score actual captain-aware points for round={round_number} strategy={strategy!r}."
        ) from exc


def _selected_ids(selected: pd.DataFrame) -> set[int]:
    if selected.empty:
        return set()
    return set(pd.to_numeric(selected["id_atleta"], errors="raise").astype(int).tolist())


def _optional_int(value: object) -> int | None:
    if pd.isna(value):
        return None
    return int(cast(Any, value))


def _mismatch_reasons(
    *,
    selected_ids_match: bool,
    captain_id_match: bool,
    formation_match: bool,
    budget_used_delta: float,
    predicted_points_delta: float,
    actual_points_delta: float,
) -> list[str]:
    reasons: list[str] = []
    if not selected_ids_match:
        reasons.append("selected_ids mismatch")
    if not captain_id_match:
        reasons.append("captain_id mismatch")
    if not formation_match:
        reasons.append("formation mismatch")
    if abs(budget_used_delta) > _TOLERANCE:
        reasons.append("budget_used mismatch")
    if abs(predicted_points_delta) > _TOLERANCE:
        reasons.append("predicted_points_with_captain mismatch")
    if abs(actual_points_delta) > _TOLERANCE:
        reasons.append("actual_points_with_captain mismatch")
    return reasons
