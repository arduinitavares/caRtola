from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, Mapping

import pandas as pd

from cartola.backtesting.config import (
    DEFAULT_FORMATIONS,
    DEFAULT_SCOUT_COLUMNS,
    MARKET_OPEN_PRICE_COLUMN,
    BacktestConfig,
    FootyStatsEvaluationScope,
    FootyStatsMode,
)
from cartola.backtesting.data import _entry_flag_mask, load_season_data
from cartola.backtesting.features import (
    FOOTYSTATS_PPG_FEATURE_COLUMNS,
    FOOTYSTATS_XG_FEATURE_COLUMNS,
    build_prediction_frame,
    build_training_frame,
    feature_columns_for_config,
)
from cartola.backtesting.footystats_features import (
    FootyStatsPPGLoadResult,
    build_footystats_join_diagnostics,
    load_footystats_feature_rows_for_recommendation,
)
from cartola.backtesting.models import BaselinePredictor, RandomForestPointPredictor
from cartola.backtesting.optimizer import optimize_squad

RecommendationMode = Literal["live", "replay"]


@dataclass(frozen=True)
class RecommendationConfig:
    season: int
    target_round: int
    mode: RecommendationMode
    budget: float = 100.0
    playable_statuses: tuple[str, ...] = ("Provavel",)
    formation_name: str = "4-3-3"
    random_seed: int = 123
    project_root: Path = Path(".")
    output_root: Path = Path("data/08_reporting/recommendations")
    footystats_mode: FootyStatsMode = "ppg"
    footystats_league_slug: str = "brazil-serie-a"
    footystats_dir: Path = Path("data/footystats")
    current_year: int | None = None
    allow_finalized_live_data: bool = False
    output_run_id: str | None = None
    live_workflow: Mapping[str, object] | None = None
    scout_columns: tuple[str, ...] = DEFAULT_SCOUT_COLUMNS
    formations: Mapping[str, Mapping[str, int]] = field(default_factory=lambda: DEFAULT_FORMATIONS)

    @property
    def output_path(self) -> Path:
        base = self.project_root / self.output_root / str(self.season) / f"round-{self.target_round}" / self.mode
        if self.output_run_id is None:
            return base
        return base / "runs" / self.output_run_id

    @property
    def selected_formation(self) -> Mapping[str, int]:
        if self.formation_name not in self.formations:
            raise ValueError(f"Unknown formation {self.formation_name!r}. Available: {sorted(self.formations)}")
        return self.formations[self.formation_name]


@dataclass(frozen=True)
class RecommendationResult:
    recommended_squad: pd.DataFrame
    candidate_predictions: pd.DataFrame
    summary: dict[str, object]
    metadata: dict[str, object]


BASE_OUTPUT_COLUMNS = [
    "rodada",
    "id_atleta",
    "apelido",
    "id_clube",
    "nome_clube",
    "posicao",
    "status",
    MARKET_OPEN_PRICE_COLUMN,
    "baseline_score",
    "random_forest_score",
    "price_score",
]


def _resolved_current_year(config: RecommendationConfig) -> int:
    return config.current_year if config.current_year is not None else datetime.now(UTC).year


def _footystats_scope(config: RecommendationConfig) -> FootyStatsEvaluationScope:
    if config.footystats_mode == "none":
        return "historical_candidate"
    if config.season == _resolved_current_year(config):
        return "live_current"
    return "historical_candidate"


def _backtest_config(config: RecommendationConfig) -> BacktestConfig:
    return BacktestConfig(
        season=config.season,
        start_round=config.target_round,
        budget=config.budget,
        playable_statuses=config.playable_statuses,
        formation_name=config.formation_name,
        random_seed=config.random_seed,
        project_root=config.project_root,
        output_root=Path("data/08_reporting/backtests"),
        fixture_mode="none",
        footystats_mode=config.footystats_mode,
        footystats_evaluation_scope=_footystats_scope(config),
        footystats_league_slug=config.footystats_league_slug,
        footystats_dir=config.footystats_dir,
        current_year=config.current_year,
        scout_columns=config.scout_columns,
        formations=config.formations,
    )


def _validate_mode_scope(config: RecommendationConfig) -> None:
    if config.mode not in {"live", "replay"}:
        raise ValueError(f"Unsupported recommendation mode: {config.mode!r}")
    if config.target_round <= 0:
        raise ValueError("target_round must be a positive integer")
    if config.output_run_id is not None:
        run_id_path = Path(config.output_run_id)
        if (
            config.output_run_id in {"", ".", ".."}
            or run_id_path.is_absolute()
            or "/" in config.output_run_id
            or "\\" in config.output_run_id
        ):
            raise ValueError(f"output_run_id must be a single path segment: {config.output_run_id!r}")
    if config.mode == "live":
        current_year = _resolved_current_year(config)
        if config.season != current_year:
            raise ValueError(f"live mode requires season {config.season} to equal current_year {current_year}")


def _validate_output_root(config: RecommendationConfig) -> None:
    project_root = config.project_root.resolve()
    protected_backtest_root = (project_root / "data" / "08_reporting" / "backtests").resolve()
    output_root = _resolve_output_root(config)
    if project_root != output_root and project_root not in output_root.parents:
        raise ValueError(f"Recommendation output_root must resolve inside project_root: output_root={config.output_root}")
    if output_root == protected_backtest_root or protected_backtest_root in output_root.parents:
        raise ValueError(
            "Recommendation output_root cannot be inside backtest reports: "
            f"output_root={config.output_root}"
        )


def _resolve_output_root(config: RecommendationConfig) -> Path:
    if config.output_root.is_absolute():
        return config.output_root.resolve()
    return (config.project_root / config.output_root).resolve()


def _visible_season_frame(season_df: pd.DataFrame, *, target_round: int) -> pd.DataFrame:
    rodada = pd.to_numeric(season_df["rodada"], errors="raise").astype(int)
    return season_df.loc[rodada.le(target_round)].copy()


def _real_club_keys(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=pd.Index(["rodada", "id_clube"]))
    source = frame.copy()
    if "nome_clube" in source.columns:
        has_name = source["nome_clube"].notna() & source["nome_clube"].map(lambda value: str(value).strip() != "")
        source = source.loc[has_name]
    keys = source[["rodada", "id_clube"]].dropna().drop_duplicates().copy()
    keys["rodada"] = pd.to_numeric(keys["rodada"], errors="raise").astype(int)
    keys["id_clube"] = pd.to_numeric(keys["id_clube"], errors="raise").astype(int)
    return keys.sort_values(["rodada", "id_clube"]).reset_index(drop=True)


def _load_recommendation_footystats(
    config: RecommendationConfig,
    visible_season_df: pd.DataFrame,
) -> FootyStatsPPGLoadResult | None:
    if config.footystats_mode == "none":
        return None
    return load_footystats_feature_rows_for_recommendation(
        season=config.season,
        project_root=config.project_root,
        footystats_dir=config.footystats_dir,
        league_slug=config.footystats_league_slug,
        current_year=config.current_year,
        target_round=config.target_round,
        footystats_mode=config.footystats_mode,
        require_complete_status=config.season != _resolved_current_year(config),
        required_keys=_real_club_keys(visible_season_df),
    )


def _finalized_live_data_evidence(
    target_frame: pd.DataFrame,
    *,
    scout_columns: tuple[str, ...] = DEFAULT_SCOUT_COLUMNS,
) -> dict[str, int]:
    pontuacao = pd.to_numeric(target_frame.get("pontuacao", pd.Series(dtype=float)), errors="coerce")
    pontuacao_non_zero_count = int(pontuacao.fillna(0.0).ne(0.0).sum())

    entrou = target_frame.get("entrou_em_campo", pd.Series(dtype=bool))
    entrou_true_count = int(_entry_flag_mask(entrou).sum())

    non_zero_scout_count = 0
    for scout in scout_columns:
        if scout in target_frame.columns:
            values = pd.to_numeric(target_frame[scout], errors="coerce").fillna(0.0)
            non_zero_scout_count += int(values.ne(0.0).sum())

    return {
        "pontuacao_non_zero_count": pontuacao_non_zero_count,
        "entrou_em_campo_true_count": entrou_true_count,
        "non_zero_scout_count": non_zero_scout_count,
    }


def _active_footystats_columns(config: RecommendationConfig) -> list[str]:
    if config.footystats_mode == "none":
        return []
    if config.footystats_mode == "ppg":
        return list(FOOTYSTATS_PPG_FEATURE_COLUMNS)
    if config.footystats_mode == "ppg_xg":
        return [*FOOTYSTATS_PPG_FEATURE_COLUMNS, *FOOTYSTATS_XG_FEATURE_COLUMNS]
    raise ValueError(f"Unsupported footystats_mode: {config.footystats_mode!r}")


def _select_columns(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    return frame[[column for column in columns if column in frame.columns]].copy()


def _replay_actual_points(selected: pd.DataFrame) -> tuple[float | None, list[str]]:
    if "pontuacao" not in selected.columns:
        return None, ["Replay actual_points is null because selected pontuacao is unavailable."]

    values = pd.to_numeric(selected["pontuacao"], errors="coerce")
    missing_count = int(values.isna().sum())
    if missing_count:
        return None, [
            "Replay actual_points is null because "
            f"{missing_count} selected players have missing pontuacao."
        ]
    return float(values.sum()), []


def _empty_oracle_metrics() -> dict[str, object]:
    return {
        "oracle_actual_points": None,
        "oracle_gap": None,
        "oracle_capture_rate": None,
        "oracle_optimizer_status": None,
    }


def _numeric_finite_series(frame: pd.DataFrame, column: str) -> pd.Series:
    values = pd.to_numeric(frame[column], errors="coerce")
    finite = values.map(lambda value: bool(pd.notna(value) and math.isfinite(float(value))))
    result = values.astype(float)
    return result.where(finite)


def _oracle_replay_metrics(
    candidates: pd.DataFrame,
    *,
    actual_points: float | None,
    config: BacktestConfig,
) -> tuple[dict[str, object], list[str]]:
    metrics = _empty_oracle_metrics()
    if "pontuacao" not in candidates.columns:
        return metrics, ["Oracle actual_points is null because candidate pontuacao is unavailable."]

    actual_values = _numeric_finite_series(candidates, "pontuacao")
    invalid_count = int(actual_values.isna().sum())
    if invalid_count:
        return metrics, [
            "Oracle actual_points is null because "
            f"{invalid_count} candidate rows have missing or non-finite pontuacao."
        ]

    oracle_candidates = candidates.copy()
    oracle_candidates["pontuacao"] = actual_values
    oracle_result = optimize_squad(oracle_candidates, score_column="pontuacao", config=config)
    metrics["oracle_optimizer_status"] = oracle_result.status
    if oracle_result.status != "Optimal":
        return metrics, [f"Oracle optimizer did not reach Optimal status: {oracle_result.status}."]

    oracle_points = float(pd.to_numeric(oracle_result.selected["pontuacao"], errors="raise").sum())
    metrics["oracle_actual_points"] = oracle_points
    if actual_points is None:
        return metrics, []

    metrics["oracle_gap"] = oracle_points - actual_points
    if oracle_points <= 0:
        return metrics, ["Oracle capture_rate is null because oracle_actual_points is not positive."]

    metrics["oracle_capture_rate"] = actual_points / oracle_points
    return metrics, []


def run_recommendation(config: RecommendationConfig) -> RecommendationResult:
    _validate_mode_scope(config)
    _validate_output_root(config)
    season_df = load_season_data(config.season, project_root=config.project_root)
    visible = _visible_season_frame(season_df, target_round=config.target_round)
    target = visible[visible["rodada"].eq(config.target_round)].copy()
    if target.empty:
        raise ValueError(f"Target round {config.target_round} not found in season {config.season} data.")
    if visible[visible["rodada"].lt(config.target_round)].empty:
        raise ValueError(f"No training history exists before target round {config.target_round}.")

    finalized_evidence = _finalized_live_data_evidence(target, scout_columns=config.scout_columns)
    finalized_detected = any(value > 0 for value in finalized_evidence.values())
    if config.mode == "live" and finalized_detected and not config.allow_finalized_live_data:
        raise ValueError(
            "live mode target-round data appears finalized: "
            f"season={config.season} target_round={config.target_round} evidence={finalized_evidence}"
        )

    footystats = _load_recommendation_footystats(config, visible)
    footystats_rows = footystats.rows if footystats is not None else None
    diagnostics = build_footystats_join_diagnostics(visible, footystats_rows) if footystats_rows is not None else None
    if diagnostics is not None and diagnostics.missing_join_keys_by_round:
        raise ValueError(f"FootyStats recommendation missing join keys: {diagnostics.missing_join_keys_by_round}")
    if diagnostics is not None and diagnostics.duplicate_join_keys_by_round:
        raise ValueError(f"FootyStats recommendation duplicate join keys: {diagnostics.duplicate_join_keys_by_round}")

    backtest_config = _backtest_config(config)
    training = build_training_frame(
        visible,
        config.target_round,
        playable_statuses=config.playable_statuses,
        fixtures=None,
        footystats_rows=footystats_rows,
    )
    candidates = build_prediction_frame(visible, config.target_round, fixtures=None, footystats_rows=footystats_rows)
    candidates = candidates[candidates["status"].isin(config.playable_statuses)].copy()
    if training.empty:
        raise ValueError(f"No training rows remain before target round {config.target_round}.")
    if candidates.empty:
        raise ValueError(f"No playable target-round candidates for round {config.target_round}.")

    feature_columns = feature_columns_for_config(backtest_config)
    scored = candidates.copy()
    baseline_model = BaselinePredictor().fit(training)
    forest_model = RandomForestPointPredictor(
        random_seed=config.random_seed,
        feature_columns=feature_columns,
    ).fit(training)
    scored["baseline_score"] = baseline_model.predict(scored)
    scored["random_forest_score"] = forest_model.predict(scored)
    scored["price_score"] = scored[MARKET_OPEN_PRICE_COLUMN].astype(float)

    optimized = optimize_squad(scored, score_column="random_forest_score", config=backtest_config)
    if optimized.status != "Optimal":
        raise ValueError(f"Recommendation optimizer failed: status={optimized.status}")

    selected = optimized.selected.copy()
    selected["predicted_points"] = selected["random_forest_score"]
    warnings: list[str] = []
    actual_points = None
    oracle_metrics = _empty_oracle_metrics()
    if config.mode == "replay":
        actual_points, actual_warnings = _replay_actual_points(selected)
        warnings.extend(actual_warnings)
        oracle_metrics, oracle_warnings = _oracle_replay_metrics(
            scored,
            actual_points=actual_points,
            config=backtest_config,
        )
        warnings.extend(oracle_warnings)

    selected_columns = [*BASE_OUTPUT_COLUMNS, "predicted_points"]
    candidate_columns = [*BASE_OUTPUT_COLUMNS, *_active_footystats_columns(config)]
    if config.mode == "replay":
        replay_columns = ["pontuacao", "entrou_em_campo", *config.scout_columns]
        selected_columns = [*selected_columns, *replay_columns]
        candidate_columns = [*candidate_columns, *replay_columns]

    recommended_squad = _select_columns(selected, selected_columns)
    candidate_predictions = _select_columns(scored, candidate_columns)
    summary: dict[str, object] = {
        "season": config.season,
        "target_round": config.target_round,
        "mode": config.mode,
        "strategy": "random_forest",
        "formation": config.formation_name,
        "budget": float(config.budget),
        "optimizer_status": optimized.status,
        "selected_count": int(optimized.selected_count),
        "budget_used": float(optimized.budget_used),
        "predicted_points": float(optimized.predicted_points),
        "actual_points": None if actual_points is None else float(actual_points),
        **oracle_metrics,
        "output_directory": str(config.output_path),
    }
    metadata = _build_metadata(
        config=config,
        visible=visible,
        feature_columns=feature_columns,
        footystats=footystats,
        finalized_detected=finalized_detected,
        finalized_evidence=finalized_evidence,
        optimizer_status=optimized.status,
        warnings=warnings,
    )

    _write_recommendation_outputs(config, recommended_squad, candidate_predictions, summary, metadata)
    return RecommendationResult(
        recommended_squad=recommended_squad,
        candidate_predictions=candidate_predictions,
        summary=summary,
        metadata=metadata,
    )


def _utc_now_z() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _build_metadata(
    *,
    config: RecommendationConfig,
    visible: pd.DataFrame,
    feature_columns: list[str],
    footystats: FootyStatsPPGLoadResult | None,
    finalized_detected: bool,
    finalized_evidence: dict[str, int],
    optimizer_status: str,
    warnings: list[str],
) -> dict[str, object]:
    training_rounds = sorted(
        int(round_number)
        for round_number in visible.loc[visible["rodada"].lt(config.target_round), "rodada"].dropna().unique()
    )
    return {
        "season": config.season,
        "target_round": config.target_round,
        "mode": config.mode,
        "current_year": _resolved_current_year(config),
        "training_rounds": training_rounds,
        "candidate_round": config.target_round,
        "visible_max_round": int(pd.to_numeric(visible["rodada"], errors="raise").max()),
        "fixture_mode": "none",
        "footystats_mode": config.footystats_mode,
        "footystats_evaluation_scope": _footystats_scope(config),
        "footystats_league_slug": config.footystats_league_slug,
        "footystats_matches_source_path": str(footystats.source_path) if footystats is not None else None,
        "footystats_matches_source_sha256": footystats.source_sha256 if footystats is not None else None,
        "feature_columns": feature_columns,
        "playable_statuses": list(config.playable_statuses),
        "formation": config.formation_name,
        "budget": float(config.budget),
        "random_seed": config.random_seed,
        "finalized_live_data_detected": finalized_detected,
        "finalized_live_data_evidence": finalized_evidence,
        "allow_finalized_live_data": config.allow_finalized_live_data,
        "live_workflow": dict(config.live_workflow) if config.live_workflow is not None else None,
        "optimizer_status": optimizer_status,
        "warnings": warnings,
        "generated_at_utc": _utc_now_z(),
    }


def _write_recommendation_outputs(
    config: RecommendationConfig,
    recommended_squad: pd.DataFrame,
    candidate_predictions: pd.DataFrame,
    summary: dict[str, object],
    metadata: dict[str, object],
) -> None:
    output_path = config.output_path
    output_path.mkdir(parents=True, exist_ok=True)
    recommended_squad.to_csv(output_path / "recommended_squad.csv", index=False)
    candidate_predictions.to_csv(output_path / "candidate_predictions.csv", index=False)
    (output_path / "recommendation_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_path / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
