from __future__ import annotations

import json
from dataclasses import dataclass

import pandas as pd

from cartola.backtesting.config import MARKET_OPEN_PRICE_COLUMN, BacktestConfig
from cartola.backtesting.data import build_round_alignment_report, load_fixtures, load_season_data
from cartola.backtesting.features import build_prediction_frame, build_training_frame, feature_columns_for_config
from cartola.backtesting.footystats_features import (
    FootyStatsJoinDiagnostics,
    FootyStatsPPGLoadResult,
    build_footystats_join_diagnostics,
    load_footystats_feature_rows,
)
from cartola.backtesting.metrics import build_diagnostics, build_summary
from cartola.backtesting.models import BaselinePredictor, RandomForestPointPredictor
from cartola.backtesting.optimizer import optimize_squad
from cartola.backtesting.scoring_contract import (
    actual_scores_with_captain,
    captain_policy_diagnostics,
    contract_fields,
)
from cartola.backtesting.strict_fixtures import load_strict_fixtures

ROUND_RESULT_COLUMNS: list[str] = [
    "rodada",
    "strategy",
    "solver_status",
    "formation",
    "selected_count",
    "budget_used",
    "predicted_points",
    "predicted_points_base",
    "captain_bonus_predicted",
    "predicted_points_with_captain",
    "actual_points",
    "actual_points_base",
    "captain_bonus_actual",
    "actual_points_with_captain",
    "captain_id",
    "captain_name",
    "captain_policy_ev_id",
    "captain_policy_safe_id",
    "captain_policy_upside_id",
    "actual_points_with_ev_captain",
    "actual_points_with_safe_captain",
    "actual_points_with_upside_captain",
]

OUTPUT_FLOAT_PRECISION = 10
CSV_FLOAT_FORMAT = f"%.{OUTPUT_FLOAT_PRECISION}f"

FLOAT_NORMALIZATION_EXCLUDED_COLUMNS: set[str] = {
    "rodada",
    "id_atleta",
    "id_clube",
    "num_jogos",
    "prior_appearances",
    "prior_num_jogos",
    "selected_count",
    "rounds",
}


@dataclass(frozen=True)
class BacktestMetadata:
    season: int
    start_round: int
    max_round: int
    scoring_contract_version: str
    captain_scoring_enabled: bool
    captain_multiplier: float
    formation_search: str
    fixture_mode: str
    strict_alignment_policy: str
    fixture_source_directory: str | None
    fixture_manifest_paths: list[str]
    fixture_manifest_sha256: dict[str, str]
    generator_versions: list[str]
    excluded_rounds: list[int]
    warnings: list[str]
    footystats_mode: str
    footystats_evaluation_scope: str
    footystats_league_slug: str
    footystats_matches_source_path: str | None
    footystats_matches_source_sha256: str | None
    footystats_feature_columns: list[str]
    footystats_missing_join_keys_by_round: dict[str, list[dict[str, int]]]
    footystats_duplicate_join_keys_by_round: dict[str, list[dict[str, int]]]
    footystats_extra_club_rows_by_round: dict[str, list[dict[str, int]]]


@dataclass(frozen=True)
class BacktestResult:
    round_results: pd.DataFrame
    selected_players: pd.DataFrame
    player_predictions: pd.DataFrame
    summary: pd.DataFrame
    diagnostics: pd.DataFrame
    metadata: BacktestMetadata


@dataclass(frozen=True)
class FixtureLoadForRun:
    fixtures: pd.DataFrame | None
    source_directory: str | None
    manifest_paths: list[str]
    manifest_sha256: dict[str, str]
    generator_versions: list[str]
    excluded_rounds: list[int]
    warnings: list[str]


def run_backtest(
    config: BacktestConfig,
    season_df: pd.DataFrame | None = None,
    fixtures: pd.DataFrame | None = None,
) -> BacktestResult:
    data = (
        season_df.copy() if season_df is not None else load_season_data(config.season, project_root=config.project_root)
    )
    resolved_fixtures = _resolve_fixtures(config, data, fixtures)
    fixture_data = resolved_fixtures.fixtures
    alignment_excluded_rounds = _validate_fixture_alignment(
        fixture_data,
        data,
        policy=config.strict_alignment_policy if config.fixture_mode == "strict" else "fail",
    )
    excluded_rounds = sorted({*resolved_fixtures.excluded_rounds, *alignment_excluded_rounds})
    if excluded_rounds:
        data = data[~pd.to_numeric(data["rodada"], errors="raise").isin(excluded_rounds)].copy()

    resolved_footystats = _resolve_footystats(config)
    footystats_rows = resolved_footystats.rows if resolved_footystats is not None else None
    footystats_diagnostics = (
        build_footystats_join_diagnostics(data, footystats_rows)
        if footystats_rows is not None
        else FootyStatsJoinDiagnostics()
    )
    _validate_footystats_join_diagnostics(footystats_diagnostics)

    round_rows: list[dict[str, object]] = []
    selected_frames: list[pd.DataFrame] = []
    prediction_frames: list[pd.DataFrame] = []

    max_round = _max_round(data)
    model_feature_columns = feature_columns_for_config(config)
    contract = contract_fields()
    metadata = BacktestMetadata(
        season=config.season,
        start_round=config.start_round,
        max_round=max_round,
        scoring_contract_version=str(contract["scoring_contract_version"]),
        captain_scoring_enabled=bool(contract["captain_scoring_enabled"]),
        captain_multiplier=float(contract["captain_multiplier"]),
        formation_search=str(contract["formation_search"]),
        fixture_mode=config.fixture_mode,
        strict_alignment_policy=config.strict_alignment_policy,
        fixture_source_directory=resolved_fixtures.source_directory,
        fixture_manifest_paths=resolved_fixtures.manifest_paths,
        fixture_manifest_sha256=resolved_fixtures.manifest_sha256,
        generator_versions=resolved_fixtures.generator_versions,
        excluded_rounds=excluded_rounds,
        warnings=resolved_fixtures.warnings,
        footystats_mode=config.footystats_mode,
        footystats_evaluation_scope=config.footystats_evaluation_scope,
        footystats_league_slug=config.footystats_league_slug,
        footystats_matches_source_path=(
            str(resolved_footystats.source_path) if resolved_footystats is not None else None
        ),
        footystats_matches_source_sha256=(
            resolved_footystats.source_sha256 if resolved_footystats is not None else None
        ),
        footystats_feature_columns=list(resolved_footystats.feature_columns) if resolved_footystats is not None else [],
        footystats_missing_join_keys_by_round=footystats_diagnostics.missing_join_keys_by_round,
        footystats_duplicate_join_keys_by_round=footystats_diagnostics.duplicate_join_keys_by_round,
        footystats_extra_club_rows_by_round=footystats_diagnostics.extra_club_rows_by_round,
    )
    for round_number in range(config.start_round, max_round + 1):
        if round_number in excluded_rounds:
            continue

        training = build_training_frame(
            data,
            round_number,
            playable_statuses=config.playable_statuses,
            fixtures=fixture_data,
            footystats_rows=footystats_rows,
        )
        candidates = build_prediction_frame(data, round_number, fixtures=fixture_data, footystats_rows=footystats_rows)
        candidates = candidates[candidates["status"].isin(config.playable_statuses)].copy()

        if training.empty or candidates.empty:
            _record_skipped_round(round_rows, round_number, "TrainingEmpty" if training.empty else "Empty")
            continue

        scored_candidates = candidates.copy()
        baseline_model = BaselinePredictor().fit(training)
        forest_model = RandomForestPointPredictor(
            random_seed=config.random_seed,
            feature_columns=model_feature_columns,
        ).fit(training)
        scored_candidates["baseline_score"] = baseline_model.predict(scored_candidates)
        scored_candidates["random_forest_score"] = forest_model.predict(scored_candidates)
        scored_candidates["price_score"] = scored_candidates[MARKET_OPEN_PRICE_COLUMN].astype(float)
        prediction_frames.append(scored_candidates.copy())

        for strategy, score_column in _strategies().items():
            strategy_candidates = scored_candidates.copy()
            strategy_candidates["predicted_points"] = strategy_candidates[score_column]
            result = optimize_squad(strategy_candidates, score_column="predicted_points", config=config)
            actual_scores = _actual_scores_for_result(
                result.selected,
                round_number=round_number,
                strategy=strategy,
                solver_status=result.status,
            )
            policy_diagnostics = _policy_diagnostics_for_result(
                result.selected,
                round_number=round_number,
                strategy=strategy,
                solver_status=result.status,
            )
            policy_summary = _policy_round_summary(policy_diagnostics)
            actual_points_with_captain = actual_scores["actual_points_with_captain"]
            round_rows.append(
                {
                    "rodada": round_number,
                    "strategy": strategy,
                    "solver_status": result.status,
                    "formation": result.formation_name,
                    "selected_count": result.selected_count,
                    "budget_used": result.budget_used,
                    "predicted_points": result.predicted_points_with_captain,
                    "predicted_points_base": result.predicted_points_base,
                    "captain_bonus_predicted": result.captain_bonus_predicted,
                    "predicted_points_with_captain": result.predicted_points_with_captain,
                    "actual_points": actual_points_with_captain,
                    "actual_points_base": actual_scores["actual_points_base"],
                    "captain_bonus_actual": actual_scores["captain_bonus_actual"],
                    "actual_points_with_captain": actual_points_with_captain,
                    "captain_id": result.captain_id,
                    "captain_name": result.captain_name,
                    **policy_summary,
                }
            )

            if not result.selected.empty:
                selected = result.selected.copy()
                _apply_captain_policy_flags(selected, policy_diagnostics)
                selected["rodada"] = round_number
                selected["strategy"] = strategy
                selected_frames.append(selected)

    round_results = pd.DataFrame(round_rows, columns=pd.Index(ROUND_RESULT_COLUMNS))
    selected_players = _concat_or_empty(selected_frames)
    player_predictions = _concat_or_empty(prediction_frames)
    summary = build_summary(round_results, benchmark_strategy="price")
    diagnostics = build_diagnostics(
        round_results,
        selected_players,
        player_predictions,
        benchmark_strategy="price",
        budget=config.budget,
        random_seed=config.random_seed,
    )

    round_results = _normalize_float_outputs(round_results)
    selected_players = _normalize_float_outputs(selected_players)
    player_predictions = _normalize_float_outputs(player_predictions)
    summary = _normalize_float_outputs(summary)
    diagnostics = _normalize_float_outputs(diagnostics)

    _write_outputs(config, round_results, selected_players, player_predictions, summary, diagnostics, metadata)
    return BacktestResult(
        round_results=round_results,
        selected_players=selected_players,
        player_predictions=player_predictions,
        summary=summary,
        diagnostics=diagnostics,
        metadata=metadata,
    )


def _resolve_footystats(config: BacktestConfig) -> FootyStatsPPGLoadResult | None:
    if config.footystats_evaluation_scope == "live_current":
        raise ValueError("live_current is not supported by the backtest runner")
    if config.footystats_mode == "none":
        return None

    return load_footystats_feature_rows(
        season=config.season,
        project_root=config.project_root,
        footystats_dir=config.footystats_dir,
        league_slug=config.footystats_league_slug,
        evaluation_scope=config.footystats_evaluation_scope,
        current_year=config.current_year,
        footystats_mode=config.footystats_mode,
    )


def _validate_footystats_join_diagnostics(diagnostics: FootyStatsJoinDiagnostics) -> None:
    if diagnostics.missing_join_keys_by_round:
        raise ValueError(f"FootyStats PPG missing join keys: {diagnostics.missing_join_keys_by_round}")
    if diagnostics.duplicate_join_keys_by_round:
        raise ValueError(f"FootyStats PPG duplicate join keys: {diagnostics.duplicate_join_keys_by_round}")


def _max_round(data: pd.DataFrame) -> int:
    if data.empty:
        return 0
    return int(data["rodada"].max())


def _load_optional_fixtures(config: BacktestConfig) -> pd.DataFrame | None:
    try:
        return load_fixtures(config.season, project_root=config.project_root)
    except FileNotFoundError:
        return None


def _strict_required_rounds(season_df: pd.DataFrame) -> list[int]:
    if season_df.empty:
        return []
    max_round = int(pd.to_numeric(season_df["rodada"], errors="raise").max())
    return list(range(1, max_round + 1))


def _resolve_fixtures(
    config: BacktestConfig,
    season_df: pd.DataFrame,
    fixtures: pd.DataFrame | None,
) -> FixtureLoadForRun:
    if config.fixture_mode == "none":
        return FixtureLoadForRun(
            fixtures=None,
            source_directory=None,
            manifest_paths=[],
            manifest_sha256={},
            generator_versions=[],
            excluded_rounds=[],
            warnings=[],
        )

    if config.fixture_mode == "strict":
        required_rounds = _strict_required_rounds(season_df)
        if config.strict_alignment_policy == "exclude_round":
            return _load_strict_fixtures_with_exclusions(config, required_rounds)

        loaded = load_strict_fixtures(
            season=config.season,
            project_root=config.project_root,
            required_rounds=required_rounds,
        )
        return FixtureLoadForRun(
            fixtures=loaded.fixtures,
            source_directory=f"data/01_raw/fixtures_strict/{config.season}",
            manifest_paths=loaded.manifest_paths,
            manifest_sha256=loaded.manifest_sha256,
            generator_versions=loaded.generator_versions,
            excluded_rounds=[],
            warnings=[],
        )

    if config.fixture_mode != "exploratory":
        raise ValueError(f"Unknown fixture_mode: {config.fixture_mode!r}")

    warnings = ["Exploratory fixture mode uses reconstructed fixture data and is not strict no-leakage."]
    if fixtures is not None:
        return FixtureLoadForRun(
            fixtures=fixtures.copy(),
            source_directory=None,
            manifest_paths=[],
            manifest_sha256={},
            generator_versions=[],
            excluded_rounds=[],
            warnings=warnings,
        )

    loaded_fixtures = _load_optional_fixtures(config)
    if loaded_fixtures is None:
        return FixtureLoadForRun(
            fixtures=None,
            source_directory=None,
            manifest_paths=[],
            manifest_sha256={},
            generator_versions=[],
            excluded_rounds=[],
            warnings=[*warnings, "Exploratory fixture files were not found; running with neutral fixture defaults."],
        )

    return FixtureLoadForRun(
        fixtures=loaded_fixtures,
        source_directory=f"data/01_raw/fixtures/{config.season}",
        manifest_paths=[],
        manifest_sha256={},
        generator_versions=[],
        excluded_rounds=[],
        warnings=warnings,
    )


def _load_strict_fixtures_with_exclusions(config: BacktestConfig, required_rounds: list[int]) -> FixtureLoadForRun:
    loaded_frames: list[pd.DataFrame] = []
    manifest_paths: list[str] = []
    manifest_sha256: dict[str, str] = {}
    generator_versions: set[str] = set()
    excluded_rounds: list[int] = []

    for round_number in required_rounds:
        try:
            loaded = load_strict_fixtures(
                season=config.season,
                project_root=config.project_root,
                required_rounds=[round_number],
            )
        except FileNotFoundError:
            excluded_rounds.append(round_number)
            continue

        round_fixtures = loaded.fixtures[
            pd.to_numeric(loaded.fixtures["rodada"], errors="raise").astype(int).eq(round_number)
        ].copy()
        loaded_frames.append(round_fixtures)
        for manifest_path in loaded.manifest_paths:
            if manifest_path not in manifest_paths:
                manifest_paths.append(manifest_path)
        manifest_sha256.update(loaded.manifest_sha256)
        generator_versions.update(loaded.generator_versions)

    return FixtureLoadForRun(
        fixtures=_concat_or_empty(loaded_frames) if loaded_frames else None,
        source_directory=f"data/01_raw/fixtures_strict/{config.season}",
        manifest_paths=manifest_paths,
        manifest_sha256=manifest_sha256,
        generator_versions=sorted(generator_versions),
        excluded_rounds=excluded_rounds,
        warnings=[],
    )


def _validate_fixture_alignment(
    fixtures: pd.DataFrame | None,
    season_df: pd.DataFrame,
    *,
    policy: str = "fail",
) -> list[int]:
    if policy not in {"fail", "exclude_round"}:
        raise ValueError(f"Unknown strict_alignment_policy: {policy!r}")

    if fixtures is None:
        return []

    report = build_round_alignment_report(fixtures, season_df)
    invalid = report[~report["is_valid"].astype(bool)]
    if invalid.empty:
        return []

    invalid_rounds = sorted(pd.to_numeric(invalid["rodada"], errors="raise").astype(int).tolist())
    if policy == "exclude_round":
        return invalid_rounds

    details = invalid[["rodada", "missing_from_fixtures", "extra_in_fixtures"]].to_dict("records")
    raise ValueError(f"Fixture alignment failed: {details}")


def _strategies() -> dict[str, str]:
    return {
        "baseline": "baseline_score",
        "random_forest": "random_forest_score",
        "price": "price_score",
    }


def _empty_score_fields() -> dict[str, float | None]:
    return {
        "actual_points_base": 0.0,
        "captain_bonus_actual": 0.0,
        "actual_points_with_captain": 0.0,
    }


def _actual_scores_for_result(
    selected: pd.DataFrame,
    *,
    round_number: int,
    strategy: str,
    solver_status: str,
) -> dict[str, float | None]:
    if solver_status != "Optimal" or selected.empty:
        return _empty_score_fields()

    try:
        return actual_scores_with_captain(selected, actual_column="pontuacao")
    except ValueError as exc:
        raise ValueError(
            f"Failed to score actual captain-aware points for round={round_number} strategy={strategy!r}."
        ) from exc


def _policy_diagnostics_for_result(
    selected: pd.DataFrame,
    *,
    round_number: int,
    strategy: str,
    solver_status: str,
) -> list[dict[str, object]]:
    if solver_status != "Optimal" or selected.empty:
        return []

    try:
        return captain_policy_diagnostics(
            selected,
            predicted_column="predicted_points",
            actual_column="pontuacao",
        )
    except ValueError as exc:
        raise ValueError(
            f"Failed to compute captain policy diagnostics for round={round_number} strategy={strategy!r}."
        ) from exc


def _policy_round_summary(policy_diagnostics: list[dict[str, object]]) -> dict[str, object]:
    by_policy = {str(record["policy"]): record for record in policy_diagnostics}
    return {
        "captain_policy_ev_id": _policy_value(by_policy, "ev", "captain_id"),
        "captain_policy_safe_id": _policy_value(by_policy, "safe", "captain_id"),
        "captain_policy_upside_id": _policy_value(by_policy, "upside", "captain_id"),
        "actual_points_with_ev_captain": _policy_value(by_policy, "ev", "actual_points_with_policy"),
        "actual_points_with_safe_captain": _policy_value(by_policy, "safe", "actual_points_with_policy"),
        "actual_points_with_upside_captain": _policy_value(by_policy, "upside", "actual_points_with_policy"),
    }


def _policy_value(
    by_policy: dict[str, dict[str, object]],
    policy: str,
    key: str,
) -> object:
    record = by_policy.get(policy)
    if record is None:
        return None
    return record[key]


def _apply_captain_policy_flags(selected: pd.DataFrame, policy_diagnostics: list[dict[str, object]]) -> None:
    policy_columns = {
        "ev": "captain_policy_ev",
        "safe": "captain_policy_safe",
        "upside": "captain_policy_upside",
    }
    for policy, column in policy_columns.items():
        if column not in selected.columns:
            selected[column] = False
        record = next((item for item in policy_diagnostics if item["policy"] == policy), None)
        if record is not None:
            selected[column] = selected["id_atleta"].eq(record["captain_id"])
        else:
            selected[column] = selected[column].fillna(False).astype(bool)


def _record_skipped_round(
    round_rows: list[dict[str, object]],
    round_number: int,
    status: str,
) -> None:
    for strategy in _strategies():
        round_rows.append(
            {
                "rodada": round_number,
                "strategy": strategy,
                "solver_status": status,
                "formation": "",
                "selected_count": 0,
                "budget_used": 0.0,
                "predicted_points": 0.0,
                "predicted_points_base": 0.0,
                "captain_bonus_predicted": 0.0,
                "predicted_points_with_captain": 0.0,
                "actual_points": 0.0,
                "actual_points_base": 0.0,
                "captain_bonus_actual": 0.0,
                "actual_points_with_captain": 0.0,
                "captain_id": None,
                "captain_name": None,
                "captain_policy_ev_id": None,
                "captain_policy_safe_id": None,
                "captain_policy_upside_id": None,
                "actual_points_with_ev_captain": None,
                "actual_points_with_safe_captain": None,
                "actual_points_with_upside_captain": None,
            }
        )


def _concat_or_empty(frames: list[pd.DataFrame]) -> pd.DataFrame:
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _normalize_float_outputs(frame: pd.DataFrame) -> pd.DataFrame:
    """Round non-identifier float outputs so repeated runs serialize identically."""
    normalized = frame.copy()
    float_columns = [
        column
        for column in normalized.select_dtypes(include=["float"]).columns
        if column not in FLOAT_NORMALIZATION_EXCLUDED_COLUMNS
    ]
    if float_columns:
        normalized.loc[:, float_columns] = normalized.loc[:, float_columns].round(OUTPUT_FLOAT_PRECISION)
    return normalized


def _write_outputs(
    config: BacktestConfig,
    round_results: pd.DataFrame,
    selected_players: pd.DataFrame,
    player_predictions: pd.DataFrame,
    summary: pd.DataFrame,
    diagnostics: pd.DataFrame,
    metadata: BacktestMetadata,
) -> None:
    output_path = config.output_path
    output_path.mkdir(parents=True, exist_ok=True)
    round_results.to_csv(output_path / "round_results.csv", index=False, float_format=CSV_FLOAT_FORMAT)
    selected_players.to_csv(output_path / "selected_players.csv", index=False, float_format=CSV_FLOAT_FORMAT)
    player_predictions.to_csv(output_path / "player_predictions.csv", index=False, float_format=CSV_FLOAT_FORMAT)
    summary.to_csv(output_path / "summary.csv", index=False, float_format=CSV_FLOAT_FORMAT)
    diagnostics.to_csv(output_path / "diagnostics.csv", index=False, float_format=CSV_FLOAT_FORMAT)
    (output_path / "run_metadata.json").write_text(
        json.dumps(metadata.__dict__, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
