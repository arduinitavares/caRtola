from __future__ import annotations

import json
from dataclasses import dataclass

import pandas as pd

from cartola.backtesting.config import MARKET_OPEN_PRICE_COLUMN, BacktestConfig
from cartola.backtesting.data import build_round_alignment_report, load_fixtures, load_season_data
from cartola.backtesting.features import build_prediction_frame, build_training_frame
from cartola.backtesting.metrics import build_diagnostics, build_summary
from cartola.backtesting.models import BaselinePredictor, RandomForestPointPredictor
from cartola.backtesting.optimizer import optimize_squad
from cartola.backtesting.strict_fixtures import load_strict_fixtures

ROUND_RESULT_COLUMNS: list[str] = [
    "rodada",
    "strategy",
    "solver_status",
    "formation",
    "selected_count",
    "budget_used",
    "predicted_points",
    "actual_points",
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

    round_rows: list[dict[str, object]] = []
    selected_frames: list[pd.DataFrame] = []
    prediction_frames: list[pd.DataFrame] = []

    max_round = _max_round(data)
    metadata = BacktestMetadata(
        season=config.season,
        start_round=config.start_round,
        max_round=max_round,
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
        footystats_matches_source_path=None,
        footystats_matches_source_sha256=None,
        footystats_feature_columns=[],
        footystats_missing_join_keys_by_round={},
        footystats_duplicate_join_keys_by_round={},
        footystats_extra_club_rows_by_round={},
    )
    for round_number in range(config.start_round, max_round + 1):
        if round_number in excluded_rounds:
            continue

        training = build_training_frame(
            data,
            round_number,
            playable_statuses=config.playable_statuses,
            fixtures=fixture_data,
        )
        candidates = build_prediction_frame(data, round_number, fixtures=fixture_data)
        candidates = candidates[candidates["status"].isin(config.playable_statuses)].copy()

        if training.empty or candidates.empty:
            _record_skipped_round(round_rows, round_number, config, "TrainingEmpty" if training.empty else "Empty")
            continue

        scored_candidates = candidates.copy()
        baseline_model = BaselinePredictor().fit(training)
        forest_model = RandomForestPointPredictor(random_seed=config.random_seed).fit(training)
        scored_candidates["baseline_score"] = baseline_model.predict(scored_candidates)
        scored_candidates["random_forest_score"] = forest_model.predict(scored_candidates)
        scored_candidates["price_score"] = scored_candidates[MARKET_OPEN_PRICE_COLUMN].astype(float)
        prediction_frames.append(scored_candidates.copy())

        for strategy, score_column in _strategies().items():
            strategy_candidates = scored_candidates.copy()
            strategy_candidates["predicted_points"] = strategy_candidates[score_column]
            result = optimize_squad(strategy_candidates, score_column="predicted_points", config=config)
            round_rows.append(
                {
                    "rodada": round_number,
                    "strategy": strategy,
                    "solver_status": result.status,
                    "formation": result.formation_name,
                    "selected_count": result.selected_count,
                    "budget_used": result.budget_used,
                    "predicted_points": result.predicted_points,
                    "actual_points": result.actual_points,
                }
            )

            if not result.selected.empty:
                selected = result.selected.copy()
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


def _record_skipped_round(
    round_rows: list[dict[str, object]],
    round_number: int,
    config: BacktestConfig,
    status: str,
) -> None:
    for strategy in _strategies():
        round_rows.append(
            {
                "rodada": round_number,
                "strategy": strategy,
                "solver_status": status,
                "formation": config.formation_name,
                "selected_count": 0,
                "budget_used": 0.0,
                "predicted_points": 0.0,
                "actual_points": 0.0,
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
