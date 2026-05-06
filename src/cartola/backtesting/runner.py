from __future__ import annotations

import json
import os
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from threading import Lock
from time import perf_counter

import pandas as pd
from threadpoolctl import threadpool_info as _raw_threadpool_info

from cartola.backtesting.budgeting import (
    BUDGET_POLICY_MOVING,
    BudgetRoundUpdate,
    BudgetState,
    advance_budget,
    initial_budget_state,
)
from cartola.backtesting.config import MARKET_OPEN_PRICE_COLUMN, BacktestConfig
from cartola.backtesting.data import build_round_alignment_report, load_fixtures, load_season_data
from cartola.backtesting.features import (
    MARKET_COLUMNS,  # noqa: F401
    MATCHUP_CONTEXT_V1_FEATURE_COLUMNS,
    build_prediction_frame,
    feature_columns_for_config,
)
from cartola.backtesting.footystats_features import (
    FootyStatsJoinDiagnostics,
    FootyStatsPPGLoadResult,
    build_footystats_join_diagnostics,
    load_footystats_feature_rows,
)
from cartola.backtesting.metrics import build_diagnostics, build_summary
from cartola.backtesting.model_registry import ModelId, create_point_predictor, resolve_model_id
from cartola.backtesting.models import BaselinePredictor
from cartola.backtesting.optimizer import optimize_squad
from cartola.backtesting.scoring_contract import (
    CAPTAIN_MULTIPLIER,
    CAPTAIN_SCORING_ENABLED,
    FORMATION_SEARCH,
    SCORING_CONTRACT_VERSION,
    actual_scores_with_captain,
    apply_captain_policy_flags,
    captain_policy_diagnostics,
)
from cartola.backtesting.strict_fixtures import load_strict_fixtures

ROUND_RESULT_COLUMNS: list[str] = [
    "rodada",
    "strategy",
    "solver_status",
    "formation",
    "selected_count",
    "budget_used",
    "budget_before_round",
    "budget_after_round",
    "budget_delta",
    "budget_remaining",
    "budget_peak",
    "budget_drawdown",
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

SORT_KEYS: dict[str, list[str]] = {
    "round_results": ["rodada", "strategy"],
    "selected_players": ["rodada", "strategy", "id_atleta"],
    "player_predictions": ["rodada", "id_atleta"],
    "summary": ["strategy"],
    "diagnostics": ["section", "strategy", "position", "metric"],
}

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

THREAD_ENV_KEYS: tuple[str, ...] = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "BLIS_NUM_THREADS",
)


@dataclass(frozen=True)
class BacktestMetadata:
    season: int
    start_round: int
    max_round: int
    cache_enabled: bool
    prediction_frames_built: int
    wall_clock_seconds: float
    backtest_jobs: int
    backtest_workers_effective: int
    model_n_jobs_effective: int
    parallel_backend: str
    budget_policy: str
    initial_budget: float
    thread_env: dict[str, str | None]
    scoring_contract_version: str
    captain_scoring_enabled: bool
    captain_multiplier: float
    formation_search: str
    fixture_mode: str
    strict_alignment_policy: str
    matchup_context_mode: str
    matchup_context_feature_columns: list[str]
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
    runtime_profile_enabled: bool = False
    round_profiles: list[dict[str, object]] = field(default_factory=list)
    threadpool_info: list[dict[str, object]] = field(default_factory=list)


@dataclass(frozen=True)
class BacktestResult:
    round_results: pd.DataFrame
    selected_players: pd.DataFrame
    player_predictions: pd.DataFrame
    summary: pd.DataFrame
    diagnostics: pd.DataFrame
    metadata: BacktestMetadata


@dataclass(frozen=True)
class RoundEvaluationResult:
    round_number: int
    round_rows: list[dict[str, object]]
    selected_frames: list[pd.DataFrame]
    prediction_frames: list[pd.DataFrame]
    budget_states: dict[str, BudgetState] = field(default_factory=dict)
    profile: dict[str, object] = field(default_factory=dict)


class BacktestRoundEvaluationError(RuntimeError):
    def __init__(self, round_number: int, message: str) -> None:
        super().__init__(f"Failed to evaluate round {round_number}: {message}")
        self.round_number = round_number


@dataclass(frozen=True)
class FixtureLoadForRun:
    fixtures: pd.DataFrame | None
    source_directory: str | None
    manifest_paths: list[str]
    manifest_sha256: dict[str, str]
    generator_versions: list[str]
    excluded_rounds: list[int]
    warnings: list[str]


class RoundFrameStore:
    def __init__(
        self,
        *,
        season_df: pd.DataFrame,
        fixtures: pd.DataFrame | None,
        footystats_rows: pd.DataFrame | None,
        matchup_context_mode: str,
    ) -> None:
        self._season_df = season_df
        self._fixtures = fixtures
        self._footystats_rows = footystats_rows
        self._matchup_context_mode = matchup_context_mode
        self._frames: dict[int, pd.DataFrame] = {}
        self._lock = Lock()

    @property
    def prediction_frames_built(self) -> int:
        return len(self._frames)

    def build_all(self, rounds: list[int]) -> None:
        for round_number in rounds:
            if round_number in self._frames:
                continue
            self._frames[round_number] = build_prediction_frame(
                self._season_df,
                round_number,
                fixtures=self._fixtures,
                footystats_rows=self._footystats_rows,
                matchup_context_mode=self._matchup_context_mode,
            ).copy(deep=True)

    def prediction_frame(self, round_number: int) -> pd.DataFrame:
        with self._lock:
            try:
                frame = self._frames[round_number]
            except KeyError as exc:
                raise KeyError(f"Prediction frame for round {round_number} was not built.") from exc
            return frame.copy(deep=True)

    def training_frame(
        self,
        *,
        target_round: int,
        playable_statuses: tuple[str, ...],
        empty_columns: list[str],
    ) -> pd.DataFrame:
        with self._lock:
            copied_frames = [
                self._frames[round_number].copy(deep=True)
                for round_number in sorted(self._frames)
                if round_number < target_round
            ]

        frames: list[pd.DataFrame] = []
        for round_frame in copied_frames:
            round_frame = round_frame[round_frame["status"].isin(playable_statuses)].copy(deep=True)
            round_frame["target"] = round_frame["pontuacao"]
            frames.append(round_frame)

        if not frames:
            return pd.DataFrame(columns=pd.Index(empty_columns))

        return pd.concat(frames, ignore_index=True)


def run_backtest(
    config: BacktestConfig,
    season_df: pd.DataFrame | None = None,
    fixtures: pd.DataFrame | None = None,
) -> BacktestResult:
    return _run_backtest(
        config,
        primary_model_id="random_forest",
        season_df=season_df,
        fixtures=fixtures,
        model_params=None,
    )


def run_backtest_for_experiment(
    config: BacktestConfig,
    *,
    primary_model_id: str,
    season_df: pd.DataFrame | None = None,
    fixtures: pd.DataFrame | None = None,
    model_params: Mapping[str, object] | None = None,
) -> BacktestResult:
    return _run_backtest(
        config,
        primary_model_id=primary_model_id,
        season_df=season_df,
        fixtures=fixtures,
        model_params=model_params,
    )


def _run_backtest(
    config: BacktestConfig,
    *,
    primary_model_id: str,
    season_df: pd.DataFrame | None = None,
    fixtures: pd.DataFrame | None = None,
    model_params: Mapping[str, object] | None = None,
) -> BacktestResult:
    started_at = perf_counter()
    resolved_primary_model_id = resolve_model_id(primary_model_id)
    data = (
        season_df.copy() if season_df is not None else load_season_data(config.season, project_root=config.project_root)
    )
    _validate_matchup_context_config(config)
    resolved_fixtures = _resolve_fixtures(config, data, fixtures)
    fixture_data = resolved_fixtures.fixtures
    if config.matchup_context_mode != "none" and fixture_data is None:
        raise ValueError("matchup_context_mode requires fixture_mode='exploratory' or 'strict' with available fixtures")
    alignment_excluded_rounds = _validate_fixture_alignment(
        fixture_data,
        data,
        policy=config.strict_alignment_policy if config.fixture_mode == "strict" else "fail",
    )
    max_round = _max_round(data)
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

    cached_rounds = _detected_rounds(data)
    cached_round_set = set(cached_rounds)
    round_frame_store = RoundFrameStore(
        season_df=data,
        fixtures=fixture_data,
        footystats_rows=footystats_rows,
        matchup_context_mode=config.matchup_context_mode,
    )
    round_frame_store.build_all(cached_rounds)

    round_rows: list[dict[str, object]] = []
    selected_frames: list[pd.DataFrame] = []
    prediction_frames: list[pd.DataFrame] = []

    model_feature_columns = feature_columns_for_config(config)
    empty_training_columns = list(dict.fromkeys([*MARKET_COLUMNS, *model_feature_columns, "target"]))
    target_rounds = list(range(config.start_round, max_round + 1))
    model_n_jobs_effective = _effective_model_n_jobs(1)
    backtest_workers_effective = 1 if target_rounds else 0
    parallel_backend = "sequential_moving_budget" if target_rounds else "none"
    moving_budget_warnings = []
    if config.jobs > 1 and target_rounds:
        moving_budget_warnings.append("Target-round parallelism is disabled by moving-budget semantics.")
    metadata = BacktestMetadata(
        season=config.season,
        start_round=config.start_round,
        max_round=max_round,
        cache_enabled=True,
        prediction_frames_built=round_frame_store.prediction_frames_built,
        wall_clock_seconds=0.0,
        backtest_jobs=config.jobs,
        backtest_workers_effective=backtest_workers_effective,
        model_n_jobs_effective=model_n_jobs_effective,
        parallel_backend=parallel_backend,
        budget_policy=BUDGET_POLICY_MOVING,
        initial_budget=float(config.budget),
        thread_env=_thread_env(),
        scoring_contract_version=SCORING_CONTRACT_VERSION,
        captain_scoring_enabled=CAPTAIN_SCORING_ENABLED,
        captain_multiplier=CAPTAIN_MULTIPLIER,
        formation_search=FORMATION_SEARCH,
        fixture_mode=config.fixture_mode,
        strict_alignment_policy=config.strict_alignment_policy,
        matchup_context_mode=config.matchup_context_mode,
        matchup_context_feature_columns=_matchup_context_feature_columns(config),
        fixture_source_directory=resolved_fixtures.source_directory,
        fixture_manifest_paths=resolved_fixtures.manifest_paths,
        fixture_manifest_sha256=resolved_fixtures.manifest_sha256,
        generator_versions=resolved_fixtures.generator_versions,
        excluded_rounds=excluded_rounds,
        warnings=[*resolved_fixtures.warnings, *moving_budget_warnings],
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
        runtime_profile_enabled=config.profile_runtime,
        round_profiles=[],
        threadpool_info=_threadpool_info(),
    )
    round_results_for_targets = [
        *_run_rounds_with_moving_budget(
            config=config,
            target_rounds=target_rounds,
            excluded_rounds=set(excluded_rounds),
            cached_round_set=cached_round_set,
            round_frame_store=round_frame_store,
            empty_training_columns=empty_training_columns,
            model_feature_columns=model_feature_columns,
            model_n_jobs_effective=model_n_jobs_effective,
            primary_model_id=resolved_primary_model_id,
            model_params=model_params,
        ),
    ]
    ordered_evaluations = sorted(round_results_for_targets, key=lambda item: item.round_number)
    round_profiles: list[dict[str, object]] = []
    for evaluation in ordered_evaluations:
        round_rows.extend(evaluation.round_rows)
        selected_frames.extend(evaluation.selected_frames)
        prediction_frames.extend(evaluation.prediction_frames)
        if config.profile_runtime and evaluation.profile:
            round_profiles.append(evaluation.profile)

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

    round_results, selected_players, player_predictions, summary, diagnostics = _sort_outputs(
        round_results,
        selected_players,
        player_predictions,
        summary,
        diagnostics,
    )

    round_results = _normalize_float_outputs(round_results)
    selected_players = _normalize_float_outputs(selected_players)
    player_predictions = _normalize_float_outputs(player_predictions)
    summary = _normalize_float_outputs(summary)
    diagnostics = _normalize_float_outputs(diagnostics)

    metadata = replace(
        metadata,
        wall_clock_seconds=round(perf_counter() - started_at, OUTPUT_FLOAT_PRECISION),
        round_profiles=round_profiles,
        threadpool_info=_threadpool_info(),
    )
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


def _validate_matchup_context_config(config: BacktestConfig) -> None:
    if config.matchup_context_mode == "none":
        return
    if config.matchup_context_mode != "cartola_matchup_v1":
        raise ValueError(f"Unsupported matchup_context_mode: {config.matchup_context_mode!r}")
    if config.fixture_mode == "none":
        raise ValueError("matchup_context_mode='cartola_matchup_v1' requires fixture_mode='exploratory' or 'strict'")


def _matchup_context_feature_columns(config: BacktestConfig) -> list[str]:
    if config.matchup_context_mode == "none":
        return []
    if config.matchup_context_mode == "cartola_matchup_v1":
        return list(MATCHUP_CONTEXT_V1_FEATURE_COLUMNS)
    raise ValueError(f"Unsupported matchup_context_mode: {config.matchup_context_mode!r}")


def _validate_footystats_join_diagnostics(diagnostics: FootyStatsJoinDiagnostics) -> None:
    if diagnostics.missing_join_keys_by_round:
        raise ValueError(f"FootyStats PPG missing join keys: {diagnostics.missing_join_keys_by_round}")
    if diagnostics.duplicate_join_keys_by_round:
        raise ValueError(f"FootyStats PPG duplicate join keys: {diagnostics.duplicate_join_keys_by_round}")


def _max_round(data: pd.DataFrame) -> int:
    if data.empty:
        return 0
    return int(data["rodada"].max())


def _detected_rounds(data: pd.DataFrame) -> list[int]:
    if data.empty:
        return []
    return sorted(
        int(round_number)
        for round_number in pd.to_numeric(data["rodada"], errors="raise").dropna().unique()
    )


def _effective_model_n_jobs(backtest_jobs: int) -> int:
    if backtest_jobs == 1:
        return -1
    return 1


def _thread_env() -> dict[str, str | None]:
    return {key: os.environ.get(key) for key in THREAD_ENV_KEYS}


def _threadpool_info() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for item in _raw_threadpool_info():
        rows.append(
            {
                "user_api": item.get("user_api"),
                "internal_api": item.get("internal_api"),
                "num_threads": item.get("num_threads"),
                "prefix": item.get("prefix"),
                "version": item.get("version"),
            }
        )
    return rows


def _run_rounds_with_moving_budget(
    *,
    config: BacktestConfig,
    target_rounds: list[int],
    excluded_rounds: set[int],
    cached_round_set: set[int],
    round_frame_store: RoundFrameStore,
    empty_training_columns: list[str],
    model_feature_columns: list[str],
    model_n_jobs_effective: int,
    primary_model_id: ModelId,
    model_params: Mapping[str, object] | None = None,
) -> list[RoundEvaluationResult]:
    budget_states = {
        strategy: initial_budget_state(config.budget)
        for strategy in _strategies(primary_model_id)
    }
    results: list[RoundEvaluationResult] = []
    for round_number in target_rounds:
        if round_number in excluded_rounds or round_number not in cached_round_set:
            result = _evaluate_skipped_target_round_with_budget_state(
                round_number=round_number,
                status="Excluded" if round_number in excluded_rounds else "Empty",
                primary_model_id=primary_model_id,
                budget_states=budget_states,
            )
        else:
            try:
                result = _evaluate_target_round(
                    round_number=round_number,
                    config=config,
                    round_frame_store=round_frame_store,
                    empty_training_columns=empty_training_columns,
                    model_feature_columns=model_feature_columns,
                    model_n_jobs_effective=model_n_jobs_effective,
                    primary_model_id=primary_model_id,
                    model_params=model_params,
                    budget_states=budget_states,
                )
            except Exception as exc:
                raise BacktestRoundEvaluationError(round_number, str(exc)) from exc
        budget_states = result.budget_states
        results.append(result)
    return results


def _evaluate_skipped_target_round_with_budget_state(
    *,
    round_number: int,
    status: str,
    primary_model_id: ModelId,
    budget_states: Mapping[str, BudgetState],
) -> RoundEvaluationResult:
    round_rows: list[dict[str, object]] = []
    next_budget_states = _record_skipped_round(
        round_rows,
        round_number,
        status,
        primary_model_id=primary_model_id,
        budget_states=budget_states,
    )
    return RoundEvaluationResult(
        round_number=round_number,
        round_rows=round_rows,
        selected_frames=[],
        prediction_frames=[],
        budget_states=next_budget_states,
    )


def _evaluate_target_round(
    *,
    round_number: int,
    config: BacktestConfig,
    round_frame_store: RoundFrameStore,
    model_feature_columns: list[str],
    empty_training_columns: list[str],
    model_n_jobs_effective: int,
    primary_model_id: ModelId,
    model_params: Mapping[str, object] | None = None,
    budget_states: Mapping[str, BudgetState] | None = None,
) -> RoundEvaluationResult:
    round_started = perf_counter()
    profile: dict[str, object] = {"round_number": round_number} if config.profile_runtime else {}
    round_rows: list[dict[str, object]] = []
    selected_frames: list[pd.DataFrame] = []
    prediction_frames: list[pd.DataFrame] = []
    next_budget_states = dict(budget_states or {})

    started = perf_counter()
    training = round_frame_store.training_frame(
        target_round=round_number,
        playable_statuses=config.playable_statuses,
        empty_columns=empty_training_columns,
    )
    training_frame_seconds = perf_counter() - started
    started = perf_counter()
    candidates = round_frame_store.prediction_frame(round_number)
    candidates = candidates[candidates["status"].isin(config.playable_statuses)].copy(deep=True)
    candidate_frame_seconds = perf_counter() - started
    if config.profile_runtime:
        profile.update(
            {
                "training_frame_seconds": training_frame_seconds,
                "candidate_frame_seconds": candidate_frame_seconds,
                "training_rows": len(training),
                "training_columns": len(training.columns),
                "candidate_rows": len(candidates),
                "candidate_columns": len(candidates.columns),
                "feature_count": len(model_feature_columns),
            }
        )

    if training.empty or candidates.empty:
        if config.profile_runtime:
            profile.update(
                {
                    "status": "TrainingEmpty" if training.empty else "Empty",
                    "total_seconds": perf_counter() - round_started,
                }
            )
        next_budget_states = _record_skipped_round(
            round_rows,
            round_number,
            "TrainingEmpty" if training.empty else "Empty",
            primary_model_id=primary_model_id,
            budget_states=next_budget_states,
        )
        return RoundEvaluationResult(
            round_number=round_number,
            round_rows=round_rows,
            selected_frames=selected_frames,
            prediction_frames=prediction_frames,
            budget_states=next_budget_states,
            profile=profile,
        )

    scored_candidates = candidates.copy()
    started = perf_counter()
    baseline_model = BaselinePredictor().fit(training)
    baseline_fit_seconds = perf_counter() - started
    started = perf_counter()
    primary_model = create_point_predictor(
        model_id=primary_model_id,
        random_seed=config.random_seed,
        feature_columns=model_feature_columns,
        n_jobs=model_n_jobs_effective,
        model_params=model_params,
    ).fit(training)
    primary_model_fit_seconds = perf_counter() - started
    primary_score_column = f"{primary_model_id}_score"
    started = perf_counter()
    scored_candidates["baseline_score"] = baseline_model.predict(scored_candidates)
    baseline_predict_seconds = perf_counter() - started
    started = perf_counter()
    scored_candidates[primary_score_column] = primary_model.predict(scored_candidates)
    primary_model_predict_seconds = perf_counter() - started
    scored_candidates["price_score"] = scored_candidates[MARKET_OPEN_PRICE_COLUMN].astype(float)
    prediction_frames.append(scored_candidates.copy())
    if config.profile_runtime:
        profile.update(
            {
                "baseline_fit_seconds": baseline_fit_seconds,
                "primary_model_fit_seconds": primary_model_fit_seconds,
                "baseline_predict_seconds": baseline_predict_seconds,
                "primary_model_predict_seconds": primary_model_predict_seconds,
                **getattr(primary_model, "last_fit_profile_", {}),
                **getattr(primary_model, "last_predict_profile_", {}),
            }
        )

    optimizer_seconds_by_strategy: dict[str, float] = {}
    for strategy, score_column in _strategies(primary_model_id).items():
        strategy_candidates = scored_candidates.copy()
        strategy_candidates["predicted_points"] = strategy_candidates[score_column]
        budget_state = next_budget_states.get(strategy, initial_budget_state(config.budget))
        started = perf_counter()
        result = optimize_squad(
            strategy_candidates,
            score_column="predicted_points",
            config=config,
            budget=budget_state.current_budget,
        )
        optimizer_seconds_by_strategy[strategy] = perf_counter() - started
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
        budget_update = advance_budget(budget_state, result.selected, budget_used=result.budget_used)
        next_budget_states[strategy] = budget_update.next_state
        round_rows.append(
            {
                "rodada": round_number,
                "strategy": strategy,
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
            apply_captain_policy_flags(selected, policy_diagnostics)
            selected["rodada"] = round_number
            selected["strategy"] = strategy
            selected_frames.append(selected)

    if config.profile_runtime:
        profile.update(
            {
                "status": "ok",
                "optimizer_seconds_by_strategy": optimizer_seconds_by_strategy,
                "optimizer_total_seconds": sum(optimizer_seconds_by_strategy.values()),
                "total_seconds": perf_counter() - round_started,
            }
        )
    return RoundEvaluationResult(
        round_number=round_number,
        round_rows=round_rows,
        selected_frames=selected_frames,
        prediction_frames=prediction_frames,
        budget_states=next_budget_states,
        profile=profile,
    )


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


def _strategies(primary_model_id: ModelId) -> dict[str, str]:
    return {
        "baseline": "baseline_score",
        primary_model_id: f"{primary_model_id}_score",
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
        scores = actual_scores_with_captain(selected, actual_column="pontuacao")
    except ValueError as exc:
        raise ValueError(
            f"Failed to score actual captain-aware points for round={round_number} strategy={strategy!r}."
        ) from exc
    return {
        "actual_points_base": scores["actual_points_base"],
        "captain_bonus_actual": scores["captain_bonus_actual"],
        "actual_points_with_captain": scores["actual_points_with_captain"],
    }


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


def _record_skipped_round(
    round_rows: list[dict[str, object]],
    round_number: int,
    status: str,
    *,
    primary_model_id: ModelId,
    budget_states: Mapping[str, BudgetState] | None = None,
) -> dict[str, BudgetState]:
    next_budget_states = dict(budget_states or {})
    empty_selected = pd.DataFrame({"variacao": pd.Series(dtype=float)})
    for strategy in _strategies(primary_model_id):
        budget_state = next_budget_states.get(strategy)
        if budget_state is None:
            budget_fields = _missing_budget_fields()
        else:
            budget_update = advance_budget(budget_state, empty_selected, budget_used=0.0)
            next_budget_states[strategy] = budget_update.next_state
            budget_fields = _budget_fields_from_update(budget_update)
        round_rows.append(
            {
                "rodada": round_number,
                "strategy": strategy,
                "solver_status": status,
                "formation": "",
                "selected_count": 0,
                "budget_used": 0.0,
                **budget_fields,
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
    return next_budget_states


def _missing_budget_fields() -> dict[str, object]:
    return {
        "budget_before_round": None,
        "budget_after_round": None,
        "budget_delta": None,
        "budget_remaining": None,
        "budget_peak": None,
        "budget_drawdown": None,
    }


def _budget_fields_from_update(budget_update: BudgetRoundUpdate) -> dict[str, object]:
    return {
        "budget_before_round": budget_update.budget_before_round,
        "budget_after_round": budget_update.budget_after_round,
        "budget_delta": budget_update.budget_delta,
        "budget_remaining": budget_update.budget_remaining,
        "budget_peak": budget_update.budget_peak,
        "budget_drawdown": budget_update.budget_drawdown,
    }


def _concat_or_empty(frames: list[pd.DataFrame]) -> pd.DataFrame:
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _sort_frame(name: str, frame: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()

    missing_columns = [key for key in keys if key not in frame.columns]
    if missing_columns:
        raise ValueError(f"{name} is missing required sort columns: {missing_columns}")

    return frame.sort_values(keys, kind="mergesort").reset_index(drop=True)


def _sort_outputs(
    round_results: pd.DataFrame,
    selected_players: pd.DataFrame,
    player_predictions: pd.DataFrame,
    summary: pd.DataFrame,
    diagnostics: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return (
        _sort_frame("round_results", round_results, SORT_KEYS["round_results"]),
        _sort_frame("selected_players", selected_players, SORT_KEYS["selected_players"]),
        _sort_frame("player_predictions", player_predictions, SORT_KEYS["player_predictions"]),
        _sort_frame("summary", summary, SORT_KEYS["summary"]),
        _sort_frame("diagnostics", diagnostics, SORT_KEYS["diagnostics"]),
    )


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
