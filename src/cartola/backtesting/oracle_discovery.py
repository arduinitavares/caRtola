from __future__ import annotations

import json
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from html import escape
from pathlib import Path
from time import perf_counter
from typing import Any, Literal, cast

import pandas as pd
from pandas.errors import EmptyDataError, ParserError

from cartola.backtesting.config import BacktestConfig, FixtureMode, FootyStatsMode, MatchupContextMode
from cartola.backtesting.data import load_fixtures
from cartola.backtesting.optimizer import SquadOptimizationResult, optimize_squad
from cartola.backtesting.oracle_profiles import build_oracle_player_profile_rows, build_profile_gap_summary_rows


class ArtifactValidationError(ValueError):
    pass


class OracleObjectiveError(ValueError):
    pass


OracleDiscoveryProgressEventType = Literal[
    "report_started",
    "child_started",
    "work_planned",
    "strategy_started",
    "round_finished",
    "strategy_finished",
    "child_finished",
    "report_finished",
]


@dataclass(frozen=True)
class OracleDiscoveryProgressEvent:
    event_type: OracleDiscoveryProgressEventType
    output_path: Path
    total_rounds: int
    completed_rounds: int
    source_child_id: str | None = None
    child_index: int | None = None
    total_children: int | None = None
    season: int | None = None
    strategy: str | None = None
    model_id: str | None = None
    feature_pack: str | None = None
    round_number: int | None = None
    elapsed_seconds: float | None = None
    message: str | None = None


OracleDiscoveryProgressCallback = Callable[[OracleDiscoveryProgressEvent], None]


@dataclass(frozen=True)
class SourceRunContext:
    source_experiment_id: str
    source_child_id: str
    source_child_path: Path
    season: int
    model_id: str
    feature_pack: str
    fixture_mode: str
    matchup_context_mode: str
    budget_policy: str
    primary_strategy: str
    strategy_score_columns: dict[str, str]
    analyzed_strategies: tuple[str, ...]
    current_year: int | None = None


@dataclass(frozen=True)
class ChildArtifacts:
    round_results: pd.DataFrame
    selected_players: pd.DataFrame
    player_predictions: pd.DataFrame
    summary: pd.DataFrame
    metadata: dict[str, Any]


ROUND_RESULTS_COLUMNS = frozenset(
    {
        "rodada",
        "strategy",
        "solver_status",
        "budget_before_round",
        "budget_after_round",
        "budget_delta",
        "budget_used",
        "actual_points_with_captain",
        "captain_id",
    }
)
SELECTED_PLAYERS_COLUMNS = frozenset(
    {
        "rodada",
        "strategy",
        "id_atleta",
        "apelido",
        "posicao",
        "id_clube",
        "nome_clube",
        "entrou_em_campo",
        "preco_pre_rodada",
        "pontuacao",
        "variacao",
        "is_captain",
    }
)
PLAYER_PREDICTIONS_COLUMNS = frozenset(
    {
        "rodada",
        "id_atleta",
        "apelido",
        "posicao",
        "id_clube",
        "nome_clube",
        "status",
        "entrou_em_campo",
        "preco_pre_rodada",
        "pontuacao",
        "variacao",
    }
)
SUMMARY_COLUMNS = frozenset({"strategy", "rounds"})
METADATA_FIELDS = frozenset(
    {
        "season",
        "start_round",
        "initial_budget",
        "budget_policy",
        "fixture_mode",
        "matchup_context_mode",
        "footystats_mode",
        "scoring_contract_version",
        "fixture_source_directory",
        "fixture_manifest_sha256",
    }
)
_MISSING = object()
_PARENT_METADATA_ARTIFACT = "experiment_metadata.json"

IDENTITY_COLUMNS = [
    "source_mode",
    "source_experiment_id",
    "source_child_id",
    "season",
    "rodada",
    "strategy",
    "model_id",
    "feature_pack",
    "fixture_mode",
    "matchup_context_mode",
    "budget_policy",
    "oracle_type",
    "candidate_universe",
    "budget_path",
]
ORACLE_ROUND_RESULT_COLUMNS = [
    *IDENTITY_COLUMNS,
    "optimizer_status",
    "optimizer_formation",
    "optimizer_budget_used",
    "budget_before_round",
    "oracle_actual_points_base",
    "oracle_captain_bonus_actual",
    "oracle_actual_points_with_captain",
    "optimizer_captain_id",
    "optimizer_selected_count",
    "full_market_status",
]
ORACLE_SELECTED_PLAYER_COLUMNS = [
    *IDENTITY_COLUMNS,
    "id_atleta",
    "apelido",
    "posicao",
    "id_clube",
    "nome_clube",
    "preco_pre_rodada",
    "oracle_actual_points",
    "is_oracle_captain",
    "model_score_column",
    "model_score",
    "model_predicted_rank_overall",
    "model_predicted_rank_position",
    "entrou_em_campo",
    "status",
]
ORACLE_CAPTAIN_PROFILE_COLUMNS = [
    *IDENTITY_COLUMNS,
    "captain_id",
    "captain_name",
    "captain_position",
    "captain_club",
    "captain_status",
    "captain_is_home",
    "captain_price_percentile_position",
    "captain_price_rank_position",
    "captain_model_score",
    "captain_model_predicted_rank_overall",
    "captain_model_predicted_rank_position",
    "captain_recent_form_percentile_position",
    "captain_oracle_actual_points",
    "model_captain_id",
    "model_captain_actual_points",
    "selected_squad_captain_regret",
    "full_market_status",
]
ORACLE_PLAYER_PROFILE_COLUMNS = [
    *IDENTITY_COLUMNS,
    "id_atleta",
    "posicao",
    "profile_section",
    "profile_metric",
    "profile_value",
    "baseline_name",
    "baseline_value",
    "sample_size",
    "full_market_status",
]
MODEL_VS_ORACLE_RECALL_COLUMNS = [
    *IDENTITY_COLUMNS,
    "id_atleta",
    "posicao",
    "in_selected_squad",
    "in_model_candidate_artifact",
    "absent_from_model_candidate_artifact",
    "in_full_market",
    "full_market_status",
    "model_predicted_rank_overall",
    "model_predicted_rank_position",
    "individually_affordable",
    "squad_budget_blocked_by_counterfactual",
    "recall_bucket",
]
PROFILE_GAP_SUMMARY_COLUMNS = [
    *IDENTITY_COLUMNS,
    "profile_section",
    "profile_metric",
    "oracle_value",
    "baseline_name",
    "baseline_value",
    "absolute_gap",
    "relative_gap",
    "sample_size",
    "season_stability_count",
    "stability_label",
    "full_market_status",
]
INVALID_ORACLE_ROW_COLUMNS = [
    *IDENTITY_COLUMNS,
    "id_atleta",
    "posicao",
    "pontuacao",
    "entrou_em_campo",
    "invalid_reason",
]


def add_oracle_actual_points(frame: pd.DataFrame) -> pd.DataFrame:
    if "pontuacao" not in frame.columns:
        raise OracleObjectiveError("Missing required oracle objective columns: ['pontuacao']")

    output = frame.copy()
    numeric_points = pd.to_numeric(output["pontuacao"], errors="coerce")
    oracle_points: list[float] = []
    invalid_ids: list[object] = []
    ambiguous_ids: list[object] = []

    for index, raw_value in output["pontuacao"].items():
        row_id = _row_identifier(output, index)
        if _is_missing_scalar(raw_value):
            if "entrou_em_campo" not in output.columns:
                raise OracleObjectiveError("Missing required oracle objective columns: ['entrou_em_campo']")
            if _bool_or_none(output.loc[index, "entrou_em_campo"]) is False:
                oracle_points.append(0.0)
            else:
                ambiguous_ids.append(row_id)
            continue

        try:
            point_value = float(numeric_points.loc[index])
        except (TypeError, ValueError, OverflowError):
            invalid_ids.append(row_id)
            continue
        if not math.isfinite(point_value):
            invalid_ids.append(row_id)
            continue
        oracle_points.append(point_value)

    if invalid_ids:
        raise OracleObjectiveError(f"Invalid pontuacao for rows: {invalid_ids}")
    if ambiguous_ids:
        raise OracleObjectiveError(f"Ambiguous missing pontuacao for rows: {ambiguous_ids}")

    output["oracle_actual_points"] = oracle_points
    return output


def adapt_oracle_result(result: SquadOptimizationResult) -> dict[str, object]:
    return {
        "optimizer_status": result.status,
        "optimizer_formation": result.formation_name,
        "optimizer_budget_used": result.budget_used,
        "optimizer_selected_count": result.selected_count,
        "optimizer_captain_id": result.captain_id,
        "oracle_actual_points_base": result.predicted_points_base,
        "oracle_captain_bonus_actual": result.captain_bonus_predicted,
        "oracle_actual_points_with_captain": result.predicted_points_with_captain,
        "oracle_objective_points": result.predicted_points,
    }


def selected_squad_captain_oracle(selected: pd.DataFrame) -> dict[str, object]:
    selected_with_actuals = add_oracle_actual_points(selected)
    _require_selected_captain_oracle_columns(selected_with_actuals)

    model_captain_mask = selected_with_actuals["is_captain"].map(_bool_or_none).eq(True)
    model_captain_count = int(model_captain_mask.sum())
    if model_captain_count != 1:
        raise OracleObjectiveError(f"Selected squad must contain exactly one model captain, got {model_captain_count}")
    model_captain = selected_with_actuals.loc[model_captain_mask].iloc[0]

    non_tecnico_mask = selected_with_actuals["posicao"].map(_is_non_tecnico_position)
    captain_candidates = selected_with_actuals.loc[non_tecnico_mask]
    if captain_candidates.empty:
        raise OracleObjectiveError("Selected squad has no non-tecnico captain candidates")

    oracle_candidate_offset = int(captain_candidates["oracle_actual_points"].astype(float).argmax())
    oracle_captain = captain_candidates.iloc[oracle_candidate_offset]
    oracle_captain_actual = float(oracle_captain["oracle_actual_points"])
    model_captain_actual = float(model_captain["oracle_actual_points"])
    captain_regret = max(0.0, 0.5 * (oracle_captain_actual - model_captain_actual))

    return {
        "captain_id": _first_available_row_value(oracle_captain, ("id_atleta", "captain_id")),
        "captain_name": _first_available_row_value(oracle_captain, ("apelido", "captain_name")),
        "captain_position": _first_available_row_value(oracle_captain, ("posicao", "captain_position")),
        "captain_club": _first_available_row_value(oracle_captain, ("nome_clube", "clube", "captain_club")),
        "captain_status": _first_available_row_value(oracle_captain, ("status", "captain_status")),
        "captain_is_home": _first_available_row_value(oracle_captain, ("is_home", "captain_is_home")),
        "captain_price_percentile_position": _first_available_row_value(
            oracle_captain,
            ("price_percentile_position", "captain_price_percentile_position"),
        ),
        "captain_price_rank_position": _first_available_row_value(
            oracle_captain,
            ("price_rank_position", "captain_price_rank_position"),
        ),
        "captain_model_score": _first_available_row_value(
            oracle_captain,
            ("model_score", "predicted_points", "captain_model_score"),
        ),
        "captain_model_predicted_rank_overall": _first_available_row_value(
            oracle_captain,
            ("model_predicted_rank_overall", "captain_model_predicted_rank_overall"),
        ),
        "captain_model_predicted_rank_position": _first_available_row_value(
            oracle_captain,
            ("model_predicted_rank_position", "captain_model_predicted_rank_position"),
        ),
        "captain_recent_form_percentile_position": _first_available_row_value(
            oracle_captain,
            ("recent_form_percentile_position", "captain_recent_form_percentile_position"),
        ),
        "captain_oracle_actual_points": oracle_captain_actual,
        "model_captain_id": _first_available_row_value(model_captain, ("id_atleta", "model_captain_id")),
        "model_captain_actual_points": model_captain_actual,
        "selected_squad_captain_regret": captain_regret,
        "full_market_status": "not_available",
    }


def run_model_candidate_oracle(
    candidates: pd.DataFrame,
    *,
    config: BacktestConfig,
    budget_before_round: float,
    score_column: str,
) -> tuple[dict[str, object], pd.DataFrame]:
    candidates = _deduplicate_model_candidates(candidates, score_column=score_column)
    _validate_model_candidate_identity(candidates)
    oracle_candidates = add_oracle_actual_points(candidates)
    _require_model_candidate_score_column(oracle_candidates, score_column)
    result = optimize_squad(
        oracle_candidates,
        score_column="oracle_actual_points",
        config=config,
        budget=budget_before_round,
    )
    row = adapt_oracle_result(result)
    selected = _oracle_selected_players(
        result.selected,
        scored_candidates=oracle_candidates,
        score_column=score_column,
    )
    if result.status != "Optimal":
        return row, selected.iloc[0:0].copy()
    return row, selected


def build_oracle_discovery_report(
    *,
    experiment_path: Path,
    output_path: Path,
    progress_callback: OracleDiscoveryProgressCallback | None = None,
) -> None:
    started = perf_counter()
    output_path.mkdir(parents=True, exist_ok=True)
    _emit_progress(
        progress_callback,
        OracleDiscoveryProgressEvent(
            event_type="report_started",
            output_path=output_path,
            total_rounds=0,
            completed_rounds=0,
            elapsed_seconds=0.0,
        ),
    )
    contexts = load_source_run_contexts(experiment_path)
    round_rows: list[dict[str, object]] = []
    selected_frames: list[pd.DataFrame] = []
    captain_rows: list[dict[str, object]] = []
    player_profile_rows: list[dict[str, object]] = []
    recall_rows: list[dict[str, object]] = []
    profile_gap_rows: list[dict[str, object]] = []
    invalid_rows: list[dict[str, object]] = []
    source_provenance: list[dict[str, object]] = []
    planned_children: list[tuple[SourceRunContext, ChildArtifacts, BacktestConfig, pd.DataFrame | None]] = []
    total_rounds = 0

    for child_index, context in enumerate(contexts, start=1):
        _emit_progress(
            progress_callback,
            _progress_event(
                "child_started",
                context=context,
                output_path=output_path,
                total_rounds=0,
                completed_rounds=0,
                child_index=child_index,
                total_children=len(contexts),
                elapsed_seconds=perf_counter() - started,
            ),
        )
        artifacts = validate_child_artifacts(context)
        config = _config_from_context(context, artifacts.metadata)
        fixtures = _load_profile_fixtures(context, config)
        source_provenance.append(_source_provenance_row(context, artifacts.metadata, config=config))
        planned_children.append((context, artifacts, config, fixtures))
        total_rounds += _count_optimal_strategy_rounds(context, artifacts)

    _emit_progress(
        progress_callback,
        OracleDiscoveryProgressEvent(
            event_type="work_planned",
            output_path=output_path,
            total_rounds=total_rounds,
            completed_rounds=0,
            elapsed_seconds=perf_counter() - started,
            message=f"oracle_rounds={total_rounds}",
        ),
    )

    completed_rounds = 0
    for child_index, (context, artifacts, config, fixtures) in enumerate(planned_children, start=1):
        for strategy in context.analyzed_strategies:
            score_column = context.strategy_score_columns[strategy]
            strategy_rounds = artifacts.round_results.loc[
                artifacts.round_results["strategy"].astype(str).eq(strategy)
                & artifacts.round_results["solver_status"].astype(str).eq("Optimal")
            ]
            _emit_progress(
                progress_callback,
                _progress_event(
                    "strategy_started",
                    context=context,
                    output_path=output_path,
                    total_rounds=total_rounds,
                    completed_rounds=completed_rounds,
                    child_index=child_index,
                    total_children=len(contexts),
                    strategy=strategy,
                    elapsed_seconds=perf_counter() - started,
                    message=f"strategy_rounds={len(strategy_rounds)}",
                ),
            )
            for _, source_round in strategy_rounds.iterrows():
                round_number = int(source_round["rodada"])
                identity = _identity(context, round_number=round_number, strategy=strategy)
                candidates = _rows_for_round(artifacts.player_predictions, round_number=round_number)
                selected = _selected_rows_for_round_and_strategy(
                    artifacts.selected_players,
                    round_number=round_number,
                    strategy=strategy,
                )
                try:
                    oracle_round, oracle_selected = run_model_candidate_oracle(
                        candidates,
                        config=config,
                        budget_before_round=float(source_round["budget_before_round"]),
                        score_column=score_column,
                    )
                except OracleObjectiveError as exc:
                    invalid_rows.extend(_invalid_oracle_rows(identity, candidates, fallback_reason=str(exc)))
                    completed_rounds += 1
                    _emit_progress(
                        progress_callback,
                        _progress_event(
                            "round_finished",
                            context=context,
                            output_path=output_path,
                            total_rounds=total_rounds,
                            completed_rounds=completed_rounds,
                            child_index=child_index,
                            total_children=len(contexts),
                            strategy=strategy,
                            round_number=round_number,
                            elapsed_seconds=perf_counter() - started,
                            message=str(exc),
                        ),
                    )
                    continue

                round_rows.append(
                    _schema_row(
                        ORACLE_ROUND_RESULT_COLUMNS,
                        {
                            **identity,
                            **oracle_round,
                            "budget_before_round": float(source_round["budget_before_round"]),
                            "full_market_status": "not_available",
                        },
                    )
                )
                selected_frames.append(_oracle_selected_output(oracle_selected, identity=identity))
                if selected.empty:
                    invalid_rows.append(_missing_selected_squad_row(_selected_squad_captain_identity(identity)))
                    completed_rounds += 1
                    _emit_progress(
                        progress_callback,
                        _progress_event(
                            "round_finished",
                            context=context,
                            output_path=output_path,
                            total_rounds=total_rounds,
                            completed_rounds=completed_rounds,
                            child_index=child_index,
                            total_children=len(contexts),
                            strategy=strategy,
                            round_number=round_number,
                            elapsed_seconds=perf_counter() - started,
                            message="selected squad is empty",
                        ),
                    )
                    continue
                try:
                    captain_profile = selected_squad_captain_oracle(selected)
                except OracleObjectiveError as exc:
                    invalid_rows.extend(
                        _invalid_oracle_rows(
                            _selected_squad_captain_identity(identity),
                            selected,
                            fallback_reason=str(exc),
                        )
                    )
                    completed_rounds += 1
                    _emit_progress(
                        progress_callback,
                        _progress_event(
                            "round_finished",
                            context=context,
                            output_path=output_path,
                            total_rounds=total_rounds,
                            completed_rounds=completed_rounds,
                            child_index=child_index,
                            total_children=len(contexts),
                            strategy=strategy,
                            round_number=round_number,
                            elapsed_seconds=perf_counter() - started,
                            message=str(exc),
                        ),
                    )
                    continue
                captain_rows.append(
                    _schema_row(
                        ORACLE_CAPTAIN_PROFILE_COLUMNS,
                        {
                            **_selected_squad_captain_identity(identity),
                            **captain_profile,
                            "full_market_status": "not_available",
                        },
                    )
                )
                recall_rows.extend(_recall_rows(identity, oracle_selected, selected))
                if oracle_round.get("optimizer_status") == "Optimal" and not oracle_selected.empty:
                    round_fixtures = _fixtures_for_round(fixtures, round_number=round_number)
                    player_profile_rows.extend(
                        build_oracle_player_profile_rows(
                            identity=identity,
                            oracle_selected=oracle_selected,
                            model_selected=selected,
                            fixtures=round_fixtures,
                        )
                    )
                    profile_gap_rows.extend(
                        build_profile_gap_summary_rows(
                            identity=identity,
                            oracle_selected=oracle_selected,
                            model_selected=selected,
                            fixtures=round_fixtures,
                        )
                    )
                completed_rounds += 1
                _emit_progress(
                    progress_callback,
                    _progress_event(
                        "round_finished",
                        context=context,
                        output_path=output_path,
                        total_rounds=total_rounds,
                        completed_rounds=completed_rounds,
                        child_index=child_index,
                        total_children=len(contexts),
                        strategy=strategy,
                        round_number=round_number,
                        elapsed_seconds=perf_counter() - started,
                    ),
                )

            _emit_progress(
                progress_callback,
                _progress_event(
                    "strategy_finished",
                    context=context,
                    output_path=output_path,
                    total_rounds=total_rounds,
                    completed_rounds=completed_rounds,
                    child_index=child_index,
                    total_children=len(contexts),
                    strategy=strategy,
                    elapsed_seconds=perf_counter() - started,
                ),
            )

        _emit_progress(
            progress_callback,
            _progress_event(
                "child_finished",
                context=context,
                output_path=output_path,
                total_rounds=total_rounds,
                completed_rounds=completed_rounds,
                child_index=child_index,
                total_children=len(contexts),
                elapsed_seconds=perf_counter() - started,
            ),
        )

    _write_outputs(
        output_path=output_path,
        round_rows=round_rows,
        selected_frames=selected_frames,
        captain_rows=captain_rows,
        player_profile_rows=player_profile_rows,
        recall_rows=recall_rows,
        profile_gap_rows=profile_gap_rows,
        invalid_rows=invalid_rows,
        experiment_path=experiment_path,
        source_context_count=len(contexts),
        source_provenance=source_provenance,
    )
    _emit_progress(
        progress_callback,
        OracleDiscoveryProgressEvent(
            event_type="report_finished",
            output_path=output_path,
            total_rounds=total_rounds,
            completed_rounds=completed_rounds,
            elapsed_seconds=perf_counter() - started,
        ),
    )


def load_source_run_contexts(experiment_path: Path) -> list[SourceRunContext]:
    metadata_path = experiment_path / "experiment_metadata.json"
    payload = _read_json_object(metadata_path)
    experiment_id = str(_require_field(payload, "experiment_id", _PARENT_METADATA_ARTIFACT, "experiment_id"))
    parent_current_year = _optional_int_field(payload, "current_year", _PARENT_METADATA_ARTIFACT, "current_year")
    child_runs = _require_list_field(payload, "child_runs", _PARENT_METADATA_ARTIFACT, "child_runs")
    contexts: list[SourceRunContext] = []
    for index, child_value in enumerate(child_runs):
        child_path = f"child_runs[{index}]"
        if not isinstance(child_value, dict):
            raise ArtifactValidationError(f"{_PARENT_METADATA_ARTIFACT}: {child_path} must be an object")
        child = cast("dict[str, Any]", child_value)
        metadata = _optional_child_metadata(child, child_path)
        model_id = str(_require_field(child, "model_id", _PARENT_METADATA_ARTIFACT, f"{child_path}.model_id"))
        strategy_roles = _require_object_field(
            child,
            "strategy_roles",
            _PARENT_METADATA_ARTIFACT,
            f"{child_path}.strategy_roles",
        )
        if not isinstance(strategy_roles, dict):
            raise ArtifactValidationError(f"{_PARENT_METADATA_ARTIFACT}: {child_path}.strategy_roles must be an object")
        budget_policy = _child_metadata_field(
            child=child,
            metadata=metadata,
            field="budget_policy",
            artifact=_PARENT_METADATA_ARTIFACT,
            child_path=child_path,
        )
        matchup_context_mode = _child_metadata_field(
            child=child,
            metadata=metadata,
            field="matchup_context_mode",
            artifact=_PARENT_METADATA_ARTIFACT,
            child_path=child_path,
        )
        analyzed = tuple(str(strategy) for strategy in strategy_roles)
        score_columns = _score_columns_from_roles(model_id=model_id, strategy_roles=strategy_roles)
        current_year = _source_current_year(
            parent_current_year=parent_current_year,
            child=child,
            metadata=metadata,
            child_path=child_path,
        )
        contexts.append(
            SourceRunContext(
                source_experiment_id=experiment_id,
                source_child_id=str(_require_field(child, "child_id", _PARENT_METADATA_ARTIFACT, f"{child_path}.child_id")),
                source_child_path=_resolve_child_output_path(
                    experiment_path,
                    str(_require_field(child, "output_path", _PARENT_METADATA_ARTIFACT, f"{child_path}.output_path"))
                ),
                season=_require_int_field(child, "season", _PARENT_METADATA_ARTIFACT, f"{child_path}.season"),
                model_id=model_id,
                feature_pack=str(
                    _require_field(child, "feature_pack", _PARENT_METADATA_ARTIFACT, f"{child_path}.feature_pack")
                ),
                fixture_mode=str(
                    _require_field(child, "fixture_mode", _PARENT_METADATA_ARTIFACT, f"{child_path}.fixture_mode")
                ),
                matchup_context_mode=str(matchup_context_mode),
                budget_policy=str(budget_policy),
                primary_strategy=model_id,
                strategy_score_columns=score_columns,
                analyzed_strategies=analyzed,
                current_year=current_year,
            )
        )
    return contexts


def _emit_progress(
    callback: OracleDiscoveryProgressCallback | None,
    event: OracleDiscoveryProgressEvent,
) -> None:
    if callback is not None:
        callback(event)


def _progress_event(
    event_type: OracleDiscoveryProgressEventType,
    *,
    context: SourceRunContext,
    output_path: Path,
    total_rounds: int,
    completed_rounds: int,
    child_index: int,
    total_children: int,
    strategy: str | None = None,
    round_number: int | None = None,
    elapsed_seconds: float | None = None,
    message: str | None = None,
) -> OracleDiscoveryProgressEvent:
    return OracleDiscoveryProgressEvent(
        event_type=event_type,
        output_path=output_path,
        total_rounds=total_rounds,
        completed_rounds=completed_rounds,
        source_child_id=context.source_child_id,
        child_index=child_index,
        total_children=total_children,
        season=context.season,
        strategy=strategy,
        model_id=context.model_id,
        feature_pack=context.feature_pack,
        round_number=round_number,
        elapsed_seconds=elapsed_seconds,
        message=message,
    )


def _count_optimal_strategy_rounds(context: SourceRunContext, artifacts: ChildArtifacts) -> int:
    total = 0
    for strategy in context.analyzed_strategies:
        total += int(
            (
                artifacts.round_results["strategy"].astype(str).eq(strategy)
                & artifacts.round_results["solver_status"].astype(str).eq("Optimal")
            ).sum()
        )
    return total


def validate_child_artifacts(context: SourceRunContext) -> ChildArtifacts:
    if context.budget_policy != "moving":
        raise ArtifactValidationError(f"Source child is not moving-budget compatible: {context.source_child_id}")
    child_path = context.source_child_path
    round_results = _read_required_csv(child_path / "round_results.csv")
    selected_players = _read_required_csv(child_path / "selected_players.csv")
    player_predictions = _read_required_csv(child_path / "player_predictions.csv")
    summary = _read_required_csv(child_path / "summary.csv")
    metadata = _read_json_object(child_path / "run_metadata.json")

    _require_columns("round_results.csv", round_results, ROUND_RESULTS_COLUMNS)
    _require_columns("selected_players.csv", selected_players, SELECTED_PLAYERS_COLUMNS)
    _require_columns("player_predictions.csv", player_predictions, PLAYER_PREDICTIONS_COLUMNS)
    _require_columns("summary.csv", summary, SUMMARY_COLUMNS)
    _require_metadata(metadata)
    _require_metadata_matches_context(context, metadata)
    for strategy in context.analyzed_strategies:
        score_column = context.strategy_score_columns.get(strategy)
        if score_column is None:
            raise ArtifactValidationError(f"Missing score-column mapping for strategy: {strategy}")
        if score_column not in player_predictions.columns:
            raise ArtifactValidationError(f"Missing score column in player_predictions.csv: {score_column}")
    return ChildArtifacts(
        round_results=round_results,
        selected_players=selected_players,
        player_predictions=player_predictions,
        summary=summary,
        metadata=metadata,
    )


def _config_from_context(context: SourceRunContext, metadata: dict[str, Any]) -> BacktestConfig:
    return BacktestConfig(
        season=context.season,
        start_round=_metadata_int(metadata, "start_round", default=context.season),
        budget=_metadata_float(metadata, ("initial_budget", "budget"), default=100.0),
        fixture_mode=cast(FixtureMode, context.fixture_mode),
        matchup_context_mode=cast(MatchupContextMode, context.matchup_context_mode),
        footystats_mode=cast(FootyStatsMode, str(metadata["footystats_mode"])),
        current_year=_config_current_year(context, metadata),
        project_root=_project_root_from_reporting_path(context.source_child_path) or Path("."),
    )


def _identity(context: SourceRunContext, *, round_number: int, strategy: str) -> dict[str, object]:
    return {
        "source_mode": "artifact",
        "source_experiment_id": context.source_experiment_id,
        "source_child_id": context.source_child_id,
        "season": context.season,
        "rodada": round_number,
        "strategy": strategy,
        "model_id": context.model_id,
        "feature_pack": context.feature_pack,
        "fixture_mode": context.fixture_mode,
        "matchup_context_mode": context.matchup_context_mode,
        "budget_policy": context.budget_policy,
        "oracle_type": "budget_constrained",
        "candidate_universe": "model_candidate",
        "budget_path": "model_budget_path",
    }


def _rows_for_round(frame: pd.DataFrame, *, round_number: int) -> pd.DataFrame:
    round_values = pd.to_numeric(frame["rodada"], errors="coerce")
    return frame.loc[round_values.eq(round_number)].copy()


def _fixtures_for_round(fixtures: pd.DataFrame | None, *, round_number: int) -> pd.DataFrame | None:
    if fixtures is None or fixtures.empty:
        return None
    round_values = pd.to_numeric(fixtures["rodada"], errors="coerce")
    return fixtures.loc[round_values.eq(round_number)].copy()


def _selected_rows_for_round_and_strategy(
    frame: pd.DataFrame,
    *,
    round_number: int,
    strategy: str,
) -> pd.DataFrame:
    round_rows = _rows_for_round(frame, round_number=round_number)
    return round_rows.loc[round_rows["strategy"].astype(str).eq(strategy)].copy()


def _oracle_selected_output(oracle_selected: pd.DataFrame, *, identity: dict[str, object]) -> pd.DataFrame:
    if oracle_selected.empty:
        return _empty_frame(ORACLE_SELECTED_PLAYER_COLUMNS)
    return _frame_with_columns(oracle_selected.assign(**identity), ORACLE_SELECTED_PLAYER_COLUMNS)


def _recall_rows(
    identity: dict[str, object],
    oracle_selected: pd.DataFrame,
    selected: pd.DataFrame,
) -> list[dict[str, object]]:
    if oracle_selected.empty:
        return []
    selected_ids = _athlete_id_set(selected)
    rows: list[dict[str, object]] = []
    for _, row in oracle_selected.iterrows():
        athlete_id = row["id_atleta"]
        in_selected_squad = _normalise_athlete_id(athlete_id) in selected_ids
        rows.append(
            _schema_row(
                MODEL_VS_ORACLE_RECALL_COLUMNS,
                {
                    **identity,
                    "id_atleta": athlete_id,
                    "posicao": row.get("posicao"),
                    "in_selected_squad": in_selected_squad,
                    "in_model_candidate_artifact": True,
                    "absent_from_model_candidate_artifact": False,
                    "in_full_market": None,
                    "full_market_status": "not_available",
                    "model_predicted_rank_overall": row.get("model_predicted_rank_overall"),
                    "model_predicted_rank_position": row.get("model_predicted_rank_position"),
                    "individually_affordable": None,
                    "squad_budget_blocked_by_counterfactual": None,
                    "recall_bucket": "selected" if in_selected_squad else "missed_inside_model_candidate",
                },
            )
        )
    return rows


def _source_provenance_row(
    context: SourceRunContext,
    metadata: dict[str, Any],
    *,
    config: BacktestConfig,
) -> dict[str, object]:
    return {
        "source_experiment_id": context.source_experiment_id,
        "source_child_id": context.source_child_id,
        "source_child_path": str(context.source_child_path),
        "season": context.season,
        "start_round": config.start_round,
        "current_year": config.current_year,
        "initial_budget": config.budget,
        "budget_policy": context.budget_policy,
        "model_id": context.model_id,
        "feature_pack": context.feature_pack,
        "fixture_mode": context.fixture_mode,
        "matchup_context_mode": context.matchup_context_mode,
        "footystats_mode": metadata["footystats_mode"],
        "scoring_contract_version": metadata["scoring_contract_version"],
        "primary_strategy": context.primary_strategy,
        "analyzed_strategies": list(context.analyzed_strategies),
        "strategy_score_columns": dict(context.strategy_score_columns),
        "fixture_source_directory": metadata.get("fixture_source_directory"),
        "fixture_manifest_sha256": metadata.get("fixture_manifest_sha256"),
    }


def _load_profile_fixtures(context: SourceRunContext, config: BacktestConfig) -> pd.DataFrame | None:
    if str(context.fixture_mode).lower() == "none":
        return None
    try:
        return load_fixtures(context.season, project_root=config.project_root)
    except (EmptyDataError, FileNotFoundError, NotADirectoryError, ParserError, ValueError):
        return None


def _write_outputs(
    *,
    output_path: Path,
    round_rows: list[dict[str, object]],
    selected_frames: list[pd.DataFrame],
    captain_rows: list[dict[str, object]],
    player_profile_rows: list[dict[str, object]],
    recall_rows: list[dict[str, object]],
    profile_gap_rows: list[dict[str, object]],
    invalid_rows: list[dict[str, object]],
    experiment_path: Path,
    source_context_count: int,
    source_provenance: list[dict[str, object]],
) -> None:
    _rows_frame(round_rows, ORACLE_ROUND_RESULT_COLUMNS).to_csv(
        output_path / "oracle_round_results.csv",
        index=False,
    )
    _selected_output_frame(selected_frames).to_csv(output_path / "oracle_selected_players.csv", index=False)
    _rows_frame(captain_rows, ORACLE_CAPTAIN_PROFILE_COLUMNS).to_csv(
        output_path / "oracle_captain_profiles.csv",
        index=False,
    )
    _rows_frame(player_profile_rows, ORACLE_PLAYER_PROFILE_COLUMNS).to_csv(
        output_path / "oracle_player_profiles.csv",
        index=False,
    )
    _rows_frame(recall_rows, MODEL_VS_ORACLE_RECALL_COLUMNS).to_csv(
        output_path / "model_vs_oracle_recall.csv",
        index=False,
    )
    _rows_frame(profile_gap_rows, PROFILE_GAP_SUMMARY_COLUMNS).to_csv(
        output_path / "profile_gap_summary.csv",
        index=False,
    )
    _rows_frame(invalid_rows, INVALID_ORACLE_ROW_COLUMNS).to_csv(output_path / "invalid_oracle_rows.csv", index=False)
    metadata = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "source_mode": "artifact",
        "source_experiment_path": str(experiment_path),
        "source_context_count": source_context_count,
        "source_experiment_ids": _unique_values(source_provenance, "source_experiment_id"),
        "source_child_ids": _unique_values(source_provenance, "source_child_id"),
        "source_child_paths": _unique_values(source_provenance, "source_child_path"),
        "seasons": _unique_values(source_provenance, "season"),
        "start_round_values": _unique_values(source_provenance, "start_round"),
        "current_year_values": _unique_values(source_provenance, "current_year"),
        "initial_budget_values": _unique_values(source_provenance, "initial_budget"),
        "budget_policies": _unique_values(source_provenance, "budget_policy"),
        "model_ids": _unique_values(source_provenance, "model_id"),
        "feature_packs": _unique_values(source_provenance, "feature_pack"),
        "fixture_modes": _unique_values(source_provenance, "fixture_mode"),
        "matchup_context_modes": _unique_values(source_provenance, "matchup_context_mode"),
        "footystats_modes": _unique_values(source_provenance, "footystats_mode"),
        "scoring_contract_versions": _unique_values(source_provenance, "scoring_contract_version"),
        "oracle_variants": ["budget_constrained", "selected_squad_captain"],
        "candidate_universes": ["model_candidate", "selected_squad"],
        "budget_paths": ["model_budget_path"],
        "full_market_status": "not_available",
        "source_children": source_provenance,
        "disclaimer": "Discovery-only hindsight analysis. Not promotion evidence.",
    }
    (output_path / "oracle_discovery_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    _write_html(output_path, round_rows=round_rows, captain_rows=captain_rows, recall_rows=recall_rows)


def _unique_values(rows: list[dict[str, object]], field: str) -> list[object]:
    values = {row[field] for row in rows if row.get(field) is not None}
    return sorted(values, key=str)


def _write_html(
    output_path: Path,
    *,
    round_rows: list[dict[str, object]],
    captain_rows: list[dict[str, object]],
    recall_rows: list[dict[str, object]],
) -> None:
    round_count = len(round_rows)
    captain_regret = pd.DataFrame(captain_rows)
    total_captain_regret = (
        float(pd.to_numeric(captain_regret["selected_squad_captain_regret"], errors="coerce").fillna(0.0).sum())
        if "selected_squad_captain_regret" in captain_regret.columns
        else 0.0
    )
    recall = pd.DataFrame(recall_rows)
    model_candidate_recall = (
        int(recall["recall_bucket"].eq("selected").sum()) if "recall_bucket" in recall.columns else 0
    )
    model_candidate_missed = (
        int(recall["recall_bucket"].eq("missed_inside_model_candidate").sum())
        if "recall_bucket" in recall.columns
        else 0
    )
    total_captain_regret_text = f"{total_captain_regret:.2f}"
    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Oracle Knowledge Discovery</title>
</head>
<body>
<h1>Oracle Knowledge Discovery</h1>
<p><strong>Discovery-only hindsight analysis. Not promotion evidence.</strong></p>
<h2>Scope</h2>
<ul>
  <li>source_mode=artifact</li>
  <li>candidate_universe=model_candidate</li>
  <li>full_market_status=not_available</li>
</ul>
<h2>Summary</h2>
<ul>
  <li>Oracle rounds: {_html_text(round_count)}</li>
  <li>Selected-squad captain regret total: {_html_text(total_captain_regret_text)}</li>
  <li>Model-candidate recall: {_html_text(model_candidate_recall)}</li>
  <li>Model-candidate missed: {_html_text(model_candidate_missed)}</li>
</ul>
</body>
</html>
"""
    (output_path / "oracle_knowledge_discovery.html").write_text(html, encoding="utf-8")


def _html_text(value: object) -> str:
    return escape(str(value), quote=True)


def _selected_output_frame(selected_frames: list[pd.DataFrame]) -> pd.DataFrame:
    if not selected_frames:
        return _empty_frame(ORACLE_SELECTED_PLAYER_COLUMNS)
    return _frame_with_columns(pd.concat(selected_frames, ignore_index=True), ORACLE_SELECTED_PLAYER_COLUMNS)


def _rows_frame(rows: list[dict[str, object]], columns: list[str]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=pd.Index(columns)) if rows else _empty_frame(columns)


def _empty_frame(columns: list[str]) -> pd.DataFrame:
    return pd.DataFrame(columns=pd.Index(columns))


def _frame_with_columns(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    output = frame.copy()
    for column in columns:
        if column not in output.columns:
            output[column] = None
    return output.loc[:, columns]


def _schema_row(columns: list[str], values: Mapping[str, object]) -> dict[str, object]:
    return {column: values.get(column) for column in columns}


def _selected_squad_captain_identity(identity: dict[str, object]) -> dict[str, object]:
    return {
        **identity,
        "oracle_type": "selected_squad_captain",
        "candidate_universe": "selected_squad",
        "budget_path": "model_budget_path",
    }


def _missing_selected_squad_row(identity: dict[str, object]) -> dict[str, object]:
    return _schema_row(INVALID_ORACLE_ROW_COLUMNS, {**identity, "invalid_reason": "missing_selected_squad"})


def _invalid_oracle_rows(
    identity: dict[str, object],
    frame: pd.DataFrame,
    *,
    fallback_reason: str,
) -> list[dict[str, object]]:
    rows = [
        _schema_row(
            INVALID_ORACLE_ROW_COLUMNS,
            {
                **identity,
                "id_atleta": row.get("id_atleta"),
                "posicao": row.get("posicao"),
                "pontuacao": row.get("pontuacao"),
                "entrou_em_campo": row.get("entrou_em_campo"),
                "invalid_reason": invalid_reason,
            },
        )
        for _, row in frame.iterrows()
        if (invalid_reason := _invalid_oracle_reason(row)) is not None
    ]
    if rows:
        return rows
    return [_schema_row(INVALID_ORACLE_ROW_COLUMNS, {**identity, "invalid_reason": fallback_reason})]


def _invalid_oracle_reason(row: pd.Series) -> str | None:
    if "pontuacao" not in row.index:
        return "missing_pontuacao"
    raw_points = row["pontuacao"]
    if _is_missing_scalar(raw_points):
        if _bool_or_none(row.get("entrou_em_campo")) is False:
            return None
        return "ambiguous_missing_pontuacao"
    numeric_points = pd.to_numeric(pd.Series([raw_points]), errors="coerce").iloc[0]
    try:
        point_value = float(numeric_points)
    except (TypeError, ValueError, OverflowError):
        return "invalid_pontuacao"
    if not math.isfinite(point_value):
        return "invalid_pontuacao"
    return None


def _athlete_id_set(frame: pd.DataFrame) -> set[object]:
    if frame.empty or "id_atleta" not in frame.columns:
        return set()
    return {_normalise_athlete_id(value) for value in frame["id_atleta"].dropna()}


def _normalise_athlete_id(value: object) -> object:
    try:
        numeric_value = float(cast("Any", value))
    except (TypeError, ValueError, OverflowError):
        return value
    if math.isfinite(numeric_value) and numeric_value.is_integer():
        return int(numeric_value)
    return value


def _metadata_int(metadata: dict[str, Any], field: str, *, default: int) -> int:
    value = metadata.get(field)
    if value is None:
        return default
    if isinstance(value, bool):
        return default
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        normalized = value.strip()
        if normalized.isdecimal():
            return int(normalized)
    return default


def _metadata_float(metadata: dict[str, Any], fields: tuple[str, ...], *, default: float) -> float:
    for field in fields:
        value = metadata.get(field)
        if value is None:
            continue
        try:
            numeric_value = float(cast("Any", value))
        except (TypeError, ValueError, OverflowError):
            continue
        if math.isfinite(numeric_value):
            return numeric_value
    return default


def _config_current_year(context: SourceRunContext, metadata: dict[str, Any]) -> int:
    if context.current_year is not None:
        return context.current_year
    return _require_int_source_value(metadata, "current_year", "run_metadata.json", "current_year")


def _source_current_year(
    *,
    parent_current_year: int | None,
    child: dict[str, Any],
    metadata: dict[str, Any],
    child_path: str,
) -> int:
    candidates = [
        ("current_year", parent_current_year),
        (
            f"{child_path}.current_year",
            _optional_int_field(child, "current_year", _PARENT_METADATA_ARTIFACT, f"{child_path}.current_year"),
        ),
        (
            f"{child_path}.metadata.current_year",
            _optional_int_field(
                metadata,
                "current_year",
                _PARENT_METADATA_ARTIFACT,
                f"{child_path}.metadata.current_year",
            ),
        ),
    ]
    values = [(path, value) for path, value in candidates if value is not None]
    if not values:
        raise ArtifactValidationError(f"Missing required field in {_PARENT_METADATA_ARTIFACT}: current_year")

    current_year = values[0][1]
    for path, value in values[1:]:
        if value != current_year:
            raise ArtifactValidationError(
                f"Conflicting field in {_PARENT_METADATA_ARTIFACT}: {path}={value!r} "
                f"disagrees with current_year={current_year!r}"
            )
    return current_year


def _optional_int_field(mapping: dict[str, Any], field: str, artifact: str, path: str) -> int | None:
    if field not in mapping or mapping[field] is None:
        return None
    return _require_int_source_value(mapping, field, artifact, path)


def _require_int_source_value(mapping: dict[str, Any], field: str, artifact: str, path: str) -> int:
    value = _require_field(mapping, field, artifact, path)
    if isinstance(value, bool):
        raise ArtifactValidationError(f"Invalid integer field in {artifact}: {path}={value!r}")
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        normalized = value.strip()
        if normalized.isdecimal():
            return int(normalized)
    raise ArtifactValidationError(f"Invalid integer field in {artifact}: {path}={value!r}")


def _require_selected_captain_oracle_columns(selected: pd.DataFrame) -> None:
    required = ("is_captain", "posicao")
    missing = [column for column in required if column not in selected.columns]
    if missing:
        raise OracleObjectiveError(f"Missing required selected-squad captain oracle columns: {missing}")


def _require_model_candidate_score_column(candidates: pd.DataFrame, score_column: str) -> None:
    if score_column not in candidates.columns:
        raise ArtifactValidationError(f"Missing model score column in candidates: {score_column}")


def _critical_candidate_columns(candidates: pd.DataFrame, score_column: str) -> list[str]:
    columns = [
        "rodada",
        "id_atleta",
        "id_clube",
        "posicao",
        "preco_pre_rodada",
        "pontuacao",
        "entrou_em_campo",
        "variacao",
        score_column,
    ]
    return [column for column in dict.fromkeys(columns) if column in candidates.columns]


def _deduplicate_model_candidates(candidates: pd.DataFrame, *, score_column: str) -> pd.DataFrame:
    identity_columns = ["rodada", "id_atleta"]
    if not set(identity_columns).issubset(candidates.columns):
        return candidates

    duplicate_mask = candidates[identity_columns].duplicated(keep=False)
    if not bool(duplicate_mask.any()):
        return candidates

    critical_columns = _critical_candidate_columns(candidates, score_column)
    winner_indices: list[object] = []
    duplicate_keys = candidates.loc[duplicate_mask, identity_columns].drop_duplicates()
    for _, key in duplicate_keys.iterrows():
        round_mask = candidates["rodada"].isna() if pd.isna(key["rodada"]) else candidates["rodada"].eq(key["rodada"])
        athlete_mask = (
            candidates["id_atleta"].isna() if pd.isna(key["id_atleta"]) else candidates["id_atleta"].eq(key["id_atleta"])
        )
        group_mask = round_mask & athlete_mask
        group = candidates.loc[group_mask]
        if len(group.loc[:, critical_columns].drop_duplicates()) > 1:
            raise OracleObjectiveError(
                f"Conflicting duplicate candidate rows for rodada={key['rodada']} id_atleta={key['id_atleta']}"
            )
        populated_counts = group.notna().sum(axis=1)
        winner_indices.append(populated_counts.sort_values(ascending=False, kind="mergesort").index[0])

    keep_indices = candidates.index[~duplicate_mask].append(pd.Index(winner_indices)).sort_values()
    return candidates.loc[keep_indices].reset_index(drop=True)


def _validate_model_candidate_identity(candidates: pd.DataFrame) -> None:
    if "rodada" not in candidates.columns:
        raise OracleObjectiveError("Model-candidate oracle candidates must contain exactly one rodada")
    if "id_atleta" not in candidates.columns:
        raise OracleObjectiveError(
            "Model-candidate oracle candidates must include id_atleta to validate unique (rodada, id_atleta)"
        )
    if candidates["rodada"].isna().any():
        raise OracleObjectiveError("Model-candidate oracle candidates must contain exactly one rodada")

    rounds = candidates["rodada"].drop_duplicates()
    if len(rounds) != 1:
        raise OracleObjectiveError(
            f"Model-candidate oracle candidates must contain exactly one rodada, got {len(rounds)}"
        )
    if candidates[["rodada", "id_atleta"]].duplicated().any():
        raise OracleObjectiveError("Duplicate candidate rows for (rodada, id_atleta)")


def _oracle_selected_players(
    selected: pd.DataFrame,
    *,
    scored_candidates: pd.DataFrame,
    score_column: str,
) -> pd.DataFrame:
    output = selected.copy()
    if "is_captain" in output.columns:
        output = output.rename(columns={"is_captain": "is_oracle_captain"})
    elif "is_oracle_captain" not in output.columns:
        output["is_oracle_captain"] = pd.Series(dtype=bool)
    output["model_score_column"] = score_column
    output["model_score"] = output[score_column]
    ranks = _prediction_ranks(scored_candidates, score_column=score_column)
    return output.merge(
        ranks,
        on=["rodada", "id_atleta"],
        how="left",
        validate="many_to_one",
    )


def _prediction_ranks(candidates: pd.DataFrame, *, score_column: str) -> pd.DataFrame:
    ranked = candidates.loc[:, ["rodada", "id_atleta", "posicao", score_column]].copy()
    ranked["model_predicted_rank_overall"] = ranked.groupby("rodada")[score_column].rank(
        method="min",
        ascending=False,
    )
    ranked["model_predicted_rank_position"] = ranked.groupby(["rodada", "posicao"])[score_column].rank(
        method="min",
        ascending=False,
    )
    return ranked.loc[:, ["rodada", "id_atleta", "model_predicted_rank_overall", "model_predicted_rank_position"]]


def _is_non_tecnico_position(value: object) -> bool:
    if _is_missing_scalar(value):
        return False
    return str(value).strip().lower() != "tec"


def _first_available_row_value(row: pd.Series, columns: tuple[str, ...]) -> object:
    for column in columns:
        value = _optional_row_value(row, column)
        if value is not None:
            return value
    return None


def _optional_row_value(row: pd.Series, column: str) -> object:
    if column not in row.index:
        return None
    value = row[column]
    if _is_missing_scalar(value):
        return None
    return value


def _score_columns_from_roles(*, model_id: str, strategy_roles: Mapping[str, object]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for strategy, role in strategy_roles.items():
        strategy_id = str(strategy)
        role_id = str(role)
        if strategy_id == "baseline":
            mapping[strategy_id] = "baseline_score"
        elif strategy_id == "price":
            mapping[strategy_id] = "price_score"
        elif role_id == "primary_model" or strategy_id == model_id:
            mapping[strategy_id] = f"{model_id}_score"
        else:
            raise ArtifactValidationError(f"Non-standard strategy requires explicit score mapping: {strategy_id}")
    return mapping


def _resolve_child_output_path(experiment_path: Path, output_path: str) -> Path:
    child_path = Path(output_path)
    if child_path.is_absolute():
        return child_path
    project_root = _project_root_from_reporting_path(experiment_path)
    if project_root is not None:
        return project_root / child_path
    parent_candidate = experiment_path.parent / child_path
    if parent_candidate.exists():
        return parent_candidate
    return child_path


def _project_root_from_reporting_path(experiment_path: Path) -> Path | None:
    parts = experiment_path.resolve().parts
    for index, part in enumerate(parts[:-1]):
        if part == "data" and parts[index + 1] == "08_reporting":
            if index == 0:
                return None
            return Path(*parts[:index])
    return None


def _optional_child_metadata(child: dict[str, Any], child_path: str) -> dict[str, Any]:
    metadata = child.get("metadata", {})
    if metadata is None:
        return {}
    if not isinstance(metadata, dict):
        raise ArtifactValidationError(f"{_PARENT_METADATA_ARTIFACT}: {child_path}.metadata must be an object")
    return cast("dict[str, Any]", metadata)


def _child_metadata_field(
    *,
    child: dict[str, Any],
    metadata: dict[str, Any],
    field: str,
    artifact: str,
    child_path: str,
) -> object:
    top_level_value = child.get(field, _MISSING)
    nested_value = metadata.get(field, _MISSING)
    if top_level_value is _MISSING and nested_value is _MISSING:
        raise ArtifactValidationError(
            f"Missing required field in {artifact}: {child_path}.{field} or {child_path}.metadata.{field}"
        )
    if top_level_value is not _MISSING and nested_value is not _MISSING and top_level_value != nested_value:
        raise ArtifactValidationError(
            f"Conflicting field in {artifact}: {child_path}.{field}={top_level_value!r} "
            f"disagrees with {child_path}.metadata.{field}={nested_value!r}"
        )
    value = top_level_value if top_level_value is not _MISSING else nested_value
    if value is None:
        raise ArtifactValidationError(f"Missing required field in {artifact}: {child_path}.{field}")
    return value


def _require_field(mapping: dict[str, Any], field: str, artifact: str, path: str) -> object:
    if field not in mapping or mapping[field] is None:
        raise ArtifactValidationError(f"Missing required field in {artifact}: {path}")
    return mapping[field]


def _require_list_field(mapping: dict[str, Any], field: str, artifact: str, path: str) -> list[object]:
    value = _require_field(mapping, field, artifact, path)
    if not isinstance(value, list):
        raise ArtifactValidationError(f"{artifact}: {path} must be a list")
    return cast("list[object]", value)


def _require_object_field(mapping: dict[str, Any], field: str, artifact: str, path: str) -> dict[str, Any]:
    value = _require_field(mapping, field, artifact, path)
    if not isinstance(value, dict):
        raise ArtifactValidationError(f"{artifact}: {path} must be an object")
    return cast("dict[str, Any]", value)


def _require_int_field(mapping: dict[str, Any], field: str, artifact: str, path: str) -> int:
    value = _require_field(mapping, field, artifact, path)
    if isinstance(value, bool):
        raise ArtifactValidationError(f"Invalid integer field in {artifact}: {path}={value!r}")
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        normalized = value.strip()
        if normalized.isdecimal():
            return int(normalized)
    raise ArtifactValidationError(f"Invalid integer field in {artifact}: {path}={value!r}")


def _require_str_field(mapping: dict[str, Any], field: str, artifact: str, path: str) -> str:
    value = _require_field(mapping, field, artifact, path)
    if not isinstance(value, str):
        raise ArtifactValidationError(f"Invalid string field in {artifact}: {path}={value!r}")
    return value


def _read_json_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ArtifactValidationError(f"Missing required JSON artifact: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ArtifactValidationError(f"Unable to read JSON artifact: {path}") from exc
    if not isinstance(payload, dict):
        raise ArtifactValidationError(f"JSON artifact must contain an object: {path}")
    return cast("dict[str, Any]", payload)


def _read_required_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise ArtifactValidationError(f"Missing required CSV artifact: {path}")
    try:
        return pd.read_csv(path)
    except (OSError, UnicodeError, EmptyDataError, ParserError, ValueError) as exc:
        raise ArtifactValidationError(f"Unable to read CSV artifact: {path}") from exc


def _require_columns(name: str, frame: pd.DataFrame, required: frozenset[str]) -> None:
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ArtifactValidationError(f"Missing required columns in {name}: {missing}")


def _require_metadata(metadata: dict[str, Any]) -> None:
    missing = sorted(METADATA_FIELDS.difference(metadata))
    if missing:
        raise ArtifactValidationError(f"Missing required fields in run_metadata.json: {missing}")


def _require_metadata_matches_context(context: SourceRunContext, metadata: dict[str, Any]) -> None:
    metadata_season = _require_int_field(metadata, "season", "run_metadata.json", "season")
    if metadata_season != context.season:
        raise ArtifactValidationError(
            f"run_metadata.json field season={metadata_season!r} disagrees with source context {context.season!r}"
        )
    _require_matching_str_metadata(metadata, "fixture_mode", context.fixture_mode)
    _require_matching_str_metadata(metadata, "matchup_context_mode", context.matchup_context_mode)
    metadata_budget_policy = _require_matching_str_metadata(metadata, "budget_policy", context.budget_policy)
    if metadata_budget_policy != "moving":
        raise ArtifactValidationError(f"Source child is not moving-budget compatible: {context.source_child_id}")
    footystats_mode = _require_str_field(metadata, "footystats_mode", "run_metadata.json", "footystats_mode")
    if _feature_pack_footystats_mode(context.feature_pack) != footystats_mode:
        raise ArtifactValidationError(
            "run_metadata.json field footystats_mode="
            f"{footystats_mode!r} disagrees with source context feature_pack={context.feature_pack!r}"
        )


def _require_matching_str_metadata(metadata: dict[str, Any], field: str, expected: str) -> str:
    actual = _require_str_field(metadata, field, "run_metadata.json", field)
    if actual != expected:
        raise ArtifactValidationError(
            f"run_metadata.json field {field}={actual!r} disagrees with source context {expected!r}"
        )
    return actual


def _feature_pack_footystats_mode(feature_pack: str) -> str:
    return feature_pack.removesuffix("_matchup")


def _row_identifier(frame: pd.DataFrame, index: object) -> object:
    if "id_atleta" in frame.columns:
        return frame.loc[index, "id_atleta"]
    return index


def _bool_or_none(value: object) -> bool | None:
    if _is_missing_scalar(value):
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1"}:
            return True
        if normalized in {"false", "0"}:
            return False
        return None
    try:
        numeric_value = float(cast("Any", value))
    except (TypeError, ValueError, OverflowError):
        return None
    if not math.isfinite(numeric_value):
        return None
    if numeric_value == 1.0:
        return True
    if numeric_value == 0.0:
        return False
    return None


def _is_missing_scalar(value: object) -> bool:
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False
