from __future__ import annotations

import argparse
import html
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import pandas as pd
from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

from cartola.backtesting.config import BacktestConfig
from cartola.backtesting.optimizer import optimize_squad
from cartola.backtesting.optimizer_policies import (
    NO_POLICY,
    FixtureCoverageError,
    OptimizerPolicy,
    get_policy_set,
    normalize_policy_candidates,
    validate_fixture_coverage,
)
from cartola.backtesting.scoring_contract import (
    CAPTAIN_MULTIPLIER,
    SCORING_CONTRACT_VERSION,
    actual_scores_with_captain,
)

_SOURCE_ARTIFACTS: tuple[str, ...] = ("player_predictions.csv", "round_results.csv", "selected_players.csv")
_DIRECT_FIXTURE_ARTIFACTS: tuple[str, ...] = ("fixtures_for_round.csv", "fixtures.csv", "round_fixtures.csv")
_FIXTURE_COLUMNS: tuple[str, ...] = ("rodada", "id_clube_home", "id_clube_away")
POLICY_INVALID_ROW_COLUMNS: tuple[str, ...] = (
    "season",
    "model_id",
    "feature_pack",
    "child_path",
    "error_type",
    "error_message",
)
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
_H001_SELECTED_SEASONS: tuple[int, ...] = (2021, 2022, 2023, 2024, 2025)
_POLICY_CONTEXT_COLUMNS: tuple[str, ...] = ("model_id", "feature_pack", "strategy")
_POLICY_GROUP_COLUMNS: tuple[str, ...] = (*_POLICY_CONTEXT_COLUMNS, "policy_variant")
_POLICY_SEASON_GROUP_COLUMNS: tuple[str, ...] = ("season", *_POLICY_GROUP_COLUMNS)
_RANKED_BENCHMARK_EVIDENCE_COLUMNS: tuple[str, ...] = (
    "benchmark_total_actual_points",
    "benchmark_final_budget",
    "benchmark_min_budget",
    "benchmark_max_budget_drawdown",
    "benchmark_non_optimal_rounds",
    "total_delta",
    "final_budget_delta",
    "min_budget_delta",
    "max_drawdown_delta",
    "non_optimal_delta",
)

POLICY_ROUND_RESULT_COLUMNS: tuple[str, ...] = (
    "season",
    "model_id",
    "feature_pack",
    "strategy",
    "policy_variant",
    "rodada",
    "solver_status",
    "formation",
    "captain_id",
    "budget_before_round",
    "budget_used",
    "budget_remaining",
    "budget_delta",
    "budget_after_round",
    "predicted_points_with_captain",
    "actual_points_with_captain",
)
POLICY_SELECTED_PLAYER_COLUMNS: tuple[str, ...] = (
    "season",
    "model_id",
    "feature_pack",
    "strategy",
    "policy_variant",
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
    "is_captain",
)
POLICY_RANKED_SUMMARY_COLUMNS: tuple[str, ...] = (
    "rank",
    "model_id",
    "feature_pack",
    "strategy",
    "policy_variant",
    "selected_seasons",
    "fixture_identity_status",
    "rounds",
    "total_actual_points",
    "benchmark_total_actual_points",
    "total_delta",
    "improved_seasons",
    "season_2025_delta",
    "top_two_positive_delta_concentration",
    "final_budget",
    "benchmark_final_budget",
    "final_budget_delta",
    "min_budget",
    "benchmark_min_budget",
    "min_budget_delta",
    "max_budget_drawdown",
    "benchmark_max_budget_drawdown",
    "max_drawdown_delta",
    "non_optimal_rounds",
    "benchmark_non_optimal_rounds",
    "non_optimal_delta",
    "decision_status",
    "decision_reason",
)
POLICY_PER_SEASON_SUMMARY_COLUMNS: tuple[str, ...] = (
    "season",
    "model_id",
    "feature_pack",
    "strategy",
    "policy_variant",
    "rounds",
    "total_actual_points",
    "benchmark_total_actual_points",
    "total_delta",
    "final_budget",
    "benchmark_final_budget",
    "final_budget_delta",
    "min_budget",
    "benchmark_min_budget",
    "min_budget_delta",
    "max_budget_drawdown",
    "benchmark_max_budget_drawdown",
    "max_drawdown_delta",
    "non_optimal_rounds",
    "benchmark_non_optimal_rounds",
    "non_optimal_delta",
)
POLICY_PROFILE_SUMMARY_COLUMNS: tuple[str, ...] = (
    "season",
    "model_id",
    "feature_pack",
    "strategy",
    "policy_variant",
    "id_atleta",
    "apelido",
    "posicao",
    "id_clube",
    "nome_clube",
    "selected_rounds",
    "captain_rounds",
    "total_pontuacao",
    "average_pontuacao",
    "total_variacao",
    "average_preco_pre_rodada",
    "first_round",
    "last_round",
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
    fixture_source_directory: Path | None


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


@dataclass(frozen=True)
class PolicyReplayResult:
    round_rows: list[dict[str, object]]
    selected_player_rows: list[dict[str, object]]
    invalid_rows: list[dict[str, object]]


@dataclass(frozen=True)
class PolicyDecision:
    status: str
    reason: str


@dataclass(frozen=True)
class PolicySimulationRunResult:
    output_path: Path
    simulation_id: str


@dataclass(frozen=True)
class _PolicySimulationChildSpec:
    child_path: Path
    season: int
    model_id: str
    feature_pack: str


def load_policy_source_context(child_path: Path) -> PolicySourceContext:
    resolved_child_path = Path(child_path)
    metadata = _read_metadata(resolved_child_path)
    _validate_metadata(metadata)
    _validate_artifact_files(resolved_child_path)

    season = _metadata_int(metadata, "season")
    model_id = _metadata_text(metadata, "primary_model_id") or _metadata_text(metadata, "model_id")
    feature_pack = _metadata_text(metadata, "feature_pack")
    path_model_id: str | None = None
    path_feature_pack: str | None = None
    if model_id is None or feature_pack is None:
        path_model_id, path_feature_pack = _canonical_child_path_context(resolved_child_path, season=season)

    if model_id is None:
        model_id = path_model_id
    if model_id is None:
        raise PolicySimulationError("Cannot determine model_id from source metadata or canonical child path.")

    if feature_pack is None:
        feature_pack = path_feature_pack
    if feature_pack is None:
        raise PolicySimulationError("Cannot determine feature_pack from source metadata or canonical child path.")

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
        season=season,
        model_id=model_id,
        feature_pack=feature_pack,
        fixture_mode=_required_metadata_text(metadata, "fixture_mode"),
        matchup_context_mode=str(metadata.get("matchup_context_mode", "none")),
        budget_policy="moving",
        scoring_contract_version=SCORING_CONTRACT_VERSION,
        score_column=score_column,
        strategy=strategy,
        fixture_source_directory=_metadata_directory(metadata, "fixture_source_directory", child_path=resolved_child_path),
    )


def run_policy_replay_for_child(*, child_path: Path, policies: tuple[OptimizerPolicy, ...]) -> PolicyReplayResult:
    context = load_policy_source_context(child_path)
    player_predictions = _read_csv(context.child_path / "player_predictions.csv")
    round_results = _read_csv(context.child_path / "round_results.csv")
    target_rounds = _target_rounds_from_predictions(player_predictions)
    initial_budget = _initial_budget_for_policy_replay(context=context, round_results=round_results)

    round_rows: list[dict[str, object]] = []
    selected_player_rows: list[dict[str, object]] = []
    for policy in policies:
        current_budget = initial_budget
        for round_number in target_rounds:
            budget_before_round = float(current_budget)
            candidates = _round_candidates(player_predictions, context=context, round_number=round_number)
            normalized_candidates = normalize_policy_candidates(candidates, score_column=context.score_column)
            fixtures_for_round = _fixtures_for_policy_round(
                context=context,
                candidates=normalized_candidates,
                policy=policy,
                round_number=round_number,
            )
            replay_config = BacktestConfig(
                season=context.season,
                start_round=round_number,
                budget=budget_before_round,
                fixture_mode="none",
                matchup_context_mode="none",
            )
            result = optimize_squad(
                normalized_candidates,
                score_column=context.score_column,
                config=replay_config,
                budget=budget_before_round,
                policy=policy,
                fixtures_for_round=fixtures_for_round,
            )

            if result.status != "Optimal" or result.selected.empty:
                budget_used = 0.0
                budget_remaining = budget_before_round
                budget_delta = 0.0
                budget_after_round = budget_before_round
                actual_points_with_captain = 0.0
                predicted_points_with_captain = 0.0
            else:
                selected = _selected_for_policy_scoring(
                    result.selected,
                    round_number=round_number,
                    policy_variant=policy.policy_variant,
                )
                budget_used = float(
                    _finite_selected_numeric(
                        selected,
                        column="preco_pre_rodada",
                        round_number=round_number,
                        policy_variant=policy.policy_variant,
                    ).sum()
                )
                budget_remaining = budget_before_round - budget_used
                budget_delta = float(selected["variacao"].sum())
                budget_after_round = budget_before_round + budget_delta
                actual_points_with_captain = _actual_points_with_captain_from_scored_selection(
                    selected,
                    round_number=round_number,
                    policy_variant=policy.policy_variant,
                )
                predicted_points_with_captain = float(result.predicted_points_with_captain)
                selected_player_rows.extend(
                    _selected_player_output_rows(
                        selected,
                        context=context,
                        policy_variant=policy.policy_variant,
                        round_number=round_number,
                    )
                )

            round_rows.append(
                _policy_replay_round_row(
                    context=context,
                    policy_variant=policy.policy_variant,
                    round_number=round_number,
                    solver_status=result.status,
                    formation=result.formation_name,
                    captain_id=result.captain_id,
                    budget_before_round=budget_before_round,
                    budget_used=budget_used,
                    budget_remaining=budget_remaining,
                    budget_delta=budget_delta,
                    budget_after_round=budget_after_round,
                    predicted_points_with_captain=predicted_points_with_captain,
                    actual_points_with_captain=actual_points_with_captain,
                )
            )
            current_budget = budget_after_round

    return PolicyReplayResult(
        round_rows=round_rows,
        selected_player_rows=selected_player_rows,
        invalid_rows=[],
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
    solver_status_match = str(source_round["solver_status"]) == result.status
    captain_id_match = _optional_int(source_round["captain_id"]) == result.captain_id
    formation_match = str(source_round["formation"]) == result.formation_name
    budget_used_delta = result.budget_used - float(source_round["budget_used"])
    predicted_points_delta = result.predicted_points_with_captain - float(source_round["predicted_points_with_captain"])
    actual_points_delta = actual_scores["actual_points_with_captain"] - float(source_round["actual_points_with_captain"])

    mismatch_reasons = _mismatch_reasons(
        solver_status_match=solver_status_match,
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


def decide_policy_variant(
    *,
    selected_seasons: tuple[int, ...],
    fixture_identity_status: str,
    total_delta: float,
    improved_seasons: int,
    season_2025_delta: float | None,
    non_optimal_delta: int,
    final_budget_delta: float,
    min_budget_delta: float,
    max_drawdown_delta: float,
    top_two_concentration: float | None,
) -> PolicyDecision:
    if selected_seasons != _H001_SELECTED_SEASONS:
        return PolicyDecision(
            status="diagnostic_only",
            reason="generation 1 requires 2021-2025 selected seasons.",
        )
    if fixture_identity_status != "verified":
        return PolicyDecision(
            status="diagnostic_only",
            reason="fixture identity unverified for policy simulation.",
        )
    if non_optimal_delta > 0:
        return PolicyDecision(
            status="ineligible",
            reason="policy introduced non-optimal solver rounds versus no_policy.",
        )
    if total_delta <= 0:
        return PolicyDecision(status="rejected", reason="total delta versus no_policy is not positive.")
    if improved_seasons < 3:
        return PolicyDecision(status="rejected", reason="policy improved fewer than three seasons.")
    if season_2025_delta is None or season_2025_delta < -25.0:
        return PolicyDecision(status="rejected", reason="2025 delta fails the regression guardrail.")
    if final_budget_delta < -5.0 or min_budget_delta < -5.0 or max_drawdown_delta > 5.0:
        return PolicyDecision(status="rejected", reason="budget path delta fails the guardrail.")
    if top_two_concentration is not None and top_two_concentration > 0.50:
        return PolicyDecision(
            status="rejected",
            reason="top two rounds concentration is above the guardrail.",
        )
    return PolicyDecision(status="candidate_policy", reason="policy clears generation 1 evidence guardrails.")


def build_policy_ranked_summary(
    round_results: pd.DataFrame,
    *,
    selected_seasons: tuple[int, ...] = (),
    fixture_identity_status: str = "unverified",
) -> pd.DataFrame:
    if round_results.empty:
        return pd.DataFrame(columns=pd.Index(POLICY_RANKED_SUMMARY_COLUMNS))

    _validate_frame_columns("round_results", round_results, POLICY_ROUND_RESULT_COLUMNS)
    per_season_summary = build_policy_per_season_summary(round_results)
    rows: list[dict[str, object]] = []
    selected_seasons_label = ",".join(str(season) for season in selected_seasons)
    for group_key, group in per_season_summary.groupby(list(_POLICY_GROUP_COLUMNS), sort=True):
        model_id, feature_pack, strategy, policy_variant = cast(tuple[object, object, object, object], group_key)
        missing_selected_seasons = _missing_selected_policy_seasons(group, selected_seasons)
        benchmark_evidence_complete = not missing_selected_seasons and _has_valid_ranked_benchmark_evidence(group)
        improved_seasons = int(pd.to_numeric(group["total_delta"], errors="coerce").gt(0.0).sum())
        total_delta: object = float("nan")
        benchmark_total_actual_points: object = float("nan")
        final_budget: object = _finite_number_or_zero(group["final_budget"].min())
        benchmark_final_budget: object = float("nan")
        final_budget_delta: object = float("nan")
        benchmark_min_budget: object = float("nan")
        min_budget_delta: object = float("nan")
        benchmark_max_budget_drawdown: object = float("nan")
        max_drawdown_delta: object = float("nan")
        non_optimal_delta: object = float("nan")
        benchmark_non_optimal_rounds: object = float("nan")
        top_two_concentration = None
        season_2025_delta = _season_delta(group, season=2025) if benchmark_evidence_complete else None
        if benchmark_evidence_complete:
            final_budget_float, benchmark_final_budget_float, final_budget_delta_float = (
                _worst_final_budget_values(group)
            )
            total_delta_float = _finite_number_or_zero(group["total_delta"].sum())
            non_optimal_delta_int = int(_finite_number_or_zero(group["non_optimal_delta"].sum()))
            min_budget_delta_float = _finite_number_or_zero(group["min_budget_delta"].min())
            max_drawdown_delta_float = _finite_number_or_zero(group["max_drawdown_delta"].max())
            final_budget = final_budget_float
            benchmark_final_budget = benchmark_final_budget_float
            final_budget_delta = final_budget_delta_float
            total_delta = total_delta_float
            benchmark_total_actual_points = _finite_number_or_zero(group["benchmark_total_actual_points"].sum())
            non_optimal_delta = non_optimal_delta_int
            benchmark_min_budget = _finite_number_or_zero(group["benchmark_min_budget"].min())
            min_budget_delta = min_budget_delta_float
            benchmark_max_budget_drawdown = _finite_number_or_zero(
                group["benchmark_max_budget_drawdown"].max()
            )
            max_drawdown_delta = max_drawdown_delta_float
            benchmark_non_optimal_rounds = int(group["benchmark_non_optimal_rounds"].sum())
            top_two_concentration = _top_two_positive_delta_concentration(
                round_results,
                model_id=str(model_id),
                feature_pack=str(feature_pack),
                strategy=str(strategy),
                policy_variant=str(policy_variant),
            )
            decision = decide_policy_variant(
                selected_seasons=selected_seasons,
                fixture_identity_status=fixture_identity_status,
                total_delta=total_delta_float,
                improved_seasons=improved_seasons,
                season_2025_delta=season_2025_delta,
                non_optimal_delta=non_optimal_delta_int,
                final_budget_delta=final_budget_delta_float,
                min_budget_delta=min_budget_delta_float,
                max_drawdown_delta=max_drawdown_delta_float,
                top_two_concentration=top_two_concentration,
            )
        elif missing_selected_seasons:
            decision = PolicyDecision(
                status="ineligible",
                reason=(
                    "missing selected season evidence for "
                    f"{','.join(str(season) for season in missing_selected_seasons)}."
                ),
            )
        else:
            decision = PolicyDecision(
                status="ineligible",
                reason="missing no_policy benchmark evidence for one or more policy seasons.",
            )
        rows.append(
            {
                "rank": 0,
                "model_id": str(model_id),
                "feature_pack": str(feature_pack),
                "strategy": str(strategy),
                "policy_variant": str(policy_variant),
                "selected_seasons": selected_seasons_label,
                "fixture_identity_status": fixture_identity_status,
                "rounds": int(group["rounds"].sum()),
                "total_actual_points": _finite_number_or_zero(group["total_actual_points"].sum()),
                "benchmark_total_actual_points": benchmark_total_actual_points,
                "total_delta": total_delta,
                "improved_seasons": improved_seasons,
                "season_2025_delta": pd.NA if season_2025_delta is None else season_2025_delta,
                "top_two_positive_delta_concentration": (
                    pd.NA if top_two_concentration is None else top_two_concentration
                ),
                "final_budget": final_budget,
                "benchmark_final_budget": benchmark_final_budget,
                "final_budget_delta": final_budget_delta,
                "min_budget": _finite_number_or_zero(group["min_budget"].min()),
                "benchmark_min_budget": benchmark_min_budget,
                "min_budget_delta": min_budget_delta,
                "max_budget_drawdown": _finite_number_or_zero(group["max_budget_drawdown"].max()),
                "benchmark_max_budget_drawdown": benchmark_max_budget_drawdown,
                "max_drawdown_delta": max_drawdown_delta,
                "non_optimal_rounds": int(group["non_optimal_rounds"].sum()),
                "benchmark_non_optimal_rounds": benchmark_non_optimal_rounds,
                "non_optimal_delta": non_optimal_delta,
                "decision_status": decision.status,
                "decision_reason": decision.reason,
            }
        )

    summary = pd.DataFrame(rows, columns=pd.Index(POLICY_RANKED_SUMMARY_COLUMNS))
    if summary.empty:
        return summary
    summary = summary.sort_values(
        [*_POLICY_CONTEXT_COLUMNS, "total_delta", "policy_variant"],
        ascending=[True, True, True, False, True],
        kind="mergesort",
        na_position="last",
    ).reset_index(drop=True)
    summary["rank"] = summary.groupby(list(_POLICY_CONTEXT_COLUMNS), sort=False).cumcount() + 1
    return summary.loc[:, POLICY_RANKED_SUMMARY_COLUMNS]


def build_policy_per_season_summary(round_results: pd.DataFrame) -> pd.DataFrame:
    if round_results.empty:
        return pd.DataFrame(columns=pd.Index(POLICY_PER_SEASON_SUMMARY_COLUMNS))

    _validate_frame_columns("round_results", round_results, POLICY_ROUND_RESULT_COLUMNS)
    variant_summaries = _policy_variant_season_summaries(round_results)
    benchmark = variant_summaries.loc[variant_summaries["policy_variant"].eq("no_policy")].rename(
        columns={
            "total_actual_points": "benchmark_total_actual_points",
            "final_budget": "benchmark_final_budget",
            "min_budget": "benchmark_min_budget",
            "max_budget_drawdown": "benchmark_max_budget_drawdown",
            "non_optimal_rounds": "benchmark_non_optimal_rounds",
        }
    )
    benchmark_columns = (
        "season",
        *_POLICY_CONTEXT_COLUMNS,
        "benchmark_total_actual_points",
        "benchmark_final_budget",
        "benchmark_min_budget",
        "benchmark_max_budget_drawdown",
        "benchmark_non_optimal_rounds",
    )
    summary = variant_summaries.merge(
        benchmark.loc[:, benchmark_columns],
        on=["season", *_POLICY_CONTEXT_COLUMNS],
        how="left",
        sort=False,
    )
    summary["total_delta"] = summary["total_actual_points"] - summary["benchmark_total_actual_points"]
    summary["final_budget_delta"] = summary["final_budget"] - summary["benchmark_final_budget"]
    summary["min_budget_delta"] = summary["min_budget"] - summary["benchmark_min_budget"]
    summary["max_drawdown_delta"] = summary["max_budget_drawdown"] - summary["benchmark_max_budget_drawdown"]
    summary["non_optimal_delta"] = summary["non_optimal_rounds"] - summary["benchmark_non_optimal_rounds"]
    summary = summary.sort_values(
        ["season", *_POLICY_CONTEXT_COLUMNS, "total_delta", "policy_variant"],
        ascending=[True, True, True, True, False, True],
        kind="mergesort",
        na_position="last",
    )
    return summary.loc[:, POLICY_PER_SEASON_SUMMARY_COLUMNS].reset_index(drop=True)


def build_policy_profile_summary(round_results: pd.DataFrame, selected_players: pd.DataFrame) -> pd.DataFrame:
    if selected_players.empty:
        return pd.DataFrame(columns=pd.Index(POLICY_PROFILE_SUMMARY_COLUMNS))

    if not round_results.empty:
        _validate_frame_columns("round_results", round_results, POLICY_ROUND_RESULT_COLUMNS)
    _validate_frame_columns("selected_players", selected_players, POLICY_SELECTED_PLAYER_COLUMNS)
    selected = selected_players.copy()
    selected["season"] = _whole_number_column(selected, artifact_name="selected_players", column="season")
    selected["rodada"] = _whole_number_column(selected, artifact_name="selected_players", column="rodada")
    selected["id_atleta"] = _whole_number_column(selected, artifact_name="selected_players", column="id_atleta")
    selected["id_clube"] = _whole_number_column(selected, artifact_name="selected_players", column="id_clube")
    selected["pontuacao"] = pd.to_numeric(selected["pontuacao"], errors="coerce").fillna(0.0).astype(float)
    selected["variacao"] = pd.to_numeric(selected["variacao"], errors="coerce").fillna(0.0).astype(float)
    selected["preco_pre_rodada"] = (
        pd.to_numeric(selected["preco_pre_rodada"], errors="coerce").fillna(0.0).astype(float)
    )
    selected["is_captain"] = _boolean_mask(selected["is_captain"]).astype(int)
    profile_group_columns = (
        "season",
        *_POLICY_GROUP_COLUMNS,
        "id_atleta",
        "apelido",
        "posicao",
        "id_clube",
        "nome_clube",
    )
    summary = (
        selected.groupby(list(profile_group_columns), sort=True, dropna=False)
        .agg(
            selected_rounds=("rodada", "nunique"),
            captain_rounds=("is_captain", "sum"),
            total_pontuacao=("pontuacao", "sum"),
            average_pontuacao=("pontuacao", "mean"),
            total_variacao=("variacao", "sum"),
            average_preco_pre_rodada=("preco_pre_rodada", "mean"),
            first_round=("rodada", "min"),
            last_round=("rodada", "max"),
        )
        .reset_index()
    )
    summary = summary.sort_values(
        [
            "season",
            *_POLICY_CONTEXT_COLUMNS,
            "policy_variant",
            "selected_rounds",
            "captain_rounds",
            "id_atleta",
        ],
        ascending=[True, True, True, True, True, False, False, True],
        kind="mergesort",
    )
    return summary.loc[:, POLICY_PROFILE_SUMMARY_COLUMNS].reset_index(drop=True)


def write_policy_simulation_report(
    output_path: Path,
    manifest: dict[str, object],
    ranked_summary: pd.DataFrame,
    per_season_summary: pd.DataFrame,
    profile_summary: pd.DataFrame,
    comparability_report: dict[str, object],
    invalid_rows: pd.DataFrame | None = None,
) -> None:
    report_path = _policy_report_path(output_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    invalid_row_table = (
        pd.DataFrame(columns=pd.Index(POLICY_INVALID_ROW_COLUMNS))
        if invalid_rows is None
        else invalid_rows
    )
    sections = [
        "<h1>Policy Simulation V1</h1>",
        "<p>H001 generation: 2021-2025. This report is research evidence only.</p>",
        "<p>Decision status vocabulary includes diagnostic_only.</p>",
        _json_section("Manifest", manifest),
        _json_section("Comparability Report", comparability_report),
        _table_section("Ranked Summary", ranked_summary),
        _table_section("Per-Season Summary", per_season_summary),
        _table_section("Invalid Rows", invalid_row_table),
        _table_section("Profile Summary", profile_summary),
    ]
    body = "\n".join(sections)
    report_path.write_text(
        "<!doctype html>\n"
        "<html lang=\"en\">\n"
        "<head><meta charset=\"utf-8\"><title>Policy Simulation V1</title></head>\n"
        f"<body>{body}</body>\n"
        "</html>\n",
        encoding="utf-8",
    )


def run_policy_simulation(args: argparse.Namespace, console: Console) -> PolicySimulationRunResult:
    selected_seasons = _parse_season_csv(str(args.seasons))
    selected_models = _parse_required_csv(str(args.models), field_name="models")
    selected_feature_packs = _parse_required_csv(str(args.feature_packs), field_name="feature_packs")
    child_specs = _selected_policy_child_specs(
        experiment_path=Path(args.experiment_path),
        seasons=selected_seasons,
        models=selected_models,
        feature_packs=selected_feature_packs,
    )
    policy_set = get_policy_set(str(args.policy_set))
    policies = _policies_with_no_policy(policy_set.policies)
    simulation_id = _timestamp_id()
    output_path = Path(args.output_root) / f"policy_simulation_started_at={simulation_id}"
    fixture_identity_status = "unverified"
    source_candidate_signature_status = "artifact_backed_unverified"

    console.print(
        "Policy simulation started "
        f"simulation_id={simulation_id} child_count={len(child_specs)} output={output_path}"
    )
    round_rows, selected_player_rows, invalid_rows = _replay_policy_children(
        child_specs=child_specs,
        policies=policies,
        console=console,
        allow_incomplete_report=bool(getattr(args, "allow_incomplete_report", False)),
    )
    if invalid_rows and not bool(getattr(args, "allow_incomplete_report", False)):
        raise PolicySimulationError(
            "Policy simulation produced invalid rows; rerun with --allow-incomplete-report to write diagnostics."
        )

    round_results = pd.DataFrame(round_rows, columns=pd.Index(POLICY_ROUND_RESULT_COLUMNS))
    selected_players = pd.DataFrame(selected_player_rows, columns=pd.Index(POLICY_SELECTED_PLAYER_COLUMNS))
    invalid_row_frame = pd.DataFrame(invalid_rows, columns=pd.Index(POLICY_INVALID_ROW_COLUMNS))
    ranked_summary = build_policy_ranked_summary(
        round_results,
        selected_seasons=selected_seasons,
        fixture_identity_status=fixture_identity_status,
    )
    per_season_summary = build_policy_per_season_summary(round_results)
    profile_summary = build_policy_profile_summary(round_results, selected_players)
    manifest = _policy_simulation_manifest(
        args=args,
        simulation_id=simulation_id,
        output_path=output_path,
        selected_seasons=selected_seasons,
        selected_models=selected_models,
        selected_feature_packs=selected_feature_packs,
        child_specs=child_specs,
        policy_variants=tuple(policy.policy_variant for policy in policies),
        fixture_identity_status=fixture_identity_status,
        source_candidate_signature_status=source_candidate_signature_status,
        invalid_row_count=len(invalid_rows),
    )
    comparability_report = _policy_comparability_report(
        args=args,
        selected_seasons=selected_seasons,
        selected_models=selected_models,
        selected_feature_packs=selected_feature_packs,
        child_count=len(child_specs),
        fixture_identity_status=fixture_identity_status,
        source_candidate_signature_status=source_candidate_signature_status,
    )
    _write_policy_simulation_artifacts(
        output_path=output_path,
        manifest=manifest,
        ranked_summary=ranked_summary,
        per_season_summary=per_season_summary,
        round_results=round_results,
        selected_players=selected_players,
        invalid_rows=invalid_row_frame,
        profile_summary=profile_summary,
        comparability_report=comparability_report,
    )
    console.print(f"Policy simulation complete simulation_id={simulation_id} output={output_path}")
    return PolicySimulationRunResult(output_path=output_path, simulation_id=simulation_id)


def _replay_policy_children(
    *,
    child_specs: tuple[_PolicySimulationChildSpec, ...],
    policies: tuple[OptimizerPolicy, ...],
    console: Console,
    allow_incomplete_report: bool,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    total_children = len(child_specs)
    round_rows: list[dict[str, object]] = []
    selected_player_rows: list[dict[str, object]] = []
    invalid_rows: list[dict[str, object]] = []
    if console.is_terminal and total_children:
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold]Policy simulation"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            TextColumn("{task.fields[current]}"),
            console=console,
            transient=False,
        ) as progress:
            task_id = progress.add_task("policy_simulation", total=total_children, current="")
            for child_index, spec in enumerate(child_specs, start=1):
                progress.update(task_id, current=_child_label(spec))
                child_result = _replay_policy_child(
                    spec=spec,
                    policies=policies,
                    child_index=child_index,
                    total_children=total_children,
                    console=progress.console,
                    allow_incomplete_report=allow_incomplete_report,
                )
                round_rows.extend(child_result.round_rows)
                selected_player_rows.extend(child_result.selected_player_rows)
                invalid_rows.extend(child_result.invalid_rows)
                progress.advance(task_id)
        return round_rows, selected_player_rows, invalid_rows

    for child_index, spec in enumerate(child_specs, start=1):
        child_result = _replay_policy_child(
            spec=spec,
            policies=policies,
            child_index=child_index,
            total_children=total_children,
            console=console,
            allow_incomplete_report=allow_incomplete_report,
        )
        round_rows.extend(child_result.round_rows)
        selected_player_rows.extend(child_result.selected_player_rows)
        invalid_rows.extend(child_result.invalid_rows)
    return round_rows, selected_player_rows, invalid_rows


def _replay_policy_child(
    *,
    spec: _PolicySimulationChildSpec,
    policies: tuple[OptimizerPolicy, ...],
    child_index: int,
    total_children: int,
    console: Console,
    allow_incomplete_report: bool,
) -> PolicyReplayResult:
    label = _child_label(spec)
    console.print(f"START child {child_index}/{total_children} {label}")
    try:
        result = run_policy_replay_for_child(child_path=spec.child_path, policies=policies)
        _verify_no_policy_replay_coverage(spec.child_path, result)
    except PolicySimulationError as exc:
        if not allow_incomplete_report:
            raise
        console.print(f"FAIL child {child_index}/{total_children} {label} error={exc}")
        return PolicyReplayResult(
            round_rows=[],
            selected_player_rows=[],
            invalid_rows=[_invalid_policy_replay_row(spec=spec, error=exc)],
        )
    console.print(
        f"DONE child {child_index}/{total_children} {label} "
        f"round_rows={len(result.round_rows)} selected_player_rows={len(result.selected_player_rows)}"
    )
    return result


def _selected_policy_child_specs(
    *,
    experiment_path: Path,
    seasons: tuple[int, ...],
    models: tuple[str, ...],
    feature_packs: tuple[str, ...],
) -> tuple[_PolicySimulationChildSpec, ...]:
    specs: list[_PolicySimulationChildSpec] = []
    missing_paths: list[Path] = []
    for season in seasons:
        for model_id in models:
            for feature_pack in feature_packs:
                child_path = (
                    experiment_path
                    / "runs"
                    / f"season={season}"
                    / f"model={model_id}"
                    / f"feature_pack={feature_pack}"
                )
                if not child_path.exists():
                    missing_paths.append(child_path)
                    continue
                specs.append(
                    _PolicySimulationChildSpec(
                        child_path=child_path,
                        season=season,
                        model_id=model_id,
                        feature_pack=feature_pack,
                    )
                )
    if missing_paths:
        missing = "\n".join(str(path) for path in missing_paths)
        raise PolicySimulationError(f"Missing requested policy simulation child run paths:\n{missing}")
    if not specs:
        raise PolicySimulationError("No policy simulation child runs were selected.")
    return tuple(specs)


def _parse_season_csv(value: str) -> tuple[int, ...]:
    parsed: list[int] = []
    for part in value.split(","):
        stripped = part.strip()
        if not stripped:
            continue
        try:
            parsed.append(int(stripped))
        except ValueError as exc:
            raise PolicySimulationError(f"Invalid season value: {stripped!r}") from exc
    if not parsed:
        raise PolicySimulationError("At least one season is required.")
    duplicates = _duplicate_values(parsed)
    if duplicates:
        raise PolicySimulationError(f"Duplicate seasons values are not allowed: {_format_duplicate_values(duplicates)}")
    return tuple(parsed)


def _parse_required_csv(value: str, *, field_name: str) -> tuple[str, ...]:
    parsed = tuple(part.strip() for part in value.split(",") if part.strip())
    if not parsed:
        raise PolicySimulationError(f"At least one {field_name} value is required.")
    duplicates = _duplicate_values(parsed)
    if duplicates:
        raise PolicySimulationError(
            f"Duplicate {field_name} values are not allowed: {_format_duplicate_values(duplicates)}"
        )
    return parsed


def _duplicate_values(values: list[int] | tuple[str, ...]) -> list[int | str]:
    seen: set[int | str] = set()
    duplicates: list[int | str] = []
    for value in values:
        if value in seen and value not in duplicates:
            duplicates.append(value)
        seen.add(value)
    return duplicates


def _format_duplicate_values(values: list[int | str]) -> str:
    return ", ".join(str(value) for value in values)


def _policies_with_no_policy(policies: tuple[OptimizerPolicy, ...]) -> tuple[OptimizerPolicy, ...]:
    if any(policy.policy_variant == NO_POLICY.policy_variant for policy in policies):
        return policies
    return (NO_POLICY, *policies)


def _verify_no_policy_replay_coverage(child_path: Path, result: PolicyReplayResult) -> None:
    player_predictions = _read_csv(child_path / "player_predictions.csv")
    target_rounds = set(_target_rounds_from_predictions(player_predictions))
    replayed_no_policy_rounds = {
        int(cast(Any, row["rodada"]))
        for row in result.round_rows
        if str(row["policy_variant"]) == NO_POLICY.policy_variant
    }
    missing_rounds = sorted(target_rounds - replayed_no_policy_rounds)
    if missing_rounds:
        raise PolicySimulationError(
            f"Policy replay missing no_policy rows for child={child_path}: rounds={missing_rounds}"
        )


def _policy_simulation_manifest(
    *,
    args: argparse.Namespace,
    simulation_id: str,
    output_path: Path,
    selected_seasons: tuple[int, ...],
    selected_models: tuple[str, ...],
    selected_feature_packs: tuple[str, ...],
    child_specs: tuple[_PolicySimulationChildSpec, ...],
    policy_variants: tuple[str, ...],
    fixture_identity_status: str,
    source_candidate_signature_status: str,
    invalid_row_count: int,
) -> dict[str, object]:
    return {
        "simulation_id": simulation_id,
        "hypothesis_id": str(args.hypothesis_id),
        "policy_set_id": str(args.policy_set),
        "policy_variants": list(policy_variants),
        "experiment_path": str(Path(args.experiment_path)),
        "output_path": str(output_path),
        "current_year": int(args.current_year),
        "selected_seasons": list(selected_seasons),
        "selected_models": list(selected_models),
        "selected_feature_packs": list(selected_feature_packs),
        "child_count": len(child_specs),
        "children": [_child_manifest_row(spec) for spec in child_specs],
        "fixture_identity_status": fixture_identity_status,
        "budget_policy": "moving",
        "source_candidate_signature_status": source_candidate_signature_status,
        "invalid_row_count": invalid_row_count,
        "allow_incomplete_report": bool(getattr(args, "allow_incomplete_report", False)),
    }


def _child_manifest_row(spec: _PolicySimulationChildSpec) -> dict[str, object]:
    return {
        "season": spec.season,
        "model_id": spec.model_id,
        "feature_pack": spec.feature_pack,
        "child_path": str(spec.child_path),
    }


def _policy_comparability_report(
    *,
    args: argparse.Namespace,
    selected_seasons: tuple[int, ...],
    selected_models: tuple[str, ...],
    selected_feature_packs: tuple[str, ...],
    child_count: int,
    fixture_identity_status: str,
    source_candidate_signature_status: str,
) -> dict[str, object]:
    return {
        "status": "diagnostic_only",
        "reason": "fixture identity is unverified for policy simulation.",
        "hypothesis_id": str(args.hypothesis_id),
        "policy_set_id": str(args.policy_set),
        "experiment_path": str(Path(args.experiment_path)),
        "selected_seasons": list(selected_seasons),
        "selected_models": list(selected_models),
        "selected_feature_packs": list(selected_feature_packs),
        "child_count": child_count,
        "fixture_identity_status": fixture_identity_status,
        "budget_policy": "moving",
        "source_candidate_signature_status": source_candidate_signature_status,
    }


def _write_policy_simulation_artifacts(
    *,
    output_path: Path,
    manifest: dict[str, object],
    ranked_summary: pd.DataFrame,
    per_season_summary: pd.DataFrame,
    round_results: pd.DataFrame,
    selected_players: pd.DataFrame,
    invalid_rows: pd.DataFrame,
    profile_summary: pd.DataFrame,
    comparability_report: dict[str, object],
) -> None:
    output_path.mkdir(parents=True, exist_ok=False)
    _write_json(output_path / "policy_simulation_manifest.json", manifest)
    ranked_summary.to_csv(output_path / "policy_ranked_summary.csv", index=False)
    per_season_summary.to_csv(output_path / "policy_per_season_summary.csv", index=False)
    round_results.to_csv(output_path / "policy_round_results.csv", index=False)
    selected_players.to_csv(output_path / "policy_selected_players.csv", index=False)
    invalid_rows.to_csv(output_path / "policy_invalid_rows.csv", index=False)
    profile_summary.to_csv(output_path / "policy_profile_summary.csv", index=False)
    _write_json(output_path / "policy_comparability_report.json", comparability_report)
    write_policy_simulation_report(
        output_path,
        manifest=manifest,
        ranked_summary=ranked_summary,
        per_season_summary=per_season_summary,
        invalid_rows=invalid_rows,
        profile_summary=profile_summary,
        comparability_report=comparability_report,
    )


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def _child_label(spec: _PolicySimulationChildSpec) -> str:
    return f"season={spec.season} model={spec.model_id} feature_pack={spec.feature_pack}"


def _invalid_policy_replay_row(*, spec: _PolicySimulationChildSpec, error: PolicySimulationError) -> dict[str, object]:
    return {
        "season": spec.season,
        "model_id": spec.model_id,
        "feature_pack": spec.feature_pack,
        "child_path": str(spec.child_path),
        "error_type": type(error).__name__,
        "error_message": str(error),
    }


def _timestamp_id() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")


def _policy_variant_season_summaries(round_results: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    normalized = round_results.copy()
    normalized["season"] = _whole_number_column(normalized, artifact_name="round_results", column="season")
    normalized["rodada"] = _whole_number_column(normalized, artifact_name="round_results", column="rodada")
    for group_key, group in normalized.groupby(list(_POLICY_SEASON_GROUP_COLUMNS), sort=True):
        season, model_id, feature_pack, strategy, policy_variant = cast(
            tuple[object, object, object, object, object],
            group_key,
        )
        budget_summary = _policy_budget_summary(group)
        rows.append(
            {
                "season": int(cast(int, season)),
                "model_id": str(model_id),
                "feature_pack": str(feature_pack),
                "strategy": str(strategy),
                "policy_variant": str(policy_variant),
                "rounds": int(group["rodada"].nunique()),
                "total_actual_points": float(
                    pd.to_numeric(group["actual_points_with_captain"], errors="coerce").fillna(0.0).sum()
                ),
                **budget_summary,
                "non_optimal_rounds": int(group["solver_status"].astype(str).ne("Optimal").sum()),
            }
        )
    return pd.DataFrame(
        rows,
        columns=pd.Index(
            (
                "season",
                *_POLICY_GROUP_COLUMNS,
                "rounds",
                "total_actual_points",
                "final_budget",
                "min_budget",
                "max_budget_drawdown",
                "non_optimal_rounds",
            )
        ),
    )


def _policy_budget_summary(rounds: pd.DataFrame) -> dict[str, float]:
    ordered = rounds.sort_values("rodada", kind="mergesort")
    before = pd.to_numeric(ordered["budget_before_round"], errors="coerce").astype(float)
    after = pd.to_numeric(ordered["budget_after_round"], errors="coerce").astype(float)
    return {
        "final_budget": float(after.iloc[-1]),
        "min_budget": float(pd.concat([before, after], ignore_index=True).min()),
        "max_budget_drawdown": float(_policy_budget_drawdown_from_path(after, initial_budget=float(before.iloc[0])).max()),
    }


def _policy_budget_drawdown_from_path(after: pd.Series, *, initial_budget: float) -> pd.Series:
    peak = initial_budget
    drawdowns: list[float] = []
    for budget_after_round in after:
        peak = max(peak, float(budget_after_round))
        drawdowns.append(peak - float(budget_after_round))
    return pd.Series(drawdowns, index=after.index, dtype=float)


def _has_valid_ranked_benchmark_evidence(group: pd.DataFrame) -> bool:
    for column in _RANKED_BENCHMARK_EVIDENCE_COLUMNS:
        values = pd.to_numeric(group[column], errors="coerce")
        if values.isna().any():
            return False
        if not np.isfinite(values.to_numpy(dtype=float)).all():
            return False
    return True


def _missing_selected_policy_seasons(
    group: pd.DataFrame,
    selected_seasons: tuple[int, ...],
) -> tuple[int, ...]:
    if selected_seasons != _H001_SELECTED_SEASONS:
        return ()
    observed_seasons = set(group["season"].astype(int).tolist())
    return tuple(season for season in selected_seasons if season not in observed_seasons)


def _worst_final_budget_values(group: pd.DataFrame) -> tuple[float, float, float]:
    metrics = group.loc[:, ["season", "final_budget", "benchmark_final_budget", "final_budget_delta"]].copy()
    for column in metrics.columns:
        metrics[column] = pd.to_numeric(metrics[column], errors="coerce")
    metrics = metrics.sort_values(
        ["final_budget_delta", "season"],
        ascending=[True, True],
        kind="mergesort",
    )
    worst = metrics.iloc[0]
    return (
        float(worst["final_budget"]),
        float(worst["benchmark_final_budget"]),
        float(worst["final_budget_delta"]),
    )


def _season_delta(group: pd.DataFrame, *, season: int) -> float | None:
    rows = group.loc[group["season"].astype(int).eq(season), "total_delta"]
    if rows.empty:
        return None
    value = rows.iloc[0]
    if pd.isna(value):
        return None
    return float(value)


def _top_two_positive_delta_concentration(
    round_results: pd.DataFrame,
    *,
    model_id: str,
    feature_pack: str,
    strategy: str,
    policy_variant: str,
) -> float | None:
    context_mask = (
        round_results["model_id"].astype(str).eq(model_id)
        & round_results["feature_pack"].astype(str).eq(feature_pack)
        & round_results["strategy"].astype(str).eq(strategy)
    )
    policy_rounds = round_results.loc[context_mask & round_results["policy_variant"].astype(str).eq(policy_variant)]
    benchmark_rounds = round_results.loc[context_mask & round_results["policy_variant"].astype(str).eq("no_policy")]
    if policy_rounds.empty or benchmark_rounds.empty:
        return None

    join_columns = ["season", "rodada"]
    policy_points = (
        policy_rounds.groupby(join_columns, sort=True)["actual_points_with_captain"].sum().reset_index()
    )
    benchmark_points = (
        benchmark_rounds.groupby(join_columns, sort=True)["actual_points_with_captain"].sum().reset_index()
    )
    merged = policy_points.merge(
        benchmark_points,
        on=join_columns,
        how="inner",
        suffixes=("_policy", "_benchmark"),
    )
    positive_deltas = sorted(
        (
            float(row["actual_points_with_captain_policy"])
            - float(row["actual_points_with_captain_benchmark"])
            for row in merged.to_dict("records")
        ),
        reverse=True,
    )
    positive_deltas = [delta for delta in positive_deltas if delta > 0.0]
    if not positive_deltas:
        return None
    positive_total = sum(positive_deltas)
    return float(sum(positive_deltas[:2]) / positive_total)


def _finite_number_or_zero(value: object) -> float:
    if pd.isna(value):
        return 0.0
    try:
        number = float(cast(Any, value))
    except (TypeError, ValueError):
        return 0.0
    if not np.isfinite(number):
        return 0.0
    return number


def _validate_frame_columns(frame_name: str, frame: pd.DataFrame, required_columns: tuple[str, ...]) -> None:
    missing = [column for column in required_columns if column not in frame.columns]
    if missing:
        raise PolicySimulationError(f"Missing required columns in {frame_name}: {', '.join(missing)}")


def _policy_report_path(output_path: Path) -> Path:
    if output_path.name == "policy_simulation_report.html":
        return output_path
    if output_path.suffix:
        return output_path.parent / "policy_simulation_report.html"
    return output_path / "policy_simulation_report.html"


def _json_section(title: str, payload: dict[str, object]) -> str:
    serialized = json.dumps(payload, indent=2, sort_keys=True, default=str)
    return f"<h2>{html.escape(title)}</h2><pre>{html.escape(serialized)}</pre>"


def _table_section(title: str, frame: pd.DataFrame) -> str:
    table = frame.to_html(index=False, escape=True)
    return f"<h2>{html.escape(title)}</h2>{table}"


def _target_rounds_from_predictions(player_predictions: pd.DataFrame) -> list[int]:
    round_values = _whole_number_column(player_predictions, artifact_name="player_predictions.csv", column="rodada")
    return sorted(round_values.astype(int).unique().tolist())


def _initial_budget_for_policy_replay(*, context: PolicySourceContext, round_results: pd.DataFrame) -> float:
    metadata = _read_metadata(context.child_path)
    metadata_budget = _optional_finite_float(metadata.get("initial_budget"), "source metadata field initial_budget")
    if metadata_budget is not None:
        return metadata_budget

    round_values = _whole_number_column(round_results, artifact_name="round_results.csv", column="rodada")
    source_rows = round_results.loc[round_results["strategy"].astype(str).eq(context.strategy)]
    if source_rows.empty:
        raise PolicySimulationError(
            f"Cannot infer replay initial budget: no round_results rows for strategy={context.strategy!r}."
        )
    ordered_indexes = round_values.loc[source_rows.index].sort_values(kind="mergesort").index
    first_budget = source_rows.loc[ordered_indexes[0], "budget_before_round"]
    return _required_finite_float(first_budget, "first source round_results budget_before_round")


def _direct_fixtures_for_round(child_path: Path, *, round_number: int) -> pd.DataFrame | None:
    for artifact_name in _DIRECT_FIXTURE_ARTIFACTS:
        artifact_path = child_path / artifact_name
        if not artifact_path.exists():
            continue
        fixtures = _read_csv(artifact_path)
        _validate_columns(artifact_path, set(fixtures.columns), _FIXTURE_COLUMNS)
        round_values = _whole_number_column(fixtures, artifact_name=artifact_name, column="rodada")
        return fixtures.loc[round_values.eq(int(round_number)), list(_FIXTURE_COLUMNS)].copy()
    return None


def _source_fixtures_for_round(context: PolicySourceContext, *, round_number: int) -> pd.DataFrame | None:
    fixture_source_directory = context.fixture_source_directory
    if fixture_source_directory is None:
        return None
    if not fixture_source_directory.exists():
        raise PolicySimulationError(f"fixture_source_directory not found: {fixture_source_directory}")
    if not fixture_source_directory.is_dir():
        raise PolicySimulationError(f"fixture_source_directory is not a directory: {fixture_source_directory}")

    fixture_files = sorted(fixture_source_directory.glob("partidas-*.csv"))
    if not fixture_files:
        raise PolicySimulationError(f"No partidas-*.csv fixture files found in {fixture_source_directory}")

    round_frames: list[pd.DataFrame] = []
    for fixture_path in fixture_files:
        fixtures = _read_csv(fixture_path)
        _validate_columns(fixture_path, set(fixtures.columns), _FIXTURE_COLUMNS)
        round_values = _whole_number_column(fixtures, artifact_name=fixture_path.name, column="rodada")
        round_frames.append(fixtures.loc[round_values.eq(int(round_number)), list(_FIXTURE_COLUMNS)].copy())

    if not round_frames:
        return pd.DataFrame(columns=pd.Index(_FIXTURE_COLUMNS))
    return pd.concat(round_frames, ignore_index=True)


def _fixtures_for_policy_round(
    *,
    context: PolicySourceContext,
    candidates: pd.DataFrame,
    policy: OptimizerPolicy,
    round_number: int,
) -> pd.DataFrame | None:
    if not _policy_requires_fixture_coverage(policy):
        return None

    try:
        fixtures_for_round = _direct_fixtures_for_round(context.child_path, round_number=round_number)
        if fixtures_for_round is None:
            fixtures_for_round = _source_fixtures_for_round(context, round_number=round_number)
    except PolicySimulationError as exc:
        raise PolicySimulationError(
            _fixture_coverage_error_message(
                policy_variant=policy.policy_variant,
                round_number=round_number,
                detail=str(exc),
            )
        ) from exc

    if fixtures_for_round is None:
        raise PolicySimulationError(
            _fixture_coverage_error_message(
                policy_variant=policy.policy_variant,
                round_number=round_number,
                detail="no fixture artifact or fixture_source_directory found",
            )
        )
    if fixtures_for_round.empty:
        raise PolicySimulationError(
            _fixture_coverage_error_message(
                policy_variant=policy.policy_variant,
                round_number=round_number,
                detail="no fixture rows found for target round",
            )
        )

    candidate_club_ids = _whole_number_column(
        candidates,
        artifact_name="player_predictions.csv",
        column="id_clube",
    ).astype(int).unique().tolist()
    try:
        validate_fixture_coverage(
            fixtures_for_round,
            candidate_club_ids=candidate_club_ids,
            round_number=round_number,
        )
    except FixtureCoverageError as exc:
        raise PolicySimulationError(
            _fixture_coverage_error_message(
                policy_variant=policy.policy_variant,
                round_number=round_number,
                detail=str(exc),
            )
        ) from exc
    return fixtures_for_round


def _policy_requires_fixture_coverage(policy: OptimizerPolicy) -> bool:
    return policy.overlap_penalty > 0.0 or policy.max_overlap_assets is not None


def _fixture_coverage_error_message(*, policy_variant: str, round_number: int, detail: str) -> str:
    return f"Invalid fixture coverage for policy_variant={policy_variant!r} round={round_number}: {detail}"


def _selected_for_policy_scoring(
    selected: pd.DataFrame,
    *,
    round_number: int,
    policy_variant: str,
) -> pd.DataFrame:
    scored = selected.copy()
    scored["variacao"] = _finite_selected_numeric(
        scored,
        column="variacao",
        round_number=round_number,
        policy_variant=policy_variant,
    )
    scored["pontuacao"] = _selected_actual_score_values(
        scored,
        round_number=round_number,
        policy_variant=policy_variant,
    )
    return scored


def _selected_actual_score_values(
    selected: pd.DataFrame,
    *,
    round_number: int,
    policy_variant: str,
) -> pd.Series:
    if "pontuacao" not in selected.columns:
        raise PolicySimulationError(
            f"Missing selected pontuacao for round={round_number} policy_variant={policy_variant!r}."
        )
    if "entrou_em_campo" not in selected.columns:
        raise PolicySimulationError(
            f"Missing selected entrou_em_campo for round={round_number} policy_variant={policy_variant!r}."
        )

    raw_scores = selected["pontuacao"]
    numeric_scores = pd.to_numeric(raw_scores, errors="coerce")
    dnp_mask = _explicit_false_mask(selected["entrou_em_campo"])
    corrupt_scores = numeric_scores.isna() & raw_scores.notna()
    null_entered_scores = numeric_scores.isna() & ~dnp_mask
    finite_scores = pd.Series(np.isfinite(numeric_scores.to_numpy(dtype=float)), index=selected.index)
    invalid_scores = corrupt_scores | null_entered_scores | (~numeric_scores.isna() & ~finite_scores)
    if bool(invalid_scores.any()):
        invalid_values = raw_scores.loc[invalid_scores].tolist()
        raise PolicySimulationError(
            "Selected pontuacao must be numeric, and null is allowed only for explicit DNP rows "
            f"for round={round_number} policy_variant={policy_variant!r}: {invalid_values}"
        )

    scored = numeric_scores.fillna(0.0).astype(float)
    scored.loc[dnp_mask] = 0.0
    return scored


def _finite_selected_numeric(
    selected: pd.DataFrame,
    *,
    column: str,
    round_number: int,
    policy_variant: str,
) -> pd.Series:
    if column not in selected.columns:
        raise PolicySimulationError(
            f"Missing selected {column} for round={round_number} policy_variant={policy_variant!r}."
        )
    numeric_values = pd.to_numeric(selected[column], errors="coerce")
    finite_values = pd.Series(np.isfinite(numeric_values.to_numpy(dtype=float)), index=selected.index)
    valid_values = numeric_values.notna() & finite_values
    if not bool(valid_values.all()):
        invalid_values = selected.loc[~valid_values, column].tolist()
        raise PolicySimulationError(
            f"Selected {column} must contain finite numeric values for "
            f"round={round_number} policy_variant={policy_variant!r}: {invalid_values}"
        )
    return numeric_values.astype(float)


def _actual_points_with_captain_from_scored_selection(
    selected: pd.DataFrame,
    *,
    round_number: int,
    policy_variant: str,
) -> float:
    captain_mask = _boolean_mask(selected["is_captain"])
    captain_count = int(captain_mask.sum())
    if captain_count != 1:
        raise PolicySimulationError(
            f"Selected squad must contain exactly one captain for round={round_number} "
            f"policy_variant={policy_variant!r}; got {captain_count}."
        )
    actual_scores = _finite_selected_numeric(
        selected,
        column="pontuacao",
        round_number=round_number,
        policy_variant=policy_variant,
    )
    captain_score = float(actual_scores.loc[captain_mask].iloc[0])
    return float(actual_scores.sum()) + (CAPTAIN_MULTIPLIER - 1.0) * captain_score


def _selected_player_output_rows(
    selected: pd.DataFrame,
    *,
    context: PolicySourceContext,
    policy_variant: str,
    round_number: int,
) -> list[dict[str, object]]:
    output = selected.copy()
    output["season"] = context.season
    output["model_id"] = context.model_id
    output["feature_pack"] = context.feature_pack
    output["strategy"] = context.strategy
    output["policy_variant"] = policy_variant
    output["rodada"] = int(round_number)
    output["id_atleta"] = _whole_number_column(output, artifact_name="policy replay selected", column="id_atleta")
    output["id_clube"] = _whole_number_column(output, artifact_name="policy replay selected", column="id_clube")
    output["posicao"] = output["posicao"].astype(str)
    output["preco_pre_rodada"] = _finite_selected_numeric(
        output,
        column="preco_pre_rodada",
        round_number=round_number,
        policy_variant=policy_variant,
    )
    output["pontuacao"] = _finite_selected_numeric(
        output,
        column="pontuacao",
        round_number=round_number,
        policy_variant=policy_variant,
    )
    output["variacao"] = _finite_selected_numeric(
        output,
        column="variacao",
        round_number=round_number,
        policy_variant=policy_variant,
    )
    output["is_captain"] = _boolean_mask(output["is_captain"])
    output["entrou_em_campo"] = _boolean_mask(output["entrou_em_campo"])
    return cast(list[dict[str, object]], output.loc[:, POLICY_SELECTED_PLAYER_COLUMNS].to_dict("records"))


def _policy_replay_round_row(
    *,
    context: PolicySourceContext,
    policy_variant: str,
    round_number: int,
    solver_status: str,
    formation: str,
    captain_id: int | None,
    budget_before_round: float,
    budget_used: float,
    budget_remaining: float,
    budget_delta: float,
    budget_after_round: float,
    predicted_points_with_captain: float,
    actual_points_with_captain: float,
) -> dict[str, object]:
    return {
        "season": context.season,
        "model_id": context.model_id,
        "feature_pack": context.feature_pack,
        "strategy": context.strategy,
        "policy_variant": policy_variant,
        "rodada": int(round_number),
        "solver_status": solver_status,
        "formation": formation,
        "captain_id": captain_id,
        "budget_before_round": float(budget_before_round),
        "budget_used": float(budget_used),
        "budget_remaining": float(budget_remaining),
        "budget_delta": float(budget_delta),
        "budget_after_round": float(budget_after_round),
        "predicted_points_with_captain": float(predicted_points_with_captain),
        "actual_points_with_captain": float(actual_points_with_captain),
    }


def _optional_finite_float(value: object, context: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, str) and not value.strip():
        return None
    return _required_finite_float(value, context)


def _required_finite_float(value: object, context: str) -> float:
    try:
        number = float(cast(Any, value))
    except (TypeError, ValueError) as exc:
        raise PolicySimulationError(f"{context} must be a finite numeric value.") from exc
    if not np.isfinite(number):
        raise PolicySimulationError(f"{context} must be a finite numeric value.")
    return number


def _explicit_false_mask(values: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(values):
        return values.eq(False).fillna(False).astype(bool)
    normalized = values.astype(str).str.strip().str.lower()
    return normalized.isin({"false", "0", "no"})


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
    try:
        return set(pd.read_csv(csv_path, nrows=0).columns)
    except (OSError, UnicodeDecodeError, pd.errors.EmptyDataError, pd.errors.ParserError) as exc:
        raise PolicySimulationError(f"Failed to read CSV header for source artifact {csv_path.name}: {csv_path}") from exc


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


def _metadata_directory(metadata: dict[str, object], key: str, *, child_path: Path) -> Path | None:
    value = _metadata_text(metadata, key)
    if value is None:
        return None
    directory = Path(value).expanduser()
    if directory.is_absolute():
        return directory

    relative_bases = (Path.cwd(), child_path, *child_path.parents)
    for base in relative_bases:
        candidate = base / directory
        if candidate.exists():
            return candidate
    return directory


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


def _canonical_child_path_context(path: Path, *, season: int) -> tuple[str, str]:
    parts = path.parts
    model_segments = [part for part in parts if part.startswith("model=")]
    feature_pack_segments = [part for part in parts if part.startswith("feature_pack=")]
    if len(model_segments) > 1 or len(feature_pack_segments) > 1:
        raise PolicySimulationError(
            "Source child path has ambiguous model= or feature_pack= segments; expected canonical child path."
        )

    expected_season = f"season={season}"
    matches: list[tuple[str, str]] = []
    for index in range(len(parts) - 3):
        if parts[index] != "runs" or parts[index + 1] != expected_season:
            continue
        model_id = _path_segment_value(parts[index + 2], "model")
        feature_pack = _path_segment_value(parts[index + 3], "feature_pack")
        if model_id is not None and feature_pack is not None:
            matches.append((model_id, feature_pack))

    if len(matches) != 1:
        raise PolicySimulationError(
            "Source child path must include canonical child path segments: "
            "runs/season=<year>/model=<model_id>/feature_pack=<feature_pack>."
        )
    return matches[0]


def _path_segment_value(part: str, key: str) -> str | None:
    prefix = f"{key}="
    if not part.startswith(prefix):
        return None
    value = part[len(prefix) :].strip()
    return value or None


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
    except (OSError, UnicodeDecodeError, pd.errors.EmptyDataError, pd.errors.ParserError) as exc:
        raise PolicySimulationError(f"Failed to read source artifact {csv_path.name}: {csv_path}") from exc


def _source_round_result(
    round_results: pd.DataFrame,
    *,
    context: PolicySourceContext,
    round_number: int,
) -> pd.Series:
    round_values = _whole_number_column(round_results, artifact_name="round_results.csv", column="rodada")
    rows = round_results.loc[
        round_values.eq(int(round_number))
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
    round_values = _whole_number_column(selected_players, artifact_name="selected_players.csv", column="rodada")
    source_selected = selected_players.loc[
        round_values.eq(int(round_number))
        & selected_players["strategy"].astype(str).eq(context.strategy)
    ].copy()
    _validate_source_selected_players(source_selected, context=context, round_number=round_number)
    return source_selected


def _round_candidates(
    player_predictions: pd.DataFrame,
    *,
    context: PolicySourceContext,
    round_number: int,
) -> pd.DataFrame:
    round_values = _whole_number_column(player_predictions, artifact_name="player_predictions.csv", column="rodada")
    candidates = player_predictions.loc[round_values.eq(int(round_number))].copy()
    if candidates.empty:
        raise PolicySimulationError(f"Missing player_predictions rows for round={round_number}.")
    if context.score_column not in candidates.columns:
        raise PolicySimulationError(f"Missing score column in player_predictions.csv: {context.score_column}")
    return candidates


def _validate_source_selected_players(
    source_selected: pd.DataFrame,
    *,
    context: PolicySourceContext,
    round_number: int,
) -> None:
    selected_ids = _whole_number_column(source_selected, artifact_name="selected_players.csv", column="id_atleta")
    duplicated_ids = sorted(selected_ids.loc[selected_ids.duplicated()].astype(int).unique().tolist())
    if duplicated_ids:
        raise PolicySimulationError(
            "Source selected_players.csv has duplicate id_atleta rows for "
            f"round={round_number} strategy={context.strategy!r}: {duplicated_ids}"
        )

    source_captain_count = int(_boolean_mask(source_selected["is_captain"]).sum())
    if source_captain_count != 1:
        raise PolicySimulationError(
            "Source selected_players.csv must contain exactly one source captain for "
            f"round={round_number} strategy={context.strategy!r}; got {source_captain_count}."
        )


def _whole_number_column(frame: pd.DataFrame, *, artifact_name: str, column: str) -> pd.Series:
    if column not in frame.columns:
        raise PolicySimulationError(f"Missing required column in {artifact_name}: {column}")
    numeric = pd.to_numeric(frame[column], errors="coerce")
    valid_values = numeric.notna() & numeric.mod(1).eq(0)
    if not bool(valid_values.all()):
        invalid_values = frame.loc[~valid_values, column].tolist()
        raise PolicySimulationError(
            f"{artifact_name} column {column} must contain non-null whole-number values: {invalid_values}"
        )
    return numeric.astype(int)


def _boolean_mask(values: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(values):
        return values.fillna(False).astype(bool)
    normalized = values.astype(str).str.strip().str.lower()
    return normalized.isin({"true", "1", "yes"})


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
    solver_status_match: bool,
    selected_ids_match: bool,
    captain_id_match: bool,
    formation_match: bool,
    budget_used_delta: float,
    predicted_points_delta: float,
    actual_points_delta: float,
) -> list[str]:
    reasons: list[str] = []
    if not solver_status_match:
        reasons.append("solver_status mismatch")
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
