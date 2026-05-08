from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd
import pulp

from cartola.backtesting.config import DEFAULT_FORMATIONS, MARKET_OPEN_PRICE_COLUMN, BacktestConfig
from cartola.backtesting.optimizer_policies import (
    NO_POLICY,
    OpponentOverlapCounts,
    OptimizerPolicy,
    count_opponent_overlap,
)
from cartola.backtesting.scoring_contract import CAPTAIN_MULTIPLIER, SCORING_CONTRACT_VERSION

_PRIMARY_OBJECTIVE_TOLERANCE = 1e-6
_BINARY_SELECTION_THRESHOLD = 0.5
_FIXTURE_COLUMNS = ("rodada", "id_clube_home", "id_clube_away")
_CLEAN_SHEET_CONTEXT_COLUMNS = ("matchup_is_home", "footystats_ppg_diff", "footystats_xg_diff")
_CLEAN_SHEET_CONTEXT_TOLERANCE = 1e-6


@dataclass(frozen=True)
class SquadOptimizationResult:
    selected: pd.DataFrame
    status: str
    budget_used: float
    predicted_points: float
    predicted_points_base: float
    captain_bonus_predicted: float
    predicted_points_with_captain: float
    formation_name: str
    selected_count: int
    captain_id: int | None
    captain_name: str | None
    captain_position: str | None
    captain_club: str | None
    captain_predicted_points: float | None
    captain_multiplier: float
    scoring_contract_version: str
    formation_scores: list[dict[str, object]]
    captain_policy_diagnostics: list[dict[str, object]]
    infeasibility_reason: str | None = None
    opponent_overlap_asset_count: int = 0
    opponent_overlap_match_count: int = 0
    gk_opponent_attack_pair_count: int = 0
    gk_opponent_captain_count: int = 0
    clean_sheet_pair_count: int = 0
    clean_sheet_pair_bonus_applied: float = 0.0
    policy_variant: str = "no_policy"


@dataclass(frozen=True)
class _PolicyTerms:
    overlap_asset_count: pulp.LpAffineExpression
    overlap_match_count: pulp.LpAffineExpression
    gk_opponent_attack_pair_count: pulp.LpAffineExpression
    gk_opponent_captain_count: pulp.LpAffineExpression
    clean_sheet_pair_count: pulp.LpAffineExpression


def optimize_squad(
    candidates: pd.DataFrame,
    score_column: str,
    config: BacktestConfig,
    *,
    budget: float | None = None,
    policy: OptimizerPolicy | None = None,
    fixtures_for_round: pd.DataFrame | None = None,
) -> SquadOptimizationResult:
    active_policy = NO_POLICY if policy is None else policy
    if candidates.empty:
        return _empty_result("Empty", "", candidates, formation_scores=[], policy_variant=active_policy.policy_variant)

    budget_limit = float(config.budget if budget is None else budget)
    results = [
        _optimize_formation(
            candidates,
            score_column=score_column,
            config=config,
            formation_name=formation_name,
            budget=budget_limit,
            active_policy=active_policy,
            fixtures_for_round=fixtures_for_round,
        )
        for formation_name in DEFAULT_FORMATIONS
    ]
    formation_scores = [_formation_score(result) for result in results]
    optimal_results = [result for result in results if result.status == "Optimal"]
    if not optimal_results:
        return _empty_result(
            "Infeasible",
            "",
            candidates,
            formation_scores=formation_scores,
            policy_variant=active_policy.policy_variant,
        )

    best = min(
        optimal_results,
        key=lambda result: (
            -_policy_adjusted_objective_value(result, active_policy),
            result.formation_name,
            tuple(sorted(result.selected["id_atleta"].astype(int).tolist())),
            result.captain_id if result.captain_id is not None else -1,
        ),
    )
    return _with_formation_scores(best, formation_scores)


def _optimize_formation(
    candidates: pd.DataFrame,
    *,
    score_column: str,
    config: BacktestConfig,
    formation_name: str,
    budget: float,
    active_policy: OptimizerPolicy,
    fixtures_for_round: pd.DataFrame | None,
) -> SquadOptimizationResult:
    required_columns = {"id_atleta", "apelido", "posicao", MARKET_OPEN_PRICE_COLUMN, score_column}
    missing_columns = sorted(required_columns - set(candidates.columns))
    if missing_columns:
        raise ValueError(f"Missing optimizer candidate columns: {', '.join(missing_columns)}")

    formation = DEFAULT_FORMATIONS[formation_name]
    player_rows = candidates.loc[~candidates["id_atleta"].duplicated()].copy()
    player_rows[MARKET_OPEN_PRICE_COLUMN] = _numeric_column(player_rows, MARKET_OPEN_PRICE_COLUMN)
    player_rows[score_column] = _numeric_column(player_rows, score_column)
    player_rows = player_rows.sort_values("id_atleta", kind="mergesort").reset_index(drop=True)

    variables = {
        index: pulp.LpVariable(f"player_{index}_{player_rows.loc[index, 'id_atleta']}", cat=pulp.LpBinary)
        for index in player_rows.index
    }
    captain_variables = {
        index: pulp.LpVariable(f"captain_{index}_{player_rows.loc[index, 'id_atleta']}", cat=pulp.LpBinary)
        for index in player_rows.index
        if player_rows.loc[index, "posicao"] != "tec"
    }

    problem = pulp.LpProblem(f"CartolaSquadOptimizer_{formation_name}", pulp.LpMaximize)
    primary_objective = pulp.lpSum(
        float(player_rows.loc[index, score_column]) * variable for index, variable in variables.items()
    ) + (CAPTAIN_MULTIPLIER - 1.0) * pulp.lpSum(
        float(player_rows.loc[index, score_column]) * captain_variable
        for index, captain_variable in captain_variables.items()
    )
    policy_terms = _build_policy_terms(
        problem=problem,
        player_rows=player_rows,
        selected_variables=variables,
        captain_variables=captain_variables,
        policy=active_policy,
        fixtures_for_round=fixtures_for_round,
        formation=formation,
    )
    policy_objective = (
        primary_objective
        - active_policy.overlap_penalty * policy_terms.overlap_asset_count
        - active_policy.gk_opponent_attack_penalty * policy_terms.gk_opponent_attack_pair_count
        - active_policy.gk_opponent_captain_penalty * policy_terms.gk_opponent_captain_count
        + active_policy.clean_sheet_pair_bonus * policy_terms.clean_sheet_pair_count
    )
    problem += policy_objective
    problem += pulp.lpSum(
        float(player_rows.loc[index, MARKET_OPEN_PRICE_COLUMN]) * variable
        for index, variable in variables.items()
    ) <= budget
    problem += pulp.lpSum(variables.values()) == sum(formation.values())

    for position, required_count in formation.items():
        problem += (
            pulp.lpSum(
                variable for index, variable in variables.items() if player_rows.loc[index, "posicao"] == position
            )
            == required_count
        )

    problem += pulp.lpSum(captain_variables.values()) == 1
    for index, captain_variable in captain_variables.items():
        problem += captain_variable <= variables[index]

    if active_policy.max_overlap_assets is not None and _has_fixture_policy_terms(active_policy):
        problem += policy_terms.overlap_asset_count <= active_policy.max_overlap_assets
    if active_policy.max_gk_opponent_attack_pairs is not None and _has_fixture_policy_terms(active_policy):
        problem += policy_terms.gk_opponent_attack_pair_count <= active_policy.max_gk_opponent_attack_pairs
    if active_policy.max_clean_sheet_pair_bonuses is not None and _has_clean_sheet_pair_terms(active_policy):
        problem += policy_terms.clean_sheet_pair_count <= active_policy.max_clean_sheet_pair_bonuses

    status_code = problem.solve(pulp.PULP_CBC_CMD(msg=False))
    status = pulp.LpStatus[status_code]
    if status == "Optimal":
        primary_optimum = float(pulp.value(policy_objective))
        problem += policy_objective >= primary_optimum - _PRIMARY_OBJECTIVE_TOLERANCE
        problem.setObjective(_tie_break_objective(player_rows, variables, captain_variables))
        status_code = problem.solve(pulp.PULP_CBC_CMD(msg=False))
        status = pulp.LpStatus[status_code]
    if status != "Optimal":
        return _empty_result(
            status,
            formation_name,
            candidates,
            formation_scores=[],
            infeasibility_reason="No feasible squad satisfies formation, budget, and captain constraints.",
            policy_variant=active_policy.policy_variant,
        )

    selected_indexes = [index for index, variable in variables.items() if _is_binary_selected(variable)]
    selected = player_rows.loc[selected_indexes].copy().sort_values("id_atleta", kind="mergesort").reset_index(drop=True)
    captain_indexes = {index for index, variable in captain_variables.items() if _is_binary_selected(variable)}
    captain_ids = set(player_rows.loc[list(captain_indexes), "id_atleta"].tolist())
    selected["is_captain"] = selected["id_atleta"].isin(captain_ids)
    selected["captain_policy_ev"] = False
    selected["captain_policy_safe"] = False
    selected["captain_policy_upside"] = False

    budget_used = float(selected[MARKET_OPEN_PRICE_COLUMN].sum())
    predicted_points_base = float(selected[score_column].sum())
    captain = selected.loc[selected["is_captain"]].iloc[0]
    captain_predicted_points = float(captain[score_column])
    captain_bonus_predicted = float((CAPTAIN_MULTIPLIER - 1.0) * captain_predicted_points)
    predicted_points_with_captain = predicted_points_base + captain_bonus_predicted
    overlap_counts = _result_overlap_counts(selected, fixtures_for_round, active_policy)
    gk_opponent_attack_pair_count = _count_gk_opponent_attack_pairs(
        selected,
        fixtures_for_round,
        attack_positions=active_policy.gk_opponent_attack_positions,
    )
    gk_opponent_captain_count = _count_gk_opponent_captain_conflicts(
        selected,
        fixtures_for_round,
        captain_positions=active_policy.gk_opponent_captain_positions,
    )
    clean_sheet_pair_count = int(round(float(pulp.value(policy_terms.clean_sheet_pair_count) or 0.0)))
    clean_sheet_pair_bonus_applied = float(active_policy.clean_sheet_pair_bonus * clean_sheet_pair_count)

    return SquadOptimizationResult(
        selected=selected,
        status=status,
        budget_used=budget_used,
        predicted_points=predicted_points_with_captain,
        predicted_points_base=predicted_points_base,
        captain_bonus_predicted=captain_bonus_predicted,
        predicted_points_with_captain=predicted_points_with_captain,
        formation_name=formation_name,
        selected_count=len(selected),
        captain_id=_optional_int(captain["id_atleta"]),
        captain_name=str(captain["apelido"]),
        captain_position=str(captain["posicao"]),
        captain_club=str(captain["clube"]) if "clube" in selected.columns and pd.notna(captain["clube"]) else None,
        captain_predicted_points=captain_predicted_points,
        captain_multiplier=CAPTAIN_MULTIPLIER,
        scoring_contract_version=SCORING_CONTRACT_VERSION,
        formation_scores=[],
        captain_policy_diagnostics=[],
        infeasibility_reason=None,
        opponent_overlap_asset_count=overlap_counts.opponent_overlap_asset_count,
        opponent_overlap_match_count=overlap_counts.opponent_overlap_match_count,
        gk_opponent_attack_pair_count=gk_opponent_attack_pair_count,
        gk_opponent_captain_count=gk_opponent_captain_count,
        clean_sheet_pair_count=clean_sheet_pair_count,
        clean_sheet_pair_bonus_applied=clean_sheet_pair_bonus_applied,
        policy_variant=active_policy.policy_variant,
    )


def _has_fixture_policy_terms(policy: OptimizerPolicy) -> bool:
    return (
        policy.overlap_penalty > 0.0
        or policy.max_overlap_assets is not None
        or policy.gk_opponent_attack_penalty > 0.0
        or policy.max_gk_opponent_attack_pairs is not None
        or policy.gk_opponent_captain_penalty > 0.0
    )


def _has_clean_sheet_pair_terms(policy: OptimizerPolicy) -> bool:
    return policy.clean_sheet_pair_bonus > 0.0 or policy.max_clean_sheet_pair_bonuses is not None


def _build_policy_terms(
    *,
    problem: pulp.LpProblem,
    player_rows: pd.DataFrame,
    selected_variables: dict[int, pulp.LpVariable],
    captain_variables: dict[int, pulp.LpVariable],
    policy: OptimizerPolicy,
    fixtures_for_round: pd.DataFrame | None,
    formation: Mapping[str, int],
) -> _PolicyTerms:
    if policy.policy_variant == NO_POLICY.policy_variant:
        return _zero_policy_terms()

    clean_sheet_pair_variables = _clean_sheet_pair_variables(
        problem=problem,
        player_rows=player_rows,
        selected_variables=selected_variables,
        policy=policy,
        formation=formation,
    )

    if not _has_fixture_policy_terms(policy):
        return _PolicyTerms(
            overlap_asset_count=pulp.lpSum([]),
            overlap_match_count=pulp.lpSum([]),
            gk_opponent_attack_pair_count=pulp.lpSum([]),
            gk_opponent_captain_count=pulp.lpSum([]),
            clean_sheet_pair_count=pulp.lpSum(clean_sheet_pair_variables),
        )

    if "id_clube" not in player_rows.columns:
        raise ValueError("Missing optimizer policy candidate columns: id_clube")
    if fixtures_for_round is None:
        raise ValueError(f"Policy {policy.policy_variant!r} requires fixtures_for_round.")

    missing_fixture_columns = [column for column in _FIXTURE_COLUMNS if column not in fixtures_for_round.columns]
    if missing_fixture_columns:
        raise ValueError(f"Missing optimizer policy fixture columns: {', '.join(missing_fixture_columns)}")

    formation_size = sum(formation.values())
    club_ids = _whole_number_series(player_rows["id_clube"], "id_clube").astype(int)
    fixture_rows = fixtures_for_round.loc[:, list(_FIXTURE_COLUMNS)].copy()
    for column in _FIXTURE_COLUMNS:
        fixture_rows[column] = _whole_number_series(fixture_rows[column], column).astype(int)

    overlap_variables: list[pulp.LpVariable] = []
    both_side_variables: list[pulp.LpVariable] = []
    gk_attack_pair_variables: list[pulp.LpVariable] = []
    gk_captain_variables: list[pulp.LpVariable] = []
    positions = player_rows["posicao"].astype(str)
    for fixture_position, fixture in enumerate(fixture_rows.to_dict("records")):
        home_club_id = int(fixture["id_clube_home"])
        away_club_id = int(fixture["id_clube_away"])
        home_count = pulp.lpSum(
            selected_variables[index] for index in selected_variables if int(club_ids.loc[index]) == home_club_id
        )
        away_count = pulp.lpSum(
            selected_variables[index] for index in selected_variables if int(club_ids.loc[index]) == away_club_id
        )
        home_present = pulp.LpVariable(f"policy_home_present_{fixture_position}", cat=pulp.LpBinary)
        away_present = pulp.LpVariable(f"policy_away_present_{fixture_position}", cat=pulp.LpBinary)
        both_sides_selected = pulp.LpVariable(
            f"policy_both_sides_selected_{fixture_position}",
            cat=pulp.LpBinary,
        )

        problem += home_count >= home_present
        problem += home_count <= formation_size * home_present
        problem += away_count >= away_present
        problem += away_count <= formation_size * away_present
        problem += both_sides_selected <= home_present
        problem += both_sides_selected <= away_present
        problem += both_sides_selected >= home_present + away_present - 1
        both_side_variables.append(both_sides_selected)

        for index, selected_variable in selected_variables.items():
            if int(club_ids.loc[index]) not in {home_club_id, away_club_id}:
                continue
            overlap_variable = pulp.LpVariable(
                f"policy_overlap_{fixture_position}_{index}",
                cat=pulp.LpBinary,
            )
            problem += overlap_variable <= selected_variable
            problem += overlap_variable <= both_sides_selected
            problem += overlap_variable >= selected_variable + both_sides_selected - 1
            overlap_variables.append(overlap_variable)

        gk_attack_pair_variables.extend(
            _gk_opponent_pair_variables(
                problem=problem,
                gk_variables=selected_variables,
                attack_variables=selected_variables,
                positions=positions,
                club_ids=club_ids,
                home_club_id=home_club_id,
                away_club_id=away_club_id,
                fixture_position=fixture_position,
                attack_positions=policy.gk_opponent_attack_positions,
                variable_prefix="policy_gk_opponent_attack",
            )
        )
        gk_captain_variables.extend(
            _gk_opponent_pair_variables(
                problem=problem,
                gk_variables=selected_variables,
                attack_variables=captain_variables,
                positions=positions,
                club_ids=club_ids,
                home_club_id=home_club_id,
                away_club_id=away_club_id,
                fixture_position=fixture_position,
                attack_positions=policy.gk_opponent_captain_positions,
                variable_prefix="policy_gk_opponent_captain",
            )
        )

    return _PolicyTerms(
        overlap_asset_count=pulp.lpSum(overlap_variables),
        overlap_match_count=pulp.lpSum(both_side_variables),
        gk_opponent_attack_pair_count=pulp.lpSum(gk_attack_pair_variables),
        gk_opponent_captain_count=pulp.lpSum(gk_captain_variables),
        clean_sheet_pair_count=pulp.lpSum(clean_sheet_pair_variables),
    )


def _zero_policy_terms() -> _PolicyTerms:
    return _PolicyTerms(
        overlap_asset_count=pulp.lpSum([]),
        overlap_match_count=pulp.lpSum([]),
        gk_opponent_attack_pair_count=pulp.lpSum([]),
        gk_opponent_captain_count=pulp.lpSum([]),
        clean_sheet_pair_count=pulp.lpSum([]),
    )


def _gk_opponent_pair_variables(
    *,
    problem: pulp.LpProblem,
    gk_variables: dict[int, pulp.LpVariable],
    attack_variables: dict[int, pulp.LpVariable],
    positions: pd.Series,
    club_ids: pd.Series,
    home_club_id: int,
    away_club_id: int,
    fixture_position: int,
    attack_positions: tuple[str, ...],
    variable_prefix: str,
) -> list[pulp.LpVariable]:
    if not attack_positions:
        return []

    attack_position_set = set(attack_positions)
    home_gk_indexes = [
        index
        for index in gk_variables
        if int(club_ids.loc[index]) == home_club_id and str(positions.loc[index]) == "gol"
    ]
    away_gk_indexes = [
        index
        for index in gk_variables
        if int(club_ids.loc[index]) == away_club_id and str(positions.loc[index]) == "gol"
    ]
    home_attack_indexes = [
        index
        for index in attack_variables
        if int(club_ids.loc[index]) == home_club_id and str(positions.loc[index]) in attack_position_set
    ]
    away_attack_indexes = [
        index
        for index in attack_variables
        if int(club_ids.loc[index]) == away_club_id and str(positions.loc[index]) in attack_position_set
    ]

    conflict_variables: list[pulp.LpVariable] = []
    for gk_index, attacker_index in [
        *((gk_index, attacker_index) for gk_index in home_gk_indexes for attacker_index in away_attack_indexes),
        *((gk_index, attacker_index) for gk_index in away_gk_indexes for attacker_index in home_attack_indexes),
    ]:
        conflict_variable = pulp.LpVariable(
            f"{variable_prefix}_{fixture_position}_{gk_index}_{attacker_index}",
            cat=pulp.LpBinary,
        )
        problem += conflict_variable <= gk_variables[gk_index]
        problem += conflict_variable <= attack_variables[attacker_index]
        problem += conflict_variable >= gk_variables[gk_index] + attack_variables[attacker_index] - 1
        conflict_variables.append(conflict_variable)
    return conflict_variables


def _clean_sheet_pair_variables(
    *,
    problem: pulp.LpProblem,
    player_rows: pd.DataFrame,
    selected_variables: dict[int, pulp.LpVariable],
    policy: OptimizerPolicy,
    formation: Mapping[str, int],
) -> list[pulp.LpVariable]:
    if not _has_clean_sheet_pair_terms(policy):
        return []
    eligible_club_ids = _eligible_clean_sheet_pair_club_ids(player_rows, policy=policy)
    if not eligible_club_ids:
        return []

    club_ids = _whole_number_series(player_rows["id_clube"], "id_clube").astype(int)
    positions = player_rows["posicao"].astype(str)
    partner_position_set = set(policy.clean_sheet_pair_partner_positions)
    max_anchor_count = max(1, int(formation.get(policy.clean_sheet_pair_anchor_position, 1)))
    max_partner_count = max(
        1,
        sum(int(count) for position, count in formation.items() if position in partner_position_set),
    )
    pair_variables: list[pulp.LpVariable] = []
    for club_id in eligible_club_ids:
        anchor_count = pulp.lpSum(
            selected_variables[index]
            for index in selected_variables
            if int(club_ids.loc[index]) == club_id and str(positions.loc[index]) == policy.clean_sheet_pair_anchor_position
        )
        partner_count = pulp.lpSum(
            selected_variables[index]
            for index in selected_variables
            if int(club_ids.loc[index]) == club_id and str(positions.loc[index]) in partner_position_set
        )
        anchor_present = pulp.LpVariable(f"policy_clean_sheet_anchor_present_{club_id}", cat=pulp.LpBinary)
        partner_present = pulp.LpVariable(f"policy_clean_sheet_partner_present_{club_id}", cat=pulp.LpBinary)
        pair_selected = pulp.LpVariable(f"policy_clean_sheet_pair_{club_id}", cat=pulp.LpBinary)
        problem += anchor_count >= anchor_present
        problem += anchor_count <= max_anchor_count * anchor_present
        problem += partner_count >= partner_present
        problem += partner_count <= max_partner_count * partner_present
        problem += pair_selected <= anchor_present
        problem += pair_selected <= partner_present
        problem += pair_selected >= anchor_present + partner_present - 1
        pair_variables.append(pair_selected)
    return pair_variables


def _eligible_clean_sheet_pair_club_ids(player_rows: pd.DataFrame, *, policy: OptimizerPolicy) -> list[int]:
    missing_columns = [
        column for column in ("id_clube", *_CLEAN_SHEET_CONTEXT_COLUMNS) if column not in player_rows.columns
    ]
    if missing_columns:
        raise ValueError(f"Missing clean-sheet policy candidate columns: {', '.join(missing_columns)}")

    rows = player_rows.copy()
    rows["id_clube"] = _whole_number_series(rows["id_clube"], "id_clube").astype(int)
    rows["matchup_is_home"] = _whole_number_series(rows["matchup_is_home"], "matchup_is_home").astype(int)
    for column in ("footystats_ppg_diff", "footystats_xg_diff"):
        rows[column] = _numeric_column(rows, column)

    eligible_club_ids: list[int] = []
    for _, club_rows in rows.groupby("id_clube", sort=True):
        club_id_int = int(club_rows["id_clube"].iloc[0])
        home_values = sorted(club_rows["matchup_is_home"].astype(int).unique().tolist())
        if len(home_values) != 1:
            raise ValueError(f"Conflicting clean-sheet context for id_clube={club_id_int}: matchup_is_home")
        home_value = int(home_values[0])
        if home_value not in {0, 1}:
            raise ValueError(f"Clean-sheet context matchup_is_home must be 0 or 1 for id_clube={club_id_int}")
        ppg_span = float(club_rows["footystats_ppg_diff"].max() - club_rows["footystats_ppg_diff"].min())
        xg_span = float(club_rows["footystats_xg_diff"].max() - club_rows["footystats_xg_diff"].min())
        if ppg_span > _CLEAN_SHEET_CONTEXT_TOLERANCE:
            raise ValueError(f"Conflicting clean-sheet context for id_clube={club_id_int}: footystats_ppg_diff")
        if xg_span > _CLEAN_SHEET_CONTEXT_TOLERANCE:
            raise ValueError(f"Conflicting clean-sheet context for id_clube={club_id_int}: footystats_xg_diff")

        is_home = home_value == 1
        ppg_diff = float(club_rows["footystats_ppg_diff"].iloc[0])
        xg_diff = float(club_rows["footystats_xg_diff"].iloc[0])
        if policy.clean_sheet_pair_home_only and not is_home:
            continue
        if ppg_diff < policy.clean_sheet_pair_min_ppg_diff:
            continue
        if xg_diff < policy.clean_sheet_pair_min_xg_diff:
            continue
        eligible_club_ids.append(club_id_int)
    return eligible_club_ids


def _whole_number_series(values: pd.Series, column: str) -> pd.Series:
    numeric_values = pd.to_numeric(values, errors="coerce")
    valid_values = numeric_values.notna() & numeric_values.mod(1).eq(0)
    if not bool(valid_values.all()):
        invalid_values = values.loc[~valid_values].tolist()
        raise ValueError(
            f"Optimizer policy column {column!r} must contain non-null whole-number values: {invalid_values}"
        )
    return numeric_values


def _result_overlap_counts(
    selected: pd.DataFrame,
    fixtures_for_round: pd.DataFrame | None,
    active_policy: OptimizerPolicy,
) -> OpponentOverlapCounts:
    if fixtures_for_round is None or selected.empty or "id_clube" not in selected.columns:
        return count_opponent_overlap(selected.iloc[0:0], None)
    try:
        return count_opponent_overlap(selected, fixtures_for_round)
    except ValueError:
        if active_policy.policy_variant == NO_POLICY.policy_variant:
            return count_opponent_overlap(selected.iloc[0:0], None)
        raise


def _count_gk_opponent_attack_pairs(
    selected: pd.DataFrame,
    fixtures_for_round: pd.DataFrame | None,
    *,
    attack_positions: tuple[str, ...],
) -> int:
    return _count_gk_opponent_conflicts(
        selected,
        fixtures_for_round,
        attack_positions=attack_positions,
        captain_only=False,
    )


def _count_gk_opponent_captain_conflicts(
    selected: pd.DataFrame,
    fixtures_for_round: pd.DataFrame | None,
    *,
    captain_positions: tuple[str, ...],
) -> int:
    return _count_gk_opponent_conflicts(
        selected,
        fixtures_for_round,
        attack_positions=captain_positions,
        captain_only=True,
    )


def _count_gk_opponent_conflicts(
    selected: pd.DataFrame,
    fixtures_for_round: pd.DataFrame | None,
    *,
    attack_positions: tuple[str, ...],
    captain_only: bool,
) -> int:
    if fixtures_for_round is None or selected.empty or not attack_positions:
        return 0
    required_columns = {"id_clube", "posicao"}
    if captain_only:
        required_columns.add("is_captain")
    if not required_columns.issubset(selected.columns):
        return 0
    missing_fixture_columns = [column for column in _FIXTURE_COLUMNS if column not in fixtures_for_round.columns]
    if missing_fixture_columns:
        return 0

    selected = selected.copy()
    selected["id_clube"] = _whole_number_series(selected["id_clube"], "id_clube").astype(int)
    selected["posicao"] = selected["posicao"].astype(str)
    fixture_rows = fixtures_for_round.loc[:, list(_FIXTURE_COLUMNS)].copy()
    for column in _FIXTURE_COLUMNS:
        fixture_rows[column] = _whole_number_series(fixture_rows[column], column).astype(int)

    attack_position_set = set(attack_positions)
    conflict_count = 0
    for fixture in fixture_rows.to_dict("records"):
        home_club_id = int(fixture["id_clube_home"])
        away_club_id = int(fixture["id_clube_away"])
        home_gk_count = _selected_position_count(selected, club_id=home_club_id, positions={"gol"})
        away_gk_count = _selected_position_count(selected, club_id=away_club_id, positions={"gol"})
        home_attack_count = _selected_position_count(
            selected,
            club_id=home_club_id,
            positions=attack_position_set,
            captain_only=captain_only,
        )
        away_attack_count = _selected_position_count(
            selected,
            club_id=away_club_id,
            positions=attack_position_set,
            captain_only=captain_only,
        )
        conflict_count += home_gk_count * away_attack_count
        conflict_count += away_gk_count * home_attack_count
    return int(conflict_count)


def _selected_position_count(
    selected: pd.DataFrame,
    *,
    club_id: int,
    positions: set[str],
    captain_only: bool = False,
) -> int:
    mask = selected["id_clube"].eq(club_id) & selected["posicao"].isin(positions)
    if captain_only:
        mask &= selected["is_captain"].astype(bool)
    return int(mask.sum())


def _policy_adjusted_objective_value(result: SquadOptimizationResult, active_policy: OptimizerPolicy) -> float:
    return (
        float(result.predicted_points_with_captain)
        - active_policy.overlap_penalty * float(result.opponent_overlap_asset_count)
        - active_policy.gk_opponent_attack_penalty * float(result.gk_opponent_attack_pair_count)
        - active_policy.gk_opponent_captain_penalty * float(result.gk_opponent_captain_count)
        + active_policy.clean_sheet_pair_bonus * float(result.clean_sheet_pair_count)
    )


def _tie_break_objective(
    player_rows: pd.DataFrame,
    variables: dict[int, pulp.LpVariable],
    captain_variables: dict[int, pulp.LpVariable],
) -> pulp.LpAffineExpression:
    # The primary EV objective is constrained to its optimum before this objective is used.
    # Maximizing negative IDs gives CBC a stable preference among otherwise equal squads.
    captain_weight = 1.0 / (float(player_rows["id_atleta"].abs().max()) + 1.0)
    selected_id_penalty = pulp.lpSum(
        -float(player_rows.loc[index, "id_atleta"]) * variable for index, variable in variables.items()
    )
    captain_id_penalty = pulp.lpSum(
        -captain_weight * float(player_rows.loc[index, "id_atleta"]) * captain_variable
        for index, captain_variable in captain_variables.items()
    )
    return selected_id_penalty + captain_id_penalty


def _is_binary_selected(variable: pulp.LpVariable) -> bool:
    value = pulp.value(variable)
    return value is not None and float(value) > _BINARY_SELECTION_THRESHOLD


def _numeric_column(frame: pd.DataFrame, column: str) -> pd.Series:
    try:
        values = pd.to_numeric(frame[column], errors="raise")
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Optimizer column {column!r} must be numeric.") from exc
    if values.isna().any() or not np.isfinite(values.astype(float)).all():
        raise ValueError(f"Optimizer column {column!r} must contain finite numeric values.")
    return values.astype(float)


def _empty_result(
    status: str,
    formation_name: str,
    candidates: pd.DataFrame,
    *,
    formation_scores: list[dict[str, object]],
    infeasibility_reason: str | None = None,
    policy_variant: str = NO_POLICY.policy_variant,
) -> SquadOptimizationResult:
    selected = candidates.iloc[0:0].copy()
    selected["is_captain"] = pd.Series(dtype=bool)
    selected["captain_policy_ev"] = pd.Series(dtype=bool)
    selected["captain_policy_safe"] = pd.Series(dtype=bool)
    selected["captain_policy_upside"] = pd.Series(dtype=bool)
    result = SquadOptimizationResult(
        selected=selected,
        status=status,
        budget_used=0.0,
        predicted_points=0.0,
        predicted_points_base=0.0,
        captain_bonus_predicted=0.0,
        predicted_points_with_captain=0.0,
        formation_name=formation_name,
        selected_count=0,
        captain_id=None,
        captain_name=None,
        captain_position=None,
        captain_club=None,
        captain_predicted_points=None,
        captain_multiplier=CAPTAIN_MULTIPLIER,
        scoring_contract_version=SCORING_CONTRACT_VERSION,
        formation_scores=formation_scores,
        captain_policy_diagnostics=[],
        infeasibility_reason=infeasibility_reason,
        policy_variant=policy_variant,
    )
    return result


def _formation_score(result: SquadOptimizationResult) -> dict[str, object]:
    if result.status != "Optimal":
        return {
            "formation": result.formation_name,
            "solver_status": result.status,
            "predicted_points_base": None,
            "captain_bonus_predicted": None,
            "predicted_points_with_captain": None,
            "captain_id": None,
            "captain_name": None,
            "infeasibility_reason": result.infeasibility_reason,
        }
    return {
        "formation": result.formation_name,
        "solver_status": result.status,
        "predicted_points_base": float(result.predicted_points_base),
        "captain_bonus_predicted": float(result.captain_bonus_predicted),
        "predicted_points_with_captain": float(result.predicted_points_with_captain),
        "captain_id": result.captain_id,
        "captain_name": result.captain_name,
        "infeasibility_reason": None,
    }


def _with_formation_scores(
    result: SquadOptimizationResult, formation_scores: list[dict[str, object]]
) -> SquadOptimizationResult:
    return SquadOptimizationResult(
        selected=result.selected,
        status=result.status,
        budget_used=result.budget_used,
        predicted_points=result.predicted_points,
        predicted_points_base=result.predicted_points_base,
        captain_bonus_predicted=result.captain_bonus_predicted,
        predicted_points_with_captain=result.predicted_points_with_captain,
        formation_name=result.formation_name,
        selected_count=result.selected_count,
        captain_id=result.captain_id,
        captain_name=result.captain_name,
        captain_position=result.captain_position,
        captain_club=result.captain_club,
        captain_predicted_points=result.captain_predicted_points,
        captain_multiplier=result.captain_multiplier,
        scoring_contract_version=result.scoring_contract_version,
        formation_scores=formation_scores,
        captain_policy_diagnostics=result.captain_policy_diagnostics,
        infeasibility_reason=result.infeasibility_reason,
        opponent_overlap_asset_count=result.opponent_overlap_asset_count,
        opponent_overlap_match_count=result.opponent_overlap_match_count,
        gk_opponent_attack_pair_count=result.gk_opponent_attack_pair_count,
        gk_opponent_captain_count=result.gk_opponent_captain_count,
        clean_sheet_pair_count=result.clean_sheet_pair_count,
        clean_sheet_pair_bonus_applied=result.clean_sheet_pair_bonus_applied,
        policy_variant=result.policy_variant,
    )


def _optional_int(value: object) -> int | None:
    if pd.isna(value):
        return None
    if isinstance(value, int | float):
        return int(value)
    if isinstance(value, np.integer | np.floating):
        return int(value.item())
    return int(str(value))
