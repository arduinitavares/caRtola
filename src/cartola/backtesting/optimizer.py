from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import pulp

from cartola.backtesting.config import DEFAULT_FORMATIONS, MARKET_OPEN_PRICE_COLUMN, BacktestConfig
from cartola.backtesting.scoring_contract import CAPTAIN_MULTIPLIER, SCORING_CONTRACT_VERSION


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


def optimize_squad(candidates: pd.DataFrame, score_column: str, config: BacktestConfig) -> SquadOptimizationResult:
    if candidates.empty:
        return _empty_result("Empty", "", candidates, formation_scores=[])

    results = [
        _optimize_formation(candidates, score_column=score_column, config=config, formation_name=formation_name)
        for formation_name in DEFAULT_FORMATIONS
    ]
    formation_scores = [_formation_score(result) for result in results]
    optimal_results = [result for result in results if result.status == "Optimal"]
    if not optimal_results:
        return _empty_result("Infeasible", "", candidates, formation_scores=formation_scores)

    best = min(
        optimal_results,
        key=lambda result: (
            -result.predicted_points_with_captain,
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
    problem += pulp.lpSum(
        float(player_rows.loc[index, score_column]) * variable for index, variable in variables.items()
    ) + (CAPTAIN_MULTIPLIER - 1.0) * pulp.lpSum(
        float(player_rows.loc[index, score_column]) * captain_variable
        for index, captain_variable in captain_variables.items()
    )
    problem += pulp.lpSum(
        float(player_rows.loc[index, MARKET_OPEN_PRICE_COLUMN]) * variable
        for index, variable in variables.items()
    ) <= float(config.budget)
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

    status_code = problem.solve(pulp.PULP_CBC_CMD(msg=False))
    status = pulp.LpStatus[status_code]
    if status != "Optimal":
        return _empty_result(
            status,
            formation_name,
            candidates,
            formation_scores=[],
            infeasibility_reason="No feasible squad satisfies formation, budget, and captain constraints.",
        )

    selected_indexes = [index for index, variable in variables.items() if pulp.value(variable) == 1]
    selected = player_rows.loc[selected_indexes].copy().sort_values("id_atleta", kind="mergesort").reset_index(drop=True)
    captain_indexes = {index for index, variable in captain_variables.items() if pulp.value(variable) == 1}
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
    )


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
    )


def _optional_int(value: Any) -> int | None:
    if pd.isna(value):
        return None
    return int(value)
