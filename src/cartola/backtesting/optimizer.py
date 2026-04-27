from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import pulp

from cartola.backtesting.config import MARKET_OPEN_PRICE_COLUMN, BacktestConfig


@dataclass(frozen=True)
class SquadOptimizationResult:
    selected: pd.DataFrame
    status: str
    budget_used: float
    predicted_points: float
    actual_points: float
    formation_name: str
    selected_count: int


def optimize_squad(candidates: pd.DataFrame, score_column: str, config: BacktestConfig) -> SquadOptimizationResult:
    formation = config.selected_formation
    if candidates.empty:
        return _empty_result("Empty", config.formation_name, candidates)

    player_rows = candidates.loc[~candidates["id_atleta"].duplicated()].copy()
    player_rows.index = range(len(player_rows))
    variables = {
        index: pulp.LpVariable(f"player_{index}_{player_rows.loc[index, 'id_atleta']}", cat=pulp.LpBinary)
        for index in player_rows.index
    }

    problem = pulp.LpProblem("CartolaSquadOptimizer", pulp.LpMaximize)
    problem += pulp.lpSum(
        float(player_rows.loc[index, score_column]) * variable for index, variable in variables.items()
    )
    problem += pulp.lpSum(
        float(player_rows.loc[index, MARKET_OPEN_PRICE_COLUMN]) * variable
        for index, variable in variables.items()
    ) <= float(config.budget)

    for position, required_count in formation.items():
        problem += (
            pulp.lpSum(
                variable for index, variable in variables.items() if player_rows.loc[index, "posicao"] == position
            )
            == required_count
        )

    status_code = problem.solve(pulp.PULP_CBC_CMD(msg=False))
    status = pulp.LpStatus[status_code]
    if status != "Optimal":
        return _empty_result(status, config.formation_name, candidates)

    selected_indexes = [index for index, variable in variables.items() if pulp.value(variable) == 1]
    selected = player_rows.loc[selected_indexes].copy()
    selected.index = range(len(selected))
    budget_used = float(selected[MARKET_OPEN_PRICE_COLUMN].sum())
    predicted_points = float(selected[score_column].sum())
    actual_points = float(selected["pontuacao"].sum()) if "pontuacao" in selected.columns else 0.0

    return SquadOptimizationResult(
        selected=selected,
        status=status,
        budget_used=budget_used,
        predicted_points=predicted_points,
        actual_points=actual_points,
        formation_name=config.formation_name,
        selected_count=len(selected),
    )


def _empty_result(status: str, formation_name: str, candidates: pd.DataFrame) -> SquadOptimizationResult:
    return SquadOptimizationResult(
        selected=candidates.iloc[0:0].copy(),
        status=status,
        budget_used=0.0,
        predicted_points=0.0,
        actual_points=0.0,
        formation_name=formation_name,
        selected_count=0,
    )
