import inspect

import pandas as pd
import pulp
import pytest

import cartola.backtesting.optimizer as optimizer
from cartola.backtesting.config import DEFAULT_FORMATIONS, BacktestConfig
from cartola.backtesting.optimizer import optimize_squad
from cartola.backtesting.scoring_contract import CAPTAIN_MULTIPLIER, SCORING_CONTRACT_VERSION


def _row(player_id: int, posicao: str, score: float, price: float = 5.0) -> dict[str, object]:
    return {
        "id_atleta": player_id,
        "apelido": f"{posicao}-{player_id}",
        "posicao": posicao,
        "clube": f"club-{posicao}",
        "preco": 999.0,
        "preco_pre_rodada": price,
        "predicted_points": score,
        "pontuacao": score - 1.0,
    }


def _candidates() -> pd.DataFrame:
    rows = []
    player_id = 1
    scores = {
        "gol": [8.0, 7.0],
        "lat": [5.0, 4.8, 4.6],
        "zag": [7.5, 7.0, 6.5, 6.0],
        "mei": [18.0, 17.0, 16.0, 15.0, 14.0, 4.0],
        "ata": [11.0, 10.0, 9.0, 3.0],
        "tec": [40.0, 6.0],
    }
    for posicao, position_scores in scores.items():
        for score in position_scores:
            rows.append(_row(player_id, posicao, score))
            player_id += 1
    return pd.DataFrame(rows)


def test_optimizer_searches_all_formations_and_returns_captain_aware_scores():
    result = optimize_squad(_candidates(), score_column="predicted_points", config=BacktestConfig(budget=80))

    assert result.status == "Optimal"
    assert result.selected_count == 12
    assert result.formation_name == "3-5-2"
    assert result.selected.groupby("posicao").size().to_dict() == {
        "ata": 2,
        "gol": 1,
        "mei": 5,
        "tec": 1,
        "zag": 3,
    }
    assert result.selected["is_captain"].sum() == 1
    captain = result.selected.loc[result.selected["is_captain"]].iloc[0]
    assert captain["posicao"] != "tec"
    assert result.captain_id == captain["id_atleta"]
    assert result.captain_name == captain["apelido"]
    assert result.captain_position == captain["posicao"]
    assert result.captain_club == captain["clube"]
    assert result.captain_predicted_points == pytest.approx(captain["predicted_points"])
    assert result.scoring_contract_version == SCORING_CONTRACT_VERSION
    assert result.captain_multiplier == CAPTAIN_MULTIPLIER

    expected_base = float(result.selected["predicted_points"].sum())
    expected_bonus = (CAPTAIN_MULTIPLIER - 1.0) * float(captain["predicted_points"])
    assert result.predicted_points_base == pytest.approx(expected_base)
    assert result.captain_bonus_predicted == pytest.approx(expected_bonus)
    assert result.predicted_points_with_captain == pytest.approx(expected_base + expected_bonus)
    assert result.predicted_points == pytest.approx(result.predicted_points_with_captain)

    assert len(result.formation_scores) == len(DEFAULT_FORMATIONS)
    assert {score["formation"] for score in result.formation_scores} == set(DEFAULT_FORMATIONS)
    chosen_score = next(score for score in result.formation_scores if score["formation"] == result.formation_name)
    assert chosen_score["solver_status"] == "Optimal"
    assert chosen_score["captain_id"] == result.captain_id
    assert result.captain_policy_diagnostics == []


def test_optimizer_never_captains_unselected_phantom_player_or_tecnico():
    candidates = _candidates()
    phantom = _row(999, "ata", score=1000.0, price=500.0)
    candidates = pd.concat([candidates, pd.DataFrame([phantom])], ignore_index=True)

    result = optimize_squad(candidates, score_column="predicted_points", config=BacktestConfig(budget=80))

    assert result.status == "Optimal"
    assert result.captain_id != 999
    assert 999 not in set(result.selected["id_atleta"])
    assert result.captain_position != "tec"
    assert result.captain_predicted_points == pytest.approx(18.0)


def test_optimizer_rejects_high_scoring_unexpected_position_from_selected_roster():
    candidates = _candidates()
    unexpected = _row(999, "ban", score=1000.0, price=0.1)
    candidates = pd.concat([candidates, pd.DataFrame([unexpected])], ignore_index=True)

    result = optimize_squad(candidates, score_column="predicted_points", config=BacktestConfig(budget=80))

    assert result.status == "Optimal"
    assert result.selected_count == 12
    assert 999 not in set(result.selected["id_atleta"])
    assert "ban" not in set(result.selected["posicao"])
    assert sum(result.selected.groupby("posicao").size().to_dict().values()) == 12


def test_optimizer_reports_per_formation_scores_for_infeasible_and_chosen_formations():
    candidates = _candidates().loc[lambda frame: frame["posicao"] != "lat"].copy()

    result = optimize_squad(candidates, score_column="predicted_points", config=BacktestConfig(budget=80))

    assert result.status == "Optimal"
    assert result.formation_name in {"3-4-3", "3-5-2"}
    assert len(result.formation_scores) == len(DEFAULT_FORMATIONS)
    assert any(score["solver_status"] == "Infeasible" for score in result.formation_scores)
    assert any(
        score["formation"] == result.formation_name and score["solver_status"] == "Optimal"
        for score in result.formation_scores
    )


def test_optimizer_breaks_exact_ties_with_lower_player_and_captain_ids():
    rows = []
    player_id = 1
    for posicao, count in {
        "gol": 2,
        "lat": 2,
        "zag": 4,
        "mei": 6,
        "ata": 4,
        "tec": 2,
    }.items():
        for _ in range(count):
            rows.append(_row(player_id, posicao, score=5.0, price=5.0))
            player_id += 1
    candidates = pd.DataFrame(rows).sample(frac=1, random_state=7).reset_index(drop=True)

    results = [
        optimize_squad(candidates, score_column="predicted_points", config=BacktestConfig(budget=80))
        for _ in range(3)
    ]

    selected_ids = [tuple(result.selected["id_atleta"].astype(int)) for result in results]
    captain_ids = [result.captain_id for result in results]
    assert len(set(selected_ids)) == 1
    assert len(set(captain_ids)) == 1
    assert selected_ids[0] == (1, 5, 6, 7, 9, 10, 11, 12, 15, 16, 17, 19)
    assert captain_ids[0] == 1


def test_tie_break_objective_penalizes_captain_variables_separately():
    player_rows = pd.DataFrame({"id_atleta": [10, 20]})
    selected_vars = {
        0: pulp.LpVariable("selected_0", cat=pulp.LpBinary),
        1: pulp.LpVariable("selected_1", cat=pulp.LpBinary),
    }
    captain_vars = {
        0: pulp.LpVariable("captain_0", cat=pulp.LpBinary),
        1: pulp.LpVariable("captain_1", cat=pulp.LpBinary),
    }

    expression = optimizer._tie_break_objective(player_rows, selected_vars, captain_vars)
    coefficients = dict(expression.items())

    assert coefficients[selected_vars[0]] == pytest.approx(-10.0)
    assert coefficients[selected_vars[1]] == pytest.approx(-20.0)
    assert coefficients[captain_vars[0]] < 0.0
    assert coefficients[captain_vars[1]] < coefficients[captain_vars[0]]


def test_optimizer_reports_all_formations_without_exposing_fixed_formation_api():
    candidates = _candidates().loc[lambda frame: frame["posicao"] != "lat"].copy()

    result = optimize_squad(candidates, score_column="predicted_points", config=BacktestConfig(budget=80))

    assert result.status == "Optimal"
    assert len(result.formation_scores) == len(DEFAULT_FORMATIONS)
    assert {score["formation"] for score in result.formation_scores} == set(DEFAULT_FORMATIONS)
    assert any(score["solver_status"] == "Infeasible" for score in result.formation_scores)
    assert "formation_name" not in inspect.signature(optimize_squad).parameters


def test_optimizer_returns_empty_result_for_empty_candidates():
    result = optimize_squad(pd.DataFrame(), score_column="predicted_points", config=BacktestConfig())

    assert result.status == "Empty"
    assert result.selected.empty
    assert result.selected_count == 0
    assert result.budget_used == 0
    assert result.predicted_points == 0
    assert result.predicted_points_base == 0
    assert result.captain_bonus_predicted == 0
    assert result.predicted_points_with_captain == 0
    assert result.formation_name == ""
    assert result.captain_id is None
    assert result.scoring_contract_version == SCORING_CONTRACT_VERSION
    assert result.captain_multiplier == CAPTAIN_MULTIPLIER


def test_optimizer_reports_infeasible_budget_with_formation_scores():
    result = optimize_squad(_candidates(), score_column="predicted_points", config=BacktestConfig(budget=1))

    assert result.status == "Infeasible"
    assert result.selected.empty
    assert result.selected_count == 0
    assert len(result.formation_scores) == len(DEFAULT_FORMATIONS)
    assert {score["solver_status"] for score in result.formation_scores} == {"Infeasible"}
    assert all(score["infeasibility_reason"] for score in result.formation_scores)


def test_optimizer_deduplicates_candidates_by_player_before_solving():
    duplicate = _candidates().iloc[[0]].copy()
    duplicate["preco"] = 0.1
    duplicate["preco_pre_rodada"] = 0.1
    duplicate["predicted_points"] = 1000.0
    candidates = pd.concat([_candidates(), duplicate], ignore_index=True)

    result = optimize_squad(candidates, score_column="predicted_points", config=BacktestConfig(budget=80))

    assert result.status == "Optimal"
    assert result.selected_count == 12
    assert result.selected["id_atleta"].is_unique
    assert (
        result.selected.loc[result.selected["id_atleta"] == duplicate.iloc[0]["id_atleta"], "preco_pre_rodada"].iloc[0]
        == 5.0
    )
    assert result.selected.loc[result.selected["id_atleta"] == duplicate.iloc[0]["id_atleta"], "predicted_points"].iloc[
        0
    ] == pytest.approx(8.0)


def test_optimizer_uses_pre_round_price_for_budget():
    candidates = _candidates()
    candidates["preco"] = 100.0
    candidates["preco_pre_rodada"] = 5.0

    result = optimize_squad(candidates, score_column="predicted_points", config=BacktestConfig(budget=80))

    assert result.status == "Optimal"
    assert result.selected_count == 12
    assert result.budget_used == 60.0


@pytest.mark.parametrize("column", ["id_atleta", "apelido", "posicao", "preco_pre_rodada", "predicted_points"])
def test_optimizer_rejects_missing_required_columns(column):
    candidates = _candidates().drop(columns=[column])

    with pytest.raises(ValueError, match=column):
        optimize_squad(candidates, score_column="predicted_points", config=BacktestConfig(budget=80))


@pytest.mark.parametrize("column", ["preco_pre_rodada", "predicted_points"])
def test_optimizer_rejects_non_numeric_solver_columns(column):
    candidates = _candidates()
    candidates[column] = candidates[column].astype(object)
    candidates.loc[0, column] = "bad"

    with pytest.raises(ValueError, match=column):
        optimize_squad(candidates, score_column="predicted_points", config=BacktestConfig(budget=80))
