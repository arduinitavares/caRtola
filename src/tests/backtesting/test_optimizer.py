import pandas as pd

from cartola.backtesting.config import BacktestConfig
from cartola.backtesting.optimizer import optimize_squad


def _candidates():
    rows = []
    player_id = 1
    for pos, count in {"gol": 2, "lat": 3, "zag": 3, "mei": 4, "ata": 4, "tec": 2}.items():
        for offset in range(count):
            rows.append(
                {
                    "id_atleta": player_id,
                    "apelido": f"{pos}-{offset}",
                    "posicao": pos,
                    "preco": 5.0 + offset,
                    "predicted_points": 10.0 - offset,
                    "pontuacao": 7.0 - offset,
                }
            )
            player_id += 1
    return pd.DataFrame(rows)


def test_optimizer_selects_legal_433_squad_under_budget():
    result = optimize_squad(_candidates(), score_column="predicted_points", config=BacktestConfig(budget=80))

    assert result.status == "Optimal"
    assert result.selected_count == 12
    assert result.selected.groupby("posicao").size().to_dict() == {
        "ata": 3,
        "gol": 1,
        "lat": 2,
        "mei": 3,
        "tec": 1,
        "zag": 2,
    }
    assert result.budget_used <= 80


def test_optimizer_reports_infeasible_budget():
    result = optimize_squad(_candidates(), score_column="predicted_points", config=BacktestConfig(budget=1))

    assert result.status == "Infeasible"
    assert result.selected.empty


def test_optimizer_returns_empty_result_for_empty_candidates():
    result = optimize_squad(pd.DataFrame(), score_column="predicted_points", config=BacktestConfig())

    assert result.status == "Empty"
    assert result.selected.empty
    assert result.selected_count == 0
    assert result.budget_used == 0
    assert result.predicted_points == 0
    assert result.actual_points == 0
    assert result.formation_name == "4-3-3"


def test_optimizer_deduplicates_candidates_by_player_before_solving():
    duplicate = _candidates().iloc[[0]].copy()
    duplicate["preco"] = 0.1
    duplicate["predicted_points"] = 1000.0
    candidates = pd.concat([_candidates(), duplicate], ignore_index=True)

    result = optimize_squad(candidates, score_column="predicted_points", config=BacktestConfig(budget=80))

    assert result.status == "Optimal"
    assert result.selected_count == 12
    assert result.selected["id_atleta"].is_unique
    assert result.selected.loc[result.selected["id_atleta"] == duplicate.iloc[0]["id_atleta"], "preco"].iloc[0] == 5.0
