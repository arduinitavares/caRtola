import pandas as pd
from pandas.testing import assert_frame_equal

from cartola.backtesting.config import DEFAULT_SCOUT_COLUMNS, BacktestConfig
from cartola.backtesting.runner import run_backtest


def _tiny_round(round_number: int) -> pd.DataFrame:
    rows = []
    player_id = 1
    for pos, count in {"gol": 2, "lat": 3, "zag": 3, "mei": 4, "ata": 4, "tec": 2}.items():
        for offset in range(count):
            row = {
                "id_atleta": player_id,
                "apelido": f"{pos}-{offset}",
                "slug": f"{pos}-{offset}",
                "id_clube": player_id,
                "nome_clube": "Club",
                "posicao": pos,
                "status": "Provavel",
                "rodada": round_number,
                "preco": 5.0,
                "pontuacao": float(10 - offset + round_number),
                "media": float(5 + offset),
                "num_jogos": round_number,
                "variacao": 0.0,
                "entrou_em_campo": True,
            }
            for scout in DEFAULT_SCOUT_COLUMNS:
                row[scout] = 0
            rows.append(row)
            player_id += 1
    return pd.DataFrame(rows)


def test_run_backtest_writes_round_players_predictions_and_summary(tmp_path):
    season_df = pd.concat([_tiny_round(round_number) for round_number in range(1, 6)], ignore_index=True)
    config = BacktestConfig(project_root=tmp_path, start_round=5, budget=100)

    result = run_backtest(config, season_df=season_df)

    assert set(result.round_results["strategy"]) == {"baseline", "random_forest", "price"}
    assert (tmp_path / "data/08_reporting/backtests/2025/round_results.csv").exists()
    assert (tmp_path / "data/08_reporting/backtests/2025/selected_players.csv").exists()
    assert (tmp_path / "data/08_reporting/backtests/2025/player_predictions.csv").exists()
    assert (tmp_path / "data/08_reporting/backtests/2025/summary.csv").exists()


def test_run_backtest_records_selected_players_and_prediction_diagnostics(tmp_path):
    season_df = pd.concat([_tiny_round(round_number) for round_number in range(1, 6)], ignore_index=True)
    config = BacktestConfig(project_root=tmp_path, start_round=5, budget=100)

    result = run_backtest(config, season_df=season_df)

    assert result.round_results["solver_status"].eq("Optimal").all()
    assert result.round_results["selected_count"].eq(12).all()
    assert set(result.selected_players["strategy"]) == {"baseline", "random_forest", "price"}
    assert result.selected_players["rodada"].eq(5).all()
    assert result.selected_players["predicted_points"].notna().all()
    assert len(result.selected_players) == 36
    assert {
        "rodada",
        "baseline_score",
        "random_forest_score",
        "price_score",
    }.issubset(result.player_predictions.columns)
    assert len(result.player_predictions) == 18
    assert set(result.summary["strategy"]) == {"baseline", "random_forest", "price"}


def test_run_backtest_normalizes_tiny_float_drift_in_returned_outputs(tmp_path, monkeypatch):
    class NoisyRandomForestPointPredictor:
        calls = 0

        def __init__(self, random_seed: int = 123) -> None:
            self.random_seed = random_seed

        def fit(self, frame: pd.DataFrame) -> "NoisyRandomForestPointPredictor":
            return self

        def predict(self, frame: pd.DataFrame) -> pd.Series:
            NoisyRandomForestPointPredictor.calls += 1
            return pd.Series(
                frame["prior_points_mean"].astype(float) + NoisyRandomForestPointPredictor.calls * 0.000000000001,
                index=frame.index,
            )

    monkeypatch.setattr(
        "cartola.backtesting.runner.RandomForestPointPredictor",
        NoisyRandomForestPointPredictor,
    )
    season_df = pd.concat([_tiny_round(round_number) for round_number in range(1, 6)], ignore_index=True)

    first = run_backtest(
        BacktestConfig(project_root=tmp_path / "first", start_round=5, budget=100), season_df=season_df
    )
    second = run_backtest(
        BacktestConfig(project_root=tmp_path / "second", start_round=5, budget=100), season_df=season_df
    )

    assert_frame_equal(first.round_results, second.round_results, check_exact=True)
    assert_frame_equal(first.selected_players, second.selected_players, check_exact=True)
    assert_frame_equal(first.player_predictions, second.player_predictions, check_exact=True)
    assert_frame_equal(first.summary, second.summary, check_exact=True)


def test_selected_players_predicted_points_match_strategy_score_column(tmp_path):
    season_df = pd.concat([_tiny_round(round_number) for round_number in range(1, 6)], ignore_index=True)
    config = BacktestConfig(project_root=tmp_path, start_round=5, budget=100)

    result = run_backtest(config, season_df=season_df)

    for strategy, score_column in {
        "baseline": "baseline_score",
        "random_forest": "random_forest_score",
        "price": "price_score",
    }.items():
        selected = result.selected_players[result.selected_players["strategy"] == strategy]
        assert selected["predicted_points"].equals(selected[score_column])
