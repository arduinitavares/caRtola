import pandas as pd

from cartola.backtesting.metrics import build_diagnostics, build_summary


def test_build_summary_computes_strategy_totals_and_benchmark_delta():
    round_results = pd.DataFrame(
        [
            {
                "strategy": "model",
                "rodada": 5,
                "actual_points": 50.0,
                "predicted_points": 55.0,
                "solver_status": "Optimal",
            },
            {
                "strategy": "model",
                "rodada": 6,
                "actual_points": 60.0,
                "predicted_points": 58.0,
                "solver_status": "Optimal",
            },
            {
                "strategy": "price",
                "rodada": 5,
                "actual_points": 45.0,
                "predicted_points": 45.0,
                "solver_status": "Optimal",
            },
            {
                "strategy": "price",
                "rodada": 6,
                "actual_points": 50.0,
                "predicted_points": 50.0,
                "solver_status": "Optimal",
            },
        ]
    )

    summary = build_summary(round_results, benchmark_strategy="price")
    model = summary[summary["strategy"] == "model"].iloc[0]

    assert model["rounds"] == 2
    assert model["total_actual_points"] == 110.0
    assert model["average_actual_points"] == 55.0
    assert model["actual_points_delta_vs_price"] == 15.0


def test_build_summary_returns_expected_columns_for_empty_input():
    summary = build_summary(pd.DataFrame(), benchmark_strategy="model")

    assert summary.empty
    assert summary.columns.tolist() == [
        "strategy",
        "rounds",
        "total_actual_points",
        "average_actual_points",
        "total_predicted_points",
        "actual_points_delta_vs_model",
    ]


def test_build_summary_ignores_non_optimal_rows():
    round_results = pd.DataFrame(
        [
            {
                "strategy": "model",
                "rodada": 1,
                "actual_points": 10.0,
                "predicted_points": 12.0,
                "solver_status": "Optimal",
            },
            {
                "strategy": "model",
                "rodada": 2,
                "actual_points": 100.0,
                "predicted_points": 120.0,
                "solver_status": "Infeasible",
            },
            {
                "strategy": "price",
                "rodada": 1,
                "actual_points": 8.0,
                "predicted_points": 9.0,
                "solver_status": "Optimal",
            },
        ]
    )

    summary = build_summary(round_results)
    model = summary[summary["strategy"] == "model"].iloc[0]

    assert model["rounds"] == 1
    assert model["total_actual_points"] == 10.0
    assert model["total_predicted_points"] == 12.0
    assert model["actual_points_delta_vs_price"] == 2.0


def test_build_summary_uses_missing_delta_when_benchmark_absent_from_optimal_rows():
    round_results = pd.DataFrame(
        [
            {
                "strategy": "model",
                "rodada": 1,
                "actual_points": 10.0,
                "predicted_points": 12.0,
                "solver_status": "Optimal",
            },
            {
                "strategy": "value",
                "rodada": 1,
                "actual_points": 8.0,
                "predicted_points": 9.0,
                "solver_status": "Optimal",
            },
        ]
    )

    summary = build_summary(round_results, benchmark_strategy="price")

    assert summary["actual_points_delta_vs_price"].isna().all()


def test_build_summary_uses_missing_delta_when_benchmark_is_only_non_optimal():
    round_results = pd.DataFrame(
        [
            {
                "strategy": "model",
                "rodada": 1,
                "actual_points": 10.0,
                "predicted_points": 12.0,
                "solver_status": "Optimal",
            },
            {
                "strategy": "price",
                "rodada": 1,
                "actual_points": 8.0,
                "predicted_points": 9.0,
                "solver_status": "Infeasible",
            },
        ]
    )

    summary = build_summary(round_results, benchmark_strategy="price")

    assert summary["actual_points_delta_vs_price"].isna().all()


def test_build_summary_returns_expected_columns_when_all_rows_are_non_optimal():
    round_results = pd.DataFrame(
        [
            {
                "strategy": "model",
                "rodada": 1,
                "actual_points": 10.0,
                "predicted_points": 12.0,
                "solver_status": "Infeasible",
            },
            {
                "strategy": "price",
                "rodada": 1,
                "actual_points": 8.0,
                "predicted_points": 9.0,
                "solver_status": "Infeasible",
            },
        ]
    )

    summary = build_summary(round_results, benchmark_strategy="price")

    assert summary.empty
    assert summary.columns.tolist() == [
        "strategy",
        "rounds",
        "total_actual_points",
        "average_actual_points",
        "total_predicted_points",
        "actual_points_delta_vs_price",
    ]


def test_build_summary_sorts_by_total_actual_points_and_resets_index():
    round_results = pd.DataFrame(
        [
            {
                "strategy": "price",
                "rodada": 1,
                "actual_points": 20.0,
                "predicted_points": 20.0,
                "solver_status": "Optimal",
            },
            {
                "strategy": "model",
                "rodada": 1,
                "actual_points": 30.0,
                "predicted_points": 31.0,
                "solver_status": "Optimal",
            },
            {
                "strategy": "value",
                "rodada": 1,
                "actual_points": 10.0,
                "predicted_points": 11.0,
                "solver_status": "Optimal",
            },
        ]
    )

    summary = build_summary(round_results, benchmark_strategy="price")

    assert summary["strategy"].tolist() == ["model", "price", "value"]
    assert summary.index.tolist() == [0, 1, 2]


def test_build_diagnostics_reports_prediction_round_selection_and_random_metrics():
    round_results = pd.DataFrame(
        [
            _round("random_forest", 1, 12.0),
            _round("random_forest", 2, 8.0),
            _round("baseline", 1, 10.0),
            _round("baseline", 2, 9.0),
            _round("price", 1, 9.0),
            _round("price", 2, 7.0),
        ]
    )
    selected_players = pd.DataFrame(
        [
            _selected("random_forest", 1, "ata", 10.0, True, 5.0),
            _selected("random_forest", 1, "mei", 2.0, True, 5.0),
            _selected("random_forest", 2, "ata", 6.0, True, 5.0),
            _selected("random_forest", 2, "mei", 2.0, False, 5.0),
            _selected("baseline", 1, "ata", 8.0, True, 5.0),
            _selected("baseline", 1, "mei", 2.0, True, 5.0),
            _selected("baseline", 2, "ata", 6.0, True, 5.0),
            _selected("baseline", 2, "mei", 3.0, True, 5.0),
            _selected("price", 1, "ata", 7.0, True, 5.0),
            _selected("price", 1, "mei", 2.0, True, 5.0),
            _selected("price", 2, "ata", 5.0, True, 5.0),
            _selected("price", 2, "mei", 2.0, False, 5.0),
        ]
    )
    player_predictions = pd.DataFrame(
        [
            _prediction(1, "ata", 10.0, 5.0, baseline=8.0, random_forest=9.0, price=12.0),
            _prediction(1, "ata", 4.0, 5.0, baseline=6.0, random_forest=7.0, price=8.0),
            _prediction(1, "mei", 2.0, 5.0, baseline=4.0, random_forest=3.0, price=5.0),
            _prediction(1, "mei", 0.0, 5.0, baseline=2.0, random_forest=2.0, price=4.0),
            _prediction(2, "ata", 6.0, 5.0, baseline=7.0, random_forest=5.0, price=9.0),
            _prediction(2, "ata", 2.0, 5.0, baseline=4.0, random_forest=3.0, price=7.0),
            _prediction(2, "mei", 4.0, 5.0, baseline=3.0, random_forest=4.0, price=6.0),
            _prediction(2, "mei", 1.0, 5.0, baseline=2.0, random_forest=1.0, price=5.0),
        ]
    )

    diagnostics = build_diagnostics(
        round_results,
        selected_players,
        player_predictions,
        budget=10.0,
        random_draws=50,
        random_seed=7,
    )

    assert _metric(diagnostics, "prediction", "random_forest", "all", "player_mae") == 1.125
    assert round(_metric(diagnostics, "prediction", "random_forest", "all", "player_r2"), 6) == 0.763478
    assert _metric(diagnostics, "prediction", "random_forest", "ata", "player_mae") == 1.5
    assert _metric(diagnostics, "prediction", "random_forest", "mei", "player_mean_error") == 0.75
    assert _metric(diagnostics, "rounds", "random_forest", "all", "round_wins_vs_price") == 2
    assert _metric(diagnostics, "rounds", "random_forest", "all", "round_total_delta_vs_price") == 4.0
    assert _metric(diagnostics, "selection", "random_forest", "all", "selected_player_actual_points_mean") == 5.0
    assert _metric(diagnostics, "selection", "random_forest", "all", "selected_entrou_em_campo_rate") == 0.75
    assert _metric(diagnostics, "selection", "random_forest", "mei", "selected_entrou_em_campo_rate") == 0.5
    assert _metric(diagnostics, "random_selection", "random_forest", "all", "successful_random_draws") == 100
    assert (
        _metric_value(
            diagnostics,
            "random_selection",
            "random_forest",
            "all",
            "random_baseline_captain_policy",
        )
        == "actual_best_non_tecnico"
    )
    assert round(
        _metric(diagnostics, "random_selection", "random_forest", "all", "actual_points_delta_vs_random_expected"),
        2,
    ) == -0.04


def test_build_diagnostics_random_expected_points_include_captain_bonus():
    round_results = pd.DataFrame([_round("model", 1, 70.0)])
    selected_players = pd.DataFrame(
        [
            _selected("model", 1, "gol", 20.0, True, 1.0),
            _selected("model", 1, "lat", 10.0, True, 1.0),
            _selected("model", 1, "tec", 30.0, True, 1.0),
        ]
    )
    player_predictions = pd.DataFrame(
        [
            _prediction(1, "gol", 20.0, 1.0, baseline=20.0, random_forest=20.0, price=20.0),
            _prediction(1, "lat", 10.0, 1.0, baseline=10.0, random_forest=10.0, price=10.0),
            _prediction(1, "tec", 30.0, 1.0, baseline=30.0, random_forest=30.0, price=30.0),
        ]
    )

    diagnostics = build_diagnostics(
        round_results,
        selected_players,
        player_predictions,
        budget=3.0,
        random_draws=1,
        random_seed=7,
    )

    assert _metric(diagnostics, "random_selection", "model", "all", "successful_random_draws") == 1
    assert _metric(diagnostics, "random_selection", "model", "all", "random_expected_actual_points_total") == 70.0
    assert (
        _metric_value(diagnostics, "random_selection", "model", "all", "random_baseline_captain_policy")
        == "actual_best_non_tecnico"
    )


def test_build_diagnostics_returns_expected_columns_for_empty_inputs():
    diagnostics = build_diagnostics(pd.DataFrame(), pd.DataFrame(), pd.DataFrame())

    assert diagnostics.empty
    assert diagnostics.columns.tolist() == ["section", "strategy", "position", "metric", "value"]


def _round(strategy: str, rodada: int, actual_points: float) -> dict[str, object]:
    return {
        "strategy": strategy,
        "rodada": rodada,
        "actual_points": actual_points,
        "predicted_points": actual_points + 1.0,
        "solver_status": "Optimal",
    }


def _selected(
    strategy: str,
    rodada: int,
    posicao: str,
    pontuacao: float,
    entrou_em_campo: bool,
    preco_pre_rodada: float,
) -> dict[str, object]:
    return {
        "strategy": strategy,
        "rodada": rodada,
        "posicao": posicao,
        "pontuacao": pontuacao,
        "entrou_em_campo": entrou_em_campo,
        "preco_pre_rodada": preco_pre_rodada,
    }


def _prediction(
    rodada: int,
    posicao: str,
    pontuacao: float,
    preco_pre_rodada: float,
    *,
    baseline: float,
    random_forest: float,
    price: float,
) -> dict[str, object]:
    return {
        "rodada": rodada,
        "posicao": posicao,
        "pontuacao": pontuacao,
        "preco_pre_rodada": preco_pre_rodada,
        "baseline_score": baseline,
        "random_forest_score": random_forest,
        "price_score": price,
    }


def _metric(
    diagnostics: pd.DataFrame,
    section: str,
    strategy: str,
    position: str,
    metric: str,
) -> float:
    return float(_metric_value(diagnostics, section, strategy, position, metric))


def _metric_value(
    diagnostics: pd.DataFrame,
    section: str,
    strategy: str,
    position: str,
    metric: str,
) -> object:
    matches = diagnostics[
        diagnostics["section"].eq(section)
        & diagnostics["strategy"].eq(strategy)
        & diagnostics["position"].eq(position)
        & diagnostics["metric"].eq(metric)
    ]
    assert len(matches) == 1
    return matches.iloc[0]["value"]
