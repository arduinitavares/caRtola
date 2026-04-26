import pandas as pd

from cartola.backtesting.metrics import build_summary


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
