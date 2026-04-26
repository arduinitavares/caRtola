from __future__ import annotations

import pandas as pd


SUMMARY_COLUMNS: list[str] = [
    "strategy",
    "rounds",
    "total_actual_points",
    "average_actual_points",
    "total_predicted_points",
]


def build_summary(round_results: pd.DataFrame, benchmark_strategy: str = "price") -> pd.DataFrame:
    delta_column = f"actual_points_delta_vs_{benchmark_strategy}"
    columns = [*SUMMARY_COLUMNS, delta_column]

    if round_results.empty:
        return pd.DataFrame(columns=columns)

    optimal_results = round_results[round_results["solver_status"] == "Optimal"]
    if optimal_results.empty:
        return pd.DataFrame(columns=columns)

    summary = (
        optimal_results.groupby("strategy", as_index=False)
        .agg(
            rounds=("rodada", "nunique"),
            total_actual_points=("actual_points", "sum"),
            average_actual_points=("actual_points", "mean"),
            total_predicted_points=("predicted_points", "sum"),
        )
        .sort_values("total_actual_points", ascending=False)
        .reset_index(drop=True)
    )

    benchmark_total = summary.loc[summary["strategy"] == benchmark_strategy, "total_actual_points"].sum()
    summary[delta_column] = summary["total_actual_points"] - benchmark_total
    return summary.loc[:, columns]
