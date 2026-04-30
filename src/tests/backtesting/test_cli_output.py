from pathlib import Path

import pandas as pd
import pytest

from cartola.backtesting.cli_output import (
    ChartOutput,
    _build_performance_figure,
    _prepare_chart_data,
    write_performance_chart,
)


def test_prepare_chart_data_filters_skipped_rows_from_score_traces() -> None:
    round_results = pd.DataFrame(
        [
            {
                "rodada": 1,
                "strategy": "random_forest",
                "solver_status": "Optimal",
                "actual_points": 10.0,
                "formation": "4-3-3",
            },
            {
                "rodada": 2,
                "strategy": "random_forest",
                "solver_status": "Empty",
                "actual_points": 0.0,
                "formation": "",
            },
            {
                "rodada": 3,
                "strategy": "random_forest",
                "solver_status": "Optimal",
                "actual_points": 7.5,
                "formation": "4-4-2",
            },
            {
                "rodada": 1,
                "strategy": "baseline",
                "solver_status": "Optimal",
                "actual_points": 8.0,
                "formation": "4-3-3",
            },
        ]
    )

    chart_data = _prepare_chart_data(round_results)

    score_rows = chart_data.score_rows.sort_values(["strategy", "rodada"]).reset_index(drop=True)
    assert score_rows[["strategy", "rodada", "actual_points", "cumulative_points"]].to_dict("records") == [
        {"strategy": "baseline", "rodada": 1, "actual_points": 8.0, "cumulative_points": 8.0},
        {"strategy": "random_forest", "rodada": 1, "actual_points": 10.0, "cumulative_points": 10.0},
        {"strategy": "random_forest", "rodada": 3, "actual_points": 7.5, "cumulative_points": 17.5},
    ]
    assert chart_data.status_rows[["strategy", "rodada", "solver_status"]].to_dict("records") == [
        {"strategy": "random_forest", "rodada": 2, "solver_status": "Empty"}
    ]
    assert chart_data.formation_rows[["rodada", "formation"]].to_dict("records") == [
        {"rodada": 1, "formation": "4-3-3"},
        {"rodada": 3, "formation": "4-4-2"},
    ]


def test_prepare_chart_data_accepts_all_non_optimal_rows_without_fake_zero_lines() -> None:
    round_results = pd.DataFrame(
        [
            {
                "rodada": 1,
                "strategy": "baseline",
                "solver_status": "Empty",
                "actual_points": "not a score",
                "formation": "",
            },
            {
                "rodada": 2,
                "strategy": "random_forest",
                "solver_status": "TrainingEmpty",
                "actual_points": None,
                "formation": "",
            },
            {
                "rodada": 3,
                "strategy": "price",
                "solver_status": "Infeasible",
                "actual_points": "",
                "formation": "",
            },
        ]
    )

    chart_data = _prepare_chart_data(round_results)

    assert chart_data.score_rows.empty
    assert chart_data.status_rows["solver_status"].tolist() == ["Empty", "TrainingEmpty", "Infeasible"]
    assert chart_data.formation_rows.empty


def test_prepare_chart_data_rejects_missing_columns() -> None:
    round_results = pd.DataFrame(
        [
            {
                "rodada": 1,
                "solver_status": "Optimal",
                "actual_points": 10.0,
            }
        ]
    )

    with pytest.raises(ValueError, match="Missing chart columns: formation, strategy"):
        _prepare_chart_data(round_results)


def test_prepare_chart_data_rejects_non_numeric_round() -> None:
    round_results = pd.DataFrame(
        [
            {
                "rodada": "first",
                "strategy": "baseline",
                "solver_status": "Optimal",
                "actual_points": 10.0,
                "formation": "4-3-3",
            }
        ]
    )

    with pytest.raises(ValueError, match="Chart column 'rodada' must be numeric"):
        _prepare_chart_data(round_results)


def test_prepare_chart_data_rejects_non_numeric_actual_points_on_optimal_rows() -> None:
    round_results = pd.DataFrame(
        [
            {
                "rodada": 1,
                "strategy": "baseline",
                "solver_status": "Optimal",
                "actual_points": "ten",
                "formation": "4-3-3",
            },
            {
                "rodada": 1,
                "strategy": "random_forest",
                "solver_status": "Empty",
                "actual_points": "not checked",
                "formation": "",
            },
        ]
    )

    with pytest.raises(ValueError, match="Chart column 'actual_points' must be numeric for Optimal rows"):
        _prepare_chart_data(round_results)


def test_write_performance_chart_skips_empty_round_results(tmp_path: Path) -> None:
    chart_output = write_performance_chart(pd.DataFrame(), tmp_path)

    assert chart_output == ChartOutput(path=None, warnings=[])
    assert (tmp_path / "charts").is_dir()


def test_performance_figure_has_three_panels_without_zero_status_trace() -> None:
    round_results = pd.DataFrame(
        [
            {
                "rodada": 1,
                "strategy": "random_forest",
                "solver_status": "Optimal",
                "actual_points": 10.0,
                "formation": "4-3-3",
            },
            {
                "rodada": 2,
                "strategy": "random_forest",
                "solver_status": "TrainingEmpty",
                "actual_points": 0.0,
                "formation": "",
            },
            {
                "rodada": 1,
                "strategy": "baseline",
                "solver_status": "Optimal",
                "actual_points": 8.0,
                "formation": "4-3-3",
            },
        ]
    )

    figure = _build_performance_figure(_prepare_chart_data(round_results))

    subplot_titles = [annotation.text for annotation in figure.layout.annotations]
    assert subplot_titles[:3] == [
        "Cumulative actual points",
        "Per-round actual points",
        "Random forest formation by round",
    ]
    assert "TrainingEmpty (1)" in subplot_titles
    assert any(trace.name == "random_forest cumulative" for trace in figure.data)
    assert any(trace.name == "random_forest per round" for trace in figure.data)
    assert any(trace.name == "random_forest formation" for trace in figure.data)
    assert not any(trace.name == "non-optimal status" and list(trace.y) == [0] for trace in figure.data)


def test_write_performance_chart_writes_standalone_html(tmp_path: Path) -> None:
    round_results = pd.DataFrame(
        [
            {
                "rodada": 1,
                "strategy": "random_forest",
                "solver_status": "Optimal",
                "actual_points": 10.0,
                "formation": "4-3-3",
            },
            {
                "rodada": 2,
                "strategy": "random_forest",
                "solver_status": "Optimal",
                "actual_points": 12.0,
                "formation": "4-4-2",
            },
            {
                "rodada": 1,
                "strategy": "baseline",
                "solver_status": "Optimal",
                "actual_points": 8.0,
                "formation": "4-3-3",
            },
            {
                "rodada": 2,
                "strategy": "baseline",
                "solver_status": "Infeasible",
                "actual_points": 0.0,
                "formation": "",
            },
        ]
    )

    chart_output = write_performance_chart(round_results, tmp_path)

    expected_path = tmp_path / "charts" / "strategy_performance_by_round.html"
    assert chart_output == ChartOutput(path=expected_path, warnings=[])
    assert expected_path.is_file()
    html = expected_path.read_text(encoding="utf-8")
    assert "Plotly.newPlot" in html
    assert "random_forest" in html
    assert "baseline" in html
    assert "https://cdn.plot.ly" not in html
