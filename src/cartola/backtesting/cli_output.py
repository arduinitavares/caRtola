from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

if TYPE_CHECKING:
    from cartola.backtesting.config import BacktestConfig
    from cartola.backtesting.runner import BacktestResult

CHART_FILENAME = "strategy_performance_by_round.html"
_REQUIRED_CHART_COLUMNS = {"rodada", "strategy", "solver_status", "actual_points", "formation"}


@dataclass(frozen=True)
class ChartOutput:
    path: Path | None
    warnings: list[str]


@dataclass(frozen=True)
class PreparedChartData:
    score_rows: pd.DataFrame
    status_rows: pd.DataFrame
    formation_rows: pd.DataFrame


def render_backtest_success(
    console: Console,
    *,
    config: BacktestConfig,
    result: BacktestResult,
    chart_output: ChartOutput,
) -> None:
    warnings = [*result.metadata.warnings, *chart_output.warnings]
    if warnings:
        console.print(Panel("\n".join(warnings), title="Backtest warnings", border_style="yellow"))
    console.print(
        Panel(
            (
                f"season={config.season}  "
                f"start_round={config.start_round}  "
                f"output={_format_path(config.output_path, project_root=config.project_root)}"
            ),
            title="Backtest complete",
            border_style="green",
        )
    )
    console.print(_build_strategy_results_table(result.summary))
    console.print(_build_run_details_table(config=config, result=result, chart_output=chart_output))


def _build_strategy_results_table(summary: pd.DataFrame) -> Table:
    table = Table(title="Strategy results")
    table.add_column("Strategy")
    table.add_column("Rounds", justify="right")
    table.add_column("Total actual", justify="right")
    table.add_column("Avg actual", justify="right")
    table.add_column("Total predicted", justify="right")
    table.add_column("Vs price", justify="right")

    for _, row in summary.iterrows():
        table.add_row(
            _format_text(row.get("strategy")),
            _format_int(row.get("rounds")),
            _format_points(row.get("total_actual_points")),
            _format_points(row.get("average_actual_points")),
            _format_points(row.get("total_predicted_points")),
            _format_points(row.get("actual_points_delta_vs_price"), signed=True),
        )
    return table


def _build_run_details_table(
    *,
    config: BacktestConfig,
    result: BacktestResult,
    chart_output: ChartOutput,
) -> Table:
    metadata = result.metadata
    rows = [
        ("Output", _format_path(config.output_path, project_root=config.project_root)),
        ("Fixture mode", _format_text(getattr(metadata, "fixture_mode", None))),
        ("FootyStats mode", _format_text(getattr(metadata, "footystats_mode", None))),
        ("Matchup context mode", _format_text(getattr(metadata, "matchup_context_mode", None))),
        ("Jobs requested", _format_int(config.jobs)),
        ("Workers effective", _format_int(getattr(metadata, "backtest_workers_effective", None))),
        ("Parallel backend", _format_text(getattr(metadata, "parallel_backend", None))),
        ("Model n_jobs", _format_int(getattr(metadata, "model_n_jobs_effective", None))),
        ("Prediction frames built", _format_int(getattr(metadata, "prediction_frames_built", None))),
        ("Wall clock seconds", _format_points(getattr(metadata, "wall_clock_seconds", None))),
        ("Performance chart", _format_chart_path(chart_output.path, project_root=config.project_root)),
        ("Scoring contract", _format_text(getattr(metadata, "scoring_contract_version", None))),
    ]

    table = Table(title="Run details")
    table.add_column("Field")
    table.add_column("Value", overflow="fold")
    for label, value in rows:
        table.add_row(label, value)
    return table


def _format_text(value: Any) -> str:
    if _is_missing(value):
        return "n/a"
    return str(value)


def _format_int(value: Any) -> str:
    if _is_missing(value):
        return "n/a"
    try:
        return str(int(value))
    except (TypeError, ValueError):
        return "n/a"


def _format_points(value: Any, *, signed: bool = False) -> str:
    if _is_missing(value):
        return "n/a"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "n/a"
    return f"{number:+.2f}" if signed else f"{number:.2f}"


def _format_path(path: Path | None, *, project_root: Path) -> str:
    if path is None:
        return "n/a"
    try:
        return str(path.relative_to(project_root))
    except ValueError:
        return str(path)


def _format_chart_path(path: Path | None, *, project_root: Path) -> str:
    if path is None:
        return "n/a"
    formatted_path = _format_path(path, project_root=project_root)
    return f"{path.name} ({formatted_path})"


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _prepare_chart_data(round_results: pd.DataFrame) -> PreparedChartData:
    if round_results.empty:
        return PreparedChartData(
            score_rows=pd.DataFrame(),
            status_rows=pd.DataFrame(),
            formation_rows=pd.DataFrame(),
        )

    missing_columns = sorted(_REQUIRED_CHART_COLUMNS.difference(round_results.columns))
    if missing_columns:
        raise ValueError(f"Missing chart columns: {', '.join(missing_columns)}")

    chart_rows = round_results.loc[:, sorted(_REQUIRED_CHART_COLUMNS)].copy()
    numeric_rounds = pd.to_numeric(chart_rows["rodada"], errors="coerce")
    if numeric_rounds.isna().any():
        raise ValueError("Chart column 'rodada' must be numeric")
    chart_rows["rodada"] = numeric_rounds

    optimal_mask = chart_rows["solver_status"].eq("Optimal")
    numeric_actual_points = pd.to_numeric(chart_rows.loc[optimal_mask, "actual_points"], errors="coerce")
    if numeric_actual_points.isna().any():
        raise ValueError("Chart column 'actual_points' must be numeric for Optimal rows")
    chart_rows.loc[optimal_mask, "actual_points"] = numeric_actual_points

    score_rows = chart_rows.loc[optimal_mask].copy()
    if not score_rows.empty:
        score_rows["actual_points"] = score_rows["actual_points"].astype(float)
        score_rows = score_rows.sort_values(["strategy", "rodada"]).reset_index(drop=True)
        score_rows["cumulative_points"] = score_rows.groupby("strategy", sort=False)["actual_points"].cumsum()

    status_rows = chart_rows.loc[~optimal_mask].copy().reset_index(drop=True)

    formations = chart_rows["formation"].fillna("").astype(str).str.strip()
    formation_mask = optimal_mask & chart_rows["strategy"].eq("random_forest") & formations.ne("")
    formation_rows = (
        chart_rows.loc[formation_mask, ["rodada", "formation"]]
        .copy()
        .sort_values("rodada")
        .reset_index(drop=True)
    )

    return PreparedChartData(
        score_rows=score_rows,
        status_rows=status_rows,
        formation_rows=formation_rows,
    )


def write_performance_chart(round_results: pd.DataFrame, output_path: Path) -> ChartOutput:
    charts_path = output_path / "charts"
    try:
        charts_path.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        return ChartOutput(path=None, warnings=[_chart_unavailable_warning(exc)])

    if round_results.empty:
        return ChartOutput(path=None, warnings=[])

    chart_data = _prepare_chart_data(round_results)
    figure = _build_performance_figure(chart_data)
    chart_path = charts_path / CHART_FILENAME
    try:
        _write_plotly_html(figure, chart_path)
    except Exception as exc:
        return ChartOutput(path=None, warnings=[_chart_unavailable_warning(exc)])
    return ChartOutput(path=chart_path, warnings=[])


def _chart_unavailable_warning(exc: Exception) -> str:
    return f"Performance chart: n/a ({exc})"


def _write_plotly_html(figure: go.Figure, chart_path: Path) -> None:
    figure.write_html(chart_path, include_plotlyjs=True, full_html=True)
    _remove_plotly_cdn_defaults(chart_path)


def _build_performance_figure(chart_data: PreparedChartData) -> go.Figure:
    figure = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.46, 0.34, 0.20],
        vertical_spacing=0.10,
        subplot_titles=(
            "Cumulative actual points",
            "Per-round actual points",
            "Random forest formation by round",
        ),
    )

    for strategy, strategy_rows in chart_data.score_rows.groupby("strategy", sort=True):
        strategy_name = str(strategy)
        figure.add_trace(
            go.Scatter(
                x=strategy_rows["rodada"],
                y=strategy_rows["cumulative_points"],
                mode="lines+markers",
                name=f"{strategy_name} cumulative",
                customdata=strategy_rows["actual_points"],
                hovertemplate=(
                    "Strategy=%{fullData.name}<br>"
                    "Round=%{x}<br>"
                    "Actual=%{customdata:.2f}<br>"
                    "Cumulative=%{y:.2f}<extra></extra>"
                ),
            ),
            row=1,
            col=1,
        )
        figure.add_trace(
            go.Scatter(
                x=strategy_rows["rodada"],
                y=strategy_rows["actual_points"],
                mode="lines+markers",
                name=f"{strategy_name} per round",
                hovertemplate=(
                    f"Strategy={strategy_name}<br>"
                    "Round=%{x}<br>"
                    "Actual=%{y:.2f}<extra></extra>"
                ),
            ),
            row=2,
            col=1,
        )

    if not chart_data.formation_rows.empty:
        figure.add_trace(
            go.Scatter(
                x=chart_data.formation_rows["rodada"],
                y=chart_data.formation_rows["formation"],
                mode="markers+lines",
                name="random_forest formation",
                hovertemplate="Round=%{x}<br>Formation=%{y}<extra></extra>",
            ),
            row=3,
            col=1,
        )
    _add_status_annotations(figure, chart_data.status_rows)

    figure.update_layout(
        title="Backtest strategy performance by round",
        template="plotly_white",
        hovermode="x unified",
        legend_title_text="Series",
    )
    figure.update_xaxes(title_text="Round", row=3, col=1)
    figure.update_yaxes(title_text="Cumulative points", row=1, col=1)
    figure.update_yaxes(title_text="Round points", row=2, col=1)
    figure.update_yaxes(title_text="Formation", row=3, col=1, type="category")
    return figure


def _add_status_annotations(figure: go.Figure, status_rows: pd.DataFrame) -> None:
    if status_rows.empty:
        return
    status_counts = (
        status_rows.groupby(["rodada", "solver_status"], sort=True)
        .size()
        .reset_index(name="count")
        .sort_values(["rodada", "solver_status"])
    )
    for _, row in status_counts.iterrows():
        figure.add_annotation(
            x=row["rodada"],
            y=1.0,
            xref="x3",
            yref="paper",
            text=f"{row['solver_status']} ({row['count']})",
            showarrow=False,
            yanchor="bottom",
            font={"size": 10, "color": "#6b7280"},
        )


def _remove_plotly_cdn_defaults(chart_path: Path) -> None:
    html = chart_path.read_text(encoding="utf-8")
    if "https://cdn.plot.ly" not in html:
        return
    chart_path.write_text(html.replace("https://cdn.plot.ly", ""), encoding="utf-8")
