# Backtest Rich CLI Output Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add Rich terminal summaries and one interactive Plotly HTML performance chart to the backtest CLI without changing scoring, optimization, feature generation, or CSV/JSON report semantics.

**Architecture:** Keep `src/cartola/backtesting/cli.py` thin and move display/chart logic into a focused `src/cartola/backtesting/cli_output.py` module. `run_backtest()` remains the only report writer; the CLI output module reads the in-memory `BacktestResult`, writes only the optional HTML chart under `<output_path>/charts/`, and prints Rich panels/tables.

**Tech Stack:** Python 3.13, pandas, Rich, Plotly, uv, pytest.

---

## File Structure

- Modify `pyproject.toml`: add Plotly as a direct runtime dependency.
- Modify `uv.lock`: refresh lockfile after adding Plotly directly.
- Create `src/cartola/backtesting/cli_output.py`: Rich rendering, chart-data preparation, Plotly chart export, chart warning handling.
- Modify `src/cartola/backtesting/cli.py`: call `render_backtest_success(...)` instead of plain `print(...)`.
- Create `src/tests/backtesting/test_cli_output.py`: focused unit tests for chart data, chart export, schema failures, non-fatal chart warnings, and Rich output fragments.
- Modify `src/tests/backtesting/test_cli.py`: update existing CLI tests that currently expect plain `WARNING:` output.

## Task 1: Add Plotly As A Direct Dependency

**Files:**
- Modify: `pyproject.toml`
- Modify: `uv.lock`

- [ ] **Step 1: Add Plotly through uv**

Run:

```bash
cd /Users/aaat/projects/caRtola/.worktrees/backtest-rich-cli-output
uv add "plotly>=6.7,<7"
```

Expected:

- `pyproject.toml` gains a direct `plotly>=6.7,<7` dependency.
- `uv.lock` is updated.

- [ ] **Step 2: Verify frozen import works**

Run:

```bash
uv run --frozen python -c "import plotly; print(plotly.__version__)"
```

Expected:

```text
6.7.0
```

Any Plotly 6.7.x version is acceptable if the lock resolves a patch release.

- [ ] **Step 3: Commit dependency change**

Run:

```bash
git add pyproject.toml uv.lock
git commit -m "build: add plotly dependency"
```

Expected:

```text
[dev/backtest-rich-cli-output <hash>] build: add plotly dependency
```

## Task 2: Build Chart Data Preparation With Strict Semantics

**Files:**
- Create: `src/cartola/backtesting/cli_output.py`
- Create: `src/tests/backtesting/test_cli_output.py`

- [ ] **Step 1: Write failing tests for chart-data preparation**

Create `src/tests/backtesting/test_cli_output.py` with:

```python
from pathlib import Path

import pandas as pd
import pytest

from cartola.backtesting.cli_output import (
    CHART_FILENAME,
    ChartOutput,
    _prepare_chart_data,
    write_performance_chart,
)


def _round_results() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "rodada": 5,
                "strategy": "random_forest",
                "solver_status": "Optimal",
                "actual_points": 60.0,
                "formation": "4-3-3",
                "captain_name": "Pedro",
                "budget_used": 99.5,
            },
            {
                "rodada": 6,
                "strategy": "random_forest",
                "solver_status": "Optimal",
                "actual_points": 70.0,
                "formation": "5-3-2",
                "captain_name": "Arrascaeta",
                "budget_used": 98.0,
            },
            {
                "rodada": 6,
                "strategy": "baseline",
                "solver_status": "Optimal",
                "actual_points": 55.0,
                "formation": "4-3-3",
                "captain_name": "Kaio Jorge",
                "budget_used": 97.0,
            },
            {
                "rodada": 7,
                "strategy": "random_forest",
                "solver_status": "TrainingEmpty",
                "actual_points": 0.0,
                "formation": "",
                "captain_name": None,
                "budget_used": 0.0,
            },
            {
                "rodada": 7,
                "strategy": "baseline",
                "solver_status": "TrainingEmpty",
                "actual_points": 0.0,
                "formation": "",
                "captain_name": None,
                "budget_used": 0.0,
            },
        ]
    )


def test_prepare_chart_data_filters_skipped_rows_from_score_traces() -> None:
    chart_data = _prepare_chart_data(_round_results())

    assert set(chart_data.optimal["solver_status"]) == {"Optimal"}
    assert set(chart_data.status["solver_status"]) == {"TrainingEmpty"}
    rf_cumulative = chart_data.cumulative.loc[chart_data.cumulative["strategy"].eq("random_forest")]
    assert rf_cumulative["actual_points"].tolist() == [60.0, 70.0]
    assert rf_cumulative["cumulative_actual_points"].tolist() == [60.0, 130.0]
    assert chart_data.formations["formation"].tolist() == ["4-3-3", "5-3-2"]


def test_prepare_chart_data_accepts_all_non_optimal_rows_without_fake_zero_lines() -> None:
    frame = pd.DataFrame(
        [
            {
                "rodada": 5,
                "strategy": "random_forest",
                "solver_status": "TrainingEmpty",
                "actual_points": 0.0,
                "formation": "",
            }
        ]
    )

    chart_data = _prepare_chart_data(frame)

    assert chart_data.optimal.empty
    assert chart_data.cumulative.empty
    assert chart_data.formations.empty
    assert chart_data.status["solver_status"].tolist() == ["TrainingEmpty"]


def test_prepare_chart_data_rejects_missing_columns() -> None:
    frame = _round_results().drop(columns=["solver_status"])

    with pytest.raises(ValueError, match="Missing chart columns: solver_status"):
        _prepare_chart_data(frame)


def test_prepare_chart_data_rejects_non_numeric_round() -> None:
    frame = _round_results()
    frame.loc[0, "rodada"] = "round-five"

    with pytest.raises(ValueError, match="Chart column 'rodada' must be numeric"):
        _prepare_chart_data(frame)


def test_prepare_chart_data_rejects_non_numeric_actual_points_on_optimal_rows() -> None:
    frame = _round_results()
    frame.loc[0, "actual_points"] = "not-a-number"

    with pytest.raises(ValueError, match="Chart column 'actual_points' must be numeric for Optimal rows"):
        _prepare_chart_data(frame)


def test_write_performance_chart_skips_empty_round_results(tmp_path: Path) -> None:
    output = write_performance_chart(pd.DataFrame(), tmp_path)

    assert output == ChartOutput(path=None, warnings=[])
    assert (tmp_path / "charts").is_dir()


def test_write_performance_chart_writes_standalone_html(tmp_path: Path) -> None:
    output = write_performance_chart(_round_results(), tmp_path)

    expected_path = tmp_path / "charts" / CHART_FILENAME
    assert output.path == expected_path
    assert output.warnings == []
    html = expected_path.read_text(encoding="utf-8")
    assert "Plotly.newPlot" in html
    assert "random_forest" in html
    assert "baseline" in html
    assert "4-3-3" in html
    assert "5-3-2" in html
    assert "TrainingEmpty" in html
    assert "https://cdn.plot.ly" not in html
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
uv run --frozen pytest src/tests/backtesting/test_cli_output.py -q
```

Expected failure:

```text
ModuleNotFoundError: No module named 'cartola.backtesting.cli_output'
```

- [ ] **Step 3: Implement chart-data preparation and HTML export**

Create `src/cartola/backtesting/cli_output.py` with:

```python
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import pandas as pd
from plotly import graph_objects as go
from plotly.subplots import make_subplots

CHART_FILENAME = "strategy_performance_by_round.html"
CHART_REQUIRED_COLUMNS: tuple[str, ...] = ("rodada", "strategy", "solver_status", "actual_points", "formation")
STRATEGY_COLORS: dict[str, str] = {
    "random_forest": "#2563eb",
    "baseline": "#16a34a",
    "price": "#dc2626",
}
STRATEGY_ORDER: tuple[str, ...] = ("random_forest", "baseline", "price")


@dataclass(frozen=True)
class ChartData:
    optimal: pd.DataFrame
    cumulative: pd.DataFrame
    formations: pd.DataFrame
    status: pd.DataFrame


@dataclass(frozen=True)
class ChartOutput:
    path: Path | None
    warnings: list[str]


def _missing_columns(frame: pd.DataFrame, required: Sequence[str]) -> list[str]:
    return [column for column in required if column not in frame.columns]


def _prepare_chart_data(round_results: pd.DataFrame) -> ChartData:
    missing = _missing_columns(round_results, CHART_REQUIRED_COLUMNS)
    if missing:
        raise ValueError(f"Missing chart columns: {', '.join(missing)}")

    chart = round_results.copy()
    try:
        chart["rodada"] = pd.to_numeric(chart["rodada"], errors="raise").astype(int)
    except (TypeError, ValueError) as exc:
        raise ValueError("Chart column 'rodada' must be numeric") from exc

    optimal_mask = chart["solver_status"].eq("Optimal")
    try:
        chart.loc[optimal_mask, "actual_points"] = pd.to_numeric(
            chart.loc[optimal_mask, "actual_points"],
            errors="raise",
        ).astype(float)
    except (TypeError, ValueError) as exc:
        raise ValueError("Chart column 'actual_points' must be numeric for Optimal rows") from exc

    chart = chart.sort_values(["strategy", "rodada"], kind="mergesort").reset_index(drop=True)
    optimal = chart.loc[optimal_mask].copy()
    status = chart.loc[~optimal_mask].copy()

    if optimal.empty:
        cumulative = optimal.copy()
    else:
        cumulative = optimal.copy()
        cumulative["cumulative_actual_points"] = cumulative.groupby("strategy", sort=False)["actual_points"].cumsum()

    formations = optimal.loc[
        optimal["strategy"].eq("random_forest") & optimal["formation"].astype(str).ne("")
    ].copy()
    return ChartData(optimal=optimal, cumulative=cumulative, formations=formations, status=status)


def _strategy_sort_key(strategy: str) -> tuple[int, str]:
    if strategy in STRATEGY_ORDER:
        return (STRATEGY_ORDER.index(strategy), strategy)
    return (len(STRATEGY_ORDER), strategy)


def _build_performance_figure(chart_data: ChartData) -> go.Figure:
    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=(
            "Cumulative Actual Points By Strategy",
            "Actual Points By Round",
            "Random Forest Formation By Round",
        ),
    )

    strategies = sorted(chart_data.optimal["strategy"].astype(str).unique().tolist(), key=_strategy_sort_key)
    for strategy in strategies:
        color = STRATEGY_COLORS.get(strategy, "#6b7280")
        cumulative = chart_data.cumulative.loc[chart_data.cumulative["strategy"].eq(strategy)]
        per_round = chart_data.optimal.loc[chart_data.optimal["strategy"].eq(strategy)]

        fig.add_trace(
            go.Scatter(
                x=cumulative["rodada"],
                y=cumulative["cumulative_actual_points"],
                mode="lines+markers",
                name=strategy,
                legendgroup=strategy,
                line={"color": color, "width": 2},
                customdata=cumulative[["strategy", "actual_points"]],
                hovertemplate=(
                    "Round %{x}<br>"
                    "Strategy %{customdata[0]}<br>"
                    "Round actual %{customdata[1]:.2f}<br>"
                    "Cumulative %{y:.2f}<extra></extra>"
                ),
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=per_round["rodada"],
                y=per_round["actual_points"],
                mode="lines+markers",
                name=f"{strategy} round",
                legendgroup=strategy,
                showlegend=False,
                line={"color": color, "width": 2},
                customdata=per_round[["strategy", "formation"]],
                hovertemplate=(
                    "Round %{x}<br>"
                    "Strategy %{customdata[0]}<br>"
                    "Formation %{customdata[1]}<br>"
                    "Actual %{y:.2f}<extra></extra>"
                ),
            ),
            row=2,
            col=1,
        )

    if not chart_data.formations.empty:
        formation_hover_columns = ["formation", "actual_points"]
        if "captain_name" in chart_data.formations.columns:
            formation_hover_columns.append("captain_name")
        if "budget_used" in chart_data.formations.columns:
            formation_hover_columns.append("budget_used")
        customdata = chart_data.formations[formation_hover_columns]
        hover_lines = [
            "Round %{x}",
            "Formation %{customdata[0]}",
            "Actual %{customdata[1]:.2f}",
        ]
        if "captain_name" in formation_hover_columns:
            hover_lines.append("Captain %{customdata[" + str(formation_hover_columns.index("captain_name")) + "]}")
        if "budget_used" in formation_hover_columns:
            hover_lines.append("Budget %{customdata[" + str(formation_hover_columns.index("budget_used")) + "]:.2f}")
        fig.add_trace(
            go.Scatter(
                x=chart_data.formations["rodada"],
                y=chart_data.formations["formation"].astype(str),
                mode="markers+text",
                name="random_forest formation",
                marker={"color": "#111827", "size": 10, "symbol": "diamond"},
                text=chart_data.formations["formation"].astype(str),
                textposition="top center",
                customdata=customdata,
                hovertemplate="<br>".join(hover_lines) + "<extra></extra>",
            ),
            row=3,
            col=1,
        )
    elif chart_data.optimal.empty:
        fig.add_annotation(
            text="No optimized rounds available for score or formation traces.",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
        )

    if not chart_data.status.empty:
        status_text = (
            chart_data.status.assign(
                label=lambda frame: frame["rodada"].astype(str)
                + " / "
                + frame["strategy"].astype(str)
                + " / "
                + frame["solver_status"].astype(str)
            )["label"]
            .drop_duplicates()
            .tolist()
        )
        if status_text:
            fig.add_annotation(
                text="Skipped/non-optimal rows: " + "; ".join(status_text[:8]),
                xref="paper",
                yref="paper",
                x=0,
                y=-0.12,
                xanchor="left",
                yanchor="top",
                showarrow=False,
                font={"size": 11, "color": "#6b7280"},
            )

    fig.update_layout(
        template="plotly_white",
        height=900,
        width=1200,
        title="Backtest Strategy Performance By Round",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
        margin={"l": 70, "r": 40, "t": 100, "b": 110},
    )
    fig.update_xaxes(title_text="Round", row=3, col=1)
    fig.update_yaxes(title_text="Cumulative points", row=1, col=1)
    fig.update_yaxes(title_text="Actual points", row=2, col=1)
    fig.update_yaxes(title_text="Formation", row=3, col=1, type="category")
    return fig


def _write_plotly_html(fig: go.Figure, path: Path) -> None:
    fig.write_html(path, include_plotlyjs=True, full_html=True)


def write_performance_chart(round_results: pd.DataFrame, output_path: Path) -> ChartOutput:
    charts_dir = output_path / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)

    if round_results.empty:
        return ChartOutput(path=None, warnings=[])

    chart_data = _prepare_chart_data(round_results)
    chart_path = charts_dir / CHART_FILENAME
    try:
        fig = _build_performance_figure(chart_data)
        _write_plotly_html(fig, chart_path)
    except (ImportError, OSError, RuntimeError) as exc:
        return ChartOutput(path=None, warnings=[f"Performance chart was not written: {exc}"])
    return ChartOutput(path=chart_path, warnings=[])
```

- [ ] **Step 4: Run chart tests**

Run:

```bash
uv run --frozen pytest src/tests/backtesting/test_cli_output.py -q
```

Expected:

```text
7 passed
```

- [ ] **Step 5: Commit chart module**

Run:

```bash
git add src/cartola/backtesting/cli_output.py src/tests/backtesting/test_cli_output.py
git commit -m "feat: generate interactive backtest chart"
```

Expected:

```text
[dev/backtest-rich-cli-output <hash>] feat: generate interactive backtest chart
```

## Task 3: Add Non-Fatal Chart Warning Coverage

**Files:**
- Modify: `src/tests/backtesting/test_cli_output.py`
- Modify: `src/cartola/backtesting/cli_output.py`

- [ ] **Step 1: Add failing tests for non-fatal chart write failures**

Append to `src/tests/backtesting/test_cli_output.py`:

```python
def test_write_performance_chart_returns_warning_for_plotly_write_failure(monkeypatch, tmp_path: Path) -> None:
    def fail_write(_fig, _path: Path) -> None:
        raise OSError("disk full")

    monkeypatch.setattr("cartola.backtesting.cli_output._write_plotly_html", fail_write)

    output = write_performance_chart(_round_results(), tmp_path)

    assert output.path is None
    assert output.warnings == ["Performance chart was not written: disk full"]
    assert (tmp_path / "charts").is_dir()
```

- [ ] **Step 2: Run the test to verify it fails if Task 2 did not already cover it**

Run:

```bash
uv run --frozen pytest src/tests/backtesting/test_cli_output.py::test_write_performance_chart_returns_warning_for_plotly_write_failure -q
```

Expected before implementation:

```text
FAILED
```

If Task 2 implementation already passes this test, record that in the task notes and continue.

- [ ] **Step 3: Ensure implementation catches best-effort export errors only**

Verify `write_performance_chart(...)` catches:

```python
except (ImportError, OSError, RuntimeError) as exc:
    return ChartOutput(path=None, warnings=[f"Performance chart was not written: {exc}"])
```

Do not catch `ValueError`; missing columns and invalid numeric data must still fail.

- [ ] **Step 4: Run focused chart tests**

Run:

```bash
uv run --frozen pytest src/tests/backtesting/test_cli_output.py -q
```

Expected:

```text
8 passed
```

- [ ] **Step 5: Commit warning behavior**

Run:

```bash
git add src/cartola/backtesting/cli_output.py src/tests/backtesting/test_cli_output.py
git commit -m "test: cover chart warning behavior"
```

Expected:

```text
[dev/backtest-rich-cli-output <hash>] test: cover chart warning behavior
```

## Task 4: Render Rich Backtest Output

**Files:**
- Modify: `src/cartola/backtesting/cli_output.py`
- Modify: `src/tests/backtesting/test_cli_output.py`

- [ ] **Step 1: Add failing tests for Rich output fragments**

Add this import near the top of `src/tests/backtesting/test_cli_output.py` with the other imports:

```python
from rich.console import Console
```

Add these imports near the existing `cartola.backtesting.cli_output` imports:

```python
from cartola.backtesting.config import BacktestConfig
from cartola.backtesting.runner import BacktestMetadata, BacktestResult
from cartola.backtesting.scoring_contract import contract_fields
from cartola.backtesting.cli_output import render_backtest_success
```

Append these helpers and tests below the existing chart tests:

```python


def _metadata(config: BacktestConfig, *, warnings: list[str] | None = None) -> BacktestMetadata:
    contract = contract_fields()
    return BacktestMetadata(
        season=config.season,
        start_round=config.start_round,
        max_round=38,
        cache_enabled=True,
        prediction_frames_built=38,
        wall_clock_seconds=76.87,
        backtest_jobs=config.jobs,
        backtest_workers_effective=config.jobs,
        model_n_jobs_effective=1 if config.jobs > 1 else -1,
        parallel_backend="threads" if config.jobs > 1 else "sequential",
        thread_env={
            "OMP_NUM_THREADS": None,
            "MKL_NUM_THREADS": None,
            "OPENBLAS_NUM_THREADS": None,
            "BLIS_NUM_THREADS": None,
        },
        scoring_contract_version=str(contract["scoring_contract_version"]),
        captain_scoring_enabled=bool(contract["captain_scoring_enabled"]),
        captain_multiplier=float(contract["captain_multiplier"]),
        formation_search=str(contract["formation_search"]),
        fixture_mode=config.fixture_mode,
        strict_alignment_policy=config.strict_alignment_policy,
        matchup_context_mode=config.matchup_context_mode,
        matchup_context_feature_columns=[],
        fixture_source_directory=None,
        fixture_manifest_paths=[],
        fixture_manifest_sha256={},
        generator_versions=[],
        excluded_rounds=[],
        warnings=[] if warnings is None else warnings,
        footystats_mode=config.footystats_mode,
        footystats_evaluation_scope=config.footystats_evaluation_scope,
        footystats_league_slug=config.footystats_league_slug,
        footystats_matches_source_path=None,
        footystats_matches_source_sha256=None,
        footystats_feature_columns=[],
        footystats_missing_join_keys_by_round={},
        footystats_duplicate_join_keys_by_round={},
        footystats_extra_club_rows_by_round={},
    )


def _summary() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "strategy": "random_forest",
                "rounds": 34,
                "total_actual_points": 2272.64,
                "average_actual_points": 66.8423529412,
                "total_predicted_points": 2805.7985547558,
                "actual_points_delta_vs_price": 603.74,
            },
            {
                "strategy": "price",
                "rounds": 34,
                "total_actual_points": 1668.90,
                "average_actual_points": 49.0852941176,
                "total_predicted_points": 3730.12,
                "actual_points_delta_vs_price": 0.0,
            },
        ]
    )


def test_render_backtest_success_prints_strategy_and_run_details(tmp_path: Path) -> None:
    config = BacktestConfig(
        season=2025,
        start_round=5,
        project_root=tmp_path,
        output_root=Path("reports"),
        fixture_mode="exploratory",
        footystats_mode="ppg",
        matchup_context_mode="cartola_matchup_v1",
        jobs=12,
    )
    result = BacktestResult(
        round_results=pd.DataFrame(),
        selected_players=pd.DataFrame(),
        player_predictions=pd.DataFrame(),
        summary=_summary(),
        diagnostics=pd.DataFrame(),
        metadata=_metadata(config),
    )
    console = Console(record=True, width=140)

    render_backtest_success(
        console,
        config=config,
        result=result,
        chart_output=ChartOutput(path=config.output_path / "charts" / CHART_FILENAME, warnings=[]),
    )

    output = console.export_text()
    assert "Backtest complete" in output
    assert "Strategy results" in output
    assert "random_forest" in output
    assert "66.84" in output
    assert "+603.74" in output
    assert "Run details" in output
    assert "Fixture mode" in output
    assert "exploratory" in output
    assert "FootyStats mode" in output
    assert "ppg" in output
    assert "Matchup context mode" in output
    assert "cartola_matchup_v1" in output
    assert "Jobs requested" in output
    assert "12" in output
    assert "Workers effective" in output
    assert "Parallel backend" in output
    assert "threads" in output
    assert "Performance chart" in output
    assert "strategy_performance_by_round.html" in output


def test_render_backtest_success_prints_warning_panel_without_plain_warning(tmp_path: Path) -> None:
    config = BacktestConfig(project_root=tmp_path)
    result = BacktestResult(
        round_results=pd.DataFrame(),
        selected_players=pd.DataFrame(),
        player_predictions=pd.DataFrame(),
        summary=_summary(),
        diagnostics=pd.DataFrame(),
        metadata=_metadata(config, warnings=["Exploratory fixture mode warning."]),
    )
    console = Console(record=True, width=140)

    render_backtest_success(
        console,
        config=config,
        result=result,
        chart_output=ChartOutput(path=None, warnings=["Performance chart was not written: disk full"]),
    )

    output = console.export_text()
    assert "Backtest warnings" in output
    assert "Exploratory fixture mode warning." in output
    assert "Performance chart was not written: disk full" in output
    assert "WARNING:" not in output
    assert "Performance chart" in output
    assert "n/a" in output
```

- [ ] **Step 2: Run Rich output tests to verify they fail**

Run:

```bash
uv run --frozen pytest src/tests/backtesting/test_cli_output.py::test_render_backtest_success_prints_strategy_and_run_details src/tests/backtesting/test_cli_output.py::test_render_backtest_success_prints_warning_panel_without_plain_warning -q
```

Expected failure:

```text
ImportError: cannot import name 'render_backtest_success'
```

- [ ] **Step 3: Implement Rich output helpers**

Add these imports near the top of `src/cartola/backtesting/cli_output.py`:

```python
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from cartola.backtesting.config import BacktestConfig
from cartola.backtesting.runner import BacktestResult
```

Append these helper functions below the chart-writing helpers:

```python


def _format_number(value: object, *, signed: bool = False) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    numeric = float(value)
    return f"{numeric:+.2f}" if signed else f"{numeric:.2f}"


def _format_int(value: object) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return str(int(value))


def _relative_path(path: Path | None, project_root: Path) -> str:
    if path is None:
        return "n/a"
    try:
        return str(path.relative_to(project_root))
    except ValueError:
        return str(path)


def _metadata_value(result: BacktestResult, name: str) -> object:
    return getattr(result.metadata, name)


def _warnings_panel(warnings: list[str]) -> Panel:
    return Panel("\n".join(warnings), title="Backtest warnings", border_style="yellow")


def _strategy_table(summary: pd.DataFrame) -> Table:
    table = Table(title="Strategy results", show_header=True, header_style="bold")
    table.add_column("Strategy", style="cyan", no_wrap=True)
    table.add_column("Rounds", justify="right")
    table.add_column("Total actual", justify="right")
    table.add_column("Avg actual", justify="right")
    table.add_column("Total predicted", justify="right")
    table.add_column("Vs price", justify="right")
    for row in summary.to_dict("records"):
        table.add_row(
            str(row.get("strategy", "n/a")),
            _format_int(row.get("rounds")),
            _format_number(row.get("total_actual_points")),
            _format_number(row.get("average_actual_points")),
            _format_number(row.get("total_predicted_points")),
            _format_number(row.get("actual_points_delta_vs_price"), signed=True),
        )
    return table


def _run_details_table(config: BacktestConfig, result: BacktestResult, chart_output: ChartOutput) -> Table:
    table = Table(title="Run details", show_header=True, header_style="bold")
    table.add_column("Field", style="cyan", no_wrap=True)
    table.add_column("Value", overflow="fold")
    chart_display = _relative_path(chart_output.path, config.project_root)
    rows = [
        ("Output", _relative_path(config.output_path, config.project_root)),
        ("Fixture mode", config.fixture_mode),
        ("FootyStats mode", config.footystats_mode),
        ("Matchup context mode", config.matchup_context_mode),
        ("Jobs requested", str(result.metadata.backtest_jobs)),
        ("Workers effective", str(result.metadata.backtest_workers_effective)),
        ("Parallel backend", str(result.metadata.parallel_backend)),
        ("Model n_jobs", str(result.metadata.model_n_jobs_effective)),
        ("Prediction frames built", str(result.metadata.prediction_frames_built)),
        ("Wall clock seconds", _format_number(result.metadata.wall_clock_seconds)),
        ("Performance chart", chart_display),
        ("Scoring contract", str(result.metadata.scoring_contract_version)),
    ]
    for field, value in rows:
        table.add_row(field, str(value))
    return table


def render_backtest_success(
    console: Console,
    *,
    config: BacktestConfig,
    result: BacktestResult,
    chart_output: ChartOutput,
) -> None:
    warnings = [*result.metadata.warnings, *chart_output.warnings]
    if warnings:
        console.print(_warnings_panel(warnings))
    console.print(
        Panel(
            f"season={config.season}  start_round={config.start_round}  output={_relative_path(config.output_path, config.project_root)}",
            title="Backtest complete",
            border_style="green",
        )
    )
    console.print(_strategy_table(result.summary))
    console.print(_run_details_table(config, result, chart_output))
```

- [ ] **Step 4: Run Rich output tests**

Run:

```bash
uv run --frozen pytest src/tests/backtesting/test_cli_output.py -q
```

Expected:

```text
10 passed
```

- [ ] **Step 5: Commit Rich rendering**

Run:

```bash
git add src/cartola/backtesting/cli_output.py src/tests/backtesting/test_cli_output.py
git commit -m "feat: render rich backtest summary"
```

Expected:

```text
[dev/backtest-rich-cli-output <hash>] feat: render rich backtest summary
```

## Task 5: Integrate Rich Output Into The CLI

**Files:**
- Modify: `src/cartola/backtesting/cli.py`
- Modify: `src/tests/backtesting/test_cli.py`

- [ ] **Step 1: Update failing CLI tests for new output contract**

In `src/tests/backtesting/test_cli.py`, update `test_main_prints_metadata_warnings` from plain `WARNING:` assertions to:

```python
def test_main_prints_metadata_warnings(monkeypatch, capsys, tmp_path):
    def fake_run_backtest(config: BacktestConfig) -> BacktestResult:
        return BacktestResult(
            round_results=pd.DataFrame(),
            selected_players=pd.DataFrame(),
            player_predictions=pd.DataFrame(),
            summary=pd.DataFrame(),
            diagnostics=pd.DataFrame(),
            metadata=_metadata_for_config(config, warnings=["first warning", "second warning"]),
        )

    monkeypatch.setattr("cartola.backtesting.cli.run_backtest", fake_run_backtest)

    exit_code = main(["--project-root", str(tmp_path)])

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "Backtest warnings" in output
    assert "first warning" in output
    assert "second warning" in output
    assert "WARNING:" not in output
    assert "Backtest complete" in output
```

Add a CLI integration test with non-empty summary/round results:

```python
def test_main_prints_rich_summary_and_writes_chart(monkeypatch, capsys, tmp_path):
    def fake_run_backtest(config: BacktestConfig) -> BacktestResult:
        return BacktestResult(
            round_results=pd.DataFrame(
                [
                    {
                        "rodada": 5,
                        "strategy": "random_forest",
                        "solver_status": "Optimal",
                        "formation": "4-3-3",
                        "selected_count": 12,
                        "budget_used": 99.5,
                        "predicted_points": 80.0,
                        "predicted_points_base": 75.0,
                        "captain_bonus_predicted": 5.0,
                        "predicted_points_with_captain": 80.0,
                        "actual_points": 70.0,
                        "actual_points_base": 66.0,
                        "captain_bonus_actual": 4.0,
                        "actual_points_with_captain": 70.0,
                        "captain_id": 1,
                        "captain_name": "Pedro",
                        "captain_policy_ev_id": 1,
                        "captain_policy_safe_id": 1,
                        "captain_policy_upside_id": 1,
                        "actual_points_with_ev_captain": 70.0,
                        "actual_points_with_safe_captain": 70.0,
                        "actual_points_with_upside_captain": 70.0,
                    },
                    {
                        "rodada": 5,
                        "strategy": "price",
                        "solver_status": "Optimal",
                        "formation": "3-4-3",
                        "selected_count": 12,
                        "budget_used": 100.0,
                        "predicted_points": 100.0,
                        "predicted_points_base": 95.0,
                        "captain_bonus_predicted": 5.0,
                        "predicted_points_with_captain": 100.0,
                        "actual_points": 40.0,
                        "actual_points_base": 38.0,
                        "captain_bonus_actual": 2.0,
                        "actual_points_with_captain": 40.0,
                        "captain_id": 2,
                        "captain_name": "Arrascaeta",
                        "captain_policy_ev_id": 2,
                        "captain_policy_safe_id": 2,
                        "captain_policy_upside_id": 2,
                        "actual_points_with_ev_captain": 40.0,
                        "actual_points_with_safe_captain": 40.0,
                        "actual_points_with_upside_captain": 40.0,
                    },
                ]
            ),
            selected_players=pd.DataFrame(),
            player_predictions=pd.DataFrame(),
            summary=pd.DataFrame(
                [
                    {
                        "strategy": "random_forest",
                        "rounds": 1,
                        "total_actual_points": 70.0,
                        "average_actual_points": 70.0,
                        "total_predicted_points": 80.0,
                        "actual_points_delta_vs_price": 30.0,
                    },
                    {
                        "strategy": "price",
                        "rounds": 1,
                        "total_actual_points": 40.0,
                        "average_actual_points": 40.0,
                        "total_predicted_points": 100.0,
                        "actual_points_delta_vs_price": 0.0,
                    },
                ]
            ),
            diagnostics=pd.DataFrame(),
            metadata=_metadata_for_config(config),
        )

    monkeypatch.setattr("cartola.backtesting.cli.run_backtest", fake_run_backtest)

    exit_code = main(["--project-root", str(tmp_path), "--jobs", "4"])

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "Backtest complete" in output
    assert "Strategy results" in output
    assert "random_forest" in output
    assert "+30.00" in output
    assert "Run details" in output
    assert "Performance chart" in output
    chart_path = tmp_path / "data" / "08_reporting" / "backtests" / "2025" / "charts" / "strategy_performance_by_round.html"
    assert chart_path.exists()
```

- [ ] **Step 2: Run CLI tests to verify they fail**

Run:

```bash
uv run --frozen pytest src/tests/backtesting/test_cli.py::test_main_prints_metadata_warnings src/tests/backtesting/test_cli.py::test_main_prints_rich_summary_and_writes_chart -q
```

Expected before CLI integration:

```text
FAILED
```

- [ ] **Step 3: Update CLI to call output helpers**

Modify `src/cartola/backtesting/cli.py`:

```python
from rich.console import Console

from cartola.backtesting.cli_output import render_backtest_success, write_performance_chart
```

Replace the end of `main(...)` with:

```python
    result = run_backtest(config)
    chart_output = write_performance_chart(result.round_results, config.output_path)
    stdout = Console()
    render_backtest_success(stdout, config=config, result=result, chart_output=chart_output)
    return 0
```

Do not add broad exception handling around `run_backtest(...)`. Mandatory chart schema failures should propagate as `ValueError`; non-fatal chart export failures are handled inside `write_performance_chart(...)`.

- [ ] **Step 4: Run CLI tests**

Run:

```bash
uv run --frozen pytest src/tests/backtesting/test_cli.py src/tests/backtesting/test_cli_output.py -q
```

Expected:

```text
all tests passed
```

- [ ] **Step 5: Commit CLI integration**

Run:

```bash
git add src/cartola/backtesting/cli.py src/tests/backtesting/test_cli.py
git commit -m "feat: integrate rich backtest cli output"
```

Expected:

```text
[dev/backtest-rich-cli-output <hash>] feat: integrate rich backtest cli output
```

## Task 6: Validate Real Command Output

**Files:**
- No source changes expected.

- [ ] **Step 1: Run focused tests**

Run:

```bash
uv run --frozen pytest src/tests/backtesting/test_cli.py src/tests/backtesting/test_cli_output.py -q
```

Expected:

```text
all tests passed
```

- [ ] **Step 2: Run a small real backtest command**

Run:

```bash
uv run --frozen python -m cartola.backtesting.cli \
  --season 2025 \
  --start-round 36 \
  --budget 100 \
  --fixture-mode exploratory \
  --footystats-mode ppg \
  --matchup-context-mode cartola_matchup_v1 \
  --current-year 2026 \
  --jobs 2 \
  --output-root data/08_reporting/backtests/rich_cli_smoke
```

Expected terminal fragments:

```text
Backtest warnings
Backtest complete
Strategy results
Run details
Performance chart
strategy_performance_by_round.html
```

Expected file:

```text
data/08_reporting/backtests/rich_cli_smoke/2025/charts/strategy_performance_by_round.html
```

- [ ] **Step 3: Verify chart file content**

Run:

```bash
python - <<'PY'
from pathlib import Path
path = Path("data/08_reporting/backtests/rich_cli_smoke/2025/charts/strategy_performance_by_round.html")
text = path.read_text(encoding="utf-8")
assert "Plotly.newPlot" in text
assert "random_forest" in text
assert "https://cdn.plot.ly" not in text
print(path)
PY
```

Expected:

```text
data/08_reporting/backtests/rich_cli_smoke/2025/charts/strategy_performance_by_round.html
```

- [ ] **Step 4: Run full quality gate**

Run:

```bash
uv run --frozen scripts/pyrepo-check --all
```

Expected:

```text
ruff: All checks passed
ty: All checks passed
bandit: No issues identified
pytest: all tests passed
```

- [ ] **Step 5: Remove smoke output**

Run:

```bash
rm -rf data/08_reporting/backtests/rich_cli_smoke
git status --short
```

Expected:

```text
<no data output tracked>
```

Do not commit generated backtest output.

## Task 7: Final Review And Commit Any Test Adjustments

**Files:**
- Modify only if Task 6 reveals minor test/formatting issues.

- [ ] **Step 1: Inspect final diff**

Run:

```bash
git diff --stat
git diff -- pyproject.toml src/cartola/backtesting/cli.py src/cartola/backtesting/cli_output.py src/tests/backtesting/test_cli.py src/tests/backtesting/test_cli_output.py
```

Expected:

- Plotly direct dependency is present.
- `cli.py` delegates output to `cli_output.py`.
- Chart tests cover skipped rows, missing columns, non-numeric `rodada`, non-numeric `actual_points`, empty results, and non-fatal export failure.
- CLI tests assert Rich output fragments and no plain `WARNING:`.

- [ ] **Step 2: Commit any remaining changes**

If `git status --short` shows uncommitted source/test changes, run:

```bash
git add pyproject.toml uv.lock src/cartola/backtesting/cli.py src/cartola/backtesting/cli_output.py src/tests/backtesting/test_cli.py src/tests/backtesting/test_cli_output.py
git commit -m "test: verify rich backtest cli output"
```

Expected:

```text
[dev/backtest-rich-cli-output <hash>] test: verify rich backtest cli output
```

If there are no uncommitted changes, do not create an empty commit.

## Self-Review Checklist

- Spec coverage:
  - Rich success panel: Task 4 and Task 5.
  - Warning panel and no plain `WARNING:`: Task 4 and Task 5.
  - Strategy table: Task 4 and Task 5.
  - Run details table with chart path/runtime/jobs/backend: Task 4 and Task 5.
  - Interactive Plotly HTML chart: Task 2 and Task 6.
  - Offline HTML export with embedded Plotly.js: Task 2 and Task 6.
  - Skipped rows excluded from score traces: Task 2.
  - Mandatory schema/numeric failures: Task 2.
  - Non-fatal Plotly/filesystem export failures: Task 3 and Task 4.
  - Direct dependency plus lockfile update: Task 1.
  - No score/optimizer/report-schema semantic changes: all tasks.
- Completeness scan:
  - This plan contains only concrete implementation steps.
- Type consistency:
  - `ChartOutput`, `ChartData`, `write_performance_chart`, `_prepare_chart_data`, and `render_backtest_success` are defined before use.
  - Tests import the exact helper names defined in `cli_output.py`.
