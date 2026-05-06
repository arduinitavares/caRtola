from __future__ import annotations

import html
import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from cartola.backtesting.budgeting import normalize_budget_policy

CONFIG_COLUMNS: tuple[str, str, str, str] = ("model_id", "feature_pack", "fixture_mode", "budget_policy")
SQUAD_REPORT_NAME = "squad_performance_comparison.html"
CALIBRATION_REPORT_NAME = "calibration_plots.html"

REQUIRED_SCHEMAS: dict[str, frozenset[str]] = {
    "ranked_summary.csv": frozenset(
        {
            "model_id",
            "feature_pack",
            "fixture_mode",
            "total_actual_points",
            "aggregate_delta",
            "average_actual_delta_per_round",
            "promotion_eligible",
            "promotion_reason",
        }
    ),
    "per_season_summary.csv": frozenset(
        {"season", "model_id", "feature_pack", "fixture_mode", "total_actual_points", "rounds"}
    ),
    "prediction_metrics.csv": frozenset(
        {
            "season",
            "model_id",
            "feature_pack",
            "fixture_mode",
            "metric_scope",
            "observed_count",
            "mae",
            "spearman",
            "calibration_slope",
        }
    ),
    "calibration_deciles.csv": frozenset(
        {"season", "model_id", "feature_pack", "fixture_mode", "decile", "row_count", "predicted_mean", "actual_mean"}
    ),
}
REQUIRED_JSON_FILES: tuple[str, ...] = ("comparability_report.json",)


@dataclass(frozen=True)
class ExperimentReportInputs:
    ranked_summary: pd.DataFrame
    per_season_summary: pd.DataFrame
    prediction_metrics: pd.DataFrame
    calibration_deciles: pd.DataFrame
    comparability_report: dict[str, object]


def build_experiment_html_reports(experiment_dir: Path) -> None:
    inputs, validation_errors = _load_report_inputs(experiment_dir)
    if validation_errors:
        _write_incomplete_reports(experiment_dir, validation_errors)
        return
    if inputs is None:
        _write_incomplete_reports(experiment_dir, ("internal validation error: report inputs were not loaded",))
        return

    _write_html(
        experiment_dir / SQUAD_REPORT_NAME,
        title="Squad performance comparison",
        body=_squad_report_body(inputs),
    )
    _write_html(
        experiment_dir / CALIBRATION_REPORT_NAME,
        title="Calibration plots",
        body=_calibration_report_body(inputs),
    )


def _load_report_inputs(experiment_dir: Path) -> tuple[ExperimentReportInputs | None, tuple[str, ...]]:
    errors: list[str] = []
    csv_frames: dict[str, pd.DataFrame] = {}
    for filename, required_columns in REQUIRED_SCHEMAS.items():
        path = experiment_dir / filename
        if not path.exists():
            errors.append(f"{filename}: missing file")
            continue
        frame = pd.read_csv(path)
        missing_columns = sorted(required_columns.difference(frame.columns))
        if missing_columns:
            errors.append(f"{filename}: missing columns: {', '.join(missing_columns)}")
            continue
        csv_frames[filename] = frame

    json_payloads: dict[str, dict[str, object]] = {}
    for filename in REQUIRED_JSON_FILES:
        path = experiment_dir / filename
        if not path.exists():
            errors.append(f"{filename}: missing file")
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            errors.append(f"{filename}: invalid JSON: {exc.msg}")
            continue
        if not isinstance(payload, dict):
            errors.append(f"{filename}: JSON root must be an object")
            continue
        json_payloads[filename] = payload

    if errors:
        return None, tuple(errors)

    return (
        ExperimentReportInputs(
            ranked_summary=_with_config_label(csv_frames["ranked_summary.csv"]),
            per_season_summary=_with_config_label(csv_frames["per_season_summary.csv"]),
            prediction_metrics=_with_config_label(csv_frames["prediction_metrics.csv"]),
            calibration_deciles=_with_config_label(csv_frames["calibration_deciles.csv"]),
            comparability_report=json_payloads["comparability_report.json"],
        ),
        (),
    )


def _with_config_label(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    if "budget_policy" not in result.columns:
        result["budget_policy"] = normalize_budget_policy(None)
    else:
        result["budget_policy"] = result["budget_policy"].map(normalize_budget_policy)
    result["config_label"] = (
        result["model_id"].astype(str)
        + " / "
        + result["feature_pack"].astype(str)
        + " / "
        + result["fixture_mode"].astype(str)
        + " / "
        + result["budget_policy"].astype(str)
    )
    return result


def _squad_report_body(inputs: ExperimentReportInputs) -> str:
    ranked = _ranked_for_display(inputs.ranked_summary)
    per_season = _sort_by_ranked_config(inputs.per_season_summary, ranked)
    config_labels = ranked["config_label"].astype(str).tolist()

    figure = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Captain-aware total actual points",
            "Persisted aggregate lift vs baseline",
            "Per-season total actual points",
            "Promotion guardrails",
        ),
        specs=[[{"type": "bar"}, {"type": "bar"}], [{"type": "bar"}, {"type": "table"}]],
        vertical_spacing=0.18,
        horizontal_spacing=0.08,
    )
    figure.add_trace(
        go.Bar(
            x=config_labels,
            y=_numeric_list(ranked["total_actual_points"]),
            marker_color="#2563eb",
            hovertemplate="%{x}<br>Total actual points=%{y:.2f}<extra></extra>",
        ),
        row=1,
        col=1,
    )
    figure.add_trace(
        go.Bar(
            x=config_labels,
            y=_numeric_list(ranked["aggregate_delta"]),
            marker_color="#059669",
            hovertemplate="%{x}<br>Aggregate lift=%{y:.2f}<extra></extra>",
        ),
        row=1,
        col=2,
    )
    for config_label in config_labels:
        rows = per_season[per_season["config_label"].eq(config_label)].sort_values("season")
        if rows.empty:
            continue
        figure.add_trace(
            go.Bar(
                name=config_label,
                x=rows["season"].astype(str).tolist(),
                y=_numeric_list(rows["total_actual_points"]),
                hovertemplate=f"{html.escape(config_label)}<br>Season=%{{x}}<br>Total=%{{y:.2f}}<extra></extra>",
            ),
            row=2,
            col=1,
        )
    table_frame = ranked[
        ["config_label", "total_actual_points", "aggregate_delta", "promotion_eligible", "promotion_reason"]
    ].copy()
    figure.add_trace(
        go.Table(
            header={"values": ["config", "total", "lift", "promotion_eligible", "promotion_reason"]},
            cells={
                "values": [
                    table_frame["config_label"].astype(str).tolist(),
                    _rounded_list(table_frame["total_actual_points"]),
                    _rounded_list(table_frame["aggregate_delta"]),
                    table_frame["promotion_eligible"].astype(str).tolist(),
                    table_frame["promotion_reason"].astype(str).tolist(),
                ]
            },
        ),
        row=2,
        col=2,
    )
    figure.update_layout(
        title="Model-feature experiment squad performance",
        template="plotly_white",
        height=1000,
        barmode="group",
        legend={"orientation": "h", "y": -0.18},
        margin={"l": 60, "r": 30, "t": 90, "b": 180},
    )
    figure.update_xaxes(tickangle=35)

    notes = [
        "Squad totals and lift are rendered from ranked_summary.csv; the HTML layer does not recompute baselines.",
        "Total actual points are the experiment's persisted captain-aware squad totals.",
        f"Comparability status: {inputs.comparability_report.get('status', 'unknown')}",
    ]
    return _report_intro(notes) + _figure_html(figure) + _data_table_html(table_frame)


def _calibration_report_body(inputs: ExperimentReportInputs) -> str:
    ranked = _ranked_for_display(inputs.ranked_summary)
    calibration = _weighted_deciles(inputs.calibration_deciles, ranked)
    selected = _weighted_metric_scope(inputs.prediction_metrics, "selected_players", ranked)
    top50 = _weighted_metric_scope(inputs.prediction_metrics, "top50_candidates", ranked)
    config_labels = ranked["config_label"].astype(str).tolist()

    figure = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Calibration deciles, row-count weighted across seasons",
            "Selected players: observed-count weighted calibration slope",
            "Top-50 candidates: observed-count weighted Spearman",
            "Selected players: observed-count weighted MAE",
        ),
        specs=[[{"type": "scatter"}, {"type": "bar"}], [{"type": "bar"}, {"type": "bar"}]],
        vertical_spacing=0.18,
        horizontal_spacing=0.08,
    )
    for config_label in config_labels:
        rows = calibration[calibration["config_label"].eq(config_label)].sort_values("decile")
        if rows.empty:
            continue
        figure.add_trace(
            go.Scatter(
                name=f"{config_label} actual",
                x=rows["decile"].astype(str).tolist(),
                y=_numeric_list(rows["actual_mean"]),
                mode="lines+markers",
                hovertemplate=f"{html.escape(config_label)} actual<br>Decile=%{{x}}<br>Mean=%{{y:.3f}}<extra></extra>",
            ),
            row=1,
            col=1,
        )
        figure.add_trace(
            go.Scatter(
                name=f"{config_label} predicted",
                x=rows["decile"].astype(str).tolist(),
                y=_numeric_list(rows["predicted_mean"]),
                mode="lines+markers",
                line={"dash": "dash"},
                hovertemplate=f"{html.escape(config_label)} predicted<br>Decile=%{{x}}<br>Mean=%{{y:.3f}}<extra></extra>",
            ),
            row=1,
            col=1,
        )

    selected = _sort_by_ranked_config(selected, ranked)
    top50 = _sort_by_ranked_config(top50, ranked)
    figure.add_trace(
        go.Bar(
            x=selected["config_label"].astype(str).tolist(),
            y=_numeric_list(selected["calibration_slope"]),
            marker_color="#7c3aed",
            hovertemplate="%{x}<br>Selected players calibration slope=%{y:.3f}<extra></extra>",
        ),
        row=1,
        col=2,
    )
    figure.add_trace(
        go.Bar(
            x=top50["config_label"].astype(str).tolist(),
            y=_numeric_list(top50["spearman"]),
            marker_color="#ea580c",
            hovertemplate="%{x}<br>Top-50 Spearman=%{y:.3f}<extra></extra>",
        ),
        row=2,
        col=1,
    )
    figure.add_trace(
        go.Bar(
            x=selected["config_label"].astype(str).tolist(),
            y=_numeric_list(selected["mae"]),
            marker_color="#0f766e",
            hovertemplate="%{x}<br>Selected players MAE=%{y:.3f}<extra></extra>",
        ),
        row=2,
        col=2,
    )
    figure.update_layout(
        title="Model-feature experiment calibration and ranking quality",
        template="plotly_white",
        height=1050,
        legend={"orientation": "h", "y": -0.2},
        margin={"l": 60, "r": 30, "t": 90, "b": 190},
    )
    figure.update_xaxes(tickangle=35)

    table = selected[["config_label", "calibration_slope", "mae"]].merge(
        top50[["config_label", "spearman"]],
        on="config_label",
        how="outer",
    )
    notes = [
        "Calibration deciles use row_count-weighted predicted and actual means across seasons.",
        "Selected players and Top-50 candidates are separate metric populations.",
        "Selected-player calibration slope, selected-player MAE, and Top-50 Spearman use observed_count-weighted season aggregation.",
    ]
    return _report_intro(notes) + _figure_html(figure) + _data_table_html(table)


def _ranked_for_display(ranked_summary: pd.DataFrame) -> pd.DataFrame:
    result = ranked_summary.copy()
    if "rank" in result.columns:
        return result.sort_values("rank", kind="stable").reset_index(drop=True)
    return result.sort_values("total_actual_points", ascending=False, kind="stable").reset_index(drop=True)


def _weighted_deciles(calibration_deciles: pd.DataFrame, ranked: pd.DataFrame) -> pd.DataFrame:
    return _sort_by_ranked_config(
        _weighted_aggregate(
            calibration_deciles,
            group_columns=("config_label", "decile"),
            value_columns=("predicted_mean", "actual_mean"),
            weight_column="row_count",
        ),
        ranked,
    )


def _weighted_metric_scope(prediction_metrics: pd.DataFrame, metric_scope: str, ranked: pd.DataFrame) -> pd.DataFrame:
    scoped = prediction_metrics[prediction_metrics["metric_scope"].eq(metric_scope)].copy()
    return _sort_by_ranked_config(
        _weighted_aggregate(
            scoped,
            group_columns=("config_label",),
            value_columns=("mae", "spearman", "calibration_slope"),
            weight_column="observed_count",
        ),
        ranked,
    )


def _weighted_aggregate(
    frame: pd.DataFrame,
    *,
    group_columns: tuple[str, ...],
    value_columns: tuple[str, ...],
    weight_column: str,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    if frame.empty:
        return pd.DataFrame(columns=pd.Index([*group_columns, *value_columns]))

    groupby_keys: str | list[str] = group_columns[0] if len(group_columns) == 1 else list(group_columns)
    for raw_keys, group in frame.groupby(groupby_keys, sort=False, dropna=False):
        keys = raw_keys if isinstance(raw_keys, tuple) else (raw_keys,)
        row = dict(zip(group_columns, keys, strict=True))
        weights = pd.to_numeric(group[weight_column], errors="coerce").fillna(0.0).astype(float)
        positive_weights = weights.clip(lower=0.0)
        weight_sum = float(positive_weights.sum())
        for value_column in value_columns:
            values = pd.to_numeric(group[value_column], errors="coerce")
            valid = values.notna()
            if weight_sum > 0 and valid.any():
                weighted_values = values[valid].astype(float) * positive_weights[valid]
                valid_weight_sum = float(positive_weights[valid].sum())
                row[value_column] = None if valid_weight_sum <= 0 else float(weighted_values.sum() / valid_weight_sum)
            elif valid.any():
                row[value_column] = float(values[valid].mean())
            else:
                row[value_column] = None
        rows.append(row)
    return pd.DataFrame(rows)


def _sort_by_ranked_config(frame: pd.DataFrame, ranked: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    order = {label: index for index, label in enumerate(ranked["config_label"].astype(str).tolist())}
    result["_config_order"] = result["config_label"].astype(str).map(order).fillna(len(order)).astype(int)
    sort_columns = ["_config_order"]
    if "season" in result.columns:
        sort_columns.append("season")
    if "decile" in result.columns:
        sort_columns.append("decile")
    return result.sort_values(sort_columns, kind="stable").drop(columns=["_config_order"]).reset_index(drop=True)


def _numeric_list(series: pd.Series) -> list[float | None]:
    return [None if pd.isna(value) else float(value) for value in series]


def _rounded_list(series: pd.Series) -> list[str]:
    return ["n/a" if pd.isna(value) else f"{float(value):.2f}" for value in series]


def _figure_html(figure: go.Figure) -> str:
    return figure.to_html(full_html=False, include_plotlyjs=True)


def _data_table_html(frame: pd.DataFrame) -> str:
    display = frame.copy()
    for column in display.columns:
        if pd.api.types.is_numeric_dtype(display[column]):
            display[column] = display[column].map(lambda value: "n/a" if pd.isna(value) else f"{float(value):.3f}")
    return display.to_html(index=False, escape=True, classes="data-table")


def _report_intro(notes: list[str]) -> str:
    escaped_notes = "\n".join(f"<li>{html.escape(note)}</li>" for note in notes)
    return f"<ul class=\"notes\">\n{escaped_notes}\n</ul>\n"


def _write_html(path: Path, *, title: str, body: str) -> None:
    path.write_text(
        "\n".join(
            [
                "<!doctype html>",
                "<html>",
                "<head>",
                f"<title>{html.escape(title)}</title>",
                "<meta charset=\"utf-8\">",
                "<style>",
                _report_css(),
                "</style>",
                "</head>",
                "<body>",
                f"<h1>{html.escape(title)}</h1>",
                body,
                "</body>",
                "</html>",
            ]
        ),
        encoding="utf-8",
    )


def _write_incomplete_reports(experiment_dir: Path, validation_errors: tuple[str, ...]) -> None:
    body = _incomplete_body(validation_errors)
    for filename, title in (
        (SQUAD_REPORT_NAME, "Squad performance comparison"),
        (CALIBRATION_REPORT_NAME, "Calibration plots"),
    ):
        _write_html(experiment_dir / filename, title=title, body=body)


def _incomplete_body(validation_errors: tuple[str, ...]) -> str:
    items = "\n".join(f"<li>{html.escape(error)}</li>" for error in validation_errors)
    return (
        "<h2>Report incomplete</h2>\n"
        "<p>The experiment completed without enough validated report inputs to render Plotly charts.</p>\n"
        f"<ul class=\"errors\">{items}</ul>\n"
    )


def _report_css() -> str:
    return """
body {
  color: #172033;
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  margin: 28px;
}
h1 {
  font-size: 28px;
  margin-bottom: 12px;
}
.notes {
  background: #f7fafc;
  border: 1px solid #d9e2ec;
  border-radius: 8px;
  line-height: 1.45;
  padding: 14px 18px 14px 32px;
}
.errors {
  background: #fff5f5;
  border: 1px solid #fecaca;
  border-radius: 8px;
  color: #7f1d1d;
  line-height: 1.45;
  padding: 14px 18px 14px 32px;
}
.data-table {
  border-collapse: collapse;
  font-size: 13px;
  margin-top: 24px;
  width: 100%;
}
.data-table th,
.data-table td {
  border-bottom: 1px solid #e5e7eb;
  padding: 8px 10px;
  text-align: left;
}
.data-table th {
  background: #f3f4f6;
  font-weight: 700;
}
"""
