import json
from pathlib import Path

import pandas as pd

from cartola.backtesting.budgeting import BUDGET_POLICY_MOVING
from cartola.backtesting.experiment_reports import build_experiment_html_reports


def _write_valid_experiment_artifacts(output_path: Path) -> None:
    output_path.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "rank": 1,
                "model_id": "ridge",
                "feature_pack": "ppg",
                "fixture_mode": "exploratory",
                "total_actual_points": 200.0,
                "aggregate_delta": 50.0,
                "average_actual_delta_per_round": 5.0,
                "promotion_eligible": True,
                "promotion_reason": "passes_v1_guardrails",
            },
            {
                "rank": 2,
                "model_id": "ridge",
                "feature_pack": "ppg",
                "fixture_mode": "none",
                "total_actual_points": 180.0,
                "aggregate_delta": 30.0,
                "average_actual_delta_per_round": 3.0,
                "promotion_eligible": True,
                "promotion_reason": "passes_v1_guardrails",
            },
            {
                "rank": 3,
                "model_id": "random_forest",
                "feature_pack": "ppg",
                "fixture_mode": "exploratory",
                "total_actual_points": 150.0,
                "aggregate_delta": 0.0,
                "average_actual_delta_per_round": 0.0,
                "promotion_eligible": False,
                "promotion_reason": "aggregate_delta_not_positive",
            },
        ]
    ).to_csv(output_path / "ranked_summary.csv", index=False)
    pd.DataFrame(
        [
            {
                "season": 2024,
                "model_id": "ridge",
                "feature_pack": "ppg",
                "fixture_mode": "exploratory",
                "total_actual_points": 90.0,
                "rounds": 10,
            },
            {
                "season": 2025,
                "model_id": "ridge",
                "feature_pack": "ppg",
                "fixture_mode": "exploratory",
                "total_actual_points": 110.0,
                "rounds": 10,
            },
            {
                "season": 2024,
                "model_id": "ridge",
                "feature_pack": "ppg",
                "fixture_mode": "none",
                "total_actual_points": 80.0,
                "rounds": 10,
            },
            {
                "season": 2025,
                "model_id": "ridge",
                "feature_pack": "ppg",
                "fixture_mode": "none",
                "total_actual_points": 100.0,
                "rounds": 10,
            },
        ]
    ).to_csv(output_path / "per_season_summary.csv", index=False)
    pd.DataFrame(
        [
            {
                "season": 2024,
                "model_id": "ridge",
                "feature_pack": "ppg",
                "fixture_mode": "exploratory",
                "metric_scope": "selected_players",
                "observed_count": 10,
                "mae": 2.0,
                "spearman": 0.2,
                "calibration_slope": 0.5,
            },
            {
                "season": 2025,
                "model_id": "ridge",
                "feature_pack": "ppg",
                "fixture_mode": "exploratory",
                "metric_scope": "selected_players",
                "observed_count": 30,
                "mae": 4.0,
                "spearman": 0.6,
                "calibration_slope": 1.0,
            },
            {
                "season": 2024,
                "model_id": "ridge",
                "feature_pack": "ppg",
                "fixture_mode": "exploratory",
                "metric_scope": "top50_candidates",
                "observed_count": 10,
                "mae": 3.0,
                "spearman": 0.1,
                "calibration_slope": 0.3,
            },
            {
                "season": 2025,
                "model_id": "ridge",
                "feature_pack": "ppg",
                "fixture_mode": "exploratory",
                "metric_scope": "top50_candidates",
                "observed_count": 30,
                "mae": 5.0,
                "spearman": 0.5,
                "calibration_slope": 0.9,
            },
            {
                "season": 2024,
                "model_id": "ridge",
                "feature_pack": "ppg",
                "fixture_mode": "none",
                "metric_scope": "selected_players",
                "observed_count": 20,
                "mae": 6.0,
                "spearman": 0.7,
                "calibration_slope": 0.8,
            },
        ]
    ).to_csv(output_path / "prediction_metrics.csv", index=False)
    pd.DataFrame(
        [
            {
                "season": 2024,
                "model_id": "ridge",
                "feature_pack": "ppg",
                "fixture_mode": "exploratory",
                "decile": 1,
                "row_count": 1,
                "predicted_mean": 2.0,
                "actual_mean": 3.0,
            },
            {
                "season": 2025,
                "model_id": "ridge",
                "feature_pack": "ppg",
                "fixture_mode": "exploratory",
                "decile": 1,
                "row_count": 3,
                "predicted_mean": 4.0,
                "actual_mean": 5.0,
            },
            {
                "season": 2024,
                "model_id": "ridge",
                "feature_pack": "ppg",
                "fixture_mode": "none",
                "decile": 1,
                "row_count": 2,
                "predicted_mean": 7.0,
                "actual_mean": 8.0,
            },
        ]
    ).to_csv(output_path / "calibration_deciles.csv", index=False)
    (output_path / "comparability_report.json").write_text(json.dumps({"status": "ok"}), encoding="utf-8")


def test_experiment_reports_generate_plotly_html_with_distinct_fixture_configs(tmp_path: Path) -> None:
    _write_valid_experiment_artifacts(tmp_path)

    build_experiment_html_reports(tmp_path)

    squad_html = (tmp_path / "squad_performance_comparison.html").read_text(encoding="utf-8")
    calibration_html = (tmp_path / "calibration_plots.html").read_text(encoding="utf-8")
    assert "Plotly.newPlot" in squad_html
    assert "Plotly.newPlot" in calibration_html
    assert "ridge / ppg / exploratory" in squad_html
    assert "ridge / ppg / none" in squad_html
    assert "promotion_eligible" in squad_html
    assert "200" in squad_html
    assert "50" in squad_html


def test_experiment_reports_label_budget_policy_and_normalize_legacy_missing_policy(tmp_path: Path) -> None:
    _write_valid_experiment_artifacts(tmp_path)
    ranked = pd.read_csv(tmp_path / "ranked_summary.csv")
    moving_row = ranked.iloc[[0]].copy()
    moving_row["budget_policy"] = BUDGET_POLICY_MOVING
    moving_row["rank"] = 4
    moving_row["total_actual_points"] = 210.0
    ranked = pd.concat([ranked, moving_row], ignore_index=True)
    ranked.to_csv(tmp_path / "ranked_summary.csv", index=False)
    for filename in ("per_season_summary.csv", "prediction_metrics.csv", "calibration_deciles.csv"):
        frame = pd.read_csv(tmp_path / filename)
        moving_rows = frame.iloc[[0]].copy()
        moving_rows["budget_policy"] = BUDGET_POLICY_MOVING
        frame = pd.concat([frame, moving_rows], ignore_index=True)
        frame.to_csv(tmp_path / filename, index=False)

    build_experiment_html_reports(tmp_path)

    squad_html = (tmp_path / "squad_performance_comparison.html").read_text(encoding="utf-8")
    assert "ridge / ppg / exploratory / fixed" in squad_html
    assert "ridge / ppg / exploratory / moving" in squad_html


def test_experiment_reports_use_weighted_metric_scope_aggregations(tmp_path: Path) -> None:
    _write_valid_experiment_artifacts(tmp_path)

    build_experiment_html_reports(tmp_path)

    calibration_html = (tmp_path / "calibration_plots.html").read_text(encoding="utf-8")
    assert "Selected players" in calibration_html
    assert "Top-50 candidates" in calibration_html
    assert "0.875" in calibration_html
    assert "0.4" in calibration_html
    assert "3.5" in calibration_html
    assert "4.5" in calibration_html


def test_experiment_reports_render_incomplete_pages_for_missing_required_columns(tmp_path: Path) -> None:
    _write_valid_experiment_artifacts(tmp_path)
    ranked_summary = pd.read_csv(tmp_path / "ranked_summary.csv")
    ranked_summary.drop(columns=["fixture_mode"]).to_csv(tmp_path / "ranked_summary.csv", index=False)

    build_experiment_html_reports(tmp_path)

    for filename in ("squad_performance_comparison.html", "calibration_plots.html"):
        html = (tmp_path / filename).read_text(encoding="utf-8")
        assert "Report incomplete" in html
        assert "ranked_summary.csv" in html
        assert "fixture_mode" in html
        assert "Plotly.newPlot" not in html
