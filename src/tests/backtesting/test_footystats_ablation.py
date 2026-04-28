from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from cartola.backtesting import footystats_ablation as ablation


def _write_backtest_outputs(
    output_path: Path,
    *,
    baseline: float = 50.0,
    rf: float = 55.0,
    r2: float = 0.1,
    corr: float = 0.2,
) -> None:
    output_path.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {"strategy": "baseline", "average_actual_points": baseline},
            {"strategy": "random_forest", "average_actual_points": rf},
        ]
    ).to_csv(output_path / "summary.csv", index=False)
    pd.DataFrame(
        [
            {
                "section": "prediction",
                "strategy": "random_forest",
                "position": "all",
                "metric": "player_r2",
                "value": r2,
            },
            {
                "section": "prediction",
                "strategy": "random_forest",
                "position": "all",
                "metric": "player_correlation",
                "value": corr,
            },
        ]
    ).to_csv(output_path / "diagnostics.csv", index=False)


def test_parse_seasons_preserves_order_and_rejects_duplicates() -> None:
    assert ablation.parse_seasons("2025,2023,2024") == (2025, 2023, 2024)

    with pytest.raises(ValueError, match="duplicate"):
        ablation.parse_seasons("2023,2024,2023")


@pytest.mark.parametrize("value", ["", "2023,", "2023,,2024", "0", "-2023"])
def test_parse_seasons_rejects_empty_entries_and_non_positive_values(value: str) -> None:
    with pytest.raises(ValueError):
        ablation.parse_seasons(value)


def test_config_from_default_args() -> None:
    config = ablation.config_from_args(ablation.parse_args([]))

    assert config.seasons == (2023, 2024, 2025)
    assert config.start_round == 5
    assert config.budget == 100.0
    assert config.project_root == Path(".")
    assert config.output_root == Path("data/08_reporting/backtests/footystats_ablation")
    assert config.footystats_league_slug == "brazil-serie-a"
    assert config.force is False


def test_parse_args_preserves_duplicate_season_error_message(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit):
        ablation.parse_args(["--seasons", "2023,2023"])

    captured = capsys.readouterr()
    assert "duplicate season" in captured.err


def test_script_imports_main_from_footystats_ablation() -> None:
    script_path = Path(__file__).resolve().parents[3] / "scripts" / "run_footystats_ppg_ablation.py"
    spec = importlib.util.spec_from_file_location("run_footystats_ppg_ablation", script_path)
    assert spec is not None
    assert spec.loader is not None
    script = importlib.util.module_from_spec(spec)

    spec.loader.exec_module(script)

    assert script.main is ablation.main


@pytest.mark.parametrize(
    "output_root",
    [
        "../footystats_ablation",
        Path("/tmp/footystats_ablation"),
    ],
)
def test_resolve_output_root_rejects_paths_outside_project_root(tmp_path: Path, output_root: Path | str) -> None:
    config = ablation.FootyStatsPPGAblationConfig(project_root=tmp_path, output_root=Path(output_root))

    with pytest.raises(ValueError, match="inside project_root"):
        ablation.resolve_output_root(config)


def test_resolve_output_root_allows_absolute_paths_inside_project_root(tmp_path: Path) -> None:
    output_root = tmp_path / "reports" / "footystats_ablation"
    config = ablation.FootyStatsPPGAblationConfig(project_root=tmp_path, output_root=output_root)

    assert ablation.resolve_output_root(config) == output_root.resolve()


@pytest.mark.parametrize(
    "output_root",
    [
        ".",
        "data",
        "data/08_reporting",
        "data/08_reporting/backtests",
        "data/08_reporting/backtests/2025",
    ],
)
def test_resolve_output_root_rejects_protected_backtest_paths(tmp_path: Path, output_root: str) -> None:
    config = ablation.FootyStatsPPGAblationConfig(
        project_root=tmp_path,
        output_root=Path(output_root),
        seasons=(2025,),
    )

    with pytest.raises(ValueError, match="protected"):
        ablation.resolve_output_root(config)


def test_resolve_output_root_requires_footystats_ablation_directory_name(tmp_path: Path) -> None:
    config = ablation.FootyStatsPPGAblationConfig(project_root=tmp_path, output_root=Path("data/08_reporting/backtests/other"))

    with pytest.raises(ValueError, match="footystats_ablation"):
        ablation.resolve_output_root(config)


def test_build_backtest_config_uses_mode_specific_output_roots(tmp_path: Path) -> None:
    resolved_output_root = tmp_path / "data" / "08_reporting" / "backtests" / "footystats_ablation"
    config = ablation.FootyStatsPPGAblationConfig(
        project_root=tmp_path,
        seasons=(2025,),
        start_round=7,
        budget=90.5,
        footystats_league_slug="custom-league",
        current_year=2026,
    )

    control = ablation.build_backtest_config(config, 2025, "none", resolved_output_root)
    treatment = ablation.build_backtest_config(config, 2025, "ppg", resolved_output_root)

    assert control.output_root == resolved_output_root / "runs" / "2025" / "footystats_mode=none"
    assert treatment.output_root == resolved_output_root / "runs" / "2025" / "footystats_mode=ppg"
    assert control.output_path == resolved_output_root / "runs" / "2025" / "footystats_mode=none" / "2025"
    assert treatment.output_path == resolved_output_root / "runs" / "2025" / "footystats_mode=ppg" / "2025"
    assert control.fixture_mode == "none"
    assert treatment.fixture_mode == "none"
    assert control.footystats_mode == "none"
    assert treatment.footystats_mode == "ppg"
    assert control.footystats_evaluation_scope == "historical_candidate"
    assert treatment.footystats_evaluation_scope == "historical_candidate"
    assert control.current_year == 2026
    assert treatment.current_year == 2026
    assert control.season == 2025
    assert treatment.season == 2025
    assert control.start_round == 7
    assert treatment.start_round == 7
    assert control.budget == 90.5
    assert treatment.budget == 90.5
    assert control.project_root == tmp_path
    assert treatment.project_root == tmp_path
    assert control.footystats_league_slug == "custom-league"
    assert treatment.footystats_league_slug == "custom-league"


def test_build_backtest_config_rejects_unsupported_mode(tmp_path: Path) -> None:
    config = ablation.FootyStatsPPGAblationConfig(project_root=tmp_path)

    with pytest.raises(ValueError, match="Unsupported footystats mode"):
        ablation.build_backtest_config(config, 2025, "live", tmp_path / "footystats_ablation")


def test_build_backtest_config_rejects_normal_backtest_output_path(tmp_path: Path) -> None:
    config = ablation.FootyStatsPPGAblationConfig(project_root=tmp_path)
    resolved_output_root = tmp_path / "footystats_ablation"
    mode_root = resolved_output_root / "runs" / "2025" / "footystats_mode=none"
    mode_root.parent.mkdir(parents=True)
    normal_backtests_root = tmp_path / "data" / "08_reporting" / "backtests"
    normal_backtests_root.mkdir(parents=True)
    mode_root.symlink_to(normal_backtests_root)

    with pytest.raises(ValueError, match="normal backtest"):
        ablation.build_backtest_config(config, 2025, "none", resolved_output_root)


def test_build_backtest_config_revalidates_output_root_inside_project_root(tmp_path: Path) -> None:
    outside_root = tmp_path.parent / f"{tmp_path.name}_outside" / "footystats_ablation"
    config = ablation.FootyStatsPPGAblationConfig(project_root=tmp_path)

    with pytest.raises(ValueError, match="inside project_root"):
        ablation.build_backtest_config(config, 2025, "none", outside_root)


def test_prepare_output_root_raises_for_existing_root_without_force(tmp_path: Path) -> None:
    output_root = tmp_path / "data" / "08_reporting" / "backtests" / "footystats_ablation"
    output_root.mkdir(parents=True)
    config = ablation.FootyStatsPPGAblationConfig(project_root=tmp_path, force=False)

    with pytest.raises(FileExistsError):
        ablation.prepare_output_root(config, output_root)


def test_prepare_output_root_with_force_removes_only_safe_ablation_root(tmp_path: Path) -> None:
    output_root = tmp_path / "data" / "08_reporting" / "backtests" / "footystats_ablation"
    output_root.mkdir(parents=True)
    (output_root / "stale.csv").write_text("old")
    sibling = tmp_path / "data" / "08_reporting" / "backtests" / "2025"
    sibling.mkdir(parents=True)
    (sibling / "summary.csv").write_text("keep")
    config = ablation.FootyStatsPPGAblationConfig(project_root=tmp_path, force=True)

    ablation.prepare_output_root(config, output_root)

    assert output_root.is_dir()
    assert not (output_root / "stale.csv").exists()
    assert (sibling / "summary.csv").read_text() == "keep"


def test_prepare_output_root_revalidates_output_root_before_force_delete(tmp_path: Path) -> None:
    outside_root = tmp_path.parent / f"{tmp_path.name}_outside" / "footystats_ablation"
    outside_root.mkdir(parents=True)
    sentinel = outside_root / "sentinel.txt"
    sentinel.write_text("keep")
    config = ablation.FootyStatsPPGAblationConfig(project_root=tmp_path, force=True)

    with pytest.raises(ValueError, match="inside project_root"):
        ablation.prepare_output_root(config, outside_root)

    assert sentinel.read_text() == "keep"


def test_eligibility_failure_skips_control_and_treatment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[object] = []

    def fail_eligibility(**kwargs: object) -> object:
        raise ValueError("not a candidate")

    def fail_run_backtest(config: object) -> object:
        calls.append(config)
        raise AssertionError("run_backtest should not be called")

    monkeypatch.setattr(ablation, "load_footystats_ppg_rows", fail_eligibility)
    monkeypatch.setattr(ablation, "run_backtest", fail_run_backtest)

    config = ablation.FootyStatsPPGAblationConfig(
        project_root=tmp_path,
        output_root=Path("reports/footystats_ablation"),
    )

    result = ablation.run_footystats_ppg_ablation(config)

    assert calls == []
    assert len(result.seasons) == 3
    first = result.seasons[0]
    assert first.control_status == "skipped"
    assert first.treatment_status == "skipped"
    assert first.error_stage == "eligibility"
    assert first.metrics_comparable is False


def test_orchestration_runs_control_then_treatment_for_eligible_season(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[int, str, str]] = []

    def load_eligible(**kwargs: object) -> object:
        return SimpleNamespace(
            source_path=Path("data/footystats/brazil-serie-a-matches-2025-to-2025-stats.csv"),
            source_sha256="abc123",
        )

    def fake_run_backtest(config: object) -> object:
        assert hasattr(config, "output_path")
        calls.append((config.season, config.fixture_mode, config.footystats_mode))
        _write_backtest_outputs(config.output_path)
        return SimpleNamespace()

    monkeypatch.setattr(ablation, "load_footystats_ppg_rows", load_eligible)
    monkeypatch.setattr(ablation, "run_backtest", fake_run_backtest)

    config = ablation.FootyStatsPPGAblationConfig(
        seasons=(2025,),
        project_root=tmp_path,
        output_root=Path("reports/footystats_ablation"),
    )

    result = ablation.run_footystats_ppg_ablation(config)

    assert calls == [(2025, "none", "none"), (2025, "none", "ppg")]
    first = result.seasons[0]
    assert first.control_status == "ok"
    assert first.treatment_status == "ok"
    assert first.treatment_source_sha256 == "abc123"


def test_metric_extraction_failure_marks_paired_comparison_failed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def load_eligible(**kwargs: object) -> object:
        return SimpleNamespace(
            source_path=Path("data/footystats/brazil-serie-a-matches-2025-to-2025-stats.csv"),
            source_sha256="abc123",
        )

    def fake_run_backtest(config: object) -> object:
        assert hasattr(config, "output_path")
        if config.footystats_mode == "none":
            _write_backtest_outputs(config.output_path)
        else:
            config.output_path.mkdir(parents=True, exist_ok=True)
            pd.DataFrame([{"strategy": "random_forest", "average_actual_points": 55.0}]).to_csv(
                config.output_path / "summary.csv",
                index=False,
            )
            pd.DataFrame(
                [
                    {
                        "section": "prediction",
                        "strategy": "random_forest",
                        "position": "all",
                        "metric": "player_r2",
                        "value": 0.1,
                    },
                    {
                        "section": "prediction",
                        "strategy": "random_forest",
                        "position": "all",
                        "metric": "player_correlation",
                        "value": 0.2,
                    },
                ]
            ).to_csv(config.output_path / "diagnostics.csv", index=False)
        return SimpleNamespace()

    monkeypatch.setattr(ablation, "load_footystats_ppg_rows", load_eligible)
    monkeypatch.setattr(ablation, "run_backtest", fake_run_backtest)

    config = ablation.FootyStatsPPGAblationConfig(
        seasons=(2025,),
        project_root=tmp_path,
        output_root=Path("reports/footystats_ablation"),
    )

    result = ablation.run_footystats_ppg_ablation(config)

    first = result.seasons[0]
    assert first.error_stage == "metric_extraction"
    assert first.metrics_comparable is False
    assert first.metric_status == "failed"
    assert (first.control_status, first.treatment_status) == ("ok", "ok")


def test_extract_run_metrics_uses_prediction_diagnostics_section(tmp_path: Path) -> None:
    output_path = tmp_path / "run"
    _write_backtest_outputs(output_path, r2=0.25)
    diagnostics = pd.read_csv(output_path / "diagnostics.csv")
    diagnostics = pd.concat(
        [
            diagnostics,
            pd.DataFrame(
                [
                    {
                        "section": "rounds",
                        "strategy": "random_forest",
                        "position": "all",
                        "metric": "player_r2",
                        "value": 0.99,
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    diagnostics.to_csv(output_path / "diagnostics.csv", index=False)

    assert ablation.extract_run_metrics(output_path)["r2"] == 0.25


@pytest.mark.parametrize("strategy", ["baseline", "random_forest"])
def test_extract_run_metrics_fails_for_missing_summary_strategy(tmp_path: Path, strategy: str) -> None:
    output_path = tmp_path / "run"
    _write_backtest_outputs(output_path)
    summary = pd.read_csv(output_path / "summary.csv")
    summary = summary[~summary["strategy"].eq(strategy)]
    summary.to_csv(output_path / "summary.csv", index=False)

    with pytest.raises(ValueError, match=strategy):
        ablation.extract_run_metrics(output_path)


@pytest.mark.parametrize("strategy", ["baseline", "random_forest"])
def test_extract_run_metrics_fails_for_duplicate_summary_strategy(tmp_path: Path, strategy: str) -> None:
    output_path = tmp_path / "run"
    _write_backtest_outputs(output_path)
    summary = pd.read_csv(output_path / "summary.csv")
    summary = pd.concat(
        [
            summary,
            summary[summary["strategy"].eq(strategy)],
        ],
        ignore_index=True,
    )
    summary.to_csv(output_path / "summary.csv", index=False)

    with pytest.raises(ValueError, match=strategy):
        ablation.extract_run_metrics(output_path)


@pytest.mark.parametrize("metric", ["player_r2", "player_correlation"])
def test_extract_run_metrics_fails_for_missing_prediction_diagnostic(tmp_path: Path, metric: str) -> None:
    output_path = tmp_path / "run"
    _write_backtest_outputs(output_path)
    diagnostics = pd.read_csv(output_path / "diagnostics.csv")
    diagnostics = diagnostics[~diagnostics["metric"].eq(metric)]
    diagnostics.to_csv(output_path / "diagnostics.csv", index=False)

    with pytest.raises(ValueError, match=metric):
        ablation.extract_run_metrics(output_path)


@pytest.mark.parametrize("metric", ["player_r2", "player_correlation"])
def test_extract_run_metrics_fails_for_duplicate_prediction_diagnostic(tmp_path: Path, metric: str) -> None:
    output_path = tmp_path / "run"
    _write_backtest_outputs(output_path)
    diagnostics = pd.read_csv(output_path / "diagnostics.csv")
    diagnostics = pd.concat([diagnostics, diagnostics[diagnostics["metric"].eq(metric)]], ignore_index=True)
    diagnostics.to_csv(output_path / "diagnostics.csv", index=False)

    with pytest.raises(ValueError, match=metric):
        ablation.extract_run_metrics(output_path)


@pytest.mark.parametrize("strategy", ["baseline", "random_forest"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_extract_run_metrics_fails_for_non_finite_summary_values(
    tmp_path: Path, strategy: str, value: float
) -> None:
    output_path = tmp_path / "run"
    _write_backtest_outputs(output_path)
    summary = pd.read_csv(output_path / "summary.csv")
    summary.loc[summary["strategy"].eq(strategy), "average_actual_points"] = value
    summary.to_csv(output_path / "summary.csv", index=False)

    with pytest.raises(ValueError, match=f"summary.*{strategy}.*average_actual_points"):
        ablation.extract_run_metrics(output_path)


@pytest.mark.parametrize("metric", ["player_r2", "player_correlation"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_extract_run_metrics_fails_for_non_finite_diagnostic_values(
    tmp_path: Path, metric: str, value: float
) -> None:
    output_path = tmp_path / "run"
    _write_backtest_outputs(output_path)
    diagnostics = pd.read_csv(output_path / "diagnostics.csv")
    diagnostics.loc[diagnostics["metric"].eq(metric), "value"] = value
    diagnostics.to_csv(output_path / "diagnostics.csv", index=False)

    with pytest.raises(ValueError, match=f"diagnostics.*{metric}.*value"):
        ablation.extract_run_metrics(output_path)


def test_populate_metrics_baseline_mismatch_preserves_metrics_and_marks_not_comparable(tmp_path: Path) -> None:
    control_path = tmp_path / "control"
    treatment_path = tmp_path / "treatment"
    _write_backtest_outputs(control_path, baseline=50.0, rf=55.0, r2=0.1, corr=0.2)
    _write_backtest_outputs(treatment_path, baseline=51.0, rf=57.0, r2=0.3, corr=0.4)
    record = ablation.SeasonAblationRecord(season=2025)

    with pytest.raises(ValueError) as exc_info:
        ablation.populate_metrics(record, control_path, treatment_path)

    assert str(exc_info.value) == "baseline average points differ"
    assert record.control_baseline_avg_points == 50.0
    assert record.treatment_baseline_avg_points == 51.0
    assert record.control_rf_avg_points == 55.0
    assert record.treatment_rf_avg_points == 57.0
    assert record.control_player_r2 == 0.1
    assert record.treatment_player_r2 == 0.3
    assert record.control_player_corr == 0.2
    assert record.treatment_player_corr == 0.4
    assert record.baseline_avg_points is None
    assert record.baseline_avg_points_equal is False
    assert record.metrics_comparable is False


def test_build_aggregate_record_averages_only_successful_comparable_records() -> None:
    included_a = ablation.SeasonAblationRecord(
        season=2023,
        metrics_comparable=True,
        control_status="ok",
        treatment_status="ok",
        metric_status="ok",
        control_baseline_avg_points=45.0,
        treatment_baseline_avg_points=45.0,
        baseline_avg_points=45.0,
        baseline_avg_points_equal=True,
        control_rf_avg_points=50.0,
        treatment_rf_avg_points=55.0,
        rf_avg_points_delta=5.0,
        control_player_r2=0.1,
        treatment_player_r2=0.2,
        player_r2_delta=0.1,
        control_player_corr=0.3,
        treatment_player_corr=0.4,
        player_corr_delta=0.1,
        rf_minus_baseline_control=5.0,
        rf_minus_baseline_treatment=10.0,
    )
    included_b = ablation.SeasonAblationRecord(
        season=2024,
        metrics_comparable=True,
        control_status="ok",
        treatment_status="ok",
        metric_status="ok",
        control_baseline_avg_points=65.0,
        treatment_baseline_avg_points=65.0,
        baseline_avg_points=65.0,
        baseline_avg_points_equal=True,
        control_rf_avg_points=70.0,
        treatment_rf_avg_points=73.0,
        rf_avg_points_delta=3.0,
        control_player_r2=0.5,
        treatment_player_r2=0.6,
        player_r2_delta=0.1,
        control_player_corr=0.7,
        treatment_player_corr=0.9,
        player_corr_delta=0.2,
        rf_minus_baseline_control=5.0,
        rf_minus_baseline_treatment=8.0,
    )
    failed = ablation.SeasonAblationRecord(
        season=2025,
        metrics_comparable=True,
        control_status="failed",
        treatment_status="ok",
        control_rf_avg_points=1000.0,
        treatment_rf_avg_points=1000.0,
        rf_avg_points_delta=1000.0,
        control_player_r2=1000.0,
        treatment_player_r2=1000.0,
        player_r2_delta=1000.0,
        control_player_corr=1000.0,
        treatment_player_corr=1000.0,
        player_corr_delta=1000.0,
    )
    non_comparable = ablation.SeasonAblationRecord(
        season=2026,
        metrics_comparable=False,
        control_status="ok",
        treatment_status="ok",
        control_rf_avg_points=1000.0,
        treatment_rf_avg_points=1000.0,
        rf_avg_points_delta=1000.0,
        control_player_r2=1000.0,
        treatment_player_r2=1000.0,
        player_r2_delta=1000.0,
        control_player_corr=1000.0,
        treatment_player_corr=1000.0,
        player_corr_delta=1000.0,
    )
    metric_failed = ablation.SeasonAblationRecord(
        season=2027,
        metrics_comparable=True,
        control_status="ok",
        treatment_status="ok",
        metric_status="failed",
        control_baseline_avg_points=1000.0,
        treatment_baseline_avg_points=1000.0,
        baseline_avg_points=1000.0,
        baseline_avg_points_equal=True,
        control_rf_avg_points=1000.0,
        treatment_rf_avg_points=1000.0,
        rf_avg_points_delta=1000.0,
        control_player_r2=1000.0,
        treatment_player_r2=1000.0,
        player_r2_delta=1000.0,
        control_player_corr=1000.0,
        treatment_player_corr=1000.0,
        player_corr_delta=1000.0,
        rf_minus_baseline_control=1000.0,
        rf_minus_baseline_treatment=1000.0,
    )

    aggregate = ablation.build_aggregate_record([included_a, failed, included_b, non_comparable, metric_failed])

    assert aggregate.metrics_comparable is True
    assert aggregate.control_baseline_avg_points == 55.0
    assert aggregate.treatment_baseline_avg_points == 55.0
    assert aggregate.baseline_avg_points == 55.0
    assert aggregate.baseline_avg_points_equal is True
    assert aggregate.control_rf_avg_points == 60.0
    assert aggregate.treatment_rf_avg_points == 64.0
    assert aggregate.rf_avg_points_delta == 4.0
    assert aggregate.control_player_r2 == 0.3
    assert aggregate.treatment_player_r2 == 0.4
    assert aggregate.player_r2_delta == 0.1
    assert aggregate.control_player_corr == 0.5
    assert aggregate.treatment_player_corr == 0.65
    assert aggregate.player_corr_delta == pytest.approx(0.15)
    assert aggregate.rf_minus_baseline_control == 5.0
    assert aggregate.rf_minus_baseline_treatment == 9.0


def test_build_aggregate_record_rejects_missing_required_metric_on_comparable_record() -> None:
    record = ablation.SeasonAblationRecord(
        season=2025,
        metrics_comparable=True,
        control_status="ok",
        treatment_status="ok",
        metric_status="ok",
        control_baseline_avg_points=50.0,
        treatment_baseline_avg_points=50.0,
        baseline_avg_points=50.0,
        baseline_avg_points_equal=True,
        control_rf_avg_points=None,
        treatment_rf_avg_points=55.0,
        rf_avg_points_delta=5.0,
        control_player_r2=0.1,
        treatment_player_r2=0.2,
        player_r2_delta=0.1,
        control_player_corr=0.3,
        treatment_player_corr=0.4,
        player_corr_delta=0.1,
        rf_minus_baseline_control=0.0,
        rf_minus_baseline_treatment=5.0,
    )

    with pytest.raises(ValueError, match="control_rf_avg_points"):
        ablation.build_aggregate_record([record])


def test_build_aggregate_record_rejects_non_finite_required_metric_on_comparable_record() -> None:
    record = ablation.SeasonAblationRecord(
        season=2025,
        metrics_comparable=True,
        control_status="ok",
        treatment_status="ok",
        metric_status="ok",
        control_baseline_avg_points=50.0,
        treatment_baseline_avg_points=50.0,
        baseline_avg_points=50.0,
        baseline_avg_points_equal=True,
        control_rf_avg_points=float("nan"),
        treatment_rf_avg_points=55.0,
        rf_avg_points_delta=5.0,
        control_player_r2=0.1,
        treatment_player_r2=0.2,
        player_r2_delta=0.1,
        control_player_corr=0.3,
        treatment_player_corr=0.4,
        player_corr_delta=0.1,
        rf_minus_baseline_control=0.0,
        rf_minus_baseline_treatment=5.0,
    )

    with pytest.raises(ValueError, match="control_rf_avg_points"):
        ablation.build_aggregate_record([record])


def _report_result(tmp_path: Path) -> ablation.FootyStatsPPGAblationResult:
    output_root = tmp_path / "reports" / "footystats_ablation"
    included = ablation.SeasonAblationRecord(
        season=2024,
        season_status="candidate",
        metrics_comparable=True,
        control_status="ok",
        treatment_status="ok",
        metric_status="ok",
        control_output_path=str(output_root / "runs" / "2024" / "footystats_mode=none" / "2024"),
        treatment_output_path=str(output_root / "runs" / "2024" / "footystats_mode=ppg" / "2024"),
        control_baseline_avg_points=45.0,
        treatment_baseline_avg_points=45.0,
        baseline_avg_points=45.0,
        baseline_avg_points_equal=True,
        control_rf_avg_points=50.0,
        treatment_rf_avg_points=55.0,
        rf_avg_points_delta=5.0,
        control_player_r2=0.1,
        treatment_player_r2=0.2,
        player_r2_delta=0.1,
        control_player_corr=0.3,
        treatment_player_corr=0.4,
        player_corr_delta=0.1,
        rf_minus_baseline_control=5.0,
        rf_minus_baseline_treatment=10.0,
        treatment_source_path="data/footystats/brazil-serie-a-matches-2024-to-2024-stats.csv",
        treatment_source_sha256="sha-2024",
    )
    excluded = ablation.SeasonAblationRecord(
        season=2025,
        season_status="candidate",
        metrics_comparable=False,
        control_status="ok",
        treatment_status="ok",
        metric_status="failed",
        control_output_path=str(output_root / "runs" / "2025" / "footystats_mode=none" / "2025"),
        treatment_output_path=str(output_root / "runs" / "2025" / "footystats_mode=ppg" / "2025"),
        error_stage="metric_extraction",
        error_message="x" * 600,
        treatment_source_sha256="sha-2025",
        errors=[
            ablation.AblationError(
                stage="metric_extraction",
                type="ValueError",
                message="x" * 600,
                traceback="Traceback...",
            )
        ],
    )
    aggregate = ablation.build_aggregate_record([included, excluded])
    config = ablation.FootyStatsPPGAblationConfig(
        seasons=(2024, 2025),
        start_round=7,
        budget=90.5,
        project_root=tmp_path,
        output_root=Path("reports/footystats_ablation"),
        current_year=2026,
        force=True,
    )
    return ablation.FootyStatsPPGAblationResult(
        config=config,
        resolved_output_root=output_root,
        seasons=[included, excluded],
        aggregate=aggregate,
    )


def test_write_reports_creates_csv_and_json_report_artifacts(tmp_path: Path) -> None:
    result = _report_result(tmp_path)
    result.resolved_output_root.mkdir(parents=True)

    ablation.write_reports(result)

    csv_path = result.resolved_output_root / "ppg_ablation.csv"
    json_path = result.resolved_output_root / "ppg_ablation.json"
    assert csv_path.is_file()
    assert json_path.is_file()

    csv = pd.read_csv(csv_path, keep_default_na=False)
    assert csv["season"].astype(str).tolist() == ["2024", "2025", "aggregate"]
    assert "metric_status" in csv.columns

    report = json.loads(json_path.read_text())
    assert report["config"]["fixture_mode"] == "none"
    assert report["config"]["control_footystats_mode"] == "none"
    assert report["config"]["treatment_footystats_mode"] == "ppg"
    assert report["config"]["footystats_evaluation_scope"] == "historical_candidate"
    assert report["seasons"][0]["treatment_source_sha256"] == "sha-2024"
    assert report["aggregate"]["included_seasons"] == [2024]
    assert report["generated_at_utc"].endswith("Z")


def test_write_reports_truncates_csv_error_message_and_keeps_null_metric_cells_empty(tmp_path: Path) -> None:
    result = _report_result(tmp_path)
    result.resolved_output_root.mkdir(parents=True)

    ablation.write_reports(result)

    csv = pd.read_csv(result.resolved_output_root / "ppg_ablation.csv", keep_default_na=False)
    failed = csv[csv["season"].astype(str).eq("2025")].iloc[0]
    assert failed["error_message"] == "x" * 500
    assert failed["baseline_avg_points"] == ""
    assert failed["rf_avg_points_delta"] == ""


def test_write_reports_json_includes_config_and_aggregate_exclusions(tmp_path: Path) -> None:
    result = _report_result(tmp_path)
    result.resolved_output_root.mkdir(parents=True)

    ablation.write_reports(result)

    report = json.loads((result.resolved_output_root / "ppg_ablation.json").read_text())
    assert report["config"] == {
        "project_root": str(tmp_path),
        "resolved_project_root": str(tmp_path.resolve()),
        "output_root": "reports/footystats_ablation",
        "resolved_output_root": str(result.resolved_output_root.resolve()),
        "seasons": [2024, 2025],
        "start_round": 7,
        "budget": 90.5,
        "current_year": 2026,
        "resolved_current_year": 2026,
        "fixture_mode": "none",
        "control_footystats_mode": "none",
        "treatment_footystats_mode": "ppg",
        "footystats_evaluation_scope": "historical_candidate",
        "footystats_league_slug": "brazil-serie-a",
        "force": True,
    }
    assert report["aggregate"]["excluded_seasons"] == [
        {
            "season": 2025,
            "reason": "metric_extraction: " + ("x" * 600),
        }
    ]
    assert report["aggregate"]["aggregation_method"] == "mean"
    assert report["aggregate"]["metrics"]["rf_avg_points_delta"] == 5.0


def test_write_reports_uses_atomic_replace_without_final_files_on_replace_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result = _report_result(tmp_path)
    result.resolved_output_root.mkdir(parents=True)

    def fail_replace(self: Path, target: Path) -> Path:
        raise OSError("replace failed")

    monkeypatch.setattr(Path, "replace", fail_replace)

    with pytest.raises(OSError, match="replace failed"):
        ablation.write_reports(result)

    assert not (result.resolved_output_root / "ppg_ablation.csv").exists()
    assert not (result.resolved_output_root / "ppg_ablation.json").exists()


def test_write_reports_restores_old_final_pair_when_json_replace_fails_after_csv_replace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result = _report_result(tmp_path)
    result.resolved_output_root.mkdir(parents=True)
    csv_path = result.resolved_output_root / "ppg_ablation.csv"
    json_path = result.resolved_output_root / "ppg_ablation.json"
    csv_path.write_text("old csv\n")
    json_path.write_text('{"old": "json"}\n')
    original_replace = Path.replace

    def fail_json_final_replace(self: Path, target: Path) -> Path:
        if self.name == ".ppg_ablation.json.tmp" and target == json_path:
            raise OSError("json replace failed")
        return original_replace(self, target)

    monkeypatch.setattr(Path, "replace", fail_json_final_replace)

    with pytest.raises(OSError, match="json replace failed"):
        ablation.write_reports(result)

    assert csv_path.read_text() == "old csv\n"
    assert json_path.read_text() == '{"old": "json"}\n'


def test_main_writes_reports_and_returns_zero_when_at_least_one_season_is_comparable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def load_eligible(**kwargs: object) -> object:
        return SimpleNamespace(
            source_path=Path("data/footystats/brazil-serie-a-matches-2025-to-2025-stats.csv"),
            source_sha256="sha-2025",
        )

    def fake_run_backtest(config: object) -> object:
        assert hasattr(config, "output_path")
        _write_backtest_outputs(config.output_path)
        return SimpleNamespace()

    monkeypatch.setattr(ablation, "load_footystats_ppg_rows", load_eligible)
    monkeypatch.setattr(ablation, "run_backtest", fake_run_backtest)

    output_root = Path("data/08_reporting/backtests/footystats_ablation")
    exit_code = ablation.main(
        [
            "--project-root",
            str(tmp_path),
            "--output-root",
            str(output_root),
            "--seasons",
            "2025",
            "--current-year",
            "2026",
        ]
    )

    resolved_output_root = tmp_path / output_root
    captured = capsys.readouterr()
    assert exit_code == 0
    assert (resolved_output_root / "ppg_ablation.csv").is_file()
    assert str(resolved_output_root) in captured.out
    assert "comparable seasons: 1" in captured.out


def test_main_writes_reports_and_returns_one_when_no_season_is_comparable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def fail_eligibility(**kwargs: object) -> object:
        raise ValueError("not comparable")

    monkeypatch.setattr(ablation, "load_footystats_ppg_rows", fail_eligibility)

    output_root = Path("data/08_reporting/backtests/footystats_ablation")
    exit_code = ablation.main(
        [
            "--project-root",
            str(tmp_path),
            "--output-root",
            str(output_root),
            "--seasons",
            "2025",
            "--current-year",
            "2026",
        ]
    )

    resolved_output_root = tmp_path / output_root
    captured = capsys.readouterr()
    assert exit_code == 1
    assert (resolved_output_root / "ppg_ablation.json").is_file()
    assert str(resolved_output_root) in captured.out
    assert "comparable seasons: 0" in captured.out
