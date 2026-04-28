from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from cartola.backtesting import footystats_ablation as ablation


def _write_backtest_outputs(output_path: Path) -> None:
    output_path.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {"strategy": "baseline", "average_actual_points": 50.0},
            {"strategy": "random_forest", "average_actual_points": 55.0},
        ]
    ).to_csv(output_path / "summary.csv", index=False)
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
    assert (first.control_status, first.treatment_status) == ("failed", "failed")
