from __future__ import annotations

import csv
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from cartola.backtesting import compatibility_audit as audit
from cartola.backtesting.config import BacktestConfig


def _touch_round(root: Path, season: int, round_name: str) -> Path:
    path = root / "data" / "01_raw" / str(season) / round_name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("x\n", encoding="utf-8")
    return path


def _season_frame(rounds: range) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "rodada": list(rounds),
            "id_atleta": list(range(1, len(list(rounds)) + 1)),
            "id_clube": [10] * len(list(rounds)),
        }
    )


def _fake_backtest_result(summary: pd.DataFrame) -> SimpleNamespace:
    return SimpleNamespace(summary=summary)


def test_discover_seasons_includes_numeric_dirs_with_round_files(tmp_path: Path) -> None:
    _touch_round(tmp_path, 2025, "rodada-1.csv")
    _touch_round(tmp_path, 2025, "rodada-2.csv")
    _touch_round(tmp_path, 2026, "rodada-1.csv")
    (tmp_path / "data" / "01_raw" / "fixtures" / "2025").mkdir(parents=True)
    (tmp_path / "data" / "01_raw" / "notes").mkdir(parents=True)
    (tmp_path / "data" / "01_raw" / "2024").mkdir(parents=True)

    config = audit.AuditConfig(project_root=tmp_path, current_year=2026)

    seasons = audit.discover_seasons(config)

    assert [season.season for season in seasons] == [2025, 2026]
    assert seasons[0].detected_rounds == [1, 2]
    assert seasons[0].round_file_count == 2
    assert seasons[0].min_round == 1
    assert seasons[0].max_round == 2


def test_discover_seasons_records_malformed_round_filename(tmp_path: Path) -> None:
    _touch_round(tmp_path, 2025, "rodada-final.csv")

    config = audit.AuditConfig(project_root=tmp_path, current_year=2026)

    seasons = audit.discover_seasons(config)

    assert len(seasons) == 1
    assert seasons[0].season == 2025
    assert seasons[0].discovery_error is not None
    assert seasons[0].discovery_error.stage == "discovery"
    assert "Invalid round CSV filename" in seasons[0].discovery_error.message


def test_parse_round_number_rejects_zero_and_non_matching_names() -> None:
    with pytest.raises(ValueError, match="positive"):
        audit.parse_round_number(Path("rodada-0.csv"))

    with pytest.raises(ValueError, match="Invalid round CSV filename"):
        audit.parse_round_number(Path("round-1.csv"))


def test_classify_season_requires_contiguous_complete_rounds() -> None:
    config = audit.AuditConfig(current_year=2026, expected_complete_rounds=38)

    complete = audit.classify_season(2025, list(range(1, 39)), config)
    gapped = audit.classify_season(2025, [*range(1, 7), *range(8, 40)], config)
    irregular_extra = audit.classify_season(2022, list(range(1, 40)), config)
    partial_current = audit.classify_season(2026, list(range(1, 14)), config)

    assert complete == ("complete_historical", True, [])
    assert gapped[0] == "irregular_historical"
    assert gapped[1] is False
    assert irregular_extra[0] == "irregular_historical"
    assert partial_current == (
        "partial_current",
        False,
        ["partial current season; metrics are smoke-test only"],
    )


def test_short_error_message_caps_to_configured_limit() -> None:
    message = audit._short_error_message("x" * 350)

    assert len(message) == audit.CSV_ERROR_MESSAGE_LIMIT


def test_load_failure_records_row_and_skips_later_stages(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _touch_round(tmp_path, 2025, "rodada-1.csv")

    def fail_load(season: int, project_root: Path) -> pd.DataFrame:
        raise ValueError(f"bad load {season} {project_root}")

    monkeypatch.setattr(audit, "load_season_data", fail_load)

    result = audit.run_compatibility_audit(audit.AuditConfig(project_root=tmp_path, current_year=2026))

    record = result.seasons[0]
    assert record.load_status == "failed"
    assert record.feature_status == "skipped"
    assert record.backtest_status == "skipped"
    assert record.error_stage == "load"
    assert record.error_type == "ValueError"
    assert record.error_detail is not None
    assert record.error_detail.stage == "load"


def test_feature_check_covers_every_eligible_target_round(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    for round_number in range(1, 8):
        _touch_round(tmp_path, 2025, f"rodada-{round_number}.csv")

    training_calls: list[int] = []
    prediction_calls: list[int] = []

    monkeypatch.setattr(audit, "load_season_data", lambda season, project_root: _season_frame(range(1, 8)))
    monkeypatch.setattr(
        audit,
        "build_training_frame",
        lambda season_df, target_round, playable_statuses, fixtures: training_calls.append(target_round)
        or pd.DataFrame(),
    )
    monkeypatch.setattr(
        audit,
        "build_prediction_frame",
        lambda season_df, target_round, fixtures: prediction_calls.append(target_round) or pd.DataFrame(),
    )
    monkeypatch.setattr(
        audit,
        "run_backtest",
        lambda config: _fake_backtest_result(
            pd.DataFrame(
                [
                    {"strategy": "baseline", "average_actual_points": 1.0},
                    {"strategy": "random_forest", "average_actual_points": 2.0},
                    {"strategy": "price", "average_actual_points": 3.0},
                ]
            )
        ),
    )

    result = audit.run_compatibility_audit(audit.AuditConfig(project_root=tmp_path, current_year=2026, start_round=5))

    assert training_calls == [5, 6, 7]
    assert prediction_calls == [5, 6, 7]
    assert result.seasons[0].feature_status == "ok"


def test_feature_failure_records_target_round_and_skips_backtest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for round_number in range(1, 8):
        _touch_round(tmp_path, 2025, f"rodada-{round_number}.csv")

    def fail_training(
        season_df: pd.DataFrame, target_round: int, playable_statuses: tuple[str, ...], fixtures: None
    ) -> pd.DataFrame:
        if target_round == 6:
            raise RuntimeError("feature broke")
        return pd.DataFrame()

    def fail_if_called(config: BacktestConfig) -> SimpleNamespace:
        raise AssertionError("backtest should be skipped")

    monkeypatch.setattr(audit, "load_season_data", lambda season, project_root: _season_frame(range(1, 8)))
    monkeypatch.setattr(audit, "build_training_frame", fail_training)
    monkeypatch.setattr(audit, "build_prediction_frame", lambda season_df, target_round, fixtures: pd.DataFrame())
    monkeypatch.setattr(audit, "run_backtest", fail_if_called)

    result = audit.run_compatibility_audit(audit.AuditConfig(project_root=tmp_path, current_year=2026, start_round=5))

    record = result.seasons[0]
    assert record.feature_status == "failed"
    assert record.backtest_status == "skipped"
    assert record.error_stage == "feature"
    assert record.error_detail is not None
    assert record.error_detail.target_round == 6


def test_max_round_before_start_round_marks_feature_not_applicable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for round_number in range(1, 4):
        _touch_round(tmp_path, 2025, f"rodada-{round_number}.csv")

    monkeypatch.setattr(audit, "load_season_data", lambda season, project_root: _season_frame(range(1, 4)))

    result = audit.run_compatibility_audit(audit.AuditConfig(project_root=tmp_path, current_year=2026, start_round=5))

    record = result.seasons[0]
    assert record.load_status == "ok"
    assert record.feature_status == "not_applicable"
    assert record.backtest_status == "skipped"
    assert record.evaluated_rounds == 0


def test_backtest_stage_uses_isolated_output_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    for round_number in range(1, 6):
        _touch_round(tmp_path, 2025, f"rodada-{round_number}.csv")

    observed_configs: list[BacktestConfig] = []
    monkeypatch.setattr(audit, "load_season_data", lambda season, project_root: _season_frame(range(1, 6)))
    monkeypatch.setattr(audit, "build_training_frame", lambda season_df, target_round, playable_statuses, fixtures: pd.DataFrame())
    monkeypatch.setattr(audit, "build_prediction_frame", lambda season_df, target_round, fixtures: pd.DataFrame())

    def fake_run_backtest(config: BacktestConfig) -> SimpleNamespace:
        observed_configs.append(config)
        return _fake_backtest_result(
            pd.DataFrame(
                [
                    {"strategy": "baseline", "average_actual_points": 10.0},
                    {"strategy": "random_forest", "average_actual_points": 20.0},
                    {"strategy": "price", "average_actual_points": 5.0},
                ]
            )
        )

    monkeypatch.setattr(audit, "run_backtest", fake_run_backtest)

    result = audit.run_compatibility_audit(
        audit.AuditConfig(project_root=tmp_path, current_year=2026, output_root=Path("data/08_reporting/backtests/compatibility"))
    )

    assert observed_configs == [
        BacktestConfig(
            season=2025,
            start_round=5,
            project_root=tmp_path,
            output_root=Path("data/08_reporting/backtests/compatibility/runs"),
            fixture_mode="none",
        )
    ]
    assert result.seasons[0].backtest_status == "ok"
    assert result.seasons[0].baseline_avg_points == 10.0
    assert result.seasons[0].random_forest_avg_points == 20.0
    assert result.seasons[0].price_avg_points == 5.0


def test_missing_strategy_metric_rows_are_null_without_failing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    for round_number in range(1, 6):
        _touch_round(tmp_path, 2025, f"rodada-{round_number}.csv")

    monkeypatch.setattr(audit, "load_season_data", lambda season, project_root: _season_frame(range(1, 6)))
    monkeypatch.setattr(audit, "build_training_frame", lambda season_df, target_round, playable_statuses, fixtures: pd.DataFrame())
    monkeypatch.setattr(audit, "build_prediction_frame", lambda season_df, target_round, fixtures: pd.DataFrame())
    monkeypatch.setattr(
        audit,
        "run_backtest",
        lambda config: _fake_backtest_result(pd.DataFrame([{"strategy": "baseline", "average_actual_points": 10.0}])),
    )

    result = audit.run_compatibility_audit(audit.AuditConfig(project_root=tmp_path, current_year=2026))

    record = result.seasons[0]
    assert record.backtest_status == "ok"
    assert record.baseline_avg_points == 10.0
    assert record.random_forest_avg_points is None
    assert record.price_avg_points is None
    assert "missing expected strategy metrics" in "; ".join(record.notes)


def test_backtest_failure_records_error_and_keeps_metrics_null(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    for round_number in range(1, 6):
        _touch_round(tmp_path, 2025, f"rodada-{round_number}.csv")

    monkeypatch.setattr(audit, "load_season_data", lambda season, project_root: _season_frame(range(1, 6)))
    monkeypatch.setattr(audit, "build_training_frame", lambda season_df, target_round, playable_statuses, fixtures: pd.DataFrame())
    monkeypatch.setattr(audit, "build_prediction_frame", lambda season_df, target_round, fixtures: pd.DataFrame())
    monkeypatch.setattr(audit, "run_backtest", lambda config: (_ for _ in ()).throw(RuntimeError("backtest broke")))

    result = audit.run_compatibility_audit(audit.AuditConfig(project_root=tmp_path, current_year=2026))

    record = result.seasons[0]
    assert record.backtest_status == "failed"
    assert record.error_stage == "backtest"
    assert record.baseline_avg_points is None
    assert record.random_forest_avg_points is None
    assert record.price_avg_points is None


def test_reports_record_current_year_detected_rounds_and_full_error_details(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _touch_round(tmp_path, 2025, "rodada-1.csv")
    long_message = "x" * 350
    monkeypatch.setattr(audit, "load_season_data", lambda season, project_root: (_ for _ in ()).throw(ValueError(long_message)))

    result = audit.run_compatibility_audit(audit.AuditConfig(project_root=tmp_path, current_year=2026))

    csv_frame = pd.read_csv(result.csv_path)
    with result.csv_path.open(newline="", encoding="utf-8") as file:
        csv_rows = list(csv.DictReader(file))
    payload = json.loads(result.json_path.read_text(encoding="utf-8"))

    assert csv_rows[0]["detected_rounds"] == "1"
    assert len(csv_frame.loc[0, "error_message"]) == 300
    assert payload["config"]["current_year"] == 2026
    assert payload["config"]["fixture_mode"] == "none"
    assert payload["seasons"][0]["detected_rounds"] == [1]
    assert payload["seasons"][0]["error_detail"]["message"] == long_message
    assert "ValueError" in payload["seasons"][0]["error_detail"]["traceback"]


def test_partial_current_metrics_are_recorded_but_not_comparable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for round_number in range(1, 14):
        _touch_round(tmp_path, 2026, f"rodada-{round_number}.csv")

    monkeypatch.setattr(audit, "load_season_data", lambda season, project_root: _season_frame(range(1, 14)))
    monkeypatch.setattr(audit, "build_training_frame", lambda season_df, target_round, playable_statuses, fixtures: pd.DataFrame())
    monkeypatch.setattr(audit, "build_prediction_frame", lambda season_df, target_round, fixtures: pd.DataFrame())
    monkeypatch.setattr(
        audit,
        "run_backtest",
        lambda config: _fake_backtest_result(
            pd.DataFrame(
                [
                    {"strategy": "baseline", "average_actual_points": 1.0},
                    {"strategy": "random_forest", "average_actual_points": 2.0},
                    {"strategy": "price", "average_actual_points": 3.0},
                ]
            )
        ),
    )

    result = audit.run_compatibility_audit(audit.AuditConfig(project_root=tmp_path, current_year=2026))

    record = result.seasons[0]
    assert record.season_status == "partial_current"
    assert record.metrics_comparable is False
    assert record.random_forest_avg_points == 2.0


def test_parse_args_accepts_current_year_and_output_root() -> None:
    args = audit.parse_args(
        [
            "--project-root",
            "/tmp/cartola",
            "--start-round",
            "6",
            "--complete-round-threshold",
            "20",
            "--expected-complete-rounds",
            "22",
            "--current-year",
            "2026",
            "--output-root",
            "/tmp/audit",
        ]
    )

    assert args.project_root == Path("/tmp/cartola")
    assert args.start_round == 6
    assert args.complete_round_threshold == 20
    assert args.expected_complete_rounds == 22
    assert args.current_year == 2026
    assert args.output_root == Path("/tmp/audit")


def test_main_runs_audit_and_prints_report_paths(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    observed_configs: list[audit.AuditConfig] = []

    def fake_run(config: audit.AuditConfig) -> audit.AuditRunResult:
        observed_configs.append(config)
        csv_path = tmp_path / "season_compatibility.csv"
        json_path = tmp_path / "season_compatibility.json"
        return audit.AuditRunResult(
            generated_at_utc="2026-04-27T00:00:00Z",
            project_root=config.project_root,
            config=config,
            seasons=[],
            csv_path=csv_path,
            json_path=json_path,
        )

    monkeypatch.setattr(audit, "run_compatibility_audit", fake_run)

    exit_code = audit.main(["--project-root", str(tmp_path), "--current-year", "2026"])

    assert exit_code == 0
    assert observed_configs == [audit.AuditConfig(project_root=tmp_path, current_year=2026)]
    output = capsys.readouterr().out
    assert "Compatibility audit complete" in output
    assert "season_compatibility.csv" in output
    assert "season_compatibility.json" in output
