from __future__ import annotations

import importlib.util
from datetime import UTC, datetime
from pathlib import Path

import pytest

from cartola.backtesting.strict_round_fixture_capture import (
    StrictFixtureCaptureError,
    StrictFixtureGenerationError,
    StrictRoundFixtureCaptureConfig,
    StrictRoundFixtureCaptureResult,
)

SCRIPT_PATH = Path(__file__).resolve().parents[3] / "scripts" / "capture_strict_round_fixture.py"
SPEC = importlib.util.spec_from_file_location("capture_strict_round_fixture", SCRIPT_PATH)
assert SPEC is not None
assert SPEC.loader is not None
cli = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(cli)


def test_parse_args_requires_auto_or_round() -> None:
    with pytest.raises(SystemExit):
        cli.parse_args(["--season", "2026", "--current-year", "2026"])


def test_parse_args_rejects_auto_and_round_together() -> None:
    with pytest.raises(SystemExit):
        cli.parse_args(["--season", "2026", "--auto", "--round", "12", "--current-year", "2026"])


def test_parse_args_rejects_non_cartola_source() -> None:
    with pytest.raises(SystemExit):
        cli.parse_args(["--season", "2026", "--auto", "--source", "thesportsdb"])


def test_main_prints_success_summary_with_relative_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    captured_config: StrictRoundFixtureCaptureConfig | None = None

    def fake_run(config: StrictRoundFixtureCaptureConfig) -> StrictRoundFixtureCaptureResult:
        nonlocal captured_config
        captured_config = config
        return _result(tmp_path)

    monkeypatch.setattr(cli, "run_strict_round_fixture_capture", fake_run)

    exit_code = cli.main(
        [
            "--season",
            "2026",
            "--auto",
            "--current-year",
            "2026",
            "--project-root",
            str(tmp_path),
        ]
    )

    assert exit_code == 0
    assert captured_config == StrictRoundFixtureCaptureConfig(
        season=2026,
        auto=True,
        current_year=2026,
        source="cartola_api",
        project_root=tmp_path,
    )
    captured = capsys.readouterr()
    assert "Strict fixture capture complete" in captured.out
    assert "Season" in captured.out
    assert "2026" in captured.out
    assert "Round" in captured.out
    assert "12" in captured.out
    assert (
        "Snapshot directory"
    ) in captured.out
    assert (
        "data/01_raw/fixtures_snapshots/2026/rodada-12/captured_at=2026-06-01T18-00-00Z"
    ) in captured.out
    assert "Strict fixture" in captured.out
    assert "data/01_raw/fixtures_strict/2026/partidas-12.csv" in captured.out
    assert "Manifest" in captured.out
    assert "data/01_raw/fixtures_strict/2026/partidas-12.manifest.json" in captured.out
    assert "Captured at UTC" in captured.out
    assert "2026-06-01T18:00:00Z" in captured.out
    assert "Deadline at UTC" in captured.out
    assert "2026-06-01T18:59:00Z" in captured.out


def test_main_maps_force_generate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_config: StrictRoundFixtureCaptureConfig | None = None

    def fake_run(config: StrictRoundFixtureCaptureConfig) -> StrictRoundFixtureCaptureResult:
        nonlocal captured_config
        captured_config = config
        return _result(tmp_path)

    monkeypatch.setattr(cli, "run_strict_round_fixture_capture", fake_run)

    exit_code = cli.main(
        [
            "--season",
            "2026",
            "--round",
            "12",
            "--current-year",
            "2026",
            "--project-root",
            str(tmp_path),
            "--force-generate",
        ]
    )

    assert exit_code == 0
    assert captured_config is not None
    assert captured_config.round_number == 12
    assert captured_config.force_generate is True


def test_main_prints_generation_failure_with_retained_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fake_run(config: StrictRoundFixtureCaptureConfig) -> StrictRoundFixtureCaptureResult:
        raise StrictFixtureGenerationError(capture_result=_result(tmp_path), original=FileExistsError("already exists"))

    monkeypatch.setattr(cli, "run_strict_round_fixture_capture", fake_run)

    exit_code = cli.main(
        [
            "--season",
            "2026",
            "--round",
            "12",
            "--current-year",
            "2026",
            "--project-root",
            str(tmp_path),
        ]
    )

    assert exit_code == 1
    captured = capsys.readouterr()
    assert "Strict fixture generation failed after snapshot capture" in captured.err
    assert (
        "Retained snapshot directory"
    ) in captured.err
    assert (
        "data/01_raw/fixtures_snapshots/2026/rodada-12/captured_at=2026-06-01T18-00-00Z"
    ) in captured.err
    assert "Captured at UTC" in captured.err
    assert "2026-06-01T18:00:00Z" in captured.err
    assert "Deadline at UTC" in captured.err
    assert "2026-06-01T18:59:00Z" in captured.err
    assert "Original error" in captured.err
    assert "FileExistsError: already exists" in captured.err
    assert "Traceback" not in captured.err


def test_main_prints_capture_failure_with_attempted_round(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fake_run(config: StrictRoundFixtureCaptureConfig) -> StrictRoundFixtureCaptureResult:
        raise StrictFixtureCaptureError(round_number=12, original=ValueError("round drift"))

    monkeypatch.setattr(cli, "run_strict_round_fixture_capture", fake_run)

    exit_code = cli.main(
        [
            "--season",
            "2026",
            "--auto",
            "--current-year",
            "2026",
            "--project-root",
            str(tmp_path),
        ]
    )

    assert exit_code == 1
    captured = capsys.readouterr()
    assert "Strict fixture capture failed" in captured.err
    assert "Round" in captured.err
    assert "12" in captured.err
    assert "Original error" in captured.err
    assert "ValueError: round drift" in captured.err
    assert "Traceback" not in captured.err


def test_main_prints_operational_error_without_traceback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fake_run(config: StrictRoundFixtureCaptureConfig) -> StrictRoundFixtureCaptureResult:
        raise ValueError("bad operational input")

    monkeypatch.setattr(cli, "run_strict_round_fixture_capture", fake_run)

    exit_code = cli.main(
        [
            "--season",
            "2026",
            "--round",
            "12",
            "--current-year",
            "2026",
            "--project-root",
            str(tmp_path),
        ]
    )

    assert exit_code == 1
    captured = capsys.readouterr()
    assert "Strict fixture capture failed" in captured.err
    assert "bad operational input" in captured.err
    assert "Traceback" not in captured.err


def _result(project_root: Path) -> StrictRoundFixtureCaptureResult:
    return StrictRoundFixtureCaptureResult(
        season=2026,
        round_number=12,
        capture_dir=project_root
        / "data"
        / "01_raw"
        / "fixtures_snapshots"
        / "2026"
        / "rodada-12"
        / "captured_at=2026-06-01T18-00-00Z",
        fixture_path=project_root / "data" / "01_raw" / "fixtures_strict" / "2026" / "partidas-12.csv",
        manifest_path=project_root
        / "data"
        / "01_raw"
        / "fixtures_strict"
        / "2026"
        / "partidas-12.manifest.json",
        captured_at_utc=datetime(2026, 6, 1, 18, 0, tzinfo=UTC),
        deadline_at_utc=datetime(2026, 6, 1, 18, 59, tzinfo=UTC),
    )
