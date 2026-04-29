from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

import cartola.backtesting.strict_round_fixture_capture as strict_round_fixture_capture
from cartola.backtesting.fixture_snapshots import CaptureResult
from cartola.backtesting.strict_fixtures import StrictFixtureLoadResult
from cartola.backtesting.strict_round_fixture_capture import (
    StrictFixtureCaptureError,
    StrictFixtureGenerationError,
    StrictRoundFixtureCaptureConfig,
    run_strict_round_fixture_capture,
)

CAPTURED_AT = datetime(2026, 6, 1, 18, 0, tzinfo=UTC)
DEADLINE_AT = datetime(2026, 6, 1, 18, 59, tzinfo=UTC)


def test_run_requires_exactly_one_round_selector(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="exactly one"):
        run_strict_round_fixture_capture(
            StrictRoundFixtureCaptureConfig(season=2026, current_year=2026, project_root=tmp_path)
        )

    with pytest.raises(ValueError, match="exactly one"):
        run_strict_round_fixture_capture(
            StrictRoundFixtureCaptureConfig(
                season=2026,
                round_number=12,
                auto=True,
                current_year=2026,
                project_root=tmp_path,
            )
        )


def test_run_rejects_wrong_current_year_before_network(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_active_fetch(*args: object, **kwargs: object) -> int:
        raise AssertionError("active-round fetch should not be called")

    def fail_capture(*args: object, **kwargs: object) -> CaptureResult:
        raise AssertionError("snapshot capture should not be called")

    monkeypatch.setattr(strict_round_fixture_capture, "fetch_cartola_active_open_round", fail_active_fetch)
    monkeypatch.setattr(strict_round_fixture_capture, "capture_cartola_snapshot", fail_capture)

    with pytest.raises(ValueError, match="season 2025 must equal current_year 2026"):
        run_strict_round_fixture_capture(
            StrictRoundFixtureCaptureConfig(
                season=2025,
                auto=True,
                current_year=2026,
                project_root=tmp_path,
            )
        )


def test_run_auto_resolves_active_round_and_generates_from_captured_timestamp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    capture_kwargs: dict[str, object] = {}
    generation_kwargs: dict[str, object] = {}

    def fake_active_fetch(*, fetch: object | None = None) -> int:
        return 12

    def fake_capture(**kwargs: object) -> CaptureResult:
        capture_kwargs.update(kwargs)
        return _capture_result(tmp_path, round_number=12)

    def fake_generate(**kwargs: object) -> StrictFixtureLoadResult:
        generation_kwargs.update(kwargs)
        return _strict_result(tmp_path, round_number=12)

    monkeypatch.setattr(strict_round_fixture_capture, "fetch_cartola_active_open_round", fake_active_fetch)
    monkeypatch.setattr(strict_round_fixture_capture, "capture_cartola_snapshot", fake_capture)
    monkeypatch.setattr(strict_round_fixture_capture, "generate_strict_fixture", fake_generate)

    result = run_strict_round_fixture_capture(
        StrictRoundFixtureCaptureConfig(
            season=2026,
            auto=True,
            current_year=2026,
            project_root=tmp_path,
        )
    )

    assert capture_kwargs["round_number"] == 12
    assert generation_kwargs["round_number"] == 12
    assert generation_kwargs["captured_at"] == CAPTURED_AT
    assert generation_kwargs["force"] is False
    assert result.round_number == 12
    assert result.capture_dir == _capture_result(tmp_path, round_number=12).capture_dir


def test_run_explicit_round_passes_round_without_active_fetch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_active_fetch(*args: object, **kwargs: object) -> int:
        raise AssertionError("active-round fetch should not be called")

    def fake_capture(**kwargs: object) -> CaptureResult:
        return _capture_result(tmp_path, round_number=int(kwargs["round_number"]))

    def fake_generate(**kwargs: object) -> StrictFixtureLoadResult:
        return _strict_result(tmp_path, round_number=int(kwargs["round_number"]))

    monkeypatch.setattr(strict_round_fixture_capture, "fetch_cartola_active_open_round", fail_active_fetch)
    monkeypatch.setattr(strict_round_fixture_capture, "capture_cartola_snapshot", fake_capture)
    monkeypatch.setattr(strict_round_fixture_capture, "generate_strict_fixture", fake_generate)

    result = run_strict_round_fixture_capture(
        StrictRoundFixtureCaptureConfig(
            season=2026,
            round_number=13,
            current_year=2026,
            project_root=tmp_path,
        )
    )

    assert result.round_number == 13


def test_run_force_generate_maps_only_to_generator(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    capture_kwargs: dict[str, object] = {}
    generation_kwargs: dict[str, object] = {}

    def fake_capture(**kwargs: object) -> CaptureResult:
        capture_kwargs.update(kwargs)
        return _capture_result(tmp_path)

    def fake_generate(**kwargs: object) -> StrictFixtureLoadResult:
        generation_kwargs.update(kwargs)
        return _strict_result(tmp_path)

    monkeypatch.setattr(strict_round_fixture_capture, "capture_cartola_snapshot", fake_capture)
    monkeypatch.setattr(strict_round_fixture_capture, "generate_strict_fixture", fake_generate)

    run_strict_round_fixture_capture(
        StrictRoundFixtureCaptureConfig(
            season=2026,
            round_number=12,
            current_year=2026,
            project_root=tmp_path,
            force_generate=True,
        )
    )

    assert "force" not in capture_kwargs
    assert generation_kwargs["force"] is True


def test_run_generation_failure_preserves_capture_context(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    capture = _capture_result(tmp_path)

    def fake_capture(**kwargs: object) -> CaptureResult:
        return capture

    def fail_generate(**kwargs: object) -> StrictFixtureLoadResult:
        raise FileExistsError("already generated")

    monkeypatch.setattr(strict_round_fixture_capture, "capture_cartola_snapshot", fake_capture)
    monkeypatch.setattr(strict_round_fixture_capture, "generate_strict_fixture", fail_generate)

    with pytest.raises(StrictFixtureGenerationError, match="already generated") as exc_info:
        run_strict_round_fixture_capture(
            StrictRoundFixtureCaptureConfig(
                season=2026,
                round_number=12,
                current_year=2026,
                project_root=tmp_path,
            )
        )

    assert exc_info.value.capture_result is capture
    assert isinstance(exc_info.value.original, FileExistsError)


def test_run_capture_failure_preserves_attempted_round(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_active_fetch(*, fetch: object | None = None) -> int:
        return 12

    def fail_capture(**kwargs: object) -> CaptureResult:
        raise ValueError("round drift")

    def fail_generate(**kwargs: object) -> StrictFixtureLoadResult:
        raise AssertionError("generator should not run")

    monkeypatch.setattr(strict_round_fixture_capture, "fetch_cartola_active_open_round", fake_active_fetch)
    monkeypatch.setattr(strict_round_fixture_capture, "capture_cartola_snapshot", fail_capture)
    monkeypatch.setattr(strict_round_fixture_capture, "generate_strict_fixture", fail_generate)

    with pytest.raises(StrictFixtureCaptureError, match="round drift") as exc_info:
        run_strict_round_fixture_capture(
            StrictRoundFixtureCaptureConfig(
                season=2026,
                auto=True,
                current_year=2026,
                project_root=tmp_path,
            )
        )

    assert exc_info.value.round_number == 12
    assert isinstance(exc_info.value.original, ValueError)


def _capture_result(project_root: Path, *, round_number: int = 12) -> CaptureResult:
    return CaptureResult(
        capture_dir=project_root
        / "data"
        / "01_raw"
        / "fixtures_snapshots"
        / "2026"
        / f"rodada-{round_number}"
        / "captured_at=2026-06-01T18-00-00Z",
        captured_at_utc=CAPTURED_AT,
        deadline_at_utc=DEADLINE_AT,
        fixture_rows=[],
    )


def _strict_result(project_root: Path, *, round_number: int = 12) -> StrictFixtureLoadResult:
    fixture_path = project_root / "data" / "01_raw" / "fixtures_strict" / "2026" / f"partidas-{round_number}.csv"
    manifest_path = fixture_path.with_suffix(".manifest.json")
    return StrictFixtureLoadResult(
        fixture_path=fixture_path,
        manifest_path=manifest_path,
        manifest={"mode": "strict"},
        captured_at_utc=CAPTURED_AT,
        deadline_at_utc=DEADLINE_AT,
        generator_version="fixture_snapshot_v1",
    )
