from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import requests

from cartola.backtesting.fixture_snapshots import (
    CaptureResult,
    Clock,
    Fetch,
    capture_cartola_snapshot,
    fetch_cartola_active_open_round,
)
from cartola.backtesting.strict_fixtures import (
    STRICT_SOURCE,
    StrictFixtureLoadResult,
    generate_strict_fixture,
)


@dataclass(frozen=True)
class StrictRoundFixtureCaptureConfig:
    season: int
    round_number: int | None = None
    auto: bool = False
    current_year: int | None = None
    source: str = STRICT_SOURCE
    project_root: Path = Path(".")
    force_generate: bool = False


@dataclass(frozen=True)
class StrictRoundFixtureCaptureResult:
    season: int
    round_number: int
    capture_dir: Path
    fixture_path: Path
    manifest_path: Path
    captured_at_utc: datetime
    deadline_at_utc: datetime


class StrictFixtureGenerationError(RuntimeError):
    def __init__(self, *, capture_result: CaptureResult, original: Exception) -> None:
        self.capture_result = capture_result
        self.original = original
        super().__init__(
            "Strict fixture generation failed after snapshot capture "
            f"at {format_utc_z(capture_result.captured_at_utc)}: {original}"
        )


class StrictFixtureCaptureError(RuntimeError):
    def __init__(self, *, round_number: int, original: Exception) -> None:
        self.round_number = round_number
        self.original = original
        super().__init__(f"Strict fixture snapshot capture failed for round {round_number}: {original}")


def run_strict_round_fixture_capture(
    config: StrictRoundFixtureCaptureConfig,
    *,
    fetch: Fetch | None = None,
    now: Clock | None = None,
) -> StrictRoundFixtureCaptureResult:
    if config.source != STRICT_SOURCE:
        raise ValueError(f"Unsupported strict fixture source: {config.source!r}")

    current_year = config.current_year if config.current_year is not None else datetime.now(UTC).year
    if config.season != current_year:
        raise ValueError(f"season {config.season} must equal current_year {current_year}")

    if config.auto == (config.round_number is not None):
        raise ValueError("Specify exactly one of auto=True or round_number")

    if config.auto:
        round_number = fetch_cartola_active_open_round(fetch=fetch)
    else:
        round_number = _positive_round(config.round_number)

    root = Path(config.project_root)
    try:
        capture_result = capture_cartola_snapshot(
            project_root=root,
            season=config.season,
            round_number=round_number,
            source=config.source,
            fetch=fetch,
            now=now,
        )
    except _CAPTURE_ERRORS as exc:
        raise StrictFixtureCaptureError(round_number=round_number, original=exc) from exc

    try:
        fixture_result = generate_strict_fixture(
            project_root=root,
            season=config.season,
            round_number=round_number,
            source=config.source,
            captured_at=capture_result.captured_at_utc,
            force=config.force_generate,
        )
    except _GENERATION_ERRORS as exc:
        raise StrictFixtureGenerationError(capture_result=capture_result, original=exc) from exc

    return _result_from_capture_and_fixture(
        season=config.season,
        round_number=round_number,
        capture_result=capture_result,
        fixture_result=fixture_result,
    )


def format_project_path(project_root: str | Path, path: str | Path) -> str:
    root = Path(project_root).resolve()
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(root).as_posix()
    except ValueError:
        return resolved.as_posix()


def format_utc_z(value: datetime) -> str:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("timestamp must be timezone-aware")
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _positive_round(round_number: int | None) -> int:
    if round_number is None:
        raise ValueError("Specify exactly one of auto=True or round_number")
    if round_number <= 0:
        raise ValueError("round_number must be a positive integer")
    return round_number


def _result_from_capture_and_fixture(
    *,
    season: int,
    round_number: int,
    capture_result: CaptureResult,
    fixture_result: StrictFixtureLoadResult,
) -> StrictRoundFixtureCaptureResult:
    return StrictRoundFixtureCaptureResult(
        season=season,
        round_number=round_number,
        capture_dir=capture_result.capture_dir,
        fixture_path=fixture_result.fixture_path,
        manifest_path=fixture_result.manifest_path,
        captured_at_utc=capture_result.captured_at_utc,
        deadline_at_utc=capture_result.deadline_at_utc,
    )


_CAPTURE_ERRORS: tuple[type[Exception], ...] = (
    FileExistsError,
    FileNotFoundError,
    ValueError,
    requests.RequestException,
)
_GENERATION_ERRORS: tuple[type[Exception], ...] = (FileExistsError, FileNotFoundError, ValueError)
