from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Callable, Sequence

DEFAULT_SEASONS = (2023, 2024, 2025)
DEFAULT_OUTPUT_ROOT = Path("data/08_reporting/fixtures")
CSV_ERROR_MESSAGE_LIMIT = 300

STATUS_OK = "ok"
STATUS_FAILED = "failed"
STATUS_SKIPPED = "skipped"


@dataclass(frozen=True)
class MatchupFixtureAuditConfig:
    seasons: tuple[int, ...] = DEFAULT_SEASONS
    project_root: Path = Path(".")
    output_root: Path = DEFAULT_OUTPUT_ROOT
    expected_complete_rounds: int = 38
    complete_round_threshold: int = 38
    current_year: int | None = None

    def resolved_current_year(self, clock: Callable[[], datetime] | None = None) -> int:
        if self.current_year is not None:
            return self.current_year
        now = clock() if clock is not None else datetime.now(UTC)
        return now.year


def parse_seasons(value: str) -> tuple[int, ...]:
    raw_parts = value.split(",")
    if any(part.strip() == "" for part in raw_parts):
        raise ValueError("seasons must not contain empty entries")

    seasons: list[int] = []
    seen: set[int] = set()
    for part in raw_parts:
        try:
            season = int(part)
        except ValueError as exc:
            raise ValueError(f"Invalid season: {part}") from exc
        if season <= 0:
            raise ValueError("seasons must be positive integers")
        if season in seen:
            raise ValueError(f"duplicate season: {season}")
        seasons.append(season)
        seen.add(season)
    return tuple(seasons)


def parse_seasons_arg(value: str) -> tuple[int, ...]:
    try:
        return parse_seasons(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def _positive_int_arg(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit Cartola matchup fixture coverage.")
    parser.add_argument("--seasons", type=parse_seasons_arg, default=DEFAULT_SEASONS)
    parser.add_argument("--current-year", type=_positive_int_arg, default=None)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--expected-complete-rounds", type=_positive_int_arg, default=38)
    parser.add_argument("--complete-round-threshold", type=_positive_int_arg, default=38)
    return parser.parse_args(argv)


def config_from_args(args: argparse.Namespace) -> MatchupFixtureAuditConfig:
    return MatchupFixtureAuditConfig(
        seasons=args.seasons,
        project_root=args.project_root,
        output_root=args.output_root,
        expected_complete_rounds=args.expected_complete_rounds,
        complete_round_threshold=args.complete_round_threshold,
        current_year=args.current_year,
    )
