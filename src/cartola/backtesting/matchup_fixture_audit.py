from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Callable, Sequence

import pandas as pd

from cartola.backtesting.data import normalize_fixture_frame
from cartola.backtesting.strict_fixtures import sha256_file, validate_strict_manifest

DEFAULT_SEASONS = (2023, 2024, 2025)
DEFAULT_OUTPUT_ROOT = Path("data/08_reporting/fixtures")
CSV_ERROR_MESSAGE_LIMIT = 300

STATUS_OK = "ok"
STATUS_FAILED = "failed"
STATUS_SKIPPED = "skipped"

CONTEXT_COLUMNS: tuple[str, ...] = (
    "season",
    "rodada",
    "id_clube",
    "opponent_id_clube",
    "opponent_nome_clube",
    "is_home",
    "fixture_source",
    "source_file",
    "source_manifest",
    "source_sha256",
    "source_manifest_sha256",
)


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


def load_round_fixture_context(
    project_root: str | Path,
    *,
    season: int,
    round_number: int,
) -> pd.DataFrame:
    root = Path(project_root)
    strict_path = root / "data" / "01_raw" / "fixtures_strict" / str(season) / f"partidas-{round_number}.csv"
    if strict_path.exists():
        validate_strict_manifest(
            project_root=root,
            fixture_path=strict_path,
            season=season,
            round_number=round_number,
        )
        strict_frame = normalize_fixture_frame(pd.read_csv(strict_path), source=strict_path)
        return _context_rows_from_fixture(
            strict_frame,
            season=season,
            source="strict",
            source_file=strict_path,
            source_manifest=strict_path.with_suffix(".manifest.json"),
            project_root=root,
        )

    exploratory_path = root / "data" / "01_raw" / "fixtures" / str(season) / f"partidas-{round_number}.csv"
    if exploratory_path.exists():
        exploratory_frame = normalize_fixture_frame(pd.read_csv(exploratory_path), source=exploratory_path)
        return _context_rows_from_fixture(
            exploratory_frame,
            season=season,
            source="exploratory",
            source_file=exploratory_path,
            source_manifest=None,
            project_root=root,
        )

    return _empty_context_frame()


def _empty_context_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=pd.Index(CONTEXT_COLUMNS))


def _context_rows_from_fixture(
    fixture_frame: pd.DataFrame,
    *,
    season: int,
    source: str,
    source_file: Path,
    source_manifest: Path | None,
    project_root: Path,
) -> pd.DataFrame:
    source_file_label = _relative_path(source_file, project_root)
    source_manifest_label = None if source_manifest is None else _relative_path(source_manifest, project_root)
    source_hash = sha256_file(source_file)
    manifest_hash = None if source_manifest is None else sha256_file(source_manifest)
    rows: list[dict[str, object]] = []

    for fixture in fixture_frame.to_dict("records"):
        round_number = int(fixture["rodada"])
        home_id = int(fixture["id_clube_home"])
        away_id = int(fixture["id_clube_away"])
        rows.append(
            _context_row(
                season=season,
                round_number=round_number,
                club_id=home_id,
                opponent_id=away_id,
                is_home=1,
                source=source,
                source_file=source_file_label,
                source_manifest=source_manifest_label,
                source_hash=source_hash,
                manifest_hash=manifest_hash,
            )
        )
        rows.append(
            _context_row(
                season=season,
                round_number=round_number,
                club_id=away_id,
                opponent_id=home_id,
                is_home=0,
                source=source,
                source_file=source_file_label,
                source_manifest=source_manifest_label,
                source_hash=source_hash,
                manifest_hash=manifest_hash,
            )
        )

    return pd.DataFrame(rows, columns=pd.Index(CONTEXT_COLUMNS))


def _context_row(
    *,
    season: int,
    round_number: int,
    club_id: int,
    opponent_id: int,
    is_home: int,
    source: str,
    source_file: str,
    source_manifest: str | None,
    source_hash: str,
    manifest_hash: str | None,
) -> dict[str, object]:
    return {
        "season": season,
        "rodada": round_number,
        "id_clube": club_id,
        "opponent_id_clube": opponent_id,
        "opponent_nome_clube": None,
        "is_home": is_home,
        "fixture_source": source,
        "source_file": source_file,
        "source_manifest": source_manifest,
        "source_sha256": source_hash,
        "source_manifest_sha256": manifest_hash,
    }


def _relative_path(path: Path, project_root: Path) -> str:
    resolved_root = project_root.resolve()
    resolved_path = path.resolve()
    try:
        return resolved_path.relative_to(resolved_root).as_posix()
    except ValueError:
        return str(resolved_path)
