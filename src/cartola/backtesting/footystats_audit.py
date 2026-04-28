from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

FootyStatsTableType = Literal["league", "matches", "players", "teams", "teams2"]

FOOTYSTATS_FILE_RE = re.compile(
    r"^(?P<league_slug>.+)-(?P<table_type>league|matches|players|teams|teams2)-"
    r"(?P<start_year>\d{4})-to-(?P<end_year>\d{4})-stats\.csv$"
)
SUPPORTED_TABLE_TYPES: tuple[FootyStatsTableType, ...] = ("league", "matches", "players", "teams", "teams2")


@dataclass(frozen=True)
class FootyStatsAuditConfig:
    project_root: Path = Path(".")
    footystats_dir: Path = Path("data/footystats")
    output_root: Path = Path("data/08_reporting/footystats")


@dataclass(frozen=True)
class ParsedFootyStatsFilename:
    league_slug: str
    table_type: FootyStatsTableType
    start_year: int
    end_year: int

    @property
    def season(self) -> int:
        return self.start_year


@dataclass(frozen=True)
class FootyStatsSeasonDiscovery:
    season: int
    league_slug: str
    files: dict[FootyStatsTableType, Path]


def parse_footystats_filename(path: Path) -> ParsedFootyStatsFilename:
    match = FOOTYSTATS_FILE_RE.match(path.name)
    if not match:
        raise ValueError(f"Invalid FootyStats CSV filename: {path.name}")

    table_type = match.group("table_type")
    if table_type not in SUPPORTED_TABLE_TYPES:
        raise ValueError(f"Unsupported FootyStats table type: {table_type}")
    table_type = cast(FootyStatsTableType, table_type)

    start_year = int(match.group("start_year"))
    end_year = int(match.group("end_year"))
    if start_year != end_year:
        raise ValueError("FootyStats audit supports single-year seasons only")

    return ParsedFootyStatsFilename(
        league_slug=match.group("league_slug"),
        table_type=table_type,
        start_year=start_year,
        end_year=end_year,
    )


def discover_footystats_files(config: FootyStatsAuditConfig) -> list[FootyStatsSeasonDiscovery]:
    footystats_dir = _resolve_path(config.project_root, config.footystats_dir)
    grouped: dict[tuple[int, str], dict[FootyStatsTableType, Path]] = {}

    for path in sorted(footystats_dir.glob("*.csv")):
        parsed = parse_footystats_filename(path)
        grouped.setdefault((parsed.season, parsed.league_slug), {})[parsed.table_type] = path

    return [
        FootyStatsSeasonDiscovery(season=season, league_slug=league_slug, files=files)
        for (season, league_slug), files in sorted(grouped.items())
    ]


def _resolve_path(project_root: Path, path: Path) -> Path:
    if path.is_absolute():
        return path
    return project_root / path
