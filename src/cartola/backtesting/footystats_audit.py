from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, cast

import pandas as pd

FootyStatsTableType = Literal["league", "matches", "players", "teams", "teams2"]

FOOTYSTATS_FILE_RE = re.compile(
    r"^(?P<league_slug>.+)-(?P<table_type>league|matches|players|teams|teams2)-"
    r"(?P<start_year>\d{4})-to-(?P<end_year>\d{4})-stats\.csv$"
)
SUPPORTED_TABLE_TYPES: tuple[FootyStatsTableType, ...] = ("league", "matches", "players", "teams", "teams2")
REQUIRED_PRE_MATCH_SAFE_COLUMNS: tuple[str, ...] = ("Pre-Match PPG (Home)", "Pre-Match PPG (Away)")
PRE_MATCH_SAFE_COLUMNS = (
    "Pre-Match PPG (Home)",
    "Pre-Match PPG (Away)",
    "Home Team Pre-Match xG",
    "Away Team Pre-Match xG",
    "average_goals_per_match_pre_match",
    "btts_percentage_pre_match",
    "over_15_percentage_pre_match",
    "over_25_percentage_pre_match",
    "over_35_percentage_pre_match",
    "over_45_percentage_pre_match",
    "over_15_HT_FHG_percentage_pre_match",
    "over_05_HT_FHG_percentage_pre_match",
    "over_15_2HG_percentage_pre_match",
    "over_05_2HG_percentage_pre_match",
    "average_corners_per_match_pre_match",
    "average_cards_per_match_pre_match",
    "odds_ft_home_team_win",
    "odds_ft_draw",
    "odds_ft_away_team_win",
    "odds_ft_over15",
    "odds_ft_over25",
    "odds_ft_over35",
    "odds_ft_over45",
    "odds_btts_yes",
    "odds_btts_no",
)
POST_MATCH_OUTCOME_COLUMNS = (
    "home_team_goal_count",
    "away_team_goal_count",
    "total_goal_count",
    "total_goals_at_half_time",
    "home_team_goal_count_half_time",
    "away_team_goal_count_half_time",
    "home_team_corner_count",
    "away_team_corner_count",
    "total_corner_count",
    "home_team_yellow_cards",
    "home_team_red_cards",
    "away_team_yellow_cards",
    "away_team_red_cards",
    "home_team_shots",
    "away_team_shots",
    "home_team_shots_on_target",
    "away_team_shots_on_target",
    "home_team_fouls",
    "away_team_fouls",
    "home_team_possession",
    "away_team_possession",
    "team_a_xg",
    "team_b_xg",
)
CARTOLA_ABBREVIATIONS = {
    "FLA": "flamengo",
    "BOT": "botafogo",
    "COR": "corinthians",
    "BAH": "bahia",
    "FLU": "fluminense",
    "VAS": "vasco da gama",
    "PAL": "palmeiras",
    "SAO": "sao paulo",
    "SAN": "santos",
    "RBB": "bragantino",
    "CAM": "atletico mineiro",
    "CRU": "cruzeiro",
    "GRE": "gremio",
    "INT": "internacional",
    "JUV": "juventude",
    "VIT": "vitoria",
    "SPT": "sport recife",
    "CEA": "ceara",
    "FOR": "fortaleza",
    "MIR": "mirassol",
    "CAP": "atletico pr",
    "CFC": "coritiba",
    "CHA": "chapecoense",
    "REM": "remo",
}
TEAM_SUFFIXES = (
    " futebol clube",
    " football club",
    " esporte clube",
    " sport club",
    " club de regatas",
    " clube de regatas",
    " fc",
    " clube",
)
TEAM_ALIASES = {
    "athletico paranaense": "atletico pr",
    "athletico pr": "atletico pr",
    "atletico paranaense": "atletico pr",
    "atletico pr": "atletico pr",
    "atletico mg": "atletico mineiro",
    "atletico mineiro": "atletico mineiro",
    "america mg": "america mineiro",
    "america mineiro": "america mineiro",
    "vasco": "vasco da gama",
    "vasco da gama": "vasco da gama",
    "sport": "sport recife",
    "sport recife": "sport recife",
    "rb bragantino": "bragantino",
    "red bull bragantino": "bragantino",
    "sao paulo": "sao paulo",
}


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


@dataclass(frozen=True)
class MatchFileProfile:
    path: Path
    row_count: int
    status_counts: dict[str, int]
    min_game_week: int | None
    max_game_week: int | None
    game_week_count: int
    team_names: list[str]
    pre_match_safe_columns: tuple[str, ...]
    post_match_outcome_columns: tuple[str, ...]
    pre_match_missing_counts: dict[str, int]
    pre_match_zero_counts: dict[str, int]


@dataclass(frozen=True)
class TeamComparison:
    cartola_clubs_by_normalized_name: dict[str, int]
    mapped_teams: dict[str, int]
    unmapped_footystats_teams: list[str]
    missing_cartola_teams: list[str]


@dataclass(frozen=True)
class FootyStatsSeasonAuditRecord:
    season: int
    league_slug: str
    available_files: list[str]
    league_status: str
    match_status: str
    team_mapping_status: str
    integration_status: str
    match_row_count: int | None
    min_game_week: int | None
    max_game_week: int | None
    game_week_count: int | None
    status_counts: dict[str, int]
    footystats_team_count: int
    mapped_team_count: int
    unmapped_footystats_teams: list[str]
    missing_cartola_teams: list[str]
    pre_match_safe_columns: list[str]
    post_match_outcome_columns: list[str]
    missing_safe_columns: list[str]
    pre_match_missing_counts: dict[str, int]
    pre_match_zero_counts: dict[str, int]
    notes: list[str] = field(default_factory=list)

    def to_csv_row(self) -> dict[str, object]:
        return {
            "season": self.season,
            "league_slug": self.league_slug,
            "available_files": _pipe_join(self.available_files),
            "league_status": self.league_status,
            "match_status": self.match_status,
            "team_mapping_status": self.team_mapping_status,
            "integration_status": self.integration_status,
            "match_row_count": self.match_row_count,
            "min_game_week": self.min_game_week,
            "max_game_week": self.max_game_week,
            "game_week_count": self.game_week_count,
            "status_counts": json.dumps(self.status_counts, sort_keys=True),
            "footystats_team_count": self.footystats_team_count,
            "mapped_team_count": self.mapped_team_count,
            "unmapped_footystats_teams": _pipe_join(self.unmapped_footystats_teams),
            "missing_cartola_teams": _pipe_join(self.missing_cartola_teams),
            "pre_match_safe_columns": _pipe_join(self.pre_match_safe_columns),
            "post_match_outcome_columns": _pipe_join(self.post_match_outcome_columns),
            "missing_safe_columns": _pipe_join(self.missing_safe_columns),
            "pre_match_missing_counts": json.dumps(self.pre_match_missing_counts, sort_keys=True),
            "pre_match_zero_counts": json.dumps(self.pre_match_zero_counts, sort_keys=True),
            "notes": "; ".join(self.notes),
        }

    def to_json_object(self) -> dict[str, object]:
        row = self.to_csv_row()
        row["available_files"] = self.available_files
        row["status_counts"] = self.status_counts
        row["unmapped_footystats_teams"] = self.unmapped_footystats_teams
        row["missing_cartola_teams"] = self.missing_cartola_teams
        row["pre_match_safe_columns"] = self.pre_match_safe_columns
        row["post_match_outcome_columns"] = self.post_match_outcome_columns
        row["missing_safe_columns"] = self.missing_safe_columns
        row["pre_match_missing_counts"] = self.pre_match_missing_counts
        row["pre_match_zero_counts"] = self.pre_match_zero_counts
        row["notes"] = self.notes
        return row


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


def profile_match_file(path: Path) -> MatchFileProfile:
    df = pd.read_csv(path)
    pre_match_safe_columns = tuple(column for column in df.columns if column in PRE_MATCH_SAFE_COLUMNS)
    post_match_outcome_columns = tuple(column for column in df.columns if column in POST_MATCH_OUTCOME_COLUMNS)

    game_weeks = pd.to_numeric(df["Game Week"], errors="coerce").dropna() if "Game Week" in df else pd.Series(dtype="float64")
    min_game_week = int(game_weeks.min()) if not game_weeks.empty else None
    max_game_week = int(game_weeks.max()) if not game_weeks.empty else None

    return MatchFileProfile(
        path=path,
        row_count=len(df),
        status_counts=_value_counts(df, "status"),
        min_game_week=min_game_week,
        max_game_week=max_game_week,
        game_week_count=int(game_weeks.nunique()) if not game_weeks.empty else 0,
        team_names=_team_names(df),
        pre_match_safe_columns=pre_match_safe_columns,
        post_match_outcome_columns=post_match_outcome_columns,
        pre_match_missing_counts=_numeric_missing_counts(df, pre_match_safe_columns),
        pre_match_zero_counts=_numeric_zero_counts(df, pre_match_safe_columns),
    )


def audit_one_footystats_season(
    discovery: FootyStatsSeasonDiscovery,
    config: FootyStatsAuditConfig,
) -> FootyStatsSeasonAuditRecord:
    available_files = [str(table_type) for table_type in sorted(discovery.files)]
    league_path = discovery.files.get("league")
    match_path = discovery.files.get("matches")
    league_status = "ok" if league_path is not None and league_path.exists() else "missing"

    if match_path is None or not match_path.exists():
        return FootyStatsSeasonAuditRecord(
            season=discovery.season,
            league_slug=discovery.league_slug,
            available_files=available_files,
            league_status=league_status,
            match_status="missing",
            team_mapping_status="skipped",
            integration_status="not_candidate",
            match_row_count=None,
            min_game_week=None,
            max_game_week=None,
            game_week_count=None,
            status_counts={},
            footystats_team_count=0,
            mapped_team_count=0,
            unmapped_footystats_teams=[],
            missing_cartola_teams=[],
            pre_match_safe_columns=[],
            post_match_outcome_columns=[],
            missing_safe_columns=[],
            pre_match_missing_counts={},
            pre_match_zero_counts={},
            notes=["missing matches file"],
        )

    profile = profile_match_file(match_path)
    comparison = compare_teams_to_cartola(discovery.season, profile.team_names, config.project_root)
    missing_safe_columns = [
        column for column in REQUIRED_PRE_MATCH_SAFE_COLUMNS if column not in profile.pre_match_safe_columns
    ]
    team_mapping_status = (
        "failed"
        if not profile.team_names or comparison.unmapped_footystats_teams or comparison.missing_cartola_teams
        else "ok"
    )
    notes: list[str] = []

    has_complete_game_week_coverage = (
        profile.min_game_week == 1 and profile.max_game_week == 38 and profile.game_week_count == 38
    )
    if not has_complete_game_week_coverage:
        notes.append("match file does not cover game weeks 1-38")

    if not profile.team_names:
        notes.append("match file has no team names")

    if comparison.missing_cartola_teams:
        notes.append("cartola teams missing from footystats match file")

    has_fixture_status_data = bool(profile.status_counts)
    if not has_fixture_status_data:
        notes.append("match file has no fixture status data")

    has_non_complete_fixtures = _has_non_complete_fixtures(profile.status_counts)
    if has_non_complete_fixtures:
        notes.append("match file contains non-complete fixtures")

    if team_mapping_status == "failed" or missing_safe_columns or not has_fixture_status_data:
        integration_status = "not_candidate"
    elif has_non_complete_fixtures or not has_complete_game_week_coverage:
        integration_status = "partial_current"
    else:
        integration_status = "candidate"

    return FootyStatsSeasonAuditRecord(
        season=discovery.season,
        league_slug=discovery.league_slug,
        available_files=available_files,
        league_status=league_status,
        match_status="ok",
        team_mapping_status=team_mapping_status,
        integration_status=integration_status,
        match_row_count=profile.row_count,
        min_game_week=profile.min_game_week,
        max_game_week=profile.max_game_week,
        game_week_count=profile.game_week_count,
        status_counts=profile.status_counts,
        footystats_team_count=len(profile.team_names),
        mapped_team_count=len(comparison.mapped_teams),
        unmapped_footystats_teams=comparison.unmapped_footystats_teams,
        missing_cartola_teams=comparison.missing_cartola_teams,
        pre_match_safe_columns=list(profile.pre_match_safe_columns),
        post_match_outcome_columns=list(profile.post_match_outcome_columns),
        missing_safe_columns=missing_safe_columns,
        pre_match_missing_counts=profile.pre_match_missing_counts,
        pre_match_zero_counts=profile.pre_match_zero_counts,
        notes=notes,
    )


def normalize_team_name(value: str) -> str:
    abbreviation = value.strip().upper()
    if abbreviation in CARTOLA_ABBREVIATIONS:
        return CARTOLA_ABBREVIATIONS[abbreviation]

    try:
        value = value.encode("latin1").decode("utf-8")
    except UnicodeError:
        pass

    normalized = unicodedata.normalize("NFKD", value)
    normalized = "".join(char for char in normalized if not unicodedata.combining(char))
    normalized = normalized.lower()
    normalized = re.sub(r"[^a-z0-9]+", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()

    for suffix in TEAM_SUFFIXES:
        if normalized.endswith(suffix):
            normalized = normalized[: -len(suffix)].strip()
            break

    return TEAM_ALIASES.get(normalized, normalized)


def compare_teams_to_cartola(season: int, footystats_team_names: list[str], project_root: Path) -> TeamComparison:
    season_dir = project_root / "data" / "01_raw" / str(season)
    cartola_clubs_by_normalized_name: dict[str, int] = {}
    required_columns = {"atletas.clube_id", "atletas.clube.id.full.name"}

    for path in sorted(season_dir.glob("rodada-*.csv")):
        df = pd.read_csv(path, usecols=lambda column: column in required_columns)
        if not required_columns.issubset(df.columns):
            continue
        for club_id, club_name in zip(df["atletas.clube_id"], df["atletas.clube.id.full.name"], strict=False):
            if pd.isna(club_id) or pd.isna(club_name):
                continue
            normalized_name = normalize_team_name(str(club_name))
            cartola_clubs_by_normalized_name.setdefault(normalized_name, int(club_id))

    mapped_teams: dict[str, int] = {}
    unmapped_footystats_teams: list[str] = []
    mapped_normalized_names: set[str] = set()
    for team_name in footystats_team_names:
        normalized_team_name = normalize_team_name(team_name)
        club_id = cartola_clubs_by_normalized_name.get(normalized_team_name)
        if club_id is None:
            unmapped_footystats_teams.append(team_name)
        else:
            mapped_teams[team_name] = club_id
            mapped_normalized_names.add(normalized_team_name)

    missing_cartola_teams = sorted(set(cartola_clubs_by_normalized_name) - mapped_normalized_names)

    return TeamComparison(
        cartola_clubs_by_normalized_name=cartola_clubs_by_normalized_name,
        mapped_teams=mapped_teams,
        unmapped_footystats_teams=sorted(unmapped_footystats_teams),
        missing_cartola_teams=missing_cartola_teams,
    )


def _resolve_path(project_root: Path, path: Path) -> Path:
    if path.is_absolute():
        return path
    return project_root / path


def _value_counts(df: pd.DataFrame, column: str) -> dict[str, int]:
    if column not in df:
        return {}
    return {str(key): int(value) for key, value in df[column].value_counts(dropna=False).items()}


def _team_names(df: pd.DataFrame) -> list[str]:
    team_columns = [column for column in ("home_team_name", "away_team_name") if column in df]
    if not team_columns:
        return []
    teams = pd.concat([df[column] for column in team_columns], ignore_index=True).dropna()
    return sorted({str(team) for team in teams})


def _numeric_missing_counts(df: pd.DataFrame, columns: tuple[str, ...]) -> dict[str, int]:
    return {column: int(pd.to_numeric(df[column], errors="coerce").isna().sum()) for column in columns}


def _numeric_zero_counts(df: pd.DataFrame, columns: tuple[str, ...]) -> dict[str, int]:
    return {column: int((pd.to_numeric(df[column], errors="coerce") == 0).sum()) for column in columns}


def _has_non_complete_fixtures(status_counts: dict[str, int]) -> bool:
    return any(status.lower() != "complete" for status in status_counts)


def _pipe_join(values: list[str]) -> str:
    return "|".join(values)
