from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

import pandas as pd

FootyStatsTableType = Literal["league", "matches", "players", "teams", "teams2"]

FOOTYSTATS_FILE_RE = re.compile(
    r"^(?P<league_slug>.+)-(?P<table_type>league|matches|players|teams|teams2)-"
    r"(?P<start_year>\d{4})-to-(?P<end_year>\d{4})-stats\.csv$"
)
SUPPORTED_TABLE_TYPES: tuple[FootyStatsTableType, ...] = ("league", "matches", "players", "teams", "teams2")
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
    "RBB": "red bull bragantino",
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
    "vasco": "vasco da gama",
    "vasco da gama": "vasco da gama",
    "sport": "sport recife",
    "sport recife": "sport recife",
    "rb bragantino": "red bull bragantino",
    "red bull bragantino": "red bull bragantino",
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


def normalize_team_name(value: str) -> str:
    abbreviation = value.strip().upper()
    if abbreviation in CARTOLA_ABBREVIATIONS:
        return CARTOLA_ABBREVIATIONS[abbreviation]

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

    for path in sorted(season_dir.glob("*.csv")):
        df = pd.read_csv(path)
        for club_id, club_name in zip(df["atletas.clube_id"], df["atletas.clube.id.full.name"], strict=False):
            if pd.isna(club_id) or pd.isna(club_name):
                continue
            normalized_name = normalize_team_name(str(club_name))
            cartola_clubs_by_normalized_name.setdefault(normalized_name, int(club_id))

    mapped_teams: dict[str, int] = {}
    unmapped_footystats_teams: list[str] = []
    for team_name in footystats_team_names:
        club_id = cartola_clubs_by_normalized_name.get(normalize_team_name(team_name))
        if club_id is None:
            unmapped_footystats_teams.append(team_name)
        else:
            mapped_teams[team_name] = club_id

    return TeamComparison(
        cartola_clubs_by_normalized_name=cartola_clubs_by_normalized_name,
        mapped_teams=mapped_teams,
        unmapped_footystats_teams=sorted(unmapped_footystats_teams),
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
