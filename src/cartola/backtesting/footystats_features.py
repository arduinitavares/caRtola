from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from cartola.backtesting.footystats_audit import compare_teams_to_cartola, parse_footystats_filename


REQUIRED_MATCH_COLUMNS: tuple[str, ...] = (
    "Game Week",
    "home_team_name",
    "away_team_name",
    "Pre-Match PPG (Home)",
    "Pre-Match PPG (Away)",
    "status",
)
PPG_COLUMNS: tuple[str, ...] = ("Pre-Match PPG (Home)", "Pre-Match PPG (Away)")
RESULT_COLUMNS: tuple[str, ...] = (
    "rodada",
    "id_clube",
    "opponent_id_clube",
    "is_home_footystats",
    "footystats_team_pre_match_ppg",
    "footystats_opponent_pre_match_ppg",
    "footystats_ppg_diff",
)


@dataclass(frozen=True)
class FootyStatsJoinDiagnostics:
    missing_join_keys_by_round: dict[str, list[dict[str, int]]] = field(default_factory=dict)
    duplicate_join_keys_by_round: dict[str, list[dict[str, int]]] = field(default_factory=dict)
    extra_club_rows_by_round: dict[str, list[dict[str, int]]] = field(default_factory=dict)


@dataclass(frozen=True)
class FootyStatsPPGLoadResult:
    rows: pd.DataFrame
    source_path: Path
    source_sha256: str
    diagnostics: FootyStatsJoinDiagnostics


def load_footystats_ppg_rows(
    *,
    season: int,
    project_root: Path,
    footystats_dir: Path,
    league_slug: str,
    evaluation_scope: str,
    current_year: int | None,
) -> FootyStatsPPGLoadResult:
    source_path = _source_path(
        project_root=project_root,
        footystats_dir=footystats_dir,
        league_slug=league_slug,
        season=season,
    )
    _validate_source_filename(source_path, season=season, league_slug=league_slug)

    if evaluation_scope not in {"historical_candidate", "live_current"}:
        raise ValueError(f"Unsupported FootyStats evaluation_scope: {evaluation_scope}")
    if evaluation_scope == "live_current":
        resolved_current_year = current_year if current_year is not None else datetime.now(UTC).year
        if season != resolved_current_year:
            raise ValueError(f"live_current requires season {season} to equal current_year {resolved_current_year}")

    df = pd.read_csv(source_path)
    _require_columns(df)

    game_weeks = _validated_game_weeks(df)
    home_ppg = _validated_ppg(df, "Pre-Match PPG (Home)")
    away_ppg = _validated_ppg(df, "Pre-Match PPG (Away)")
    statuses = _validated_statuses(df, evaluation_scope)

    if evaluation_scope == "historical_candidate":
        if any(status != "complete" for status in statuses):
            raise ValueError("historical_candidate requires all statuses to be complete")
        if sorted(set(game_weeks.astype(int).tolist())) != list(range(1, 39)):
            raise ValueError("historical_candidate requires exact game-week coverage 1..38")

    team_names = _team_names(df)
    comparison = compare_teams_to_cartola(season=season, footystats_team_names=team_names, project_root=project_root)
    _validate_team_mapping(comparison)

    rows = _build_feature_rows(df, game_weeks, home_ppg, away_ppg, comparison.mapped_teams)
    _reject_duplicate_join_keys(rows)

    return FootyStatsPPGLoadResult(
        rows=rows,
        source_path=source_path,
        source_sha256=_sha256_file(source_path),
        diagnostics=FootyStatsJoinDiagnostics(),
    )


def _source_path(*, project_root: Path, footystats_dir: Path, league_slug: str, season: int) -> Path:
    resolved_dir = footystats_dir if footystats_dir.is_absolute() else project_root / footystats_dir
    return resolved_dir / f"{league_slug}-matches-{season}-to-{season}-stats.csv"


def _validate_source_filename(source_path: Path, *, season: int, league_slug: str) -> None:
    parsed = parse_footystats_filename(source_path)
    if parsed.season != season:
        raise ValueError(f"FootyStats filename season {parsed.season} does not match requested season {season}")
    if parsed.league_slug != league_slug:
        raise ValueError(
            f"FootyStats filename league slug {parsed.league_slug!r} does not match requested league slug "
            f"{league_slug!r}"
        )
    if parsed.table_type != "matches":
        raise ValueError(f"FootyStats filename table type {parsed.table_type!r} does not match required 'matches'")


def _require_columns(df: pd.DataFrame) -> None:
    missing_columns = [column for column in REQUIRED_MATCH_COLUMNS if column not in df.columns]
    if missing_columns:
        raise ValueError(f"FootyStats matches file missing required columns: {', '.join(missing_columns)}")


def _validated_game_weeks(df: pd.DataFrame) -> pd.Series:
    game_weeks = pd.to_numeric(df["Game Week"], errors="coerce")
    valid = game_weeks.notna() & game_weeks.mod(1).eq(0) & game_weeks.gt(0)
    if not bool(valid.all()):
        raise ValueError("FootyStats matches file requires positive integer Game Week values")
    return game_weeks.astype(int)


def _validated_ppg(df: pd.DataFrame, column: str) -> pd.Series:
    values = pd.to_numeric(df[column], errors="coerce")
    if bool(values.isna().any()):
        raise ValueError(f"FootyStats matches file has missing or non-numeric PPG values in {column}")
    return values.astype(float)


def _validated_statuses(df: pd.DataFrame, evaluation_scope: str) -> list[str]:
    statuses = df["status"].map(lambda value: "" if pd.isna(value) else str(value).strip().lower()).tolist()
    if evaluation_scope == "live_current":
        invalid_statuses = sorted({status for status in statuses if status not in {"complete", "incomplete"}})
        if invalid_statuses:
            raise ValueError("live_current requires statuses to be only complete or incomplete")
    return statuses


def _team_names(df: pd.DataFrame) -> list[str]:
    teams = pd.concat([df["home_team_name"], df["away_team_name"]], ignore_index=True).dropna()
    return sorted({str(team) for team in teams})


def _validate_team_mapping(comparison) -> None:
    failures: list[str] = []
    if comparison.unmapped_footystats_teams:
        failures.append(f"unmapped FootyStats teams: {', '.join(comparison.unmapped_footystats_teams)}")
    if comparison.missing_cartola_teams:
        failures.append(f"missing Cartola teams: {', '.join(comparison.missing_cartola_teams)}")
    if comparison.duplicate_mapped_cartola_teams:
        failures.append(f"duplicate mapped Cartola teams: {', '.join(comparison.duplicate_mapped_cartola_teams)}")
    if failures:
        raise ValueError("FootyStats team mapping failed: " + "; ".join(failures))


def _build_feature_rows(
    df: pd.DataFrame,
    game_weeks: pd.Series,
    home_ppg: pd.Series,
    away_ppg: pd.Series,
    mapped_teams: dict[str, int],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for index, match in df.iterrows():
        home_team = str(match["home_team_name"])
        away_team = str(match["away_team_name"])
        home_id = mapped_teams[home_team]
        away_id = mapped_teams[away_team]
        rodada = int(game_weeks.loc[index])
        home_value = float(home_ppg.loc[index])
        away_value = float(away_ppg.loc[index])

        rows.append(
            {
                "rodada": rodada,
                "id_clube": home_id,
                "opponent_id_clube": away_id,
                "is_home_footystats": True,
                "footystats_team_pre_match_ppg": home_value,
                "footystats_opponent_pre_match_ppg": away_value,
                "footystats_ppg_diff": home_value - away_value,
            }
        )
        rows.append(
            {
                "rodada": rodada,
                "id_clube": away_id,
                "opponent_id_clube": home_id,
                "is_home_footystats": False,
                "footystats_team_pre_match_ppg": away_value,
                "footystats_opponent_pre_match_ppg": home_value,
                "footystats_ppg_diff": away_value - home_value,
            }
        )

    result = pd.DataFrame(rows, columns=pd.Index(RESULT_COLUMNS))
    result["is_home_footystats"] = result["is_home_footystats"].astype(object)
    return result


def _reject_duplicate_join_keys(rows: pd.DataFrame) -> None:
    duplicates = rows.duplicated(subset=["rodada", "id_clube"], keep=False)
    if bool(duplicates.any()):
        raise ValueError("FootyStats PPG rows contain duplicate normalized (rodada, id_clube) rows")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
