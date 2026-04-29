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
FOOTYSTATS_XG_SOURCE_COLUMNS: tuple[str, ...] = ("Home Team Pre-Match xG", "Away Team Pre-Match xG")
SOURCE_COLUMNS_BY_MODE: dict[str, tuple[str, ...]] = {
    "ppg": REQUIRED_MATCH_COLUMNS,
    "ppg_xg": (*REQUIRED_MATCH_COLUMNS, *FOOTYSTATS_XG_SOURCE_COLUMNS),
}
RESULT_COLUMNS: tuple[str, ...] = (
    "rodada",
    "id_clube",
    "opponent_id_clube",
    "is_home_footystats",
    "footystats_team_pre_match_ppg",
    "footystats_opponent_pre_match_ppg",
    "footystats_ppg_diff",
)
XG_RESULT_COLUMNS: tuple[str, ...] = (
    "footystats_team_pre_match_xg",
    "footystats_opponent_pre_match_xg",
    "footystats_xg_diff",
)
PPG_FEATURE_COLUMNS: tuple[str, ...] = (
    "footystats_team_pre_match_ppg",
    "footystats_opponent_pre_match_ppg",
    "footystats_ppg_diff",
)
XG_FEATURE_COLUMNS: tuple[str, ...] = XG_RESULT_COLUMNS
ALL_FEATURE_COLUMNS: tuple[str, ...] = (*PPG_FEATURE_COLUMNS, *XG_FEATURE_COLUMNS)
FEATURE_COLUMNS_BY_MODE: dict[str, tuple[str, ...]] = {
    "ppg": PPG_FEATURE_COLUMNS,
    "ppg_xg": (*PPG_FEATURE_COLUMNS, *XG_FEATURE_COLUMNS),
}
RESULT_COLUMNS_BY_MODE: dict[str, tuple[str, ...]] = {
    "ppg": RESULT_COLUMNS,
    "ppg_xg": (*RESULT_COLUMNS, *XG_RESULT_COLUMNS),
}


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
    footystats_mode: str = "ppg"
    feature_columns: tuple[str, ...] = PPG_FEATURE_COLUMNS


def merge_footystats_ppg(
    frame: pd.DataFrame,
    footystats_rows: pd.DataFrame | None,
    *,
    target_round: int,
) -> pd.DataFrame:
    return merge_footystats_features(frame, footystats_rows, target_round=target_round)


def merge_footystats_features(
    frame: pd.DataFrame,
    footystats_rows: pd.DataFrame | None,
    *,
    target_round: int,
) -> pd.DataFrame:
    if footystats_rows is None:
        return frame

    round_rows = footystats_rows[footystats_rows["rodada"].eq(target_round)].copy()
    duplicate_mask = round_rows.duplicated(subset=["rodada", "id_clube"], keep=False)
    if bool(duplicate_mask.any()):
        duplicates = _duplicate_key_records(round_rows[duplicate_mask])
        raise ValueError(f"Duplicate FootyStats PPG rows for round {target_round}: {duplicates}")

    candidate_clubs = sorted(
        int(club_id) for club_id in _rows_with_club_identity(frame)["id_clube"].dropna().unique()
    )
    matched_clubs = {int(club_id) for club_id in round_rows["id_clube"].dropna().unique()}
    missing_clubs = [club_id for club_id in candidate_clubs if club_id not in matched_clubs]
    if missing_clubs:
        raise ValueError(f"missing FootyStats PPG rows for round {target_round} clubs: {missing_clubs}")

    feature_columns = [column for column in ALL_FEATURE_COLUMNS if column in round_rows.columns]
    feature_rows = round_rows[["id_clube", *feature_columns]]
    return frame.merge(feature_rows, on="id_clube", how="left", validate="many_to_one")


def build_footystats_join_diagnostics(
    season_df: pd.DataFrame,
    footystats_rows: pd.DataFrame | None,
) -> FootyStatsJoinDiagnostics:
    candidate_keys = _unique_key_frame(season_df, require_club_identity=True)
    footystats_key_source = pd.DataFrame(columns=pd.Index(["rodada", "id_clube"]))
    if footystats_rows is not None:
        footystats_key_source = footystats_rows[["rodada", "id_clube"]].copy()
    footystats_keys = _unique_key_frame(footystats_key_source)

    missing = candidate_keys.merge(footystats_keys, on=["rodada", "id_clube"], how="left", indicator=True)
    missing = missing[missing["_merge"].eq("left_only")][["rodada", "id_clube"]]

    extra = footystats_keys.merge(candidate_keys, on=["rodada", "id_clube"], how="left", indicator=True)
    extra = extra[extra["_merge"].eq("left_only")][["rodada", "id_clube"]]

    duplicate_counts = _duplicate_count_frame(footystats_key_source)
    duplicate_counts = duplicate_counts[duplicate_counts["count"].gt(1)]

    return FootyStatsJoinDiagnostics(
        missing_join_keys_by_round=_group_key_records_by_round(missing),
        duplicate_join_keys_by_round=_group_key_records_by_round(duplicate_counts, include_count=True),
        extra_club_rows_by_round=_group_key_records_by_round(extra),
    )


def load_footystats_ppg_rows(
    *,
    season: int,
    project_root: Path,
    footystats_dir: Path,
    league_slug: str,
    evaluation_scope: str,
    current_year: int | None,
) -> FootyStatsPPGLoadResult:
    return load_footystats_feature_rows(
        season=season,
        project_root=project_root,
        footystats_dir=footystats_dir,
        league_slug=league_slug,
        evaluation_scope=evaluation_scope,
        current_year=current_year,
        footystats_mode="ppg",
    )


def load_footystats_feature_rows(
    *,
    season: int,
    project_root: Path,
    footystats_dir: Path,
    league_slug: str,
    evaluation_scope: str,
    current_year: int | None,
    footystats_mode: str,
) -> FootyStatsPPGLoadResult:
    if footystats_mode not in SOURCE_COLUMNS_BY_MODE:
        raise ValueError(f"Unsupported footystats_mode: {footystats_mode}")

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

    required_columns = SOURCE_COLUMNS_BY_MODE[footystats_mode]
    df = _read_source_frame(source_path, required_columns)

    game_weeks = _validated_game_weeks(df)
    home_ppg = _validated_ppg(df, "Pre-Match PPG (Home)")
    away_ppg = _validated_ppg(df, "Pre-Match PPG (Away)")
    home_xg = _validated_xg(df, "Home Team Pre-Match xG") if footystats_mode == "ppg_xg" else None
    away_xg = _validated_xg(df, "Away Team Pre-Match xG") if footystats_mode == "ppg_xg" else None
    statuses = _validated_statuses(df, evaluation_scope)
    _validate_team_names_present(df)

    if evaluation_scope == "historical_candidate":
        if any(status != "complete" for status in statuses):
            raise ValueError("historical_candidate requires all statuses to be complete")
        if sorted(set(game_weeks.astype(int).tolist())) != list(range(1, 39)):
            raise ValueError("historical_candidate requires exact game-week coverage 1..38")

    team_names = _team_names(df)
    comparison = compare_teams_to_cartola(season=season, footystats_team_names=team_names, project_root=project_root)
    _validate_team_mapping(comparison)

    rows = _build_feature_rows(
        df,
        game_weeks,
        home_ppg,
        away_ppg,
        comparison.mapped_teams,
        footystats_mode=footystats_mode,
        home_xg=home_xg,
        away_xg=away_xg,
    )
    _reject_duplicate_join_keys(rows)

    return FootyStatsPPGLoadResult(
        rows=rows,
        source_path=source_path,
        source_sha256=_sha256_file(source_path),
        diagnostics=FootyStatsJoinDiagnostics(),
        footystats_mode=footystats_mode,
        feature_columns=FEATURE_COLUMNS_BY_MODE[footystats_mode],
    )


def load_footystats_feature_rows_for_recommendation(
    *,
    season: int,
    project_root: Path,
    footystats_dir: Path,
    league_slug: str,
    current_year: int | None,
    target_round: int,
    footystats_mode: str,
    require_complete_status: bool,
    required_keys: pd.DataFrame,
) -> FootyStatsPPGLoadResult:
    if footystats_mode not in SOURCE_COLUMNS_BY_MODE:
        raise ValueError(f"Unsupported footystats_mode: {footystats_mode}")
    if target_round <= 0:
        raise ValueError("target_round must be positive")

    source_path = _source_path(
        project_root=project_root,
        footystats_dir=footystats_dir,
        league_slug=league_slug,
        season=season,
    )
    _validate_source_filename(source_path, season=season, league_slug=league_slug)

    required_columns = SOURCE_COLUMNS_BY_MODE[footystats_mode]
    df = _read_source_frame(source_path, required_columns)

    game_weeks = _validated_game_weeks(df)
    visible = game_weeks.le(target_round)
    df = df.loc[visible].copy()
    game_weeks = game_weeks.loc[visible]

    required = _unique_key_frame(required_keys)
    team_names = _team_names(df)
    comparison = compare_teams_to_cartola(season=season, footystats_team_names=team_names, project_root=project_root)

    relevant = _relevant_recommendation_match_rows(df, game_weeks, comparison.mapped_teams, required)
    df = df.loc[relevant].copy()
    game_weeks = game_weeks.loc[relevant]

    _validate_team_names_present(df)
    relevant_team_names = _team_names(df)
    relevant_comparison = compare_teams_to_cartola(
        season=season,
        footystats_team_names=relevant_team_names,
        project_root=project_root,
    )
    _validate_team_mapping(relevant_comparison, require_all_cartola_teams=False)

    home_ppg = _validated_ppg(df, "Pre-Match PPG (Home)")
    away_ppg = _validated_ppg(df, "Pre-Match PPG (Away)")
    home_xg = _validated_xg(df, "Home Team Pre-Match xG") if footystats_mode == "ppg_xg" else None
    away_xg = _validated_xg(df, "Away Team Pre-Match xG") if footystats_mode == "ppg_xg" else None
    statuses = _status_values(df)
    if require_complete_status and any(status != "complete" for status in statuses):
        raise ValueError("FootyStats recommendation rows require all visible statuses to be complete")
    if not require_complete_status:
        target_statuses = {
            status
            for status, game_week in zip(statuses, game_weeks.astype(int).tolist(), strict=True)
            if int(game_week) == target_round
        }
        invalid_target_statuses = sorted(
            status for status in target_statuses if status not in {"complete", "incomplete"}
        )
        if invalid_target_statuses:
            raise ValueError(
                "target-round FootyStats recommendation rows require status complete or incomplete"
            )

    rows = _build_feature_rows(
        df,
        game_weeks,
        home_ppg,
        away_ppg,
        relevant_comparison.mapped_teams,
        footystats_mode=footystats_mode,
        home_xg=home_xg,
        away_xg=away_xg,
    )
    rows = _filter_recommendation_rows_to_required_keys(rows, required)
    _validate_required_recommendation_keys(rows, required_keys)

    return FootyStatsPPGLoadResult(
        rows=rows,
        source_path=source_path,
        source_sha256=_sha256_file(source_path),
        diagnostics=FootyStatsJoinDiagnostics(),
        footystats_mode=footystats_mode,
        feature_columns=FEATURE_COLUMNS_BY_MODE[footystats_mode],
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


def _read_source_frame(source_path: Path, required_columns: tuple[str, ...]) -> pd.DataFrame:
    header = pd.read_csv(source_path, nrows=0)
    _require_columns(header, required_columns)
    allowed_columns = set(required_columns)
    return pd.read_csv(source_path, usecols=lambda column: column in allowed_columns)


def _require_columns(df: pd.DataFrame, required_columns: tuple[str, ...]) -> None:
    missing_columns = [column for column in required_columns if column not in df.columns]
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


def _validated_xg(df: pd.DataFrame, column: str) -> pd.Series:
    values = pd.to_numeric(df[column], errors="coerce")
    if bool(values.isna().any()):
        raise ValueError(f"FootyStats matches file has missing or non-numeric xG values in {column}")
    return values.astype(float)


def _validated_statuses(df: pd.DataFrame, evaluation_scope: str) -> list[str]:
    statuses = _status_values(df)
    if evaluation_scope == "live_current":
        invalid_statuses = sorted({status for status in statuses if status not in {"complete", "incomplete"}})
        if invalid_statuses:
            raise ValueError("live_current requires statuses to be only complete or incomplete")
    return statuses


def _status_values(df: pd.DataFrame) -> list[str]:
    return df["status"].map(lambda value: "" if pd.isna(value) else str(value).strip().lower()).tolist()


def _validate_team_names_present(df: pd.DataFrame) -> None:
    missing_columns = []
    for column in ("home_team_name", "away_team_name"):
        names = df[column]
        missing = names.isna() | names.map(lambda value: isinstance(value, str) and value.strip() == "")
        if bool(missing.any()):
            missing_columns.append(column)
    if missing_columns:
        raise ValueError(f"FootyStats matches file has missing team names in {', '.join(missing_columns)}")


def _team_names(df: pd.DataFrame) -> list[str]:
    teams = pd.concat([df["home_team_name"], df["away_team_name"]], ignore_index=True).dropna()
    return sorted({str(team) for team in teams})


def _validate_team_mapping(comparison, *, require_all_cartola_teams: bool = True) -> None:
    failures: list[str] = []
    if comparison.unmapped_footystats_teams:
        failures.append(f"unmapped FootyStats teams: {', '.join(comparison.unmapped_footystats_teams)}")
    if require_all_cartola_teams and comparison.missing_cartola_teams:
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
    *,
    footystats_mode: str,
    home_xg: pd.Series | None,
    away_xg: pd.Series | None,
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
        home_row: dict[str, object] = {
            "rodada": rodada,
            "id_clube": home_id,
            "opponent_id_clube": away_id,
            "is_home_footystats": 1,
            "footystats_team_pre_match_ppg": home_value,
            "footystats_opponent_pre_match_ppg": away_value,
            "footystats_ppg_diff": home_value - away_value,
        }
        away_row: dict[str, object] = {
            "rodada": rodada,
            "id_clube": away_id,
            "opponent_id_clube": home_id,
            "is_home_footystats": 0,
            "footystats_team_pre_match_ppg": away_value,
            "footystats_opponent_pre_match_ppg": home_value,
            "footystats_ppg_diff": away_value - home_value,
        }
        if footystats_mode == "ppg_xg":
            if home_xg is None or away_xg is None:
                raise ValueError("ppg_xg mode requires pre-match xG values")
            home_xg_value = float(home_xg.loc[index])
            away_xg_value = float(away_xg.loc[index])
            home_row.update(
                {
                    "footystats_team_pre_match_xg": home_xg_value,
                    "footystats_opponent_pre_match_xg": away_xg_value,
                    "footystats_xg_diff": home_xg_value - away_xg_value,
                }
            )
            away_row.update(
                {
                    "footystats_team_pre_match_xg": away_xg_value,
                    "footystats_opponent_pre_match_xg": home_xg_value,
                    "footystats_xg_diff": away_xg_value - home_xg_value,
                }
            )

        rows.append(home_row)
        rows.append(away_row)

    result = pd.DataFrame(rows, columns=pd.Index(RESULT_COLUMNS_BY_MODE[footystats_mode]))
    return result


def _reject_duplicate_join_keys(rows: pd.DataFrame) -> None:
    duplicates = rows.duplicated(subset=["rodada", "id_clube"], keep=False)
    if bool(duplicates.any()):
        raise ValueError("FootyStats PPG rows contain duplicate normalized (rodada, id_clube) rows")


def _relevant_recommendation_match_rows(
    df: pd.DataFrame,
    game_weeks: pd.Series,
    mapped_teams: dict[str, int],
    required: pd.DataFrame,
) -> pd.Series:
    if df.empty or required.empty:
        return pd.Series(False, index=df.index)

    required_pairs = {
        (int(row["rodada"]), int(row["id_clube"]))
        for _, row in required[["rodada", "id_clube"]].iterrows()
    }

    relevant_values: list[bool] = []
    for index, match in df.iterrows():
        rodada = int(game_weeks.loc[index])
        home_id = _mapped_team_id(match["home_team_name"], mapped_teams)
        away_id = _mapped_team_id(match["away_team_name"], mapped_teams)
        relevant_values.append((rodada, home_id) in required_pairs or (rodada, away_id) in required_pairs)
    return pd.Series(relevant_values, index=df.index)


def _mapped_team_id(team_name: object, mapped_teams: dict[str, int]) -> int | None:
    if pd.isna(team_name):
        return None
    return mapped_teams.get(str(team_name))


def _filter_recommendation_rows_to_required_keys(rows: pd.DataFrame, required: pd.DataFrame) -> pd.DataFrame:
    if rows.empty or required.empty:
        return rows.iloc[0:0].copy()

    required_pairs = {
        (int(row["rodada"]), int(row["id_clube"]))
        for _, row in required[["rodada", "id_clube"]].iterrows()
    }
    keep = [
        (int(row["rodada"]), int(row["id_clube"])) in required_pairs
        for _, row in rows[["rodada", "id_clube"]].iterrows()
    ]
    return rows.loc[keep].copy()


def _validate_required_recommendation_keys(rows: pd.DataFrame, required_keys: pd.DataFrame) -> None:
    if required_keys.empty:
        return

    required = _unique_key_frame(required_keys)
    available = _unique_key_frame(rows)

    missing = required.merge(available, on=["rodada", "id_clube"], how="left", indicator=True)
    missing = missing[missing["_merge"].eq("left_only")][["rodada", "id_clube"]]
    if not missing.empty:
        raise ValueError(
            "missing FootyStats recommendation rows: "
            f"{_group_key_records_by_round(missing)}"
        )

    duplicate_counts = _duplicate_count_frame(rows)
    duplicates = duplicate_counts.merge(required, on=["rodada", "id_clube"], how="inner")
    duplicates = duplicates[duplicates["count"].gt(1)]
    if not duplicates.empty:
        raise ValueError(
            "duplicate FootyStats recommendation rows: "
            f"{_group_key_records_by_round(duplicates, include_count=True)}"
        )


def _unique_key_frame(df: pd.DataFrame, *, require_club_identity: bool = False) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=pd.Index(["rodada", "id_clube"]))
    source = _rows_with_club_identity(df) if require_club_identity else df
    keys = source[["rodada", "id_clube"]].dropna().drop_duplicates().copy()
    keys["rodada"] = keys["rodada"].astype(int)
    keys["id_clube"] = keys["id_clube"].astype(int)
    return keys.sort_values(["rodada", "id_clube"]).reset_index(drop=True)


def _rows_with_club_identity(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "nome_clube" not in df.columns:
        return df

    club_names = df["nome_clube"]
    has_name = club_names.notna() & club_names.map(lambda value: str(value).strip() != "")
    return df.loc[has_name]


def _duplicate_key_records(df: pd.DataFrame) -> list[dict[str, int]]:
    duplicate_counts = _duplicate_count_frame(df).sort_values(["rodada", "id_clube"])
    return [
        {"rodada": int(row["rodada"]), "id_clube": int(row["id_clube"]), "count": int(row["count"])}
        for _, row in duplicate_counts.iterrows()
    ]


def _duplicate_count_frame(df: pd.DataFrame) -> pd.DataFrame:
    counts: dict[tuple[int, int], int] = {}
    if not df.empty:
        for rodada, id_clube in df[["rodada", "id_clube"]].dropna().itertuples(index=False, name=None):
            key = (int(rodada), int(id_clube))
            counts[key] = counts.get(key, 0) + 1
    return pd.DataFrame(
        [
            {"rodada": rodada, "id_clube": id_clube, "count": count}
            for (rodada, id_clube), count in sorted(counts.items())
        ],
        columns=pd.Index(["rodada", "id_clube", "count"]),
    )


def _group_key_records_by_round(
    df: pd.DataFrame,
    *,
    include_count: bool = False,
) -> dict[str, list[dict[str, int]]]:
    if df.empty:
        return {}

    grouped: dict[str, list[dict[str, int]]] = {}
    ordered = df.sort_values(["rodada", "id_clube"]).reset_index(drop=True)
    for _, row in ordered.iterrows():
        rodada = int(row["rodada"])
        record = {"rodada": rodada, "id_clube": int(row["id_clube"])}
        if include_count:
            record["count"] = int(row["count"])
        grouped.setdefault(str(rodada), []).append(record)
    return grouped


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
