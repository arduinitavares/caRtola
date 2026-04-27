from __future__ import annotations

import re
import unicodedata
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from cartola.backtesting.config import (
    DEFAULT_SCOUT_COLUMNS,
    MARKET_OPEN_PRICE_COLUMN,
    POSITION_ID_TO_CODE,
    STATUS_ID_TO_NAME,
)

RAW_COLUMN_RENAMES: Mapping[str, str] = {
    "atletas.rodada_id": "rodada",
    "atletas.status_id": "status",
    "atletas.posicao_id": "posicao",
    "atletas.atleta_id": "id_atleta",
    "atletas.apelido": "apelido",
    "atletas.slug": "slug",
    "atletas.clube_id": "id_clube",
    "atletas.clube.id.full.name": "nome_clube",
    "atletas.preco_num": "preco",
    "atletas.pontos_num": "pontuacao",
    "atletas.media_num": "media",
    "atletas.jogos_num": "num_jogos",
    "atletas.variacao_num": "variacao",
    "atletas.entrou_em_campo": "entrou_em_campo",
    "atletas.minimo_para_valorizar": "minimo_para_valorizar",
    "atletas.apelido_abreviado": "apelido_abreviado",
    "atletas.nome": "nome",
    "atletas.foto": "foto",
}

REQUIRED_COLUMNS: tuple[str, ...] = (
    "rodada",
    "status",
    "posicao",
    "id_atleta",
    "apelido",
    "id_clube",
    "preco",
    "pontuacao",
    "media",
    "num_jogos",
    "variacao",
)

NUMERIC_COLUMNS: tuple[str, ...] = (
    "rodada",
    "status",
    "posicao",
    "id_atleta",
    "id_clube",
    "preco",
    "pontuacao",
    "media",
    "num_jogos",
    "variacao",
    "minimo_para_valorizar",
    MARKET_OPEN_PRICE_COLUMN,
    *DEFAULT_SCOUT_COLUMNS,
)

_ROUND_FILE_RE = re.compile(r"rodada-(\d+)\.csv$")
FIXTURE_REQUIRED_COLUMNS: tuple[str, ...] = ("rodada", "id_clube_home", "id_clube_away", "data")
_FIXTURE_FILE_RE = re.compile(r"partidas-(\d+)\.csv$")


def load_round_file(path: str | Path) -> pd.DataFrame:
    csv_path = Path(path)
    return normalize_round_frame(pd.read_csv(csv_path), source=csv_path)


def load_season_data(season: int, project_root: str | Path = ".") -> pd.DataFrame:
    season_dir = Path(project_root) / "data" / "01_raw" / str(season)
    if not season_dir.exists():
        raise FileNotFoundError(f"Season directory not found: {season_dir}")
    if not season_dir.is_dir():
        raise NotADirectoryError(f"Season path is not a directory: {season_dir}")

    round_files = sorted(season_dir.glob("rodada-*.csv"), key=_round_number)
    if not round_files:
        raise FileNotFoundError(f"No round CSV files found in season directory: {season_dir}")

    return pd.concat((load_round_file(path) for path in round_files), ignore_index=True)


def load_fixtures(season: int, project_root: str | Path = ".") -> pd.DataFrame:
    fixture_dir = Path(project_root) / "data" / "01_raw" / "fixtures" / str(season)
    if not fixture_dir.exists():
        raise FileNotFoundError(f"Fixture directory not found: {fixture_dir}")
    if not fixture_dir.is_dir():
        raise NotADirectoryError(f"Fixture path is not a directory: {fixture_dir}")

    fixture_files = sorted(fixture_dir.glob("partidas-*.csv"), key=_fixture_round_number)
    if not fixture_files:
        raise FileNotFoundError(f"No fixture CSV files found in fixture directory: {fixture_dir}")

    fixture_frames = []
    for path in fixture_files:
        fixture_frame = normalize_fixture_frame(pd.read_csv(path), source=path)
        file_round = _fixture_round_number(path)
        mismatched_rounds = sorted(set(fixture_frame.loc[fixture_frame["rodada"] != file_round, "rodada"].tolist()))
        if mismatched_rounds:
            raise ValueError(f"Fixture row rodada does not match fixture filename {path}: {mismatched_rounds}")
        fixture_frames.append(fixture_frame)

    fixtures = pd.concat(fixture_frames, ignore_index=True)
    _validate_fixture_club_entries(fixtures)
    return fixtures


def normalize_fixture_frame(frame: pd.DataFrame, source: str | Path) -> pd.DataFrame:
    source_path = Path(source)
    normalized = frame.copy()
    normalized = normalized.drop(columns=[column for column in normalized.columns if _is_saved_index_column(column)])

    missing = [column for column in FIXTURE_REQUIRED_COLUMNS if column not in normalized.columns]
    if missing:
        raise ValueError(f"Missing required fixture columns in {source_path}: {missing}")

    normalized = normalized.loc[:, list(FIXTURE_REQUIRED_COLUMNS)]
    for column in ("rodada", "id_clube_home", "id_clube_away"):
        normalized[column] = pd.to_numeric(normalized[column], errors="raise").astype(int)
    normalized["data"] = pd.to_datetime(normalized["data"], errors="raise").dt.date

    _validate_fixture_club_entries(normalized)
    return normalized


def build_round_alignment_report(
    fixtures: pd.DataFrame,
    season_df: pd.DataFrame,
    official_fixtures: pd.DataFrame | None = None,
) -> pd.DataFrame:
    fixtures = fixtures.copy()
    season_df = season_df.copy()
    fixtures["rodada"] = pd.to_numeric(fixtures["rodada"], errors="raise").astype(int)
    season_df["rodada"] = pd.to_numeric(season_df["rodada"], errors="raise").astype(int)
    if official_fixtures is not None:
        official_fixtures = official_fixtures.copy()
        official_fixtures["rodada"] = pd.to_numeric(official_fixtures["rodada"], errors="raise").astype(int)

    rows: list[dict[str, object]] = []
    fixture_rounds = set(fixtures["rodada"].tolist())
    played_rounds = set(season_df["rodada"].tolist())

    for round_number in sorted(fixture_rounds | played_rounds):
        round_fixtures = fixtures.loc[fixtures["rodada"] == round_number]
        fixture_clubs = _fixture_club_set(round_fixtures)
        played_clubs = played_club_set(season_df, round_number)
        missing = sorted(played_clubs - fixture_clubs)
        extra = sorted(fixture_clubs - played_clubs)

        discarded_match_count = 0
        discarded_clubs: list[int] = []
        if official_fixtures is not None:
            round_official = official_fixtures.loc[official_fixtures["rodada"] == round_number]
            discarded_match_count, discarded_clubs = _discarded_official_summary(round_official, fixture_clubs)

        rows.append(
            {
                "rodada": round_number,
                "fixture_club_count": len(fixture_clubs),
                "played_club_count": len(played_clubs),
                "missing_from_fixtures": _format_club_set(missing),
                "extra_in_fixtures": _format_club_set(extra),
                "discarded_official_match_count": discarded_match_count,
                "discarded_official_clubs": _format_club_set(discarded_clubs),
                "is_valid": not missing and not extra,
            }
        )

    return pd.DataFrame(
        rows,
        columns=[
            "rodada",
            "fixture_club_count",
            "played_club_count",
            "missing_from_fixtures",
            "extra_in_fixtures",
            "discarded_official_match_count",
            "discarded_official_clubs",
            "is_valid",
        ],
    )


def normalize_round_frame(frame: pd.DataFrame, source: str | Path) -> pd.DataFrame:
    source_path = Path(source)
    normalized = frame.copy()
    normalized = normalized.drop(columns=[column for column in normalized.columns if _is_saved_index_column(column)])
    normalized = normalized.rename(columns=RAW_COLUMN_RENAMES)

    missing = [column for column in REQUIRED_COLUMNS if column not in normalized.columns]
    if missing:
        raise ValueError(f"Missing required columns in {source_path}: {missing}")

    normalized["status"] = _map_values(
        normalized["status"],
        STATUS_ID_TO_NAME,
        known_names=set(STATUS_ID_TO_NAME.values()),
        source=source_path,
        label="status_id",
    )
    normalized["posicao"] = _map_values(
        normalized["posicao"],
        POSITION_ID_TO_CODE,
        known_names=set(POSITION_ID_TO_CODE.values()),
        source=source_path,
        label="position_id",
    )

    for column in DEFAULT_SCOUT_COLUMNS:
        if column not in normalized.columns:
            normalized[column] = 0

    if "slug" not in normalized.columns:
        normalized["slug"] = normalized["apelido"].map(_slugify)
    if "nome_clube" not in normalized.columns:
        normalized["nome_clube"] = normalized["id_clube"].astype("string")
    if "entrou_em_campo" not in normalized.columns:
        normalized["entrou_em_campo"] = normalized["pontuacao"].notna()

    _convert_numeric_columns(normalized)
    _add_market_open_price(normalized)
    return normalized


def _round_number(path: Path) -> int:
    match = _ROUND_FILE_RE.match(path.name)
    if not match:
        raise ValueError(f"Invalid round CSV filename: {path}")
    return int(match.group(1))


def _fixture_round_number(path: Path) -> int:
    match = _FIXTURE_FILE_RE.match(path.name)
    if not match:
        raise ValueError(f"Invalid fixture CSV filename: {path}")
    return int(match.group(1))


def _validate_fixture_club_entries(fixtures: pd.DataFrame) -> None:
    self_matches = fixtures["id_clube_home"] == fixtures["id_clube_away"]
    if self_matches.any():
        raise ValueError("Fixture rows cannot have the same home and away club")

    for round_number, round_fixtures in fixtures.groupby("rodada", sort=True):
        clubs = pd.concat(
            [round_fixtures["id_clube_home"], round_fixtures["id_clube_away"]],
            ignore_index=True,
        )
        duplicated_clubs = sorted(clubs.loc[clubs.duplicated()].astype(int).unique().tolist())
        if duplicated_clubs:
            raise ValueError(f"Duplicate fixture club entries in round {round_number}: {duplicated_clubs}")


def _fixture_club_set(fixtures: pd.DataFrame) -> set[int]:
    if fixtures.empty:
        return set()
    clubs = pd.concat([fixtures["id_clube_home"], fixtures["id_clube_away"]], ignore_index=True)
    return set(pd.to_numeric(clubs, errors="raise").astype(int).tolist())


def _discarded_official_summary(official_fixtures: pd.DataFrame, fixture_clubs: set[int]) -> tuple[int, list[int]]:
    discarded_match_count = 0
    discarded_clubs: set[int] = set()
    for _, row in official_fixtures.iterrows():
        match_clubs = {int(row["id_clube_home"]), int(row["id_clube_away"])}
        extra_clubs = match_clubs - fixture_clubs
        if extra_clubs:
            discarded_match_count += 1
            discarded_clubs.update(extra_clubs)
    return discarded_match_count, sorted(discarded_clubs)


def played_club_set(season_df: pd.DataFrame, round_number: int) -> set[int]:
    round_mask = pd.to_numeric(season_df["rodada"], errors="raise").eq(round_number)
    round_players = season_df.loc[round_mask]
    if "entrou_em_campo" in round_players.columns:
        round_players = round_players.loc[_entry_flag_mask(round_players["entrou_em_campo"])]
    return set(pd.to_numeric(round_players["id_clube"], errors="raise").dropna().astype(int).tolist())


def _entry_flag_mask(series: pd.Series) -> pd.Series:
    true_values = {True, 1, "1", "true"}
    false_values = {False, 0, "0", "false", ""}
    parsed: list[bool] = []
    invalid_values: list[Any] = []

    for value in series.tolist():
        if pd.isna(value):
            parsed.append(False)
            continue

        comparable = value.strip().lower() if isinstance(value, str) else value
        if comparable in true_values:
            parsed.append(True)
        elif comparable in false_values:
            parsed.append(False)
        else:
            invalid_values.append(value)
            parsed.append(False)

    if invalid_values:
        raise ValueError(f"Invalid entrou_em_campo values: {sorted(set(invalid_values), key=repr)}")
    return pd.Series(parsed, index=series.index)


def _format_club_set(values: list[int]) -> str:
    return ",".join(str(value) for value in sorted(values))


def _is_saved_index_column(column: Any) -> bool:
    if column is None:
        return True
    column_name = str(column)
    return column_name == "" or re.fullmatch(r"Unnamed: \d+", column_name) is not None


def _map_values(
    series: pd.Series,
    mapping: Mapping[int, str],
    *,
    known_names: set[str],
    source: Path,
    label: str,
) -> pd.Series:
    def map_one(value: Any) -> str | None:
        if pd.isna(value):
            return None
        if isinstance(value, str) and value in known_names:
            return value

        numeric_value = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
        if pd.notna(numeric_value):
            numeric_key = int(numeric_value)
            if numeric_key in mapping:
                return mapping[numeric_key]

        return None

    mapped = series.map(map_one)
    unknown_values = sorted(
        {value for value, mapped_value in zip(series.tolist(), mapped.tolist()) if mapped_value is None},
        key=repr,
    )
    if unknown_values:
        raise ValueError(f"Unknown {label} values in {source}: {unknown_values}")
    return mapped


def _convert_numeric_columns(frame: pd.DataFrame) -> None:
    for column in NUMERIC_COLUMNS:
        if column in frame.columns and column not in {"status", "posicao"}:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    for column in DEFAULT_SCOUT_COLUMNS:
        frame[column] = frame[column].fillna(0)


def _add_market_open_price(frame: pd.DataFrame) -> None:
    # Raw historical preco is post-round; subtract same-round variation to reconstruct market-open price.
    frame[MARKET_OPEN_PRICE_COLUMN] = (frame["preco"] - frame["variacao"].fillna(0)).round(2)


def _slugify(value: Any) -> str:
    text = "" if pd.isna(value) else str(value)
    ascii_text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    slug = re.sub(r"[^a-z0-9]+", "-", ascii_text.lower()).strip("-")
    return slug
