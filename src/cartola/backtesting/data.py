from __future__ import annotations

import re
import unicodedata
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from cartola.backtesting.config import DEFAULT_SCOUT_COLUMNS, POSITION_ID_TO_CODE, STATUS_ID_TO_NAME

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
    "atletas.pontos_num": "pontos",
    "atletas.media_num": "media",
    "atletas.jogos_num": "jogos",
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
    "pontos",
    "media",
    "jogos",
    "variacao",
)

NUMERIC_COLUMNS: tuple[str, ...] = (
    "rodada",
    "status",
    "posicao",
    "id_atleta",
    "id_clube",
    "preco",
    "pontos",
    "media",
    "jogos",
    "variacao",
    "minimo_para_valorizar",
    *DEFAULT_SCOUT_COLUMNS,
)

_ROUND_FILE_RE = re.compile(r"rodada-(\d+)\.csv$")


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
        normalized["entrou_em_campo"] = normalized["pontos"].notna()

    _convert_numeric_columns(normalized)
    return normalized


def _round_number(path: Path) -> int:
    match = _ROUND_FILE_RE.match(path.name)
    if not match:
        raise ValueError(f"Invalid round CSV filename: {path}")
    return int(match.group(1))


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


def _slugify(value: Any) -> str:
    text = "" if pd.isna(value) else str(value)
    ascii_text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    slug = re.sub(r"[^a-z0-9]+", "-", ascii_text.lower()).strip("-")
    return slug
