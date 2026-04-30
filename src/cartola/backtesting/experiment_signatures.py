from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Mapping, cast

import pandas as pd

from cartola.backtesting.runner import CSV_FLOAT_FORMAT

_CANDIDATE_SIGNATURE_COLUMNS: tuple[str, ...] = (
    "id_atleta",
    "posicao",
    "id_clube",
    "status",
    "preco_pre_rodada",
    "rodada",
)
_CANDIDATE_SIGNATURE_SORT_COLUMNS: tuple[str, ...] = (
    "id_atleta",
    "rodada",
    "posicao",
    "id_clube",
    "status",
    "preco_pre_rodada",
)


class ComparabilityError(ValueError):
    """Raised when experiment outputs cannot be compared safely."""


def candidate_pool_signature(frame: pd.DataFrame) -> str:
    missing_columns = [column for column in _CANDIDATE_SIGNATURE_COLUMNS if column not in frame.columns]
    if missing_columns:
        raise ComparabilityError(f"Missing required candidate columns: {', '.join(missing_columns)}")

    records = []
    for row in frame.loc[:, _CANDIDATE_SIGNATURE_COLUMNS].to_dict(orient="records"):
        record = dict(row)
        price = record["preco_pre_rodada"]
        if pd.isna(price):
            raise ComparabilityError(
                "Missing preco_pre_rodada for candidate "
                f"id_atleta={record['id_atleta']} rodada={record['rodada']}"
            )
        record["preco_pre_rodada"] = CSV_FLOAT_FORMAT % float(price)
        records.append(_json_ready(record))
    records.sort(key=_candidate_signature_sort_key)

    return _sha256_json(records)


def solver_status_signature(round_results: pd.DataFrame, *, primary_model_id: str) -> dict[str, str]:
    required_columns = ("rodada", "strategy", "solver_status")
    missing_columns = [column for column in required_columns if column not in round_results.columns]
    if missing_columns:
        raise ComparabilityError(f"Missing required solver-status columns: {', '.join(missing_columns)}")

    role_by_strategy = {
        "baseline": "baseline",
        "price": "price",
        primary_model_id: "primary_model",
    }
    signatures: dict[str, str] = {}

    for row in round_results.loc[:, required_columns].to_dict(orient="records"):
        strategy = str(row["strategy"])
        role = role_by_strategy.get(strategy)
        if role is None:
            raise ComparabilityError(f"Unexpected strategy in solver-status comparison: {strategy}")

        key = f"{_round_key(row['rodada'])}:{role}"
        solver_status = str(row["solver_status"])
        existing_status = signatures.get(key)
        if existing_status is not None and existing_status != solver_status:
            raise ComparabilityError(f"Conflicting solver statuses for {key}: {existing_status} != {solver_status}")
        signatures[key] = solver_status

    return dict(sorted(signatures.items()))


def compare_signature_sets(label: str, signatures: Mapping[str, object]) -> None:
    iterator = iter(signatures.items())
    try:
        baseline_name, baseline = next(iterator)
    except StopIteration:
        return

    for name, signature in iterator:
        if signature != baseline:
            raise ComparabilityError(f"{label} differ: {baseline_name} != {name}")


def raw_cartola_source_identity(project_root: Path, season: int) -> dict[str, object]:
    project_root = project_root.resolve()
    source_dir = project_root / "data" / "01_raw" / str(season)
    files = [_file_identity(path=path, project_root=project_root) for path in sorted(_source_files(source_dir))]
    sha256 = _sha256_json(files)

    return {
        "season": season,
        "directory": source_dir.relative_to(project_root).as_posix(),
        "files": files,
        "sha256": sha256,
    }


def _source_files(source_dir: Path) -> list[Path]:
    if not source_dir.exists():
        return []
    return [
        path
        for path in source_dir.rglob("*")
        if path.is_file() and not path.name.endswith(".capture.json")
    ]


def _file_identity(*, path: Path, project_root: Path) -> dict[str, object]:
    contents = path.read_bytes()
    return {
        "path": path.relative_to(project_root).as_posix(),
        "sha256": hashlib.sha256(contents).hexdigest(),
        "size_bytes": len(contents),
    }


def _sha256_json(payload: object) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _round_key(value: object) -> str:
    if pd.isna(value):
        raise ComparabilityError("Missing rodada in solver-status comparison")
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _candidate_signature_sort_key(record: object) -> tuple[str, ...]:
    if not isinstance(record, Mapping):
        raise TypeError("Candidate signature record must be a mapping")
    candidate_record = cast("Mapping[str, object]", record)
    return tuple(_canonical_sort_value(candidate_record[column]) for column in _CANDIDATE_SIGNATURE_SORT_COLUMNS)


def _canonical_sort_value(value: object) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
    )


def _json_ready(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_json_ready(item) for item in value]
    if value is pd.NA or pd.isna(value):
        return None
    if isinstance(value, Path):
        return value.as_posix()
    if value is None or isinstance(value, str | int | float | bool):
        return value
    return str(value)
