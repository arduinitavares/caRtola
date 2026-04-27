from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from cartola.backtesting.data import normalize_fixture_frame
from cartola.backtesting.fixture_snapshots import (
    CARTOLA_DEADLINE_ENDPOINT,
    CARTOLA_FIXTURE_ENDPOINT,
    cartola_deadline_at,
    cartola_fixture_rows,
    parse_iso_utc_z,
    validate_capture_metadata,
)

GENERATOR_VERSION = "fixture_snapshot_v1"
STRICT_SOURCE = "cartola_api"

_REQUIRED_MANIFEST_FIELDS = {
    "mode",
    "season",
    "rodada",
    "source",
    "capture_metadata_path",
    "capture_metadata_sha256",
    "fixture_snapshot_path",
    "fixture_snapshot_sha256",
    "deadline_snapshot_path",
    "deadline_snapshot_sha256",
    "captured_at_utc",
    "deadline_at_utc",
    "deadline_source",
    "fixture_endpoint",
    "fixture_final_url",
    "deadline_endpoint",
    "deadline_final_url",
    "generator_version",
    "club_mapping_path",
    "club_mapping_sha256",
    "club_id_allowlist_path",
    "club_id_allowlist_sha256",
    "canonical_fixture_path",
    "canonical_fixture_sha256",
}


@dataclass(frozen=True)
class StrictFixtureLoadResult:
    fixture_path: Path
    manifest_path: Path
    manifest: dict[str, Any]
    captured_at_utc: datetime
    deadline_at_utc: datetime
    generator_version: str


@dataclass(frozen=True)
class StrictFixturesLoadResult:
    fixtures: pd.DataFrame
    manifest_paths: list[str]
    manifest_sha256: dict[str, str]
    generator_versions: list[str]


@dataclass(frozen=True)
class _SnapshotCandidate:
    directory: Path
    capture_path: Path
    fixture_path: Path
    deadline_path: Path
    capture: dict[str, Any]
    fixture_payload: dict[str, Any]
    deadline_payload: dict[str, Any]
    captured_at_utc: datetime
    deadline_at_utc: datetime
    rows: list[dict[str, Any]]


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_strict_manifest(
    *,
    project_root: str | Path,
    fixture_path: str | Path,
    season: int,
    round_number: int,
    source: str = STRICT_SOURCE,
) -> StrictFixtureLoadResult:
    root = Path(project_root).resolve(strict=True)
    canonical_path = _resolve_under_root(root, fixture_path)
    manifest_path = canonical_path.with_suffix(".manifest.json")
    if not manifest_path.exists():
        raise FileNotFoundError(f"Strict fixture manifest does not exist: {manifest_path}")

    manifest = _read_json(manifest_path)
    missing = sorted(_REQUIRED_MANIFEST_FIELDS - manifest.keys())
    if missing:
        raise ValueError(f"Strict fixture manifest is missing required fields: {', '.join(missing)}")

    if manifest["mode"] != "strict":
        raise ValueError("Strict fixture manifest mode must be 'strict'")
    if manifest["season"] != season:
        raise ValueError(f"Strict fixture manifest season {manifest['season']!r} does not match {season}")
    if manifest["rodada"] != round_number:
        raise ValueError(f"Strict fixture manifest rodada {manifest['rodada']!r} does not match {round_number}")
    if manifest["source"] != source:
        raise ValueError(f"Strict fixture manifest source {manifest['source']!r} does not match {source!r}")
    if manifest["generator_version"] != GENERATOR_VERSION:
        raise ValueError(
            f"Strict fixture manifest generator_version {manifest['generator_version']!r} "
            f"does not match {GENERATOR_VERSION!r}"
        )
    if _fixture_round_from_name(canonical_path) != round_number:
        raise ValueError(f"Strict fixture filename does not match rodada {round_number}: {canonical_path.name}")

    manifest_canonical = _resolve_under_root(root, manifest["canonical_fixture_path"])
    if manifest_canonical != canonical_path:
        raise ValueError("Strict fixture manifest canonical_fixture_path does not identify the loaded fixture file")

    capture_path = _resolve_under_root(root, manifest["capture_metadata_path"])
    fixture_snapshot_path = _resolve_under_root(root, manifest["fixture_snapshot_path"])
    deadline_path = _resolve_under_root(root, manifest["deadline_snapshot_path"])
    _verify_hash(canonical_path, manifest["canonical_fixture_sha256"], field_name="canonical_fixture_sha256")
    _verify_hash(capture_path, manifest["capture_metadata_sha256"], field_name="capture_metadata_sha256")
    _verify_hash(fixture_snapshot_path, manifest["fixture_snapshot_sha256"], field_name="fixture_snapshot_sha256")
    _verify_hash(deadline_path, manifest["deadline_snapshot_sha256"], field_name="deadline_snapshot_sha256")
    _verify_canonical_rows_from_snapshot(
        canonical_path,
        fixture_snapshot_path,
        round_number=round_number,
    )
    _verify_optional_path_hash(
        root,
        manifest["club_mapping_path"],
        manifest["club_mapping_sha256"],
        path_field="club_mapping_path",
        hash_field="club_mapping_sha256",
    )
    _verify_optional_path_hash(
        root,
        manifest["club_id_allowlist_path"],
        manifest["club_id_allowlist_sha256"],
        path_field="club_id_allowlist_path",
        hash_field="club_id_allowlist_sha256",
    )

    captured_at = _parse_manifest_utc(manifest["captured_at_utc"], field_name="captured_at_utc")
    capture = _read_json(capture_path)
    capture_evidence = validate_capture_metadata(
        capture,
        source=source,
        season=season,
        round_number=round_number,
    )
    source_captured_at = capture_evidence.captured_at_utc
    if captured_at != source_captured_at:
        raise ValueError("captured_at_utc does not match hashed capture metadata")

    deadline_at = _parse_manifest_utc(manifest["deadline_at_utc"], field_name="deadline_at_utc")
    deadline_payload = _read_json(deadline_path)
    source_deadline_at = cartola_deadline_at(deadline_payload, season=season, round_number=round_number)
    if deadline_at != source_deadline_at:
        raise ValueError("deadline_at_utc does not match hashed deadline snapshot")
    if captured_at >= deadline_at:
        raise ValueError("captured_at_utc must be strictly before deadline_at_utc")
    expected_fixture_endpoint = CARTOLA_FIXTURE_ENDPOINT.format(round_number=round_number)
    if manifest["fixture_endpoint"] != expected_fixture_endpoint:
        raise ValueError("fixture_endpoint does not match requested Cartola fixture endpoint")
    if manifest["deadline_endpoint"] != CARTOLA_DEADLINE_ENDPOINT:
        raise ValueError("deadline_endpoint does not match Cartola market status endpoint")
    if manifest["fixture_final_url"] != capture_evidence.fixture_final_url:
        raise ValueError("fixture_final_url does not match hashed capture metadata")
    if manifest["deadline_final_url"] != capture_evidence.deadline_final_url:
        raise ValueError("deadline_final_url does not match hashed capture metadata")

    return StrictFixtureLoadResult(
        fixture_path=canonical_path,
        manifest_path=manifest_path,
        manifest=manifest,
        captured_at_utc=captured_at,
        deadline_at_utc=deadline_at,
        generator_version=str(manifest["generator_version"]),
    )


def load_strict_fixtures(
    *,
    season: int,
    project_root: str | Path,
    required_rounds: list[int],
    source: str = STRICT_SOURCE,
) -> StrictFixturesLoadResult:
    if not required_rounds:
        return StrictFixturesLoadResult(
            fixtures=pd.DataFrame(columns=pd.Index(["rodada", "id_clube_home", "id_clube_away", "data"])),
            manifest_paths=[],
            manifest_sha256={},
            generator_versions=[],
        )

    root = Path(project_root).resolve(strict=True)
    fixture_dir = root / "data" / "01_raw" / "fixtures_strict" / str(season)
    fixture_frames: list[pd.DataFrame] = []
    manifest_paths: list[str] = []
    manifest_sha256: dict[str, str] = {}
    generator_versions: set[str] = set()

    for round_number in sorted(set(required_rounds)):
        fixture_path = fixture_dir / f"partidas-{round_number}.csv"
        if not fixture_path.exists():
            raise FileNotFoundError(f"Required strict fixture does not exist: {fixture_path}")

        validation = validate_strict_manifest(
            project_root=root,
            fixture_path=fixture_path,
            season=season,
            round_number=round_number,
            source=source,
        )
        manifest_path = validation.manifest_path.relative_to(root).as_posix()
        manifest_paths.append(manifest_path)
        manifest_sha256[manifest_path] = sha256_file(validation.manifest_path)
        generator_versions.add(validation.generator_version)
        fixture_frames.append(normalize_fixture_frame(pd.read_csv(validation.fixture_path), source=validation.fixture_path))

    return StrictFixturesLoadResult(
        fixtures=pd.concat(fixture_frames, ignore_index=True),
        manifest_paths=manifest_paths,
        manifest_sha256=manifest_sha256,
        generator_versions=sorted(generator_versions),
    )


def generate_strict_fixture(
    *,
    project_root: str | Path,
    season: int,
    round_number: int,
    source: str = STRICT_SOURCE,
    captured_at: datetime | str | None = None,
    force: bool = False,
) -> StrictFixtureLoadResult:
    if source != STRICT_SOURCE:
        raise ValueError(f"Unsupported strict fixture source: {source!r}")

    root = Path(project_root).resolve()
    snapshot = _select_snapshot(
        project_root=root,
        season=season,
        round_number=round_number,
        source=source,
        captured_at=captured_at,
    )
    target_dir = root / "data" / "01_raw" / "fixtures_strict" / str(season)
    target_path = target_dir / f"partidas-{round_number}.csv"
    manifest_path = target_dir / f"partidas-{round_number}.manifest.json"
    if not force and (target_path.exists() or manifest_path.exists()):
        raise FileExistsError(f"Strict fixture target already exists; pass force=True to overwrite: {target_path}")

    target_dir.mkdir(parents=True, exist_ok=True)
    allowlist_path = _allowlist_path(root, season)
    allowlist_relative: str | None = None
    allowlist_sha: str | None = None
    if allowlist_path.exists():
        _validate_allowlist(allowlist_path, snapshot.rows)
        allowlist_relative = _relative(root, allowlist_path)
        allowlist_sha = sha256_file(allowlist_path)

    staged_fixture_path = target_dir / f".{target_path.name}.tmp"
    staged_manifest_path = target_dir / f".{manifest_path.name}.tmp"
    _write_canonical_csv(staged_fixture_path, snapshot.rows)
    manifest = {
        "mode": "strict",
        "season": season,
        "rodada": round_number,
        "source": source,
        "capture_metadata_path": _relative(root, snapshot.capture_path),
        "capture_metadata_sha256": sha256_file(snapshot.capture_path),
        "fixture_snapshot_path": _relative(root, snapshot.fixture_path),
        "fixture_snapshot_sha256": sha256_file(snapshot.fixture_path),
        "deadline_snapshot_path": _relative(root, snapshot.deadline_path),
        "deadline_snapshot_sha256": sha256_file(snapshot.deadline_path),
        "captured_at_utc": _iso_utc_z(snapshot.captured_at_utc),
        "deadline_at_utc": _iso_utc_z(snapshot.deadline_at_utc),
        "deadline_source": "cartola_api_market_status",
        "fixture_endpoint": CARTOLA_FIXTURE_ENDPOINT.format(round_number=round_number),
        "fixture_final_url": str(snapshot.capture.get("fixture_final_url", "")),
        "deadline_endpoint": CARTOLA_DEADLINE_ENDPOINT,
        "deadline_final_url": str(snapshot.capture.get("deadline_final_url", "")),
        "generator_version": GENERATOR_VERSION,
        "club_mapping_path": None,
        "club_mapping_sha256": None,
        "club_id_allowlist_path": allowlist_relative,
        "club_id_allowlist_sha256": allowlist_sha,
        "canonical_fixture_path": _relative(root, target_path),
        "canonical_fixture_sha256": sha256_file(staged_fixture_path),
    }
    try:
        _write_json(staged_manifest_path, manifest)
    except Exception:
        staged_fixture_path.unlink(missing_ok=True)
        raise

    _publish_fixture_pair(
        staged_fixture_path=staged_fixture_path,
        staged_manifest_path=staged_manifest_path,
        target_path=target_path,
        manifest_path=manifest_path,
    )
    return validate_strict_manifest(
        project_root=root,
        fixture_path=target_path,
        season=season,
        round_number=round_number,
        source=source,
    )


def _select_snapshot(
    *,
    project_root: Path,
    season: int,
    round_number: int,
    source: str,
    captured_at: datetime | str | None = None,
) -> _SnapshotCandidate:
    round_dir = project_root / "data" / "01_raw" / "fixtures_snapshots" / str(season) / f"rodada-{round_number}"
    if not round_dir.exists():
        raise FileNotFoundError(f"Fixture snapshot round directory does not exist: {round_dir}")

    requested_at = _coerce_requested_captured_at(captured_at)
    candidates: list[_SnapshotCandidate] = []
    seen: dict[datetime, Path] = {}
    for capture_dir in sorted(round_dir.glob("captured_at=*")):
        if not capture_dir.is_dir():
            continue
        try:
            candidate = _snapshot_candidate(capture_dir, season=season, round_number=round_number, source=source)
        except (FileNotFoundError, ValueError, json.JSONDecodeError):
            if requested_at is None:
                continue
            raise
        if requested_at is not None and candidate.captured_at_utc != requested_at:
            continue
        existing = seen.get(candidate.captured_at_utc)
        if existing is not None:
            raise ValueError(
                "Multiple strict fixture snapshots have the same captured_at_utc: "
                f"{existing} and {capture_dir}"
            )
        seen[candidate.captured_at_utc] = capture_dir
        if candidate.captured_at_utc < candidate.deadline_at_utc:
            candidates.append(candidate)

    if not candidates:
        if requested_at is not None:
            raise FileNotFoundError(f"No strict-valid snapshot found for captured_at={_iso_utc_z(requested_at)}")
        raise FileNotFoundError("No strict-valid fixture snapshot exists before deadline")

    return max(candidates, key=lambda item: item.captured_at_utc)


def _snapshot_candidate(
    capture_dir: Path,
    *,
    season: int,
    round_number: int,
    source: str,
) -> _SnapshotCandidate:
    capture_path = capture_dir / "capture.json"
    fixture_path = capture_dir / "fixtures.json"
    deadline_path = capture_dir / "deadline.json"
    for path in (capture_path, fixture_path, deadline_path):
        if not path.exists():
            raise FileNotFoundError(f"Strict fixture snapshot is incomplete: {path}")

    capture = _read_json(capture_path)
    fixture_payload = _read_json(fixture_path)
    deadline_payload = _read_json(deadline_path)
    capture_evidence = validate_capture_metadata(
        capture,
        source=source,
        season=season,
        round_number=round_number,
    )
    captured_at = capture_evidence.captured_at_utc
    deadline_at = cartola_deadline_at(deadline_payload, season=season, round_number=round_number)
    if captured_at >= deadline_at:
        raise ValueError("captured_at_utc must be strictly before deadline_at_utc")
    rows = cartola_fixture_rows(fixture_payload, round_number=round_number)

    return _SnapshotCandidate(
        directory=capture_dir,
        capture_path=capture_path,
        fixture_path=fixture_path,
        deadline_path=deadline_path,
        capture=capture,
        fixture_payload=fixture_payload,
        deadline_payload=deadline_payload,
        captured_at_utc=captured_at,
        deadline_at_utc=deadline_at,
        rows=rows,
    )


def _resolve_under_root(project_root: Path, path_value: str | Path) -> Path:
    path = Path(path_value)
    candidate = path if path.is_absolute() else project_root / path
    resolved = candidate.resolve(strict=True)
    try:
        resolved.relative_to(project_root)
    except ValueError as exc:
        raise ValueError(f"Manifest path resolves outside project_root: {path_value}") from exc
    return resolved


def _parse_manifest_utc(value: Any, *, field_name: str) -> datetime:
    return parse_iso_utc_z(value, field_name=field_name)


def _verify_hash(path: Path, expected_hash: Any, *, field_name: str) -> None:
    if not isinstance(expected_hash, str) or not expected_hash:
        raise ValueError(f"{field_name} must be a non-empty SHA-256 hash")
    actual_hash = sha256_file(path)
    if actual_hash != expected_hash:
        raise ValueError(f"{field_name} mismatch for {path}")


def _verify_optional_path_hash(
    root: Path,
    path_value: Any,
    expected_hash: Any,
    *,
    path_field: str,
    hash_field: str,
) -> None:
    if path_value is None:
        if expected_hash is not None:
            raise ValueError(f"{hash_field} must be null when {path_field} is null")
        return
    if expected_hash is None:
        raise ValueError(f"{hash_field} is required when {path_field} is set")
    _verify_hash(_resolve_under_root(root, path_value), expected_hash, field_name=hash_field)


def _verify_canonical_rows_from_snapshot(
    canonical_path: Path,
    fixture_snapshot_path: Path,
    *,
    round_number: int,
) -> None:
    expected_rows = cartola_fixture_rows(_read_json(fixture_snapshot_path), round_number=round_number)
    actual_rows = _read_canonical_fixture_rows(canonical_path)
    if actual_rows != expected_rows:
        raise ValueError("canonical fixture rows do not match hashed fixture snapshot")


def _read_canonical_fixture_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["rodada", "id_clube_home", "id_clube_away", "data"]:
            raise ValueError("canonical fixture rows must use expected columns")
        for row in reader:
            rows.append(
                {
                    "rodada": _csv_integer(row.get("rodada"), field_name="rodada"),
                    "id_clube_home": _csv_integer(row.get("id_clube_home"), field_name="id_clube_home"),
                    "id_clube_away": _csv_integer(row.get("id_clube_away"), field_name="id_clube_away"),
                    "data": _csv_text(row.get("data"), field_name="data"),
                }
            )
    return rows


def _csv_integer(value: str | None, *, field_name: str) -> int:
    if value is None or not value.strip().lstrip("-").isdigit():
        raise ValueError(f"canonical fixture rows field {field_name} must be an integer")
    return int(value)


def _csv_text(value: str | None, *, field_name: str) -> str:
    if value is None or not value.strip():
        raise ValueError(f"canonical fixture rows field {field_name} must be non-empty")
    return value


def _fixture_round_from_name(path: Path) -> int:
    prefix = "partidas-"
    suffix = ".csv"
    if not path.name.startswith(prefix) or not path.name.endswith(suffix):
        raise ValueError(f"Strict fixture filename must match partidas-<round>.csv: {path.name}")
    value = path.name[len(prefix) : -len(suffix)]
    if not value.isdigit():
        raise ValueError(f"Strict fixture filename round must be numeric: {path.name}")
    return int(value)


def _coerce_requested_captured_at(value: datetime | str | None) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("captured_at must be timezone-aware")
        return value.astimezone(UTC)
    return _parse_manifest_utc(value, field_name="captured_at")


def _write_canonical_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    tmp_path = path.with_name(f".{path.name}.tmp")
    fieldnames = ["rodada", "id_clube_home", "id_clube_away", "data"]
    with tmp_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    _replace_path(tmp_path, path)


def _publish_fixture_pair(
    *,
    staged_fixture_path: Path,
    staged_manifest_path: Path,
    target_path: Path,
    manifest_path: Path,
) -> None:
    fixture_backup = target_path.with_name(f".{target_path.name}.bak")
    manifest_backup = manifest_path.with_name(f".{manifest_path.name}.bak")
    fixture_existed = target_path.exists()
    manifest_existed = manifest_path.exists()
    fixture_backup_ready = False
    manifest_backup_ready = False
    _cleanup_paths(fixture_backup, manifest_backup, _backup_tmp_path(fixture_backup), _backup_tmp_path(manifest_backup))
    try:
        if fixture_existed:
            _copy_backup(target_path, fixture_backup)
            fixture_backup_ready = True
        if manifest_existed:
            _copy_backup(manifest_path, manifest_backup)
            manifest_backup_ready = True

        _replace_path(staged_fixture_path, target_path)
        _replace_path(staged_manifest_path, manifest_path)
    except Exception:
        _restore_or_remove(target_path, fixture_backup, existed=fixture_existed, backup_ready=fixture_backup_ready)
        _restore_or_remove(manifest_path, manifest_backup, existed=manifest_existed, backup_ready=manifest_backup_ready)
        raise
    finally:
        _cleanup_paths(
            staged_fixture_path,
            staged_manifest_path,
            fixture_backup,
            manifest_backup,
            _backup_tmp_path(fixture_backup),
            _backup_tmp_path(manifest_backup),
        )


def _copy_backup(source: Path, backup_path: Path) -> None:
    tmp_backup_path = _backup_tmp_path(backup_path)
    _cleanup_paths(tmp_backup_path)
    try:
        shutil.copy2(source, tmp_backup_path)
        os.replace(tmp_backup_path, backup_path)
    except Exception:
        tmp_backup_path.unlink(missing_ok=True)
        raise


def _backup_tmp_path(backup_path: Path) -> Path:
    return backup_path.with_name(f"{backup_path.name}.tmp")


def _restore_or_remove(path: Path, backup_path: Path, *, existed: bool, backup_ready: bool) -> None:
    if existed:
        if backup_ready and backup_path.exists():
            os.replace(backup_path, path)
    else:
        path.unlink(missing_ok=True)


def _cleanup_paths(*paths: Path) -> None:
    for path in paths:
        path.unlink(missing_ok=True)


def _validate_allowlist(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        allowed: set[int] = set()
        for row in reader:
            raw = row.get("id_clube") or row.get("id_clube_cartola") or row.get("clube_id")
            if raw is not None and str(raw).strip().isdigit():
                allowed.add(int(raw))
    if not allowed:
        raise ValueError(f"Club ID allowlist has no parseable club IDs: {path}")
    used = {int(row["id_clube_home"]) for row in rows} | {int(row["id_clube_away"]) for row in rows}
    unexpected = sorted(used - allowed)
    if unexpected:
        raise ValueError(f"Fixture contains club IDs outside allowlist: {unexpected}")


def _allowlist_path(root: Path, season: int) -> Path:
    return root / "data" / "01_raw" / "fixtures" / "club_ids" / f"{season}.csv"


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    tmp_path = path.with_name(f".{path.name}.tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _replace_path(tmp_path, path)


def _replace_path(source: Path, destination: Path) -> None:
    os.replace(source, destination)


def _relative(root: Path, path: Path) -> str:
    return str(path.resolve().relative_to(root))


def _iso_utc_z(value: datetime) -> str:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("timestamp must be timezone-aware")
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")
