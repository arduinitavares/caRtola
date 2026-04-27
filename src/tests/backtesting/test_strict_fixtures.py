from __future__ import annotations

import importlib.util
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

import cartola.backtesting.strict_fixtures as strict_fixtures
from cartola.backtesting.strict_fixtures import (
    generate_strict_fixture,
    load_strict_fixtures,
    sha256_file,
    validate_strict_manifest,
)

FROZEN_FIXTURE_PAYLOAD = {
    "rodada": 12,
    "clubes": {},
    "partidas": [
        {
            "clube_casa_id": 262,
            "clube_visitante_id": 277,
            "partida_data": "2026-06-01 19:00:00",
            "timestamp": 1780340400,
            "valida": True,
            "placar_oficial_mandante": 3,
            "placar_oficial_visitante": 1,
        },
        {
            "clube_casa_id": 275,
            "clube_visitante_id": 284,
            "partida_data": "2026-06-01 21:30:00",
            "timestamp": 1780349400,
            "valida": False,
        },
    ],
}

FROZEN_DEADLINE_PAYLOAD = {
    "temporada": 2026,
    "rodada_atual": 12,
    "status_mercado": 1,
    "fechamento": {
        "dia": 1,
        "mes": 6,
        "ano": 2026,
        "hora": 18,
        "minuto": 59,
        "timestamp": 1780340340,
    },
}


def test_validate_strict_manifest_rejects_missing_manifest(tmp_path: Path) -> None:
    fixture_path = _write_valid_fixture(tmp_path)

    with pytest.raises(FileNotFoundError, match="manifest"):
        validate_strict_manifest(
            project_root=tmp_path,
            fixture_path=fixture_path,
            season=2026,
            round_number=12,
            source="cartola_api",
        )


def test_validate_strict_manifest_rejects_edited_fixture_hash(tmp_path: Path) -> None:
    fixture_path, _ = _write_valid_fixture_and_manifest(tmp_path)
    fixture_path.write_text("rodada,id_clube_home,id_clube_away,data\n12,999,277,2026-06-01\n", encoding="utf-8")

    with pytest.raises(ValueError, match="canonical_fixture_sha256"):
        validate_strict_manifest(
            project_root=tmp_path,
            fixture_path=fixture_path,
            season=2026,
            round_number=12,
            source="cartola_api",
        )


def test_validate_strict_manifest_rejects_canonical_rows_not_derived_from_snapshot(tmp_path: Path) -> None:
    fixture_path, manifest_path = _write_valid_fixture_and_manifest(tmp_path)
    fixture_path.write_text("rodada,id_clube_home,id_clube_away,data\n12,999,277,2026-06-01\n", encoding="utf-8")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["canonical_fixture_sha256"] = sha256_file(fixture_path)
    manifest_path.write_text(json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="canonical fixture rows"):
        validate_strict_manifest(
            project_root=tmp_path,
            fixture_path=fixture_path,
            season=2026,
            round_number=12,
            source="cartola_api",
        )


def test_validate_strict_manifest_rejects_round_mismatch(tmp_path: Path) -> None:
    fixture_path, manifest_path = _write_valid_fixture_and_manifest(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["rodada"] = 13
    manifest_path.write_text(json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="rodada"):
        validate_strict_manifest(
            project_root=tmp_path,
            fixture_path=fixture_path,
            season=2026,
            round_number=12,
            source="cartola_api",
        )


def test_validate_strict_manifest_rejects_non_utc_manifest_timestamp(tmp_path: Path) -> None:
    fixture_path, manifest_path = _write_valid_fixture_and_manifest(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["captured_at_utc"] = "2026-06-01T15:00:00-03:00"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="captured_at_utc"):
        validate_strict_manifest(
            project_root=tmp_path,
            fixture_path=fixture_path,
            season=2026,
            round_number=12,
            source="cartola_api",
        )


def test_validate_strict_manifest_rejects_tampered_captured_at_even_when_hashes_match(tmp_path: Path) -> None:
    fixture_path, manifest_path = _write_valid_fixture_and_manifest(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["captured_at_utc"] = "2026-06-01T18:30:00Z"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="captured_at_utc"):
        validate_strict_manifest(
            project_root=tmp_path,
            fixture_path=fixture_path,
            season=2026,
            round_number=12,
            source="cartola_api",
        )


def test_validate_strict_manifest_rejects_missing_capture_http_date_evidence(tmp_path: Path) -> None:
    fixture_path, manifest_path = _write_valid_fixture_and_manifest(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    capture_path = tmp_path / manifest["capture_metadata_path"]
    capture = json.loads(capture_path.read_text(encoding="utf-8"))
    del capture["fixture_http_date_header"]
    _write_json(capture_path, capture)
    manifest["capture_metadata_sha256"] = sha256_file(capture_path)
    manifest_path.write_text(json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="fixture_http_date_header"):
        validate_strict_manifest(
            project_root=tmp_path,
            fixture_path=fixture_path,
            season=2026,
            round_number=12,
            source="cartola_api",
        )


def test_validate_strict_manifest_rejects_capture_status_not_ok(tmp_path: Path) -> None:
    fixture_path, manifest_path = _write_valid_fixture_and_manifest(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    capture_path = tmp_path / manifest["capture_metadata_path"]
    capture = json.loads(capture_path.read_text(encoding="utf-8"))
    capture["fixture_http_status"] = 500
    _write_json(capture_path, capture)
    manifest["capture_metadata_sha256"] = sha256_file(capture_path)
    manifest_path.write_text(json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="fixture_http_status"):
        validate_strict_manifest(
            project_root=tmp_path,
            fixture_path=fixture_path,
            season=2026,
            round_number=12,
            source="cartola_api",
        )


def test_validate_strict_manifest_rejects_manifest_final_url_mismatch(tmp_path: Path) -> None:
    fixture_path, manifest_path = _write_valid_fixture_and_manifest(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["fixture_final_url"] = "https://example.invalid/wrong"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="fixture_final_url"):
        validate_strict_manifest(
            project_root=tmp_path,
            fixture_path=fixture_path,
            season=2026,
            round_number=12,
            source="cartola_api",
        )


def test_validate_strict_manifest_rejects_manifest_endpoint_mismatch(tmp_path: Path) -> None:
    fixture_path, manifest_path = _write_valid_fixture_and_manifest(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["fixture_endpoint"] = "https://example.invalid/partidas/12"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="fixture_endpoint"):
        validate_strict_manifest(
            project_root=tmp_path,
            fixture_path=fixture_path,
            season=2026,
            round_number=12,
            source="cartola_api",
        )


def test_validate_strict_manifest_rejects_tampered_deadline_at_even_when_hashes_match(tmp_path: Path) -> None:
    fixture_path, manifest_path = _write_valid_fixture_and_manifest(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["deadline_at_utc"] = "2026-06-01T20:00:00Z"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="deadline_at_utc"):
        validate_strict_manifest(
            project_root=tmp_path,
            fixture_path=fixture_path,
            season=2026,
            round_number=12,
            source="cartola_api",
        )


def test_validate_strict_manifest_rejects_unknown_generator_version(tmp_path: Path) -> None:
    fixture_path, manifest_path = _write_valid_fixture_and_manifest(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["generator_version"] = "fixture_snapshot_v2"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="generator_version"):
        validate_strict_manifest(
            project_root=tmp_path,
            fixture_path=fixture_path,
            season=2026,
            round_number=12,
            source="cartola_api",
        )


def test_validate_strict_manifest_rejects_path_escape(tmp_path: Path) -> None:
    fixture_path, manifest_path = _write_valid_fixture_and_manifest(tmp_path)
    outside = tmp_path.parent / f"{tmp_path.name}-outside-capture.json"
    outside.write_text("{}", encoding="utf-8")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["capture_metadata_path"] = str(outside)
    manifest["capture_metadata_sha256"] = sha256_file(outside)
    manifest_path.write_text(json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="project_root"):
        validate_strict_manifest(
            project_root=tmp_path,
            fixture_path=fixture_path,
            season=2026,
            round_number=12,
            source="cartola_api",
        )


def test_generate_strict_fixture_uses_latest_valid_snapshot_before_deadline(tmp_path: Path) -> None:
    _write_snapshot(tmp_path, captured_at=datetime(2026, 6, 1, 17, 0, tzinfo=UTC), home_id=262)
    _write_snapshot(tmp_path, captured_at=datetime(2026, 6, 1, 18, 30, tzinfo=UTC), home_id=263)

    result = generate_strict_fixture(project_root=tmp_path, season=2026, round_number=12, source="cartola_api")

    frame = pd.read_csv(result.fixture_path)
    assert frame["id_clube_home"].tolist() == [263]
    assert result.captured_at_utc == datetime(2026, 6, 1, 18, 30, tzinfo=UTC)
    validate_strict_manifest(
        project_root=tmp_path,
        fixture_path=result.fixture_path,
        season=2026,
        round_number=12,
        source="cartola_api",
    )


def test_generate_strict_fixture_skips_post_deadline_snapshot_for_automatic_selection(tmp_path: Path) -> None:
    _write_snapshot(tmp_path, captured_at=datetime(2026, 6, 1, 18, 0, tzinfo=UTC), home_id=262)
    _write_snapshot(tmp_path, captured_at=datetime(2026, 6, 1, 19, 0, tzinfo=UTC), home_id=999)

    result = generate_strict_fixture(project_root=tmp_path, season=2026, round_number=12, source="cartola_api")

    frame = pd.read_csv(result.fixture_path)
    assert frame["id_clube_home"].tolist() == [262]
    assert result.captured_at_utc == datetime(2026, 6, 1, 18, 0, tzinfo=UTC)


def test_generate_strict_fixture_rejects_explicit_post_deadline_snapshot(tmp_path: Path) -> None:
    _write_snapshot(tmp_path, captured_at=datetime(2026, 6, 1, 19, 0, tzinfo=UTC), home_id=999)

    with pytest.raises(ValueError, match="captured_at_utc"):
        generate_strict_fixture(
            project_root=tmp_path,
            season=2026,
            round_number=12,
            source="cartola_api",
            captured_at=datetime(2026, 6, 1, 19, 0, tzinfo=UTC),
        )


def test_generate_strict_fixture_rejects_explicit_snapshot_missing_capture_evidence(tmp_path: Path) -> None:
    snapshot_dir = _write_snapshot(tmp_path, captured_at=datetime(2026, 6, 1, 18, 0, tzinfo=UTC))
    capture_path = snapshot_dir / "capture.json"
    capture = json.loads(capture_path.read_text(encoding="utf-8"))
    del capture["deadline_http_date_header"]
    _write_json(capture_path, capture)

    with pytest.raises(ValueError, match="deadline_http_date_header"):
        generate_strict_fixture(
            project_root=tmp_path,
            season=2026,
            round_number=12,
            source="cartola_api",
            captured_at=datetime(2026, 6, 1, 18, 0, tzinfo=UTC),
        )


def test_generate_strict_fixture_skips_later_incomplete_snapshot_for_automatic_selection(tmp_path: Path) -> None:
    _write_snapshot(tmp_path, captured_at=datetime(2026, 6, 1, 18, 0, tzinfo=UTC), home_id=262)
    incomplete_dir = (
        tmp_path
        / "data"
        / "01_raw"
        / "fixtures_snapshots"
        / "2026"
        / "rodada-12"
        / "captured_at=2026-06-01T18-30-00Z"
    )
    incomplete_dir.mkdir(parents=True)
    _write_json(incomplete_dir / "capture.json", {"captured_at_utc": "2026-06-01T18:30:00Z"})

    result = generate_strict_fixture(project_root=tmp_path, season=2026, round_number=12, source="cartola_api")

    frame = pd.read_csv(result.fixture_path)
    assert frame["id_clube_home"].tolist() == [262]
    assert result.captured_at_utc == datetime(2026, 6, 1, 18, 0, tzinfo=UTC)


def test_generate_strict_fixture_refuses_overwrite_without_force(tmp_path: Path) -> None:
    _write_snapshot(tmp_path, captured_at=datetime(2026, 6, 1, 18, 0, tzinfo=UTC))
    generate_strict_fixture(project_root=tmp_path, season=2026, round_number=12, source="cartola_api")

    with pytest.raises(FileExistsError, match="force"):
        generate_strict_fixture(project_root=tmp_path, season=2026, round_number=12, source="cartola_api")


def test_generate_strict_fixture_force_failure_preserves_existing_fixture_and_manifest(tmp_path: Path) -> None:
    _write_snapshot(tmp_path, captured_at=datetime(2026, 6, 1, 18, 0, tzinfo=UTC), home_id=262)
    initial = generate_strict_fixture(project_root=tmp_path, season=2026, round_number=12, source="cartola_api")
    initial_fixture_hash = sha256_file(initial.fixture_path)
    initial_manifest_hash = sha256_file(initial.manifest_path)
    _write_snapshot(tmp_path, captured_at=datetime(2026, 6, 1, 18, 30, tzinfo=UTC), home_id=999)
    allowlist_path = tmp_path / "data" / "01_raw" / "fixtures" / "club_ids" / "2026.csv"
    allowlist_path.parent.mkdir(parents=True)
    allowlist_path.write_text("id_clube\n262\n277\n", encoding="utf-8")

    with pytest.raises(ValueError, match="allowlist"):
        generate_strict_fixture(project_root=tmp_path, season=2026, round_number=12, source="cartola_api", force=True)

    assert sha256_file(initial.fixture_path) == initial_fixture_hash
    assert sha256_file(initial.manifest_path) == initial_manifest_hash


def test_generate_strict_fixture_rolls_back_when_final_manifest_replace_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_snapshot(tmp_path, captured_at=datetime(2026, 6, 1, 18, 0, tzinfo=UTC), home_id=262)
    initial = generate_strict_fixture(project_root=tmp_path, season=2026, round_number=12, source="cartola_api")
    initial_fixture_hash = sha256_file(initial.fixture_path)
    initial_manifest_hash = sha256_file(initial.manifest_path)
    _write_snapshot(tmp_path, captured_at=datetime(2026, 6, 1, 18, 30, tzinfo=UTC), home_id=263)
    original_replace = strict_fixtures._replace_path

    def fail_manifest_replace(source: Path, destination: Path) -> None:
        if destination == initial.manifest_path:
            raise RuntimeError("manifest replace failed")
        original_replace(source, destination)

    monkeypatch.setattr(strict_fixtures, "_replace_path", fail_manifest_replace)

    with pytest.raises(RuntimeError, match="manifest replace failed"):
        generate_strict_fixture(project_root=tmp_path, season=2026, round_number=12, source="cartola_api", force=True)

    assert sha256_file(initial.fixture_path) == initial_fixture_hash
    assert sha256_file(initial.manifest_path) == initial_manifest_hash
    validate_strict_manifest(
        project_root=tmp_path,
        fixture_path=initial.fixture_path,
        season=2026,
        round_number=12,
        source="cartola_api",
    )


def test_generate_strict_fixture_keeps_existing_pair_when_backup_copy_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_snapshot(tmp_path, captured_at=datetime(2026, 6, 1, 18, 0, tzinfo=UTC), home_id=262)
    initial = generate_strict_fixture(project_root=tmp_path, season=2026, round_number=12, source="cartola_api")
    initial_fixture_hash = sha256_file(initial.fixture_path)
    initial_manifest_hash = sha256_file(initial.manifest_path)
    _write_snapshot(tmp_path, captured_at=datetime(2026, 6, 1, 18, 30, tzinfo=UTC), home_id=263)
    original_copy2 = strict_fixtures.shutil.copy2

    def fail_fixture_backup_copy(source: Path, destination: Path) -> Path:
        if destination.name == f".{initial.fixture_path.name}.bak.tmp":
            destination.write_text("partial backup", encoding="utf-8")
            raise RuntimeError("fixture backup failed")
        return original_copy2(source, destination)

    monkeypatch.setattr(strict_fixtures.shutil, "copy2", fail_fixture_backup_copy)

    with pytest.raises(RuntimeError, match="fixture backup failed"):
        generate_strict_fixture(project_root=tmp_path, season=2026, round_number=12, source="cartola_api", force=True)

    assert sha256_file(initial.fixture_path) == initial_fixture_hash
    assert sha256_file(initial.manifest_path) == initial_manifest_hash
    validate_strict_manifest(
        project_root=tmp_path,
        fixture_path=initial.fixture_path,
        season=2026,
        round_number=12,
        source="cartola_api",
    )


def test_generate_strict_fixtures_cli_parses_round_and_force() -> None:
    module = _load_generate_script()

    args = module.parse_args(["--season", "2026", "--round", "12", "--source", "cartola_api", "--force"])

    assert args.season == 2026
    assert args.round_number == 12
    assert args.source == "cartola_api"
    assert args.force is True


def test_load_strict_fixtures_validates_required_rounds(tmp_path: Path) -> None:
    _write_snapshot(tmp_path, captured_at=datetime(2026, 6, 1, 18, 0, tzinfo=UTC))
    generate_strict_fixture(project_root=tmp_path, season=2026, round_number=12, source="cartola_api")

    result = load_strict_fixtures(season=2026, project_root=tmp_path, required_rounds=[12], source="cartola_api")

    assert result.fixtures["rodada"].tolist() == [12]
    assert result.manifest_paths == ["data/01_raw/fixtures_strict/2026/partidas-12.manifest.json"]
    assert set(result.manifest_sha256) == set(result.manifest_paths)
    assert result.manifest_sha256[result.manifest_paths[0]]
    assert result.generator_versions == ["fixture_snapshot_v1"]


def test_load_strict_fixtures_rejects_missing_required_round(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="partidas-12"):
        load_strict_fixtures(season=2026, project_root=tmp_path, required_rounds=[12], source="cartola_api")


def test_load_strict_fixtures_rejects_tampered_fixture(tmp_path: Path) -> None:
    _write_snapshot(tmp_path, captured_at=datetime(2026, 6, 1, 18, 0, tzinfo=UTC))
    generated = generate_strict_fixture(project_root=tmp_path, season=2026, round_number=12, source="cartola_api")
    generated.fixture_path.write_text("rodada,id_clube_home,id_clube_away,data\n12,999,277,2026-06-01\n", encoding="utf-8")

    with pytest.raises(ValueError, match="canonical_fixture_sha256"):
        load_strict_fixtures(season=2026, project_root=tmp_path, required_rounds=[12], source="cartola_api")


def test_load_strict_fixtures_returns_empty_result_for_no_required_rounds(tmp_path: Path) -> None:
    result = load_strict_fixtures(season=2026, project_root=tmp_path, required_rounds=[], source="cartola_api")

    assert result.fixtures.empty
    assert result.fixtures.columns.tolist() == ["rodada", "id_clube_home", "id_clube_away", "data"]
    assert result.manifest_paths == []
    assert result.manifest_sha256 == {}
    assert result.generator_versions == []


def _write_valid_fixture(tmp_path: Path) -> Path:
    fixture_path = tmp_path / "data" / "01_raw" / "fixtures_strict" / "2026" / "partidas-12.csv"
    fixture_path.parent.mkdir(parents=True)
    fixture_path.write_text("rodada,id_clube_home,id_clube_away,data\n12,262,277,2026-06-01\n", encoding="utf-8")
    return fixture_path


def _write_valid_fixture_and_manifest(tmp_path: Path) -> tuple[Path, Path]:
    fixture_path = _write_valid_fixture(tmp_path)
    snapshot_dir = tmp_path / "data" / "01_raw" / "fixtures_snapshots" / "2026" / "rodada-12" / (
        "captured_at=2026-06-01T18-00-00Z"
    )
    snapshot_dir.mkdir(parents=True)
    capture_path = snapshot_dir / "capture.json"
    fixture_snapshot_path = snapshot_dir / "fixtures.json"
    deadline_path = snapshot_dir / "deadline.json"
    _write_json(
        capture_path,
        {
            "capture_started_at_utc": "2026-06-01T17:59:58Z",
            "captured_at_utc": "2026-06-01T18:00:00Z",
            "fixture_http_date_header": "Mon, 01 Jun 2026 18:00:00 GMT",
            "fixture_http_date_utc": "2026-06-01T18:00:00Z",
            "fixture_http_status": 200,
            "fixture_final_url": "https://api.cartola.globo.com/partidas/12",
            "deadline_http_date_header": "Mon, 01 Jun 2026 18:00:00 GMT",
            "deadline_http_date_utc": "2026-06-01T18:00:00Z",
            "deadline_http_status": 200,
            "deadline_final_url": "https://api.cartola.globo.com/mercado/status",
            "clock_skew_tolerance_seconds": 300,
            "max_observed_clock_skew_seconds": 0.0,
            "source": "cartola_api",
            "season": 2026,
            "rodada": 12,
            "capture_version": "fixture_capture_v1",
        },
    )
    _write_json(fixture_snapshot_path, FROZEN_FIXTURE_PAYLOAD)
    _write_json(deadline_path, FROZEN_DEADLINE_PAYLOAD)
    manifest_path = fixture_path.with_suffix(".manifest.json")
    _write_json(
        manifest_path,
        {
            "mode": "strict",
            "season": 2026,
            "rodada": 12,
            "source": "cartola_api",
            "capture_metadata_path": _relative(tmp_path, capture_path),
            "capture_metadata_sha256": sha256_file(capture_path),
            "fixture_snapshot_path": _relative(tmp_path, fixture_snapshot_path),
            "fixture_snapshot_sha256": sha256_file(fixture_snapshot_path),
            "deadline_snapshot_path": _relative(tmp_path, deadline_path),
            "deadline_snapshot_sha256": sha256_file(deadline_path),
            "captured_at_utc": "2026-06-01T18:00:00Z",
            "deadline_at_utc": "2026-06-01T18:59:00Z",
            "deadline_source": "cartola_api_market_status",
            "fixture_endpoint": "https://api.cartola.globo.com/partidas/12",
            "fixture_final_url": "https://api.cartola.globo.com/partidas/12",
            "deadline_endpoint": "https://api.cartola.globo.com/mercado/status",
            "deadline_final_url": "https://api.cartola.globo.com/mercado/status",
            "generator_version": "fixture_snapshot_v1",
            "club_mapping_path": None,
            "club_mapping_sha256": None,
            "club_id_allowlist_path": None,
            "club_id_allowlist_sha256": None,
            "canonical_fixture_path": _relative(tmp_path, fixture_path),
            "canonical_fixture_sha256": sha256_file(fixture_path),
        },
    )
    return fixture_path, manifest_path


def _write_snapshot(
    tmp_path: Path,
    *,
    captured_at: datetime,
    home_id: int = 262,
    deadline_at: datetime = datetime(2026, 6, 1, 18, 59, tzinfo=UTC),
) -> Path:
    captured_label = captured_at.strftime("%Y-%m-%dT%H-%M-%SZ")
    snapshot_dir = (
        tmp_path
        / "data"
        / "01_raw"
        / "fixtures_snapshots"
        / "2026"
        / "rodada-12"
        / f"captured_at={captured_label}"
    )
    snapshot_dir.mkdir(parents=True)
    fixture_payload: dict[str, Any] = {
        **FROZEN_FIXTURE_PAYLOAD,
        "partidas": [
            {
                **FROZEN_FIXTURE_PAYLOAD["partidas"][0],
                "clube_casa_id": home_id,
            }
        ],
    }
    deadline_payload: dict[str, Any] = {
        **FROZEN_DEADLINE_PAYLOAD,
        "fechamento": {
            **FROZEN_DEADLINE_PAYLOAD["fechamento"],
            "timestamp": int(deadline_at.timestamp()),
        },
    }
    http_date_header = captured_at.strftime("%a, %d %b %Y %H:%M:%S GMT")
    http_date_utc = captured_at.isoformat().replace("+00:00", "Z")
    _write_json(snapshot_dir / "fixtures.json", fixture_payload)
    _write_json(snapshot_dir / "deadline.json", deadline_payload)
    _write_json(
        snapshot_dir / "capture.json",
        {
            "capture_started_at_utc": http_date_utc,
            "captured_at_utc": captured_at.isoformat().replace("+00:00", "Z"),
            "fixture_http_date_header": http_date_header,
            "fixture_http_date_utc": http_date_utc,
            "fixture_http_status": 200,
            "fixture_final_url": "https://api.cartola.globo.com/partidas/12",
            "deadline_http_date_header": http_date_header,
            "deadline_http_date_utc": http_date_utc,
            "deadline_http_status": 200,
            "deadline_final_url": "https://api.cartola.globo.com/mercado/status",
            "clock_skew_tolerance_seconds": 300,
            "max_observed_clock_skew_seconds": 0.0,
            "source": "cartola_api",
            "season": 2026,
            "rodada": 12,
            "capture_version": "fixture_capture_v1",
        },
    )
    return snapshot_dir


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _relative(root: Path, path: Path) -> str:
    return str(path.relative_to(root))


def _load_generate_script() -> Any:
    script_path = Path(__file__).parents[3] / "scripts" / "generate_strict_fixtures.py"
    spec = importlib.util.spec_from_file_location("generate_strict_fixtures", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
