from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import NoReturn, cast

import pytest

from cartola.backtesting.squad_submission import (
    CONTRACT_UNVERIFIED,
    ContractUnverifiedError,
    RecommendationArtifact,
    SquadSubmissionError,
    SubmissionConfig,
    canonical_payload_sha256,
    load_recommendation_artifact,
    parse_schemes,
    run_submission,
    validate_artifact_against_public_market,
)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_canonical_live_recommendation_run(project_root: Path) -> Path:
    run_dir = (
        project_root
        / "data/08_reporting/recommendations/2026/round-16/live/runs/"
        "run_started_at=20260516T130042922935Z"
    )
    run_dir.mkdir(parents=True)
    rows = [
        (1, "Gol 1", "gol", 101, 5.0, False),
        (2, "Lat 1", "lat", 102, 6.0, False),
        (3, "Lat 2", "lat", 103, 7.0, False),
        (4, "Zag 1", "zag", 104, 8.0, False),
        (5, "Zag 2", "zag", 105, 9.0, False),
        (6, "Mei 1", "mei", 106, 10.0, True),
        (7, "Mei 2", "mei", 107, 11.0, False),
        (8, "Mei 3", "mei", 108, 12.0, False),
        (9, "Ata 1", "ata", 109, 13.0, False),
        (10, "Ata 2", "ata", 110, 14.0, False),
        (11, "Ata 3", "ata", 111, 15.0, False),
        (12, "Tec 1", "tec", 112, 16.0, False),
    ]
    csv_lines = ["id_atleta,apelido,posicao,id_clube,preco_pre_rodada,status,is_captain"]
    csv_lines.extend(
        f"{athlete_id},{name},{position},{club_id},{price},Provavel,{is_captain}"
        for athlete_id, name, position, club_id, price, is_captain in rows
    )
    (run_dir / "recommended_squad.csv").write_text("\n".join(csv_lines) + "\n", encoding="utf-8")
    _write_json(
        run_dir / "recommendation_summary.json",
        {
            "season": 2026,
            "target_round": 16,
            "mode": "live",
            "formation": "4-3-3",
            "budget": 100.0,
            "budget_used": 60.0,
            "selected_count": 12,
            "captain_id": 6,
            "captain_name": "Mei 1",
            "strategy": "xgboost_depth2_l2_heavy",
            "scoring_contract_version": "cartola_standard_2026_v1",
        },
    )
    _write_json(
        run_dir / "run_metadata.json",
        {
            "season": 2026,
            "target_round": 16,
            "mode": "live",
            "model_id": "xgboost_depth2_l2_heavy",
            "footystats_mode": "ppg_xg",
            "fixture_mode": "none",
            "matchup_context_mode": "none",
            "scoring_contract_version": "cartola_standard_2026_v1",
            "playable_statuses": ["Provavel"],
            "formation": "4-3-3",
            "budget": 100.0,
        },
    )
    _write_json(run_dir / "live_workflow_metadata.json", {"status": "ok"})
    return run_dir


def _status_payload(
    round_number: int = 16,
    status_mercado: int = 1,
    deadline: int = 1_778_966_940,
) -> dict[str, object]:
    return {
        "temporada": 2026,
        "rodada_atual": round_number,
        "status_mercado": status_mercado,
        "game_over": False,
        "fechamento": {"timestamp": deadline},
    }


def _schemes_payload() -> list[dict[str, object]]:
    return [
        {
            "nome": "4-3-3",
            "esquema_id": 3,
            "posicoes": {"gol": 1, "lat": 2, "zag": 2, "mei": 3, "ata": 3, "tec": 1},
        },
        {
            "nome": "4-4-2",
            "esquema_id": 4,
            "posicoes": {"gol": 1, "lat": 2, "zag": 2, "mei": 4, "ata": 2, "tec": 1},
        },
    ]


def _market_payload_from_artifact(artifact: RecommendationArtifact) -> dict[str, object]:
    position_code_to_id = {"gol": 1, "lat": 2, "zag": 3, "mei": 4, "ata": 5, "tec": 6}
    athlete_rows: list[dict[str, object]] = []
    for row in artifact.selected.to_dict("records"):
        position_code = str(row["posicao"])
        athlete_rows.append(
            {
                "atleta_id": row["id_atleta"],
                "apelido": row["apelido"],
                "clube_id": row["id_clube"],
                "posicao_id": position_code_to_id[position_code],
                "status_id": 7,
                "status": {"id": 7, "nome": "Provável"},
                "preco_num": row["preco_pre_rodada"],
                "rodada_id": 15,
            },
        )
    return {
        "posicoes": {
            "1": {"id": 1, "abreviacao": "gol"},
            "2": {"id": 2, "abreviacao": "lat"},
            "3": {"id": 3, "abreviacao": "zag"},
            "4": {"id": 4, "abreviacao": "mei"},
            "5": {"id": 5, "abreviacao": "ata"},
            "6": {"id": 6, "abreviacao": "tec"},
        },
        "atletas": athlete_rows,
    }


def test_canonical_payload_sha256_is_stable_and_preserves_athlete_order() -> None:
    payload_a = {"capitao": 3, "atletas": [3, 1, 2], "esquema": 4}
    payload_b = {"esquema": 4, "atletas": [3, 1, 2], "capitao": 3}
    payload_c = {"esquema": 4, "atletas": [1, 2, 3], "capitao": 3}

    assert canonical_payload_sha256(payload_a) == canonical_payload_sha256(payload_b)
    assert canonical_payload_sha256(payload_a) != canonical_payload_sha256(payload_c)


def test_canonical_payload_sha256_normalizes_submission_ids_to_integers() -> None:
    string_payload = {"esquema": "4", "atletas": ["3", 1, "2"], "capitao": "3"}
    integer_payload = {"esquema": 4, "atletas": [3, 1, 2], "capitao": 3}

    assert canonical_payload_sha256(string_payload) == canonical_payload_sha256(integer_payload)


def test_parse_schemes_extracts_formation_id_and_counts() -> None:
    schemes = parse_schemes(_schemes_payload())

    assert schemes["4-3-3"].scheme_id == 3
    assert schemes["4-3-3"].position_counts == {"gol": 1, "lat": 2, "zag": 2, "mei": 3, "ata": 3, "tec": 1}
    assert schemes["4-4-2"].scheme_id == 4
    assert schemes["4-4-2"].position_counts == {"gol": 1, "lat": 2, "zag": 2, "mei": 4, "ata": 2, "tec": 1}


def test_validate_artifact_against_public_market_accepts_valid_current_market(tmp_path: Path) -> None:
    run_dir = _write_canonical_live_recommendation_run(tmp_path)
    artifact = load_recommendation_artifact(project_root=tmp_path, recommendation_path=run_dir)

    report = validate_artifact_against_public_market(
        artifact,
        _status_payload(deadline=4_102_444_800),
        _schemes_payload(),
        _market_payload_from_artifact(artifact),
        now=datetime(2026, 5, 16, 12, 0, tzinfo=UTC),
        safety_margin_seconds=120,
    )

    assert report["formation_scheme_id"] == 3
    assert report["market_round"] == 16
    assert report["account_budget_verified"] is False
    assert report["not_comparable_fields"] == []


def test_validate_artifact_against_public_market_rejects_closed_market(tmp_path: Path) -> None:
    run_dir = _write_canonical_live_recommendation_run(tmp_path)
    artifact = load_recommendation_artifact(project_root=tmp_path, recommendation_path=run_dir)

    with pytest.raises(SquadSubmissionError, match="market is not open"):
        validate_artifact_against_public_market(
            artifact,
            _status_payload(status_mercado=2, deadline=4_102_444_800),
            _schemes_payload(),
            _market_payload_from_artifact(artifact),
            now=datetime(2026, 5, 16, 12, 0, tzinfo=UTC),
            safety_margin_seconds=120,
        )


def test_validate_artifact_against_public_market_rejects_price_drift(tmp_path: Path) -> None:
    run_dir = _write_canonical_live_recommendation_run(tmp_path)
    artifact = load_recommendation_artifact(project_root=tmp_path, recommendation_path=run_dir)
    market_payload = _market_payload_from_artifact(artifact)
    athlete_rows = market_payload["atletas"]
    assert isinstance(athlete_rows, list)
    typed_athlete_rows = cast("list[dict[str, object]]", athlete_rows)
    first_athlete = typed_athlete_rows[0]
    first_athlete["preco_num"] = 99.0

    with pytest.raises(SquadSubmissionError, match="price drift"):
        validate_artifact_against_public_market(
            artifact,
            _status_payload(deadline=4_102_444_800),
            _schemes_payload(),
            market_payload,
            now=datetime(2026, 5, 16, 12, 0, tzinfo=UTC),
            safety_margin_seconds=120,
        )


def test_confirm_submit_fails_contract_unverified_before_fetch_or_auth(tmp_path: Path) -> None:
    calls: list[str] = []

    def fetch(url: str, timeout_seconds: float) -> NoReturn:
        calls.append(url)
        raise AssertionError("Phase 1 submit must not fetch anything")

    config = SubmissionConfig(
        project_root=tmp_path,
        submission_plan=tmp_path / "submission_plan.json",
        confirm_submit=True,
        confirm_payload_sha256="abc123",
    )

    with pytest.raises(ContractUnverifiedError, match=CONTRACT_UNVERIFIED):
        run_submission(config, fetch=fetch)

    assert calls == []


def test_load_recommendation_artifact_reads_canonical_live_run(tmp_path: Path) -> None:
    run_dir = _write_canonical_live_recommendation_run(tmp_path)

    artifact = load_recommendation_artifact(project_root=tmp_path, recommendation_path=run_dir)

    assert artifact.season == 2026
    assert artifact.target_round == 16
    assert artifact.selected.shape[0] == 12
    assert artifact.summary["formation"] == "4-3-3"
    assert artifact.source_artifact_hashes == {
        "recommended_squad.csv": _file_sha256(run_dir / "recommended_squad.csv"),
        "recommendation_summary.json": _file_sha256(run_dir / "recommendation_summary.json"),
        "run_metadata.json": _file_sha256(run_dir / "run_metadata.json"),
        "live_workflow_metadata.json": _file_sha256(run_dir / "live_workflow_metadata.json"),
    }


def test_load_recommendation_artifact_rejects_non_canonical_backtest_path(tmp_path: Path) -> None:
    backtest_path = tmp_path / "data/08_reporting/backtests/run"
    backtest_path.mkdir(parents=True)

    with pytest.raises(SquadSubmissionError, match="canonical live recommendation"):
        load_recommendation_artifact(project_root=tmp_path, recommendation_path=backtest_path)


def test_load_recommendation_artifact_rejects_artifact_symlink_outside_project(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    run_dir = _write_canonical_live_recommendation_run(project_root)
    outside_csv = tmp_path / "outside-recommended-squad.csv"
    outside_csv.write_text((run_dir / "recommended_squad.csv").read_text(encoding="utf-8"), encoding="utf-8")
    artifact_path = run_dir / "recommended_squad.csv"
    artifact_path.unlink()
    try:
        artifact_path.symlink_to(outside_csv)
    except (NotImplementedError, OSError) as exc:
        pytest.skip(msg=f"Symlink creation is unsupported: {exc}")

    with pytest.raises(SquadSubmissionError, match="inside project_root"):
        load_recommendation_artifact(project_root=project_root, recommendation_path=run_dir)
