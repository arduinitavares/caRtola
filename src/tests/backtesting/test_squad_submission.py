from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Callable, NoReturn, cast

import pytest

from cartola.backtesting.squad_submission import (
    CARTOLA_MARKET_ENDPOINT,
    CARTOLA_SCHEMES_ENDPOINT,
    CARTOLA_STATUS_ENDPOINT,
    CONTRACT_UNVERIFIED,
    ContractUnverifiedError,
    JsonValue,
    RecommendationArtifact,
    SquadSubmissionError,
    SubmissionConfig,
    canonical_payload_sha256,
    fetch_public_json,
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


def _fetch_public_for_artifact(artifact: RecommendationArtifact) -> Callable[[str, float], JsonValue]:
    payloads: dict[str, JsonValue] = {
        CARTOLA_STATUS_ENDPOINT: cast("JsonValue", _status_payload(deadline=4_102_444_800)),
        CARTOLA_SCHEMES_ENDPOINT: cast("JsonValue", _schemes_payload()),
        CARTOLA_MARKET_ENDPOINT: cast("JsonValue", _market_payload_from_artifact(artifact)),
    }

    def fetch(url: str, timeout_seconds: float) -> JsonValue:
        assert timeout_seconds > 0
        try:
            return payloads[url]
        except KeyError as exc:
            raise AssertionError(f"unexpected public endpoint: {url}") from exc

    return fetch


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


def test_fetch_public_json_disables_ambient_requests_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    import requests  # type: ignore[import-untyped]

    observed: dict[str, object] = {}

    class FakeResponse:
        status_code = 200

        def json(self) -> dict[str, object]:
            return {"ok": True}

    class FakeSession:
        trust_env = True

        def get(self, url: str, timeout: float) -> FakeResponse:
            observed["url"] = url
            observed["timeout"] = timeout
            observed["trust_env_at_get"] = self.trust_env
            return FakeResponse()

    def top_level_get(url: str, timeout: float) -> NoReturn:
        raise AssertionError("fetch_public_json must use an explicit requests.Session")

    monkeypatch.setattr(requests, "Session", FakeSession)
    monkeypatch.setattr(requests, "get", top_level_get)

    payload = fetch_public_json(CARTOLA_STATUS_ENDPOINT, 12.5)

    assert payload == {"ok": True}
    assert observed == {
        "url": CARTOLA_STATUS_ENDPOINT,
        "timeout": 12.5,
        "trust_env_at_get": False,
    }


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


def test_validate_artifact_against_public_market_accepts_valid_zero_position_formation(tmp_path: Path) -> None:
    run_dir = _write_canonical_live_recommendation_run(tmp_path)
    artifact = load_recommendation_artifact(project_root=tmp_path, recommendation_path=run_dir)
    artifact.summary["formation"] = "3-4-3"
    artifact.metadata["formation"] = "3-4-3"
    lat_index = artifact.selected.index[artifact.selected["posicao"] == "lat"].tolist()
    artifact.selected.loc[lat_index[0], "posicao"] = "zag"
    artifact.selected.loc[lat_index[1], "posicao"] = "mei"

    report = validate_artifact_against_public_market(
        artifact,
        _status_payload(deadline=4_102_444_800),
        [
            {
                "nome": "3-4-3",
                "esquema_id": 5,
                "posicoes": {"gol": 1, "lat": 0, "zag": 3, "mei": 4, "ata": 3, "tec": 1},
            },
        ],
        _market_payload_from_artifact(artifact),
        now=datetime(2026, 5, 16, 12, 0, tzinfo=UTC),
        safety_margin_seconds=120,
    )

    assert report["formation_scheme_id"] == 5
    assert report["selected_position_counts"] == {"ata": 3, "gol": 1, "lat": 0, "mei": 4, "tec": 1, "zag": 3}


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


def test_validate_artifact_against_public_market_rejects_current_market_status_drift(tmp_path: Path) -> None:
    run_dir = _write_canonical_live_recommendation_run(tmp_path)
    artifact = load_recommendation_artifact(project_root=tmp_path, recommendation_path=run_dir)
    market_payload = _market_payload_from_artifact(artifact)
    athlete_rows = market_payload["atletas"]
    assert isinstance(athlete_rows, list)
    typed_athlete_rows = cast("list[dict[str, object]]", athlete_rows)
    first_athlete = typed_athlete_rows[0]
    first_athlete["status_id"] = 2
    first_athlete["status"] = {"id": 2, "nome": "Dúvida"}

    with pytest.raises(SquadSubmissionError, match="status drift"):
        validate_artifact_against_public_market(
            artifact,
            _status_payload(deadline=4_102_444_800),
            _schemes_payload(),
            market_payload,
            now=datetime(2026, 5, 16, 12, 0, tzinfo=UTC),
            safety_margin_seconds=120,
        )


def test_validate_artifact_against_public_market_rejects_authoritative_status_id_drift(tmp_path: Path) -> None:
    run_dir = _write_canonical_live_recommendation_run(tmp_path)
    artifact = load_recommendation_artifact(project_root=tmp_path, recommendation_path=run_dir)
    market_payload = _market_payload_from_artifact(artifact)
    athlete_rows = market_payload["atletas"]
    assert isinstance(athlete_rows, list)
    typed_athlete_rows = cast("list[dict[str, object]]", athlete_rows)
    first_athlete = typed_athlete_rows[0]
    first_athlete["status_id"] = 2
    first_athlete["status"] = {"id": 7, "nome": "Provável"}

    with pytest.raises(SquadSubmissionError, match="status drift"):
        validate_artifact_against_public_market(
            artifact,
            _status_payload(deadline=4_102_444_800),
            _schemes_payload(),
            market_payload,
            now=datetime(2026, 5, 16, 12, 0, tzinfo=UTC),
            safety_margin_seconds=120,
        )


def test_validate_artifact_against_public_market_rejects_duplicate_selected_athletes(tmp_path: Path) -> None:
    run_dir = _write_canonical_live_recommendation_run(tmp_path)
    artifact = load_recommendation_artifact(project_root=tmp_path, recommendation_path=run_dir)
    market_payload = _market_payload_from_artifact(artifact)
    artifact.selected.loc[artifact.selected.index[1], "id_atleta"] = artifact.selected.loc[
        artifact.selected.index[0],
        "id_atleta",
    ]

    with pytest.raises(SquadSubmissionError, match="Duplicate selected athlete"):
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


def test_run_submission_writes_plan_and_result_under_unique_attempt(tmp_path: Path) -> None:
    run_dir = _write_canonical_live_recommendation_run(tmp_path)
    artifact = load_recommendation_artifact(project_root=tmp_path, recommendation_path=run_dir)

    result = run_submission(
        SubmissionConfig(project_root=tmp_path, recommendation_path=run_dir),
        fetch=_fetch_public_for_artifact(artifact),
        clock=lambda: datetime(2026, 5, 16, 13, 0, 42, tzinfo=UTC),
    )

    assert result.status == "plan_only"
    assert result.attempt_directory is not None
    assert result.submission_plan_path is not None
    assert result.submission_result_path is not None
    assert result.submission_plan_path.exists()
    assert result.submission_result_path.exists()
    assert result.submission_plan_path.parent == result.attempt_directory
    assert result.submission_result_path.parent == result.attempt_directory
    assert result.attempt_directory.parent == run_dir / "submission_attempts"
    assert result.attempt_directory.name.startswith("attempt_started_at=")

    plan = json.loads(result.submission_plan_path.read_text(encoding="utf-8"))
    audit = json.loads(result.submission_result_path.read_text(encoding="utf-8"))

    assert plan["plan_status"] == "ready_for_review"
    assert plan["payload"]["esquema"] == 3
    assert len(plan["payload"]["atletas"]) == 12
    assert plan["payload"]["capitao"] == 6
    assert plan["payload_sha256"] == result.payload_sha256
    assert plan["validation_report"]["account_budget_verified"] is False
    assert audit["submission_status"] == "plan_only"
    assert audit["would_submit"] is False
    assert audit["auth_token_present"] is False
    assert audit["auth_token_source"] == "not_required"


def test_run_submission_rejects_non_approved_model_without_override(tmp_path: Path) -> None:
    run_dir = _write_canonical_live_recommendation_run(tmp_path)
    metadata_path = run_dir / "run_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["model_id"] = "ridge"
    _write_json(metadata_path, metadata)
    artifact = load_recommendation_artifact(project_root=tmp_path, recommendation_path=run_dir)

    with pytest.raises(SquadSubmissionError, match="non-approved model"):
        run_submission(
            SubmissionConfig(project_root=tmp_path, recommendation_path=run_dir),
            fetch=_fetch_public_for_artifact(artifact),
        )


def test_run_submission_allows_non_approved_model_for_plan_with_override_reason(tmp_path: Path) -> None:
    run_dir = _write_canonical_live_recommendation_run(tmp_path)
    metadata_path = run_dir / "run_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["model_id"] = "ridge"
    _write_json(metadata_path, metadata)
    artifact = load_recommendation_artifact(project_root=tmp_path, recommendation_path=run_dir)

    result = run_submission(
        SubmissionConfig(
            project_root=tmp_path,
            recommendation_path=run_dir,
            allow_non_approved_model=True,
            override_reason="manual comparison plan",
        ),
        fetch=_fetch_public_for_artifact(artifact),
    )

    assert result.status == "plan_only"


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
