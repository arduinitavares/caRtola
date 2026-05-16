from __future__ import annotations

import json
from pathlib import Path
from typing import NoReturn

import pytest

from cartola.backtesting.squad_submission import (
    CONTRACT_UNVERIFIED,
    ContractUnverifiedError,
    SquadSubmissionError,
    SubmissionConfig,
    canonical_payload_sha256,
    load_recommendation_artifact,
    run_submission,
)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_canonical_live_recommendation_run(project_root: Path) -> Path:
    run_dir = (
        project_root
        / "data/08_reporting/recommendations/2026/round-16/live/runs/"
        "run_started_at=20260516T130042922935Z"
    )
    run_dir.mkdir(parents=True)
    rows = [
        (1, "Gol 1", "gol", False),
        (2, "Lat 1", "lat", False),
        (3, "Lat 2", "lat", False),
        (4, "Zag 1", "zag", False),
        (5, "Zag 2", "zag", False),
        (6, "Mei 1", "mei", True),
        (7, "Mei 2", "mei", False),
        (8, "Mei 3", "mei", False),
        (9, "Ata 1", "ata", False),
        (10, "Ata 2", "ata", False),
        (11, "Ata 3", "ata", False),
        (12, "Tec 1", "tec", False),
    ]
    csv_lines = ["id_atleta,apelido,posicao,status,is_captain"]
    csv_lines.extend(f"{athlete_id},{name},{position},Provavel,{is_captain}" for athlete_id, name, position, is_captain in rows)
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
    assert artifact.source_artifact_hashes.keys() == {
        "recommended_squad.csv",
        "recommendation_summary.json",
        "run_metadata.json",
        "live_workflow_metadata.json",
    }


def test_load_recommendation_artifact_rejects_non_canonical_backtest_path(tmp_path: Path) -> None:
    backtest_path = tmp_path / "data/08_reporting/backtests/run"
    backtest_path.mkdir(parents=True)

    with pytest.raises(SquadSubmissionError, match="canonical live recommendation"):
        load_recommendation_artifact(project_root=tmp_path, recommendation_path=backtest_path)
