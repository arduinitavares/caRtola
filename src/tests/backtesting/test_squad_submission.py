from __future__ import annotations

from pathlib import Path
from typing import NoReturn

import pytest

from cartola.backtesting.squad_submission import (
    CONTRACT_UNVERIFIED,
    ContractUnverifiedError,
    SubmissionConfig,
    canonical_payload_sha256,
    run_submission,
)


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
