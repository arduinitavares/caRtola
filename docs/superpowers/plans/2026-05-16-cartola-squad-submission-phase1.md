# Cartola Squad Submission Phase 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a safe Phase 1 artifact-to-submission-plan command for Cartola recommendations, with real authenticated submission hard-disabled by `CONTRACT_UNVERIFIED`.

**Architecture:** Add a focused `cartola.backtesting.squad_submission` module that loads one live recommendation artifact, validates it against public Cartola market data, builds a provisional payload, hashes it canonically, and writes a timestamped `submission_plan.json` plus `submission_result.json`. Add `scripts/submit_recommended_squad.py` as a thin CLI wrapper. Phase 1 must not load `.env`, read `CARTOLA_GLB_TOKEN`, construct an authenticated HTTP client, or construct a POST request.

**Tech Stack:** Python 3.13, `uv`, pandas, stdlib `json`/`hashlib`/`datetime`, existing `requests` pattern for public GETs, pytest, Rich for CLI output.

---

## File Structure

- Create `src/cartola/backtesting/squad_submission.py`
  - Owns Phase 1 dataclasses, public API GETs, artifact loading, validation, current-market drift checks, canonical payload hashing, attempt directory creation, and audit writing.
  - Exposes `run_submission(config, fetch=fetch_public_json, clock=utc_now) -> SquadSubmissionResult`.
  - Raises `ContractUnverifiedError("CONTRACT_UNVERIFIED")` before auth or POST work for any `confirm_submit=True`.

- Create `scripts/submit_recommended_squad.py`
  - Parses CLI arguments.
  - Calls `run_submission`.
  - Prints a compact Rich summary.
  - Returns `1` for expected validation errors and `CONTRACT_UNVERIFIED`.

- Create `src/tests/backtesting/test_squad_submission.py`
  - Unit and integration-style mocked tests for payload hashing, artifact validation, public API parsing, market drift checks, attempt writing, and `CONTRACT_UNVERIFIED`.

- Create `src/tests/backtesting/test_submit_recommended_squad_cli.py`
  - CLI parsing and main behavior tests, including the Phase 1 guarantee that `.env` is not loaded and no POST path exists.

- Modify `AGENTS.md`
  - Add the Phase 1 plan-generation command and state that `--confirm-submit` is hard-disabled.

- Modify `roadmap.md`
  - Add the milestone result and next Phase 2 blocker: verified save/read-back contract.

---

### Task 1: Core Types, Canonical Hashing, And Contract Gate

**Files:**
- Create: `src/cartola/backtesting/squad_submission.py`
- Create: `src/tests/backtesting/test_squad_submission.py`

- [ ] **Step 1: Write failing tests for canonical payload hashing and the Phase 1 contract gate**

Add this to `src/tests/backtesting/test_squad_submission.py`:

```python
from __future__ import annotations

from pathlib import Path

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


def test_confirm_submit_fails_contract_unverified_before_fetch_or_auth(tmp_path: Path) -> None:
    calls: list[str] = []

    def fetch(url: str, timeout_seconds: float) -> object:
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
```

- [ ] **Step 2: Run tests and verify they fail because the module does not exist**

Run:

```bash
uv run --frozen pytest src/tests/backtesting/test_squad_submission.py::test_canonical_payload_sha256_is_stable_and_preserves_athlete_order src/tests/backtesting/test_squad_submission.py::test_confirm_submit_fails_contract_unverified_before_fetch_or_auth -q
```

Expected: FAIL with `ModuleNotFoundError: No module named 'cartola.backtesting.squad_submission'`.

- [ ] **Step 3: Implement the minimal core module**

Create `src/cartola/backtesting/squad_submission.py`:

```python
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Mapping

CONTRACT_UNVERIFIED = "CONTRACT_UNVERIFIED"
CARTOLA_STATUS_ENDPOINT = "https://api.cartola.globo.com/mercado/status"
CARTOLA_MARKET_ENDPOINT = "https://api.cartola.globo.com/atletas/mercado"
CARTOLA_SCHEMES_ENDPOINT = "https://api.cartola.globo.com/esquemas"

APPROVED_PROFILE: dict[str, str] = {
    "model_id": "xgboost_depth2_l2_heavy",
    "footystats_mode": "ppg_xg",
    "fixture_mode": "none",
    "matchup_context_mode": "none",
    "scoring_contract_version": "cartola_standard_2026_v1",
}

JsonValue = dict[str, Any] | list[Any]
Fetch = Callable[[str, float], JsonValue]
Clock = Callable[[], datetime]


@dataclass(frozen=True)
class SubmissionConfig:
    project_root: Path = Path(".")
    recommendation_path: Path | None = None
    submission_plan: Path | None = None
    timeout_seconds: float = 30.0
    confirm_submit: bool = False
    confirm_payload_sha256: str | None = None
    allow_non_approved_model: bool = False
    override_reason: str | None = None
    safety_margin_seconds: int = 120


@dataclass(frozen=True)
class SquadSubmissionResult:
    attempt_directory: Path | None
    submission_plan_path: Path | None
    submission_result_path: Path | None
    payload_sha256: str | None
    status: str


class SquadSubmissionError(ValueError):
    """Expected user-actionable submission planning failure."""


class ContractUnverifiedError(SquadSubmissionError):
    """Raised for Phase 1 submit attempts before auth or POST setup."""


def utc_now() -> datetime:
    return datetime.now(UTC)


def canonical_payload_bytes(payload: Mapping[str, object]) -> bytes:
    normalized = {
        "atletas": [int(value) for value in payload["atletas"]],  # type: ignore[index]
        "capitao": int(payload["capitao"]),
        "esquema": int(payload["esquema"]),
    }
    return json.dumps(normalized, sort_keys=True, separators=(",", ":")).encode("utf-8")


def canonical_payload_sha256(payload: Mapping[str, object]) -> str:
    return hashlib.sha256(canonical_payload_bytes(payload)).hexdigest()


def fetch_public_json(url: str, timeout_seconds: float) -> JsonValue:
    import requests  # type: ignore[import-untyped]

    response = requests.get(url, timeout=timeout_seconds)
    if response.status_code != 200:
        raise SquadSubmissionError(f"Cartola public request failed: url={url} status={response.status_code}")
    try:
        payload = response.json()
    except ValueError as exc:
        raise SquadSubmissionError(f"Cartola public response is not valid JSON: url={url}") from exc
    if not isinstance(payload, (dict, list)):
        raise SquadSubmissionError(f"Cartola public JSON payload must be an object or array: url={url}")
    return payload


def run_submission(
    config: SubmissionConfig,
    *,
    fetch: Fetch = fetch_public_json,
    clock: Clock = utc_now,
) -> SquadSubmissionResult:
    if config.confirm_submit:
        raise ContractUnverifiedError(CONTRACT_UNVERIFIED)
    raise SquadSubmissionError("recommendation_path is required for Phase 1 plan generation")
```

- [ ] **Step 4: Run tests and verify they pass**

Run:

```bash
uv run --frozen pytest src/tests/backtesting/test_squad_submission.py::test_canonical_payload_sha256_is_stable_and_preserves_athlete_order src/tests/backtesting/test_squad_submission.py::test_confirm_submit_fails_contract_unverified_before_fetch_or_auth -q
```

Expected: `2 passed`.

- [ ] **Step 5: Commit**

```bash
git add src/cartola/backtesting/squad_submission.py src/tests/backtesting/test_squad_submission.py
git commit -m "feat: add squad submission phase one core"
```

---

### Task 2: Artifact Loading, Path Validation, And Source Hashes

**Files:**
- Modify: `src/cartola/backtesting/squad_submission.py`
- Modify: `src/tests/backtesting/test_squad_submission.py`

- [ ] **Step 1: Add fixture helpers and failing artifact-loading tests**

Append this to `src/tests/backtesting/test_squad_submission.py`:

```python
import json

import pandas as pd

from cartola.backtesting.squad_submission import load_recommendation_artifact


def _valid_run_dir(tmp_path: Path) -> Path:
    run_dir = (
        tmp_path
        / "data"
        / "08_reporting"
        / "recommendations"
        / "2026"
        / "round-16"
        / "live"
        / "runs"
        / "run_started_at=20260516T130042922935Z"
    )
    run_dir.mkdir(parents=True)
    rows = [
        {"rodada": 16, "id_atleta": 1, "apelido": "Goalie", "id_clube": 10, "nome_clube": "AAA", "posicao": "gol", "status": "Provavel", "preco_pre_rodada": 5.0, "predicted_points": 4.0, "is_captain": False},
        {"rodada": 16, "id_atleta": 2, "apelido": "Lat 1", "id_clube": 10, "nome_clube": "AAA", "posicao": "lat", "status": "Provavel", "preco_pre_rodada": 5.0, "predicted_points": 4.0, "is_captain": False},
        {"rodada": 16, "id_atleta": 3, "apelido": "Lat 2", "id_clube": 11, "nome_clube": "BBB", "posicao": "lat", "status": "Provavel", "preco_pre_rodada": 5.0, "predicted_points": 4.0, "is_captain": False},
        {"rodada": 16, "id_atleta": 4, "apelido": "Zag 1", "id_clube": 12, "nome_clube": "CCC", "posicao": "zag", "status": "Provavel", "preco_pre_rodada": 5.0, "predicted_points": 4.0, "is_captain": False},
        {"rodada": 16, "id_atleta": 5, "apelido": "Zag 2", "id_clube": 13, "nome_clube": "DDD", "posicao": "zag", "status": "Provavel", "preco_pre_rodada": 5.0, "predicted_points": 4.0, "is_captain": False},
        {"rodada": 16, "id_atleta": 6, "apelido": "Mei 1", "id_clube": 14, "nome_clube": "EEE", "posicao": "mei", "status": "Provavel", "preco_pre_rodada": 5.0, "predicted_points": 4.0, "is_captain": True},
        {"rodada": 16, "id_atleta": 7, "apelido": "Mei 2", "id_clube": 15, "nome_clube": "FFF", "posicao": "mei", "status": "Provavel", "preco_pre_rodada": 5.0, "predicted_points": 4.0, "is_captain": False},
        {"rodada": 16, "id_atleta": 8, "apelido": "Mei 3", "id_clube": 16, "nome_clube": "GGG", "posicao": "mei", "status": "Provavel", "preco_pre_rodada": 5.0, "predicted_points": 4.0, "is_captain": False},
        {"rodada": 16, "id_atleta": 9, "apelido": "Ata 1", "id_clube": 17, "nome_clube": "HHH", "posicao": "ata", "status": "Provavel", "preco_pre_rodada": 5.0, "predicted_points": 4.0, "is_captain": False},
        {"rodada": 16, "id_atleta": 10, "apelido": "Ata 2", "id_clube": 18, "nome_clube": "III", "posicao": "ata", "status": "Provavel", "preco_pre_rodada": 5.0, "predicted_points": 4.0, "is_captain": False},
        {"rodada": 16, "id_atleta": 11, "apelido": "Ata 3", "id_clube": 19, "nome_clube": "JJJ", "posicao": "ata", "status": "Provavel", "preco_pre_rodada": 5.0, "predicted_points": 4.0, "is_captain": False},
        {"rodada": 16, "id_atleta": 12, "apelido": "Coach", "id_clube": 20, "nome_clube": "KKK", "posicao": "tec", "status": "Provavel", "preco_pre_rodada": 5.0, "predicted_points": 4.0, "is_captain": False},
    ]
    pd.DataFrame(rows).to_csv(run_dir / "recommended_squad.csv", index=False)
    summary = {
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
    }
    metadata = {
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
    }
    (run_dir / "recommendation_summary.json").write_text(json.dumps(summary), encoding="utf-8")
    (run_dir / "run_metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    (run_dir / "live_workflow_metadata.json").write_text(json.dumps({"status": "ok"}), encoding="utf-8")
    return run_dir


def test_load_recommendation_artifact_reads_live_files_and_hashes(tmp_path: Path) -> None:
    run_dir = _valid_run_dir(tmp_path)

    artifact = load_recommendation_artifact(project_root=tmp_path, recommendation_path=run_dir)

    assert artifact.season == 2026
    assert artifact.target_round == 16
    assert artifact.summary["formation"] == "4-3-3"
    assert artifact.selected.shape[0] == 12
    assert set(artifact.source_artifact_hashes) == {
        "recommended_squad.csv",
        "recommendation_summary.json",
        "run_metadata.json",
        "live_workflow_metadata.json",
    }


def test_load_recommendation_artifact_rejects_non_canonical_path(tmp_path: Path) -> None:
    bad_dir = tmp_path / "data" / "08_reporting" / "backtests" / "run"
    bad_dir.mkdir(parents=True)

    with pytest.raises(ValueError, match="canonical live recommendation"):
        load_recommendation_artifact(project_root=tmp_path, recommendation_path=bad_dir)
```

- [ ] **Step 2: Run tests and verify they fail on missing `load_recommendation_artifact`**

Run:

```bash
uv run --frozen pytest src/tests/backtesting/test_squad_submission.py::test_load_recommendation_artifact_reads_live_files_and_hashes src/tests/backtesting/test_squad_submission.py::test_load_recommendation_artifact_rejects_non_canonical_path -q
```

Expected: FAIL with `ImportError` or `AttributeError` for `load_recommendation_artifact`.

- [ ] **Step 3: Implement artifact dataclass, path validation, JSON/CSV loading, and file hashes**

Add this to `src/cartola/backtesting/squad_submission.py`:

```python
import pandas as pd


@dataclass(frozen=True)
class RecommendationArtifact:
    path: Path
    selected: pd.DataFrame
    summary: dict[str, Any]
    metadata: dict[str, Any]
    live_workflow_metadata: dict[str, Any] | None
    source_artifact_hashes: dict[str, str]

    @property
    def season(self) -> int:
        return int(self.summary["season"])

    @property
    def target_round(self) -> int:
        return int(self.summary["target_round"])


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SquadSubmissionError(f"Invalid JSON file: {path}") from exc
    if not isinstance(payload, dict):
        raise SquadSubmissionError(f"JSON file must contain an object: {path}")
    return payload


def _resolve_project_path(project_root: Path, value: Path) -> Path:
    root = project_root.resolve()
    path = value if value.is_absolute() else root / value
    resolved = path.resolve()
    if resolved != root and root not in resolved.parents:
        raise SquadSubmissionError(f"path must resolve inside project_root: {value}")
    return resolved


def _validate_recommendation_path(project_root: Path, recommendation_path: Path) -> Path:
    resolved = _resolve_project_path(project_root, recommendation_path)
    root = project_root.resolve()
    try:
        relative = resolved.relative_to(root)
    except ValueError as exc:
        raise SquadSubmissionError("recommendation path must be inside project_root") from exc
    parts = relative.parts
    forbidden = {"backtests", "experiments", "policy_simulations", "blend_diagnostics", "oracle_discovery", "ebm_diagnostics"}
    if forbidden.intersection(parts):
        raise SquadSubmissionError("recommendation path must use canonical live recommendation tree")
    if len(parts) < 8:
        raise SquadSubmissionError("recommendation path must use canonical live recommendation tree")
    expected_prefix = ("data", "08_reporting", "recommendations")
    if parts[:3] != expected_prefix:
        raise SquadSubmissionError("recommendation path must use canonical live recommendation tree")
    if parts[5] != "live" or parts[6] != "runs" or not parts[7].startswith("run_started_at="):
        raise SquadSubmissionError("recommendation path must use canonical live recommendation tree")
    return resolved


def load_recommendation_artifact(*, project_root: Path, recommendation_path: Path) -> RecommendationArtifact:
    run_dir = _validate_recommendation_path(project_root, recommendation_path)
    required = {
        "recommended_squad.csv": run_dir / "recommended_squad.csv",
        "recommendation_summary.json": run_dir / "recommendation_summary.json",
        "run_metadata.json": run_dir / "run_metadata.json",
    }
    missing = [name for name, path in required.items() if not path.exists()]
    if missing:
        raise SquadSubmissionError(f"missing required recommendation artifact files: {missing}")
    selected = pd.read_csv(required["recommended_squad.csv"])
    summary = _read_json_object(required["recommendation_summary.json"])
    metadata = _read_json_object(required["run_metadata.json"])
    workflow_path = run_dir / "live_workflow_metadata.json"
    workflow = _read_json_object(workflow_path) if workflow_path.exists() else None
    source_hashes = {name: _sha256_file(path) for name, path in required.items()}
    if workflow_path.exists():
        source_hashes["live_workflow_metadata.json"] = _sha256_file(workflow_path)
    return RecommendationArtifact(
        path=run_dir,
        selected=selected,
        summary=summary,
        metadata=metadata,
        live_workflow_metadata=workflow,
        source_artifact_hashes=source_hashes,
    )
```

- [ ] **Step 4: Run tests and verify Task 2 passes**

Run:

```bash
uv run --frozen pytest src/tests/backtesting/test_squad_submission.py::test_load_recommendation_artifact_reads_live_files_and_hashes src/tests/backtesting/test_squad_submission.py::test_load_recommendation_artifact_rejects_non_canonical_path -q
```

Expected: `2 passed`.

- [ ] **Step 5: Commit**

```bash
git add src/cartola/backtesting/squad_submission.py src/tests/backtesting/test_squad_submission.py
git commit -m "feat: load recommendation artifacts for squad plans"
```

---

### Task 3: Market Status, Formation, And Current-Athlete Validation

**Files:**
- Modify: `src/cartola/backtesting/squad_submission.py`
- Modify: `src/tests/backtesting/test_squad_submission.py`

- [ ] **Step 1: Add failing tests for public market validation**

Append this to `src/tests/backtesting/test_squad_submission.py`:

```python
from datetime import UTC, datetime

from cartola.backtesting.squad_submission import (
    RecommendationArtifact,
    parse_schemes,
    validate_artifact_against_public_market,
)


def _status_payload(*, round_number: int = 16, status_mercado: int = 1, deadline: int = 1778966940) -> dict[str, object]:
    return {
        "temporada": 2026,
        "rodada_atual": round_number,
        "status_mercado": status_mercado,
        "game_over": False,
        "fechamento": {"timestamp": deadline},
    }


def _schemes_payload() -> list[dict[str, object]]:
    return [
        {"esquema_id": 3, "nome": "4-3-3", "posicoes": {"gol": 1, "lat": 2, "zag": 2, "mei": 3, "ata": 3, "tec": 1}},
        {"esquema_id": 4, "nome": "4-4-2", "posicoes": {"gol": 1, "lat": 2, "zag": 2, "mei": 4, "ata": 2, "tec": 1}},
    ]


def _market_payload_from_artifact(artifact: RecommendationArtifact) -> dict[str, object]:
    posicao_ids = {"gol": 1, "lat": 2, "zag": 3, "mei": 4, "ata": 5, "tec": 6}
    rows = []
    for row in artifact.selected.to_dict("records"):
        posicao = str(row["posicao"])
        rows.append(
            {
                "atleta_id": int(row["id_atleta"]),
                "apelido": str(row["apelido"]),
                "clube_id": int(row["id_clube"]),
                "posicao_id": posicao_ids[posicao],
                "status_id": 7,
                "status": {"id": 7, "nome": "Provável"},
                "preco_num": float(row["preco_pre_rodada"]),
                "rodada_id": 15,
            }
        )
    return {
        "posicoes": {str(value): {"id": value, "abreviacao": key} for key, value in posicao_ids.items()},
        "atletas": rows,
    }


def test_parse_schemes_extracts_formation_id_and_counts() -> None:
    schemes = parse_schemes(_schemes_payload())

    assert schemes["4-3-3"].scheme_id == 3
    assert schemes["4-3-3"].position_counts == {"gol": 1, "lat": 2, "zag": 2, "mei": 3, "ata": 3, "tec": 1}


def test_validate_artifact_against_public_market_accepts_valid_current_market(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run_dir = _valid_run_dir(tmp_path)
    artifact = load_recommendation_artifact(project_root=tmp_path, recommendation_path=run_dir)
    market_payload = _market_payload_from_artifact(artifact)

    report = validate_artifact_against_public_market(
        artifact,
        status_payload=_status_payload(deadline=4102444800),
        schemes_payload=_schemes_payload(),
        market_payload=market_payload,
        now=datetime(2026, 5, 16, 12, 0, tzinfo=UTC),
        safety_margin_seconds=120,
    )

    assert report["formation_scheme_id"] == 3
    assert report["market_round"] == 16
    assert report["account_budget_verified"] is False
    assert report["not_comparable_fields"] == []


def test_validate_artifact_against_public_market_rejects_closed_market(tmp_path: Path) -> None:
    run_dir = _valid_run_dir(tmp_path)
    artifact = load_recommendation_artifact(project_root=tmp_path, recommendation_path=run_dir)

    with pytest.raises(ValueError, match="market is not open"):
        validate_artifact_against_public_market(
            artifact,
            status_payload=_status_payload(status_mercado=2, deadline=4102444800),
            schemes_payload=_schemes_payload(),
            market_payload=_market_payload_from_artifact(artifact),
            now=datetime(2026, 5, 16, 12, 0, tzinfo=UTC),
            safety_margin_seconds=120,
        )


def test_validate_artifact_against_public_market_rejects_price_drift(tmp_path: Path) -> None:
    run_dir = _valid_run_dir(tmp_path)
    artifact = load_recommendation_artifact(project_root=tmp_path, recommendation_path=run_dir)
    market_payload = _market_payload_from_artifact(artifact)
    market_payload["atletas"][0]["preco_num"] = 999.0  # type: ignore[index]

    with pytest.raises(ValueError, match="price drift"):
        validate_artifact_against_public_market(
            artifact,
            status_payload=_status_payload(deadline=4102444800),
            schemes_payload=_schemes_payload(),
            market_payload=market_payload,
            now=datetime(2026, 5, 16, 12, 0, tzinfo=UTC),
            safety_margin_seconds=120,
        )
```

- [ ] **Step 2: Run tests and verify they fail on missing public-market functions**

Run:

```bash
uv run --frozen pytest src/tests/backtesting/test_squad_submission.py::test_parse_schemes_extracts_formation_id_and_counts src/tests/backtesting/test_squad_submission.py::test_validate_artifact_against_public_market_accepts_valid_current_market src/tests/backtesting/test_squad_submission.py::test_validate_artifact_against_public_market_rejects_closed_market src/tests/backtesting/test_squad_submission.py::test_validate_artifact_against_public_market_rejects_price_drift -q
```

Expected: FAIL because `parse_schemes` and `validate_artifact_against_public_market` do not exist.

- [ ] **Step 3: Implement formation parsing and current-market validation**

Add this to `src/cartola/backtesting/squad_submission.py`:

```python
import math
import unicodedata


POSITION_ID_TO_CODE = {1: "gol", 2: "lat", 3: "zag", 4: "mei", 5: "ata", 6: "tec"}


@dataclass(frozen=True)
class FormationScheme:
    scheme_id: int
    name: str
    position_counts: dict[str, int]


def _int_value(value: object, field_name: str) -> int:
    if isinstance(value, bool):
        raise SquadSubmissionError(f"{field_name} must be an integer")
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise SquadSubmissionError(f"{field_name} must be an integer") from exc


def _float_value(value: object, field_name: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise SquadSubmissionError(f"{field_name} must be numeric") from exc
    if not math.isfinite(parsed):
        raise SquadSubmissionError(f"{field_name} must be finite")
    return parsed


def _strip_accents(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value)
    return "".join(ch for ch in normalized if not unicodedata.combining(ch)).lower().strip()


def parse_schemes(payload: JsonValue) -> dict[str, FormationScheme]:
    rows: list[Any]
    if isinstance(payload, list):
        rows = payload
    elif isinstance(payload, dict) and isinstance(payload.get("esquemas"), list):
        rows = payload["esquemas"]
    else:
        raise SquadSubmissionError("/esquemas payload must be an array or contain an esquemas array")
    schemes: dict[str, FormationScheme] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise SquadSubmissionError("every esquema row must be an object")
        name = str(row.get("nome") or row.get("esquema") or "").strip()
        scheme_id = _int_value(row.get("esquema_id", row.get("id")), "esquema_id")
        raw_positions = row.get("posicoes")
        if not isinstance(raw_positions, dict):
            raise SquadSubmissionError(f"esquema {name!r} must include posicoes")
        counts: dict[str, int] = {}
        for key, value in raw_positions.items():
            code = str(key).lower().strip()
            counts[code] = _int_value(value, f"posicoes.{code}")
        schemes[name] = FormationScheme(scheme_id=scheme_id, name=name, position_counts=counts)
    return schemes


def _market_position_map(market_payload: JsonValue) -> dict[int, str]:
    if not isinstance(market_payload, dict):
        raise SquadSubmissionError("/atletas/mercado payload must be an object")
    raw_positions = market_payload.get("posicoes", {})
    result = dict(POSITION_ID_TO_CODE)
    if isinstance(raw_positions, dict):
        for key, value in raw_positions.items():
            if isinstance(value, dict) and "abreviacao" in value:
                result[int(key)] = str(value["abreviacao"]).lower().strip()
    return result


def _market_athlete_index(market_payload: JsonValue) -> dict[int, dict[str, Any]]:
    if not isinstance(market_payload, dict):
        raise SquadSubmissionError("/atletas/mercado payload must be an object")
    raw = market_payload.get("atletas")
    if not isinstance(raw, list):
        raise SquadSubmissionError("/atletas/mercado atletas must be an array")
    result: dict[int, dict[str, Any]] = {}
    for row in raw:
        if not isinstance(row, dict):
            raise SquadSubmissionError("market athlete row must be an object")
        athlete_id = _int_value(row.get("atleta_id"), "atleta_id")
        result[athlete_id] = row
    return result


def _selected_position_counts(selected: pd.DataFrame) -> dict[str, int]:
    return {str(key): int(value) for key, value in selected["posicao"].value_counts().to_dict().items()}


def _validate_open_status(
    artifact: RecommendationArtifact,
    status_payload: JsonValue,
    *,
    now: datetime,
    safety_margin_seconds: int,
) -> None:
    if not isinstance(status_payload, dict):
        raise SquadSubmissionError("/mercado/status payload must be an object")
    season = _int_value(status_payload.get("temporada"), "temporada")
    rodada = _int_value(status_payload.get("rodada_atual"), "rodada_atual")
    status_mercado = _int_value(status_payload.get("status_mercado"), "status_mercado")
    if season != artifact.season:
        raise SquadSubmissionError(f"market season mismatch: artifact={artifact.season} market={season}")
    if rodada != artifact.target_round:
        raise SquadSubmissionError(f"market round mismatch: artifact={artifact.target_round} market={rodada}")
    if status_mercado != 1 or status_payload.get("game_over") is True:
        raise SquadSubmissionError(f"market is not open: rodada_atual={rodada} status_mercado={status_mercado}")
    fechamento = status_payload.get("fechamento")
    if not isinstance(fechamento, dict):
        raise SquadSubmissionError("market fechamento timestamp is missing")
    deadline = _int_value(fechamento.get("timestamp"), "fechamento.timestamp")
    if deadline - int(now.timestamp()) < safety_margin_seconds:
        raise SquadSubmissionError("market deadline is inside the configured safety margin")


def _status_is_playable(market_row: dict[str, Any], artifact_status: object) -> bool:
    status_id = market_row.get("status_id")
    if status_id is not None and _int_value(status_id, "status_id") == 7:
        return True
    raw_status = market_row.get("status")
    if isinstance(raw_status, dict):
        name = raw_status.get("nome")
        if isinstance(name, str) and _strip_accents(name) == "provavel":
            return True
    return _strip_accents(str(artifact_status)) == "provavel"


def validate_artifact_against_public_market(
    artifact: RecommendationArtifact,
    *,
    status_payload: JsonValue,
    schemes_payload: JsonValue,
    market_payload: JsonValue,
    now: datetime,
    safety_margin_seconds: int,
) -> dict[str, object]:
    _validate_open_status(artifact, status_payload, now=now, safety_margin_seconds=safety_margin_seconds)
    schemes = parse_schemes(schemes_payload)
    formation = str(artifact.summary.get("formation") or artifact.metadata.get("formation"))
    if formation not in schemes:
        raise SquadSubmissionError(f"unsupported or unmapped formation: {formation}")
    selected_counts = _selected_position_counts(artifact.selected)
    expected_counts = schemes[formation].position_counts
    if selected_counts != expected_counts:
        raise SquadSubmissionError(f"selected positions do not match formation {formation}: {selected_counts}")

    position_map = _market_position_map(market_payload)
    athletes = _market_athlete_index(market_payload)
    not_comparable: list[str] = []
    for row in artifact.selected.to_dict("records"):
        athlete_id = _int_value(row.get("id_atleta"), "id_atleta")
        market_row = athletes.get(athlete_id)
        if market_row is None:
            raise SquadSubmissionError(f"selected athlete missing from current market: {athlete_id}")
        if str(row.get("apelido", "")).strip() != str(market_row.get("apelido", "")).strip():
            raise SquadSubmissionError(f"nickname drift for athlete {athlete_id}")
        market_position = position_map.get(_int_value(market_row.get("posicao_id"), "posicao_id"))
        if market_position != str(row.get("posicao")).lower().strip():
            raise SquadSubmissionError(f"position drift for athlete {athlete_id}")
        if "id_clube" in row and int(row["id_clube"]) != _int_value(market_row.get("clube_id"), "clube_id"):
            raise SquadSubmissionError(f"club drift for athlete {athlete_id}")
        elif "id_clube" not in row:
            not_comparable.append("id_clube")
        if abs(_float_value(row.get("preco_pre_rodada"), "preco_pre_rodada") - _float_value(market_row.get("preco_num"), "preco_num")) > 0.01:
            raise SquadSubmissionError(f"price drift for athlete {athlete_id}")
        if not _status_is_playable(market_row, row.get("status")):
            raise SquadSubmissionError(f"status drift for athlete {athlete_id}")
    return {
        "market_round": artifact.target_round,
        "market_season": artifact.season,
        "formation": formation,
        "formation_scheme_id": schemes[formation].scheme_id,
        "selected_position_counts": selected_counts,
        "account_budget_verified": False,
        "not_comparable_fields": sorted(set(not_comparable)),
    }
```

- [ ] **Step 4: Run the public-market tests**

Run:

```bash
uv run --frozen pytest src/tests/backtesting/test_squad_submission.py::test_parse_schemes_extracts_formation_id_and_counts src/tests/backtesting/test_squad_submission.py::test_validate_artifact_against_public_market_accepts_valid_current_market src/tests/backtesting/test_squad_submission.py::test_validate_artifact_against_public_market_rejects_closed_market src/tests/backtesting/test_squad_submission.py::test_validate_artifact_against_public_market_rejects_price_drift -q
```

Expected: `4 passed`.

- [ ] **Step 5: Commit**

```bash
git add src/cartola/backtesting/squad_submission.py src/tests/backtesting/test_squad_submission.py
git commit -m "feat: validate squad plans against public market"
```

---

### Task 4: Squad Shape, Approved Profile, Payload Building, And Plan Writing

**Files:**
- Modify: `src/cartola/backtesting/squad_submission.py`
- Modify: `src/tests/backtesting/test_squad_submission.py`

- [ ] **Step 1: Add failing tests for full plan generation**

Append this to `src/tests/backtesting/test_squad_submission.py`:

```python
from cartola.backtesting.squad_submission import (
    CARTOLA_MARKET_ENDPOINT,
    CARTOLA_SCHEMES_ENDPOINT,
    CARTOLA_STATUS_ENDPOINT,
)


def _fetch_public_for_artifact(artifact: RecommendationArtifact) -> object:
    def fetch(url: str, timeout_seconds: float) -> object:
        if url == CARTOLA_STATUS_ENDPOINT:
            return _status_payload(deadline=4102444800)
        if url == CARTOLA_SCHEMES_ENDPOINT:
            return _schemes_payload()
        if url == CARTOLA_MARKET_ENDPOINT:
            return _market_payload_from_artifact(artifact)
        raise AssertionError(f"unexpected URL: {url}")

    return fetch


def test_run_submission_writes_plan_and_result_under_unique_attempt(tmp_path: Path) -> None:
    run_dir = _valid_run_dir(tmp_path)
    artifact = load_recommendation_artifact(project_root=tmp_path, recommendation_path=run_dir)

    result = run_submission(
        SubmissionConfig(project_root=tmp_path, recommendation_path=run_dir),
        fetch=_fetch_public_for_artifact(artifact),
        clock=lambda: datetime(2026, 5, 16, 13, 0, 42, tzinfo=UTC),
    )

    assert result.status == "plan_only"
    assert result.submission_plan_path is not None
    assert result.submission_result_path is not None
    assert result.submission_plan_path.exists()
    assert result.submission_result_path.exists()
    assert result.submission_plan_path.parent.name.startswith("attempt_started_at=")
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
    run_dir = _valid_run_dir(tmp_path)
    metadata_path = run_dir / "run_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["model_id"] = "ridge"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    artifact = load_recommendation_artifact(project_root=tmp_path, recommendation_path=run_dir)

    with pytest.raises(ValueError, match="non-approved model"):
        run_submission(
            SubmissionConfig(project_root=tmp_path, recommendation_path=run_dir),
            fetch=_fetch_public_for_artifact(artifact),
        )


def test_run_submission_allows_non_approved_model_for_plan_with_override_reason(tmp_path: Path) -> None:
    run_dir = _valid_run_dir(tmp_path)
    metadata_path = run_dir / "run_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["model_id"] = "ridge"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    artifact = load_recommendation_artifact(project_root=tmp_path, recommendation_path=run_dir)

    result = run_submission(
        SubmissionConfig(
            project_root=tmp_path,
            recommendation_path=run_dir,
            allow_non_approved_model=True,
            override_reason="manual review for dry-run plan",
        ),
        fetch=_fetch_public_for_artifact(artifact),
    )

    assert result.status == "plan_only"
```

- [ ] **Step 2: Run tests and verify they fail because plan generation is still minimal**

Run:

```bash
uv run --frozen pytest src/tests/backtesting/test_squad_submission.py::test_run_submission_writes_plan_and_result_under_unique_attempt src/tests/backtesting/test_squad_submission.py::test_run_submission_rejects_non_approved_model_without_override src/tests/backtesting/test_squad_submission.py::test_run_submission_allows_non_approved_model_for_plan_with_override_reason -q
```

Expected: FAIL because `run_submission` still raises `recommendation_path is required for Phase 1 plan generation` or lacks plan writing.

- [ ] **Step 3: Implement squad validation, approved-profile validation, payload building, and plan writing**

Add these functions to `src/cartola/backtesting/squad_submission.py`, then replace the non-submit branch of `run_submission` with the final implementation below:

```python
def _bool_series(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() == "true"


def _validate_selected_squad(artifact: RecommendationArtifact) -> None:
    selected = artifact.selected
    required = {"id_atleta", "apelido", "id_clube", "posicao", "status", "preco_pre_rodada", "is_captain"}
    missing = sorted(required - set(selected.columns))
    if missing:
        raise SquadSubmissionError(f"recommended_squad.csv missing columns: {missing}")
    if len(selected.index) != 12:
        raise SquadSubmissionError(f"recommendation must contain exactly 12 selected rows: {len(selected.index)}")
    position_counts = _selected_position_counts(selected)
    if position_counts.get("tec", 0) != 1:
        raise SquadSubmissionError("recommendation must contain exactly one tecnico row")
    if sum(count for position, count in position_counts.items() if position != "tec") != 11:
        raise SquadSubmissionError("recommendation must contain exactly 11 non-tecnico rows")
    ids = [_int_value(value, "id_atleta") for value in selected["id_atleta"].tolist()]
    if len(set(ids)) != len(ids):
        raise SquadSubmissionError("selected athlete IDs must be unique")
    captain_rows = [row for row in selected.to_dict("records") if _bool_series(row.get("is_captain"))]
    if len(captain_rows) != 1:
        raise SquadSubmissionError("recommendation must contain exactly one captain")
    if str(captain_rows[0].get("posicao")) == "tec":
        raise SquadSubmissionError("captain cannot be tecnico")
    if str(artifact.summary.get("mode")) != "live" or str(artifact.metadata.get("mode")) != "live":
        raise SquadSubmissionError("recommendation artifact must be live mode")
    budget_used = _float_value(artifact.summary.get("budget_used"), "budget_used")
    budget = _float_value(artifact.summary.get("budget"), "budget")
    if budget_used > budget + 1e-9:
        raise SquadSubmissionError(f"budget_used exceeds budget: {budget_used} > {budget}")


def _validate_approved_profile(artifact: RecommendationArtifact, config: SubmissionConfig) -> None:
    observed = {
        "model_id": str(artifact.metadata.get("model_id", artifact.summary.get("strategy", ""))),
        "footystats_mode": str(artifact.metadata.get("footystats_mode", "")),
        "fixture_mode": str(artifact.metadata.get("fixture_mode", "")),
        "matchup_context_mode": str(artifact.metadata.get("matchup_context_mode", "")),
        "scoring_contract_version": str(artifact.metadata.get("scoring_contract_version", artifact.summary.get("scoring_contract_version", ""))),
    }
    mismatches = {key: {"expected": expected, "actual": observed.get(key)} for key, expected in APPROVED_PROFILE.items() if observed.get(key) != expected}
    if mismatches and not config.allow_non_approved_model:
        raise SquadSubmissionError(f"non-approved model artifact requires --allow-non-approved-model: {mismatches}")
    if mismatches and not (config.override_reason or "").strip():
        raise SquadSubmissionError("--override-reason is required for non-approved model artifacts")


def _build_payload(artifact: RecommendationArtifact, validation_report: Mapping[str, object]) -> dict[str, object]:
    selected = artifact.selected
    athlete_ids = [_int_value(value, "id_atleta") for value in selected["id_atleta"].tolist()]
    captain = selected.loc[selected["is_captain"].map(_bool_series)].iloc[0]
    return {
        "esquema": _int_value(validation_report["formation_scheme_id"], "formation_scheme_id"),
        "atletas": athlete_ids,
        "capitao": _int_value(captain["id_atleta"], "capitao"),
    }


def _attempt_directory(run_dir: Path, now: datetime) -> Path:
    stamp = now.astimezone(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    base = run_dir / "submission_attempts" / f"attempt_started_at={stamp}"
    candidate = base
    suffix = 1
    while candidate.exists():
        candidate = base.with_name(f"{base.name}_{suffix}")
        suffix += 1
    candidate.mkdir(parents=True, exist_ok=False)
    return candidate


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    try:
        path.chmod(0o600)
    except OSError:
        pass


def _public_payloads(config: SubmissionConfig, fetch: Fetch) -> tuple[JsonValue, JsonValue, JsonValue]:
    return (
        fetch(CARTOLA_STATUS_ENDPOINT, config.timeout_seconds),
        fetch(CARTOLA_SCHEMES_ENDPOINT, config.timeout_seconds),
        fetch(CARTOLA_MARKET_ENDPOINT, config.timeout_seconds),
    )


def run_submission(
    config: SubmissionConfig,
    *,
    fetch: Fetch = fetch_public_json,
    clock: Clock = utc_now,
) -> SquadSubmissionResult:
    if config.confirm_submit:
        raise ContractUnverifiedError(CONTRACT_UNVERIFIED)
    if config.recommendation_path is None:
        raise SquadSubmissionError("recommendation_path is required for Phase 1 plan generation")
    if config.submission_plan is not None:
        raise SquadSubmissionError("submission_plan is only accepted with future Phase 2 submit flow")

    artifact = load_recommendation_artifact(project_root=config.project_root, recommendation_path=config.recommendation_path)
    _validate_selected_squad(artifact)
    _validate_approved_profile(artifact, config)
    status_payload, schemes_payload, market_payload = _public_payloads(config, fetch)
    validation_report = validate_artifact_against_public_market(
        artifact,
        status_payload=status_payload,
        schemes_payload=schemes_payload,
        market_payload=market_payload,
        now=clock(),
        safety_margin_seconds=config.safety_margin_seconds,
    )
    payload = _build_payload(artifact, validation_report)
    payload_hash = canonical_payload_sha256(payload)
    attempt_dir = _attempt_directory(artifact.path, clock())
    plan = {
        "plan_status": "ready_for_review",
        "phase": "phase1_plan_only",
        "recommendation_path": str(artifact.path),
        "payload": payload,
        "payload_sha256": payload_hash,
        "source_artifact_hashes": artifact.source_artifact_hashes,
        "target_round": artifact.target_round,
        "season": artifact.season,
        "formation": validation_report["formation"],
        "selected_count": int(len(artifact.selected.index)),
        "captain_id": payload["capitao"],
        "captain_name": str(artifact.summary.get("captain_name", "")),
        "model_id": str(artifact.metadata.get("model_id", "")),
        "footystats_mode": str(artifact.metadata.get("footystats_mode", "")),
        "fixture_mode": str(artifact.metadata.get("fixture_mode", "")),
        "matchup_context_mode": str(artifact.metadata.get("matchup_context_mode", "")),
        "validation_report": dict(validation_report),
    }
    result_payload = {
        "submission_status": "plan_only",
        "would_submit": False,
        "submitted_at_utc": None,
        "http_status": None,
        "auth_token_present": False,
        "auth_token_source": "not_required",
        "payload_sha256": payload_hash,
    }
    plan_path = attempt_dir / "submission_plan.json"
    result_path = attempt_dir / "submission_result.json"
    _write_json(plan_path, plan)
    _write_json(result_path, result_payload)
    return SquadSubmissionResult(
        attempt_directory=attempt_dir,
        submission_plan_path=plan_path,
        submission_result_path=result_path,
        payload_sha256=payload_hash,
        status="plan_only",
    )
```

- [ ] **Step 4: Run Task 4 tests**

Run:

```bash
uv run --frozen pytest src/tests/backtesting/test_squad_submission.py::test_run_submission_writes_plan_and_result_under_unique_attempt src/tests/backtesting/test_squad_submission.py::test_run_submission_rejects_non_approved_model_without_override src/tests/backtesting/test_squad_submission.py::test_run_submission_allows_non_approved_model_for_plan_with_override_reason -q
```

Expected: `3 passed`.

- [ ] **Step 5: Run all squad submission module tests so far**

Run:

```bash
uv run --frozen pytest src/tests/backtesting/test_squad_submission.py -q
```

Expected: all tests in `test_squad_submission.py` pass.

- [ ] **Step 6: Commit**

```bash
git add src/cartola/backtesting/squad_submission.py src/tests/backtesting/test_squad_submission.py
git commit -m "feat: write Cartola submission plans"
```

---

### Task 5: CLI Wrapper

**Files:**
- Create: `scripts/submit_recommended_squad.py`
- Create: `src/tests/backtesting/test_submit_recommended_squad_cli.py`

- [ ] **Step 1: Write failing CLI tests**

Create `src/tests/backtesting/test_submit_recommended_squad_cli.py`:

```python
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from cartola.backtesting.squad_submission import SquadSubmissionResult, SubmissionConfig

SCRIPT_PATH = Path(__file__).resolve().parents[3] / "scripts" / "submit_recommended_squad.py"
SPEC = importlib.util.spec_from_file_location("submit_recommended_squad", SCRIPT_PATH)
assert SPEC is not None
assert SPEC.loader is not None
submit_recommended_squad = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(submit_recommended_squad)
main = submit_recommended_squad.main
parse_args = submit_recommended_squad.parse_args


def test_parse_args_accepts_recommendation_path() -> None:
    args = parse_args(["--recommendation-path", "data/08_reporting/recommendations/2026/round-16/live/runs/run_started_at=x"])

    assert args.recommendation_path == Path("data/08_reporting/recommendations/2026/round-16/live/runs/run_started_at=x")
    assert args.project_root == Path(".")
    assert args.confirm_submit is False


def test_parse_args_rejects_both_recommendation_and_plan() -> None:
    with pytest.raises(SystemExit):
        parse_args(["--recommendation-path", "run", "--submission-plan", "plan.json"])


def test_main_prints_plan_summary(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    observed: list[SubmissionConfig] = []

    def fake_run_submission(config: SubmissionConfig) -> SquadSubmissionResult:
        observed.append(config)
        return SquadSubmissionResult(
            attempt_directory=tmp_path / "attempt",
            submission_plan_path=tmp_path / "attempt" / "submission_plan.json",
            submission_result_path=tmp_path / "attempt" / "submission_result.json",
            payload_sha256="abc123",
            status="plan_only",
        )

    monkeypatch.setattr(submit_recommended_squad, "run_submission", fake_run_submission)

    exit_code = main(["--recommendation-path", "run", "--project-root", str(tmp_path)])

    assert exit_code == 0
    assert observed == [SubmissionConfig(project_root=tmp_path, recommendation_path=Path("run"))]
    output = capsys.readouterr().out
    assert "Submission plan ready" in output
    assert "abc123" in output


def test_main_contract_unverified_does_not_load_dotenv(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    calls: list[str] = []

    def fake_run_submission(config: SubmissionConfig) -> SquadSubmissionResult:
        calls.append("run_submission")
        from cartola.backtesting.squad_submission import CONTRACT_UNVERIFIED, ContractUnverifiedError

        raise ContractUnverifiedError(CONTRACT_UNVERIFIED)

    def fake_load_dotenv(*args: object, **kwargs: object) -> None:
        calls.append("load_dotenv")
        raise AssertionError("Phase 1 must not load dotenv")

    monkeypatch.setattr(submit_recommended_squad, "run_submission", fake_run_submission)
    monkeypatch.setattr(submit_recommended_squad, "load_dotenv", fake_load_dotenv, raising=False)

    exit_code = main(
        [
            "--submission-plan",
            str(tmp_path / "submission_plan.json"),
            "--confirm-payload-sha256",
            "abc123",
            "--confirm-submit",
            "--project-root",
            str(tmp_path),
        ]
    )

    assert exit_code == 1
    assert calls == ["run_submission"]
    assert "CONTRACT_UNVERIFIED" in capsys.readouterr().err
```

- [ ] **Step 2: Run CLI tests and verify they fail because the script does not exist**

Run:

```bash
uv run --frozen pytest src/tests/backtesting/test_submit_recommended_squad_cli.py -q
```

Expected: FAIL with `FileNotFoundError` for `scripts/submit_recommended_squad.py`.

- [ ] **Step 3: Implement the CLI script**

Create `scripts/submit_recommended_squad.py`:

```python
#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from cartola.backtesting.squad_submission import (
    CONTRACT_UNVERIFIED,
    ContractUnverifiedError,
    SquadSubmissionError,
    SquadSubmissionResult,
    SubmissionConfig,
    run_submission,
)


def _positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a Cartola squad submission plan from a recommendation artifact.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--recommendation-path", type=Path)
    source.add_argument("--submission-plan", type=Path)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--timeout-seconds", type=_positive_float, default=30.0)
    parser.add_argument("--confirm-submit", action="store_true")
    parser.add_argument("--confirm-payload-sha256")
    parser.add_argument("--allow-non-approved-model", action="store_true")
    parser.add_argument("--override-reason")
    return parser.parse_args(argv)


def _config_from_args(args: argparse.Namespace) -> SubmissionConfig:
    return SubmissionConfig(
        project_root=args.project_root,
        recommendation_path=args.recommendation_path,
        submission_plan=args.submission_plan,
        timeout_seconds=args.timeout_seconds,
        confirm_submit=args.confirm_submit,
        confirm_payload_sha256=args.confirm_payload_sha256,
        allow_non_approved_model=args.allow_non_approved_model,
        override_reason=args.override_reason,
    )


def _print_success(console: Console, result: SquadSubmissionResult) -> None:
    console.print(Panel("Phase 1 plan generated; no authenticated submit was attempted.", title="Submission plan ready", border_style="green"))
    table = Table(show_header=True, header_style="bold")
    table.add_column("Field", style="cyan", no_wrap=True)
    table.add_column("Value", overflow="fold")
    table.add_row("Status", result.status)
    table.add_row("Payload SHA-256", str(result.payload_sha256))
    table.add_row("Attempt directory", str(result.attempt_directory))
    table.add_row("Submission plan", str(result.submission_plan_path))
    table.add_row("Submission result", str(result.submission_result_path))
    console.print(table)


def _print_error(console: Console, error: Exception) -> None:
    console.print(Panel(str(error), title="Submission plan failed", border_style="red"))


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    stdout = Console()
    stderr = Console(stderr=True)
    try:
        result = run_submission(_config_from_args(args))
    except ContractUnverifiedError:
        _print_error(stderr, ContractUnverifiedError(CONTRACT_UNVERIFIED))
        return 1
    except SquadSubmissionError as error:
        _print_error(stderr, error)
        return 1
    _print_success(stdout, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run CLI tests and verify they pass**

Run:

```bash
uv run --frozen pytest src/tests/backtesting/test_submit_recommended_squad_cli.py -q
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add scripts/submit_recommended_squad.py src/tests/backtesting/test_submit_recommended_squad_cli.py
git commit -m "feat: add squad submission plan CLI"
```

---

### Task 6: Documentation And Final Verification

**Files:**
- Modify: `AGENTS.md`
- Modify: `roadmap.md`

- [ ] **Step 1: Update `AGENTS.md` with the Phase 1 workflow**

Add a short section under the live recommendation workflow:

```markdown
## Cartola Submission Plan Workflow

- Phase 1 only: build a sanitized submission plan from a reviewed live recommendation artifact:
  `uv run --frozen python scripts/submit_recommended_squad.py --recommendation-path data/08_reporting/recommendations/2026/round-16/live/runs/run_started_at=20260516T130042922935Z`
- Outputs are written under timestamped attempt directories such as `submission_attempts/attempt_started_at=20260516T130042000000Z/` beside the recommendation run and include `submission_plan.json` and `submission_result.json`.
- Real submit is intentionally disabled in Phase 1. Any invocation with `--confirm-submit` must fail with `CONTRACT_UNVERIFIED` before loading `.env`, reading `CARTOLA_GLB_TOKEN`, constructing an authenticated HTTP client, or constructing any POST request.
- Phase 2 requires a separate spec with verified save/read-back contract, authenticated team identity preflight, account budget verification, and `CARTOLA_EXPECTED_TEAM_ID`.
```

- [ ] **Step 2: Update `roadmap.md` with the milestone result**

Add or update the next-milestone entry:

```markdown
### M010 Cartola Submission Plan Phase 1

Build the safe artifact-to-plan workflow for reviewed live recommendations. Scope is plan generation only: public market validation, current-athlete drift checks, canonical payload hashing, and timestamped audit artifacts. Real authenticated POST remains blocked with `CONTRACT_UNVERIFIED`; Phase 2 needs verified save/read-back contract and authenticated team identity preflight before any POST can be enabled.
```

- [ ] **Step 3: Run focused tests**

Run:

```bash
uv run --frozen pytest src/tests/backtesting/test_squad_submission.py src/tests/backtesting/test_submit_recommended_squad_cli.py -q
```

Expected: all tests pass.

- [ ] **Step 4: Run quality gate**

Run:

```bash
uv run --frozen pyrepo-check --all
```

Expected: command exits `0`.

- [ ] **Step 5: Manual smoke command against a recent artifact**

Run:

```bash
uv run --frozen python scripts/submit_recommended_squad.py \
  --recommendation-path data/08_reporting/recommendations/2026/round-16/live/runs/run_started_at=20260516T130042922935Z
```

Expected: command exits `0` while the market is open and public API checks pass, writing:

```text
data/08_reporting/recommendations/2026/round-16/live/runs/run_started_at=20260516T130042922935Z/submission_attempts/attempt_started_at=20260516T130042000000Z/submission_plan.json
data/08_reporting/recommendations/2026/round-16/live/runs/run_started_at=20260516T130042922935Z/submission_attempts/attempt_started_at=20260516T130042000000Z/submission_result.json
```

The exact timestamp segment will match the command runtime.

If the market has closed or player prices/statuses changed, this smoke command may fail with a validation error. That is acceptable; document the exact error in the final implementation summary.

- [ ] **Step 6: Manual contract gate smoke**

Run:

```bash
plan_path="$(find data/08_reporting/recommendations/2026/round-16/live/runs/run_started_at=20260516T130042922935Z/submission_attempts -name submission_plan.json -print | sort | tail -n 1)"
uv run --frozen python scripts/submit_recommended_squad.py \
  --submission-plan "$plan_path" \
  --confirm-payload-sha256 abc123 \
  --confirm-submit
```

Expected: command exits nonzero with `CONTRACT_UNVERIFIED`.

- [ ] **Step 7: Commit docs and final implementation**

```bash
git add AGENTS.md roadmap.md
git commit -m "docs: document Cartola submission plan workflow"
```

- [ ] **Step 8: Final status**

Report that Phase 1 is implemented, real submit remains disabled with
`CONTRACT_UNVERIFIED`, and include the exact outcomes from the focused tests,
`pyrepo-check`, manual plan smoke, and manual contract-gate smoke.

---

## Self-Review Checklist

- Spec coverage:
  - Phase 1 plan generation: Task 4.
  - Public market status, schemes, athlete drift: Task 3.
  - Artifact hashes and path safety: Task 2.
  - Canonical payload hash: Task 1.
  - `CONTRACT_UNVERIFIED` before token/auth/POST: Tasks 1 and 5.
  - CLI and audit outputs: Tasks 4 and 5.
  - Documentation: Task 6.

- Phase 2 exclusion:
  - No implementation task enables POST.
  - Phase 2 tests are listed as future scope only.
  - Phase 1 CLI submit attempts fail before auth.

- Verification:
  - Focused pytest command is included.
  - Full `pyrepo-check --all` command is included.
  - Manual smoke commands are included with expected outcomes.
