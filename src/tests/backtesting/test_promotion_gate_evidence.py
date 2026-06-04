from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from cartola.backtesting.promotion_gate_evidence import (
    discover_decision_artifacts,
    scan_promotion_gate_evidence,
    write_markdown_report,
)


def test_scan_reports_pass_and_fail_by_required_field_presence(tmp_path: Path) -> None:
    root = tmp_path / "data"
    complete = root / "complete_decision.json"
    incomplete = root / "incomplete_decision.json"
    _write_json(complete, _complete_decision())
    _write_json(incomplete, {"decision_status": "rejected"})

    checks = scan_promotion_gate_evidence(root, project_root=tmp_path)

    by_name = {check.path.name: check for check in checks}
    assert by_name["complete_decision.json"].status == "pass"
    assert by_name["complete_decision.json"].missing_fields == ()
    assert by_name["incomplete_decision.json"].status == "fail"
    assert "candidate_model_id" in by_name["incomplete_decision.json"].missing_fields


def test_discovery_skips_generated_verification_report_json(tmp_path: Path) -> None:
    root = tmp_path / "data"
    _write_json(root / "promotion_gate_evidence_verification.json", {"decision_status": "rejected"})
    _write_json(root / "ridge_promotion_decision.json", _complete_decision())

    discovered = discover_decision_artifacts(root)

    assert [path.name for path in discovered] == ["ridge_promotion_decision.json"]


def test_markdown_report_contains_per_artifact_summary(tmp_path: Path) -> None:
    root = tmp_path / "data"
    artifact = root / "ridge_promotion_decision.json"
    _write_json(artifact, _complete_decision())
    report = tmp_path / "promotion_gate_evidence_verification.md"

    checks = scan_promotion_gate_evidence(root, project_root=tmp_path)
    write_markdown_report(
        checks,
        report,
        generated_at=datetime(2026, 6, 4, 12, 0, tzinfo=UTC),
    )

    text = report.read_text(encoding="utf-8")
    assert "Generated at: `2026-06-04T12:00:00Z`" in text
    assert "- Report generated at UTC: `2026-06-04T12:00:00Z`" in text
    assert "Artifact identifiers are deterministic links" in text
    assert "| [`artifact-" in text
    assert "`data/ridge_promotion_decision.json` | `pass`" in text
    assert "- Artifact ID: `artifact-" in text
    assert "- Source artifact: `data/ridge_promotion_decision.json`" in text
    assert "`candidate_model_id` | `present` | `candidate_strategy.model_id`" in text


def _write_json(path: Path, data: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _complete_decision() -> dict[str, object]:
    return {
        "decision_artifact_id": "decision-1",
        "decision_artifact_sha256": "sha256:abc",
        "generated_at_utc": "2026-06-04T12:00:00Z",
        "candidate_strategy": {"model_id": "ridge", "feature_pack": "ppg_xg"},
        "control_strategy": {"model_id": "xgboost_depth2_l2_heavy", "feature_pack": "ppg_xg"},
        "promotion_seasons": [2020, 2021, 2022, 2023, 2024, 2025],
        "budget_policy": "moving",
        "candidate_vs_control_summary": {"total_actual_points_delta": 450.82},
        "budget_risk_summary": {"budget_risk_pass": False},
        "dnp_or_availability_checks": "not_applicable: historical backtest does not measure live DNP exposure",
        "gate_results": {"calibration_pass": True, "comparability_ok": True},
        "decision_status": "candidate_requires_budget_guardrail",
        "recommendation": "keep_xgboost_default_until_live_budget_risk_guardrails",
        "authority_refs": ["REQ.default-promotion-gate", "DATA.promotion-decision"],
        "source_experiment_path": "data/08_reporting/experiments/model_feature/source",
    }
