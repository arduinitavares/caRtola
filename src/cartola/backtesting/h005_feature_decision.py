from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping, SupportsFloat, SupportsIndex, cast

import pandas as pd

CONTROL_MODEL_ID = "xgboost_depth2_slow"
CONTROL_FEATURE_PACK = "ppg_xg_matchup"
CHALLENGER_FEATURE_PACK = "ppg_xg_matchup_h005"
REQUIRED_SEASONS = (2021, 2022, 2023, 2024, 2025)
SOURCE_EBM_DIAGNOSTIC_PATH = "data/08_reporting/ebm_diagnostics/ebm_diagnostic_started_at=20260511T004620197204Z"


class H005FeatureDecisionError(ValueError):
    """Raised when H005 feature decision artifacts cannot be interpreted."""


def write_h005_feature_decision(*, experiment_path: Path, audit_decision_path: Path) -> Path:
    decision = build_h005_feature_decision(
        experiment_path=experiment_path,
        audit_decision_path=audit_decision_path,
    )
    output_path = experiment_path / "h005_feature_decision.json"
    output_path.write_text(json.dumps(decision, indent=2, sort_keys=True), encoding="utf-8")
    return output_path


def build_h005_feature_decision(*, experiment_path: Path, audit_decision_path: Path) -> dict[str, Any]:
    artifacts = _load_required_artifacts(Path(experiment_path))
    audit = _read_json(Path(audit_decision_path), label="H005 mechanism audit decision")
    mechanism_audit_status = str(audit.get("audit_status", "missing"))
    fixture_identity_status = _fixture_identity_status(artifacts["metadata"])
    candidate_signature_status = _candidate_signature_status(artifacts["metadata"])
    validation_errors = _validation_errors(
        artifacts=artifacts,
        audit=audit,
        fixture_identity_status=fixture_identity_status,
        candidate_signature_status=candidate_signature_status,
    )
    gate_payload = _build_gate_payload(artifacts)
    gate_results = gate_payload["gate_results"]
    failed_gates = [name for name, passed in gate_results.items() if not bool(passed)]
    decision_status = _decision_status(
        validation_errors=validation_errors,
        mechanism_audit_status=mechanism_audit_status,
        fixture_identity_status=fixture_identity_status,
        candidate_signature_status=candidate_signature_status,
        gate_payload=gate_payload,
    )

    return {
        "hypothesis_id": "H005",
        "h005_design_revision": "reliability_v1",
        "manual_points_shrinkage": False,
        "decision_status": decision_status,
        "mechanism_audit_status": mechanism_audit_status,
        "fixture_identity_status": fixture_identity_status,
        "candidate_signature_status": candidate_signature_status,
        "control_strategy": {"model_id": CONTROL_MODEL_ID, "feature_pack": CONTROL_FEATURE_PACK},
        "challenger_strategy": {"model_id": CONTROL_MODEL_ID, "feature_pack": CHALLENGER_FEATURE_PACK},
        "gate_results": gate_results,
        "season_deltas": gate_payload["season_deltas"],
        "metric_deltas": gate_payload["metric_deltas"],
        "budget_deltas": gate_payload["budget_deltas"],
        "aggregate_actual_points_delta": gate_payload["aggregate_actual_points_delta"],
        "failed_gates": failed_gates,
        "validation_errors": validation_errors,
        "source_ebm_diagnostic_path": SOURCE_EBM_DIAGNOSTIC_PATH,
    }


def _decision_status(
    *,
    validation_errors: list[str],
    mechanism_audit_status: str,
    fixture_identity_status: str,
    candidate_signature_status: str,
    gate_payload: Mapping[str, Any],
) -> str:
    if validation_errors or mechanism_audit_status == "invalid":
        return "invalid"
    if mechanism_audit_status != "supports_reliability_hypothesis" or fixture_identity_status != "verified":
        return "diagnostic_only"
    if candidate_signature_status != "ok":
        return "invalid"

    gate_results = gate_payload["gate_results"]
    candidate_gates = (
        "aggregate_delta_pass",
        "improved_seasons_pass",
        "worst_season_delta_pass",
        "recent_season_delta_pass",
        "final_budget_pass",
        "season_final_budget_pass",
        "budget_integrity_pass",
        "top50_spearman_pass",
        "selected_calibration_pass",
        "concentration_pass",
    )
    if all(bool(gate_results[gate]) for gate in candidate_gates):
        return "candidate_research_profile"

    aggregate_delta = float(gate_payload["aggregate_actual_points_delta"])
    improved_seasons = int(gate_payload["improved_seasons"])
    worst_season_delta = float(gate_payload["worst_season_delta"])
    season_2025_delta = float(gate_payload["season_2025_delta"])
    concentration = float(gate_payload["concentration"])
    budget_integrity_pass = bool(gate_results["budget_integrity_pass"])
    if (
        aggregate_delta >= 40.0
        and improved_seasons >= 3
        and worst_season_delta >= -20.0
        and season_2025_delta >= -10.0
        and concentration < 0.75
        and budget_integrity_pass
    ):
        return "weak_positive_research_lead"
    if (
        -20.0 <= aggregate_delta < 40.0
        and worst_season_delta >= -20.0
        and season_2025_delta >= -10.0
        and budget_integrity_pass
    ):
        return "inconclusive"
    return "rejected"


def _load_required_artifacts(experiment_path: Path) -> dict[str, Any]:
    required_files = {
        "ranked": "ranked_summary.csv",
        "season": "per_season_summary.csv",
        "metrics": "prediction_metrics.csv",
        "comparability": "comparability_report.json",
        "metadata": "experiment_metadata.json",
    }
    missing = [filename for filename in required_files.values() if not (experiment_path / filename).exists()]
    if missing:
        raise H005FeatureDecisionError(f"Missing required H005 feature decision artifacts: {', '.join(missing)}")
    return {
        "ranked": pd.read_csv(experiment_path / required_files["ranked"]),
        "season": pd.read_csv(experiment_path / required_files["season"]),
        "metrics": pd.read_csv(experiment_path / required_files["metrics"]),
        "comparability": _read_json(experiment_path / required_files["comparability"], label="comparability report"),
        "metadata": _read_json(experiment_path / required_files["metadata"], label="experiment metadata"),
    }


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise H005FeatureDecisionError(f"Missing {label}: {path}") from exc
    except json.JSONDecodeError as exc:
        raise H005FeatureDecisionError(f"Invalid JSON in {label}: {path}") from exc
    if not isinstance(data, dict):
        raise H005FeatureDecisionError(f"{label} must be a JSON object: {path}")
    return data


def _validation_errors(
    *,
    artifacts: Mapping[str, Any],
    audit: Mapping[str, Any],
    fixture_identity_status: str,
    candidate_signature_status: str,
) -> list[str]:
    errors: list[str] = []
    metadata = artifacts["metadata"]
    if audit.get("hypothesis_id") != "H005":
        errors.append("audit_decision hypothesis_id must be H005")
    if metadata.get("budget_policy") != "moving":
        errors.append("experiment budget_policy must be moving")
    if metadata.get("group") != "h005-count-aware-matchup-shrinkage":
        errors.append("experiment group must be h005-count-aware-matchup-shrinkage")
    if tuple(metadata.get("seasons", [])) != REQUIRED_SEASONS:
        errors.append("experiment seasons must be 2021,2022,2023,2024,2025")
    if artifacts["comparability"].get("status") != "ok":
        errors.append("comparability_report status must be ok")
    if candidate_signature_status != "ok":
        errors.append(f"candidate_signature_status={candidate_signature_status}")
    if fixture_identity_status in {"mismatch", "missing"}:
        errors.append(f"fixture_identity_status={fixture_identity_status}")
    required_season_columns = {
        "child_id",
        "season",
        "model_id",
        "feature_pack",
        "fixture_mode",
        "budget_policy",
        "total_actual_points",
        "final_budget",
        "min_budget",
        "max_budget_drawdown",
        "budget_constrained_rounds",
    }
    required_metric_columns = {
        "child_id",
        "season",
        "model_id",
        "feature_pack",
        "fixture_mode",
        "budget_policy",
        "metric_scope",
        "observed_count",
        "spearman",
        "calibration_slope",
    }
    errors.extend(_missing_column_errors(artifacts["season"], required_season_columns, "per_season_summary.csv"))
    errors.extend(_missing_column_errors(artifacts["metrics"], required_metric_columns, "prediction_metrics.csv"))
    errors.extend(_child_context_errors(metadata))
    return errors


def _missing_column_errors(frame: pd.DataFrame, required: set[str], label: str) -> list[str]:
    missing = sorted(required.difference(frame.columns))
    return [f"{label} missing columns: {', '.join(missing)}"] if missing else []


def _build_gate_payload(artifacts: Mapping[str, Any]) -> dict[str, Any]:
    season = artifacts["season"].copy()
    metrics = artifacts["metrics"].copy()
    control = _primary_season_rows(season, feature_pack=CONTROL_FEATURE_PACK)
    challenger = _primary_season_rows(season, feature_pack=CHALLENGER_FEATURE_PACK)
    merged = control.merge(challenger, on="season", suffixes=("_control", "_challenger"), validate="one_to_one")
    season_deltas: list[dict[str, Any]] = []
    budget_deltas: list[dict[str, Any]] = []
    for row in merged.to_dict(orient="records"):
        delta = _finite(row["total_actual_points_challenger"]) - _finite(row["total_actual_points_control"])
        budget_constrained_rounds_delta = int(row["budget_constrained_rounds_challenger"]) - int(
            row["budget_constrained_rounds_control"]
        )
        season_deltas.append({"season": int(row["season"]), "actual_points_delta": delta})
        budget_deltas.append(
            {
                "season": int(row["season"]),
                "final_budget_delta": _finite(row["final_budget_challenger"]) - _finite(row["final_budget_control"]),
                "min_budget_delta": _finite(row["min_budget_challenger"]) - _finite(row["min_budget_control"]),
                "max_budget_drawdown_delta": _finite(row["max_budget_drawdown_challenger"])
                - _finite(row["max_budget_drawdown_control"]),
                "budget_constrained_rounds_delta": budget_constrained_rounds_delta,
            }
        )

    metric_deltas = _metric_deltas(metrics)
    aggregate_delta = sum(float(item["actual_points_delta"]) for item in season_deltas)
    improved_seasons = sum(1 for item in season_deltas if float(item["actual_points_delta"]) > 0.0)
    worst_season_delta = min(float(item["actual_points_delta"]) for item in season_deltas)
    season_2025_delta = next(float(item["actual_points_delta"]) for item in season_deltas if item["season"] == 2025)
    aggregate_final_budget_delta = sum(float(item["final_budget_delta"]) for item in budget_deltas)
    min_season_final_budget_delta = min(float(item["final_budget_delta"]) for item in budget_deltas)
    additional_budget_constrained_rounds = sum(int(item["budget_constrained_rounds_delta"]) for item in budget_deltas)
    concentration = _positive_delta_concentration(season_deltas)
    top50_nonnegative_seasons = sum(
        1 for row in metric_deltas if row["metric_scope"] == "top50_candidates" and float(row["delta"]) >= 0.0
    )
    challenger_calibration_rows = _metric_rows(
        metrics,
        feature_pack=CHALLENGER_FEATURE_PACK,
        metric_scope="selected_players",
    )
    selected_calibration_pass = all(
        0.50 <= _finite(row["calibration_slope"]) <= 1.50 and int(row["observed_count"]) >= 120
        for row in challenger_calibration_rows.to_dict(orient="records")
    )
    gate_results = {
        "aggregate_delta_pass": aggregate_delta >= 85.0,
        "improved_seasons_pass": improved_seasons >= 4,
        "weak_improved_seasons_pass": improved_seasons >= 3,
        "worst_season_delta_pass": worst_season_delta >= -20.0,
        "recent_season_delta_pass": season_2025_delta >= -10.0,
        "final_budget_pass": aggregate_final_budget_delta >= 0.0,
        "season_final_budget_pass": min_season_final_budget_delta >= -2.0,
        "budget_integrity_pass": min_season_final_budget_delta >= -2.0 and additional_budget_constrained_rounds <= 0,
        "top50_spearman_pass": top50_nonnegative_seasons >= 4,
        "selected_calibration_pass": selected_calibration_pass,
        "concentration_pass": concentration < 0.70,
        "weak_concentration_pass": concentration < 0.75,
    }
    return {
        "gate_results": gate_results,
        "season_deltas": season_deltas,
        "metric_deltas": metric_deltas,
        "budget_deltas": budget_deltas,
        "aggregate_actual_points_delta": aggregate_delta,
        "improved_seasons": improved_seasons,
        "worst_season_delta": worst_season_delta,
        "season_2025_delta": season_2025_delta,
        "aggregate_final_budget_delta": aggregate_final_budget_delta,
        "min_season_final_budget_delta": min_season_final_budget_delta,
        "additional_budget_constrained_rounds": additional_budget_constrained_rounds,
        "top50_nonnegative_seasons": top50_nonnegative_seasons,
        "selected_calibration_pass": selected_calibration_pass,
        "concentration": concentration,
    }


def _positive_delta_concentration(season_deltas: list[dict[str, Any]]) -> float:
    aggregate_delta = sum(float(item["actual_points_delta"]) for item in season_deltas)
    positive_deltas = sorted(
        [float(item["actual_points_delta"]) for item in season_deltas if float(item["actual_points_delta"]) > 0.0],
        reverse=True,
    )
    positive_sum = sum(positive_deltas)
    return math.inf if aggregate_delta <= 0.0 or positive_sum <= 0.0 else sum(positive_deltas[:2]) / positive_sum


def _primary_season_rows(frame: pd.DataFrame, *, feature_pack: str) -> pd.DataFrame:
    rows = frame[
        frame["model_id"].eq(CONTROL_MODEL_ID)
        & frame["feature_pack"].eq(feature_pack)
        & frame["fixture_mode"].eq("exploratory")
        & frame["budget_policy"].eq("moving")
    ].copy()
    if set(rows["season"].astype(int)) != set(REQUIRED_SEASONS):
        raise H005FeatureDecisionError(f"Missing primary rows for feature_pack={feature_pack}")
    return rows


def _metric_rows(frame: pd.DataFrame, *, feature_pack: str, metric_scope: str) -> pd.DataFrame:
    rows = frame[
        frame["model_id"].eq(CONTROL_MODEL_ID)
        & frame["feature_pack"].eq(feature_pack)
        & frame["fixture_mode"].eq("exploratory")
        & frame["budget_policy"].eq("moving")
        & frame["metric_scope"].eq(metric_scope)
    ].copy()
    if set(rows["season"].astype(int)) != set(REQUIRED_SEASONS):
        raise H005FeatureDecisionError(f"Missing {metric_scope} rows for feature_pack={feature_pack}")
    return rows


def _metric_deltas(metrics: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for metric_scope, metric_name in (("top50_candidates", "spearman"), ("selected_players", "calibration_slope")):
        control = _metric_rows(metrics, feature_pack=CONTROL_FEATURE_PACK, metric_scope=metric_scope)
        challenger = _metric_rows(metrics, feature_pack=CHALLENGER_FEATURE_PACK, metric_scope=metric_scope)
        merged = control.merge(challenger, on="season", suffixes=("_control", "_challenger"), validate="one_to_one")
        for row in merged.to_dict(orient="records"):
            control_value = _finite(row[f"{metric_name}_control"])
            challenger_value = _finite(row[f"{metric_name}_challenger"])
            rows.append(
                {
                    "season": int(row["season"]),
                    "metric_scope": metric_scope,
                    "metric": metric_name,
                    "control": control_value,
                    "challenger": challenger_value,
                    "delta": challenger_value - control_value,
                }
            )
    return rows


def _candidate_signature_status(metadata: Mapping[str, Any]) -> str:
    signatures = metadata.get("candidate_pool_signatures")
    if not isinstance(signatures, dict):
        return "missing"
    for season in REQUIRED_SEASONS:
        control_id = f"season={season}/model={CONTROL_MODEL_ID}/feature_pack={CONTROL_FEATURE_PACK}"
        challenger_id = f"season={season}/model={CONTROL_MODEL_ID}/feature_pack={CHALLENGER_FEATURE_PACK}"
        control = signatures.get(control_id)
        challenger = signatures.get(challenger_id)
        if control is None or challenger is None:
            return "missing"
        if control != challenger:
            return "mismatch"
    return "ok"


def _fixture_identity_status(metadata: Mapping[str, Any]) -> str:
    child_by_id = {
        str(child.get("child_id")): child
        for child in metadata.get("child_runs", [])
        if isinstance(child, dict) and child.get("child_id") is not None
    }
    saw_source_hashes = False
    for season in REQUIRED_SEASONS:
        control = child_by_id.get(f"season={season}/model={CONTROL_MODEL_ID}/feature_pack={CONTROL_FEATURE_PACK}")
        challenger = child_by_id.get(
            f"season={season}/model={CONTROL_MODEL_ID}/feature_pack={CHALLENGER_FEATURE_PACK}"
        )
        if control is None or challenger is None:
            return "missing"
        control_hashes = _fixture_hashes(control)
        challenger_hashes = _fixture_hashes(challenger)
        if control_hashes is None or challenger_hashes is None:
            return "unverified"
        saw_source_hashes = True
        if control_hashes != challenger_hashes:
            return "mismatch"
    return "verified" if saw_source_hashes else "unverified"


def _fixture_hashes(child: Mapping[str, Any]) -> dict[str, str] | None:
    metadata = child.get("metadata")
    if not isinstance(metadata, dict):
        return None
    hashes = metadata.get("fixture_source_sha256") or metadata.get("fixture_manifest_sha256")
    return hashes if isinstance(hashes, dict) and hashes else None


def _child_context_errors(metadata: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    child_runs = metadata.get("child_runs")
    if not isinstance(child_runs, list):
        return ["experiment_metadata.json child_runs must be a list"]
    for child in child_runs:
        if not isinstance(child, dict):
            errors.append("child_runs entry must be an object")
            continue
        if child.get("model_id") != CONTROL_MODEL_ID:
            continue
        if child.get("feature_pack") not in {CONTROL_FEATURE_PACK, CHALLENGER_FEATURE_PACK}:
            continue
        child_metadata = child.get("metadata")
        if not isinstance(child_metadata, dict):
            errors.append(f"{child.get('child_id')}: missing metadata")
            continue
        expected_mode = "h005_matchup_reliability_v1" if child.get("feature_pack") == CHALLENGER_FEATURE_PACK else "none"
        observed_mode = child.get("feature_augmentation_mode", child_metadata.get("feature_augmentation_mode"))
        checks = {
            "budget_policy": "moving",
            "fixture_mode": "exploratory",
            "footystats_mode": "ppg_xg",
            "matchup_context_mode": "cartola_matchup_v1",
            "scoring_contract_version": "cartola_standard_2026_v1",
        }
        for key, expected in checks.items():
            if child_metadata.get(key) != expected:
                errors.append(f"{child.get('child_id')}: {key} must be {expected}")
        if observed_mode != expected_mode:
            errors.append(f"{child.get('child_id')}: feature_augmentation_mode must be {expected_mode}")
    return errors


def _finite(value: object) -> float:
    number = float(cast("str | bytes | bytearray | SupportsFloat | SupportsIndex", value))
    if not math.isfinite(number):
        raise H005FeatureDecisionError(f"Expected finite numeric value, got {value!r}")
    return number
