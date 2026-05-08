from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping, SupportsFloat, SupportsIndex, cast

import pandas as pd

CONTROL_MODEL_ID = "xgboost_depth2_slow"
CONTROL_FEATURE_PACK = "ppg_xg_matchup"
CHALLENGER_FEATURE_PACK = "ppg_xg_matchup_h004"
REQUIRED_SEASONS = (2021, 2022, 2023, 2024, 2025)


class H004FeatureDecisionError(ValueError):
    """Raised when H004 Phase 2 decision artifacts cannot be interpreted."""


def write_h004_phase2_decision(*, experiment_path: Path, phase1_decision_path: Path) -> Path:
    decision = build_h004_phase2_decision(
        experiment_path=experiment_path,
        phase1_decision_path=phase1_decision_path,
    )
    output_path = experiment_path / "h004_phase2_decision.json"
    output_path.write_text(json.dumps(decision, indent=2, sort_keys=True), encoding="utf-8")
    return output_path


def build_h004_phase2_decision(*, experiment_path: Path, phase1_decision_path: Path) -> dict[str, Any]:
    artifacts = _load_required_artifacts(Path(experiment_path))
    phase1 = _read_json(Path(phase1_decision_path), label="Phase 1 decision")
    passed_families = phase1.get("passed_families", [])
    phase1_ok = phase1.get("diagnostic_status") == "passes" and isinstance(passed_families, list) and "C" in passed_families

    fixture_identity_status = _fixture_identity_status(artifacts["metadata"])
    candidate_signature_status = _candidate_signature_status(artifacts["metadata"])
    validation_errors = _validation_errors(
        artifacts=artifacts,
        phase1_ok=phase1_ok,
        fixture_identity_status=fixture_identity_status,
        candidate_signature_status=candidate_signature_status,
    )
    gate_payload = _build_gate_payload(artifacts)
    gate_results = gate_payload["gate_results"]
    failed_gates = [name for name, passed in gate_results.items() if not bool(passed)]

    if validation_errors:
        final_status = "invalid"
        reasons = validation_errors
    elif failed_gates:
        final_status = "rejected"
        reasons = failed_gates
    elif fixture_identity_status == "verified":
        final_status = "candidate_research"
        reasons: list[str] = []
    else:
        final_status = "diagnostic_only"
        reasons = [f"fixture_identity_status={fixture_identity_status}"]

    return {
        "hypothesis_id": "H004",
        "phase": "feature_pack_phase2",
        "control": {"model_id": CONTROL_MODEL_ID, "feature_pack": CONTROL_FEATURE_PACK},
        "challenger": {"model_id": CONTROL_MODEL_ID, "feature_pack": CHALLENGER_FEATURE_PACK},
        "phase1_precondition_status": "passes" if phase1_ok else "failed",
        "fixture_identity_status": fixture_identity_status,
        "candidate_signature_status": candidate_signature_status,
        "final_status": final_status,
        "gate_results": gate_results,
        "season_deltas": gate_payload["season_deltas"],
        "metric_deltas": gate_payload["metric_deltas"],
        "budget_deltas": gate_payload["budget_deltas"],
        "reasons": reasons,
    }


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
        raise H004FeatureDecisionError(f"Missing required H004 Phase 2 artifacts: {', '.join(missing)}")
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
        raise H004FeatureDecisionError(f"Missing {label}: {path}") from exc
    except json.JSONDecodeError as exc:
        raise H004FeatureDecisionError(f"Invalid JSON in {label}: {path}") from exc
    if not isinstance(data, dict):
        raise H004FeatureDecisionError(f"{label} must be a JSON object: {path}")
    return data


def _validation_errors(
    *,
    artifacts: Mapping[str, Any],
    phase1_ok: bool,
    fixture_identity_status: str,
    candidate_signature_status: str,
) -> list[str]:
    errors: list[str] = []
    metadata = artifacts["metadata"]
    if not phase1_ok:
        errors.append("phase1_precondition_failed")
    if metadata.get("budget_policy") != "moving":
        errors.append("experiment budget_policy must be moving")
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
        season_deltas.append({"season": int(row["season"]), "actual_points_delta": delta})
        budget_deltas.append(
            {
                "season": int(row["season"]),
                "final_budget_delta": _finite(row["final_budget_challenger"]) - _finite(row["final_budget_control"]),
                "min_budget_delta": _finite(row["min_budget_challenger"]) - _finite(row["min_budget_control"]),
                "max_budget_drawdown_delta": _finite(row["max_budget_drawdown_challenger"])
                - _finite(row["max_budget_drawdown_control"]),
                "budget_constrained_rounds_delta": int(row["budget_constrained_rounds_challenger"])
                - int(row["budget_constrained_rounds_control"]),
            }
        )

    metric_deltas = _metric_deltas(metrics)
    aggregate_delta = sum(float(item["actual_points_delta"]) for item in season_deltas)
    positive_deltas = sorted(
        [float(item["actual_points_delta"]) for item in season_deltas if float(item["actual_points_delta"]) > 0.0],
        reverse=True,
    )
    positive_sum = sum(positive_deltas)
    concentration = math.inf if aggregate_delta <= 0.0 or positive_sum <= 0.0 else sum(positive_deltas[:2]) / positive_sum
    top50_regressions = sum(
        1 for row in metric_deltas if row["metric_scope"] == "top50_candidates" and float(row["delta"]) < -0.02
    )
    challenger_calibration_rows = _metric_rows(
        metrics,
        feature_pack=CHALLENGER_FEATURE_PACK,
        metric_scope="selected_players",
    )
    calibration_pass = all(
        0.50 <= _finite(row["calibration_slope"]) <= 1.50 and int(row["observed_count"]) >= 120
        for row in challenger_calibration_rows.to_dict(orient="records")
    )
    gate_results = {
        "aggregate_delta_pass": aggregate_delta >= 85.0,
        "improved_seasons_pass": sum(1 for item in season_deltas if float(item["actual_points_delta"]) > 0.0) >= 4,
        "worst_season_delta_pass": min(float(item["actual_points_delta"]) for item in season_deltas) >= -20.0,
        "recent_season_delta_pass": next(
            float(item["actual_points_delta"]) for item in season_deltas if item["season"] == 2025
        )
        >= -10.0,
        "final_budget_pass": min(float(item["final_budget_delta"]) for item in budget_deltas) >= -15.0,
        "min_budget_pass": min(float(item["min_budget_delta"]) for item in budget_deltas) >= -15.0,
        "max_drawdown_pass": max(float(item["max_budget_drawdown_delta"]) for item in budget_deltas) <= 15.0,
        "budget_constrained_rounds_pass": sum(int(item["budget_constrained_rounds_delta"]) for item in budget_deltas)
        <= 2,
        "top50_spearman_pass": top50_regressions <= 1,
        "selected_calibration_pass": calibration_pass,
        "concentration_pass": concentration < 0.70,
    }
    return {
        "gate_results": gate_results,
        "season_deltas": season_deltas,
        "metric_deltas": metric_deltas,
        "budget_deltas": budget_deltas,
    }


def _primary_season_rows(frame: pd.DataFrame, *, feature_pack: str) -> pd.DataFrame:
    rows = frame[
        frame["model_id"].eq(CONTROL_MODEL_ID)
        & frame["feature_pack"].eq(feature_pack)
        & frame["fixture_mode"].eq("exploratory")
        & frame["budget_policy"].eq("moving")
    ].copy()
    if set(rows["season"].astype(int)) != set(REQUIRED_SEASONS):
        raise H004FeatureDecisionError(f"Missing primary rows for feature_pack={feature_pack}")
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
        raise H004FeatureDecisionError(f"Missing {metric_scope} rows for feature_pack={feature_pack}")
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
        expected_mode = "h004_attack_defense_v1" if child.get("feature_pack") == CHALLENGER_FEATURE_PACK else "none"
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
        raise H004FeatureDecisionError(f"Expected finite numeric value, got {value!r}")
    return number
