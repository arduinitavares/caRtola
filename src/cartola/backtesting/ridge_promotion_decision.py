from __future__ import annotations

import json
import math
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal, Mapping, SupportsFloat, SupportsIndex, cast

import pandas as pd

DecisionStatus = Literal[
    "promote_candidate",
    "candidate_requires_budget_guardrail",
    "candidate_requires_calibration_review",
    "rejected",
    "invalid",
]

EXPECTED_ROUNDS_PER_SEASON = 34
_FloatConvertible = str | bytes | bytearray | SupportsFloat | SupportsIndex


class RidgePromotionDecisionError(ValueError):
    """Raised when Ridge promotion decision artifacts cannot be interpreted."""


def write_ridge_promotion_decision(
    *,
    experiment_path: Path,
    candidate_model: str,
    candidate_feature_pack: str,
    control_model: str,
    control_feature_pack: str,
    baseline_model: str,
    baseline_feature_pack: str,
    promotion_seasons: tuple[int, ...],
) -> Path:
    decision = build_ridge_promotion_decision(
        experiment_path=experiment_path,
        candidate_model=candidate_model,
        candidate_feature_pack=candidate_feature_pack,
        control_model=control_model,
        control_feature_pack=control_feature_pack,
        baseline_model=baseline_model,
        baseline_feature_pack=baseline_feature_pack,
        promotion_seasons=promotion_seasons,
    )
    json_path = Path(experiment_path) / "ridge_promotion_decision.json"
    markdown_path = Path(experiment_path) / "ridge_promotion_decision.md"
    json_path.write_text(json.dumps(_json_ready(decision), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown_path.write_text(_markdown_report(decision), encoding="utf-8")
    return json_path


def build_ridge_promotion_decision(
    *,
    experiment_path: Path,
    candidate_model: str,
    candidate_feature_pack: str,
    control_model: str,
    control_feature_pack: str,
    baseline_model: str,
    baseline_feature_pack: str,
    promotion_seasons: tuple[int, ...],
) -> dict[str, Any]:
    artifacts = _load_required_artifacts(Path(experiment_path))
    ranked = artifacts["ranked"]
    season = artifacts["season"]
    comparability = artifacts["comparability"]

    strategies = {
        "candidate": {"model_id": candidate_model, "feature_pack": candidate_feature_pack},
        "control": {"model_id": control_model, "feature_pack": control_feature_pack},
        "baseline": {"model_id": baseline_model, "feature_pack": baseline_feature_pack},
    }
    ranked_rows = {
        name: _strategy_ranked_row(ranked, model_id=strategy["model_id"], feature_pack=strategy["feature_pack"])
        for name, strategy in strategies.items()
    }
    season_rows = {
        name: _strategy_season_rows(
            season,
            model_id=strategy["model_id"],
            feature_pack=strategy["feature_pack"],
            promotion_seasons=promotion_seasons,
        )
        for name, strategy in strategies.items()
    }

    validation_errors = _validation_errors(
        comparability=comparability,
        ranked_rows=ranked_rows,
        season_rows=season_rows,
        promotion_seasons=promotion_seasons,
    )
    source_gates = _source_gates(
        comparability=comparability,
        ranked_rows=ranked_rows,
        season_rows=season_rows,
        promotion_seasons=promotion_seasons,
    )
    direct_summary = _direct_summary(season_rows["candidate"], season_rows["control"])
    baseline_summary = _direct_summary(season_rows["candidate"], season_rows["baseline"])
    point_gates = _point_gates(ranked_rows["candidate"], direct_summary, baseline_summary)
    calibration_gates = _calibration_gates(ranked_rows["candidate"], point_gates)
    budget_gates = _budget_gates(ranked_rows["candidate"], ranked_rows["control"])
    gate_results = {**source_gates, **point_gates, **calibration_gates, **budget_gates}
    failed_gates = [name for name, passed in gate_results.items() if not bool(passed)]
    decision_status = _decision_status(
        source_pass=all(source_gates.values()),
        point_pass=all(point_gates.values()),
        calibration_pass=bool(calibration_gates["calibration_pass"]),
        budget_pass=bool(budget_gates["budget_risk_pass"]),
    )

    return {
        "decision_status": decision_status,
        "candidate_strategy": strategies["candidate"],
        "control_strategy": strategies["control"],
        "baseline_strategy": strategies["baseline"],
        "promotion_seasons": [int(season_value) for season_value in promotion_seasons],
        "candidate_vs_control_summary": direct_summary,
        "candidate_vs_baseline_summary": baseline_summary,
        "gate_results": gate_results,
        "failed_gates": failed_gates,
        "recommendation": _recommendation(decision_status),
        "validation_errors": validation_errors,
        "source_experiment_path": str(Path(experiment_path)),
        "source_comparability_status": str(comparability.get("status", "missing")),
        "prediction_scale_warning": _prediction_scale_warning(ranked_rows["candidate"], calibration_gates),
        "budget_risk_summary": _budget_risk_summary(ranked_rows["candidate"], ranked_rows["control"]),
    }


def _load_required_artifacts(experiment_path: Path) -> dict[str, Any]:
    if not experiment_path.is_dir():
        raise RidgePromotionDecisionError(f"experiment path does not exist: {experiment_path}")
    required_files = {
        "ranked": "ranked_summary.csv",
        "season": "per_season_summary.csv",
        "metrics": "prediction_metrics.csv",
        "comparability": "comparability_report.json",
    }
    missing = [filename for filename in required_files.values() if not (experiment_path / filename).is_file()]
    if missing:
        raise RidgePromotionDecisionError(
            f"Missing required Ridge promotion decision artifacts: {', '.join(missing)}"
        )
    return {
        "ranked": pd.read_csv(experiment_path / required_files["ranked"]),
        "season": pd.read_csv(experiment_path / required_files["season"]),
        "metrics": pd.read_csv(experiment_path / required_files["metrics"]),
        "comparability": _read_json(experiment_path / required_files["comparability"]),
    }


def _read_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RidgePromotionDecisionError(f"Invalid JSON: {path}") from exc
    if not isinstance(data, dict):
        raise RidgePromotionDecisionError(f"JSON artifact must be an object: {path}")
    return data


def _strategy_ranked_row(frame: pd.DataFrame, *, model_id: str, feature_pack: str) -> pd.Series | None:
    rows = frame.loc[frame["model_id"].astype(str).eq(model_id) & frame["feature_pack"].astype(str).eq(feature_pack)]
    if rows.empty:
        return None
    return rows.iloc[0]


def _strategy_season_rows(
    frame: pd.DataFrame,
    *,
    model_id: str,
    feature_pack: str,
    promotion_seasons: tuple[int, ...],
) -> pd.DataFrame:
    rows = frame.loc[
        frame["model_id"].astype(str).eq(model_id)
        & frame["feature_pack"].astype(str).eq(feature_pack)
        & frame["season"].astype(int).isin(set(promotion_seasons))
    ]
    return rows.copy()


def _validation_errors(
    *,
    comparability: Mapping[str, Any],
    ranked_rows: Mapping[str, pd.Series | None],
    season_rows: Mapping[str, pd.DataFrame],
    promotion_seasons: tuple[int, ...],
) -> list[str]:
    errors: list[str] = []
    if comparability.get("status") != "ok":
        errors.append("comparability_report status must be ok")
    for name, row in ranked_rows.items():
        if row is None:
            errors.append(f"missing {name} ranked_summary row")
    for name, rows in season_rows.items():
        present = set(rows["season"].astype(int).tolist()) if "season" in rows else set()
        for season in promotion_seasons:
            if int(season) not in present:
                errors.append(f"missing {name} per-season row for season {season}")
    if not _budget_policy_moving(ranked_rows=ranked_rows, season_rows=season_rows):
        errors.append("candidate/control/baseline rows must all use budget_policy=moving")
    if not _total_rounds_expected(ranked_rows=ranked_rows, season_rows=season_rows, promotion_seasons=promotion_seasons):
        expected = len(promotion_seasons) * EXPECTED_ROUNDS_PER_SEASON
        errors.append(f"candidate/control/baseline rows must each evaluate {expected} rounds")
    return errors


def _source_gates(
    *,
    comparability: Mapping[str, Any],
    ranked_rows: Mapping[str, pd.Series | None],
    season_rows: Mapping[str, pd.DataFrame],
    promotion_seasons: tuple[int, ...],
) -> dict[str, bool]:
    return {
        "comparability_ok": comparability.get("status") == "ok",
        "required_rows_present": _required_rows_present(
            ranked_rows=ranked_rows,
            season_rows=season_rows,
            promotion_seasons=promotion_seasons,
        ),
        "budget_policy_moving": _budget_policy_moving(ranked_rows=ranked_rows, season_rows=season_rows),
        "total_rounds_expected": _total_rounds_expected(
            ranked_rows=ranked_rows,
            season_rows=season_rows,
            promotion_seasons=promotion_seasons,
        ),
    }


def _required_rows_present(
    *,
    ranked_rows: Mapping[str, pd.Series | None],
    season_rows: Mapping[str, pd.DataFrame],
    promotion_seasons: tuple[int, ...],
) -> bool:
    if any(row is None for row in ranked_rows.values()):
        return False
    expected = set(int(season) for season in promotion_seasons)
    return all(set(rows["season"].astype(int).tolist()) == expected for rows in season_rows.values())


def _budget_policy_moving(
    *,
    ranked_rows: Mapping[str, pd.Series | None],
    season_rows: Mapping[str, pd.DataFrame],
) -> bool:
    ranked_ok = all(row is not None and str(row.get("budget_policy")) == "moving" for row in ranked_rows.values())
    season_ok = all(
        (not rows.empty) and rows["budget_policy"].astype(str).eq("moving").all() for rows in season_rows.values()
    )
    return bool(ranked_ok and season_ok)


def _total_rounds_expected(
    *,
    ranked_rows: Mapping[str, pd.Series | None],
    season_rows: Mapping[str, pd.DataFrame],
    promotion_seasons: tuple[int, ...],
) -> bool:
    expected = len(promotion_seasons) * EXPECTED_ROUNDS_PER_SEASON
    ranked_ok = all(row is not None and _optional_int(row.get("total_rounds")) == expected for row in ranked_rows.values())
    season_ok = all(_optional_int(rows["rounds"].sum()) == expected for rows in season_rows.values() if "rounds" in rows)
    return bool(ranked_ok and season_ok and len(season_rows) == 3)


def _direct_summary(candidate: pd.DataFrame, other: pd.DataFrame) -> dict[str, Any]:
    if candidate.empty or other.empty:
        return {
            "total_actual_points_delta": None,
            "improved_seasons": 0,
            "season_deltas": [],
            "worst_season_delta": None,
            "season_2025_delta": None,
        }
    merged = candidate.merge(
        other,
        on="season",
        suffixes=("_candidate", "_other"),
        how="inner",
    ).sort_values("season")
    season_deltas = [
        {
            "season": int(row["season"]),
            "actual_points_delta": _round_float(
                float(row["total_actual_points_candidate"]) - float(row["total_actual_points_other"])
            ),
        }
        for row in merged.to_dict(orient="records")
    ]
    deltas = [float(row["actual_points_delta"]) for row in season_deltas]
    season_2025_delta = next(
        (float(row["actual_points_delta"]) for row in season_deltas if int(row["season"]) == 2025),
        None,
    )
    return {
        "candidate_actual_points": _round_float(float(candidate["total_actual_points"].sum())),
        "other_actual_points": _round_float(float(other["total_actual_points"].sum())),
        "total_actual_points_delta": _round_float(sum(deltas)),
        "improved_seasons": sum(1 for delta in deltas if delta > 0),
        "season_deltas": season_deltas,
        "worst_season_delta": _round_float(min(deltas)) if deltas else None,
        "season_2025_delta": _round_float(season_2025_delta) if season_2025_delta is not None else None,
    }


def _point_gates(candidate: pd.Series | None, control_summary: Mapping[str, Any], baseline_summary: Mapping[str, Any]) -> dict[str, bool]:
    total_control_delta = _optional_float(control_summary.get("total_actual_points_delta"))
    total_baseline_delta = _optional_float(baseline_summary.get("total_actual_points_delta"))
    improved_seasons = _optional_int(control_summary.get("improved_seasons"))
    season_2025_delta = _optional_float(control_summary.get("season_2025_delta"))
    worst_season_delta = _optional_float(control_summary.get("worst_season_delta"))
    return {
        "candidate_rank_1": candidate is not None and _optional_int(candidate.get("rank")) == 1,
        "candidate_beats_control_by_250": total_control_delta is not None and total_control_delta >= 250.0,
        "candidate_beats_baseline_by_1000": total_baseline_delta is not None and total_baseline_delta >= 1000.0,
        "candidate_improves_3_of_6_direct_seasons": improved_seasons is not None and improved_seasons >= 3,
        "candidate_2025_delta_at_least_50": season_2025_delta is not None and season_2025_delta >= 50.0,
        "candidate_no_direct_season_loss_below_150": worst_season_delta is not None and worst_season_delta >= -150.0,
    }


def _calibration_gates(candidate: pd.Series | None, point_gates: Mapping[str, bool]) -> dict[str, bool]:
    slope = None if candidate is None else _optional_float(candidate.get("selected_calibration_slope"))
    top50_delta = None if candidate is None else _optional_float(candidate.get("top50_spearman_delta"))
    strict_pass = slope is not None and 0.75 <= slope <= 1.25
    exception_pass = (
        slope is not None
        and top50_delta is not None
        and 0.60 <= slope < 0.75
        and top50_delta >= -0.01
        and all(point_gates.values())
    )
    return {
        "calibration_strict_pass": strict_pass,
        "calibration_exception_pass": exception_pass,
        "calibration_pass": strict_pass or exception_pass,
    }


def _budget_gates(candidate: pd.Series | None, control: pd.Series | None) -> dict[str, bool]:
    candidate_min_budget = None if candidate is None else _optional_float(candidate.get("worst_min_budget"))
    candidate_drawdown = None if candidate is None else _optional_float(candidate.get("worst_max_budget_drawdown"))
    control_drawdown = None if control is None else _optional_float(control.get("worst_max_budget_drawdown"))
    candidate_constrained = None if candidate is None else _optional_int(candidate.get("total_budget_constrained_rounds"))
    control_constrained = None if control is None else _optional_int(control.get("total_budget_constrained_rounds"))
    min_budget_pass = candidate_min_budget is not None and candidate_min_budget >= 75.0
    drawdown_pass = (
        candidate_drawdown is not None
        and control_drawdown is not None
        and candidate_drawdown <= control_drawdown + 15.0
    )
    constrained_pass = (
        candidate_constrained is not None
        and control_constrained is not None
        and candidate_constrained <= control_constrained + 2
    )
    return {
        "candidate_worst_min_budget_at_least_75": min_budget_pass,
        "candidate_drawdown_within_control_plus_15": drawdown_pass,
        "candidate_budget_constrained_within_control_plus_2": constrained_pass,
        "budget_risk_pass": min_budget_pass and drawdown_pass and constrained_pass,
    }


def _decision_status(*, source_pass: bool, point_pass: bool, calibration_pass: bool, budget_pass: bool) -> DecisionStatus:
    if not source_pass:
        return "invalid"
    if not point_pass:
        return "rejected"
    if calibration_pass and budget_pass:
        return "promote_candidate"
    if calibration_pass and not budget_pass:
        return "candidate_requires_budget_guardrail"
    return "candidate_requires_calibration_review"


def _recommendation(status: str) -> str:
    if status == "promote_candidate":
        return "promote_ridge_ppg_xg_live_default"
    if status == "candidate_requires_budget_guardrail":
        return "keep_xgboost_default_until_live_budget_risk_guardrails"
    if status == "candidate_requires_calibration_review":
        return "keep_xgboost_default_until_ridge_calibration_review"
    if status == "rejected":
        return "keep_xgboost_default"
    return "fix_source_artifacts_before_decision"


def _prediction_scale_warning(candidate: pd.Series | None, calibration_gates: Mapping[str, bool]) -> str | None:
    if candidate is None:
        return None
    slope = _optional_float(candidate.get("selected_calibration_slope"))
    if bool(calibration_gates["calibration_strict_pass"]):
        return None
    if bool(calibration_gates["calibration_exception_pass"]):
        return f"selected_calibration_slope={slope:.4f} passed only by exception gate"
    return f"selected_calibration_slope={slope:.4f} failed calibration gate" if slope is not None else None


def _budget_risk_summary(candidate: pd.Series | None, control: pd.Series | None) -> dict[str, float | int | None]:
    candidate_min = None if candidate is None else _optional_float(candidate.get("worst_min_budget"))
    candidate_drawdown = None if candidate is None else _optional_float(candidate.get("worst_max_budget_drawdown"))
    control_drawdown = None if control is None else _optional_float(control.get("worst_max_budget_drawdown"))
    candidate_constrained = None if candidate is None else _optional_int(candidate.get("total_budget_constrained_rounds"))
    control_constrained = None if control is None else _optional_int(control.get("total_budget_constrained_rounds"))
    return {
        "candidate_worst_min_budget": candidate_min,
        "candidate_worst_max_budget_drawdown": candidate_drawdown,
        "control_worst_max_budget_drawdown": control_drawdown,
        "max_budget_drawdown_delta": None
        if candidate_drawdown is None or control_drawdown is None
        else _round_float(candidate_drawdown - control_drawdown),
        "candidate_budget_constrained_rounds": candidate_constrained,
        "control_budget_constrained_rounds": control_constrained,
        "budget_constrained_rounds_delta": None
        if candidate_constrained is None or control_constrained is None
        else candidate_constrained - control_constrained,
    }


def _markdown_report(decision: Mapping[str, Any]) -> str:
    gate_results = decision["gate_results"]
    failed_gates = decision["failed_gates"]
    return "\n".join(
        [
            "# Ridge PPG_XG Promotion Decision",
            "",
            f"- Decision status: `{decision['decision_status']}`",
            f"- Recommendation: `{decision['recommendation']}`",
            f"- Source experiment: `{decision['source_experiment_path']}`",
            f"- Comparability: `{decision['source_comparability_status']}`",
            "",
            "## Candidate Vs Control",
            "",
            f"```json\n{json.dumps(_json_ready(decision['candidate_vs_control_summary']), indent=2, sort_keys=True)}\n```",
            "",
            "## Budget Risk",
            "",
            f"```json\n{json.dumps(_json_ready(decision['budget_risk_summary']), indent=2, sort_keys=True)}\n```",
            "",
            "## Gates",
            "",
            f"```json\n{json.dumps(_json_ready(gate_results), indent=2, sort_keys=True)}\n```",
            "",
            "## Failed Gates",
            "",
            f"```json\n{json.dumps(_json_ready(failed_gates), indent=2)}\n```",
            "",
        ]
    )


def _round_float(value: float) -> float:
    return round(float(value), 10)


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    parsed = float(cast("_FloatConvertible", value))
    if math.isnan(parsed):
        return None
    return parsed


def _optional_int(value: object) -> int | None:
    parsed = _optional_float(value)
    if parsed is None:
        return None
    return int(parsed)


def _json_ready(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, bool | str | int) or value is None:
        return value
    if isinstance(value, float):
        return None if math.isnan(value) else value
    if hasattr(value, "item"):
        item = cast("Callable[[], object]", getattr(value, "item"))
        return _json_ready(item())
    return value
