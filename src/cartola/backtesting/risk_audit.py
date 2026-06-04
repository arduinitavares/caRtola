from __future__ import annotations

import math
import unicodedata
from typing import Mapping

import pandas as pd

RISK_AUDIT_SCHEMA_VERSION = "cartola.risk_audit.v1"
SAFE_STATUSES = {"provavel", "provavel titular"}
CAPTAIN_MEDIUM_RISK_STD = 5.0
BUDGET_MEDIUM_UTILIZATION_PCT = 98.0


def _ascii_lower(value: object) -> str:
    normalized = unicodedata.normalize("NFKD", str(value).strip().lower())
    return normalized.encode("ascii", "ignore").decode("ascii")


def _finite_float(value: object, *, default: float = 0.0) -> float:
    try:
        result = float(str(value))
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _bool_value(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "t", "yes", "y"}


def _captain_row(selected: pd.DataFrame) -> pd.Series:
    captain_mask = selected["is_captain"].map(_bool_value)
    if int(captain_mask.sum()) != 1:
        raise ValueError("Risk audit requires exactly one selected captain")
    return selected.loc[captain_mask].iloc[0]


def _captain_policy(captain: pd.Series) -> str:
    for policy in ("ev", "safe", "upside"):
        if _bool_value(captain.get(f"captain_policy_{policy}")):
            return policy
    return "ev"


def _dnp_risk_rows(selected: pd.DataFrame) -> list[dict[str, object]]:
    risks: list[dict[str, object]] = []
    for row in selected.to_dict("records"):
        status = str(row.get("status", ""))
        normalized_status = _ascii_lower(status)
        dnp_risk = normalized_status not in SAFE_STATUSES
        risks.append(
            {
                "player_id": int(row["id_atleta"]),
                "player_name": str(row["apelido"]),
                "position": str(row["posicao"]),
                "status": status,
                "dnp_risk": dnp_risk,
                "reason": None if not dnp_risk else f"pre-lock status is {status!r}",
            }
        )
    return risks


def _overall_risk_level(
    *,
    dnp_risk: list[dict[str, object]],
    captain_risk_score: float,
    budget_utilization_pct: float,
) -> tuple[str, list[str]]:
    warnings: list[str] = []
    risky_players = [risk for risk in dnp_risk if risk["dnp_risk"]]
    if risky_players:
        names = ", ".join(str(risk["player_name"]) for risk in risky_players)
        warnings.append(f"Selected squad includes unavailable or uncertain pre-lock statuses: {names}")
        return "high", warnings
    if captain_risk_score >= CAPTAIN_MEDIUM_RISK_STD:
        return "medium", warnings
    if budget_utilization_pct >= BUDGET_MEDIUM_UTILIZATION_PCT:
        return "medium", warnings
    return "low", warnings


def build_risk_audit(
    *,
    selected: pd.DataFrame,
    summary: Mapping[str, object],
    generated_timestamp: str,
) -> dict[str, object]:
    budget = _finite_float(summary.get("budget"))
    budget_used = _finite_float(summary.get("budget_used"))
    if budget <= 0:
        raise ValueError("Risk audit requires a positive operator-provided budget")

    captain = _captain_row(selected)
    captain_risk_score = _finite_float(captain.get("prior_points_std"))
    budget_utilization_pct = budget_used / budget * 100.0
    dnp_risk = _dnp_risk_rows(selected)
    overall_risk_level, warnings = _overall_risk_level(
        dnp_risk=dnp_risk,
        captain_risk_score=captain_risk_score,
        budget_utilization_pct=budget_utilization_pct,
    )

    return {
        "schema_version": RISK_AUDIT_SCHEMA_VERSION,
        "generated_timestamp": generated_timestamp,
        "advisory_only": True,
        "budget": budget,
        "budget_used": budget_used,
        "budget_utilization_pct": budget_utilization_pct,
        "captain_risk_policy": _captain_policy(captain),
        "captain_risk_score": captain_risk_score,
        "dnp_risk": dnp_risk,
        "overall_risk_level": overall_risk_level,
        "warnings": warnings,
    }
