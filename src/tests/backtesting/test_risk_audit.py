from __future__ import annotations

from time import perf_counter

import pandas as pd
import pytest

from cartola.backtesting.risk_audit import build_risk_audit


def _selected_squad(*, status: str = "Provavel", captain_std: float = 2.5) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "id_atleta": 1,
                "apelido": "Captain",
                "posicao": "ata",
                "status": status,
                "is_captain": True,
                "captain_policy_ev": True,
                "captain_policy_safe": False,
                "captain_policy_upside": False,
                "prior_points_std": captain_std,
            },
            {
                "id_atleta": 2,
                "apelido": "Mid",
                "posicao": "mei",
                "status": "Provavel",
                "is_captain": False,
                "captain_policy_ev": False,
                "captain_policy_safe": False,
                "captain_policy_upside": False,
                "prior_points_std": 1.0,
            },
        ]
    )


def _large_selected_squad(player_count: int) -> pd.DataFrame:
    rows = []
    for player_id in range(1, player_count + 1):
        rows.append(
            {
                "id_atleta": player_id,
                "apelido": f"Player {player_id}",
                "posicao": "ata" if player_id == 1 else "mei",
                "status": "Provavel",
                "is_captain": player_id == 1,
                "captain_policy_ev": player_id == 1,
                "captain_policy_safe": False,
                "captain_policy_upside": False,
                "prior_points_std": 2.0 if player_id == 1 else 1.0,
            }
        )
    return pd.DataFrame(rows)


def test_build_risk_audit_emits_required_advisory_fields() -> None:
    audit = build_risk_audit(
        selected=_selected_squad(),
        summary={"budget": 100.0, "budget_used": 92.5},
        generated_timestamp="2026-06-04T12:00:00Z",
    )

    assert audit["schema_version"] == "cartola.risk_audit.v1"
    assert audit["generated_timestamp"] == "2026-06-04T12:00:00Z"
    assert audit["advisory_only"] is True
    assert audit["budget_utilization_pct"] == 92.5
    assert audit["captain_risk_policy"] == "ev"
    assert audit["captain_risk_score"] == 2.5
    assert audit["overall_risk_level"] == "low"
    assert audit["warnings"] == []
    assert [risk["dnp_risk"] for risk in audit["dnp_risk"]] == [False, False]


@pytest.mark.parametrize(
    ("captain_std", "budget_used"),
    [
        (5.0, 88.0),
        (2.0, 98.0),
    ],
)
def test_build_risk_audit_marks_medium_risk_for_volatility_or_high_budget_use(
    captain_std: float,
    budget_used: float,
) -> None:
    audit = build_risk_audit(
        selected=_selected_squad(captain_std=captain_std),
        summary={"budget": 100.0, "budget_used": budget_used},
        generated_timestamp="2026-06-04T12:00:00Z",
    )

    assert audit["overall_risk_level"] == "medium"
    assert audit["warnings"] == []


def test_build_risk_audit_forces_high_risk_for_unavailable_status() -> None:
    audit = build_risk_audit(
        selected=_selected_squad(status="Suspenso"),
        summary={"budget": 100.0, "budget_used": 88.0},
        generated_timestamp="2026-06-04T12:00:00Z",
    )

    assert audit["overall_risk_level"] == "high"
    assert audit["dnp_risk"][0]["dnp_risk"] is True
    assert audit["dnp_risk"][0]["reason"] == "pre-lock status is 'Suspenso'"
    assert "Captain" in audit["warnings"][0]


def test_build_risk_audit_rejects_missing_positive_budget() -> None:
    with pytest.raises(ValueError, match="positive operator-provided budget"):
        build_risk_audit(
            selected=_selected_squad(),
            summary={"budget": 0.0, "budget_used": 0.0},
            generated_timestamp="2026-06-04T12:00:00Z",
        )


def test_build_risk_audit_completes_under_two_seconds_for_large_selected_squad() -> None:
    started = perf_counter()

    audit = build_risk_audit(
        selected=_large_selected_squad(1_000),
        summary={"budget": 100.0, "budget_used": 92.5},
        generated_timestamp="2026-06-04T12:00:00Z",
    )

    elapsed = perf_counter() - started
    assert elapsed < 2.0
    assert len(audit["dnp_risk"]) == 1_000
