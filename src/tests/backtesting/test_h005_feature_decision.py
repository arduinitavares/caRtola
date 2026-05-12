from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from cartola.backtesting.h005_feature_decision import (
    build_h005_feature_decision,
    write_h005_feature_decision,
)

CONTROL_CHILD_2021 = "season=2021/model=xgboost_depth2_slow/feature_pack=ppg_xg_matchup"
CHALLENGER_CHILD_2021 = "season=2021/model=xgboost_depth2_slow/feature_pack=ppg_xg_matchup_h005"


def test_h005_decision_is_candidate_research_profile_when_all_gates_pass(tmp_path: Path) -> None:
    audit = _write_audit_decision(tmp_path, status="supports_reliability_hypothesis")
    experiment = _write_experiment(
        tmp_path,
        fixture_hashes={"fixture.csv": "same"},
        season_deltas={2021: 25.0, 2022: 22.0, 2023: 20.0, 2024: 15.0, 2025: 12.0},
    )

    decision = build_h005_feature_decision(experiment_path=experiment, audit_decision_path=audit)

    assert decision["decision_status"] == "candidate_research_profile"
    assert decision["mechanism_audit_status"] == "supports_reliability_hypothesis"
    assert decision["manual_points_shrinkage"] is False
    assert decision["h005_design_revision"] == "reliability_v1"
    assert decision["gate_results"]["aggregate_delta_pass"] is True


def test_h005_decision_is_weak_positive_for_stable_small_lift(tmp_path: Path) -> None:
    audit = _write_audit_decision(tmp_path, status="supports_reliability_hypothesis")
    experiment = _write_experiment(
        tmp_path,
        fixture_hashes={"fixture.csv": "same"},
        season_deltas={2021: 12.0, 2022: 10.0, 2023: 8.0, 2024: 7.0, 2025: 6.0},
    )

    decision = build_h005_feature_decision(experiment_path=experiment, audit_decision_path=audit)

    assert decision["decision_status"] == "weak_positive_research_lead"
    assert decision["gate_results"]["aggregate_delta_pass"] is False
    assert decision["gate_results"]["weak_improved_seasons_pass"] is True


def test_h005_decision_is_inconclusive_inside_noise_band(tmp_path: Path) -> None:
    audit = _write_audit_decision(tmp_path, status="supports_reliability_hypothesis")
    experiment = _write_experiment(
        tmp_path,
        fixture_hashes={"fixture.csv": "same"},
        season_deltas={2021: 6.0, 2022: 5.0, 2023: 4.0, 2024: 3.0, 2025: 2.0},
    )

    decision = build_h005_feature_decision(experiment_path=experiment, audit_decision_path=audit)

    assert decision["decision_status"] == "inconclusive"


def test_h005_decision_is_diagnostic_only_when_audit_is_mixed(tmp_path: Path) -> None:
    audit = _write_audit_decision(tmp_path, status="mixed_or_weak")
    experiment = _write_experiment(
        tmp_path,
        fixture_hashes={"fixture.csv": "same"},
        season_deltas={2021: 25.0, 2022: 22.0, 2023: 20.0, 2024: 15.0, 2025: 12.0},
    )

    decision = build_h005_feature_decision(experiment_path=experiment, audit_decision_path=audit)

    assert decision["decision_status"] == "diagnostic_only"
    assert decision["mechanism_audit_status"] == "mixed_or_weak"


def test_h005_decision_is_diagnostic_only_when_fixture_identity_unverified(tmp_path: Path) -> None:
    audit = _write_audit_decision(tmp_path, status="supports_reliability_hypothesis")
    experiment = _write_experiment(
        tmp_path,
        fixture_hashes={},
        season_deltas={2021: 25.0, 2022: 22.0, 2023: 20.0, 2024: 15.0, 2025: 12.0},
    )

    decision = build_h005_feature_decision(experiment_path=experiment, audit_decision_path=audit)

    assert decision["decision_status"] == "diagnostic_only"
    assert decision["fixture_identity_status"] == "unverified"


def test_h005_decision_invalidates_candidate_signature_mismatch(tmp_path: Path) -> None:
    audit = _write_audit_decision(tmp_path, status="supports_reliability_hypothesis")
    experiment = _write_experiment(
        tmp_path,
        fixture_hashes={"fixture.csv": "same"},
        season_deltas={2021: 25.0, 2022: 22.0, 2023: 20.0, 2024: 15.0, 2025: 12.0},
    )
    metadata = json.loads((experiment / "experiment_metadata.json").read_text(encoding="utf-8"))
    metadata["candidate_pool_signatures"][CHALLENGER_CHILD_2021]["5"] = "different"
    (experiment / "experiment_metadata.json").write_text(json.dumps(metadata), encoding="utf-8")

    decision = build_h005_feature_decision(experiment_path=experiment, audit_decision_path=audit)

    assert decision["decision_status"] == "invalid"
    assert decision["candidate_signature_status"] == "mismatch"


def test_h005_decision_rejects_failed_season_gate(tmp_path: Path) -> None:
    audit = _write_audit_decision(tmp_path, status="supports_reliability_hypothesis")
    experiment = _write_experiment(
        tmp_path,
        fixture_hashes={"fixture.csv": "same"},
        season_deltas={2021: 80.0, 2022: 20.0, 2023: -25.0, 2024: 18.0, 2025: 12.0},
    )

    decision = build_h005_feature_decision(experiment_path=experiment, audit_decision_path=audit)

    assert decision["decision_status"] == "rejected"
    assert decision["gate_results"]["worst_season_delta_pass"] is False
    assert "worst_season_delta_pass" in decision["failed_gates"]


def test_h005_candidate_profile_requires_tight_selected_calibration(tmp_path: Path) -> None:
    audit = _write_audit_decision(tmp_path, status="supports_reliability_hypothesis")
    experiment = _write_experiment(
        tmp_path,
        fixture_hashes={"fixture.csv": "same"},
        season_deltas={2021: 25.0, 2022: 22.0, 2023: 20.0, 2024: 15.0, 2025: 12.0},
        challenger_selected_calibration_slope=1.3,
    )

    decision = build_h005_feature_decision(experiment_path=experiment, audit_decision_path=audit)

    assert decision["decision_status"] != "candidate_research_profile"
    assert decision["gate_results"]["selected_calibration_pass"] is False


def test_h005_weak_positive_requires_aggregate_final_budget_pass(tmp_path: Path) -> None:
    audit = _write_audit_decision(tmp_path, status="supports_reliability_hypothesis")
    experiment = _write_experiment(
        tmp_path,
        fixture_hashes={"fixture.csv": "same"},
        season_deltas={2021: 9.0, 2022: 9.0, 2023: 9.0, 2024: 8.0, 2025: 8.0},
        challenger_final_budget_delta=-1.0,
    )

    decision = build_h005_feature_decision(experiment_path=experiment, audit_decision_path=audit)

    assert decision["decision_status"] != "weak_positive_research_lead"
    assert decision["decision_status"] == "rejected"
    assert decision["gate_results"]["final_budget_pass"] is False
    assert decision["gate_results"]["season_final_budget_pass"] is True
    assert decision["gate_results"]["budget_integrity_pass"] is True


def test_write_h005_feature_decision_writes_json(tmp_path: Path) -> None:
    audit = _write_audit_decision(tmp_path, status="supports_reliability_hypothesis")
    experiment = _write_experiment(
        tmp_path,
        fixture_hashes={"fixture.csv": "same"},
        season_deltas={2021: 6.0, 2022: 5.0, 2023: 4.0, 2024: 3.0, 2025: 2.0},
    )

    output = write_h005_feature_decision(experiment_path=experiment, audit_decision_path=audit)

    assert output == experiment / "h005_feature_decision.json"
    assert json.loads(output.read_text(encoding="utf-8"))["hypothesis_id"] == "H005"


def _write_audit_decision(tmp_path: Path, *, status: str) -> Path:
    path = tmp_path / "h005_mechanism_audit_decision.json"
    path.write_text(
        json.dumps({"hypothesis_id": "H005", "audit_status": status}),
        encoding="utf-8",
    )
    return path


def _write_experiment(
    tmp_path: Path,
    *,
    fixture_hashes: dict[str, str],
    season_deltas: dict[int, float],
    challenger_final_budget_delta: float = 1.0,
    challenger_selected_calibration_slope: float = 1.00,
) -> Path:
    experiment = tmp_path / "experiment"
    experiment.mkdir()
    control_rows = []
    challenger_rows = []
    metric_rows = []
    child_runs = []
    signatures = {}
    for season, delta in season_deltas.items():
        control_child = f"season={season}/model=xgboost_depth2_slow/feature_pack=ppg_xg_matchup"
        challenger_child = f"season={season}/model=xgboost_depth2_slow/feature_pack=ppg_xg_matchup_h005"
        signatures[control_child] = {"5": "same", "6": "same"}
        signatures[challenger_child] = {"5": "same", "6": "same"}
        child_runs.extend(
            [
                _child_record(
                    child_id=control_child,
                    season=season,
                    feature_pack="ppg_xg_matchup",
                    feature_augmentation_mode="none",
                    fixture_hashes=fixture_hashes,
                ),
                _child_record(
                    child_id=challenger_child,
                    season=season,
                    feature_pack="ppg_xg_matchup_h005",
                    feature_augmentation_mode="h005_matchup_reliability_v1",
                    fixture_hashes=fixture_hashes,
                ),
            ]
        )
        control_rows.append(_season_row(control_child, season, "ppg_xg_matchup", 1000.0, 120.0, 100.0, 10.0, 1))
        challenger_rows.append(
            _season_row(
                challenger_child,
                season,
                "ppg_xg_matchup_h005",
                1000.0 + delta,
                120.0 + challenger_final_budget_delta,
                100.5,
                9.0,
                1,
            )
        )
        metric_rows.extend(
            [
                _metric_row(control_child, season, "ppg_xg_matchup", "top50_candidates", 0.10, None, 800),
                _metric_row(challenger_child, season, "ppg_xg_matchup_h005", "top50_candidates", 0.11, None, 800),
                _metric_row(control_child, season, "ppg_xg_matchup", "selected_players", 0.05, 0.90, 408),
                _metric_row(
                    challenger_child,
                    season,
                    "ppg_xg_matchup_h005",
                    "selected_players",
                    0.05,
                    challenger_selected_calibration_slope,
                    408,
                ),
            ]
        )
    pd.DataFrame(
        [
            {
                "rank": 1,
                "model_id": "xgboost_depth2_slow",
                "feature_pack": "ppg_xg_matchup_h005",
                "fixture_mode": "exploratory",
                "budget_policy": "moving",
                "seasons_evaluated": 5,
                "total_rounds": 170,
                "total_actual_points": sum(row["total_actual_points"] for row in challenger_rows),
            }
        ]
    ).to_csv(experiment / "ranked_summary.csv", index=False)
    pd.DataFrame([*control_rows, *challenger_rows]).to_csv(experiment / "per_season_summary.csv", index=False)
    pd.DataFrame(metric_rows).to_csv(experiment / "prediction_metrics.csv", index=False)
    (experiment / "comparability_report.json").write_text(json.dumps({"status": "ok"}), encoding="utf-8")
    (experiment / "experiment_metadata.json").write_text(
        json.dumps(
            {
                "status": "ok",
                "budget_policy": "moving",
                "group": "h005-count-aware-matchup-shrinkage",
                "seasons": [2021, 2022, 2023, 2024, 2025],
                "child_runs": child_runs,
                "candidate_pool_signatures": signatures,
            }
        ),
        encoding="utf-8",
    )
    return experiment


def _child_record(
    *,
    child_id: str,
    season: int,
    feature_pack: str,
    feature_augmentation_mode: str,
    fixture_hashes: dict[str, str],
) -> dict[str, object]:
    return {
        "child_id": child_id,
        "season": season,
        "model_id": "xgboost_depth2_slow",
        "feature_pack": feature_pack,
        "fixture_mode": "exploratory",
        "feature_augmentation_mode": feature_augmentation_mode,
        "metadata": {
            "budget_policy": "moving",
            "fixture_mode": "exploratory",
            "footystats_mode": "ppg_xg",
            "matchup_context_mode": "cartola_matchup_v1",
            "scoring_contract_version": "cartola_standard_2026_v1",
            "fixture_manifest_sha256": {},
            "fixture_source_sha256": fixture_hashes,
        },
    }


def _season_row(
    child_id: str,
    season: int,
    feature_pack: str,
    total_actual_points: float,
    final_budget: float,
    min_budget: float,
    max_budget_drawdown: float,
    budget_constrained_rounds: int,
) -> dict[str, object]:
    return {
        "child_id": child_id,
        "season": season,
        "model_id": "xgboost_depth2_slow",
        "feature_pack": feature_pack,
        "fixture_mode": "exploratory",
        "budget_policy": "moving",
        "strategy": "xgboost_depth2_slow",
        "rounds": 34,
        "total_actual_points": total_actual_points,
        "final_budget": final_budget,
        "min_budget": min_budget,
        "max_budget_drawdown": max_budget_drawdown,
        "budget_constrained_rounds": budget_constrained_rounds,
    }


def _metric_row(
    child_id: str,
    season: int,
    feature_pack: str,
    metric_scope: str,
    spearman: float | None,
    calibration_slope: float | None,
    observed_count: int,
) -> dict[str, object]:
    return {
        "child_id": child_id,
        "season": season,
        "model_id": "xgboost_depth2_slow",
        "feature_pack": feature_pack,
        "fixture_mode": "exploratory",
        "budget_policy": "moving",
        "metric_scope": metric_scope,
        "observed_count": observed_count,
        "spearman": spearman,
        "calibration_slope": calibration_slope,
    }
