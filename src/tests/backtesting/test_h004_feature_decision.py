from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from cartola.backtesting.h004_feature_decision import (
    H004FeatureDecisionError,
    build_h004_phase2_decision,
    write_h004_phase2_decision,
)

CONTROL_CHILD_2021 = "season=2021/model=xgboost_depth2_slow/feature_pack=ppg_xg_matchup"
CHALLENGER_CHILD_2021 = "season=2021/model=xgboost_depth2_slow/feature_pack=ppg_xg_matchup_h004"


def test_h004_decision_is_diagnostic_only_when_fixture_identity_unverified(tmp_path: Path) -> None:
    phase1 = _write_phase1_decision(tmp_path, status="passes", passed_families=["C"])
    experiment = _write_experiment(
        tmp_path,
        fixture_hashes_control={},
        fixture_hashes_challenger={},
        season_deltas={2021: 25.0, 2022: 22.0, 2023: 20.0, 2024: 15.0, 2025: 12.0},
    )

    decision = build_h004_phase2_decision(experiment_path=experiment, phase1_decision_path=phase1)

    assert decision["final_status"] == "diagnostic_only"
    assert decision["fixture_identity_status"] == "unverified"
    assert decision["gate_results"]["aggregate_delta_pass"] is True


def test_h004_decision_is_candidate_research_when_all_gates_and_fixture_identity_pass(tmp_path: Path) -> None:
    phase1 = _write_phase1_decision(tmp_path, status="passes", passed_families=["C"])
    hashes = {"data/01_raw/fixtures/2021/partidas-1.csv": "same-sha"}
    experiment = _write_experiment(
        tmp_path,
        fixture_hashes_control=hashes,
        fixture_hashes_challenger=hashes,
        season_deltas={2021: 25.0, 2022: 22.0, 2023: 20.0, 2024: 15.0, 2025: 12.0},
    )

    decision = build_h004_phase2_decision(experiment_path=experiment, phase1_decision_path=phase1)

    assert decision["final_status"] == "candidate_research"
    assert decision["fixture_identity_status"] == "verified"


def test_h004_decision_rejects_failed_metric_gate(tmp_path: Path) -> None:
    phase1 = _write_phase1_decision(tmp_path, status="passes", passed_families=["C"])
    hashes = {"data/01_raw/fixtures/2021/partidas-1.csv": "same-sha"}
    experiment = _write_experiment(
        tmp_path,
        fixture_hashes_control=hashes,
        fixture_hashes_challenger=hashes,
        season_deltas={2021: 60.0, 2022: 20.0, 2023: -25.0, 2024: 18.0, 2025: 12.0},
    )

    decision = build_h004_phase2_decision(experiment_path=experiment, phase1_decision_path=phase1)

    assert decision["final_status"] == "rejected"
    assert decision["gate_results"]["worst_season_delta_pass"] is False
    assert "worst_season_delta_pass" in decision["reasons"]


def test_h004_decision_invalidates_candidate_signature_mismatch(tmp_path: Path) -> None:
    phase1 = _write_phase1_decision(tmp_path, status="passes", passed_families=["C"])
    hashes = {"data/01_raw/fixtures/2021/partidas-1.csv": "same-sha"}
    experiment = _write_experiment(
        tmp_path,
        fixture_hashes_control=hashes,
        fixture_hashes_challenger=hashes,
        season_deltas={2021: 25.0, 2022: 22.0, 2023: 20.0, 2024: 15.0, 2025: 12.0},
    )
    metadata = json.loads((experiment / "experiment_metadata.json").read_text(encoding="utf-8"))
    metadata["candidate_pool_signatures"][CHALLENGER_CHILD_2021]["5"] = "different"
    (experiment / "experiment_metadata.json").write_text(json.dumps(metadata), encoding="utf-8")

    decision = build_h004_phase2_decision(experiment_path=experiment, phase1_decision_path=phase1)

    assert decision["final_status"] == "invalid"
    assert decision["candidate_signature_status"] == "mismatch"


def test_write_h004_phase2_decision_writes_json(tmp_path: Path) -> None:
    phase1 = _write_phase1_decision(tmp_path, status="passes", passed_families=["C"])
    experiment = _write_experiment(
        tmp_path,
        fixture_hashes_control={},
        fixture_hashes_challenger={},
        season_deltas={2021: 25.0, 2022: 22.0, 2023: 20.0, 2024: 15.0, 2025: 12.0},
    )

    output = write_h004_phase2_decision(experiment_path=experiment, phase1_decision_path=phase1)

    assert output == experiment / "h004_phase2_decision.json"
    assert json.loads(output.read_text(encoding="utf-8"))["hypothesis_id"] == "H004"


def test_h004_decision_requires_phase1_passed_family_c(tmp_path: Path) -> None:
    phase1 = _write_phase1_decision(tmp_path, status="passes", passed_families=["A"])
    experiment = _write_experiment(
        tmp_path,
        fixture_hashes_control={},
        fixture_hashes_challenger={},
        season_deltas={2021: 25.0, 2022: 22.0, 2023: 20.0, 2024: 15.0, 2025: 12.0},
    )

    decision = build_h004_phase2_decision(experiment_path=experiment, phase1_decision_path=phase1)

    assert decision["final_status"] == "invalid"
    assert decision["phase1_precondition_status"] == "failed"


def test_h004_decision_rejects_missing_artifacts(tmp_path: Path) -> None:
    phase1 = _write_phase1_decision(tmp_path, status="passes", passed_families=["C"])

    with pytest.raises(H004FeatureDecisionError, match="ranked_summary.csv"):
        build_h004_phase2_decision(experiment_path=tmp_path / "missing", phase1_decision_path=phase1)


def _write_phase1_decision(tmp_path: Path, *, status: str, passed_families: list[str]) -> Path:
    path = tmp_path / "h004_diagnostic_decision.json"
    path.write_text(
        json.dumps({"diagnostic_status": status, "passed_families": passed_families}),
        encoding="utf-8",
    )
    return path


def _write_experiment(
    tmp_path: Path,
    *,
    fixture_hashes_control: dict[str, str],
    fixture_hashes_challenger: dict[str, str],
    season_deltas: dict[int, float],
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
        challenger_child = f"season={season}/model=xgboost_depth2_slow/feature_pack=ppg_xg_matchup_h004"
        signatures[control_child] = {"5": "same", "6": "same"}
        signatures[challenger_child] = {"5": "same", "6": "same"}
        child_runs.extend(
            [
                _child_record(
                    child_id=control_child,
                    season=season,
                    feature_pack="ppg_xg_matchup",
                    feature_augmentation_mode="none",
                    fixture_hashes=fixture_hashes_control,
                ),
                _child_record(
                    child_id=challenger_child,
                    season=season,
                    feature_pack="ppg_xg_matchup_h004",
                    feature_augmentation_mode="h004_attack_defense_v1",
                    fixture_hashes=fixture_hashes_challenger,
                ),
            ]
        )
        control_rows.append(_season_row(control_child, season, "ppg_xg_matchup", 1000.0, 120.0, 100.0, 10.0, 1))
        challenger_rows.append(
            _season_row(challenger_child, season, "ppg_xg_matchup_h004", 1000.0 + delta, 118.0, 99.0, 11.0, 1)
        )
        metric_rows.extend(
            [
                _metric_row(control_child, season, "ppg_xg_matchup", "top50_candidates", 0.10, None, 800),
                _metric_row(challenger_child, season, "ppg_xg_matchup_h004", "top50_candidates", 0.09, None, 800),
                _metric_row(control_child, season, "ppg_xg_matchup", "selected_players", 0.05, 0.90, 408),
                _metric_row(challenger_child, season, "ppg_xg_matchup_h004", "selected_players", 0.05, 1.00, 408),
            ]
        )
    pd.DataFrame(
        [
            {
                "rank": 1,
                "model_id": "xgboost_depth2_slow",
                "feature_pack": "ppg_xg_matchup_h004",
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
                "group": "h004-attack-defense-mismatch",
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
