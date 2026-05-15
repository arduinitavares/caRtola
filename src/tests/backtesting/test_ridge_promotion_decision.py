from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from cartola.backtesting.ridge_promotion_decision import build_ridge_promotion_decision

SEASONS = (2020, 2021, 2022, 2023, 2024, 2025)


def test_invalid_comparability_produces_invalid(tmp_path: Path) -> None:
    experiment = _write_experiment(tmp_path, comparability_status="not_comparable")

    decision = _build_decision(experiment)

    assert decision["decision_status"] == "invalid"
    assert decision["gate_results"]["comparability_ok"] is False
    assert "comparability_ok" in decision["failed_gates"]


def test_missing_strategy_season_rows_produces_invalid(tmp_path: Path) -> None:
    experiment = _write_experiment(tmp_path, omit_candidate_season=2024)

    decision = _build_decision(experiment)

    assert decision["decision_status"] == "invalid"
    assert decision["gate_results"]["required_rows_present"] is False
    assert any("missing candidate per-season row" in error for error in decision["validation_errors"])


def test_full_pass_promotes_candidate(tmp_path: Path) -> None:
    experiment = _write_experiment(
        tmp_path,
        candidate_total=12_300.0,
        control_total=11_900.0,
        baseline_total=10_800.0,
        candidate_rank=1,
        candidate_calibration=0.91,
        candidate_top50_delta=0.02,
        candidate_worst_min_budget=84.0,
        candidate_worst_drawdown=31.0,
        control_worst_drawdown=25.0,
        candidate_budget_constrained=1,
        control_budget_constrained=0,
        season_deltas_vs_control={2020: 80.0, 2021: 50.0, 2022: 60.0, 2023: -40.0, 2024: 55.0, 2025: 90.0},
    )

    decision = _build_decision(experiment)

    assert decision["decision_status"] == "promote_candidate"
    assert decision["recommendation"] == "promote_ridge_ppg_xg_live_default"
    assert decision["gate_results"]["calibration_strict_pass"] is True
    assert decision["gate_results"]["budget_risk_pass"] is True


def test_point_winner_with_budget_failure_requires_budget_guardrail(tmp_path: Path) -> None:
    experiment = _write_experiment(
        tmp_path,
        candidate_calibration=0.9,
        candidate_worst_min_budget=68.0,
        candidate_worst_drawdown=55.0,
        control_worst_drawdown=30.0,
        candidate_budget_constrained=5,
        control_budget_constrained=0,
    )

    decision = _build_decision(experiment)

    assert decision["decision_status"] == "candidate_requires_budget_guardrail"
    assert decision["gate_results"]["budget_risk_pass"] is False
    assert decision["recommendation"] == "keep_xgboost_default_until_live_budget_risk_guardrails"


def test_point_winner_with_calibration_failure_requires_calibration_review(tmp_path: Path) -> None:
    experiment = _write_experiment(
        tmp_path,
        candidate_calibration=0.55,
        candidate_top50_delta=0.02,
        candidate_worst_min_budget=85.0,
        candidate_worst_drawdown=32.0,
        control_worst_drawdown=25.0,
    )

    decision = _build_decision(experiment)

    assert decision["decision_status"] == "candidate_requires_calibration_review"
    assert decision["gate_results"]["calibration_pass"] is False
    assert decision["recommendation"] == "keep_xgboost_default_until_ridge_calibration_review"


def test_weak_point_result_is_rejected(tmp_path: Path) -> None:
    experiment = _write_experiment(
        tmp_path,
        candidate_total=11_000.0,
        control_total=10_900.0,
        baseline_total=10_100.0,
        season_deltas_vs_control={2020: 20.0, 2021: -20.0, 2022: 25.0, 2023: -10.0, 2024: 15.0, 2025: 30.0},
    )

    decision = _build_decision(experiment)

    assert decision["decision_status"] == "rejected"
    assert decision["gate_results"]["candidate_beats_control_by_250"] is False
    assert decision["gate_results"]["candidate_beats_baseline_by_1000"] is False


def _build_decision(experiment: Path) -> dict[str, object]:
    return build_ridge_promotion_decision(
        experiment_path=experiment,
        candidate_model="ridge",
        candidate_feature_pack="ppg_xg",
        control_model="xgboost_depth2_l2_heavy",
        control_feature_pack="ppg_xg",
        baseline_model="random_forest",
        baseline_feature_pack="ppg",
        promotion_seasons=SEASONS,
    )


def _write_experiment(
    tmp_path: Path,
    *,
    comparability_status: str = "ok",
    omit_candidate_season: int | None = None,
    candidate_total: float = 12_000.0,
    control_total: float = 11_600.0,
    baseline_total: float = 10_800.0,
    candidate_rank: int = 1,
    candidate_calibration: float = 0.66,
    candidate_top50_delta: float = 0.01,
    candidate_worst_min_budget: float = 82.0,
    candidate_worst_drawdown: float = 35.0,
    control_worst_drawdown: float = 25.0,
    candidate_budget_constrained: int = 1,
    control_budget_constrained: int = 0,
    season_deltas_vs_control: dict[int, float] | None = None,
) -> Path:
    experiment = tmp_path / "experiment"
    experiment.mkdir()
    (experiment / "comparability_report.json").write_text(
        json.dumps({"status": comparability_status}, indent=2),
        encoding="utf-8",
    )
    deltas = season_deltas_vs_control or {
        2020: 90.0,
        2021: -10.0,
        2022: 80.0,
        2023: -60.0,
        2024: 70.0,
        2025: 110.0,
    }
    _ranked_frame(
        candidate_total=candidate_total,
        control_total=control_total,
        baseline_total=baseline_total,
        candidate_rank=candidate_rank,
        candidate_calibration=candidate_calibration,
        candidate_top50_delta=candidate_top50_delta,
        candidate_worst_min_budget=candidate_worst_min_budget,
        candidate_worst_drawdown=candidate_worst_drawdown,
        control_worst_drawdown=control_worst_drawdown,
        candidate_budget_constrained=candidate_budget_constrained,
        control_budget_constrained=control_budget_constrained,
    ).to_csv(experiment / "ranked_summary.csv", index=False)
    _season_frame(
        deltas,
        baseline_total=baseline_total,
        control_total=control_total,
        omit_candidate_season=omit_candidate_season,
    ).to_csv(experiment / "per_season_summary.csv", index=False)
    _prediction_metrics_frame(candidate_calibration).to_csv(experiment / "prediction_metrics.csv", index=False)
    return experiment


def _ranked_frame(
    *,
    candidate_total: float,
    control_total: float,
    baseline_total: float,
    candidate_rank: int,
    candidate_calibration: float,
    candidate_top50_delta: float,
    candidate_worst_min_budget: float,
    candidate_worst_drawdown: float,
    control_worst_drawdown: float,
    candidate_budget_constrained: int,
    control_budget_constrained: int,
) -> pd.DataFrame:
    rows = [
        _ranked_row(
            rank=candidate_rank,
            model_id="ridge",
            feature_pack="ppg_xg",
            total_actual_points=candidate_total,
            baseline_total=baseline_total,
            worst_min_budget=candidate_worst_min_budget,
            worst_drawdown=candidate_worst_drawdown,
            budget_constrained=candidate_budget_constrained,
            calibration=candidate_calibration,
            top50_delta=candidate_top50_delta,
        ),
        _ranked_row(
            rank=2,
            model_id="xgboost_depth2_l2_heavy",
            feature_pack="ppg_xg",
            total_actual_points=control_total,
            baseline_total=baseline_total,
            worst_min_budget=90.0,
            worst_drawdown=control_worst_drawdown,
            budget_constrained=control_budget_constrained,
            calibration=0.9,
            top50_delta=0.0,
        ),
        _ranked_row(
            rank=3,
            model_id="random_forest",
            feature_pack="ppg",
            total_actual_points=baseline_total,
            baseline_total=baseline_total,
            worst_min_budget=90.0,
            worst_drawdown=30.0,
            budget_constrained=0,
            calibration=0.8,
            top50_delta=0.0,
        ),
    ]
    return pd.DataFrame(rows)


def _ranked_row(
    *,
    rank: int,
    model_id: str,
    feature_pack: str,
    total_actual_points: float,
    baseline_total: float,
    worst_min_budget: float,
    worst_drawdown: float,
    budget_constrained: int,
    calibration: float,
    top50_delta: float,
) -> dict[str, object]:
    return {
        "rank": rank,
        "model_id": model_id,
        "feature_pack": feature_pack,
        "fixture_mode": "none",
        "budget_policy": "moving",
        "seasons_evaluated": 6,
        "total_rounds": 204,
        "total_actual_points": total_actual_points,
        "average_actual_points": total_actual_points / 204.0,
        "worst_min_budget": worst_min_budget,
        "worst_max_budget_drawdown": worst_drawdown,
        "total_budget_constrained_rounds": budget_constrained,
        "baseline_total_actual_points": baseline_total,
        "aggregate_delta": total_actual_points - baseline_total,
        "improved_seasons": 5,
        "worst_season_avg_delta": -0.1,
        "selected_calibration_slope": calibration,
        "top50_spearman_delta": top50_delta,
        "promotion_eligible": False,
        "promotion_reason": "synthetic",
    }


def _season_frame(
    deltas_vs_control: dict[int, float],
    *,
    baseline_total: float,
    control_total: float,
    omit_candidate_season: int | None,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    baseline_per_season = baseline_total / len(SEASONS)
    control_per_season = control_total / len(SEASONS)
    candidate_budget_by_season = {
        2020: (95.0, 95.0, 18.0, 0),
        2021: (102.0, 92.0, 20.0, 1),
        2022: (110.0, 96.0, 12.0, 0),
        2023: (98.0, 91.0, 22.0, 0),
        2024: (82.0, 82.0, 35.0, 0),
        2025: (105.0, 98.0, 15.0, 0),
    }
    for season in SEASONS:
        rows.append(
            _season_row(
                season,
                model_id="random_forest",
                feature_pack="ppg",
                total_actual_points=baseline_per_season,
                final_budget=100.0,
                min_budget=90.0,
                max_budget_drawdown=20.0,
                budget_constrained_rounds=0,
            )
        )
        rows.append(
            _season_row(
                season,
                model_id="xgboost_depth2_l2_heavy",
                feature_pack="ppg_xg",
                total_actual_points=control_per_season,
                final_budget=100.0,
                min_budget=90.0,
                max_budget_drawdown=25.0,
                budget_constrained_rounds=0,
            )
        )
        if season == omit_candidate_season:
            continue
        final_budget, min_budget, drawdown, constrained = candidate_budget_by_season[season]
        rows.append(
            _season_row(
                season,
                model_id="ridge",
                feature_pack="ppg_xg",
                total_actual_points=control_per_season + deltas_vs_control[season],
                final_budget=final_budget,
                min_budget=min_budget,
                max_budget_drawdown=drawdown,
                budget_constrained_rounds=constrained,
            )
        )
    return pd.DataFrame(rows)


def _season_row(
    season: int,
    *,
    model_id: str,
    feature_pack: str,
    total_actual_points: float,
    final_budget: float,
    min_budget: float,
    max_budget_drawdown: float,
    budget_constrained_rounds: int,
) -> dict[str, object]:
    return {
        "child_id": f"season={season}/model={model_id}/feature_pack={feature_pack}",
        "season": season,
        "model_id": model_id,
        "feature_pack": feature_pack,
        "fixture_mode": "none",
        "budget_policy": "moving",
        "strategy": model_id,
        "rounds": 34,
        "total_actual_points": total_actual_points,
        "average_actual_points": total_actual_points / 34.0,
        "initial_budget": 100.0,
        "final_budget": final_budget,
        "min_budget": min_budget,
        "max_budget_drawdown": max_budget_drawdown,
        "budget_constrained_rounds": budget_constrained_rounds,
    }


def _prediction_metrics_frame(candidate_calibration: float) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "season": season,
                "model_id": "ridge",
                "feature_pack": "ppg_xg",
                "fixture_mode": "none",
                "budget_policy": "moving",
                "metric_scope": "selected_players",
                "observed_count": 408,
                "spearman": 0.1,
                "calibration_slope": candidate_calibration,
            }
            for season in SEASONS
        ]
    )
