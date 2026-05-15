from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from cartola.backtesting.xgboost_optuna_tuning import (
    TrialAggregateMetrics,
    balanced_objective_score,
    suggest_xgboost_parameters,
    summarize_trial_results,
)


class FakeTrial:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def suggest_int(self, name: str, low: int, high: int, **kwargs: object) -> int:
        self.calls.append(("int", name))
        if name == "max_depth":
            return 2
        return low + ((high - low) // 2)

    def suggest_float(self, name: str, low: float, high: float, **kwargs: object) -> float:
        self.calls.append(("float", name))
        return round((low + high) / 2, 5)


def test_suggest_xgboost_parameters_uses_bounded_cartola_search_space() -> None:
    trial = FakeTrial()

    params = suggest_xgboost_parameters(trial)

    assert params == {
        "n_estimators": 350,
        "max_depth": 2,
        "learning_rate": 0.045,
        "min_child_weight": 10.5,
        "subsample": 0.825,
        "colsample_bytree": 0.825,
        "reg_lambda": 102.5,
        "reg_alpha": 10.0,
        "gamma": 5.0,
    }
    assert "scale_pos_weight" not in params
    assert ("int", "max_depth") in trial.calls
    assert ("float", "reg_lambda") in trial.calls


def test_balanced_objective_penalizes_budget_risk_enough_to_prefer_safer_candidate() -> None:
    control = TrialAggregateMetrics(
        total_actual_points=11_200.0,
        total_rounds=204,
        worst_min_budget=73.0,
        worst_max_budget_drawdown=32.0,
        total_budget_constrained_rounds=0,
        selected_calibration_slope=0.9,
        season_actual_points={2025: 1_900.0},
    )
    safe = TrialAggregateMetrics(
        total_actual_points=11_650.0,
        total_rounds=204,
        worst_min_budget=78.0,
        worst_max_budget_drawdown=40.0,
        total_budget_constrained_rounds=1,
        selected_calibration_slope=0.95,
        season_actual_points={2025: 2_010.0},
    )
    risky = TrialAggregateMetrics(
        total_actual_points=11_730.0,
        total_rounds=204,
        worst_min_budget=60.0,
        worst_max_budget_drawdown=62.0,
        total_budget_constrained_rounds=5,
        selected_calibration_slope=0.55,
        season_actual_points={2025: 1_910.0},
    )

    assert balanced_objective_score(safe, control) > balanced_objective_score(risky, control)


def test_summarize_trial_results_aggregates_season_budget_and_calibration() -> None:
    results = {
        2024: _result(
            strategy="xgboost_depth2_l2_heavy",
            total_actual_points=100.0,
            total_predicted_points=110.0,
            min_budget=82.0,
            max_budget_drawdown=18.0,
            budget_constrained_rounds=0,
            predicted=[5.0, 6.0],
            actual=[4.0, 7.0],
        ),
        2025: _result(
            strategy="xgboost_depth2_l2_heavy",
            total_actual_points=120.0,
            total_predicted_points=115.0,
            min_budget=76.0,
            max_budget_drawdown=24.0,
            budget_constrained_rounds=1,
            predicted=[3.0, 8.0],
            actual=[2.0, 9.0],
        ),
    }

    summary = summarize_trial_results(
        results,
        model_id="xgboost_depth2_l2_heavy",
        model_params={"max_depth": 2},
        output_path=Path("trial"),
    )

    assert summary.total_actual_points == 220.0
    assert summary.total_rounds == 4
    assert summary.worst_min_budget == 76.0
    assert summary.worst_max_budget_drawdown == 24.0
    assert summary.total_budget_constrained_rounds == 1
    assert summary.season_actual_points == {2024: 100.0, 2025: 120.0}
    assert summary.selected_calibration_slope is not None


def _result(
    *,
    strategy: str,
    total_actual_points: float,
    total_predicted_points: float,
    min_budget: float,
    max_budget_drawdown: float,
    budget_constrained_rounds: int,
    predicted: list[float],
    actual: list[float],
) -> SimpleNamespace:
    return SimpleNamespace(
        summary=pd.DataFrame(
            [
                {
                    "strategy": strategy,
                    "rounds": len(predicted),
                    "total_actual_points": total_actual_points,
                    "total_predicted_points": total_predicted_points,
                    "min_budget": min_budget,
                    "max_budget_drawdown": max_budget_drawdown,
                    "budget_constrained_rounds": budget_constrained_rounds,
                }
            ]
        ),
        selected_players=pd.DataFrame(
            {
                "predicted_points": predicted,
                "pontuacao": actual,
            }
        ),
    )
