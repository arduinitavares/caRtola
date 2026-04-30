import math

import pandas as pd
import pytest

from cartola.backtesting.experiment_metrics import (
    calibration_slope_intercept,
    promotion_status,
    top_k_rows_by_round,
)


def test_top_k_rows_are_selected_per_round() -> None:
    frame = pd.DataFrame(
        {
            "rodada": [1, 1, 1, 2, 2, 2],
            "id_atleta": [1, 2, 3, 4, 5, 6],
            "model_score": [1, 3, 2, 10, 8, 9],
        }
    )

    selected = top_k_rows_by_round(frame, score_column="model_score", k=2)

    assert selected["id_atleta"].to_list() == [2, 3, 4, 6]


def test_top_k_rejects_invalid_k() -> None:
    frame = pd.DataFrame({"rodada": [1], "model_score": [1.0]})

    with pytest.raises(ValueError, match="k must be at least 1"):
        top_k_rows_by_round(frame, score_column="model_score", k=0)


def test_calibration_slope_and_intercept() -> None:
    result = calibration_slope_intercept(
        predicted=pd.Series([1, 2, 3, 4]),
        actual=pd.Series([2, 4, 6, 8]),
    )

    assert result == {
        "calibration_intercept": 0.0,
        "calibration_slope": 2.0,
        "warning": None,
    }


def test_calibration_returns_null_for_constant_predictions() -> None:
    result = calibration_slope_intercept(
        predicted=pd.Series([1, 1, 1]),
        actual=pd.Series([2, 3, 4]),
    )

    assert result == {
        "calibration_intercept": None,
        "calibration_slope": None,
        "warning": "constant_prediction",
    }


def test_promotion_status_passes_when_all_guardrails_pass() -> None:
    result = promotion_status(
        aggregate_delta=1.0,
        improved_seasons=2,
        worst_season_avg_delta=-1.5,
        selected_calibration_slope=1.0,
        top50_spearman_delta=-0.03,
        comparable=True,
    )

    assert result == {"eligible": True, "reason": "passes_v1_guardrails"}


def test_promotion_status_fails_null_guardrail() -> None:
    result = promotion_status(
        aggregate_delta=None,
        improved_seasons=2,
        worst_season_avg_delta=-1.5,
        selected_calibration_slope=1.0,
        top50_spearman_delta=-0.03,
        comparable=True,
    )

    assert result == {"eligible": False, "reason": "insufficient_metric_data"}


def test_promotion_status_fails_nan_guardrail() -> None:
    result = promotion_status(
        aggregate_delta=math.nan,
        improved_seasons=2,
        worst_season_avg_delta=-1.5,
        selected_calibration_slope=1.0,
        top50_spearman_delta=-0.03,
        comparable=True,
    )

    assert result == {"eligible": False, "reason": "insufficient_metric_data"}


def test_promotion_status_fails_aggregate_only_win() -> None:
    result = promotion_status(
        aggregate_delta=1.0,
        improved_seasons=1,
        worst_season_avg_delta=-1.5,
        selected_calibration_slope=1.0,
        top50_spearman_delta=-0.03,
        comparable=True,
    )

    assert result == {
        "eligible": False,
        "reason": "fewer_than_two_seasons_improved",
    }


def test_promotion_status_fails_worst_season_regression() -> None:
    result = promotion_status(
        aggregate_delta=1.0,
        improved_seasons=2,
        worst_season_avg_delta=-1.51,
        selected_calibration_slope=1.0,
        top50_spearman_delta=-0.03,
        comparable=True,
    )

    assert result == {
        "eligible": False,
        "reason": "worst_season_regression_exceeds_threshold",
    }
