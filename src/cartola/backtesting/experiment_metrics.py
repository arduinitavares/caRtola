from __future__ import annotations

import pandas as pd


def top_k_rows_by_round(frame: pd.DataFrame, *, score_column: str, k: int) -> pd.DataFrame:
    if k < 1:
        msg = "k must be at least 1"
        raise ValueError(msg)

    if frame.empty:
        return frame.copy()

    sorted_frame = frame.sort_values(
        by=["rodada", score_column],
        ascending=[True, False],
        kind="mergesort",
    )
    return sorted_frame.groupby("rodada", sort=False).head(k).reset_index(drop=True)


def calibration_slope_intercept(predicted: pd.Series, actual: pd.Series) -> dict[str, float | str | None]:
    paired = pd.DataFrame({"predicted": predicted, "actual": actual}).dropna()
    if paired.empty:
        return {
            "calibration_intercept": None,
            "calibration_slope": None,
            "warning": "empty_input",
        }

    predicted_values = paired["predicted"].astype(float)
    actual_values = paired["actual"].astype(float)
    if predicted_values.nunique() == 1:
        return {
            "calibration_intercept": None,
            "calibration_slope": None,
            "warning": "constant_prediction",
        }

    predicted_mean = predicted_values.mean()
    actual_mean = actual_values.mean()
    slope = ((predicted_values - predicted_mean) * (actual_values - actual_mean)).sum() / (
        (predicted_values - predicted_mean) ** 2
    ).sum()
    intercept = actual_mean - (slope * predicted_mean)

    return {
        "calibration_intercept": round(float(intercept), 10),
        "calibration_slope": round(float(slope), 10),
        "warning": None,
    }


def promotion_status(
    *,
    aggregate_delta: float | None,
    improved_seasons: int | None,
    worst_season_avg_delta: float | None,
    selected_calibration_slope: float | None,
    top50_spearman_delta: float | None,
    comparable: bool,
) -> dict[str, object]:
    if not comparable:
        return {"eligible": False, "reason": "not_comparable"}

    guardrails = (
        aggregate_delta,
        improved_seasons,
        worst_season_avg_delta,
        selected_calibration_slope,
        top50_spearman_delta,
    )
    if any(_is_missing(value) for value in guardrails):
        return {"eligible": False, "reason": "insufficient_metric_data"}

    assert aggregate_delta is not None
    assert improved_seasons is not None
    assert worst_season_avg_delta is not None
    assert selected_calibration_slope is not None
    assert top50_spearman_delta is not None

    if aggregate_delta <= 0:
        return {"eligible": False, "reason": "aggregate_delta_not_positive"}
    if improved_seasons < 2:
        return {"eligible": False, "reason": "fewer_than_two_seasons_improved"}
    if worst_season_avg_delta < -1.5:
        return {"eligible": False, "reason": "worst_season_regression_exceeds_threshold"}
    if selected_calibration_slope < 0.75 or selected_calibration_slope > 1.25:
        return {"eligible": False, "reason": "selected_calibration_slope_out_of_range"}
    if top50_spearman_delta < -0.03:
        return {"eligible": False, "reason": "top50_spearman_regression_exceeds_threshold"}
    return {"eligible": True, "reason": "passes_v1_guardrails"}


def _is_missing(value: object) -> bool:
    if value is None:
        return True

    missing = pd.isna(value)
    if isinstance(missing, bool):
        return missing

    return False
