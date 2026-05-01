import math

import pandas as pd
import pytest

from cartola.backtesting.ridge_tuning_metrics import promotion_decision, rank_tuning_summary


def _passing_promotion_inputs(**overrides: object) -> dict[str, object]:
    values: dict[str, object] = {
        "comparable": True,
        "final_reproducible": True,
        "aggregate_delta_vs_primary_incumbent": 51.0,
        "total_rounds": 102,
        "improved_seasons_vs_primary_incumbent": 2,
        "worst_season_avg_delta_vs_primary_incumbent": -0.5,
        "selected_calibration_slope": 1.0,
        "top50_spearman_delta_vs_primary_incumbent": -0.03,
        "candidate_pool_mae_delta_pct_vs_primary_incumbent": 0.05,
        "selected_players_mae_delta_pct_vs_primary_incumbent": 0.05,
    }
    values.update(overrides)
    return values


def test_promotion_decision_rejects_lift_below_practical_threshold() -> None:
    result = promotion_decision(
        **_passing_promotion_inputs(aggregate_delta_vs_primary_incumbent=50.99),
    )

    assert result == {
        "eligible": False,
        "reason": "lift_below_practical_threshold",
    }


def test_promotion_decision_accepts_lift_at_practical_threshold() -> None:
    result = promotion_decision(**_passing_promotion_inputs())

    assert result == {"eligible": True, "reason": "passes_tuning_guardrails"}


def test_promotion_decision_rejects_null_required_metric() -> None:
    result = promotion_decision(
        **_passing_promotion_inputs(selected_calibration_slope=math.nan),
    )

    assert result == {"eligible": False, "reason": "insufficient_metric_data"}


def test_rank_tuning_summary_keeps_alpha_candidates_distinct() -> None:
    per_season_summary = pd.DataFrame(
        [
            _season_row("ridge_alpha_1_0__ppg_xg", 2023, alpha=1.0, actual=100.0, predicted=105.0),
            _season_row("ridge_alpha_1_0__ppg_xg", 2024, alpha=1.0, actual=100.0, predicted=105.0),
            _season_row("ridge_alpha_3_0__ppg_xg", 2023, alpha=3.0, actual=126.0, predicted=126.0),
            _season_row("ridge_alpha_3_0__ppg_xg", 2024, alpha=3.0, actual=126.0, predicted=126.0),
            _season_row("ridge_alpha_10_0__ppg_xg", 2023, alpha=10.0, actual=127.0, predicted=127.0),
            _season_row("ridge_alpha_10_0__ppg_xg", 2024, alpha=10.0, actual=127.0, predicted=127.0),
        ]
    )
    prediction_metrics = pd.DataFrame(
        [
            *_metric_rows("ridge_alpha_1_0__ppg_xg", alpha=1.0, mae=10.0, spearman=0.8),
            *_metric_rows("ridge_alpha_3_0__ppg_xg", alpha=3.0, mae=10.0, spearman=0.8),
            *_metric_rows("ridge_alpha_10_0__ppg_xg", alpha=10.0, mae=10.0, spearman=0.8),
        ]
    )

    ranked = rank_tuning_summary(
        per_season_summary,
        prediction_metrics,
        primary_incumbent_candidate_id="ridge_alpha_1_0__ppg_xg",
        final_reproducibility_by_candidate={
            "ridge_alpha_1_0__ppg_xg": True,
            "ridge_alpha_3_0__ppg_xg": True,
            "ridge_alpha_10_0__ppg_xg": True,
        },
    )

    assert set(ranked["candidate_id"]) == {
        "ridge_alpha_1_0__ppg_xg",
        "ridge_alpha_3_0__ppg_xg",
        "ridge_alpha_10_0__ppg_xg",
    }
    ridge_rows = ranked[ranked["feature_pack"].eq("ppg_xg") & ranked["model_id"].eq("ridge")]
    assert ridge_rows["alpha"].to_list() == [10.0, 3.0, 1.0]
    assert ridge_rows["aggregate_delta_vs_primary_incumbent"].to_list() == [54.0, 52.0, 0.0]


def test_rank_tuning_summary_mae_regression_percentage_rejects_candidate() -> None:
    per_season_summary = pd.DataFrame(
        [
            _season_row("ridge_alpha_1_0__ppg_xg", 2023, alpha=1.0, actual=100.0, predicted=105.0),
            _season_row("ridge_alpha_1_0__ppg_xg", 2024, alpha=1.0, actual=100.0, predicted=105.0),
            _season_row("ridge_alpha_3_0__ppg_xg", 2023, alpha=3.0, actual=126.0, predicted=126.0),
            _season_row("ridge_alpha_3_0__ppg_xg", 2024, alpha=3.0, actual=126.0, predicted=126.0),
        ]
    )
    prediction_metrics = pd.DataFrame(
        [
            *_metric_rows_for_seasons("ridge_alpha_1_0__ppg_xg", alpha=1.0, mae=10.0, spearman=0.8, seasons=(2023, 2024)),
            *_metric_rows_for_seasons("ridge_alpha_3_0__ppg_xg", alpha=3.0, mae=10.6, spearman=0.8, seasons=(2023, 2024)),
        ]
    )

    ranked = rank_tuning_summary(
        per_season_summary,
        prediction_metrics,
        primary_incumbent_candidate_id="ridge_alpha_1_0__ppg_xg",
        final_reproducibility_by_candidate={
            "ridge_alpha_1_0__ppg_xg": True,
            "ridge_alpha_3_0__ppg_xg": True,
        },
    )

    candidate = ranked.loc[ranked["candidate_id"].eq("ridge_alpha_3_0__ppg_xg")].iloc[0]
    assert candidate["candidate_pool_mae_delta_pct_vs_primary_incumbent"] == 0.06
    assert candidate["selected_players_mae_delta_pct_vs_primary_incumbent"] == 0.06
    assert candidate["promotion_eligible"] is False
    assert candidate["promotion_reason"] == "candidate_pool_mae_regression_exceeds_threshold"


def test_rank_tuning_summary_missing_incumbent_fails_without_crashing() -> None:
    per_season_summary = pd.DataFrame(
        [
            _season_row("ridge_alpha_3_0__ppg_xg", 2023, alpha=3.0, actual=126.0, predicted=126.0),
            _season_row("ridge_alpha_3_0__ppg_xg", 2024, alpha=3.0, actual=126.0, predicted=126.0),
        ]
    )
    prediction_metrics = pd.DataFrame([*_metric_rows("ridge_alpha_3_0__ppg_xg", alpha=3.0, mae=10.0, spearman=0.8)])

    ranked = rank_tuning_summary(
        per_season_summary,
        prediction_metrics,
        primary_incumbent_candidate_id="ridge_alpha_1_0__ppg_xg",
        final_reproducibility_by_candidate={"ridge_alpha_3_0__ppg_xg": True},
    )

    candidate = ranked.iloc[0]
    assert pd.isna(candidate["aggregate_delta_vs_primary_incumbent"])
    assert candidate["promotion_eligible"] is False
    assert candidate["promotion_reason"] == "insufficient_metric_data"


def test_rank_tuning_summary_rejects_duplicate_candidate_season_rows() -> None:
    per_season_summary = pd.DataFrame(
        [
            _season_row("ridge_alpha_1_0__ppg_xg", 2023, alpha=1.0, actual=100.0, predicted=105.0),
            _season_row("ridge_alpha_1_0__ppg_xg", 2023, alpha=1.0, actual=101.0, predicted=106.0),
        ]
    )

    with pytest.raises(ValueError, match="Duplicate per-season summary rows"):
        rank_tuning_summary(
            per_season_summary,
            pd.DataFrame(),
            primary_incumbent_candidate_id="ridge_alpha_1_0__ppg_xg",
            final_reproducibility_by_candidate={"ridge_alpha_1_0__ppg_xg": True},
        )


def test_rank_tuning_summary_candidate_missing_incumbent_season_is_not_comparable() -> None:
    per_season_summary = pd.DataFrame(
        [
            _season_row("ridge_alpha_1_0__ppg_xg", 2023, alpha=1.0, actual=100.0, predicted=105.0),
            _season_row("ridge_alpha_1_0__ppg_xg", 2024, alpha=1.0, actual=100.0, predicted=105.0),
            _season_row("ridge_alpha_3_0__ppg_xg", 2023, alpha=3.0, actual=152.0, predicted=152.0),
        ]
    )
    prediction_metrics = pd.DataFrame(
        [
            *_metric_rows_for_seasons("ridge_alpha_1_0__ppg_xg", alpha=1.0, mae=10.0, spearman=0.8, seasons=(2023, 2024)),
            *_metric_rows_for_seasons("ridge_alpha_3_0__ppg_xg", alpha=3.0, mae=10.0, spearman=0.8, seasons=(2023,)),
        ]
    )

    ranked = rank_tuning_summary(
        per_season_summary,
        prediction_metrics,
        primary_incumbent_candidate_id="ridge_alpha_1_0__ppg_xg",
        final_reproducibility_by_candidate={
            "ridge_alpha_1_0__ppg_xg": True,
            "ridge_alpha_3_0__ppg_xg": True,
        },
    )

    candidate = ranked.loc[ranked["candidate_id"].eq("ridge_alpha_3_0__ppg_xg")].iloc[0]
    assert candidate["promotion_eligible"] is False
    assert candidate["promotion_reason"] == "not_comparable"


def test_rank_tuning_summary_rejects_duplicate_candidate_season_metric_scope_rows() -> None:
    per_season_summary = pd.DataFrame(
        [
            _season_row("ridge_alpha_1_0__ppg_xg", 2023, alpha=1.0, actual=100.0, predicted=105.0),
            _season_row("ridge_alpha_3_0__ppg_xg", 2023, alpha=3.0, actual=152.0, predicted=152.0),
        ]
    )
    duplicate_metric = _metric_rows_for_seasons(
        "ridge_alpha_1_0__ppg_xg",
        alpha=1.0,
        mae=10.0,
        spearman=0.8,
        seasons=(2023,),
    )
    prediction_metrics = pd.DataFrame(
        [
            *duplicate_metric,
            duplicate_metric[0],
            *_metric_rows_for_seasons("ridge_alpha_3_0__ppg_xg", alpha=3.0, mae=10.0, spearman=0.8, seasons=(2023,)),
        ]
    )

    with pytest.raises(ValueError, match="Duplicate prediction metric rows"):
        rank_tuning_summary(
            per_season_summary,
            prediction_metrics,
            primary_incumbent_candidate_id="ridge_alpha_1_0__ppg_xg",
            final_reproducibility_by_candidate={
                "ridge_alpha_1_0__ppg_xg": True,
                "ridge_alpha_3_0__ppg_xg": True,
            },
        )


def test_rank_tuning_summary_missing_required_metric_for_incumbent_season_is_insufficient() -> None:
    per_season_summary = pd.DataFrame(
        [
            _season_row("ridge_alpha_1_0__ppg_xg", 2023, alpha=1.0, actual=100.0, predicted=105.0),
            _season_row("ridge_alpha_1_0__ppg_xg", 2024, alpha=1.0, actual=100.0, predicted=105.0),
            _season_row("ridge_alpha_3_0__ppg_xg", 2023, alpha=3.0, actual=126.0, predicted=126.0),
            _season_row("ridge_alpha_3_0__ppg_xg", 2024, alpha=3.0, actual=126.0, predicted=126.0),
        ]
    )
    candidate_metrics = _metric_rows_for_seasons(
        "ridge_alpha_3_0__ppg_xg",
        alpha=3.0,
        mae=10.0,
        spearman=0.8,
        seasons=(2023, 2024),
    )
    candidate_metrics = [
        metric
        for metric in candidate_metrics
        if not (metric["season"] == 2024 and metric["metric_scope"] == "selected_players")
    ]
    prediction_metrics = pd.DataFrame(
        [
            *_metric_rows_for_seasons("ridge_alpha_1_0__ppg_xg", alpha=1.0, mae=10.0, spearman=0.8, seasons=(2023, 2024)),
            *candidate_metrics,
        ]
    )

    ranked = rank_tuning_summary(
        per_season_summary,
        prediction_metrics,
        primary_incumbent_candidate_id="ridge_alpha_1_0__ppg_xg",
        final_reproducibility_by_candidate={
            "ridge_alpha_1_0__ppg_xg": True,
            "ridge_alpha_3_0__ppg_xg": True,
        },
    )

    candidate = ranked.loc[ranked["candidate_id"].eq("ridge_alpha_3_0__ppg_xg")].iloc[0]
    assert candidate["promotion_eligible"] is False
    assert candidate["promotion_reason"] == "insufficient_metric_data"


def _season_row(candidate_id: str, season: int, *, alpha: float, actual: float, predicted: float) -> dict[str, object]:
    return {
        "candidate_id": candidate_id,
        "model_id": "ridge",
        "feature_pack": "ppg_xg",
        "alpha": alpha,
        "season": season,
        "rounds": 51,
        "total_actual_points": actual,
        "total_predicted_points": predicted,
    }


def _metric_rows(candidate_id: str, *, alpha: float, mae: float, spearman: float) -> list[dict[str, object]]:
    return _metric_rows_for_seasons(candidate_id, alpha=alpha, mae=mae, spearman=spearman, seasons=(2023,))


def _metric_rows_for_seasons(
    candidate_id: str,
    *,
    alpha: float,
    mae: float,
    spearman: float,
    seasons: tuple[int, ...],
) -> list[dict[str, object]]:
    rows = []
    for season in seasons:
        rows.extend(_metric_rows_for_season(candidate_id, alpha=alpha, mae=mae, spearman=spearman, season=season))
    return rows


def _metric_rows_for_season(
    candidate_id: str,
    *,
    alpha: float,
    mae: float,
    spearman: float,
    season: int,
) -> list[dict[str, object]]:
    common = {
        "candidate_id": candidate_id,
        "model_id": "ridge",
        "feature_pack": "ppg_xg",
        "alpha": alpha,
        "season": season,
    }
    return [
        {
            **common,
            "metric_scope": "selected_players",
            "mae": mae,
            "calibration_slope": 1.0,
            "spearman": 0.7,
        },
        {
            **common,
            "metric_scope": "candidate_pool",
            "mae": mae,
            "calibration_slope": None,
            "spearman": 0.7,
        },
        {
            **common,
            "metric_scope": "top50_candidates",
            "mae": mae,
            "calibration_slope": None,
            "spearman": spearman,
        },
    ]
