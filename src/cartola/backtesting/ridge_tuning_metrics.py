from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import cast

import pandas as pd

PRACTICAL_LIFT_PER_ROUND = 0.5
_REQUIRED_PROMOTION_METRIC_SCOPES = frozenset({"candidate_pool", "selected_players", "top50_candidates"})

_RANKED_TUNING_SUMMARY_COLUMNS = [
    "candidate_id",
    "model_id",
    "feature_pack",
    "alpha",
    "seasons_evaluated",
    "total_rounds",
    "total_actual_points",
    "average_actual_points",
    "total_predicted_points",
    "average_predicted_points",
    "primary_incumbent_total_actual_points",
    "aggregate_delta_vs_primary_incumbent",
    "average_delta_per_round_vs_primary_incumbent",
    "improved_seasons_vs_primary_incumbent",
    "worst_season_avg_delta_vs_primary_incumbent",
    "selected_calibration_slope",
    "top50_spearman_delta_vs_primary_incumbent",
    "candidate_pool_mae_delta_pct_vs_primary_incumbent",
    "selected_players_mae_delta_pct_vs_primary_incumbent",
    "promotion_eligible",
    "promotion_reason",
]


def promotion_decision(
    *,
    comparable: bool,
    final_reproducible: bool,
    aggregate_delta_vs_primary_incumbent: float | None,
    total_rounds: int | None,
    improved_seasons_vs_primary_incumbent: int | None,
    worst_season_avg_delta_vs_primary_incumbent: float | None,
    selected_calibration_slope: float | None,
    top50_spearman_delta_vs_primary_incumbent: float | None,
    candidate_pool_mae_delta_pct_vs_primary_incumbent: float | None,
    selected_players_mae_delta_pct_vs_primary_incumbent: float | None,
) -> dict[str, object]:
    if not comparable:
        return {"eligible": False, "reason": "not_comparable"}
    if not final_reproducible:
        return {"eligible": False, "reason": "non_reproducible"}

    guardrails = (
        aggregate_delta_vs_primary_incumbent,
        total_rounds,
        improved_seasons_vs_primary_incumbent,
        worst_season_avg_delta_vs_primary_incumbent,
        selected_calibration_slope,
        top50_spearman_delta_vs_primary_incumbent,
        candidate_pool_mae_delta_pct_vs_primary_incumbent,
        selected_players_mae_delta_pct_vs_primary_incumbent,
    )
    if any(_is_missing(value) for value in guardrails):
        return {"eligible": False, "reason": "insufficient_metric_data"}

    aggregate_delta = cast("float", aggregate_delta_vs_primary_incumbent)
    rounds = cast("int", total_rounds)
    improved_seasons = cast("int", improved_seasons_vs_primary_incumbent)
    worst_season_avg_delta = cast("float", worst_season_avg_delta_vs_primary_incumbent)
    selected_calibration_slope_value = cast("float", selected_calibration_slope)
    top50_spearman_delta = cast("float", top50_spearman_delta_vs_primary_incumbent)
    candidate_pool_mae_delta_pct = cast("float", candidate_pool_mae_delta_pct_vs_primary_incumbent)
    selected_players_mae_delta_pct = cast("float", selected_players_mae_delta_pct_vs_primary_incumbent)

    required_aggregate_lift = PRACTICAL_LIFT_PER_ROUND * rounds
    if aggregate_delta < required_aggregate_lift:
        return {"eligible": False, "reason": "lift_below_practical_threshold"}
    if improved_seasons < 2:
        return {"eligible": False, "reason": "fewer_than_two_seasons_improved"}
    if worst_season_avg_delta < -0.5:
        return {"eligible": False, "reason": "worst_season_regression_exceeds_threshold"}
    if selected_calibration_slope_value < 0.75 or selected_calibration_slope_value > 1.25:
        return {"eligible": False, "reason": "selected_calibration_slope_out_of_range"}
    if top50_spearman_delta < -0.03:
        return {"eligible": False, "reason": "top50_spearman_regression_exceeds_threshold"}
    if candidate_pool_mae_delta_pct > 0.05:
        return {"eligible": False, "reason": "candidate_pool_mae_regression_exceeds_threshold"}
    if selected_players_mae_delta_pct > 0.05:
        return {"eligible": False, "reason": "selected_players_mae_regression_exceeds_threshold"}

    return {"eligible": True, "reason": "passes_tuning_guardrails"}


def rank_tuning_summary(
    per_season_summary: pd.DataFrame,
    prediction_metrics: pd.DataFrame,
    *,
    primary_incumbent_candidate_id: str,
    final_reproducibility_by_candidate: Mapping[str, bool],
) -> pd.DataFrame:
    if per_season_summary.empty:
        ranked = pd.DataFrame(columns=pd.Index(_RANKED_TUNING_SUMMARY_COLUMNS))
        ranked.insert(0, "rank", pd.Series(dtype="int64"))
        return ranked

    _validate_per_season_summary(per_season_summary)
    _validate_prediction_metrics(prediction_metrics)
    incumbent_by_season = _incumbent_actual_points_by_season(
        per_season_summary,
        incumbent_candidate_id=primary_incumbent_candidate_id,
    )
    incumbent_seasons = set(incumbent_by_season)
    primary_incumbent_total_actual_points = _sum_or_none(list(incumbent_by_season.values()))
    incumbent_selected_players_mae = _season_aligned_mean_metric(
        prediction_metrics,
        candidate_id=primary_incumbent_candidate_id,
        metric_scope="selected_players",
        metric_column="mae",
        required_seasons=incumbent_seasons,
    )
    incumbent_candidate_pool_mae = _season_aligned_mean_metric(
        prediction_metrics,
        candidate_id=primary_incumbent_candidate_id,
        metric_scope="candidate_pool",
        metric_column="mae",
        required_seasons=incumbent_seasons,
    )
    incumbent_top50_spearman = _season_aligned_mean_metric(
        prediction_metrics,
        candidate_id=primary_incumbent_candidate_id,
        metric_scope="top50_candidates",
        metric_column="spearman",
        required_seasons=incumbent_seasons,
    )

    rows = [
        _aggregate_tuning_summary_row(
            group_frame,
            prediction_metrics=prediction_metrics,
            incumbent_by_season=incumbent_by_season,
            incumbent_seasons=incumbent_seasons,
            primary_incumbent_total_actual_points=primary_incumbent_total_actual_points,
            incumbent_selected_players_mae=incumbent_selected_players_mae,
            incumbent_candidate_pool_mae=incumbent_candidate_pool_mae,
            incumbent_top50_spearman=incumbent_top50_spearman,
            final_reproducibility_by_candidate=final_reproducibility_by_candidate,
        )
        for _group_key, group_frame in per_season_summary.groupby(
            ["candidate_id", "model_id", "feature_pack", "alpha"],
            sort=False,
            dropna=False,
        )
    ]
    ranked = pd.DataFrame(rows)
    ranked = ranked.sort_values(
        by=[
            "promotion_eligible",
            "aggregate_delta_vs_primary_incumbent",
            "total_actual_points",
            "candidate_id",
        ],
        ascending=[False, False, False, True],
        na_position="last",
        kind="mergesort",
    ).reset_index(drop=True)
    ranked.insert(0, "rank", pd.Series(range(1, len(ranked) + 1), dtype="int64"))
    ranked["promotion_eligible"] = ranked["promotion_eligible"].astype(object)
    return ranked.loc[:, ["rank", *_RANKED_TUNING_SUMMARY_COLUMNS]]


def _aggregate_tuning_summary_row(
    group_frame: pd.DataFrame,
    *,
    prediction_metrics: pd.DataFrame,
    incumbent_by_season: Mapping[int, float],
    incumbent_seasons: set[int],
    primary_incumbent_total_actual_points: float | None,
    incumbent_selected_players_mae: float | None,
    incumbent_candidate_pool_mae: float | None,
    incumbent_top50_spearman: float | None,
    final_reproducibility_by_candidate: Mapping[str, bool],
) -> dict[str, object]:
    first = group_frame.iloc[0]
    candidate_id = str(first["candidate_id"])
    candidate_seasons = set(group_frame["season"].astype(int))
    comparable = True if not incumbent_seasons else candidate_seasons == incumbent_seasons
    total_rounds = int(group_frame["rounds"].sum())
    total_actual_points = float(group_frame["total_actual_points"].sum())
    total_predicted_points = float(group_frame["total_predicted_points"].sum())
    season_deltas = _season_deltas(group_frame, incumbent_by_season=incumbent_by_season)
    aggregate_delta = _sum_or_none([delta for delta, _rounds in season_deltas])
    average_delta_per_round = None if aggregate_delta is None or total_rounds == 0 else aggregate_delta / total_rounds
    season_average_deltas = [delta / rounds for delta, rounds in season_deltas if rounds > 0]
    worst_season_avg_delta = min(season_average_deltas) if season_average_deltas else None
    selected_calibration_slope = _season_aligned_mean_metric(
        prediction_metrics,
        candidate_id=candidate_id,
        metric_scope="selected_players",
        metric_column="calibration_slope",
        required_seasons=incumbent_seasons,
    )
    top50_spearman = _season_aligned_mean_metric(
        prediction_metrics,
        candidate_id=candidate_id,
        metric_scope="top50_candidates",
        metric_column="spearman",
        required_seasons=incumbent_seasons,
    )
    selected_players_mae = _season_aligned_mean_metric(
        prediction_metrics,
        candidate_id=candidate_id,
        metric_scope="selected_players",
        metric_column="mae",
        required_seasons=incumbent_seasons,
    )
    candidate_pool_mae = _season_aligned_mean_metric(
        prediction_metrics,
        candidate_id=candidate_id,
        metric_scope="candidate_pool",
        metric_column="mae",
        required_seasons=incumbent_seasons,
    )
    top50_spearman_delta = (
        None if top50_spearman is None or incumbent_top50_spearman is None else top50_spearman - incumbent_top50_spearman
    )
    candidate_pool_mae_delta_pct = _percent_delta(candidate_pool_mae, incumbent_candidate_pool_mae)
    selected_players_mae_delta_pct = _percent_delta(selected_players_mae, incumbent_selected_players_mae)
    promotion = promotion_decision(
        comparable=comparable,
        final_reproducible=final_reproducibility_by_candidate.get(candidate_id, False),
        aggregate_delta_vs_primary_incumbent=aggregate_delta,
        total_rounds=total_rounds,
        improved_seasons_vs_primary_incumbent=None
        if aggregate_delta is None
        else sum(1 for delta, _rounds in season_deltas if delta > 0),
        worst_season_avg_delta_vs_primary_incumbent=worst_season_avg_delta,
        selected_calibration_slope=selected_calibration_slope,
        top50_spearman_delta_vs_primary_incumbent=top50_spearman_delta,
        candidate_pool_mae_delta_pct_vs_primary_incumbent=candidate_pool_mae_delta_pct,
        selected_players_mae_delta_pct_vs_primary_incumbent=selected_players_mae_delta_pct,
    )

    return {
        "candidate_id": candidate_id,
        "model_id": str(first["model_id"]),
        "feature_pack": str(first["feature_pack"]),
        "alpha": float(first["alpha"]),
        "seasons_evaluated": int(group_frame["season"].nunique()),
        "total_rounds": total_rounds,
        "total_actual_points": total_actual_points,
        "average_actual_points": None if total_rounds == 0 else total_actual_points / total_rounds,
        "total_predicted_points": total_predicted_points,
        "average_predicted_points": None if total_rounds == 0 else total_predicted_points / total_rounds,
        "primary_incumbent_total_actual_points": primary_incumbent_total_actual_points,
        "aggregate_delta_vs_primary_incumbent": aggregate_delta,
        "average_delta_per_round_vs_primary_incumbent": average_delta_per_round,
        "improved_seasons_vs_primary_incumbent": None
        if aggregate_delta is None
        else sum(1 for delta, _rounds in season_deltas if delta > 0),
        "worst_season_avg_delta_vs_primary_incumbent": worst_season_avg_delta,
        "selected_calibration_slope": selected_calibration_slope,
        "top50_spearman_delta_vs_primary_incumbent": top50_spearman_delta,
        "candidate_pool_mae_delta_pct_vs_primary_incumbent": candidate_pool_mae_delta_pct,
        "selected_players_mae_delta_pct_vs_primary_incumbent": selected_players_mae_delta_pct,
        "promotion_eligible": bool(promotion["eligible"]),
        "promotion_reason": str(promotion["reason"]),
    }


def _incumbent_actual_points_by_season(
    per_season_summary: pd.DataFrame,
    *,
    incumbent_candidate_id: str,
) -> dict[int, float]:
    incumbent = per_season_summary[per_season_summary["candidate_id"].eq(incumbent_candidate_id)]
    return {int(row["season"]): float(row["total_actual_points"]) for row in incumbent.to_dict(orient="records")}


def _season_deltas(
    group_frame: pd.DataFrame,
    *,
    incumbent_by_season: Mapping[int, float],
) -> list[tuple[float, int]]:
    deltas: list[tuple[float, int]] = []
    for row in group_frame.to_dict(orient="records"):
        season = int(row["season"])
        incumbent_actual_points = incumbent_by_season.get(season)
        if incumbent_actual_points is None:
            return []
        deltas.append((float(row["total_actual_points"]) - incumbent_actual_points, int(row["rounds"])))
    return deltas


def _validate_per_season_summary(per_season_summary: pd.DataFrame) -> None:
    required_columns = {"candidate_id", "season"}
    if not required_columns.issubset(per_season_summary.columns):
        return

    duplicate_mask = per_season_summary.duplicated(subset=["candidate_id", "season"], keep=False)
    if duplicate_mask.any():
        raise ValueError("Duplicate per-season summary rows")


def _validate_prediction_metrics(prediction_metrics: pd.DataFrame) -> None:
    required_columns = {"candidate_id", "season", "metric_scope"}
    if prediction_metrics.empty or not required_columns.issubset(prediction_metrics.columns):
        return

    scoped_metrics = prediction_metrics[prediction_metrics["metric_scope"].isin(_REQUIRED_PROMOTION_METRIC_SCOPES)]
    duplicate_mask = scoped_metrics.duplicated(subset=["candidate_id", "season", "metric_scope"], keep=False)
    if duplicate_mask.any():
        raise ValueError("Duplicate prediction metric rows")


def _season_aligned_mean_metric(
    prediction_metrics: pd.DataFrame,
    *,
    candidate_id: str,
    metric_scope: str,
    metric_column: str,
    required_seasons: set[int],
) -> float | None:
    required_columns = {"candidate_id", "season", "metric_scope", metric_column}
    if not required_seasons or prediction_metrics.empty or not required_columns.issubset(prediction_metrics.columns):
        return None

    rows = prediction_metrics[
        prediction_metrics["candidate_id"].eq(candidate_id) & prediction_metrics["metric_scope"].eq(metric_scope)
    ].copy()
    if set(rows["season"].astype(int)) != required_seasons:
        return None
    rows = rows.sort_values("season", kind="mergesort")
    values = rows[metric_column]
    if values.isna().any():
        return None
    return float(values.mean())


def _percent_delta(candidate_value: float | None, incumbent_value: float | None) -> float | None:
    if candidate_value is None or incumbent_value is None or incumbent_value == 0:
        return None
    return round((candidate_value - incumbent_value) / abs(incumbent_value), 10)


def _sum_or_none(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return float(sum(values))


def _is_missing(value: object) -> bool:
    if value is None:
        return True

    missing = pd.isna(value)
    if isinstance(missing, bool):
        return missing

    return False
