from __future__ import annotations

import math
from collections.abc import Mapping

import pandas as pd

from cartola.backtesting.config import MARKET_OPEN_PRICE_COLUMN
from cartola.backtesting.scoring_contract import CAPTAIN_MULTIPLIER

SUMMARY_COLUMNS: list[str] = [
    "strategy",
    "rounds",
    "total_actual_points",
    "average_actual_points",
    "total_predicted_points",
]

DIAGNOSTIC_COLUMNS: list[str] = ["section", "strategy", "position", "metric", "value"]


def build_summary(round_results: pd.DataFrame, benchmark_strategy: str = "price") -> pd.DataFrame:
    delta_column = f"actual_points_delta_vs_{benchmark_strategy}"
    columns = [*SUMMARY_COLUMNS, delta_column]

    if round_results.empty:
        return pd.DataFrame(columns=pd.Index(columns))

    optimal_results = round_results[round_results["solver_status"] == "Optimal"]
    if optimal_results.empty:
        return pd.DataFrame(columns=pd.Index(columns))

    summary = (
        optimal_results.groupby("strategy", as_index=False)
        .agg(
            rounds=("rodada", "nunique"),
            total_actual_points=("actual_points", "sum"),
            average_actual_points=("actual_points", "mean"),
            total_predicted_points=("predicted_points", "sum"),
        )
        .sort_values("total_actual_points", ascending=False)
        .reset_index(drop=True)
    )

    benchmark_rows = summary.loc[summary["strategy"] == benchmark_strategy, "total_actual_points"]
    if benchmark_rows.empty:
        summary[delta_column] = pd.NA
        return summary.loc[:, columns]

    benchmark_total = benchmark_rows.iloc[0]
    summary[delta_column] = summary["total_actual_points"] - benchmark_total
    return summary.loc[:, columns]


def build_diagnostics(
    round_results: pd.DataFrame,
    selected_players: pd.DataFrame,
    player_predictions: pd.DataFrame,
    *,
    benchmark_strategy: str = "price",
    budget: float = 100.0,
    random_draws: int = 100,
    random_seed: int = 123,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    _append_prediction_diagnostics(rows, player_predictions)
    _append_round_diagnostics(rows, round_results, benchmark_strategy)
    _append_selection_diagnostics(rows, selected_players)
    _append_random_selection_diagnostics(
        rows,
        round_results,
        selected_players,
        player_predictions,
        budget=budget,
        random_draws=random_draws,
        random_seed=random_seed,
    )
    return pd.DataFrame(rows, columns=pd.Index(DIAGNOSTIC_COLUMNS))


def _append_prediction_diagnostics(rows: list[dict[str, object]], player_predictions: pd.DataFrame) -> None:
    if player_predictions.empty:
        return

    for strategy, score_column in _score_columns(player_predictions).items():
        _append_prediction_group(rows, strategy, "all", player_predictions, score_column)
        for position, position_frame in player_predictions.groupby("posicao"):
            _append_prediction_group(rows, strategy, str(position), position_frame, score_column)


def _append_prediction_group(
    rows: list[dict[str, object]],
    strategy: str,
    position: str,
    frame: pd.DataFrame,
    score_column: str,
) -> None:
    paired = frame[[score_column, "pontuacao"]].dropna()
    predicted = paired[score_column].astype(float)
    actual = paired["pontuacao"].astype(float)

    _append_metric(rows, "prediction", strategy, position, "player_count", len(paired))
    if paired.empty:
        _append_metric(rows, "prediction", strategy, position, "player_mae", pd.NA)
        _append_metric(rows, "prediction", strategy, position, "player_mean_error", pd.NA)
        _append_metric(rows, "prediction", strategy, position, "player_correlation", pd.NA)
        _append_metric(rows, "prediction", strategy, position, "player_r2", pd.NA)
        return

    errors = predicted - actual
    _append_metric(rows, "prediction", strategy, position, "player_mae", errors.abs().mean())
    _append_metric(rows, "prediction", strategy, position, "player_mean_error", errors.mean())
    _append_metric(rows, "prediction", strategy, position, "player_correlation", _correlation(predicted, actual))
    _append_metric(rows, "prediction", strategy, position, "player_r2", _r2(predicted, actual))


def _append_round_diagnostics(
    rows: list[dict[str, object]],
    round_results: pd.DataFrame,
    benchmark_strategy: str,
) -> None:
    optimal = _optimal_round_results(round_results)
    if optimal.empty:
        return

    actual_by_round = optimal.pivot(index="rodada", columns="strategy", values="actual_points")
    for strategy, strategy_rounds in optimal.groupby("strategy"):
        actual_points = strategy_rounds["actual_points"].astype(float)
        _append_metric(rows, "rounds", str(strategy), "all", "round_count", strategy_rounds["rodada"].nunique())
        _append_metric(rows, "rounds", str(strategy), "all", "actual_points_total", actual_points.sum())
        _append_metric(rows, "rounds", str(strategy), "all", "actual_points_mean", actual_points.mean())
        _append_metric(rows, "rounds", str(strategy), "all", "actual_points_stddev", _sample_std(actual_points))
        _append_metric(rows, "rounds", str(strategy), "all", "best_round_actual", actual_points.max())
        _append_metric(rows, "rounds", str(strategy), "all", "worst_round_actual", actual_points.min())

        if benchmark_strategy in actual_by_round.columns and strategy in actual_by_round.columns:
            deltas = (actual_by_round[strategy] - actual_by_round[benchmark_strategy]).dropna()
            _append_metric(rows, "rounds", str(strategy), "all", f"round_wins_vs_{benchmark_strategy}", (deltas > 0).sum())
            _append_metric(
                rows, "rounds", str(strategy), "all", f"round_losses_vs_{benchmark_strategy}", (deltas < 0).sum()
            )
            _append_metric(rows, "rounds", str(strategy), "all", f"round_ties_vs_{benchmark_strategy}", (deltas == 0).sum())
            _append_metric(rows, "rounds", str(strategy), "all", f"round_total_delta_vs_{benchmark_strategy}", deltas.sum())
            _append_metric(rows, "rounds", str(strategy), "all", f"round_average_delta_vs_{benchmark_strategy}", deltas.mean())
            _append_metric(rows, "rounds", str(strategy), "all", f"round_delta_stddev_vs_{benchmark_strategy}", _sample_std(deltas))


def _append_selection_diagnostics(rows: list[dict[str, object]], selected_players: pd.DataFrame) -> None:
    if selected_players.empty:
        return

    for strategy, strategy_frame in selected_players.groupby("strategy"):
        _append_selection_group(rows, str(strategy), "all", strategy_frame)
        for position, position_frame in strategy_frame.groupby("posicao"):
            _append_selection_group(rows, str(strategy), str(position), position_frame)


def _append_selection_group(
    rows: list[dict[str, object]],
    strategy: str,
    position: str,
    frame: pd.DataFrame,
) -> None:
    actual_points = frame["pontuacao"].astype(float)
    entered = frame["entrou_em_campo"].fillna(False).astype(bool)

    _append_metric(rows, "selection", strategy, position, "selected_player_count", len(frame))
    _append_metric(rows, "selection", strategy, position, "selected_player_actual_points_mean", actual_points.mean())
    _append_metric(rows, "selection", strategy, position, "selected_player_actual_points_median", actual_points.median())
    _append_metric(rows, "selection", strategy, position, "selected_entrou_em_campo_rate", entered.mean())


def _append_random_selection_diagnostics(
    rows: list[dict[str, object]],
    round_results: pd.DataFrame,
    selected_players: pd.DataFrame,
    player_predictions: pd.DataFrame,
    *,
    budget: float,
    random_draws: int,
    random_seed: int,
) -> None:
    optimal = _optimal_round_results(round_results)
    if optimal.empty or selected_players.empty or player_predictions.empty:
        return

    for strategy, strategy_rounds in optimal.groupby("strategy"):
        expected_points_by_round: list[float] = []
        successful_draws = 0
        requested_draws = 0

        for _, round_row in strategy_rounds.iterrows():
            round_number = int(round_row["rodada"])
            selected_round = selected_players[
                selected_players["strategy"].eq(strategy) & selected_players["rodada"].eq(round_number)
            ]
            candidate_pool = player_predictions[player_predictions["rodada"].eq(round_number)]
            if selected_round.empty or candidate_pool.empty:
                continue

            formation = selected_round["posicao"].value_counts().to_dict()
            draws = _random_valid_squad_points(
                candidate_pool,
                formation,
                budget=budget,
                random_draws=random_draws,
                random_seed=_round_seed(random_seed, round_number, formation),
            )
            requested_draws += random_draws
            successful_draws += len(draws)
            if draws:
                expected_points_by_round.append(float(pd.Series(draws).mean()))

        actual_total = float(strategy_rounds["actual_points"].sum())
        expected_total = float(pd.Series(expected_points_by_round).sum()) if expected_points_by_round else None
        expected_average = float(pd.Series(expected_points_by_round).mean()) if expected_points_by_round else pd.NA
        delta = actual_total - expected_total if expected_total is not None else pd.NA

        _append_metric(rows, "random_selection", str(strategy), "all", "requested_random_draws", requested_draws)
        _append_metric(rows, "random_selection", str(strategy), "all", "successful_random_draws", successful_draws)
        _append_metric(
            rows,
            "random_selection",
            str(strategy),
            "all",
            "random_baseline_captain_policy",
            "actual_best_non_tecnico",
        )
        _append_metric(
            rows,
            "random_selection",
            str(strategy),
            "all",
            "random_expected_actual_points_total",
            expected_total if expected_total is not None else pd.NA,
        )
        _append_metric(
            rows,
            "random_selection",
            str(strategy),
            "all",
            "random_expected_actual_points_average",
            expected_average,
        )
        _append_metric(rows, "random_selection", str(strategy), "all", "actual_points_delta_vs_random_expected", delta)


def _random_valid_squad_points(
    candidate_pool: pd.DataFrame,
    formation: Mapping[object, int],
    *,
    budget: float,
    random_draws: int,
    random_seed: int,
) -> list[float]:
    if random_draws <= 0 or MARKET_OPEN_PRICE_COLUMN not in candidate_pool.columns:
        return []

    by_position = {
        position: position_frame.reset_index(drop=True)
        for position, position_frame in candidate_pool.groupby("posicao")
    }
    if any(position not in by_position or len(by_position[position]) < count for position, count in formation.items()):
        return []

    points: list[float] = []
    max_attempts = max(random_draws * 100, 100)
    attempts = 0

    while len(points) < random_draws and attempts < max_attempts:
        attempts += 1
        sampled_frames = []
        for position_index, (position, count) in enumerate(formation.items()):
            position_frame = by_position[position]
            sampled_frames.append(
                position_frame.sample(
                    n=int(count),
                    replace=False,
                    random_state=random_seed + attempts * 1009 + position_index,
                )
            )

        squad = pd.concat(sampled_frames, ignore_index=True)
        if float(squad[MARKET_OPEN_PRICE_COLUMN].sum()) <= budget:
            actual_points = pd.to_numeric(squad["pontuacao"], errors="raise").astype(float)
            captain_candidates = squad["posicao"].ne("tec")
            if not captain_candidates.any():
                continue
            base_actual = float(actual_points.sum())
            captain_actual = float(actual_points.loc[captain_candidates].max())
            points.append(base_actual + (CAPTAIN_MULTIPLIER - 1.0) * captain_actual)

    return points


def _score_columns(player_predictions: pd.DataFrame) -> dict[str, str]:
    return {
        column.removesuffix("_score"): column
        for column in player_predictions.columns
        if column.endswith("_score")
    }


def _optimal_round_results(round_results: pd.DataFrame) -> pd.DataFrame:
    if round_results.empty or "solver_status" not in round_results.columns:
        return pd.DataFrame()
    return round_results[round_results["solver_status"] == "Optimal"]


def _correlation(predicted: pd.Series, actual: pd.Series) -> object:
    if len(predicted) < 2 or math.isclose(float(predicted.std()), 0.0) or math.isclose(float(actual.std()), 0.0):
        return pd.NA
    return float(predicted.corr(actual))


def _r2(predicted: pd.Series, actual: pd.Series) -> object:
    total_sum_of_squares = float(((actual - actual.mean()) ** 2).sum())
    if math.isclose(total_sum_of_squares, 0.0):
        return pd.NA
    residual_sum_of_squares = float(((actual - predicted) ** 2).sum())
    return 1.0 - residual_sum_of_squares / total_sum_of_squares


def _sample_std(values: pd.Series) -> object:
    if len(values) < 2:
        return pd.NA
    return float(values.std())


def _round_seed(random_seed: int, round_number: int, formation: Mapping[object, int]) -> int:
    formation_offset = sum((index + 1) * int(count) for index, (_, count) in enumerate(sorted(formation.items())))
    return random_seed + round_number * 1009 + formation_offset


def _append_metric(
    rows: list[dict[str, object]],
    section: str,
    strategy: str,
    position: str,
    metric: str,
    value: object,
) -> None:
    rows.append(
        {
            "section": section,
            "strategy": strategy,
            "position": position,
            "metric": metric,
            "value": value,
        }
    )
