from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from cartola.backtesting.config import MARKET_OPEN_PRICE_COLUMN, BacktestConfig
from cartola.backtesting.data import load_season_data
from cartola.backtesting.features import build_prediction_frame, build_training_frame
from cartola.backtesting.metrics import build_diagnostics, build_summary
from cartola.backtesting.models import BaselinePredictor, RandomForestPointPredictor
from cartola.backtesting.optimizer import optimize_squad

ROUND_RESULT_COLUMNS: list[str] = [
    "rodada",
    "strategy",
    "solver_status",
    "formation",
    "selected_count",
    "budget_used",
    "predicted_points",
    "actual_points",
]

OUTPUT_FLOAT_PRECISION = 10
CSV_FLOAT_FORMAT = f"%.{OUTPUT_FLOAT_PRECISION}f"

FLOAT_NORMALIZATION_EXCLUDED_COLUMNS: set[str] = {
    "rodada",
    "id_atleta",
    "id_clube",
    "num_jogos",
    "prior_appearances",
    "prior_num_jogos",
    "selected_count",
    "rounds",
}


@dataclass(frozen=True)
class BacktestResult:
    round_results: pd.DataFrame
    selected_players: pd.DataFrame
    player_predictions: pd.DataFrame
    summary: pd.DataFrame
    diagnostics: pd.DataFrame


def run_backtest(config: BacktestConfig, season_df: pd.DataFrame | None = None) -> BacktestResult:
    data = (
        season_df.copy() if season_df is not None else load_season_data(config.season, project_root=config.project_root)
    )

    round_rows: list[dict[str, object]] = []
    selected_frames: list[pd.DataFrame] = []
    prediction_frames: list[pd.DataFrame] = []

    max_round = _max_round(data)
    for round_number in range(config.start_round, max_round + 1):
        training = build_training_frame(data, round_number, playable_statuses=config.playable_statuses)
        candidates = build_prediction_frame(data, round_number)
        candidates = candidates[candidates["status"].isin(config.playable_statuses)].copy()

        if training.empty or candidates.empty:
            _record_skipped_round(round_rows, round_number, config, "TrainingEmpty" if training.empty else "Empty")
            continue

        scored_candidates = candidates.copy()
        baseline_model = BaselinePredictor().fit(training)
        forest_model = RandomForestPointPredictor(random_seed=config.random_seed).fit(training)
        scored_candidates["baseline_score"] = baseline_model.predict(scored_candidates)
        scored_candidates["random_forest_score"] = forest_model.predict(scored_candidates)
        scored_candidates["price_score"] = scored_candidates[MARKET_OPEN_PRICE_COLUMN].astype(float)
        prediction_frames.append(scored_candidates.copy())

        for strategy, score_column in _strategies().items():
            strategy_candidates = scored_candidates.copy()
            strategy_candidates["predicted_points"] = strategy_candidates[score_column]
            result = optimize_squad(strategy_candidates, score_column="predicted_points", config=config)
            round_rows.append(
                {
                    "rodada": round_number,
                    "strategy": strategy,
                    "solver_status": result.status,
                    "formation": result.formation_name,
                    "selected_count": result.selected_count,
                    "budget_used": result.budget_used,
                    "predicted_points": result.predicted_points,
                    "actual_points": result.actual_points,
                }
            )

            if not result.selected.empty:
                selected = result.selected.copy()
                selected["rodada"] = round_number
                selected["strategy"] = strategy
                selected_frames.append(selected)

    round_results = pd.DataFrame(round_rows, columns=pd.Index(ROUND_RESULT_COLUMNS))
    selected_players = _concat_or_empty(selected_frames)
    player_predictions = _concat_or_empty(prediction_frames)
    summary = build_summary(round_results, benchmark_strategy="price")
    diagnostics = build_diagnostics(
        round_results,
        selected_players,
        player_predictions,
        benchmark_strategy="price",
        budget=config.budget,
        random_seed=config.random_seed,
    )

    round_results = _normalize_float_outputs(round_results)
    selected_players = _normalize_float_outputs(selected_players)
    player_predictions = _normalize_float_outputs(player_predictions)
    summary = _normalize_float_outputs(summary)
    diagnostics = _normalize_float_outputs(diagnostics)

    _write_outputs(config, round_results, selected_players, player_predictions, summary, diagnostics)
    return BacktestResult(
        round_results=round_results,
        selected_players=selected_players,
        player_predictions=player_predictions,
        summary=summary,
        diagnostics=diagnostics,
    )


def _max_round(data: pd.DataFrame) -> int:
    if data.empty:
        return 0
    return int(data["rodada"].max())


def _strategies() -> dict[str, str]:
    return {
        "baseline": "baseline_score",
        "random_forest": "random_forest_score",
        "price": "price_score",
    }


def _record_skipped_round(
    round_rows: list[dict[str, object]],
    round_number: int,
    config: BacktestConfig,
    status: str,
) -> None:
    for strategy in _strategies():
        round_rows.append(
            {
                "rodada": round_number,
                "strategy": strategy,
                "solver_status": status,
                "formation": config.formation_name,
                "selected_count": 0,
                "budget_used": 0.0,
                "predicted_points": 0.0,
                "actual_points": 0.0,
            }
        )


def _concat_or_empty(frames: list[pd.DataFrame]) -> pd.DataFrame:
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _normalize_float_outputs(frame: pd.DataFrame) -> pd.DataFrame:
    """Round non-identifier float outputs so repeated runs serialize identically."""
    normalized = frame.copy()
    float_columns = [
        column
        for column in normalized.select_dtypes(include=["float"]).columns
        if column not in FLOAT_NORMALIZATION_EXCLUDED_COLUMNS
    ]
    if float_columns:
        normalized.loc[:, float_columns] = normalized.loc[:, float_columns].round(OUTPUT_FLOAT_PRECISION)
    return normalized


def _write_outputs(
    config: BacktestConfig,
    round_results: pd.DataFrame,
    selected_players: pd.DataFrame,
    player_predictions: pd.DataFrame,
    summary: pd.DataFrame,
    diagnostics: pd.DataFrame,
) -> None:
    output_path = config.output_path
    output_path.mkdir(parents=True, exist_ok=True)
    round_results.to_csv(output_path / "round_results.csv", index=False, float_format=CSV_FLOAT_FORMAT)
    selected_players.to_csv(output_path / "selected_players.csv", index=False, float_format=CSV_FLOAT_FORMAT)
    player_predictions.to_csv(output_path / "player_predictions.csv", index=False, float_format=CSV_FLOAT_FORMAT)
    summary.to_csv(output_path / "summary.csv", index=False, float_format=CSV_FLOAT_FORMAT)
    diagnostics.to_csv(output_path / "diagnostics.csv", index=False, float_format=CSV_FLOAT_FORMAT)
