from __future__ import annotations

import pandas as pd

from cartola.backtesting.config import DEFAULT_SCOUT_COLUMNS, MARKET_OPEN_PRICE_COLUMN

MARKET_COLUMNS: list[str] = [
    "id_atleta",
    "apelido",
    "slug",
    "id_clube",
    "nome_clube",
    "posicao",
    "status",
    "rodada",
    "preco",
    MARKET_OPEN_PRICE_COLUMN,
    "pontuacao",
    "media",
    "num_jogos",
    "variacao",
    "entrou_em_campo",
]

FEATURE_COLUMNS: list[str] = [
    MARKET_OPEN_PRICE_COLUMN,
    "id_clube",
    "rodada",
    "posicao",
    "prior_appearances",
    "prior_points_mean",
    "prior_points_roll3",
    "prior_points_roll5",
    "prior_price_mean",
    "prior_variation_mean",
    "prior_media",
    "prior_num_jogos",
    *[f"prior_{scout}_mean" for scout in DEFAULT_SCOUT_COLUMNS],
]

NUMERIC_PRIOR_COLUMNS: list[str] = [
    "position_points_prior",
    "prior_appearances",
    "prior_points_mean",
    "prior_points_roll3",
    "prior_points_roll5",
    "prior_price_mean",
    "prior_variation_mean",
    "prior_media",
    "prior_num_jogos",
    *[f"prior_{scout}_mean" for scout in DEFAULT_SCOUT_COLUMNS],
]


def build_prediction_frame(season_df: pd.DataFrame, target_round: int) -> pd.DataFrame:
    candidates = season_df[season_df["rodada"] == target_round].copy()
    history = _played_history(season_df, target_round)
    return _add_prior_features(candidates, history)


def build_training_frame(
    season_df: pd.DataFrame,
    target_round: int,
    playable_statuses: tuple[str, ...] | None = None,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    historical_rounds = sorted(
        round_number for round_number in season_df["rodada"].dropna().unique() if round_number < target_round
    )

    for round_number in historical_rounds:
        round_frame = build_prediction_frame(season_df, int(round_number))
        if playable_statuses is not None:
            round_frame = round_frame[round_frame["status"].isin(playable_statuses)].copy()
        round_frame["target"] = round_frame["pontuacao"]
        frames.append(round_frame)

    if not frames:
        return pd.DataFrame(
            columns=pd.Index(
                list(dict.fromkeys([*MARKET_COLUMNS, *FEATURE_COLUMNS, "target"]))
            )
        )
    return pd.concat(frames, ignore_index=True)


def _played_history(season_df: pd.DataFrame, target_round: int) -> pd.DataFrame:
    history = season_df[season_df["rodada"] < target_round].copy()
    if "entrou_em_campo" in history.columns:
        history = history[history["entrou_em_campo"].fillna(False)]
    return history


def _add_prior_features(candidates: pd.DataFrame, history: pd.DataFrame) -> pd.DataFrame:
    frame = candidates.merge(_player_history_features(history), on="id_atleta", how="left")
    frame = frame.merge(_position_priors(history), on="posicao", how="left")

    for column in NUMERIC_PRIOR_COLUMNS:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

    global_points_prior = float(history["pontuacao"].mean()) if not history.empty else 0.0
    frame["position_points_prior"] = frame["position_points_prior"].fillna(global_points_prior)
    frame["prior_points_mean"] = frame["prior_points_mean"].fillna(frame["position_points_prior"])
    frame["prior_points_roll3"] = frame["prior_points_roll3"].fillna(frame["prior_points_mean"])
    frame["prior_points_roll5"] = frame["prior_points_roll5"].fillna(frame["prior_points_mean"])
    frame["prior_appearances"] = frame["prior_appearances"].fillna(0)
    frame["prior_price_mean"] = frame["prior_price_mean"].fillna(frame[MARKET_OPEN_PRICE_COLUMN])
    frame["prior_variation_mean"] = frame["prior_variation_mean"].fillna(0)
    frame["prior_media"] = frame["prior_media"].fillna(frame["prior_points_mean"])
    frame["prior_num_jogos"] = frame["prior_num_jogos"].fillna(0)

    for scout in DEFAULT_SCOUT_COLUMNS:
        column = f"prior_{scout}_mean"
        if column not in frame.columns:
            frame[column] = 0.0
        frame[column] = frame[column].fillna(0.0)

    return frame


def _player_history_features(history: pd.DataFrame) -> pd.DataFrame:
    if history.empty:
        return pd.DataFrame(columns=pd.Index(_player_feature_columns()))

    ordered = history.sort_values(["id_atleta", "rodada"])
    features = (
        ordered.groupby("id_atleta")
        .agg(
            prior_appearances=("rodada", "count"),
            prior_points_mean=("pontuacao", "mean"),
            prior_points_roll3=("pontuacao", lambda values: float(values.tail(3).mean())),
            prior_points_roll5=("pontuacao", lambda values: float(values.tail(5).mean())),
            prior_price_mean=(MARKET_OPEN_PRICE_COLUMN, "mean"),
            prior_variation_mean=("variacao", "mean"),
            prior_media=("media", "last"),
            prior_num_jogos=("num_jogos", "last"),
        )
        .reset_index()
    )

    scout_columns = [scout for scout in DEFAULT_SCOUT_COLUMNS if scout in ordered.columns]
    if scout_columns:
        scout_means = (
            ordered.groupby("id_atleta")[scout_columns]
            .mean()
            .rename(columns={scout: f"prior_{scout}_mean" for scout in scout_columns})
            .reset_index()
        )
        features = features.merge(scout_means, on="id_atleta", how="left")

    return features


def _player_feature_columns() -> list[str]:
    return [
        "id_atleta",
        "prior_appearances",
        "prior_points_mean",
        "prior_points_roll3",
        "prior_points_roll5",
        "prior_price_mean",
        "prior_variation_mean",
        "prior_media",
        "prior_num_jogos",
        *[f"prior_{scout}_mean" for scout in DEFAULT_SCOUT_COLUMNS],
    ]


def _position_priors(history: pd.DataFrame) -> pd.DataFrame:
    if history.empty:
        return pd.DataFrame(columns=pd.Index(["posicao", "position_points_prior"]))
    return history.groupby("posicao", as_index=False).agg(position_points_prior=("pontuacao", "mean"))
