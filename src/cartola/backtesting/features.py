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
    "prior_appearance_rate",
    "prior_points_mean",
    "prior_points_roll3",
    "prior_points_roll5",
    "prior_points_weighted3",
    "prior_points_std",
    "prior_price_mean",
    "prior_variation_mean",
    "club_points_roll3",
    "is_home",
    "opponent_club_points_roll3",
    "prior_media",
    "prior_num_jogos",
    *[f"prior_{scout}_mean" for scout in DEFAULT_SCOUT_COLUMNS],
]

NUMERIC_PRIOR_COLUMNS: list[str] = [
    "position_points_prior",
    "prior_appearances",
    "prior_appearance_rate",
    "prior_points_mean",
    "prior_points_roll3",
    "prior_points_roll5",
    "prior_points_weighted3",
    "prior_points_std",
    "prior_price_mean",
    "prior_variation_mean",
    "club_points_roll3",
    "is_home",
    "opponent_club_points_roll3",
    "prior_media",
    "prior_num_jogos",
    *[f"prior_{scout}_mean" for scout in DEFAULT_SCOUT_COLUMNS],
]


def build_prediction_frame(
    season_df: pd.DataFrame,
    target_round: int,
    fixtures: pd.DataFrame | None = None,
) -> pd.DataFrame:
    candidates = season_df[season_df["rodada"] == target_round].copy()
    played_history = _played_history(season_df, target_round)
    all_history = season_df[season_df["rodada"] < target_round].copy()
    return _add_prior_features(candidates, played_history, all_history, fixtures, target_round)


def build_training_frame(
    season_df: pd.DataFrame,
    target_round: int,
    playable_statuses: tuple[str, ...] | None = None,
    fixtures: pd.DataFrame | None = None,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    historical_rounds = sorted(
        round_number for round_number in season_df["rodada"].dropna().unique() if round_number < target_round
    )

    for round_number in historical_rounds:
        round_frame = build_prediction_frame(season_df, int(round_number), fixtures=fixtures)
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


def _appearance_history_features(all_history: pd.DataFrame) -> pd.DataFrame:
    if all_history.empty:
        return pd.DataFrame(columns=pd.Index(["id_atleta", "prior_appearance_rate"]))

    history = all_history[["id_atleta", "rodada"]].copy()
    if "entrou_em_campo" in all_history.columns:
        history["entrou_em_campo"] = all_history["entrou_em_campo"].fillna(False).astype(bool)
    else:
        history["entrou_em_campo"] = True

    appearances = (
        history.groupby("id_atleta")
        .agg(
            total_rounds=("rodada", "count"),
            played_rounds=("entrou_em_campo", "sum"),
        )
        .reset_index()
    )
    appearances["prior_appearance_rate"] = appearances["played_rounds"] / appearances["total_rounds"]
    return appearances[["id_atleta", "prior_appearance_rate"]]


def _club_history_features(played_history: pd.DataFrame) -> pd.DataFrame:
    if played_history.empty:
        return pd.DataFrame(columns=pd.Index(["id_clube", "club_points_roll3"]))

    club_round = _club_round_points(played_history)
    club_round["club_points_roll3"] = (
        club_round.groupby("id_clube")["club_round_points"].transform(
            lambda values: values.rolling(3, min_periods=1).mean()
        )
    )
    return club_round.groupby("id_clube", as_index=False).agg(
        club_points_roll3=("club_points_roll3", "last")
    )


def _fixture_context_features(
    fixtures: pd.DataFrame | None,
    *,
    target_round: int,
    played_history: pd.DataFrame,
) -> pd.DataFrame:
    columns = pd.Index(["id_clube", "is_home", "opponent_club_points_roll3"])
    if fixtures is None or fixtures.empty:
        return pd.DataFrame(columns=columns)

    fixture_frame = fixtures.copy()
    fixture_frame["rodada"] = pd.to_numeric(fixture_frame["rodada"], errors="raise").astype(int)
    round_fixtures = fixture_frame[fixture_frame["rodada"].eq(target_round)]
    if round_fixtures.empty:
        return pd.DataFrame(columns=columns)

    home_context = round_fixtures[["id_clube_home", "id_clube_away"]].rename(
        columns={"id_clube_home": "id_clube", "id_clube_away": "opponent_id"}
    )
    home_context["is_home"] = 1
    away_context = round_fixtures[["id_clube_away", "id_clube_home"]].rename(
        columns={"id_clube_away": "id_clube", "id_clube_home": "opponent_id"}
    )
    away_context["is_home"] = 0
    context = pd.concat([home_context, away_context], ignore_index=True)
    duplicate_clubs = context.loc[context["id_clube"].duplicated(), "id_clube"].drop_duplicates().to_list()
    if duplicate_clubs:
        raise ValueError(f"Duplicate fixture club context for round {target_round}: {duplicate_clubs}")

    opponent_roll = _club_history_features(played_history).rename(
        columns={"id_clube": "opponent_id", "club_points_roll3": "opponent_club_points_roll3"}
    )
    context = context.merge(opponent_roll, on="opponent_id", how="left")
    return context[["id_clube", "is_home", "opponent_club_points_roll3"]]


def _global_club_points_prior(played_history: pd.DataFrame) -> float:
    if played_history.empty:
        return 0.0
    return float(_club_round_points(played_history)["club_round_points"].mean())


def _club_round_points(played_history: pd.DataFrame) -> pd.DataFrame:
    return (
        played_history.groupby(["id_clube", "rodada"], as_index=False)
        .agg(club_round_points=("pontuacao", "sum"))
        .sort_values(["id_clube", "rodada"])
    )


def _add_prior_features(
    candidates: pd.DataFrame,
    played_history: pd.DataFrame,
    all_history: pd.DataFrame,
    fixtures: pd.DataFrame | None,
    target_round: int,
) -> pd.DataFrame:
    frame = candidates.merge(_player_history_features(played_history), on="id_atleta", how="left")
    frame = frame.merge(_position_priors(played_history), on="posicao", how="left")
    frame = frame.merge(_appearance_history_features(all_history), on="id_atleta", how="left")
    frame = frame.merge(_club_history_features(played_history), on="id_clube", how="left")
    frame = frame.merge(
        _fixture_context_features(fixtures, target_round=target_round, played_history=played_history),
        on="id_clube",
        how="left",
        validate="many_to_one",
    )

    for column in NUMERIC_PRIOR_COLUMNS:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

    global_points_prior = float(played_history["pontuacao"].mean()) if not played_history.empty else 0.0
    global_club_points_prior = _global_club_points_prior(played_history)
    frame["position_points_prior"] = frame["position_points_prior"].fillna(global_points_prior)
    frame["prior_points_mean"] = frame["prior_points_mean"].fillna(frame["position_points_prior"])
    frame["prior_points_roll3"] = frame["prior_points_roll3"].fillna(frame["prior_points_mean"])
    frame["prior_points_roll5"] = frame["prior_points_roll5"].fillna(frame["prior_points_mean"])
    frame["prior_points_weighted3"] = frame["prior_points_weighted3"].fillna(frame["prior_points_mean"])
    frame["prior_points_std"] = frame["prior_points_std"].fillna(0.0)
    frame["prior_appearances"] = frame["prior_appearances"].fillna(0)
    frame["prior_appearance_rate"] = frame["prior_appearance_rate"].fillna(1.0)
    frame["prior_price_mean"] = frame["prior_price_mean"].fillna(frame[MARKET_OPEN_PRICE_COLUMN])
    frame["prior_variation_mean"] = frame["prior_variation_mean"].fillna(0)
    frame["club_points_roll3"] = frame["club_points_roll3"].fillna(global_club_points_prior)
    # Missing fixture context is encoded as the neutral away-like default; runner
    # alignment validation requires fixture coverage for clubs that actually played.
    frame["is_home"] = frame["is_home"].fillna(0).astype(int)
    frame["opponent_club_points_roll3"] = frame["opponent_club_points_roll3"].fillna(global_club_points_prior)
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
            prior_points_weighted3=("pontuacao", _weighted_recent_mean),
            prior_points_std=("pontuacao", "std"),
            prior_price_mean=(MARKET_OPEN_PRICE_COLUMN, "mean"),
            prior_variation_mean=("variacao", "mean"),
            prior_media=("media", "last"),
            prior_num_jogos=("num_jogos", "last"),
        )
        .reset_index()
    )

    scout_columns = [scout for scout in DEFAULT_SCOUT_COLUMNS if scout in ordered.columns]
    if scout_columns:
        scout_deltas = _scout_delta_frame(ordered, scout_columns)
        scout_means = (
            scout_deltas.groupby("id_atleta")[scout_columns]
            .mean()
            .rename(columns={scout: f"prior_{scout}_mean" for scout in scout_columns})
            .reset_index()
        )
        features = features.merge(scout_means, on="id_atleta", how="left")

    return features


def _scout_delta_frame(history: pd.DataFrame, scout_columns: list[str]) -> pd.DataFrame:
    deltas = history[["id_atleta", *scout_columns]].copy()
    for scout in scout_columns:
        values = pd.to_numeric(deltas[scout], errors="coerce").fillna(0)
        deltas[scout] = values.groupby(deltas["id_atleta"]).diff().fillna(values).clip(lower=0)
    return deltas


def _weighted_recent_mean(values: pd.Series) -> float:
    recent_values = [float(value) for value in values.dropna().tail(3).to_list()]
    if not recent_values:
        return float("nan")
    weights = [0.2, 0.3, 0.5][-len(recent_values):]
    weighted_sum = sum(value * weight for value, weight in zip(recent_values, weights, strict=True))
    return float(weighted_sum / sum(weights))


def _player_feature_columns() -> list[str]:
    return [
        "id_atleta",
        "prior_appearances",
        "prior_points_mean",
        "prior_points_roll3",
        "prior_points_roll5",
        "prior_points_weighted3",
        "prior_points_std",
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
