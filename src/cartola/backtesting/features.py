from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

from cartola.backtesting.config import DEFAULT_SCOUT_COLUMNS, MARKET_OPEN_PRICE_COLUMN
from cartola.backtesting.footystats_features import merge_footystats_features

if TYPE_CHECKING:
    from cartola.backtesting.config import BacktestConfig

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
    "prior_media",
    "prior_num_jogos",
    *[f"prior_{scout}_mean" for scout in DEFAULT_SCOUT_COLUMNS],
]

FOOTYSTATS_PPG_FEATURE_COLUMNS: list[str] = [
    "footystats_team_pre_match_ppg",
    "footystats_opponent_pre_match_ppg",
    "footystats_ppg_diff",
]

FOOTYSTATS_XG_FEATURE_COLUMNS: list[str] = [
    "footystats_team_pre_match_xg",
    "footystats_opponent_pre_match_xg",
    "footystats_xg_diff",
]

MATCHUP_CONTEXT_V1_FEATURE_COLUMNS: list[str] = [
    "matchup_is_home",
    "matchup_opponent_allowed_points_roll5",
    "matchup_opponent_allowed_position_points_roll5",
    "matchup_club_position_points_roll5",
    "matchup_opponent_allowed_position_count",
    "matchup_club_position_count",
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
    "prior_media",
    "prior_num_jogos",
    *MATCHUP_CONTEXT_V1_FEATURE_COLUMNS,
    *[f"prior_{scout}_mean" for scout in DEFAULT_SCOUT_COLUMNS],
]


def feature_columns_for_config(config: BacktestConfig) -> list[str]:
    columns = list(FEATURE_COLUMNS)
    if config.footystats_mode == "none":
        pass
    elif config.footystats_mode == "ppg":
        columns.extend(FOOTYSTATS_PPG_FEATURE_COLUMNS)
    elif config.footystats_mode == "ppg_xg":
        columns.extend([*FOOTYSTATS_PPG_FEATURE_COLUMNS, *FOOTYSTATS_XG_FEATURE_COLUMNS])
    else:
        raise ValueError(f"Unsupported footystats_mode: {config.footystats_mode!r}")

    if config.matchup_context_mode == "none":
        return columns
    if config.matchup_context_mode == "cartola_matchup_v1":
        return [*columns, *MATCHUP_CONTEXT_V1_FEATURE_COLUMNS]
    raise ValueError(f"Unsupported matchup_context_mode: {config.matchup_context_mode!r}")


def build_prediction_frame(
    season_df: pd.DataFrame,
    target_round: int,
    fixtures: pd.DataFrame | None = None,
    footystats_rows: pd.DataFrame | None = None,
    matchup_context_mode: str = "none",
) -> pd.DataFrame:
    candidates = season_df[season_df["rodada"] == target_round].copy()
    played_history = _played_history(season_df, target_round)
    all_history = season_df[season_df["rodada"] < target_round].copy()
    frame = _add_prior_features(
        candidates,
        played_history,
        all_history,
        fixtures,
        target_round,
        matchup_context_mode=matchup_context_mode,
    )
    return merge_footystats_features(frame, footystats_rows, target_round=target_round)


def build_training_frame(
    season_df: pd.DataFrame,
    target_round: int,
    playable_statuses: tuple[str, ...] | None = None,
    fixtures: pd.DataFrame | None = None,
    footystats_rows: pd.DataFrame | None = None,
    matchup_context_mode: str = "none",
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    historical_rounds = sorted(
        round_number for round_number in season_df["rodada"].dropna().unique() if round_number < target_round
    )

    for round_number in historical_rounds:
        round_frame = build_prediction_frame(
            season_df,
            int(round_number),
            fixtures=fixtures,
            footystats_rows=footystats_rows,
            matchup_context_mode=matchup_context_mode,
        )
        if playable_statuses is not None:
            round_frame = round_frame[round_frame["status"].isin(playable_statuses)].copy()
        round_frame["target"] = round_frame["pontuacao"]
        frames.append(round_frame)

    if not frames:
        feature_columns = list(FEATURE_COLUMNS)
        if matchup_context_mode == "cartola_matchup_v1":
            feature_columns.extend(MATCHUP_CONTEXT_V1_FEATURE_COLUMNS)
        return pd.DataFrame(
            columns=pd.Index(
                list(dict.fromkeys([*MARKET_COLUMNS, *feature_columns, "target"]))
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


def _round_fixture_context(fixtures: pd.DataFrame | None, target_round: int) -> pd.DataFrame:
    columns = pd.Index(["id_clube", "opponent_id_clube", "matchup_is_home"])
    if fixtures is None or fixtures.empty:
        return pd.DataFrame(columns=columns)

    fixture_frame = fixtures.copy()
    fixture_frame["rodada"] = pd.to_numeric(fixture_frame["rodada"], errors="raise").astype(int)
    round_fixtures = fixture_frame[fixture_frame["rodada"].eq(target_round)]
    if round_fixtures.empty:
        return pd.DataFrame(columns=columns)

    home_context = round_fixtures[["id_clube_home", "id_clube_away"]].rename(
        columns={"id_clube_home": "id_clube", "id_clube_away": "opponent_id_clube"}
    )
    home_context["matchup_is_home"] = 1
    away_context = round_fixtures[["id_clube_away", "id_clube_home"]].rename(
        columns={"id_clube_away": "id_clube", "id_clube_home": "opponent_id_clube"}
    )
    away_context["matchup_is_home"] = 0
    context = pd.concat([home_context, away_context], ignore_index=True)
    duplicate_clubs = context.loc[context["id_clube"].duplicated(), "id_clube"].drop_duplicates().to_list()
    if duplicate_clubs:
        raise ValueError(f"Duplicate fixture club context for round {target_round}: {duplicate_clubs}")
    return context[["id_clube", "opponent_id_clube", "matchup_is_home"]]


def _historical_fixture_context(fixtures: pd.DataFrame | None, target_round: int) -> pd.DataFrame:
    columns = pd.Index(["rodada", "id_clube", "opponent_id_clube"])
    if fixtures is None or fixtures.empty:
        return pd.DataFrame(columns=columns)

    fixture_frame = fixtures.copy()
    fixture_frame["rodada"] = pd.to_numeric(fixture_frame["rodada"], errors="raise").astype(int)
    fixture_frame = fixture_frame[fixture_frame["rodada"].lt(target_round)]
    if fixture_frame.empty:
        return pd.DataFrame(columns=columns)

    home_context = fixture_frame[["rodada", "id_clube_home", "id_clube_away"]].rename(
        columns={"id_clube_home": "id_clube", "id_clube_away": "opponent_id_clube"}
    )
    away_context = fixture_frame[["rodada", "id_clube_away", "id_clube_home"]].rename(
        columns={"id_clube_away": "id_clube", "id_clube_home": "opponent_id_clube"}
    )
    return pd.concat([home_context, away_context], ignore_index=True)[["rodada", "id_clube", "opponent_id_clube"]]


def _roll5_last(
    frame: pd.DataFrame,
    group_columns: list[str],
    value_column: str,
    output_column: str,
    count_column: str | None = None,
) -> pd.DataFrame:
    if frame.empty:
        columns = [*group_columns, output_column]
        if count_column is not None:
            columns.append(count_column)
        return pd.DataFrame(columns=pd.Index(columns))

    rows: list[dict[str, object]] = []
    for key, group in frame.sort_values([*group_columns, "rodada"]).groupby(group_columns, sort=False, dropna=False):
        key_values = key if isinstance(key, tuple) else (key,)
        recent = group.tail(5)
        row = {column: value for column, value in zip(group_columns, key_values, strict=True)}
        row[output_column] = float(recent[value_column].mean())
        if count_column is not None:
            row[count_column] = int(recent["sample_count"].sum())
        rows.append(row)
    return pd.DataFrame(rows)


def _club_position_roll5(played_history: pd.DataFrame) -> pd.DataFrame:
    columns = pd.Index(
        [
            "id_clube",
            "posicao",
            "matchup_club_position_points_roll5",
            "matchup_club_position_count",
        ]
    )
    if played_history.empty:
        return pd.DataFrame(columns=columns)

    club_position_round = (
        played_history.groupby(["id_clube", "posicao", "rodada"], as_index=False)
        .agg(position_points=("pontuacao", "mean"), sample_count=("pontuacao", "count"))
        .sort_values(["id_clube", "posicao", "rodada"])
    )
    result = _roll5_last(
        club_position_round,
        ["id_clube", "posicao"],
        "position_points",
        "matchup_club_position_points_roll5",
        "matchup_club_position_count",
    )
    return result if not result.empty else pd.DataFrame(columns=columns)


def _opponent_allowed_roll5(played_history: pd.DataFrame, fixtures: pd.DataFrame | None, target_round: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    all_columns = pd.Index(["opponent_id_clube", "matchup_opponent_allowed_points_roll5"])
    position_columns = pd.Index(
        [
            "opponent_id_clube",
            "posicao",
            "matchup_opponent_allowed_position_points_roll5",
            "matchup_opponent_allowed_position_count",
        ]
    )
    fixture_context = _historical_fixture_context(fixtures, target_round)
    if played_history.empty or fixture_context.empty:
        return pd.DataFrame(columns=all_columns), pd.DataFrame(columns=position_columns)

    scored_against = played_history.merge(
        fixture_context,
        on=["rodada", "id_clube"],
        how="inner",
        validate="many_to_one",
    )
    if scored_against.empty:
        return pd.DataFrame(columns=all_columns), pd.DataFrame(columns=position_columns)

    opponent_round = (
        scored_against.groupby(["opponent_id_clube", "rodada"], as_index=False)
        .agg(allowed_points=("pontuacao", "mean"), sample_count=("pontuacao", "count"))
        .sort_values(["opponent_id_clube", "rodada"])
    )
    opponent_position_round = (
        scored_against.groupby(["opponent_id_clube", "posicao", "rodada"], as_index=False)
        .agg(allowed_position_points=("pontuacao", "mean"), sample_count=("pontuacao", "count"))
        .sort_values(["opponent_id_clube", "posicao", "rodada"])
    )
    opponent_all = _roll5_last(
        opponent_round,
        ["opponent_id_clube"],
        "allowed_points",
        "matchup_opponent_allowed_points_roll5",
    )
    opponent_position = _roll5_last(
        opponent_position_round,
        ["opponent_id_clube", "posicao"],
        "allowed_position_points",
        "matchup_opponent_allowed_position_points_roll5",
        "matchup_opponent_allowed_position_count",
    )
    return opponent_all, opponent_position


def _add_matchup_context_features(
    frame: pd.DataFrame,
    played_history: pd.DataFrame,
    fixtures: pd.DataFrame | None,
    target_round: int,
) -> pd.DataFrame:
    result = frame.merge(
        _round_fixture_context(fixtures, target_round),
        on="id_clube",
        how="left",
        validate="many_to_one",
    )
    opponent_allowed, opponent_position_allowed = _opponent_allowed_roll5(played_history, fixtures, target_round)
    result = result.merge(opponent_allowed, on="opponent_id_clube", how="left", validate="many_to_one")
    result = result.merge(
        opponent_position_allowed,
        on=["opponent_id_clube", "posicao"],
        how="left",
        validate="many_to_one",
    )
    result = result.merge(_club_position_roll5(played_history), on=["id_clube", "posicao"], how="left", validate="many_to_one")

    global_points_prior = float(played_history["pontuacao"].mean()) if not played_history.empty else 0.0
    position_priors = _position_priors(played_history).rename(
        columns={"position_points_prior": "_matchup_position_points_prior"}
    )
    result = result.merge(position_priors, on="posicao", how="left", validate="many_to_one")
    for column in [*MATCHUP_CONTEXT_V1_FEATURE_COLUMNS, "_matchup_position_points_prior"]:
        if column not in result.columns:
            result[column] = pd.NA
        result[column] = pd.to_numeric(result[column], errors="coerce")

    result["matchup_is_home"] = result["matchup_is_home"].fillna(0).astype(int)
    result["matchup_opponent_allowed_points_roll5"] = result["matchup_opponent_allowed_points_roll5"].fillna(
        global_points_prior
    )
    result["matchup_opponent_allowed_position_points_roll5"] = (
        result["matchup_opponent_allowed_position_points_roll5"]
        .fillna(result["matchup_opponent_allowed_points_roll5"])
        .fillna(result["_matchup_position_points_prior"])
        .fillna(global_points_prior)
    )
    result["matchup_club_position_points_roll5"] = (
        result["matchup_club_position_points_roll5"]
        .fillna(result["_matchup_position_points_prior"])
        .fillna(global_points_prior)
    )
    result["matchup_opponent_allowed_position_count"] = (
        result["matchup_opponent_allowed_position_count"].fillna(0).astype(int)
    )
    result["matchup_club_position_count"] = result["matchup_club_position_count"].fillna(0).astype(int)
    return result.drop(columns=["opponent_id_clube", "_matchup_position_points_prior"], errors="ignore")


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
    matchup_context_mode: str = "none",
) -> pd.DataFrame:
    frame = candidates.merge(_player_history_features(played_history), on="id_atleta", how="left")
    frame = frame.merge(_position_priors(played_history), on="posicao", how="left")
    frame = frame.merge(_appearance_history_features(all_history), on="id_atleta", how="left")
    frame = frame.merge(_club_history_features(played_history), on="id_clube", how="left")
    if matchup_context_mode == "cartola_matchup_v1":
        frame = _add_matchup_context_features(frame, played_history, fixtures, target_round)
    elif matchup_context_mode != "none":
        raise ValueError(f"Unsupported matchup_context_mode: {matchup_context_mode!r}")

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
