import warnings

import pandas as pd
import pytest

from cartola.backtesting.config import DEFAULT_SCOUT_COLUMNS
from cartola.backtesting.features import FEATURE_COLUMNS, build_prediction_frame, build_training_frame


def _season_df() -> pd.DataFrame:
    rows = [
        {
            "rodada": 1,
            "id_atleta": 1,
            "apelido": "A",
            "slug": "a",
            "posicao": "ata",
            "status": "Provavel",
            "preco": 10,
            "preco_pre_rodada": 10,
            "pontuacao": 2,
            "media": 2,
            "num_jogos": 1,
            "variacao": 0,
            "id_clube": 10,
            "nome_clube": "Clube 10",
            "entrou_em_campo": True,
            "G": 0,
            "A": 0,
            "DS": 1,
            "V": 0,
        },
        {
            "rodada": 2,
            "id_atleta": 1,
            "apelido": "A",
            "slug": "a",
            "posicao": "ata",
            "status": "Provavel",
            "preco": 11,
            "preco_pre_rodada": 10,
            "pontuacao": 8,
            "media": 5,
            "num_jogos": 2,
            "variacao": 1,
            "id_clube": 10,
            "nome_clube": "Clube 10",
            "entrou_em_campo": True,
            "G": 1,
            "A": 0,
            "DS": 0,
            "V": 0,
        },
        {
            "rodada": 3,
            "id_atleta": 1,
            "apelido": "A",
            "slug": "a",
            "posicao": "ata",
            "status": "Provavel",
            "preco": 12,
            "preco_pre_rodada": 11,
            "pontuacao": 100,
            "media": 36.7,
            "num_jogos": 3,
            "variacao": 1,
            "id_clube": 10,
            "nome_clube": "Clube 10",
            "entrou_em_campo": True,
            "G": 5,
            "A": 5,
            "DS": 0,
            "V": 0,
        },
        {
            "rodada": 1,
            "id_atleta": 2,
            "apelido": "B",
            "slug": "b",
            "posicao": "mei",
            "status": "Provavel",
            "preco": 8,
            "preco_pre_rodada": 8,
            "pontuacao": 4,
            "media": 4,
            "num_jogos": 1,
            "variacao": 0,
            "id_clube": 20,
            "nome_clube": "Clube 20",
            "entrou_em_campo": True,
            "G": 0,
            "A": 1,
            "DS": 1,
            "V": 1,
        },
        {
            "rodada": 2,
            "id_atleta": 2,
            "apelido": "B",
            "slug": "b",
            "posicao": "mei",
            "status": "Provavel",
            "preco": 9,
            "preco_pre_rodada": 8,
            "pontuacao": 6,
            "media": 5,
            "num_jogos": 2,
            "variacao": 1,
            "id_clube": 20,
            "nome_clube": "Clube 20",
            "entrou_em_campo": True,
            "G": 0,
            "A": 0,
            "DS": 2,
            "V": 1,
        },
        {
            "rodada": 3,
            "id_atleta": 2,
            "apelido": "B",
            "slug": "b",
            "posicao": "mei",
            "status": "Provavel",
            "preco": 10,
            "preco_pre_rodada": 9,
            "pontuacao": 7,
            "media": 5.7,
            "num_jogos": 3,
            "variacao": 1,
            "id_clube": 20,
            "nome_clube": "Clube 20",
            "entrou_em_campo": True,
            "G": 0,
            "A": 1,
            "DS": 2,
            "V": 2,
        },
    ]
    frame = pd.DataFrame(rows)
    for scout in DEFAULT_SCOUT_COLUMNS:
        if scout not in frame.columns:
            frame[scout] = 0
    return frame


def test_prediction_features_use_only_prior_rounds() -> None:
    frame = build_prediction_frame(_season_df(), target_round=3)
    player = frame.loc[frame["id_atleta"] == 1].iloc[0]

    assert player["prior_points_mean"] == 5
    assert player["prior_points_roll3"] == 5
    assert player["prior_G_mean"] == 0.5
    assert player["pontuacao"] == 100


def test_prior_points_weighted3_weights_recent_rounds_higher() -> None:
    season_df = _season_df()
    next_round = season_df[season_df["rodada"] == 3].copy()
    next_round["rodada"] = 4
    season_df = pd.concat([season_df, next_round], ignore_index=True)
    season_df.loc[season_df["id_atleta"].eq(1), "pontuacao"] = [2, 4, 10, 999]

    frame = build_prediction_frame(season_df, target_round=4)
    player = frame.loc[frame["id_atleta"] == 1].iloc[0]

    assert player["prior_points_weighted3"] == pytest.approx(6.6)


def test_prior_points_std_measures_volatility() -> None:
    season_df = _season_df()
    season_df.loc[season_df["id_atleta"].eq(1), "pontuacao"] = [0, 10, 999]
    season_df.loc[season_df["id_atleta"].eq(2), "pontuacao"] = [5, 5, 999]

    frame = build_prediction_frame(season_df, target_round=3)
    volatile_player = frame.loc[frame["id_atleta"] == 1].iloc[0]
    steady_player = frame.loc[frame["id_atleta"] == 2].iloc[0]

    assert volatile_player["prior_points_std"] == pytest.approx(7.0710678119)
    assert steady_player["prior_points_std"] == 0


def test_prior_appearance_rate_counts_dnp_correctly() -> None:
    season_df = _season_df()
    round_4 = season_df[season_df["rodada"] == 3].copy()
    round_4["rodada"] = 4
    round_5 = season_df[season_df["rodada"] == 3].copy()
    round_5["rodada"] = 5
    season_df = pd.concat([season_df, round_4, round_5], ignore_index=True)
    season_df.loc[season_df["id_atleta"].eq(1), "entrou_em_campo"] = [
        True,
        False,
        True,
        True,
        True,
    ]

    frame = build_prediction_frame(season_df, target_round=5)
    player = frame.loc[frame["id_atleta"] == 1].iloc[0]

    assert player["prior_appearance_rate"] == pytest.approx(0.75)


def test_prior_appearance_rate_defaults_to_one_when_entry_flag_is_missing() -> None:
    season_df = _season_df().drop(columns=["entrou_em_campo"])

    frame = build_prediction_frame(season_df, target_round=3)
    player = frame.loc[frame["id_atleta"] == 1].iloc[0]

    assert player["prior_appearance_rate"] == 1.0


def test_club_points_roll3_captures_team_form() -> None:
    season_df = _season_df()
    next_round = season_df[season_df["rodada"] == 3].copy()
    next_round["rodada"] = 4
    season_df = pd.concat([season_df, next_round], ignore_index=True)
    season_df.loc[season_df["id_atleta"].eq(1), "pontuacao"] = [10, 20, 30, 999]
    season_df.loc[season_df["id_atleta"].eq(2), "pontuacao"] = [90, 60, 30, 999]

    frame = build_prediction_frame(season_df, target_round=4)
    club_10_player = frame.loc[frame["id_atleta"] == 1].iloc[0]
    club_20_player = frame.loc[frame["id_atleta"] == 2].iloc[0]

    assert club_10_player["club_points_roll3"] == pytest.approx(20)
    assert club_20_player["club_points_roll3"] == pytest.approx(60)


def test_prior_scout_features_use_per_round_deltas_from_cumulative_scouts() -> None:
    season_df = _season_df()
    season_df.loc[season_df["id_atleta"].eq(1), "G"] = [1, 1, 2]
    season_df.loc[season_df["id_atleta"].eq(1), "DS"] = [3, 5, 5]

    frame = build_prediction_frame(season_df, target_round=3)
    player = frame.loc[frame["id_atleta"] == 1].iloc[0]

    assert player["prior_G_mean"] == 0.5
    assert player["prior_DS_mean"] == 2.5


def test_prior_scout_deltas_span_dnp_gaps() -> None:
    season_df = _season_df()
    next_round = season_df.loc[(season_df["id_atleta"] == 1) & (season_df["rodada"] == 3)].copy()
    next_round["rodada"] = 4
    season_df = pd.concat([season_df, next_round], ignore_index=True)
    season_df.loc[season_df["id_atleta"].eq(1), "G"] = [1, 1, 2, 9]
    season_df.loc[(season_df["id_atleta"] == 1) & (season_df["rodada"] == 2), "entrou_em_campo"] = False

    frame = build_prediction_frame(season_df, target_round=4)
    player = frame.loc[frame["id_atleta"] == 1].iloc[0]

    assert player["prior_G_mean"] == 1.0


def test_prior_scout_deltas_clip_cumulative_counter_decreases() -> None:
    season_df = _season_df()
    season_df.loc[season_df["id_atleta"].eq(1), "G"] = [3, 2, 7]

    frame = build_prediction_frame(season_df, target_round=3)
    player = frame.loc[frame["id_atleta"] == 1].iloc[0]

    assert player["prior_G_mean"] == 1.5


def test_training_frame_excludes_target_round_from_feature_history() -> None:
    frame = build_training_frame(_season_df(), target_round=3)
    player_round_2 = frame[(frame["id_atleta"] == 1) & (frame["rodada"] == 2)].iloc[0]

    assert player_round_2["prior_points_mean"] == 2
    assert player_round_2["target"] == 8


def test_feature_fill_does_not_emit_pandas_downcast_future_warning() -> None:
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always", FutureWarning)
        build_training_frame(_season_df(), target_round=3)

    assert not [
        warning
        for warning in captured
        if "Downcasting object dtype arrays" in str(warning.message)
    ]


def test_feature_columns_exist_in_prediction_frame() -> None:
    frame = build_prediction_frame(_season_df(), target_round=3)

    for column in FEATURE_COLUMNS:
        assert column in frame.columns


def test_feature_columns_exclude_target_round_cumulative_fields() -> None:
    assert "preco" not in FEATURE_COLUMNS
    assert "preco_pre_rodada" in FEATURE_COLUMNS
    assert "pontuacao" not in FEATURE_COLUMNS
    assert "target" not in FEATURE_COLUMNS
    assert "media" not in FEATURE_COLUMNS
    assert "num_jogos" not in FEATURE_COLUMNS
    assert "prior_media" in FEATURE_COLUMNS
    assert "prior_num_jogos" in FEATURE_COLUMNS


def test_training_frame_can_filter_to_playable_statuses() -> None:
    season_df = _season_df()
    season_df.loc[(season_df["id_atleta"] == 2) & (season_df["rodada"] == 2), "status"] = "Suspenso"

    frame = build_training_frame(season_df, target_round=3, playable_statuses=("Provavel",))

    assert frame[(frame["id_atleta"] == 2) & (frame["rodada"] == 2)].empty
    assert not frame["status"].eq("Suspenso").any()


def test_prior_features_ignore_rows_where_player_did_not_enter_field() -> None:
    season_df = _season_df()
    season_df.loc[(season_df["id_atleta"] == 1) & (season_df["rodada"] == 2), "entrou_em_campo"] = False
    season_df.loc[(season_df["id_atleta"] == 1) & (season_df["rodada"] == 2), "pontuacao"] = 80

    frame = build_prediction_frame(season_df, target_round=3)
    player = frame.loc[frame["id_atleta"] == 1].iloc[0]

    assert player["prior_points_mean"] == 2
    assert player["prior_points_roll3"] == 2


def test_prior_cumulative_replacements_use_only_prior_rounds() -> None:
    frame = build_prediction_frame(_season_df(), target_round=3)
    player = frame.loc[frame["id_atleta"] == 1].iloc[0]

    assert player["media"] == 36.7
    assert player["num_jogos"] == 3
    assert player["prior_media"] == 5
    assert player["prior_num_jogos"] == 2
