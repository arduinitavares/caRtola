import pandas as pd

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


def test_training_frame_excludes_target_round_from_feature_history() -> None:
    frame = build_training_frame(_season_df(), target_round=3)
    player_round_2 = frame[(frame["id_atleta"] == 1) & (frame["rodada"] == 2)].iloc[0]

    assert player_round_2["prior_points_mean"] == 2
    assert player_round_2["target"] == 8


def test_feature_columns_exist_in_prediction_frame() -> None:
    frame = build_prediction_frame(_season_df(), target_round=3)

    for column in FEATURE_COLUMNS:
        assert column in frame.columns


def test_feature_columns_exclude_target_round_cumulative_fields() -> None:
    assert "media" not in FEATURE_COLUMNS
    assert "num_jogos" not in FEATURE_COLUMNS
    assert "prior_media" in FEATURE_COLUMNS
    assert "prior_num_jogos" in FEATURE_COLUMNS


def test_prior_cumulative_replacements_use_only_prior_rounds() -> None:
    frame = build_prediction_frame(_season_df(), target_round=3)
    player = frame.loc[frame["id_atleta"] == 1].iloc[0]

    assert player["media"] == 36.7
    assert player["num_jogos"] == 3
    assert player["prior_media"] == 5
    assert player["prior_num_jogos"] == 2
