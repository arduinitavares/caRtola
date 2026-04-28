import pandas as pd
import pytest

from cartola.backtesting.config import DEFAULT_SCOUT_COLUMNS
from cartola.backtesting.features import FEATURE_COLUMNS, FOOTYSTATS_PPG_FEATURE_COLUMNS
from cartola.backtesting.models import BaselinePredictor, RandomForestPointPredictor


def test_baseline_predictor_uses_prior_player_mean_with_position_fallback() -> None:
    train = pd.DataFrame(
        {
            "id_atleta": [1, 2],
            "posicao": ["ata", "mei"],
            "target": [6.0, 4.0],
            "prior_points_mean": [5.0, 3.0],
        }
    )
    predict = pd.DataFrame(
        {
            "id_atleta": [1, 3],
            "posicao": ["ata", "mei"],
            "prior_points_mean": [5.0, 99.0],
        }
    )

    model = BaselinePredictor().fit(train)
    predictions = model.predict(predict)

    assert predictions.tolist() == [5.0, 4.0]


def test_random_forest_point_predictor_fit_predict_smoke() -> None:
    train = _model_frame()
    predict = train.drop(columns=["target"]).copy()

    model = RandomForestPointPredictor(random_seed=7, feature_columns=FEATURE_COLUMNS).fit(train)
    predictions = model.predict(predict)

    assert len(predictions) == len(predict)
    assert predictions.notna().all()


def test_random_forest_point_predictor_uses_explicit_feature_columns() -> None:
    feature_columns = [*FEATURE_COLUMNS, *FOOTYSTATS_PPG_FEATURE_COLUMNS]
    train = _model_frame()
    train["footystats_team_pre_match_ppg"] = [1.0, 2.0, 1.5, 2.5]
    train["footystats_opponent_pre_match_ppg"] = [2.0, 1.0, 2.5, 1.5]
    train["footystats_ppg_diff"] = [-1.0, 1.0, -1.0, 1.0]
    predict = train.drop(columns=["target"]).copy()

    model = RandomForestPointPredictor(random_seed=7, feature_columns=feature_columns).fit(train)
    predictions = model.predict(predict)

    assert model.feature_columns == feature_columns
    assert len(predictions) == len(predict)
    assert predictions.notna().all()


def test_random_forest_point_predictor_requires_feature_columns() -> None:
    with pytest.raises(ValueError, match="feature_columns must be provided"):
        RandomForestPointPredictor()


def _model_frame() -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "preco_pre_rodada": [10.0, 11.0, 8.0, 8.5],
            "id_clube": [10, 10, 20, 20],
            "rodada": [2, 3, 2, 3],
            "posicao": ["ata", "ata", "mei", "mei"],
            "prior_appearances": [1, 2, 1, 2],
            "prior_appearance_rate": [1.0, 1.0, 1.0, 1.0],
            "prior_points_mean": [2.0, 5.0, 4.0, 5.0],
            "prior_points_roll3": [2.0, 5.0, 4.0, 5.0],
            "prior_points_roll5": [2.0, 5.0, 4.0, 5.0],
            "prior_points_weighted3": [2.0, 5.5, 4.0, 5.5],
            "prior_points_std": [0.0, 4.24, 0.0, 1.41],
            "prior_price_mean": [10.0, 10.5, 8.0, 8.5],
            "prior_variation_mean": [0.0, 0.5, 0.0, 0.5],
            "club_points_roll3": [30.0, 32.0, 24.0, 25.0],
            "is_home": [1, 0, 1, 0],
            "opponent_club_points_roll3": [24.0, 25.0, 30.0, 32.0],
            "prior_media": [2.0, 5.0, 4.0, 5.0],
            "prior_num_jogos": [1, 2, 1, 2],
            "target": [8.0, 10.0, 6.0, 7.0],
        }
    )
    for scout in DEFAULT_SCOUT_COLUMNS:
        frame[f"prior_{scout}_mean"] = 0.0
    return frame[[*FEATURE_COLUMNS, "target"]]
