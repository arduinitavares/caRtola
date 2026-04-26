import pandas as pd

from cartola.backtesting.models import BaselinePredictor


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
