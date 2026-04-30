import pandas as pd
import pytest
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import OneHotEncoder

from cartola.backtesting.features import FEATURE_COLUMNS
from cartola.backtesting.model_registry import (
    MODEL_SPECS,
    create_point_predictor,
    model_n_jobs_for_metadata,
)


def test_model_registry_contains_exact_v1_models() -> None:
    assert tuple(MODEL_SPECS) == ("random_forest", "extra_trees", "hist_gradient_boosting", "ridge")


def test_random_forest_spec_matches_contract() -> None:
    model = create_point_predictor(
        model_id="random_forest",
        random_seed=7,
        feature_columns=FEATURE_COLUMNS,
        n_jobs=3,
    )

    estimator = model.pipeline.named_steps["model"]
    assert isinstance(estimator, RandomForestRegressor)
    assert estimator.n_estimators == 200
    assert estimator.min_samples_leaf == 3
    assert estimator.random_state == 7
    assert estimator.n_jobs == 3


def test_extra_trees_spec_matches_contract() -> None:
    model = create_point_predictor(
        model_id="extra_trees",
        random_seed=7,
        feature_columns=FEATURE_COLUMNS,
        n_jobs=2,
    )

    estimator = model.pipeline.named_steps["model"]
    assert isinstance(estimator, ExtraTreesRegressor)
    assert estimator.n_estimators == 200
    assert estimator.min_samples_leaf == 3
    assert estimator.random_state == 7
    assert estimator.n_jobs == 2


def test_hist_gradient_boosting_uses_dense_one_hot() -> None:
    model = create_point_predictor(
        model_id="hist_gradient_boosting",
        random_seed=7,
        feature_columns=FEATURE_COLUMNS,
        n_jobs=99,
    )

    estimator = model.pipeline.named_steps["model"]
    assert isinstance(estimator, HistGradientBoostingRegressor)
    assert estimator.max_iter == 200
    assert estimator.learning_rate == 0.05
    assert estimator.min_samples_leaf == 20
    assert estimator.random_state == 7

    categorical_pipeline = model.pipeline.named_steps["preprocess"].transformers[1][1]
    encoder = categorical_pipeline.named_steps["encoder"]
    assert isinstance(encoder, OneHotEncoder)
    assert encoder.sparse_output is False


def test_ridge_spec_matches_contract() -> None:
    model = create_point_predictor(
        model_id="ridge",
        random_seed=7,
        feature_columns=FEATURE_COLUMNS,
        n_jobs=99,
    )

    estimator = model.pipeline.named_steps["model"]
    assert isinstance(estimator, Ridge)
    assert estimator.alpha == 1.0


@pytest.mark.parametrize("model_id", ["hist_gradient_boosting", "ridge"])
def test_non_parallel_models_record_null_n_jobs(model_id: str) -> None:
    assert model_n_jobs_for_metadata(model_id, requested_n_jobs=4) is None


@pytest.mark.parametrize("model_id", ["random_forest", "extra_trees"])
def test_parallel_models_record_effective_n_jobs(model_id: str) -> None:
    assert model_n_jobs_for_metadata(model_id, requested_n_jobs=4) == 4


def test_unknown_model_id_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unsupported model_id"):
        create_point_predictor(
            model_id="xgboost",
            random_seed=7,
            feature_columns=FEATURE_COLUMNS,
            n_jobs=1,
        )


def test_created_models_fit_and_predict() -> None:
    frame = pd.DataFrame(
        {
            "preco_pre_rodada": [10.0, 11.0, 8.0, 8.5, 9.0, 7.0],
            "id_clube": [10, 10, 20, 20, 30, 30],
            "rodada": [2, 3, 2, 3, 2, 3],
            "posicao": ["ata", "ata", "mei", "mei", "zag", "zag"],
            "prior_appearances": [1, 2, 1, 2, 1, 2],
            "prior_appearance_rate": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            "prior_points_mean": [2.0, 5.0, 4.0, 5.0, 3.0, 4.0],
            "prior_points_roll3": [2.0, 5.0, 4.0, 5.0, 3.0, 4.0],
            "prior_points_roll5": [2.0, 5.0, 4.0, 5.0, 3.0, 4.0],
            "prior_points_weighted3": [2.0, 5.5, 4.0, 5.5, 3.0, 4.0],
            "prior_points_std": [0.0, 4.24, 0.0, 1.41, 0.0, 1.0],
            "prior_price_mean": [10.0, 10.5, 8.0, 8.5, 9.0, 8.0],
            "prior_variation_mean": [0.0, 0.5, 0.0, 0.5, 0.0, 0.1],
            "club_points_roll3": [30.0, 32.0, 24.0, 25.0, 20.0, 21.0],
            "is_home": [1, 0, 1, 0, 1, 0],
            "opponent_club_points_roll3": [24.0, 25.0, 30.0, 32.0, 22.0, 23.0],
            "prior_media": [2.0, 5.0, 4.0, 5.0, 3.0, 4.0],
            "prior_num_jogos": [1, 2, 1, 2, 1, 2],
            "target": [8.0, 10.0, 6.0, 7.0, 3.0, 5.0],
        }
    )
    for scout in (
        "G",
        "A",
        "DS",
        "SG",
        "CA",
        "FC",
        "FS",
        "FF",
        "FD",
        "FT",
        "I",
        "GS",
        "DE",
        "DP",
        "V",
        "CV",
        "PP",
        "PS",
        "PC",
        "GC",
    ):
        frame[f"prior_{scout}_mean"] = 0.0
    frame = frame[[*FEATURE_COLUMNS, "target"]]

    for model_id in MODEL_SPECS:
        model = create_point_predictor(
            model_id=model_id,
            random_seed=7,
            feature_columns=FEATURE_COLUMNS,
            n_jobs=1,
        ).fit(frame)
        predictions = model.predict(frame)
        assert len(predictions) == len(frame)
        assert predictions.notna().all()
