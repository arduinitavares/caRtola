from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal, Protocol, cast

import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge

from cartola.backtesting.models import (
    ExtraTreesPointPredictor,
    HistGradientBoostingPointPredictor,
    RandomForestPointPredictor,
    RidgePointPredictor,
)

ModelId = Literal["random_forest", "extra_trees", "hist_gradient_boosting", "ridge"]


class PointPredictor(Protocol):
    pipeline: object

    def fit(self, frame: pd.DataFrame) -> PointPredictor: ...

    def predict(self, frame: pd.DataFrame) -> pd.Series: ...


@dataclass(frozen=True)
class ModelSpec:
    predictor_type: type[PointPredictor]
    supports_n_jobs: bool
    parameters: dict[str, object]


MODEL_SPECS: dict[ModelId, ModelSpec] = {
    "random_forest": ModelSpec(
        predictor_type=RandomForestPointPredictor,
        supports_n_jobs=True,
        parameters={
            "estimator": RandomForestRegressor,
            "n_estimators": 200,
            "min_samples_leaf": 3,
        },
    ),
    "extra_trees": ModelSpec(
        predictor_type=ExtraTreesPointPredictor,
        supports_n_jobs=True,
        parameters={
            "estimator": ExtraTreesRegressor,
            "n_estimators": 200,
            "min_samples_leaf": 3,
        },
    ),
    "hist_gradient_boosting": ModelSpec(
        predictor_type=HistGradientBoostingPointPredictor,
        supports_n_jobs=False,
        parameters={
            "estimator": HistGradientBoostingRegressor,
            "max_iter": 200,
            "learning_rate": 0.05,
            "min_samples_leaf": 20,
            "l2_regularization": 0.0,
        },
    ),
    "ridge": ModelSpec(
        predictor_type=RidgePointPredictor,
        supports_n_jobs=False,
        parameters={
            "estimator": Ridge,
            "alpha": 1.0,
        },
    ),
}


def resolve_model_id(model_id: str) -> ModelId:
    if model_id not in MODEL_SPECS:
        raise ValueError(f"Unsupported model_id: {model_id!r}")
    return cast(ModelId, model_id)


def create_point_predictor(
    *,
    model_id: str,
    random_seed: int,
    feature_columns: list[str],
    n_jobs: int,
    model_params: Mapping[str, object] | None = None,
) -> PointPredictor:
    resolved_model_id = resolve_model_id(model_id)
    spec = MODEL_SPECS[resolved_model_id]
    validated_params = _validate_model_param_overrides(resolved_model_id, model_params)

    return spec.predictor_type(
        random_seed=random_seed,
        feature_columns=feature_columns,
        n_jobs=n_jobs,
        model_params=validated_params,
    )


def effective_model_parameters(
    model_id: str,
    model_params: Mapping[str, object] | None = None,
) -> dict[str, object]:
    resolved_model_id = resolve_model_id(model_id)
    defaults = dict(MODEL_SPECS[resolved_model_id].parameters)
    overrides = _validate_model_param_overrides(resolved_model_id, model_params)
    return {**defaults, **overrides}


def _validate_model_param_overrides(
    model_id: ModelId,
    model_params: Mapping[str, object] | None,
) -> dict[str, object]:
    if not model_params:
        return {}
    if model_id != "ridge":
        raise ValueError("Model parameter overrides are only supported for ridge in v1")

    allowed = {"alpha"}
    unknown = sorted(set(model_params) - allowed)
    if unknown:
        raise ValueError(f"Unsupported model parameter for ridge: {unknown[0]}")

    alpha = float(cast("object", model_params["alpha"]))
    if alpha <= 0:
        raise ValueError("ridge alpha must be positive")
    return {"alpha": alpha}


def model_n_jobs_for_metadata(model_id: str, *, requested_n_jobs: int) -> int | None:
    spec = MODEL_SPECS[resolve_model_id(model_id)]
    return requested_n_jobs if spec.supports_n_jobs else None
