from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

import pandas as pd

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


MODEL_SPECS: dict[ModelId, ModelSpec] = {
    "random_forest": ModelSpec(
        predictor_type=RandomForestPointPredictor,
        supports_n_jobs=True,
    ),
    "extra_trees": ModelSpec(
        predictor_type=ExtraTreesPointPredictor,
        supports_n_jobs=True,
    ),
    "hist_gradient_boosting": ModelSpec(
        predictor_type=HistGradientBoostingPointPredictor,
        supports_n_jobs=False,
    ),
    "ridge": ModelSpec(
        predictor_type=RidgePointPredictor,
        supports_n_jobs=False,
    ),
}


def create_point_predictor(
    *,
    model_id: ModelId,
    random_seed: int,
    feature_columns: list[str],
    n_jobs: int,
) -> PointPredictor:
    try:
        spec = MODEL_SPECS[model_id]
    except KeyError as exc:
        raise ValueError(f"Unsupported model_id: {model_id}") from exc

    return spec.predictor_type(
        random_seed=random_seed,
        feature_columns=feature_columns,
        n_jobs=n_jobs,
    )


def model_n_jobs_for_metadata(model_id: ModelId, *, requested_n_jobs: int) -> int | None:
    try:
        spec = MODEL_SPECS[model_id]
    except KeyError as exc:
        raise ValueError(f"Unsupported model_id: {model_id}") from exc
    return requested_n_jobs if spec.supports_n_jobs else None
