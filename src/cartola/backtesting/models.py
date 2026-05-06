from __future__ import annotations

import importlib
from collections.abc import Mapping
from time import perf_counter
from typing import Self, cast

import pandas as pd
from scipy import sparse
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class BaselinePredictor:
    def __init__(self) -> None:
        self.player_means_: pd.Series | None = None
        self.position_means_: pd.Series | None = None
        self.global_mean_: float = 0.0
        self.known_player_ids_: set[object] = set()

    def fit(self, frame: pd.DataFrame) -> BaselinePredictor:
        self.player_means_ = frame.groupby("id_atleta")["target"].mean()
        self.position_means_ = frame.groupby("posicao")["target"].mean()
        self.global_mean_ = float(frame["target"].mean()) if not frame.empty else 0.0
        self.known_player_ids_ = set(frame["id_atleta"].dropna().unique())
        return self

    def predict(self, frame: pd.DataFrame) -> pd.Series:
        if self.player_means_ is None or self.position_means_ is None:
            raise RuntimeError("BaselinePredictor must be fitted before predict().")

        predictions = pd.Series(index=frame.index, dtype=float)
        known_mask = frame["id_atleta"].isin(self.known_player_ids_)
        if "prior_points_mean" in frame.columns:
            predictions.loc[known_mask] = frame.loc[known_mask, "prior_points_mean"]

        learned_player_mean = frame["id_atleta"].map(self.player_means_)
        position_fallback = frame["posicao"].map(self.position_means_).fillna(self.global_mean_)
        predictions = predictions.fillna(learned_player_mean).fillna(position_fallback)
        return predictions.astype(float)


class SklearnPointPredictor:
    def __init__(
        self,
        random_seed: int = 123,
        feature_columns: list[str] | None = None,
        n_jobs: int = -1,
        model_params: Mapping[str, object] | None = None,
    ) -> None:
        if feature_columns is None:
            raise ValueError("feature_columns must be provided")

        self.feature_columns = feature_columns
        self.n_jobs = n_jobs
        self.model_params = dict(model_params or {})
        self.last_fit_profile_: dict[str, object] = {}
        self.last_predict_profile_: dict[str, object] = {}
        numeric_features = [column for column in self.feature_columns if column != "posicao"]
        categorical_features = ["posicao"] if "posicao" in self.feature_columns else []

        self.pipeline = Pipeline(
            steps=[
                (
                    "preprocess",
                    ColumnTransformer(
                        transformers=[
                            ("numeric", self._make_numeric_transformer(), numeric_features),
                            (
                                "categorical",
                                Pipeline(
                                    steps=[
                                        ("imputer", SimpleImputer(strategy="most_frequent")),
                                        ("encoder", self._make_categorical_encoder()),
                                    ]
                                ),
                                categorical_features,
                            ),
                        ]
                    ),
                ),
                ("model", self._make_model(random_seed=random_seed, n_jobs=n_jobs)),
            ]
        )

    def _make_numeric_transformer(self) -> object:
        return SimpleImputer(strategy="median")

    def _make_categorical_encoder(self) -> OneHotEncoder:
        return OneHotEncoder(handle_unknown="ignore")

    def _make_model(self, *, random_seed: int, n_jobs: int) -> object:
        raise NotImplementedError

    def fit(self, frame: pd.DataFrame) -> Self:
        x_train = frame[self.feature_columns]
        y_train = frame["target"]
        preprocess = self.pipeline.named_steps["preprocess"]
        model = self.pipeline.named_steps["model"]

        started = perf_counter()
        transformed = preprocess.fit_transform(x_train, y_train)
        preprocess_seconds = perf_counter() - started

        started = perf_counter()
        model.fit(transformed, y_train)
        model_fit_seconds = perf_counter() - started

        self.last_fit_profile_ = {
            "training_rows": len(frame),
            "training_columns": len(frame.columns),
            "feature_count": len(self.feature_columns),
            "preprocess_fit_transform_seconds": preprocess_seconds,
            "model_fit_seconds": model_fit_seconds,
            "model_n_iter": getattr(model, "n_iter_", None),
            **_transformed_matrix_profile(transformed, prefix="transformed_feature"),
        }
        return self

    def predict(self, frame: pd.DataFrame) -> pd.Series:
        preprocess = self.pipeline.named_steps["preprocess"]
        model = self.pipeline.named_steps["model"]

        started = perf_counter()
        transformed = preprocess.transform(frame[self.feature_columns])
        transform_seconds = perf_counter() - started

        started = perf_counter()
        predictions = model.predict(transformed)
        predict_seconds = perf_counter() - started

        self.last_predict_profile_ = {
            "prediction_rows": len(frame),
            "candidate_transform_seconds": transform_seconds,
            "model_predict_seconds": predict_seconds,
            **_transformed_matrix_profile(transformed, prefix="candidate_transformed_feature"),
        }
        return pd.Series(predictions, index=frame.index, dtype=float)


class RandomForestPointPredictor(SklearnPointPredictor):
    def _make_model(self, *, random_seed: int, n_jobs: int) -> RandomForestRegressor:
        return RandomForestRegressor(
            n_estimators=200,
            min_samples_leaf=3,
            random_state=random_seed,
            n_jobs=n_jobs,
        )


class ExtraTreesPointPredictor(SklearnPointPredictor):
    def _make_model(self, *, random_seed: int, n_jobs: int) -> ExtraTreesRegressor:
        return ExtraTreesRegressor(
            n_estimators=200,
            min_samples_leaf=3,
            random_state=random_seed,
            n_jobs=n_jobs,
        )


class HistGradientBoostingPointPredictor(SklearnPointPredictor):
    def _make_categorical_encoder(self) -> OneHotEncoder:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)

    def _make_model(self, *, random_seed: int, n_jobs: int) -> HistGradientBoostingRegressor:
        return HistGradientBoostingRegressor(
            max_iter=200,
            learning_rate=0.05,
            min_samples_leaf=20,
            random_state=random_seed,
        )


class RidgePointPredictor(SklearnPointPredictor):
    def _make_numeric_transformer(self) -> Pipeline:
        return Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )

    def _make_model(self, *, random_seed: int, n_jobs: int) -> Ridge:
        alpha = float(self.model_params.get("alpha", 1.0))
        return Ridge(alpha=alpha)


class XGBoostPointPredictor(SklearnPointPredictor):
    def _make_model(self, *, random_seed: int, n_jobs: int) -> object:
        xgb_regressor = _load_xgb_regressor()
        return xgb_regressor(
            objective="reg:squarederror",
            tree_method="hist",
            n_estimators=int(self.model_params["n_estimators"]),
            max_depth=int(self.model_params["max_depth"]),
            learning_rate=float(self.model_params["learning_rate"]),
            min_child_weight=float(self.model_params["min_child_weight"]),
            subsample=float(self.model_params["subsample"]),
            colsample_bytree=float(self.model_params["colsample_bytree"]),
            reg_lambda=float(self.model_params["reg_lambda"]),
            reg_alpha=float(self.model_params["reg_alpha"]),
            gamma=float(self.model_params.get("gamma", 0.0)),
            random_state=random_seed,
            n_jobs=n_jobs,
            verbosity=0,
        )


def _transformed_matrix_profile(matrix: object, *, prefix: str) -> dict[str, object]:
    shape = getattr(matrix, "shape", (None, None))
    rows = int(shape[0]) if len(shape) >= 1 and shape[0] is not None else None
    columns = int(shape[1]) if len(shape) >= 2 and shape[1] is not None else None
    return {
        f"{prefix}_rows": rows,
        f"{prefix}_columns": columns,
        f"{prefix}_type": type(matrix).__name__,
        f"{prefix}_sparse": bool(sparse.issparse(matrix)),
        f"{prefix}_mb": _matrix_size_mb(matrix),
    }


def _matrix_size_mb(matrix: object) -> float | None:
    if sparse.issparse(matrix):
        sparse_matrix = sparse.csr_matrix(matrix)
        return float(
            (sparse_matrix.data.nbytes + sparse_matrix.indices.nbytes + sparse_matrix.indptr.nbytes) / (1024 * 1024)
        )
    nbytes = getattr(matrix, "nbytes", None)
    if nbytes is None:
        return None
    return float(nbytes) / (1024 * 1024)


def _load_xgb_regressor() -> type[object]:
    try:
        xgboost = importlib.import_module("xgboost")
        xgb_regressor = getattr(xgboost, "XGBRegressor")
    except Exception as exc:
        raise RuntimeError(
            "XGBoost is installed as an optional research dependency, but its native runtime could not be loaded. "
            "On macOS, install the OpenMP runtime with `brew install libomp` before running XGBoost experiments."
        ) from exc
    return cast(type[object], xgb_regressor)
