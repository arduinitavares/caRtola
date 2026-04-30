from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


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


class RandomForestPointPredictor:
    def __init__(
        self,
        random_seed: int = 123,
        feature_columns: list[str] | None = None,
        n_jobs: int = -1,
    ) -> None:
        if feature_columns is None:
            raise ValueError("feature_columns must be provided")

        self.feature_columns = feature_columns
        self.n_jobs = n_jobs
        numeric_features = [column for column in self.feature_columns if column != "posicao"]
        categorical_features = ["posicao"] if "posicao" in self.feature_columns else []

        self.pipeline = Pipeline(
            steps=[
                (
                    "preprocess",
                    ColumnTransformer(
                        transformers=[
                            ("numeric", SimpleImputer(strategy="median"), numeric_features),
                            (
                                "categorical",
                                Pipeline(
                                    steps=[
                                        ("imputer", SimpleImputer(strategy="most_frequent")),
                                        ("encoder", OneHotEncoder(handle_unknown="ignore")),
                                    ]
                                ),
                                categorical_features,
                            ),
                        ]
                    ),
                ),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=200,
                        min_samples_leaf=3,
                        random_state=random_seed,
                        n_jobs=n_jobs,
                    ),
                ),
            ]
        )

    def fit(self, frame: pd.DataFrame) -> RandomForestPointPredictor:
        self.pipeline.fit(frame[self.feature_columns], frame["target"])
        return self

    def predict(self, frame: pd.DataFrame) -> pd.Series:
        predictions = self.pipeline.predict(frame[self.feature_columns])
        return pd.Series(predictions, index=frame.index, dtype=float)
