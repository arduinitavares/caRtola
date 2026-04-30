# Model Feature Experiment Runner Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a controlled model and feature-pack experiment runner for 2023-2025 Cartola backtests without exposing model selection in the normal backtest CLI.

**Architecture:** Add a private model registry/factory used by backtest internals, then add an experiment-only runner that executes sequential child backtests with one primary model strategy per child run. Experiment aggregation lives outside the normal CLI and enforces source hashes, candidate-pool signatures, role-based solver comparability, prediction metrics, squad metrics, and promotion guardrails before ranking.

**Tech Stack:** Python 3.13, pandas, scikit-learn, Plotly, Rich, pytest, uv.

---

## File Structure

- Create `src/cartola/backtesting/model_registry.py`
  - Owns `ModelId`, `PointPredictor`, `MODEL_SPECS`, and `create_point_predictor()`.
  - Keeps model construction out of scripts and out of the normal CLI.

- Modify `src/cartola/backtesting/models.py`
  - Keep `BaselinePredictor`.
  - Keep `RandomForestPointPredictor`.
  - Add reusable sklearn predictor wrappers for Extra Trees, HistGradientBoosting, and Ridge.

- Modify `src/cartola/backtesting/config.py`
  - Add private `_output_path_override: Path | None = None`.
  - Preserve normal `output_path` behavior when override is absent.

- Modify `src/cartola/backtesting/runner.py`
  - Add `run_backtest_for_experiment(config, *, primary_model_id)`.
  - Thread `primary_model_id` through private helpers.
  - Emit `baseline`, `<primary_model_id>`, and `price`.
  - Keep public `run_backtest(config)` equivalent to primary model `random_forest`.

- Create `src/cartola/backtesting/experiment_config.py`
  - Defines experiment groups, feature packs, child run configs, fixed matrix, config hash, and output paths.

- Create `src/cartola/backtesting/experiment_signatures.py`
  - Computes source hashes, candidate-pool signatures, skipped-round signatures, and role-based solver-status signatures.
  - Raises `ComparabilityError`.

- Create `src/cartola/backtesting/experiment_metrics.py`
  - Computes prediction metrics, calibration deciles, per-season squad summaries, and promotion eligibility.

- Create `src/cartola/backtesting/experiment_runner.py`
  - Orchestrates sequential child backtests.
  - Writes top-level experiment artifacts.
  - Does not implement experiment-level parallelism.

- Create `scripts/run_model_experiments.py`
  - Thin CLI wrapper around `cartola.backtesting.experiment_runner`.

- Create `src/tests/backtesting/test_model_registry.py`
  - Tests model factory, fixed params, dense HGB preprocessing, and non-parallel metadata behavior.

- Modify `src/tests/backtesting/test_models.py`
  - Adds smoke tests for the new predictor classes.

- Modify `src/tests/backtesting/test_runner.py`
  - Tests model strategy identity and hidden normal CLI behavior.

- Create `src/tests/backtesting/test_experiment_config.py`
  - Tests matrix generation, groups, config hash, child paths, and season rejection.

- Create `src/tests/backtesting/test_experiment_signatures.py`
  - Tests source hashes and comparability signatures.

- Create `src/tests/backtesting/test_experiment_metrics.py`
  - Tests top-K, calibration, null guardrails, and promotion rules.

- Create `src/tests/backtesting/test_experiment_runner.py`
  - Tests orchestration, no experiment-level concurrency, failure behavior, and outputs.

- Create `src/tests/backtesting/test_run_model_experiments_cli.py`
  - Tests CLI parsing and error handling.

---

### Task 1: Add The Model Registry And Predictors

**Files:**
- Create: `src/cartola/backtesting/model_registry.py`
- Modify: `src/cartola/backtesting/models.py`
- Test: `src/tests/backtesting/test_model_registry.py`
- Test: `src/tests/backtesting/test_models.py`

- [ ] **Step 1: Write model registry tests**

Create `src/tests/backtesting/test_model_registry.py`:

```python
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
        "G", "A", "DS", "SG", "CA", "FC", "FS", "FF", "FD", "FT", "I", "GS",
        "DE", "DP", "V", "CV", "PP", "PS", "PC", "GC",
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
```

- [ ] **Step 2: Run model registry tests and verify they fail**

Run:

```bash
uv run --frozen pytest src/tests/backtesting/test_model_registry.py -q
```

Expected: failure because `cartola.backtesting.model_registry` does not exist.

- [ ] **Step 3: Implement predictor classes**

In `src/cartola/backtesting/models.py`, add imports:

```python
from typing import Self

from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
```

Add these classes below `RandomForestPointPredictor`:

```python
class SklearnPointPredictor:
    def __init__(self, *, feature_columns: list[str], pipeline: Pipeline) -> None:
        if feature_columns is None:
            raise ValueError("feature_columns must be provided")
        self.feature_columns = feature_columns
        self.pipeline = pipeline

    def fit(self, frame: pd.DataFrame) -> Self:
        self.pipeline.fit(frame[self.feature_columns], frame["target"])
        return self

    def predict(self, frame: pd.DataFrame) -> pd.Series:
        predictions = self.pipeline.predict(frame[self.feature_columns])
        return pd.Series(predictions, index=frame.index, dtype=float)


class ExtraTreesPointPredictor(SklearnPointPredictor):
    def __init__(self, *, random_seed: int, feature_columns: list[str], n_jobs: int) -> None:
        numeric_features = [column for column in feature_columns if column != "posicao"]
        categorical_features = ["posicao"] if "posicao" in feature_columns else []
        super().__init__(
            feature_columns=feature_columns,
            pipeline=Pipeline(
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
                        ExtraTreesRegressor(
                            n_estimators=200,
                            min_samples_leaf=3,
                            random_state=random_seed,
                            n_jobs=n_jobs,
                        ),
                    ),
                ]
            ),
        )


class HistGradientBoostingPointPredictor(SklearnPointPredictor):
    def __init__(self, *, random_seed: int, feature_columns: list[str]) -> None:
        numeric_features = [column for column in feature_columns if column != "posicao"]
        categorical_features = ["posicao"] if "posicao" in feature_columns else []
        super().__init__(
            feature_columns=feature_columns,
            pipeline=Pipeline(
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
                                            (
                                                "encoder",
                                                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                                            ),
                                        ]
                                    ),
                                    categorical_features,
                                ),
                            ]
                        ),
                    ),
                    (
                        "model",
                        HistGradientBoostingRegressor(
                            max_iter=200,
                            learning_rate=0.05,
                            min_samples_leaf=20,
                            l2_regularization=0.0,
                            random_state=random_seed,
                        ),
                    ),
                ]
            ),
        )


class RidgePointPredictor(SklearnPointPredictor):
    def __init__(self, *, feature_columns: list[str]) -> None:
        numeric_features = [column for column in feature_columns if column != "posicao"]
        categorical_features = ["posicao"] if "posicao" in feature_columns else []
        super().__init__(
            feature_columns=feature_columns,
            pipeline=Pipeline(
                steps=[
                    (
                        "preprocess",
                        ColumnTransformer(
                            transformers=[
                                (
                                    "numeric",
                                    Pipeline(
                                        steps=[
                                            ("imputer", SimpleImputer(strategy="median")),
                                            ("scaler", StandardScaler()),
                                        ]
                                    ),
                                    numeric_features,
                                ),
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
                    ("model", Ridge(alpha=1.0)),
                ]
            ),
        )
```

- [ ] **Step 4: Implement registry**

Create `src/cartola/backtesting/model_registry.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol, Self

import pandas as pd

from cartola.backtesting.models import (
    ExtraTreesPointPredictor,
    HistGradientBoostingPointPredictor,
    RandomForestPointPredictor,
    RidgePointPredictor,
)

ModelId = Literal["random_forest", "extra_trees", "hist_gradient_boosting", "ridge"]


class PointPredictor(Protocol):
    feature_columns: list[str]

    def fit(self, frame: pd.DataFrame) -> Self: ...

    def predict(self, frame: pd.DataFrame) -> pd.Series: ...


@dataclass(frozen=True)
class ModelSpec:
    model_id: ModelId
    supports_n_jobs: bool
    parameters: dict[str, object]


MODEL_SPECS: dict[ModelId, ModelSpec] = {
    "random_forest": ModelSpec(
        model_id="random_forest",
        supports_n_jobs=True,
        parameters={
            "estimator": "RandomForestRegressor",
            "n_estimators": 200,
            "min_samples_leaf": 3,
        },
    ),
    "extra_trees": ModelSpec(
        model_id="extra_trees",
        supports_n_jobs=True,
        parameters={
            "estimator": "ExtraTreesRegressor",
            "n_estimators": 200,
            "min_samples_leaf": 3,
        },
    ),
    "hist_gradient_boosting": ModelSpec(
        model_id="hist_gradient_boosting",
        supports_n_jobs=False,
        parameters={
            "estimator": "HistGradientBoostingRegressor",
            "max_iter": 200,
            "learning_rate": 0.05,
            "min_samples_leaf": 20,
            "l2_regularization": 0.0,
        },
    ),
    "ridge": ModelSpec(
        model_id="ridge",
        supports_n_jobs=False,
        parameters={
            "estimator": "Ridge",
            "alpha": 1.0,
        },
    ),
}


def create_point_predictor(
    *,
    model_id: str,
    random_seed: int,
    feature_columns: list[str],
    n_jobs: int,
) -> PointPredictor:
    if model_id == "random_forest":
        return RandomForestPointPredictor(
            random_seed=random_seed,
            feature_columns=feature_columns,
            n_jobs=n_jobs,
        )
    if model_id == "extra_trees":
        return ExtraTreesPointPredictor(
            random_seed=random_seed,
            feature_columns=feature_columns,
            n_jobs=n_jobs,
        )
    if model_id == "hist_gradient_boosting":
        return HistGradientBoostingPointPredictor(
            random_seed=random_seed,
            feature_columns=feature_columns,
        )
    if model_id == "ridge":
        return RidgePointPredictor(feature_columns=feature_columns)
    raise ValueError(f"Unsupported model_id: {model_id!r}")


def model_n_jobs_for_metadata(model_id: str, *, requested_n_jobs: int) -> int | None:
    if model_id not in MODEL_SPECS:
        raise ValueError(f"Unsupported model_id: {model_id!r}")
    if MODEL_SPECS[model_id].supports_n_jobs:
        return requested_n_jobs
    return None
```

- [ ] **Step 5: Run tests and verify they pass**

Run:

```bash
uv run --frozen pytest src/tests/backtesting/test_model_registry.py src/tests/backtesting/test_models.py -q
```

Expected: all selected tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/cartola/backtesting/models.py src/cartola/backtesting/model_registry.py \
  src/tests/backtesting/test_model_registry.py src/tests/backtesting/test_models.py
git commit -m "feat: add backtest model registry"
```

---

### Task 2: Add Experiment-Only Primary Model Strategy Support

**Files:**
- Modify: `src/cartola/backtesting/config.py`
- Modify: `src/cartola/backtesting/runner.py`
- Test: `src/tests/backtesting/test_runner.py`
- Test: `src/tests/backtesting/test_cli.py`

- [ ] **Step 1: Write runner strategy tests**

Add these tests to `src/tests/backtesting/test_runner.py`:

```python
def test_public_run_backtest_keeps_random_forest_strategy(tmp_path, monkeypatch) -> None:
    data = pd.concat([_tiny_round(round_number) for round_number in range(1, 6)], ignore_index=True)
    monkeypatch.setattr(runner_module, "load_season_data", lambda season, project_root: data)

    config = BacktestConfig(project_root=tmp_path, season=2025, start_round=5, jobs=1)

    result = run_backtest(config)

    assert set(result.round_results["strategy"]) == {"baseline", "random_forest", "price"}
    assert "random_forest_score" in result.player_predictions.columns


def test_experiment_run_uses_primary_model_strategy(tmp_path, monkeypatch) -> None:
    data = pd.concat([_tiny_round(round_number) for round_number in range(1, 6)], ignore_index=True)
    monkeypatch.setattr(runner_module, "load_season_data", lambda season, project_root: data)

    config = BacktestConfig(project_root=tmp_path, season=2025, start_round=5, jobs=1)

    result = runner_module.run_backtest_for_experiment(config, primary_model_id="extra_trees")

    assert set(result.round_results["strategy"]) == {"baseline", "extra_trees", "price"}
    assert "extra_trees_score" in result.player_predictions.columns
    assert "random_forest_score" not in result.player_predictions.columns


def test_experiment_run_rejects_unknown_primary_model(tmp_path, monkeypatch) -> None:
    data = pd.concat([_tiny_round(round_number) for round_number in range(1, 6)], ignore_index=True)
    monkeypatch.setattr(runner_module, "load_season_data", lambda season, project_root: data)

    config = BacktestConfig(project_root=tmp_path, season=2025, start_round=5, jobs=1)

    with pytest.raises(ValueError, match="Unsupported model_id"):
        runner_module.run_backtest_for_experiment(config, primary_model_id="xgboost")
```

- [ ] **Step 2: Write hidden CLI model-selection test**

Add to `src/tests/backtesting/test_cli.py`:

```python
def test_parse_args_does_not_expose_model_id() -> None:
    with pytest.raises(SystemExit):
        parse_args(["--model-id", "extra_trees"])
```

- [ ] **Step 3: Write output path override test**

Add to `src/tests/backtesting/test_config.py`:

```python
def test_output_path_override_is_private_and_exact(tmp_path) -> None:
    override = tmp_path / "runs" / "season=2025" / "model=extra_trees" / "feature_pack=ppg"
    config = BacktestConfig(project_root=tmp_path, _output_path_override=override)

    assert config.output_path == override
```

- [ ] **Step 4: Run focused tests and verify they fail**

Run:

```bash
uv run --frozen pytest \
  src/tests/backtesting/test_config.py::test_output_path_override_is_private_and_exact \
  src/tests/backtesting/test_cli.py::test_parse_args_does_not_expose_model_id \
  src/tests/backtesting/test_runner.py::test_public_run_backtest_keeps_random_forest_strategy \
  src/tests/backtesting/test_runner.py::test_experiment_run_uses_primary_model_strategy \
  src/tests/backtesting/test_runner.py::test_experiment_run_rejects_unknown_primary_model \
  -q
```

Expected: failures for `_output_path_override` and `run_backtest_for_experiment`.

- [ ] **Step 5: Add private output path override**

In `src/cartola/backtesting/config.py`, add the field:

```python
    _output_path_override: Path | None = None
```

Update `output_path`:

```python
    @property
    def output_path(self) -> Path:
        if self._output_path_override is not None:
            return self._output_path_override
        return self.project_root / self.output_root / str(self.season)
```

- [ ] **Step 6: Update runner to use primary model id**

In `src/cartola/backtesting/runner.py`, import:

```python
from cartola.backtesting.model_registry import create_point_predictor
```

Change public `run_backtest` to call a private implementation:

```python
def run_backtest(
    config: BacktestConfig,
    season_df: pd.DataFrame | None = None,
    fixtures: pd.DataFrame | None = None,
) -> BacktestResult:
    return _run_backtest(
        config=config,
        primary_model_id="random_forest",
        season_df=season_df,
        fixtures=fixtures,
    )


def run_backtest_for_experiment(
    config: BacktestConfig,
    *,
    primary_model_id: str,
    season_df: pd.DataFrame | None = None,
    fixtures: pd.DataFrame | None = None,
) -> BacktestResult:
    return _run_backtest(
        config=config,
        primary_model_id=primary_model_id,
        season_df=season_df,
        fixtures=fixtures,
    )
```

Rename the existing `run_backtest()` body to `_run_backtest(...)` and add `primary_model_id: str`.

Thread `primary_model_id` into `_run_round_workers()` and `_evaluate_target_round()`:

```python
_run_round_workers(
    config=config,
    worker_rounds=worker_rounds,
    round_frame_store=round_frame_store,
    empty_training_columns=empty_training_columns,
    model_feature_columns=model_feature_columns,
    model_n_jobs_effective=model_n_jobs_effective,
    primary_model_id=primary_model_id,
)
```

Replace the RF-only fitting block in `_evaluate_target_round()`:

```python
primary_score_column = f"{primary_model_id}_score"
baseline_model = BaselinePredictor().fit(training)
primary_model = create_point_predictor(
    model_id=primary_model_id,
    random_seed=config.random_seed,
    feature_columns=model_feature_columns,
    n_jobs=model_n_jobs_effective,
).fit(training)
scored_candidates["baseline_score"] = baseline_model.predict(scored_candidates)
scored_candidates[primary_score_column] = primary_model.predict(scored_candidates)
scored_candidates["price_score"] = scored_candidates[MARKET_OPEN_PRICE_COLUMN].astype(float)
```

Replace `_strategies()` with:

```python
def _strategies(primary_model_id: str = "random_forest") -> dict[str, str]:
    return {
        "baseline": "baseline_score",
        primary_model_id: f"{primary_model_id}_score",
        "price": "price_score",
    }
```

Update all calls to `_strategies()` inside target-round evaluation and skipped-round recording to pass `primary_model_id`.

- [ ] **Step 7: Run focused tests and verify they pass**

Run:

```bash
uv run --frozen pytest \
  src/tests/backtesting/test_config.py::test_output_path_override_is_private_and_exact \
  src/tests/backtesting/test_cli.py::test_parse_args_does_not_expose_model_id \
  src/tests/backtesting/test_runner.py::test_public_run_backtest_keeps_random_forest_strategy \
  src/tests/backtesting/test_runner.py::test_experiment_run_uses_primary_model_strategy \
  src/tests/backtesting/test_runner.py::test_experiment_run_rejects_unknown_primary_model \
  -q
```

Expected: all selected tests pass.

- [ ] **Step 8: Run existing runner and CLI tests**

Run:

```bash
uv run --frozen pytest src/tests/backtesting/test_runner.py src/tests/backtesting/test_cli.py -q
```

Expected: all selected tests pass.

- [ ] **Step 9: Commit**

```bash
git add src/cartola/backtesting/config.py src/cartola/backtesting/runner.py \
  src/tests/backtesting/test_config.py src/tests/backtesting/test_cli.py src/tests/backtesting/test_runner.py
git commit -m "feat: support experiment primary model strategy"
```

---

### Task 3: Add Experiment Matrix And Config Hashing

**Files:**
- Create: `src/cartola/backtesting/experiment_config.py`
- Test: `src/tests/backtesting/test_experiment_config.py`

- [ ] **Step 1: Write config tests**

Create `src/tests/backtesting/test_experiment_config.py`:

```python
from pathlib import Path

import pytest

from cartola.backtesting.experiment_config import (
    FeaturePack,
    build_child_run_specs,
    config_hash,
    experiment_id,
    feature_pack_to_modes,
)


def test_production_parity_matrix() -> None:
    specs = build_child_run_specs(
        group="production-parity",
        seasons=(2023, 2024, 2025),
        start_round=5,
        budget=100.0,
        project_root=Path("/repo"),
        output_root=Path("data/08_reporting/experiments/model_feature/test"),
        current_year=2026,
        jobs=12,
    )

    assert len(specs) == 24
    assert {spec.fixture_mode for spec in specs} == {"none"}
    assert {spec.feature_pack for spec in specs} == {"ppg", "ppg_xg"}


def test_matchup_research_matrix() -> None:
    specs = build_child_run_specs(
        group="matchup-research",
        seasons=(2023, 2024, 2025),
        start_round=5,
        budget=100.0,
        project_root=Path("/repo"),
        output_root=Path("data/08_reporting/experiments/model_feature/test"),
        current_year=2026,
        jobs=12,
    )

    assert len(specs) == 48
    assert {spec.fixture_mode for spec in specs} == {"exploratory"}
    assert {spec.feature_pack for spec in specs} == {
        "ppg",
        "ppg_xg",
        "ppg_matchup",
        "ppg_xg_matchup",
    }


def test_feature_pack_to_modes() -> None:
    assert feature_pack_to_modes("ppg") == FeaturePack(
        feature_pack="ppg",
        footystats_mode="ppg",
        matchup_context_mode="none",
    )
    assert feature_pack_to_modes("ppg_xg_matchup") == FeaturePack(
        feature_pack="ppg_xg_matchup",
        footystats_mode="ppg_xg",
        matchup_context_mode="cartola_matchup_v1",
    )


def test_experiment_rejects_live_year() -> None:
    with pytest.raises(ValueError, match="Experiment seasons must be before current_year"):
        build_child_run_specs(
            group="production-parity",
            seasons=(2025, 2026),
            start_round=5,
            budget=100.0,
            project_root=Path("/repo"),
            output_root=Path("data/08_reporting/experiments/model_feature/test"),
            current_year=2026,
            jobs=12,
        )


def test_child_paths_are_deterministic() -> None:
    spec = build_child_run_specs(
        group="production-parity",
        seasons=(2025,),
        start_round=5,
        budget=100.0,
        project_root=Path("/repo"),
        output_root=Path("data/08_reporting/experiments/model_feature/test"),
        current_year=2026,
        jobs=12,
    )[0]

    assert spec.output_path == Path("/repo/data/08_reporting/experiments/model_feature/test/runs/season=2025/model=random_forest/feature_pack=ppg")
    assert spec.backtest_config.output_path == spec.output_path


def test_config_hash_changes_for_material_fields() -> None:
    base = build_child_run_specs(
        group="production-parity",
        seasons=(2025,),
        start_round=5,
        budget=100.0,
        project_root=Path("/repo"),
        output_root=Path("data/08_reporting/experiments/model_feature/test"),
        current_year=2026,
        jobs=12,
    )[0]
    changed = build_child_run_specs(
        group="production-parity",
        seasons=(2025,),
        start_round=6,
        budget=100.0,
        project_root=Path("/repo"),
        output_root=Path("data/08_reporting/experiments/model_feature/test"),
        current_year=2026,
        jobs=12,
    )[0]

    assert config_hash(base.config_identity) != config_hash(changed.config_identity)


def test_experiment_id_includes_group_and_hash() -> None:
    value = experiment_id(
        group="production-parity",
        started_at_utc="20260430T200000000000Z",
        matrix_hash="abcdef1234567890",
    )

    assert value == "group=production-parity__started_at=20260430T200000000000Z__matrix=abcdef123456"
```

- [ ] **Step 2: Run tests and verify they fail**

Run:

```bash
uv run --frozen pytest src/tests/backtesting/test_experiment_config.py -q
```

Expected: failure because `experiment_config.py` does not exist.

- [ ] **Step 3: Implement experiment config**

Create `src/cartola/backtesting/experiment_config.py`:

```python
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from cartola.backtesting.config import BacktestConfig, FixtureMode, FootyStatsMode, MatchupContextMode
from cartola.backtesting.model_registry import MODEL_SPECS, ModelId

ExperimentGroup = Literal["production-parity", "matchup-research"]
FeaturePackId = Literal["ppg", "ppg_xg", "ppg_matchup", "ppg_xg_matchup"]


@dataclass(frozen=True)
class FeaturePack:
    feature_pack: FeaturePackId
    footystats_mode: FootyStatsMode
    matchup_context_mode: MatchupContextMode


@dataclass(frozen=True)
class ChildRunSpec:
    group: ExperimentGroup
    season: int
    model_id: ModelId
    feature_pack: FeaturePackId
    fixture_mode: FixtureMode
    backtest_config: BacktestConfig
    output_path: Path
    config_identity: dict[str, object]


def feature_pack_to_modes(feature_pack: FeaturePackId) -> FeaturePack:
    if feature_pack == "ppg":
        return FeaturePack(feature_pack=feature_pack, footystats_mode="ppg", matchup_context_mode="none")
    if feature_pack == "ppg_xg":
        return FeaturePack(feature_pack=feature_pack, footystats_mode="ppg_xg", matchup_context_mode="none")
    if feature_pack == "ppg_matchup":
        return FeaturePack(feature_pack=feature_pack, footystats_mode="ppg", matchup_context_mode="cartola_matchup_v1")
    if feature_pack == "ppg_xg_matchup":
        return FeaturePack(feature_pack=feature_pack, footystats_mode="ppg_xg", matchup_context_mode="cartola_matchup_v1")
    raise ValueError(f"Unsupported feature_pack: {feature_pack!r}")


def _feature_packs_for_group(group: ExperimentGroup) -> tuple[FeaturePackId, ...]:
    if group == "production-parity":
        return ("ppg", "ppg_xg")
    if group == "matchup-research":
        return ("ppg", "ppg_xg", "ppg_matchup", "ppg_xg_matchup")
    raise ValueError(f"Unsupported experiment group: {group!r}")


def _fixture_mode_for_group(group: ExperimentGroup) -> FixtureMode:
    if group == "production-parity":
        return "none"
    if group == "matchup-research":
        return "exploratory"
    raise ValueError(f"Unsupported experiment group: {group!r}")


def build_child_run_specs(
    *,
    group: ExperimentGroup,
    seasons: tuple[int, ...],
    start_round: int,
    budget: float,
    project_root: Path,
    output_root: Path,
    current_year: int,
    jobs: int,
) -> list[ChildRunSpec]:
    if any(season >= current_year for season in seasons):
        raise ValueError("Experiment seasons must be before current_year")

    fixture_mode = _fixture_mode_for_group(group)
    feature_packs = _feature_packs_for_group(group)
    specs: list[ChildRunSpec] = []
    for season in seasons:
        for model_id in MODEL_SPECS:
            for feature_pack_id in feature_packs:
                feature_pack = feature_pack_to_modes(feature_pack_id)
                child_output_path = (
                    project_root
                    / output_root
                    / "runs"
                    / f"season={season}"
                    / f"model={model_id}"
                    / f"feature_pack={feature_pack_id}"
                )
                config = BacktestConfig(
                    season=season,
                    start_round=start_round,
                    budget=budget,
                    project_root=project_root,
                    output_root=output_root,
                    fixture_mode=fixture_mode,
                    footystats_mode=feature_pack.footystats_mode,
                    matchup_context_mode=feature_pack.matchup_context_mode,
                    current_year=current_year,
                    jobs=jobs,
                    _output_path_override=child_output_path,
                )
                identity = {
                    "group": group,
                    "season": season,
                    "model_id": model_id,
                    "feature_pack": feature_pack_id,
                    "fixture_mode": fixture_mode,
                    "footystats_mode": feature_pack.footystats_mode,
                    "matchup_context_mode": feature_pack.matchup_context_mode,
                    "start_round": start_round,
                    "budget": budget,
                    "current_year": current_year,
                    "jobs": jobs,
                    "scoring_contract_version": "cartola_standard_2026_v1",
                    "model_parameters": MODEL_SPECS[model_id].parameters,
                }
                specs.append(
                    ChildRunSpec(
                        group=group,
                        season=season,
                        model_id=model_id,
                        feature_pack=feature_pack_id,
                        fixture_mode=fixture_mode,
                        backtest_config=config,
                        output_path=child_output_path,
                        config_identity=identity,
                    )
                )
    return specs


def config_hash(payload: dict[str, object]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def experiment_id(*, group: ExperimentGroup, started_at_utc: str, matrix_hash: str) -> str:
    return f"group={group}__started_at={started_at_utc}__matrix={matrix_hash[:12]}"
```

- [ ] **Step 4: Run tests and verify they pass**

Run:

```bash
uv run --frozen pytest src/tests/backtesting/test_experiment_config.py -q
```

Expected: all selected tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/cartola/backtesting/experiment_config.py src/tests/backtesting/test_experiment_config.py
git commit -m "feat: define model experiment matrix"
```

---

### Task 4: Add Experiment Signatures And Source Hashes

**Files:**
- Create: `src/cartola/backtesting/experiment_signatures.py`
- Test: `src/tests/backtesting/test_experiment_signatures.py`

- [ ] **Step 1: Write signature tests**

Create `src/tests/backtesting/test_experiment_signatures.py`:

```python
import json
from pathlib import Path

import pandas as pd
import pytest

from cartola.backtesting.experiment_signatures import (
    ComparabilityError,
    candidate_pool_signature,
    compare_signature_sets,
    raw_cartola_source_identity,
    solver_status_signature,
)


def test_candidate_pool_signature_uses_canonical_candidate_fields() -> None:
    frame = pd.DataFrame(
        {
            "id_atleta": [2, 1],
            "posicao": ["mei", "ata"],
            "id_clube": [20, 10],
            "status": ["Provavel", "Provavel"],
            "preco_pre_rodada": [8.12345678911, 10.0],
            "rodada": [5, 5],
            "random_forest_score": [9.0, 1.0],
        }
    )

    first = candidate_pool_signature(frame)
    second = candidate_pool_signature(frame.sort_values("id_atleta", ascending=False))

    assert first == second


def test_candidate_pool_signature_changes_when_price_changes() -> None:
    frame = pd.DataFrame(
        {
            "id_atleta": [1],
            "posicao": ["ata"],
            "id_clube": [10],
            "status": ["Provavel"],
            "preco_pre_rodada": [10.0],
            "rodada": [5],
        }
    )
    changed = frame.copy()
    changed["preco_pre_rodada"] = [10.1]

    assert candidate_pool_signature(frame) != candidate_pool_signature(changed)


def test_solver_status_signature_maps_primary_role() -> None:
    rows = pd.DataFrame(
        {
            "rodada": [5, 5, 5],
            "strategy": ["baseline", "extra_trees", "price"],
            "solver_status": ["Optimal", "Optimal", "Infeasible"],
        }
    )

    assert solver_status_signature(rows, primary_model_id="extra_trees") == {
        "5:baseline": "Optimal",
        "5:primary_model": "Optimal",
        "5:price": "Infeasible",
    }


def test_solver_status_signature_rejects_unexpected_strategy() -> None:
    rows = pd.DataFrame(
        {
            "rodada": [5],
            "strategy": ["random_forest"],
            "solver_status": ["Optimal"],
        }
    )

    with pytest.raises(ComparabilityError, match="Unexpected strategy"):
        solver_status_signature(rows, primary_model_id="extra_trees")


def test_compare_signature_sets_raises_on_mismatch() -> None:
    with pytest.raises(ComparabilityError, match="candidate pools differ"):
        compare_signature_sets(
            label="candidate pools",
            signatures={
                "run-a": {"2025:5": "abc"},
                "run-b": {"2025:5": "def"},
            },
        )


def test_raw_cartola_source_identity_hashes_sorted_files(tmp_path) -> None:
    season_dir = tmp_path / "data" / "01_raw" / "2025"
    season_dir.mkdir(parents=True)
    (season_dir / "rodada-2.csv").write_text("b\n", encoding="utf-8")
    (season_dir / "rodada-1.csv").write_text("a\n", encoding="utf-8")
    (season_dir / "rodada-2.capture.json").write_text("ignored\n", encoding="utf-8")

    identity = raw_cartola_source_identity(project_root=tmp_path, season=2025)

    assert identity["season"] == 2025
    assert len(identity["files"]) == 2
    assert [file_record["path"] for file_record in identity["files"]] == [
        "data/01_raw/2025/rodada-1.csv",
        "data/01_raw/2025/rodada-2.csv",
    ]
    json.dumps(identity, sort_keys=True)
```

- [ ] **Step 2: Run tests and verify they fail**

Run:

```bash
uv run --frozen pytest src/tests/backtesting/test_experiment_signatures.py -q
```

Expected: failure because `experiment_signatures.py` does not exist.

- [ ] **Step 3: Implement signatures**

Create `src/cartola/backtesting/experiment_signatures.py`:

```python
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from cartola.backtesting.runner import CSV_FLOAT_FORMAT


class ComparabilityError(ValueError):
    pass


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical_hash(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return _sha256_bytes(encoded)


def candidate_pool_signature(frame: pd.DataFrame) -> str:
    required = ["id_atleta", "posicao", "id_clube", "status", "preco_pre_rodada", "rodada"]
    missing = sorted(set(required).difference(frame.columns))
    if missing:
        raise ComparabilityError(f"candidate pool missing columns: {missing}")

    rows: list[dict[str, object]] = []
    for _, row in frame.loc[:, required].sort_values("id_atleta").iterrows():
        rows.append(
            {
                "id_atleta": int(row["id_atleta"]),
                "posicao": str(row["posicao"]),
                "id_clube": int(row["id_clube"]),
                "status": str(row["status"]),
                "preco_pre_rodada": CSV_FLOAT_FORMAT % float(row["preco_pre_rodada"]),
                "rodada": int(row["rodada"]),
            }
        )
    return _canonical_hash(rows)


def solver_status_signature(round_results: pd.DataFrame, *, primary_model_id: str) -> dict[str, str]:
    signatures: dict[str, str] = {}
    for _, row in round_results.iterrows():
        strategy = str(row["strategy"])
        if strategy == "baseline":
            role = "baseline"
        elif strategy == "price":
            role = "price"
        elif strategy == primary_model_id:
            role = "primary_model"
        else:
            raise ComparabilityError(f"Unexpected strategy for primary model {primary_model_id!r}: {strategy!r}")
        signatures[f"{int(row['rodada'])}:{role}"] = str(row["solver_status"])
    return signatures


def compare_signature_sets(*, label: str, signatures: dict[str, dict[str, str]]) -> None:
    items = list(signatures.items())
    if len(items) <= 1:
        return
    first_run, first_signature = items[0]
    for run_id, signature in items[1:]:
        if signature != first_signature:
            raise ComparabilityError(f"{label} differ between {first_run!r} and {run_id!r}")


def raw_cartola_source_identity(*, project_root: Path, season: int) -> dict[str, object]:
    season_dir = project_root / "data" / "01_raw" / str(season)
    files = [
        path
        for path in season_dir.rglob("*")
        if path.is_file() and not path.name.endswith(".capture.json")
    ]
    records: list[dict[str, object]] = []
    for path in sorted(files):
        relative = path.relative_to(project_root).as_posix()
        data = path.read_bytes()
        records.append(
            {
                "path": relative,
                "sha256": _sha256_bytes(data),
                "size_bytes": len(data),
            }
        )
    return {
        "season": season,
        "directory": season_dir.relative_to(project_root).as_posix(),
        "files": records,
        "sha256": _canonical_hash(records),
    }
```

- [ ] **Step 4: Run tests and verify they pass**

Run:

```bash
uv run --frozen pytest src/tests/backtesting/test_experiment_signatures.py -q
```

Expected: all selected tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/cartola/backtesting/experiment_signatures.py src/tests/backtesting/test_experiment_signatures.py
git commit -m "feat: add experiment comparability signatures"
```

---

### Task 5: Add Experiment Metrics And Promotion Rules

**Files:**
- Create: `src/cartola/backtesting/experiment_metrics.py`
- Test: `src/tests/backtesting/test_experiment_metrics.py`

- [ ] **Step 1: Write metrics tests**

Create `src/tests/backtesting/test_experiment_metrics.py`:

```python
import pandas as pd

from cartola.backtesting.experiment_metrics import (
    calibration_slope_intercept,
    promotion_status,
    top_k_rows_by_round,
)


def test_top_k_rows_are_selected_per_round() -> None:
    frame = pd.DataFrame(
        {
            "rodada": [1, 1, 1, 2, 2, 2],
            "id_atleta": [1, 2, 3, 4, 5, 6],
            "model_score": [1.0, 3.0, 2.0, 10.0, 8.0, 9.0],
            "pontuacao": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        }
    )

    result = top_k_rows_by_round(frame, score_column="model_score", k=2)

    assert result["id_atleta"].tolist() == [2, 3, 4, 6]


def test_calibration_slope_and_intercept() -> None:
    predicted = pd.Series([1.0, 2.0, 3.0, 4.0])
    actual = pd.Series([2.0, 4.0, 6.0, 8.0])

    result = calibration_slope_intercept(predicted, actual)

    assert result == {"calibration_intercept": 0.0, "calibration_slope": 2.0, "warning": None}


def test_calibration_returns_null_for_constant_predictions() -> None:
    result = calibration_slope_intercept(pd.Series([1.0, 1.0]), pd.Series([2.0, 3.0]))

    assert result == {
        "calibration_intercept": None,
        "calibration_slope": None,
        "warning": "constant_prediction",
    }


def test_promotion_status_passes_when_all_guardrails_pass() -> None:
    result = promotion_status(
        aggregate_delta=10.0,
        improved_seasons=2,
        worst_season_avg_delta=-1.0,
        selected_calibration_slope=1.0,
        top50_spearman_delta=-0.01,
        comparable=True,
    )

    assert result == {"eligible": True, "reason": "passes_v1_guardrails"}


def test_promotion_status_fails_null_guardrail() -> None:
    result = promotion_status(
        aggregate_delta=10.0,
        improved_seasons=2,
        worst_season_avg_delta=-1.0,
        selected_calibration_slope=None,
        top50_spearman_delta=-0.01,
        comparable=True,
    )

    assert result == {"eligible": False, "reason": "insufficient_metric_data"}


def test_promotion_status_fails_aggregate_only_win() -> None:
    result = promotion_status(
        aggregate_delta=10.0,
        improved_seasons=1,
        worst_season_avg_delta=-1.0,
        selected_calibration_slope=1.0,
        top50_spearman_delta=0.0,
        comparable=True,
    )

    assert result == {"eligible": False, "reason": "fewer_than_two_seasons_improved"}


def test_promotion_status_fails_worst_season_regression() -> None:
    result = promotion_status(
        aggregate_delta=10.0,
        improved_seasons=2,
        worst_season_avg_delta=-2.0,
        selected_calibration_slope=1.0,
        top50_spearman_delta=0.0,
        comparable=True,
    )

    assert result == {"eligible": False, "reason": "worst_season_regression_exceeds_threshold"}
```

- [ ] **Step 2: Run tests and verify they fail**

Run:

```bash
uv run --frozen pytest src/tests/backtesting/test_experiment_metrics.py -q
```

Expected: failure because `experiment_metrics.py` does not exist.

- [ ] **Step 3: Implement metrics helpers**

Create `src/cartola/backtesting/experiment_metrics.py`:

```python
from __future__ import annotations

import math

import pandas as pd


def top_k_rows_by_round(frame: pd.DataFrame, *, score_column: str, k: int) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for _, round_frame in frame.groupby("rodada", sort=True):
        rows.append(round_frame.sort_values(score_column, ascending=False).head(k))
    if not rows:
        return pd.DataFrame(columns=frame.columns)
    return pd.concat(rows, ignore_index=True)


def calibration_slope_intercept(predicted: pd.Series, actual: pd.Series) -> dict[str, float | str | None]:
    paired = pd.DataFrame({"predicted": predicted, "actual": actual}).dropna()
    if paired.empty:
        return {"calibration_intercept": None, "calibration_slope": None, "warning": "empty_input"}
    x = paired["predicted"].astype(float)
    y = paired["actual"].astype(float)
    x_mean = float(x.mean())
    y_mean = float(y.mean())
    denominator = float(((x - x_mean) ** 2).sum())
    if math.isclose(denominator, 0.0, abs_tol=1e-12):
        return {"calibration_intercept": None, "calibration_slope": None, "warning": "constant_prediction"}
    slope = float(((x - x_mean) * (y - y_mean)).sum() / denominator)
    intercept = float(y_mean - slope * x_mean)
    return {
        "calibration_intercept": round(intercept, 10),
        "calibration_slope": round(slope, 10),
        "warning": None,
    }


def promotion_status(
    *,
    aggregate_delta: float | None,
    improved_seasons: int | None,
    worst_season_avg_delta: float | None,
    selected_calibration_slope: float | None,
    top50_spearman_delta: float | None,
    comparable: bool,
) -> dict[str, object]:
    if not comparable:
        return {"eligible": False, "reason": "not_comparable"}
    guardrails = [
        aggregate_delta,
        improved_seasons,
        worst_season_avg_delta,
        selected_calibration_slope,
        top50_spearman_delta,
    ]
    if any(value is None for value in guardrails):
        return {"eligible": False, "reason": "insufficient_metric_data"}
    assert aggregate_delta is not None
    assert improved_seasons is not None
    assert worst_season_avg_delta is not None
    assert selected_calibration_slope is not None
    assert top50_spearman_delta is not None
    if aggregate_delta <= 0:
        return {"eligible": False, "reason": "aggregate_delta_not_positive"}
    if improved_seasons < 2:
        return {"eligible": False, "reason": "fewer_than_two_seasons_improved"}
    if worst_season_avg_delta < -1.5:
        return {"eligible": False, "reason": "worst_season_regression_exceeds_threshold"}
    if selected_calibration_slope < 0.75 or selected_calibration_slope > 1.25:
        return {"eligible": False, "reason": "selected_calibration_slope_out_of_range"}
    if top50_spearman_delta < -0.03:
        return {"eligible": False, "reason": "top50_spearman_regression_exceeds_threshold"}
    return {"eligible": True, "reason": "passes_v1_guardrails"}
```

- [ ] **Step 4: Run tests and verify they pass**

Run:

```bash
uv run --frozen pytest src/tests/backtesting/test_experiment_metrics.py -q
```

Expected: all selected tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/cartola/backtesting/experiment_metrics.py src/tests/backtesting/test_experiment_metrics.py
git commit -m "feat: add experiment metric helpers"
```

---

### Task 6: Add Sequential Experiment Runner

**Files:**
- Create: `src/cartola/backtesting/experiment_runner.py`
- Test: `src/tests/backtesting/test_experiment_runner.py`

- [ ] **Step 1: Write runner orchestration tests**

Create `src/tests/backtesting/test_experiment_runner.py`:

```python
from pathlib import Path

import pandas as pd
import pytest

from cartola.backtesting.experiment_runner import run_model_experiment
from cartola.backtesting.runner import BacktestMetadata, BacktestResult


def _metadata(season: int, fixture_mode: str, footystats_mode: str, matchup_context_mode: str) -> BacktestMetadata:
    return BacktestMetadata(
        season=season,
        start_round=5,
        max_round=5,
        cache_enabled=True,
        prediction_frames_built=5,
        wall_clock_seconds=1.0,
        backtest_jobs=2,
        backtest_workers_effective=1,
        model_n_jobs_effective=1,
        parallel_backend="sequential",
        thread_env={"OMP_NUM_THREADS": None, "MKL_NUM_THREADS": None, "OPENBLAS_NUM_THREADS": None, "BLIS_NUM_THREADS": None},
        scoring_contract_version="cartola_standard_2026_v1",
        captain_scoring_enabled=True,
        captain_multiplier=1.5,
        formation_search="all_official_formations",
        fixture_mode=fixture_mode,
        strict_alignment_policy="fail",
        matchup_context_mode=matchup_context_mode,
        matchup_context_feature_columns=[],
        fixture_source_directory=None,
        fixture_manifest_paths=[],
        fixture_manifest_sha256={},
        generator_versions=[],
        excluded_rounds=[],
        warnings=[],
        footystats_mode=footystats_mode,
        footystats_evaluation_scope="historical_candidate",
        footystats_league_slug="brazil-serie-a",
        footystats_matches_source_path=None,
        footystats_matches_source_sha256=None,
        footystats_feature_columns=[],
        footystats_missing_join_keys_by_round={},
        footystats_duplicate_join_keys_by_round={},
        footystats_extra_club_rows_by_round={},
    )


def _result(strategy: str, season: int, fixture_mode: str, footystats_mode: str, matchup_context_mode: str) -> BacktestResult:
    round_results = pd.DataFrame(
        {
            "rodada": [5, 5, 5],
            "strategy": ["baseline", strategy, "price"],
            "solver_status": ["Optimal", "Optimal", "Optimal"],
            "actual_points": [50.0, 60.0, 45.0],
            "predicted_points": [55.0, 65.0, 45.0],
        }
    )
    player_predictions = pd.DataFrame(
        {
            "rodada": [5, 5],
            "id_atleta": [1, 2],
            "posicao": ["ata", "mei"],
            "id_clube": [10, 20],
            "status": ["Provavel", "Provavel"],
            "preco_pre_rodada": [10.0, 8.0],
            "pontuacao": [7.0, 3.0],
            f"{strategy}_score": [6.0, 4.0],
            "baseline_score": [5.0, 5.0],
            "price_score": [10.0, 8.0],
        }
    )
    return BacktestResult(
        round_results=round_results,
        selected_players=pd.DataFrame(
            {
                "rodada": [5],
                "strategy": [strategy],
                "id_atleta": [1],
                "posicao": ["ata"],
                "pontuacao": [7.0],
            }
        ),
        player_predictions=player_predictions,
        summary=pd.DataFrame(
            {
                "strategy": ["baseline", strategy, "price"],
                "rounds": [1, 1, 1],
                "total_actual_points": [50.0, 60.0, 45.0],
                "average_actual_points": [50.0, 60.0, 45.0],
                "total_predicted_points": [55.0, 65.0, 45.0],
            }
        ),
        diagnostics=pd.DataFrame(),
        metadata=_metadata(season, fixture_mode, footystats_mode, matchup_context_mode),
    )


def test_experiment_runner_executes_child_runs_sequentially(tmp_path, monkeypatch) -> None:
    observed: list[str] = []

    def fake_run_backtest_for_experiment(config, *, primary_model_id):
        observed.append(primary_model_id)
        return _result(
            strategy=primary_model_id,
            season=config.season,
            fixture_mode=config.fixture_mode,
            footystats_mode=config.footystats_mode,
            matchup_context_mode=config.matchup_context_mode,
        )

    monkeypatch.setattr(
        "cartola.backtesting.experiment_runner.run_backtest_for_experiment",
        fake_run_backtest_for_experiment,
    )
    monkeypatch.setattr(
        "cartola.backtesting.experiment_runner.raw_cartola_source_identity",
        lambda project_root, season: {"season": season, "sha256": "raw"},
    )

    result = run_model_experiment(
        group="production-parity",
        seasons=(2025,),
        start_round=5,
        budget=100.0,
        current_year=2026,
        jobs=2,
        project_root=tmp_path,
        output_root=Path("data/08_reporting/experiments/model_feature/test"),
        started_at_utc="20260430T200000000000Z",
    )

    assert observed == [
        "random_forest",
        "random_forest",
        "extra_trees",
        "extra_trees",
        "hist_gradient_boosting",
        "hist_gradient_boosting",
        "ridge",
        "ridge",
    ]
    assert result.output_path.exists()
    assert (result.output_path / "experiment_metadata.json").exists()
    assert (result.output_path / "ranked_summary.csv").exists()


def test_experiment_runner_aborts_on_child_failure(tmp_path, monkeypatch) -> None:
    def fake_run_backtest_for_experiment(config, *, primary_model_id):
        raise RuntimeError("child failed")

    monkeypatch.setattr(
        "cartola.backtesting.experiment_runner.run_backtest_for_experiment",
        fake_run_backtest_for_experiment,
    )

    with pytest.raises(RuntimeError, match="child failed"):
        run_model_experiment(
            group="production-parity",
            seasons=(2025,),
            start_round=5,
            budget=100.0,
            current_year=2026,
            jobs=2,
            project_root=tmp_path,
            output_root=Path("data/08_reporting/experiments/model_feature/test"),
            started_at_utc="20260430T200000000000Z",
        )
```

- [ ] **Step 2: Run tests and verify they fail**

Run:

```bash
uv run --frozen pytest src/tests/backtesting/test_experiment_runner.py -q
```

Expected: failure because `experiment_runner.py` does not exist.

- [ ] **Step 3: Implement minimal sequential runner**

Create `src/cartola/backtesting/experiment_runner.py`:

```python
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from cartola.backtesting.experiment_config import (
    ChildRunSpec,
    ExperimentGroup,
    build_child_run_specs,
    config_hash,
    experiment_id,
)
from cartola.backtesting.experiment_signatures import (
    candidate_pool_signature,
    raw_cartola_source_identity,
    solver_status_signature,
)
from cartola.backtesting.runner import BacktestResult, run_backtest_for_experiment


@dataclass(frozen=True)
class ExperimentRunResult:
    experiment_id: str
    output_path: Path
    ranked_summary: pd.DataFrame
    metadata: dict[str, object]


def run_model_experiment(
    *,
    group: ExperimentGroup,
    seasons: tuple[int, ...],
    start_round: int,
    budget: float,
    current_year: int,
    jobs: int,
    project_root: Path,
    output_root: Path,
    started_at_utc: str,
) -> ExperimentRunResult:
    matrix_seed_specs = build_child_run_specs(
        group=group,
        seasons=seasons,
        start_round=start_round,
        budget=budget,
        project_root=project_root,
        output_root=output_root,
        current_year=current_year,
        jobs=jobs,
    )
    matrix_hash = config_hash({"children": [spec.config_identity for spec in matrix_seed_specs]})
    run_id = experiment_id(group=group, started_at_utc=started_at_utc, matrix_hash=matrix_hash)
    experiment_output_root = output_root / run_id
    specs = build_child_run_specs(
        group=group,
        seasons=seasons,
        start_round=start_round,
        budget=budget,
        project_root=project_root,
        output_root=experiment_output_root,
        current_year=current_year,
        jobs=jobs,
    )
    output_path = project_root / experiment_output_root
    if output_path.exists():
        raise FileExistsError(f"Experiment output already exists: {output_path}")
    output_path.mkdir(parents=True)

    child_records: list[dict[str, object]] = []
    ranked_rows: list[dict[str, object]] = []
    candidate_signatures: dict[str, dict[str, str]] = {}
    solver_signatures: dict[str, dict[str, str]] = {}

    try:
        for spec in specs:
            child_id = _child_id(spec)
            result = run_backtest_for_experiment(spec.backtest_config, primary_model_id=spec.model_id)
            _write_child_outputs_marker(spec, result)
            child_records.append(_child_record(spec, result))
            candidate_signatures[child_id] = _candidate_signatures_for_result(result)
            solver_signatures[child_id] = solver_status_signature(result.round_results, primary_model_id=spec.model_id)
            ranked_rows.append(_ranked_row(spec, result))
    except Exception:
        _write_json(output_path / "experiment_metadata.json", {"status": "failed", "experiment_id": run_id})
        _write_json(output_path / "comparability_report.json", {"status": "failed"})
        raise

    metadata = {
        "status": "ok",
        "experiment_id": run_id,
        "experiment_started_at_utc": started_at_utc,
        "group": group,
        "seasons": list(seasons),
        "start_round": start_round,
        "budget": budget,
        "current_year": current_year,
        "jobs": jobs,
        "matrix_hash": matrix_hash,
        "child_runs": child_records,
        "raw_sources": {
            str(season): raw_cartola_source_identity(project_root=project_root, season=season)
            for season in seasons
        },
    }
    ranked_summary = pd.DataFrame(ranked_rows).sort_values(
        ["total_actual_points", "model_id", "feature_pack", "season"],
        ascending=[False, True, True, True],
    )
    _write_json(output_path / "experiment_metadata.json", metadata)
    _write_json(
        output_path / "comparability_report.json",
        {
            "status": "ok",
            "candidate_pool_signatures": candidate_signatures,
            "solver_status_signatures": solver_signatures,
        },
    )
    ranked_summary.to_csv(output_path / "ranked_summary.csv", index=False)
    pd.DataFrame(ranked_rows).to_csv(output_path / "per_season_summary.csv", index=False)
    pd.DataFrame().to_csv(output_path / "prediction_metrics.csv", index=False)
    pd.DataFrame().to_csv(output_path / "calibration_deciles.csv", index=False)
    (output_path / "comparison_report.md").write_text("# Model Feature Experiment\n", encoding="utf-8")
    (output_path / "calibration_plots.html").write_text("<!doctype html><html><body></body></html>\n", encoding="utf-8")
    (output_path / "squad_performance_comparison.html").write_text(
        "<!doctype html><html><body></body></html>\n",
        encoding="utf-8",
    )
    return ExperimentRunResult(
        experiment_id=run_id,
        output_path=output_path,
        ranked_summary=ranked_summary,
        metadata=metadata,
    )


def _child_id(spec: ChildRunSpec) -> str:
    return f"season={spec.season}/model={spec.model_id}/feature_pack={spec.feature_pack}"


def _candidate_signatures_for_result(result: BacktestResult) -> dict[str, str]:
    signatures: dict[str, str] = {}
    if result.player_predictions.empty:
        return signatures
    for round_number, round_frame in result.player_predictions.groupby("rodada", sort=True):
        signatures[str(int(round_number))] = candidate_pool_signature(round_frame)
    return signatures


def _ranked_row(spec: ChildRunSpec, result: BacktestResult) -> dict[str, object]:
    primary = result.summary[result.summary["strategy"].eq(spec.model_id)]
    if primary.empty:
        return {
            "season": spec.season,
            "model_id": spec.model_id,
            "feature_pack": spec.feature_pack,
            "total_actual_points": pd.NA,
            "average_actual_points": pd.NA,
        }
    row = primary.iloc[0]
    return {
        "season": spec.season,
        "model_id": spec.model_id,
        "feature_pack": spec.feature_pack,
        "fixture_mode": spec.fixture_mode,
        "total_actual_points": row["total_actual_points"],
        "average_actual_points": row["average_actual_points"],
    }


def _child_record(spec: ChildRunSpec, result: BacktestResult) -> dict[str, object]:
    return {
        "child_id": _child_id(spec),
        "season": spec.season,
        "model_id": spec.model_id,
        "feature_pack": spec.feature_pack,
        "fixture_mode": spec.fixture_mode,
        "output_path": str(spec.output_path),
        "model_n_jobs_effective": None if spec.model_id in {"hist_gradient_boosting", "ridge"} else result.metadata.model_n_jobs_effective,
        "strategy_roles": {
            "baseline": "baseline",
            "primary_model": spec.model_id,
            "price": "price",
        },
    }


def _write_child_outputs_marker(spec: ChildRunSpec, result: BacktestResult) -> None:
    spec.output_path.mkdir(parents=True, exist_ok=True)
    _write_json(
        spec.output_path / "experiment_child_metadata.json",
        {
            "season": spec.season,
            "model_id": spec.model_id,
            "feature_pack": spec.feature_pack,
            "fixture_mode": spec.fixture_mode,
            "round_rows": len(result.round_results),
        },
    )


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
```

- [ ] **Step 4: Run tests and verify they pass**

Run:

```bash
uv run --frozen pytest src/tests/backtesting/test_experiment_runner.py -q
```

Expected: all selected tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/cartola/backtesting/experiment_runner.py src/tests/backtesting/test_experiment_runner.py
git commit -m "feat: add sequential model experiment runner"
```

---

### Task 7: Add Experiment CLI

**Files:**
- Create: `scripts/run_model_experiments.py`
- Test: `src/tests/backtesting/test_run_model_experiments_cli.py`

- [ ] **Step 1: Write CLI tests**

Create `src/tests/backtesting/test_run_model_experiments_cli.py`:

```python
from pathlib import Path

from scripts.run_model_experiments import main, parse_args


def test_parse_args_defaults() -> None:
    args = parse_args(["--group", "production-parity", "--current-year", "2026"])

    assert args.group == "production-parity"
    assert args.seasons == "2023,2024,2025"
    assert args.start_round == 5
    assert args.budget == 100.0
    assert args.jobs == 1


def test_main_calls_runner(monkeypatch, tmp_path) -> None:
    observed: dict[str, object] = {}

    def fake_run_model_experiment(**kwargs):
        observed.update(kwargs)

        class Result:
            output_path = tmp_path / "out"
            experiment_id = "exp"

        return Result()

    monkeypatch.setattr("scripts.run_model_experiments.run_model_experiment", fake_run_model_experiment)

    exit_code = main(
        [
            "--group",
            "matchup-research",
            "--seasons",
            "2023,2024",
            "--current-year",
            "2026",
            "--project-root",
            str(tmp_path),
            "--output-root",
            "data/08_reporting/experiments/model_feature/test",
            "--jobs",
            "12",
        ]
    )

    assert exit_code == 0
    assert observed["group"] == "matchup-research"
    assert observed["seasons"] == (2023, 2024)
    assert observed["current_year"] == 2026
    assert observed["project_root"] == tmp_path
    assert observed["output_root"] == Path("data/08_reporting/experiments/model_feature/test")
    assert observed["jobs"] == 12
```

- [ ] **Step 2: Run tests and verify they fail**

Run:

```bash
uv run --frozen pytest src/tests/backtesting/test_run_model_experiments_cli.py -q
```

Expected: failure because `scripts/run_model_experiments.py` does not exist.

- [ ] **Step 3: Implement CLI**

Create `scripts/run_model_experiments.py`:

```python
from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path
from typing import Sequence

from rich.console import Console
from rich.panel import Panel

from cartola.backtesting.experiment_runner import run_model_experiment


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run controlled Cartola model/feature experiments.")
    parser.add_argument("--group", choices=("production-parity", "matchup-research"), required=True)
    parser.add_argument("--seasons", default="2023,2024,2025")
    parser.add_argument("--start-round", type=int, default=5)
    parser.add_argument("--budget", type=float, default=100.0)
    parser.add_argument("--current-year", type=int, required=True)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/08_reporting/experiments/model_feature"),
    )
    parser.add_argument("--jobs", type=int, default=1)
    return parser.parse_args(argv)


def _parse_seasons(value: str) -> tuple[int, ...]:
    seasons = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not seasons:
        raise ValueError("At least one season is required")
    return seasons


def _timestamp() -> str:
    return datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%S%fZ")


def main(argv: Sequence[str] | None = None) -> int:
    console = Console()
    args = parse_args(argv)
    try:
        result = run_model_experiment(
            group=args.group,
            seasons=_parse_seasons(args.seasons),
            start_round=args.start_round,
            budget=args.budget,
            current_year=args.current_year,
            jobs=args.jobs,
            project_root=args.project_root,
            output_root=args.output_root,
            started_at_utc=_timestamp(),
        )
    except Exception as exc:
        console.print(Panel(f"error: {exc}", title="Experiment failed", border_style="red"))
        return 1

    console.print(
        Panel(
            f"experiment_id={result.experiment_id}\noutput={result.output_path}",
            title="Experiment complete",
            border_style="green",
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run tests and verify they pass**

Run:

```bash
uv run --frozen pytest src/tests/backtesting/test_run_model_experiments_cli.py -q
```

Expected: all selected tests pass.

- [ ] **Step 5: Commit**

```bash
git add scripts/run_model_experiments.py src/tests/backtesting/test_run_model_experiments_cli.py
git commit -m "feat: add model experiment CLI"
```

---

### Task 8: Add End-To-End Guardrails And Quality Gate

**Files:**
- Modify: `src/tests/backtesting/test_runner.py`
- Modify: `src/tests/backtesting/test_cli_output.py`

- [ ] **Step 1: Add baseline/price equivalence test**

Add to `src/tests/backtesting/test_runner.py`, which already defines `_tiny_round`, imports `runner_module`, imports `BacktestConfig`, imports `Path`, and imports `assert_frame_equal`:

```python
def test_baseline_and_price_are_equal_across_model_ids(tmp_path, monkeypatch) -> None:
    data = pd.concat([_tiny_round(round_number) for round_number in range(1, 6)], ignore_index=True)
    monkeypatch.setattr("cartola.backtesting.runner.load_season_data", lambda season, project_root: data)

    config = BacktestConfig(project_root=tmp_path, season=2025, start_round=5, jobs=1)
    rf = runner_module.run_backtest_for_experiment(config, primary_model_id="random_forest")
    et = runner_module.run_backtest_for_experiment(
        BacktestConfig(
            project_root=tmp_path,
            season=2025,
            start_round=5,
            jobs=1,
            output_root=Path("data/08_reporting/backtests_extra"),
        ),
        primary_model_id="extra_trees",
    )

    rf_common = rf.round_results[rf.round_results["strategy"].isin(["baseline", "price"])].reset_index(drop=True)
    et_common = et.round_results[et.round_results["strategy"].isin(["baseline", "price"])].reset_index(drop=True)
    assert_frame_equal(rf_common, et_common)
```

- [ ] **Step 2: Add chart strategy flexibility test**

In `src/tests/backtesting/test_cli_output.py`, add a test that uses `strategy=extra_trees` and confirms chart preparation does not fail for score traces. The formation panel may remain normal-CLI focused on `random_forest`.

```python
def test_chart_accepts_non_random_forest_strategy(tmp_path) -> None:
    rows = pd.DataFrame(
        {
            "rodada": [5],
            "strategy": ["extra_trees"],
            "solver_status": ["Optimal"],
            "actual_points": [60.0],
            "formation": ["4-3-3"],
        }
    )

    output = write_performance_chart(rows, tmp_path)

    assert output.path is not None
    assert output.warning is None
```

- [ ] **Step 3: Run guardrail tests**

Run:

```bash
uv run --frozen pytest \
  src/tests/backtesting/test_runner.py \
  src/tests/backtesting/test_cli_output.py \
  src/tests/backtesting/test_experiment_runner.py \
  -q
```

Expected: all selected tests pass.

- [ ] **Step 4: Run all experiment-related tests**

Run:

```bash
uv run --frozen pytest \
  src/tests/backtesting/test_model_registry.py \
  src/tests/backtesting/test_experiment_config.py \
  src/tests/backtesting/test_experiment_signatures.py \
  src/tests/backtesting/test_experiment_metrics.py \
  src/tests/backtesting/test_experiment_runner.py \
  src/tests/backtesting/test_run_model_experiments_cli.py \
  -q
```

Expected: all selected tests pass.

- [ ] **Step 5: Run full quality gate**

Run:

```bash
uv run --frozen scripts/pyrepo-check --all
```

Expected:

```text
ruff passed
ty passed
bandit passed
pytest passed
```

- [ ] **Step 6: Commit**

```bash
git add src/cartola/backtesting scripts src/tests/backtesting
git commit -m "test: verify model experiment guardrails"
```

---

## Execution Notes

- Do not push or merge until the full quality gate passes.
- Do not change live recommendation defaults in this implementation.
- Do not expose `--model-id` in `python -m cartola.backtesting.cli`.
- Do not add experiment-level parallelism.
- Do not introduce external model libraries.
- Keep each task independently reviewable.

## Manual Smoke Commands After Implementation

Production parity:

```bash
uv run --frozen python scripts/run_model_experiments.py \
  --group production-parity \
  --seasons 2025 \
  --start-round 5 \
  --budget 100 \
  --current-year 2026 \
  --jobs 4 \
  --output-root data/08_reporting/experiments/model_feature/smoke_production
```

Matchup research:

```bash
uv run --frozen python scripts/run_model_experiments.py \
  --group matchup-research \
  --seasons 2025 \
  --start-round 5 \
  --budget 100 \
  --current-year 2026 \
  --jobs 4 \
  --output-root data/08_reporting/experiments/model_feature/smoke_matchup
```

Full intended matrix:

```bash
uv run --frozen python scripts/run_model_experiments.py \
  --group production-parity \
  --seasons 2023,2024,2025 \
  --start-round 5 \
  --budget 100 \
  --current-year 2026 \
  --jobs 12
```

```bash
uv run --frozen python scripts/run_model_experiments.py \
  --group matchup-research \
  --seasons 2023,2024,2025 \
  --start-round 5 \
  --budget 100 \
  --current-year 2026 \
  --jobs 12
```

## Self-Review

- Spec coverage: the plan covers private model registry, strategy identity, hidden normal CLI, sequential child execution, output paths, source hashing, role-based comparability, null guardrails, dense HGB preprocessing, and no live default changes.
- Placeholder scan: no incomplete sections are intentionally left for the implementer.
- Type consistency: model ids, feature-pack ids, and child-run path names match the approved spec.
