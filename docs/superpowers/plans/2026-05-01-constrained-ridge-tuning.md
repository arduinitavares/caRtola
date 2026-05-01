# Constrained Ridge Tuning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a fixed Ridge alpha tuning runner that evaluates `ppg` and `ppg_xg` over 2023-2025, reruns finalists, and produces promotion-safe reports without changing normal backtest or live behavior.

**Architecture:** Add a private model-parameter override path for Ridge, then build a separate tuning runner around the existing walk-forward backtest. Keep model-feature experiments intact, represent tuned variants with `candidate_id` plus `model_params`, and compute promotion against freshly rerun Ridge incumbents.

**Tech Stack:** Python 3.13, pandas, existing Cartola backtesting runner, existing experiment signatures/metrics helpers, pytest, Ruff/ty/Bandit through `scripts/pyrepo-check`.

---

## Preflight

Create a dedicated implementation branch or worktree before changing code:

```bash
cd /Users/aaat/projects/caRtola
git switch -c dev/constrained-ridge-tuning
```

If `master` already has unpushed commits that should stay on `master`, use a worktree instead:

```bash
cd /Users/aaat/projects/caRtola
git worktree add ../caRtola-constrained-ridge-tuning -b dev/constrained-ridge-tuning
cd ../caRtola-constrained-ridge-tuning
```

Expected: clean working tree on `dev/constrained-ridge-tuning`.

Do not edit historical report folders under `data/08_reporting/experiments/model_feature/...`. Tests must use `tmp_path`.

## File Structure

- Modify `src/cartola/backtesting/models.py`
  - Add optional `model_params` plumbing to sklearn predictors.
  - Keep defaults unchanged.
  - Let `RidgePointPredictor` consume validated `alpha`.
- Modify `src/cartola/backtesting/model_registry.py`
  - Validate model parameter overrides.
  - Add `model_params` to `create_point_predictor`.
  - Keep `MODEL_SPECS` defaults unchanged.
- Modify `src/cartola/backtesting/runner.py`
  - Add `model_params` only to `run_backtest_for_experiment` and the internal round-evaluation path.
  - Keep public `run_backtest(config)` unchanged.
- Create `src/cartola/backtesting/ridge_tuning_config.py`
  - Own fixed alpha list, feature-pack list, candidate ids, specs, and generation hashes.
- Create `src/cartola/backtesting/ridge_tuning_metrics.py`
  - Own tuning-specific prediction rows, calibration rows, ranked summaries, promotion gates, and final-rerun comparison.
- Create `src/cartola/backtesting/ridge_tuning_runner.py`
  - Orchestrate screen and final stages, write artifacts, and emit progress events.
- Create `scripts/run_ridge_tuning.py`
  - CLI entry point with Rich progress display modeled on `scripts/run_model_experiments.py`.
- Add tests:
  - `src/tests/backtesting/test_model_registry.py`
  - `src/tests/backtesting/test_runner.py`
  - `src/tests/backtesting/test_ridge_tuning_config.py`
  - `src/tests/backtesting/test_ridge_tuning_metrics.py`
  - `src/tests/backtesting/test_ridge_tuning_runner.py`
  - `src/tests/backtesting/test_run_ridge_tuning_cli.py`

## Task 1: Ridge Model Parameter Overrides

**Files:**
- Modify: `src/cartola/backtesting/models.py`
- Modify: `src/cartola/backtesting/model_registry.py`
- Test: `src/tests/backtesting/test_model_registry.py`

- [ ] **Step 1: Write failing model-registry tests for Ridge alpha override**

Append tests to `src/tests/backtesting/test_model_registry.py`:

```python
def test_ridge_model_params_override_alpha() -> None:
    model = create_point_predictor(
        model_id="ridge",
        random_seed=7,
        feature_columns=FEATURE_COLUMNS,
        n_jobs=99,
        model_params={"alpha": 3.0},
    )

    estimator = model.pipeline.named_steps["model"]
    assert isinstance(estimator, Ridge)
    assert estimator.alpha == 3.0


def test_ridge_model_params_reject_unknown_key() -> None:
    with pytest.raises(ValueError, match="Unsupported model parameter for ridge: fit_intercept"):
        create_point_predictor(
            model_id="ridge",
            random_seed=7,
            feature_columns=FEATURE_COLUMNS,
            n_jobs=1,
            model_params={"fit_intercept": False},
        )


@pytest.mark.parametrize("alpha", [0.0, -1.0])
def test_ridge_model_params_reject_non_positive_alpha(alpha: float) -> None:
    with pytest.raises(ValueError, match="ridge alpha must be positive"):
        create_point_predictor(
            model_id="ridge",
            random_seed=7,
            feature_columns=FEATURE_COLUMNS,
            n_jobs=1,
            model_params={"alpha": alpha},
        )


def test_non_ridge_model_params_are_rejected_for_v1() -> None:
    with pytest.raises(ValueError, match="Model parameter overrides are only supported for ridge"):
        create_point_predictor(
            model_id="random_forest",
            random_seed=7,
            feature_columns=FEATURE_COLUMNS,
            n_jobs=1,
            model_params={"min_samples_leaf": 10},
        )


def test_model_params_none_preserves_current_defaults() -> None:
    model = create_point_predictor(
        model_id="ridge",
        random_seed=7,
        feature_columns=FEATURE_COLUMNS,
        n_jobs=99,
        model_params=None,
    )

    estimator = model.pipeline.named_steps["model"]
    assert isinstance(estimator, Ridge)
    assert estimator.alpha == 1.0
```

- [ ] **Step 2: Run the focused failing tests**

```bash
uv run --frozen pytest src/tests/backtesting/test_model_registry.py \
  -k "ridge_model_params or non_ridge_model_params or model_params_none" -q
```

Expected: tests fail because `create_point_predictor` does not accept `model_params`.

- [ ] **Step 3: Add optional model params to the predictor base class**

In `src/cartola/backtesting/models.py`, update imports:

```python
from collections.abc import Mapping
from typing import Self
```

Update `SklearnPointPredictor.__init__`:

```python
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
        numeric_features = [column for column in self.feature_columns if column != "posicao"]
        categorical_features = ["posicao"] if "posicao" in self.feature_columns else []
```

Update `RidgePointPredictor._make_model`:

```python
    def _make_model(self, *, random_seed: int, n_jobs: int) -> Ridge:
        alpha = float(self.model_params.get("alpha", 1.0))
        return Ridge(alpha=alpha)
```

- [ ] **Step 4: Add registry validation**

In `src/cartola/backtesting/model_registry.py`, import `Mapping`:

```python
from typing import Literal, Mapping, Protocol, cast
```

Add helper functions:

```python
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
```

Update `create_point_predictor`:

```python
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
    overrides = _validate_model_param_overrides(resolved_model_id, model_params)

    return spec.predictor_type(
        random_seed=random_seed,
        feature_columns=feature_columns,
        n_jobs=n_jobs,
        model_params=overrides,
    )
```

- [ ] **Step 5: Run model-registry tests**

```bash
uv run --frozen pytest src/tests/backtesting/test_model_registry.py -q
```

Expected: all tests in `test_model_registry.py` pass.

- [ ] **Step 6: Commit**

```bash
git add src/cartola/backtesting/models.py src/cartola/backtesting/model_registry.py src/tests/backtesting/test_model_registry.py
git commit -m "feat: support ridge model parameter overrides"
```

## Task 2: Backtest Runner Model Params Plumbing

**Files:**
- Modify: `src/cartola/backtesting/runner.py`
- Test: `src/tests/backtesting/test_runner.py`

- [ ] **Step 1: Write failing runner test for experiment-only model params**

Add a test near existing predictor-factory monkeypatch tests in `src/tests/backtesting/test_runner.py`:

```python
def test_experiment_runner_passes_model_params_to_predictor(monkeypatch: pytest.MonkeyPatch, minimal_backtest_config: BacktestConfig) -> None:
    observed: list[dict[str, object]] = []

    class RecordingPointPredictor:
        def __init__(
            self,
            *,
            random_seed: int,
            feature_columns: list[str],
            n_jobs: int,
            model_params: dict[str, object] | None = None,
        ) -> None:
            observed.append(
                {
                    "random_seed": random_seed,
                    "feature_columns": feature_columns,
                    "n_jobs": n_jobs,
                    "model_params": model_params,
                }
            )

        def fit(self, frame: pd.DataFrame) -> "RecordingPointPredictor":
            return self

        def predict(self, frame: pd.DataFrame) -> pd.Series:
            return pd.Series(1.0, index=frame.index)

    def recording_create_point_predictor(**kwargs: object) -> RecordingPointPredictor:
        return RecordingPointPredictor(
            random_seed=int(kwargs["random_seed"]),
            feature_columns=list(kwargs["feature_columns"]),
            n_jobs=int(kwargs["n_jobs"]),
            model_params=kwargs.get("model_params"),
        )

    monkeypatch.setattr("cartola.backtesting.runner.create_point_predictor", recording_create_point_predictor)

    run_backtest_for_experiment(
        minimal_backtest_config,
        primary_model_id="ridge",
        model_params={"alpha": 3.0},
    )

    assert observed
    assert all(call["model_params"] == {"alpha": 3.0} for call in observed)
```

If the local fixture names differ, reuse the smallest existing runner test fixture that creates a playable mini season. Do not add raw files under `data/`.

- [ ] **Step 2: Run the focused failing test**

```bash
uv run --frozen pytest src/tests/backtesting/test_runner.py::test_experiment_runner_passes_model_params_to_predictor -q
```

Expected: fail because `run_backtest_for_experiment` does not accept `model_params`.

- [ ] **Step 3: Add model params to experiment-only runner path**

In `src/cartola/backtesting/runner.py`, import `Mapping` if not already imported.

Update signatures:

```python
def run_backtest_for_experiment(
    config: BacktestConfig,
    *,
    primary_model_id: str,
    model_params: Mapping[str, object] | None = None,
    season_df: pd.DataFrame | None = None,
    fixtures: pd.DataFrame | None = None,
) -> BacktestResult:
    return _run_backtest(
        config,
        primary_model_id=primary_model_id,
        model_params=model_params,
        season_df=season_df,
        fixtures=fixtures,
    )
```

Keep `run_backtest(config)` passing no params:

```python
def run_backtest(...) -> BacktestResult:
    return _run_backtest(
        config,
        primary_model_id="random_forest",
        model_params=None,
        season_df=season_df,
        fixtures=fixtures,
    )
```

Pass `model_params` through `_run_backtest`, `_run_round_workers`, and `_evaluate_target_round`, then into `create_point_predictor`:

```python
primary_model = create_point_predictor(
    model_id=primary_model_id,
    random_seed=config.random_seed,
    feature_columns=model_feature_columns,
    n_jobs=model_n_jobs_effective,
    model_params=model_params,
).fit(training)
```

- [ ] **Step 4: Run runner and model-feature experiment tests**

```bash
uv run --frozen pytest \
  src/tests/backtesting/test_runner.py::test_experiment_runner_passes_model_params_to_predictor \
  src/tests/backtesting/test_experiment_runner.py \
  src/tests/backtesting/test_run_model_experiments_cli.py \
  -q
```

Expected: pass. This confirms existing model-feature experiments still call the runner without model params.

- [ ] **Step 5: Commit**

```bash
git add src/cartola/backtesting/runner.py src/tests/backtesting/test_runner.py
git commit -m "feat: pass experiment model params into backtests"
```

## Task 3: Fixed Ridge Tuning Candidate Matrix

**Files:**
- Create: `src/cartola/backtesting/ridge_tuning_config.py`
- Test: `src/tests/backtesting/test_ridge_tuning_config.py`

- [ ] **Step 1: Write failing candidate-matrix tests**

Create `src/tests/backtesting/test_ridge_tuning_config.py`:

```python
from __future__ import annotations

from pathlib import Path

import pytest

from cartola.backtesting.ridge_tuning_config import (
    PRIMARY_INCUMBENT_CANDIDATE_ID,
    RIDGE_ALPHA_VALUES,
    RIDGE_TUNING_FEATURE_PACKS,
    RidgeTuningStage,
    build_ridge_tuning_specs,
    candidate_id_for,
)


def test_fixed_alpha_and_feature_pack_lists() -> None:
    assert RIDGE_ALPHA_VALUES == (0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0)
    assert RIDGE_TUNING_FEATURE_PACKS == ("ppg", "ppg_xg")


@pytest.mark.parametrize(
    ("alpha", "feature_pack", "candidate_id"),
    [
        (0.01, "ppg", "ridge_alpha_0_01__ppg"),
        (1.0, "ppg_xg", "ridge_alpha_1_0__ppg_xg"),
        (300.0, "ppg_xg", "ridge_alpha_300_0__ppg_xg"),
    ],
)
def test_candidate_id_for_is_stable(alpha: float, feature_pack: str, candidate_id: str) -> None:
    assert candidate_id_for(alpha=alpha, feature_pack=feature_pack) == candidate_id


def test_build_ridge_tuning_specs_contains_every_alpha_feature_season(tmp_path: Path) -> None:
    specs = build_ridge_tuning_specs(
        seasons=(2023, 2024),
        start_round=5,
        budget=100.0,
        project_root=tmp_path,
        output_root=Path("data/08_reporting/experiments/model_tuning/exp"),
        current_year=2026,
        jobs=12,
        stage="screen",
    )

    assert len(specs) == 2 * len(RIDGE_ALPHA_VALUES) * len(RIDGE_TUNING_FEATURE_PACKS)
    assert {spec.model_id for spec in specs} == {"ridge"}
    assert {spec.feature_pack for spec in specs} == {"ppg", "ppg_xg"}
    assert {spec.stage for spec in specs} == {"screen"}
    assert PRIMARY_INCUMBENT_CANDIDATE_ID in {spec.candidate_id for spec in specs}


def test_build_ridge_tuning_specs_rejects_current_year(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Tuning seasons must be before current_year"):
        build_ridge_tuning_specs(
            seasons=(2026,),
            start_round=5,
            budget=100.0,
            project_root=tmp_path,
            output_root=Path("data/08_reporting/experiments/model_tuning/exp"),
            current_year=2026,
            jobs=12,
            stage="screen",
        )


def test_model_param_and_generation_hash_change_with_alpha(tmp_path: Path) -> None:
    specs = build_ridge_tuning_specs(
        seasons=(2025,),
        start_round=5,
        budget=100.0,
        project_root=tmp_path,
        output_root=Path("data/08_reporting/experiments/model_tuning/exp"),
        current_year=2026,
        jobs=12,
        stage="screen",
    )

    ppg_xg = [spec for spec in specs if spec.feature_pack == "ppg_xg"]
    assert len({spec.model_params_hash for spec in ppg_xg}) == len(RIDGE_ALPHA_VALUES)
    assert len({spec.tuning_generation_hash for spec in specs}) == 1
```

- [ ] **Step 2: Run failing tests**

```bash
uv run --frozen pytest src/tests/backtesting/test_ridge_tuning_config.py -q
```

Expected: import failure because `ridge_tuning_config.py` does not exist.

- [ ] **Step 3: Implement tuning config module**

Create `src/cartola/backtesting/ridge_tuning_config.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Mapping

from cartola.backtesting.config import BacktestConfig
from cartola.backtesting.experiment_config import FeaturePackId, config_hash, feature_pack_to_modes
from cartola.backtesting.scoring_contract import SCORING_CONTRACT_VERSION

RidgeTuningStage = Literal["screen", "final"]
RidgeTuningFeaturePack = Literal["ppg", "ppg_xg"]

RIDGE_ALPHA_VALUES: tuple[float, ...] = (0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0)
RIDGE_TUNING_FEATURE_PACKS: tuple[RidgeTuningFeaturePack, ...] = ("ppg", "ppg_xg")
PRIMARY_INCUMBENT_CANDIDATE_ID = "ridge_alpha_1_0__ppg_xg"
SECONDARY_CONTROL_CANDIDATE_ID = "ridge_alpha_1_0__ppg"


@dataclass(frozen=True)
class RidgeTuningSpec:
    stage: RidgeTuningStage
    season: int
    candidate_id: str
    model_id: str
    feature_pack: RidgeTuningFeaturePack
    alpha: float
    start_round: int
    budget: float
    current_year: int
    jobs: int
    model_parameters: Mapping[str, object]
    model_params_hash: str
    tuning_generation_hash: str
    output_path: Path
    backtest_config: BacktestConfig
    config_identity: Mapping[str, object]
```

Add functions:

```python
def candidate_id_for(*, alpha: float, feature_pack: str) -> str:
    encoded_alpha = str(float(alpha)).replace(".", "_")
    return f"ridge_alpha_{encoded_alpha}__{feature_pack}"


def build_ridge_tuning_specs(
    *,
    seasons: tuple[int, ...],
    start_round: int,
    budget: float,
    project_root: Path,
    output_root: Path,
    current_year: int,
    jobs: int,
    stage: RidgeTuningStage,
    candidate_ids: set[str] | None = None,
) -> list[RidgeTuningSpec]:
    if any(season >= current_year for season in seasons):
        raise ValueError("Tuning seasons must be before current_year")

    generation_identity = {
        "experiment_type": "ridge-alpha-tuning",
        "alphas": RIDGE_ALPHA_VALUES,
        "feature_packs": RIDGE_TUNING_FEATURE_PACKS,
        "seasons": seasons,
        "start_round": start_round,
        "budget": budget,
        "current_year": current_year,
        "jobs": jobs,
        "fixture_mode": "none",
        "matchup_context_mode": "none",
        "scoring_contract_version": SCORING_CONTRACT_VERSION,
    }
    generation_hash = config_hash(generation_identity)

    specs: list[RidgeTuningSpec] = []
    for season in seasons:
        for feature_pack_id in RIDGE_TUNING_FEATURE_PACKS:
            feature_pack = feature_pack_to_modes(feature_pack_id)
            for alpha in RIDGE_ALPHA_VALUES:
                candidate_id = candidate_id_for(alpha=alpha, feature_pack=feature_pack_id)
                if candidate_ids is not None and candidate_id not in candidate_ids:
                    continue
                model_parameters = {"estimator": "sklearn.linear_model.Ridge", "alpha": alpha}
                model_params_hash = config_hash(model_parameters)
                child_output_path = (
                    project_root
                    / output_root
                    / "runs"
                    / f"stage={stage}"
                    / f"season={season}"
                    / f"candidate={candidate_id}"
                )
                backtest_config = BacktestConfig(
                    season=season,
                    start_round=start_round,
                    budget=budget,
                    project_root=project_root,
                    output_root=output_root,
                    fixture_mode="none",
                    matchup_context_mode="none",
                    footystats_mode=feature_pack.footystats_mode,
                    current_year=current_year,
                    jobs=jobs,
                    _output_path_override=child_output_path,
                )
                config_identity = {
                    **generation_identity,
                    "stage": stage,
                    "season": season,
                    "candidate_id": candidate_id,
                    "model_id": "ridge",
                    "feature_pack": feature_pack_id,
                    "footystats_mode": feature_pack.footystats_mode,
                    "model_parameters": model_parameters,
                    "model_params_hash": model_params_hash,
                }
                specs.append(
                    RidgeTuningSpec(
                        stage=stage,
                        season=season,
                        candidate_id=candidate_id,
                        model_id="ridge",
                        feature_pack=feature_pack_id,
                        alpha=alpha,
                        start_round=start_round,
                        budget=budget,
                        current_year=current_year,
                        jobs=jobs,
                        model_parameters=model_parameters,
                        model_params_hash=model_params_hash,
                        tuning_generation_hash=generation_hash,
                        output_path=child_output_path,
                        backtest_config=backtest_config,
                        config_identity=config_identity,
                    )
                )
    return specs
```

- [ ] **Step 4: Run candidate config tests**

```bash
uv run --frozen pytest src/tests/backtesting/test_ridge_tuning_config.py -q
```

Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add src/cartola/backtesting/ridge_tuning_config.py src/tests/backtesting/test_ridge_tuning_config.py
git commit -m "feat: add fixed ridge tuning matrix"
```

## Task 4: Tuning Metrics And Promotion Gates

**Files:**
- Create: `src/cartola/backtesting/ridge_tuning_metrics.py`
- Test: `src/tests/backtesting/test_ridge_tuning_metrics.py`

- [ ] **Step 1: Write failing promotion and ranking tests**

Create `src/tests/backtesting/test_ridge_tuning_metrics.py`:

```python
from __future__ import annotations

import pandas as pd

from cartola.backtesting.ridge_tuning_metrics import (
    PRACTICAL_LIFT_PER_ROUND,
    promotion_decision,
    rank_tuning_summary,
)


def test_promotion_decision_requires_practical_lift() -> None:
    decision = promotion_decision(
        aggregate_delta=50.99,
        total_rounds=102,
        improved_seasons=3,
        worst_season_avg_delta=0.0,
        selected_calibration_slope=0.9,
        top50_spearman_delta=0.0,
        candidate_pool_mae_delta_pct=0.0,
        selected_players_mae_delta_pct=0.0,
        comparable=True,
        final_reproducible=True,
    )

    assert PRACTICAL_LIFT_PER_ROUND == 0.5
    assert decision == {"eligible": False, "reason": "lift_below_practical_threshold"}


def test_promotion_decision_accepts_clean_practical_lift() -> None:
    decision = promotion_decision(
        aggregate_delta=51.0,
        total_rounds=102,
        improved_seasons=2,
        worst_season_avg_delta=-0.5,
        selected_calibration_slope=0.9,
        top50_spearman_delta=-0.03,
        candidate_pool_mae_delta_pct=0.05,
        selected_players_mae_delta_pct=0.05,
        comparable=True,
        final_reproducible=True,
    )

    assert decision == {"eligible": True, "reason": "passes_tuning_guardrails"}


def test_promotion_decision_rejects_null_metric() -> None:
    decision = promotion_decision(
        aggregate_delta=None,
        total_rounds=102,
        improved_seasons=3,
        worst_season_avg_delta=0.0,
        selected_calibration_slope=0.9,
        top50_spearman_delta=0.0,
        candidate_pool_mae_delta_pct=0.0,
        selected_players_mae_delta_pct=0.0,
        comparable=True,
        final_reproducible=True,
    )

    assert decision == {"eligible": False, "reason": "insufficient_metric_data"}


def test_rank_tuning_summary_groups_by_candidate_id() -> None:
    per_season = pd.DataFrame(
        [
            {
                "candidate_id": "ridge_alpha_1_0__ppg_xg",
                "season": 2025,
                "model_id": "ridge",
                "feature_pack": "ppg_xg",
                "alpha": 1.0,
                "rounds": 34,
                "total_actual_points": 2100.0,
                "total_predicted_points": 2110.0,
            },
            {
                "candidate_id": "ridge_alpha_3_0__ppg_xg",
                "season": 2025,
                "model_id": "ridge",
                "feature_pack": "ppg_xg",
                "alpha": 3.0,
                "rounds": 34,
                "total_actual_points": 2120.0,
                "total_predicted_points": 2115.0,
            },
            {
                "candidate_id": "ridge_alpha_3_0__ppg",
                "season": 2025,
                "model_id": "ridge",
                "feature_pack": "ppg",
                "alpha": 3.0,
                "rounds": 34,
                "total_actual_points": 2050.0,
                "total_predicted_points": 2060.0,
            },
        ]
    )
    prediction_metrics = pd.DataFrame(
        [
            {
                "candidate_id": candidate_id,
                "metric_scope": metric_scope,
                "mae": 1.0,
                "spearman": 0.1,
                "calibration_slope": 0.9,
            }
            for candidate_id in per_season["candidate_id"].unique()
            for metric_scope in ("candidate_pool", "selected_players", "top50_candidates")
        ]
    )

    ranked = rank_tuning_summary(
        per_season,
        prediction_metrics,
        primary_incumbent_candidate_id="ridge_alpha_1_0__ppg_xg",
        final_reproducibility_by_candidate={candidate_id: True for candidate_id in per_season["candidate_id"].unique()},
    )

    assert list(ranked["candidate_id"]) == [
        "ridge_alpha_3_0__ppg_xg",
        "ridge_alpha_1_0__ppg_xg",
        "ridge_alpha_3_0__ppg",
    ]
    assert set(ranked["alpha"]) == {1.0, 3.0}
```

- [ ] **Step 2: Run failing metrics tests**

```bash
uv run --frozen pytest src/tests/backtesting/test_ridge_tuning_metrics.py -q
```

Expected: import failure.

- [ ] **Step 3: Implement promotion gate function**

Create `src/cartola/backtesting/ridge_tuning_metrics.py` with:

```python
from __future__ import annotations

from typing import Mapping, SupportsFloat, SupportsInt, cast

import pandas as pd

PRACTICAL_LIFT_PER_ROUND = 0.5


def promotion_decision(
    *,
    aggregate_delta: float | None,
    total_rounds: int | None,
    improved_seasons: int | None,
    worst_season_avg_delta: float | None,
    selected_calibration_slope: float | None,
    top50_spearman_delta: float | None,
    candidate_pool_mae_delta_pct: float | None,
    selected_players_mae_delta_pct: float | None,
    comparable: bool,
    final_reproducible: bool,
) -> dict[str, object]:
    if not comparable:
        return {"eligible": False, "reason": "not_comparable"}
    if not final_reproducible:
        return {"eligible": False, "reason": "non_reproducible"}

    required = (
        aggregate_delta,
        total_rounds,
        improved_seasons,
        worst_season_avg_delta,
        selected_calibration_slope,
        top50_spearman_delta,
        candidate_pool_mae_delta_pct,
        selected_players_mae_delta_pct,
    )
    if any(_is_missing(value) for value in required):
        return {"eligible": False, "reason": "insufficient_metric_data"}

    aggregate_delta_value = float(cast("SupportsFloat", aggregate_delta))
    total_rounds_value = int(cast("SupportsInt", total_rounds))
    improved_seasons_value = int(cast("SupportsInt", improved_seasons))
    worst_season_avg_delta_value = float(cast("SupportsFloat", worst_season_avg_delta))
    selected_calibration_slope_value = float(cast("SupportsFloat", selected_calibration_slope))
    top50_spearman_delta_value = float(cast("SupportsFloat", top50_spearman_delta))
    candidate_pool_mae_delta_pct_value = float(cast("SupportsFloat", candidate_pool_mae_delta_pct))
    selected_players_mae_delta_pct_value = float(cast("SupportsFloat", selected_players_mae_delta_pct))

    if total_rounds_value <= 0:
        return {"eligible": False, "reason": "insufficient_metric_data"}
    if aggregate_delta_value < PRACTICAL_LIFT_PER_ROUND * total_rounds_value:
        return {"eligible": False, "reason": "lift_below_practical_threshold"}
    if improved_seasons_value < 2:
        return {"eligible": False, "reason": "fewer_than_two_seasons_improved"}
    if worst_season_avg_delta_value < -0.5:
        return {"eligible": False, "reason": "worst_season_regression_exceeds_threshold"}
    if selected_calibration_slope_value < 0.75 or selected_calibration_slope_value > 1.25:
        return {"eligible": False, "reason": "selected_calibration_slope_out_of_range"}
    if top50_spearman_delta_value < -0.03:
        return {"eligible": False, "reason": "top50_spearman_regression_exceeds_threshold"}
    if candidate_pool_mae_delta_pct_value > 0.05:
        return {"eligible": False, "reason": "candidate_pool_mae_regression_exceeds_threshold"}
    if selected_players_mae_delta_pct_value > 0.05:
        return {"eligible": False, "reason": "selected_players_mae_regression_exceeds_threshold"}
    return {"eligible": True, "reason": "passes_tuning_guardrails"}
```

Include helpers:

```python
def _is_missing(value: object) -> bool:
    if value is None:
        return True
    missing = pd.isna(value)
    if isinstance(missing, bool):
        return missing
    return False
```

- [ ] **Step 4: Implement ranked summary**

In `ridge_tuning_metrics.py`, add `rank_tuning_summary(...)` that:

- groups by `candidate_id`, `model_id`, `feature_pack`, and `alpha`;
- computes totals and average actual points;
- compares every candidate to `primary_incumbent_candidate_id`;
- computes `aggregate_delta_vs_primary_incumbent`;
- computes `average_delta_per_round_vs_primary_incumbent`;
- computes `improved_seasons_vs_primary_incumbent`;
- computes `worst_season_avg_delta_vs_primary_incumbent`;
- computes candidate-pool and selected-player MAE delta percentages;
- computes selected-player calibration slope and top-50 Spearman delta;
- calls `promotion_decision`;
- sorts eligible candidates first, then by aggregate delta, total actual points, candidate id.

Use exact output columns from the spec.

- [ ] **Step 5: Run metrics tests**

```bash
uv run --frozen pytest src/tests/backtesting/test_ridge_tuning_metrics.py -q
```

Expected: pass.

- [ ] **Step 6: Commit**

```bash
git add src/cartola/backtesting/ridge_tuning_metrics.py src/tests/backtesting/test_ridge_tuning_metrics.py
git commit -m "feat: add ridge tuning promotion metrics"
```

## Task 5: Ridge Tuning Runner Orchestration

**Files:**
- Create: `src/cartola/backtesting/ridge_tuning_runner.py`
- Test: `src/tests/backtesting/test_ridge_tuning_runner.py`

- [ ] **Step 1: Write failing orchestration tests with monkeypatched backtests**

Create `src/tests/backtesting/test_ridge_tuning_runner.py`:

```python
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from cartola.backtesting.ridge_tuning_runner import run_ridge_tuning
from cartola.backtesting.runner import BacktestMetadata, BacktestResult


def _fake_result(*, total_actual_points: float, strategy: str = "ridge") -> BacktestResult:
    round_results = pd.DataFrame(
        [
            {
                "rodada": 5,
                "strategy": strategy,
                "solver_status": "Optimal",
                "formation": "4-3-3",
                "selected_count": 12,
                "budget_used": 100.0,
                "predicted_points": total_actual_points,
                "predicted_points_base": total_actual_points,
                "captain_bonus_predicted": 0.0,
                "predicted_points_with_captain": total_actual_points,
                "actual_points": total_actual_points,
                "actual_points_base": total_actual_points,
                "captain_bonus_actual": 0.0,
                "actual_points_with_captain": total_actual_points,
                "captain_id": 1,
                "captain_name": "A",
                "captain_policy_ev_id": 1,
                "captain_policy_safe_id": 1,
                "captain_policy_upside_id": 1,
                "actual_points_with_ev_captain": total_actual_points,
                "actual_points_with_safe_captain": total_actual_points,
                "actual_points_with_upside_captain": total_actual_points,
            }
        ]
    )
    selected_players = pd.DataFrame(
        [
            {
                "rodada": 5,
                "strategy": strategy,
                "id_atleta": 1,
                "apelido": "A",
                "posicao": "ata",
                "id_clube": 1,
                "status": "Provavel",
                "preco_pre_rodada": 10.0,
                "predicted_points": total_actual_points,
                "pontuacao": total_actual_points,
                "is_captain": True,
            }
        ]
    )
    player_predictions = pd.DataFrame(
        [
            {
                "rodada": 5,
                "id_atleta": 1,
                "apelido": "A",
                "posicao": "ata",
                "id_clube": 1,
                "status": "Provavel",
                "preco_pre_rodada": 10.0,
                "ridge_score": total_actual_points,
                "pontuacao": total_actual_points,
            }
        ]
    )
    summary = pd.DataFrame(
        [
            {
                "strategy": strategy,
                "rounds": 1,
                "total_actual_points": total_actual_points,
                "average_actual_points": total_actual_points,
                "total_predicted_points": total_actual_points,
                "average_predicted_points": total_actual_points,
            }
        ]
    )
    metadata = BacktestMetadata(
        season=2025,
        start_round=5,
        max_round=5,
        cache_enabled=True,
        prediction_frames_built=1,
        wall_clock_seconds=1.0,
        backtest_jobs=1,
        backtest_workers_effective=1,
        model_n_jobs_effective=-1,
        parallel_backend="sequential",
        thread_env={},
        scoring_contract_version="cartola_standard_2026_v1",
        captain_scoring_enabled=True,
        captain_multiplier=1.5,
        formation_search="all_official_formations",
        fixture_mode="none",
        strict_alignment_policy="fail",
        matchup_context_mode="none",
        matchup_context_feature_columns=[],
        fixture_source_directory=None,
        fixture_manifest_paths=[],
        fixture_manifest_sha256={},
        generator_versions=[],
        excluded_rounds=[],
        warnings=[],
        footystats_mode="ppg_xg",
        footystats_evaluation_scope="historical_candidate",
        footystats_league_slug="brazil-serie-a",
        footystats_matches_source_path=None,
        footystats_matches_source_sha256=None,
        footystats_feature_columns=[],
        footystats_missing_join_keys_by_round={},
        footystats_duplicate_join_keys_by_round={},
        footystats_extra_club_rows_by_round={},
    )
    return BacktestResult(
        round_results=round_results,
        selected_players=selected_players,
        player_predictions=player_predictions,
        summary=summary,
        diagnostics=pd.DataFrame(),
        metadata=metadata,
    )


def test_run_ridge_tuning_passes_alpha_params_and_writes_reports(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    observed: list[dict[str, object]] = []

    def fake_run_backtest_for_experiment(config: object, *, primary_model_id: str, model_params: dict[str, object]) -> BacktestResult:
        observed.append({"primary_model_id": primary_model_id, "model_params": model_params})
        return _fake_result(total_actual_points=100.0 + float(model_params["alpha"]))

    monkeypatch.setattr("cartola.backtesting.ridge_tuning_runner.run_backtest_for_experiment", fake_run_backtest_for_experiment)

    result = run_ridge_tuning(
        seasons=(2025,),
        start_round=5,
        budget=100.0,
        current_year=2026,
        jobs=1,
        project_root=tmp_path,
        output_root=Path("data/08_reporting/experiments/model_tuning"),
        started_at_utc="20260501T120000000000Z",
        skip_final_rerun=True,
    )

    assert result.experiment_id.startswith("group=ridge-alpha-tuning__")
    assert (result.output_path / "tuning_generation_manifest.json").exists()
    assert (result.output_path / "ranked_summary.csv").exists()
    assert (result.output_path / "promotion_report.json").exists()
    assert observed
    assert {call["primary_model_id"] for call in observed} == {"ridge"}
    assert {call["model_params"]["alpha"] for call in observed} >= {0.01, 1.0, 300.0}


def test_run_ridge_tuning_rejects_current_year(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Tuning seasons must be before current_year"):
        run_ridge_tuning(
            seasons=(2026,),
            start_round=5,
            budget=100.0,
            current_year=2026,
            jobs=1,
            project_root=tmp_path,
            output_root=Path("data/08_reporting/experiments/model_tuning"),
            started_at_utc="20260501T120000000000Z",
        )
```

- [ ] **Step 2: Run failing runner tests**

```bash
uv run --frozen pytest src/tests/backtesting/test_ridge_tuning_runner.py -q
```

Expected: import failure.

- [ ] **Step 3: Implement runner result and progress event dataclasses**

Create `src/cartola/backtesting/ridge_tuning_runner.py` with:

```python
from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Literal

import pandas as pd

from cartola.backtesting.experiment_config import config_hash
from cartola.backtesting.ridge_tuning_config import (
    PRIMARY_INCUMBENT_CANDIDATE_ID,
    SECONDARY_CONTROL_CANDIDATE_ID,
    RidgeTuningSpec,
    build_ridge_tuning_specs,
)
from cartola.backtesting.runner import CSV_FLOAT_FORMAT, BacktestResult, run_backtest_for_experiment


@dataclass(frozen=True)
class RidgeTuningRunResult:
    experiment_id: str
    output_path: Path
    ranked_summary: pd.DataFrame
    metadata: dict[str, object]


@dataclass(frozen=True)
class RidgeTuningProgressEvent:
    event_type: Literal["experiment_started", "child_started", "child_finished", "child_failed", "experiment_finished"]
    experiment_id: str
    output_path: Path
    total_children: int
    completed_children: int
    child_index: int | None = None
    child_id: str | None = None
    stage: str | None = None
    season: int | None = None
    candidate_id: str | None = None
    feature_pack: str | None = None
    alpha: float | None = None
    elapsed_seconds: float | None = None
    child_duration_seconds: float | None = None
    phase: str | None = None
    message: str | None = None


RidgeTuningProgressCallback = Callable[[RidgeTuningProgressEvent], None]
```

- [ ] **Step 4: Implement screen-stage orchestration**

In `ridge_tuning_runner.py`, implement `run_ridge_tuning(...)`:

- build screen specs with `build_ridge_tuning_specs(stage="screen")`;
- create experiment id `group=ridge-alpha-tuning__started_at=<timestamp>__matrix=<hash[:12]>`;
- reject existing output path;
- write `tuning_generation_manifest.json` before child runs;
- loop children sequentially;
- call:

```python
result = run_backtest_for_experiment(
    spec.backtest_config,
    primary_model_id="ridge",
    model_params={"alpha": spec.alpha},
)
```

- write each child output through the normal backtest runner output path;
- collect per-season rows, prediction metrics, calibration deciles, candidate-pool signatures, and solver-status signatures.

Use local helper functions in this module for:

- `_child_id(spec) -> str`;
- `_primary_summary_rows(spec, result, child_id)`;
- `_prediction_metric_rows(spec, result, child_id)`;
- `_calibration_decile_rows(spec, result, child_id)`;
- `_candidate_signatures_by_round(result.player_predictions)`;
- `_solver_status_signature(result.round_results)`.

Do not import private helpers from `experiment_runner.py` unless this module first extracts them into a shared public helper module with tests. The lower-risk v1 path is local helpers with tuning-specific candidate columns.

- [ ] **Step 5: Implement final-stage selection and rerun**

After screen ranking:

- if `skip_final_rerun=True`, write `promotion_report.json` with `keep_incumbent` and reason `final_rerun_skipped`;
- otherwise choose candidates for final stage:
  - always include `PRIMARY_INCUMBENT_CANDIDATE_ID`;
  - always include `SECONDARY_CONTROL_CANDIDATE_ID`;
  - include top two screen-eligible challengers by `aggregate_delta_vs_primary_incumbent`;
- rerun only those candidate ids with `stage="final"`;
- build final ranked summary;
- compare final totals to screen totals by candidate id;
- mark candidates non-reproducible if total actual points differs by more than `0.01`;
- write `promotion_report.json` from final results.

- [ ] **Step 6: Write top-level artifacts**

Write these files:

```text
tuning_generation_manifest.json
ranked_summary.csv
per_season_summary.csv
prediction_metrics.csv
calibration_deciles.csv
comparability_report.json
promotion_report.json
comparison_report.md
calibration_plots.html
squad_performance_comparison.html
experiment_metadata.json
```

For HTML files, v1 may match the current model-feature stub behavior:

```html
<!doctype html><title>Ridge tuning calibration plots</title>
```

The key acceptance artifact is the CSV/JSON output, not chart polish.

- [ ] **Step 7: Run runner tests**

```bash
uv run --frozen pytest src/tests/backtesting/test_ridge_tuning_runner.py -q
```

Expected: pass.

- [ ] **Step 8: Commit**

```bash
git add src/cartola/backtesting/ridge_tuning_runner.py src/tests/backtesting/test_ridge_tuning_runner.py
git commit -m "feat: add ridge tuning runner"
```

## Task 6: Ridge Tuning CLI

**Files:**
- Create: `scripts/run_ridge_tuning.py`
- Test: `src/tests/backtesting/test_run_ridge_tuning_cli.py`

- [ ] **Step 1: Write failing CLI tests**

Create `src/tests/backtesting/test_run_ridge_tuning_cli.py`:

```python
from __future__ import annotations

import sys
from pathlib import Path
from typing import NoReturn

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_ridge_tuning import main, parse_args  # noqa: E402


def test_parse_args_defaults() -> None:
    args = parse_args(["--current-year", "2026"])

    assert args.seasons == "2023,2024,2025"
    assert args.start_round == 5
    assert args.budget == 100.0
    assert args.current_year == 2026
    assert args.jobs == 1
    assert args.output_root == Path("data/08_reporting/experiments/model_tuning")
    assert args.skip_final_rerun is False


def test_main_calls_runner(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    observed: dict[str, object] = {}

    def fake_run_ridge_tuning(**kwargs: object) -> object:
        observed.update(kwargs)

        class Result:
            output_path = tmp_path / "out"
            experiment_id = "exp"

        return Result()

    monkeypatch.setattr("scripts.run_ridge_tuning.run_ridge_tuning", fake_run_ridge_tuning)

    exit_code = main(
        [
            "--seasons",
            "2023,2024",
            "--current-year",
            "2026",
            "--project-root",
            str(tmp_path),
            "--output-root",
            "data/08_reporting/experiments/model_tuning/test",
            "--jobs",
            "12",
        ]
    )

    assert exit_code == 0
    assert observed["seasons"] == (2023, 2024)
    assert observed["current_year"] == 2026
    assert observed["project_root"] == tmp_path
    assert observed["output_root"] == Path("data/08_reporting/experiments/model_tuning/test")
    assert observed["jobs"] == 12
    assert callable(observed["progress_callback"])


def test_main_rejects_empty_seasons_without_traceback(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main(["--seasons", "", "--current-year", "2026"])

    assert exit_code == 1
    captured = capsys.readouterr()
    assert "At least one season is required" in captured.err
    assert "Traceback" not in captured.err


def test_main_reports_runner_failure_without_traceback(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    def fake_run_ridge_tuning(**_kwargs: object) -> NoReturn:
        raise RuntimeError("tuning failed")

    monkeypatch.setattr("scripts.run_ridge_tuning.run_ridge_tuning", fake_run_ridge_tuning)

    exit_code = main(["--current-year", "2026"])

    assert exit_code == 1
    captured = capsys.readouterr()
    assert "tuning failed" in captured.err
    assert "Traceback" not in captured.err
```

- [ ] **Step 2: Run failing CLI tests**

```bash
uv run --frozen pytest src/tests/backtesting/test_run_ridge_tuning_cli.py -q
```

Expected: import failure.

- [ ] **Step 3: Implement CLI**

Create `scripts/run_ridge_tuning.py` by adapting `scripts/run_model_experiments.py`:

- parser description: `"Run Cartola constrained Ridge tuning."`;
- args:
  - `--seasons`, default `2023,2024,2025`;
  - `--start-round`, default `5`;
  - `--budget`, default `100.0`;
  - `--current-year`, required;
  - `--project-root`, default `Path(".")`;
  - `--output-root`, default `Path("data/08_reporting/experiments/model_tuning")`;
  - `--jobs`, default `1`;
  - `--skip-final-rerun`, action `store_true`;
- progress display labels include `stage`, `season`, `candidate_id`, `feature_pack`, and `alpha`;
- on success, print:

```text
experiment_id=<id>
output_path=<path>
```

- on error, print a red panel and return `1`.

- [ ] **Step 4: Run CLI tests**

```bash
uv run --frozen pytest src/tests/backtesting/test_run_ridge_tuning_cli.py -q
```

Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add scripts/run_ridge_tuning.py src/tests/backtesting/test_run_ridge_tuning_cli.py
git commit -m "feat: add ridge tuning CLI"
```

## Task 7: Full Verification And Documentation Update

**Files:**
- Modify: `roadmap.md`
- Optional Modify: `docs/superpowers/specs/2026-05-01-constrained-ridge-tuning-design.md` only if implementation exposes a necessary clarified behavior.

- [ ] **Step 1: Run focused test suite**

```bash
uv run --frozen pytest \
  src/tests/backtesting/test_model_registry.py \
  src/tests/backtesting/test_runner.py \
  src/tests/backtesting/test_ridge_tuning_config.py \
  src/tests/backtesting/test_ridge_tuning_metrics.py \
  src/tests/backtesting/test_ridge_tuning_runner.py \
  src/tests/backtesting/test_run_ridge_tuning_cli.py \
  src/tests/backtesting/test_experiment_runner.py \
  src/tests/backtesting/test_run_model_experiments_cli.py \
  -q
```

Expected: all selected tests pass.

- [ ] **Step 2: Run repository quality gate**

```bash
uv run --frozen scripts/pyrepo-check --all
```

Expected: Ruff, ty, Bandit, and pytest all pass.

- [ ] **Step 3: Run a smoke tuning command with final rerun skipped**

Use one completed historical season to verify CLI wiring and artifacts without paying the full matrix cost:

```bash
uv run --frozen python scripts/run_ridge_tuning.py \
  --seasons 2025 \
  --start-round 5 \
  --budget 100 \
  --current-year 2026 \
  --jobs 4 \
  --skip-final-rerun
```

Expected:

- command exits `0`;
- output path is under `data/08_reporting/experiments/model_tuning/`;
- `ranked_summary.csv` exists;
- `promotion_report.json` exists and does not promote because final rerun was skipped;
- `experiment_metadata.json` records all candidate ids and alpha params.

- [ ] **Step 4: Update roadmap after implementation**

In `roadmap.md`, mark constrained Ridge tuning as implemented and add the smoke command. Do not claim a tuned alpha is better until the full 2023-2025 run completes and passes final rerun gates.

- [ ] **Step 5: Commit verification/docs**

```bash
git add roadmap.md docs/superpowers/specs/2026-05-01-constrained-ridge-tuning-design.md
git commit -m "docs: update roadmap for ridge tuning runner"
```

If neither file changed, skip this commit.

## Execution Notes

- Keep `scripts/run_model_experiments.py` behavior unchanged.
- Keep `scripts/run_live_round.py` behavior unchanged.
- Keep public `python -m cartola.backtesting.cli` behavior unchanged.
- Do not add `--alphas` in v1.
- Do not expose arbitrary live model parameters in v1.
- Do not introduce Optuna, RidgeCV, HGB tuning, external libraries, or tree calibration in this implementation.

## Final Run Command

After implementation and smoke testing, the real tuning run is:

```bash
/usr/bin/time -p uv run --frozen python scripts/run_ridge_tuning.py \
  --seasons 2023,2024,2025 \
  --start-round 5 \
  --budget 100 \
  --current-year 2026 \
  --jobs 20
```

Interpret only final-stage results for promotion.
