# H004 Feature Pack Phase 2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the frozen `ppg_xg_matchup_h004` feature pack and run the side-by-side H004 control/challenger experiment.

**Architecture:** Add an internal `feature_augmentation_mode` to `BacktestConfig` so normal `ppg_xg_matchup` remains unchanged while `ppg_xg_matchup_h004` receives exactly four H004 model columns. Thread that mode through feature-frame builders and the model-feature experiment config; reuse the existing experiment runner and reports instead of building a custom H004 runner.

**Tech Stack:** Python 3.13, pandas, existing Cartola backtesting feature pipeline, existing model-feature experiment runner, pytest, Ruff, ty.

---

## File Structure

- Modify `src/cartola/backtesting/config.py`
  - Add `FeatureAugmentationMode = Literal["none", "h004_attack_defense_v1"]`.
  - Add `feature_augmentation_mode` to `BacktestConfig`, defaulting to `"none"`.
- Modify `src/cartola/backtesting/features.py`
  - Add `H004_ATTACK_DEFENSE_FEATURE_COLUMNS`.
  - Add `feature_augmentation_mode` parameters to prediction/training frame builders.
  - Add `_add_h004_attack_defense_features`.
  - Extend `feature_columns_for_config` only when H004 mode is enabled.
- Modify `src/cartola/backtesting/runner.py`
  - Pass `config.feature_augmentation_mode` into prediction/training frame builders.
  - Record H004 feature columns in metadata when enabled.
- Modify `src/cartola/backtesting/experiment_config.py`
  - Add `FeaturePackId = "ppg_xg_matchup_h004"`.
  - Add experiment group `h004-attack-defense-mismatch`.
  - Map control and challenger feature packs without changing existing groups.
- Create `src/cartola/backtesting/h004_feature_decision.py`
  - Read a completed Phase 2 experiment.
  - Validate Phase 1 diagnostic precondition, control/challenger comparability, fixture identity, and frozen H004 gates.
  - Write deterministic `h004_phase2_decision.json`.
- Create `scripts/run_h004_feature_decision.py`
  - CLI wrapper around the decision module.
  - Prints a concise decision summary after the experiment run.
- Modify tests:
  - `src/tests/backtesting/test_features.py`
  - `src/tests/backtesting/test_runner.py`
  - `src/tests/backtesting/test_experiment_config.py`
  - `src/tests/backtesting/test_h004_feature_decision.py`
  - `src/tests/backtesting/test_run_model_experiments_cli.py` if the group validation has CLI coverage.
- Modify `roadmap.md` after the real experiment run.

## Frozen H004 Columns

```text
h004_position_softness_delta =
  matchup_opponent_allowed_position_points_roll5 - matchup_opponent_allowed_points_roll5

h004_position_mismatch_score =
  (matchup_club_position_points_roll5 - position_points_prior)
  + (matchup_opponent_allowed_position_points_roll5 - position_points_prior)

h004_home_xg_edge =
  matchup_is_home * footystats_xg_diff

h004_role_xg_mismatch =
  footystats_xg_diff * h004_position_softness_delta
```

For `posicao == "tec"`, all four H004 values must be `0.0`.

## Frozen Phase 2 Decision Contract

The Phase 2 result is not interpreted manually. The run writes `h004_phase2_decision.json` with these fields:

```json
{
  "hypothesis_id": "H004",
  "phase": "feature_pack_phase2",
  "control": {
    "model_id": "xgboost_depth2_slow",
    "feature_pack": "ppg_xg_matchup"
  },
  "challenger": {
    "model_id": "xgboost_depth2_slow",
    "feature_pack": "ppg_xg_matchup_h004"
  },
  "phase1_precondition_status": "passes",
  "fixture_identity_status": "verified|unverified|mismatch|missing",
  "candidate_signature_status": "ok|mismatch|missing",
  "final_status": "candidate_research|diagnostic_only|rejected|invalid",
  "gate_results": {},
  "season_deltas": [],
  "metric_deltas": [],
  "budget_deltas": [],
  "reasons": []
}
```

Source constraints:

- The Phase 1 precondition must read `h004_diagnostic_decision.json` from `data/08_reporting/hypotheses/h004_residual_diagnostic_started_at=20260508T182202655139Z/` and require `diagnostic_status == "passes"` plus `"C"` in `passed_families`.
- The Phase 2 experiment must evaluate exactly seasons `2021,2022,2023,2024,2025`.
- The control must be `xgboost_depth2_slow + ppg_xg_matchup`.
- The challenger must be `xgboost_depth2_slow + ppg_xg_matchup_h004`.
- Both children for each season must have `budget_policy == "moving"`, `fixture_mode == "exploratory"`, `scoring_contract_version == "cartola_standard_2026_v1"`, `footystats_mode == "ppg_xg"`, and `matchup_context_mode == "cartola_matchup_v1"`.
- Candidate signatures are the existing experiment-level identity signatures over `id_atleta`, `posicao`, `id_clube`, `status`, `preco_pre_rodada`, and `rodada`; the decision module validates they match for control/challenger by season and round. Feature columns and model predictions are intentionally excluded from the candidate signature.
- Fixture identity is `verified` only when the child metadata contains source fixture hashes and the source hashes match between control and challenger for every season. Missing hashes produce `fixture_identity_status="unverified"`, not a promotion-capable result.

Frozen gates:

```text
gate.aggregate_delta_pass:
  challenger_total_actual_points - control_total_actual_points >= +85.0

gate.improved_seasons_pass:
  count(season_delta > 0.0) >= 4

gate.worst_season_delta_pass:
  min(season_delta) >= -20.0

gate.recent_season_delta_pass:
  season_delta[2025] >= -10.0

gate.final_budget_pass:
  min(challenger_final_budget - control_final_budget by season) >= -15.0

gate.min_budget_pass:
  min(challenger_min_budget - control_min_budget by season) >= -15.0

gate.max_drawdown_pass:
  max(challenger_max_budget_drawdown - control_max_budget_drawdown by season) <= +15.0

gate.budget_constrained_rounds_pass:
  sum(challenger_budget_constrained_rounds) - sum(control_budget_constrained_rounds) <= +2

gate.top50_spearman_pass:
  count((challenger_top50_spearman - control_top50_spearman) < -0.02) <= 1

gate.selected_calibration_pass:
  every finite challenger selected-player calibration slope is in [0.50, 1.50]
  and every challenger selected-player observed_count >= 120

gate.concentration_pass:
  if aggregate_delta <= 0.0: false
  else sum(two largest positive season_deltas) / sum(all positive season_deltas) < 0.70
```

Decision status:

```text
invalid:
  missing required artifacts, missing required columns, missing Phase 1 decision, failed Phase 1 precondition,
  candidate signature mismatch, scoring-contract mismatch, budget-policy mismatch, fixture-mode mismatch,
  or missing required control/challenger rows.

rejected:
  artifacts are valid but one or more frozen gates fail.

diagnostic_only:
  every frozen gate passes and fixture_identity_status == "unverified".

candidate_research:
  every frozen gate passes and fixture_identity_status == "verified".
```

This decision artifact is the only source used for the roadmap conclusion. Existing `ranked_summary.csv` promotion fields are generic model-experiment fields and must not be used as the H004 Phase 2 decision.

## Task 1: Add Feature-Augmentation Config Plumbing

**Files:**
- Modify: `src/cartola/backtesting/config.py`
- Modify: `src/cartola/backtesting/experiment_config.py`
- Test: `src/tests/backtesting/test_experiment_config.py`

- [ ] **Step 1: Add failing tests for H004 feature pack mapping and group matrix**

Append to `src/tests/backtesting/test_experiment_config.py`:

```python
def test_h004_feature_pack_to_modes() -> None:
    feature_pack = feature_pack_to_modes("ppg_xg_matchup_h004")

    assert feature_pack == FeaturePack(
        feature_pack="ppg_xg_matchup_h004",
        footystats_mode="ppg_xg",
        matchup_context_mode="cartola_matchup_v1",
        feature_augmentation_mode="h004_attack_defense_v1",
    )


def test_h004_attack_defense_mismatch_matrix_is_control_vs_challenger_only() -> None:
    specs = build_child_run_specs(
        group="h004-attack-defense-mismatch",
        seasons=(2021, 2022, 2023, 2024, 2025),
        start_round=5,
        budget=100.0,
        project_root=Path("/repo"),
        output_root=Path("data/08_reporting/experiments/model_feature/test"),
        current_year=2026,
        jobs=12,
    )

    assert len(specs) == 10
    assert {spec.fixture_mode for spec in specs} == {"exploratory"}
    assert {spec.model_id for spec in specs} == {"xgboost_depth2_slow"}
    assert {spec.feature_pack for spec in specs} == {
        "ppg_xg_matchup",
        "ppg_xg_matchup_h004",
    }
    h004_specs = [spec for spec in specs if spec.feature_pack == "ppg_xg_matchup_h004"]
    control_specs = [spec for spec in specs if spec.feature_pack == "ppg_xg_matchup"]
    assert {spec.backtest_config.feature_augmentation_mode for spec in h004_specs} == {
        "h004_attack_defense_v1"
    }
    assert {spec.backtest_config.feature_augmentation_mode for spec in control_specs} == {"none"}
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
uv run --frozen pytest src/tests/backtesting/test_experiment_config.py::test_h004_feature_pack_to_modes src/tests/backtesting/test_experiment_config.py::test_h004_attack_defense_mismatch_matrix_is_control_vs_challenger_only -q
```

Expected: FAIL with unsupported feature pack/group or dataclass field mismatch.

- [ ] **Step 3: Add config type and feature-pack mapping**

In `src/cartola/backtesting/config.py`, add:

```python
FeatureAugmentationMode = Literal["none", "h004_attack_defense_v1"]
```

Add to `BacktestConfig`:

```python
    feature_augmentation_mode: FeatureAugmentationMode = "none"
```

In `src/cartola/backtesting/experiment_config.py`, update imports:

```python
from cartola.backtesting.config import (
    BacktestConfig,
    FeatureAugmentationMode,
    FixtureMode,
    FootyStatsMode,
    MatchupContextMode,
)
```

Update literals:

```python
ExperimentGroup = Literal[
    "production-parity",
    "matchup-research",
    "xgboost-research",
    "xgboost-sensitivity-v2",
    "h004-attack-defense-mismatch",
]
FeaturePackId = Literal["ppg", "ppg_xg", "ppg_matchup", "ppg_xg_matchup", "ppg_xg_matchup_h004"]
```

Update `FeaturePack`:

```python
@dataclass(frozen=True)
class FeaturePack:
    feature_pack: FeaturePackId
    footystats_mode: FootyStatsMode
    matchup_context_mode: MatchupContextMode
    feature_augmentation_mode: FeatureAugmentationMode = "none"
```

Add group entries:

```python
_GROUP_FIXTURE_MODES: Mapping[ExperimentGroup, FixtureMode] = {
    "production-parity": "none",
    "matchup-research": "exploratory",
    "xgboost-research": "exploratory",
    "xgboost-sensitivity-v2": "exploratory",
    "h004-attack-defense-mismatch": "exploratory",
}
```

```python
_GROUP_FEATURE_PACKS: Mapping[ExperimentGroup, tuple[FeaturePackId, ...]] = {
    "production-parity": ("ppg", "ppg_xg"),
    "matchup-research": ("ppg", "ppg_xg", "ppg_matchup", "ppg_xg_matchup"),
    "xgboost-research": ("ppg_xg", "ppg_xg_matchup"),
    "xgboost-sensitivity-v2": ("ppg_xg_matchup",),
    "h004-attack-defense-mismatch": ("ppg_xg_matchup", "ppg_xg_matchup_h004"),
}
```

```python
_GROUP_MODEL_IDS: Mapping[ExperimentGroup, tuple[ModelId, ...]] = {
    "production-parity": ("random_forest", "extra_trees", "hist_gradient_boosting", "ridge"),
    "matchup-research": ("random_forest", "extra_trees", "hist_gradient_boosting", "ridge"),
    "xgboost-research": ("xgboost_conservative", "xgboost_balanced", "xgboost_capacity"),
    "xgboost-sensitivity-v2": (
        "ridge",
        "xgboost_conservative",
        "xgboost_depth1_stumps",
        "xgboost_depth2_slow",
        "xgboost_depth2_fast",
        "xgboost_depth2_more_trees",
        "xgboost_depth2_heavy_child",
        "xgboost_depth2_subsample",
        "xgboost_depth2_l2_heavy",
        "xgboost_depth2_l1_gamma",
        "xgboost_depth3_slow",
    ),
    "h004-attack-defense-mismatch": ("xgboost_depth2_slow",),
}
```

Add feature pack mapping:

```python
    "ppg_xg_matchup_h004": FeaturePack(
        feature_pack="ppg_xg_matchup_h004",
        footystats_mode="ppg_xg",
        matchup_context_mode="cartola_matchup_v1",
        feature_augmentation_mode="h004_attack_defense_v1",
    ),
```

When constructing `BacktestConfig`, pass:

```python
                    feature_augmentation_mode=feature_pack.feature_augmentation_mode,
```

Add to `config_identity`:

```python
                    "feature_augmentation_mode": feature_pack.feature_augmentation_mode,
```

- [ ] **Step 4: Run tests**

Run:

```bash
uv run --frozen pytest src/tests/backtesting/test_experiment_config.py::test_h004_feature_pack_to_modes src/tests/backtesting/test_experiment_config.py::test_h004_attack_defense_mismatch_matrix_is_control_vs_challenger_only -q
```

Expected: `2 passed`.

- [ ] **Step 5: Run focused static checks**

Run:

```bash
uv run --frozen ruff check src/cartola/backtesting/config.py src/cartola/backtesting/experiment_config.py src/tests/backtesting/test_experiment_config.py
uv run --frozen ty check src/cartola/backtesting/config.py src/cartola/backtesting/experiment_config.py src/tests/backtesting/test_experiment_config.py
```

Expected: both pass.

- [ ] **Step 6: Commit**

Run:

```bash
git add src/cartola/backtesting/config.py src/cartola/backtesting/experiment_config.py src/tests/backtesting/test_experiment_config.py
git commit -m "feat: add h004 feature pack config"
```

Expected: commit succeeds.

## Task 2: Implement H004 Feature Formulas

**Files:**
- Modify: `src/cartola/backtesting/features.py`
- Test: `src/tests/backtesting/test_features.py`

- [ ] **Step 1: Add failing tests for H004 feature columns**

Append to `src/tests/backtesting/test_features.py`:

```python
from cartola.backtesting.features import (
    H004_ATTACK_DEFENSE_FEATURE_COLUMNS,
    _add_h004_attack_defense_features,
    build_prediction_frame,
    feature_columns_for_config,
)
from cartola.backtesting.config import BacktestConfig


def test_h004_feature_columns_are_added_only_for_h004_augmentation() -> None:
    base_columns = feature_columns_for_config(
        BacktestConfig(footystats_mode="ppg_xg", matchup_context_mode="cartola_matchup_v1")
    )
    h004_columns = feature_columns_for_config(
        BacktestConfig(
            footystats_mode="ppg_xg",
            matchup_context_mode="cartola_matchup_v1",
            feature_augmentation_mode="h004_attack_defense_v1",
        )
    )

    assert not set(H004_ATTACK_DEFENSE_FEATURE_COLUMNS).intersection(base_columns)
    assert set(H004_ATTACK_DEFENSE_FEATURE_COLUMNS).issubset(h004_columns)


def test_h004_feature_columns_require_matchup_and_xg_context() -> None:
    with pytest.raises(ValueError, match="requires footystats_mode='ppg_xg'"):
        feature_columns_for_config(
            BacktestConfig(
                footystats_mode="ppg",
                matchup_context_mode="cartola_matchup_v1",
                feature_augmentation_mode="h004_attack_defense_v1",
            )
        )

    with pytest.raises(ValueError, match="requires matchup_context_mode='cartola_matchup_v1'"):
        feature_columns_for_config(
            BacktestConfig(
                footystats_mode="ppg_xg",
                matchup_context_mode="none",
                feature_augmentation_mode="h004_attack_defense_v1",
            )
        )


def test_h004_feature_formulas_are_finite_and_zero_for_tecnico() -> None:
    season_df = pd.DataFrame(
        [
            {
                "rodada": 1,
                "id_atleta": 1,
                "id_clube": 10,
                "posicao": "ata",
                "status": "Provavel",
                "pontuacao": 6.0,
                "entrou_em_campo": True,
                "preco": 10.0,
                "preco_pre_rodada": 10.0,
                "media": 6.0,
                "num_jogos": 1,
                "variacao": 0.0,
            },
            {
                "rodada": 1,
                "id_atleta": 2,
                "id_clube": 20,
                "posicao": "zag",
                "status": "Provavel",
                "pontuacao": 2.0,
                "entrou_em_campo": True,
                "preco": 8.0,
                "preco_pre_rodada": 8.0,
                "media": 2.0,
                "num_jogos": 1,
                "variacao": 0.0,
            },
            {
                "rodada": 2,
                "id_atleta": 3,
                "id_clube": 10,
                "posicao": "ata",
                "status": "Provavel",
                "pontuacao": 9.0,
                "entrou_em_campo": True,
                "preco": 11.0,
                "preco_pre_rodada": 11.0,
                "media": 9.0,
                "num_jogos": 1,
                "variacao": 0.0,
            },
            {
                "rodada": 2,
                "id_atleta": 4,
                "id_clube": 20,
                "posicao": "tec",
                "status": "Provavel",
                "pontuacao": 4.0,
                "entrou_em_campo": True,
                "preco": 5.0,
                "preco_pre_rodada": 5.0,
                "media": 4.0,
                "num_jogos": 1,
                "variacao": 0.0,
            },
        ]
    )
    fixtures = pd.DataFrame(
        {
            "rodada": [1, 2],
            "id_clube_home": [10, 10],
            "id_clube_away": [20, 20],
        }
    )
    footystats_rows = pd.DataFrame(
        {
            "rodada": [2, 2],
            "id_clube": [10, 20],
            "footystats_team_pre_match_ppg": [2.0, 1.0],
            "footystats_opponent_pre_match_ppg": [1.0, 2.0],
            "footystats_ppg_diff": [1.0, -1.0],
            "footystats_team_pre_match_xg": [1.8, 0.7],
            "footystats_opponent_pre_match_xg": [0.7, 1.8],
            "footystats_xg_diff": [1.1, -1.1],
        }
    )

    frame = build_prediction_frame(
        season_df,
        target_round=2,
        fixtures=fixtures,
        footystats_rows=footystats_rows,
        matchup_context_mode="cartola_matchup_v1",
        feature_augmentation_mode="h004_attack_defense_v1",
    )

    ata = frame.loc[frame["posicao"].eq("ata")].iloc[0]
    tec = frame.loc[frame["posicao"].eq("tec")].iloc[0]
    assert ata["h004_position_softness_delta"] == pytest.approx(
        ata["matchup_opponent_allowed_position_points_roll5"]
        - ata["matchup_opponent_allowed_points_roll5"]
    )
    assert ata["h004_position_mismatch_score"] == pytest.approx(
        (ata["matchup_club_position_points_roll5"] - ata["position_points_prior"])
        + (ata["matchup_opponent_allowed_position_points_roll5"] - ata["position_points_prior"])
    )
    assert ata["h004_home_xg_edge"] == pytest.approx(ata["matchup_is_home"] * ata["footystats_xg_diff"])
    assert ata["h004_role_xg_mismatch"] == pytest.approx(
        ata["footystats_xg_diff"] * ata["h004_position_softness_delta"]
    )
    assert tec[list(H004_ATTACK_DEFENSE_FEATURE_COLUMNS)].tolist() == [0.0, 0.0, 0.0, 0.0]


def test_h004_feature_formulas_reject_missing_dependency_columns() -> None:
    frame = pd.DataFrame(
        {
            "posicao": ["ata"],
            "position_points_prior": [5.0],
            "matchup_club_position_points_roll5": [6.0],
            "matchup_opponent_allowed_position_points_roll5": [7.0],
            "matchup_opponent_allowed_points_roll5": [5.5],
            "matchup_is_home": [1.0],
        }
    )

    with pytest.raises(ValueError, match="footystats_xg_diff"):
        _add_h004_attack_defense_features(frame)


def test_h004_feature_formulas_reject_infinite_context_values() -> None:
    frame = pd.DataFrame(
        {
            "posicao": ["ata"],
            "position_points_prior": [5.0],
            "matchup_club_position_points_roll5": [6.0],
            "matchup_opponent_allowed_position_points_roll5": [float("inf")],
            "matchup_opponent_allowed_points_roll5": [5.5],
            "matchup_is_home": [1.0],
            "footystats_xg_diff": [0.7],
        }
    )

    with pytest.raises(ValueError, match="non-finite numeric context"):
        _add_h004_attack_defense_features(frame)
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
uv run --frozen pytest \
  src/tests/backtesting/test_features.py::test_h004_feature_columns_are_added_only_for_h004_augmentation \
  src/tests/backtesting/test_features.py::test_h004_feature_columns_require_matchup_and_xg_context \
  src/tests/backtesting/test_features.py::test_h004_feature_formulas_are_finite_and_zero_for_tecnico \
  src/tests/backtesting/test_features.py::test_h004_feature_formulas_reject_missing_dependency_columns \
  src/tests/backtesting/test_features.py::test_h004_feature_formulas_reject_infinite_context_values \
  -q
```

Expected: FAIL with missing `H004_ATTACK_DEFENSE_FEATURE_COLUMNS` or unsupported `feature_augmentation_mode`.

- [ ] **Step 3: Implement H004 feature formulas**

In `src/cartola/backtesting/features.py`, add:

```python
import numpy as np

H004_ATTACK_DEFENSE_FEATURE_COLUMNS: list[str] = [
    "h004_position_softness_delta",
    "h004_position_mismatch_score",
    "h004_home_xg_edge",
    "h004_role_xg_mismatch",
]
```

In `feature_columns_for_config`, after matchup context handling, include:

```python
    if config.feature_augmentation_mode == "h004_attack_defense_v1":
        if config.footystats_mode != "ppg_xg":
            raise ValueError("feature_augmentation_mode='h004_attack_defense_v1' requires footystats_mode='ppg_xg'")
        if config.matchup_context_mode != "cartola_matchup_v1":
            raise ValueError(
                "feature_augmentation_mode='h004_attack_defense_v1' "
                "requires matchup_context_mode='cartola_matchup_v1'"
            )

    if config.feature_augmentation_mode == "none":
        return columns
    if config.feature_augmentation_mode == "h004_attack_defense_v1":
        return [*columns, *H004_ATTACK_DEFENSE_FEATURE_COLUMNS]
    raise ValueError(f"Unsupported feature_augmentation_mode: {config.feature_augmentation_mode!r}")
```

Update `build_prediction_frame` signature:

```python
def build_prediction_frame(
    season_df: pd.DataFrame,
    target_round: int,
    fixtures: pd.DataFrame | None = None,
    footystats_rows: pd.DataFrame | None = None,
    matchup_context_mode: str = "none",
    feature_augmentation_mode: str = "none",
) -> pd.DataFrame:
```

Update body:

```python
    frame = _add_prior_features(
        candidates,
        played_history,
        all_history,
        fixtures,
        target_round,
        matchup_context_mode=matchup_context_mode,
    )
    frame = merge_footystats_features(frame, footystats_rows, target_round=target_round)
    if feature_augmentation_mode == "none":
        return frame
    if feature_augmentation_mode == "h004_attack_defense_v1":
        return _add_h004_attack_defense_features(frame)
    raise ValueError(f"Unsupported feature_augmentation_mode: {feature_augmentation_mode!r}")
```

Update `build_training_frame` signature and pass through to `build_prediction_frame`.

Add:

```python
def _add_h004_attack_defense_features(frame: pd.DataFrame) -> pd.DataFrame:
    required = [
        "posicao",
        "position_points_prior",
        "matchup_club_position_points_roll5",
        "matchup_opponent_allowed_position_points_roll5",
        "matchup_opponent_allowed_points_roll5",
        "matchup_is_home",
        "footystats_xg_diff",
    ]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"H004 feature augmentation requires columns: {', '.join(missing)}")

    result = frame.copy()
    numeric_columns = [column for column in required if column != "posicao"]
    for column in numeric_columns:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    invalid = result[numeric_columns].isna().any(axis=1) | ~np.isfinite(result[numeric_columns]).all(axis=1)
    if bool(invalid.any()):
        invalid_columns = [
            column
            for column in numeric_columns
            if bool(result.loc[invalid, column].isna().any())
            or bool((~np.isfinite(result.loc[invalid, column])).any())
        ]
        raise ValueError(f"H004 feature augmentation has non-finite numeric context: {', '.join(invalid_columns)}")

    result["h004_position_softness_delta"] = (
        result["matchup_opponent_allowed_position_points_roll5"]
        - result["matchup_opponent_allowed_points_roll5"]
    )
    result["h004_position_mismatch_score"] = (
        result["matchup_club_position_points_roll5"] - result["position_points_prior"]
    ) + (
        result["matchup_opponent_allowed_position_points_roll5"] - result["position_points_prior"]
    )
    result["h004_home_xg_edge"] = result["matchup_is_home"] * result["footystats_xg_diff"]
    result["h004_role_xg_mismatch"] = (
        result["footystats_xg_diff"] * result["h004_position_softness_delta"]
    )
    tecnico_mask = result["posicao"].astype(str).eq("tec")
    result.loc[tecnico_mask, H004_ATTACK_DEFENSE_FEATURE_COLUMNS] = 0.0
    for column in H004_ATTACK_DEFENSE_FEATURE_COLUMNS:
        result[column] = pd.to_numeric(result[column], errors="raise").astype(float)
    return result
```

- [ ] **Step 4: Run tests**

Run:

```bash
uv run --frozen pytest \
  src/tests/backtesting/test_features.py::test_h004_feature_columns_are_added_only_for_h004_augmentation \
  src/tests/backtesting/test_features.py::test_h004_feature_columns_require_matchup_and_xg_context \
  src/tests/backtesting/test_features.py::test_h004_feature_formulas_are_finite_and_zero_for_tecnico \
  src/tests/backtesting/test_features.py::test_h004_feature_formulas_reject_missing_dependency_columns \
  src/tests/backtesting/test_features.py::test_h004_feature_formulas_reject_infinite_context_values \
  -q
```

Expected: `2 passed`.

- [ ] **Step 5: Run focused static checks**

Run:

```bash
uv run --frozen ruff check src/cartola/backtesting/features.py src/tests/backtesting/test_features.py
uv run --frozen ty check src/cartola/backtesting/features.py src/tests/backtesting/test_features.py
```

Expected: both pass.

- [ ] **Step 6: Commit**

Run:

```bash
git add src/cartola/backtesting/features.py src/tests/backtesting/test_features.py
git commit -m "feat: compute h004 attack-defense features"
```

Expected: commit succeeds.

## Task 3: Thread H004 Mode Through Runner And Metadata

**Files:**
- Modify: `src/cartola/backtesting/runner.py`
- Test: `src/tests/backtesting/test_runner.py`

- [ ] **Step 1: Add failing runner test**

Append to `src/tests/backtesting/test_runner.py`:

```python
def test_run_backtest_records_h004_feature_augmentation_columns(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    season_df = pd.concat([_tiny_round(round_number) for round_number in range(1, 6)], ignore_index=True)
    rows = _tiny_footystats_rows(range(1, 6))
    rows["footystats_team_pre_match_xg"] = 1.2
    rows["footystats_opponent_pre_match_xg"] = 0.8
    rows["footystats_xg_diff"] = 0.4
    fixtures = pd.DataFrame(
        {
            "rodada": list(range(1, 6)),
            "id_clube_home": [18] * 5,
            "id_clube_away": [19] * 5,
        }
    )

    def fake_load_footystats_feature_rows(**kwargs: object) -> FootyStatsPPGLoadResult:
        return FootyStatsPPGLoadResult(
            rows=rows,
            source_path=tmp_path / "data/footystats/brazil-serie-a-matches-2025-to-2025-stats.csv",
            source_sha256="fake-sha",
            diagnostics=FootyStatsJoinDiagnostics(),
            footystats_mode="ppg_xg",
            feature_columns=(*FOOTYSTATS_PPG_FEATURE_COLUMNS, *FOOTYSTATS_XG_FEATURE_COLUMNS),
        )

    monkeypatch.setattr(
        "cartola.backtesting.runner.load_footystats_feature_rows",
        fake_load_footystats_feature_rows,
        raising=False,
    )
    monkeypatch.setattr(
        "cartola.backtesting.runner.load_fixtures",
        lambda **_: fixtures,
        raising=False,
    )

    config = BacktestConfig(
        project_root=tmp_path,
        start_round=5,
        budget=100,
        fixture_mode="exploratory",
        footystats_mode="ppg_xg",
        matchup_context_mode="cartola_matchup_v1",
        feature_augmentation_mode="h004_attack_defense_v1",
    )
    result = run_backtest(config, season_df=season_df)

    assert set(H004_ATTACK_DEFENSE_FEATURE_COLUMNS).issubset(result.player_predictions.columns)
    assert result.metadata.feature_augmentation_mode == "h004_attack_defense_v1"
    assert result.metadata.feature_augmentation_columns == H004_ATTACK_DEFENSE_FEATURE_COLUMNS


def test_run_backtest_control_mode_does_not_emit_h004_columns(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    season_df = pd.concat([_tiny_round(round_number) for round_number in range(1, 6)], ignore_index=True)
    rows = _tiny_footystats_rows(range(1, 6))
    rows["footystats_team_pre_match_xg"] = 1.2
    rows["footystats_opponent_pre_match_xg"] = 0.8
    rows["footystats_xg_diff"] = 0.4
    fixtures = pd.DataFrame(
        {
            "rodada": list(range(1, 6)),
            "id_clube_home": [18] * 5,
            "id_clube_away": [19] * 5,
        }
    )

    def fake_load_footystats_feature_rows(**kwargs: object) -> FootyStatsPPGLoadResult:
        return FootyStatsPPGLoadResult(
            rows=rows,
            source_path=tmp_path / "data/footystats/brazil-serie-a-matches-2025-to-2025-stats.csv",
            source_sha256="fake-sha",
            diagnostics=FootyStatsJoinDiagnostics(),
            footystats_mode="ppg_xg",
            feature_columns=(*FOOTYSTATS_PPG_FEATURE_COLUMNS, *FOOTYSTATS_XG_FEATURE_COLUMNS),
        )

    monkeypatch.setattr(
        "cartola.backtesting.runner.load_footystats_feature_rows",
        fake_load_footystats_feature_rows,
        raising=False,
    )
    monkeypatch.setattr(
        "cartola.backtesting.runner.load_fixtures",
        lambda **_: fixtures,
        raising=False,
    )

    config = BacktestConfig(
        project_root=tmp_path,
        start_round=5,
        budget=100,
        fixture_mode="exploratory",
        footystats_mode="ppg_xg",
        matchup_context_mode="cartola_matchup_v1",
        feature_augmentation_mode="none",
    )
    result = run_backtest(config, season_df=season_df)

    assert set(H004_ATTACK_DEFENSE_FEATURE_COLUMNS).isdisjoint(result.player_predictions.columns)
    assert result.metadata.feature_augmentation_mode == "none"
    assert result.metadata.feature_augmentation_columns == []
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
uv run --frozen pytest \
  src/tests/backtesting/test_runner.py::test_run_backtest_records_h004_feature_augmentation_columns \
  src/tests/backtesting/test_runner.py::test_run_backtest_control_mode_does_not_emit_h004_columns \
  -q
```

Expected: FAIL with missing metadata fields or unsupported mode.

- [ ] **Step 3: Thread config mode through runner**

In `src/cartola/backtesting/runner.py`:

- Import `H004_ATTACK_DEFENSE_FEATURE_COLUMNS` from `features.py`.
- Pass `feature_augmentation_mode=config.feature_augmentation_mode` into every `build_training_frame` and `build_prediction_frame` call.
- Add metadata fields to the run metadata dataclass if it exists:

```python
    feature_augmentation_mode: str
    feature_augmentation_columns: list[str]
```

- Populate metadata:

```python
feature_augmentation_columns = (
    H004_ATTACK_DEFENSE_FEATURE_COLUMNS
    if config.feature_augmentation_mode == "h004_attack_defense_v1"
    else []
)
```

- Ensure `run_metadata.json` writes both fields.
- Ensure `player_predictions.csv` for `feature_augmentation_mode="none"` does not contain any `h004_*` columns.

In `src/cartola/backtesting/experiment_runner.py`, add `feature_augmentation_mode` to child-level experiment metadata, experiment index rows, and MLflow/file tracker params:

```python
"feature_augmentation_mode": spec.backtest_config.feature_augmentation_mode,
```

The parent `experiment_metadata.json` child records must expose this field outside the nested backtest metadata so the H004 decision module can validate control/challenger identity without opening every child artifact first.

- [ ] **Step 4: Run test**

Run:

```bash
uv run --frozen pytest \
  src/tests/backtesting/test_runner.py::test_run_backtest_records_h004_feature_augmentation_columns \
  src/tests/backtesting/test_runner.py::test_run_backtest_control_mode_does_not_emit_h004_columns \
  -q
```

Expected: `1 passed`.

- [ ] **Step 5: Run focused static checks**

Run:

```bash
uv run --frozen ruff check src/cartola/backtesting/runner.py src/cartola/backtesting/experiment_runner.py src/tests/backtesting/test_runner.py
uv run --frozen ty check src/cartola/backtesting/runner.py src/cartola/backtesting/experiment_runner.py src/tests/backtesting/test_runner.py
```

Expected: both pass.

- [ ] **Step 6: Commit**

Run:

```bash
git add src/cartola/backtesting/runner.py src/cartola/backtesting/experiment_runner.py src/tests/backtesting/test_runner.py
git commit -m "feat: thread h004 feature mode through runner"
```

Expected: commit succeeds.

## Task 4: Verify Experiment CLI Accepts H004 Group

**Files:**
- Verify: `scripts/run_model_experiments.py`
- Test: `src/tests/backtesting/test_run_model_experiments_cli.py`
- Test: `src/tests/backtesting/test_experiment_config.py`

- [ ] **Step 1: Run existing group-validation tests**

Run:

```bash
uv run --frozen pytest src/tests/backtesting/test_run_model_experiments_cli.py src/tests/backtesting/test_experiment_config.py -q
```

Expected: pass. If this fails because `parse_args` explicitly enumerates groups, continue with Step 2.

- [ ] **Step 2: If CLI rejects the new group, add a focused test**

If needed, add to `src/tests/backtesting/test_run_model_experiments_cli.py`:

```python
def test_parse_args_accepts_h004_group() -> None:
    args = cli.parse_args(
        [
            "--group",
            "h004-attack-defense-mismatch",
            "--seasons",
            "2021,2022",
            "--start-round",
            "5",
            "--budget",
            "100",
            "--current-year",
            "2026",
        ]
    )

    assert args.group == "h004-attack-defense-mismatch"
```

- [ ] **Step 3: Implement the CLI update only when Step 1 proves it is required**

If `scripts/run_model_experiments.py` uses explicit group choices, add `"h004-attack-defense-mismatch"` to that choice list. If it does not, do not edit the script.

- [ ] **Step 4: Run focused tests**

Run:

```bash
uv run --frozen pytest src/tests/backtesting/test_run_model_experiments_cli.py src/tests/backtesting/test_experiment_config.py -q
```

Expected: pass.

- [ ] **Step 5: Commit when files changed**

If files changed, run:

```bash
git add scripts/run_model_experiments.py src/tests/backtesting/test_run_model_experiments_cli.py src/tests/backtesting/test_experiment_config.py
git commit -m "feat: expose h004 experiment group"
```

Expected: commit succeeds only if files changed.

## Task 5: Focused Verification

**Files:** verify only.

- [ ] **Step 1: Run focused H004 feature tests**

Run:

```bash
uv run --frozen pytest \
  src/tests/backtesting/test_features.py \
  src/tests/backtesting/test_runner.py::test_run_backtest_records_h004_feature_augmentation_columns \
  src/tests/backtesting/test_experiment_config.py \
  src/tests/backtesting/test_run_model_experiments_cli.py \
  -q
```

Expected: pass.

- [ ] **Step 2: Run static checks on touched files**

Run:

```bash
uv run --frozen ruff check src/cartola/backtesting/config.py src/cartola/backtesting/experiment_config.py src/cartola/backtesting/features.py src/cartola/backtesting/runner.py src/tests/backtesting/test_experiment_config.py src/tests/backtesting/test_features.py src/tests/backtesting/test_runner.py src/tests/backtesting/test_run_model_experiments_cli.py scripts/run_model_experiments.py
uv run --frozen ty check src/cartola/backtesting/config.py src/cartola/backtesting/experiment_config.py src/cartola/backtesting/features.py src/cartola/backtesting/runner.py src/tests/backtesting/test_experiment_config.py src/tests/backtesting/test_features.py src/tests/backtesting/test_runner.py src/tests/backtesting/test_run_model_experiments_cli.py scripts/run_model_experiments.py
```

Expected: both pass.

- [ ] **Step 3: Commit fixes when verification required code changes**

If verification required fixes, run:

```bash
git add src/cartola/backtesting/config.py src/cartola/backtesting/experiment_config.py src/cartola/backtesting/features.py src/cartola/backtesting/runner.py src/tests/backtesting/test_experiment_config.py src/tests/backtesting/test_features.py src/tests/backtesting/test_runner.py src/tests/backtesting/test_run_model_experiments_cli.py scripts/run_model_experiments.py
git commit -m "fix: stabilize h004 feature pack"
```

Expected: commit succeeds only if fixes were made.

## Task 6: Add Deterministic H004 Phase 2 Decision Artifact

**Files:**
- Create: `src/cartola/backtesting/h004_feature_decision.py`
- Create: `scripts/run_h004_feature_decision.py`
- Test: `src/tests/backtesting/test_h004_feature_decision.py`

- [ ] **Step 1: Add failing tests for decision status**

Create `src/tests/backtesting/test_h004_feature_decision.py`:

```python
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from cartola.backtesting.h004_feature_decision import (
    H004FeatureDecisionError,
    build_h004_phase2_decision,
    write_h004_phase2_decision,
)


CONTROL_CHILD_2021 = "season=2021/model=xgboost_depth2_slow/feature_pack=ppg_xg_matchup"
CHALLENGER_CHILD_2021 = "season=2021/model=xgboost_depth2_slow/feature_pack=ppg_xg_matchup_h004"


def test_h004_decision_is_diagnostic_only_when_fixture_identity_unverified(tmp_path: Path) -> None:
    phase1 = _write_phase1_decision(tmp_path, status="passes", passed_families=["C"])
    experiment = _write_experiment(
        tmp_path,
        fixture_hashes_control={},
        fixture_hashes_challenger={},
        season_deltas={2021: 25.0, 2022: 22.0, 2023: 20.0, 2024: 15.0, 2025: 12.0},
    )

    decision = build_h004_phase2_decision(experiment_path=experiment, phase1_decision_path=phase1)

    assert decision["final_status"] == "diagnostic_only"
    assert decision["fixture_identity_status"] == "unverified"
    assert decision["gate_results"]["aggregate_delta_pass"] is True


def test_h004_decision_is_candidate_research_when_all_gates_and_fixture_identity_pass(tmp_path: Path) -> None:
    phase1 = _write_phase1_decision(tmp_path, status="passes", passed_families=["C"])
    hashes = {"data/01_raw/fixtures/2021/partidas-1.csv": "same-sha"}
    experiment = _write_experiment(
        tmp_path,
        fixture_hashes_control=hashes,
        fixture_hashes_challenger=hashes,
        season_deltas={2021: 25.0, 2022: 22.0, 2023: 20.0, 2024: 15.0, 2025: 12.0},
    )

    decision = build_h004_phase2_decision(experiment_path=experiment, phase1_decision_path=phase1)

    assert decision["final_status"] == "candidate_research"
    assert decision["fixture_identity_status"] == "verified"


def test_h004_decision_rejects_failed_metric_gate(tmp_path: Path) -> None:
    phase1 = _write_phase1_decision(tmp_path, status="passes", passed_families=["C"])
    hashes = {"data/01_raw/fixtures/2021/partidas-1.csv": "same-sha"}
    experiment = _write_experiment(
        tmp_path,
        fixture_hashes_control=hashes,
        fixture_hashes_challenger=hashes,
        season_deltas={2021: 60.0, 2022: 20.0, 2023: -25.0, 2024: 18.0, 2025: 12.0},
    )

    decision = build_h004_phase2_decision(experiment_path=experiment, phase1_decision_path=phase1)

    assert decision["final_status"] == "rejected"
    assert decision["gate_results"]["worst_season_delta_pass"] is False
    assert "worst_season_delta_pass" in decision["reasons"]


def test_h004_decision_invalidates_candidate_signature_mismatch(tmp_path: Path) -> None:
    phase1 = _write_phase1_decision(tmp_path, status="passes", passed_families=["C"])
    hashes = {"data/01_raw/fixtures/2021/partidas-1.csv": "same-sha"}
    experiment = _write_experiment(
        tmp_path,
        fixture_hashes_control=hashes,
        fixture_hashes_challenger=hashes,
        season_deltas={2021: 25.0, 2022: 22.0, 2023: 20.0, 2024: 15.0, 2025: 12.0},
    )
    metadata = json.loads((experiment / "experiment_metadata.json").read_text(encoding="utf-8"))
    metadata["candidate_pool_signatures"][CHALLENGER_CHILD_2021]["5"] = "different"
    (experiment / "experiment_metadata.json").write_text(json.dumps(metadata), encoding="utf-8")

    decision = build_h004_phase2_decision(experiment_path=experiment, phase1_decision_path=phase1)

    assert decision["final_status"] == "invalid"
    assert decision["candidate_signature_status"] == "mismatch"


def test_write_h004_phase2_decision_writes_json(tmp_path: Path) -> None:
    phase1 = _write_phase1_decision(tmp_path, status="passes", passed_families=["C"])
    experiment = _write_experiment(
        tmp_path,
        fixture_hashes_control={},
        fixture_hashes_challenger={},
        season_deltas={2021: 25.0, 2022: 22.0, 2023: 20.0, 2024: 15.0, 2025: 12.0},
    )

    output = write_h004_phase2_decision(experiment_path=experiment, phase1_decision_path=phase1)

    assert output == experiment / "h004_phase2_decision.json"
    assert json.loads(output.read_text(encoding="utf-8"))["hypothesis_id"] == "H004"


def test_h004_decision_requires_phase1_passed_family_c(tmp_path: Path) -> None:
    phase1 = _write_phase1_decision(tmp_path, status="passes", passed_families=["A"])
    experiment = _write_experiment(
        tmp_path,
        fixture_hashes_control={},
        fixture_hashes_challenger={},
        season_deltas={2021: 25.0, 2022: 22.0, 2023: 20.0, 2024: 15.0, 2025: 12.0},
    )

    decision = build_h004_phase2_decision(experiment_path=experiment, phase1_decision_path=phase1)

    assert decision["final_status"] == "invalid"
    assert decision["phase1_precondition_status"] == "failed"


def test_h004_decision_rejects_missing_artifacts(tmp_path: Path) -> None:
    phase1 = _write_phase1_decision(tmp_path, status="passes", passed_families=["C"])

    with pytest.raises(H004FeatureDecisionError, match="ranked_summary.csv"):
        build_h004_phase2_decision(experiment_path=tmp_path / "missing", phase1_decision_path=phase1)


def _write_phase1_decision(tmp_path: Path, *, status: str, passed_families: list[str]) -> Path:
    path = tmp_path / "h004_diagnostic_decision.json"
    path.write_text(
        json.dumps({"diagnostic_status": status, "passed_families": passed_families}),
        encoding="utf-8",
    )
    return path


def _write_experiment(
    tmp_path: Path,
    *,
    fixture_hashes_control: dict[str, str],
    fixture_hashes_challenger: dict[str, str],
    season_deltas: dict[int, float],
) -> Path:
    experiment = tmp_path / "experiment"
    experiment.mkdir()
    control_rows = []
    challenger_rows = []
    metric_rows = []
    child_runs = []
    signatures = {}
    for season, delta in season_deltas.items():
        control_child = f"season={season}/model=xgboost_depth2_slow/feature_pack=ppg_xg_matchup"
        challenger_child = f"season={season}/model=xgboost_depth2_slow/feature_pack=ppg_xg_matchup_h004"
        signatures[control_child] = {"5": "same", "6": "same"}
        signatures[challenger_child] = {"5": "same", "6": "same"}
        child_runs.extend(
            [
                _child_record(
                    child_id=control_child,
                    season=season,
                    feature_pack="ppg_xg_matchup",
                    feature_augmentation_mode="none",
                    fixture_hashes=fixture_hashes_control,
                ),
                _child_record(
                    child_id=challenger_child,
                    season=season,
                    feature_pack="ppg_xg_matchup_h004",
                    feature_augmentation_mode="h004_attack_defense_v1",
                    fixture_hashes=fixture_hashes_challenger,
                ),
            ]
        )
        control_rows.append(_season_row(control_child, season, "ppg_xg_matchup", 1000.0, 120.0, 100.0, 10.0, 1))
        challenger_rows.append(
            _season_row(challenger_child, season, "ppg_xg_matchup_h004", 1000.0 + delta, 118.0, 99.0, 11.0, 1)
        )
        metric_rows.extend(
            [
                _metric_row(control_child, season, "ppg_xg_matchup", "top50_candidates", 0.10, None, 800),
                _metric_row(challenger_child, season, "ppg_xg_matchup_h004", "top50_candidates", 0.09, None, 800),
                _metric_row(control_child, season, "ppg_xg_matchup", "selected_players", 0.05, 0.90, 408),
                _metric_row(challenger_child, season, "ppg_xg_matchup_h004", "selected_players", 0.05, 1.00, 408),
            ]
        )
    pd.DataFrame(
        [
            {
                "rank": 1,
                "model_id": "xgboost_depth2_slow",
                "feature_pack": "ppg_xg_matchup_h004",
                "fixture_mode": "exploratory",
                "budget_policy": "moving",
                "seasons_evaluated": 5,
                "total_rounds": 170,
                "total_actual_points": sum(row["total_actual_points"] for row in challenger_rows),
            }
        ]
    ).to_csv(experiment / "ranked_summary.csv", index=False)
    pd.DataFrame([*control_rows, *challenger_rows]).to_csv(experiment / "per_season_summary.csv", index=False)
    pd.DataFrame(metric_rows).to_csv(experiment / "prediction_metrics.csv", index=False)
    (experiment / "comparability_report.json").write_text(json.dumps({"status": "ok"}), encoding="utf-8")
    (experiment / "experiment_metadata.json").write_text(
        json.dumps(
            {
                "status": "ok",
                "budget_policy": "moving",
                "group": "h004-attack-defense-mismatch",
                "seasons": [2021, 2022, 2023, 2024, 2025],
                "child_runs": child_runs,
                "candidate_pool_signatures": signatures,
            }
        ),
        encoding="utf-8",
    )
    return experiment


def _child_record(
    *,
    child_id: str,
    season: int,
    feature_pack: str,
    feature_augmentation_mode: str,
    fixture_hashes: dict[str, str],
) -> dict[str, object]:
    return {
        "child_id": child_id,
        "season": season,
        "model_id": "xgboost_depth2_slow",
        "feature_pack": feature_pack,
        "fixture_mode": "exploratory",
        "feature_augmentation_mode": feature_augmentation_mode,
        "metadata": {
            "budget_policy": "moving",
            "fixture_mode": "exploratory",
            "footystats_mode": "ppg_xg",
            "matchup_context_mode": "cartola_matchup_v1",
            "scoring_contract_version": "cartola_standard_2026_v1",
            "fixture_manifest_sha256": {},
            "fixture_source_sha256": fixture_hashes,
        },
    }


def _season_row(
    child_id: str,
    season: int,
    feature_pack: str,
    total_actual_points: float,
    final_budget: float,
    min_budget: float,
    max_budget_drawdown: float,
    budget_constrained_rounds: int,
) -> dict[str, object]:
    return {
        "child_id": child_id,
        "season": season,
        "model_id": "xgboost_depth2_slow",
        "feature_pack": feature_pack,
        "fixture_mode": "exploratory",
        "budget_policy": "moving",
        "strategy": "xgboost_depth2_slow",
        "rounds": 34,
        "total_actual_points": total_actual_points,
        "final_budget": final_budget,
        "min_budget": min_budget,
        "max_budget_drawdown": max_budget_drawdown,
        "budget_constrained_rounds": budget_constrained_rounds,
    }


def _metric_row(
    child_id: str,
    season: int,
    feature_pack: str,
    metric_scope: str,
    spearman: float | None,
    calibration_slope: float | None,
    observed_count: int,
) -> dict[str, object]:
    return {
        "child_id": child_id,
        "season": season,
        "model_id": "xgboost_depth2_slow",
        "feature_pack": feature_pack,
        "fixture_mode": "exploratory",
        "budget_policy": "moving",
        "metric_scope": metric_scope,
        "observed_count": observed_count,
        "spearman": spearman,
        "calibration_slope": calibration_slope,
    }
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
uv run --frozen pytest src/tests/backtesting/test_h004_feature_decision.py -q
```

Expected: FAIL with missing module or missing functions.

- [ ] **Step 3: Implement the decision module**

Create `src/cartola/backtesting/h004_feature_decision.py`:

```python
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping

import pandas as pd


class H004FeatureDecisionError(ValueError):
    """Raised when the H004 Phase 2 decision cannot be built from artifacts."""


CONTROL_MODEL_ID = "xgboost_depth2_slow"
CONTROL_FEATURE_PACK = "ppg_xg_matchup"
CHALLENGER_FEATURE_PACK = "ppg_xg_matchup_h004"
REQUIRED_SEASONS = (2021, 2022, 2023, 2024, 2025)


def write_h004_phase2_decision(*, experiment_path: Path, phase1_decision_path: Path) -> Path:
    decision = build_h004_phase2_decision(
        experiment_path=experiment_path,
        phase1_decision_path=phase1_decision_path,
    )
    output_path = experiment_path / "h004_phase2_decision.json"
    output_path.write_text(json.dumps(decision, indent=2, sort_keys=True), encoding="utf-8")
    return output_path


def build_h004_phase2_decision(*, experiment_path: Path, phase1_decision_path: Path) -> dict[str, Any]:
    experiment_path = Path(experiment_path)
    phase1_decision_path = Path(phase1_decision_path)
    artifacts = _load_required_artifacts(experiment_path)
    phase1 = _read_json(phase1_decision_path, label="Phase 1 decision")
    phase1_ok = phase1.get("diagnostic_status") == "passes" and "C" in set(phase1.get("passed_families", []))
    validation_errors = _validation_errors(artifacts=artifacts, phase1_ok=phase1_ok)
    candidate_signature_status = _candidate_signature_status(artifacts["metadata"])
    fixture_identity_status = _fixture_identity_status(artifacts["metadata"])
    if candidate_signature_status != "ok":
        validation_errors.append(f"candidate_signature_status={candidate_signature_status}")
    gate_payload = _build_gate_payload(artifacts)
    gate_results = gate_payload["gate_results"]
    failed_gates = [name for name, passed in gate_results.items() if not bool(passed)]

    if validation_errors:
        final_status = "invalid"
        reasons = validation_errors
    elif failed_gates:
        final_status = "rejected"
        reasons = failed_gates
    elif fixture_identity_status == "verified":
        final_status = "candidate_research"
        reasons = []
    else:
        final_status = "diagnostic_only"
        reasons = [f"fixture_identity_status={fixture_identity_status}"]

    return {
        "hypothesis_id": "H004",
        "phase": "feature_pack_phase2",
        "control": {"model_id": CONTROL_MODEL_ID, "feature_pack": CONTROL_FEATURE_PACK},
        "challenger": {"model_id": CONTROL_MODEL_ID, "feature_pack": CHALLENGER_FEATURE_PACK},
        "phase1_precondition_status": "passes" if phase1_ok else "failed",
        "fixture_identity_status": fixture_identity_status,
        "candidate_signature_status": candidate_signature_status,
        "final_status": final_status,
        "gate_results": gate_results,
        "season_deltas": gate_payload["season_deltas"],
        "metric_deltas": gate_payload["metric_deltas"],
        "budget_deltas": gate_payload["budget_deltas"],
        "reasons": reasons,
    }
```

Continue the module with small private helpers:

```python
def _load_required_artifacts(experiment_path: Path) -> dict[str, Any]:
    required_files = {
        "ranked": "ranked_summary.csv",
        "season": "per_season_summary.csv",
        "metrics": "prediction_metrics.csv",
        "comparability": "comparability_report.json",
        "metadata": "experiment_metadata.json",
    }
    missing = [filename for filename in required_files.values() if not (experiment_path / filename).exists()]
    if missing:
        raise H004FeatureDecisionError(f"Missing required H004 Phase 2 artifacts: {', '.join(missing)}")
    return {
        "ranked": pd.read_csv(experiment_path / required_files["ranked"]),
        "season": pd.read_csv(experiment_path / required_files["season"]),
        "metrics": pd.read_csv(experiment_path / required_files["metrics"]),
        "comparability": _read_json(experiment_path / required_files["comparability"], label="comparability report"),
        "metadata": _read_json(experiment_path / required_files["metadata"], label="experiment metadata"),
    }


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise H004FeatureDecisionError(f"Missing {label}: {path}") from exc
    except json.JSONDecodeError as exc:
        raise H004FeatureDecisionError(f"Invalid JSON in {label}: {path}") from exc
    if not isinstance(data, dict):
        raise H004FeatureDecisionError(f"{label} must be a JSON object: {path}")
    return data


def _validation_errors(*, artifacts: Mapping[str, Any], phase1_ok: bool) -> list[str]:
    errors: list[str] = []
    if not phase1_ok:
        errors.append("phase1_precondition_failed")
    metadata = artifacts["metadata"]
    if metadata.get("budget_policy") != "moving":
        errors.append("experiment budget_policy must be moving")
    if tuple(metadata.get("seasons", [])) != REQUIRED_SEASONS:
        errors.append("experiment seasons must be 2021,2022,2023,2024,2025")
    if artifacts["comparability"].get("status") != "ok":
        errors.append("comparability_report status must be ok")
    required_season_columns = {
        "child_id",
        "season",
        "model_id",
        "feature_pack",
        "fixture_mode",
        "budget_policy",
        "total_actual_points",
        "final_budget",
        "min_budget",
        "max_budget_drawdown",
        "budget_constrained_rounds",
    }
    required_metric_columns = {
        "child_id",
        "season",
        "model_id",
        "feature_pack",
        "fixture_mode",
        "budget_policy",
        "metric_scope",
        "observed_count",
        "spearman",
        "calibration_slope",
    }
    errors.extend(_missing_column_errors(artifacts["season"], required_season_columns, "per_season_summary.csv"))
    errors.extend(_missing_column_errors(artifacts["metrics"], required_metric_columns, "prediction_metrics.csv"))
    child_errors = _child_context_errors(metadata)
    errors.extend(child_errors)
    return errors


def _missing_column_errors(frame: pd.DataFrame, required: set[str], label: str) -> list[str]:
    missing = sorted(required.difference(frame.columns))
    return [f"{label} missing columns: {', '.join(missing)}"] if missing else []
```

Then implement the gate helpers with the frozen formulas:

```python
def _build_gate_payload(artifacts: Mapping[str, Any]) -> dict[str, Any]:
    season = artifacts["season"].copy()
    metrics = artifacts["metrics"].copy()
    control = _primary_season_rows(season, feature_pack=CONTROL_FEATURE_PACK)
    challenger = _primary_season_rows(season, feature_pack=CHALLENGER_FEATURE_PACK)
    merged = control.merge(challenger, on="season", suffixes=("_control", "_challenger"), validate="one_to_one")
    season_deltas = []
    budget_deltas = []
    for row in merged.to_dict(orient="records"):
        delta = _finite(row["total_actual_points_challenger"]) - _finite(row["total_actual_points_control"])
        season_deltas.append({"season": int(row["season"]), "actual_points_delta": delta})
        budget_deltas.append(
            {
                "season": int(row["season"]),
                "final_budget_delta": _finite(row["final_budget_challenger"]) - _finite(row["final_budget_control"]),
                "min_budget_delta": _finite(row["min_budget_challenger"]) - _finite(row["min_budget_control"]),
                "max_budget_drawdown_delta": _finite(row["max_budget_drawdown_challenger"])
                - _finite(row["max_budget_drawdown_control"]),
                "budget_constrained_rounds_delta": int(row["budget_constrained_rounds_challenger"])
                - int(row["budget_constrained_rounds_control"]),
            }
        )
    metric_deltas = _metric_deltas(metrics)
    aggregate_delta = sum(item["actual_points_delta"] for item in season_deltas)
    positive_deltas = sorted(
        [item["actual_points_delta"] for item in season_deltas if item["actual_points_delta"] > 0.0],
        reverse=True,
    )
    positive_sum = sum(positive_deltas)
    concentration = math.inf if aggregate_delta <= 0.0 or positive_sum <= 0.0 else sum(positive_deltas[:2]) / positive_sum
    top50_regressions = sum(1 for row in metric_deltas if row["metric_scope"] == "top50_candidates" and row["delta"] < -0.02)
    challenger_calibration_rows = _metric_rows(metrics, feature_pack=CHALLENGER_FEATURE_PACK, metric_scope="selected_players")
    calibration_pass = all(
        0.50 <= _finite(row["calibration_slope"]) <= 1.50 and int(row["observed_count"]) >= 120
        for row in challenger_calibration_rows.to_dict(orient="records")
    )
    gate_results = {
        "aggregate_delta_pass": aggregate_delta >= 85.0,
        "improved_seasons_pass": sum(1 for item in season_deltas if item["actual_points_delta"] > 0.0) >= 4,
        "worst_season_delta_pass": min(item["actual_points_delta"] for item in season_deltas) >= -20.0,
        "recent_season_delta_pass": next(item["actual_points_delta"] for item in season_deltas if item["season"] == 2025)
        >= -10.0,
        "final_budget_pass": min(item["final_budget_delta"] for item in budget_deltas) >= -15.0,
        "min_budget_pass": min(item["min_budget_delta"] for item in budget_deltas) >= -15.0,
        "max_drawdown_pass": max(item["max_budget_drawdown_delta"] for item in budget_deltas) <= 15.0,
        "budget_constrained_rounds_pass": sum(item["budget_constrained_rounds_delta"] for item in budget_deltas) <= 2,
        "top50_spearman_pass": top50_regressions <= 1,
        "selected_calibration_pass": calibration_pass,
        "concentration_pass": concentration < 0.70,
    }
    return {
        "gate_results": gate_results,
        "season_deltas": season_deltas,
        "metric_deltas": metric_deltas,
        "budget_deltas": budget_deltas,
    }
```

Finish helpers for source identity, signatures, and numeric safety:

```python
def _primary_season_rows(frame: pd.DataFrame, *, feature_pack: str) -> pd.DataFrame:
    rows = frame[
        frame["model_id"].eq(CONTROL_MODEL_ID)
        & frame["feature_pack"].eq(feature_pack)
        & frame["fixture_mode"].eq("exploratory")
        & frame["budget_policy"].eq("moving")
    ].copy()
    if set(rows["season"].astype(int)) != set(REQUIRED_SEASONS):
        raise H004FeatureDecisionError(f"Missing primary rows for feature_pack={feature_pack}")
    return rows


def _metric_rows(frame: pd.DataFrame, *, feature_pack: str, metric_scope: str) -> pd.DataFrame:
    rows = frame[
        frame["model_id"].eq(CONTROL_MODEL_ID)
        & frame["feature_pack"].eq(feature_pack)
        & frame["fixture_mode"].eq("exploratory")
        & frame["budget_policy"].eq("moving")
        & frame["metric_scope"].eq(metric_scope)
    ].copy()
    if set(rows["season"].astype(int)) != set(REQUIRED_SEASONS):
        raise H004FeatureDecisionError(f"Missing {metric_scope} rows for feature_pack={feature_pack}")
    return rows


def _metric_deltas(metrics: pd.DataFrame) -> list[dict[str, Any]]:
    rows = []
    for metric_scope, metric_name in (("top50_candidates", "spearman"), ("selected_players", "calibration_slope")):
        control = _metric_rows(metrics, feature_pack=CONTROL_FEATURE_PACK, metric_scope=metric_scope)
        challenger = _metric_rows(metrics, feature_pack=CHALLENGER_FEATURE_PACK, metric_scope=metric_scope)
        merged = control.merge(challenger, on="season", suffixes=("_control", "_challenger"), validate="one_to_one")
        for row in merged.to_dict(orient="records"):
            rows.append(
                {
                    "season": int(row["season"]),
                    "metric_scope": metric_scope,
                    "metric": metric_name,
                    "control": _finite(row[f"{metric_name}_control"]),
                    "challenger": _finite(row[f"{metric_name}_challenger"]),
                    "delta": _finite(row[f"{metric_name}_challenger"]) - _finite(row[f"{metric_name}_control"]),
                }
            )
    return rows


def _candidate_signature_status(metadata: Mapping[str, Any]) -> str:
    signatures = metadata.get("candidate_pool_signatures")
    if not isinstance(signatures, dict):
        return "missing"
    for season in REQUIRED_SEASONS:
        control_id = f"season={season}/model={CONTROL_MODEL_ID}/feature_pack={CONTROL_FEATURE_PACK}"
        challenger_id = f"season={season}/model={CONTROL_MODEL_ID}/feature_pack={CHALLENGER_FEATURE_PACK}"
        control = signatures.get(control_id)
        challenger = signatures.get(challenger_id)
        if control is None or challenger is None:
            return "missing"
        if control != challenger:
            return "mismatch"
    return "ok"


def _fixture_identity_status(metadata: Mapping[str, Any]) -> str:
    child_by_id = {
        str(child.get("child_id")): child
        for child in metadata.get("child_runs", [])
        if isinstance(child, dict) and child.get("child_id") is not None
    }
    saw_source_hashes = False
    for season in REQUIRED_SEASONS:
        control = child_by_id.get(f"season={season}/model={CONTROL_MODEL_ID}/feature_pack={CONTROL_FEATURE_PACK}")
        challenger = child_by_id.get(
            f"season={season}/model={CONTROL_MODEL_ID}/feature_pack={CHALLENGER_FEATURE_PACK}"
        )
        if control is None or challenger is None:
            return "missing"
        control_hashes = _fixture_hashes(control)
        challenger_hashes = _fixture_hashes(challenger)
        if control_hashes is None or challenger_hashes is None:
            return "unverified"
        saw_source_hashes = True
        if control_hashes != challenger_hashes:
            return "mismatch"
    return "verified" if saw_source_hashes else "unverified"


def _fixture_hashes(child: Mapping[str, Any]) -> dict[str, str] | None:
    metadata = child.get("metadata")
    if not isinstance(metadata, dict):
        return None
    hashes = metadata.get("fixture_source_sha256") or metadata.get("fixture_manifest_sha256")
    return hashes if isinstance(hashes, dict) and hashes else None


def _child_context_errors(metadata: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    child_runs = metadata.get("child_runs")
    if not isinstance(child_runs, list):
        return ["experiment_metadata.json child_runs must be a list"]
    for child in child_runs:
        if not isinstance(child, dict):
            errors.append("child_runs entry must be an object")
            continue
        if child.get("model_id") != CONTROL_MODEL_ID:
            continue
        if child.get("feature_pack") not in {CONTROL_FEATURE_PACK, CHALLENGER_FEATURE_PACK}:
            continue
        child_metadata = child.get("metadata")
        if not isinstance(child_metadata, dict):
            errors.append(f"{child.get('child_id')}: missing metadata")
            continue
        expected_mode = "h004_attack_defense_v1" if child.get("feature_pack") == CHALLENGER_FEATURE_PACK else "none"
        observed_mode = child.get("feature_augmentation_mode", child_metadata.get("feature_augmentation_mode"))
        checks = {
            "budget_policy": "moving",
            "fixture_mode": "exploratory",
            "footystats_mode": "ppg_xg",
            "matchup_context_mode": "cartola_matchup_v1",
            "scoring_contract_version": "cartola_standard_2026_v1",
        }
        for key, expected in checks.items():
            if child_metadata.get(key) != expected:
                errors.append(f"{child.get('child_id')}: {key} must be {expected}")
        if observed_mode != expected_mode:
            errors.append(f"{child.get('child_id')}: feature_augmentation_mode must be {expected_mode}")
    return errors


def _finite(value: object) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise H004FeatureDecisionError(f"Expected finite numeric value, got {value!r}")
    return number
```

- [ ] **Step 4: Add the CLI wrapper**

Create `scripts/run_h004_feature_decision.py`:

```python
from __future__ import annotations

import argparse
from pathlib import Path

from rich.console import Console

from cartola.backtesting.h004_feature_decision import write_h004_phase2_decision


DEFAULT_PHASE1_DECISION = Path(
    "data/08_reporting/hypotheses/"
    "h004_residual_diagnostic_started_at=20260508T182202655139Z/"
    "h004_diagnostic_decision.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the deterministic H004 Phase 2 decision artifact.")
    parser.add_argument("--experiment-path", required=True, type=Path)
    parser.add_argument("--phase1-decision-path", type=Path, default=DEFAULT_PHASE1_DECISION)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = write_h004_phase2_decision(
        experiment_path=args.experiment_path,
        phase1_decision_path=args.phase1_decision_path,
    )
    console = Console()
    console.print(f"H004 Phase 2 decision written: {output_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run focused tests**

Run:

```bash
uv run --frozen pytest src/tests/backtesting/test_h004_feature_decision.py -q
```

Expected: all H004 decision tests pass.

- [ ] **Step 6: Run static checks**

Run:

```bash
uv run --frozen ruff check src/cartola/backtesting/h004_feature_decision.py src/tests/backtesting/test_h004_feature_decision.py scripts/run_h004_feature_decision.py
uv run --frozen ty check src/cartola/backtesting/h004_feature_decision.py src/tests/backtesting/test_h004_feature_decision.py scripts/run_h004_feature_decision.py
```

Expected: both pass.

- [ ] **Step 7: Commit**

Run:

```bash
git add src/cartola/backtesting/h004_feature_decision.py src/tests/backtesting/test_h004_feature_decision.py scripts/run_h004_feature_decision.py
git commit -m "feat: add h004 phase2 decision artifact"
```

Expected: commit succeeds.

## Task 7: Run Real H004 Phase 2 Experiment

**Files:** generated experiment output under `data/08_reporting/experiments/model_feature/`.

- [ ] **Step 1: Verify the Phase 1 precondition artifact exists and passed**

Run:

```bash
uv run --frozen python - <<'PY'
import json
from pathlib import Path

path = Path(
    "data/08_reporting/hypotheses/"
    "h004_residual_diagnostic_started_at=20260508T182202655139Z/"
    "h004_diagnostic_decision.json"
)
decision = json.loads(path.read_text(encoding="utf-8"))
print(decision["diagnostic_status"], decision["passed_families"])
assert decision["diagnostic_status"] == "passes"
assert "C" in decision["passed_families"]
PY
```

Expected: prints `passes` with family `C`.

- [ ] **Step 2: Run the side-by-side experiment**

Run:

```bash
uv run --frozen python scripts/run_model_experiments.py \
  --group h004-attack-defense-mismatch \
  --seasons 2021,2022,2023,2024,2025 \
  --start-round 5 \
  --budget 100 \
  --current-year 2026 \
  --jobs 12 \
  --profile-runtime
```

Expected: 10 child runs complete and output path is printed.

- [ ] **Step 3: Write the deterministic H004 decision**

```bash
export OUTPUT_PATH="$(ls -td data/08_reporting/experiments/model_feature/group=h004-attack-defense-mismatch__* | head -1)"
uv run --frozen python scripts/run_h004_feature_decision.py \
  --experiment-path "$OUTPUT_PATH"
```

Expected: writes `$OUTPUT_PATH/h004_phase2_decision.json`.

- [ ] **Step 4: Inspect experiment result and decision**

```bash
export OUTPUT_PATH="$(ls -td data/08_reporting/experiments/model_feature/group=h004-attack-defense-mismatch__* | head -1)"
uv run --frozen python - <<'PY'
from pathlib import Path
import pandas as pd
import json
import os

path = Path(os.environ["OUTPUT_PATH"])
ranked = pd.read_csv(path / "ranked_summary.csv")
season = pd.read_csv(path / "per_season_summary.csv")
metrics = pd.read_csv(path / "prediction_metrics.csv")
decision = json.loads((path / "h004_phase2_decision.json").read_text(encoding="utf-8"))
print("output_path=", path)
print("decision=", decision["final_status"], decision["reasons"])
print(ranked.to_string(index=False))
print()
print(season.to_string(index=False))
print()
print(metrics[["model_id", "feature_pack", "season", "metric_scope", "spearman", "calibration_slope"]].to_string(index=False))
PY
```

Expected: the challenger `ppg_xg_matchup_h004` and control `ppg_xg_matchup` are both present.

- [ ] **Step 5: Commit code fixes if real run exposes defects**

If the real run exposes code defects, fix them, rerun focused tests, then:

```bash
git add src/cartola/backtesting/config.py src/cartola/backtesting/experiment_config.py src/cartola/backtesting/features.py src/cartola/backtesting/runner.py src/tests/backtesting/test_experiment_config.py src/tests/backtesting/test_features.py src/tests/backtesting/test_runner.py src/tests/backtesting/test_run_model_experiments_cli.py scripts/run_model_experiments.py
git commit -m "fix: handle real h004 feature experiment"
```

Expected: commit succeeds only if code changed.

## Task 8: Update Roadmap With Phase 2 Result

**Files:**
- Modify: `roadmap.md`

- [ ] **Step 1: Add result summary**

After the H004 Phase 1 interpretation, add:

```markdown
Latest H004 Phase 2 feature-pack experiment: `final_status` from `h004_phase2_decision.json`, output path from the real experiment run.
Control: `xgboost_depth2_slow + ppg_xg_matchup`.
Challenger: `xgboost_depth2_slow + ppg_xg_matchup_h004`.
Aggregate delta: `sum(season_deltas[].actual_points_delta)`.
Improved seasons: `count(season_deltas[].actual_points_delta > 0) / 5`.
2025 delta: `season_deltas` entry where `season == 2025`.
Decision: `final_status` from `h004_phase2_decision.json`.
Fixture identity: `fixture_identity_status` from `h004_phase2_decision.json`.
Candidate signatures: `candidate_signature_status` from `h004_phase2_decision.json`.
```

- [ ] **Step 2: Add run command to How To Run if missing**

Ensure `roadmap.md` includes:

```bash
uv run --frozen python scripts/run_model_experiments.py \
  --group h004-attack-defense-mismatch \
  --seasons 2021,2022,2023,2024,2025 \
  --start-round 5 \
  --budget 100 \
  --current-year 2026 \
  --jobs 12 \
  --profile-runtime
```

- [ ] **Step 3: Verify docs diff**

Run:

```bash
git diff --check roadmap.md
```

Expected: no output.

- [ ] **Step 4: Commit roadmap**

Run:

```bash
git add roadmap.md
git commit -m "docs: update roadmap for h004 feature experiment"
```

Expected: commit succeeds.

## Task 9: Final Verification

**Files:** verify only.

- [ ] **Step 1: Run full gate**

Run:

```bash
uv run --frozen scripts/pyrepo-check --all
```

Expected: Ruff, ty, Bandit, and pytest pass.

- [ ] **Step 2: Check git status**

Run:

```bash
git status --short --branch
```

Expected: clean branch, or only ignored generated report outputs.

- [ ] **Step 3: Final response**

Report:

```text
Implemented H004 Phase 2 feature pack.
Experiment output: the generated `group=h004-attack-defense-mismatch__...` directory.
Decision: the `final_status` value from `h004_phase2_decision.json`.
Verification: exact commands run and pass counts.
Next step: the action implied by `candidate_research`, `diagnostic_only`, `rejected`, or `invalid`.
```
