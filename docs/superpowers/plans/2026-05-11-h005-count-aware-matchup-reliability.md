# H005 Count-Aware Matchup Reliability Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement H005 as a research-only feature pack with a source-anchored mechanism audit and deterministic experiment decision artifact.

**Architecture:** Add a new `h005_matchup_reliability_v1` feature augmentation mode that computes position-normalized opponent-matchup reliability from cutoff-safe season history. Add a Phase 0 audit that reads source predictions, recomputes existing matchup counts from raw data/fixtures, validates source identity, and writes the H005 mechanism decision. Add a post-experiment decision helper modeled on H004, but with H005-specific statuses and gates.

**Tech Stack:** Python 3.13.12, pandas, pytest, uv, existing Cartola backtesting/experiment infrastructure.

---

## File Structure

- Modify `src/cartola/backtesting/config.py`
  - Add `h005_matchup_reliability_v1` to `FeatureAugmentationMode`.
- Modify `src/cartola/backtesting/features.py`
  - Add `H005_MATCHUP_RELIABILITY_FEATURE_COLUMNS`.
  - Add H005 helper functions for available opponent-match count, zero-count-densified position priors, and augmentation.
  - Wire `build_prediction_frame`, `build_training_frame`, and `feature_columns_for_config`.
- Modify `src/cartola/backtesting/runner.py`
  - Include H005 augmentation columns in metadata.
- Modify `src/cartola/backtesting/experiment_config.py`
  - Add group `h005-count-aware-matchup-shrinkage`.
  - Add feature pack `ppg_xg_matchup_h005`.
  - Matrix: `xgboost_depth2_slow` with control/challenger only.
- Create `src/cartola/backtesting/h005_mechanism_audit.py`
  - Source child discovery, source prediction loading, raw/fixture hash recording, H005 recomputation, residual bin summaries, audit decision.
- Create `scripts/run_h005_mechanism_audit.py`
  - CLI with progress output and artifact path.
- Create `src/cartola/backtesting/h005_feature_decision.py`
  - Deterministic post-experiment gates and statuses.
- Create `scripts/run_h005_feature_decision.py`
  - CLI for writing `h005_feature_decision.json`.
- Modify `src/tests/backtesting/test_features.py`
  - H005 feature mode and formula tests.
- Modify `src/tests/backtesting/test_experiment_config.py`
  - H005 matrix tests.
- Create `src/tests/backtesting/test_h005_mechanism_audit.py`
  - Audit source/provenance/bin/status tests.
- Create `src/tests/backtesting/test_h005_feature_decision.py`
  - Decision gate/status tests.
- Optionally update `AGENTS.md`
  - Add H005 commands after implementation is verified.

---

### Task 1: Wire H005 Feature Pack And Metadata

**Files:**
- Modify: `src/cartola/backtesting/config.py`
- Modify: `src/cartola/backtesting/features.py`
- Modify: `src/cartola/backtesting/runner.py`
- Modify: `src/cartola/backtesting/experiment_config.py`
- Test: `src/tests/backtesting/test_features.py`
- Test: `src/tests/backtesting/test_experiment_config.py`

- [ ] **Step 1: Write failing feature-column tests**

Add imports in `src/tests/backtesting/test_features.py`:

```python
from cartola.backtesting.features import (
    FEATURE_COLUMNS,
    FOOTYSTATS_PPG_FEATURE_COLUMNS,
    FOOTYSTATS_XG_FEATURE_COLUMNS,
    H004_ATTACK_DEFENSE_FEATURE_COLUMNS,
    H005_MATCHUP_RELIABILITY_FEATURE_COLUMNS,
    MATCHUP_CONTEXT_V1_FEATURE_COLUMNS,
    _add_h004_attack_defense_features,
    build_prediction_frame,
    build_training_frame,
    feature_columns_for_config,
)
```

Add tests near the H004 feature-column tests:

```python
def test_h005_feature_columns_are_added_only_for_h005_augmentation() -> None:
    base_columns = feature_columns_for_config(
        BacktestConfig(footystats_mode="ppg_xg", matchup_context_mode="cartola_matchup_v1")
    )
    h005_columns = feature_columns_for_config(
        BacktestConfig(
            footystats_mode="ppg_xg",
            matchup_context_mode="cartola_matchup_v1",
            feature_augmentation_mode="h005_matchup_reliability_v1",
        )
    )

    assert set(H005_MATCHUP_RELIABILITY_FEATURE_COLUMNS).isdisjoint(base_columns)
    assert h005_columns == [*base_columns, *H005_MATCHUP_RELIABILITY_FEATURE_COLUMNS]


def test_h005_feature_columns_require_matchup_context() -> None:
    with pytest.raises(ValueError, match="requires matchup_context_mode='cartola_matchup_v1'"):
        feature_columns_for_config(
            BacktestConfig(
                footystats_mode="ppg_xg",
                matchup_context_mode="none",
                feature_augmentation_mode="h005_matchup_reliability_v1",
            )
        )
```

- [ ] **Step 2: Write failing experiment-config tests**

Add tests to `src/tests/backtesting/test_experiment_config.py`:

```python
def test_h005_feature_pack_to_modes() -> None:
    feature_pack = feature_pack_to_modes("ppg_xg_matchup_h005")

    assert feature_pack.feature_pack == "ppg_xg_matchup_h005"
    assert feature_pack.footystats_mode == "ppg_xg"
    assert feature_pack.matchup_context_mode == "cartola_matchup_v1"
    assert feature_pack.feature_augmentation_mode == "h005_matchup_reliability_v1"


def test_h005_count_aware_matchup_reliability_matrix_is_control_vs_challenger_only() -> None:
    specs = build_child_run_specs(
        group="h005-count-aware-matchup-shrinkage",
        seasons=(2021, 2022, 2023, 2024, 2025),
        start_round=5,
        budget=100.0,
        current_year=2026,
        jobs=12,
        output_root=Path("out"),
    )

    assert len(specs) == 10
    assert {spec.model_id for spec in specs} == {"xgboost_depth2_slow"}
    assert {spec.feature_pack for spec in specs} == {"ppg_xg_matchup", "ppg_xg_matchup_h005"}
    h005_specs = [spec for spec in specs if spec.feature_pack == "ppg_xg_matchup_h005"]
    control_specs = [spec for spec in specs if spec.feature_pack == "ppg_xg_matchup"]
    assert {spec.backtest_config.feature_augmentation_mode for spec in h005_specs} == {
        "h005_matchup_reliability_v1"
    }
    assert {spec.backtest_config.feature_augmentation_mode for spec in control_specs} == {"none"}
```

- [ ] **Step 3: Run failing tests**

Run:

```bash
uv run --frozen pytest \
  src/tests/backtesting/test_features.py::test_h005_feature_columns_are_added_only_for_h005_augmentation \
  src/tests/backtesting/test_features.py::test_h005_feature_columns_require_matchup_context \
  src/tests/backtesting/test_experiment_config.py::test_h005_feature_pack_to_modes \
  src/tests/backtesting/test_experiment_config.py::test_h005_count_aware_matchup_reliability_matrix_is_control_vs_challenger_only \
  -q
```

Expected: FAIL because `H005_MATCHUP_RELIABILITY_FEATURE_COLUMNS`, feature pack `ppg_xg_matchup_h005`, group `h005-count-aware-matchup-shrinkage`, and mode `h005_matchup_reliability_v1` do not exist yet.

- [ ] **Step 4: Implement config and feature-column wiring**

In `src/cartola/backtesting/config.py`, update:

```python
FeatureAugmentationMode = Literal["none", "h004_attack_defense_v1", "h005_matchup_reliability_v1"]
```

In `src/cartola/backtesting/features.py`, add after `H004_ATTACK_DEFENSE_FEATURE_COLUMNS`:

```python
H005_MATCHUP_RELIABILITY_FEATURE_COLUMNS: list[str] = [
    "h005_opponent_position_available_match_count_roll5",
    "h005_opponent_position_expected_count_roll5",
    "h005_opponent_position_count_ratio",
]
```

In `feature_columns_for_config`, add after the H004 branch:

```python
    if config.feature_augmentation_mode == "h005_matchup_reliability_v1":
        if config.matchup_context_mode != "cartola_matchup_v1":
            raise ValueError(
                "feature_augmentation_mode='h005_matchup_reliability_v1' "
                "requires matchup_context_mode='cartola_matchup_v1'"
            )
        return [*columns, *H005_MATCHUP_RELIABILITY_FEATURE_COLUMNS]
```

In the empty-frame branch of `build_training_frame`, extend augmentation columns:

```python
        if feature_augmentation_mode == "h005_matchup_reliability_v1":
            feature_columns.extend(H005_MATCHUP_RELIABILITY_FEATURE_COLUMNS)
```

In `src/cartola/backtesting/runner.py`, import the H005 constant and update `_feature_augmentation_columns`:

```python
    if config.feature_augmentation_mode == "h005_matchup_reliability_v1":
        return list(H005_MATCHUP_RELIABILITY_FEATURE_COLUMNS)
```

In `src/cartola/backtesting/experiment_config.py`:

```python
ExperimentGroup = Literal[
    "production-parity",
    "matchup-research",
    "xgboost-research",
    "xgboost-sensitivity-v2",
    "h004-attack-defense-mismatch",
    "h005-count-aware-matchup-shrinkage",
]

FeaturePackId = Literal[
    "ppg",
    "ppg_xg",
    "ppg_matchup",
    "ppg_xg_matchup",
    "ppg_xg_matchup_h004",
    "ppg_xg_matchup_h005",
]
```

Add mappings:

```python
    "h005-count-aware-matchup-shrinkage": "exploratory",
```

```python
    "h005-count-aware-matchup-shrinkage": ("ppg_xg_matchup", "ppg_xg_matchup_h005"),
```

```python
    "h005-count-aware-matchup-shrinkage": ("xgboost_depth2_slow",),
```

Add feature pack:

```python
    "ppg_xg_matchup_h005": FeaturePack(
        feature_pack="ppg_xg_matchup_h005",
        footystats_mode="ppg_xg",
        matchup_context_mode="cartola_matchup_v1",
        feature_augmentation_mode="h005_matchup_reliability_v1",
    ),
```

- [ ] **Step 5: Run green tests**

Run:

```bash
uv run --frozen pytest \
  src/tests/backtesting/test_features.py::test_h005_feature_columns_are_added_only_for_h005_augmentation \
  src/tests/backtesting/test_features.py::test_h005_feature_columns_require_matchup_context \
  src/tests/backtesting/test_experiment_config.py::test_h005_feature_pack_to_modes \
  src/tests/backtesting/test_experiment_config.py::test_h005_count_aware_matchup_reliability_matrix_is_control_vs_challenger_only \
  -q
```

Expected: PASS.

- [ ] **Step 6: Commit Task 1**

```bash
git add src/cartola/backtesting/config.py src/cartola/backtesting/features.py src/cartola/backtesting/runner.py src/cartola/backtesting/experiment_config.py src/tests/backtesting/test_features.py src/tests/backtesting/test_experiment_config.py
git commit -m "feat: wire h005 reliability feature pack"
```

---

### Task 2: Implement H005 Feature Computation

**Files:**
- Modify: `src/cartola/backtesting/features.py`
- Test: `src/tests/backtesting/test_features.py`

- [ ] **Step 1: Write failing formula and denominator tests**

Add helper functions to `src/tests/backtesting/test_features.py`:

```python
def _h005_season_df() -> pd.DataFrame:
    rows = [
        # Round 1: both clubs active; club 10 has no lat, which must count as zero in the position prior.
        _h005_player(1, 101, 10, "gol", 3.0),
        _h005_player(1, 102, 10, "ata", 7.0),
        _h005_player(1, 201, 20, "gol", 4.0),
        _h005_player(1, 202, 20, "lat", 5.0),
        _h005_player(1, 203, 20, "ata", 8.0),
        # Round 2: both clubs active; club 20 has no lat, another zero for lat prior.
        _h005_player(2, 101, 10, "gol", 2.0),
        _h005_player(2, 102, 10, "ata", 6.0),
        _h005_player(2, 103, 10, "lat", 1.0),
        _h005_player(2, 201, 20, "gol", 5.0),
        _h005_player(2, 203, 20, "ata", 7.0),
        # Round 3 target candidates.
        _h005_player(3, 301, 10, "lat", 0.0, entered=False),
        _h005_player(3, 302, 10, "tec", 0.0, entered=False),
    ]
    frame = pd.DataFrame(rows)
    for scout in DEFAULT_SCOUT_COLUMNS:
        if scout not in frame.columns:
            frame[scout] = 0
    return frame


def _h005_player(round_number: int, athlete_id: int, club_id: int, position: str, points: float, *, entered: bool = True) -> dict[str, object]:
    return {
        "rodada": round_number,
        "id_atleta": athlete_id,
        "apelido": f"P{athlete_id}",
        "slug": f"p-{athlete_id}",
        "posicao": position,
        "status": "Provavel",
        "preco": 10.0,
        "preco_pre_rodada": 10.0,
        "pontuacao": points,
        "media": points,
        "num_jogos": round_number,
        "variacao": 0.0,
        "id_clube": club_id,
        "nome_clube": f"Clube {club_id}",
        "entrou_em_campo": entered,
        "G": 0,
        "A": 0,
        "DS": 0,
        "V": 0,
    }


def _h005_fixture_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"rodada": 1, "id_clube_home": 10, "id_clube_away": 20, "data": "2025-03-29"},
            {"rodada": 2, "id_clube_home": 20, "id_clube_away": 10, "data": "2025-04-05"},
            {"rodada": 3, "id_clube_home": 10, "id_clube_away": 20, "data": "2025-04-12"},
        ]
    )
```

Add tests:

```python
def test_h005_reliability_counts_position_zero_opponent_matches_as_available() -> None:
    frame = build_prediction_frame(
        _h005_season_df(),
        target_round=3,
        fixtures=_h005_fixture_df(),
        matchup_context_mode="cartola_matchup_v1",
        feature_augmentation_mode="h005_matchup_reliability_v1",
    )

    lat = frame.loc[frame["posicao"].eq("lat")].iloc[0]

    assert lat["matchup_opponent_allowed_position_count"] == 1
    assert lat["h005_opponent_position_available_match_count_roll5"] == 2
    assert lat["h005_opponent_position_expected_count_roll5"] == pytest.approx(1.0)
    assert lat["h005_opponent_position_count_ratio"] == pytest.approx(1.0)


def test_h005_position_prior_densifies_zero_count_team_position_rounds() -> None:
    frame = build_prediction_frame(
        _h005_season_df(),
        target_round=3,
        fixtures=_h005_fixture_df(),
        matchup_context_mode="cartola_matchup_v1",
        feature_augmentation_mode="h005_matchup_reliability_v1",
    )

    lat = frame.loc[frame["posicao"].eq("lat")].iloc[0]

    assert lat["h005_opponent_position_expected_count_roll5"] == pytest.approx(1.0)


def test_h005_reliability_sets_tecnico_columns_to_zero() -> None:
    frame = build_prediction_frame(
        _h005_season_df(),
        target_round=3,
        fixtures=_h005_fixture_df(),
        matchup_context_mode="cartola_matchup_v1",
        feature_augmentation_mode="h005_matchup_reliability_v1",
    )

    tecnico = frame.loc[frame["posicao"].eq("tec")].iloc[0]
    assert tecnico[list(H005_MATCHUP_RELIABILITY_FEATURE_COLUMNS)].tolist() == [0.0, 0.0, 0.0]
```

- [ ] **Step 2: Run failing tests**

Run:

```bash
uv run --frozen pytest \
  src/tests/backtesting/test_features.py::test_h005_reliability_counts_position_zero_opponent_matches_as_available \
  src/tests/backtesting/test_features.py::test_h005_position_prior_densifies_zero_count_team_position_rounds \
  src/tests/backtesting/test_features.py::test_h005_reliability_sets_tecnico_columns_to_zero \
  -q
```

Expected: FAIL because `build_prediction_frame` does not implement H005 augmentation.

- [ ] **Step 3: Implement H005 helpers**

In `src/cartola/backtesting/features.py`, add:

```python
STANDARD_PLAYER_POSITIONS: tuple[str, ...] = ("gol", "lat", "zag", "mei", "ata")
```

Add helper functions after `_opponent_allowed_roll5`:

```python
def _opponent_available_match_count_roll5(
    played_history: pd.DataFrame,
    fixtures: pd.DataFrame | None,
    target_round: int,
) -> pd.DataFrame:
    columns = pd.Index(["opponent_id_clube", "h005_opponent_position_available_match_count_roll5"])
    fixture_context = _historical_fixture_context(fixtures, target_round)
    if played_history.empty or fixture_context.empty:
        return pd.DataFrame(columns=columns)

    scored_against = played_history.merge(
        fixture_context,
        on=["rodada", "id_clube"],
        how="inner",
        validate="many_to_one",
    )
    if scored_against.empty:
        return pd.DataFrame(columns=columns)

    opponent_round = (
        scored_against[["opponent_id_clube", "rodada"]]
        .drop_duplicates()
        .sort_values(["opponent_id_clube", "rodada"])
    )
    rows: list[dict[str, object]] = []
    for opponent_id, group in opponent_round.groupby("opponent_id_clube", sort=False, dropna=False):
        recent = group.tail(5)
        rows.append(
            {
                "opponent_id_clube": opponent_id,
                "h005_opponent_position_available_match_count_roll5": int(recent["rodada"].nunique()),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def _h005_position_expected_counts(played_history: pd.DataFrame) -> pd.DataFrame:
    columns = pd.Index(["posicao", "h005_position_players_per_team_round_prior"])
    if played_history.empty:
        return pd.DataFrame(columns=columns)

    played = played_history.loc[played_history["posicao"].isin(STANDARD_PLAYER_POSITIONS)].copy()
    if played.empty:
        return pd.DataFrame(columns=columns)

    active_clubs = played.groupby(["rodada", "id_clube"], as_index=False).size()[["rodada", "id_clube"]]
    positions = pd.DataFrame({"posicao": list(STANDARD_PLAYER_POSITIONS)})
    dense = active_clubs.merge(positions, how="cross")
    observed = (
        played.groupby(["rodada", "id_clube", "posicao"], as_index=False)
        .agg(observed_players=("id_atleta", "nunique"))
    )
    dense = dense.merge(observed, on=["rodada", "id_clube", "posicao"], how="left")
    dense["observed_players"] = pd.to_numeric(dense["observed_players"], errors="coerce").fillna(0.0)
    return dense.groupby("posicao", as_index=False).agg(
        h005_position_players_per_team_round_prior=("observed_players", "mean")
    )
```

Add H005 augmentation helper:

```python
def _add_h005_matchup_reliability_features(
    frame: pd.DataFrame,
    played_history: pd.DataFrame,
    fixtures: pd.DataFrame | None,
    target_round: int,
) -> pd.DataFrame:
    required_columns = ["id_clube", "posicao", "matchup_opponent_allowed_position_count"]
    missing_columns = [column for column in required_columns if column not in frame.columns]
    if missing_columns:
        raise ValueError(f"H005 feature augmentation requires columns: {', '.join(missing_columns)}")

    result = frame.copy()
    round_context = _round_fixture_context(fixtures, target_round)
    if round_context.empty:
        result["opponent_id_clube"] = pd.NA
    else:
        result = result.merge(
            round_context[["id_clube", "opponent_id_clube"]],
            on="id_clube",
            how="left",
            validate="many_to_one",
        )

    available = _opponent_available_match_count_roll5(played_history, fixtures, target_round)
    expected = _h005_position_expected_counts(played_history)
    result = result.merge(available, on="opponent_id_clube", how="left", validate="many_to_one")
    result = result.merge(expected, on="posicao", how="left", validate="many_to_one")

    global_expected = float(expected["h005_position_players_per_team_round_prior"].mean()) if not expected.empty else 1.0
    if not np.isfinite(global_expected) or global_expected <= 0.0:
        global_expected = 1.0

    available_column = "h005_opponent_position_available_match_count_roll5"
    prior_column = "h005_position_players_per_team_round_prior"
    result[available_column] = pd.to_numeric(result[available_column], errors="coerce").fillna(0.0).clip(lower=0.0)
    result[prior_column] = pd.to_numeric(result[prior_column], errors="coerce").fillna(global_expected).clip(lower=0.0)
    result["h005_opponent_position_expected_count_roll5"] = (
        result[available_column] * result[prior_column]
    ).clip(lower=1.0)
    count = pd.to_numeric(result["matchup_opponent_allowed_position_count"], errors="coerce").fillna(0.0).clip(lower=0.0)
    result["h005_opponent_position_count_ratio"] = count / result["h005_opponent_position_expected_count_roll5"]

    tecnico_mask = result["posicao"].astype(str).eq("tec")
    result.loc[tecnico_mask, H005_MATCHUP_RELIABILITY_FEATURE_COLUMNS] = 0.0
    for column in H005_MATCHUP_RELIABILITY_FEATURE_COLUMNS:
        result[column] = pd.to_numeric(result[column], errors="raise").astype(float)
    numeric_context = result[H005_MATCHUP_RELIABILITY_FEATURE_COLUMNS]
    invalid = numeric_context.isna().any(axis=1) | ~np.isfinite(numeric_context).all(axis=1)
    if bool(invalid.any()):
        raise ValueError("H005 feature augmentation produced non-finite reliability context")
    return result.drop(columns=["opponent_id_clube", prior_column], errors="ignore")
```

In `build_prediction_frame`, add after H004 branch:

```python
    if feature_augmentation_mode == "h005_matchup_reliability_v1":
        if matchup_context_mode != "cartola_matchup_v1":
            raise ValueError(
                "feature_augmentation_mode='h005_matchup_reliability_v1' "
                "requires matchup_context_mode='cartola_matchup_v1'"
            )
        return _add_h005_matchup_reliability_features(frame, played_history, fixtures, target_round)
```

- [ ] **Step 4: Run green tests**

Run:

```bash
uv run --frozen pytest \
  src/tests/backtesting/test_features.py::test_h005_reliability_counts_position_zero_opponent_matches_as_available \
  src/tests/backtesting/test_features.py::test_h005_position_prior_densifies_zero_count_team_position_rounds \
  src/tests/backtesting/test_h005_reliability_sets_tecnico_columns_to_zero \
  src/tests/backtesting/test_features.py::test_h005_feature_columns_are_added_only_for_h005_augmentation \
  -q
```

Expected: PASS.

- [ ] **Step 5: Run broader feature tests**

Run:

```bash
uv run --frozen pytest src/tests/backtesting/test_features.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit Task 2**

```bash
git add src/cartola/backtesting/features.py src/tests/backtesting/test_features.py
git commit -m "feat: compute h005 matchup reliability features"
```

---

### Task 3: Implement Phase 0 Mechanism Audit

**Files:**
- Create: `src/cartola/backtesting/h005_mechanism_audit.py`
- Create: `scripts/run_h005_mechanism_audit.py`
- Test: `src/tests/backtesting/test_h005_mechanism_audit.py`

- [ ] **Step 1: Write failing source discovery and mismatch tests**

Create `src/tests/backtesting/test_h005_mechanism_audit.py`:

```python
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from cartola.backtesting.h005_mechanism_audit import (
    H005MechanismAuditError,
    build_h005_mechanism_audit,
    discover_h005_source_children,
)


def test_discover_h005_source_children_requires_one_child_per_season(tmp_path: Path) -> None:
    experiment = tmp_path / "experiment"
    child = experiment / "runs" / "season=2021" / "model=xgboost_depth2_slow" / "feature_pack=ppg_xg_matchup"
    child.mkdir(parents=True)
    (child / "run_metadata.json").write_text(
        json.dumps(
            {
                "season": 2021,
                "model_id": "xgboost_depth2_slow",
                "feature_pack": "ppg_xg_matchup",
                "fixture_mode": "exploratory",
                "matchup_context_mode": "cartola_matchup_v1",
                "footystats_mode": "ppg_xg",
            }
        ),
        encoding="utf-8",
    )

    children = discover_h005_source_children(
        experiment_path=experiment,
        seasons=(2021,),
        model_id="xgboost_depth2_slow",
        feature_pack="ppg_xg_matchup",
    )

    assert len(children) == 1
    assert children[0].season == 2021


def test_h005_mechanism_audit_invalidates_recomputed_count_mismatch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    experiment = _write_source_experiment(tmp_path)
    output_path = tmp_path / "audit"

    monkeypatch.setattr(
        "cartola.backtesting.h005_mechanism_audit.load_season_data",
        lambda season, project_root: _raw_season(),
    )
    monkeypatch.setattr(
        "cartola.backtesting.h005_mechanism_audit.load_fixtures",
        lambda season, project_root: _fixtures(),
    )

    # Persisted count is intentionally wrong; recomputed count should be 1.
    predictions = pd.read_csv(
        experiment
        / "runs"
        / "season=2021"
        / "model=xgboost_depth2_slow"
        / "feature_pack=ppg_xg_matchup"
        / "player_predictions.csv"
    )
    predictions["matchup_opponent_allowed_position_count"] = 99
    predictions.to_csv(
        experiment
        / "runs"
        / "season=2021"
        / "model=xgboost_depth2_slow"
        / "feature_pack=ppg_xg_matchup"
        / "player_predictions.csv",
        index=False,
    )

    result = build_h005_mechanism_audit(
        experiment_path=experiment,
        output_path=output_path,
        seasons=(2021,),
        model_id="xgboost_depth2_slow",
        feature_pack="ppg_xg_matchup",
        project_root=tmp_path,
    )

    assert result.decision["audit_status"] == "invalid"
    assert "recomputed_count_mismatch" in result.decision["failed_checks"]
```

Add helper fixtures in the same test file:

```python
def _write_source_experiment(tmp_path: Path) -> Path:
    experiment = tmp_path / "experiment"
    child = experiment / "runs" / "season=2021" / "model=xgboost_depth2_slow" / "feature_pack=ppg_xg_matchup"
    child.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "rodada": 3,
                "id_atleta": 301,
                "apelido": "LAT",
                "id_clube": 10,
                "posicao": "lat",
                "status": "Provavel",
                "preco_pre_rodada": 10.0,
                "pontuacao": 4.0,
                "entrou_em_campo": True,
                "xgboost_depth2_slow_score": 5.0,
                "matchup_opponent_allowed_position_count": 1,
                "matchup_opponent_allowed_position_points_roll5": 5.0,
                "matchup_opponent_allowed_points_roll5": 6.0,
            }
        ]
    ).to_csv(child / "player_predictions.csv", index=False)
    (child / "selected_players.csv").write_text("rodada,id_atleta,entrou_em_campo\n3,301,true\n", encoding="utf-8")
    (child / "run_metadata.json").write_text(
        json.dumps(
            {
                "season": 2021,
                "model_id": "xgboost_depth2_slow",
                "feature_pack": "ppg_xg_matchup",
                "fixture_mode": "exploratory",
                "matchup_context_mode": "cartola_matchup_v1",
                "footystats_mode": "ppg_xg",
                "fixture_source_sha256": {"fixture.csv": "fixture-sha"},
            }
        ),
        encoding="utf-8",
    )
    return experiment


def _raw_season() -> pd.DataFrame:
    rows = [
        _raw_player(1, 101, 10, "lat", 5.0),
        _raw_player(1, 201, 20, "gol", 3.0),
        _raw_player(2, 102, 10, "ata", 7.0),
        _raw_player(2, 202, 20, "gol", 4.0),
        _raw_player(3, 301, 10, "lat", 4.0),
    ]
    frame = pd.DataFrame(rows)
    for scout in ("G", "A", "DS", "V"):
        if scout not in frame.columns:
            frame[scout] = 0
    return frame


def _raw_player(round_number: int, athlete_id: int, club_id: int, position: str, points: float) -> dict[str, object]:
    return {
        "rodada": round_number,
        "id_atleta": athlete_id,
        "apelido": str(athlete_id),
        "slug": str(athlete_id),
        "posicao": position,
        "status": "Provavel",
        "preco": 10.0,
        "preco_pre_rodada": 10.0,
        "pontuacao": points,
        "media": points,
        "num_jogos": 1,
        "variacao": 0.0,
        "id_clube": club_id,
        "nome_clube": str(club_id),
        "entrou_em_campo": True,
    }


def _fixtures() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"rodada": 1, "id_clube_home": 10, "id_clube_away": 20, "data": "2021-01-01"},
            {"rodada": 2, "id_clube_home": 20, "id_clube_away": 10, "data": "2021-01-08"},
            {"rodada": 3, "id_clube_home": 10, "id_clube_away": 20, "data": "2021-01-15"},
        ]
    )
```

- [ ] **Step 2: Run failing audit tests**

Run:

```bash
uv run --frozen pytest src/tests/backtesting/test_h005_mechanism_audit.py -q
```

Expected: FAIL because the module does not exist.

- [ ] **Step 3: Implement audit module skeleton and source loading**

Create `src/cartola/backtesting/h005_mechanism_audit.py` with:

```python
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from cartola.backtesting.data import load_fixtures, load_season_data
from cartola.backtesting.features import H005_MATCHUP_RELIABILITY_FEATURE_COLUMNS, build_prediction_frame

H005_REQUIRED_PREDICTION_COLUMNS: tuple[str, ...] = (
    "rodada",
    "id_atleta",
    "apelido",
    "id_clube",
    "posicao",
    "status",
    "preco_pre_rodada",
    "pontuacao",
    "entrou_em_campo",
    "matchup_opponent_allowed_position_count",
    "matchup_opponent_allowed_position_points_roll5",
    "matchup_opponent_allowed_points_roll5",
)


class H005MechanismAuditError(ValueError):
    """Raised when H005 mechanism audit artifacts cannot be interpreted."""


@dataclass(frozen=True)
class H005SourceChild:
    season: int
    model_id: str
    feature_pack: str
    child_path: Path
    score_column: str


@dataclass(frozen=True)
class H005MechanismAuditResult:
    output_path: Path
    audit: pd.DataFrame
    raw_count_audit: pd.DataFrame
    decision: dict[str, Any]


def discover_h005_source_children(
    *,
    experiment_path: Path,
    seasons: tuple[int, ...],
    model_id: str,
    feature_pack: str,
) -> tuple[H005SourceChild, ...]:
    children: list[H005SourceChild] = []
    for season in seasons:
        child_path = (
            Path(experiment_path)
            / "runs"
            / f"season={season}"
            / f"model={model_id}"
            / f"feature_pack={feature_pack}"
        )
        if not child_path.is_dir():
            raise FileNotFoundError(f"Missing H005 source child: {child_path}")
        metadata = _read_json(child_path / "run_metadata.json")
        children.append(
            H005SourceChild(
                season=int(metadata.get("season", season)),
                model_id=str(metadata.get("model_id", model_id)),
                feature_pack=str(metadata.get("feature_pack", feature_pack)),
                child_path=child_path,
                score_column=f"{model_id}_score",
            )
        )
    return tuple(children)
```

Implement `build_h005_mechanism_audit` minimally:

```python
def build_h005_mechanism_audit(
    *,
    experiment_path: Path,
    output_path: Path,
    seasons: tuple[int, ...],
    model_id: str,
    feature_pack: str,
    project_root: Path = Path("."),
) -> H005MechanismAuditResult:
    children = discover_h005_source_children(
        experiment_path=Path(experiment_path),
        seasons=seasons,
        model_id=model_id,
        feature_pack=feature_pack,
    )
    frames = []
    failed_checks: list[str] = []
    for child in children:
        source = _load_source_predictions(child)
        recomputed = _recompute_h005_features(child, project_root=project_root)
        merged = source.merge(
            recomputed,
            on=["rodada", "id_atleta"],
            suffixes=("_source", "_recomputed"),
            how="outer",
            indicator=True,
            validate="one_to_one",
        )
        if not merged["_merge"].eq("both").all():
            failed_checks.append("row_identity_mismatch")
        count_match = (
            merged["matchup_opponent_allowed_position_count_source"].fillna(-1).astype(int)
            == merged["matchup_opponent_allowed_position_count_recomputed"].fillna(-2).astype(int)
        )
        if not bool(count_match.all()):
            failed_checks.append("recomputed_count_mismatch")
        frames.append(merged)

    joined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    audit = _build_ratio_audit(joined)
    raw_count_audit = _build_raw_count_audit(joined)
    decision = _build_audit_decision(audit=audit, raw_count_audit=raw_count_audit, failed_checks=failed_checks)
    result = H005MechanismAuditResult(output_path=Path(output_path), audit=audit, raw_count_audit=raw_count_audit, decision=decision)
    write_h005_mechanism_audit_artifacts(result)
    return result
```

Add helpers:

```python
def _load_source_predictions(child: H005SourceChild) -> pd.DataFrame:
    path = child.child_path / "player_predictions.csv"
    frame = pd.read_csv(path)
    required = (*H005_REQUIRED_PREDICTION_COLUMNS, child.score_column)
    _validate_columns("player_predictions.csv", frame, required)
    output = frame.loc[:, list(required)].copy()
    output["season"] = child.season
    output["predicted_points"] = pd.to_numeric(output[child.score_column], errors="coerce")
    output["actual_points"] = pd.to_numeric(output["pontuacao"], errors="coerce")
    output["source_residual"] = output["actual_points"] - output["predicted_points"]
    return output


def _recompute_h005_features(child: H005SourceChild, *, project_root: Path) -> pd.DataFrame:
    season_df = load_season_data(child.season, project_root=project_root)
    fixtures = load_fixtures(child.season, project_root=project_root)
    source = pd.read_csv(child.child_path / "player_predictions.csv")
    frames = []
    for round_number in sorted(source["rodada"].dropna().astype(int).unique()):
        recomputed = build_prediction_frame(
            season_df,
            round_number,
            fixtures=fixtures,
            matchup_context_mode="cartola_matchup_v1",
            feature_augmentation_mode="h005_matchup_reliability_v1",
        )
        keep = [
            "rodada",
            "id_atleta",
            "matchup_opponent_allowed_position_count",
            *H005_MATCHUP_RELIABILITY_FEATURE_COLUMNS,
        ]
        frames.append(recomputed.loc[:, keep])
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _validate_columns(label: str, frame: pd.DataFrame, required: tuple[str, ...]) -> None:
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise H005MechanismAuditError(f"{label} missing columns: {', '.join(missing)}")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))
```

Add bin and artifact helpers:

```python
def _build_ratio_audit(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    working = frame.copy()
    working["ratio_bin"] = pd.cut(
        working["h005_opponent_position_count_ratio"],
        bins=[-0.001, 0.0, 0.5, 0.8, 1.0, 1.5, float("inf")],
        labels=["0", "(0, 0.5]", "(0.5, 0.8]", "(0.8, 1.0]", "(1.0, 1.5]", "> 1.5"],
        include_lowest=True,
    )
    return _audit_group(working, bin_column="ratio_bin")


def _build_raw_count_audit(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    working = frame.copy()
    working["raw_count_bin"] = pd.cut(
        working["matchup_opponent_allowed_position_count_source"],
        bins=[-0.001, 0.0, 5.0, 10.0, 20.0, 30.0, float("inf")],
        labels=["0", "(0, 5]", "(5, 10]", "(10, 20]", "(20, 30]", "> 30"],
        include_lowest=True,
    )
    return _audit_group(working, bin_column="raw_count_bin")


def _audit_group(frame: pd.DataFrame, *, bin_column: str) -> pd.DataFrame:
    grouped = frame.groupby(["season", "posicao", bin_column], observed=False)
    return grouped.agg(
        row_count=("id_atleta", "size"),
        round_count=("rodada", "nunique"),
        source_residual_mean=("source_residual", "mean"),
        source_overprediction_rate=("source_residual", lambda values: float((values < 0).mean())),
        mean_matchup_opponent_allowed_position_count=("matchup_opponent_allowed_position_count_source", "mean"),
        mean_h005_opponent_position_available_match_count_roll5=(
            "h005_opponent_position_available_match_count_roll5",
            "mean",
        ),
        mean_h005_opponent_position_expected_count_roll5=("h005_opponent_position_expected_count_roll5", "mean"),
        mean_h005_opponent_position_count_ratio=("h005_opponent_position_count_ratio", "mean"),
        mean_matchup_opponent_allowed_position_points_roll5=(
            "matchup_opponent_allowed_position_points_roll5",
            "mean",
        ),
        mean_matchup_opponent_allowed_points_roll5=("matchup_opponent_allowed_points_roll5", "mean"),
    ).reset_index()


def _build_audit_decision(*, audit: pd.DataFrame, raw_count_audit: pd.DataFrame, failed_checks: list[str]) -> dict[str, Any]:
    if failed_checks:
        return {"hypothesis_id": "H005", "audit_status": "invalid", "failed_checks": sorted(set(failed_checks))}
    supported = _audit_supports_reliability(audit, raw_count_audit)
    return {
        "hypothesis_id": "H005",
        "audit_status": "supports_reliability_hypothesis" if supported else "mixed_or_weak",
        "failed_checks": [],
    }


def _audit_supports_reliability(audit: pd.DataFrame, raw_count_audit: pd.DataFrame) -> bool:
    if audit.empty or raw_count_audit.empty:
        return False
    low = audit[audit["ratio_bin"].astype(str).isin(["0", "(0, 0.5]", "(0.5, 0.8]"])]
    positions = set(low.loc[low["row_count"].ge(100), "posicao"].astype(str))
    return len(positions.difference({"tec"})) >= 4


def write_h005_mechanism_audit_artifacts(result: H005MechanismAuditResult) -> None:
    result.output_path.mkdir(parents=True, exist_ok=True)
    result.audit.to_csv(result.output_path / "h005_mechanism_audit.csv", index=False)
    result.raw_count_audit.to_csv(result.output_path / "h005_raw_count_audit.csv", index=False)
    (result.output_path / "h005_mechanism_audit_decision.json").write_text(
        json.dumps(result.decision, indent=2, sort_keys=True),
        encoding="utf-8",
    )
```

- [ ] **Step 4: Add CLI**

Create `scripts/run_h005_mechanism_audit.py`:

```python
#!/usr/bin/env python
from __future__ import annotations

import argparse
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.panel import Panel

build_h005_mechanism_audit: Callable[..., Any] | None = None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run H005 mechanism audit from source experiment artifacts.")
    parser.add_argument("--experiment-path", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("data/08_reporting/hypotheses"))
    parser.add_argument("--seasons", type=_parse_seasons, default=(2021, 2022, 2023, 2024, 2025))
    parser.add_argument("--model-id", default="xgboost_depth2_slow")
    parser.add_argument("--feature-pack", default="ppg_xg_matchup")
    parser.add_argument("--project-root", type=Path, default=Path("."))
    return parser.parse_args(argv)


def _parse_seasons(value: str) -> tuple[int, ...]:
    seasons = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    duplicates = sorted({season for season in seasons if seasons.count(season) > 1})
    if duplicates:
        raise argparse.ArgumentTypeError(f"Duplicate seasons are not allowed: {duplicates}")
    return seasons


def _timestamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")


def _load_runtime_dependencies() -> None:
    global build_h005_mechanism_audit
    if build_h005_mechanism_audit is None:
        from cartola.backtesting.h005_mechanism_audit import build_h005_mechanism_audit as imported

        build_h005_mechanism_audit = imported


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    console = Console()
    output_path = args.output_root / f"h005_mechanism_audit_started_at={_timestamp()}"
    console.print(
        f"H005 mechanism audit started: seasons={','.join(str(season) for season in args.seasons)} "
        f"model_id={args.model_id} feature_pack={args.feature_pack} output={output_path}"
    )
    _load_runtime_dependencies()
    if build_h005_mechanism_audit is None:
        raise RuntimeError("H005 mechanism audit dependencies were not loaded.")
    result = build_h005_mechanism_audit(
        experiment_path=args.experiment_path,
        output_path=output_path,
        seasons=args.seasons,
        model_id=str(args.model_id),
        feature_pack=str(args.feature_pack),
        project_root=args.project_root,
    )
    console.print(
        Panel(
            f"audit_status={result.decision.get('audit_status')}\noutput_path={result.output_path}",
            title="H005 mechanism audit complete",
            border_style="green",
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 5: Run audit tests**

Run:

```bash
uv run --frozen pytest src/tests/backtesting/test_h005_mechanism_audit.py -q
```

Expected: PASS after the minimal audit implementation.

- [ ] **Step 6: Strengthen audit gate tests**

Add a test in `src/tests/backtesting/test_h005_mechanism_audit.py`:

```python
def test_h005_mechanism_audit_writes_required_artifacts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    experiment = _write_source_experiment(tmp_path)
    output_path = tmp_path / "audit"
    monkeypatch.setattr(
        "cartola.backtesting.h005_mechanism_audit.load_season_data",
        lambda season, project_root: _raw_season(),
    )
    monkeypatch.setattr(
        "cartola.backtesting.h005_mechanism_audit.load_fixtures",
        lambda season, project_root: _fixtures(),
    )

    result = build_h005_mechanism_audit(
        experiment_path=experiment,
        output_path=output_path,
        seasons=(2021,),
        model_id="xgboost_depth2_slow",
        feature_pack="ppg_xg_matchup",
        project_root=tmp_path,
    )

    assert result.output_path == output_path
    assert (output_path / "h005_mechanism_audit.csv").is_file()
    assert (output_path / "h005_raw_count_audit.csv").is_file()
    assert (output_path / "h005_mechanism_audit_decision.json").is_file()
```

Run:

```bash
uv run --frozen pytest src/tests/backtesting/test_h005_mechanism_audit.py -q
```

Expected: PASS.

- [ ] **Step 7: Commit Task 3**

```bash
git add src/cartola/backtesting/h005_mechanism_audit.py scripts/run_h005_mechanism_audit.py src/tests/backtesting/test_h005_mechanism_audit.py
git commit -m "feat: add h005 mechanism audit"
```

---

### Task 4: Implement H005 Post-Experiment Decision

**Files:**
- Create: `src/cartola/backtesting/h005_feature_decision.py`
- Create: `scripts/run_h005_feature_decision.py`
- Test: `src/tests/backtesting/test_h005_feature_decision.py`

- [ ] **Step 1: Write failing decision tests**

Create `src/tests/backtesting/test_h005_feature_decision.py`:

```python
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from cartola.backtesting.h005_feature_decision import (
    H005FeatureDecisionError,
    build_h005_feature_decision,
    write_h005_feature_decision,
)


def test_h005_decision_is_candidate_research_profile_when_all_gates_pass(tmp_path: Path) -> None:
    audit = _write_audit_decision(tmp_path, status="supports_reliability_hypothesis")
    experiment = _write_experiment(
        tmp_path,
        fixture_hashes={"fixture.csv": "same"},
        season_deltas={2021: 25.0, 2022: 22.0, 2023: 20.0, 2024: 15.0, 2025: 12.0},
    )

    decision = build_h005_feature_decision(experiment_path=experiment, audit_decision_path=audit)

    assert decision["decision_status"] == "candidate_research_profile"
    assert decision["mechanism_audit_status"] == "supports_reliability_hypothesis"


def test_h005_decision_is_weak_positive_for_stable_small_lift(tmp_path: Path) -> None:
    audit = _write_audit_decision(tmp_path, status="supports_reliability_hypothesis")
    experiment = _write_experiment(
        tmp_path,
        fixture_hashes={"fixture.csv": "same"},
        season_deltas={2021: 12.0, 2022: 10.0, 2023: 8.0, 2024: 7.0, 2025: 6.0},
    )

    decision = build_h005_feature_decision(experiment_path=experiment, audit_decision_path=audit)

    assert decision["decision_status"] == "weak_positive_research_lead"


def test_h005_decision_is_inconclusive_inside_noise_band(tmp_path: Path) -> None:
    audit = _write_audit_decision(tmp_path, status="supports_reliability_hypothesis")
    experiment = _write_experiment(
        tmp_path,
        fixture_hashes={"fixture.csv": "same"},
        season_deltas={2021: 6.0, 2022: 5.0, 2023: 4.0, 2024: 3.0, 2025: 2.0},
    )

    decision = build_h005_feature_decision(experiment_path=experiment, audit_decision_path=audit)

    assert decision["decision_status"] == "inconclusive"


def test_h005_decision_is_diagnostic_only_when_audit_is_mixed(tmp_path: Path) -> None:
    audit = _write_audit_decision(tmp_path, status="mixed_or_weak")
    experiment = _write_experiment(
        tmp_path,
        fixture_hashes={"fixture.csv": "same"},
        season_deltas={2021: 25.0, 2022: 22.0, 2023: 20.0, 2024: 15.0, 2025: 12.0},
    )

    decision = build_h005_feature_decision(experiment_path=experiment, audit_decision_path=audit)

    assert decision["decision_status"] == "diagnostic_only"
```

Add helpers patterned after `test_h004_feature_decision.py`, replacing feature packs:

```python
CONTROL_CHILD_2021 = "season=2021/model=xgboost_depth2_slow/feature_pack=ppg_xg_matchup"
CHALLENGER_CHILD_2021 = "season=2021/model=xgboost_depth2_slow/feature_pack=ppg_xg_matchup_h005"


def _write_audit_decision(tmp_path: Path, *, status: str) -> Path:
    path = tmp_path / "h005_mechanism_audit_decision.json"
    path.write_text(json.dumps({"hypothesis_id": "H005", "audit_status": status}), encoding="utf-8")
    return path
```

Use the same `_write_experiment`, `_child_record`, `_season_row`, and `_metric_row` shape from `test_h004_feature_decision.py`, but:

```python
"group": "h005-count-aware-matchup-shrinkage"
```

and challenger feature pack:

```python
"ppg_xg_matchup_h005"
```

with challenger feature augmentation mode:

```python
"h005_matchup_reliability_v1"
```

- [ ] **Step 2: Run failing decision tests**

Run:

```bash
uv run --frozen pytest src/tests/backtesting/test_h005_feature_decision.py -q
```

Expected: FAIL because `cartola.backtesting.h005_feature_decision` does not exist.

- [ ] **Step 3: Implement decision helper**

Create `src/cartola/backtesting/h005_feature_decision.py` using the H004 structure with these constants:

```python
CONTROL_MODEL_ID = "xgboost_depth2_slow"
CONTROL_FEATURE_PACK = "ppg_xg_matchup"
CHALLENGER_FEATURE_PACK = "ppg_xg_matchup_h005"
REQUIRED_SEASONS = (2021, 2022, 2023, 2024, 2025)
```

Public functions:

```python
def write_h005_feature_decision(*, experiment_path: Path, audit_decision_path: Path) -> Path:
    decision = build_h005_feature_decision(experiment_path=experiment_path, audit_decision_path=audit_decision_path)
    output_path = experiment_path / "h005_feature_decision.json"
    output_path.write_text(json.dumps(decision, indent=2, sort_keys=True), encoding="utf-8")
    return output_path


def build_h005_feature_decision(*, experiment_path: Path, audit_decision_path: Path) -> dict[str, Any]:
    artifacts = _load_required_artifacts(Path(experiment_path))
    audit = _read_json(Path(audit_decision_path), label="H005 mechanism audit decision")
    mechanism_audit_status = str(audit.get("audit_status", "invalid"))
    fixture_identity_status = _fixture_identity_status(artifacts["metadata"])
    candidate_signature_status = _candidate_signature_status(artifacts["metadata"])
    validation_errors = _validation_errors(
        artifacts=artifacts,
        mechanism_audit_status=mechanism_audit_status,
        fixture_identity_status=fixture_identity_status,
        candidate_signature_status=candidate_signature_status,
    )
    payload = _build_gate_payload(artifacts)
    decision_status = _decision_status(
        validation_errors=validation_errors,
        mechanism_audit_status=mechanism_audit_status,
        fixture_identity_status=fixture_identity_status,
        gate_results=payload["gate_results"],
        aggregate_delta=float(payload["aggregate_actual_points_delta"]),
    )
    return {
        "hypothesis_id": "H005",
        "h005_design_revision": "reliability_v1",
        "manual_points_shrinkage": False,
        "decision_status": decision_status,
        "mechanism_audit_status": mechanism_audit_status,
        "fixture_identity_status": fixture_identity_status,
        "candidate_signature_status": candidate_signature_status,
        "control_strategy": {"model_id": CONTROL_MODEL_ID, "feature_pack": CONTROL_FEATURE_PACK},
        "challenger_strategy": {"model_id": CONTROL_MODEL_ID, "feature_pack": CHALLENGER_FEATURE_PACK},
        "gate_results": payload["gate_results"],
        "season_deltas": payload["season_deltas"],
        "metric_deltas": payload["metric_deltas"],
        "budget_deltas": payload["budget_deltas"],
        "aggregate_actual_points_delta": payload["aggregate_actual_points_delta"],
        "failed_gates": [name for name, passed in payload["gate_results"].items() if not bool(passed)],
        "validation_errors": validation_errors,
        "source_ebm_diagnostic_path": "data/08_reporting/ebm_diagnostics/ebm_diagnostic_started_at=20260511T004620197204Z",
    }
```

Implement `_load_required_artifacts`, `_candidate_signature_status`, `_fixture_identity_status`, `_child_context_errors`, `_primary_season_rows`, `_metric_rows`, `_metric_deltas`, and `_finite` by copying H004 and replacing constants/modes.

Decision status logic:

```python
def _decision_status(
    *,
    validation_errors: list[str],
    mechanism_audit_status: str,
    fixture_identity_status: str,
    gate_results: dict[str, bool],
    aggregate_delta: float,
) -> str:
    if validation_errors or mechanism_audit_status == "invalid":
        return "invalid"
    if mechanism_audit_status != "supports_reliability_hypothesis" or fixture_identity_status != "verified":
        return "diagnostic_only"
    if all(bool(value) for value in gate_results.values()):
        return "candidate_research_profile"
    weak_required = (
        aggregate_delta >= 40.0
        and bool(gate_results["weak_improved_seasons_pass"])
        and bool(gate_results["worst_season_delta_pass"])
        and bool(gate_results["recent_season_delta_pass"])
        and bool(gate_results["weak_concentration_pass"])
    )
    if weak_required:
        return "weak_positive_research_lead"
    inconclusive = (
        -20.0 <= aggregate_delta < 40.0
        and bool(gate_results["worst_season_delta_pass"])
        and bool(gate_results["recent_season_delta_pass"])
        and bool(gate_results["budget_integrity_pass"])
    )
    if inconclusive:
        return "inconclusive"
    return "rejected"
```

Gate payload must include:

```python
gate_results = {
    "aggregate_delta_pass": aggregate_delta >= 85.0,
    "improved_seasons_pass": improved_seasons >= 4,
    "weak_improved_seasons_pass": improved_seasons >= 3,
    "worst_season_delta_pass": worst_season_delta >= -20.0,
    "recent_season_delta_pass": season_2025_delta >= -10.0,
    "final_budget_pass": aggregate_final_budget_delta >= 0.0,
    "season_final_budget_pass": min_season_final_budget_delta >= -2.0,
    "budget_integrity_pass": min_season_final_budget_delta >= -2.0 and additional_budget_constrained_rounds <= 0,
    "top50_spearman_pass": top50_nonnegative_seasons >= 4,
    "selected_calibration_pass": selected_calibration_pass,
    "concentration_pass": concentration < 0.70,
    "weak_concentration_pass": concentration < 0.75,
}
```

- [ ] **Step 4: Add CLI**

Create `scripts/run_h005_feature_decision.py`:

```python
#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

from rich.console import Console

from cartola.backtesting.h005_feature_decision import write_h005_feature_decision


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the deterministic H005 feature decision artifact.")
    parser.add_argument("--experiment-path", required=True, type=Path)
    parser.add_argument("--audit-decision-path", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = write_h005_feature_decision(
        experiment_path=args.experiment_path,
        audit_decision_path=args.audit_decision_path,
    )
    Console().print(f"H005 feature decision written: {output_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run decision tests**

Run:

```bash
uv run --frozen pytest src/tests/backtesting/test_h005_feature_decision.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit Task 4**

```bash
git add src/cartola/backtesting/h005_feature_decision.py scripts/run_h005_feature_decision.py src/tests/backtesting/test_h005_feature_decision.py
git commit -m "feat: add h005 feature decision gates"
```

---

### Task 5: End-To-End Smoke And Documentation

**Files:**
- Modify: `AGENTS.md`
- Test: targeted pytest and repo checks

- [ ] **Step 1: Add workflow commands to `AGENTS.md`**

Add under the H004 research section:

```markdown
## H005 Count-Aware Matchup Reliability Research

- Source-anchored H005 mechanism audit:
  `uv run --frozen python scripts/run_h005_mechanism_audit.py --experiment-path data/08_reporting/experiments/model_feature/group=xgboost-sensitivity-v2__started_at=20260507T231127138806Z__matrix=f019652c883d --seasons 2021,2022,2023,2024,2025 --model-id xgboost_depth2_slow --feature-pack ppg_xg_matchup`
- H005 feature experiment:
  `uv run --frozen python scripts/run_model_experiments.py --group h005-count-aware-matchup-shrinkage --seasons 2021,2022,2023,2024,2025 --start-round 5 --budget 100 --current-year 2026 --jobs 12 --profile-runtime`
- H005 feature decision:
  `uv run --frozen python scripts/run_h005_feature_decision.py --experiment-path data/08_reporting/experiments/model_feature/<h005-experiment-id> --audit-decision-path data/08_reporting/hypotheses/<h005-audit-id>/h005_mechanism_audit_decision.json`
- H005 is research-only. Do not change live defaults from H005 unless a separate promotion protocol is explicitly approved.
```

- [ ] **Step 2: Run targeted test suite**

Run:

```bash
uv run --frozen pytest \
  src/tests/backtesting/test_features.py \
  src/tests/backtesting/test_experiment_config.py \
  src/tests/backtesting/test_h005_mechanism_audit.py \
  src/tests/backtesting/test_h005_feature_decision.py \
  -q
```

Expected: PASS.

- [ ] **Step 3: Run annotation gate for touched code**

Run:

```bash
uv run --frozen ruff check \
  src/cartola/backtesting/features.py \
  src/cartola/backtesting/config.py \
  src/cartola/backtesting/runner.py \
  src/cartola/backtesting/experiment_config.py \
  src/cartola/backtesting/h005_mechanism_audit.py \
  src/cartola/backtesting/h005_feature_decision.py \
  scripts/run_h005_mechanism_audit.py \
  scripts/run_h005_feature_decision.py \
  src/tests/backtesting/test_features.py \
  src/tests/backtesting/test_experiment_config.py \
  src/tests/backtesting/test_h005_mechanism_audit.py \
  src/tests/backtesting/test_h005_feature_decision.py \
  --select ANN
```

Expected: PASS.

- [ ] **Step 4: Run repository gate**

Run:

```bash
uv run --frozen scripts/pyrepo-check --all
```

Expected: PASS.

- [ ] **Step 5: Commit final docs**

```bash
git add AGENTS.md
git commit -m "docs: add h005 research workflow"
```

---

## Self-Review Checklist

- Spec coverage:
  - H005 feature columns: Task 1 and Task 2.
  - Position-normalized expected count: Task 2.
  - Opportunity denominator with zero-position rounds: Task 2.
  - Source-anchored Phase 0 audit: Task 3.
  - Raw-vs-ratio audit comparison: Task 3.
  - Experiment group and feature pack: Task 1.
  - Post-experiment decision statuses/gates: Task 4.
  - CLI commands and project guidance: Task 5.
- Placeholder scan:
  - No unresolved placeholder markers or unnamed implementation steps.
- Type consistency:
  - Feature augmentation mode is consistently `h005_matchup_reliability_v1`.
  - Feature pack is consistently `ppg_xg_matchup_h005`.
  - Group is consistently `h005-count-aware-matchup-shrinkage`.
  - H005 columns are consistently:
    - `h005_opponent_position_available_match_count_roll5`
    - `h005_opponent_position_expected_count_roll5`
    - `h005_opponent_position_count_ratio`
