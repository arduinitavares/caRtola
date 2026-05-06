# Moving Budget Backtesting Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace fixed per-round Cartola backtesting with moving-budget semantics where each strategy's future budget changes according to selected-player `variacao`.

**Architecture:** Keep the optimizer as a single-round solver, and move budget state into the backtest runner. Run target rounds sequentially, update each strategy's budget after scoring, persist budget-path columns in artifacts, and mark all new runs with `budget_policy="moving"` so old fixed-budget evidence cannot be mixed with new results.

**V1 scope:** Historical backtests, historical replay paths that evaluate multiple rounds, model experiments, and tuning runs use moving-budget semantics. Live open-round recommendations and single-round recommendation replay remain one-round workflows: `--budget` means the caller's current available patrimonio, and there is no post-selection budget update until finalized replay data has historical `variacao`.

**Tech Stack:** Python 3.13, pandas, PuLP/CBC optimizer, existing Cartola backtesting runner, SQLite experiment index, pytest, Ruff, ty, Bandit via `uv run --frozen scripts/pyrepo-check --all`.

---

## Preflight

Work in the current development branch or a dedicated worktree. Do not edit generated historical experiment folders under `data/08_reporting/experiments/...`.

Run this before code changes to capture baseline state:

```bash
git status --short
uv run --frozen pytest src/tests/backtesting/test_runner.py::test_price_strategy_scores_market_open_price_not_post_round_price -q
```

Expected:

- working tree may already contain unrelated local edits;
- focused test passes before moving-budget changes.

## File Structure

- Create `src/cartola/backtesting/budgeting.py`
  - Own budget policy constants, budget state, budget update validation, summary helpers.
- Create `src/tests/backtesting/test_budgeting.py`
  - Unit-test budget state independent of model fitting and optimizer.
- Modify `src/cartola/backtesting/config.py`
  - Update budget comments/help text if needed. Budget policy constants live in `budgeting.py`.
- Modify `src/cartola/backtesting/runner.py`
  - Add metadata fields.
  - Add round result budget columns.
  - Replace normal target-round worker execution with sequential moving-budget replay.
  - Update each strategy budget after selected squad scoring.
- Modify `src/cartola/backtesting/metrics.py`
  - Add summary budget columns.
- Modify `src/cartola/backtesting/cli_output.py`
  - Display "Initial budget" / moving-budget policy where current run details are shown.
- Modify `src/cartola/backtesting/experiment_runner.py`
  - Carry new budget summary columns through per-season and ranked summaries.
  - Add budget policy to experiment metadata and tracker tags.
- Modify `src/cartola/backtesting/experiment_index.py`
  - Bump schema version and add `budget_policy` to experiments and child runs.
  - Migrate existing v1 rows as `fixed` so old rows are not mislabeled.
- Modify diagnostics that consume a budget scalar
  - Random-selection diagnostics must use each row's `budget_before_round`, not the initial budget.
- Modify tests:
  - `src/tests/backtesting/test_runner.py`
  - `src/tests/backtesting/test_metrics.py`
  - `src/tests/backtesting/test_cli_output.py`
  - `src/tests/backtesting/test_experiment_runner.py`
  - `src/tests/backtesting/test_experiment_index.py`
  - CLI tests that assert budget labels or metadata.
- Modify `roadmap.md`
  - State fixed-budget evidence is superseded and all incumbents/challengers need moving-budget reruns.

## Task 1: Budget State Primitive

**Files:**
- Create: `src/cartola/backtesting/budgeting.py`
- Create: `src/tests/backtesting/test_budgeting.py`

- [ ] **Step 1: Write failing budget primitive tests**

Create `src/tests/backtesting/test_budgeting.py`:

```python
import math

import pandas as pd
import pytest

from cartola.backtesting.budgeting import (
    BUDGET_CONSTRAINT_TOLERANCE,
    BUDGET_POLICY_FIXED,
    BUDGET_POLICY_MOVING,
    BudgetState,
    advance_budget,
    initial_budget_state,
    normalize_budget_policy,
)


def test_initial_budget_state_sets_current_peak_and_minimum() -> None:
    state = initial_budget_state(100.0)

    assert state.current_budget == 100.0
    assert state.peak_budget == 100.0
    assert state.min_budget == 100.0
    assert state.max_drawdown == 0.0


def test_advance_budget_sums_selected_variacao_including_tecnico() -> None:
    selected = pd.DataFrame(
        {
            "id_atleta": [1, 2, 3],
            "posicao": ["gol", "ata", "tec"],
            "variacao": [-1.5, 2.0, -0.2],
        }
    )
    state = initial_budget_state(100.0)

    update = advance_budget(state, selected, budget_used=96.0)

    assert update.budget_before_round == 100.0
    assert update.budget_delta == pytest.approx(0.3)
    assert update.budget_after_round == pytest.approx(100.3)
    assert update.budget_remaining == pytest.approx(4.0)
    assert update.next_state.current_budget == pytest.approx(100.3)
    assert update.next_state.peak_budget == pytest.approx(100.3)
    assert update.next_state.min_budget == pytest.approx(100.0)
    assert update.next_state.max_drawdown == pytest.approx(0.0)


def test_advance_budget_tracks_drawdown_from_peak() -> None:
    state = BudgetState(current_budget=105.0, peak_budget=110.0, min_budget=100.0, max_drawdown=2.0)
    selected = pd.DataFrame({"variacao": [-8.0]})

    update = advance_budget(state, selected, budget_used=90.0)

    assert update.budget_after_round == 97.0
    assert update.budget_peak == 110.0
    assert update.budget_drawdown == 13.0
    assert update.next_state.min_budget == 97.0
    assert update.next_state.max_drawdown == 13.0


def test_advance_budget_preserves_zero_or_negative_budget_path() -> None:
    selected = pd.DataFrame({"variacao": [-105.0]})

    update = advance_budget(initial_budget_state(100.0), selected, budget_used=99.0)

    assert update.budget_after_round == -5.0
    assert update.next_state.current_budget == -5.0
    assert update.next_state.min_budget == -5.0


def test_advance_budget_fails_when_selected_variacao_column_is_missing() -> None:
    selected = pd.DataFrame({"id_atleta": [1]})

    with pytest.raises(ValueError, match="Selected squad is missing required variacao column"):
        advance_budget(initial_budget_state(100.0), selected, budget_used=10.0)


@pytest.mark.parametrize("bad_value", [None, float("nan"), float("inf"), "bad"])
def test_advance_budget_fails_when_selected_variacao_is_invalid(bad_value: object) -> None:
    selected = pd.DataFrame({"id_atleta": [1], "variacao": [bad_value]})

    with pytest.raises(ValueError, match="Selected squad variacao must contain finite numeric values"):
        advance_budget(initial_budget_state(100.0), selected, budget_used=10.0)


def test_advance_budget_allows_empty_selected_squad_without_budget_change() -> None:
    selected = pd.DataFrame(columns=["variacao"])

    update = advance_budget(initial_budget_state(100.0), selected, budget_used=0.0)

    assert update.budget_before_round == 100.0
    assert update.budget_delta == 0.0
    assert update.budget_after_round == 100.0
    assert update.budget_remaining == 100.0
    assert update.is_budget_constrained is False


def test_budget_constrained_flag_uses_tolerance() -> None:
    selected = pd.DataFrame({"variacao": [0.0]})

    constrained = advance_budget(
        initial_budget_state(100.0),
        selected,
        budget_used=100.0 - (BUDGET_CONSTRAINT_TOLERANCE / 2),
    )
    not_constrained = advance_budget(
        initial_budget_state(100.0),
        selected,
        budget_used=100.0 - (BUDGET_CONSTRAINT_TOLERANCE * 2),
    )

    assert constrained.is_budget_constrained is True
    assert not_constrained.is_budget_constrained is False
    assert math.isclose(constrained.budget_remaining, BUDGET_CONSTRAINT_TOLERANCE / 2)


def test_normalize_budget_policy_treats_missing_as_fixed() -> None:
    assert normalize_budget_policy(None) == BUDGET_POLICY_FIXED
    assert normalize_budget_policy("") == BUDGET_POLICY_FIXED
    assert normalize_budget_policy(BUDGET_POLICY_MOVING) == BUDGET_POLICY_MOVING
```

- [ ] **Step 2: Run the failing tests**

```bash
uv run --frozen pytest src/tests/backtesting/test_budgeting.py -q
```

Expected: import failure because `cartola.backtesting.budgeting` does not exist.

- [ ] **Step 3: Implement the budget primitive**

Create `src/cartola/backtesting/budgeting.py`:

```python
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

BUDGET_POLICY_MOVING = "moving"
BUDGET_POLICY_FIXED = "fixed"
BUDGET_CONSTRAINT_TOLERANCE = 1e-6


@dataclass(frozen=True)
class BudgetState:
    current_budget: float
    peak_budget: float
    min_budget: float
    max_drawdown: float


@dataclass(frozen=True)
class BudgetRoundUpdate:
    budget_before_round: float
    budget_after_round: float
    budget_delta: float
    budget_remaining: float
    budget_peak: float
    budget_drawdown: float
    is_budget_constrained: bool
    next_state: BudgetState


def initial_budget_state(initial_budget: float) -> BudgetState:
    budget = float(initial_budget)
    if not np.isfinite(budget) or budget <= 0:
        raise ValueError("initial budget must be a positive finite value")
    return BudgetState(
        current_budget=budget,
        peak_budget=budget,
        min_budget=budget,
        max_drawdown=0.0,
    )


def normalize_budget_policy(value: object) -> str:
    if value is None:
        return BUDGET_POLICY_FIXED
    policy = str(value).strip()
    if not policy:
        return BUDGET_POLICY_FIXED
    if policy not in {BUDGET_POLICY_MOVING, BUDGET_POLICY_FIXED}:
        raise ValueError(f"Unknown budget policy: {policy}")
    return policy


def advance_budget(state: BudgetState, selected: pd.DataFrame, *, budget_used: float) -> BudgetRoundUpdate:
    budget_before = float(state.current_budget)
    used = float(budget_used)
    if not np.isfinite(used) or used < 0:
        raise ValueError("budget_used must be a non-negative finite value")

    budget_delta = _selected_variation_sum(selected)
    budget_after = budget_before + budget_delta
    budget_remaining = budget_before - used
    budget_peak = max(float(state.peak_budget), budget_after)
    min_budget = min(float(state.min_budget), budget_before, budget_after)
    budget_drawdown = budget_peak - budget_after
    max_drawdown = max(float(state.max_drawdown), budget_drawdown)
    next_state = BudgetState(
        current_budget=budget_after,
        peak_budget=budget_peak,
        min_budget=min_budget,
        max_drawdown=max_drawdown,
    )
    return BudgetRoundUpdate(
        budget_before_round=budget_before,
        budget_after_round=budget_after,
        budget_delta=budget_delta,
        budget_remaining=budget_remaining,
        budget_peak=budget_peak,
        budget_drawdown=budget_drawdown,
        is_budget_constrained=budget_remaining <= BUDGET_CONSTRAINT_TOLERANCE,
        next_state=next_state,
    )


def _selected_variation_sum(selected: pd.DataFrame) -> float:
    if selected.empty:
        return 0.0
    if "variacao" not in selected.columns:
        raise ValueError("Selected squad is missing required variacao column")
    try:
        values = pd.to_numeric(selected["variacao"], errors="raise").astype(float)
    except (TypeError, ValueError) as exc:
        raise ValueError("Selected squad variacao must contain finite numeric values") from exc
    if values.isna().any() or not np.isfinite(values).all():
        raise ValueError("Selected squad variacao must contain finite numeric values")
    return float(values.sum())
```

- [ ] **Step 4: Verify budget primitive tests pass**

```bash
uv run --frozen pytest src/tests/backtesting/test_budgeting.py -q
```

Expected: all tests pass.

## Task 2: Runner Round Results Use Moving Budget

**Files:**
- Modify: `src/cartola/backtesting/runner.py`
- Modify: `src/tests/backtesting/test_runner.py`

- [ ] **Step 1: Write a failing runner test for next-round purchasing power**

Append to `src/tests/backtesting/test_runner.py`:

```python
def test_moving_budget_uses_prior_round_variation_for_next_round(tmp_path: Path) -> None:
    season_df = pd.DataFrame(
        [
            # Training round before start_round.
            _player(1, 1, "gol", 10.0, 5.0, 0.0),
            _player(1, 2, "zag", 10.0, 5.0, 0.0),
            _player(1, 3, "zag", 10.0, 5.0, 0.0),
            _player(1, 4, "zag", 10.0, 5.0, 0.0),
            _player(1, 5, "mei", 10.0, 5.0, 0.0),
            _player(1, 6, "mei", 10.0, 5.0, 0.0),
            _player(1, 7, "mei", 10.0, 5.0, 0.0),
            _player(1, 8, "mei", 10.0, 5.0, 0.0),
            _player(1, 9, "ata", 10.0, 5.0, 0.0),
            _player(1, 10, "ata", 10.0, 5.0, 0.0),
            _player(1, 11, "ata", 10.0, 5.0, 0.0),
            _player(1, 12, "tec", 0.0, 1.0, 0.0),
            # Round 2 selected squad spends 99 and loses 20.
            _player(2, 1, "gol", 9.0, 5.0, -2.0),
            _player(2, 2, "zag", 9.0, 5.0, -2.0),
            _player(2, 3, "zag", 9.0, 5.0, -2.0),
            _player(2, 4, "zag", 9.0, 5.0, -2.0),
            _player(2, 5, "mei", 9.0, 5.0, -2.0),
            _player(2, 6, "mei", 9.0, 5.0, -2.0),
            _player(2, 7, "mei", 9.0, 5.0, -2.0),
            _player(2, 8, "mei", 9.0, 5.0, -2.0),
            _player(2, 9, "ata", 9.0, 5.0, -2.0),
            _player(2, 10, "ata", 9.0, 5.0, -2.0),
            _player(2, 11, "ata", 9.0, 5.0, 0.0),
            _player(2, 12, "tec", 0.0, 1.0, 0.0),
            # Round 3 has an expensive 99-cost formation and a cheaper 77-cost formation.
            _player(3, 1, "gol", 9.0, 5.0, 0.0),
            _player(3, 2, "zag", 9.0, 5.0, 0.0),
            _player(3, 3, "zag", 9.0, 5.0, 0.0),
            _player(3, 4, "zag", 9.0, 5.0, 0.0),
            _player(3, 5, "mei", 9.0, 5.0, 0.0),
            _player(3, 6, "mei", 9.0, 5.0, 0.0),
            _player(3, 7, "mei", 9.0, 5.0, 0.0),
            _player(3, 8, "mei", 9.0, 5.0, 0.0),
            _player(3, 9, "ata", 9.0, 5.0, 0.0),
            _player(3, 10, "ata", 9.0, 5.0, 0.0),
            _player(3, 11, "ata", 9.0, 5.0, 0.0),
            _player(3, 12, "tec", 0.0, 1.0, 0.0),
            _player(3, 101, "gol", 7.0, 1.0, 0.0),
            _player(3, 102, "zag", 7.0, 1.0, 0.0),
            _player(3, 103, "zag", 7.0, 1.0, 0.0),
            _player(3, 104, "zag", 7.0, 1.0, 0.0),
            _player(3, 105, "mei", 7.0, 1.0, 0.0),
            _player(3, 106, "mei", 7.0, 1.0, 0.0),
            _player(3, 107, "mei", 7.0, 1.0, 0.0),
            _player(3, 108, "mei", 7.0, 1.0, 0.0),
            _player(3, 109, "ata", 7.0, 1.0, 0.0),
            _player(3, 110, "ata", 7.0, 1.0, 0.0),
            _player(3, 111, "ata", 7.0, 1.0, 0.0),
            _player(3, 112, "tec", 0.0, 1.0, 0.0),
        ]
    )

    result = run_backtest(
        BacktestConfig(project_root=tmp_path, start_round=2, budget=100.0, jobs=12),
        season_df=season_df,
    )

    baseline_rounds = result.round_results[result.round_results["strategy"].eq("baseline")]
    round_2 = baseline_rounds[baseline_rounds["rodada"].eq(2)].iloc[0]
    round_3 = baseline_rounds[baseline_rounds["rodada"].eq(3)].iloc[0]

    assert round_2["budget_before_round"] == pytest.approx(100.0)
    assert round_2["solver_status"] == "Optimal"
    assert round_2["budget_used"] == pytest.approx(99.0)
    assert round_2["budget_delta"] == pytest.approx(-20.0)
    assert round_2["budget_after_round"] == pytest.approx(80.0)
    assert round_3["budget_before_round"] == pytest.approx(80.0)
    assert round_3["solver_status"] == "Optimal"
    assert round_3["budget_used"] <= 80.0
    assert round_3["budget_used"] == pytest.approx(77.0)
    round_3_selected = result.selected_players[
        result.selected_players["strategy"].eq("baseline")
        & result.selected_players["rodada"].eq(3)
    ]
    assert set(round_3_selected["id_atleta"].astype(int)) == set(range(101, 113))
    assert result.metadata.parallel_backend == "sequential_moving_budget"
    assert result.metadata.backtest_workers_effective == 1
```

Use the existing test helper shape. If `_player` does not exist with this signature, add this helper near existing runner test helpers:

```python
def _player(
    rodada: int,
    athlete_id: int,
    posicao: str,
    preco_pre_rodada: float,
    pontuacao: float,
    variacao: float,
) -> dict[str, object]:
    return {
        "rodada": rodada,
        "id_atleta": athlete_id,
        "apelido": f"Player {athlete_id}",
        "slug": f"player-{athlete_id}",
        "id_clube": 100 + athlete_id,
        "nome_clube": f"Club {athlete_id}",
        "posicao": posicao,
        "status": "Provavel",
        "preco": preco_pre_rodada + variacao,
        "preco_pre_rodada": preco_pre_rodada,
        "pontuacao": pontuacao,
        "media": pontuacao,
        "num_jogos": rodada,
        "variacao": variacao,
        "entrou_em_campo": True,
        "G": 0,
        "A": 0,
        "DS": 0,
        "SG": 0,
        "CA": 0,
        "FC": 0,
        "FS": 0,
        "FF": 0,
        "FD": 0,
        "FT": 0,
        "I": 0,
        "GS": 0,
        "DE": 0,
        "DP": 0,
        "V": 0,
        "CV": 0,
        "PP": 0,
        "PS": 0,
        "PC": 0,
        "GC": 0,
    }
```

- [ ] **Step 2: Run the failing runner test**

```bash
uv run --frozen pytest src/tests/backtesting/test_runner.py::test_moving_budget_uses_prior_round_variation_for_next_round -q
```

Expected: fails because the output has no moving-budget columns or because round 3 still uses fixed 100.

- [ ] **Step 3: Add metadata and round-result columns**

In `src/cartola/backtesting/runner.py`, import:

```python
from cartola.backtesting.budgeting import (
    BUDGET_POLICY_MOVING,
    BudgetState,
    advance_budget,
    initial_budget_state,
)
```

Add to `ROUND_RESULT_COLUMNS` after `selected_count`:

```python
    "budget_before_round",
    "budget_after_round",
    "budget_delta",
    "budget_remaining",
    "budget_peak",
    "budget_drawdown",
```

Add to `BacktestMetadata`:

```python
    budget_policy: str
    initial_budget: float
```

Set these in metadata construction:

```python
        budget_policy=BUDGET_POLICY_MOVING,
        initial_budget=float(config.budget),
```

Set moving-budget execution metadata:

```python
    backtest_workers_effective = 1 if target_rounds else 0
    parallel_backend = "sequential_moving_budget" if target_rounds else "none"
    model_n_jobs_effective = -1
```

If `config.jobs > 1`, append a warning:

```python
moving_budget_warnings = []
if config.jobs > 1:
    moving_budget_warnings.append("Target-round parallelism is disabled by moving-budget semantics.")
```

Merge that into metadata `warnings`.

- [ ] **Step 4: Thread budget state through sequential round evaluation**

Extend `RoundEvaluationResult`:

```python
    budget_states: dict[str, BudgetState] = field(default_factory=dict)
```

Replace the `_run_round_workers(...)` call in `_run_backtest()` with a sequential helper. Do not keep `skipped_results` outside this helper; every target round must flow through one ordered budget state machine so each strategy has continuous budget fields.

Create one ordered list:

```python
target_rounds = sorted(detected_target_rounds)
excluded_rounds = set(skipped_round_numbers)
```

`target_rounds` includes normal, skipped, empty, and evidence-excluded rounds. The helper decides how to emit each round.

```python
    round_results_for_targets = [
        *_run_rounds_with_moving_budget(
            config=config,
            target_rounds=target_rounds,
            round_frame_store=round_frame_store,
            empty_training_columns=empty_training_columns,
            model_feature_columns=model_feature_columns,
            model_n_jobs_effective=model_n_jobs_effective,
            primary_model_id=resolved_primary_model_id,
            model_params=model_params,
            excluded_rounds=excluded_rounds,
        ),
    ]
```

Add helper:

```python
def _run_rounds_with_moving_budget(
    *,
    config: BacktestConfig,
    target_rounds: list[int],
    round_frame_store: RoundFrameStore,
    empty_training_columns: list[str],
    model_feature_columns: list[str],
    model_n_jobs_effective: int,
    primary_model_id: ModelId,
    model_params: Mapping[str, object] | None = None,
    excluded_rounds: set[int] | None = None,
) -> list[RoundEvaluationResult]:
    budget_states = {
        strategy: initial_budget_state(config.budget)
        for strategy in _strategies(primary_model_id)
    }
    results: list[RoundEvaluationResult] = []
    for round_number in target_rounds:
        if excluded_rounds and round_number in excluded_rounds:
            result = _evaluate_skipped_target_round_with_budget_state(
                config=config,
                round_number=round_number,
                primary_model_id=primary_model_id,
                budget_states=budget_states,
            )
            budget_states = result.budget_states
            results.append(result)
            continue
        result = _evaluate_target_round(
            config=config,
            round_number=round_number,
            round_frame_store=round_frame_store,
            empty_training_columns=empty_training_columns,
            model_feature_columns=model_feature_columns,
            model_n_jobs_effective=model_n_jobs_effective,
            primary_model_id=primary_model_id,
            model_params=model_params,
            budget_states=budget_states,
        )
        budget_states = result.budget_states
        results.append(result)
    return results
```

If the existing runner currently returns `skipped_results, worker_rounds`, replace that split with `target_rounds, excluded_rounds`. A round excluded because fixture evidence or required source evidence is unavailable must still emit unchanged budget rows, but the run or child result must be marked promotion-ineligible through the existing skipped-round/comparability pathway.

Update `_evaluate_target_round(...)` signature:

```python
    budget_states: Mapping[str, BudgetState] | None = None,
```

Inside `_evaluate_target_round`, initialize if missing:

```python
    next_budget_states = dict(budget_states or {})
```

Before optimizing each strategy:

```python
        budget_state = next_budget_states.get(strategy, initial_budget_state(config.budget))
        round_config = replace(config, budget=budget_state.current_budget)
        result = optimize_squad(strategy_candidates, score_column="predicted_points", config=round_config)
```

After scoring:

```python
        budget_update = advance_budget(budget_state, result.selected, budget_used=result.budget_used)
        next_budget_states[strategy] = budget_update.next_state
```

Add to `round_rows.append(...)`:

```python
                "budget_before_round": budget_update.budget_before_round,
                "budget_after_round": budget_update.budget_after_round,
                "budget_delta": budget_update.budget_delta,
                "budget_remaining": budget_update.budget_remaining,
                "budget_peak": budget_update.budget_peak,
                "budget_drawdown": budget_update.budget_drawdown,
```

Return:

```python
        budget_states=next_budget_states,
```

Skipped/empty/infeasible rows must use unchanged budget state, not `pd.NA`:

```python
budget_update = advance_budget(budget_state, pd.DataFrame(columns=["variacao"]), budget_used=0.0)
```

Use the existing skipped/empty/infeasible solver status, but write `budget_before_round`, `budget_after_round`, `budget_delta=0`, `budget_remaining`, `budget_peak`, and `budget_drawdown`. These rows preserve the budget path and make skipped rounds auditable.

- [ ] **Step 5: Run the runner test**

```bash
uv run --frozen pytest src/tests/backtesting/test_runner.py::test_moving_budget_uses_prior_round_variation_for_next_round -q
```

Expected: pass.

## Task 3: Missing Variation Fails For Selected Assets

**Files:**
- Modify: `src/tests/backtesting/test_runner.py`

- [ ] **Step 1: Add failing runner tests for invalid selected variation**

Append:

```python
def test_moving_budget_fails_when_selected_asset_variacao_is_missing(tmp_path: Path) -> None:
    season_df = _minimal_two_round_feasible_season()
    season_df = season_df.drop(columns=["variacao"])

    with pytest.raises(ValueError, match="Selected squad is missing required variacao column"):
        run_backtest(
            BacktestConfig(project_root=tmp_path, start_round=2, budget=100.0),
            season_df=season_df,
        )


def test_moving_budget_fails_when_selected_asset_variacao_is_null(tmp_path: Path) -> None:
    season_df = _minimal_two_round_feasible_season()
    season_df.loc[season_df["rodada"].eq(2), "variacao"] = pd.NA

    with pytest.raises(ValueError, match="Selected squad variacao must contain finite numeric values"):
        run_backtest(
            BacktestConfig(project_root=tmp_path, start_round=2, budget=100.0),
            season_df=season_df,
        )
```

Add helper if absent:

```python
def _minimal_two_round_feasible_season() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for rodada in [1, 2]:
        for athlete_id, posicao in [
            (1, "gol"),
            (2, "zag"),
            (3, "zag"),
            (4, "zag"),
            (5, "mei"),
            (6, "mei"),
            (7, "mei"),
            (8, "mei"),
            (9, "ata"),
            (10, "ata"),
            (11, "ata"),
            (12, "tec"),
        ]:
            rows.append(_player(rodada, athlete_id, posicao, 8.0, 5.0, 0.0))
    return pd.DataFrame(rows)
```

- [ ] **Step 2: Run the tests**

```bash
uv run --frozen pytest src/tests/backtesting/test_runner.py \
  -k "moving_budget_fails_when_selected_asset_variacao" -q
```

Expected: pass if Task 2 uses `advance_budget()` directly.

## Task 4: Summary Budget Columns

**Files:**
- Modify: `src/cartola/backtesting/metrics.py`
- Modify: `src/tests/backtesting/test_metrics.py`

- [ ] **Step 1: Write failing summary test**

Append to `src/tests/backtesting/test_metrics.py`:

```python
def test_build_summary_includes_moving_budget_metrics() -> None:
    round_results = pd.DataFrame(
        [
            {
                "rodada": 5,
                "strategy": "ridge",
                "solver_status": "Optimal",
                "actual_points": 50.0,
                "predicted_points": 55.0,
                "budget_before_round": 100.0,
                "budget_after_round": 96.0,
                "budget_delta": -4.0,
                "budget_remaining": 0.0,
                "budget_drawdown": 4.0,
            },
            {
                "rodada": 6,
                "strategy": "ridge",
                "solver_status": "Optimal",
                "actual_points": 60.0,
                "predicted_points": 58.0,
                "budget_before_round": 96.0,
                "budget_after_round": 101.0,
                "budget_delta": 5.0,
                "budget_remaining": 2.0,
                "budget_drawdown": 0.0,
            },
            {
                "rodada": 7,
                "strategy": "ridge",
                "solver_status": "Infeasible",
                "actual_points": 0.0,
                "predicted_points": 0.0,
                "budget_before_round": 101.0,
                "budget_after_round": 101.0,
                "budget_delta": 0.0,
                "budget_remaining": 101.0,
                "budget_drawdown": 0.0,
            },
            {
                "rodada": 5,
                "strategy": "price",
                "solver_status": "Optimal",
                "actual_points": 40.0,
                "predicted_points": 40.0,
                "budget_before_round": 100.0,
                "budget_after_round": 99.0,
                "budget_delta": -1.0,
                "budget_remaining": 1.0,
                "budget_drawdown": 1.0,
            },
        ]
    )

    summary = build_summary(round_results, benchmark_strategy="price")
    ridge = summary[summary["strategy"].eq("ridge")].iloc[0]

    assert ridge["initial_budget"] == 100.0
    assert ridge["final_budget"] == 101.0
    assert ridge["total_budget_delta"] == 1.0
    assert ridge["min_budget"] == 96.0
    assert ridge["max_budget_drawdown"] == 4.0
    assert ridge["budget_constrained_rounds"] == 1
    assert ridge["rounds"] == 2


def test_build_summary_uses_missing_delta_when_benchmark_has_zero_optimal_rows() -> None:
    round_results = pd.DataFrame(
        [
            {
                "rodada": 5,
                "strategy": "ridge",
                "solver_status": "Optimal",
                "actual_points": 50.0,
                "predicted_points": 55.0,
                "budget_before_round": 100.0,
                "budget_after_round": 101.0,
                "budget_delta": 1.0,
                "budget_remaining": 2.0,
                "budget_drawdown": 0.0,
            },
            {
                "rodada": 5,
                "strategy": "price",
                "solver_status": "Infeasible",
                "actual_points": 0.0,
                "predicted_points": 0.0,
                "budget_before_round": 100.0,
                "budget_after_round": 100.0,
                "budget_delta": 0.0,
                "budget_remaining": 100.0,
                "budget_drawdown": 0.0,
            },
        ]
    )

    summary = build_summary(round_results, benchmark_strategy="price")

    assert summary["actual_points_delta_vs_price"].isna().all()
```

- [ ] **Step 2: Run the failing test**

```bash
uv run --frozen pytest src/tests/backtesting/test_metrics.py \
  -k "moving_budget_metrics or benchmark_has_zero_optimal_rows" -q
```

Expected: fails because summary columns do not exist or because an all-infeasible benchmark is treated as a real zero baseline.

- [ ] **Step 3: Add summary columns**

In `src/cartola/backtesting/metrics.py`, extend `SUMMARY_COLUMNS`:

```python
SUMMARY_COLUMNS: list[str] = [
    "strategy",
    "rounds",
    "total_actual_points",
    "average_actual_points",
    "total_predicted_points",
    "initial_budget",
    "final_budget",
    "total_budget_delta",
    "min_budget",
    "max_budget_drawdown",
    "budget_constrained_rounds",
]
```

Add helper:

```python
def _budget_summary(strategy_rounds: pd.DataFrame) -> dict[str, object]:
    # Budget path metrics use all target-round rows, not only optimal rows.
    ordered = strategy_rounds.sort_values("rodada", kind="mergesort")
    before = ordered["budget_before_round"].astype(float)
    after = ordered["budget_after_round"].astype(float)
    optimal = ordered[ordered["solver_status"].eq("Optimal")]
    remaining = optimal["budget_remaining"].astype(float)
    drawdown = ordered["budget_drawdown"].astype(float)
    return {
        "initial_budget": float(before.iloc[0]),
        "final_budget": float(after.iloc[-1]),
        "total_budget_delta": float(ordered["budget_delta"].astype(float).sum()),
        "min_budget": float(pd.concat([before, after], ignore_index=True).min()),
        "max_budget_drawdown": float(drawdown.max()),
        "budget_constrained_rounds": int((remaining <= 1e-6).sum()),
    }
```

Replace the groupby aggregation in `build_summary()` with explicit rows:

```python
    rows: list[dict[str, object]] = []
    for strategy, strategy_rounds in round_results.groupby("strategy", sort=False):
        optimal_strategy_rounds = strategy_rounds[strategy_rounds["solver_status"].eq("Optimal")]
        actual_points = optimal_strategy_rounds["actual_points"].astype(float)
        predicted_points = optimal_strategy_rounds["predicted_points"].astype(float)
        rows.append(
            {
                "strategy": strategy,
                "rounds": optimal_strategy_rounds["rodada"].nunique(),
                "total_actual_points": actual_points.sum(),
                "average_actual_points": actual_points.mean(),
                "total_predicted_points": predicted_points.sum(),
                **_budget_summary(strategy_rounds),
            }
        )

    summary = (
        pd.DataFrame(rows)
        .sort_values("total_actual_points", ascending=False)
        .reset_index(drop=True)
    )
```

Benchmark-delta behavior:

```python
    benchmark_rows = summary.loc[
        summary["strategy"].eq(benchmark_strategy) & summary["rounds"].gt(0),
        "total_actual_points",
    ]
    if benchmark_rows.empty:
        summary[delta_column] = pd.NA
    else:
        benchmark_total = float(benchmark_rows.iloc[0])
        summary[delta_column] = summary["total_actual_points"] - benchmark_total
```

This preserves the existing total-points delta convention: each strategy's total uses its own optimal rows, not only rounds where both strategy and benchmark are optimal. Solver-status comparability remains the promotion gate for skipped/infeasible asymmetry.

- [ ] **Step 4: Run metrics tests**

```bash
uv run --frozen pytest src/tests/backtesting/test_metrics.py -q
```

Expected: pass.

## Task 4.5: Budget-Aware Diagnostics

**Files:**
- Modify diagnostics builder module that currently receives `budget=config.budget`
- Modify the associated diagnostics tests

- [ ] **Step 1: Find diagnostics budget usage**

```bash
rg -n "build_diagnostics|random.*budget|budget=config\\.budget|config\\.budget" src/cartola src/tests/backtesting
```

Expected: identify any random-selection or benchmark diagnostics that still use the initial budget as if every round has the same budget.

- [ ] **Step 2: Add a failing diagnostics test**

Create or update a diagnostics test with two strategy/round rows:

```python
round_results = pd.DataFrame(
    [
        {"rodada": 5, "strategy": "ridge", "budget_before_round": 100.0, "budget_used": 99.0},
        {"rodada": 6, "strategy": "ridge", "budget_before_round": 80.0, "budget_used": 79.0},
    ]
)
```

Assert any random-selection or budget-feasibility diagnostic reads `budget_before_round` per row, not the initial budget `100.0` for both rounds. If the current diagnostics API cannot express this directly, add the smallest assertion on the serialized diagnostics output that proves round 6 used `80.0`.

- [ ] **Step 3: Update diagnostics implementation**

Pass the per-round budget path into diagnostics. Do not keep `budget=config.budget` for any diagnostic that simulates or evaluates round-level affordability. Use:

```python
round_budget = row["budget_before_round"]
```

or a mapping keyed by `(strategy, rodada)` if the diagnostic works from selected players rather than round rows.

- [ ] **Step 4: Run diagnostics tests**

```bash
uv run --frozen pytest src/tests/backtesting -k "diagnostic or random" -q
```

Expected: pass, and no diagnostics path retains fixed-budget assumptions inside a moving-budget run.

## Task 5: Metadata And CLI Output

**Files:**
- Modify: `src/cartola/backtesting/cli_output.py`
- Modify: `src/tests/backtesting/test_cli_output.py`
- Modify: `src/tests/backtesting/test_runner.py`

- [ ] **Step 1: Add metadata assertion to runner test**

Append or extend an existing metadata test in `src/tests/backtesting/test_runner.py`:

```python
def test_run_metadata_records_moving_budget_policy(tmp_path: Path) -> None:
    result = run_backtest(
        BacktestConfig(project_root=tmp_path, start_round=2, budget=100.0),
        season_df=_minimal_two_round_feasible_season(),
    )

    assert result.metadata.budget_policy == "moving"
    assert result.metadata.initial_budget == 100.0
```

- [ ] **Step 2: Add CLI output expectation**

In `src/tests/backtesting/test_cli_output.py`, add a test around the existing Rich summary helper:

```python
def test_cli_output_labels_budget_as_initial_budget() -> None:
    metadata = SimpleNamespace(
        season=2025,
        start_round=5,
        fixture_mode="none",
        footystats_mode="ppg",
        matchup_context_mode="none",
        backtest_jobs=12,
        backtest_workers_effective=1,
        parallel_backend="sequential_moving_budget",
        model_n_jobs_effective=-1,
        prediction_frames_built=38,
        wall_clock_seconds=1.2,
        scoring_contract_version="cartola_standard_2026_v1",
        budget_policy="moving",
        initial_budget=100.0,
        warnings=["Target-round parallelism is disabled by moving-budget semantics."],
    )

    rendered = _render_cli_output_for_test(metadata=metadata)

    assert "Budget policy" in rendered
    assert "moving" in rendered
    assert "Initial budget" in rendered
    assert "100.00" in rendered
```

If the test file uses a different rendering helper, follow that local pattern and assert the same labels.

- [ ] **Step 3: Update CLI output**

In `src/cartola/backtesting/cli_output.py`, add rows to the run details table:

```python
    table.add_row("Budget policy", _format_text(getattr(metadata, "budget_policy", None)))
    table.add_row("Initial budget", _format_float(getattr(metadata, "initial_budget", None)))
```

Make sure warnings from metadata still render.

- [ ] **Step 4: Run focused tests**

```bash
uv run --frozen pytest src/tests/backtesting/test_runner.py::test_run_metadata_records_moving_budget_policy src/tests/backtesting/test_cli_output.py -q
```

Expected: pass after adapting the CLI test to the existing helper style.

- [ ] **Step 5: Guard live recommendation semantics**

Do not add moving-budget replay state to `scripts/recommend_squad.py` or `RecommendationConfig` in this feature. Live open-round recommendations are one-round optimization calls; `--budget` means the user's current available patrimonio for that open round. If a live command writes metadata or console text, label the value as "Current budget" or "Available budget", not "Initial budget".

Single-round recommendation replay does not own a budget path. Multi-round historical replay can use moving-budget semantics only when it goes through the historical backtesting runner and finalized `variacao` exists. Add or update one CLI/recommendation test if current output labels make live/replay budget ambiguous.

## Task 6: Experiment Artifacts And Index Policy

**Files:**
- Modify: `src/cartola/backtesting/experiment_runner.py`
- Modify: `src/cartola/backtesting/experiment_index.py`
- Modify: `src/tests/backtesting/test_experiment_runner.py`
- Modify: `src/tests/backtesting/test_experiment_index.py`

- [ ] **Step 1: Add experiment index migration tests**

Append to `src/tests/backtesting/test_experiment_index.py`:

```python
def test_experiment_index_initializes_budget_policy_columns(tmp_path: Path) -> None:
    index = ExperimentIndex(tmp_path / "experiment_index.sqlite")

    index.initialize()

    with sqlite3.connect(index.path) as connection:
        experiment_columns = {row[1] for row in connection.execute("PRAGMA table_info(experiments)").fetchall()}
        child_columns = {row[1] for row in connection.execute("PRAGMA table_info(child_runs)").fetchall()}
        version = int(connection.execute("PRAGMA user_version").fetchone()[0])

    assert version >= 2
    assert "budget_policy" in experiment_columns
    assert "budget_policy" in child_columns


def test_experiment_index_migrates_existing_rows_as_fixed(tmp_path: Path) -> None:
    path = tmp_path / "experiment_index.sqlite"
    with sqlite3.connect(path) as connection:
        connection.execute("PRAGMA user_version = 1")
        connection.execute(
            'CREATE TABLE experiments (experiment_id TEXT PRIMARY KEY, "group" TEXT NOT NULL, started_at_utc TEXT NOT NULL, finished_at_utc TEXT, status TEXT NOT NULL, output_path TEXT NOT NULL, matrix_hash TEXT NOT NULL, seasons TEXT NOT NULL, start_round INTEGER NOT NULL, budget REAL NOT NULL, current_year INTEGER NOT NULL, jobs INTEGER NOT NULL, scoring_contract_version TEXT NOT NULL, git_commit TEXT, git_branch TEXT, git_dirty INTEGER NOT NULL, python_version TEXT NOT NULL, uv_lock_hash TEXT, mlflow_enabled INTEGER NOT NULL, mlflow_status TEXT NOT NULL, mlflow_parent_run_id TEXT, warning_count INTEGER NOT NULL, child_run_count INTEGER NOT NULL, completed_child_run_count INTEGER NOT NULL, failed_child_run_count INTEGER NOT NULL)'
        )
        connection.execute(
            "INSERT INTO experiments VALUES ('e1','g','s',NULL,'ok','out','m','2025',5,100,2026,1,'contract',NULL,NULL,0,'py',NULL,0,'disabled',NULL,0,0,0,0)"
        )
        connection.execute(
            "CREATE TABLE child_runs (experiment_id TEXT NOT NULL, child_run_id TEXT NOT NULL, season INTEGER NOT NULL, model_id TEXT NOT NULL, feature_pack TEXT NOT NULL, fixture_mode TEXT NOT NULL, footystats_mode TEXT NOT NULL, matchup_context_mode TEXT NOT NULL, output_path TEXT NOT NULL, status TEXT NOT NULL, wall_clock_seconds REAL, backtest_jobs INTEGER NOT NULL, backtest_workers_effective INTEGER, model_n_jobs_effective INTEGER, total_actual_points REAL, avg_actual_points REAL, total_predicted_points REAL, prediction_mae REAL, prediction_rmse REAL, prediction_r2 REAL, prediction_pearson REAL, prediction_spearman REAL, selected_calibration_slope REAL, top50_spearman REAL, optimal_round_count INTEGER, skipped_round_count INTEGER, candidate_pool_signature_hash TEXT, solver_status_signature_hash TEXT, comparability_partition TEXT NOT NULL, comparable_within_partition INTEGER NOT NULL, ineligibility_reason TEXT, source_hash_summary TEXT, mlflow_child_run_id TEXT, PRIMARY KEY (experiment_id, child_run_id))"
        )

    index = ExperimentIndex(path)
    index.initialize()

    with sqlite3.connect(path) as connection:
        policy = connection.execute("SELECT budget_policy FROM experiments WHERE experiment_id='e1'").fetchone()[0]

    assert policy == "fixed"


def test_missing_budget_policy_is_excluded_from_moving_budget_promotion_inputs(tmp_path: Path) -> None:
    output_path = tmp_path / "old_experiment"
    output_path.mkdir()
    (output_path / "experiment_metadata.json").write_text('{"experiment_id": "old"}')

    metadata = _load_experiment_metadata_for_promotion(output_path)

    assert metadata.budget_policy == "fixed"
    assert metadata.is_moving_budget_comparable is False
```

Adapt the last test to the repo's actual artifact reader or promotion/comparison helper. The required behavior is the same even if helper names differ: artifact/index readers call `normalize_budget_policy()`, missing `budget_policy` becomes `fixed`, and fixed rows cannot enter moving-budget promotion aggregation.

- [ ] **Step 2: Run failing index tests**

```bash
uv run --frozen pytest src/tests/backtesting/test_experiment_index.py \
  -k "budget_policy" -q
```

Expected: fails because schema is v1 and columns do not exist.

- [ ] **Step 3: Update experiment index schema**

In `src/cartola/backtesting/experiment_index.py`:

```python
SCHEMA_VERSION = 2
```

Add `"budget_policy"` after `"budget"` in `EXPERIMENT_COLUMNS`.

Add `"budget_policy"` after `"fixture_mode"` in `CHILD_RUN_COLUMNS`.

Import and use the central normalizer for all row reads/writes that may touch historical rows:

```python
from cartola.backtesting.budgeting import BUDGET_POLICY_FIXED, normalize_budget_policy
```

When materializing experiment or child-run rows from SQLite or artifact metadata, normalize before filtering:

```python
row["budget_policy"] = normalize_budget_policy(row.get("budget_policy"))
```

In `initialize()`, before `_create_schema(connection)`, call:

```python
            if user_version == 1:
                _migrate_v1_to_v2(connection)
```

Add:

```python
def _migrate_v1_to_v2(connection: sqlite3.Connection) -> None:
    experiment_columns = _table_columns(connection, "experiments")
    if experiment_columns and "budget_policy" not in experiment_columns:
        connection.execute("ALTER TABLE experiments ADD COLUMN budget_policy TEXT NOT NULL DEFAULT 'fixed'")
    child_columns = _table_columns(connection, "child_runs")
    if child_columns and "budget_policy" not in child_columns:
        connection.execute("ALTER TABLE child_runs ADD COLUMN budget_policy TEXT NOT NULL DEFAULT 'fixed'")


def _table_columns(connection: sqlite3.Connection, table: str) -> set[str]:
    return {str(row[1]) for row in connection.execute(f"PRAGMA table_info({table})").fetchall()}  # nosec B608
```

Update CREATE TABLE definitions with:

```sql
budget_policy TEXT NOT NULL,
```

for experiments and child runs.

- [ ] **Step 4: Add experiment runner metadata tests**

In `src/tests/backtesting/test_experiment_runner.py`, extend an existing success test:

```python
metadata = json.loads((output_path / "experiment_metadata.json").read_text())
assert metadata["budget_policy"] == "moving"
assert metadata["initial_budget"] == 100.0
```

Assert child index row includes `"budget_policy": "moving"` by inspecting the fake index row if tests already monkeypatch the index, or by reading SQLite if the test uses a real temp index.

- [ ] **Step 5: Update experiment runner rows**

In `src/cartola/backtesting/experiment_runner.py`, import:

```python
from cartola.backtesting.budgeting import BUDGET_POLICY_MOVING
```

Add `budget_policy=BUDGET_POLICY_MOVING` and `initial_budget=budget` to `_metadata(...)`.

Add `budget_policy` to experiment index row:

```python
            "budget_policy": BUDGET_POLICY_MOVING,
```

Add `budget_policy` to child index row:

```python
        "budget_policy": result.metadata.budget_policy,
```

In `_primary_summary_rows(...)`, carry through new summary columns because `row.to_dict()` already contains them. Ensure the returned row also includes `budget_policy`.

If a child has skipped or evidence-excluded target rounds under moving budget, mark it promotion-ineligible through the existing ineligibility/comparability fields. Budget carry-forward keeps artifacts coherent, but it does not make the child comparable:

```python
if skipped_round_count > 0:
    ineligibility_reason = _append_reason(ineligibility_reason, "skipped_or_evidence_excluded_rounds")
```

In `_rank_summary(...)`, aggregate moving-budget diagnostics:

```python
total_budget_delta=("total_budget_delta", "sum")
average_final_budget=("final_budget", "mean")
worst_min_budget=("min_budget", "min")
worst_max_budget_drawdown=("max_budget_drawdown", "max")
budget_constrained_rounds=("budget_constrained_rounds", "sum")
```

Use explicit row construction if the current helper is not an aggregation dict.

- [ ] **Step 6: Run experiment tests**

```bash
uv run --frozen pytest src/tests/backtesting/test_experiment_index.py src/tests/backtesting/test_experiment_runner.py -q
```

Expected: pass.

## Task 7: Refresh Tests Encoding Fixed-Budget Assumptions

**Files:**
- Modify tests under `src/tests/backtesting/` that fail after Tasks 1-6.

- [ ] **Step 1: Run runner and CLI test subsets**

```bash
uv run --frozen pytest src/tests/backtesting/test_runner.py src/tests/backtesting/test_cli.py src/tests/backtesting/test_cli_output.py -q
```

Expected: some tests may fail because they assumed constant per-round budget or exact metadata.

- [ ] **Step 2: Update tests to assert moving-budget contract**

For any test that expected repeated fixed budget:

- keep the test if it validates optimizer single-round behavior;
- update backtest-level assertions to use `budget_before_round`;
- assert `budget_before_round` equals initial budget only for the first evaluated round;
- assert `budget_after_round` equals next round's `budget_before_round` for the same strategy;
- assert `budget_policy == "moving"`.

For tests that only checked column sets, add required budget columns:

```python
{
    "budget_before_round",
    "budget_after_round",
    "budget_delta",
    "budget_remaining",
}
```

- [ ] **Step 3: Run all backtesting tests**

```bash
uv run --frozen pytest src/tests/backtesting -q
```

Expected: pass.

## Task 8: Real Smoke Runs

**Files:**
- No source edits unless smoke exposes a bug.

- [ ] **Step 1: Run a bounded real backtest smoke**

```bash
uv run --frozen python -m cartola.backtesting.cli \
  --season 2023 \
  --start-round 5 \
  --budget 100 \
  --fixture-mode none \
  --footystats-mode ppg_xg \
  --current-year 2026 \
  --jobs 12 \
  --output-root data/08_reporting/backtests/2023_moving_budget_smoke
```

Expected:

- command succeeds;
- run details show `Budget policy = moving`;
- run details show `Initial budget = 100.00`;
- metadata shows `parallel_backend = sequential_moving_budget`;
- `round_results.csv` contains moving-budget columns;
- `summary.csv` contains summary budget columns.

- [ ] **Step 2: Inspect generated artifact columns**

```bash
uv run --frozen python - <<'PY'
from pathlib import Path
import pandas as pd

root = Path("data/08_reporting/backtests/2023_moving_budget_smoke/2023")
round_results = pd.read_csv(root / "round_results.csv")
summary = pd.read_csv(root / "summary.csv")
print(round_results[["rodada", "strategy", "budget_before_round", "budget_delta", "budget_after_round", "budget_remaining"]].head(12).to_string(index=False))
print(summary[["strategy", "total_actual_points", "initial_budget", "final_budget", "total_budget_delta", "min_budget", "max_budget_drawdown", "budget_constrained_rounds"]].to_string(index=False))
PY
```

Expected:

- first evaluated round for each strategy starts at `100`;
- later rounds show strategy-specific budget paths;
- summary columns are non-null for optimal strategies.

## Task 9: Roadmap And Final Gate

**Files:**
- Modify: `roadmap.md`

- [ ] **Step 1: Update roadmap interpretation**

Edit `roadmap.md`:

```markdown
- Backtests now use moving-budget semantics. `--budget` is the initial patrimonio only.
- Old fixed-budget model/feature experiments are non-comparable and must be rerun before promotion decisions.
- Production-parity and matchup-research incumbents need fresh moving-budget generations over the selected season set.
- Promotion reruns remain on `2023,2024,2025` until `2021` and `2022` pass the same compatibility/comparability audit.
```

- [ ] **Step 2: Run full quality gate**

```bash
uv run --frozen scripts/pyrepo-check --all
```

Expected:

- Ruff passes;
- ty passes;
- Bandit passes;
- pytest passes.

- [ ] **Step 3: Record recommended rerun command**

Use this after the implementation is merged locally:

```bash
uv run --frozen python scripts/run_model_experiments.py \
  --group production-parity \
  --seasons 2023,2024,2025 \
  --start-round 5 \
  --budget 100 \
  --current-year 2026 \
  --jobs 12
```

Expected:

- output reports moving-budget artifacts;
- old 2023-2025 fixed-budget conclusions are treated as historical context only;
- 2021/2022 are not part of promotion until their audits pass and the season set is explicitly expanded.

## Self-Review

Spec coverage:

- Hard replacement: covered in Tasks 2, 7, and 9.
- `--budget` as initial budget: covered in Tasks 2, 4, 5, and 8.
- Per-strategy budget paths: covered in Tasks 1 and 2.
- Post-selection `variacao` updates: covered in Tasks 1, 2, and 3.
- Missing variation invalid: covered in Task 3.
- Sequential correctness over parallelism: covered in Tasks 2 and 5.
- Skipped/empty/infeasible budget continuity: covered in Task 2.
- Required selected-ID proof for reduced-budget selection: covered in Task 2.
- Benchmark delta semantics for zero-optimal benchmark: covered in Task 4.
- Diagnostics budget use: covered in Task 4.5.
- Live and single-round replay scope: covered in Task 5.
- Artifact columns: covered in Tasks 2, 4, 5, 6, and 8.
- Central budget-policy normalization: covered in Tasks 1 and 6.
- Evidence-excluded promotion ineligibility: covered in Tasks 2 and 6.
- Comparability and old fixed artifacts: covered in Tasks 6 and 9.
- Approved promotion season set remains `2023,2024,2025`: covered in Task 9.

Unresolved-marker scan:

- No unresolved TODO-style tasks remain.
- Every code-changing task includes exact target files, test command, and expected result.

Type consistency:

- `BudgetState`, `BudgetRoundUpdate`, `advance_budget`, and `initial_budget_state` are introduced before runner tasks use them.
- `budget_policy`, `initial_budget`, and moving-budget column names match the design spec.
