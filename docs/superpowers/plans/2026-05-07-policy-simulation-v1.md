# Policy Simulation V1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an artifact-backed policy simulation runner for H001 opponent-overlap exposure, holding predictions/candidates fixed while replaying optimizer variants under moving-budget semantics.

**Architecture:** Add small policy definition and fixture/signature helpers first, then extend the optimizer with an optional policy-aware path that preserves current no-policy behavior. Build the simulation runner around persisted experiment artifacts, fail-closed validation, explicit diagnostic-only semantics for unverified fixture identity, and CSV/HTML report outputs.

**Tech Stack:** Python 3.13, pandas, PuLP/CBC, Rich CLI progress, Plotly offline HTML, pytest, Ruff, ty, Bandit, `uv`.

---

## File Structure

- Create `src/cartola/backtesting/optimizer_policies.py`
  - Owns policy definitions, policy-set lookup, fixture signatures, fixture coverage validation, duplicate normalization, and overlap counting helpers.
- Modify `src/cartola/backtesting/optimizer.py`
  - Adds optional policy-aware optimization while leaving existing `optimize_squad()` call sites unchanged.
- Create `src/cartola/backtesting/policy_simulation.py`
  - Owns artifact loading, source context validation, no-policy reproduction, moving-budget policy replay, summaries, comparability report, and HTML generation.
- Create `scripts/run_policy_simulation.py`
  - Thin CLI wrapper with dotenv bootstrap, Rich progress, filters, and output-path reporting.
- Create `src/tests/backtesting/test_optimizer_policies.py`
  - Unit tests for policy definitions, fixture signatures, coverage validation, duplicate normalization, overlap metrics, and hard-cap MILP behavior.
- Create `src/tests/backtesting/test_policy_simulation.py`
  - Unit/integration tests for artifact validation, no-policy reproduction, moving-budget replay, decision logic, output schemas, and fixture identity status.
- Create `src/tests/backtesting/test_run_policy_simulation_cli.py`
  - CLI parser, progress smoke, filter validation, and failure-mode tests.
- Modify `AGENTS.md`
  - Add command and workflow notes after implementation completes.
- Modify `roadmap.md`
  - Add delivered item only after implementation and real artifact acceptance pass.

## Task 1: Policy Definitions, Fixture Signatures, And Duplicate Normalization

**Files:**
- Create: `src/cartola/backtesting/optimizer_policies.py`
- Test: `src/tests/backtesting/test_optimizer_policies.py`

- [ ] **Step 1: Write failing tests for the frozen H001 policy set**

Add tests:

```python
from cartola.backtesting.optimizer_policies import get_policy_set


def test_opponent_overlap_v1_policy_set_is_frozen() -> None:
    policy_set = get_policy_set("opponent-overlap-v1")

    assert [policy.policy_variant for policy in policy_set.policies] == [
        "no_policy",
        "soft_overlap_penalty_low",
        "soft_overlap_penalty_medium",
        "hard_max_overlap_3",
        "hard_max_overlap_2",
    ]
    assert policy_set.policies[0].overlap_penalty == 0.0
    assert policy_set.policies[1].overlap_penalty == 0.15
    assert policy_set.policies[2].overlap_penalty == 0.35
    assert policy_set.policies[3].max_overlap_assets == 3
    assert policy_set.policies[4].max_overlap_assets == 2
```

- [ ] **Step 2: Run the policy-set test and verify it fails**

Run:

```bash
uv run --frozen pytest src/tests/backtesting/test_optimizer_policies.py::test_opponent_overlap_v1_policy_set_is_frozen -q
```

Expected: fail with `ModuleNotFoundError` or missing `get_policy_set`.

- [ ] **Step 3: Implement frozen policy definitions**

Create `optimizer_policies.py` with:

```python
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class OptimizerPolicy:
    policy_variant: str
    overlap_penalty: float = 0.0
    max_overlap_assets: int | None = None


@dataclass(frozen=True)
class OptimizerPolicySet:
    policy_set_id: str
    policies: tuple[OptimizerPolicy, ...]


NO_POLICY = OptimizerPolicy(policy_variant="no_policy")

_OPPONENT_OVERLAP_V1 = OptimizerPolicySet(
    policy_set_id="opponent-overlap-v1",
    policies=(
        NO_POLICY,
        OptimizerPolicy(policy_variant="soft_overlap_penalty_low", overlap_penalty=0.15),
        OptimizerPolicy(policy_variant="soft_overlap_penalty_medium", overlap_penalty=0.35),
        OptimizerPolicy(policy_variant="hard_max_overlap_3", max_overlap_assets=3),
        OptimizerPolicy(policy_variant="hard_max_overlap_2", max_overlap_assets=2),
    ),
)


def get_policy_set(policy_set_id: str) -> OptimizerPolicySet:
    if policy_set_id == _OPPONENT_OVERLAP_V1.policy_set_id:
        return _OPPONENT_OVERLAP_V1
    raise ValueError(f"Unknown policy set: {policy_set_id}")
```

- [ ] **Step 4: Run the policy-set test and verify it passes**

Run:

```bash
uv run --frozen pytest src/tests/backtesting/test_optimizer_policies.py::test_opponent_overlap_v1_policy_set_is_frozen -q
```

Expected: pass.

- [ ] **Step 5: Write failing tests for fixture signature and coverage validation**

Add tests:

```python
import pandas as pd
import pytest

from cartola.backtesting.optimizer_policies import (
    FixtureCoverageError,
    fixture_signature,
    validate_fixture_coverage,
)


def test_fixture_signature_is_order_stable() -> None:
    left = pd.DataFrame(
        [
            {"rodada": 2, "id_clube_home": 30, "id_clube_away": 40},
            {"rodada": 1, "id_clube_home": 10, "id_clube_away": 20},
        ]
    )
    right = left.iloc[[1, 0]].reset_index(drop=True)

    assert fixture_signature(left) == fixture_signature(right)


def test_fixture_coverage_rejects_duplicate_club_in_round() -> None:
    fixtures = pd.DataFrame(
        [
            {"rodada": 5, "id_clube_home": 1, "id_clube_away": 2},
            {"rodada": 5, "id_clube_home": 1, "id_clube_away": 3},
        ]
    )

    with pytest.raises(FixtureCoverageError, match="more than one fixture"):
        validate_fixture_coverage(fixtures, candidate_club_ids={1, 2, 3}, round_number=5)


def test_fixture_coverage_rejects_missing_candidate_club() -> None:
    fixtures = pd.DataFrame([{"rodada": 5, "id_clube_home": 1, "id_clube_away": 2}])

    with pytest.raises(FixtureCoverageError, match="missing fixture coverage"):
        validate_fixture_coverage(fixtures, candidate_club_ids={1, 2, 3}, round_number=5)
```

- [ ] **Step 6: Run fixture validation tests and verify they fail**

Run:

```bash
uv run --frozen pytest src/tests/backtesting/test_optimizer_policies.py -q
```

Expected: fail on missing fixture helpers.

- [ ] **Step 7: Implement fixture signature and coverage validation**

Add to `optimizer_policies.py`:

```python
import hashlib
import json
from typing import Iterable

import pandas as pd


class FixtureCoverageError(ValueError):
    pass


def fixture_signature(fixtures: pd.DataFrame) -> str:
    required = {"rodada", "id_clube_home", "id_clube_away"}
    missing = sorted(required - set(fixtures.columns))
    if missing:
        raise ValueError(f"Missing fixture signature columns: {', '.join(missing)}")
    records = (
        fixtures.loc[:, ["rodada", "id_clube_home", "id_clube_away"]]
        .astype({"rodada": int, "id_clube_home": int, "id_clube_away": int})
        .sort_values(["rodada", "id_clube_home", "id_clube_away"], kind="mergesort")
        .to_dict("records")
    )
    payload = json.dumps(records, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def validate_fixture_coverage(
    fixtures: pd.DataFrame,
    *,
    candidate_club_ids: Iterable[int],
    round_number: int,
) -> None:
    required = {"rodada", "id_clube_home", "id_clube_away"}
    missing = sorted(required - set(fixtures.columns))
    if missing:
        raise FixtureCoverageError(f"Missing fixture coverage columns: {', '.join(missing)}")
    round_fixtures = fixtures.loc[fixtures["rodada"].astype(int).eq(int(round_number))]
    club_counts: dict[int, int] = {}
    for _, row in round_fixtures.iterrows():
        for column in ("id_clube_home", "id_clube_away"):
            club_id = int(row[column])
            club_counts[club_id] = club_counts.get(club_id, 0) + 1
    duplicated = sorted(club_id for club_id, count in club_counts.items() if count > 1)
    if duplicated:
        raise FixtureCoverageError(
            f"Club appears in more than one fixture for round {round_number}: {duplicated}"
        )
    missing_clubs = sorted(int(club_id) for club_id in candidate_club_ids if int(club_id) not in club_counts)
    if missing_clubs:
        raise FixtureCoverageError(
            f"Round {round_number} has missing fixture coverage for candidate clubs: {missing_clubs}"
        )
```

- [ ] **Step 8: Write failing tests for duplicate candidate normalization**

Add tests:

```python
from cartola.backtesting.optimizer_policies import (
    DuplicateCandidateError,
    normalize_policy_candidates,
)


def test_normalize_policy_candidates_keeps_richest_equivalent_duplicate() -> None:
    rows = pd.DataFrame(
        [
            {
                "rodada": 5,
                "id_atleta": 10,
                "id_clube": 1,
                "posicao": "ata",
                "preco_pre_rodada": 8.0,
                "model_score": 4.0,
                "apelido": None,
            },
            {
                "rodada": 5,
                "id_atleta": 10,
                "id_clube": 1,
                "posicao": "ata",
                "preco_pre_rodada": 8.0,
                "model_score": 4.0,
                "apelido": "A10",
            },
        ]
    )

    normalized = normalize_policy_candidates(rows, score_column="model_score")

    assert len(normalized) == 1
    assert normalized.iloc[0]["apelido"] == "A10"


def test_normalize_policy_candidates_rejects_conflicting_duplicate() -> None:
    rows = pd.DataFrame(
        [
            {
                "rodada": 5,
                "id_atleta": 10,
                "id_clube": 1,
                "posicao": "ata",
                "preco_pre_rodada": 8.0,
                "model_score": 4.0,
            },
            {
                "rodada": 5,
                "id_atleta": 10,
                "id_clube": 1,
                "posicao": "ata",
                "preco_pre_rodada": 9.0,
                "model_score": 4.0,
            },
        ]
    )

    with pytest.raises(DuplicateCandidateError, match="Conflicting duplicate candidate"):
        normalize_policy_candidates(rows, score_column="model_score")
```

- [ ] **Step 9: Implement duplicate candidate normalization**

Add:

```python
class DuplicateCandidateError(ValueError):
    pass


def normalize_policy_candidates(candidates: pd.DataFrame, *, score_column: str) -> pd.DataFrame:
    critical = ["rodada", "id_atleta", "id_clube", "posicao", "preco_pre_rodada", score_column]
    missing = sorted(set(critical) - set(candidates.columns))
    if missing:
        raise DuplicateCandidateError(f"Missing duplicate-normalization columns: {', '.join(missing)}")
    kept_rows: list[pd.Series] = []
    for key, group in candidates.groupby(["rodada", "id_atleta"], dropna=False, sort=False):
        comparison = group.loc[:, critical].drop_duplicates()
        if len(comparison) > 1:
            raise DuplicateCandidateError(f"Conflicting duplicate candidate rows for {key}")
        richest_index = group.notna().sum(axis=1).sort_values(ascending=False, kind="mergesort").index[0]
        kept_rows.append(group.loc[richest_index])
    if not kept_rows:
        return candidates.iloc[0:0].copy()
    return (
        pd.DataFrame(kept_rows)
        .sort_values(["rodada", "id_atleta", "id_clube", "posicao"], kind="mergesort")
        .reset_index(drop=True)
    )
```

- [ ] **Step 10: Run Task 1 tests**

Run:

```bash
uv run --frozen pytest src/tests/backtesting/test_optimizer_policies.py -q
```

Expected: all Task 1 tests pass.

- [ ] **Step 11: Commit Task 1**

Run:

```bash
git add src/cartola/backtesting/optimizer_policies.py src/tests/backtesting/test_optimizer_policies.py
git commit -m "feat: add optimizer policy foundations"
```

## Task 2: Policy-Aware Optimizer MILP

**Files:**
- Modify: `src/cartola/backtesting/optimizer.py`
- Modify: `src/cartola/backtesting/optimizer_policies.py`
- Test: `src/tests/backtesting/test_optimizer_policies.py`
- Test: `src/tests/backtesting/test_optimizer.py`

- [ ] **Step 1: Write a failing hard-cap optimizer test**

Add a test that proves the MILP cannot cheat overlap variables:

```python
import pandas as pd

from cartola.backtesting.config import BacktestConfig
from cartola.backtesting.optimizer import optimize_squad
from cartola.backtesting.optimizer_policies import OptimizerPolicy


def test_hard_overlap_cap_forces_different_squad() -> None:
    candidates = pd.DataFrame(
        [
            {"id_atleta": 1, "apelido": "G1", "posicao": "gol", "preco_pre_rodada": 1.0, "score": 10.0, "id_clube": 1},
            {"id_atleta": 2, "apelido": "Z1", "posicao": "zag", "preco_pre_rodada": 1.0, "score": 9.0, "id_clube": 1},
            {"id_atleta": 3, "apelido": "Z2", "posicao": "zag", "preco_pre_rodada": 1.0, "score": 8.0, "id_clube": 2},
            {"id_atleta": 4, "apelido": "L1", "posicao": "lat", "preco_pre_rodada": 1.0, "score": 7.0, "id_clube": 2},
            {"id_atleta": 5, "apelido": "L2", "posicao": "lat", "preco_pre_rodada": 1.0, "score": 6.0, "id_clube": 3},
            {"id_atleta": 6, "apelido": "M1", "posicao": "mei", "preco_pre_rodada": 1.0, "score": 5.0, "id_clube": 3},
            {"id_atleta": 7, "apelido": "M2", "posicao": "mei", "preco_pre_rodada": 1.0, "score": 4.0, "id_clube": 4},
            {"id_atleta": 8, "apelido": "M3", "posicao": "mei", "preco_pre_rodada": 1.0, "score": 3.0, "id_clube": 4},
            {"id_atleta": 9, "apelido": "A1", "posicao": "ata", "preco_pre_rodada": 1.0, "score": 2.0, "id_clube": 5},
            {"id_atleta": 10, "apelido": "A2", "posicao": "ata", "preco_pre_rodada": 1.0, "score": 1.9, "id_clube": 5},
            {"id_atleta": 11, "apelido": "A3", "posicao": "ata", "preco_pre_rodada": 1.0, "score": 1.8, "id_clube": 6},
            {"id_atleta": 12, "apelido": "T1", "posicao": "tec", "preco_pre_rodada": 1.0, "score": 1.7, "id_clube": 6},
            {"id_atleta": 13, "apelido": "Alt", "posicao": "zag", "preco_pre_rodada": 1.0, "score": 0.1, "id_clube": 7},
        ]
    )
    fixtures = pd.DataFrame([{"rodada": 5, "id_clube_home": 1, "id_clube_away": 2}])
    config = BacktestConfig(season=2025, start_round=5, budget=100)

    no_policy = optimize_squad(candidates, "score", config, policy=None, fixtures_for_round=fixtures)
    capped = optimize_squad(
        candidates,
        "score",
        config,
        policy=OptimizerPolicy("hard_test", max_overlap_assets=2),
        fixtures_for_round=fixtures,
    )

    assert no_policy.status == "Optimal"
    assert capped.status == "Optimal"
    assert no_policy.selected["id_atleta"].astype(int).tolist() != capped.selected["id_atleta"].astype(int).tolist()
    assert int(capped.selected["id_clube"].isin([1, 2]).sum()) <= 2
```

- [ ] **Step 2: Run hard-cap test and verify it fails**

Run:

```bash
uv run --frozen pytest src/tests/backtesting/test_optimizer.py::test_hard_overlap_cap_forces_different_squad -q
```

Expected: fail because `optimize_squad()` does not accept policy arguments.

- [ ] **Step 3: Add optional policy parameters without changing existing callers**

Modify signatures in `optimizer.py`:

```python
from cartola.backtesting.optimizer_policies import OptimizerPolicy, NO_POLICY


def optimize_squad(
    candidates: pd.DataFrame,
    score_column: str,
    config: BacktestConfig,
    *,
    budget: float | None = None,
    policy: OptimizerPolicy | None = None,
    fixtures_for_round: pd.DataFrame | None = None,
) -> SquadOptimizationResult:
    active_policy = NO_POLICY if policy is None else policy
```

Pass `active_policy` and `fixtures_for_round` into `_optimize_formation()`.

- [ ] **Step 4: Implement policy MILP variables and constraints**

Inside `_optimize_formation()`, after selection/captain variables are created:

```python
policy_terms = _build_policy_terms(
    player_rows=player_rows,
    selected_variables=variables,
    policy=active_policy,
    fixtures_for_round=fixtures_for_round,
)
```

Then change the objective:

```python
problem += primary_objective - active_policy.overlap_penalty * policy_terms.overlap_asset_count
```

If `active_policy.max_overlap_assets is not None`:

```python
problem += policy_terms.overlap_asset_count <= active_policy.max_overlap_assets
```

Add helper objects:

```python
@dataclass(frozen=True)
class _PolicyTerms:
    overlap_asset_count: pulp.LpAffineExpression
    overlap_match_count: pulp.LpAffineExpression
```

Implement `_build_policy_terms()` using the exact constraints from the design spec.

- [ ] **Step 5: Preserve no-policy behavior**

If `policy.policy_variant == "no_policy"` or `fixtures_for_round is None`, `_build_policy_terms()` must return zero expressions and add no constraints:

```python
return _PolicyTerms(overlap_asset_count=pulp.lpSum([]), overlap_match_count=pulp.lpSum([]))
```

- [ ] **Step 6: Add overlap diagnostics to optimization result**

Extend `SquadOptimizationResult` with:

```python
opponent_overlap_asset_count: int = 0
opponent_overlap_match_count: int = 0
policy_variant: str = "no_policy"
```

Populate these values after solve by calculating overlap from the selected squad and fixtures using a pure helper. Do not rely on raw PuLP variable values for reporting when a no-policy run has no policy variables.

- [ ] **Step 7: Run optimizer tests**

Run:

```bash
uv run --frozen pytest src/tests/backtesting/test_optimizer.py src/tests/backtesting/test_optimizer_policies.py -q
```

Expected: pass.

- [ ] **Step 8: Run type and lint checks for touched modules**

Run:

```bash
uv run --frozen ruff check src/cartola/backtesting/optimizer.py src/cartola/backtesting/optimizer_policies.py src/tests/backtesting/test_optimizer.py src/tests/backtesting/test_optimizer_policies.py
uv run --frozen ty check
```

Expected: both pass.

- [ ] **Step 9: Commit Task 2**

Run:

```bash
git add src/cartola/backtesting/optimizer.py src/cartola/backtesting/optimizer_policies.py src/tests/backtesting/test_optimizer.py src/tests/backtesting/test_optimizer_policies.py
git commit -m "feat: add opponent overlap optimizer policy"
```

## Task 3: Source Artifact Validation And No-Policy Reproduction

**Files:**
- Create: `src/cartola/backtesting/policy_simulation.py`
- Test: `src/tests/backtesting/test_policy_simulation.py`

- [ ] **Step 1: Write failing tests for source validation**

Add tests that build a minimal source directory and assert:

```python
from pathlib import Path

import pandas as pd
import pytest

from cartola.backtesting.policy_simulation import PolicySimulationError, load_policy_source_context


def test_policy_source_rejects_fixed_budget(tmp_path: Path) -> None:
    child = tmp_path / "child"
    child.mkdir()
    (child / "run_metadata.json").write_text(
        '{"season": 2025, "budget_policy": "fixed", "scoring_contract_version": "cartola_standard_2026_v1"}',
        encoding="utf-8",
    )
    pd.DataFrame().to_csv(child / "player_predictions.csv", index=False)
    pd.DataFrame().to_csv(child / "round_results.csv", index=False)
    pd.DataFrame().to_csv(child / "selected_players.csv", index=False)

    with pytest.raises(PolicySimulationError, match="budget_policy=moving"):
        load_policy_source_context(child)
```

- [ ] **Step 2: Implement source validation skeleton**

In `policy_simulation.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json

import pandas as pd


class PolicySimulationError(ValueError):
    pass


@dataclass(frozen=True)
class PolicySourceContext:
    child_path: Path
    season: int
    model_id: str
    feature_pack: str
    fixture_mode: str
    matchup_context_mode: str
    budget_policy: str
    scoring_contract_version: str
    score_column: str


def load_policy_source_context(child_path: Path) -> PolicySourceContext:
    metadata_path = child_path / "run_metadata.json"
    if not metadata_path.exists():
        raise PolicySimulationError(f"Missing run metadata: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("budget_policy") != "moving":
        raise PolicySimulationError("Policy simulation requires budget_policy=moving source artifacts.")
    if metadata.get("scoring_contract_version") != "cartola_standard_2026_v1":
        raise PolicySimulationError("Policy simulation requires scoring_contract_version=cartola_standard_2026_v1.")
    return PolicySourceContext(
        child_path=child_path,
        season=int(metadata["season"]),
        model_id=str(metadata.get("primary_model_id", metadata.get("model_id", ""))),
        feature_pack=str(metadata.get("feature_pack", "")),
        fixture_mode=str(metadata.get("fixture_mode", "")),
        matchup_context_mode=str(metadata.get("matchup_context_mode", "none")),
        budget_policy=str(metadata["budget_policy"]),
        scoring_contract_version=str(metadata["scoring_contract_version"]),
        score_column=_score_column_from_metadata(metadata),
    )
```

- [ ] **Step 3: Add deterministic score-column mapping**

Add:

```python
def _score_column_from_metadata(metadata: dict[str, object]) -> str:
    strategy_roles = metadata.get("strategy_roles")
    model_id = str(metadata.get("primary_model_id", metadata.get("model_id", "")))
    if isinstance(strategy_roles, dict):
        for strategy, role in strategy_roles.items():
            if role == "primary_model":
                return f"{strategy}_score"
    if model_id:
        return f"{model_id}_score"
    raise PolicySimulationError("Cannot determine primary model score column from source metadata.")
```

If real metadata stores this differently, adjust after inspecting a real child and add a regression test for the observed shape.

- [ ] **Step 4: Write no-policy reproduction tests**

Use a synthetic child with one round where `player_predictions.csv`, `selected_players.csv`, and `round_results.csv` match an optimizer result. Test:

```python
from cartola.backtesting.policy_simulation import reproduce_no_policy_round


def test_no_policy_reproduces_selected_ids_and_captain(synthetic_policy_child: Path) -> None:
    result = reproduce_no_policy_round(synthetic_policy_child, round_number=5)

    assert result.status == "ok"
    assert result.selected_ids_match
    assert result.captain_id_match
    assert result.budget_used_delta == 0.0
```

- [ ] **Step 5: Implement no-policy reproduction**

Implement:

```python
@dataclass(frozen=True)
class NoPolicyReproductionResult:
    status: str
    selected_ids_match: bool
    captain_id_match: bool
    formation_match: bool
    budget_used_delta: float
    predicted_points_delta: float
    actual_points_delta: float
    failure_reason: str | None
```

Read artifact rows for one round, call `optimize_squad(..., policy=NO_POLICY)`, and compare:

```python
TOLERANCE = 1e-6
```

- [ ] **Step 6: Run Task 3 tests**

Run:

```bash
uv run --frozen pytest src/tests/backtesting/test_policy_simulation.py -q
```

Expected: source validation and no-policy reproduction tests pass.

- [ ] **Step 7: Commit Task 3**

Run:

```bash
git add src/cartola/backtesting/policy_simulation.py src/tests/backtesting/test_policy_simulation.py
git commit -m "feat: validate policy simulation sources"
```

## Task 4: Moving-Budget Policy Replay Engine

**Files:**
- Modify: `src/cartola/backtesting/policy_simulation.py`
- Test: `src/tests/backtesting/test_policy_simulation.py`

- [ ] **Step 1: Write moving-budget replay test**

Add a two-round synthetic test:

```python
from cartola.backtesting.optimizer_policies import get_policy_set
from cartola.backtesting.policy_simulation import run_policy_replay_for_child


def test_policy_replay_tracks_independent_moving_budget(synthetic_two_round_child: Path) -> None:
    policy_set = get_policy_set("opponent-overlap-v1")

    result = run_policy_replay_for_child(
        child_path=synthetic_two_round_child,
        policies=policy_set.policies[:2],
    )

    no_policy_rounds = [row for row in result.round_rows if row["policy_variant"] == "no_policy"]
    soft_rounds = [row for row in result.round_rows if row["policy_variant"] == "soft_overlap_penalty_low"]
    assert no_policy_rounds[0]["budget_after_round"] != soft_rounds[0]["budget_after_round"]
    assert no_policy_rounds[1]["budget_before_round"] == no_policy_rounds[0]["budget_after_round"]
    assert soft_rounds[1]["budget_before_round"] == soft_rounds[0]["budget_after_round"]
```

- [ ] **Step 2: Implement replay result container**

Add:

```python
@dataclass(frozen=True)
class PolicyReplayResult:
    round_rows: list[dict[str, object]]
    selected_player_rows: list[dict[str, object]]
    invalid_rows: list[dict[str, object]]
```

- [ ] **Step 3: Implement policy replay loop**

Implement:

```python
def run_policy_replay_for_child(
    *,
    child_path: Path,
    policies: tuple[OptimizerPolicy, ...],
) -> PolicyReplayResult:
    context = load_policy_source_context(child_path)
    predictions = _read_required_csv(child_path / "player_predictions.csv")
    current_budget_by_policy = {policy.policy_variant: _initial_budget(context, child_path) for policy in policies}
    round_rows: list[dict[str, object]] = []
    selected_rows: list[dict[str, object]] = []
    invalid_rows: list[dict[str, object]] = []
    for policy in policies:
        for round_number in sorted(predictions["rodada"].astype(int).unique()):
            round_candidates = _round_candidates(predictions, round_number, context.score_column)
            result = optimize_squad(
                round_candidates,
                context.score_column,
                _config_from_context(context, current_budget_by_policy[policy.policy_variant]),
                budget=current_budget_by_policy[policy.policy_variant],
                policy=policy,
                fixtures_for_round=_fixtures_for_round(context, round_number),
            )
            budget_update = _score_and_update_budget(result, current_budget_by_policy[policy.policy_variant])
            current_budget_by_policy[policy.policy_variant] = budget_update["budget_after_round"]
            round_rows.append(_round_result_row(context, policy, round_number, result, budget_update))
            selected_rows.extend(_selected_player_rows(context, policy, round_number, result))
    return PolicyReplayResult(round_rows=round_rows, selected_player_rows=selected_rows, invalid_rows=invalid_rows)
```

Keep helper functions private and focused.

- [ ] **Step 4: Implement scoring and budget update**

Add helper:

```python
def _score_and_update_budget(result: SquadOptimizationResult, budget_before: float) -> dict[str, float]:
    if result.status != "Optimal" or result.selected.empty:
        return {
            "budget_before_round": budget_before,
            "budget_used": 0.0,
            "budget_remaining": budget_before,
            "budget_delta": 0.0,
            "budget_after_round": budget_before,
            "actual_points_with_captain": 0.0,
        }
    selected = result.selected.copy()
    variacao = pd.to_numeric(selected["variacao"], errors="raise")
    if variacao.isna().any():
        raise PolicySimulationError("Selected assets must have finite variacao.")
    pontuacao = pd.to_numeric(selected["pontuacao"], errors="raise").fillna(0.0)
    captain_bonus = 0.5 * float(pontuacao.loc[selected["is_captain"]].iloc[0])
    actual = float(pontuacao.sum() + captain_bonus)
    delta = float(variacao.sum())
    used = float(pd.to_numeric(selected["preco_pre_rodada"], errors="raise").sum())
    return {
        "budget_before_round": budget_before,
        "budget_used": used,
        "budget_remaining": budget_before - used,
        "budget_delta": delta,
        "budget_after_round": budget_before + delta,
        "actual_points_with_captain": actual,
    }
```

- [ ] **Step 5: Run replay tests**

Run:

```bash
uv run --frozen pytest src/tests/backtesting/test_policy_simulation.py -q
```

Expected: pass.

- [ ] **Step 6: Commit Task 4**

Run:

```bash
git add src/cartola/backtesting/policy_simulation.py src/tests/backtesting/test_policy_simulation.py
git commit -m "feat: replay policy variants with moving budget"
```

## Task 5: Summaries, Decision Logic, And Reports

**Files:**
- Modify: `src/cartola/backtesting/policy_simulation.py`
- Test: `src/tests/backtesting/test_policy_simulation.py`

- [ ] **Step 1: Write decision logic tests**

Add tests:

```python
from cartola.backtesting.policy_simulation import decide_policy_variant


def test_decision_is_diagnostic_only_for_non_h001_seasons() -> None:
    decision = decide_policy_variant(
        selected_seasons=(2024, 2025),
        fixture_identity_status="verified",
        total_delta=50.0,
        improved_seasons=2,
        season_2025_delta=10.0,
        non_optimal_delta=0,
        final_budget_delta=0.0,
        min_budget_delta=0.0,
        max_drawdown_delta=0.0,
        top_two_concentration=0.1,
    )

    assert decision.status == "diagnostic_only"


def test_decision_rejects_2025_regression() -> None:
    decision = decide_policy_variant(
        selected_seasons=(2021, 2022, 2023, 2024, 2025),
        fixture_identity_status="verified",
        total_delta=100.0,
        improved_seasons=4,
        season_2025_delta=-30.0,
        non_optimal_delta=0,
        final_budget_delta=0.0,
        min_budget_delta=0.0,
        max_drawdown_delta=0.0,
        top_two_concentration=0.1,
    )

    assert decision.status == "rejected"
    assert "2025" in decision.reason
```

- [ ] **Step 2: Implement decision logic**

Add:

```python
@dataclass(frozen=True)
class PolicyDecision:
    status: str
    reason: str


def decide_policy_variant(
    *,
    selected_seasons: tuple[int, ...],
    fixture_identity_status: str,
    total_delta: float,
    improved_seasons: int,
    season_2025_delta: float | None,
    non_optimal_delta: int,
    final_budget_delta: float,
    min_budget_delta: float,
    max_drawdown_delta: float,
    top_two_concentration: float | None,
) -> PolicyDecision:
    if tuple(selected_seasons) != (2021, 2022, 2023, 2024, 2025):
        return PolicyDecision("diagnostic_only", "H001 generation 1 decisions require seasons 2021-2025.")
    if fixture_identity_status != "verified":
        return PolicyDecision("diagnostic_only", "Fixture identity is unverified.")
    if non_optimal_delta > 0:
        return PolicyDecision("ineligible", "Non-optimal solver rounds increased versus no_policy.")
    if total_delta <= 0:
        return PolicyDecision("rejected", "Total points did not improve versus no_policy.")
    if improved_seasons < 3:
        return PolicyDecision("rejected", "Improved fewer than 3 of 5 seasons.")
    if season_2025_delta is None or season_2025_delta < -25.0:
        return PolicyDecision("rejected", "2025 delta breached the -25 point threshold.")
    if final_budget_delta < -5.0 or min_budget_delta < -5.0 or max_drawdown_delta > 5.0:
        return PolicyDecision("rejected", "Budget path worsened beyond H001 thresholds.")
    if top_two_concentration is not None and top_two_concentration > 0.50:
        return PolicyDecision("rejected", "Top two rounds explain more than 50% of positive lift.")
    return PolicyDecision("candidate_policy", "All H001 generation 1 criteria passed.")
```

- [ ] **Step 3: Write output schema tests**

Test that `build_policy_outputs()` returns DataFrames with exact required columns for:

```text
policy_ranked_summary.csv
policy_per_season_summary.csv
policy_round_results.csv
policy_selected_players.csv
policy_profile_summary.csv
```

- [ ] **Step 4: Implement summary builders**

Implement functions:

```python
def build_policy_ranked_summary(round_results: pd.DataFrame) -> pd.DataFrame: ...
def build_policy_per_season_summary(round_results: pd.DataFrame) -> pd.DataFrame: ...
def build_policy_profile_summary(round_results: pd.DataFrame, selected_players: pd.DataFrame) -> pd.DataFrame: ...
```

Use `no_policy` as the benchmark inside each `(source_child_id, model_id, feature_pack)` group.

- [ ] **Step 5: Implement HTML report**

Add:

```python
def write_policy_simulation_report(
    *,
    output_path: Path,
    manifest: dict[str, object],
    ranked_summary: pd.DataFrame,
    per_season_summary: pd.DataFrame,
    profile_summary: pd.DataFrame,
    comparability_report: dict[str, object],
) -> None:
    ...
```

The HTML must include literal text:

```text
Policy Simulation V1
H001
research evidence only
diagnostic_only
```

- [ ] **Step 6: Run report tests**

Run:

```bash
uv run --frozen pytest src/tests/backtesting/test_policy_simulation.py -q
```

Expected: pass.

- [ ] **Step 7: Commit Task 5**

Run:

```bash
git add src/cartola/backtesting/policy_simulation.py src/tests/backtesting/test_policy_simulation.py
git commit -m "feat: summarize policy simulation results"
```

## Task 6: CLI With Rich Progress

**Files:**
- Create: `scripts/run_policy_simulation.py`
- Test: `src/tests/backtesting/test_run_policy_simulation_cli.py`

- [ ] **Step 1: Write CLI parser tests**

Add tests:

```python
from scripts.run_policy_simulation import parse_args


def test_parse_policy_simulation_args() -> None:
    args = parse_args(
        [
            "--experiment-path",
            "data/08_reporting/experiments/model_feature/example",
            "--hypothesis-id",
            "H001",
            "--policy-set",
            "opponent-overlap-v1",
            "--models",
            "xgboost_depth2_slow",
            "--feature-packs",
            "ppg_xg_matchup",
            "--current-year",
            "2026",
        ]
    )

    assert args.hypothesis_id == "H001"
    assert args.models == "xgboost_depth2_slow"
```

- [ ] **Step 2: Implement CLI skeleton**

Create:

```python
from __future__ import annotations

import argparse
from pathlib import Path

from rich.console import Console


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run optimizer policy simulation from experiment artifacts.")
    parser.add_argument("--experiment-path", type=Path, required=True)
    parser.add_argument("--hypothesis-id", required=True)
    parser.add_argument("--policy-set", required=True)
    parser.add_argument("--models", required=True)
    parser.add_argument("--feature-packs", required=True)
    parser.add_argument("--seasons", default="2021,2022,2023,2024,2025")
    parser.add_argument("--current-year", type=int, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("data/08_reporting/policy_simulations"))
    parser.add_argument("--allow-incomplete-report", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    from cartola.backtesting.policy_simulation import run_policy_simulation

    console = Console()
    result = run_policy_simulation(args=args, console=console)
    console.print(f"Policy simulation complete: output_path={result.output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 3: Implement `run_policy_simulation()` orchestration**

Add to `policy_simulation.py`:

```python
@dataclass(frozen=True)
class PolicySimulationRunResult:
    output_path: Path
    simulation_id: str
```

Function must:

1. select child runs by model, feature pack, and seasons;
2. load `opponent-overlap-v1`;
3. replay policies;
4. write all artifacts;
5. return output path.

- [ ] **Step 4: Add Rich progress**

Use Rich progress with fields:

```text
selected child count
current child
current policy variant
current season
current round
elapsed
output path
failure reason
```

The CLI must print at least:

```text
Policy simulation started
START child
DONE child
Policy simulation complete
```

- [ ] **Step 5: Run CLI tests**

Run:

```bash
uv run --frozen pytest src/tests/backtesting/test_run_policy_simulation_cli.py -q
```

Expected: pass.

- [ ] **Step 6: Commit Task 6**

Run:

```bash
git add scripts/run_policy_simulation.py src/cartola/backtesting/policy_simulation.py src/tests/backtesting/test_run_policy_simulation_cli.py
git commit -m "feat: add policy simulation cli"
```

## Task 7: End-To-End Smoke And Real Artifact Acceptance

**Files:**
- Modify: `src/tests/backtesting/test_policy_simulation.py`
- Modify: `AGENTS.md`
- Modify: `roadmap.md`

- [ ] **Step 1: Add synthetic end-to-end test**

Test command:

```bash
uv run --frozen pytest src/tests/backtesting/test_policy_simulation.py::test_policy_simulation_writes_required_artifacts -q
```

The test must assert these files exist:

```text
policy_simulation_manifest.json
policy_ranked_summary.csv
policy_per_season_summary.csv
policy_round_results.csv
policy_selected_players.csv
policy_profile_summary.csv
policy_comparability_report.json
policy_simulation_report.html
```

- [ ] **Step 2: Run targeted test suite**

Run:

```bash
uv run --frozen pytest \
  src/tests/backtesting/test_optimizer.py \
  src/tests/backtesting/test_optimizer_policies.py \
  src/tests/backtesting/test_policy_simulation.py \
  src/tests/backtesting/test_run_policy_simulation_cli.py \
  -q
```

Expected: pass.

- [ ] **Step 3: Run real artifact smoke**

Run:

```bash
uv run --frozen python scripts/run_policy_simulation.py \
  --experiment-path data/08_reporting/experiments/model_feature/group=xgboost-sensitivity-v2__started_at=20260505T211914592073Z__matrix=d95948374d75 \
  --hypothesis-id H001 \
  --policy-set opponent-overlap-v1 \
  --models xgboost_depth2_slow \
  --feature-packs ppg_xg_matchup \
  --seasons 2021,2022,2023,2024,2025 \
  --current-year 2026
```

Expected:

- command shows progress;
- all required artifacts are written;
- `no_policy` reproduction status is visible in `policy_comparability_report.json`;
- if fixture identity is unavailable, decisions are `diagnostic_only`, not `candidate_policy`.

- [ ] **Step 4: Inspect real artifact outputs**

Run:

```bash
latest=$(ls -td data/08_reporting/policy_simulations/policy_simulation_started_at=* | head -1)
uv run --frozen python - <<'PY'
from pathlib import Path
import json
import pandas as pd

latest = Path(sorted(Path("data/08_reporting/policy_simulations").glob("policy_simulation_started_at=*"))[-1])
ranked = pd.read_csv(latest / "policy_ranked_summary.csv")
comparability = json.loads((latest / "policy_comparability_report.json").read_text(encoding="utf-8"))
html = (latest / "policy_simulation_report.html").read_text(encoding="utf-8")

print("output", latest)
print("ranked_rows", len(ranked))
print("decision_statuses", sorted(ranked["decision_status"].astype(str).unique()))
print("comparability_status", comparability.get("status"))
print("fixture_identity_status", comparability.get("fixture_identity_status"))
print("html_has_h001", "H001" in html)
print("html_has_research_warning", "research evidence only" in html)
PY
```

Expected:

- `ranked_rows > 0`;
- no unexpected crash on real artifacts;
- HTML checks are `True`;
- decision statuses match fixture identity quality.

- [ ] **Step 5: Update `AGENTS.md`**

Add command:

```text
uv run --frozen python scripts/run_policy_simulation.py \
  --experiment-path data/08_reporting/experiments/model_feature/<experiment_id> \
  --hypothesis-id H001 \
  --policy-set opponent-overlap-v1 \
  --models xgboost_depth2_slow \
  --feature-packs ppg_xg_matchup \
  --current-year 2026
```

State:

- policy simulation is research-only;
- unverified fixture identity produces `diagnostic_only`;
- do not change live defaults from policy simulation without frozen validation.

- [ ] **Step 6: Update `roadmap.md`**

Add delivered item only after real artifact smoke passes:

```text
- Policy Simulation V1:
  - replays H001 opponent-overlap policy variants from persisted experiment artifacts;
  - keeps model predictions and candidate pools fixed;
  - uses independent moving-budget paths per policy;
  - writes policy summaries, selected players, comparability, and HTML reports;
  - treats unverified fixture identity as diagnostic_only.
```

- [ ] **Step 7: Run full quality gate**

Run:

```bash
uv run --frozen scripts/pyrepo-check --all
```

Expected:

- Ruff passes;
- ty passes;
- Bandit passes;
- pytest passes.

- [ ] **Step 8: Commit Task 7**

Run:

```bash
git add \
  src/tests/backtesting/test_policy_simulation.py \
  AGENTS.md \
  roadmap.md
git commit -m "docs: document policy simulation workflow"
```

If `AGENTS.md` contains unrelated local edits, inspect with `git diff -- AGENTS.md` and stage only the policy-simulation hunk.

## Task 8: Final Review And Handoff

**Files:**
- No required source changes.

- [ ] **Step 1: Verify final commit list**

Run:

```bash
git log --oneline --decorate -8
```

Expected: task commits appear in order.

- [ ] **Step 2: Verify working tree**

Run:

```bash
git status --short --branch
```

Expected: only intentional local files remain.

- [ ] **Step 3: Summarize implementation outcome**

Final summary must include:

- feature branch name;
- output path from real artifact smoke;
- full quality-gate result;
- whether real artifact run was `candidate_policy`, `diagnostic_only`, or failed;
- any performance caveat from solver runtime.

- [ ] **Step 4: Ask for merge/push decision**

Use the finishing branch workflow. Do not merge or push without the user's explicit instruction.

## Self-Review Checklist

- Spec coverage:
  - H001 hypothesis mapping: Task 1 and Task 5.
  - Fixture identity and coverage: Task 1 and Task 3.
  - Exact MILP linearization: Task 2.
  - Moving-budget replay: Task 4.
  - Artifact schemas and outputs: Task 5 and Task 7.
  - Rich progress CLI: Task 6.
  - Real artifact acceptance: Task 7.
- Placeholder scan:
  - No incomplete-marker terms or unbounded edge-case instructions are present.
- Type consistency:
  - `OptimizerPolicy`, `OptimizerPolicySet`, `PolicySourceContext`, `PolicyReplayResult`, and `PolicyDecision` are introduced before later tasks reference them.
- Implementation sequencing:
  - Validation and hashing precede optimizer changes.
  - No-policy reproduction precedes policy report generation.
  - Real artifact smoke precedes roadmap delivery claims.
