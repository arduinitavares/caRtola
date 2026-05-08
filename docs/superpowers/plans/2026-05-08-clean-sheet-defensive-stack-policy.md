# Clean-Sheet Defensive Stack Policy Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement and test H003 `clean-sheet-stack-v1`, a policy-simulation replay that adds a small optimizer bonus for selecting an eligible same-team `GOL + LAT/ZAG` defensive pair.

**Architecture:** The optimizer owns only the MILP policy mechanics and candidate-context validation. `optimizer_policies.py` owns the frozen policy set and shared policy data. `policy_simulation.py` owns artifact validation, replay output columns, changed-round summaries, and hypothesis-specific decision gates.

**Tech Stack:** Python 3.13, pandas, PuLP CBC MILP, pytest, uv, existing Cartola backtesting artifacts.

---

## Context

Primary spec:

```text
docs/superpowers/specs/2026-05-08-clean-sheet-defensive-stack-policy-design.md
```

Primary source run for the real acceptance command:

```text
data/08_reporting/experiments/model_feature/group=xgboost-sensitivity-v2__started_at=20260507T231127138806Z__matrix=f019652c883d
```

Required run command after implementation:

```bash
uv run --frozen python scripts/run_policy_simulation.py \
  --experiment-path data/08_reporting/experiments/model_feature/group=xgboost-sensitivity-v2__started_at=20260507T231127138806Z__matrix=f019652c883d \
  --hypothesis-id H003 \
  --policy-set clean-sheet-stack-v1 \
  --models xgboost_depth2_slow \
  --feature-packs ppg_xg_matchup \
  --seasons 2021,2022,2023,2024,2025 \
  --current-year 2026
```

This implementation must not change model training, feature generation, live recommendations, or experiment-ranking logic.

## File Map

- Modify `src/cartola/backtesting/optimizer_policies.py`
  - Add H003 policy fields to `OptimizerPolicy`.
  - Add frozen `clean-sheet-stack-v1` policy set.
  - Add optional policy helpers for clean-sheet context constants.

- Modify `src/cartola/backtesting/optimizer.py`
  - Add MILP variables and constraints for H003 pair bonus.
  - Keep H001/H002 fixture policy behavior unchanged.
  - Add result diagnostics for `clean_sheet_pair_count` and `clean_sheet_pair_bonus_applied`.

- Modify `src/cartola/backtesting/policy_simulation.py`
  - Validate H003 candidate-context columns.
  - Add H003 output columns.
  - Compute selected-ID changed rounds versus `no_policy`.
  - Add H003-specific decision gates.
  - Preserve existing H001/H002 behavior.

- Modify `src/tests/backtesting/test_optimizer_policies.py`
  - Freeze `clean-sheet-stack-v1` policy set.

- Modify `src/tests/backtesting/test_optimizer.py`
  - Test H003 bonus eligibility, ineligibility, cap, and linearization.

- Modify `src/tests/backtesting/test_policy_simulation.py`
  - Test H003 source column requirements, output schemas, changed-round metrics, and decision gates.

- Modify `src/tests/backtesting/test_run_policy_simulation_cli.py`
  - Smoke-test CLI parsing and artifact write path for `clean-sheet-stack-v1`.

- Modify `roadmap.md`
  - Mark H003 implementation as available after tests and the real replay command pass.

## Implementation Notes

- Do not make H003 require `fixtures_for_round` inside `optimize_squad()`. H003 derives eligible clubs from persisted candidate columns.
- Policy simulation still requires verified fixture identity before `candidate_policy` because the persisted H003 columns were fixture-derived.
- H003 context agreement tolerance is `1e-6`.
- Missing or conflicting H003 context columns must produce strict failure or invalid rows in incomplete mode. They must never become zero bonus silently.
- `no_policy` tie-break behavior must stay reproducible.

---

### Task 1: Freeze the H003 Policy Set

**Files:**
- Modify: `src/cartola/backtesting/optimizer_policies.py`
- Test: `src/tests/backtesting/test_optimizer_policies.py`

- [ ] **Step 1: Write the failing policy registry test**

Add this test after `test_gk_conflict_v1_policy_set_is_frozen`:

```python
def test_clean_sheet_stack_v1_policy_set_is_frozen() -> None:
    policy_set = get_policy_set("clean-sheet-stack-v1")

    assert [policy.policy_variant for policy in policy_set.policies] == [
        "no_policy",
        "home_cs_pair_bonus_025",
        "home_cs_pair_bonus_050",
        "home_cs_pair_bonus_075",
        "home_cs_pair_bonus_100",
    ]
    assert [policy.clean_sheet_pair_bonus for policy in policy_set.policies] == [
        0.0,
        0.25,
        0.50,
        0.75,
        1.00,
    ]
    for policy in policy_set.policies[1:]:
        assert policy.clean_sheet_pair_anchor_position == "gol"
        assert policy.clean_sheet_pair_partner_positions == ("lat", "zag")
        assert policy.clean_sheet_pair_min_ppg_diff == 0.75
        assert policy.clean_sheet_pair_min_xg_diff == 0.20
        assert policy.clean_sheet_pair_home_only is True
        assert policy.max_clean_sheet_pair_bonuses == 1
```

- [ ] **Step 2: Run the new test and verify it fails**

Run:

```bash
uv run --frozen pytest src/tests/backtesting/test_optimizer_policies.py::test_clean_sheet_stack_v1_policy_set_is_frozen -q
```

Expected: FAIL with `Unknown policy set: clean-sheet-stack-v1` or missing `OptimizerPolicy` attributes.

- [ ] **Step 3: Add H003 fields and policy set**

In `src/cartola/backtesting/optimizer_policies.py`, extend `OptimizerPolicy`:

```python
@dataclass(frozen=True)
class OptimizerPolicy:
    policy_variant: str
    overlap_penalty: float = 0.0
    max_overlap_assets: int | None = None
    gk_opponent_attack_penalty: float = 0.0
    gk_opponent_attack_positions: tuple[str, ...] = ()
    max_gk_opponent_attack_pairs: int | None = None
    gk_opponent_captain_penalty: float = 0.0
    gk_opponent_captain_positions: tuple[str, ...] = ()
    clean_sheet_pair_bonus: float = 0.0
    clean_sheet_pair_anchor_position: str = "gol"
    clean_sheet_pair_partner_positions: tuple[str, ...] = ("lat", "zag")
    clean_sheet_pair_min_ppg_diff: float = 0.75
    clean_sheet_pair_min_xg_diff: float = 0.20
    clean_sheet_pair_home_only: bool = True
    max_clean_sheet_pair_bonuses: int | None = None
```

Add the frozen policy set after `_GK_CONFLICT_V1`:

```python
_CLEAN_SHEET_STACK_V1 = OptimizerPolicySet(
    policy_set_id="clean-sheet-stack-v1",
    policies=(
        NO_POLICY,
        OptimizerPolicy(
            policy_variant="home_cs_pair_bonus_025",
            clean_sheet_pair_bonus=0.25,
            max_clean_sheet_pair_bonuses=1,
        ),
        OptimizerPolicy(
            policy_variant="home_cs_pair_bonus_050",
            clean_sheet_pair_bonus=0.50,
            max_clean_sheet_pair_bonuses=1,
        ),
        OptimizerPolicy(
            policy_variant="home_cs_pair_bonus_075",
            clean_sheet_pair_bonus=0.75,
            max_clean_sheet_pair_bonuses=1,
        ),
        OptimizerPolicy(
            policy_variant="home_cs_pair_bonus_100",
            clean_sheet_pair_bonus=1.00,
            max_clean_sheet_pair_bonuses=1,
        ),
    ),
)
```

Update `get_policy_set()`:

```python
def get_policy_set(policy_set_id: str) -> OptimizerPolicySet:
    if policy_set_id == _OPPONENT_OVERLAP_V1.policy_set_id:
        return _OPPONENT_OVERLAP_V1
    if policy_set_id == _GK_CONFLICT_V1.policy_set_id:
        return _GK_CONFLICT_V1
    if policy_set_id == _CLEAN_SHEET_STACK_V1.policy_set_id:
        return _CLEAN_SHEET_STACK_V1
    raise ValueError(f"Unknown policy set: {policy_set_id}")
```

- [ ] **Step 4: Run the registry tests**

Run:

```bash
uv run --frozen pytest src/tests/backtesting/test_optimizer_policies.py::test_clean_sheet_stack_v1_policy_set_is_frozen src/tests/backtesting/test_optimizer_policies.py::test_get_policy_set_rejects_unknown_policy_set_id -q
```

Expected: PASS.

- [ ] **Step 5: Commit Task 1**

If the working tree contains unrelated changes, stage only this task’s hunks with `git add -p`.

```bash
git add src/cartola/backtesting/optimizer_policies.py src/tests/backtesting/test_optimizer_policies.py
git commit -m "feat: add clean sheet stack policy set"
```

---

### Task 2: Add Optimizer Diagnostics Fields

**Files:**
- Modify: `src/cartola/backtesting/optimizer.py`
- Test: `src/tests/backtesting/test_optimizer.py`

- [ ] **Step 1: Write failing tests for empty/no-policy diagnostics**

Add these assertions to `test_optimizer_returns_empty_result_for_empty_candidates`:

```python
assert result.clean_sheet_pair_count == 0
assert result.clean_sheet_pair_bonus_applied == 0.0
```

Add these assertions to `test_no_policy_selection_is_unchanged_with_fixture_context`:

```python
assert baseline.clean_sheet_pair_count == 0
assert baseline.clean_sheet_pair_bonus_applied == 0.0
assert with_none_policy.clean_sheet_pair_count == 0
assert with_none_policy.clean_sheet_pair_bonus_applied == 0.0
assert with_no_policy.clean_sheet_pair_count == 0
assert with_no_policy.clean_sheet_pair_bonus_applied == 0.0
```

- [ ] **Step 2: Run the diagnostics tests and verify they fail**

Run:

```bash
uv run --frozen pytest \
  src/tests/backtesting/test_optimizer.py::test_optimizer_returns_empty_result_for_empty_candidates \
  src/tests/backtesting/test_optimizer.py::test_no_policy_selection_is_unchanged_with_fixture_context \
  -q
```

Expected: FAIL with missing `clean_sheet_pair_count` or `clean_sheet_pair_bonus_applied`.

- [ ] **Step 3: Add result fields and zero propagation**

In `SquadOptimizationResult`, add:

```python
    clean_sheet_pair_count: int = 0
    clean_sheet_pair_bonus_applied: float = 0.0
```

In `_with_formation_scores()`, propagate:

```python
        clean_sheet_pair_count=result.clean_sheet_pair_count,
        clean_sheet_pair_bonus_applied=result.clean_sheet_pair_bonus_applied,
```

No explicit change is needed in `_empty_result()` because dataclass defaults apply.

- [ ] **Step 4: Run diagnostics tests**

Run:

```bash
uv run --frozen pytest \
  src/tests/backtesting/test_optimizer.py::test_optimizer_returns_empty_result_for_empty_candidates \
  src/tests/backtesting/test_optimizer.py::test_no_policy_selection_is_unchanged_with_fixture_context \
  -q
```

Expected: PASS.

- [ ] **Step 5: Commit Task 2**

```bash
git add src/cartola/backtesting/optimizer.py src/tests/backtesting/test_optimizer.py
git commit -m "feat: expose clean sheet policy diagnostics"
```

---

### Task 3: Implement H003 MILP Pair Bonus

**Files:**
- Modify: `src/cartola/backtesting/optimizer.py`
- Test: `src/tests/backtesting/test_optimizer.py`

- [ ] **Step 1: Add synthetic H003 candidates helper**

Add this helper below `_gk_conflict_candidates()`:

```python
def _clean_sheet_stack_candidates(*, home: int = 1, ppg_diff: float = 0.80, xg_diff: float = 0.25) -> pd.DataFrame:
    rows = [
        _policy_row(1, "gol", 10.00, 100),
        _policy_row(2, "gol", 9.95, 200),
        _policy_row(3, "lat", 7.00, 300),
        _policy_row(4, "lat", 6.95, 100),
        _policy_row(5, "lat", 6.90, 400),
        _policy_row(6, "zag", 6.80, 500),
        _policy_row(7, "zag", 6.70, 600),
        _policy_row(8, "zag", 6.60, 700),
        _policy_row(9, "zag", 6.50, 800),
        _policy_row(10, "mei", 8.00, 900),
        _policy_row(11, "mei", 7.90, 901),
        _policy_row(12, "mei", 7.80, 902),
        _policy_row(13, "mei", 7.70, 903),
        _policy_row(14, "mei", 7.60, 904),
        _policy_row(15, "mei", 7.50, 905),
        _policy_row(16, "ata", 8.50, 906),
        _policy_row(17, "ata", 8.40, 907),
        _policy_row(18, "ata", 8.30, 908),
        _policy_row(19, "ata", 8.20, 909),
        _policy_row(20, "tec", 5.00, 910),
        _policy_row(21, "tec", 4.90, 911),
    ]
    candidates = pd.DataFrame(rows)
    candidates["matchup_is_home"] = 0
    candidates["footystats_ppg_diff"] = 0.0
    candidates["footystats_xg_diff"] = 0.0
    candidates.loc[candidates["id_clube"].eq(100), "matchup_is_home"] = home
    candidates.loc[candidates["id_clube"].eq(100), "footystats_ppg_diff"] = ppg_diff
    candidates.loc[candidates["id_clube"].eq(100), "footystats_xg_diff"] = xg_diff
    return candidates
```

- [ ] **Step 2: Write failing optimizer tests for H003 bonus and thresholds**

Add these tests:

```python
def test_clean_sheet_pair_bonus_can_select_eligible_goalkeeper_defender_pair_without_fixtures() -> None:
    candidates = _clean_sheet_stack_candidates()
    config = BacktestConfig(season=2025, start_round=5, budget=100)

    no_policy = optimize_squad(candidates, "score", config)
    stacked = optimize_squad(
        candidates,
        "score",
        config,
        policy=OptimizerPolicy(
            "clean_sheet_stack_test",
            clean_sheet_pair_bonus=0.25,
            max_clean_sheet_pair_bonuses=1,
        ),
    )

    assert no_policy.status == "Optimal"
    assert stacked.status == "Optimal"
    assert {1, 4}.issubset(set(stacked.selected["id_atleta"].astype(int)))
    assert {1, 4}.issubset(set(no_policy.selected["id_atleta"].astype(int))) is False
    assert stacked.clean_sheet_pair_count == 1
    assert stacked.clean_sheet_pair_bonus_applied == pytest.approx(0.25)
```

```python
@pytest.mark.parametrize(
    ("home", "ppg_diff", "xg_diff"),
    [
        (0, 0.80, 0.25),
        (1, 0.74, 0.25),
        (1, 0.80, 0.19),
    ],
)
def test_clean_sheet_pair_bonus_does_not_apply_when_context_is_ineligible(
    home: int,
    ppg_diff: float,
    xg_diff: float,
) -> None:
    candidates = _clean_sheet_stack_candidates(home=home, ppg_diff=ppg_diff, xg_diff=xg_diff)
    config = BacktestConfig(season=2025, start_round=5, budget=100)

    no_policy = optimize_squad(candidates, "score", config)
    stacked = optimize_squad(
        candidates,
        "score",
        config,
        policy=OptimizerPolicy(
            "clean_sheet_stack_test",
            clean_sheet_pair_bonus=1.00,
            max_clean_sheet_pair_bonuses=1,
        ),
    )

    assert stacked.status == "Optimal"
    assert stacked.clean_sheet_pair_count == 0
    assert stacked.clean_sheet_pair_bonus_applied == pytest.approx(0.0)
    assert stacked.selected["id_atleta"].astype(int).tolist() == no_policy.selected["id_atleta"].astype(int).tolist()
```

- [ ] **Step 3: Run the new optimizer tests and verify they fail**

Run:

```bash
uv run --frozen pytest \
  src/tests/backtesting/test_optimizer.py::test_clean_sheet_pair_bonus_can_select_eligible_goalkeeper_defender_pair_without_fixtures \
  src/tests/backtesting/test_optimizer.py::test_clean_sheet_pair_bonus_does_not_apply_when_context_is_ineligible \
  -q
```

Expected: FAIL because the policy has no MILP terms yet.

- [ ] **Step 4: Extend `_PolicyTerms` and zero terms**

In `src/cartola/backtesting/optimizer.py`, change `_PolicyTerms` to:

```python
@dataclass(frozen=True)
class _PolicyTerms:
    overlap_asset_count: pulp.LpAffineExpression
    overlap_match_count: pulp.LpAffineExpression
    gk_opponent_attack_pair_count: pulp.LpAffineExpression
    gk_opponent_captain_count: pulp.LpAffineExpression
    clean_sheet_pair_count: pulp.LpAffineExpression
```

Update `_zero_policy_terms()`:

```python
def _zero_policy_terms() -> _PolicyTerms:
    return _PolicyTerms(
        overlap_asset_count=pulp.lpSum([]),
        overlap_match_count=pulp.lpSum([]),
        gk_opponent_attack_pair_count=pulp.lpSum([]),
        gk_opponent_captain_count=pulp.lpSum([]),
        clean_sheet_pair_count=pulp.lpSum([]),
    )
```

- [ ] **Step 5: Add policy activity helpers**

Replace `_is_policy_active()` with helpers that do not force candidate-context policies to have fixtures:

```python
def _has_fixture_policy_terms(policy: OptimizerPolicy) -> bool:
    return (
        policy.overlap_penalty > 0.0
        or policy.max_overlap_assets is not None
        or policy.gk_opponent_attack_penalty > 0.0
        or policy.max_gk_opponent_attack_pairs is not None
        or policy.gk_opponent_captain_penalty > 0.0
    )


def _has_clean_sheet_pair_terms(policy: OptimizerPolicy) -> bool:
    return policy.clean_sheet_pair_bonus > 0.0 or policy.max_clean_sheet_pair_bonuses is not None
```

Update hard-cap checks in `_optimize_formation()`:

```python
    if active_policy.max_overlap_assets is not None and _has_fixture_policy_terms(active_policy):
        problem += policy_terms.overlap_asset_count <= active_policy.max_overlap_assets
    if active_policy.max_gk_opponent_attack_pairs is not None and _has_fixture_policy_terms(active_policy):
        problem += policy_terms.gk_opponent_attack_pair_count <= active_policy.max_gk_opponent_attack_pairs
    if active_policy.max_clean_sheet_pair_bonuses is not None and _has_clean_sheet_pair_terms(active_policy):
        problem += policy_terms.clean_sheet_pair_count <= active_policy.max_clean_sheet_pair_bonuses
```

- [ ] **Step 6: Add objective bonus and adjusted objective value**

Update `policy_objective`:

```python
    policy_objective = (
        primary_objective
        - active_policy.overlap_penalty * policy_terms.overlap_asset_count
        - active_policy.gk_opponent_attack_penalty * policy_terms.gk_opponent_attack_pair_count
        - active_policy.gk_opponent_captain_penalty * policy_terms.gk_opponent_captain_count
        + active_policy.clean_sheet_pair_bonus * policy_terms.clean_sheet_pair_count
    )
```

Update `_policy_adjusted_objective_value()`:

```python
def _policy_adjusted_objective_value(result: SquadOptimizationResult, active_policy: OptimizerPolicy) -> float:
    return (
        float(result.predicted_points_with_captain)
        - active_policy.overlap_penalty * float(result.opponent_overlap_asset_count)
        - active_policy.gk_opponent_attack_penalty * float(result.gk_opponent_attack_pair_count)
        - active_policy.gk_opponent_captain_penalty * float(result.gk_opponent_captain_count)
        + active_policy.clean_sheet_pair_bonus * float(result.clean_sheet_pair_count)
    )
```

- [ ] **Step 7: Build clean-sheet pair variables**

Add constants near `_FIXTURE_COLUMNS`:

```python
_CLEAN_SHEET_CONTEXT_COLUMNS = ("matchup_is_home", "footystats_ppg_diff", "footystats_xg_diff")
_CLEAN_SHEET_CONTEXT_TOLERANCE = 1e-6
```

Add helpers after `_gk_opponent_pair_variables()`:

```python
def _clean_sheet_pair_variables(
    *,
    problem: pulp.LpProblem,
    player_rows: pd.DataFrame,
    selected_variables: dict[int, pulp.LpVariable],
    policy: OptimizerPolicy,
    formation: dict[str, int],
) -> list[pulp.LpVariable]:
    if not _has_clean_sheet_pair_terms(policy):
        return []
    eligible_club_ids = _eligible_clean_sheet_pair_club_ids(player_rows, policy=policy)
    if not eligible_club_ids:
        return []

    club_ids = _whole_number_series(player_rows["id_clube"], "id_clube").astype(int)
    positions = player_rows["posicao"].astype(str)
    partner_position_set = set(policy.clean_sheet_pair_partner_positions)
    max_partner_count = max(
        1,
        sum(int(count) for position, count in formation.items() if position in partner_position_set),
    )
    pair_variables: list[pulp.LpVariable] = []
    for club_id in eligible_club_ids:
        gk_count = pulp.lpSum(
            selected_variables[index]
            for index in selected_variables
            if int(club_ids.loc[index]) == club_id and str(positions.loc[index]) == policy.clean_sheet_pair_anchor_position
        )
        partner_count = pulp.lpSum(
            selected_variables[index]
            for index in selected_variables
            if int(club_ids.loc[index]) == club_id and str(positions.loc[index]) in partner_position_set
        )
        gk_present = pulp.LpVariable(f"policy_clean_sheet_gk_present_{club_id}", cat=pulp.LpBinary)
        partner_present = pulp.LpVariable(f"policy_clean_sheet_partner_present_{club_id}", cat=pulp.LpBinary)
        pair_selected = pulp.LpVariable(f"policy_clean_sheet_pair_{club_id}", cat=pulp.LpBinary)
        problem += gk_count >= gk_present
        problem += gk_count <= gk_present
        problem += partner_count >= partner_present
        problem += partner_count <= max_partner_count * partner_present
        problem += pair_selected <= gk_present
        problem += pair_selected <= partner_present
        problem += pair_selected >= gk_present + partner_present - 1
        pair_variables.append(pair_selected)
    return pair_variables
```

Add context validation helper:

```python
def _eligible_clean_sheet_pair_club_ids(player_rows: pd.DataFrame, *, policy: OptimizerPolicy) -> list[int]:
    missing_columns = [
        column
        for column in ("id_clube", *_CLEAN_SHEET_CONTEXT_COLUMNS)
        if column not in player_rows.columns
    ]
    if missing_columns:
        raise ValueError(f"Missing clean-sheet policy candidate columns: {', '.join(missing_columns)}")

    rows = player_rows.copy()
    rows["id_clube"] = _whole_number_series(rows["id_clube"], "id_clube").astype(int)
    rows["matchup_is_home"] = _whole_number_series(rows["matchup_is_home"], "matchup_is_home").astype(int)
    for column in ("footystats_ppg_diff", "footystats_xg_diff"):
        rows[column] = _numeric_column(rows, column)

    eligible_club_ids: list[int] = []
    for club_id, club_rows in rows.groupby("id_clube", sort=True):
        home_values = sorted(club_rows["matchup_is_home"].astype(int).unique().tolist())
        if len(home_values) != 1:
            raise ValueError(f"Conflicting clean-sheet context for id_clube={int(club_id)}: matchup_is_home")
        ppg_span = float(club_rows["footystats_ppg_diff"].max() - club_rows["footystats_ppg_diff"].min())
        xg_span = float(club_rows["footystats_xg_diff"].max() - club_rows["footystats_xg_diff"].min())
        if ppg_span > _CLEAN_SHEET_CONTEXT_TOLERANCE:
            raise ValueError(f"Conflicting clean-sheet context for id_clube={int(club_id)}: footystats_ppg_diff")
        if xg_span > _CLEAN_SHEET_CONTEXT_TOLERANCE:
            raise ValueError(f"Conflicting clean-sheet context for id_clube={int(club_id)}: footystats_xg_diff")
        is_home = bool(home_values[0])
        ppg_diff = float(club_rows["footystats_ppg_diff"].iloc[0])
        xg_diff = float(club_rows["footystats_xg_diff"].iloc[0])
        if policy.clean_sheet_pair_home_only and not is_home:
            continue
        if ppg_diff < policy.clean_sheet_pair_min_ppg_diff:
            continue
        if xg_diff < policy.clean_sheet_pair_min_xg_diff:
            continue
        eligible_club_ids.append(int(club_id))
    return eligible_club_ids
```

- [ ] **Step 8: Attach clean-sheet variables inside `_build_policy_terms()`**

Refactor `_build_policy_terms()` so fixture terms and candidate-context terms are independent:

```python
def _build_policy_terms(
    *,
    problem: pulp.LpProblem,
    player_rows: pd.DataFrame,
    selected_variables: dict[int, pulp.LpVariable],
    captain_variables: dict[int, pulp.LpVariable],
    policy: OptimizerPolicy,
    fixtures_for_round: pd.DataFrame | None,
    formation: dict[str, int],
) -> _PolicyTerms:
    if policy.policy_variant == NO_POLICY.policy_variant:
        return _zero_policy_terms()

    clean_sheet_pair_variables = _clean_sheet_pair_variables(
        problem=problem,
        player_rows=player_rows,
        selected_variables=selected_variables,
        policy=policy,
        formation=formation,
    )

    if not _has_fixture_policy_terms(policy):
        return _PolicyTerms(
            overlap_asset_count=pulp.lpSum([]),
            overlap_match_count=pulp.lpSum([]),
            gk_opponent_attack_pair_count=pulp.lpSum([]),
            gk_opponent_captain_count=pulp.lpSum([]),
            clean_sheet_pair_count=pulp.lpSum(clean_sheet_pair_variables),
        )

    if "id_clube" not in player_rows.columns:
        raise ValueError("Missing optimizer policy candidate columns: id_clube")
    if fixtures_for_round is None:
        raise ValueError(f"Policy {policy.policy_variant!r} requires fixtures_for_round.")
```

Keep the existing fixture term body after that point. When returning existing fixture terms, include:

```python
        clean_sheet_pair_count=pulp.lpSum(clean_sheet_pair_variables),
```

Update the call site to pass `formation=formation` instead of `formation_size=sum(formation.values())`. Inside fixture overlap logic, define:

```python
    formation_size = sum(formation.values())
```

- [ ] **Step 9: Return clean-sheet diagnostics**

After solving and before `return SquadOptimizationResult(...)`, compute:

```python
    clean_sheet_pair_count = int(round(float(pulp.value(policy_terms.clean_sheet_pair_count) or 0.0)))
    clean_sheet_pair_bonus_applied = float(active_policy.clean_sheet_pair_bonus * clean_sheet_pair_count)
```

Add to the returned result:

```python
        clean_sheet_pair_count=clean_sheet_pair_count,
        clean_sheet_pair_bonus_applied=clean_sheet_pair_bonus_applied,
```

- [ ] **Step 10: Run H003 optimizer tests**

Run:

```bash
uv run --frozen pytest \
  src/tests/backtesting/test_optimizer.py::test_clean_sheet_pair_bonus_can_select_eligible_goalkeeper_defender_pair_without_fixtures \
  src/tests/backtesting/test_optimizer.py::test_clean_sheet_pair_bonus_does_not_apply_when_context_is_ineligible \
  -q
```

Expected: PASS.

- [ ] **Step 11: Run existing optimizer policy tests**

Run:

```bash
uv run --frozen pytest \
  src/tests/backtesting/test_optimizer.py::test_hard_overlap_cap_forces_different_squad \
  src/tests/backtesting/test_optimizer.py::test_gk_opponent_attack_soft_penalty_can_remove_conflicting_gk_pick \
  src/tests/backtesting/test_optimizer.py::test_gk_opponent_captain_penalty_targets_attacking_midfielder_captain_only \
  -q
```

Expected: PASS.

- [ ] **Step 12: Commit Task 3**

```bash
git add src/cartola/backtesting/optimizer.py src/tests/backtesting/test_optimizer.py
git commit -m "feat: add clean sheet pair optimizer bonus"
```

---

### Task 4: Add H003 Context Validation In Policy Replay

**Files:**
- Modify: `src/cartola/backtesting/policy_simulation.py`
- Test: `src/tests/backtesting/test_policy_simulation.py`

- [ ] **Step 1: Add H003 context columns to synthetic candidates**

In `_candidate_row()` in `src/tests/backtesting/test_policy_simulation.py`, add:

```python
        "matchup_is_home": 1,
        "footystats_ppg_diff": 0.80,
        "footystats_xg_diff": 0.25,
```

- [ ] **Step 2: Write failing replay validation test**

Add this test near the fixture-dependent replay tests:

```python
def test_policy_replay_requires_clean_sheet_context_columns_for_h003(tmp_path: Path) -> None:
    child = _write_two_round_policy_child(tmp_path)
    predictions_path = child / "player_predictions.csv"
    predictions = pd.read_csv(predictions_path).drop(columns=["footystats_xg_diff"])
    predictions.to_csv(predictions_path, index=False)
    policy = get_policy_set("clean-sheet-stack-v1").policies[1]

    with pytest.raises(PolicySimulationError, match="clean-sheet.*footystats_xg_diff"):
        run_policy_replay_for_child(child_path=child, policies=(policy,))
```

Add conflict test:

```python
def test_policy_replay_rejects_conflicting_clean_sheet_context_for_club_round(tmp_path: Path) -> None:
    child = _write_two_round_policy_child(tmp_path)
    predictions_path = child / "player_predictions.csv"
    predictions = pd.read_csv(predictions_path)
    club_mask = predictions["rodada"].eq(5) & predictions["id_clube"].eq(101)
    assert int(club_mask.sum()) == 1
    duplicate = predictions.loc[club_mask].copy()
    duplicate["id_atleta"] = 999001
    duplicate["footystats_ppg_diff"] = 0.95
    predictions = pd.concat([predictions, duplicate], ignore_index=True)
    predictions.to_csv(predictions_path, index=False)
    policy = get_policy_set("clean-sheet-stack-v1").policies[1]

    with pytest.raises(PolicySimulationError, match="Conflicting clean-sheet context"):
        run_policy_replay_for_child(child_path=child, policies=(policy,))
```

- [ ] **Step 3: Run tests and verify they fail**

Run:

```bash
uv run --frozen pytest \
  src/tests/backtesting/test_policy_simulation.py::test_policy_replay_requires_clean_sheet_context_columns_for_h003 \
  src/tests/backtesting/test_policy_simulation.py::test_policy_replay_rejects_conflicting_clean_sheet_context_for_club_round \
  -q
```

Expected: FAIL because H003 context validation is not wired through replay.

- [ ] **Step 4: Add H003 context helpers in policy simulation**

In `src/cartola/backtesting/policy_simulation.py`, add constants near `_PLAYER_PREDICTION_COLUMNS`:

```python
_CLEAN_SHEET_CONTEXT_COLUMNS: tuple[str, ...] = (
    "matchup_is_home",
    "footystats_ppg_diff",
    "footystats_xg_diff",
)
```

Add helpers near `_policy_requires_fixture_coverage()`:

```python
def _policy_requires_clean_sheet_context(policy: OptimizerPolicy) -> bool:
    return policy.clean_sheet_pair_bonus > 0.0 or policy.max_clean_sheet_pair_bonuses is not None


def _validate_clean_sheet_context_columns(candidates: pd.DataFrame, *, policy_variant: str, round_number: int) -> None:
    missing = [column for column in _CLEAN_SHEET_CONTEXT_COLUMNS if column not in candidates.columns]
    if missing:
        raise PolicySimulationError(
            "Missing clean-sheet policy candidate columns for "
            f"policy_variant={policy_variant!r} round={round_number}: {', '.join(missing)}"
        )
```

- [ ] **Step 5: Call validation before optimization**

In `run_policy_replay_for_child()`, after `normalized_candidates = normalize_policy_candidates(...)`, add:

```python
            if _policy_requires_clean_sheet_context(policy):
                _validate_clean_sheet_context_columns(
                    normalized_candidates,
                    policy_variant=policy.policy_variant,
                    round_number=round_number,
                )
```

The optimizer will enforce finite values and context agreement. The replay layer enforces the source-artifact contract and gives policy-specific error messages.

- [ ] **Step 6: Wrap optimizer context errors as replay errors**

Around `optimize_squad(...)`, keep the existing exception behavior. `ValueError` is already included in `_INCOMPLETE_REPLAY_ERRORS`, and strict mode raises it. No extra broad `except` is needed in `run_policy_replay_for_child()`.

- [ ] **Step 7: Run H003 context tests**

Run:

```bash
uv run --frozen pytest \
  src/tests/backtesting/test_policy_simulation.py::test_policy_replay_requires_clean_sheet_context_columns_for_h003 \
  src/tests/backtesting/test_policy_simulation.py::test_policy_replay_rejects_conflicting_clean_sheet_context_for_club_round \
  -q
```

Expected: PASS.

- [ ] **Step 8: Commit Task 4**

```bash
git add src/cartola/backtesting/policy_simulation.py src/tests/backtesting/test_policy_simulation.py
git commit -m "feat: validate clean sheet policy context"
```

---

### Task 5: Add H003 Replay Output Columns

**Files:**
- Modify: `src/cartola/backtesting/policy_simulation.py`
- Test: `src/tests/backtesting/test_policy_simulation.py`

- [ ] **Step 1: Update schema test expectation first**

In `test_policy_summary_output_schemas_are_stable`, change `POLICY_ROUND_RESULT_COLUMNS` expected tuple to include:

```python
        "predicted_points_with_captain",
        "actual_points_with_captain",
        "clean_sheet_pair_count",
        "clean_sheet_pair_bonus_applied",
        "selected_ids_changed_vs_no_policy",
```

Change `POLICY_RANKED_SUMMARY_COLUMNS` expected tuple to include these after `"rounds"`:

```python
        "changed_rounds",
        "changed_round_rate",
```

- [ ] **Step 2: Run schema test and verify it fails**

Run:

```bash
uv run --frozen pytest src/tests/backtesting/test_policy_simulation.py::test_policy_summary_output_schemas_are_stable -q
```

Expected: FAIL because constants are not updated.

- [ ] **Step 3: Add round output constants**

In `POLICY_ROUND_RESULT_COLUMNS`, append:

```python
    "clean_sheet_pair_count",
    "clean_sheet_pair_bonus_applied",
    "selected_ids_changed_vs_no_policy",
```

In `POLICY_RANKED_SUMMARY_COLUMNS`, add after `"rounds"`:

```python
    "changed_rounds",
    "changed_round_rate",
```

- [ ] **Step 4: Add row fields with default changed flag**

Change `_policy_replay_round_row()` signature:

```python
    clean_sheet_pair_count: int,
    clean_sheet_pair_bonus_applied: float,
    selected_ids_changed_vs_no_policy: bool | None = None,
```

Add to returned dict:

```python
        "clean_sheet_pair_count": int(clean_sheet_pair_count),
        "clean_sheet_pair_bonus_applied": float(clean_sheet_pair_bonus_applied),
        "selected_ids_changed_vs_no_policy": selected_ids_changed_vs_no_policy,
```

At the call site in `run_policy_replay_for_child()`, pass:

```python
                    clean_sheet_pair_count=result.clean_sheet_pair_count,
                    clean_sheet_pair_bonus_applied=result.clean_sheet_pair_bonus_applied,
```

- [ ] **Step 5: Add changed-round annotation helper**

Add helper after `_verify_no_policy_replay_coverage()`:

```python
def _annotate_selected_id_changes(
    round_rows: list[dict[str, object]],
    selected_player_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    if not round_rows:
        return round_rows
    if not selected_player_rows:
        annotated: list[dict[str, object]] = []
        for row in round_rows:
            changed = False if str(row["policy_variant"]) == NO_POLICY.policy_variant else pd.NA
            annotated.append({**row, "selected_ids_changed_vs_no_policy": changed})
        return annotated

    selected = pd.DataFrame(selected_player_rows)
    key_columns = ["season", "model_id", "feature_pack", "strategy", "rodada"]
    selected_sets: dict[tuple[object, ...], set[int]] = {}
    for key, group in selected.groupby([*key_columns, "policy_variant"], sort=False):
        season, model_id, feature_pack, strategy, rodada, policy_variant = cast(
            tuple[object, object, object, object, object, object],
            key,
        )
        selected_sets[(season, model_id, feature_pack, strategy, rodada, policy_variant)] = set(
            pd.to_numeric(group["id_atleta"], errors="coerce").dropna().astype(int).tolist()
        )

    annotated_rows: list[dict[str, object]] = []
    for row in round_rows:
        policy_variant = str(row["policy_variant"])
        if policy_variant == NO_POLICY.policy_variant:
            changed: object = False
        else:
            benchmark_key = tuple(row[column] for column in key_columns) + (NO_POLICY.policy_variant,)
            policy_key = tuple(row[column] for column in key_columns) + (policy_variant,)
            benchmark_ids = selected_sets.get(benchmark_key)
            policy_ids = selected_sets.get(policy_key)
            changed = pd.NA if benchmark_ids is None or policy_ids is None else policy_ids != benchmark_ids
        annotated_rows.append({**row, "selected_ids_changed_vs_no_policy": changed})
    return annotated_rows
```

In `run_policy_simulation()`, after replay returns:

```python
    round_rows = _annotate_selected_id_changes(round_rows, selected_player_rows)
```

In `_replay_policy_child()`, before returning successful `result`, annotate child-local rows too:

```python
        annotated_round_rows = _annotate_selected_id_changes(result.round_rows, result.selected_player_rows)
        result = PolicyReplayResult(
            round_rows=annotated_round_rows,
            selected_player_rows=result.selected_player_rows,
            invalid_rows=result.invalid_rows,
        )
```

- [ ] **Step 6: Add changed-round summary helper**

Add after `_top_two_positive_delta_concentration()`:

```python
def _changed_round_metrics(
    round_results: pd.DataFrame,
    *,
    model_id: str,
    feature_pack: str,
    strategy: str,
    policy_variant: str,
) -> tuple[int, float]:
    if policy_variant == NO_POLICY.policy_variant:
        return 0, 0.0
    context_mask = (
        round_results["model_id"].astype(str).eq(model_id)
        & round_results["feature_pack"].astype(str).eq(feature_pack)
        & round_results["strategy"].astype(str).eq(strategy)
        & round_results["policy_variant"].astype(str).eq(policy_variant)
        & round_results["solver_status"].astype(str).eq("Optimal")
    )
    policy_rounds = round_results.loc[context_mask]
    if policy_rounds.empty:
        return 0, 0.0
    changed = policy_rounds["selected_ids_changed_vs_no_policy"].astype("boolean")
    comparable = changed.notna()
    comparable_count = int(comparable.sum())
    if comparable_count == 0:
        return 0, 0.0
    changed_count = int(changed.loc[comparable].sum())
    return changed_count, float(changed_count / comparable_count)
```

Use it in `build_policy_ranked_summary()` before appending each row:

```python
        changed_rounds, changed_round_rate = _changed_round_metrics(
            round_results,
            model_id=str(model_id),
            feature_pack=str(feature_pack),
            strategy=str(strategy),
            policy_variant=str(policy_variant),
        )
```

Add row keys:

```python
                "changed_rounds": changed_rounds,
                "changed_round_rate": changed_round_rate,
```

- [ ] **Step 7: Update synthetic summary data**

In `_policy_summary_round_results()`, add these fields to each row:

```python
                        "clean_sheet_pair_count": 0,
                        "clean_sheet_pair_bonus_applied": 0.0,
                        "selected_ids_changed_vs_no_policy": policy_variant != "no_policy",
```

- [ ] **Step 8: Run schema and replay schema tests**

Run:

```bash
uv run --frozen pytest \
  src/tests/backtesting/test_policy_simulation.py::test_policy_summary_output_schemas_are_stable \
  src/tests/backtesting/test_policy_simulation.py::test_policy_replay_output_schemas_match_policy_contract \
  -q
```

Expected: PASS.

- [ ] **Step 9: Commit Task 5**

```bash
git add src/cartola/backtesting/policy_simulation.py src/tests/backtesting/test_policy_simulation.py
git commit -m "feat: report clean sheet policy replay metrics"
```

---

### Task 6: Add H003 Decision Gates

**Files:**
- Modify: `src/cartola/backtesting/policy_simulation.py`
- Test: `src/tests/backtesting/test_policy_simulation.py`

- [ ] **Step 1: Write failing decision tests**

Add tests near existing decision tests:

```python
def test_h003_decision_requires_practical_total_delta() -> None:
    decision = decide_policy_variant(
        hypothesis_id="H003",
        policy_set_id="clean-sheet-stack-v1",
        selected_seasons=(2021, 2022, 2023, 2024, 2025),
        fixture_identity_status="verified",
        total_delta=74.99,
        improved_seasons=5,
        season_2025_delta=0.0,
        season_deltas=(10.0, 20.0, 15.0, 20.0, 9.99),
        non_optimal_delta=0,
        final_budget_delta=0.0,
        min_budget_delta=0.0,
        max_drawdown_delta=0.0,
        top_two_concentration=0.25,
        changed_rounds=20,
        changed_round_rate=0.20,
    )

    assert decision.status == "rejected"
    assert "+75" in decision.reason
```

```python
def test_h003_decision_requires_changed_round_window() -> None:
    low_change = decide_policy_variant(
        hypothesis_id="H003",
        policy_set_id="clean-sheet-stack-v1",
        selected_seasons=(2021, 2022, 2023, 2024, 2025),
        fixture_identity_status="verified",
        total_delta=90.0,
        improved_seasons=5,
        season_2025_delta=1.0,
        season_deltas=(15.0, 15.0, 20.0, 20.0, 20.0),
        non_optimal_delta=0,
        final_budget_delta=0.0,
        min_budget_delta=0.0,
        max_drawdown_delta=0.0,
        top_two_concentration=0.25,
        changed_rounds=14,
        changed_round_rate=0.20,
    )
    high_change = decide_policy_variant(
        hypothesis_id="H003",
        policy_set_id="clean-sheet-stack-v1",
        selected_seasons=(2021, 2022, 2023, 2024, 2025),
        fixture_identity_status="verified",
        total_delta=90.0,
        improved_seasons=5,
        season_2025_delta=1.0,
        season_deltas=(15.0, 15.0, 20.0, 20.0, 20.0),
        non_optimal_delta=0,
        final_budget_delta=0.0,
        min_budget_delta=0.0,
        max_drawdown_delta=0.0,
        top_two_concentration=0.25,
        changed_rounds=50,
        changed_round_rate=0.41,
    )

    assert low_change.status == "rejected"
    assert "15" in low_change.reason
    assert high_change.status == "rejected"
    assert "40%" in high_change.reason
```

```python
def test_h003_decision_accepts_candidate_when_all_gates_pass() -> None:
    decision = decide_policy_variant(
        hypothesis_id="H003",
        policy_set_id="clean-sheet-stack-v1",
        selected_seasons=(2021, 2022, 2023, 2024, 2025),
        fixture_identity_status="verified",
        total_delta=90.0,
        improved_seasons=5,
        season_2025_delta=0.0,
        season_deltas=(20.0, 15.0, 20.0, 20.0, 15.0),
        non_optimal_delta=0,
        final_budget_delta=0.0,
        min_budget_delta=0.0,
        max_drawdown_delta=0.0,
        top_two_concentration=0.25,
        changed_rounds=20,
        changed_round_rate=0.20,
    )

    assert decision.status == "candidate_policy"
    assert "H003" in decision.reason
```

- [ ] **Step 2: Run decision tests and verify they fail**

Run:

```bash
uv run --frozen pytest \
  src/tests/backtesting/test_policy_simulation.py::test_h003_decision_requires_practical_total_delta \
  src/tests/backtesting/test_policy_simulation.py::test_h003_decision_requires_changed_round_window \
  src/tests/backtesting/test_policy_simulation.py::test_h003_decision_accepts_candidate_when_all_gates_pass \
  -q
```

Expected: FAIL because `decide_policy_variant()` does not accept H003 parameters.

- [ ] **Step 3: Extend decision signature**

Change `decide_policy_variant()` signature to:

```python
def decide_policy_variant(
    *,
    hypothesis_id: str = "H001",
    policy_set_id: str = "opponent-overlap-v1",
    selected_seasons: tuple[int, ...],
    fixture_identity_status: str,
    total_delta: float,
    improved_seasons: int,
    season_2025_delta: float | None,
    season_deltas: tuple[float, ...] = (),
    non_optimal_delta: int,
    final_budget_delta: float,
    min_budget_delta: float,
    max_drawdown_delta: float,
    top_two_concentration: float | None,
    changed_rounds: int = 0,
    changed_round_rate: float = 0.0,
) -> PolicyDecision:
```

Keep defaults so existing H001/H002 tests keep calling the function with old arguments until tests are updated.

- [ ] **Step 4: Implement shared preconditions**

At the top of `decide_policy_variant()`, keep:

```python
    if selected_seasons != _H001_SELECTED_SEASONS:
        return PolicyDecision(
            status="diagnostic_only",
            reason="generation 1 requires 2021-2025 selected seasons.",
        )
    if fixture_identity_status != "verified":
        return PolicyDecision(
            status="diagnostic_only",
            reason="fixture identity unverified for policy simulation.",
        )
    if non_optimal_delta > 0:
        return PolicyDecision(
            status="ineligible",
            reason="policy introduced non-optimal solver rounds versus no_policy.",
        )
    if final_budget_delta < -5.0 or min_budget_delta < -5.0 or max_drawdown_delta > 5.0:
        return PolicyDecision(status="rejected", reason="budget path delta fails the guardrail.")
    if top_two_concentration is not None and top_two_concentration > 0.50:
        return PolicyDecision(
            status="rejected",
            reason="top two rounds concentration is above the guardrail.",
        )
```

- [ ] **Step 5: Add H003 branch**

Add after shared preconditions:

```python
    if hypothesis_id == "H003" or policy_set_id == "clean-sheet-stack-v1":
        finite_season_deltas = [float(delta) for delta in season_deltas if np.isfinite(float(delta))]
        if total_delta < 75.0:
            return PolicyDecision(status="rejected", reason="H003 requires total delta of at least +75.")
        if improved_seasons < 3:
            return PolicyDecision(status="rejected", reason="H003 improved fewer than three seasons.")
        if not finite_season_deltas or float(np.median(finite_season_deltas)) <= 0.0:
            return PolicyDecision(status="rejected", reason="H003 median season delta is not positive.")
        if season_2025_delta is None or season_2025_delta < -15.0:
            return PolicyDecision(status="rejected", reason="H003 2025 delta fails the -15 guardrail.")
        if min(finite_season_deltas) < -25.0:
            return PolicyDecision(status="rejected", reason="H003 season delta fails the -25 guardrail.")
        if changed_rounds < 15:
            return PolicyDecision(status="rejected", reason="H003 changed fewer than 15 rounds.")
        if changed_round_rate > 0.40:
            return PolicyDecision(status="rejected", reason="H003 changed more than 40% of evaluated rounds.")
        if top_two_concentration is None:
            return PolicyDecision(status="rejected", reason="H003 has no positive round-delta concentration evidence.")
        return PolicyDecision(status="candidate_policy", reason="H003 clears clean-sheet-stack evidence guardrails.")
```

Leave existing H001/H002 positive-total branch after H003:

```python
    if total_delta <= 0:
        return PolicyDecision(status="rejected", reason="total delta versus no_policy is not positive.")
    if improved_seasons < 3:
        return PolicyDecision(status="rejected", reason="policy improved fewer than three seasons.")
    if season_2025_delta is None or season_2025_delta < -25.0:
        return PolicyDecision(status="rejected", reason="2025 delta fails the regression guardrail.")
    return PolicyDecision(status="candidate_policy", reason="policy clears generation 1 evidence guardrails.")
```

- [ ] **Step 6: Pass H003 decision inputs from ranked summary**

In `build_policy_ranked_summary()`, compute season deltas before calling the decision:

```python
            season_deltas = tuple(
                float(value)
                for value in pd.to_numeric(group["total_delta"], errors="coerce").dropna().tolist()
            )
```

Update the call:

```python
            decision = decide_policy_variant(
                hypothesis_id="H003" if str(policy_variant).startswith("home_cs_pair_bonus_") else "H001",
                policy_set_id="clean-sheet-stack-v1" if str(policy_variant).startswith("home_cs_pair_bonus_") else "opponent-overlap-v1",
                selected_seasons=selected_seasons,
                fixture_identity_status=fixture_identity_status,
                total_delta=total_delta_float,
                improved_seasons=improved_seasons,
                season_2025_delta=season_2025_delta,
                season_deltas=season_deltas,
                non_optimal_delta=non_optimal_delta_int,
                final_budget_delta=final_budget_delta_float,
                min_budget_delta=min_budget_delta_float,
                max_drawdown_delta=max_drawdown_delta_float,
                top_two_concentration=top_two_concentration,
                changed_rounds=changed_rounds,
                changed_round_rate=changed_round_rate,
            )
```

This variant-name mapping is acceptable for V1 because `policy_variant` values are frozen by `get_policy_set()`. A broader hypothesis registry is not part of H003.

- [ ] **Step 7: Run decision tests**

Run:

```bash
uv run --frozen pytest \
  src/tests/backtesting/test_policy_simulation.py::test_h003_decision_requires_practical_total_delta \
  src/tests/backtesting/test_policy_simulation.py::test_h003_decision_requires_changed_round_window \
  src/tests/backtesting/test_policy_simulation.py::test_h003_decision_accepts_candidate_when_all_gates_pass \
  src/tests/backtesting/test_policy_simulation.py::test_policy_decision_rejects_2025_regression \
  -q
```

Expected: PASS.

- [ ] **Step 8: Commit Task 6**

```bash
git add src/cartola/backtesting/policy_simulation.py src/tests/backtesting/test_policy_simulation.py
git commit -m "feat: add h003 policy decision gates"
```

---

### Task 7: Add CLI Smoke Coverage For H003

**Files:**
- Modify: `src/tests/backtesting/test_run_policy_simulation_cli.py`

- [ ] **Step 1: Add parse test for H003 policy set**

Add a test near existing parse tests:

```python
def test_parse_args_accepts_clean_sheet_stack_policy_set() -> None:
    args = parse_args(
        [
            "--experiment-path",
            "data/08_reporting/experiments/model_feature/example",
            "--hypothesis-id",
            "H003",
            "--policy-set",
            "clean-sheet-stack-v1",
            "--models",
            "xgboost_depth2_slow",
            "--feature-packs",
            "ppg_xg_matchup",
            "--seasons",
            "2021,2022,2023,2024,2025",
            "--current-year",
            "2026",
        ]
    )

    assert args.hypothesis_id == "H003"
    assert args.policy_set == "clean-sheet-stack-v1"
```

- [ ] **Step 2: Run new parse test**

Run:

```bash
uv run --frozen pytest src/tests/backtesting/test_run_policy_simulation_cli.py::test_parse_args_accepts_clean_sheet_stack_policy_set -q
```

Expected: PASS if CLI accepts policy-set strings generically. If it fails due `choices`, add `"clean-sheet-stack-v1"` to the parser choices in `scripts/run_policy_simulation.py`, then rerun.

- [ ] **Step 3: Update artifact smoke test if it asserts policy-set literals**

If `test_run_policy_simulation_cli.py` has an artifact smoke test that hard-codes only H001 variants, add a second smoke with `policy_set="clean-sheet-stack-v1"` and assert:

```python
ranked_summary = pd.read_csv(output_path / "policy_ranked_summary.csv")
assert "home_cs_pair_bonus_025" in set(ranked_summary["policy_variant"].astype(str))
```

- [ ] **Step 4: Run CLI test file**

Run:

```bash
uv run --frozen pytest src/tests/backtesting/test_run_policy_simulation_cli.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit Task 7**

```bash
git add scripts/run_policy_simulation.py src/tests/backtesting/test_run_policy_simulation_cli.py
git commit -m "test: cover clean sheet policy simulation cli"
```

---

### Task 8: Run Focused Policy Simulation Tests

**Files:**
- No source edits expected unless tests reveal a bug.

- [ ] **Step 1: Run optimizer and policy simulator focused suite**

Run:

```bash
uv run --frozen pytest \
  src/tests/backtesting/test_optimizer_policies.py \
  src/tests/backtesting/test_optimizer.py \
  src/tests/backtesting/test_policy_simulation.py \
  src/tests/backtesting/test_run_policy_simulation_cli.py \
  -q
```

Expected: PASS.

- [ ] **Step 2: Run annotation quality check**

Run:

```bash
uv run --frozen ruff check src/cartola/backtesting/optimizer.py src/cartola/backtesting/optimizer_policies.py src/cartola/backtesting/policy_simulation.py src/tests/backtesting/test_optimizer.py src/tests/backtesting/test_optimizer_policies.py src/tests/backtesting/test_policy_simulation.py src/tests/backtesting/test_run_policy_simulation_cli.py --select ANN
```

Expected: PASS. Add missing annotations where Ruff reports them.

- [ ] **Step 3: Run standard lint on touched files**

Run:

```bash
uv run --frozen ruff check src/cartola/backtesting/optimizer.py src/cartola/backtesting/optimizer_policies.py src/cartola/backtesting/policy_simulation.py src/tests/backtesting/test_optimizer.py src/tests/backtesting/test_optimizer_policies.py src/tests/backtesting/test_policy_simulation.py src/tests/backtesting/test_run_policy_simulation_cli.py
```

Expected: PASS.

- [ ] **Step 4: Commit any fixes from focused verification**

If fixes were needed:

```bash
git add src/cartola/backtesting/optimizer.py src/cartola/backtesting/optimizer_policies.py src/cartola/backtesting/policy_simulation.py src/tests/backtesting/test_optimizer.py src/tests/backtesting/test_optimizer_policies.py src/tests/backtesting/test_policy_simulation.py src/tests/backtesting/test_run_policy_simulation_cli.py
git commit -m "fix: stabilize clean sheet policy simulation"
```

If no fixes were needed, do not create an empty commit.

---

### Task 9: Run Real H003 Simulation

**Files:**
- Generated output under `data/08_reporting/policy_simulations/`.
- No source edits expected.

- [ ] **Step 1: Run the real H003 replay**

Run:

```bash
uv run --frozen python scripts/run_policy_simulation.py \
  --experiment-path data/08_reporting/experiments/model_feature/group=xgboost-sensitivity-v2__started_at=20260507T231127138806Z__matrix=f019652c883d \
  --hypothesis-id H003 \
  --policy-set clean-sheet-stack-v1 \
  --models xgboost_depth2_slow \
  --feature-packs ppg_xg_matchup \
  --seasons 2021,2022,2023,2024,2025 \
  --current-year 2026
```

Expected:

```text
Policy simulation started ...
...
Policy simulation complete simulation_id=... output=data/08_reporting/policy_simulations/policy_simulation_started_at=...
```

- [ ] **Step 2: Inspect comparability status**

Replace `<output>` with the output path from Step 1:

```bash
uv run --frozen python - <<'PY'
import json
from pathlib import Path
path = Path("<output>") / "policy_comparability_report.json"
payload = json.loads(path.read_text())
print("status=", payload["status"])
print("fixture_identity_status=", payload["fixture_identity_status"])
print("reason=", payload["reason"])
PY
```

Expected for promotion-grade evidence:

```text
status= ok
fixture_identity_status= verified
```

If status is `diagnostic_only`, do not interpret H003 as final evidence. Report the exact reason.

- [ ] **Step 3: Inspect H003 ranked summary**

Run:

```bash
uv run --frozen python - <<'PY'
import pandas as pd
from pathlib import Path
path = Path("<output>") / "policy_ranked_summary.csv"
frame = pd.read_csv(path)
cols = [
    "policy_variant",
    "total_delta",
    "improved_seasons",
    "season_2025_delta",
    "changed_rounds",
    "changed_round_rate",
    "top_two_positive_delta_concentration",
    "final_budget_delta",
    "min_budget_delta",
    "max_drawdown_delta",
    "decision_status",
    "decision_reason",
]
print(frame.loc[:, cols].to_string(index=False))
PY
```

Expected:

```text
Rows for no_policy plus home_cs_pair_bonus_025, home_cs_pair_bonus_050, home_cs_pair_bonus_075, home_cs_pair_bonus_100.
decision_status is candidate_policy only if all frozen H003 gates pass.
```

- [ ] **Step 4: Inspect invalid rows**

Run:

```bash
uv run --frozen python - <<'PY'
import pandas as pd
from pathlib import Path
path = Path("<output>") / "policy_invalid_rows.csv"
frame = pd.read_csv(path)
print("invalid_rows=", len(frame))
if len(frame):
    print(frame.to_string(index=False))
PY
```

Expected:

```text
invalid_rows= 0
```

- [ ] **Step 5: Record result in the final answer**

Use this exact interpretation rule:

```text
If fixture_identity_status=verified, invalid_rows=0, and a H003 variant has decision_status=candidate_policy, then H003 is a candidate policy for a frozen validation run.
If no H003 variant has candidate_policy, H003 is rejected for this generation.
If fixture identity is not verified or invalid rows are present, H003 is diagnostic only.
```

---

### Task 10: Update Roadmap

**Files:**
- Modify: `roadmap.md`

- [ ] **Step 1: Find the policy simulation section**

Run:

```bash
rg -n "Policy|H001|H002|H003|oracle|hypoth" roadmap.md
```

Expected: at least one policy or research section appears.

- [ ] **Step 2: Update the roadmap with H003 status**

Add or update a concise entry:

```markdown
- H003 clean-sheet defensive stack policy: implemented as artifact-backed `clean-sheet-stack-v1` replay. Uses persisted `matchup_is_home`, `footystats_ppg_diff`, and `footystats_xg_diff` context, adds a bounded `GOL + LAT/ZAG` same-club bonus, and remains research-only until a verified fixture-identity run clears frozen gates.
```

If Task 9 produced a final result, append one sentence:

```markdown
  Latest run: `<output path>`; decision status: `<status summary>`.
```

- [ ] **Step 3: Commit roadmap**

```bash
git add roadmap.md
git commit -m "docs: update roadmap for h003 policy simulation"
```

---

### Task 11: Full Verification

**Files:**
- No source edits expected unless checks fail.

- [ ] **Step 1: Run policy-focused tests**

Run:

```bash
uv run --frozen pytest \
  src/tests/backtesting/test_optimizer_policies.py \
  src/tests/backtesting/test_optimizer.py \
  src/tests/backtesting/test_policy_simulation.py \
  src/tests/backtesting/test_run_policy_simulation_cli.py \
  -q
```

Expected: PASS.

- [ ] **Step 2: Run repository quality gate**

Run:

```bash
uv run --frozen scripts/pyrepo-check --all
```

Expected: PASS. If the command is too slow, keep it running; this is the merge gate.

- [ ] **Step 3: Inspect git status**

Run:

```bash
git status --short
```

Expected: no unstaged source/test/doc changes from this feature except generated reports under ignored paths. If generated CSVs appear, update `.gitignore` only if the file pattern is generated output and not a source artifact.

- [ ] **Step 4: Final implementation note**

Final response must include:

```text
Implemented H003 `clean-sheet-stack-v1`.
Verification: <commands run and result>.
Real simulation output: <path or not run with reason>.
Decision: <candidate_policy/rejected/diagnostic_only with exact reason>.
```

## Self-Review

Spec coverage:

- Frozen policy set is covered by Task 1.
- MILP linearization and cap are covered by Task 3.
- Candidate context columns and conflict validation are covered by Task 4.
- Output columns and changed-round metrics are covered by Task 5.
- H003 frozen gates are covered by Task 6.
- CLI and real run are covered by Tasks 7 and 9.
- Roadmap is covered by Task 10.

Placeholder scan:

- No step relies on unspecified implementation.
- No task uses post-result threshold changes.
- No task asks the worker to infer policy semantics from existing code.

Type consistency:

- `clean_sheet_pair_count` is an integer diagnostic on `SquadOptimizationResult` and `policy_round_results.csv`.
- `clean_sheet_pair_bonus_applied` is a float diagnostic on `SquadOptimizationResult` and `policy_round_results.csv`.
- `selected_ids_changed_vs_no_policy` is a nullable boolean in round results, summarized into integer `changed_rounds` and float `changed_round_rate`.
