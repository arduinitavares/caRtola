# Captain-Aware Optimizer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the fixed-formation/no-captain optimizer with the standard Cartola 2026 scoring contract: all official formations, one captain selected inside the optimizer, captain-adjusted reporting, and no public legacy scoring path.

**Architecture:** Keep prediction unchanged and make optimization/scoring honest. `optimizer.py` owns predicted lineup optimization and captain choice; a new scoring-contract helper owns fixed constants, actual-score evaluation, and report-contract validation; `runner.py`, `recommendation.py`, ablations, audits, and live workflow consume that single contract.

**Tech Stack:** Python 3.13, pandas, PuLP/CBC, Rich, pytest, Ruff, ty, Bandit.

---

## File Map

- Modify `src/cartola/backtesting/config.py`: add official formation constants and scoring contract constants; remove public fixed-formation config fields from `BacktestConfig`.
- Create `src/cartola/backtesting/scoring_contract.py`: one source for `SCORING_CONTRACT_VERSION`, `CAPTAIN_MULTIPLIER`, report contract fields, contract validation, captain actual scoring, and captain policy diagnostics.
- Modify `src/cartola/backtesting/optimizer.py`: search all official formations, add captain binary variables, return captain-aware predicted fields and per-formation scores.
- Modify `src/cartola/backtesting/recommendation.py`: remove public formation config fields, use captain-aware optimizer output, compute replay actual totals outside the optimizer, write captain fields.
- Modify `scripts/recommend_squad.py` and `scripts/run_live_round.py`: print captain-adjusted fields.
- Modify `src/cartola/backtesting/runner.py`: write captain-adjusted round totals, keep selected-player scores raw, write contract metadata.
- Modify `src/cartola/backtesting/metrics.py`: use active-contract round totals; make random selection diagnostics captain-aware with `actual_best_non_tecnico`.
- Modify `src/cartola/backtesting/footystats_ablation.py`: validate `run_metadata.json` before reading metrics and carry contract columns through CSV/JSON.
- Modify `src/cartola/backtesting/compatibility_audit.py`: smoke backtests use the new contract and audit rows record contract fields.
- Modify tests under `src/tests/backtesting/`: optimizer, config, recommendation, runner, metrics, ablation, audit, CLI/live workflow.
- Modify `README.md` and `roadmap.md` only if command output examples or roadmap status need the new scoring terminology.

## Task 1: Scoring Contract Constants And Config Surface

**Files:**
- Modify: `src/cartola/backtesting/config.py`
- Create: `src/cartola/backtesting/scoring_contract.py`
- Modify: `src/tests/backtesting/test_config.py`
- Create: `src/tests/backtesting/test_scoring_contract.py`

- [ ] **Step 1: Add failing config tests for all official formations and no public fixed-formation fields**

Add these tests to `src/tests/backtesting/test_config.py`:

```python
from dataclasses import fields

from cartola.backtesting.config import DEFAULT_FORMATIONS, BacktestConfig


def test_default_formations_are_all_official_cartola_formations() -> None:
    assert DEFAULT_FORMATIONS == {
        "3-4-3": {"gol": 1, "lat": 0, "zag": 3, "mei": 4, "ata": 3, "tec": 1},
        "3-5-2": {"gol": 1, "lat": 0, "zag": 3, "mei": 5, "ata": 2, "tec": 1},
        "4-3-3": {"gol": 1, "lat": 2, "zag": 2, "mei": 3, "ata": 3, "tec": 1},
        "4-4-2": {"gol": 1, "lat": 2, "zag": 2, "mei": 4, "ata": 2, "tec": 1},
        "4-5-1": {"gol": 1, "lat": 2, "zag": 2, "mei": 5, "ata": 1, "tec": 1},
        "5-3-2": {"gol": 1, "lat": 2, "zag": 3, "mei": 3, "ata": 2, "tec": 1},
        "5-4-1": {"gol": 1, "lat": 2, "zag": 3, "mei": 4, "ata": 1, "tec": 1},
    }


def test_backtest_config_has_no_public_fixed_formation_fields() -> None:
    field_names = {field.name for field in fields(BacktestConfig)}
    assert "formation_name" not in field_names
    assert "formations" not in field_names
    assert not hasattr(BacktestConfig(), "selected_formation")
```

- [ ] **Step 2: Add failing scoring contract tests**

Create `src/tests/backtesting/test_scoring_contract.py`:

```python
import json
from pathlib import Path

import pytest

from cartola.backtesting.scoring_contract import (
    CAPTAIN_MULTIPLIER,
    CAPTAIN_SCORING_ENABLED,
    FORMATION_SEARCH,
    SCORING_CONTRACT_VERSION,
    contract_fields,
    validate_report_contract,
)


def test_scoring_contract_constants_are_standard_cartola_2026() -> None:
    assert SCORING_CONTRACT_VERSION == "cartola_standard_2026_v1"
    assert CAPTAIN_SCORING_ENABLED is True
    assert CAPTAIN_MULTIPLIER == 1.5
    assert FORMATION_SEARCH == "all_official_formations"


def test_contract_fields_are_flat_report_columns() -> None:
    assert contract_fields() == {
        "scoring_contract_version": "cartola_standard_2026_v1",
        "captain_scoring_enabled": True,
        "captain_multiplier": 1.5,
        "formation_search": "all_official_formations",
    }


def test_validate_report_contract_rejects_missing_metadata(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="run_metadata.json"):
        validate_report_contract(tmp_path)


def test_validate_report_contract_rejects_old_report_without_contract(tmp_path: Path) -> None:
    (tmp_path / "run_metadata.json").write_text(json.dumps({"season": 2025}) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="scoring_contract_version"):
        validate_report_contract(tmp_path)


def test_validate_report_contract_accepts_standard_contract(tmp_path: Path) -> None:
    (tmp_path / "run_metadata.json").write_text(json.dumps(contract_fields()) + "\n", encoding="utf-8")

    assert validate_report_contract(tmp_path) == contract_fields()
```

- [ ] **Step 3: Run the new tests and verify they fail**

Run:

```bash
uv run --frozen pytest src/tests/backtesting/test_config.py src/tests/backtesting/test_scoring_contract.py -q
```

Expected: failures because `DEFAULT_FORMATIONS` only has `4-3-3`, `BacktestConfig` still exposes formation fields, and `scoring_contract.py` does not exist.

- [ ] **Step 4: Implement scoring constants and remove public fixed-formation config fields**

In `src/cartola/backtesting/config.py`, replace the current `DEFAULT_FORMATIONS` block and remove `formation_name`, `formations`, and `selected_formation` from `BacktestConfig`.

Use this shape:

```python
DEFAULT_FORMATIONS: Mapping[str, Mapping[str, int]] = {
    "3-4-3": {"gol": 1, "lat": 0, "zag": 3, "mei": 4, "ata": 3, "tec": 1},
    "3-5-2": {"gol": 1, "lat": 0, "zag": 3, "mei": 5, "ata": 2, "tec": 1},
    "4-3-3": {"gol": 1, "lat": 2, "zag": 2, "mei": 3, "ata": 3, "tec": 1},
    "4-4-2": {"gol": 1, "lat": 2, "zag": 2, "mei": 4, "ata": 2, "tec": 1},
    "4-5-1": {"gol": 1, "lat": 2, "zag": 2, "mei": 5, "ata": 1, "tec": 1},
    "5-3-2": {"gol": 1, "lat": 2, "zag": 3, "mei": 3, "ata": 2, "tec": 1},
    "5-4-1": {"gol": 1, "lat": 2, "zag": 3, "mei": 4, "ata": 1, "tec": 1},
}


@dataclass(frozen=True)
class BacktestConfig:
    season: int = 2025
    start_round: int = 5
    budget: float = 100.0
    playable_statuses: tuple[str, ...] = ("Provavel",)
    random_seed: int = 123
    project_root: Path = Path(".")
    output_root: Path = Path("data/08_reporting/backtests")
    fixture_mode: FixtureMode = "none"
    strict_alignment_policy: StrictAlignmentPolicy = "fail"
    footystats_mode: FootyStatsMode = "none"
    footystats_evaluation_scope: FootyStatsEvaluationScope = "historical_candidate"
    footystats_league_slug: str = "brazil-serie-a"
    footystats_dir: Path = Path("data/footystats")
    current_year: int | None = None
    scout_columns: tuple[str, ...] = DEFAULT_SCOUT_COLUMNS
```

Create `src/cartola/backtesting/scoring_contract.py`:

```python
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

SCORING_CONTRACT_VERSION = "cartola_standard_2026_v1"
CAPTAIN_SCORING_ENABLED = True
CAPTAIN_MULTIPLIER = 1.5
FORMATION_SEARCH = "all_official_formations"


def contract_fields() -> dict[str, object]:
    return {
        "scoring_contract_version": SCORING_CONTRACT_VERSION,
        "captain_scoring_enabled": CAPTAIN_SCORING_ENABLED,
        "captain_multiplier": CAPTAIN_MULTIPLIER,
        "formation_search": FORMATION_SEARCH,
    }


def validate_contract_mapping(mapping: dict[str, Any]) -> dict[str, object]:
    expected = contract_fields()
    for key, expected_value in expected.items():
        if key not in mapping:
            raise ValueError(f"Missing scoring contract field: {key}")
        if mapping[key] != expected_value:
            raise ValueError(
                f"Unsupported scoring contract field {key}: expected {expected_value!r}, got {mapping[key]!r}"
            )
    return expected


def validate_report_contract(output_path: Path) -> dict[str, object]:
    metadata_path = output_path / "run_metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing run_metadata.json beside report outputs: {metadata_path}")
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"run_metadata.json must contain an object: {metadata_path}")
    return validate_contract_mapping(payload)
```

- [ ] **Step 5: Run the task tests**

Run:

```bash
uv run --frozen pytest src/tests/backtesting/test_config.py src/tests/backtesting/test_scoring_contract.py -q
```

Expected: pass for the new tests. Other project tests may still fail because downstream code still references removed formation fields.

- [ ] **Step 6: Commit Task 1**

Run:

```bash
git add src/cartola/backtesting/config.py src/cartola/backtesting/scoring_contract.py src/tests/backtesting/test_config.py src/tests/backtesting/test_scoring_contract.py
git commit -m "feat: define standard captain scoring contract"
```

## Task 2: Captain-Aware Optimizer Across All Formations

**Files:**
- Modify: `src/cartola/backtesting/optimizer.py`
- Modify: `src/tests/backtesting/test_optimizer.py`

- [ ] **Step 1: Replace optimizer tests with captain-aware contract tests**

In `src/tests/backtesting/test_optimizer.py`, keep the `_candidates()` helper and add/update these tests:

```python
from cartola.backtesting.config import DEFAULT_FORMATIONS, BacktestConfig
from cartola.backtesting.optimizer import optimize_squad
from cartola.backtesting.scoring_contract import CAPTAIN_MULTIPLIER, SCORING_CONTRACT_VERSION


def test_optimizer_searches_all_official_formations_and_marks_one_captain():
    result = optimize_squad(_candidates(), score_column="predicted_points", config=BacktestConfig(budget=100))

    assert result.status == "Optimal"
    assert result.formation_name in DEFAULT_FORMATIONS
    assert result.selected_count == 12
    assert result.selected["is_captain"].sum() == 1
    assert result.selected.loc[result.selected["is_captain"], "posicao"].iloc[0] != "tec"
    assert result.scoring_contract_version == SCORING_CONTRACT_VERSION
    assert result.captain_multiplier == CAPTAIN_MULTIPLIER


def test_optimizer_predicted_total_is_base_plus_captain_bonus():
    result = optimize_squad(_candidates(), score_column="predicted_points", config=BacktestConfig(budget=100))

    captain_score = float(result.selected.loc[result.selected["is_captain"], "predicted_points"].iloc[0])
    assert result.predicted_points_base == float(result.selected["predicted_points"].sum())
    assert result.captain_bonus_predicted == (CAPTAIN_MULTIPLIER - 1.0) * captain_score
    assert result.predicted_points_with_captain == result.predicted_points_base + result.captain_bonus_predicted
    assert result.predicted_points == result.predicted_points_with_captain


def test_optimizer_does_not_choose_unselected_phantom_captain():
    candidates = _candidates()
    phantom = candidates.iloc[[0]].copy()
    phantom["id_atleta"] = 999999
    phantom["apelido"] = "phantom"
    phantom["posicao"] = "ata"
    phantom["preco_pre_rodada"] = 999.0
    phantom["predicted_points"] = 5000.0
    candidates = pd.concat([candidates, phantom], ignore_index=True)

    result = optimize_squad(candidates, score_column="predicted_points", config=BacktestConfig(budget=80))

    assert result.status == "Optimal"
    assert result.captain_id in set(result.selected["id_atleta"])
    assert result.captain_id != 999999


def test_optimizer_never_uses_tecnico_as_captain_even_if_highest_score():
    candidates = _candidates()
    candidates.loc[candidates["posicao"].eq("tec"), "predicted_points"] = 1000.0

    result = optimize_squad(candidates, score_column="predicted_points", config=BacktestConfig(budget=120))

    assert result.status == "Optimal"
    assert result.selected.loc[result.selected["is_captain"], "posicao"].iloc[0] != "tec"


def test_optimizer_returns_per_formation_scores_for_infeasible_formations():
    candidates = _candidates()
    candidates = candidates[candidates["posicao"].ne("lat")].copy()

    result = optimize_squad(candidates, score_column="predicted_points", config=BacktestConfig(budget=100))

    assert result.status == "Optimal"
    assert result.formation_name in {"3-4-3", "3-5-2"}
    by_formation = {record["formation"]: record["solver_status"] for record in result.formation_scores}
    assert by_formation["4-3-3"] != "Optimal"
    assert by_formation[result.formation_name] == "Optimal"
```

- [ ] **Step 2: Run optimizer tests and verify failure**

Run:

```bash
uv run --frozen pytest src/tests/backtesting/test_optimizer.py -q
```

Expected: fail because optimizer still solves only `config.selected_formation` and has no captain fields.

- [ ] **Step 3: Implement captain-aware optimizer dataclasses**

In `src/cartola/backtesting/optimizer.py`, update imports and dataclass:

```python
from typing import Any, Mapping

from cartola.backtesting.config import DEFAULT_FORMATIONS, MARKET_OPEN_PRICE_COLUMN, BacktestConfig
from cartola.backtesting.scoring_contract import (
    CAPTAIN_MULTIPLIER,
    SCORING_CONTRACT_VERSION,
)


@dataclass(frozen=True)
class SquadOptimizationResult:
    selected: pd.DataFrame
    status: str
    budget_used: float
    predicted_points: float
    predicted_points_base: float
    captain_bonus_predicted: float
    predicted_points_with_captain: float
    formation_name: str
    selected_count: int
    captain_id: int | None
    captain_name: str | None
    captain_position: str | None
    captain_club: str | None
    captain_predicted_points: float | None
    captain_multiplier: float
    scoring_contract_version: str
    formation_scores: list[dict[str, object]]
    captain_policy_diagnostics: list[dict[str, object]]
```

- [ ] **Step 4: Implement all-formation search wrapper**

Replace `optimize_squad()` with this public shape:

```python
def optimize_squad(candidates: pd.DataFrame, score_column: str, config: BacktestConfig) -> SquadOptimizationResult:
    if candidates.empty:
        return _empty_result("Empty", "", candidates, formation_scores=[])

    results = [
        _optimize_single_formation(candidates, score_column, config, formation_name, formation)
        for formation_name, formation in DEFAULT_FORMATIONS.items()
    ]
    formation_scores = [record for result in results for record in result.formation_scores]
    optimal = [result for result in results if result.status == "Optimal"]
    if not optimal:
        return _empty_result("Infeasible", "", candidates, formation_scores=formation_scores)

    return sorted(
        optimal,
        key=lambda result: (
            -result.predicted_points_with_captain,
            result.formation_name,
            tuple(int(value) for value in result.selected["id_atleta"].sort_values().tolist()),
            int(result.captain_id) if result.captain_id is not None else 10**12,
        ),
    )[0]
```

- [ ] **Step 5: Implement single-formation MILP with selected and captain variables**

Add `_optimize_single_formation()` in `optimizer.py`:

```python
def _optimize_single_formation(
    candidates: pd.DataFrame,
    score_column: str,
    config: BacktestConfig,
    formation_name: str,
    formation: Mapping[str, int],
) -> SquadOptimizationResult:
    _validate_optimizer_inputs(candidates, score_column)
    player_rows = candidates.loc[~candidates["id_atleta"].duplicated()].copy()
    player_rows = player_rows.sort_values(["id_atleta"]).reset_index(drop=True)
    selected_vars = {
        index: pulp.LpVariable(f"player_{index}_{player_rows.loc[index, 'id_atleta']}", cat=pulp.LpBinary)
        for index in player_rows.index
    }
    captain_vars = {
        index: pulp.LpVariable(f"captain_{index}_{player_rows.loc[index, 'id_atleta']}", cat=pulp.LpBinary)
        for index in player_rows.index
        if player_rows.loc[index, "posicao"] != "tec"
    }

    problem = pulp.LpProblem(f"CartolaSquadOptimizer_{formation_name}", pulp.LpMaximize)
    base_objective = pulp.lpSum(
        float(player_rows.loc[index, score_column]) * selected_vars[index] for index in player_rows.index
    )
    captain_objective = (CAPTAIN_MULTIPLIER - 1.0) * pulp.lpSum(
        float(player_rows.loc[index, score_column]) * captain_vars[index] for index in captain_vars
    )
    problem += base_objective + captain_objective
    problem += pulp.lpSum(
        float(player_rows.loc[index, MARKET_OPEN_PRICE_COLUMN]) * selected_vars[index]
        for index in player_rows.index
    ) <= float(config.budget)

    for position, required_count in formation.items():
        problem += (
            pulp.lpSum(
                selected_vars[index]
                for index in player_rows.index
                if player_rows.loc[index, "posicao"] == position
            )
            == int(required_count)
        )

    problem += pulp.lpSum(captain_vars.values()) == 1
    for index, captain_var in captain_vars.items():
        problem += captain_var <= selected_vars[index]

    status_code = problem.solve(pulp.PULP_CBC_CMD(msg=False))
    status = pulp.LpStatus[status_code]
    if status != "Optimal":
        return _empty_result(status, formation_name, candidates, formation_scores=[_formation_score(formation_name, status)])

    selected_indexes = [index for index, variable in selected_vars.items() if pulp.value(variable) == 1]
    captain_indexes = [index for index, variable in captain_vars.items() if pulp.value(variable) == 1]
    selected = player_rows.loc[selected_indexes].copy().reset_index(drop=True)
    captain_source_index = captain_indexes[0]
    captain_id = int(player_rows.loc[captain_source_index, "id_atleta"])
    selected["is_captain"] = selected["id_atleta"].astype(int).eq(captain_id)
    selected["captain_policy_ev"] = selected["is_captain"]
    selected["captain_policy_safe"] = False
    selected["captain_policy_upside"] = False
    return _result_from_selected(selected, score_column, formation_name)
```

- [ ] **Step 6: Implement validation and result helpers**

Add helpers in `optimizer.py`:

```python
def _validate_optimizer_inputs(candidates: pd.DataFrame, score_column: str) -> None:
    required = {"id_atleta", "apelido", "posicao", MARKET_OPEN_PRICE_COLUMN, score_column}
    missing = sorted(required.difference(candidates.columns))
    if missing:
        raise ValueError(f"Optimizer candidates missing required columns: {missing}")
    for column in (MARKET_OPEN_PRICE_COLUMN, score_column):
        values = pd.to_numeric(candidates[column], errors="coerce")
        if values.isna().any():
            raise ValueError(f"Optimizer column has missing or non-numeric values: {column}")


def _result_from_selected(selected: pd.DataFrame, score_column: str, formation_name: str) -> SquadOptimizationResult:
    budget_used = float(selected[MARKET_OPEN_PRICE_COLUMN].sum())
    predicted_points_base = float(selected[score_column].sum())
    captain_row = selected.loc[selected["is_captain"]].iloc[0]
    captain_predicted_points = float(captain_row[score_column])
    captain_bonus_predicted = (CAPTAIN_MULTIPLIER - 1.0) * captain_predicted_points
    predicted_points_with_captain = predicted_points_base + captain_bonus_predicted
    formation_score = _formation_score(
        formation_name,
        "Optimal",
        predicted_points_base=predicted_points_base,
        captain_bonus_predicted=captain_bonus_predicted,
        predicted_points_with_captain=predicted_points_with_captain,
        captain_id=int(captain_row["id_atleta"]),
        captain_name=str(captain_row.get("apelido", "")),
    )
    return SquadOptimizationResult(
        selected=selected,
        status="Optimal",
        budget_used=budget_used,
        predicted_points=predicted_points_with_captain,
        predicted_points_base=predicted_points_base,
        captain_bonus_predicted=captain_bonus_predicted,
        predicted_points_with_captain=predicted_points_with_captain,
        formation_name=formation_name,
        selected_count=len(selected),
        captain_id=int(captain_row["id_atleta"]),
        captain_name=str(captain_row.get("apelido", "")),
        captain_position=str(captain_row.get("posicao", "")),
        captain_club=str(captain_row.get("nome_clube", "")) if "nome_clube" in captain_row else None,
        captain_predicted_points=captain_predicted_points,
        captain_multiplier=CAPTAIN_MULTIPLIER,
        scoring_contract_version=SCORING_CONTRACT_VERSION,
        formation_scores=[formation_score],
        captain_policy_diagnostics=[],
    )
```

- [ ] **Step 7: Run optimizer tests**

Run:

```bash
uv run --frozen pytest src/tests/backtesting/test_optimizer.py -q
```

Expected: pass.

- [ ] **Step 8: Commit Task 2**

Run:

```bash
git add src/cartola/backtesting/optimizer.py src/tests/backtesting/test_optimizer.py
git commit -m "feat: optimize squads with captain and all formations"
```

## Task 3: Captain Policy And Actual Scoring Helpers

**Files:**
- Modify: `src/cartola/backtesting/scoring_contract.py`
- Modify: `src/tests/backtesting/test_scoring_contract.py`

- [ ] **Step 1: Add failing tests for captain policy diagnostics and actual scoring**

Append to `src/tests/backtesting/test_scoring_contract.py`:

```python
import pandas as pd

from cartola.backtesting.scoring_contract import (
    actual_scores_with_captain,
    captain_policy_diagnostics,
)


def _selected_for_captain_policy() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"id_atleta": 1, "apelido": "A", "posicao": "ata", "nome_clube": "Club", "predicted_points": 8.0, "prior_points_std": 1.0, "pontuacao": 4.0, "is_captain": True},
            {"id_atleta": 2, "apelido": "B", "posicao": "mei", "nome_clube": "Club", "predicted_points": 7.5, "prior_points_std": 0.1, "pontuacao": 10.0, "is_captain": False},
            {"id_atleta": 3, "apelido": "C", "posicao": "tec", "nome_clube": "Club", "predicted_points": 12.0, "prior_points_std": 0.0, "pontuacao": 3.0, "is_captain": False},
        ]
    )


def test_actual_scores_with_captain_uses_selected_captain_once() -> None:
    result = actual_scores_with_captain(_selected_for_captain_policy(), actual_column="pontuacao")

    assert result["actual_points_base"] == 17.0
    assert result["captain_bonus_actual"] == 2.0
    assert result["actual_points_with_captain"] == 19.0


def test_captain_policy_diagnostics_use_same_selected_squad_and_exclude_tecnico() -> None:
    diagnostics = captain_policy_diagnostics(
        _selected_for_captain_policy(),
        predicted_column="predicted_points",
        actual_column="pontuacao",
    )

    by_policy = {record["policy"]: record for record in diagnostics}
    assert by_policy["ev"]["captain_id"] == 1
    assert by_policy["safe"]["captain_id"] == 2
    assert by_policy["upside"]["captain_id"] == 1
    assert all(record["captain_position"] != "tec" for record in diagnostics)
    assert by_policy["safe"]["actual_delta_vs_ev"] == 3.0
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```bash
uv run --frozen pytest src/tests/backtesting/test_scoring_contract.py -q
```

Expected: fail because the helper functions are missing.

- [ ] **Step 3: Implement actual score helper**

Add to `src/cartola/backtesting/scoring_contract.py`:

```python
import math

import pandas as pd


def _finite_numeric_series(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        raise ValueError(f"Missing required score column: {column}")
    values = pd.to_numeric(frame[column], errors="coerce")
    finite = values.map(lambda value: bool(pd.notna(value) and math.isfinite(float(value))))
    if not finite.all():
        raise ValueError(f"Score column has missing or non-finite values: {column}")
    return values.astype(float)


def _selected_captain_row(selected: pd.DataFrame) -> pd.Series:
    if "is_captain" not in selected.columns:
        raise ValueError("selected frame must include is_captain")
    captain_rows = selected.loc[selected["is_captain"].astype(bool)]
    if len(captain_rows) != 1:
        raise ValueError(f"selected frame must contain exactly one captain, found {len(captain_rows)}")
    return captain_rows.iloc[0]


def actual_scores_with_captain(selected: pd.DataFrame, *, actual_column: str = "pontuacao") -> dict[str, float]:
    values = _finite_numeric_series(selected, actual_column)
    captain = _selected_captain_row(selected)
    captain_actual = float(pd.to_numeric(pd.Series([captain[actual_column]]), errors="raise").iloc[0])
    base = float(values.sum())
    bonus = (CAPTAIN_MULTIPLIER - 1.0) * captain_actual
    return {
        "actual_points_base": base,
        "captain_bonus_actual": bonus,
        "actual_points_with_captain": base + bonus,
    }
```

- [ ] **Step 4: Implement captain policy diagnostics**

Add to `src/cartola/backtesting/scoring_contract.py`:

```python
def _captain_candidates(selected: pd.DataFrame, predicted_column: str) -> pd.DataFrame:
    candidates = selected.loc[selected["posicao"].ne("tec")].copy()
    if candidates.empty:
        raise ValueError("selected squad has no non-tecnico captain candidates")
    candidates[predicted_column] = pd.to_numeric(candidates[predicted_column], errors="raise").astype(float)
    if "prior_points_std" not in candidates.columns:
        candidates["prior_points_std"] = 0.0
    candidates["prior_points_std"] = pd.to_numeric(candidates["prior_points_std"], errors="coerce").fillna(0.0).astype(float)
    return candidates


def _policy_pick(candidates: pd.DataFrame, policy: str, predicted_column: str) -> pd.Series:
    source = candidates.copy()
    if policy == "ev":
        source["policy_score"] = source[predicted_column]
    elif policy == "safe":
        source["policy_score"] = source[predicted_column] - source["prior_points_std"]
    elif policy == "upside":
        source["policy_score"] = source[predicted_column] + source["prior_points_std"]
    else:
        raise ValueError(f"Unsupported captain policy: {policy}")
    source = source.sort_values(["policy_score", predicted_column, "id_atleta"], ascending=[False, False, True])
    return source.iloc[0]


def captain_policy_diagnostics(
    selected: pd.DataFrame,
    *,
    predicted_column: str,
    actual_column: str | None = None,
) -> list[dict[str, object]]:
    candidates = _captain_candidates(selected, predicted_column)
    base_predicted = float(pd.to_numeric(selected[predicted_column], errors="raise").sum())
    ev_actual_total = None
    if actual_column is not None:
        ev_captain = _policy_pick(candidates, "ev", predicted_column)
        ev_bonus = (CAPTAIN_MULTIPLIER - 1.0) * float(ev_captain[actual_column])
        ev_actual_total = float(pd.to_numeric(selected[actual_column], errors="raise").sum()) + ev_bonus

    records: list[dict[str, object]] = []
    for policy in ("ev", "safe", "upside"):
        captain = _policy_pick(candidates, policy, predicted_column)
        predicted_points = float(captain[predicted_column])
        predicted_bonus = (CAPTAIN_MULTIPLIER - 1.0) * predicted_points
        actual_captain_points = None
        actual_bonus = None
        actual_total = None
        actual_delta = None
        if actual_column is not None:
            actual_captain_points = float(captain[actual_column])
            actual_bonus = (CAPTAIN_MULTIPLIER - 1.0) * actual_captain_points
            actual_total = float(pd.to_numeric(selected[actual_column], errors="raise").sum()) + actual_bonus
            actual_delta = None if ev_actual_total is None else actual_total - ev_actual_total
        records.append(
            {
                "policy": policy,
                "captain_id": int(captain["id_atleta"]),
                "captain_name": str(captain.get("apelido", "")),
                "captain_position": str(captain.get("posicao", "")),
                "captain_club": str(captain.get("nome_clube", "")),
                "captain_predicted_points": predicted_points,
                "predicted_captain_bonus": predicted_bonus,
                "predicted_points_with_policy": base_predicted + predicted_bonus,
                "actual_captain_points": actual_captain_points,
                "actual_captain_bonus": actual_bonus,
                "actual_points_with_policy": actual_total,
                "actual_delta_vs_ev": actual_delta,
            }
        )
    return records
```

- [ ] **Step 5: Run scoring helper tests**

Run:

```bash
uv run --frozen pytest src/tests/backtesting/test_scoring_contract.py -q
```

Expected: pass.

- [ ] **Step 6: Commit Task 3**

Run:

```bash
git add src/cartola/backtesting/scoring_contract.py src/tests/backtesting/test_scoring_contract.py
git commit -m "feat: add captain scoring helpers"
```

## Task 4: Recommendation Integration And Live Output

**Files:**
- Modify: `src/cartola/backtesting/recommendation.py`
- Modify: `scripts/recommend_squad.py`
- Modify: `scripts/run_live_round.py`
- Modify: `src/cartola/backtesting/live_workflow.py`
- Modify: `src/tests/backtesting/test_recommendation.py`
- Modify: `src/tests/backtesting/test_recommend_squad_cli.py`
- Modify: `src/tests/backtesting/test_live_workflow.py`

- [ ] **Step 1: Add failing recommendation tests for captain fields and no fixed formation config**

In `src/tests/backtesting/test_recommendation.py`, remove `test_recommendation_config_selected_formation` and add:

```python
from dataclasses import fields

from cartola.backtesting.scoring_contract import SCORING_CONTRACT_VERSION


def test_recommendation_config_has_no_public_fixed_formation_fields() -> None:
    field_names = {field.name for field in fields(RecommendationConfig)}
    assert "formation_name" not in field_names
    assert "formations" not in field_names
    assert not hasattr(
        RecommendationConfig(season=2026, target_round=14, mode="live"),
        "selected_formation",
    )


def test_recommendation_outputs_captain_contract_fields(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    season_df = _season_frame(range(1, 4), target_round=3, live_target=True)
    monkeypatch.setattr("cartola.backtesting.recommendation.load_season_data", lambda *a, **k: season_df)
    monkeypatch.setattr(
        "cartola.backtesting.recommendation.load_footystats_feature_rows_for_recommendation",
        lambda **kwargs: None,
    )
    config = RecommendationConfig(
        season=2026,
        target_round=3,
        mode="live",
        project_root=tmp_path,
        current_year=2026,
        footystats_mode="none",
    )

    result = run_recommendation(config)

    assert result.summary["scoring_contract_version"] == SCORING_CONTRACT_VERSION
    assert result.summary["captain_scoring_enabled"] is True
    assert result.summary["formation_search"] == "all_official_formations"
    assert result.summary["captain_id"] in set(result.recommended_squad["id_atleta"])
    assert result.recommended_squad["is_captain"].sum() == 1
    assert "predicted_points_base" in result.summary
    assert "captain_bonus_predicted" in result.summary
    assert result.summary["predicted_points"] == result.summary["predicted_points_with_captain"]
    assert result.metadata["scoring_contract_version"] == SCORING_CONTRACT_VERSION
```

- [ ] **Step 2: Run recommendation tests and verify failure**

Run:

```bash
uv run --frozen pytest src/tests/backtesting/test_recommendation.py -q
```

Expected: fail because `RecommendationConfig` still has formation fields and summary lacks captain fields.

- [ ] **Step 3: Remove public fixed-formation fields from RecommendationConfig and backtest conversion**

In `src/cartola/backtesting/recommendation.py`, remove `formation_name`, `formations`, and `selected_formation` from `RecommendationConfig`. In `_backtest_config()`, stop passing `formation_name` and `formations`.

Use this shape:

```python
@dataclass(frozen=True)
class RecommendationConfig:
    season: int
    target_round: int
    mode: RecommendationMode
    budget: float = 100.0
    playable_statuses: tuple[str, ...] = ("Provavel",)
    random_seed: int = 123
    project_root: Path = Path(".")
    output_root: Path = Path("data/08_reporting/recommendations")
    footystats_mode: FootyStatsMode = "ppg"
    footystats_league_slug: str = "brazil-serie-a"
    footystats_dir: Path = Path("data/footystats")
    current_year: int | None = None
    allow_finalized_live_data: bool = False
    output_run_id: str | None = None
    live_workflow: Mapping[str, object] | None = None
    scout_columns: tuple[str, ...] = DEFAULT_SCOUT_COLUMNS
```

- [ ] **Step 4: Add captain summary/metadata fields in recommendation**

Import `actual_scores_with_captain`, `captain_policy_diagnostics`, and `contract_fields`. After `optimized = optimize_squad(...)`, compute:

```python
selected = optimized.selected.copy()
selected["predicted_points"] = selected["random_forest_score"]
warnings: list[str] = []
actual_points_base = None
captain_bonus_actual = None
actual_points_with_captain = None
policy_diagnostics = captain_policy_diagnostics(
    selected,
    predicted_column="predicted_points",
    actual_column="pontuacao" if config.mode == "replay" else None,
)
if config.mode == "replay":
    try:
        actual_scores = actual_scores_with_captain(selected, actual_column="pontuacao")
        actual_points_base = actual_scores["actual_points_base"]
        captain_bonus_actual = actual_scores["captain_bonus_actual"]
        actual_points_with_captain = actual_scores["actual_points_with_captain"]
    except ValueError as exc:
        warnings.append(f"Replay actual points are null: {exc}")
```

Build summary with:

```python
summary = {
    **contract_fields(),
    "season": config.season,
    "target_round": config.target_round,
    "mode": config.mode,
    "strategy": "random_forest",
    "formation": optimized.formation_name,
    "budget": float(config.budget),
    "optimizer_status": optimized.status,
    "selected_count": int(optimized.selected_count),
    "budget_used": float(optimized.budget_used),
    "predicted_points": float(optimized.predicted_points_with_captain),
    "predicted_points_base": float(optimized.predicted_points_base),
    "captain_bonus_predicted": float(optimized.captain_bonus_predicted),
    "predicted_points_with_captain": float(optimized.predicted_points_with_captain),
    "actual_points": actual_points_with_captain,
    "actual_points_base": actual_points_base,
    "captain_bonus_actual": captain_bonus_actual,
    "actual_points_with_captain": actual_points_with_captain,
    "captain_id": optimized.captain_id,
    "captain_name": optimized.captain_name,
    "captain_position": optimized.captain_position,
    "captain_club": optimized.captain_club,
    "captain_policy_diagnostics": policy_diagnostics,
    "output_directory": str(config.output_path),
}
```

In `_build_metadata()`, merge `contract_fields()` and add `formation_search`, `allowed_formations`, and `captain_policy_definitions`.

- [ ] **Step 5: Update output columns**

In `recommendation.py`, add captain columns to selected output:

```python
selected_columns = [
    *BASE_OUTPUT_COLUMNS,
    "predicted_points",
    "is_captain",
    "captain_policy_ev",
    "captain_policy_safe",
    "captain_policy_upside",
]
```

Keep `predicted_points` raw per-player.

- [ ] **Step 6: Update terminal output scripts**

In `scripts/recommend_squad.py`, change table labels:

```python
table.add_row("Predicted total", _format_points(_float_summary(result_summary, "predicted_points_with_captain")))
table.add_row("Predicted base", _format_points(_float_summary(result_summary, "predicted_points_base")))
table.add_row("Captain bonus", _format_points(_float_summary(result_summary, "captain_bonus_predicted")))
table.add_row("Captain", str(result_summary.get("captain_name", "n/a")))
```

In `scripts/run_live_round.py`, replace `"Predicted points"` with:

```python
table.add_row("Predicted total", _format_float(metadata.get("predicted_points")))
table.add_row("Captain", str(metadata.get("captain_name")))
table.add_row("Captain bonus", _format_float(metadata.get("captain_bonus_predicted")))
```

In `live_workflow.py`, copy the new fields into workflow metadata:

```python
"predicted_points": None if recommendation is None else recommendation.summary.get("predicted_points"),
"predicted_points_base": None if recommendation is None else recommendation.summary.get("predicted_points_base"),
"captain_bonus_predicted": None if recommendation is None else recommendation.summary.get("captain_bonus_predicted"),
"captain_id": None if recommendation is None else recommendation.summary.get("captain_id"),
"captain_name": None if recommendation is None else recommendation.summary.get("captain_name"),
```

- [ ] **Step 7: Run recommendation and live workflow tests**

Run:

```bash
uv run --frozen pytest src/tests/backtesting/test_recommendation.py src/tests/backtesting/test_recommend_squad_cli.py src/tests/backtesting/test_live_workflow.py -q
```

Expected: pass after updating assertions to the new labels and fields.

- [ ] **Step 8: Commit Task 4**

Run:

```bash
git add src/cartola/backtesting/recommendation.py scripts/recommend_squad.py scripts/run_live_round.py src/cartola/backtesting/live_workflow.py src/tests/backtesting/test_recommendation.py src/tests/backtesting/test_recommend_squad_cli.py src/tests/backtesting/test_live_workflow.py
git commit -m "feat: show captain-aware live recommendations"
```

## Task 5: Backtest Runner Outputs And Metadata

**Files:**
- Modify: `src/cartola/backtesting/runner.py`
- Modify: `src/tests/backtesting/test_runner.py`

- [ ] **Step 1: Add failing runner tests for contract metadata and round aliases**

In `src/tests/backtesting/test_runner.py`, add:

```python
from cartola.backtesting.scoring_contract import SCORING_CONTRACT_VERSION


def test_run_backtest_outputs_standard_scoring_contract(tmp_path):
    season_df = pd.concat([_tiny_round(round_number) for round_number in range(1, 6)], ignore_index=True)
    config = BacktestConfig(project_root=tmp_path, start_round=5, budget=100)

    result = run_backtest(config, season_df=season_df)

    assert result.metadata.scoring_contract_version == SCORING_CONTRACT_VERSION
    assert result.metadata.captain_scoring_enabled is True
    assert result.metadata.captain_multiplier == 1.5
    assert result.metadata.formation_search == "all_official_formations"
    assert "predicted_points_base" in result.round_results.columns
    assert "predicted_points_with_captain" in result.round_results.columns
    assert "captain_id" in result.round_results.columns
    assert (result.round_results["predicted_points"] == result.round_results["predicted_points_with_captain"]).all()
    assert "is_captain" in result.selected_players.columns
    assert result.selected_players.groupby(["rodada", "strategy"])["is_captain"].sum().eq(1).all()
```

- [ ] **Step 2: Run runner tests and verify failure**

Run:

```bash
uv run --frozen pytest src/tests/backtesting/test_runner.py -q
```

Expected: fail because metadata and round columns are missing.

- [ ] **Step 3: Extend `BacktestMetadata` and round columns**

In `runner.py`, import `contract_fields`, `actual_scores_with_captain`, and `captain_policy_diagnostics`.

Add fields to `ROUND_RESULT_COLUMNS`:

```python
"predicted_points_base",
"captain_bonus_predicted",
"predicted_points_with_captain",
"actual_points_base",
"captain_bonus_actual",
"actual_points_with_captain",
"captain_id",
"captain_name",
"captain_policy_ev_id",
"captain_policy_safe_id",
"captain_policy_upside_id",
"actual_points_with_ev_captain",
"actual_points_with_safe_captain",
"actual_points_with_upside_captain",
```

Add metadata fields:

```python
scoring_contract_version: str
captain_scoring_enabled: bool
captain_multiplier: float
formation_search: str
```

- [ ] **Step 4: Populate metadata contract fields**

When building `BacktestMetadata`, merge `contract_fields()`:

```python
contract = contract_fields()
metadata = BacktestMetadata(
    season=config.season,
    start_round=config.start_round,
    max_round=max_round,
    scoring_contract_version=str(contract["scoring_contract_version"]),
    captain_scoring_enabled=bool(contract["captain_scoring_enabled"]),
    captain_multiplier=float(contract["captain_multiplier"]),
    formation_search=str(contract["formation_search"]),
    ...
)
```

- [ ] **Step 5: Populate round rows with captain-aware predicted and actual fields**

Inside the strategy loop, after `result = optimize_squad(...)`, compute actual scores:

```python
selected = result.selected.copy()
selected["rodada"] = round_number
selected["strategy"] = strategy
actual_scores: dict[str, float | None]
try:
    actual_scores = actual_scores_with_captain(selected, actual_column="pontuacao")
except ValueError:
    actual_scores = {
        "actual_points_base": None,
        "captain_bonus_actual": None,
        "actual_points_with_captain": None,
    }
policy_records = captain_policy_diagnostics(selected, predicted_column="predicted_points", actual_column="pontuacao")
policy_by_name = {record["policy"]: record for record in policy_records}
```

Use active-contract aliases:

```python
"predicted_points": result.predicted_points_with_captain,
"actual_points": actual_scores["actual_points_with_captain"],
"predicted_points_base": result.predicted_points_base,
"captain_bonus_predicted": result.captain_bonus_predicted,
"predicted_points_with_captain": result.predicted_points_with_captain,
"actual_points_base": actual_scores["actual_points_base"],
"captain_bonus_actual": actual_scores["captain_bonus_actual"],
"actual_points_with_captain": actual_scores["actual_points_with_captain"],
"captain_id": result.captain_id,
"captain_name": result.captain_name,
"captain_policy_ev_id": policy_by_name["ev"]["captain_id"],
"captain_policy_safe_id": policy_by_name["safe"]["captain_id"],
"captain_policy_upside_id": policy_by_name["upside"]["captain_id"],
"actual_points_with_ev_captain": policy_by_name["ev"]["actual_points_with_policy"],
"actual_points_with_safe_captain": policy_by_name["safe"]["actual_points_with_policy"],
"actual_points_with_upside_captain": policy_by_name["upside"]["actual_points_with_policy"],
```

- [ ] **Step 6: Preserve selected-player raw scores**

Before appending selected players, assert selected-player semantics:

```python
selected["predicted_points"] = selected["predicted_points"].astype(float)
selected["captain_policy_diagnostics"] = json.dumps(policy_records, sort_keys=True)
```

Do not replace selected-player `predicted_points` with captain-adjusted row values.

- [ ] **Step 7: Run runner tests**

Run:

```bash
uv run --frozen pytest src/tests/backtesting/test_runner.py -q
```

Expected: pass after updating any old tests that expected `formation == "4-3-3"` to accept official formation names.

- [ ] **Step 8: Commit Task 5**

Run:

```bash
git add src/cartola/backtesting/runner.py src/tests/backtesting/test_runner.py
git commit -m "feat: write captain-aware backtest outputs"
```

## Task 6: Metrics And Random Baseline Contract

**Files:**
- Modify: `src/cartola/backtesting/metrics.py`
- Modify: `src/tests/backtesting/test_metrics.py`

- [ ] **Step 1: Add failing metrics test for captain-aware random baseline**

In `src/tests/backtesting/test_metrics.py`, add:

```python
def test_random_selection_diagnostics_uses_captain_adjusted_random_points() -> None:
    round_results = pd.DataFrame(
        [
            {"rodada": 1, "strategy": "random_forest", "solver_status": "Optimal", "actual_points": 30.0},
        ]
    )
    selected_players = pd.DataFrame(
        [
            {"rodada": 1, "strategy": "random_forest", "posicao": "gol", "pontuacao": 1.0, "preco_pre_rodada": 1.0},
            {"rodada": 1, "strategy": "random_forest", "posicao": "lat", "pontuacao": 2.0, "preco_pre_rodada": 1.0},
            {"rodada": 1, "strategy": "random_forest", "posicao": "tec", "pontuacao": 3.0, "preco_pre_rodada": 1.0},
        ]
    )
    player_predictions = pd.DataFrame(
        [
            {"rodada": 1, "posicao": "gol", "pontuacao": 10.0, "preco_pre_rodada": 1.0},
            {"rodada": 1, "posicao": "lat", "pontuacao": 20.0, "preco_pre_rodada": 1.0},
            {"rodada": 1, "posicao": "tec", "pontuacao": 30.0, "preco_pre_rodada": 1.0},
        ]
    )

    diagnostics = build_diagnostics(
        round_results,
        selected_players,
        player_predictions,
        budget=10,
        random_draws=1,
        random_seed=1,
    )

    row = diagnostics[
        diagnostics["section"].eq("random_selection")
        & diagnostics["metric"].eq("random_baseline_captain_policy")
    ]
    assert row["value"].iloc[0] == "actual_best_non_tecnico"
```

- [ ] **Step 2: Run metrics tests and verify failure**

Run:

```bash
uv run --frozen pytest src/tests/backtesting/test_metrics.py -q
```

Expected: fail because random diagnostics have no captain policy metric.

- [ ] **Step 3: Update random squad point calculation**

In `metrics.py`, import `CAPTAIN_MULTIPLIER`. Change `_random_valid_squad_points()` so each random squad adds the captain bonus:

```python
non_tecnico = squad.loc[squad["posicao"].ne("tec")]
if non_tecnico.empty:
    continue
captain_actual = float(pd.to_numeric(non_tecnico["pontuacao"], errors="raise").max())
base_actual = float(pd.to_numeric(squad["pontuacao"], errors="raise").sum())
points.append(base_actual + (CAPTAIN_MULTIPLIER - 1.0) * captain_actual)
```

- [ ] **Step 4: Record random baseline captain policy**

In `_append_random_selection_diagnostics()`, add:

```python
_append_metric(
    rows,
    "random_selection",
    str(strategy),
    "all",
    "random_baseline_captain_policy",
    "actual_best_non_tecnico",
)
```

- [ ] **Step 5: Run metrics tests**

Run:

```bash
uv run --frozen pytest src/tests/backtesting/test_metrics.py -q
```

Expected: pass.

- [ ] **Step 6: Commit Task 6**

Run:

```bash
git add src/cartola/backtesting/metrics.py src/tests/backtesting/test_metrics.py
git commit -m "feat: make random diagnostics captain aware"
```

## Task 7: Report Reader Contract Validation In Ablation And Compatibility Audit

**Files:**
- Modify: `src/cartola/backtesting/footystats_ablation.py`
- Modify: `src/cartola/backtesting/compatibility_audit.py`
- Modify: `src/tests/backtesting/test_footystats_ablation.py`
- Modify: `src/tests/backtesting/test_compatibility_audit.py`

- [ ] **Step 1: Add failing ablation tests for metadata validation**

In `src/tests/backtesting/test_footystats_ablation.py`, add:

```python
from cartola.backtesting.scoring_contract import contract_fields


def test_extract_run_metrics_rejects_report_without_contract_metadata(tmp_path: Path) -> None:
    output = tmp_path / "run"
    output.mkdir()
    pd.DataFrame([{"strategy": "random_forest", "average_actual_points": 10.0}]).to_csv(output / "summary.csv", index=False)
    pd.DataFrame(columns=["section", "strategy", "position", "metric", "value"]).to_csv(output / "diagnostics.csv", index=False)

    with pytest.raises(FileNotFoundError, match="run_metadata.json"):
        extract_run_metrics(output)


def test_extract_run_metrics_accepts_standard_contract_metadata(tmp_path: Path) -> None:
    output = tmp_path / "run"
    output.mkdir()
    (output / "run_metadata.json").write_text(json.dumps(contract_fields()) + "\n", encoding="utf-8")
    pd.DataFrame(
        [
            {"strategy": "baseline", "average_actual_points": 8.0},
            {"strategy": "random_forest", "average_actual_points": 10.0},
        ]
    ).to_csv(output / "summary.csv", index=False)
    pd.DataFrame(
        [
            {"section": "prediction", "strategy": "random_forest", "position": "all", "metric": "player_r2", "value": 0.1},
            {"section": "prediction", "strategy": "random_forest", "position": "all", "metric": "player_correlation", "value": 0.2},
        ]
    ).to_csv(output / "diagnostics.csv", index=False)

    assert extract_run_metrics(output)["rf"] == 10.0
```

- [ ] **Step 2: Run ablation tests and verify failure**

Run:

```bash
uv run --frozen pytest src/tests/backtesting/test_footystats_ablation.py -q
```

Expected: fail because `extract_run_metrics()` reads `summary.csv` directly.

- [ ] **Step 3: Validate report contract in ablation metric extraction**

In `footystats_ablation.py`, import `contract_fields` and `validate_report_contract`. At the top of `extract_run_metrics()`:

```python
validate_report_contract(output_path)
summary = pd.read_csv(output_path / "summary.csv")
diagnostics = pd.read_csv(output_path / "diagnostics.csv")
```

Add contract fields to `CSV_COLUMNS`:

```python
"scoring_contract_version",
"captain_scoring_enabled",
"captain_multiplier",
"formation_search",
```

Add matching fields to `SeasonAblationRecord`, set them from `contract_fields()` for season rows and aggregate rows, and include them in `to_csv_row()` / `to_json_object()`.

- [ ] **Step 4: Update compatibility audit records**

In `compatibility_audit.py`, import `contract_fields`. Add the same flat contract fields to the season record and CSV/JSON output. In `_run_backtest_stage()`, after `result = run_backtest(backtest_config)`, copy:

```python
contract = contract_fields()
record.scoring_contract_version = str(contract["scoring_contract_version"])
record.captain_scoring_enabled = bool(contract["captain_scoring_enabled"])
record.captain_multiplier = float(contract["captain_multiplier"])
record.formation_search = str(contract["formation_search"])
```

- [ ] **Step 5: Run ablation and compatibility tests**

Run:

```bash
uv run --frozen pytest src/tests/backtesting/test_footystats_ablation.py src/tests/backtesting/test_compatibility_audit.py -q
```

Expected: pass after updating expected CSV column assertions.

- [ ] **Step 6: Commit Task 7**

Run:

```bash
git add src/cartola/backtesting/footystats_ablation.py src/cartola/backtesting/compatibility_audit.py src/tests/backtesting/test_footystats_ablation.py src/tests/backtesting/test_compatibility_audit.py
git commit -m "feat: validate scoring contracts in report readers"
```

## Task 8: CLI, Tests, And Removed Formation Surface Cleanup

**Files:**
- Modify: `src/cartola/backtesting/cli.py`
- Modify: tests that construct `BacktestConfig(... formation_name=...)` or `RecommendationConfig(... formation_name=...)`
- Modify: `README.md`
- Modify: `roadmap.md`

- [ ] **Step 1: Find remaining public fixed-formation references**

Run:

```bash
rg -n "formation_name|formations|selected_formation|captain_multiplier|captain_mode|formation_mode" src scripts README.md roadmap.md
```

Expected: only private helper names or output field names remain. Public config constructors and CLIs must not accept these fields.

- [ ] **Step 2: Remove or update stale tests**

For tests such as `test_recommendation_config_selected_formation`, delete the test or replace it with the no-public-field assertion from Task 4. For any runner or optimizer test expecting `formation == "4-3-3"`, update the assertion to:

```python
assert result.formation_name in DEFAULT_FORMATIONS
```

For CLI tests, assert there is no `--formation` option:

```python
help_text = subprocess.run(
    ["uv", "run", "--frozen", "python", "scripts/recommend_squad.py", "--help"],
    check=True,
    text=True,
    capture_output=True,
).stdout
assert "--formation" not in help_text
```

- [ ] **Step 3: Update docs to describe standard scoring**

In `README.md`, update recommendation/backtest wording to say:

```markdown
Recommendations and backtests use the standard Cartola 2026 scoring contract:
all official formations are searched, one non-tecnico captain is selected, and
round-level predicted/actual totals include the 1.5x captain multiplier.
Selected-player `predicted_points` remains the raw per-athlete model score.
```

In `roadmap.md`, mark the captain-aware optimizer as the active next milestone.

- [ ] **Step 4: Run cleanup search again**

Run:

```bash
rg -n "formation_name|formations|selected_formation|captain_mode|formation_mode" src scripts README.md roadmap.md
```

Expected: no public config/CLI references. Mentions in docs are acceptable only when stating that fixed-formation or disabled-captain modes are not supported in v1.

- [ ] **Step 5: Run affected tests**

Run:

```bash
uv run --frozen pytest src/tests/backtesting/test_config.py src/tests/backtesting/test_optimizer.py src/tests/backtesting/test_recommendation.py src/tests/backtesting/test_runner.py src/tests/backtesting/test_footystats_ablation.py src/tests/backtesting/test_compatibility_audit.py -q
```

Expected: pass.

- [ ] **Step 6: Commit Task 8**

Run:

```bash
git add src scripts src/tests README.md roadmap.md
git commit -m "chore: remove fixed formation public surface"
```

## Task 9: Full Verification

**Files:**
- No planned source edits unless verification finds a defect.

- [x] **Step 1: Run focused live recommendation smoke in replay mode**

Run:

```bash
uv run --frozen python scripts/recommend_squad.py \
  --season 2026 \
  --target-round 13 \
  --mode replay \
  --budget 100 \
  --footystats-mode ppg \
  --current-year 2026
```

Expected: command succeeds and terminal shows formation, captain, predicted base, captain bonus, predicted total, actual total when available, and output path.

- [x] **Step 2: Inspect generated recommendation metadata**

Run:

```bash
python - <<'PY'
import json
from pathlib import Path
p = Path("data/08_reporting/recommendations/2026/round-13/replay/run_metadata.json")
payload = json.loads(p.read_text())
for key in ("scoring_contract_version", "captain_scoring_enabled", "captain_multiplier", "formation_search"):
    print(key, payload[key])
PY
```

Expected:

```text
scoring_contract_version cartola_standard_2026_v1
captain_scoring_enabled True
captain_multiplier 1.5
formation_search all_official_formations
```

- [x] **Step 3: Run full quality gate**

Run:

```bash
uv run --frozen scripts/pyrepo-check --all
```

Expected: Ruff, ty, Bandit, and pytest all pass.

- [x] **Step 4: Handle verification failures without a generic catch-all commit**

If Step 3 fails, return to the task that owns the failing area and add a focused fix commit using that task's file list and commit-message style. For example, a metrics failure returns to Task 6 and uses:

```bash
git add src/cartola/backtesting/metrics.py src/tests/backtesting/test_metrics.py
git commit -m "fix: make captain random diagnostics pass"
```

If Step 3 passes, do not create an empty commit.

## Self-Review Checklist

- Spec coverage:
  - all 7 formations: Task 1 and Task 2;
  - fixed `CAPTAIN_MULTIPLIER = 1.5`: Task 1 and Task 3;
  - no public legacy fixed-formation path: Task 1, Task 4, Task 8;
  - captain MILP constraints: Task 2;
  - selected-player raw score semantics: Task 4 and Task 5;
  - replay/backtest actual captain scoring outside optimizer: Task 3, Task 4, Task 5;
  - report metadata validation: Task 7;
  - random baseline policy: Task 6;
  - terminal/report output: Task 4 and Task 8.
- Placeholder scan: no unresolved markers, no unspecified "add tests", no deferred implementation details.
- Type consistency:
  - contract fields use `scoring_contract_version`, `captain_scoring_enabled`, `captain_multiplier`, `formation_search`;
  - optimizer predicted fields use `predicted_points_base`, `captain_bonus_predicted`, `predicted_points_with_captain`;
  - actual fields are computed by scoring helpers, not optimizer internals.
