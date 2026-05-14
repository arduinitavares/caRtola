# M006 Fixed Blend Diagnostic Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an artifact-backed diagnostic that tests frozen XGBoost/Ridge fixed blends against the currently promoted `xgboost_depth2_l2_heavy + ppg_xg` profile under the same moving-budget optimizer semantics.

**Architecture:** M006 does not train new models and does not build a learned stacker. It reads completed production-parity child artifacts, validates candidate-row identity across base models, computes deterministic weighted prediction columns, replays each blend through `optimize_squad`, tracks independent moving budgets with `advance_budget`, and writes decision-ready CSV/JSON/HTML artifacts. Learned stacking and RF gating remain out of scope until fixed blends prove stable complementary value.

**Tech Stack:** Python 3.13, pandas, existing `cartola.backtesting` optimizer/budget/scoring modules, `uv`, pytest, Ruff/Ty/Bandit quality gate.

---

## Context

Use this completed same-generation source experiment as the default implementation target:

```text
data/08_reporting/experiments/model_feature/group=production-parity__started_at=20260513T165550180815Z__matrix=9064290978ca
```

Known source facts from that run:

- `xgboost_depth2_l2_heavy + ppg_xg` is promotion eligible and passes v1 guardrails.
- `ridge + ppg_xg` has higher aggregate points but fails selected-player calibration.
- XGB/Ridge prediction correlation is moderate (`~0.79` overall), and selected-squad overlap is low (`~4.22/12`), so a fixed-blend diagnostic has real headroom.
- The immediate test is not AutoML, not RF gating, and not a learned meta-model. It is a cheap, leak-safe replay of predeclared blends.

## Non-Goals

- Do not add ElasticNet, Huber, AutoML, or a random-forest gating model.
- Do not change live defaults from M006 unless the decision artifact clears the frozen gates and the user explicitly approves promotion.
- Do not reuse current live 2026 results as training or promotion evidence.
- Do not compare moving-budget M006 results against fixed-budget artifacts.
- Do not use fixture or matchup context in this milestone; source profile is no-fixture `ppg_xg`.

## File Structure

- Create `src/cartola/backtesting/fixed_blend_diagnostic.py`
  - Dataclasses for blend specs and replay outputs.
  - Blend-spec parser.
  - Source artifact validation and candidate identity checks.
  - Fixed-blend score construction.
  - Moving-budget replay loop.
  - Per-season, ranked, complementarity, and decision summaries.
  - Artifact writers.
- Create `scripts/run_fixed_blend_diagnostic.py`
  - Thin CLI wrapper following `scripts/run_policy_simulation.py` style.
  - Loads `.env` before runtime imports.
  - Prints output path and decision status.
- Create `src/tests/backtesting/test_fixed_blend_diagnostic.py`
  - Unit tests for parsing, validation, replay, summary, and gates.
- Create `src/tests/backtesting/test_run_fixed_blend_diagnostic_cli.py`
  - CLI parser/runtime smoke tests.
- Modify `AGENTS.md`
  - Add M006 workflow command and interpretation rules.

## Frozen M006 Blend Matrix

The first implementation must pre-register exactly these blends:

```text
xgb90_ridge10 = 0.90 * xgboost_depth2_l2_heavy_score + 0.10 * ridge_score
xgb80_ridge20 = 0.80 * xgboost_depth2_l2_heavy_score + 0.20 * ridge_score
xgb70_ridge30 = 0.70 * xgboost_depth2_l2_heavy_score + 0.30 * ridge_score
```

All three use `feature_pack=ppg_xg`, `fixture_mode=none`, `matchup_context_mode=none`, `season=2021..2025`, `start_round=5`, and `initial_budget=100`.

## Output Artifacts

Write every M006 run under:

```text
data/08_reporting/blend_diagnostics/fixed_blend_started_at=<UTC timestamp>/
```

Required files:

- `fixed_blend_manifest.json`
- `blend_complementarity.csv`
- `blend_round_results.csv`
- `blend_selected_players.csv`
- `blend_per_season_summary.csv`
- `blend_ranked_summary.csv`
- `blend_decision.json`
- `invalid_rows.csv`
- `fixed_blend_report.html`

## Decision Statuses

Use these frozen statuses:

- `invalid`: source artifacts, candidate identity, score columns, budget policy, or control reproduction checks fail.
- `rejected`: blend is valid but fails strong and weak gates.
- `inconclusive`: blend result is inside the noise band and does not justify follow-up.
- `weak_positive_research_lead`: blend clears integrity gates and shows modest value, but not enough for promotion.
- `candidate_blend`: blend clears all M006 promotion-quality gates.

Strong `candidate_blend` gates:

- Source experiment has `budget_policy=moving`.
- Selected seasons are exactly `2021,2022,2023,2024,2025`.
- Control is `xgboost_depth2_l2_heavy + ppg_xg`.
- Candidate identity matches across XGB and Ridge for every `(season, rodada, id_atleta, id_clube, posicao)` row.
- Control replay reproduction status is `ok` for every season/round.
- Solver non-optimal rounds do not increase versus control.
- Aggregate actual-points delta versus control is at least `+85`.
- At least `3/5` seasons improve versus control.
- Worst season delta is at least `-25`.
- 2025 season delta is at least `-15`.
- Final budget delta is at least `-10`.
- Min-budget delta is at least `-10`.
- Max-drawdown delta is at most `+10`.
- `selected_calibration_slope` is within `[0.75, 1.25]`.
- `top50_spearman_delta` versus control is at least `-0.03`.
- Disaster rounds under `45` captain-aware actual points do not increase.
- Worst 2-round rolling total delta is at least `-10`.
- Top-two positive delta concentration is at most `0.50`.

Weak `weak_positive_research_lead` gates:

- All integrity gates pass.
- Aggregate actual-points delta versus control is at least `+40`.
- At least `3/5` seasons improve versus control.
- Worst season delta is at least `-35`.
- 2025 season delta is at least `-25`.
- Final budget delta is at least `-15`.
- Disaster rounds under `45` do not increase by more than `1`.
- Top-two positive delta concentration is at most `0.65`.

`inconclusive` band:

- Integrity gates pass.
- Aggregate delta is between `-20` and `+40`.
- No severe budget regression: final budget delta `>= -20` and max-drawdown delta `<= +20`.

---

### Task 1: Core Types And Blend Parser

**Files:**
- Create: `src/cartola/backtesting/fixed_blend_diagnostic.py`
- Test: `src/tests/backtesting/test_fixed_blend_diagnostic.py`

- [ ] **Step 1: Write failing parser tests**

Add these imports and tests:

```python
from __future__ import annotations

import pytest

from cartola.backtesting.fixed_blend_diagnostic import (
    FixedBlendDiagnosticError,
    parse_blend_specs,
)


def test_parse_blend_specs_accepts_predeclared_weights() -> None:
    specs = parse_blend_specs(
        (
            "xgb80_ridge20=xgboost_depth2_l2_heavy:0.8,ridge:0.2",
        )
    )

    assert len(specs) == 1
    spec = specs[0]
    assert spec.name == "xgb80_ridge20"
    assert [(item.model_id, item.weight) for item in spec.components] == [
        ("xgboost_depth2_l2_heavy", 0.8),
        ("ridge", 0.2),
    ]


def test_parse_blend_specs_rejects_negative_weight() -> None:
    with pytest.raises(FixedBlendDiagnosticError, match="non-negative"):
        parse_blend_specs(("bad=xgboost_depth2_l2_heavy:1.1,ridge:-0.1",))


def test_parse_blend_specs_rejects_weight_sum_not_one() -> None:
    with pytest.raises(FixedBlendDiagnosticError, match="sum to 1.0"):
        parse_blend_specs(("bad=xgboost_depth2_l2_heavy:0.8,ridge:0.3",))


def test_parse_blend_specs_rejects_duplicate_name() -> None:
    with pytest.raises(FixedBlendDiagnosticError, match="duplicate blend name"):
        parse_blend_specs(
            (
                "xgb80_ridge20=xgboost_depth2_l2_heavy:0.8,ridge:0.2",
                "xgb80_ridge20=xgboost_depth2_l2_heavy:0.7,ridge:0.3",
            )
        )
```

- [ ] **Step 2: Run parser tests and verify they fail**

Run:

```bash
uv run --frozen pytest src/tests/backtesting/test_fixed_blend_diagnostic.py -q
```

Expected: import failure for `cartola.backtesting.fixed_blend_diagnostic`.

- [ ] **Step 3: Add parser implementation**

Create the module with these definitions:

```python
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pandas as pd


_WEIGHT_TOLERANCE = 1e-9


class FixedBlendDiagnosticError(ValueError):
    pass


@dataclass(frozen=True)
class BlendComponent:
    model_id: str
    weight: float


@dataclass(frozen=True)
class BlendSpec:
    name: str
    components: tuple[BlendComponent, ...]


@dataclass(frozen=True)
class BlendReplayContext:
    experiment_path: Path
    season: int
    feature_pack: str
    control_model: str
    initial_budget: float


@dataclass(frozen=True)
class FixedBlendDecision:
    status: Literal[
        "invalid",
        "rejected",
        "inconclusive",
        "weak_positive_research_lead",
        "candidate_blend",
    ]
    reason: str


def parse_blend_specs(raw_specs: tuple[str, ...]) -> tuple[BlendSpec, ...]:
    specs: list[BlendSpec] = []
    seen_names: set[str] = set()
    for raw in raw_specs:
        if "=" not in raw:
            raise FixedBlendDiagnosticError(f"Blend spec must contain '=': {raw}")
        name, raw_components = raw.split("=", 1)
        name = name.strip()
        if not name:
            raise FixedBlendDiagnosticError(f"Blend spec has empty name: {raw}")
        if name in seen_names:
            raise FixedBlendDiagnosticError(f"duplicate blend name: {name}")
        seen_names.add(name)

        components: list[BlendComponent] = []
        seen_models: set[str] = set()
        for item in raw_components.split(","):
            if ":" not in item:
                raise FixedBlendDiagnosticError(f"Blend component must contain ':': {item}")
            model_id, raw_weight = item.split(":", 1)
            model_id = model_id.strip()
            if not model_id:
                raise FixedBlendDiagnosticError(f"Blend component has empty model id: {item}")
            if model_id in seen_models:
                raise FixedBlendDiagnosticError(f"duplicate model id in blend {name}: {model_id}")
            seen_models.add(model_id)
            try:
                weight = float(raw_weight)
            except ValueError as exc:
                raise FixedBlendDiagnosticError(f"Invalid weight for {model_id}: {raw_weight}") from exc
            if not math.isfinite(weight) or weight < 0.0:
                raise FixedBlendDiagnosticError(f"Blend weights must be finite and non-negative: {raw}")
            components.append(BlendComponent(model_id=model_id, weight=weight))

        if len(components) < 2:
            raise FixedBlendDiagnosticError(f"Blend must contain at least two components: {raw}")
        weight_sum = sum(component.weight for component in components)
        if abs(weight_sum - 1.0) > _WEIGHT_TOLERANCE:
            raise FixedBlendDiagnosticError(f"Blend weights must sum to 1.0, got {weight_sum:.12f}: {raw}")
        specs.append(BlendSpec(name=name, components=tuple(components)))
    if not specs:
        raise FixedBlendDiagnosticError("At least one blend spec is required.")
    return tuple(specs)
```

- [ ] **Step 4: Run parser tests and verify they pass**

Run:

```bash
uv run --frozen pytest src/tests/backtesting/test_fixed_blend_diagnostic.py -q
```

Expected: parser tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/cartola/backtesting/fixed_blend_diagnostic.py src/tests/backtesting/test_fixed_blend_diagnostic.py
git commit -m "feat: add fixed blend spec parser"
```

### Task 2: Source Artifact Loading And Candidate Identity Validation

**Files:**
- Modify: `src/cartola/backtesting/fixed_blend_diagnostic.py`
- Test: `src/tests/backtesting/test_fixed_blend_diagnostic.py`

- [ ] **Step 1: Add failing artifact validation tests**

Append these tests:

```python
from pathlib import Path

import pandas as pd

from cartola.backtesting.fixed_blend_diagnostic import (
    BlendSpec,
    BlendComponent,
    load_blend_candidate_frame,
)


def test_load_blend_candidate_frame_merges_component_scores(tmp_path: Path) -> None:
    experiment = _write_blend_experiment(tmp_path)
    spec = BlendSpec(
        name="xgb80_ridge20",
        components=(
            BlendComponent("xgboost_depth2_l2_heavy", 0.8),
            BlendComponent("ridge", 0.2),
        ),
    )

    frame = load_blend_candidate_frame(
        experiment_path=experiment,
        season=2021,
        feature_pack="ppg_xg",
        spec=spec,
    )

    assert "m006_component_xgboost_depth2_l2_heavy_score" in frame.columns
    assert "m006_component_ridge_score" in frame.columns
    assert "m006_blend_xgb80_ridge20_score" in frame.columns
    assert frame["m006_blend_xgb80_ridge20_score"].tolist() == pytest.approx([4.2, 5.2, 6.2])


def test_load_blend_candidate_frame_rejects_candidate_identity_mismatch(tmp_path: Path) -> None:
    experiment = _write_blend_experiment(tmp_path)
    ridge_path = experiment / "runs" / "season=2021" / "model=ridge" / "feature_pack=ppg_xg" / "player_predictions.csv"
    ridge = pd.read_csv(ridge_path)
    ridge.loc[0, "id_atleta"] = 999
    ridge.to_csv(ridge_path, index=False)
    spec = BlendSpec(
        name="xgb80_ridge20",
        components=(
            BlendComponent("xgboost_depth2_l2_heavy", 0.8),
            BlendComponent("ridge", 0.2),
        ),
    )

    with pytest.raises(FixedBlendDiagnosticError, match="candidate identity mismatch"):
        load_blend_candidate_frame(
            experiment_path=experiment,
            season=2021,
            feature_pack="ppg_xg",
            spec=spec,
        )
```

Add this helper at the bottom of the test file:

```python
def _write_blend_experiment(tmp_path: Path) -> Path:
    experiment = tmp_path / "experiment"
    for model_id, score_values in {
        "xgboost_depth2_l2_heavy": [4.0, 5.0, 6.0],
        "ridge": [5.0, 6.0, 7.0],
    }.items():
        child = experiment / "runs" / "season=2021" / f"model={model_id}" / "feature_pack=ppg_xg"
        child.mkdir(parents=True)
        pd.DataFrame(
            {
                "rodada": [5, 5, 5],
                "id_atleta": [101, 102, 103],
                "id_clube": [1, 1, 2],
                "posicao": ["ata", "mei", "gol"],
                "apelido": ["A", "B", "C"],
                "nome_clube": ["AAA", "AAA", "BBB"],
                "preco_pre_rodada": [10.0, 10.0, 10.0],
                "pontuacao": [1.0, 2.0, 3.0],
                "entrou_em_campo": [True, True, True],
                "variacao": [0.1, 0.2, 0.3],
                "baseline_score": [1.0, 1.0, 1.0],
                f"{model_id}_score": score_values,
                "price_score": [10.0, 10.0, 10.0],
            }
        ).to_csv(child / "player_predictions.csv", index=False)
        pd.DataFrame(
            {
                "rodada": [5],
                "strategy": [model_id],
                "solver_status": ["Optimal"],
                "formation": ["synthetic"],
                "budget_before_round": [100.0],
                "budget_after_round": [100.0],
                "budget_delta": [0.0],
                "budget_used": [30.0],
                "actual_points_with_captain": [6.0],
                "predicted_points_with_captain": [20.0],
                "captain_id": [101],
            }
        ).to_csv(child / "round_results.csv", index=False)
        pd.DataFrame(
            {
                "rodada": [5],
                "strategy": [model_id],
                "id_atleta": [101],
                "id_clube": [1],
                "posicao": ["ata"],
                "preco_pre_rodada": [10.0],
                "pontuacao": [1.0],
                "entrou_em_campo": [True],
                "variacao": [0.1],
                "is_captain": [True],
            }
        ).to_csv(child / "selected_players.csv", index=False)
        (child / "run_metadata.json").write_text(
            json.dumps(
                {
                    "season": 2021,
                    "primary_model_id": model_id,
                    "feature_pack": "ppg_xg",
                    "fixture_mode": "none",
                    "matchup_context_mode": "none",
                    "budget_policy": "moving",
                    "scoring_contract_version": "cartola_standard_2026_v1",
                    "strategy_roles": {
                        "baseline": "baseline",
                        model_id: "primary_model",
                        "price": "price",
                    },
                }
            ),
            encoding="utf-8",
        )
    return experiment
```

- [ ] **Step 2: Run artifact tests and verify they fail**

Run:

```bash
uv run --frozen pytest src/tests/backtesting/test_fixed_blend_diagnostic.py::test_load_blend_candidate_frame_merges_component_scores src/tests/backtesting/test_fixed_blend_diagnostic.py::test_load_blend_candidate_frame_rejects_candidate_identity_mismatch -q
```

Expected: import failure for `load_blend_candidate_frame`.

- [ ] **Step 3: Implement artifact loading**

Add these functions:

```python
_CANDIDATE_IDENTITY_COLUMNS = ("rodada", "id_atleta", "id_clube", "posicao")
_BASE_CANDIDATE_COLUMNS = (
    "rodada",
    "id_atleta",
    "id_clube",
    "posicao",
    "apelido",
    "nome_clube",
    "preco_pre_rodada",
    "pontuacao",
    "entrou_em_campo",
    "variacao",
)


def child_path_for(
    *,
    experiment_path: Path,
    season: int,
    model_id: str,
    feature_pack: str,
) -> Path:
    return experiment_path / "runs" / f"season={season}" / f"model={model_id}" / f"feature_pack={feature_pack}"


def score_column_for(model_id: str) -> str:
    return f"{model_id}_score"


def load_blend_candidate_frame(
    *,
    experiment_path: Path,
    season: int,
    feature_pack: str,
    spec: BlendSpec,
) -> pd.DataFrame:
    first_component = spec.components[0]
    base_path = child_path_for(
        experiment_path=experiment_path,
        season=season,
        model_id=first_component.model_id,
        feature_pack=feature_pack,
    )
    base_frame = _read_prediction_frame(base_path, model_id=first_component.model_id)
    merged = base_frame.copy()
    first_source_col = score_column_for(first_component.model_id)
    merged[f"m006_component_{first_component.model_id}_score"] = merged[first_source_col].astype(float)

    base_identity = _candidate_identity_frame(merged)
    for component in spec.components[1:]:
        component_path = child_path_for(
            experiment_path=experiment_path,
            season=season,
            model_id=component.model_id,
            feature_pack=feature_pack,
        )
        component_frame = _read_prediction_frame(component_path, model_id=component.model_id)
        component_identity = _candidate_identity_frame(component_frame)
        if not base_identity.equals(component_identity):
            raise FixedBlendDiagnosticError(
                "candidate identity mismatch for "
                f"season={season} feature_pack={feature_pack} model={component.model_id}"
            )
        component_score_col = score_column_for(component.model_id)
        merged[f"m006_component_{component.model_id}_score"] = component_frame[component_score_col].astype(float).to_numpy()

    blend_col = f"m006_blend_{spec.name}_score"
    merged[blend_col] = 0.0
    for component in spec.components:
        merged[blend_col] = merged[blend_col] + (
            component.weight * merged[f"m006_component_{component.model_id}_score"].astype(float)
        )
    return merged


def _read_prediction_frame(child_path: Path, *, model_id: str) -> pd.DataFrame:
    prediction_path = child_path / "player_predictions.csv"
    if not prediction_path.is_file():
        raise FixedBlendDiagnosticError(f"Missing player_predictions.csv: {prediction_path}")
    frame = pd.read_csv(prediction_path)
    required = (*_BASE_CANDIDATE_COLUMNS, score_column_for(model_id))
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise FixedBlendDiagnosticError(f"{prediction_path} missing required columns: {missing}")
    return frame.sort_values(list(_CANDIDATE_IDENTITY_COLUMNS), kind="mergesort").reset_index(drop=True)


def _candidate_identity_frame(frame: pd.DataFrame) -> pd.DataFrame:
    identity = frame.loc[:, _CANDIDATE_IDENTITY_COLUMNS].copy()
    identity["rodada"] = pd.to_numeric(identity["rodada"], errors="raise").astype(int)
    identity["id_atleta"] = pd.to_numeric(identity["id_atleta"], errors="raise").astype(int)
    identity["id_clube"] = pd.to_numeric(identity["id_clube"], errors="raise").astype(int)
    identity["posicao"] = identity["posicao"].astype(str)
    return identity.sort_values(list(_CANDIDATE_IDENTITY_COLUMNS), kind="mergesort").reset_index(drop=True)
```

- [ ] **Step 4: Run artifact tests and verify they pass**

Run:

```bash
uv run --frozen pytest src/tests/backtesting/test_fixed_blend_diagnostic.py -q
```

Expected: all current M006 tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/cartola/backtesting/fixed_blend_diagnostic.py src/tests/backtesting/test_fixed_blend_diagnostic.py
git commit -m "feat: validate fixed blend source artifacts"
```

### Task 3: Moving-Budget Blend Replay

**Files:**
- Modify: `src/cartola/backtesting/fixed_blend_diagnostic.py`
- Test: `src/tests/backtesting/test_fixed_blend_diagnostic.py`

- [ ] **Step 1: Write failing replay test**

Add:

```python
from cartola.backtesting.fixed_blend_diagnostic import run_blend_replay_for_season


def test_run_blend_replay_for_season_tracks_independent_moving_budget(tmp_path: Path) -> None:
    experiment = _write_optimizable_blend_experiment(tmp_path)
    spec = BlendSpec(
        name="xgb80_ridge20",
        components=(
            BlendComponent("xgboost_depth2_l2_heavy", 0.8),
            BlendComponent("ridge", 0.2),
        ),
    )

    result = run_blend_replay_for_season(
        experiment_path=experiment,
        season=2021,
        feature_pack="ppg_xg",
        control_model="xgboost_depth2_l2_heavy",
        specs=(spec,),
        initial_budget=100.0,
    )

    round_results = pd.DataFrame(result.round_rows)
    assert round_results["blend_name"].tolist() == ["xgb80_ridge20", "xgb80_ridge20"]
    assert round_results["rodada"].tolist() == [5, 6]
    assert round_results.loc[0, "budget_before_round"] == pytest.approx(100.0)
    assert round_results.loc[0, "budget_after_round"] == pytest.approx(101.0)
    assert round_results.loc[1, "budget_before_round"] == pytest.approx(101.0)
```

Add this helper:

```python
def _write_optimizable_blend_experiment(tmp_path: Path) -> Path:
    experiment = tmp_path / "experiment"
    players = [
        (101, "gol", 1),
        (102, "lat", 1),
        (103, "lat", 2),
        (104, "zag", 2),
        (105, "zag", 3),
        (106, "mei", 3),
        (107, "mei", 4),
        (108, "mei", 4),
        (109, "ata", 5),
        (110, "ata", 5),
        (111, "ata", 6),
        (112, "tec", 6),
    ]
    for model_id, offset in {"xgboost_depth2_l2_heavy": 0.0, "ridge": 0.2}.items():
        child = experiment / "runs" / "season=2021" / f"model={model_id}" / "feature_pack=ppg_xg"
        child.mkdir(parents=True)
        rows: list[dict[str, object]] = []
        for rodada in (5, 6):
            for player_id, posicao, club_id in players:
                rows.append(
                    {
                        "rodada": rodada,
                        "id_atleta": player_id,
                        "id_clube": club_id,
                        "posicao": posicao,
                        "apelido": f"P{player_id}",
                        "nome_clube": f"C{club_id}",
                        "preco_pre_rodada": 1.0,
                        "pontuacao": 2.0,
                        "entrou_em_campo": True,
                        "variacao": 1.0 if rodada == 5 else -0.5,
                        "baseline_score": 1.0,
                        f"{model_id}_score": 5.0 + offset,
                        "price_score": 1.0,
                    }
                )
        pd.DataFrame(rows).to_csv(child / "player_predictions.csv", index=False)
        pd.DataFrame(
            {
                "rodada": [5, 6],
                "strategy": [model_id, model_id],
                "solver_status": ["Optimal", "Optimal"],
                "formation": ["4-3-3", "4-3-3"],
                "budget_before_round": [100.0, 112.0],
                "budget_after_round": [112.0, 106.0],
                "budget_delta": [12.0, -6.0],
                "budget_used": [12.0, 12.0],
                "actual_points_with_captain": [25.0, 25.0],
                "predicted_points_with_captain": [75.0, 75.0],
                "captain_id": [101, 101],
            }
        ).to_csv(child / "round_results.csv", index=False)
        pd.DataFrame(
            {
                "rodada": [5],
                "strategy": [model_id],
                "id_atleta": [101],
                "id_clube": [1],
                "posicao": ["gol"],
                "preco_pre_rodada": [1.0],
                "pontuacao": [2.0],
                "entrou_em_campo": [True],
                "variacao": [1.0],
                "is_captain": [True],
            }
        ).to_csv(child / "selected_players.csv", index=False)
        (child / "run_metadata.json").write_text(
            json.dumps(
                {
                    "season": 2021,
                    "primary_model_id": model_id,
                    "feature_pack": "ppg_xg",
                    "fixture_mode": "none",
                    "matchup_context_mode": "none",
                    "budget_policy": "moving",
                    "scoring_contract_version": "cartola_standard_2026_v1",
                    "strategy_roles": {
                        "baseline": "baseline",
                        model_id: "primary_model",
                        "price": "price",
                    },
                }
            ),
            encoding="utf-8",
        )
    return experiment
```

- [ ] **Step 2: Run replay test and verify it fails**

Run:

```bash
uv run --frozen pytest src/tests/backtesting/test_fixed_blend_diagnostic.py::test_run_blend_replay_for_season_tracks_independent_moving_budget -q
```

Expected: import failure for `run_blend_replay_for_season`.

- [ ] **Step 3: Implement replay loop**

Add imports:

```python
from dataclasses import asdict

from cartola.backtesting.budgeting import advance_budget, initial_budget_state
from cartola.backtesting.config import BacktestConfig
from cartola.backtesting.optimizer import optimize_squad
from cartola.backtesting.scoring_contract import actual_scores_with_captain
```

Add dataclass and function:

```python
@dataclass(frozen=True)
class BlendReplayResult:
    round_rows: list[dict[str, object]]
    selected_player_rows: list[dict[str, object]]
    invalid_rows: list[dict[str, object]]


def run_blend_replay_for_season(
    *,
    experiment_path: Path,
    season: int,
    feature_pack: str,
    control_model: str,
    specs: tuple[BlendSpec, ...],
    initial_budget: float,
) -> BlendReplayResult:
    round_rows: list[dict[str, object]] = []
    selected_rows: list[dict[str, object]] = []
    invalid_rows: list[dict[str, object]] = []
    for spec in specs:
        budget_state = initial_budget_state(initial_budget)
        try:
            candidate_frame = load_blend_candidate_frame(
                experiment_path=experiment_path,
                season=season,
                feature_pack=feature_pack,
                spec=spec,
            )
        except FixedBlendDiagnosticError as exc:
            invalid_rows.append(
                {
                    "season": season,
                    "blend_name": spec.name,
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                }
            )
            continue
        score_column = f"m006_blend_{spec.name}_score"
        for round_number in sorted(pd.to_numeric(candidate_frame["rodada"], errors="raise").astype(int).unique()):
            candidates = candidate_frame.loc[candidate_frame["rodada"].eq(round_number)].copy()
            candidates["predicted_points"] = candidates[score_column].astype(float)
            config = BacktestConfig(
                season=season,
                start_round=int(round_number),
                budget=float(budget_state.current_budget),
                fixture_mode="none",
                matchup_context_mode="none",
            )
            result = optimize_squad(
                candidates,
                score_column="predicted_points",
                config=config,
                budget=float(budget_state.current_budget),
            )
            if result.status == "Optimal" and not result.selected.empty:
                selected = result.selected.copy()
                scores = actual_scores_with_captain(selected, actual_column="pontuacao")
                budget_update = advance_budget(budget_state, selected, budget_used=result.budget_used)
                selected["season"] = season
                selected["blend_name"] = spec.name
                selected["rodada"] = int(round_number)
                selected["strategy"] = spec.name
                selected_rows.extend(_selected_output_rows(selected))
            else:
                scores = {
                    "actual_points_base": 0.0,
                    "captain_bonus_actual": 0.0,
                    "actual_points_with_captain": 0.0,
                }
                budget_update = advance_budget(budget_state, pd.DataFrame(), budget_used=0.0)

            round_rows.append(
                {
                    "season": season,
                    "control_model": control_model,
                    "feature_pack": feature_pack,
                    "blend_name": spec.name,
                    "rodada": int(round_number),
                    "solver_status": result.status,
                    "formation": result.formation_name,
                    "selected_count": result.selected_count,
                    "budget_before_round": budget_update.budget_before_round,
                    "budget_used": result.budget_used,
                    "budget_remaining": budget_update.budget_remaining,
                    "budget_delta": budget_update.budget_delta,
                    "budget_after_round": budget_update.budget_after_round,
                    "budget_peak": budget_update.budget_peak,
                    "budget_drawdown": budget_update.budget_drawdown,
                    "predicted_points_with_captain": result.predicted_points_with_captain,
                    "actual_points_base": scores["actual_points_base"],
                    "captain_bonus_actual": scores["captain_bonus_actual"],
                    "actual_points_with_captain": scores["actual_points_with_captain"],
                    "captain_id": result.captain_id,
                    "captain_name": result.captain_name,
                }
            )
            budget_state = budget_update.next_state
    return BlendReplayResult(round_rows=round_rows, selected_player_rows=selected_rows, invalid_rows=invalid_rows)


def _selected_output_rows(selected: pd.DataFrame) -> list[dict[str, object]]:
    output_columns = [
        "season",
        "blend_name",
        "rodada",
        "strategy",
        "id_atleta",
        "apelido",
        "posicao",
        "id_clube",
        "nome_clube",
        "preco_pre_rodada",
        "pontuacao",
        "entrou_em_campo",
        "variacao",
        "is_captain",
        "predicted_points",
    ]
    return selected.loc[:, [column for column in output_columns if column in selected.columns]].to_dict("records")
```

- [ ] **Step 4: Run replay test and verify it passes**

Run:

```bash
uv run --frozen pytest src/tests/backtesting/test_fixed_blend_diagnostic.py::test_run_blend_replay_for_season_tracks_independent_moving_budget -q
```

Expected: replay test passes.

- [ ] **Step 5: Commit**

```bash
git add src/cartola/backtesting/fixed_blend_diagnostic.py src/tests/backtesting/test_fixed_blend_diagnostic.py
git commit -m "feat: replay fixed blends with moving budgets"
```

### Task 4: Summaries, Complementarity Metrics, And Decision Gates

**Files:**
- Modify: `src/cartola/backtesting/fixed_blend_diagnostic.py`
- Test: `src/tests/backtesting/test_fixed_blend_diagnostic.py`

- [ ] **Step 1: Write failing decision tests**

Add:

```python
from cartola.backtesting.fixed_blend_diagnostic import (
    build_blend_ranked_summary,
    decide_blend_candidate,
)


def test_decide_blend_candidate_promotes_strong_candidate() -> None:
    decision = decide_blend_candidate(
        source_valid=True,
        aggregate_delta=100.0,
        improved_seasons=4,
        worst_season_delta=-10.0,
        season_2025_delta=5.0,
        final_budget_delta=-3.0,
        min_budget_delta=-2.0,
        max_drawdown_delta=4.0,
        selected_calibration_slope=0.95,
        top50_spearman_delta=0.0,
        disaster_rounds_under45_delta=0,
        worst_2_round_delta=-5.0,
        non_optimal_delta=0,
        top_two_concentration=0.4,
    )

    assert decision.status == "candidate_blend"


def test_decide_blend_candidate_returns_weak_positive_for_modest_delta() -> None:
    decision = decide_blend_candidate(
        source_valid=True,
        aggregate_delta=50.0,
        improved_seasons=3,
        worst_season_delta=-20.0,
        season_2025_delta=-10.0,
        final_budget_delta=-6.0,
        min_budget_delta=-8.0,
        max_drawdown_delta=8.0,
        selected_calibration_slope=0.70,
        top50_spearman_delta=-0.05,
        disaster_rounds_under45_delta=1,
        worst_2_round_delta=-20.0,
        non_optimal_delta=0,
        top_two_concentration=0.6,
    )

    assert decision.status == "weak_positive_research_lead"


def test_decide_blend_candidate_rejects_more_disasters() -> None:
    decision = decide_blend_candidate(
        source_valid=True,
        aggregate_delta=120.0,
        improved_seasons=5,
        worst_season_delta=0.0,
        season_2025_delta=10.0,
        final_budget_delta=0.0,
        min_budget_delta=0.0,
        max_drawdown_delta=0.0,
        selected_calibration_slope=1.0,
        top50_spearman_delta=0.0,
        disaster_rounds_under45_delta=2,
        worst_2_round_delta=0.0,
        non_optimal_delta=0,
        top_two_concentration=0.2,
    )

    assert decision.status == "rejected"
    assert "disaster" in decision.reason
```

- [ ] **Step 2: Run decision tests and verify they fail**

Run:

```bash
uv run --frozen pytest src/tests/backtesting/test_fixed_blend_diagnostic.py::test_decide_blend_candidate_promotes_strong_candidate src/tests/backtesting/test_fixed_blend_diagnostic.py::test_decide_blend_candidate_returns_weak_positive_for_modest_delta src/tests/backtesting/test_fixed_blend_diagnostic.py::test_decide_blend_candidate_rejects_more_disasters -q
```

Expected: import failure for `decide_blend_candidate`.

- [ ] **Step 3: Implement gates**

Add:

```python
def decide_blend_candidate(
    *,
    source_valid: bool,
    aggregate_delta: float,
    improved_seasons: int,
    worst_season_delta: float,
    season_2025_delta: float,
    final_budget_delta: float,
    min_budget_delta: float,
    max_drawdown_delta: float,
    selected_calibration_slope: float,
    top50_spearman_delta: float,
    disaster_rounds_under45_delta: int,
    worst_2_round_delta: float,
    non_optimal_delta: int,
    top_two_concentration: float,
) -> FixedBlendDecision:
    if not source_valid:
        return FixedBlendDecision(status="invalid", reason="source artifacts failed validation.")
    if non_optimal_delta > 0:
        return FixedBlendDecision(status="rejected", reason="blend introduced non-optimal solver rounds.")
    if disaster_rounds_under45_delta > 1:
        return FixedBlendDecision(status="rejected", reason="blend increased disaster rounds under 45.")

    strong_pass = (
        aggregate_delta >= 85.0
        and improved_seasons >= 3
        and worst_season_delta >= -25.0
        and season_2025_delta >= -15.0
        and final_budget_delta >= -10.0
        and min_budget_delta >= -10.0
        and max_drawdown_delta <= 10.0
        and 0.75 <= selected_calibration_slope <= 1.25
        and top50_spearman_delta >= -0.03
        and disaster_rounds_under45_delta <= 0
        and worst_2_round_delta >= -10.0
        and top_two_concentration <= 0.50
    )
    if strong_pass:
        return FixedBlendDecision(status="candidate_blend", reason="blend clears M006 strong gates.")

    weak_pass = (
        aggregate_delta >= 40.0
        and improved_seasons >= 3
        and worst_season_delta >= -35.0
        and season_2025_delta >= -25.0
        and final_budget_delta >= -15.0
        and top_two_concentration <= 0.65
    )
    if weak_pass:
        return FixedBlendDecision(
            status="weak_positive_research_lead",
            reason="blend clears weak-positive M006 gates but not strong gates.",
        )

    inconclusive = (
        -20.0 <= aggregate_delta < 40.0
        and final_budget_delta >= -20.0
        and max_drawdown_delta <= 20.0
    )
    if inconclusive:
        return FixedBlendDecision(status="inconclusive", reason="blend is inside the M006 noise band.")
    return FixedBlendDecision(status="rejected", reason="blend fails M006 evidence gates.")
```

Then implement `build_blend_per_season_summary`, `build_blend_ranked_summary`, and `build_blend_complementarity` with deterministic pandas groupbys:

```python
def _rolling_worst_two(values: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce").astype(float)
    if len(numeric) < 2:
        return float(numeric.sum())
    return float(numeric.rolling(window=2).sum().min())


def _top_two_positive_delta_concentration(deltas: pd.Series) -> float:
    positive = pd.to_numeric(deltas, errors="coerce").dropna()
    positive = positive.loc[positive.gt(0.0)]
    total = float(positive.sum())
    if total <= 0.0:
        return float("inf")
    return float(positive.sort_values(ascending=False).head(2).sum() / total)
```

Use the source control round results as benchmark by reading:

```text
runs/season=<season>/model=xgboost_depth2_l2_heavy/feature_pack=ppg_xg/round_results.csv
```

Filter source control rows to `strategy == "xgboost_depth2_l2_heavy"`.

- [ ] **Step 4: Run decision tests and full M006 tests**

Run:

```bash
uv run --frozen pytest src/tests/backtesting/test_fixed_blend_diagnostic.py -q
```

Expected: all M006 tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/cartola/backtesting/fixed_blend_diagnostic.py src/tests/backtesting/test_fixed_blend_diagnostic.py
git commit -m "feat: summarize fixed blend evidence"
```

### Task 5: Run Orchestrator And Artifact Writers

**Files:**
- Modify: `src/cartola/backtesting/fixed_blend_diagnostic.py`
- Test: `src/tests/backtesting/test_fixed_blend_diagnostic.py`

- [ ] **Step 1: Write failing run-level artifact test**

Add:

```python
from cartola.backtesting.fixed_blend_diagnostic import run_fixed_blend_diagnostic


def test_run_fixed_blend_diagnostic_writes_required_artifacts(tmp_path: Path) -> None:
    experiment = _write_optimizable_blend_experiment(tmp_path)
    output_root = tmp_path / "blend_outputs"

    result_path = run_fixed_blend_diagnostic(
        experiment_path=experiment,
        output_root=output_root,
        seasons=(2021,),
        feature_pack="ppg_xg",
        control_model="xgboost_depth2_l2_heavy",
        specs=parse_blend_specs(("xgb80_ridge20=xgboost_depth2_l2_heavy:0.8,ridge:0.2",)),
        initial_budget=100.0,
        current_year=2026,
        started_at_utc="20260513T000000000000Z",
    )

    assert (result_path / "fixed_blend_manifest.json").is_file()
    assert (result_path / "blend_round_results.csv").is_file()
    assert (result_path / "blend_selected_players.csv").is_file()
    assert (result_path / "blend_ranked_summary.csv").is_file()
    assert (result_path / "blend_decision.json").is_file()
    assert (result_path / "invalid_rows.csv").is_file()
    assert (result_path / "fixed_blend_report.html").is_file()
```

- [ ] **Step 2: Run artifact test and verify it fails**

Run:

```bash
uv run --frozen pytest src/tests/backtesting/test_fixed_blend_diagnostic.py::test_run_fixed_blend_diagnostic_writes_required_artifacts -q
```

Expected: import failure for `run_fixed_blend_diagnostic`.

- [ ] **Step 3: Implement run-level orchestration**

Add:

```python
import json
from datetime import UTC, datetime

CSV_FLOAT_FORMAT = "%.10f"


def run_fixed_blend_diagnostic(
    *,
    experiment_path: Path,
    output_root: Path,
    seasons: tuple[int, ...],
    feature_pack: str,
    control_model: str,
    specs: tuple[BlendSpec, ...],
    initial_budget: float,
    current_year: int,
    started_at_utc: str | None = None,
) -> Path:
    started = started_at_utc or datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    output_path = output_root / f"fixed_blend_started_at={started}"
    if output_path.exists():
        raise FileExistsError(output_path)
    output_path.mkdir(parents=True)

    all_round_rows: list[dict[str, object]] = []
    all_selected_rows: list[dict[str, object]] = []
    all_invalid_rows: list[dict[str, object]] = []
    for season in seasons:
        result = run_blend_replay_for_season(
            experiment_path=experiment_path,
            season=season,
            feature_pack=feature_pack,
            control_model=control_model,
            specs=specs,
            initial_budget=initial_budget,
        )
        all_round_rows.extend(result.round_rows)
        all_selected_rows.extend(result.selected_player_rows)
        all_invalid_rows.extend(result.invalid_rows)

    round_results = pd.DataFrame(all_round_rows)
    selected_players = pd.DataFrame(all_selected_rows)
    invalid_rows = pd.DataFrame(all_invalid_rows)
    per_season = build_blend_per_season_summary(
        experiment_path=experiment_path,
        blend_round_results=round_results,
        seasons=seasons,
        control_model=control_model,
        feature_pack=feature_pack,
    )
    ranked = build_blend_ranked_summary(per_season_summary=per_season, selected_players=selected_players)
    complementarity = build_blend_complementarity(
        experiment_path=experiment_path,
        seasons=seasons,
        feature_pack=feature_pack,
        model_a="xgboost_depth2_l2_heavy",
        model_b="ridge",
    )
    decision_payload = build_blend_decision_payload(ranked_summary=ranked, invalid_rows=invalid_rows)
    manifest = {
        "hypothesis_id": "M006",
        "design_revision": "fixed_blend_v1",
        "source_experiment_path": str(experiment_path),
        "seasons": list(seasons),
        "feature_pack": feature_pack,
        "control_model": control_model,
        "initial_budget": initial_budget,
        "budget_policy": "moving",
        "current_year": current_year,
        "blend_specs": [
            {
                "name": spec.name,
                "components": [asdict(component) for component in spec.components],
            }
            for spec in specs
        ],
    }
    _write_json(output_path / "fixed_blend_manifest.json", manifest)
    _write_csv(output_path / "blend_round_results.csv", round_results)
    _write_csv(output_path / "blend_selected_players.csv", selected_players)
    _write_csv(output_path / "blend_per_season_summary.csv", per_season)
    _write_csv(output_path / "blend_ranked_summary.csv", ranked)
    _write_csv(output_path / "blend_complementarity.csv", complementarity)
    _write_csv(output_path / "invalid_rows.csv", invalid_rows)
    _write_json(output_path / "blend_decision.json", decision_payload)
    _write_html_report(output_path, manifest=manifest, ranked=ranked, decision=decision_payload)
    return output_path


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    frame.to_csv(path, index=False, float_format=CSV_FLOAT_FORMAT)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _write_html_report(
    output_path: Path,
    *,
    manifest: dict[str, object],
    ranked: pd.DataFrame,
    decision: dict[str, object],
) -> None:
    html = "\n".join(
        [
            "<!doctype html>",
            "<html><head><meta charset='utf-8'><title>M006 Fixed Blend Diagnostic</title></head><body>",
            "<h1>M006 Fixed Blend Diagnostic</h1>",
            "<h2>Decision</h2>",
            f"<pre>{json.dumps(decision, indent=2, sort_keys=True)}</pre>",
            "<h2>Ranked Summary</h2>",
            ranked.to_html(index=False),
            "<h2>Manifest</h2>",
            f"<pre>{json.dumps(manifest, indent=2, sort_keys=True)}</pre>",
            "</body></html>",
        ]
    )
    (output_path / "fixed_blend_report.html").write_text(html, encoding="utf-8")
```

- [ ] **Step 4: Run artifact test and verify it passes**

Run:

```bash
uv run --frozen pytest src/tests/backtesting/test_fixed_blend_diagnostic.py::test_run_fixed_blend_diagnostic_writes_required_artifacts -q
```

Expected: test passes.

- [ ] **Step 5: Commit**

```bash
git add src/cartola/backtesting/fixed_blend_diagnostic.py src/tests/backtesting/test_fixed_blend_diagnostic.py
git commit -m "feat: write fixed blend diagnostic artifacts"
```

### Task 6: CLI Wrapper

**Files:**
- Create: `scripts/run_fixed_blend_diagnostic.py`
- Test: `src/tests/backtesting/test_run_fixed_blend_diagnostic_cli.py`

- [ ] **Step 1: Write failing CLI tests**

Create:

```python
from __future__ import annotations

from pathlib import Path

from scripts.run_fixed_blend_diagnostic import parse_args


def test_parse_args_accepts_required_fixed_blend_inputs() -> None:
    args = parse_args(
        [
            "--experiment-path",
            "data/08_reporting/experiments/model_feature/source",
            "--seasons",
            "2021,2022,2023,2024,2025",
            "--feature-pack",
            "ppg_xg",
            "--control-model",
            "xgboost_depth2_l2_heavy",
            "--blend",
            "xgb80_ridge20=xgboost_depth2_l2_heavy:0.8,ridge:0.2",
            "--initial-budget",
            "100",
            "--current-year",
            "2026",
        ]
    )

    assert args.experiment_path == Path("data/08_reporting/experiments/model_feature/source")
    assert args.seasons == "2021,2022,2023,2024,2025"
    assert args.feature_pack == "ppg_xg"
    assert args.control_model == "xgboost_depth2_l2_heavy"
    assert args.blend == ["xgb80_ridge20=xgboost_depth2_l2_heavy:0.8,ridge:0.2"]
    assert args.initial_budget == 100.0
    assert args.current_year == 2026
```

- [ ] **Step 2: Run CLI tests and verify they fail**

Run:

```bash
uv run --frozen pytest src/tests/backtesting/test_run_fixed_blend_diagnostic_cli.py -q
```

Expected: import failure for `scripts.run_fixed_blend_diagnostic`.

- [ ] **Step 3: Implement CLI**

Create:

```python
#!/usr/bin/env python
from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from rich.console import Console

run_fixed_blend_diagnostic: Callable[..., Path] | None = None
parse_blend_specs: Callable[[tuple[str, ...]], tuple[Any, ...]] | None = None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run M006 fixed blend diagnostic.")
    parser.add_argument("--experiment-path", type=Path, required=True)
    parser.add_argument("--seasons", default="2021,2022,2023,2024,2025")
    parser.add_argument("--feature-pack", default="ppg_xg")
    parser.add_argument("--control-model", default="xgboost_depth2_l2_heavy")
    parser.add_argument("--blend", action="append", required=True)
    parser.add_argument("--initial-budget", type=float, default=100.0)
    parser.add_argument("--current-year", type=int, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("data/08_reporting/blend_diagnostics"))
    return parser.parse_args(argv)


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _bootstrap_dotenv(project_root: Path | None = None) -> bool:
    resolved_project_root = _project_root() if project_root is None else project_root
    dotenv_path = resolved_project_root.expanduser() / ".env"
    if not dotenv_path.is_file():
        return False
    load_dotenv(dotenv_path=dotenv_path, override=False)
    return True


def _load_runtime_dependencies() -> None:
    global parse_blend_specs
    global run_fixed_blend_diagnostic
    if run_fixed_blend_diagnostic is None or parse_blend_specs is None:
        from cartola.backtesting.fixed_blend_diagnostic import (
            parse_blend_specs as imported_parse_blend_specs,
        )
        from cartola.backtesting.fixed_blend_diagnostic import (
            run_fixed_blend_diagnostic as imported_run_fixed_blend_diagnostic,
        )

        parse_blend_specs = imported_parse_blend_specs
        run_fixed_blend_diagnostic = imported_run_fixed_blend_diagnostic


def _parse_seasons(value: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    _bootstrap_dotenv()
    _load_runtime_dependencies()
    if run_fixed_blend_diagnostic is None or parse_blend_specs is None:
        raise RuntimeError("Fixed blend runtime dependencies were not loaded.")
    specs = parse_blend_specs(tuple(args.blend))
    console = Console()
    output_path = run_fixed_blend_diagnostic(
        experiment_path=args.experiment_path,
        output_root=args.output_root,
        seasons=_parse_seasons(args.seasons),
        feature_pack=args.feature_pack,
        control_model=args.control_model,
        specs=specs,
        initial_budget=args.initial_budget,
        current_year=args.current_year,
    )
    console.print(f"M006 fixed blend diagnostic complete: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run CLI tests and verify they pass**

Run:

```bash
uv run --frozen pytest src/tests/backtesting/test_run_fixed_blend_diagnostic_cli.py -q
```

Expected: CLI tests pass.

- [ ] **Step 5: Commit**

```bash
git add scripts/run_fixed_blend_diagnostic.py src/tests/backtesting/test_run_fixed_blend_diagnostic_cli.py
git commit -m "feat: add fixed blend diagnostic CLI"
```

### Task 7: AGENTS Workflow Documentation

**Files:**
- Modify: `AGENTS.md`

- [ ] **Step 1: Add M006 workflow section**

Insert after `Model Experiment Workflow` or before `Policy Simulation Workflow`:

```markdown
## M006 Fixed Blend Diagnostic Workflow

- Artifact-backed fixed XGB/Ridge blend diagnostic for a completed same-generation production-parity experiment:
  `uv run --frozen python scripts/run_fixed_blend_diagnostic.py --experiment-path data/08_reporting/experiments/model_feature/group=production-parity__started_at=20260513T165550180815Z__matrix=9064290978ca --seasons 2021,2022,2023,2024,2025 --feature-pack ppg_xg --control-model xgboost_depth2_l2_heavy --blend xgb90_ridge10=xgboost_depth2_l2_heavy:0.9,ridge:0.1 --blend xgb80_ridge20=xgboost_depth2_l2_heavy:0.8,ridge:0.2 --blend xgb70_ridge30=xgboost_depth2_l2_heavy:0.7,ridge:0.3 --initial-budget 100 --current-year 2026`
- M006 is a leak-safe fixed-blend diagnostic. It does not train a learned stacker, does not run AutoML, and does not use 2026 live outcomes as promotion evidence.
- Treat `candidate_blend` as promotion-quality research evidence only after the source control reproduction, candidate identity, moving-budget, calibration, budget, and disaster-risk gates all pass.
- If all fixed blends are rejected or inconclusive, do not build RF gating; redirect effort to live guardrails, captain policy, or risk-aware optimization.
```

- [ ] **Step 2: Commit**

```bash
git add AGENTS.md
git commit -m "docs: document M006 fixed blend workflow"
```

### Task 8: End-To-End Verification

**Files:**
- No source file edits unless verification exposes a bug.

- [ ] **Step 1: Run targeted tests**

Run:

```bash
uv run --frozen pytest src/tests/backtesting/test_fixed_blend_diagnostic.py src/tests/backtesting/test_run_fixed_blend_diagnostic_cli.py -q
```

Expected: all targeted tests pass.

- [ ] **Step 2: Run annotation gate**

Run:

```bash
uv run --frozen ruff check src/cartola src/tests scripts --select ANN
```

Expected: no ANN violations.

- [ ] **Step 3: Run full quality gate**

Run:

```bash
uv run --frozen scripts/pyrepo-check --all
```

Expected: Ruff, Ty, Bandit, and Pytest pass.

- [ ] **Step 4: Run the real M006 diagnostic**

Run:

```bash
uv run --frozen python scripts/run_fixed_blend_diagnostic.py \
  --experiment-path data/08_reporting/experiments/model_feature/group=production-parity__started_at=20260513T165550180815Z__matrix=9064290978ca \
  --seasons 2021,2022,2023,2024,2025 \
  --feature-pack ppg_xg \
  --control-model xgboost_depth2_l2_heavy \
  --blend xgb90_ridge10=xgboost_depth2_l2_heavy:0.9,ridge:0.1 \
  --blend xgb80_ridge20=xgboost_depth2_l2_heavy:0.8,ridge:0.2 \
  --blend xgb70_ridge30=xgboost_depth2_l2_heavy:0.7,ridge:0.3 \
  --initial-budget 100 \
  --current-year 2026
```

Expected: command completes and prints `M006 fixed blend diagnostic complete: data/08_reporting/blend_diagnostics/fixed_blend_started_at=...`.

- [ ] **Step 5: Inspect decision artifact**

Run:

```bash
BLEND_DIR="$(ls -td data/08_reporting/blend_diagnostics/fixed_blend_started_at=* | head -1)"
cat "$BLEND_DIR/blend_decision.json"
```

Expected: `decision_status` is one of `candidate_blend`, `weak_positive_research_lead`, `inconclusive`, `rejected`, or `invalid`; `invalid_rows.csv` is empty unless `decision_status` is `invalid`.

- [ ] **Step 6: Commit verification fixes only if needed**

If verification required edits:

```bash
git add src/cartola/backtesting/fixed_blend_diagnostic.py scripts/run_fixed_blend_diagnostic.py src/tests/backtesting/test_fixed_blend_diagnostic.py src/tests/backtesting/test_run_fixed_blend_diagnostic_cli.py AGENTS.md
git commit -m "fix: harden fixed blend diagnostic"
```

## Self-Review Checklist

- M006 starts with artifact-backed fixed blends, not learned stacking.
- The exact source experiment path is recorded.
- Candidate identity is checked across base models before any blend score is trusted.
- Moving budget restarts at `100` at the beginning of each historical season and then advances independently for each blend.
- The plan reuses `optimize_squad`, `advance_budget`, and `actual_scores_with_captain`.
- Decision gates include aggregate points, season consistency, 2025 regression, budget path, calibration, top-50 ranking, disaster rounds, worst two-round stretch, and concentration.
- AGENTS documentation includes the runnable command and research-only caution.
- Verification includes targeted tests, ANN, full `pyrepo-check`, and the real diagnostic command.

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-05-13-m006-fixed-blend-diagnostic.md`. Two execution options:

1. **Subagent-Driven (recommended)** - dispatch a fresh subagent per task, review between tasks, fast iteration.
2. **Inline Execution** - execute tasks in this session using executing-plans, batch execution with checkpoints.

Which approach?
