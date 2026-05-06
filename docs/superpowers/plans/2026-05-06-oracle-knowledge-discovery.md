# Oracle Knowledge Discovery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an artifact-backed, discovery-only oracle diagnostics report for Cartola model experiment outputs.

**Architecture:** Implement a standalone `oracle_discovery` module that validates source artifacts and source-run context before computing any oracle metrics. V1 uses persisted `player_predictions.csv`, `selected_players.csv`, `round_results.csv`, and metadata; it does not rebuild model frames, does not run normal experiments, and does not implement full-market or reconstructed mode in the first pass.

**Tech Stack:** Python 3.13, pandas, existing `optimize_squad`, existing scoring contract helpers, uv/pytest/ruff/ty.

---

## Required V1 Contracts

Source mode:

- V1 implements `source_mode=artifact` only.
- Reconstructed mode, full-market oracle, unlimited-budget oracle, independent oracle budget path, deterministic hypothesis generation, and policy learning are explicitly deferred.
- Normal experiment runners must not import or call oracle discovery.

Source-run context:

- Every analyzed child run must resolve a validated `SourceRunContext` before reading player rows.
- Required fields: `source_experiment_id`, `source_child_id`, `source_child_path`, `season`, `model_id`, `feature_pack`, `fixture_mode`, `matchup_context_mode`, `budget_policy`, `primary_strategy`, `strategy_score_columns`, `analyzed_strategies`.
- `strategy_score_columns` may be constructed from parent experiment `strategy_roles` only with this deterministic rule:
  - `baseline -> baseline_score`
  - `price -> price_score`
  - primary model strategy -> {model_id}_score
- Every mapped score column must exist in `player_predictions.csv`; non-standard strategies require explicit mapping and fail without it.
- Missing `budget_policy` or any non-`moving` policy means the source artifact is old fixed-budget evidence and must be rejected.

DNP/null objective policy:

- Finite `pontuacao` becomes `oracle_actual_points`.
- Missing `pontuacao` with `entrou_em_campo == false` becomes `oracle_actual_points = 0.0`.
- Missing `pontuacao` with absent, null, or ambiguous `entrou_em_campo` is invalid.
- Non-finite or non-numeric `pontuacao` is invalid.
- Strict mode fails affected rounds on invalid rows; permissive mode records invalid rows and disables affected sections instead of emitting zero-valued metrics.

Rank semantics:

- Prediction ranks are descending by the validated strategy score column.
- Overall rank is within `season + rodada + strategy`.
- Position rank is within `season + rodada + strategy + posicao`.
- Ties use pandas `rank(method="min", ascending=False)` for a deterministic shared rank.
- `tec` rows participate in player-selection rank, but captain-specific comparisons exclude `tec`.

Shared identity columns:

```text
source_mode, source_experiment_id, source_child_id, season, rodada,
model_id, feature_pack, fixture_mode, matchup_context_mode, budget_policy,
oracle_type, candidate_universe, budget_path
```

`oracle_round_results.csv` columns:

```text
<shared identity columns>,
optimizer_status, optimizer_formation, optimizer_budget_used,
budget_before_round, oracle_actual_points_base,
oracle_captain_bonus_actual, oracle_actual_points_with_captain,
optimizer_captain_id, optimizer_selected_count, full_market_status
```

`oracle_selected_players.csv` columns:

```text
<shared identity columns>,
id_atleta, apelido, posicao, id_clube, nome_clube, preco_pre_rodada,
oracle_actual_points, is_oracle_captain, model_score_column, model_score,
model_predicted_rank_overall, model_predicted_rank_position,
entrou_em_campo, status
```

`oracle_captain_profiles.csv` columns:

```text
<shared identity columns>,
captain_id, captain_name, captain_position, captain_club, captain_status,
captain_is_home, captain_price_percentile_position, captain_price_rank_position,
captain_model_score, captain_model_predicted_rank_overall,
captain_model_predicted_rank_position, captain_recent_form_percentile_position,
captain_oracle_actual_points, model_captain_id, model_captain_actual_points,
selected_squad_captain_regret, full_market_status
```

`oracle_player_profiles.csv` columns:

```text
<shared identity columns>,
id_atleta, posicao, profile_section, profile_metric, profile_value,
baseline_name, baseline_value, sample_size, full_market_status
```

`model_vs_oracle_recall.csv` columns:

```text
<shared identity columns>,
id_atleta, posicao, in_selected_squad, in_model_candidate_artifact,
absent_from_model_candidate_artifact, in_full_market, full_market_status,
model_predicted_rank_overall, model_predicted_rank_position,
individually_affordable, squad_budget_blocked_by_counterfactual, recall_bucket
```

`profile_gap_summary.csv` columns:

```text
<shared identity columns>,
profile_section, profile_metric, oracle_value, baseline_name, baseline_value,
absolute_gap, relative_gap, sample_size, season_stability_count,
stability_label, full_market_status
```

`invalid_oracle_rows.csv` columns:

```text
<shared identity columns when available>,
id_atleta, posicao, pontuacao, entrou_em_campo, invalid_reason
```

Full-market language:

- V1 must write `full_market_status=not_available`.
- V1 must use `absent_from_model_candidate_artifact`, not “not visible”, “not eligible”, or “candidate-generation failure”.
- Any chart or table that lacks full-market support must label that limitation.

## File Structure

- Create `src/cartola/backtesting/oracle_discovery.py`
  - Dataclasses for source context, loaded artifacts, oracle outputs.
  - Strict artifact/schema validation.
  - DNP/null objective policy.
  - Oracle result adapter around `optimize_squad`.
  - Selected-squad captain oracle.
  - Model-candidate oracle.
  - CSV/HTML artifact writer.

- Create `scripts/run_oracle_knowledge_discovery.py`
  - CLI wrapper.
  - Accepts `--experiment-path`, `--output-root`, `--models`, `--seasons`, `--allow-incomplete`.
  - Calls `build_oracle_discovery_report`.

- Create `src/tests/backtesting/test_oracle_discovery.py`
  - Unit tests for validators, source context, score mapping, DNP policy, adapter, oracles, output schemas.

- Create `src/tests/backtesting/test_run_oracle_knowledge_discovery_cli.py`
  - CLI import isolation and argument tests.

---

### Task 1: Source Context And Schema Validation

**Files:**
- Create: `src/cartola/backtesting/oracle_discovery.py`
- Create: `src/tests/backtesting/test_oracle_discovery.py`

- [ ] **Step 1: Write failing tests for source context and schema validation**

Add to `src/tests/backtesting/test_oracle_discovery.py`:

```python
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from cartola.backtesting.oracle_discovery import (
    ArtifactValidationError,
    SourceRunContext,
    load_source_run_contexts,
    validate_child_artifacts,
)


def _write_csv(path: Path, columns: list[str]) -> None:
    pd.DataFrame([{column: 1 for column in columns}]).to_csv(path, index=False)


def _valid_child_dir(tmp_path: Path) -> Path:
    child = tmp_path / "runs" / "season=2025" / "model=xgboost_depth2_l2_heavy" / "feature_pack=ppg_xg_matchup"
    child.mkdir(parents=True)
    _write_csv(
        child / "round_results.csv",
        [
            "rodada",
            "strategy",
            "solver_status",
            "budget_before_round",
            "budget_after_round",
            "budget_delta",
            "budget_used",
            "actual_points_with_captain",
            "captain_id",
        ],
    )
    _write_csv(
        child / "selected_players.csv",
        [
            "rodada",
            "strategy",
            "id_atleta",
            "apelido",
            "posicao",
            "id_clube",
            "nome_clube",
            "entrou_em_campo",
            "preco_pre_rodada",
            "pontuacao",
            "variacao",
            "is_captain",
        ],
    )
    _write_csv(
        child / "player_predictions.csv",
        [
            "rodada",
            "id_atleta",
            "apelido",
            "posicao",
            "id_clube",
            "nome_clube",
            "status",
            "entrou_em_campo",
            "preco_pre_rodada",
            "pontuacao",
            "variacao",
            "baseline_score",
            "price_score",
            "xgboost_depth2_l2_heavy_score",
        ],
    )
    _write_csv(child / "summary.csv", ["strategy", "rounds"])
    (child / "run_metadata.json").write_text(
        json.dumps(
            {
                "season": 2025,
                "start_round": 5,
                "initial_budget": 100.0,
                "budget_policy": "moving",
                "fixture_mode": "exploratory",
                "matchup_context_mode": "cartola_matchup_v1",
                "footystats_mode": "ppg_xg",
                "scoring_contract_version": "cartola_standard_2026_v1",
                "fixture_source_directory": "data/01_raw/fixtures/2025",
                "fixture_manifest_sha256": {},
            }
        ),
        encoding="utf-8",
    )
    return child


def test_load_source_run_contexts_derives_score_columns_from_parent_metadata(tmp_path: Path) -> None:
    child = _valid_child_dir(tmp_path)
    experiment = tmp_path
    (experiment / "experiment_metadata.json").write_text(
        json.dumps(
            {
                "experiment_id": "exp-1",
                "child_runs": [
                    {
                        "child_id": "season=2025/model=xgboost_depth2_l2_heavy/feature_pack=ppg_xg_matchup",
                        "output_path": str(child),
                        "season": 2025,
                        "model_id": "xgboost_depth2_l2_heavy",
                        "feature_pack": "ppg_xg_matchup",
                        "fixture_mode": "exploratory",
                        "matchup_context_mode": "cartola_matchup_v1",
                        "budget_policy": "moving",
                        "strategy_roles": {
                            "baseline": "baseline",
                            "price": "price",
                            "xgboost_depth2_l2_heavy": "primary_model",
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    contexts = load_source_run_contexts(experiment)

    assert contexts == [
        SourceRunContext(
            source_experiment_id="exp-1",
            source_child_id="season=2025/model=xgboost_depth2_l2_heavy/feature_pack=ppg_xg_matchup",
            source_child_path=child,
            season=2025,
            model_id="xgboost_depth2_l2_heavy",
            feature_pack="ppg_xg_matchup",
            fixture_mode="exploratory",
            matchup_context_mode="cartola_matchup_v1",
            budget_policy="moving",
            primary_strategy="xgboost_depth2_l2_heavy",
            strategy_score_columns={
                "baseline": "baseline_score",
                "price": "price_score",
                "xgboost_depth2_l2_heavy": "xgboost_depth2_l2_heavy_score",
            },
            analyzed_strategies=("baseline", "price", "xgboost_depth2_l2_heavy"),
        )
    ]


def test_validate_child_artifacts_rejects_missing_score_column(tmp_path: Path) -> None:
    child = _valid_child_dir(tmp_path)
    predictions = pd.read_csv(child / "player_predictions.csv").drop(columns=["xgboost_depth2_l2_heavy_score"])
    predictions.to_csv(child / "player_predictions.csv", index=False)
    context = SourceRunContext(
        source_experiment_id="exp-1",
        source_child_id="child-1",
        source_child_path=child,
        season=2025,
        model_id="xgboost_depth2_l2_heavy",
        feature_pack="ppg_xg_matchup",
        fixture_mode="exploratory",
        matchup_context_mode="cartola_matchup_v1",
        budget_policy="moving",
        primary_strategy="xgboost_depth2_l2_heavy",
        strategy_score_columns={"xgboost_depth2_l2_heavy": "xgboost_depth2_l2_heavy_score"},
        analyzed_strategies=("xgboost_depth2_l2_heavy",),
    )

    with pytest.raises(ArtifactValidationError, match="xgboost_depth2_l2_heavy_score"):
        validate_child_artifacts(context)


def test_validate_child_artifacts_rejects_old_fixed_budget_artifacts(tmp_path: Path) -> None:
    child = _valid_child_dir(tmp_path)
    round_results = pd.read_csv(child / "round_results.csv").drop(columns=["budget_before_round"])
    round_results.to_csv(child / "round_results.csv", index=False)
    context = SourceRunContext(
        source_experiment_id="exp-1",
        source_child_id="child-1",
        source_child_path=child,
        season=2025,
        model_id="ridge",
        feature_pack="ppg_xg",
        fixture_mode="none",
        matchup_context_mode="none",
        budget_policy="fixed",
        primary_strategy="ridge",
        strategy_score_columns={"ridge": "ridge_score"},
        analyzed_strategies=("ridge",),
    )

    with pytest.raises(ArtifactValidationError, match="not moving-budget compatible"):
        validate_child_artifacts(context)
```

- [ ] **Step 2: Run tests and verify they fail**

Run:

```bash
uv run --frozen pytest src/tests/backtesting/test_oracle_discovery.py -q
```

Expected: FAIL because `cartola.backtesting.oracle_discovery` does not exist.

- [ ] **Step 3: Implement source context and artifact validation**

Create `src/cartola/backtesting/oracle_discovery.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import pandas as pd


class ArtifactValidationError(ValueError):
    pass


@dataclass(frozen=True)
class SourceRunContext:
    source_experiment_id: str
    source_child_id: str
    source_child_path: Path
    season: int
    model_id: str
    feature_pack: str
    fixture_mode: str
    matchup_context_mode: str
    budget_policy: str
    primary_strategy: str
    strategy_score_columns: dict[str, str]
    analyzed_strategies: tuple[str, ...]


@dataclass(frozen=True)
class ChildArtifacts:
    round_results: pd.DataFrame
    selected_players: pd.DataFrame
    player_predictions: pd.DataFrame
    summary: pd.DataFrame
    metadata: dict[str, Any]


ROUND_RESULTS_COLUMNS = frozenset(
    {
        "rodada",
        "strategy",
        "solver_status",
        "budget_before_round",
        "budget_after_round",
        "budget_delta",
        "budget_used",
        "actual_points_with_captain",
        "captain_id",
    }
)
SELECTED_PLAYERS_COLUMNS = frozenset(
    {
        "rodada",
        "strategy",
        "id_atleta",
        "apelido",
        "posicao",
        "id_clube",
        "nome_clube",
        "entrou_em_campo",
        "preco_pre_rodada",
        "pontuacao",
        "variacao",
        "is_captain",
    }
)
PLAYER_PREDICTIONS_COLUMNS = frozenset(
    {
        "rodada",
        "id_atleta",
        "apelido",
        "posicao",
        "id_clube",
        "nome_clube",
        "status",
        "entrou_em_campo",
        "preco_pre_rodada",
        "pontuacao",
        "variacao",
    }
)
METADATA_FIELDS = frozenset(
    {
        "season",
        "start_round",
        "initial_budget",
        "budget_policy",
        "fixture_mode",
        "matchup_context_mode",
        "footystats_mode",
        "scoring_contract_version",
        "fixture_source_directory",
        "fixture_manifest_sha256",
    }
)


def load_source_run_contexts(experiment_path: Path) -> list[SourceRunContext]:
    metadata_path = experiment_path / "experiment_metadata.json"
    payload = _read_json_object(metadata_path)
    experiment_id = str(payload["experiment_id"])
    contexts: list[SourceRunContext] = []
    for child in payload.get("child_runs", []):
        if not isinstance(child, dict):
            raise ArtifactValidationError("experiment_metadata child_runs must contain objects")
        model_id = str(child["model_id"])
        strategy_roles = child.get("strategy_roles", {})
        if not isinstance(strategy_roles, dict):
            raise ArtifactValidationError("strategy_roles must be an object")
        analyzed = tuple(str(strategy) for strategy in strategy_roles)
        score_columns = _score_columns_from_roles(model_id=model_id, strategy_roles=strategy_roles)
        contexts.append(
            SourceRunContext(
                source_experiment_id=experiment_id,
                source_child_id=str(child["child_id"]),
                source_child_path=Path(str(child["output_path"])),
                season=int(child["season"]),
                model_id=model_id,
                feature_pack=str(child["feature_pack"]),
                fixture_mode=str(child["fixture_mode"]),
                matchup_context_mode=str(child["matchup_context_mode"]),
                budget_policy=str(child["budget_policy"]),
                primary_strategy=model_id,
                strategy_score_columns=score_columns,
                analyzed_strategies=analyzed,
            )
        )
    return contexts


def validate_child_artifacts(context: SourceRunContext) -> ChildArtifacts:
    if context.budget_policy != "moving":
        raise ArtifactValidationError(f"Source child is not moving-budget compatible: {context.source_child_id}")
    child_path = context.source_child_path
    round_results = _read_required_csv(child_path / "round_results.csv")
    selected_players = _read_required_csv(child_path / "selected_players.csv")
    player_predictions = _read_required_csv(child_path / "player_predictions.csv")
    summary = _read_required_csv(child_path / "summary.csv")
    metadata = _read_json_object(child_path / "run_metadata.json")

    _require_columns("round_results.csv", round_results, ROUND_RESULTS_COLUMNS)
    _require_columns("selected_players.csv", selected_players, SELECTED_PLAYERS_COLUMNS)
    _require_columns("player_predictions.csv", player_predictions, PLAYER_PREDICTIONS_COLUMNS)
    _require_metadata(metadata)
    if metadata.get("budget_policy") != "moving":
        raise ArtifactValidationError(f"Source child is not moving-budget compatible: {context.source_child_id}")
    for strategy in context.analyzed_strategies:
        score_column = context.strategy_score_columns.get(strategy)
        if score_column is None:
            raise ArtifactValidationError(f"Missing score-column mapping for strategy: {strategy}")
        if score_column not in player_predictions.columns:
            raise ArtifactValidationError(f"Missing score column in player_predictions.csv: {score_column}")
    return ChildArtifacts(
        round_results=round_results,
        selected_players=selected_players,
        player_predictions=player_predictions,
        summary=summary,
        metadata=metadata,
    )


def _score_columns_from_roles(*, model_id: str, strategy_roles: dict[str, object]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for strategy, role in strategy_roles.items():
        strategy_id = str(strategy)
        role_id = str(role)
        if strategy_id == "baseline":
            mapping[strategy_id] = "baseline_score"
        elif strategy_id == "price":
            mapping[strategy_id] = "price_score"
        elif role_id == "primary_model" or strategy_id == model_id:
            mapping[strategy_id] = f"{model_id}_score"
        else:
            raise ArtifactValidationError(f"Non-standard strategy requires explicit score mapping: {strategy_id}")
    return mapping


def _read_json_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ArtifactValidationError(f"Missing required JSON artifact: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ArtifactValidationError(f"JSON artifact must contain an object: {path}")
    return payload


def _read_required_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise ArtifactValidationError(f"Missing required CSV artifact: {path}")
    return pd.read_csv(path)


def _require_columns(name: str, frame: pd.DataFrame, required: frozenset[str]) -> None:
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ArtifactValidationError(f"Missing required columns in {name}: {missing}")


def _require_metadata(metadata: dict[str, Any]) -> None:
    missing = sorted(METADATA_FIELDS.difference(metadata))
    if missing:
        raise ArtifactValidationError(f"Missing required fields in run_metadata.json: {missing}")
```

- [ ] **Step 4: Run tests and verify they pass**

Run:

```bash
uv run --frozen pytest src/tests/backtesting/test_oracle_discovery.py -q
```

Expected: PASS for the three validation tests.

---

### Task 2: DNP Policy And Oracle Result Adapter

**Files:**
- Modify: `src/cartola/backtesting/oracle_discovery.py`
- Modify: `src/tests/backtesting/test_oracle_discovery.py`

- [ ] **Step 1: Add failing tests for DNP/null policy and result adapter**

Append to `src/tests/backtesting/test_oracle_discovery.py`:

```python
from cartola.backtesting.optimizer import SquadOptimizationResult
from cartola.backtesting.oracle_discovery import (
    OracleObjectiveError,
    adapt_oracle_result,
    add_oracle_actual_points,
)


def test_add_oracle_actual_points_maps_explicit_dnp_null_to_zero() -> None:
    frame = pd.DataFrame(
        [
            {"id_atleta": 1, "pontuacao": None, "entrou_em_campo": False},
            {"id_atleta": 2, "pontuacao": 7.5, "entrou_em_campo": True},
        ]
    )

    result = add_oracle_actual_points(frame)

    assert result["oracle_actual_points"].tolist() == [0.0, 7.5]


def test_add_oracle_actual_points_rejects_ambiguous_null() -> None:
    frame = pd.DataFrame([{"id_atleta": 1, "pontuacao": None, "entrou_em_campo": None}])

    with pytest.raises(OracleObjectiveError, match="Ambiguous missing pontuacao"):
        add_oracle_actual_points(frame)


def test_adapt_oracle_result_renames_prediction_fields() -> None:
    result = SquadOptimizationResult(
        selected=pd.DataFrame(),
        status="Optimal",
        budget_used=98.0,
        predicted_points=75.0,
        predicted_points_base=70.0,
        captain_bonus_predicted=5.0,
        predicted_points_with_captain=75.0,
        formation_name="4-3-3",
        selected_count=12,
        captain_id=10,
        captain_name="A",
        captain_position="ata",
        captain_club="FLA",
        captain_predicted_points=10.0,
        captain_multiplier=1.5,
        scoring_contract_version="cartola_standard_2026_v1",
        formation_scores=[],
        captain_policy_diagnostics=[],
    )

    adapted = adapt_oracle_result(result)

    assert adapted["oracle_actual_points_base"] == 70.0
    assert adapted["oracle_captain_bonus_actual"] == 5.0
    assert adapted["oracle_actual_points_with_captain"] == 75.0
    assert adapted["optimizer_status"] == "Optimal"
    assert "predicted_points" not in adapted
```

- [ ] **Step 2: Run tests and verify they fail**

Run:

```bash
uv run --frozen pytest src/tests/backtesting/test_oracle_discovery.py::test_add_oracle_actual_points_maps_explicit_dnp_null_to_zero src/tests/backtesting/test_oracle_discovery.py::test_add_oracle_actual_points_rejects_ambiguous_null src/tests/backtesting/test_oracle_discovery.py::test_adapt_oracle_result_renames_prediction_fields -q
```

Expected: FAIL because functions/classes are not implemented.

- [ ] **Step 3: Implement DNP policy and adapter**

Add to `src/cartola/backtesting/oracle_discovery.py`:

```python
import numpy as np

from cartola.backtesting.optimizer import SquadOptimizationResult


class OracleObjectiveError(ValueError):
    pass


def add_oracle_actual_points(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"pontuacao", "entrou_em_campo"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise OracleObjectiveError(f"Missing required oracle objective columns: {missing}")
    output = frame.copy()
    points = pd.to_numeric(output["pontuacao"], errors="coerce")
    entered = output["entrou_em_campo"].map(_bool_or_none)
    oracle_points: list[float] = []
    invalid_ids: list[object] = []
    for index, value in points.items():
        if pd.notna(value) and np.isfinite(float(value)):
            oracle_points.append(float(value))
            continue
        entry_value = entered.loc[index]
        if entry_value is False:
            oracle_points.append(0.0)
            continue
        invalid_ids.append(output.loc[index, "id_atleta"] if "id_atleta" in output.columns else index)
        oracle_points.append(float("nan"))
    if invalid_ids:
        raise OracleObjectiveError(f"Ambiguous missing pontuacao for rows: {invalid_ids}")
    output["oracle_actual_points"] = oracle_points
    return output


def adapt_oracle_result(result: SquadOptimizationResult) -> dict[str, object]:
    return {
        "optimizer_status": result.status,
        "optimizer_formation": result.formation_name,
        "optimizer_budget_used": result.budget_used,
        "optimizer_selected_count": result.selected_count,
        "optimizer_captain_id": result.captain_id,
        "oracle_actual_points_base": result.predicted_points_base,
        "oracle_captain_bonus_actual": result.captain_bonus_predicted,
        "oracle_actual_points_with_captain": result.predicted_points_with_captain,
        "oracle_objective_points": result.predicted_points,
    }


def _bool_or_none(value: object) -> bool | None:
    if pd.isna(value):
        return None
    if value in (True, 1, "1"):
        return True
    if value in (False, 0, "0", ""):
        return False
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered == "true":
            return True
        if lowered == "false":
            return False
    return None
```

- [ ] **Step 4: Run tests and verify they pass**

Run:

```bash
uv run --frozen pytest src/tests/backtesting/test_oracle_discovery.py -q
```

Expected: all oracle discovery tests pass.

---

### Task 3: Selected-Squad Captain Oracle

**Files:**
- Modify: `src/cartola/backtesting/oracle_discovery.py`
- Modify: `src/tests/backtesting/test_oracle_discovery.py`

- [ ] **Step 1: Add failing tests for selected-squad captain oracle**

Append to `src/tests/backtesting/test_oracle_discovery.py`:

```python
from cartola.backtesting.oracle_discovery import selected_squad_captain_oracle


def test_selected_squad_captain_oracle_ignores_tecnico_and_computes_regret() -> None:
    selected = pd.DataFrame(
        [
            {
                "id_atleta": 1,
                "apelido": "Coach",
                "posicao": "tec",
                "nome_clube": "A",
                "pontuacao": 20.0,
                "entrou_em_campo": True,
                "is_captain": False,
            },
            {
                "id_atleta": 2,
                "apelido": "Chosen",
                "posicao": "mei",
                "nome_clube": "A",
                "pontuacao": 8.0,
                "entrou_em_campo": True,
                "is_captain": True,
            },
            {
                "id_atleta": 3,
                "apelido": "Best",
                "posicao": "ata",
                "nome_clube": "B",
                "pontuacao": 14.0,
                "entrou_em_campo": True,
                "is_captain": False,
            },
        ]
    )

    result = selected_squad_captain_oracle(selected)

    assert result["captain_id"] == 3
    assert result["selected_squad_captain_regret"] == 3.0
```

- [ ] **Step 2: Run test and verify it fails**

Run:

```bash
uv run --frozen pytest src/tests/backtesting/test_oracle_discovery.py::test_selected_squad_captain_oracle_ignores_tecnico_and_computes_regret -q
```

Expected: FAIL because `selected_squad_captain_oracle` is missing.

- [ ] **Step 3: Implement selected-squad captain oracle**

Add to `src/cartola/backtesting/oracle_discovery.py`:

```python
from cartola.backtesting.scoring_contract import CAPTAIN_MULTIPLIER


def selected_squad_captain_oracle(selected: pd.DataFrame) -> dict[str, object]:
    scored = add_oracle_actual_points(selected)
    captain_rows = scored.loc[scored["is_captain"].eq(True)]
    if len(captain_rows) != 1:
        raise OracleObjectiveError(f"Selected squad must contain exactly one captain, got {len(captain_rows)}")
    eligible = scored.loc[scored["posicao"].ne("tec")].copy()
    if eligible.empty:
        raise OracleObjectiveError("Selected squad has no non-tecnico captain candidates")
    best = eligible.sort_values(
        ["oracle_actual_points", "id_atleta"],
        ascending=[False, True],
        kind="mergesort",
    ).iloc[0]
    selected_captain = captain_rows.iloc[0]
    regret = (CAPTAIN_MULTIPLIER - 1.0) * (
        float(best["oracle_actual_points"]) - float(selected_captain["oracle_actual_points"])
    )
    return {
        "captain_id": int(best["id_atleta"]),
        "captain_name": str(best["apelido"]),
        "captain_position": str(best["posicao"]),
        "captain_club": str(best["nome_clube"]),
        "captain_oracle_actual_points": float(best["oracle_actual_points"]),
        "model_captain_id": int(selected_captain["id_atleta"]),
        "model_captain_actual_points": float(selected_captain["oracle_actual_points"]),
        "selected_squad_captain_regret": float(regret),
    }
```

- [ ] **Step 4: Run tests and verify they pass**

Run:

```bash
uv run --frozen pytest src/tests/backtesting/test_oracle_discovery.py -q
```

Expected: PASS.

---

### Task 4: Model-Candidate Oracle

**Files:**
- Modify: `src/cartola/backtesting/oracle_discovery.py`
- Modify: `src/tests/backtesting/test_oracle_discovery.py`

- [ ] **Step 1: Add failing tests for model-candidate oracle**

Append to `src/tests/backtesting/test_oracle_discovery.py`:

```python
from cartola.backtesting.config import BacktestConfig
from cartola.backtesting.oracle_discovery import run_model_candidate_oracle


def _candidate_rows() -> pd.DataFrame:
    rows = []
    positions = ["gol", "lat", "lat", "zag", "zag", "mei", "mei", "mei", "ata", "ata", "ata", "tec"]
    for index, position in enumerate(positions, start=1):
        rows.append(
            {
                "rodada": 5,
                "id_atleta": index,
                "apelido": f"P{index}",
                "posicao": position,
                "id_clube": 100 + index,
                "nome_clube": f"C{index}",
                "status": "Provavel",
                "entrou_em_campo": True,
                "preco_pre_rodada": 1.0,
                "pontuacao": float(index),
                "variacao": 0.0,
                "model_score": float(100 - index),
            }
        )
    return pd.DataFrame(rows)


def test_run_model_candidate_oracle_uses_actual_points_objective() -> None:
    candidates = _candidate_rows()
    cfg = BacktestConfig(season=2025, start_round=5, budget=100, project_root=Path("."))

    row, selected = run_model_candidate_oracle(
        candidates,
        config=cfg,
        budget_before_round=100.0,
        score_column="model_score",
    )

    assert row["optimizer_status"] == "Optimal"
    assert row["oracle_actual_points_with_captain"] == 84.0
    assert selected["is_oracle_captain"].sum() == 1
    assert int(selected.loc[selected["is_oracle_captain"], "id_atleta"].iloc[0]) == 11
```

- [ ] **Step 2: Run test and verify it fails**

Run:

```bash
uv run --frozen pytest src/tests/backtesting/test_oracle_discovery.py::test_run_model_candidate_oracle_uses_actual_points_objective -q
```

Expected: FAIL because `run_model_candidate_oracle` is missing.

- [ ] **Step 3: Implement model-candidate oracle**

Add to `src/cartola/backtesting/oracle_discovery.py`:

```python
from cartola.backtesting.config import BacktestConfig
from cartola.backtesting.optimizer import optimize_squad


def run_model_candidate_oracle(
    candidates: pd.DataFrame,
    *,
    config: BacktestConfig,
    budget_before_round: float,
    score_column: str,
) -> tuple[dict[str, object], pd.DataFrame]:
    oracle_candidates = add_oracle_actual_points(candidates)
    result = optimize_squad(
        oracle_candidates,
        score_column="oracle_actual_points",
        config=config,
        budget=budget_before_round,
    )
    row = adapt_oracle_result(result)
    selected = result.selected.copy()
    if selected.empty:
        selected["is_oracle_captain"] = pd.Series(dtype=bool)
        return row, selected
    selected = selected.rename(columns={"is_captain": "is_oracle_captain"})
    selected["model_score_column"] = score_column
    selected["model_score"] = selected[score_column]
    ranks = _prediction_ranks(oracle_candidates, score_column=score_column)
    selected = selected.merge(
        ranks,
        on=["rodada", "id_atleta"],
        how="left",
        validate="many_to_one",
    )
    return row, selected


def _prediction_ranks(candidates: pd.DataFrame, *, score_column: str) -> pd.DataFrame:
    ranked = candidates.loc[:, ["rodada", "id_atleta", "posicao", score_column]].copy()
    ranked["model_predicted_rank_overall"] = ranked.groupby("rodada")[score_column].rank(
        method="min",
        ascending=False,
    )
    ranked["model_predicted_rank_position"] = ranked.groupby(["rodada", "posicao"])[score_column].rank(
        method="min",
        ascending=False,
    )
    return ranked.loc[:, ["rodada", "id_atleta", "model_predicted_rank_overall", "model_predicted_rank_position"]]
```

- [ ] **Step 4: Run tests and verify they pass**

Run:

```bash
uv run --frozen pytest src/tests/backtesting/test_oracle_discovery.py -q
```

Expected: PASS.

---

### Task 5: Report Builder And CSV Artifacts

**Files:**
- Modify: `src/cartola/backtesting/oracle_discovery.py`
- Modify: `src/tests/backtesting/test_oracle_discovery.py`

- [ ] **Step 1: Add failing integration test for report builder**

Append to `src/tests/backtesting/test_oracle_discovery.py`:

```python
from cartola.backtesting.oracle_discovery import build_oracle_discovery_report


def test_build_oracle_discovery_report_writes_expected_artifacts(tmp_path: Path) -> None:
    child = _valid_child_dir(tmp_path)
    predictions = _candidate_rows()
    predictions["baseline_score"] = predictions["model_score"]
    predictions["price_score"] = predictions["preco_pre_rodada"]
    predictions["xgboost_depth2_l2_heavy_score"] = predictions["model_score"]
    predictions.to_csv(child / "player_predictions.csv", index=False)
    selected = predictions.head(12).copy()
    selected["strategy"] = "xgboost_depth2_l2_heavy"
    selected["is_captain"] = selected["id_atleta"].eq(10)
    selected.to_csv(child / "selected_players.csv", index=False)
    pd.DataFrame(
        [
            {
                "rodada": 5,
                "strategy": "xgboost_depth2_l2_heavy",
                "solver_status": "Optimal",
                "budget_before_round": 100.0,
                "budget_after_round": 100.0,
                "budget_delta": 0.0,
                "budget_used": 12.0,
                "actual_points_with_captain": 83.5,
                "captain_id": 10,
            }
        ]
    ).to_csv(child / "round_results.csv", index=False)
    experiment = tmp_path
    (experiment / "experiment_metadata.json").write_text(
        json.dumps(
            {
                "experiment_id": "exp-1",
                "child_runs": [
                    {
                        "child_id": "season=2025/model=xgboost_depth2_l2_heavy/feature_pack=ppg_xg_matchup",
                        "output_path": str(child),
                        "season": 2025,
                        "model_id": "xgboost_depth2_l2_heavy",
                        "feature_pack": "ppg_xg_matchup",
                        "fixture_mode": "exploratory",
                        "matchup_context_mode": "cartola_matchup_v1",
                        "budget_policy": "moving",
                        "strategy_roles": {"xgboost_depth2_l2_heavy": "primary_model"},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "oracle_out"

    build_oracle_discovery_report(experiment_path=experiment, output_path=output)

    assert (output / "oracle_round_results.csv").is_file()
    assert (output / "oracle_selected_players.csv").is_file()
    assert (output / "oracle_captain_profiles.csv").is_file()
    assert (output / "oracle_player_profiles.csv").is_file()
    assert (output / "model_vs_oracle_recall.csv").is_file()
    assert (output / "profile_gap_summary.csv").is_file()
    assert (output / "invalid_oracle_rows.csv").is_file()
    assert (output / "oracle_discovery_metadata.json").is_file()
    round_results = pd.read_csv(output / "oracle_round_results.csv")
    assert "oracle_actual_points_with_captain" in round_results.columns
    assert "predicted_points" not in round_results.columns
```

- [ ] **Step 2: Run test and verify it fails**

Run:

```bash
uv run --frozen pytest src/tests/backtesting/test_oracle_discovery.py::test_build_oracle_discovery_report_writes_expected_artifacts -q
```

Expected: FAIL because `build_oracle_discovery_report` is missing.

- [ ] **Step 3: Implement report builder**

Add to `src/cartola/backtesting/oracle_discovery.py`:

```python
from datetime import UTC, datetime


def build_oracle_discovery_report(*, experiment_path: Path, output_path: Path) -> None:
    output_path.mkdir(parents=True, exist_ok=True)
    contexts = load_source_run_contexts(experiment_path)
    round_rows: list[dict[str, object]] = []
    selected_rows: list[pd.DataFrame] = []
    captain_rows: list[dict[str, object]] = []
    recall_rows: list[dict[str, object]] = []
    invalid_rows: list[dict[str, object]] = []
    for context in contexts:
        artifacts = validate_child_artifacts(context)
        config = _config_from_context(context, artifacts.metadata)
        for strategy in context.analyzed_strategies:
            if strategy not in context.strategy_score_columns:
                continue
            score_column = context.strategy_score_columns[strategy]
            strategy_rounds = artifacts.round_results.loc[
                artifacts.round_results["strategy"].eq(strategy)
                & artifacts.round_results["solver_status"].eq("Optimal")
            ]
            for _, round_row in strategy_rounds.iterrows():
                round_number = int(round_row["rodada"])
                candidates = artifacts.player_predictions.loc[
                    artifacts.player_predictions["rodada"].astype(int).eq(round_number)
                ].copy()
                identity = _identity(context, round_number=round_number)
                identity.update(
                    {
                        "oracle_type": "budget_constrained",
                        "candidate_universe": "model_candidate",
                        "budget_path": "model_budget_path",
                    }
                )
                try:
                    oracle_row, oracle_selected = run_model_candidate_oracle(
                        candidates,
                        config=config,
                        budget_before_round=float(round_row["budget_before_round"]),
                        score_column=score_column,
                    )
                except OracleObjectiveError as exc:
                    invalid_rows.append({**identity, "error": str(exc)})
                    continue
                round_rows.append({**identity, **oracle_row, "budget_before_round": float(round_row["budget_before_round"]), "full_market_status": "not_available"})
                if not oracle_selected.empty:
                    oracle_selected = oracle_selected.assign(**identity)
                    selected_rows.append(oracle_selected)
                selected = artifacts.selected_players.loc[
                    artifacts.selected_players["rodada"].astype(int).eq(round_number)
                    & artifacts.selected_players["strategy"].eq(strategy)
                ].copy()
                if not selected.empty:
                    captain_rows.append({**identity, **selected_squad_captain_oracle(selected), "full_market_status": "not_available"})
                recall_rows.extend(_recall_rows(identity, oracle_selected, selected))
    _write_outputs(
        output_path=output_path,
        round_rows=round_rows,
        selected_frames=selected_rows,
        captain_rows=captain_rows,
        recall_rows=recall_rows,
        invalid_rows=invalid_rows,
        experiment_path=experiment_path,
    )


def _config_from_context(context: SourceRunContext, metadata: dict[str, Any]) -> BacktestConfig:
    return BacktestConfig(
        season=context.season,
        start_round=int(metadata["start_round"]),
        budget=float(metadata["initial_budget"]),
        fixture_mode=context.fixture_mode,  # type: ignore[arg-type]
        matchup_context_mode=context.matchup_context_mode,  # type: ignore[arg-type]
        footystats_mode=str(metadata["footystats_mode"]),  # type: ignore[arg-type]
        current_year=2026,
        project_root=Path("."),
    )


def _identity(context: SourceRunContext, *, round_number: int) -> dict[str, object]:
    return {
        "source_mode": "artifact",
        "source_experiment_id": context.source_experiment_id,
        "source_child_id": context.source_child_id,
        "season": context.season,
        "rodada": round_number,
        "model_id": context.model_id,
        "feature_pack": context.feature_pack,
        "fixture_mode": context.fixture_mode,
        "matchup_context_mode": context.matchup_context_mode,
        "budget_policy": context.budget_policy,
    }


def _recall_rows(identity: dict[str, object], oracle_selected: pd.DataFrame, selected: pd.DataFrame) -> list[dict[str, object]]:
    if oracle_selected.empty:
        return []
    selected_ids = set(selected["id_atleta"].astype(int).tolist()) if not selected.empty else set()
    rows: list[dict[str, object]] = []
    for _, row in oracle_selected.iterrows():
        athlete_id = int(row["id_atleta"])
        rows.append(
            {
                **identity,
                "id_atleta": athlete_id,
                "posicao": row["posicao"],
                "in_selected_squad": athlete_id in selected_ids,
                "in_model_candidate_artifact": True,
                "absent_from_model_candidate_artifact": False,
                "in_full_market": None,
                "full_market_status": "not_available",
                "model_predicted_rank_overall": row.get("model_predicted_rank_overall"),
                "model_predicted_rank_position": row.get("model_predicted_rank_position"),
                "individually_affordable": None,
                "squad_budget_blocked_by_counterfactual": None,
                "recall_bucket": "selected" if athlete_id in selected_ids else "missed_inside_model_candidate",
            }
        )
    return rows


def _write_outputs(
    *,
    output_path: Path,
    round_rows: list[dict[str, object]],
    selected_frames: list[pd.DataFrame],
    captain_rows: list[dict[str, object]],
    recall_rows: list[dict[str, object]],
    invalid_rows: list[dict[str, object]],
    experiment_path: Path,
) -> None:
    pd.DataFrame(round_rows).to_csv(output_path / "oracle_round_results.csv", index=False)
    pd.concat(selected_frames, ignore_index=True).to_csv(output_path / "oracle_selected_players.csv", index=False) if selected_frames else pd.DataFrame().to_csv(output_path / "oracle_selected_players.csv", index=False)
    pd.DataFrame(captain_rows).to_csv(output_path / "oracle_captain_profiles.csv", index=False)
    pd.DataFrame().to_csv(output_path / "oracle_player_profiles.csv", index=False)
    pd.DataFrame(recall_rows).to_csv(output_path / "model_vs_oracle_recall.csv", index=False)
    pd.DataFrame().to_csv(output_path / "profile_gap_summary.csv", index=False)
    pd.DataFrame(invalid_rows).to_csv(output_path / "invalid_oracle_rows.csv", index=False)
    metadata = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "source_mode": "artifact",
        "source_experiment_path": str(experiment_path),
        "disclaimer": "Discovery-only hindsight analysis. Not promotion evidence.",
    }
    (output_path / "oracle_discovery_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
```

- [ ] **Step 4: Run tests and verify they pass**

Run:

```bash
uv run --frozen pytest src/tests/backtesting/test_oracle_discovery.py -q
```

Expected: PASS.

---

### Task 6: HTML Report

**Files:**
- Modify: `src/cartola/backtesting/oracle_discovery.py`
- Modify: `src/tests/backtesting/test_oracle_discovery.py`

- [ ] **Step 1: Add failing test for HTML report**

Append to `src/tests/backtesting/test_oracle_discovery.py`:

```python
def test_build_oracle_discovery_report_writes_html_disclaimer(tmp_path: Path) -> None:
    child = _valid_child_dir(tmp_path)
    predictions = _candidate_rows()
    predictions["baseline_score"] = predictions["model_score"]
    predictions["price_score"] = predictions["preco_pre_rodada"]
    predictions["xgboost_depth2_l2_heavy_score"] = predictions["model_score"]
    predictions.to_csv(child / "player_predictions.csv", index=False)
    selected = predictions.head(12).copy()
    selected["strategy"] = "xgboost_depth2_l2_heavy"
    selected["is_captain"] = selected["id_atleta"].eq(10)
    selected.to_csv(child / "selected_players.csv", index=False)
    pd.DataFrame(
        [
            {
                "rodada": 5,
                "strategy": "xgboost_depth2_l2_heavy",
                "solver_status": "Optimal",
                "budget_before_round": 100.0,
                "budget_after_round": 100.0,
                "budget_delta": 0.0,
                "budget_used": 12.0,
                "actual_points_with_captain": 83.5,
                "captain_id": 10,
            }
        ]
    ).to_csv(child / "round_results.csv", index=False)
    (tmp_path / "experiment_metadata.json").write_text(
        json.dumps(
            {
                "experiment_id": "exp-1",
                "child_runs": [
                    {
                        "child_id": "season=2025/model=xgboost_depth2_l2_heavy/feature_pack=ppg_xg_matchup",
                        "output_path": str(child),
                        "season": 2025,
                        "model_id": "xgboost_depth2_l2_heavy",
                        "feature_pack": "ppg_xg_matchup",
                        "fixture_mode": "exploratory",
                        "matchup_context_mode": "cartola_matchup_v1",
                        "budget_policy": "moving",
                        "strategy_roles": {"xgboost_depth2_l2_heavy": "primary_model"},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "oracle_out"

    build_oracle_discovery_report(experiment_path=tmp_path, output_path=output)

    html = (output / "oracle_knowledge_discovery.html").read_text(encoding="utf-8")
    assert "Discovery-only hindsight analysis" in html
    assert "Not promotion evidence" in html
    assert "model_candidate" in html
```

- [ ] **Step 2: Run test and verify it fails**

Run:

```bash
uv run --frozen pytest src/tests/backtesting/test_oracle_discovery.py::test_build_oracle_discovery_report_writes_html_disclaimer -q
```

Expected: FAIL because HTML is not written.

- [ ] **Step 3: Implement simple HTML writer**

Add to `_write_outputs(...)` in `src/cartola/backtesting/oracle_discovery.py` after metadata write:

```python
    _write_html(output_path, round_rows=round_rows, captain_rows=captain_rows, recall_rows=recall_rows)
```

Add helper:

```python
def _write_html(
    output_path: Path,
    *,
    round_rows: list[dict[str, object]],
    captain_rows: list[dict[str, object]],
    recall_rows: list[dict[str, object]],
) -> None:
    round_count = len(round_rows)
    captain_regret = pd.DataFrame(captain_rows)
    total_captain_regret = (
        float(captain_regret["selected_squad_captain_regret"].sum())
        if "selected_squad_captain_regret" in captain_regret.columns
        else 0.0
    )
    recall = pd.DataFrame(recall_rows)
    missed = int(recall["recall_bucket"].eq("missed_inside_model_candidate").sum()) if "recall_bucket" in recall.columns else 0
    html = f"""<!doctype html>
<html>
<head><meta charset="utf-8"><title>Oracle Knowledge Discovery</title></head>
<body>
<h1>Oracle Knowledge Discovery</h1>
<p><strong>Discovery-only hindsight analysis. Not promotion evidence.</strong></p>
<h2>Scope</h2>
<p>Source mode: artifact. Candidate universe: model_candidate. Full-market status: not_available.</p>
<h2>Summary</h2>
<ul>
  <li>Oracle rounds: {round_count}</li>
  <li>Total selected-squad captain regret: {total_captain_regret:.2f}</li>
  <li>Oracle players missed inside model candidate pool: {missed}</li>
</ul>
</body>
</html>
"""
    (output_path / "oracle_knowledge_discovery.html").write_text(html, encoding="utf-8")
```

- [ ] **Step 4: Run tests and verify they pass**

Run:

```bash
uv run --frozen pytest src/tests/backtesting/test_oracle_discovery.py -q
```

Expected: PASS.

---

### Task 7: CLI Wrapper

**Files:**
- Create: `scripts/run_oracle_knowledge_discovery.py`
- Create: `src/tests/backtesting/test_run_oracle_knowledge_discovery_cli.py`

- [ ] **Step 1: Add failing CLI tests**

Create `src/tests/backtesting/test_run_oracle_knowledge_discovery_cli.py`:

```python
from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_oracle_knowledge_discovery import main, parse_args  # noqa: E402


def test_importing_cli_does_not_import_oracle_module() -> None:
    code = "\n".join(
        [
            "import importlib",
            "import sys",
            "importlib.import_module('scripts.run_oracle_knowledge_discovery')",
            "print('cartola.backtesting.oracle_discovery' in sys.modules)",
        ]
    )

    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.stdout.strip() == "False"


def test_parse_args_defaults() -> None:
    args = parse_args(["--experiment-path", "exp", "--current-year", "2026"])

    assert args.experiment_path == Path("exp")
    assert args.current_year == 2026
    assert args.output_root == Path("data/08_reporting/oracle_discovery")
    assert args.allow_incomplete is False


def test_main_calls_report_builder(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import scripts.run_oracle_knowledge_discovery as cli

    observed: dict[str, object] = {}

    def fake_builder(**kwargs: object) -> None:
        observed.update(kwargs)

    monkeypatch.setattr(cli, "build_oracle_discovery_report", fake_builder)

    exit_code = main(
        [
            "--experiment-path",
            str(tmp_path / "exp"),
            "--output-root",
            str(tmp_path / "oracle"),
            "--current-year",
            "2026",
        ]
    )

    assert exit_code == 0
    assert observed["experiment_path"] == tmp_path / "exp"
    assert str(observed["output_path"]).startswith(str(tmp_path / "oracle"))
```

- [ ] **Step 2: Run tests and verify they fail**

Run:

```bash
uv run --frozen pytest src/tests/backtesting/test_run_oracle_knowledge_discovery_cli.py -q
```

Expected: FAIL because script does not exist.

- [ ] **Step 3: Implement CLI**

Create `scripts/run_oracle_knowledge_discovery.py`:

```python
#!/usr/bin/env python
from __future__ import annotations

import argparse
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable

from rich.console import Console
from rich.panel import Panel

build_oracle_discovery_report: Callable[..., Any] | None = None


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Cartola oracle knowledge discovery report.")
    parser.add_argument("--experiment-path", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("data/08_reporting/oracle_discovery"))
    parser.add_argument("--current-year", type=int, required=True)
    parser.add_argument("--allow-incomplete", action="store_true")
    return parser.parse_args(argv)


def _timestamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")


def _load_runtime_dependencies() -> None:
    global build_oracle_discovery_report
    if build_oracle_discovery_report is None:
        from cartola.backtesting.oracle_discovery import (
            build_oracle_discovery_report as imported_build_oracle_discovery_report,
        )

        build_oracle_discovery_report = imported_build_oracle_discovery_report


def main(argv: Sequence[str] | None = None) -> int:
    console = Console(stderr=True)
    args = parse_args(argv)
    _load_runtime_dependencies()
    output_path = args.output_root / f"oracle_discovery_started_at={_timestamp()}"
    try:
        assert build_oracle_discovery_report is not None
        build_oracle_discovery_report(experiment_path=args.experiment_path, output_path=output_path)
    except Exception as exc:
        console.print(Panel(str(exc), title="Oracle discovery failed", border_style="red"))
        return 1
    console.print(Panel(f"output_path={output_path}", title="Oracle discovery complete", border_style="green"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run CLI tests**

Run:

```bash
uv run --frozen pytest src/tests/backtesting/test_run_oracle_knowledge_discovery_cli.py -q
```

Expected: PASS.

---

### Task 8: Verification And Smoke Run

**Files:**
- Read/execute only unless failures require fixes.

- [ ] **Step 1: Run targeted tests**

Run:

```bash
uv run --frozen pytest src/tests/backtesting/test_oracle_discovery.py src/tests/backtesting/test_run_oracle_knowledge_discovery_cli.py -q
```

Expected: PASS.

- [ ] **Step 2: Run full quality gate**

Run:

```bash
uv run --frozen scripts/pyrepo-check --all
```

Expected: ruff, ty, bandit, and pytest pass.

- [ ] **Step 3: Run smoke report on the completed XGBoost sensitivity experiment**

Run:

```bash
uv run --frozen python scripts/run_oracle_knowledge_discovery.py \
  --experiment-path data/08_reporting/experiments/model_feature/group=xgboost-sensitivity-v2__started_at=20260505T211914592073Z__matrix=d95948374d75 \
  --current-year 2026
```

Expected: command exits `0` and writes an output directory under `data/08_reporting/oracle_discovery/`.

- [ ] **Step 4: Inspect generated files**

Run:

```bash
find data/08_reporting/oracle_discovery -maxdepth 2 -type f | tail -20
```

Expected: includes:

- `oracle_round_results.csv`
- `oracle_selected_players.csv`
- `oracle_captain_profiles.csv`
- `model_vs_oracle_recall.csv`
- `oracle_knowledge_discovery.html`
- `oracle_discovery_metadata.json`

---

## Self-Review Notes

Spec coverage:

- Artifact-backed default mode: Task 1.
- Source context and score mapping: Task 1.
- Moving-budget compatibility rejection: Task 1.
- DNP/null policy: Task 2.
- Oracle result adapter: Task 2.
- Selected-squad captain oracle: Task 3.
- Model-candidate oracle: Task 4.
- Required CSV artifacts: Task 5.
- HTML disclaimer: Task 6.
- CLI isolation: Task 7.
- Verification: Task 8.

Deferred intentionally:

- Reconstructed mode.
- Full-market oracle.
- Unlimited-budget appendix.
- Independent oracle budget path.
- Deterministic hypothesis candidates.
- Rich Plotly HTML.

Those are v1.1+ after artifact-backed diagnostics are trusted.
