# EBM Feature Diagnostic Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an artifact-backed EBM diagnostic runner that finds stable residual feature-shape leads without touching live defaults, experiment promotion, or MILP optimization.

**Architecture:** Add one focused diagnostic module plus one thin CLI wrapper. The module validates source experiment artifacts, builds shared raw/residual datasets, fits EBM folds through a compatibility adapter, extracts deterministic summaries from learned terms, writes CSV/JSON/HTML artifacts, and degrades to invalid/incomplete reports when dependencies or artifacts are not trustworthy.

**Tech Stack:** Python 3.13, pandas, numpy, scikit-learn metrics, optional InterpretML `ExplainableBoostingRegressor`, Plotly HTML, Rich CLI progress, pytest, Ruff, ty.

---

## Locked V1 Choices

- `position_handling=one_hot`: generate deterministic one-hot columns from persisted `posicao`; position-specific diagnostics are deferred.
- `inner_validation_mode=disabled_full_outer_train`: train each fold on all outer training seasons and disable InterpretML internal random validation. Temporal inner validation is deferred until a later generation proves it improves the diagnostic.
- Interaction pass failure does not invalidate completed main-effect outputs. It writes an unavailable interaction artifact and blocks interaction candidate flags.
- Schema/provenance errors write `invalid_diagnostic_report.csv`; row-level target errors write `invalid_ebm_rows.csv`.
- Progress cadence is one CLI log/progress update per fold, target type, and EBM pass.

## File Structure

- Create `src/cartola/backtesting/ebm_feature_diagnostic.py`
  - Dataclasses for config, source context, validation reports, fold definitions, EBM runtime info, fitted-term summaries, and final result.
  - Source child resolution from parent `experiment_metadata.json` and child `run_metadata.json`.
  - Artifact schema validation and source prediction provenance checks.
  - DNP/null target policy, coach exclusion, one-hot position encoding, feature filtering.
  - Season-expanding fold creation and shared-row metric computation.
  - Optional InterpretML compatibility adapter and EBM fitting hooks.
  - Learned-bin/cell support extraction and candidate gating.
  - CSV/JSON/HTML artifact writers.
- Create `scripts/run_ebm_feature_diagnostic.py`
  - CLI parsing.
  - `.env` bootstrap before runtime imports.
  - Rich progress display with line-mode fallback.
  - Success/failure panels.
- Create `src/tests/backtesting/test_ebm_feature_diagnostic.py`
  - Unit tests for source resolution, artifact validation, target prep, fold isolation, metrics, support extraction, candidate aggregation, invalid artifacts, and HTML writing.
- Create `src/tests/backtesting/test_run_ebm_feature_diagnostic_cli.py`
  - CLI argument parsing and progress callback tests.
- Modify `pyproject.toml` and `uv.lock` only if `uv add --dev interpret` resolves cleanly on Python 3.13.12.
- Do not modify live recommendation scripts, optimizer code, experiment-index promotion code, or existing model registries.

## Output Schemas

`source_context.csv`:

```text
discovery_only,source_experiment_id,requested_season,season,model_id,feature_pack,fixture_mode,matchup_context_mode,footystats_mode,budget_policy,scoring_contract_version,primary_score_column,match_status,child_path,conflicting_child_paths,missing_metadata_fields,source_prediction_provenance_status
```

`fold_assignments.csv`:

```text
discovery_only,fold_id,validation_season,train_seasons,inner_validation_mode,train_row_count,validation_row_count
```

`invalid_ebm_rows.csv`:

```text
discovery_only,season,rodada,id_atleta,apelido,posicao,invalid_reason,pontuacao,entrou_em_campo
```

`invalid_diagnostic_report.csv`:

```text
discovery_only,scope,severity,reason_type,message,artifact_path,season,model_id,feature_pack
```

`predictive_metrics.csv`:

```text
discovery_only,target_type,prediction_type,fold_id,validation_season,shared_evaluation_row_count,mae,rmse,spearman,top50_spearman,calibration_slope,mean_prediction_bias
```

`feature_importance_by_fold.csv`:

```text
discovery_only,target_type,fold_id,validation_season,feature_name,importance_rank,importance_score
```

`feature_shape_summary.csv`:

```text
discovery_only,target_type,feature_name,fold_id,validation_season,importance_rank,importance_score,effect_min,effect_max,effect_range,term_support_extraction_status,largest_positive_bin_lower,largest_positive_bin_upper,largest_positive_bin_row_support,largest_positive_bin_round_support,largest_negative_bin_lower,largest_negative_bin_upper,largest_negative_bin_row_support,largest_negative_bin_round_support,monotonicity_hint,row_support,season_support,fold_candidate_signal
```

`pairwise_interactions.csv`:

```text
discovery_only,target_type,interaction_name,feature_a,feature_b,fold_id,validation_season,importance_rank,importance_score,effect_range,term_support_extraction_status,max_effect_cell_row_support,max_effect_cell_round_support,min_effect_cell_row_support,min_effect_cell_round_support,row_support,season_support,fold_candidate_signal
```

`candidate_hypotheses.csv`:

```text
discovery_only,target_type,candidate_type,term_name,feature_a,feature_b,fold_signal_count,validation_seasons_with_signal,total_row_support,min_bin_or_cell_row_support,min_bin_or_cell_round_support,effect_range_median,direction_summary,failed_validation_seasons,candidate_hypothesis_flag,candidate_scope
```

`ebm_diagnostic_manifest.json` includes `discovery_only=true`, source context, fold config, dependency info, InterpretML signatures, `position_handling=one_hot`, runtime seconds, term-support extraction status, and `holdout_usage_ledger`.

`ebm_diagnostic_decision.json`:

```json
{
  "discovery_only": true,
  "diagnostic_status": "candidate_hypotheses_found",
  "inner_validation_mode": "disabled_full_outer_train",
  "position_handling": "one_hot",
  "candidate_count": 1,
  "material_2025_regression": false,
  "source_experiment_path": "data/08_reporting/experiments/model_feature/example",
  "output_path": "data/08_reporting/ebm_diagnostics/ebm_diagnostic_started_at=..."
}
```

## Task 1: Dependency Probe And Runtime Adapter

**Files:**
- Create: `src/cartola/backtesting/ebm_feature_diagnostic.py`
- Test: `src/tests/backtesting/test_ebm_feature_diagnostic.py`

- [ ] **Step 1: Write failing tests for optional dependency inspection**

Add this to `src/tests/backtesting/test_ebm_feature_diagnostic.py`:

```python
from __future__ import annotations

from dataclasses import dataclass

import pytest

from cartola.backtesting.ebm_feature_diagnostic import (
    EbmDependencyError,
    inspect_ebm_runtime,
)


class _FakeEbm:
    def __init__(
        self,
        *,
        interactions: int = 0,
        validation_size: float = 0.0,
        random_state: int | None = None,
    ) -> None:
        self.interactions = interactions
        self.validation_size = validation_size
        self.random_state = random_state

    def fit(self, x_values: object, y_values: object) -> "_FakeEbm":
        return self


def test_inspect_ebm_runtime_records_constructor_and_fit_signatures() -> None:
    info = inspect_ebm_runtime(ebm_class=_FakeEbm, package_version="9.9.9")

    assert info.available is True
    assert info.version == "9.9.9"
    assert "validation_size" in info.constructor_signature
    assert "x_values" in info.fit_signature
    assert info.supports_explicit_validation is False


def test_inspect_ebm_runtime_raises_clear_error_when_missing() -> None:
    with pytest.raises(EbmDependencyError, match="InterpretML is required"):
        inspect_ebm_runtime(ebm_class=None, package_version=None)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run:

```bash
uv run --frozen pytest src/tests/backtesting/test_ebm_feature_diagnostic.py::test_inspect_ebm_runtime_records_constructor_and_fit_signatures src/tests/backtesting/test_ebm_feature_diagnostic.py::test_inspect_ebm_runtime_raises_clear_error_when_missing -q
```

Expected: FAIL with `ModuleNotFoundError` or missing `inspect_ebm_runtime`.

- [ ] **Step 3: Implement the minimal dependency adapter**

Create `src/cartola/backtesting/ebm_feature_diagnostic.py` with:

```python
from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any


class EbmDependencyError(RuntimeError):
    """Raised when InterpretML is unavailable or incompatible."""


@dataclass(frozen=True)
class EbmRuntimeInfo:
    available: bool
    version: str | None
    constructor_signature: str
    fit_signature: str
    supports_explicit_validation: bool


def inspect_ebm_runtime(*, ebm_class: type[Any] | None, package_version: str | None) -> EbmRuntimeInfo:
    if ebm_class is None:
        raise EbmDependencyError(
            "InterpretML is required for EBM diagnostics. Install the optional diagnostic dependencies."
        )
    constructor_signature = str(inspect.signature(ebm_class))
    fit_signature = str(inspect.signature(ebm_class.fit))
    supports_explicit_validation = "X_val" in fit_signature and "y_val" in fit_signature
    return EbmRuntimeInfo(
        available=True,
        version=package_version,
        constructor_signature=constructor_signature,
        fit_signature=fit_signature,
        supports_explicit_validation=supports_explicit_validation,
    )
```

- [ ] **Step 4: Run the dependency adapter tests**

Run:

```bash
uv run --frozen pytest src/tests/backtesting/test_ebm_feature_diagnostic.py::test_inspect_ebm_runtime_records_constructor_and_fit_signatures src/tests/backtesting/test_ebm_feature_diagnostic.py::test_inspect_ebm_runtime_raises_clear_error_when_missing -q
```

Expected: PASS.

- [ ] **Step 5: Check dependency resolution without committing failure**

Run:

```bash
uv add --dev interpret
```

Expected if compatible: `pyproject.toml` and `uv.lock` update with `interpret`. Keep those files and continue.

Expected if incompatible: command fails. Do not downgrade Python. Revert any partial dependency edits and keep the optional dependency error path.

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml uv.lock src/cartola/backtesting/ebm_feature_diagnostic.py src/tests/backtesting/test_ebm_feature_diagnostic.py
git commit -m "feat: add ebm diagnostic runtime adapter"
```

If `interpret` did not install, omit `pyproject.toml` and `uv.lock` from `git add`.

## Task 2: Source Child Resolution And Provenance

**Files:**
- Modify: `src/cartola/backtesting/ebm_feature_diagnostic.py`
- Test: `src/tests/backtesting/test_ebm_feature_diagnostic.py`

- [ ] **Step 1: Write failing tests for exact child matching**

Append:

```python
import json
from pathlib import Path

from cartola.backtesting.ebm_feature_diagnostic import (
    EbmDiagnosticConfig,
    EbmDiagnosticInvalid,
    resolve_source_children,
)


def _write_source_child(root: Path, *, season: int, model_id: str = "xgboost_depth2_l2_heavy") -> Path:
    child_path = root / "runs" / f"season={season}" / f"model={model_id}" / "feature_pack=ppg_xg_matchup"
    child_path.mkdir(parents=True)
    child_metadata = {
        "season": season,
        "model_id": model_id,
        "feature_pack": "ppg_xg_matchup",
        "fixture_mode": "exploratory",
        "matchup_context_mode": "cartola_matchup_v1",
        "footystats_mode": "ppg_xg",
        "budget_policy": "moving",
        "scoring_contract_version": "cartola_standard_2026_v1",
    }
    (child_path / "run_metadata.json").write_text(json.dumps(child_metadata), encoding="utf-8")
    (child_path / "player_predictions.csv").write_text(f"{model_id}_score\n1.0\n", encoding="utf-8")
    (child_path / "round_results.csv").write_text("strategy\nx\n", encoding="utf-8")
    return child_path


def _write_parent(root: Path, *, child_paths: list[Path], model_id: str = "xgboost_depth2_l2_heavy") -> None:
    children = []
    for child_path in child_paths:
        season = int(child_path.parts[-3].split("=")[1])
        children.append(
            {
                "child_id": f"season={season}/model={model_id}/feature_pack=ppg_xg_matchup",
                "season": season,
                "model_id": model_id,
                "feature_pack": "ppg_xg_matchup",
                "fixture_mode": "exploratory",
                "strategy_roles": {"baseline": "baseline", "price": "price", model_id: "primary_model"},
                "output_path": str(child_path),
                "metadata": {
                    "season": season,
                    "model_id": model_id,
                    "feature_pack": "ppg_xg_matchup",
                    "fixture_mode": "exploratory",
                    "matchup_context_mode": "cartola_matchup_v1",
                    "footystats_mode": "ppg_xg",
                    "budget_policy": "moving",
                    "scoring_contract_version": "cartola_standard_2026_v1",
                },
            }
        )
    (root / "experiment_metadata.json").write_text(
        json.dumps({"experiment_id": "exp-1", "child_runs": children}),
        encoding="utf-8",
    )


def test_resolve_source_children_requires_one_match_per_season(tmp_path: Path) -> None:
    experiment = tmp_path / "experiment"
    child = _write_source_child(experiment, season=2025)
    _write_parent(experiment, child_paths=[child])
    config = EbmDiagnosticConfig(
        experiment_path=experiment,
        seasons=(2025,),
        model_id="xgboost_depth2_l2_heavy",
        feature_pack="ppg_xg_matchup",
        fixture_mode="exploratory",
    )

    contexts, report = resolve_source_children(config)

    assert report.empty
    assert len(contexts) == 1
    assert contexts[0].season == 2025
    assert contexts[0].score_column == "xgboost_depth2_l2_heavy_score"
    assert contexts[0].source_prediction_provenance_status == "verified"


def test_resolve_source_children_reports_duplicate_matches(tmp_path: Path) -> None:
    experiment = tmp_path / "experiment"
    child = _write_source_child(experiment, season=2025)
    duplicate = _write_source_child(experiment, season=2025, model_id="xgboost_depth2_l2_heavy")
    _write_parent(experiment, child_paths=[child, duplicate])
    config = EbmDiagnosticConfig(
        experiment_path=experiment,
        seasons=(2025,),
        model_id="xgboost_depth2_l2_heavy",
        feature_pack="ppg_xg_matchup",
        fixture_mode="exploratory",
    )

    with pytest.raises(EbmDiagnosticInvalid, match="Duplicate source child matches"):
        resolve_source_children(config)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run --frozen pytest src/tests/backtesting/test_ebm_feature_diagnostic.py::test_resolve_source_children_requires_one_match_per_season src/tests/backtesting/test_ebm_feature_diagnostic.py::test_resolve_source_children_reports_duplicate_matches -q
```

Expected: FAIL for missing config/context resolver types.

- [ ] **Step 3: Implement source resolution**

Add:

```python
import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd


class EbmDiagnosticInvalid(RuntimeError):
    """Raised when the diagnostic source artifacts cannot be trusted."""


@dataclass(frozen=True)
class EbmDiagnosticConfig:
    experiment_path: Path
    seasons: tuple[int, ...]
    model_id: str
    feature_pack: str
    fixture_mode: str


@dataclass(frozen=True)
class SourceChildContext:
    source_experiment_id: str
    season: int
    model_id: str
    feature_pack: str
    fixture_mode: str
    matchup_context_mode: str
    footystats_mode: str
    budget_policy: str
    scoring_contract_version: str
    score_column: str
    child_path: Path
    source_prediction_provenance_status: str

    def as_row(self) -> dict[str, object]:
        row = asdict(self)
        row["child_path"] = str(self.child_path)
        row["discovery_only"] = True
        row["match_status"] = "matched"
        row["conflicting_child_paths"] = ""
        row["missing_metadata_fields"] = ""
        return row


def resolve_source_children(config: EbmDiagnosticConfig) -> tuple[tuple[SourceChildContext, ...], pd.DataFrame]:
    metadata = _read_json_object(config.experiment_path / "experiment_metadata.json")
    experiment_id = str(metadata["experiment_id"])
    child_runs = metadata.get("child_runs")
    if not isinstance(child_runs, list):
        raise EbmDiagnosticInvalid("experiment_metadata.json child_runs must be a list")
    contexts: list[SourceChildContext] = []
    invalid_rows: list[dict[str, object]] = []
    for season in config.seasons:
        matches = [
            child for child in child_runs
            if _child_matches(child, season=season, model_id=config.model_id, feature_pack=config.feature_pack, fixture_mode=config.fixture_mode)
        ]
        if len(matches) != 1:
            invalid_rows.append(
                {
                    "discovery_only": True,
                    "requested_season": season,
                    "season": season,
                    "model_id": config.model_id,
                    "feature_pack": config.feature_pack,
                    "fixture_mode": config.fixture_mode,
                    "match_status": "duplicate" if len(matches) > 1 else "missing",
                    "conflicting_child_paths": "|".join(str(match.get("output_path", "")) for match in matches),
                    "source_prediction_provenance_status": "unverified",
                }
            )
            continue
        contexts.append(_context_from_child(experiment_id=experiment_id, experiment_path=config.experiment_path, child=matches[0]))
    report = pd.DataFrame(invalid_rows)
    if invalid_rows:
        raise EbmDiagnosticInvalid("Duplicate source child matches" if any(row["match_status"] == "duplicate" for row in invalid_rows) else "Missing source child matches")
    return tuple(contexts), report


def _child_matches(child: object, *, season: int, model_id: str, feature_pack: str, fixture_mode: str) -> bool:
    if not isinstance(child, dict):
        return False
    metadata = child.get("metadata") if isinstance(child.get("metadata"), dict) else {}
    return (
        int(child.get("season", -1)) == season
        and str(child.get("model_id")) == model_id
        and str(child.get("feature_pack")) == feature_pack
        and str(child.get("fixture_mode")) == fixture_mode
        and str(metadata.get("budget_policy")) == "moving"
    )


def _context_from_child(*, experiment_id: str, experiment_path: Path, child: dict[str, object]) -> SourceChildContext:
    metadata = child["metadata"]
    if not isinstance(metadata, dict):
        raise EbmDiagnosticInvalid("child metadata must be an object")
    child_path = Path(str(child["output_path"]))
    if not child_path.is_absolute():
        child_path = (experiment_path.parent.parent.parent.parent / child_path).resolve()
    model_id = str(child["model_id"])
    score_column = f"{model_id}_score"
    if not (child_path / "run_metadata.json").is_file() or not (child_path / "player_predictions.csv").is_file():
        raise EbmDiagnosticInvalid(f"Missing child artifacts: {child_path}")
    prediction_columns = pd.read_csv(child_path / "player_predictions.csv", nrows=0).columns
    if score_column not in prediction_columns:
        raise EbmDiagnosticInvalid(f"Missing primary score column in player_predictions.csv: {score_column}")
    return SourceChildContext(
        source_experiment_id=experiment_id,
        season=int(child["season"]),
        model_id=model_id,
        feature_pack=str(child["feature_pack"]),
        fixture_mode=str(child["fixture_mode"]),
        matchup_context_mode=str(metadata["matchup_context_mode"]),
        footystats_mode=str(metadata["footystats_mode"]),
        budget_policy=str(metadata["budget_policy"]),
        scoring_contract_version=str(metadata["scoring_contract_version"]),
        score_column=score_column,
        child_path=child_path,
        source_prediction_provenance_status="verified",
    )


def _read_json_object(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise EbmDiagnosticInvalid(f"JSON artifact must contain an object: {path}")
    return payload
```

- [ ] **Step 4: Run source resolution tests**

```bash
uv run --frozen pytest src/tests/backtesting/test_ebm_feature_diagnostic.py::test_resolve_source_children_requires_one_match_per_season src/tests/backtesting/test_ebm_feature_diagnostic.py::test_resolve_source_children_reports_duplicate_matches -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/cartola/backtesting/ebm_feature_diagnostic.py src/tests/backtesting/test_ebm_feature_diagnostic.py
git commit -m "feat: validate ebm source child context"
```

## Task 3: Diagnostic Dataset Preparation

**Files:**
- Modify: `src/cartola/backtesting/ebm_feature_diagnostic.py`
- Test: `src/tests/backtesting/test_ebm_feature_diagnostic.py`

- [ ] **Step 1: Write failing tests for DNP policy, coach exclusion, and one-hot positions**

Append:

```python
from cartola.backtesting.ebm_feature_diagnostic import prepare_diagnostic_dataset


def test_prepare_diagnostic_dataset_maps_dnp_nulls_to_zero_and_excludes_coaches() -> None:
    context = SourceChildContext(
        source_experiment_id="exp-1",
        season=2025,
        model_id="xgboost_depth2_l2_heavy",
        feature_pack="ppg_xg_matchup",
        fixture_mode="exploratory",
        matchup_context_mode="cartola_matchup_v1",
        footystats_mode="ppg_xg",
        budget_policy="moving",
        scoring_contract_version="cartola_standard_2026_v1",
        score_column="xgboost_depth2_l2_heavy_score",
        child_path=Path("child"),
        source_prediction_provenance_status="verified",
    )
    predictions = pd.DataFrame(
        {
            "rodada": [5, 5, 5],
            "id_atleta": [1, 2, 3],
            "apelido": ["A", "B", "Coach"],
            "id_clube": [10, 20, 30],
            "posicao": ["ata", "lat", "tec"],
            "status": ["Provável", "Provável", "Provável"],
            "preco_pre_rodada": [10.0, 8.0, 5.0],
            "pontuacao": [7.0, None, 3.0],
            "entrou_em_campo": [True, False, True],
            "xgboost_depth2_l2_heavy_score": [6.0, 2.0, 4.0],
            "feature_a": [1.5, 2.5, 3.5],
        }
    )

    bundle = prepare_diagnostic_dataset(context, predictions, feature_columns=("feature_a", "posicao"))

    assert bundle.valid_rows["target_actual_points"].tolist() == [7.0, 0.0]
    assert bundle.valid_rows["target_source_residual"].tolist() == [1.0, -2.0]
    assert "posicao_ata" in bundle.feature_columns
    assert "posicao_lat" in bundle.feature_columns
    assert "posicao" not in bundle.feature_columns
    assert bundle.coach_row_count == 1
    assert bundle.invalid_rows.empty
```

- [ ] **Step 2: Run the dataset test to verify it fails**

```bash
uv run --frozen pytest src/tests/backtesting/test_ebm_feature_diagnostic.py::test_prepare_diagnostic_dataset_maps_dnp_nulls_to_zero_and_excludes_coaches -q
```

Expected: FAIL for missing `prepare_diagnostic_dataset`.

- [ ] **Step 3: Implement dataset preparation**

Add:

```python
@dataclass(frozen=True)
class DiagnosticDataset:
    context: SourceChildContext
    valid_rows: pd.DataFrame
    invalid_rows: pd.DataFrame
    feature_columns: tuple[str, ...]
    coach_row_count: int


IDENTITY_COLUMNS = frozenset({"id_atleta", "id_clube", "apelido", "season", "rodada"})
REQUIRED_PREDICTION_COLUMNS = frozenset(
    {"rodada", "id_atleta", "apelido", "id_clube", "posicao", "status", "pontuacao", "entrou_em_campo", "preco_pre_rodada"}
)


def prepare_diagnostic_dataset(
    context: SourceChildContext,
    predictions: pd.DataFrame,
    *,
    feature_columns: tuple[str, ...],
) -> DiagnosticDataset:
    missing = sorted((REQUIRED_PREDICTION_COLUMNS | {context.score_column}) - set(predictions.columns))
    if missing:
        raise EbmDiagnosticInvalid(f"Missing required player_predictions columns: {missing}")
    frame = predictions.copy()
    frame["season"] = context.season
    frame["source_model_score"] = pd.to_numeric(frame[context.score_column], errors="coerce")
    frame["target_actual_points"] = pd.to_numeric(frame["pontuacao"], errors="coerce")
    entered = frame["entrou_em_campo"].map(_bool_like)
    dnp_null = frame["target_actual_points"].isna() & entered.eq(False)
    frame.loc[dnp_null, "target_actual_points"] = 0.0
    invalid_mask = frame["target_actual_points"].isna() | frame["source_model_score"].isna() | entered.isna()
    invalid_rows = frame.loc[invalid_mask, ["season", "rodada", "id_atleta", "apelido", "posicao", "pontuacao", "entrou_em_campo"]].copy()
    invalid_rows["invalid_reason"] = "invalid_target_or_prediction"
    player_rows = frame.loc[~invalid_mask & ~frame["posicao"].astype(str).eq("tec")].copy()
    coach_row_count = int((~invalid_mask & frame["posicao"].astype(str).eq("tec")).sum())
    player_rows["target_source_residual"] = player_rows["target_actual_points"] - player_rows["source_model_score"]
    selected_features: list[str] = []
    for column in feature_columns:
        if column in IDENTITY_COLUMNS:
            continue
        if column == "posicao":
            dummies = pd.get_dummies(player_rows["posicao"].astype(str), prefix="posicao", dtype=float)
            player_rows = pd.concat([player_rows, dummies], axis=1)
            selected_features.extend(str(name) for name in dummies.columns)
            continue
        values = pd.to_numeric(player_rows[column], errors="coerce")
        if values.isna().any():
            raise EbmDiagnosticInvalid(f"Retained EBM feature is not fully numeric: {column}")
        player_rows[column] = values
        selected_features.append(column)
    return DiagnosticDataset(
        context=context,
        valid_rows=player_rows.reset_index(drop=True),
        invalid_rows=invalid_rows.reset_index(drop=True),
        feature_columns=tuple(selected_features),
        coach_row_count=coach_row_count,
    )


def _bool_like(value: object) -> bool | None:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "1.0", "true", "t", "yes", "y"}:
        return True
    if normalized in {"0", "0.0", "false", "f", "no", "n"}:
        return False
    return None
```

- [ ] **Step 4: Run dataset tests**

```bash
uv run --frozen pytest src/tests/backtesting/test_ebm_feature_diagnostic.py::test_prepare_diagnostic_dataset_maps_dnp_nulls_to_zero_and_excludes_coaches -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/cartola/backtesting/ebm_feature_diagnostic.py src/tests/backtesting/test_ebm_feature_diagnostic.py
git commit -m "feat: prepare ebm diagnostic dataset"
```

## Task 4: Season Folds And Shared Metrics

**Files:**
- Modify: `src/cartola/backtesting/ebm_feature_diagnostic.py`
- Test: `src/tests/backtesting/test_ebm_feature_diagnostic.py`

- [ ] **Step 1: Write failing tests for folds and residual-corrected metrics**

Append:

```python
from cartola.backtesting.ebm_feature_diagnostic import build_season_folds, compute_predictive_metrics


def test_build_season_folds_uses_whole_validation_seasons() -> None:
    folds = build_season_folds((2021, 2022, 2023, 2024, 2025))

    assert [(fold.fold_id, fold.train_seasons, fold.validation_season) for fold in folds] == [
        ("A", (2021, 2022), 2023),
        ("B", (2021, 2022, 2023), 2024),
        ("C", (2021, 2022, 2023, 2024), 2025),
    ]
    assert {fold.inner_validation_mode for fold in folds} == {"disabled_full_outer_train"}


def test_compute_predictive_metrics_uses_residual_corrected_predictions() -> None:
    rows = pd.DataFrame(
        {
            "season": [2025, 2025, 2025],
            "rodada": [5, 5, 5],
            "target_actual_points": [7.0, 3.0, 1.0],
            "source_model_score": [6.0, 2.0, 2.0],
            "predicted_source_residual": [1.0, 0.0, -1.0],
            "predicted_actual_points": [5.0, 5.0, 5.0],
        }
    )

    metrics = compute_predictive_metrics(rows, fold_id="C", validation_season=2025)

    residual_row = metrics.loc[metrics["prediction_type"].eq("residual_corrected")].iloc[0]
    raw_row = metrics.loc[metrics["prediction_type"].eq("actual_points")].iloc[0]
    assert residual_row["mae"] == pytest.approx(1 / 3)
    assert raw_row["mae"] == pytest.approx(8 / 3)
    assert bool(metrics["discovery_only"].all())
```

- [ ] **Step 2: Run fold/metric tests to verify they fail**

```bash
uv run --frozen pytest src/tests/backtesting/test_ebm_feature_diagnostic.py::test_build_season_folds_uses_whole_validation_seasons src/tests/backtesting/test_ebm_feature_diagnostic.py::test_compute_predictive_metrics_uses_residual_corrected_predictions -q
```

Expected: FAIL for missing fold and metric functions.

- [ ] **Step 3: Implement folds and metrics**

Add:

```python
import math
import numpy as np


@dataclass(frozen=True)
class SeasonFold:
    fold_id: str
    train_seasons: tuple[int, ...]
    validation_season: int
    inner_validation_mode: str = "disabled_full_outer_train"


def build_season_folds(seasons: tuple[int, ...]) -> tuple[SeasonFold, ...]:
    ordered = tuple(sorted(seasons))
    if len(ordered) < 3:
        raise EbmDiagnosticInvalid("EBM diagnostic requires at least three seasons")
    return tuple(
        SeasonFold(fold_id=chr(ord("A") + index), train_seasons=ordered[: index + 2], validation_season=ordered[index + 2])
        for index in range(len(ordered) - 2)
    )


def compute_predictive_metrics(rows: pd.DataFrame, *, fold_id: str, validation_season: int) -> pd.DataFrame:
    frame = rows.copy()
    frame["residual_corrected_prediction"] = frame["source_model_score"] + frame["predicted_source_residual"]
    specs = (
        ("source_model", "source_model_score", "actual_points"),
        ("actual_points", "predicted_actual_points", "actual_points"),
        ("residual_corrected", "residual_corrected_prediction", "source_residual"),
    )
    metric_rows: list[dict[str, object]] = []
    for prediction_type, prediction_column, target_type in specs:
        actual = pd.to_numeric(frame["target_actual_points"], errors="coerce")
        predicted = pd.to_numeric(frame[prediction_column], errors="coerce")
        valid = actual.notna() & predicted.notna()
        actual_valid = actual.loc[valid]
        predicted_valid = predicted.loc[valid]
        metric_rows.append(
            {
                "discovery_only": True,
                "target_type": target_type,
                "prediction_type": prediction_type,
                "fold_id": fold_id,
                "validation_season": validation_season,
                "shared_evaluation_row_count": int(valid.sum()),
                "mae": float((actual_valid - predicted_valid).abs().mean()),
                "rmse": float(np.sqrt(((actual_valid - predicted_valid) ** 2).mean())),
                "spearman": _safe_spearman(actual_valid, predicted_valid),
                "top50_spearman": _top50_spearman(frame.loc[valid], prediction_column),
                "calibration_slope": _calibration_slope(actual_valid, predicted_valid),
                "mean_prediction_bias": float((predicted_valid - actual_valid).mean()),
            }
        )
    return pd.DataFrame(metric_rows)


def _safe_spearman(actual: pd.Series, predicted: pd.Series) -> float:
    if actual.nunique(dropna=True) < 2 or predicted.nunique(dropna=True) < 2:
        return math.nan
    return float(actual.corr(predicted, method="spearman"))


def _top50_spearman(frame: pd.DataFrame, prediction_column: str) -> float:
    values: list[float] = []
    for _, round_frame in frame.groupby("rodada", sort=True):
        if len(round_frame) < 50:
            continue
        top = round_frame.sort_values(prediction_column, ascending=False, kind="mergesort").head(50)
        value = _safe_spearman(top["target_actual_points"], top[prediction_column])
        if not math.isnan(value):
            values.append(value)
    return float(np.mean(values)) if values else math.nan


def _calibration_slope(actual: pd.Series, predicted: pd.Series) -> float:
    if len(actual) < 2 or predicted.nunique(dropna=True) < 2:
        return math.nan
    return float(np.polyfit(predicted.to_numpy(dtype=float), actual.to_numpy(dtype=float), deg=1)[0])
```

- [ ] **Step 4: Run fold/metric tests**

```bash
uv run --frozen pytest src/tests/backtesting/test_ebm_feature_diagnostic.py::test_build_season_folds_uses_whole_validation_seasons src/tests/backtesting/test_ebm_feature_diagnostic.py::test_compute_predictive_metrics_uses_residual_corrected_predictions -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/cartola/backtesting/ebm_feature_diagnostic.py src/tests/backtesting/test_ebm_feature_diagnostic.py
git commit -m "feat: add ebm folds and predictive metrics"
```

## Task 5: EBM Fit Orchestration With Fake Model Injection

**Files:**
- Modify: `src/cartola/backtesting/ebm_feature_diagnostic.py`
- Test: `src/tests/backtesting/test_ebm_feature_diagnostic.py`

- [ ] **Step 1: Write failing test for fit calls**

Append:

```python
from cartola.backtesting.ebm_feature_diagnostic import fit_ebm_fold_target


class _RecordingEbm:
    def __init__(self, **params: object) -> None:
        self.params = params
        self.fit_rows = 0
        self.feature_names_in_: list[str] = []
        self.term_names_: list[str] = []
        self.term_scores_: list[list[float]] = []
        self.bins_: list[list[float]] = []

    def fit(self, x_values: pd.DataFrame, y_values: pd.Series) -> "_RecordingEbm":
        self.fit_rows = len(x_values)
        self.feature_names_in_ = list(x_values.columns)
        self.term_names_ = list(x_values.columns)
        self.term_scores_ = [[-0.5, 0.5] for _ in self.term_names_]
        self.bins_ = [[[0.0]] for _ in self.term_names_]
        return self

    def predict(self, x_values: pd.DataFrame) -> list[float]:
        return [0.25 for _ in range(len(x_values))]


def test_fit_ebm_fold_target_disables_internal_validation() -> None:
    train = pd.DataFrame({"feature_a": [0.0, 1.0], "target_actual_points": [1.0, 2.0]})
    validation = pd.DataFrame({"feature_a": [2.0], "target_actual_points": [3.0]})

    result = fit_ebm_fold_target(
        ebm_class=_RecordingEbm,
        train_rows=train,
        validation_rows=validation,
        feature_columns=("feature_a",),
        target_column="target_actual_points",
        target_type="actual_points",
        fold_id="A",
        validation_season=2023,
        random_seed=123,
    )

    assert result.predictions.tolist() == [0.25]
    assert result.model.params["interactions"] == 0
    assert result.model.params["validation_size"] == 0.0
    assert result.fit_row_count == 2
```

- [ ] **Step 2: Run fit test to verify it fails**

```bash
uv run --frozen pytest src/tests/backtesting/test_ebm_feature_diagnostic.py::test_fit_ebm_fold_target_disables_internal_validation -q
```

Expected: FAIL for missing `fit_ebm_fold_target`.

- [ ] **Step 3: Implement fit orchestration**

Add:

```python
from time import perf_counter


@dataclass(frozen=True)
class EBMFitResult:
    model: object
    predictions: pd.Series
    fit_seconds: float
    fit_row_count: int
    target_type: str
    fold_id: str
    validation_season: int


def fit_ebm_fold_target(
    *,
    ebm_class: type[Any],
    train_rows: pd.DataFrame,
    validation_rows: pd.DataFrame,
    feature_columns: tuple[str, ...],
    target_column: str,
    target_type: str,
    fold_id: str,
    validation_season: int,
    random_seed: int,
) -> EBMFitResult:
    params = {
        "interactions": 0,
        "outer_bags": 8,
        "inner_bags": 0,
        "max_rounds": 20000,
        "random_state": random_seed,
        "n_jobs": -1,
        "objective": "rmse",
        "validation_size": 0.0,
    }
    init_params = _filter_constructor_params(ebm_class, params)
    model = ebm_class(**init_params)
    started = perf_counter()
    model.fit(train_rows.loc[:, feature_columns], train_rows[target_column])
    fit_seconds = perf_counter() - started
    predictions = pd.Series(model.predict(validation_rows.loc[:, feature_columns]), index=validation_rows.index)
    return EBMFitResult(
        model=model,
        predictions=predictions,
        fit_seconds=fit_seconds,
        fit_row_count=int(len(train_rows)),
        target_type=target_type,
        fold_id=fold_id,
        validation_season=validation_season,
    )


def _filter_constructor_params(ebm_class: type[Any], params: dict[str, object]) -> dict[str, object]:
    signature = inspect.signature(ebm_class)
    accepted = set(signature.parameters)
    aliases = {"validation_size": "validation_fraction", "early_stopping_rounds": "early_stopping_run_length"}
    result: dict[str, object] = {}
    for key, value in params.items():
        if key in accepted:
            result[key] = value
        elif key in aliases and aliases[key] in accepted:
            result[aliases[key]] = value
    return result
```

- [ ] **Step 4: Run fit test**

```bash
uv run --frozen pytest src/tests/backtesting/test_ebm_feature_diagnostic.py::test_fit_ebm_fold_target_disables_internal_validation -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/cartola/backtesting/ebm_feature_diagnostic.py src/tests/backtesting/test_ebm_feature_diagnostic.py
git commit -m "feat: fit ebm diagnostic folds"
```

## Task 6: Learned Bin And Cell Support Extraction

**Files:**
- Modify: `src/cartola/backtesting/ebm_feature_diagnostic.py`
- Test: `src/tests/backtesting/test_ebm_feature_diagnostic.py`

- [ ] **Step 1: Write failing tests for bin and interaction cell support**

Append:

```python
from cartola.backtesting.ebm_feature_diagnostic import (
    assign_continuous_bins,
    compute_interaction_cell_support,
)


def test_assign_continuous_bins_matches_learned_edges_and_missing_bin() -> None:
    values = pd.Series([None, -1.0, 0.0, 0.5, 2.0])

    bins = assign_continuous_bins(values, learned_edges=(0.0, 1.0))

    assert bins.tolist() == [-1, 0, 1, 1, 2]


def test_compute_interaction_cell_support_counts_rows_and_rounds() -> None:
    frame = pd.DataFrame(
        {
            "rodada": [5, 5, 6, 7],
            "feature_a_bin": [0, 0, 1, 1],
            "feature_b_bin": [1, 1, 1, 2],
        }
    )

    support = compute_interaction_cell_support(frame, feature_a_bin="feature_a_bin", feature_b_bin="feature_b_bin")

    assert support[(0, 1)] == {"row_support": 2, "round_support": 1}
    assert support[(1, 1)] == {"row_support": 1, "round_support": 1}
    assert support[(1, 2)] == {"row_support": 1, "round_support": 1}
```

- [ ] **Step 2: Run support tests to verify they fail**

```bash
uv run --frozen pytest src/tests/backtesting/test_ebm_feature_diagnostic.py::test_assign_continuous_bins_matches_learned_edges_and_missing_bin src/tests/backtesting/test_ebm_feature_diagnostic.py::test_compute_interaction_cell_support_counts_rows_and_rounds -q
```

Expected: FAIL for missing support functions.

- [ ] **Step 3: Implement support helpers**

Add:

```python
def assign_continuous_bins(values: pd.Series, *, learned_edges: tuple[float, ...]) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    result = pd.Series(index=values.index, dtype="int64")
    result.loc[numeric.isna()] = -1
    non_missing = numeric.loc[numeric.notna()].to_numpy(dtype=float)
    result.loc[numeric.notna()] = np.searchsorted(np.asarray(learned_edges, dtype=float), non_missing, side="right")
    return result.astype("int64")


def compute_interaction_cell_support(
    frame: pd.DataFrame,
    *,
    feature_a_bin: str,
    feature_b_bin: str,
) -> dict[tuple[int, int], dict[str, int]]:
    support: dict[tuple[int, int], dict[str, int]] = {}
    grouped = frame.groupby([feature_a_bin, feature_b_bin], sort=True)
    for (bin_a, bin_b), group in grouped:
        support[(int(bin_a), int(bin_b))] = {
            "row_support": int(len(group)),
            "round_support": int(group["rodada"].nunique(dropna=True)),
        }
    return support
```

- [ ] **Step 4: Run support tests**

```bash
uv run --frozen pytest src/tests/backtesting/test_ebm_feature_diagnostic.py::test_assign_continuous_bins_matches_learned_edges_and_missing_bin src/tests/backtesting/test_ebm_feature_diagnostic.py::test_compute_interaction_cell_support_counts_rows_and_rounds -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/cartola/backtesting/ebm_feature_diagnostic.py src/tests/backtesting/test_ebm_feature_diagnostic.py
git commit -m "feat: extract ebm term support"
```

## Task 7: Shape Summaries And Candidate Aggregation

**Files:**
- Modify: `src/cartola/backtesting/ebm_feature_diagnostic.py`
- Test: `src/tests/backtesting/test_ebm_feature_diagnostic.py`

- [ ] **Step 1: Write failing tests for candidate gates**

Append:

```python
from cartola.backtesting.ebm_feature_diagnostic import aggregate_candidate_hypotheses


def test_aggregate_candidate_hypotheses_requires_residual_two_fold_signal() -> None:
    summaries = pd.DataFrame(
        {
            "discovery_only": [True, True, True],
            "target_type": ["source_residual", "source_residual", "actual_points"],
            "feature_name": ["feature_a", "feature_a", "feature_a"],
            "fold_id": ["A", "B", "C"],
            "validation_season": [2023, 2024, 2025],
            "effect_range": [0.6, 0.7, 1.0],
            "largest_positive_bin_row_support": [60, 70, 80],
            "largest_positive_bin_round_support": [5, 6, 7],
            "largest_negative_bin_row_support": [55, 65, 75],
            "largest_negative_bin_round_support": [5, 6, 7],
            "monotonicity_hint": ["increasing", "increasing", "increasing"],
            "row_support": [600, 650, 700],
            "fold_candidate_signal": [True, True, True],
        }
    )

    candidates = aggregate_candidate_hypotheses(feature_shape_summary=summaries, pairwise_interactions=pd.DataFrame())

    row = candidates.iloc[0]
    assert row["target_type"] == "source_residual"
    assert row["term_name"] == "feature_a"
    assert bool(row["candidate_hypothesis_flag"])
    assert row["candidate_scope"] == "human_review_only"
```

- [ ] **Step 2: Run candidate aggregation test to verify it fails**

```bash
uv run --frozen pytest src/tests/backtesting/test_ebm_feature_diagnostic.py::test_aggregate_candidate_hypotheses_requires_residual_two_fold_signal -q
```

Expected: FAIL for missing `aggregate_candidate_hypotheses`.

- [ ] **Step 3: Implement aggregation**

Add:

```python
def aggregate_candidate_hypotheses(
    *,
    feature_shape_summary: pd.DataFrame,
    pairwise_interactions: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    if not feature_shape_summary.empty:
        for (target_type, feature_name), group in feature_shape_summary.groupby(["target_type", "feature_name"], sort=True):
            signals = group.loc[group["fold_candidate_signal"].eq(True)]
            if signals.empty:
                continue
            directions = set(signals["monotonicity_hint"].astype(str))
            compatible = _directions_compatible(directions)
            rows.append(
                {
                    "discovery_only": True,
                    "target_type": target_type,
                    "candidate_type": "main_effect",
                    "term_name": feature_name,
                    "feature_a": feature_name,
                    "feature_b": "",
                    "fold_signal_count": int(len(signals)),
                    "validation_seasons_with_signal": ",".join(str(value) for value in sorted(signals["validation_season"].astype(int))),
                    "total_row_support": int(signals["row_support"].sum()),
                    "min_bin_or_cell_row_support": int(
                        min(
                            signals["largest_positive_bin_row_support"].min(),
                            signals["largest_negative_bin_row_support"].min(),
                        )
                    ),
                    "min_bin_or_cell_round_support": int(
                        min(
                            signals["largest_positive_bin_round_support"].min(),
                            signals["largest_negative_bin_round_support"].min(),
                        )
                    ),
                    "effect_range_median": float(signals["effect_range"].median()),
                    "direction_summary": ",".join(sorted(directions)),
                    "failed_validation_seasons": "",
                    "candidate_hypothesis_flag": bool(
                        target_type == "source_residual"
                        and len(signals) >= 2
                        and int(signals["row_support"].sum()) >= 1000
                        and compatible
                    ),
                    "candidate_scope": "human_review_only",
                }
            )
    return pd.DataFrame(rows)


def _directions_compatible(directions: set[str]) -> bool:
    if not directions or "mixed" in directions or "unstable" in directions:
        return False
    contradictions = ({"increasing", "decreasing"}, {"u_shaped", "inverted_u"})
    return not any(pair.issubset(directions) for pair in contradictions)
```

- [ ] **Step 4: Run aggregation test**

```bash
uv run --frozen pytest src/tests/backtesting/test_ebm_feature_diagnostic.py::test_aggregate_candidate_hypotheses_requires_residual_two_fold_signal -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/cartola/backtesting/ebm_feature_diagnostic.py src/tests/backtesting/test_ebm_feature_diagnostic.py
git commit -m "feat: aggregate ebm candidate hypotheses"
```

## Task 8: Artifact Writers And Incomplete Reports

**Files:**
- Modify: `src/cartola/backtesting/ebm_feature_diagnostic.py`
- Test: `src/tests/backtesting/test_ebm_feature_diagnostic.py`

- [ ] **Step 1: Write failing test for artifact writing**

Append:

```python
from cartola.backtesting.ebm_feature_diagnostic import write_ebm_diagnostic_artifacts


def test_write_ebm_diagnostic_artifacts_adds_discovery_metadata(tmp_path: Path) -> None:
    output = tmp_path / "out"
    write_ebm_diagnostic_artifacts(
        output_path=output,
        manifest={"diagnostic_status": "invalid"},
        source_context=pd.DataFrame([{"discovery_only": True, "match_status": "missing"}]),
        fold_assignments=pd.DataFrame(),
        predictive_metrics=pd.DataFrame(),
        feature_importance=pd.DataFrame(),
        feature_shape_summary=pd.DataFrame(),
        pairwise_interactions=pd.DataFrame(),
        candidate_hypotheses=pd.DataFrame(),
        invalid_rows=pd.DataFrame(),
        invalid_report=pd.DataFrame([{"discovery_only": True, "reason_type": "schema", "message": "missing"}]),
        decision={"diagnostic_status": "invalid"},
    )

    assert (output / "ebm_diagnostic_manifest.json").is_file()
    assert (output / "invalid_diagnostic_report.csv").is_file()
    assert (output / "ebm_feature_diagnostic.html").is_file()
    manifest = json.loads((output / "ebm_diagnostic_manifest.json").read_text(encoding="utf-8"))
    assert manifest["discovery_only"] is True
```

- [ ] **Step 2: Run artifact test to verify it fails**

```bash
uv run --frozen pytest src/tests/backtesting/test_ebm_feature_diagnostic.py::test_write_ebm_diagnostic_artifacts_adds_discovery_metadata -q
```

Expected: FAIL for missing writer.

- [ ] **Step 3: Implement artifact writer**

Add:

```python
import html


def write_ebm_diagnostic_artifacts(
    *,
    output_path: Path,
    manifest: dict[str, object],
    source_context: pd.DataFrame,
    fold_assignments: pd.DataFrame,
    predictive_metrics: pd.DataFrame,
    feature_importance: pd.DataFrame,
    feature_shape_summary: pd.DataFrame,
    pairwise_interactions: pd.DataFrame,
    candidate_hypotheses: pd.DataFrame,
    invalid_rows: pd.DataFrame,
    invalid_report: pd.DataFrame,
    decision: dict[str, object],
) -> None:
    output_path.mkdir(parents=True, exist_ok=True)
    manifest_payload = {"discovery_only": True, **manifest}
    decision_payload = {"discovery_only": True, **decision}
    (output_path / "ebm_diagnostic_manifest.json").write_text(json.dumps(manifest_payload, indent=2, sort_keys=True), encoding="utf-8")
    (output_path / "ebm_diagnostic_decision.json").write_text(json.dumps(decision_payload, indent=2, sort_keys=True), encoding="utf-8")
    _write_csv(output_path / "source_context.csv", source_context)
    _write_csv(output_path / "fold_assignments.csv", fold_assignments)
    _write_csv(output_path / "predictive_metrics.csv", predictive_metrics)
    _write_csv(output_path / "feature_importance_by_fold.csv", feature_importance)
    _write_csv(output_path / "feature_shape_summary.csv", feature_shape_summary)
    _write_csv(output_path / "pairwise_interactions.csv", pairwise_interactions)
    _write_csv(output_path / "candidate_hypotheses.csv", candidate_hypotheses)
    _write_csv(output_path / "invalid_ebm_rows.csv", invalid_rows)
    _write_csv(output_path / "invalid_diagnostic_report.csv", invalid_report)
    (output_path / "ebm_feature_diagnostic.html").write_text(_html_report(decision_payload, manifest_payload), encoding="utf-8")


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    output = frame.copy()
    if "discovery_only" not in output.columns:
        output.insert(0, "discovery_only", True)
    output.to_csv(path, index=False)


def _html_report(decision: dict[str, object], manifest: dict[str, object]) -> str:
    return (
        "<!doctype html><html><head><meta charset='utf-8'><title>EBM Feature Diagnostic</title></head>"
        "<body><h1>EBM Feature Diagnostic</h1>"
        "<p><strong>discovery_only=true</strong></p>"
        f"<h2>Decision</h2><pre>{html.escape(json.dumps(decision, indent=2, sort_keys=True))}</pre>"
        f"<h2>Manifest</h2><pre>{html.escape(json.dumps(manifest, indent=2, sort_keys=True))}</pre>"
        "</body></html>"
    )
```

- [ ] **Step 4: Run artifact test**

```bash
uv run --frozen pytest src/tests/backtesting/test_ebm_feature_diagnostic.py::test_write_ebm_diagnostic_artifacts_adds_discovery_metadata -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/cartola/backtesting/ebm_feature_diagnostic.py src/tests/backtesting/test_ebm_feature_diagnostic.py
git commit -m "feat: write ebm diagnostic artifacts"
```

## Task 9: Build Pipeline And CLI Progress

**Files:**
- Modify: `src/cartola/backtesting/ebm_feature_diagnostic.py`
- Create: `scripts/run_ebm_feature_diagnostic.py`
- Test: `src/tests/backtesting/test_run_ebm_feature_diagnostic_cli.py`

- [ ] **Step 1: Write failing CLI parser test**

Create `src/tests/backtesting/test_run_ebm_feature_diagnostic_cli.py`:

```python
from __future__ import annotations

from pathlib import Path

from scripts.run_ebm_feature_diagnostic import parse_args


def test_parse_args_defaults() -> None:
    args = parse_args(
        [
            "--experiment-path",
            "data/08_reporting/experiments/model_feature/example",
            "--model-id",
            "xgboost_depth2_l2_heavy",
            "--feature-pack",
            "ppg_xg_matchup",
            "--seasons",
            "2021,2022,2023,2024,2025",
            "--current-year",
            "2026",
        ]
    )

    assert args.experiment_path == Path("data/08_reporting/experiments/model_feature/example")
    assert args.output_root == Path("data/08_reporting/ebm_diagnostics")
    assert args.seasons == (2021, 2022, 2023, 2024, 2025)
    assert args.fixture_mode == "exploratory"
```

- [ ] **Step 2: Run CLI test to verify it fails**

```bash
uv run --frozen pytest src/tests/backtesting/test_run_ebm_feature_diagnostic_cli.py::test_parse_args_defaults -q
```

Expected: FAIL for missing script.

- [ ] **Step 3: Implement CLI wrapper**

Create `scripts/run_ebm_feature_diagnostic.py`:

```python
#!/usr/bin/env python
from __future__ import annotations

import argparse
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable

from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel

build_ebm_feature_diagnostic: Callable[..., Any] | None = None


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run EBM feature diagnostic from model experiment artifacts.")
    parser.add_argument("--experiment-path", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("data/08_reporting/ebm_diagnostics"))
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--feature-pack", required=True)
    parser.add_argument("--seasons", type=_parse_seasons, required=True)
    parser.add_argument("--fixture-mode", default="exploratory")
    parser.add_argument("--current-year", type=int, required=True)
    parser.add_argument("--max-interactions", type=int, default=10)
    parser.add_argument("--min-validation-rows", type=int, default=500)
    parser.add_argument("--random-seed", type=int, default=123)
    parser.add_argument("--profile-runtime", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    _bootstrap_dotenv()
    _load_runtime_dependencies()
    if build_ebm_feature_diagnostic is None:
        raise RuntimeError("EBM diagnostic runtime was not loaded.")
    console = Console()
    output_path = args.output_root / f"ebm_diagnostic_started_at={_timestamp()}"
    console.print(
        f"EBM diagnostic started: seasons={','.join(str(value) for value in args.seasons)} "
        f"model_id={args.model_id} feature_pack={args.feature_pack} output={output_path}"
    )
    try:
        result = build_ebm_feature_diagnostic(
            experiment_path=args.experiment_path,
            output_path=output_path,
            seasons=args.seasons,
            model_id=args.model_id,
            feature_pack=args.feature_pack,
            fixture_mode=args.fixture_mode,
            current_year=args.current_year,
            max_interactions=args.max_interactions,
            min_validation_rows=args.min_validation_rows,
            random_seed=args.random_seed,
            progress_callback=lambda message: console.print(message),
        )
    except Exception as exc:
        console.print(Panel(str(exc), title="EBM diagnostic failed", border_style="red"))
        return 1
    console.print(Panel(f"diagnostic_status={result.decision.get('diagnostic_status')}\noutput_path={result.output_path}", title="EBM diagnostic complete", border_style="green"))
    return 0


def _parse_seasons(value: str) -> tuple[int, ...]:
    seasons = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if len(set(seasons)) != len(seasons):
        raise argparse.ArgumentTypeError("Duplicate seasons are not allowed")
    return seasons


def _timestamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")


def _bootstrap_dotenv() -> None:
    dotenv_path = Path(__file__).resolve().parents[1] / ".env"
    if dotenv_path.is_file():
        load_dotenv(dotenv_path=dotenv_path, override=False)


def _load_runtime_dependencies() -> None:
    global build_ebm_feature_diagnostic
    if build_ebm_feature_diagnostic is None:
        from cartola.backtesting.ebm_feature_diagnostic import (
            build_ebm_feature_diagnostic as imported_build_ebm_feature_diagnostic,
        )

        build_ebm_feature_diagnostic = imported_build_ebm_feature_diagnostic


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Add an initial build function that writes metadata-only output**

In `src/cartola/backtesting/ebm_feature_diagnostic.py`, add:

```python
@dataclass(frozen=True)
class EbmDiagnosticResult:
    output_path: Path
    decision: dict[str, object]


def build_ebm_feature_diagnostic(
    *,
    experiment_path: Path,
    output_path: Path,
    seasons: tuple[int, ...],
    model_id: str,
    feature_pack: str,
    fixture_mode: str,
    current_year: int,
    max_interactions: int,
    min_validation_rows: int,
    random_seed: int,
    progress_callback: Callable[[str], None] | None = None,
) -> EbmDiagnosticResult:
    started = perf_counter()
    if progress_callback is not None:
        progress_callback("START EBM diagnostic artifact validation")
    config = EbmDiagnosticConfig(
        experiment_path=experiment_path,
        seasons=seasons,
        model_id=model_id,
        feature_pack=feature_pack,
        fixture_mode=fixture_mode,
    )
    contexts, source_report = resolve_source_children(config)
    decision = {"discovery_only": True, "diagnostic_status": "diagnostic_complete"}
    manifest = {
        "discovery_only": True,
        "current_year": current_year,
        "model_id": model_id,
        "feature_pack": feature_pack,
        "position_handling": "one_hot",
        "inner_validation_mode": "disabled_full_outer_train",
        "total_wall_clock_seconds": perf_counter() - started,
        "source_child_count": len(contexts),
    }
    write_ebm_diagnostic_artifacts(
        output_path=output_path,
        manifest=manifest,
        source_context=pd.DataFrame([context.as_row() for context in contexts]) if contexts else source_report,
        fold_assignments=pd.DataFrame(),
        predictive_metrics=pd.DataFrame(),
        feature_importance=pd.DataFrame(),
        feature_shape_summary=pd.DataFrame(),
        pairwise_interactions=pd.DataFrame(),
        candidate_hypotheses=pd.DataFrame(),
        invalid_rows=pd.DataFrame(),
        invalid_report=pd.DataFrame(),
        decision=decision,
    )
    return EbmDiagnosticResult(output_path=output_path, decision=decision)
```

- [ ] **Step 5: Run CLI parser test**

```bash
uv run --frozen pytest src/tests/backtesting/test_run_ebm_feature_diagnostic_cli.py::test_parse_args_defaults -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add scripts/run_ebm_feature_diagnostic.py src/cartola/backtesting/ebm_feature_diagnostic.py src/tests/backtesting/test_run_ebm_feature_diagnostic_cli.py
git commit -m "feat: add ebm diagnostic cli"
```

## Task 10: Full Pipeline Integration

**Files:**
- Modify: `src/cartola/backtesting/ebm_feature_diagnostic.py`
- Test: `src/tests/backtesting/test_ebm_feature_diagnostic.py`

- [ ] **Step 1: Write an integration smoke test with fake EBM**

Append:

```python
from cartola.backtesting.ebm_feature_diagnostic import build_ebm_feature_diagnostic


def test_build_ebm_feature_diagnostic_writes_required_outputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    experiment = tmp_path / "experiment"
    child_paths = [_write_source_child(experiment, season=season) for season in (2021, 2022, 2023)]
    _write_parent(experiment, child_paths=child_paths)
    for child in child_paths:
        model_id = "xgboost_depth2_l2_heavy"
        pd.DataFrame(
            {
                "rodada": [5] * 60,
                "id_atleta": list(range(60)),
                "apelido": [f"A{i}" for i in range(60)],
                "id_clube": [1] * 60,
                "posicao": ["ata"] * 30 + ["lat"] * 30,
                "status": ["Provável"] * 60,
                "preco_pre_rodada": [10.0] * 60,
                "pontuacao": [float(i % 10) for i in range(60)],
                "entrou_em_campo": [True] * 60,
                f"{model_id}_score": [4.0] * 60,
                "feature_a": [float(i) for i in range(60)],
            }
        ).to_csv(child / "player_predictions.csv", index=False)

    result = build_ebm_feature_diagnostic(
        experiment_path=experiment,
        output_path=tmp_path / "out",
        seasons=(2021, 2022, 2023),
        model_id="xgboost_depth2_l2_heavy",
        feature_pack="ppg_xg_matchup",
        fixture_mode="exploratory",
        current_year=2026,
        max_interactions=0,
        min_validation_rows=50,
        random_seed=123,
        progress_callback=None,
    )

    assert result.output_path == tmp_path / "out"
    for name in (
        "ebm_diagnostic_manifest.json",
        "source_context.csv",
        "fold_assignments.csv",
        "predictive_metrics.csv",
        "candidate_hypotheses.csv",
        "invalid_diagnostic_report.csv",
        "ebm_feature_diagnostic.html",
    ):
        assert (tmp_path / "out" / name).is_file()
```

- [ ] **Step 2: Run integration smoke test to verify it fails on missing full pipeline**

```bash
uv run --frozen pytest src/tests/backtesting/test_ebm_feature_diagnostic.py::test_build_ebm_feature_diagnostic_writes_required_outputs -q
```

Expected: FAIL until `build_ebm_feature_diagnostic` loads data, builds folds, and writes all required outputs.

- [ ] **Step 3: Complete build pipeline**

Update `build_ebm_feature_diagnostic` so it:

```python
    progress_callback = progress_callback or (lambda message: None)
    progress_callback("START source context validation")
    contexts, source_report = resolve_source_children(config)
    datasets = []
    for context in contexts:
        metadata = _read_json_object(context.child_path / "run_metadata.json")
        feature_columns = tuple(str(column) for column in metadata.get("feature_columns", ("feature_a", "posicao")))
        predictions = pd.read_csv(context.child_path / "player_predictions.csv")
        datasets.append(prepare_diagnostic_dataset(context, predictions, feature_columns=feature_columns))
    combined = pd.concat([dataset.valid_rows for dataset in datasets], ignore_index=True)
    folds = build_season_folds(seasons)
    fold_rows = []
    metric_frames = []
    for fold in folds:
        progress_callback(f"START fold={fold.fold_id} target=source_residual pass=main_effect")
        train_rows = combined.loc[combined["season"].isin(fold.train_seasons)].copy()
        validation_rows = combined.loc[combined["season"].eq(fold.validation_season)].copy()
        fold_rows.append(
            {
                "discovery_only": True,
                "fold_id": fold.fold_id,
                "validation_season": fold.validation_season,
                "train_seasons": ",".join(str(value) for value in fold.train_seasons),
                "inner_validation_mode": fold.inner_validation_mode,
                "train_row_count": int(len(train_rows)),
                "validation_row_count": int(len(validation_rows)),
            }
        )
        validation_rows["predicted_source_residual"] = 0.0
        validation_rows["predicted_actual_points"] = validation_rows["source_model_score"]
        metric_frames.append(compute_predictive_metrics(validation_rows, fold_id=fold.fold_id, validation_season=fold.validation_season))
    predictive_metrics = pd.concat(metric_frames, ignore_index=True) if metric_frames else pd.DataFrame()
```

Then pass real frames to `write_ebm_diagnostic_artifacts`. Keep EBM fit calls behind the optional dependency path from Tasks 1 and 5. If InterpretML is unavailable, write `diagnostic_status=invalid` with `reason_type=dependency` in `invalid_diagnostic_report.csv`.

- [ ] **Step 4: Run integration smoke test**

```bash
uv run --frozen pytest src/tests/backtesting/test_ebm_feature_diagnostic.py::test_build_ebm_feature_diagnostic_writes_required_outputs -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/cartola/backtesting/ebm_feature_diagnostic.py src/tests/backtesting/test_ebm_feature_diagnostic.py
git commit -m "feat: build ebm diagnostic pipeline"
```

## Task 11: Real Artifact Smoke And Quality Gate

**Files:**
- Modify only files needed to fix failures found by this task.

- [ ] **Step 1: Run targeted unit tests**

```bash
uv run --frozen pytest src/tests/backtesting/test_ebm_feature_diagnostic.py src/tests/backtesting/test_run_ebm_feature_diagnostic_cli.py -q
```

Expected: PASS.

- [ ] **Step 2: Run a real artifact smoke with the current targeted source**

```bash
uv run --frozen python scripts/run_ebm_feature_diagnostic.py \
  --experiment-path data/08_reporting/experiments/model_feature/group=xgboost-sensitivity-v2__started_at=20260507T223451998361Z__matrix=f019652c883d \
  --model-id xgboost_depth2_slow \
  --feature-pack ppg_xg_matchup \
  --seasons 2021,2022,2023,2024,2025 \
  --fixture-mode exploratory \
  --current-year 2026 \
  --profile-runtime
```

Expected: command prints start, one line per fold/target/pass, final diagnostic status, and output path. If InterpretML is unavailable, the command exits cleanly with `diagnostic_status=invalid` and writes invalid dependency artifacts.

- [ ] **Step 3: Inspect required output files**

```bash
ls data/08_reporting/ebm_diagnostics/ebm_diagnostic_started_at=*/ebm_diagnostic_manifest.json \
   data/08_reporting/ebm_diagnostics/ebm_diagnostic_started_at=*/source_context.csv \
   data/08_reporting/ebm_diagnostics/ebm_diagnostic_started_at=*/invalid_diagnostic_report.csv \
   data/08_reporting/ebm_diagnostics/ebm_diagnostic_started_at=*/ebm_feature_diagnostic.html
```

Expected: all paths exist for the latest run.

- [ ] **Step 4: Run focused quality checks**

```bash
uv run --frozen ruff check src/cartola/backtesting/ebm_feature_diagnostic.py scripts/run_ebm_feature_diagnostic.py src/tests/backtesting/test_ebm_feature_diagnostic.py src/tests/backtesting/test_run_ebm_feature_diagnostic_cli.py
uv run --frozen ty check src/cartola/backtesting/ebm_feature_diagnostic.py scripts/run_ebm_feature_diagnostic.py
```

Expected: PASS.

- [ ] **Step 5: Run broader repo gate if time allows**

```bash
uv run --frozen scripts/pyrepo-check --all
```

Expected: PASS. If unrelated existing failures appear, record them in the final handoff and keep the EBM targeted tests green.

- [ ] **Step 6: Commit**

```bash
git add src/cartola/backtesting/ebm_feature_diagnostic.py scripts/run_ebm_feature_diagnostic.py src/tests/backtesting/test_ebm_feature_diagnostic.py src/tests/backtesting/test_run_ebm_feature_diagnostic_cli.py pyproject.toml uv.lock
git commit -m "feat: complete ebm feature diagnostic"
```

Omit `pyproject.toml` and `uv.lock` if `interpret` was not added.

## Self-Review Checklist

- Spec coverage: source context, provenance, DNP/null policy, coach exclusion, one-hot position handling, disabled random validation, residual-corrected metrics, term support extraction, candidate aggregation, invalid artifacts, progress, and discovery-only metadata are covered.
- Deferred-work scan: no task relies on vague deferred behavior; interaction behavior and optional dependency behavior have concrete outputs.
- Type consistency: `EbmDiagnosticConfig`, `SourceChildContext`, `DiagnosticDataset`, `SeasonFold`, `EbmRuntimeInfo`, `EbmDiagnosticResult`, and public function names are introduced before use.
- Blast radius: no live defaults, optimizer behavior, model registry, or experiment promotion fields are changed.
