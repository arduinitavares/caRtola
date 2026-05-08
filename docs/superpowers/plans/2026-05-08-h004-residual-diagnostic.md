# H004 Residual Diagnostic Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build Phase 1 of H004: an artifact-backed residual diagnostic that decides whether attack-vs-defense mismatch signal is strong enough to justify implementing the frozen H004 feature pack.

**Architecture:** Add a pure reporting module under `src/cartola/backtesting/h004_residual_diagnostic.py` and a thin CLI wrapper under `scripts/run_h004_residual_diagnostic.py`. The module reads persisted experiment artifacts, validates source context, computes residual metrics and deterministic pass/fail gates, then writes CSV/JSON/HTML artifacts under `data/08_reporting/hypotheses/`.

**Tech Stack:** Python 3.13, pandas, scipy/pandas Spearman ranking, Rich CLI progress, pytest, Ruff, ty.

---

## File Structure

- Create `src/cartola/backtesting/h004_residual_diagnostic.py`
  - Source child discovery and validation.
  - Artifact loading and required-column checks.
  - Residual correlation, quintile, top-actual recall, selected-player profile, and DNP profile metrics.
  - Deterministic diagnostic decision.
  - Artifact writers and HTML report.
- Create `scripts/run_h004_residual_diagnostic.py`
  - CLI argument parsing.
  - `.env` bootstrap before runtime imports.
  - Rich progress/logging and success/failure panels.
- Create `src/tests/backtesting/test_h004_residual_diagnostic.py`
  - Unit tests for source context, validation, metrics, decision gates, schemas, and artifact writing.
- Create `src/tests/backtesting/test_run_h004_residual_diagnostic_cli.py`
  - CLI parse/bootstrap/smoke tests.
- No changes in live recommendation scripts, optimizer policies, or experiment promotion logic.

## Output Schemas

`h004_residual_correlations.csv`:

```text
season,position,signal_family,context_column,row_count,spearman,quintile_residual_spread,passes_signal
```

`h004_residual_quintiles.csv`:

```text
season,position,context_column,quintile,row_count,context_min,context_max,mean_residual,median_residual
```

`h004_top_actual_recall.csv`:

```text
season,position,row_count,median_predicted_rank_percentile,median_context_edge,passes_signal
```

`h004_selected_residual_profile.csv`:

```text
season,position,scope,row_count,mean_residual,median_residual,mean_predicted_points,mean_actual_points
```

`h004_dnp_context_profile.csv`:

```text
season,position,row_count,dnp_rate,mean_footystats_xg_diff,mean_matchup_opponent_allowed_position_points_roll5
```

`h004_diagnostic_decision.json`:

```json
{
  "diagnostic_status": "passes",
  "passed_families": ["A"],
  "family_results": {
    "A": {"passed": true, "passed_seasons": [2021, 2022, 2024]},
    "B": {"passed": false, "passed_seasons": []},
    "C": {"passed": false, "passed_seasons": [2022]}
  },
  "source_experiment_path": "...",
  "source_children": [],
  "score_column_mapping": {"xgboost_depth2_slow": "xgboost_depth2_slow_score"},
  "fixture_identity_status": "verified",
  "footystats_source_identity": {},
  "missing_or_invalid_columns": []
}
```

`h004_residual_diagnostic.html`:

```text
Self-contained HTML summary with decision JSON, family result table, correlations, recall, selected residual profile, DNP profile, and source paths.
```

## Task 1: Add Source Context Discovery

**Files:**
- Create: `src/cartola/backtesting/h004_residual_diagnostic.py`
- Test: `src/tests/backtesting/test_h004_residual_diagnostic.py`

- [ ] **Step 1: Write failing tests for source child discovery**

Add this to `src/tests/backtesting/test_h004_residual_diagnostic.py`:

```python
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from cartola.backtesting.h004_residual_diagnostic import (
    H004_CONTROL_FEATURE_PACK,
    H004_CONTROL_MODEL_ID,
    H004_PRIMARY_SCORE_COLUMN,
    H004SourceChild,
    discover_h004_source_children,
)


def _write_child(tmp_path: Path, *, season: int = 2025) -> Path:
    child = (
        tmp_path
        / "experiment"
        / "runs"
        / f"season={season}"
        / "model=xgboost_depth2_slow"
        / "feature_pack=ppg_xg_matchup"
    )
    child.mkdir(parents=True)
    (child / "run_metadata.json").write_text(
        json.dumps(
            {
                "season": season,
                "model_id": "xgboost_depth2_slow",
                "feature_pack": "ppg_xg_matchup",
                "fixture_mode": "exploratory",
                "matchup_context_mode": "cartola_matchup_v1",
                "footystats_mode": "ppg_xg",
                "scoring_contract_version": "cartola_standard_2026_v1",
                "budget_policy": "moving",
                "fixture_identity_status": "verified",
                "footystats_matches_source_sha256": "footy-sha",
            }
        ),
        encoding="utf-8",
    )
    return child


def test_discover_h004_source_children_derives_season_from_context_not_prediction_csv(tmp_path: Path) -> None:
    child = _write_child(tmp_path, season=2025)

    children = discover_h004_source_children(
        experiment_path=tmp_path / "experiment",
        seasons=(2025,),
        model_id=H004_CONTROL_MODEL_ID,
        feature_pack=H004_CONTROL_FEATURE_PACK,
    )

    assert children == (
        H004SourceChild(
            season=2025,
            model_id="xgboost_depth2_slow",
            feature_pack="ppg_xg_matchup",
            child_path=child,
            score_column=H004_PRIMARY_SCORE_COLUMN,
            fixture_mode="exploratory",
            matchup_context_mode="cartola_matchup_v1",
            footystats_mode="ppg_xg",
            fixture_identity_status="verified",
            footystats_source_identity={"footystats_matches_source_sha256": "footy-sha"},
        ),
    )


def test_discover_h004_source_children_fails_when_child_is_missing(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="season=2025.*xgboost_depth2_slow.*ppg_xg_matchup"):
        discover_h004_source_children(
            experiment_path=tmp_path / "experiment",
            seasons=(2025,),
            model_id=H004_CONTROL_MODEL_ID,
            feature_pack=H004_CONTROL_FEATURE_PACK,
        )
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
uv run --frozen pytest src/tests/backtesting/test_h004_residual_diagnostic.py::test_discover_h004_source_children_derives_season_from_context_not_prediction_csv src/tests/backtesting/test_h004_residual_diagnostic.py::test_discover_h004_source_children_fails_when_child_is_missing -q
```

Expected: FAIL with `ModuleNotFoundError: No module named 'cartola.backtesting.h004_residual_diagnostic'`.

- [ ] **Step 3: Implement source context discovery**

Create `src/cartola/backtesting/h004_residual_diagnostic.py`:

```python
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

H004_CONTROL_MODEL_ID = "xgboost_depth2_slow"
H004_CONTROL_FEATURE_PACK = "ppg_xg_matchup"
H004_PRIMARY_SCORE_COLUMN = "xgboost_depth2_slow_score"
H004_REQUIRED_SEASONS: tuple[int, ...] = (2021, 2022, 2023, 2024, 2025)


@dataclass(frozen=True)
class H004SourceChild:
    season: int
    model_id: str
    feature_pack: str
    child_path: Path
    score_column: str
    fixture_mode: str
    matchup_context_mode: str
    footystats_mode: str
    fixture_identity_status: str
    footystats_source_identity: dict[str, str]

    def as_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["child_path"] = str(self.child_path)
        return payload


def discover_h004_source_children(
    *,
    experiment_path: Path,
    seasons: tuple[int, ...],
    model_id: str,
    feature_pack: str,
) -> tuple[H004SourceChild, ...]:
    children: list[H004SourceChild] = []
    for season in seasons:
        child_path = experiment_path / "runs" / f"season={season}" / f"model={model_id}" / f"feature_pack={feature_pack}"
        if not child_path.is_dir():
            raise FileNotFoundError(
                f"Missing H004 source child for season={season} model={model_id} feature_pack={feature_pack}: "
                f"{child_path}"
            )
        metadata = _read_json(child_path / "run_metadata.json")
        children.append(
            H004SourceChild(
                season=_metadata_int(metadata, "season", fallback=season),
                model_id=_metadata_str(metadata, "model_id", fallback=model_id),
                feature_pack=_metadata_str(metadata, "feature_pack", fallback=feature_pack),
                child_path=child_path,
                score_column=f"{model_id}_score",
                fixture_mode=_metadata_str(metadata, "fixture_mode"),
                matchup_context_mode=_metadata_str(metadata, "matchup_context_mode"),
                footystats_mode=_metadata_str(metadata, "footystats_mode"),
                fixture_identity_status=_metadata_str(metadata, "fixture_identity_status", fallback="unverified"),
                footystats_source_identity=_footystats_source_identity(metadata),
            )
        )
    return tuple(children)


def _read_json(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def _metadata_str(metadata: dict[str, object], key: str, *, fallback: str | None = None) -> str:
    value = metadata.get(key, fallback)
    if value is None or str(value).strip() == "":
        raise ValueError(f"Missing H004 source metadata field: {key}")
    return str(value)


def _metadata_int(metadata: dict[str, object], key: str, *, fallback: int) -> int:
    value = metadata.get(key, fallback)
    try:
        return int(str(value))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid H004 source metadata field {key}: {value!r}") from exc


def _footystats_source_identity(metadata: dict[str, object]) -> dict[str, str]:
    return {
        str(key): str(value)
        for key, value in metadata.items()
        if str(key).startswith("footystats_") and ("sha" in str(key) or "source" in str(key))
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run:

```bash
uv run --frozen pytest src/tests/backtesting/test_h004_residual_diagnostic.py::test_discover_h004_source_children_derives_season_from_context_not_prediction_csv src/tests/backtesting/test_h004_residual_diagnostic.py::test_discover_h004_source_children_fails_when_child_is_missing -q
```

Expected: `2 passed`.

- [ ] **Step 5: Commit**

Run:

```bash
git add src/cartola/backtesting/h004_residual_diagnostic.py src/tests/backtesting/test_h004_residual_diagnostic.py
git commit -m "feat: discover h004 diagnostic source artifacts"
```

Expected: commit succeeds.

## Task 2: Load And Validate Prediction Artifacts

**Files:**
- Modify: `src/cartola/backtesting/h004_residual_diagnostic.py`
- Test: `src/tests/backtesting/test_h004_residual_diagnostic.py`

- [ ] **Step 1: Add tests for prediction artifact validation**

Append:

```python
from cartola.backtesting.h004_residual_diagnostic import (
    H004PredictionBundle,
    load_h004_prediction_bundle,
)


def _prediction_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "rodada": [5, 5, 5],
            "id_atleta": [1, 2, 3],
            "posicao": ["ata", "mei", "tec"],
            "id_clube": [10, 20, 30],
            "pontuacao": [8.0, 2.0, None],
            "entrou_em_campo": [True, True, False],
            "xgboost_depth2_slow_score": [5.0, 4.0, 3.0],
            "matchup_is_home": [1, 0, 1],
            "footystats_xg_diff": [0.6, -0.2, 0.1],
            "footystats_ppg_diff": [0.8, -0.4, 0.2],
            "matchup_opponent_allowed_points_roll5": [4.0, 5.0, 3.0],
            "matchup_opponent_allowed_position_points_roll5": [6.0, 4.0, 3.0],
            "matchup_club_position_points_roll5": [7.0, 3.5, 2.0],
            "matchup_opponent_allowed_position_count": [5, 5, 0],
            "matchup_club_position_count": [5, 5, 0],
            "position_points_prior": [4.0, 3.0, 2.0],
        }
    )


def _selected_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "rodada": [5],
            "id_atleta": [1],
            "posicao": ["ata"],
            "pontuacao": [8.0],
            "entrou_em_campo": [True],
        }
    )


def test_load_h004_prediction_bundle_adds_context_season_and_residuals(tmp_path: Path) -> None:
    child_path = _write_child(tmp_path, season=2025)
    _prediction_rows().to_csv(child_path / "player_predictions.csv", index=False)
    _selected_rows().to_csv(child_path / "selected_players.csv", index=False)
    child = discover_h004_source_children(
        experiment_path=tmp_path / "experiment",
        seasons=(2025,),
        model_id=H004_CONTROL_MODEL_ID,
        feature_pack=H004_CONTROL_FEATURE_PACK,
    )[0]

    bundle = load_h004_prediction_bundle(child)

    assert isinstance(bundle, H004PredictionBundle)
    assert bundle.played["season"].tolist() == [2025, 2025]
    assert bundle.played["predicted_points"].tolist() == [5.0, 4.0]
    assert bundle.played["prediction_residual"].tolist() == [3.0, -2.0]
    assert bundle.dnp["id_atleta"].tolist() == [3]
    assert bundle.selected_players["season"].tolist() == [2025]
    assert bundle.selected_players["id_atleta"].tolist() == [1]


def test_load_h004_prediction_bundle_fails_for_missing_score_column(tmp_path: Path) -> None:
    child_path = _write_child(tmp_path, season=2025)
    _prediction_rows().drop(columns=["xgboost_depth2_slow_score"]).to_csv(
        child_path / "player_predictions.csv",
        index=False,
    )
    _selected_rows().to_csv(child_path / "selected_players.csv", index=False)
    child = discover_h004_source_children(
        experiment_path=tmp_path / "experiment",
        seasons=(2025,),
        model_id=H004_CONTROL_MODEL_ID,
        feature_pack=H004_CONTROL_FEATURE_PACK,
    )[0]

    with pytest.raises(ValueError, match="xgboost_depth2_slow_score"):
        load_h004_prediction_bundle(child)
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
uv run --frozen pytest src/tests/backtesting/test_h004_residual_diagnostic.py::test_load_h004_prediction_bundle_adds_context_season_and_residuals src/tests/backtesting/test_h004_residual_diagnostic.py::test_load_h004_prediction_bundle_fails_for_missing_score_column -q
```

Expected: FAIL with missing `H004PredictionBundle` or `load_h004_prediction_bundle`.

- [ ] **Step 3: Implement prediction bundle loading**

Add to `h004_residual_diagnostic.py`:

```python
import pandas as pd

H004_REQUIRED_PREDICTION_COLUMNS: tuple[str, ...] = (
    "rodada",
    "id_atleta",
    "posicao",
    "id_clube",
    "pontuacao",
    "entrou_em_campo",
    "matchup_is_home",
    "footystats_xg_diff",
    "footystats_ppg_diff",
    "matchup_opponent_allowed_points_roll5",
    "matchup_opponent_allowed_position_points_roll5",
    "matchup_club_position_points_roll5",
    "matchup_opponent_allowed_position_count",
    "matchup_club_position_count",
    "position_points_prior",
)
H004_REQUIRED_SELECTED_COLUMNS: tuple[str, ...] = (
    "rodada",
    "id_atleta",
    "posicao",
    "pontuacao",
    "entrou_em_campo",
)


@dataclass(frozen=True)
class H004PredictionBundle:
    child: H004SourceChild
    all_candidates: pd.DataFrame
    played: pd.DataFrame
    dnp: pd.DataFrame
    selected_players: pd.DataFrame


def load_h004_prediction_bundle(child: H004SourceChild) -> H004PredictionBundle:
    predictions_path = child.child_path / "player_predictions.csv"
    selected_path = child.child_path / "selected_players.csv"
    if not predictions_path.is_file():
        raise FileNotFoundError(predictions_path)
    if not selected_path.is_file():
        raise FileNotFoundError(selected_path)
    predictions = pd.read_csv(predictions_path)
    selected_players = pd.read_csv(selected_path)
    required = (*H004_REQUIRED_PREDICTION_COLUMNS, child.score_column)
    _validate_columns("player_predictions.csv", predictions, required)
    _validate_columns("selected_players.csv", selected_players, H004_REQUIRED_SELECTED_COLUMNS)

    frame = predictions.copy()
    frame["season"] = child.season
    frame["model_id"] = child.model_id
    frame["feature_pack"] = child.feature_pack
    frame["predicted_points"] = pd.to_numeric(frame[child.score_column], errors="coerce")
    frame["actual_points"] = pd.to_numeric(frame["pontuacao"], errors="coerce")
    frame["entered_field"] = frame["entrou_em_campo"].fillna(False).astype(bool)

    numeric_columns = (
        "rodada",
        "id_atleta",
        "id_clube",
        "predicted_points",
        "actual_points",
        "matchup_is_home",
        "footystats_xg_diff",
        "footystats_ppg_diff",
        "matchup_opponent_allowed_points_roll5",
        "matchup_opponent_allowed_position_points_roll5",
        "matchup_club_position_points_roll5",
        "matchup_opponent_allowed_position_count",
        "matchup_club_position_count",
        "position_points_prior",
    )
    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    invalid_prediction = frame["predicted_points"].isna()
    if bool(invalid_prediction.any()):
        raise ValueError(f"Non-finite H004 predicted points in {predictions_path}")

    played = frame.loc[frame["entered_field"] & frame["actual_points"].notna()].copy()
    played["prediction_residual"] = played["actual_points"] - played["predicted_points"]
    dnp = frame.loc[~frame["entered_field"]].copy()
    selected_frame = selected_players.copy()
    selected_frame["season"] = child.season
    selected_frame["model_id"] = child.model_id
    selected_frame["feature_pack"] = child.feature_pack
    selected_frame["rodada"] = pd.to_numeric(selected_frame["rodada"], errors="coerce")
    selected_frame["id_atleta"] = pd.to_numeric(selected_frame["id_atleta"], errors="coerce")
    return H004PredictionBundle(
        child=child,
        all_candidates=frame,
        played=played,
        dnp=dnp,
        selected_players=selected_frame,
    )


def _validate_columns(frame_name: str, frame: pd.DataFrame, required_columns: tuple[str, ...]) -> None:
    missing = [column for column in required_columns if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing required columns in {frame_name}: {', '.join(missing)}")
```

- [ ] **Step 4: Run tests to verify they pass**

Run:

```bash
uv run --frozen pytest src/tests/backtesting/test_h004_residual_diagnostic.py::test_load_h004_prediction_bundle_adds_context_season_and_residuals src/tests/backtesting/test_h004_residual_diagnostic.py::test_load_h004_prediction_bundle_fails_for_missing_score_column -q
```

Expected: `2 passed`.

- [ ] **Step 5: Commit**

Run:

```bash
git add src/cartola/backtesting/h004_residual_diagnostic.py src/tests/backtesting/test_h004_residual_diagnostic.py
git commit -m "feat: load h004 prediction artifacts"
```

Expected: commit succeeds.

## Task 3: Compute Residual Correlations And Quintiles

**Files:**
- Modify: `src/cartola/backtesting/h004_residual_diagnostic.py`
- Test: `src/tests/backtesting/test_h004_residual_diagnostic.py`

- [ ] **Step 1: Add tests for correlation and quintile outputs**

Append:

```python
from cartola.backtesting.h004_residual_diagnostic import (
    H004_SIGNAL_COLUMNS,
    build_h004_residual_correlations,
    build_h004_residual_quintiles,
)


def _played_signal_rows() -> pd.DataFrame:
    rows = []
    for index in range(120):
        rows.append(
            {
                "season": 2025,
                "posicao": "ata",
                "prediction_residual": float(index) / 20.0,
                "footystats_xg_diff": float(index) / 100.0,
                "matchup_opponent_allowed_position_points_roll5": float(index) / 30.0,
                "matchup_is_home": 1,
            }
        )
    return pd.DataFrame(rows)


def test_build_h004_residual_correlations_requires_minimum_rows_and_flags_signal() -> None:
    correlations = build_h004_residual_correlations(_played_signal_rows())

    row = correlations[
        correlations["context_column"].eq("footystats_xg_diff")
        & correlations["position"].eq("ata")
        & correlations["season"].eq(2025)
    ].iloc[0]
    assert row["row_count"] == 120
    assert row["spearman"] > 0.99
    assert bool(row["passes_signal"])


def test_build_h004_residual_quintiles_outputs_deterministic_quintile_rows() -> None:
    quintiles = build_h004_residual_quintiles(_played_signal_rows())

    subset = quintiles[
        quintiles["context_column"].eq("footystats_xg_diff")
        & quintiles["position"].eq("ata")
        & quintiles["season"].eq(2025)
    ]
    assert subset["quintile"].tolist() == [1, 2, 3, 4, 5]
    assert subset["row_count"].sum() == 120
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
uv run --frozen pytest src/tests/backtesting/test_h004_residual_diagnostic.py::test_build_h004_residual_correlations_requires_minimum_rows_and_flags_signal src/tests/backtesting/test_h004_residual_diagnostic.py::test_build_h004_residual_quintiles_outputs_deterministic_quintile_rows -q
```

Expected: FAIL with missing functions.

- [ ] **Step 3: Implement correlation and quintile builders**

Add:

```python
H004_MIN_CORRELATION_ROWS = 100
H004_MIN_ABS_SPEARMAN = 0.05
H004_MIN_QUINTILE_SPREAD = 0.25
H004_SIGNAL_COLUMNS: tuple[str, ...] = (
    "footystats_xg_diff",
    "matchup_opponent_allowed_position_points_roll5",
    "diagnostic_home_xg_edge",
)


def build_h004_residual_correlations(played: pd.DataFrame) -> pd.DataFrame:
    frame = played.copy()
    frame["diagnostic_home_xg_edge"] = frame["matchup_is_home"] * frame["footystats_xg_diff"]
    rows: list[dict[str, object]] = []
    for season, position, column in _season_position_column_keys(frame, H004_SIGNAL_COLUMNS):
        group = _valid_metric_group(frame, season=season, position=position, column=column)
        row_count = int(len(group))
        spearman = float("nan")
        spread = float("nan")
        passes_signal = False
        if row_count >= H004_MIN_CORRELATION_ROWS and group[column].nunique(dropna=True) > 1:
            spearman = float(group["prediction_residual"].corr(group[column], method="spearman"))
            spread = _quintile_spread(group, column)
            passes_signal = bool(
                pd.notna(spearman)
                and spearman >= H004_MIN_ABS_SPEARMAN
                and pd.notna(spread)
                and spread >= H004_MIN_QUINTILE_SPREAD
            )
        rows.append(
            {
                "season": int(season),
                "position": str(position),
                "signal_family": _signal_family(position=str(position), column=column),
                "context_column": column,
                "row_count": row_count,
                "spearman": spearman,
                "quintile_residual_spread": spread,
                "passes_signal": passes_signal,
            }
        )
    return pd.DataFrame(rows).sort_values(["season", "position", "context_column"], kind="mergesort").reset_index(drop=True)


def build_h004_residual_quintiles(played: pd.DataFrame) -> pd.DataFrame:
    frame = played.copy()
    frame["diagnostic_home_xg_edge"] = frame["matchup_is_home"] * frame["footystats_xg_diff"]
    rows: list[dict[str, object]] = []
    for season, position, column in _season_position_column_keys(frame, H004_SIGNAL_COLUMNS):
        group = _valid_metric_group(frame, season=season, position=position, column=column)
        if group.empty:
            continue
        ranked = group.sort_values([column, "prediction_residual"], kind="mergesort").copy()
        ranked["quintile"] = pd.qcut(ranked[column].rank(method="first"), q=min(5, len(ranked)), labels=False) + 1
        for quintile, quintile_group in ranked.groupby("quintile", sort=True):
            rows.append(
                {
                    "season": int(season),
                    "position": str(position),
                    "context_column": column,
                    "quintile": int(quintile),
                    "row_count": int(len(quintile_group)),
                    "context_min": float(quintile_group[column].min()),
                    "context_max": float(quintile_group[column].max()),
                    "mean_residual": float(quintile_group["prediction_residual"].mean()),
                    "median_residual": float(quintile_group["prediction_residual"].median()),
                }
            )
    return pd.DataFrame(rows).sort_values(["season", "position", "context_column", "quintile"], kind="mergesort").reset_index(drop=True)


def _season_position_column_keys(frame: pd.DataFrame, columns: tuple[str, ...]) -> list[tuple[int, str, str]]:
    seasons = sorted(int(value) for value in frame["season"].dropna().unique())
    positions = sorted(str(value) for value in frame["posicao"].dropna().unique())
    return [(season, position, column) for season in seasons for position in positions for column in columns]


def _valid_metric_group(frame: pd.DataFrame, *, season: int, position: str, column: str) -> pd.DataFrame:
    group = frame.loc[frame["season"].eq(season) & frame["posicao"].eq(position)].copy()
    values = pd.to_numeric(group[column], errors="coerce")
    residuals = pd.to_numeric(group["prediction_residual"], errors="coerce")
    valid = values.notna() & residuals.notna()
    group = group.loc[valid].copy()
    group[column] = values.loc[valid]
    group["prediction_residual"] = residuals.loc[valid]
    return group


def _quintile_spread(group: pd.DataFrame, column: str) -> float:
    if len(group) < 5 or group[column].nunique(dropna=True) < 2:
        return float("nan")
    ranked = group.sort_values([column, "prediction_residual"], kind="mergesort").copy()
    ranked["quintile"] = pd.qcut(ranked[column].rank(method="first"), q=5, labels=False) + 1
    means = ranked.groupby("quintile")["prediction_residual"].mean()
    return float(means.loc[5] - means.loc[1])


def _signal_family(*, position: str, column: str) -> str:
    if position in {"ata", "mei"} and column in {"footystats_xg_diff", "matchup_opponent_allowed_position_points_roll5"}:
        return "A"
    if position in {"gol", "lat", "zag"} and column == "diagnostic_home_xg_edge":
        return "B"
    return "descriptive"
```

- [ ] **Step 4: Run tests**

Run:

```bash
uv run --frozen pytest src/tests/backtesting/test_h004_residual_diagnostic.py::test_build_h004_residual_correlations_requires_minimum_rows_and_flags_signal src/tests/backtesting/test_h004_residual_diagnostic.py::test_build_h004_residual_quintiles_outputs_deterministic_quintile_rows -q
```

Expected: `2 passed`.

- [ ] **Step 5: Commit**

Run:

```bash
git add src/cartola/backtesting/h004_residual_diagnostic.py src/tests/backtesting/test_h004_residual_diagnostic.py
git commit -m "feat: compute h004 residual diagnostics"
```

Expected: commit succeeds.

## Task 4: Add Top-Actual Recall And Decision Gate

**Files:**
- Modify: `src/cartola/backtesting/h004_residual_diagnostic.py`
- Test: `src/tests/backtesting/test_h004_residual_diagnostic.py`

- [ ] **Step 1: Add tests for Family C and final decision**

Append:

```python
from cartola.backtesting.h004_residual_diagnostic import (
    build_h004_diagnostic_decision,
    build_h004_top_actual_recall,
)


def _top_actual_rows() -> pd.DataFrame:
    rows = []
    for season in (2021, 2022, 2023):
        for round_number in range(5, 8):
            for index in range(12):
                rows.append(
                    {
                        "season": season,
                        "rodada": round_number,
                        "posicao": "ata",
                        "id_atleta": season * 1000 + round_number * 100 + index,
                        "actual_points": 20.0 - index if index < 5 else 1.0,
                        "predicted_points": 1.0 + index,
                        "footystats_xg_diff": 2.0 if index < 5 else -1.0,
                        "matchup_opponent_allowed_position_points_roll5": 8.0 if index < 5 else 2.0,
                    }
                )
    return pd.DataFrame(rows)


def test_build_h004_top_actual_recall_detects_context_gap() -> None:
    recall = build_h004_top_actual_recall(_top_actual_rows())

    assert set(recall.loc[recall["passes_signal"], "season"]) == {2021, 2022, 2023}
    assert recall["median_predicted_rank_percentile"].min() >= 0.35
    assert recall["median_context_edge"].min() >= 0.25


def test_build_h004_diagnostic_decision_passes_when_one_family_clears_three_seasons() -> None:
    correlations = pd.DataFrame(
        {
            "season": [2021, 2022, 2023],
            "position": ["ata", "ata", "ata"],
            "signal_family": ["A", "A", "A"],
            "context_column": ["footystats_xg_diff"] * 3,
            "row_count": [120, 120, 120],
            "spearman": [0.08, 0.07, 0.06],
            "quintile_residual_spread": [0.3, 0.4, 0.25],
            "passes_signal": [True, True, True],
        }
    )
    recall = pd.DataFrame(
        columns=[
            "season",
            "position",
            "row_count",
            "median_predicted_rank_percentile",
            "median_context_edge",
            "passes_signal",
        ]
    )

    decision = build_h004_diagnostic_decision(
        correlations=correlations,
        top_actual_recall=recall,
        source_experiment_path=Path("experiment"),
        children=(),
        missing_or_invalid_columns=(),
    )

    assert decision["diagnostic_status"] == "passes"
    assert decision["passed_families"] == ["A"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
uv run --frozen pytest src/tests/backtesting/test_h004_residual_diagnostic.py::test_build_h004_top_actual_recall_detects_context_gap src/tests/backtesting/test_h004_residual_diagnostic.py::test_build_h004_diagnostic_decision_passes_when_one_family_clears_three_seasons -q
```

Expected: FAIL with missing functions.

- [ ] **Step 3: Implement top-actual recall and decision**

Add:

```python
def build_h004_top_actual_recall(played: pd.DataFrame) -> pd.DataFrame:
    frame = played.copy()
    frame["context_edge"] = _context_edge(frame)
    rows: list[pd.DataFrame] = []
    for _, group in frame.groupby(["season", "rodada", "posicao"], sort=True):
        played_count = len(group)
        ranked_prediction = group.copy()
        ranked_prediction["predicted_rank"] = ranked_prediction["predicted_points"].rank(
            method="average",
            ascending=False,
        )
        ranked_prediction["predicted_rank_percentile"] = (
            (ranked_prediction["predicted_rank"] - 1.0) / max(float(played_count - 1), 1.0)
        )
        top_n = min(5, played_count)
        actual_top = ranked_prediction.sort_values(
            ["actual_points", "id_atleta"],
            ascending=[False, True],
            kind="mergesort",
        ).head(top_n)
        rows.append(actual_top)
    if not rows:
        return pd.DataFrame(
            columns=pd.Index(
                [
                    "season",
                    "position",
                    "row_count",
                    "median_predicted_rank_percentile",
                    "median_context_edge",
                    "passes_signal",
                ]
            )
        )
    actual_top = pd.concat(rows, ignore_index=True)
    summary = (
        actual_top.groupby(["season", "posicao"], as_index=False)
        .agg(
            row_count=("id_atleta", "count"),
            median_predicted_rank_percentile=("predicted_rank_percentile", "median"),
            median_context_edge=("context_edge", "median"),
        )
        .rename(columns={"posicao": "position"})
    )
    summary["passes_signal"] = (
        summary["median_predicted_rank_percentile"].ge(0.35)
        & summary["median_context_edge"].ge(0.25)
    )
    return summary.sort_values(["season", "position"], kind="mergesort").reset_index(drop=True)


def build_h004_diagnostic_decision(
    *,
    correlations: pd.DataFrame,
    top_actual_recall: pd.DataFrame,
    source_experiment_path: Path,
    children: tuple[H004SourceChild, ...],
    missing_or_invalid_columns: tuple[str, ...],
) -> dict[str, object]:
    family_a = _family_result(correlations, family="A")
    family_b = _family_result(correlations, family="B")
    family_c_seasons = sorted(
        int(value)
        for value in top_actual_recall.loc[top_actual_recall["passes_signal"], "season"].dropna().unique()
    )
    family_c = {"passed": len(family_c_seasons) >= 3, "passed_seasons": family_c_seasons}
    family_results = {"A": family_a, "B": family_b, "C": family_c}
    passed_families = [family for family, result in family_results.items() if bool(result["passed"])]
    status = "invalid" if missing_or_invalid_columns else ("passes" if passed_families else "rejected")
    return {
        "diagnostic_status": status,
        "passed_families": passed_families,
        "family_results": family_results,
        "source_experiment_path": str(source_experiment_path),
        "source_children": [child.as_dict() for child in children],
        "score_column_mapping": {H004_CONTROL_MODEL_ID: H004_PRIMARY_SCORE_COLUMN},
        "fixture_identity_status": _fixture_identity_status(children),
        "footystats_source_identity": {str(child.season): child.footystats_source_identity for child in children},
        "missing_or_invalid_columns": list(missing_or_invalid_columns),
    }


def _context_edge(frame: pd.DataFrame) -> pd.Series:
    xg = _zscore_by_season_position(frame, "footystats_xg_diff")
    position_allowed = _zscore_by_season_position(frame, "matchup_opponent_allowed_position_points_roll5")
    return xg + position_allowed


def _zscore_by_season_position(frame: pd.DataFrame, column: str) -> pd.Series:
    values = pd.to_numeric(frame[column], errors="coerce")
    grouped = frame.assign(_value=values).groupby(["season", "posicao"])["_value"]
    mean = grouped.transform("mean")
    std = grouped.transform("std").replace(0.0, pd.NA)
    result = (values - mean) / std
    return result.fillna(0.0)


def _family_result(correlations: pd.DataFrame, *, family: str) -> dict[str, object]:
    if correlations.empty:
        return {"passed": False, "passed_seasons": []}
    subset = correlations.loc[correlations["signal_family"].eq(family) & correlations["passes_signal"]]
    seasons = sorted(int(value) for value in subset["season"].dropna().unique())
    return {"passed": len(seasons) >= 3, "passed_seasons": seasons}


def _fixture_identity_status(children: tuple[H004SourceChild, ...]) -> str:
    if not children:
        return "unavailable"
    statuses = {child.fixture_identity_status for child in children}
    return "verified" if statuses == {"verified"} else "unverified"
```

- [ ] **Step 4: Run tests**

Run:

```bash
uv run --frozen pytest src/tests/backtesting/test_h004_residual_diagnostic.py::test_build_h004_top_actual_recall_detects_context_gap src/tests/backtesting/test_h004_residual_diagnostic.py::test_build_h004_diagnostic_decision_passes_when_one_family_clears_three_seasons -q
```

Expected: `2 passed`.

- [ ] **Step 5: Commit**

Run:

```bash
git add src/cartola/backtesting/h004_residual_diagnostic.py src/tests/backtesting/test_h004_residual_diagnostic.py
git commit -m "feat: decide h004 diagnostic families"
```

Expected: commit succeeds.

## Task 5: Add Profiles, Artifact Writer, And HTML Report

**Files:**
- Modify: `src/cartola/backtesting/h004_residual_diagnostic.py`
- Test: `src/tests/backtesting/test_h004_residual_diagnostic.py`

- [ ] **Step 1: Add tests for output artifacts**

Append:

```python
from cartola.backtesting.h004_residual_diagnostic import (
    H004DiagnosticResult,
    build_h004_dnp_context_profile,
    build_h004_selected_residual_profile,
    write_h004_diagnostic_artifacts,
)


def test_build_h004_selected_residual_profile_separates_all_candidates_and_selected() -> None:
    played = pd.DataFrame(
        {
            "season": [2025, 2025],
            "rodada": [5, 5],
            "id_atleta": [1, 2],
            "posicao": ["ata", "ata"],
            "prediction_residual": [3.0, -2.0],
            "predicted_points": [5.0, 4.0],
            "actual_points": [8.0, 2.0],
        }
    )
    selected_players = pd.DataFrame({"season": [2025], "rodada": [5], "id_atleta": [1]})

    profile = build_h004_selected_residual_profile(played, selected_players)

    all_row = profile.loc[profile["scope"].eq("all_candidates")].iloc[0]
    selected_row = profile.loc[profile["scope"].eq("selected_players")].iloc[0]
    assert all_row["row_count"] == 2
    assert all_row["mean_residual"] == 0.5
    assert selected_row["row_count"] == 1
    assert selected_row["mean_residual"] == 3.0


def test_write_h004_diagnostic_artifacts_creates_required_files(tmp_path: Path) -> None:
    output_path = tmp_path / "h004"
    result = H004DiagnosticResult(
        output_path=output_path,
        residual_correlations=pd.DataFrame({"season": [2025], "position": ["ata"]}),
        residual_quintiles=pd.DataFrame({"season": [2025], "position": ["ata"]}),
        top_actual_recall=pd.DataFrame({"season": [2025], "position": ["ata"]}),
        selected_residual_profile=pd.DataFrame({"season": [2025], "position": ["ata"], "scope": ["selected"]}),
        dnp_context_profile=pd.DataFrame({"season": [2025], "position": ["ata"]}),
        decision={"diagnostic_status": "rejected", "passed_families": []},
    )

    write_h004_diagnostic_artifacts(result)

    assert (output_path / "h004_residual_correlations.csv").is_file()
    assert (output_path / "h004_residual_quintiles.csv").is_file()
    assert (output_path / "h004_top_actual_recall.csv").is_file()
    assert (output_path / "h004_selected_residual_profile.csv").is_file()
    assert (output_path / "h004_dnp_context_profile.csv").is_file()
    assert (output_path / "h004_diagnostic_decision.json").is_file()
    html = (output_path / "h004_residual_diagnostic.html").read_text(encoding="utf-8")
    assert "H004 Residual Diagnostic" in html
    assert "diagnostic_status" in html
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
uv run --frozen pytest src/tests/backtesting/test_h004_residual_diagnostic.py::test_build_h004_selected_residual_profile_separates_all_candidates_and_selected src/tests/backtesting/test_h004_residual_diagnostic.py::test_write_h004_diagnostic_artifacts_creates_required_files -q
```

Expected: FAIL with missing `H004DiagnosticResult` or writer.

- [ ] **Step 3: Implement profiles and artifact writer**

Add:

```python
import html


@dataclass(frozen=True)
class H004DiagnosticResult:
    output_path: Path
    residual_correlations: pd.DataFrame
    residual_quintiles: pd.DataFrame
    top_actual_recall: pd.DataFrame
    selected_residual_profile: pd.DataFrame
    dnp_context_profile: pd.DataFrame
    decision: dict[str, object]


def build_h004_selected_residual_profile(played: pd.DataFrame, selected_players: pd.DataFrame) -> pd.DataFrame:
    selected_keys = selected_players[["season", "rodada", "id_atleta"]].drop_duplicates()
    selected_played = played.merge(
        selected_keys,
        on=["season", "rodada", "id_atleta"],
        how="inner",
        validate="many_to_one",
    )
    return pd.concat(
        [
            _profile_frame(played, scope="all_candidates"),
            _profile_frame(selected_played, scope="selected_players"),
        ],
        ignore_index=True,
    )


def build_h004_dnp_context_profile(all_candidates: pd.DataFrame) -> pd.DataFrame:
    frame = all_candidates.copy()
    frame["is_dnp"] = ~frame["entered_field"]
    grouped = (
        frame.groupby(["season", "posicao"], as_index=False)
        .agg(
            row_count=("id_atleta", "count"),
            dnp_count=("is_dnp", "sum"),
            mean_footystats_xg_diff=("footystats_xg_diff", "mean"),
            mean_matchup_opponent_allowed_position_points_roll5=(
                "matchup_opponent_allowed_position_points_roll5",
                "mean",
            ),
        )
        .rename(columns={"posicao": "position"})
    )
    grouped["dnp_rate"] = grouped["dnp_count"] / grouped["row_count"].clip(lower=1)
    return grouped[
        [
            "season",
            "position",
            "row_count",
            "dnp_rate",
            "mean_footystats_xg_diff",
            "mean_matchup_opponent_allowed_position_points_roll5",
        ]
    ].sort_values(["season", "position"], kind="mergesort").reset_index(drop=True)


def write_h004_diagnostic_artifacts(result: H004DiagnosticResult) -> None:
    result.output_path.mkdir(parents=True, exist_ok=True)
    result.residual_correlations.to_csv(result.output_path / "h004_residual_correlations.csv", index=False)
    result.residual_quintiles.to_csv(result.output_path / "h004_residual_quintiles.csv", index=False)
    result.top_actual_recall.to_csv(result.output_path / "h004_top_actual_recall.csv", index=False)
    result.selected_residual_profile.to_csv(result.output_path / "h004_selected_residual_profile.csv", index=False)
    result.dnp_context_profile.to_csv(result.output_path / "h004_dnp_context_profile.csv", index=False)
    (result.output_path / "h004_diagnostic_decision.json").write_text(
        json.dumps(result.decision, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    _write_h004_html_report(result)


def _profile_frame(played: pd.DataFrame, *, scope: str) -> pd.DataFrame:
    if played.empty:
        return pd.DataFrame(
            columns=pd.Index(
                [
                    "season",
                    "position",
                    "scope",
                    "row_count",
                    "mean_residual",
                    "median_residual",
                    "mean_predicted_points",
                    "mean_actual_points",
                ]
            )
        )
    grouped = (
        played.groupby(["season", "posicao"], as_index=False)
        .agg(
            row_count=("id_atleta", "count"),
            mean_residual=("prediction_residual", "mean"),
            median_residual=("prediction_residual", "median"),
            mean_predicted_points=("predicted_points", "mean"),
            mean_actual_points=("actual_points", "mean"),
        )
        .rename(columns={"posicao": "position"})
    )
    grouped["scope"] = scope
    return grouped[
        [
            "season",
            "position",
            "scope",
            "row_count",
            "mean_residual",
            "median_residual",
            "mean_predicted_points",
            "mean_actual_points",
        ]
    ].sort_values(["season", "position", "scope"], kind="mergesort").reset_index(drop=True)


def _write_h004_html_report(result: H004DiagnosticResult) -> None:
    sections = [
        "<h1>H004 Residual Diagnostic</h1>",
        _json_section("Decision", result.decision),
        _table_section("Residual Correlations", result.residual_correlations),
        _table_section("Residual Quintiles", result.residual_quintiles),
        _table_section("Top Actual Recall", result.top_actual_recall),
        _table_section("Selected Residual Profile", result.selected_residual_profile),
        _table_section("DNP Context Profile", result.dnp_context_profile),
    ]
    body = "\n".join(sections)
    (result.output_path / "h004_residual_diagnostic.html").write_text(
        f"<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\"><title>H004 Residual Diagnostic</title></head><body>{body}</body></html>",
        encoding="utf-8",
    )


def _json_section(title: str, payload: dict[str, object]) -> str:
    serialized = json.dumps(payload, indent=2, sort_keys=True, default=str)
    return f"<h2>{html.escape(title)}</h2><pre>{html.escape(serialized)}</pre>"


def _table_section(title: str, frame: pd.DataFrame) -> str:
    return f"<h2>{html.escape(title)}</h2>{frame.to_html(index=False, escape=True)}"
```

- [ ] **Step 4: Run test**

Run:

```bash
uv run --frozen pytest src/tests/backtesting/test_h004_residual_diagnostic.py::test_build_h004_selected_residual_profile_separates_all_candidates_and_selected src/tests/backtesting/test_h004_residual_diagnostic.py::test_write_h004_diagnostic_artifacts_creates_required_files -q
```

Expected: `2 passed`.

- [ ] **Step 5: Commit**

Run:

```bash
git add src/cartola/backtesting/h004_residual_diagnostic.py src/tests/backtesting/test_h004_residual_diagnostic.py
git commit -m "feat: write h004 diagnostic artifacts"
```

Expected: commit succeeds.

## Task 6: Add End-to-End Diagnostic Builder

**Files:**
- Modify: `src/cartola/backtesting/h004_residual_diagnostic.py`
- Test: `src/tests/backtesting/test_h004_residual_diagnostic.py`

- [ ] **Step 1: Add end-to-end test**

Append:

```python
from cartola.backtesting.h004_residual_diagnostic import build_h004_residual_diagnostic


def test_build_h004_residual_diagnostic_writes_decision_artifacts(tmp_path: Path) -> None:
    child_path = _write_child(tmp_path, season=2025)
    _prediction_rows().to_csv(child_path / "player_predictions.csv", index=False)
    _selected_rows().to_csv(child_path / "selected_players.csv", index=False)

    result = build_h004_residual_diagnostic(
        experiment_path=tmp_path / "experiment",
        output_path=tmp_path / "out",
        seasons=(2025,),
        model_id=H004_CONTROL_MODEL_ID,
        feature_pack=H004_CONTROL_FEATURE_PACK,
    )

    assert result.output_path == tmp_path / "out"
    assert (tmp_path / "out" / "h004_diagnostic_decision.json").is_file()
    assert result.decision["diagnostic_status"] in {"passes", "rejected", "invalid"}
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
uv run --frozen pytest src/tests/backtesting/test_h004_residual_diagnostic.py::test_build_h004_residual_diagnostic_writes_decision_artifacts -q
```

Expected: FAIL with missing `build_h004_residual_diagnostic`.

- [ ] **Step 3: Implement end-to-end builder**

Add:

```python
def build_h004_residual_diagnostic(
    *,
    experiment_path: Path,
    output_path: Path,
    seasons: tuple[int, ...],
    model_id: str,
    feature_pack: str,
) -> H004DiagnosticResult:
    children = discover_h004_source_children(
        experiment_path=experiment_path,
        seasons=seasons,
        model_id=model_id,
        feature_pack=feature_pack,
    )
    bundles = tuple(load_h004_prediction_bundle(child) for child in children)
    played = pd.concat([bundle.played for bundle in bundles], ignore_index=True)
    all_candidates = pd.concat([bundle.all_candidates for bundle in bundles], ignore_index=True)
    selected_players = pd.concat([bundle.selected_players for bundle in bundles], ignore_index=True)
    correlations = build_h004_residual_correlations(played)
    quintiles = build_h004_residual_quintiles(played)
    recall = build_h004_top_actual_recall(played)
    selected_profile = build_h004_selected_residual_profile(played, selected_players)
    dnp_profile = build_h004_dnp_context_profile(all_candidates)
    decision = build_h004_diagnostic_decision(
        correlations=correlations,
        top_actual_recall=recall,
        source_experiment_path=experiment_path,
        children=children,
        missing_or_invalid_columns=(),
    )
    result = H004DiagnosticResult(
        output_path=output_path,
        residual_correlations=correlations,
        residual_quintiles=quintiles,
        top_actual_recall=recall,
        selected_residual_profile=selected_profile,
        dnp_context_profile=dnp_profile,
        decision=decision,
    )
    write_h004_diagnostic_artifacts(result)
    return result
```

- [ ] **Step 4: Run test**

Run:

```bash
uv run --frozen pytest src/tests/backtesting/test_h004_residual_diagnostic.py::test_build_h004_residual_diagnostic_writes_decision_artifacts -q
```

Expected: `1 passed`.

- [ ] **Step 5: Commit**

Run:

```bash
git add src/cartola/backtesting/h004_residual_diagnostic.py src/tests/backtesting/test_h004_residual_diagnostic.py
git commit -m "feat: build h004 residual diagnostic"
```

Expected: commit succeeds.

## Task 7: Add CLI With Progress And Dotenv Bootstrap

**Files:**
- Create: `scripts/run_h004_residual_diagnostic.py`
- Test: `src/tests/backtesting/test_run_h004_residual_diagnostic_cli.py`

- [ ] **Step 1: Add CLI tests**

Create `src/tests/backtesting/test_run_h004_residual_diagnostic_cli.py`:

```python
from __future__ import annotations

from pathlib import Path

import scripts.run_h004_residual_diagnostic as cli


def test_parse_args_accepts_h004_source_options() -> None:
    args = cli.parse_args(
        [
            "--experiment-path",
            "data/experiment",
            "--seasons",
            "2021,2022",
            "--output-root",
            "data/out",
        ]
    )

    assert args.experiment_path == Path("data/experiment")
    assert args.seasons == "2021,2022"
    assert args.output_root == Path("data/out")
    assert args.model_id == "xgboost_depth2_slow"
    assert args.feature_pack == "ppg_xg_matchup"


def test_parse_seasons_rejects_duplicates() -> None:
    try:
        cli._parse_seasons("2021,2021")
    except ValueError as exc:
        assert "Duplicate seasons" in str(exc)
    else:
        raise AssertionError("Expected duplicate season validation failure")
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
uv run --frozen pytest src/tests/backtesting/test_run_h004_residual_diagnostic_cli.py -q
```

Expected: FAIL with `ModuleNotFoundError` or missing script.

- [ ] **Step 3: Implement CLI**

Create `scripts/run_h004_residual_diagnostic.py`:

```python
#!/usr/bin/env python
from __future__ import annotations

import argparse
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel

build_h004_residual_diagnostic: Callable[..., Any] | None = None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run H004 residual diagnostic from model experiment artifacts.")
    parser.add_argument("--experiment-path", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("data/08_reporting/hypotheses"))
    parser.add_argument("--seasons", default="2021,2022,2023,2024,2025")
    parser.add_argument("--model-id", default="xgboost_depth2_slow")
    parser.add_argument("--feature-pack", default="ppg_xg_matchup")
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
    global build_h004_residual_diagnostic
    if build_h004_residual_diagnostic is None:
        from cartola.backtesting.h004_residual_diagnostic import (
            build_h004_residual_diagnostic as imported_build_h004_residual_diagnostic,
        )

        build_h004_residual_diagnostic = imported_build_h004_residual_diagnostic


def _parse_seasons(value: str) -> tuple[int, ...]:
    seasons = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    duplicates = sorted({season for season in seasons if seasons.count(season) > 1})
    if duplicates:
        raise ValueError(f"Duplicate seasons are not allowed: {duplicates}")
    return seasons


def _timestamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    _bootstrap_dotenv()
    _load_runtime_dependencies()
    if build_h004_residual_diagnostic is None:
        raise RuntimeError("H004 diagnostic runtime dependencies were not loaded.")

    console = Console()
    seasons = _parse_seasons(str(args.seasons))
    output_path = args.output_root / f"h004_residual_diagnostic_started_at={_timestamp()}"
    console.print(
        f"H004 residual diagnostic started: seasons={','.join(str(season) for season in seasons)} "
        f"output={output_path}"
    )
    with console.status("Loading artifacts and computing H004 residual diagnostics..."):
        result = build_h004_residual_diagnostic(
            experiment_path=args.experiment_path,
            output_path=output_path,
            seasons=seasons,
            model_id=str(args.model_id),
            feature_pack=str(args.feature_pack),
        )
    console.print(
        Panel(
            f"diagnostic_status={result.decision.get('diagnostic_status')}\\noutput_path={result.output_path}",
            title="H004 residual diagnostic complete",
            border_style="green",
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run CLI tests**

Run:

```bash
uv run --frozen pytest src/tests/backtesting/test_run_h004_residual_diagnostic_cli.py -q
```

Expected: `2 passed`.

- [ ] **Step 5: Commit**

Run:

```bash
git add scripts/run_h004_residual_diagnostic.py src/tests/backtesting/test_run_h004_residual_diagnostic_cli.py
git commit -m "feat: add h004 residual diagnostic cli"
```

Expected: commit succeeds.

## Task 8: Run Focused Verification

**Files:**
- Verify only.

- [ ] **Step 1: Run diagnostic test suite**

Run:

```bash
uv run --frozen pytest src/tests/backtesting/test_h004_residual_diagnostic.py src/tests/backtesting/test_run_h004_residual_diagnostic_cli.py -q
```

Expected: all tests pass.

- [ ] **Step 2: Run lint/type checks for touched files**

Run:

```bash
uv run --frozen ruff check src/cartola/backtesting/h004_residual_diagnostic.py scripts/run_h004_residual_diagnostic.py src/tests/backtesting/test_h004_residual_diagnostic.py src/tests/backtesting/test_run_h004_residual_diagnostic_cli.py
uv run --frozen ty check src/cartola/backtesting/h004_residual_diagnostic.py scripts/run_h004_residual_diagnostic.py src/tests/backtesting/test_h004_residual_diagnostic.py src/tests/backtesting/test_run_h004_residual_diagnostic_cli.py
```

Expected: both commands pass.

- [ ] **Step 3: Commit fixes if needed**

If Step 1 or Step 2 required fixes, run:

```bash
git add src/cartola/backtesting/h004_residual_diagnostic.py scripts/run_h004_residual_diagnostic.py src/tests/backtesting/test_h004_residual_diagnostic.py src/tests/backtesting/test_run_h004_residual_diagnostic_cli.py
git commit -m "fix: stabilize h004 residual diagnostic"
```

Expected: commit succeeds only if fixes were made. If no fixes were made, do not create an empty commit.

## Task 9: Run Real H004 Residual Diagnostic

**Files:**
- Generated output under `data/08_reporting/hypotheses/`.

- [ ] **Step 1: Run real diagnostic**

Run:

```bash
uv run --frozen python scripts/run_h004_residual_diagnostic.py \
  --experiment-path data/08_reporting/experiments/model_feature/group=xgboost-sensitivity-v2__started_at=20260507T231127138806Z__matrix=f019652c883d \
  --seasons 2021,2022,2023,2024,2025
```

Expected output includes:

```text
H004 residual diagnostic started
H004 residual diagnostic complete
diagnostic_status=<passes|rejected|invalid>
output_path=data/08_reporting/hypotheses/h004_residual_diagnostic_started_at=...
```

- [ ] **Step 2: Inspect decision**

Run:

```bash
uv run --frozen python - <<'PY'
import json
from pathlib import Path

outputs = sorted(Path("data/08_reporting/hypotheses").glob("h004_residual_diagnostic_started_at=*"))
path = outputs[-1]
decision = json.loads((path / "h004_diagnostic_decision.json").read_text())
print("output_path=", path)
print("diagnostic_status=", decision["diagnostic_status"])
print("passed_families=", decision["passed_families"])
print("fixture_identity_status=", decision["fixture_identity_status"])
print(json.dumps(decision["family_results"], indent=2, sort_keys=True))
PY
```

Expected: prints the latest diagnostic status and family results.

- [ ] **Step 3: Interpret result**

Use this rule:

```text
diagnostic_status=passes -> proceed to a separate Phase 2 feature-pack implementation plan.
diagnostic_status=rejected -> stop H004 and record a null discovery result.
diagnostic_status=invalid -> fix artifact/context issues before interpreting the diagnostic.
```

- [ ] **Step 4: Commit code if real run exposed code/report fixes**

If Task 9 required code or report fixes, run focused tests again, then:

```bash
git add src/cartola/backtesting/h004_residual_diagnostic.py scripts/run_h004_residual_diagnostic.py src/tests/backtesting/test_h004_residual_diagnostic.py src/tests/backtesting/test_run_h004_residual_diagnostic_cli.py
git commit -m "fix: handle real h004 diagnostic artifacts"
```

Expected: commit succeeds only if fixes were made.

## Task 10: Update Roadmap With Phase 1 Result

**Files:**
- Modify: `roadmap.md`

- [ ] **Step 1: Add H004 Phase 1 status**

Find the policy/research interpretation section:

```bash
rg -n "H004|Policy-simulation|oracle|hypothesis|Current Interpretation" roadmap.md
```

Add one short paragraph after the policy-simulation interpretation:

```markdown
H004 attack-vs-defense mismatch is being tested as model-signal research, not
an optimizer policy. Phase 1 residual diagnostics read persisted
`xgboost_depth2_slow + ppg_xg_matchup` artifacts and decide whether residuals
correlate with pre-match xG/home/position matchup context strongly enough to
justify a frozen feature-pack experiment.
```

If the real diagnostic has completed, add the actual status:

```markdown
Latest H004 residual diagnostic: `<diagnostic_status>`, passed families:
`<families>`, output `<path>`. If rejected, no H004 feature pack should be
implemented for this generation.
```

- [ ] **Step 2: Run docs diff check**

Run:

```bash
git diff --check roadmap.md
```

Expected: no output.

- [ ] **Step 3: Commit roadmap**

Run:

```bash
git add roadmap.md
git commit -m "docs: update roadmap for h004 diagnostic"
```

Expected: commit succeeds.

## Task 11: Final Verification

**Files:**
- Verify only.

- [ ] **Step 1: Run full project gate**

Run:

```bash
uv run --frozen scripts/pyrepo-check --all
```

Expected:

```text
All checks passed!
...
pytest ... passed
```

- [ ] **Step 2: Check git status**

Run:

```bash
git status --short --branch
```

Expected: clean branch, or only intentionally untracked generated reports under ignored paths.

- [ ] **Step 3: Final response**

Report:

```text
Implemented H004 Phase 1 residual diagnostic.
Output path: <latest diagnostic output>
Decision: <passes|rejected|invalid>
Verification: <commands and pass counts>
Next step: <Phase 2 feature pack plan if passes, stop H004 if rejected, fix artifacts if invalid>
```
