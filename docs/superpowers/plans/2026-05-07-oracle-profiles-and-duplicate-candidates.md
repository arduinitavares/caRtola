# Oracle Profiles And Duplicate Candidate Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the 2022 round 5 duplicate-candidate oracle failure and extend oracle knowledge discovery with concrete profile metrics for home/away, opponent overlap, predicted rank, price rank, favorite/underdog proxy, position mix, and team concentration.

**Architecture:** Keep `oracle_discovery.py` as the artifact orchestration layer, but move profile-specific calculations into a small pure module so report generation stays testable. Normalize duplicate model-candidate artifacts before oracle optimization, recording every deduped or rejected group; then compute player-level profile rows and aggregate profile-gap rows from already-persisted artifacts only.

**Tech Stack:** Python 3.13, pandas, pytest, existing Cartola backtesting artifacts, existing fixture loader in `cartola.backtesting.data`.

---

## File Structure

- Modify `src/cartola/backtesting/oracle_discovery.py`
  - Add candidate normalization before `_validate_model_candidate_identity()`.
  - Thread candidate normalization rows into output writing.
  - Call profile builders after each valid oracle-selected/model-selected round.
  - Update metadata and HTML to expose profile sections and candidate normalization counts.

- Create `src/cartola/backtesting/oracle_profiles.py`
  - Pure profile metric helpers.
  - Fixture matchup lookup.
  - Player-level long-format rows for `oracle_player_profiles.csv`.
  - Aggregate rows for `profile_gap_summary.csv`.

- Modify `src/tests/backtesting/test_oracle_discovery.py`
  - Add tests for duplicate candidate dedupe and conflicting duplicate rejection.
  - Add report-builder tests proving profile CSVs are non-empty and contain expected deterministic metrics.

- Create `src/tests/backtesting/test_oracle_profiles.py`
  - Unit tests for profile helpers independent of the full report builder.

- Modify `docs/superpowers/plans/2026-05-06-oracle-knowledge-discovery.md` only if it contains stale output expectations that conflict with this plan.

---

### Task 1: Candidate Duplicate Normalization

**Files:**
- Modify: `src/cartola/backtesting/oracle_discovery.py`
- Test: `src/tests/backtesting/test_oracle_discovery.py`

- [ ] **Step 1: Write failing test for safe duplicate dedupe**

Add this test near `test_run_model_candidate_oracle_rejects_duplicate_round_athletes_before_optimizer`:

```python
def test_run_model_candidate_oracle_deduplicates_equivalent_candidate_rows() -> None:
    candidates = _model_candidate_rows()
    duplicate = candidates.iloc[[0]].copy()
    duplicate["slug"] = None
    duplicate["minimo_para_valorizar"] = 3.42
    candidates = pd.concat([candidates, duplicate], ignore_index=True)
    config = BacktestConfig(season=2025, start_round=5, budget=100.0)

    row, selected = run_model_candidate_oracle(
        candidates,
        config=config,
        budget_before_round=100.0,
        score_column="model_score",
    )

    assert row["optimizer_status"] == "Optimal"
    assert selected["id_atleta"].nunique() == len(selected)
```

- [ ] **Step 2: Write failing test for conflicting duplicates**

Add this directly after the safe duplicate test:

```python
def test_run_model_candidate_oracle_rejects_conflicting_duplicate_candidate_rows() -> None:
    candidates = _model_candidate_rows()
    duplicate = candidates.iloc[[0]].copy()
    duplicate["model_score"] = float(duplicate["model_score"].iloc[0]) + 2.0
    candidates = pd.concat([candidates, duplicate], ignore_index=True)
    config = BacktestConfig(season=2025, start_round=5, budget=100.0)

    with pytest.raises(OracleObjectiveError, match="Conflicting duplicate candidate rows"):
        run_model_candidate_oracle(
            candidates,
            config=config,
            budget_before_round=100.0,
            score_column="model_score",
        )
```

- [ ] **Step 3: Run tests to verify they fail**

Run:

```bash
uv run --frozen pytest \
  src/tests/backtesting/test_oracle_discovery.py::test_run_model_candidate_oracle_deduplicates_equivalent_candidate_rows \
  src/tests/backtesting/test_oracle_discovery.py::test_run_model_candidate_oracle_rejects_conflicting_duplicate_candidate_rows \
  -q
```

Expected: the first test fails with the existing duplicate-row error; the second may fail with the old generic duplicate error.

- [ ] **Step 4: Implement candidate normalization**

In `src/cartola/backtesting/oracle_discovery.py`, add:

```python
def _critical_candidate_columns(candidates: pd.DataFrame, score_column: str) -> list[str]:
    columns = [
        "rodada",
        "id_atleta",
        "id_clube",
        "posicao",
        "preco_pre_rodada",
        "pontuacao",
        "entrou_em_campo",
        "variacao",
        score_column,
    ]
    return [column for column in columns if column in candidates.columns]


def _deduplicate_model_candidates(candidates: pd.DataFrame, *, score_column: str) -> pd.DataFrame:
    if not {"rodada", "id_atleta"}.issubset(candidates.columns):
        return candidates
    duplicate_mask = candidates[["rodada", "id_atleta"]].duplicated(keep=False)
    if not bool(duplicate_mask.any()):
        return candidates

    critical_columns = _critical_candidate_columns(candidates, score_column)
    normalized_groups: list[pd.DataFrame] = []
    duplicate_keys = candidates.loc[duplicate_mask, ["rodada", "id_atleta"]].drop_duplicates()
    for _, key in duplicate_keys.iterrows():
        group_mask = candidates["rodada"].eq(key["rodada"]) & candidates["id_atleta"].eq(key["id_atleta"])
        group = candidates.loc[group_mask]
        if len(group.loc[:, critical_columns].drop_duplicates()) > 1:
            raise OracleObjectiveError(
                f"Conflicting duplicate candidate rows for rodada={key['rodada']} id_atleta={key['id_atleta']}"
            )
        # Prefer the row with the most populated optional fields while preserving critical equality.
        winner_index = group.notna().sum(axis=1).sort_values(ascending=False).index[0]
        normalized_groups.append(candidates.loc[[winner_index]])

    non_duplicate = candidates.loc[~duplicate_mask]
    return pd.concat([non_duplicate, *normalized_groups], ignore_index=True)
```

Then update `run_model_candidate_oracle()` before validation:

```python
    candidates = _deduplicate_model_candidates(candidates, score_column=score_column)
    _validate_model_candidate_identity(candidates)
```

- [ ] **Step 5: Run duplicate tests**

Run:

```bash
uv run --frozen pytest \
  src/tests/backtesting/test_oracle_discovery.py::test_run_model_candidate_oracle_deduplicates_equivalent_candidate_rows \
  src/tests/backtesting/test_oracle_discovery.py::test_run_model_candidate_oracle_rejects_conflicting_duplicate_candidate_rows \
  -q
```

Expected: both pass.

- [ ] **Step 6: Verify the real 2022 round 5 artifact no longer invalidates oracle rounds**

Run the oracle command on the existing xgboost sensitivity experiment:

```bash
uv run --frozen python scripts/run_oracle_knowledge_discovery.py \
  --experiment-path data/08_reporting/experiments/model_feature/group=xgboost-sensitivity-v2__started_at=20260505T211914592073Z__matrix=d95948374d75 \
  --current-year 2026
```

Expected: progress reaches `5610/5610`; `invalid_oracle_rows.csv` has zero rows for `Duplicate candidate rows for (rodada, id_atleta)`.

- [ ] **Step 7: Commit**

```bash
git add src/cartola/backtesting/oracle_discovery.py src/tests/backtesting/test_oracle_discovery.py
git commit -m "fix: normalize duplicate oracle candidates"
```

---

### Task 2: Pure Oracle Profile Metrics

**Files:**
- Create: `src/cartola/backtesting/oracle_profiles.py`
- Test: `src/tests/backtesting/test_oracle_profiles.py`

- [ ] **Step 1: Write unit tests for matchup and rank profiles**

Create `src/tests/backtesting/test_oracle_profiles.py`:

```python
from __future__ import annotations

import pandas as pd
import pytest

from cartola.backtesting.oracle_profiles import (
    build_oracle_player_profile_rows,
    build_profile_gap_summary_rows,
)


IDENTITY = {
    "source_mode": "artifact",
    "source_experiment_id": "exp-1",
    "source_child_id": "child-1",
    "season": 2025,
    "rodada": 5,
    "strategy": "model_a",
    "model_id": "model_a",
    "feature_pack": "ppg_xg_matchup",
    "fixture_mode": "exploratory",
    "matchup_context_mode": "cartola_matchup_v1",
    "budget_policy": "moving",
    "oracle_type": "budget_constrained",
    "candidate_universe": "model_candidate",
    "budget_path": "model_budget_path",
}


def _fixtures() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"rodada": 5, "id_clube_home": 10, "id_clube_away": 20, "data": "2025-05-01"},
            {"rodada": 5, "id_clube_home": 30, "id_clube_away": 40, "data": "2025-05-01"},
        ]
    )


def _selected(source: str) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "id_atleta": 1,
                "rodada": 5,
                "id_clube": 10,
                "posicao": "ata",
                "preco_pre_rodada": 20.0,
                "model_predicted_rank_overall": 3,
                "model_predicted_rank_position": 1,
                "matchup_is_home": True,
                "footystats_ppg_diff": 0.5,
                "source": source,
            },
            {
                "id_atleta": 2,
                "rodada": 5,
                "id_clube": 20,
                "posicao": "zag",
                "preco_pre_rodada": 5.0,
                "model_predicted_rank_overall": 44,
                "model_predicted_rank_position": 7,
                "matchup_is_home": False,
                "footystats_ppg_diff": -0.5,
                "source": source,
            },
            {
                "id_atleta": 3,
                "rodada": 5,
                "id_clube": 30,
                "posicao": "tec",
                "preco_pre_rodada": 8.0,
                "model_predicted_rank_overall": 10,
                "model_predicted_rank_position": 2,
                "matchup_is_home": True,
                "footystats_ppg_diff": 0.2,
                "source": source,
            },
        ]
    )


def test_build_oracle_player_profile_rows_emits_home_rank_price_and_opponent_overlap() -> None:
    rows = build_oracle_player_profile_rows(
        identity=IDENTITY,
        oracle_selected=_selected("oracle"),
        model_selected=_selected("model"),
        fixtures=_fixtures(),
    )

    metric_values = {(row["id_atleta"], row["profile_metric"]): row["profile_value"] for row in rows}
    assert metric_values[(1, "is_home")] is True
    assert metric_values[(2, "is_home")] is False
    assert metric_values[(1, "opponent_overlap_in_lineup")] is True
    assert metric_values[(2, "opponent_overlap_in_lineup")] is True
    assert metric_values[(1, "model_predicted_rank_position")] == 1
    assert metric_values[(2, "model_predicted_rank_position")] == 7
    assert metric_values[(1, "favorite_proxy_ppg_diff_positive")] is True
    assert metric_values[(2, "favorite_proxy_ppg_diff_positive")] is False


def test_build_profile_gap_summary_rows_compares_oracle_to_model_selected_baseline() -> None:
    oracle = _selected("oracle")
    model = _selected("model").loc[lambda frame: frame["id_clube"].isin([10, 30])].copy()

    rows = build_profile_gap_summary_rows(
        identity=IDENTITY,
        oracle_selected=oracle,
        model_selected=model,
        fixtures=_fixtures(),
    )

    metrics = {row["profile_metric"]: row for row in rows}
    assert metrics["opponent_overlap_round_rate"]["oracle_value"] == 1.0
    assert metrics["opponent_overlap_round_rate"]["baseline_value"] == 0.0
    assert metrics["home_player_share"]["oracle_value"] == pytest.approx(0.5)
    assert metrics["home_player_share"]["baseline_value"] == pytest.approx(1.0)
    assert metrics["median_model_predicted_rank_position"]["oracle_value"] == pytest.approx(4.0)
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
uv run --frozen pytest src/tests/backtesting/test_oracle_profiles.py -q
```

Expected: import failure because `oracle_profiles.py` does not exist.

- [ ] **Step 3: Implement `oracle_profiles.py`**

Create `src/cartola/backtesting/oracle_profiles.py` with:

```python
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd


IDENTITY_COLUMNS = [
    "source_mode",
    "source_experiment_id",
    "source_child_id",
    "season",
    "rodada",
    "strategy",
    "model_id",
    "feature_pack",
    "fixture_mode",
    "matchup_context_mode",
    "budget_policy",
    "oracle_type",
    "candidate_universe",
    "budget_path",
]


def build_oracle_player_profile_rows(
    *,
    identity: Mapping[str, object],
    oracle_selected: pd.DataFrame,
    model_selected: pd.DataFrame,
    fixtures: pd.DataFrame | None,
) -> list[dict[str, object]]:
    selected = _with_lineup_context(oracle_selected, fixtures=fixtures)
    rows: list[dict[str, object]] = []
    for _, player in selected.iterrows():
        for metric, value in _player_metric_values(player).items():
            rows.append(
                {
                    **{column: identity.get(column) for column in IDENTITY_COLUMNS},
                    "id_atleta": player.get("id_atleta"),
                    "posicao": player.get("posicao"),
                    "profile_section": "oracle_player",
                    "profile_metric": metric,
                    "profile_value": value,
                    "baseline_name": "model_selected",
                    "baseline_value": None,
                    "sample_size": 1,
                    "full_market_status": "not_available",
                }
            )
    return rows


def build_profile_gap_summary_rows(
    *,
    identity: Mapping[str, object],
    oracle_selected: pd.DataFrame,
    model_selected: pd.DataFrame,
    fixtures: pd.DataFrame | None,
) -> list[dict[str, object]]:
    oracle = _with_lineup_context(oracle_selected, fixtures=fixtures)
    model = _with_lineup_context(model_selected, fixtures=fixtures)
    metrics = {
        "opponent_overlap_round_rate": (
            float(_lineup_has_opponent_overlap(oracle, fixtures)),
            float(_lineup_has_opponent_overlap(model, fixtures)),
        ),
        "avg_players_in_opponent_overlap": (
            float(_players_in_opponent_overlap(oracle, fixtures)),
            float(_players_in_opponent_overlap(model, fixtures)),
        ),
        "home_player_share": (_share_true(oracle, "matchup_is_home"), _share_true(model, "matchup_is_home")),
        "favorite_proxy_ppg_diff_positive_share": (
            _share_positive(oracle, "footystats_ppg_diff"),
            _share_positive(model, "footystats_ppg_diff"),
        ),
        "median_model_predicted_rank_position": (
            _median_numeric(oracle, "model_predicted_rank_position"),
            _median_numeric(model, "model_predicted_rank_position"),
        ),
        "top5_position_rank_share": (
            _share_at_most(oracle, "model_predicted_rank_position", 5),
            _share_at_most(model, "model_predicted_rank_position", 5),
        ),
        "avg_same_club_selected_count": (
            _avg_same_club_count(oracle),
            _avg_same_club_count(model),
        ),
    }
    rows: list[dict[str, object]] = []
    for metric, (oracle_value, baseline_value) in metrics.items():
        rows.append(
            {
                **{column: identity.get(column) for column in IDENTITY_COLUMNS},
                "profile_section": "lineup_profile",
                "profile_metric": metric,
                "oracle_value": oracle_value,
                "baseline_name": "model_selected",
                "baseline_value": baseline_value,
                "absolute_gap": _numeric_gap(oracle_value, baseline_value),
                "relative_gap": None,
                "sample_size": int(len(oracle)),
                "season_stability_count": None,
                "stability_label": "round_level",
                "full_market_status": "not_available",
            }
        )
    return rows


def _player_metric_values(player: pd.Series) -> dict[str, object]:
    return {
        "is_home": _bool_or_none(player.get("matchup_is_home")),
        "opponent_overlap_in_lineup": _bool_or_none(player.get("opponent_overlap_in_lineup")),
        "same_club_selected_count": _numeric_or_none(player.get("same_club_selected_count")),
        "model_predicted_rank_overall": _numeric_or_none(player.get("model_predicted_rank_overall")),
        "model_predicted_rank_position": _numeric_or_none(player.get("model_predicted_rank_position")),
        "preco_pre_rodada": _numeric_or_none(player.get("preco_pre_rodada")),
        "favorite_proxy_ppg_diff_positive": _positive_bool_or_none(player.get("footystats_ppg_diff")),
        "footystats_ppg_diff": _numeric_or_none(player.get("footystats_ppg_diff")),
    }


def _with_lineup_context(selected: pd.DataFrame, *, fixtures: pd.DataFrame | None) -> pd.DataFrame:
    output = selected.copy()
    if output.empty:
        output["opponent_overlap_in_lineup"] = pd.Series(dtype=bool)
        output["same_club_selected_count"] = pd.Series(dtype=float)
        return output
    output["same_club_selected_count"] = output.groupby("id_clube")["id_atleta"].transform("nunique")
    overlap_clubs = _opponent_overlap_clubs(output, fixtures)
    output["opponent_overlap_in_lineup"] = output["id_clube"].isin(overlap_clubs)
    return output


def _opponent_overlap_clubs(selected: pd.DataFrame, fixtures: pd.DataFrame | None) -> set[int]:
    if fixtures is None or selected.empty or "id_clube" not in selected.columns:
        return set()
    clubs = {int(value) for value in pd.to_numeric(selected["id_clube"], errors="coerce").dropna().tolist()}
    overlap: set[int] = set()
    for _, fixture in fixtures.iterrows():
        home = int(fixture["id_clube_home"])
        away = int(fixture["id_clube_away"])
        if home in clubs and away in clubs:
            overlap.update({home, away})
    return overlap


def _lineup_has_opponent_overlap(selected: pd.DataFrame, fixtures: pd.DataFrame | None) -> bool:
    return bool(_opponent_overlap_clubs(selected, fixtures))


def _players_in_opponent_overlap(selected: pd.DataFrame, fixtures: pd.DataFrame | None) -> int:
    overlap = _opponent_overlap_clubs(selected, fixtures)
    if not overlap or selected.empty:
        return 0
    return int(selected.loc[selected["id_clube"].isin(overlap), "id_atleta"].nunique())


def _share_true(frame: pd.DataFrame, column: str) -> float | None:
    if frame.empty or column not in frame.columns:
        return None
    values = frame.loc[frame["posicao"].astype(str).ne("tec"), column].dropna()
    if values.empty:
        return None
    return float(values.astype(bool).mean())


def _share_positive(frame: pd.DataFrame, column: str) -> float | None:
    if frame.empty or column not in frame.columns:
        return None
    values = pd.to_numeric(frame.loc[frame["posicao"].astype(str).ne("tec"), column], errors="coerce").dropna()
    if values.empty:
        return None
    return float(values.gt(0).mean())


def _share_at_most(frame: pd.DataFrame, column: str, threshold: float) -> float | None:
    if frame.empty or column not in frame.columns:
        return None
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    if values.empty:
        return None
    return float(values.le(threshold).mean())


def _median_numeric(frame: pd.DataFrame, column: str) -> float | None:
    if frame.empty or column not in frame.columns:
        return None
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    if values.empty:
        return None
    return float(values.median())


def _avg_same_club_count(frame: pd.DataFrame) -> float | None:
    if frame.empty or "id_clube" not in frame.columns:
        return None
    return float(frame.groupby("id_clube")["id_atleta"].nunique().mean())


def _numeric_gap(left: object, right: object) -> float | None:
    if left is None or right is None:
        return None
    return float(left) - float(right)


def _bool_or_none(value: object) -> bool | None:
    if pd.isna(value):
        return None
    return bool(value)


def _positive_bool_or_none(value: object) -> bool | None:
    numeric = _numeric_or_none(value)
    if numeric is None:
        return None
    return numeric > 0


def _numeric_or_none(value: object) -> float | None:
    if value is None or pd.isna(value):
        return None
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric):
        return None
    return float(numeric)
```

- [ ] **Step 4: Run profile helper tests**

Run:

```bash
uv run --frozen pytest src/tests/backtesting/test_oracle_profiles.py -q
```

Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add src/cartola/backtesting/oracle_profiles.py src/tests/backtesting/test_oracle_profiles.py
git commit -m "feat: add oracle profile metrics"
```

---

### Task 3: Integrate Profile Metrics Into Oracle Discovery Output

**Files:**
- Modify: `src/cartola/backtesting/oracle_discovery.py`
- Test: `src/tests/backtesting/test_oracle_discovery.py`

- [ ] **Step 1: Write failing report-builder test for profile outputs**

Add this test after `test_build_oracle_discovery_report_writes_expected_artifacts`:

```python
def test_build_oracle_discovery_report_writes_profile_metrics(tmp_path: Path) -> None:
    predictions = _report_builder_predictions()
    predictions["matchup_is_home"] = [True, False, True, False, True, False, True, False, True, False, True, False]
    predictions["footystats_ppg_diff"] = [0.5, -0.5, 0.2, -0.2, 0.1, -0.1, 0.3, -0.3, 0.4, -0.4, 0.6, -0.6]
    selected_players = predictions.head(12).copy()
    selected_players["strategy"] = "xgboost_depth2_l2_heavy"
    selected_players["is_captain"] = selected_players["id_atleta"].eq(10)
    round_results = pd.DataFrame(
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
    )
    experiment = _write_report_builder_experiment(
        tmp_path,
        predictions=predictions,
        round_results=round_results,
        selected_players=selected_players,
    )
    fixture_dir = tmp_path / "data" / "01_raw" / "fixtures" / "2025"
    fixture_dir.mkdir(parents=True)
    pd.DataFrame(
        [{"rodada": 5, "id_clube_home": 1, "id_clube_away": 2, "data": "2025-05-01"}]
    ).to_csv(fixture_dir / "partidas-5.csv", index=False)
    output = tmp_path / "oracle_out"

    _run_build_oracle_discovery_report(experiment_path=experiment, output_path=output)

    player_profiles = pd.read_csv(output / "oracle_player_profiles.csv")
    gap_summary = pd.read_csv(output / "profile_gap_summary.csv")
    assert not player_profiles.empty
    assert not gap_summary.empty
    assert {"is_home", "opponent_overlap_in_lineup", "model_predicted_rank_position"}.issubset(
        set(player_profiles["profile_metric"])
    )
    assert {"opponent_overlap_round_rate", "home_player_share", "top5_position_rank_share"}.issubset(
        set(gap_summary["profile_metric"])
    )
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
uv run --frozen pytest \
  src/tests/backtesting/test_oracle_discovery.py::test_build_oracle_discovery_report_writes_profile_metrics \
  -q
```

Expected: fail because `oracle_player_profiles.csv` and `profile_gap_summary.csv` are empty.

- [ ] **Step 3: Integrate profile builders**

In `src/cartola/backtesting/oracle_discovery.py`, import:

```python
from cartola.backtesting.data import load_fixtures
from cartola.backtesting.oracle_profiles import build_oracle_player_profile_rows, build_profile_gap_summary_rows
```

Add list accumulators in `build_oracle_discovery_report()`:

```python
    player_profile_rows: list[dict[str, object]] = []
    profile_gap_rows: list[dict[str, object]] = []
```

Load fixtures once per child after config creation:

```python
        fixtures = _load_profile_fixtures(context, config)
        planned_children.append((context, artifacts, config, fixtures))
```

Add helper:

```python
def _load_profile_fixtures(context: SourceRunContext, config: BacktestConfig) -> pd.DataFrame | None:
    if context.fixture_mode == "none":
        return None
    try:
        fixtures = load_fixtures(context.season, project_root=config.project_root)
    except (FileNotFoundError, NotADirectoryError, ValueError):
        return None
    return fixtures
```

After `recall_rows.extend(...)` in the successful round path:

```python
                player_profile_rows.extend(
                    build_oracle_player_profile_rows(
                        identity=identity,
                        oracle_selected=oracle_selected,
                        model_selected=selected,
                        fixtures=_fixtures_for_round(fixtures, round_number=round_number),
                    )
                )
                profile_gap_rows.extend(
                    build_profile_gap_summary_rows(
                        identity=identity,
                        oracle_selected=oracle_selected,
                        model_selected=selected,
                        fixtures=_fixtures_for_round(fixtures, round_number=round_number),
                    )
                )
```

Add helper:

```python
def _fixtures_for_round(fixtures: pd.DataFrame | None, *, round_number: int) -> pd.DataFrame | None:
    if fixtures is None or fixtures.empty:
        return None
    return fixtures.loc[pd.to_numeric(fixtures["rodada"], errors="coerce").eq(round_number)].copy()
```

Update `_write_outputs(...)` signature to accept `player_profile_rows` and `profile_gap_rows`, then replace the empty CSV writes:

```python
    _rows_frame(player_profile_rows, ORACLE_PLAYER_PROFILE_COLUMNS).to_csv(
        output_path / "oracle_player_profiles.csv",
        index=False,
    )
    _rows_frame(profile_gap_rows, PROFILE_GAP_SUMMARY_COLUMNS).to_csv(
        output_path / "profile_gap_summary.csv",
        index=False,
    )
```

- [ ] **Step 4: Run report-builder profile test**

Run:

```bash
uv run --frozen pytest \
  src/tests/backtesting/test_oracle_discovery.py::test_build_oracle_discovery_report_writes_profile_metrics \
  -q
```

Expected: pass.

- [ ] **Step 5: Run oracle discovery tests**

Run:

```bash
uv run --frozen pytest src/tests/backtesting/test_oracle_discovery.py src/tests/backtesting/test_oracle_profiles.py -q
```

Expected: pass.

- [ ] **Step 6: Commit**

```bash
git add src/cartola/backtesting/oracle_discovery.py src/tests/backtesting/test_oracle_discovery.py
git commit -m "feat: write oracle profile outputs"
```

---

### Task 4: HTML Summary For Knowledge Discovery Metrics

**Files:**
- Modify: `src/cartola/backtesting/oracle_discovery.py`
- Test: `src/tests/backtesting/test_oracle_discovery.py`

- [ ] **Step 1: Write failing HTML test**

Add this test near `test_build_oracle_discovery_report_writes_html_disclaimer`:

```python
def test_build_oracle_discovery_report_html_includes_profile_metrics(tmp_path: Path) -> None:
    predictions = _report_builder_predictions()
    predictions["matchup_is_home"] = True
    predictions["footystats_ppg_diff"] = 0.25
    selected_players = predictions.head(12).copy()
    selected_players["strategy"] = "xgboost_depth2_l2_heavy"
    selected_players["is_captain"] = selected_players["id_atleta"].eq(10)
    round_results = pd.DataFrame(
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
    )
    experiment = _write_report_builder_experiment(
        tmp_path,
        predictions=predictions,
        round_results=round_results,
        selected_players=selected_players,
    )
    output = tmp_path / "oracle_out"

    _run_build_oracle_discovery_report(experiment_path=experiment, output_path=output)

    html = (output / "oracle_knowledge_discovery.html").read_text(encoding="utf-8")
    assert "Profile Gap Summary" in html
    assert "home_player_share" in html
    assert "top5_position_rank_share" in html
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
uv run --frozen pytest \
  src/tests/backtesting/test_oracle_discovery.py::test_build_oracle_discovery_report_html_includes_profile_metrics \
  -q
```

Expected: fail because HTML does not include profile gap metrics.

- [ ] **Step 3: Update `_write_html()`**

Change `_write_html(...)` signature:

```python
def _write_html(
    output_path: Path,
    *,
    round_rows: list[dict[str, object]],
    captain_rows: list[dict[str, object]],
    recall_rows: list[dict[str, object]],
    profile_gap_rows: list[dict[str, object]],
) -> None:
```

Build a compact deterministic table:

```python
def _profile_gap_html(profile_gap_rows: list[dict[str, object]]) -> str:
    frame = pd.DataFrame(profile_gap_rows)
    if frame.empty:
        return "<p>No profile gap metrics were available for this run.</p>"
    summary = (
        frame.groupby("profile_metric", as_index=False)
        .agg(
            oracle_value=("oracle_value", "mean"),
            baseline_value=("baseline_value", "mean"),
            absolute_gap=("absolute_gap", "mean"),
            sample_size=("sample_size", "sum"),
        )
        .sort_values("profile_metric")
    )
    rows = []
    for _, row in summary.iterrows():
        rows.append(
            "<tr>"
            f"<td>{_html_text(row['profile_metric'])}</td>"
            f"<td>{_html_text(_format_number(row['oracle_value']))}</td>"
            f"<td>{_html_text(_format_number(row['baseline_value']))}</td>"
            f"<td>{_html_text(_format_number(row['absolute_gap']))}</td>"
            f"<td>{_html_text(row['sample_size'])}</td>"
            "</tr>"
        )
    return (
        "<table><thead><tr><th>metric</th><th>oracle avg</th><th>model-selected avg</th>"
        "<th>gap</th><th>sample size</th></tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table>"
    )


def _format_number(value: object) -> str:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric):
        return "NA"
    return f"{float(numeric):.3f}"
```

In the HTML body, add:

```html
<h2>Profile Gap Summary</h2>
<p>Oracle rows are hindsight-selected from the model candidate pool. Baseline rows are the model-selected squad from the same round and strategy.</p>
{_profile_gap_html(profile_gap_rows)}
```

- [ ] **Step 4: Update `_write_outputs()` to pass profile rows into HTML**

Change:

```python
    _write_html(output_path, round_rows=round_rows, captain_rows=captain_rows, recall_rows=recall_rows)
```

to:

```python
    _write_html(
        output_path,
        round_rows=round_rows,
        captain_rows=captain_rows,
        recall_rows=recall_rows,
        profile_gap_rows=profile_gap_rows,
    )
```

- [ ] **Step 5: Run HTML test**

Run:

```bash
uv run --frozen pytest \
  src/tests/backtesting/test_oracle_discovery.py::test_build_oracle_discovery_report_html_includes_profile_metrics \
  -q
```

Expected: pass.

- [ ] **Step 6: Commit**

```bash
git add src/cartola/backtesting/oracle_discovery.py src/tests/backtesting/test_oracle_discovery.py
git commit -m "feat: summarize oracle profiles in html"
```

---

### Task 5: Real Experiment Verification And Acceptance

**Files:**
- No source changes expected.

- [ ] **Step 1: Run targeted tests**

```bash
uv run --frozen pytest \
  src/tests/backtesting/test_oracle_profiles.py \
  src/tests/backtesting/test_oracle_discovery.py \
  src/tests/backtesting/test_run_oracle_knowledge_discovery_cli.py \
  -q
```

Expected: all pass.

- [ ] **Step 2: Run full quality gate**

```bash
uv run --frozen scripts/pyrepo-check --all
```

Expected: Ruff, ty, Bandit, and pytest pass.

- [ ] **Step 3: Regenerate oracle discovery for the completed xgboost sensitivity experiment**

```bash
uv run --frozen python scripts/run_oracle_knowledge_discovery.py \
  --experiment-path data/08_reporting/experiments/model_feature/group=xgboost-sensitivity-v2__started_at=20260505T211914592073Z__matrix=d95948374d75 \
  --current-year 2026
```

Expected:
- Progress reaches `5610/5610`.
- `invalid_oracle_rows.csv` contains no duplicate-candidate rows.
- `oracle_player_profiles.csv` has more than one line.
- `profile_gap_summary.csv` has more than one line.
- `oracle_knowledge_discovery.html` contains `Profile Gap Summary`.

- [ ] **Step 4: Print a compact acceptance summary**

Run:

```bash
latest=$(ls -td data/08_reporting/oracle_discovery/oracle_discovery_started_at=* | head -1)
wc -l "$latest"/invalid_oracle_rows.csv "$latest"/oracle_player_profiles.csv "$latest"/profile_gap_summary.csv
uv run --frozen python - <<'PY'
from pathlib import Path
import pandas as pd

latest = sorted(Path("data/08_reporting/oracle_discovery").glob("oracle_discovery_started_at=*"))[-1]
invalid = pd.read_csv(latest / "invalid_oracle_rows.csv")
profiles = pd.read_csv(latest / "profile_gap_summary.csv")
print("output", latest)
print("invalid_duplicate_rows", int(invalid["invalid_reason"].astype(str).str.contains("Duplicate candidate").sum()) if not invalid.empty else 0)
print(
    profiles.groupby("profile_metric")
    .agg(oracle_avg=("oracle_value", "mean"), model_selected_avg=("baseline_value", "mean"), gap=("absolute_gap", "mean"))
    .round(3)
    .sort_index()
    .to_string()
)
PY
```

Expected: duplicate invalid count is `0`; profile metrics include `opponent_overlap_round_rate`, `home_player_share`, `favorite_proxy_ppg_diff_positive_share`, `median_model_predicted_rank_position`, and `top5_position_rank_share`.

- [ ] **Step 5: Commit final verification notes if docs changed**

Only if documentation was changed:

```bash
git add docs/superpowers/plans/2026-05-07-oracle-profiles-and-duplicate-candidates.md
git commit -m "docs: plan oracle profile discovery fix"
```

---

## Self-Review

- Spec coverage: duplicate candidate issue is covered by Task 1; profile metrics are covered by Tasks 2-4; real experiment rerun and acceptance are covered by Task 5.
- Leakage guardrail: all metrics use persisted source artifacts and same-source fixtures only; full-market claims remain unavailable.
- No hard policy promotion: report remains discovery-only and compares oracle-selected rows to model-selected rows, not production recommendations.
- Known non-goal: no automatic lineup policy penalty is added in this plan. The output should inform a future policy proposal after the profile metrics are stable.

