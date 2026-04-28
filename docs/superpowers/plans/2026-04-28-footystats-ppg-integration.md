# FootyStats PPG Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an explicit, leakage-safe `footystats_mode="ppg"` that joins FootyStats pre-match PPG context into RandomForest backtests without changing default `none` behavior.

**Architecture:** Keep `FEATURE_COLUMNS` as the base model feature list, add a mode-aware resolver, and pass the resolved columns into `RandomForestPointPredictor`. Put FootyStats parsing and validation in a new `footystats_features.py` module, then wire the runner to load validated rows, merge them into walk-forward frames, and record source/hash/join diagnostics in `run_metadata.json`.

**Tech Stack:** Python 3.13, pandas, scikit-learn RandomForest pipeline, pytest, uv.

---

## File Structure

- Modify `src/cartola/backtesting/config.py`
  - Add FootyStats config literals and fields: `footystats_mode`, `footystats_evaluation_scope`, `footystats_league_slug`, `footystats_dir`, `current_year`.
- Modify `src/cartola/backtesting/cli.py`
  - Add `--output-root`, `--footystats-mode`, `--footystats-evaluation-scope`, `--footystats-league-slug`, `--footystats-dir`, and `--current-year`.
- Modify `src/cartola/backtesting/features.py`
  - Add `FOOTYSTATS_PPG_FEATURE_COLUMNS` and `feature_columns_for_config`.
  - Add optional `footystats_rows` arguments to frame builders.
  - Merge validated FootyStats rows into prediction/training frames.
- Modify `src/cartola/backtesting/models.py`
  - Make `RandomForestPointPredictor` accept explicit `feature_columns`.
- Create `src/cartola/backtesting/footystats_features.py`
  - Load a selected FootyStats matches file.
  - Reuse audit helpers for filename parsing and team mapping.
  - Normalize match rows into club-side PPG rows.
  - Validate status, PPG values, uniqueness, join coverage, and source hash.
- Modify `src/cartola/backtesting/runner.py`
  - Resolve FootyStats rows based on config.
  - Reject `live_current` in the existing backtest CLI path.
  - Pass rows into frame builders.
  - Pass resolved feature columns into RF.
  - Extend `BacktestMetadata`.
- Modify tests:
  - `src/tests/backtesting/test_cli.py`
  - `src/tests/backtesting/test_features.py`
  - `src/tests/backtesting/test_models.py`
  - `src/tests/backtesting/test_runner.py`
  - Create `src/tests/backtesting/test_footystats_features.py`
- Modify `README.md`
  - Add isolated ablation commands.

---

### Task 1: Config, CLI, and Metadata Shape

**Files:**
- Modify: `src/cartola/backtesting/config.py`
- Modify: `src/cartola/backtesting/cli.py`
- Modify: `src/cartola/backtesting/runner.py`
- Modify: `src/tests/backtesting/test_cli.py`
- Modify: `src/tests/backtesting/test_runner.py`

- [ ] **Step 1: Write failing CLI/config tests**

Add these tests to `src/tests/backtesting/test_cli.py`:

```python
from pathlib import Path

from cartola.backtesting import cli


def test_parse_args_accepts_footystats_and_output_options() -> None:
    args = cli.parse_args(
        [
            "--season",
            "2025",
            "--output-root",
            "data/08_reporting/backtests/footystats_ppg",
            "--footystats-mode",
            "ppg",
            "--footystats-evaluation-scope",
            "historical_candidate",
            "--footystats-league-slug",
            "brazil-serie-a",
            "--footystats-dir",
            "data/footystats",
            "--current-year",
            "2026",
        ]
    )

    assert args.output_root == Path("data/08_reporting/backtests/footystats_ppg")
    assert args.footystats_mode == "ppg"
    assert args.footystats_evaluation_scope == "historical_candidate"
    assert args.footystats_league_slug == "brazil-serie-a"
    assert args.footystats_dir == Path("data/footystats")
    assert args.current_year == 2026
```

Add this assertion block to the fake config captured in the existing CLI run test that monkeypatches `run_backtest`:

```python
assert observed_config.output_root == Path("data/08_reporting/backtests")
assert observed_config.footystats_mode == "none"
assert observed_config.footystats_evaluation_scope == "historical_candidate"
assert observed_config.footystats_league_slug == "brazil-serie-a"
assert observed_config.footystats_dir == Path("data/footystats")
assert observed_config.current_year is None
```

Add this test to `src/tests/backtesting/test_runner.py`:

```python
import json


def test_run_backtest_metadata_records_default_footystats_mode(tmp_path):
    season_df = pd.concat([_tiny_round(round_number) for round_number in range(1, 6)], ignore_index=True)
    config = BacktestConfig(project_root=tmp_path, start_round=5, budget=100)

    run_backtest(config, season_df=season_df)

    metadata = json.loads((tmp_path / "data/08_reporting/backtests/2025/run_metadata.json").read_text())
    assert metadata["footystats_mode"] == "none"
    assert metadata["footystats_evaluation_scope"] == "historical_candidate"
    assert metadata["footystats_league_slug"] == "brazil-serie-a"
    assert metadata["footystats_matches_source_path"] is None
    assert metadata["footystats_matches_source_sha256"] is None
    assert metadata["footystats_feature_columns"] == []
    assert metadata["footystats_missing_join_keys_by_round"] == {}
    assert metadata["footystats_duplicate_join_keys_by_round"] == {}
    assert metadata["footystats_extra_club_rows_by_round"] == {}
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
uv run --frozen pytest \
  src/tests/backtesting/test_cli.py::test_parse_args_accepts_footystats_and_output_options \
  src/tests/backtesting/test_runner.py::test_run_backtest_metadata_records_default_footystats_mode \
  -q
```

Expected: fail because CLI args and metadata fields do not exist.

- [ ] **Step 3: Add config literals and fields**

In `src/cartola/backtesting/config.py`, add:

```python
FootyStatsMode = Literal["none", "ppg"]
FootyStatsEvaluationScope = Literal["historical_candidate", "live_current"]
```

Then add these fields to `BacktestConfig`:

```python
    footystats_mode: FootyStatsMode = "none"
    footystats_evaluation_scope: FootyStatsEvaluationScope = "historical_candidate"
    footystats_league_slug: str = "brazil-serie-a"
    footystats_dir: Path = Path("data/footystats")
    current_year: int | None = None
```

- [ ] **Step 4: Add CLI arguments and config wiring**

In `src/cartola/backtesting/cli.py`, add parser arguments:

```python
    parser.add_argument("--output-root", type=Path, default=Path("data/08_reporting/backtests"))
    parser.add_argument("--footystats-mode", choices=("none", "ppg"), default="none")
    parser.add_argument(
        "--footystats-evaluation-scope",
        choices=("historical_candidate", "live_current"),
        default="historical_candidate",
    )
    parser.add_argument("--footystats-league-slug", default="brazil-serie-a")
    parser.add_argument("--footystats-dir", type=Path, default=Path("data/footystats"))
    parser.add_argument("--current-year", type=int, default=None)
```

Pass them into `BacktestConfig`:

```python
        output_root=args.output_root,
        footystats_mode=args.footystats_mode,
        footystats_evaluation_scope=args.footystats_evaluation_scope,
        footystats_league_slug=args.footystats_league_slug,
        footystats_dir=args.footystats_dir,
        current_year=args.current_year,
```

- [ ] **Step 5: Extend metadata defaults**

In `src/cartola/backtesting/runner.py`, extend `BacktestMetadata`:

```python
    footystats_mode: str
    footystats_evaluation_scope: str
    footystats_league_slug: str
    footystats_matches_source_path: str | None
    footystats_matches_source_sha256: str | None
    footystats_feature_columns: list[str]
    footystats_missing_join_keys_by_round: dict[str, list[dict[str, int]]]
    footystats_duplicate_join_keys_by_round: dict[str, list[dict[str, int]]]
    footystats_extra_club_rows_by_round: dict[str, list[dict[str, int]]]
```

Populate neutral defaults where metadata is created:

```python
        footystats_mode=config.footystats_mode,
        footystats_evaluation_scope=config.footystats_evaluation_scope,
        footystats_league_slug=config.footystats_league_slug,
        footystats_matches_source_path=None,
        footystats_matches_source_sha256=None,
        footystats_feature_columns=[],
        footystats_missing_join_keys_by_round={},
        footystats_duplicate_join_keys_by_round={},
        footystats_extra_club_rows_by_round={},
```

- [ ] **Step 6: Run tests to verify they pass**

Run:

```bash
uv run --frozen pytest \
  src/tests/backtesting/test_cli.py::test_parse_args_accepts_footystats_and_output_options \
  src/tests/backtesting/test_runner.py::test_run_backtest_metadata_records_default_footystats_mode \
  -q
```

Expected: pass.

- [ ] **Step 7: Commit**

```bash
git add src/cartola/backtesting/config.py src/cartola/backtesting/cli.py src/cartola/backtesting/runner.py src/tests/backtesting/test_cli.py src/tests/backtesting/test_runner.py
git commit -m "feat: add footystats backtest config metadata"
```

---

### Task 2: Dynamic Feature Columns and RF Model Contract

**Files:**
- Modify: `src/cartola/backtesting/features.py`
- Modify: `src/cartola/backtesting/models.py`
- Modify: `src/cartola/backtesting/runner.py`
- Modify: `src/tests/backtesting/test_features.py`
- Modify: `src/tests/backtesting/test_models.py`

- [ ] **Step 1: Write failing feature resolver tests**

Add to `src/tests/backtesting/test_features.py`:

```python
from cartola.backtesting.config import BacktestConfig
from cartola.backtesting.features import FOOTYSTATS_PPG_FEATURE_COLUMNS, feature_columns_for_config


def test_feature_columns_for_none_excludes_footystats_columns() -> None:
    columns = feature_columns_for_config(BacktestConfig(footystats_mode="none"))

    for column in FOOTYSTATS_PPG_FEATURE_COLUMNS:
        assert column not in columns


def test_feature_columns_for_ppg_includes_footystats_columns_after_base_columns() -> None:
    base_columns = feature_columns_for_config(BacktestConfig(footystats_mode="none"))
    ppg_columns = feature_columns_for_config(BacktestConfig(footystats_mode="ppg"))

    assert ppg_columns[: len(base_columns)] == base_columns
    assert ppg_columns[-3:] == FOOTYSTATS_PPG_FEATURE_COLUMNS
```

- [ ] **Step 2: Write failing model contract test**

Modify `test_random_forest_point_predictor_fit_predict_smoke` in `src/tests/backtesting/test_models.py`:

```python
def test_random_forest_point_predictor_fit_predict_smoke() -> None:
    train = _model_frame()
    predict = train.drop(columns=["target"]).copy()

    model = RandomForestPointPredictor(random_seed=7, feature_columns=FEATURE_COLUMNS).fit(train)
    predictions = model.predict(predict)

    assert len(predictions) == len(predict)
    assert predictions.notna().all()
```

Add:

```python
def test_random_forest_point_predictor_uses_explicit_feature_columns() -> None:
    train = _model_frame()
    train["footystats_team_pre_match_ppg"] = [1.0, 2.0, 1.5, 1.2]
    train["footystats_opponent_pre_match_ppg"] = [1.2, 1.0, 1.1, 1.4]
    train["footystats_ppg_diff"] = (
        train["footystats_team_pre_match_ppg"] - train["footystats_opponent_pre_match_ppg"]
    )
    feature_columns = [*FEATURE_COLUMNS, "footystats_team_pre_match_ppg", "footystats_opponent_pre_match_ppg", "footystats_ppg_diff"]

    model = RandomForestPointPredictor(random_seed=7, feature_columns=feature_columns).fit(train)
    predictions = model.predict(train.drop(columns=["target"]))

    assert len(predictions) == len(train)
    assert predictions.notna().all()
```

- [ ] **Step 3: Run tests to verify they fail**

Run:

```bash
uv run --frozen pytest \
  src/tests/backtesting/test_features.py::test_feature_columns_for_none_excludes_footystats_columns \
  src/tests/backtesting/test_features.py::test_feature_columns_for_ppg_includes_footystats_columns_after_base_columns \
  src/tests/backtesting/test_models.py::test_random_forest_point_predictor_uses_explicit_feature_columns \
  -q
```

Expected: fail because resolver and model constructor argument do not exist.

- [ ] **Step 4: Add resolver**

In `src/cartola/backtesting/features.py`, import `BacktestConfig` under `TYPE_CHECKING` to avoid runtime cycles:

```python
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cartola.backtesting.config import BacktestConfig
```

Add after `FEATURE_COLUMNS`:

```python
FOOTYSTATS_PPG_FEATURE_COLUMNS: list[str] = [
    "footystats_team_pre_match_ppg",
    "footystats_opponent_pre_match_ppg",
    "footystats_ppg_diff",
]


def feature_columns_for_config(config: "BacktestConfig") -> list[str]:
    if config.footystats_mode == "none":
        return list(FEATURE_COLUMNS)
    if config.footystats_mode == "ppg":
        return [*FEATURE_COLUMNS, *FOOTYSTATS_PPG_FEATURE_COLUMNS]
    raise ValueError(f"Unsupported footystats_mode={config.footystats_mode!r}")
```

- [ ] **Step 5: Make RF model use explicit columns**

In `src/cartola/backtesting/models.py`, remove the `FEATURE_COLUMNS` import and update the class:

```python
class RandomForestPointPredictor:
    def __init__(self, random_seed: int = 123, feature_columns: list[str] | None = None) -> None:
        if feature_columns is None:
            raise ValueError("feature_columns must be provided")
        self.feature_columns = list(feature_columns)
        numeric_features = [column for column in self.feature_columns if column != "posicao"]
        categorical_features = ["posicao"] if "posicao" in self.feature_columns else []
```

Keep the existing pipeline structure, but use `categorical_features` and `numeric_features` from the explicit list. Update fit/predict:

```python
    def fit(self, frame: pd.DataFrame) -> RandomForestPointPredictor:
        self.pipeline.fit(frame[self.feature_columns], frame["target"])
        return self

    def predict(self, frame: pd.DataFrame) -> pd.Series:
        predictions = self.pipeline.predict(frame[self.feature_columns])
        return pd.Series(predictions, index=frame.index, dtype=float)
```

- [ ] **Step 6: Wire runner to pass resolved columns**

In `src/cartola/backtesting/runner.py`, import:

```python
from cartola.backtesting.features import build_prediction_frame, build_training_frame, feature_columns_for_config
```

Before the round loop, add:

```python
    model_feature_columns = feature_columns_for_config(config)
```

Update RF creation:

```python
        forest_model = RandomForestPointPredictor(
            random_seed=config.random_seed,
            feature_columns=model_feature_columns,
        ).fit(training)
```

- [ ] **Step 7: Run targeted tests**

Run:

```bash
uv run --frozen pytest src/tests/backtesting/test_features.py src/tests/backtesting/test_models.py -q
```

Expected: pass.

- [ ] **Step 8: Commit**

```bash
git add src/cartola/backtesting/features.py src/cartola/backtesting/models.py src/cartola/backtesting/runner.py src/tests/backtesting/test_features.py src/tests/backtesting/test_models.py
git commit -m "feat: resolve backtest feature columns by config"
```

---

### Task 3: FootyStats PPG Loader and Validation

**Files:**
- Create: `src/cartola/backtesting/footystats_features.py`
- Create: `src/tests/backtesting/test_footystats_features.py`

- [ ] **Step 1: Write loader tests**

Create `src/tests/backtesting/test_footystats_features.py` with:

```python
from pathlib import Path

import pandas as pd
import pytest

from cartola.backtesting import footystats_features as features


def _write_cartola_round(root: Path, season: int = 2025) -> None:
    season_dir = root / "data" / "01_raw" / str(season)
    season_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "atletas.clube_id": [262, 275],
            "atletas.clube.id.full.name": ["Flamengo", "Palmeiras"],
        }
    ).to_csv(season_dir / "rodada-1.csv", index=False)


def _write_matches(root: Path, rows: list[dict[str, object]], season: int = 2025) -> Path:
    footystats_dir = root / "data" / "footystats"
    footystats_dir.mkdir(parents=True)
    path = footystats_dir / f"brazil-serie-a-matches-{season}-to-{season}-stats.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _complete_rows(rounds: range = range(1, 39)) -> list[dict[str, object]]:
    return [
        {
            "Game Week": round_number,
            "home_team_name": "Flamengo",
            "away_team_name": "Palmeiras",
            "Pre-Match PPG (Home)": 2.1,
            "Pre-Match PPG (Away)": 1.4,
            "status": "complete",
            "home_team_goal_count": 3,
        }
        for round_number in rounds
    ]


def test_load_footystats_ppg_rows_builds_home_and_away_rows_and_excludes_outcomes(tmp_path: Path) -> None:
    _write_cartola_round(tmp_path)
    source_path = _write_matches(tmp_path, _complete_rows())

    result = features.load_footystats_ppg_rows(
        season=2025,
        project_root=tmp_path,
        footystats_dir=Path("data/footystats"),
        league_slug="brazil-serie-a",
        evaluation_scope="historical_candidate",
        current_year=2026,
    )

    assert result.source_path == source_path
    assert len(result.rows) == 76
    assert result.source_sha256
    assert "home_team_goal_count" not in result.rows.columns
    home = result.rows[result.rows["id_clube"].eq(262)].iloc[0]
    away = result.rows[result.rows["id_clube"].eq(275)].iloc[0]
    assert home["rodada"] == 1
    assert home["is_home_footystats"] == 1
    assert home["footystats_team_pre_match_ppg"] == 2.1
    assert home["footystats_opponent_pre_match_ppg"] == 1.4
    assert home["footystats_ppg_diff"] == pytest.approx(0.7)
    assert away["is_home_footystats"] == 0
    assert away["footystats_ppg_diff"] == pytest.approx(-0.7)


def test_load_footystats_ppg_rows_rejects_historical_incomplete_game_week_coverage(tmp_path: Path) -> None:
    _write_cartola_round(tmp_path)
    _write_matches(tmp_path, _complete_rows(range(1, 20)))

    with pytest.raises(ValueError, match="Historical FootyStats rows must cover game weeks 1-38"):
        features.load_footystats_ppg_rows(
            season=2025,
            project_root=tmp_path,
            footystats_dir=Path("data/footystats"),
            league_slug="brazil-serie-a",
            evaluation_scope="historical_candidate",
            current_year=2026,
        )


def test_load_footystats_ppg_rows_rejects_missing_ppg_column(tmp_path: Path) -> None:
    _write_cartola_round(tmp_path)
    rows = _complete_rows()
    del rows[0]["Pre-Match PPG (Away)"]
    _write_matches(tmp_path, rows)

    with pytest.raises(ValueError, match="Missing required FootyStats columns"):
        features.load_footystats_ppg_rows(
            season=2025,
            project_root=tmp_path,
            footystats_dir=Path("data/footystats"),
            league_slug="brazil-serie-a",
            evaluation_scope="historical_candidate",
            current_year=2026,
        )


def test_load_footystats_ppg_rows_rejects_missing_ppg_value(tmp_path: Path) -> None:
    _write_cartola_round(tmp_path)
    rows = _complete_rows()
    rows[0]["Pre-Match PPG (Away)"] = None
    _write_matches(tmp_path, rows)

    with pytest.raises(ValueError, match="Missing FootyStats PPG values"):
        features.load_footystats_ppg_rows(
            season=2025,
            project_root=tmp_path,
            footystats_dir=Path("data/footystats"),
            league_slug="brazil-serie-a",
            evaluation_scope="historical_candidate",
            current_year=2026,
        )


def test_load_footystats_ppg_rows_rejects_historical_non_complete_status(tmp_path: Path) -> None:
    _write_cartola_round(tmp_path)
    rows = _complete_rows()
    rows[0]["status"] = "incomplete"
    _write_matches(tmp_path, rows)

    with pytest.raises(ValueError, match="Historical FootyStats rows must be complete"):
        features.load_footystats_ppg_rows(
            season=2025,
            project_root=tmp_path,
            footystats_dir=Path("data/footystats"),
            league_slug="brazil-serie-a",
            evaluation_scope="historical_candidate",
            current_year=2026,
        )


def test_load_footystats_ppg_rows_rejects_live_current_in_loader_for_non_current_year(tmp_path: Path) -> None:
    _write_cartola_round(tmp_path)
    _write_matches(tmp_path, _complete_rows())

    with pytest.raises(ValueError, match="live_current requires season to equal current_year"):
        features.load_footystats_ppg_rows(
            season=2025,
            project_root=tmp_path,
            footystats_dir=Path("data/footystats"),
            league_slug="brazil-serie-a",
            evaluation_scope="live_current",
            current_year=2026,
        )
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
uv run --frozen pytest src/tests/backtesting/test_footystats_features.py -q
```

Expected: fail because `footystats_features.py` does not exist.

- [ ] **Step 3: Implement loader module**

Create `src/cartola/backtesting/footystats_features.py`:

```python
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from cartola.backtesting.footystats_audit import compare_teams_to_cartola, parse_footystats_filename

REQUIRED_FOOTYSTATS_COLUMNS: tuple[str, ...] = (
    "Game Week",
    "home_team_name",
    "away_team_name",
    "Pre-Match PPG (Home)",
    "Pre-Match PPG (Away)",
    "status",
)

FOOTYSTATS_PPG_COLUMNS: tuple[str, ...] = (
    "footystats_team_pre_match_ppg",
    "footystats_opponent_pre_match_ppg",
    "footystats_ppg_diff",
)


@dataclass(frozen=True)
class FootyStatsJoinDiagnostics:
    missing_join_keys_by_round: dict[str, list[dict[str, int]]] = field(default_factory=dict)
    duplicate_join_keys_by_round: dict[str, list[dict[str, int]]] = field(default_factory=dict)
    extra_club_rows_by_round: dict[str, list[dict[str, int]]] = field(default_factory=dict)


@dataclass(frozen=True)
class FootyStatsPPGLoadResult:
    rows: pd.DataFrame
    source_path: Path
    source_sha256: str
    diagnostics: FootyStatsJoinDiagnostics


def load_footystats_ppg_rows(
    *,
    season: int,
    project_root: Path,
    footystats_dir: Path,
    league_slug: str,
    evaluation_scope: str,
    current_year: int | None,
) -> FootyStatsPPGLoadResult:
    if evaluation_scope == "live_current" and season != _resolved_current_year(current_year):
        raise ValueError("live_current requires season to equal current_year")
    source_path = _source_path(project_root, footystats_dir, league_slug, season)
    if not source_path.exists():
        raise FileNotFoundError(source_path)
    parsed = parse_footystats_filename(source_path)
    if parsed.season != season or parsed.league_slug != league_slug or parsed.table_type != "matches":
        raise ValueError("FootyStats source filename does not match requested season, league, and matches table")

    df = pd.read_csv(source_path)
    missing_columns = [column for column in REQUIRED_FOOTYSTATS_COLUMNS if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required FootyStats columns: {missing_columns}")

    _validate_scope_status(df, evaluation_scope)
    _validate_historical_game_weeks(df["Game Week"], evaluation_scope)
    _validate_ppg_values(df)
    team_names = sorted({*df["home_team_name"].dropna().astype(str), *df["away_team_name"].dropna().astype(str)})
    comparison = compare_teams_to_cartola(season, team_names, project_root)
    if comparison.unmapped_footystats_teams or comparison.missing_cartola_teams or comparison.duplicate_mapped_cartola_teams:
        raise ValueError(
            "FootyStats team mapping failed: "
            f"unmapped={comparison.unmapped_footystats_teams}, "
            f"missing_cartola={comparison.missing_cartola_teams}, "
            f"duplicates={comparison.duplicate_mapped_cartola_teams}"
        )

    rows = _club_side_rows(df, comparison.mapped_teams)
    duplicates = _duplicate_keys(rows)
    if duplicates:
        raise ValueError(f"Duplicate FootyStats club rows: {duplicates}")

    return FootyStatsPPGLoadResult(
        rows=rows,
        source_path=source_path,
        source_sha256=sha256_file(source_path),
        diagnostics=FootyStatsJoinDiagnostics(),
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_path(project_root: Path, footystats_dir: Path, league_slug: str, season: int) -> Path:
    root = footystats_dir if footystats_dir.is_absolute() else project_root / footystats_dir
    return root / f"{league_slug}-matches-{season}-to-{season}-stats.csv"


def _resolved_current_year(current_year: int | None) -> int:
    if current_year is not None:
        return current_year
    from datetime import UTC, datetime

    return datetime.now(UTC).year


def _validate_scope_status(df: pd.DataFrame, evaluation_scope: str) -> None:
    statuses = df["status"].dropna().astype(str).str.lower()
    if evaluation_scope == "historical_candidate" and not statuses.eq("complete").all():
        raise ValueError("Historical FootyStats rows must be complete")
    if evaluation_scope == "live_current" and not statuses.isin(["complete", "incomplete"]).all():
        raise ValueError("live_current FootyStats rows must be complete or incomplete")
    if evaluation_scope not in {"historical_candidate", "live_current"}:
        raise ValueError(f"Unsupported footystats_evaluation_scope={evaluation_scope!r}")


def _validate_historical_game_weeks(rows: pd.Series, evaluation_scope: str) -> None:
    if evaluation_scope != "historical_candidate":
        return
    game_weeks = pd.to_numeric(rows, errors="coerce")
    valid = game_weeks.dropna()
    integer_weeks = valid[valid.mod(1).eq(0) & valid.gt(0)]
    detected = sorted({int(value) for value in integer_weeks})
    if len(integer_weeks) != len(game_weeks) or detected != list(range(1, 39)):
        raise ValueError("Historical FootyStats rows must cover game weeks 1-38")


def _validate_ppg_values(df: pd.DataFrame) -> None:
    ppg_columns = ["Pre-Match PPG (Home)", "Pre-Match PPG (Away)"]
    numeric = df[ppg_columns].apply(pd.to_numeric, errors="coerce")
    if numeric.isna().any().any():
        raise ValueError("Missing FootyStats PPG values")


def _club_side_rows(df: pd.DataFrame, mapped_teams: dict[str, int]) -> pd.DataFrame:
    rounds = pd.to_numeric(df["Game Week"], errors="coerce")
    if rounds.isna().any() or not rounds.mod(1).eq(0).all() or not rounds.gt(0).all():
        raise ValueError("Game Week must contain positive integers")

    home_ppg = pd.to_numeric(df["Pre-Match PPG (Home)"], errors="raise").astype(float)
    away_ppg = pd.to_numeric(df["Pre-Match PPG (Away)"], errors="raise").astype(float)
    home = pd.DataFrame(
        {
            "rodada": rounds.astype(int),
            "id_clube": df["home_team_name"].map(mapped_teams).astype(int),
            "opponent_id_clube": df["away_team_name"].map(mapped_teams).astype(int),
            "is_home_footystats": 1,
            "footystats_team_pre_match_ppg": home_ppg,
            "footystats_opponent_pre_match_ppg": away_ppg,
            "footystats_ppg_diff": home_ppg - away_ppg,
        }
    )
    away = pd.DataFrame(
        {
            "rodada": rounds.astype(int),
            "id_clube": df["away_team_name"].map(mapped_teams).astype(int),
            "opponent_id_clube": df["home_team_name"].map(mapped_teams).astype(int),
            "is_home_footystats": 0,
            "footystats_team_pre_match_ppg": away_ppg,
            "footystats_opponent_pre_match_ppg": home_ppg,
            "footystats_ppg_diff": away_ppg - home_ppg,
        }
    )
    return pd.concat([home, away], ignore_index=True).sort_values(["rodada", "id_clube"]).reset_index(drop=True)


def _duplicate_keys(rows: pd.DataFrame) -> list[dict[str, int]]:
    duplicates = rows.groupby(["rodada", "id_clube"], as_index=False).size()
    duplicates = duplicates[duplicates["size"] > 1]
    return [
        {"rodada": int(row.rodada), "id_clube": int(row.id_clube), "count": int(row.size)}
        for row in duplicates.itertuples(index=False)
    ]
```

- [ ] **Step 4: Run tests to verify they pass**

Run:

```bash
uv run --frozen pytest src/tests/backtesting/test_footystats_features.py -q
```

Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add src/cartola/backtesting/footystats_features.py src/tests/backtesting/test_footystats_features.py
git commit -m "feat: load footystats ppg rows"
```

---

### Task 4: Merge FootyStats Rows Into Feature Frames

**Files:**
- Modify: `src/cartola/backtesting/features.py`
- Modify: `src/cartola/backtesting/footystats_features.py`
- Modify: `src/tests/backtesting/test_features.py`
- Modify: `src/tests/backtesting/test_footystats_features.py`

- [ ] **Step 1: Write failing frame-merge tests**

Add to `src/tests/backtesting/test_features.py`:

```python
def _footystats_rows() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "rodada": 3,
                "id_clube": 10,
                "opponent_id_clube": 20,
                "is_home_footystats": 1,
                "footystats_team_pre_match_ppg": 2.0,
                "footystats_opponent_pre_match_ppg": 1.1,
                "footystats_ppg_diff": 0.9,
            },
            {
                "rodada": 3,
                "id_clube": 20,
                "opponent_id_clube": 10,
                "is_home_footystats": 0,
                "footystats_team_pre_match_ppg": 1.1,
                "footystats_opponent_pre_match_ppg": 2.0,
                "footystats_ppg_diff": -0.9,
            },
        ]
    )


def test_prediction_frame_merges_footystats_ppg_rows() -> None:
    frame = build_prediction_frame(_season_df(), target_round=3, footystats_rows=_footystats_rows())
    club_10 = frame[frame["id_clube"].eq(10)].iloc[0]

    assert club_10["footystats_team_pre_match_ppg"] == 2.0
    assert club_10["footystats_opponent_pre_match_ppg"] == 1.1
    assert club_10["footystats_ppg_diff"] == pytest.approx(0.9)


def test_prediction_frame_fails_when_candidate_club_missing_footystats_row() -> None:
    rows = _footystats_rows()
    rows = rows[rows["id_clube"].ne(20)]

    with pytest.raises(ValueError, match="Missing FootyStats rows"):
        build_prediction_frame(_season_df(), target_round=3, footystats_rows=rows)
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
uv run --frozen pytest \
  src/tests/backtesting/test_features.py::test_prediction_frame_merges_footystats_ppg_rows \
  src/tests/backtesting/test_features.py::test_prediction_frame_fails_when_candidate_club_missing_footystats_row \
  -q
```

Expected: fail because `footystats_rows` argument does not exist.

- [ ] **Step 3: Add merge helper**

In `src/cartola/backtesting/footystats_features.py`, add:

```python
def merge_footystats_ppg(
    frame: pd.DataFrame,
    footystats_rows: pd.DataFrame | None,
    *,
    target_round: int,
) -> pd.DataFrame:
    if footystats_rows is None:
        return frame

    rows = footystats_rows.copy()
    rows["rodada"] = pd.to_numeric(rows["rodada"], errors="raise").astype(int)
    rows["id_clube"] = pd.to_numeric(rows["id_clube"], errors="raise").astype(int)
    round_rows = rows[rows["rodada"].eq(target_round)].copy()
    duplicates = _duplicate_keys(round_rows)
    if duplicates:
        raise ValueError(f"Duplicate FootyStats rows for round {target_round}: {duplicates}")

    candidate_keys = set(pd.to_numeric(frame["id_clube"], errors="raise").astype(int).dropna().unique())
    row_keys = set(round_rows["id_clube"].dropna().astype(int).unique())
    missing = sorted(candidate_keys - row_keys)
    if missing:
        raise ValueError(f"Missing FootyStats rows for round {target_round}: {missing}")

    feature_rows = round_rows[
        [
            "id_clube",
            "footystats_team_pre_match_ppg",
            "footystats_opponent_pre_match_ppg",
            "footystats_ppg_diff",
        ]
    ].copy()
    merged = frame.merge(feature_rows, on="id_clube", how="left", validate="many_to_one")
    return merged


def build_footystats_join_diagnostics(
    season_df: pd.DataFrame,
    footystats_rows: pd.DataFrame,
) -> FootyStatsJoinDiagnostics:
    missing: dict[str, list[dict[str, int]]] = {}
    extra: dict[str, list[dict[str, int]]] = {}
    duplicate: dict[str, list[dict[str, int]]] = {}

    candidates = season_df[["rodada", "id_clube"]].copy()
    candidates["rodada"] = pd.to_numeric(candidates["rodada"], errors="raise").astype(int)
    candidates["id_clube"] = pd.to_numeric(candidates["id_clube"], errors="raise").astype(int)
    footy = footystats_rows[["rodada", "id_clube"]].copy()
    footy["rodada"] = pd.to_numeric(footy["rodada"], errors="raise").astype(int)
    footy["id_clube"] = pd.to_numeric(footy["id_clube"], errors="raise").astype(int)

    duplicate_rows = footy.groupby(["rodada", "id_clube"], as_index=False).size()
    duplicate_rows = duplicate_rows[duplicate_rows["size"] > 1]
    for row in duplicate_rows.itertuples(index=False):
        duplicate.setdefault(str(int(row.rodada)), []).append(
            {"rodada": int(row.rodada), "id_clube": int(row.id_clube), "count": int(row.size)}
        )

    for round_number in sorted(candidates["rodada"].unique()):
        candidate_clubs = set(candidates.loc[candidates["rodada"].eq(round_number), "id_clube"].unique())
        footy_clubs = set(footy.loc[footy["rodada"].eq(round_number), "id_clube"].unique())
        missing_clubs = sorted(candidate_clubs - footy_clubs)
        extra_clubs = sorted(footy_clubs - candidate_clubs)
        if missing_clubs:
            missing[str(int(round_number))] = [
                {"rodada": int(round_number), "id_clube": int(club_id)} for club_id in missing_clubs
            ]
        if extra_clubs:
            extra[str(int(round_number))] = [
                {"rodada": int(round_number), "id_clube": int(club_id)} for club_id in extra_clubs
            ]

    return FootyStatsJoinDiagnostics(
        missing_join_keys_by_round=missing,
        duplicate_join_keys_by_round=duplicate,
        extra_club_rows_by_round=extra,
    )
```

- [ ] **Step 4: Wire frame builders**

In `src/cartola/backtesting/features.py`, import:

```python
from cartola.backtesting.footystats_features import merge_footystats_ppg
```

Update signatures:

```python
def build_prediction_frame(
    season_df: pd.DataFrame,
    target_round: int,
    fixtures: pd.DataFrame | None = None,
    footystats_rows: pd.DataFrame | None = None,
) -> pd.DataFrame:
```

```python
def build_training_frame(
    season_df: pd.DataFrame,
    target_round: int,
    playable_statuses: tuple[str, ...] | None = None,
    fixtures: pd.DataFrame | None = None,
    footystats_rows: pd.DataFrame | None = None,
) -> pd.DataFrame:
```

Pass `footystats_rows` inside training:

```python
        round_frame = build_prediction_frame(
            season_df,
            int(round_number),
            fixtures=fixtures,
            footystats_rows=footystats_rows,
        )
```

In `build_prediction_frame`, merge after prior features:

```python
    frame = _add_prior_features(candidates, played_history, all_history, fixtures, target_round)
    return merge_footystats_ppg(frame, footystats_rows, target_round=target_round)
```

- [ ] **Step 5: Update compatibility audit monkeypatch lambdas**

Where tests monkeypatch `build_prediction_frame` or `build_training_frame`, update lambdas to accept `footystats_rows=None`:

```python
lambda season_df, target_round, fixtures=None, footystats_rows=None: pd.DataFrame()
```

and:

```python
lambda season_df, target_round, playable_statuses=None, fixtures=None, footystats_rows=None: pd.DataFrame()
```

- [ ] **Step 6: Run targeted tests**

Run:

```bash
uv run --frozen pytest src/tests/backtesting/test_features.py src/tests/backtesting/test_footystats_features.py -q
```

Expected: pass.

- [ ] **Step 7: Commit**

```bash
git add src/cartola/backtesting/features.py src/cartola/backtesting/footystats_features.py src/tests/backtesting/test_features.py src/tests/backtesting/test_compatibility_audit.py src/tests/backtesting/test_footystats_features.py
git commit -m "feat: merge footystats ppg into feature frames"
```

---

### Task 5: Runner Integration, Historical Guard, and Metadata

**Files:**
- Modify: `src/cartola/backtesting/runner.py`
- Modify: `src/tests/backtesting/test_runner.py`

- [ ] **Step 1: Write failing runner tests**

Add to `src/tests/backtesting/test_runner.py`:

```python
def test_run_backtest_rejects_live_current_scope(tmp_path):
    season_df = pd.concat([_tiny_round(round_number) for round_number in range(1, 6)], ignore_index=True)
    config = BacktestConfig(
        project_root=tmp_path,
        start_round=5,
        budget=100,
        footystats_mode="ppg",
        footystats_evaluation_scope="live_current",
        current_year=2025,
    )

    with pytest.raises(ValueError, match="live_current is not supported by the backtest runner"):
        run_backtest(config, season_df=season_df)


def test_run_backtest_ppg_passes_footystats_rows_and_metadata(tmp_path, monkeypatch):
    season_df = pd.concat([_tiny_round(round_number) for round_number in range(1, 6)], ignore_index=True)
    rows = []
    for round_number in range(1, 6):
        for club_id in range(1, 19):
            rows.append(
                {
                    "rodada": round_number,
                    "id_clube": club_id,
                    "opponent_id_clube": 999,
                    "is_home_footystats": 1,
                    "footystats_team_pre_match_ppg": 2.0,
                    "footystats_opponent_pre_match_ppg": 1.0,
                    "footystats_ppg_diff": 1.0,
                }
            )
    footystats_rows = pd.DataFrame(rows)

    class FakeLoadResult:
        rows = footystats_rows
        source_path = tmp_path / "data/footystats/brazil-serie-a-matches-2025-to-2025-stats.csv"
        source_sha256 = "abc123"
        diagnostics = type(
            "Diagnostics",
            (),
            {
                "missing_join_keys_by_round": {},
                "duplicate_join_keys_by_round": {},
                "extra_club_rows_by_round": {},
            },
        )()

    observed_kwargs = {}

    def fake_load_footystats_ppg_rows(**kwargs):
        observed_kwargs.update(kwargs)
        return FakeLoadResult()

    monkeypatch.setattr("cartola.backtesting.runner.load_footystats_ppg_rows", fake_load_footystats_ppg_rows)
    config = BacktestConfig(project_root=tmp_path, start_round=5, budget=100, footystats_mode="ppg")

    result = run_backtest(config, season_df=season_df)

    assert observed_kwargs["season"] == 2025
    assert observed_kwargs["league_slug"] == "brazil-serie-a"
    assert result.metadata.footystats_mode == "ppg"
    assert result.metadata.footystats_matches_source_sha256 == "abc123"
    assert result.metadata.footystats_feature_columns == [
        "footystats_team_pre_match_ppg",
        "footystats_opponent_pre_match_ppg",
        "footystats_ppg_diff",
    ]
    assert "footystats_ppg_diff" in result.player_predictions.columns
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
uv run --frozen pytest \
  src/tests/backtesting/test_runner.py::test_run_backtest_rejects_live_current_scope \
  src/tests/backtesting/test_runner.py::test_run_backtest_ppg_passes_footystats_rows_and_metadata \
  -q
```

Expected: fail because runner does not load FootyStats or reject live scope.

- [ ] **Step 3: Add FootyStats load state**

In `src/cartola/backtesting/runner.py`, import:

```python
from cartola.backtesting.features import FOOTYSTATS_PPG_FEATURE_COLUMNS
from cartola.backtesting.footystats_features import (
    FootyStatsJoinDiagnostics,
    FootyStatsPPGLoadResult,
    build_footystats_join_diagnostics,
    load_footystats_ppg_rows,
)
```

Add helper:

```python
def _resolve_footystats(config: BacktestConfig) -> FootyStatsPPGLoadResult | None:
    if config.footystats_mode == "none":
        return None
    if config.footystats_mode != "ppg":
        raise ValueError(f"Unknown footystats_mode: {config.footystats_mode!r}")
    if config.footystats_evaluation_scope == "live_current":
        raise ValueError("live_current is not supported by the backtest runner")
    return load_footystats_ppg_rows(
        season=config.season,
        project_root=config.project_root,
        footystats_dir=config.footystats_dir,
        league_slug=config.footystats_league_slug,
        evaluation_scope=config.footystats_evaluation_scope,
        current_year=config.current_year,
    )
```

- [ ] **Step 4: Wire runner**

At the top of `run_backtest`, after fixture resolution:

```python
    footystats_load = _resolve_footystats(config)
    footystats_rows = footystats_load.rows if footystats_load is not None else None
    footystats_diagnostics = (
        build_footystats_join_diagnostics(data, footystats_rows)
        if footystats_rows is not None
        else FootyStatsJoinDiagnostics()
    )
    if footystats_diagnostics.missing_join_keys_by_round:
        raise ValueError(f"Missing FootyStats join keys: {footystats_diagnostics.missing_join_keys_by_round}")
    if footystats_diagnostics.duplicate_join_keys_by_round:
        raise ValueError(f"Duplicate FootyStats join keys: {footystats_diagnostics.duplicate_join_keys_by_round}")
```

Pass `footystats_rows` into frame builders:

```python
        training = build_training_frame(
            data,
            round_number,
            playable_statuses=config.playable_statuses,
            fixtures=fixture_data,
            footystats_rows=footystats_rows,
        )
        candidates = build_prediction_frame(
            data,
            round_number,
            fixtures=fixture_data,
            footystats_rows=footystats_rows,
        )
```

Populate metadata:

```python
        footystats_matches_source_path=str(footystats_load.source_path) if footystats_load is not None else None,
        footystats_matches_source_sha256=footystats_load.source_sha256 if footystats_load is not None else None,
        footystats_feature_columns=list(FOOTYSTATS_PPG_FEATURE_COLUMNS) if footystats_load is not None else [],
        footystats_missing_join_keys_by_round=(
            footystats_diagnostics.missing_join_keys_by_round
        ),
        footystats_duplicate_join_keys_by_round=(
            footystats_diagnostics.duplicate_join_keys_by_round
        ),
        footystats_extra_club_rows_by_round=(
            footystats_diagnostics.extra_club_rows_by_round
        ),
```

- [ ] **Step 5: Run runner tests**

Run:

```bash
uv run --frozen pytest src/tests/backtesting/test_runner.py -q
```

Expected: pass.

- [ ] **Step 6: Commit**

```bash
git add src/cartola/backtesting/runner.py src/tests/backtesting/test_runner.py
git commit -m "feat: wire footystats ppg into backtest runner"
```

---

### Task 6: CLI Integration, Docs, and Measurement Commands

**Files:**
- Modify: `README.md`
- Modify: `src/tests/backtesting/test_cli.py`

- [ ] **Step 1: Add CLI integration test for output root**

In `src/tests/backtesting/test_cli.py`, add a monkeypatched `run_backtest` assertion:

```python
def test_main_passes_footystats_options_to_config(monkeypatch) -> None:
    observed = {}

    def fake_run_backtest(config):
        observed["config"] = config
        return _fake_result()

    monkeypatch.setattr(cli, "run_backtest", fake_run_backtest)

    cli.main(
        [
            "--season",
            "2025",
            "--output-root",
            "data/08_reporting/backtests/footystats_ppg",
            "--footystats-mode",
            "ppg",
            "--footystats-evaluation-scope",
            "historical_candidate",
            "--footystats-league-slug",
            "brazil-serie-a",
            "--current-year",
            "2026",
        ]
    )

    config = observed["config"]
    assert config.output_root == Path("data/08_reporting/backtests/footystats_ppg")
    assert config.output_path == Path("data/08_reporting/backtests/footystats_ppg/2025")
    assert config.footystats_mode == "ppg"
    assert config.footystats_evaluation_scope == "historical_candidate"
    assert config.footystats_league_slug == "brazil-serie-a"
    assert config.current_year == 2026
```

- [ ] **Step 2: Run CLI tests to verify failure or pass**

Run:

```bash
uv run --frozen pytest src/tests/backtesting/test_cli.py -q
```

Expected: pass if Task 1 covered all wiring; otherwise fail with the missing config field.

- [ ] **Step 3: Update README commands**

Add a section to `README.md`:

````markdown
### FootyStats PPG ablation

Run the no-FootyStats baseline and the PPG feature run into isolated output roots:

```bash
uv run --frozen python -m cartola.backtesting.cli \
  --season 2025 \
  --start-round 5 \
  --budget 100 \
  --fixture-mode exploratory \
  --footystats-mode none \
  --output-root data/08_reporting/backtests/footystats_none

uv run --frozen python -m cartola.backtesting.cli \
  --season 2025 \
  --start-round 5 \
  --budget 100 \
  --fixture-mode exploratory \
  --footystats-mode ppg \
  --footystats-evaluation-scope historical_candidate \
  --footystats-league-slug brazil-serie-a \
  --current-year 2026 \
  --output-root data/08_reporting/backtests/footystats_ppg
```

The outputs land in:

- `data/08_reporting/backtests/footystats_none/2025/`
- `data/08_reporting/backtests/footystats_ppg/2025/`
````

- [ ] **Step 4: Run docs and CLI tests**

Run:

```bash
uv run --frozen pytest src/tests/backtesting/test_cli.py -q
```

Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add README.md src/tests/backtesting/test_cli.py
git commit -m "docs: add footystats ppg ablation commands"
```

---

### Task 7: Verification and Ablation Run

**Files:**
- No source edits unless verification reveals a bug.

- [ ] **Step 1: Run focused test suite**

Run:

```bash
uv run --frozen pytest \
  src/tests/backtesting/test_footystats_features.py \
  src/tests/backtesting/test_features.py \
  src/tests/backtesting/test_models.py \
  src/tests/backtesting/test_runner.py \
  src/tests/backtesting/test_cli.py \
  -q
```

Expected: all selected tests pass.

- [ ] **Step 2: Run full project gate**

Run:

```bash
uv run --frozen scripts/pyrepo-check --all
```

Expected: Ruff, ty, Bandit, and pytest pass.

- [ ] **Step 3: Run no-FootyStats baseline**

Run:

```bash
uv run --frozen python -m cartola.backtesting.cli \
  --season 2025 \
  --start-round 5 \
  --budget 100 \
  --fixture-mode exploratory \
  --footystats-mode none \
  --output-root data/08_reporting/backtests/footystats_none
```

Expected: output written to `data/08_reporting/backtests/footystats_none/2025/`.

- [ ] **Step 4: Run PPG ablation**

Run:

```bash
uv run --frozen python -m cartola.backtesting.cli \
  --season 2025 \
  --start-round 5 \
  --budget 100 \
  --fixture-mode exploratory \
  --footystats-mode ppg \
  --footystats-evaluation-scope historical_candidate \
  --footystats-league-slug brazil-serie-a \
  --current-year 2026 \
  --output-root data/08_reporting/backtests/footystats_ppg
```

Expected: output written to `data/08_reporting/backtests/footystats_ppg/2025/`.

- [ ] **Step 5: Compare metrics**

Run:

```bash
python - <<'PY'
from pathlib import Path
import pandas as pd

roots = {
    "none": Path("data/08_reporting/backtests/footystats_none/2025"),
    "ppg": Path("data/08_reporting/backtests/footystats_ppg/2025"),
}
for label, root in roots.items():
    summary = pd.read_csv(root / "summary.csv")
    diagnostics = pd.read_csv(root / "diagnostics.csv")
    rf_avg = summary.loc[summary["strategy"].eq("random_forest"), "average_actual_points"].iloc[0]
    r2 = diagnostics.loc[
        diagnostics["metric"].eq("player_r2") & diagnostics["strategy"].eq("random_forest") & diagnostics["group"].eq("all"),
        "value",
    ].iloc[0]
    corr = diagnostics.loc[
        diagnostics["metric"].eq("player_correlation") & diagnostics["strategy"].eq("random_forest") & diagnostics["group"].eq("all"),
        "value",
    ].iloc[0]
    print(f"{label}: rf_avg={rf_avg:.4f} player_r2={r2:.6f} player_corr={corr:.6f}")
PY
```

Expected: both runs print RF average, player R2, and player correlation. The result is descriptive; there is no automatic merge gate on model improvement.

- [ ] **Step 6: Commit verification-only docs if needed**

If README or roadmap needs a result note, commit it:

```bash
git add README.md roadmap.md
git commit -m "docs: record footystats ppg ablation result"
```

If no docs changed, do not create a commit.

---

## Self-Review Checklist

- [ ] Config and CLI cover `footystats_mode`, `footystats_evaluation_scope`, `footystats_league_slug`, `footystats_dir`, `current_year`, and `output_root`.
- [ ] `FEATURE_COLUMNS` remains the base list and does not include FootyStats fields.
- [ ] `feature_columns_for_config` returns PPG fields only when `footystats_mode="ppg"`.
- [ ] `RandomForestPointPredictor` receives explicit feature columns and no longer imports global `FEATURE_COLUMNS`.
- [ ] FootyStats loader validates filename season/league/table type.
- [ ] FootyStats loader rejects missing PPG columns, missing PPG values, invalid `Game Week`, duplicate club rows, bad mapping, and historical non-complete statuses.
- [ ] Post-match outcome columns are not included in the FootyStats feature frame.
- [ ] Feature-frame merges use `many_to_one` and fail when candidate clubs lack target-round FootyStats rows.
- [ ] Existing `footystats_mode="none"` behavior remains neutral and does not require FootyStats files.
- [ ] Existing backtest CLI rejects `live_current` scope in this milestone.
- [ ] `run_metadata.json` records FootyStats mode, scope, source path/hash, feature columns, and join diagnostics.
- [ ] Measurement commands use isolated output roots.
