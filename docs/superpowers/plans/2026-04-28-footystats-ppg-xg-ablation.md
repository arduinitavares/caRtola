# FootyStats PPG + xG Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add leakage-safe FootyStats pre-match xG features and measure their marginal value over the existing PPG feature pack.

**Architecture:** Generalize the existing PPG-specific FootyStats loader and ablation report into mode-aware components, then add `ppg_xg` as an additive mode. The model path remains explicit: `feature_columns_for_config()` resolves the feature list, `run_backtest()` passes the resolved columns into RandomForest, and the ablation runner compares paired control/treatment modes with `fixture_mode="none"`.

**Tech Stack:** Python 3.13, pandas, scikit-learn RandomForest, pytest, uv, existing Cartola backtesting modules.

---

## File Structure

- Modify `src/cartola/backtesting/config.py`
  - Add `ppg_xg` to `FootyStatsMode`.

- Modify `src/cartola/backtesting/features.py`
  - Add `FOOTYSTATS_XG_FEATURE_COLUMNS`.
  - Update `feature_columns_for_config()` for `none`, `ppg`, `ppg_xg`.
  - Switch merge call from PPG-specific naming to generic FootyStats feature naming.

- Modify `src/cartola/backtesting/footystats_features.py`
  - Keep PPG behavior compatible.
  - Add mode-aware safe source-column allowlists.
  - Add `load_footystats_feature_rows(..., footystats_mode=...)`.
  - Keep `load_footystats_ppg_rows()` as a compatibility wrapper.
  - Add xG feature row construction for `ppg_xg`.
  - Add generic merge behavior that merges all known feature columns present in loaded rows.

- Modify `src/cartola/backtesting/runner.py`
  - Resolve FootyStats rows through the generic loader.
  - Record the correct feature columns for `ppg_xg`.

- Modify `src/cartola/backtesting/cli.py`
  - Accept `--footystats-mode ppg_xg`.

- Modify `src/cartola/backtesting/footystats_ablation.py`
  - Preserve current PPG report fixes before generalizing:
    - JSON includes summary/diagnostics paths for control and treatment.
    - Aggregate JSON uses `aggregation_method="unweighted_mean_across_successful_comparable_seasons"`.
  - Rename PPG-specific dataclass/function names to generic FootyStats ablation names.
  - Add `control_footystats_mode` and `treatment_footystats_mode`.
  - Validate control/treatment modes for both CLI and programmatic callers.
  - Emit generic `footystats_ablation.csv` and `footystats_ablation.json`.
  - Keep default comparison as `none -> ppg`.
  - Support xG comparison as `ppg -> ppg_xg`.

- Add `scripts/run_footystats_ablation.py`
  - Generic ablation entrypoint.

- Modify `scripts/run_footystats_ppg_ablation.py`
  - Compatibility wrapper that still calls the generic ablation `main()`.

- Modify tests:
  - `src/tests/backtesting/test_config.py`
  - `src/tests/backtesting/test_features.py`
  - `src/tests/backtesting/test_footystats_features.py`
  - `src/tests/backtesting/test_runner.py`
  - `src/tests/backtesting/test_cli.py`
  - `src/tests/backtesting/test_footystats_ablation.py`

- Modify docs:
  - `README.md`
  - `roadmap.md`

---

### Task 0: Lock In Existing PPG Ablation Review Fixes

**Files:**
- Modify: `src/tests/backtesting/test_footystats_ablation.py`
- Modify only if tests fail: `src/cartola/backtesting/footystats_ablation.py`

This task prevents the generic xG work from regressing the two active PPG ablation report-contract fixes:

- JSON season records include `control_summary_path`, `treatment_summary_path`, `control_diagnostics_path`, and `treatment_diagnostics_path`.
- Aggregate JSON emits `aggregation_method="unweighted_mean_across_successful_comparable_seasons"`.

- [ ] **Step 1: Write or verify regression tests for report paths and aggregation method**

Add this test to `src/tests/backtesting/test_footystats_ablation.py` if equivalent coverage is not already present:

```python
def test_write_reports_includes_run_artifact_paths_and_explicit_aggregation_method(tmp_path: Path) -> None:
    config = ablation.FootyStatsPPGAblationConfig(project_root=tmp_path, force=True)
    resolved_output_root = tmp_path / "data" / "08_reporting" / "backtests" / "footystats_ablation"
    control_output = resolved_output_root / "runs" / "2025" / "footystats_mode=none" / "2025"
    treatment_output = resolved_output_root / "runs" / "2025" / "footystats_mode=ppg" / "2025"
    record = ablation.SeasonAblationRecord(
        season=2025,
        season_status="candidate",
        metrics_comparable=True,
        control_status="ok",
        treatment_status="ok",
        metric_status="ok",
        control_output_path=str(control_output),
        treatment_output_path=str(treatment_output),
        control_summary_path=str(control_output / "summary.csv"),
        treatment_summary_path=str(treatment_output / "summary.csv"),
        control_diagnostics_path=str(control_output / "diagnostics.csv"),
        treatment_diagnostics_path=str(treatment_output / "diagnostics.csv"),
    )
    record.control_baseline_avg_points = 50.0
    record.treatment_baseline_avg_points = 50.0
    record.baseline_avg_points = 50.0
    record.baseline_avg_points_equal = True
    record.control_rf_avg_points = 52.0
    record.treatment_rf_avg_points = 55.0
    record.rf_avg_points_delta = 3.0
    record.control_player_r2 = 0.01
    record.treatment_player_r2 = 0.02
    record.player_r2_delta = 0.01
    record.control_player_corr = 0.10
    record.treatment_player_corr = 0.12
    record.player_corr_delta = 0.02
    record.rf_minus_baseline_control = 2.0
    record.rf_minus_baseline_treatment = 5.0
    aggregate = ablation.build_aggregate_record([record])
    result = ablation.FootyStatsPPGAblationResult(
        config=config,
        resolved_output_root=resolved_output_root,
        seasons=[record],
        aggregate=aggregate,
    )

    ablation.write_reports(result)

    report = json.loads((resolved_output_root / "ppg_ablation.json").read_text())
    season = report["seasons"][0]
    assert season["control_summary_path"] == str(control_output / "summary.csv")
    assert season["treatment_summary_path"] == str(treatment_output / "summary.csv")
    assert season["control_diagnostics_path"] == str(control_output / "diagnostics.csv")
    assert season["treatment_diagnostics_path"] == str(treatment_output / "diagnostics.csv")
    assert report["aggregate"]["aggregation_method"] == "unweighted_mean_across_successful_comparable_seasons"
```

- [ ] **Step 2: Run the regression test**

Run:

```bash
uv run --frozen pytest src/tests/backtesting/test_footystats_ablation.py::test_write_reports_includes_run_artifact_paths_and_explicit_aggregation_method -q
```

Expected: fail if the active review fixes have not been applied yet; otherwise pass. If it fails, fix `SeasonAblationRecord`, `write_reports()`, or `_aggregate_json()` before starting Task 1.

- [ ] **Step 3: Commit only if this task added or fixed test/code**

If files changed:

```bash
git add src/cartola/backtesting/footystats_ablation.py src/tests/backtesting/test_footystats_ablation.py
git commit -m "test: preserve footystats ablation report contract"
```

---

### Task 1: Add `ppg_xg` To Config And Feature Column Resolution

**Files:**
- Modify: `src/cartola/backtesting/config.py`
- Modify: `src/cartola/backtesting/features.py`
- Test: `src/tests/backtesting/test_features.py`
- Test: `src/tests/backtesting/test_config.py`

- [ ] **Step 1: Write failing feature-column tests**

Add to `src/tests/backtesting/test_features.py`:

```python
def test_feature_columns_for_ppg_xg_includes_ppg_then_xg_columns_after_base_columns() -> None:
    columns = feature_columns_for_config(BacktestConfig(footystats_mode="ppg_xg"))

    base_count = len(FEATURE_COLUMNS)
    ppg_count = len(FOOTYSTATS_PPG_FEATURE_COLUMNS)
    assert columns[:base_count] == FEATURE_COLUMNS
    assert columns[base_count : base_count + ppg_count] == FOOTYSTATS_PPG_FEATURE_COLUMNS
    assert columns[base_count + ppg_count :] == FOOTYSTATS_XG_FEATURE_COLUMNS
```

Update the import in the same test file:

```python
from cartola.backtesting.features import (
    FEATURE_COLUMNS,
    FOOTYSTATS_PPG_FEATURE_COLUMNS,
    FOOTYSTATS_XG_FEATURE_COLUMNS,
    build_prediction_frame,
    build_training_frame,
    feature_columns_for_config,
)
```

Add to `src/tests/backtesting/test_config.py`:

```python
def test_backtest_config_accepts_ppg_xg_footystats_mode() -> None:
    config = BacktestConfig(footystats_mode="ppg_xg")

    assert config.footystats_mode == "ppg_xg"
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
uv run --frozen pytest src/tests/backtesting/test_features.py::test_feature_columns_for_ppg_xg_includes_ppg_then_xg_columns_after_base_columns src/tests/backtesting/test_config.py::test_backtest_config_accepts_ppg_xg_footystats_mode -q
```

Expected: failure because `FOOTYSTATS_XG_FEATURE_COLUMNS` and `ppg_xg` support do not exist yet.

- [ ] **Step 3: Implement config and feature-column support**

In `src/cartola/backtesting/config.py`, change:

```python
FootyStatsMode = Literal["none", "ppg"]
```

to:

```python
FootyStatsMode = Literal["none", "ppg", "ppg_xg"]
```

In `src/cartola/backtesting/features.py`, add after `FOOTYSTATS_PPG_FEATURE_COLUMNS`:

```python
FOOTYSTATS_XG_FEATURE_COLUMNS: list[str] = [
    "footystats_team_pre_match_xg",
    "footystats_opponent_pre_match_xg",
    "footystats_xg_diff",
]
```

Update `feature_columns_for_config()`:

```python
def feature_columns_for_config(config: BacktestConfig) -> list[str]:
    if config.footystats_mode == "none":
        return list(FEATURE_COLUMNS)
    if config.footystats_mode == "ppg":
        return [*FEATURE_COLUMNS, *FOOTYSTATS_PPG_FEATURE_COLUMNS]
    if config.footystats_mode == "ppg_xg":
        return [*FEATURE_COLUMNS, *FOOTYSTATS_PPG_FEATURE_COLUMNS, *FOOTYSTATS_XG_FEATURE_COLUMNS]
    raise ValueError(f"Unsupported footystats_mode: {config.footystats_mode!r}")
```

- [ ] **Step 4: Run tests to verify green**

Run:

```bash
uv run --frozen pytest src/tests/backtesting/test_features.py::test_feature_columns_for_ppg_xg_includes_ppg_then_xg_columns_after_base_columns src/tests/backtesting/test_config.py::test_backtest_config_accepts_ppg_xg_footystats_mode -q
```

Expected: both tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/cartola/backtesting/config.py src/cartola/backtesting/features.py src/tests/backtesting/test_features.py src/tests/backtesting/test_config.py
git commit -m "feat: add ppg_xg footystats feature mode"
```

---

### Task 2: Make FootyStats Loader Mode-Aware With Strict Safe Column Allowlists

**Files:**
- Modify: `src/cartola/backtesting/footystats_features.py`
- Test: `src/tests/backtesting/test_footystats_features.py`

- [ ] **Step 1: Confirm selected xG columns are treated as pre-match-safe**

Run:

```bash
uv run --frozen python - <<'PY'
from pathlib import Path
from cartola.backtesting import footystats_audit as audit

path = Path("data/footystats/brazil-serie-a-matches-2025-to-2025-stats.csv")
profile = audit.profile_match_file(path)
required = {"Home Team Pre-Match xG", "Away Team Pre-Match xG"}
print("safe columns contain xG:", sorted(required & set(profile.safe_pre_match_columns)))
print("post-match columns contain xG:", sorted(required & set(profile.post_match_columns)))
if not required.issubset(set(profile.safe_pre_match_columns)):
    raise SystemExit("selected xG columns are not classified as safe pre-match columns")
if required & set(profile.post_match_columns):
    raise SystemExit("selected xG columns are classified as post-match columns")
PY
```

Expected:

```text
safe columns contain xG: ['Away Team Pre-Match xG', 'Home Team Pre-Match xG']
post-match columns contain xG: []
```

If this fails, stop and inspect `src/cartola/backtesting/footystats_audit.py` plus the source CSV headers before adding model features.

- [ ] **Step 2: Write failing loader tests**

Add imports in `src/tests/backtesting/test_footystats_features.py`:

```python
from cartola.backtesting.footystats_features import (
    FOOTYSTATS_XG_SOURCE_COLUMNS,
    PPG_SOURCE_COLUMNS,
    REQUIRED_MATCH_IDENTITY_COLUMNS,
    REQUIRED_MATCH_COLUMNS,
    build_footystats_join_diagnostics,
    load_footystats_feature_rows,
    load_footystats_ppg_rows,
    merge_footystats_ppg,
)
```

Add tests:

```python
def test_load_footystats_feature_rows_ppg_xg_builds_xg_features(tmp_path: Path) -> None:
    _write_matches_csv(tmp_path, [_match_row(week=week, home_xg=1.4, away_xg=0.8) for week in range(1, 39)])
    _write_cartola_round(tmp_path)

    result = load_footystats_feature_rows(
        season=SEASON,
        project_root=tmp_path,
        footystats_dir=Path("data/footystats"),
        league_slug=LEAGUE_SLUG,
        evaluation_scope="historical_candidate",
        current_year=None,
        footystats_mode="ppg_xg",
    )

    home_row = result.rows[(result.rows["rodada"] == 1) & (result.rows["id_clube"] == 262)].iloc[0]
    away_row = result.rows[(result.rows["rodada"] == 1) & (result.rows["id_clube"] == 275)].iloc[0]
    assert home_row["footystats_team_pre_match_xg"] == 1.4
    assert home_row["footystats_opponent_pre_match_xg"] == 0.8
    assert home_row["footystats_xg_diff"] == 0.6
    assert away_row["footystats_team_pre_match_xg"] == 0.8
    assert away_row["footystats_opponent_pre_match_xg"] == 1.4
    assert away_row["footystats_xg_diff"] == -0.6
    assert "team_a_xg" not in result.rows.columns
    assert "team_b_xg" not in result.rows.columns


def test_load_footystats_feature_rows_ppg_xg_reads_only_required_safe_columns(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_path = _write_matches_csv(tmp_path, [_match_row(week=week) for week in range(1, 39)])
    _write_cartola_round(tmp_path)
    original_read_csv = pd.read_csv
    read_calls: list[dict[str, object]] = []

    def capture_read_csv(*args, **kwargs):
        if args and Path(args[0]) == source_path:
            read_calls.append(dict(kwargs))
        return original_read_csv(*args, **kwargs)

    monkeypatch.setattr(pd, "read_csv", capture_read_csv)

    load_footystats_feature_rows(
        season=SEASON,
        project_root=tmp_path,
        footystats_dir=Path("data/footystats"),
        league_slug=LEAGUE_SLUG,
        evaluation_scope="historical_candidate",
        current_year=None,
        footystats_mode="ppg_xg",
    )

    assert read_calls[0]["nrows"] == 0
    assert set(read_calls[1]["usecols"]) == {
        *REQUIRED_MATCH_IDENTITY_COLUMNS,
        *PPG_SOURCE_COLUMNS,
        *FOOTYSTATS_XG_SOURCE_COLUMNS,
    }


def test_load_footystats_feature_rows_ppg_does_not_require_xg_columns(tmp_path: Path) -> None:
    _write_matches_csv(
        tmp_path,
        [_match_row(week=week) for week in range(1, 39)],
        drop_columns=["Home Team Pre-Match xG", "Away Team Pre-Match xG"],
    )
    _write_cartola_round(tmp_path)

    result = load_footystats_feature_rows(
        season=SEASON,
        project_root=tmp_path,
        footystats_dir=Path("data/footystats"),
        league_slug=LEAGUE_SLUG,
        evaluation_scope="historical_candidate",
        current_year=None,
        footystats_mode="ppg",
    )

    assert "footystats_team_pre_match_ppg" in result.rows.columns
    assert "footystats_team_pre_match_xg" not in result.rows.columns


def test_load_footystats_feature_rows_ppg_xg_rejects_missing_xg_column(tmp_path: Path) -> None:
    _write_matches_csv(
        tmp_path,
        [_match_row(week=week) for week in range(1, 39)],
        drop_columns=["Away Team Pre-Match xG"],
    )
    _write_cartola_round(tmp_path)

    with pytest.raises(ValueError, match="Away Team Pre-Match xG"):
        load_footystats_feature_rows(
            season=SEASON,
            project_root=tmp_path,
            footystats_dir=Path("data/footystats"),
            league_slug=LEAGUE_SLUG,
            evaluation_scope="historical_candidate",
            current_year=None,
            footystats_mode="ppg_xg",
        )


def test_load_footystats_feature_rows_ppg_xg_rejects_missing_xg_value(tmp_path: Path) -> None:
    rows = [_match_row(week=week) for week in range(1, 39)]
    rows[0]["Home Team Pre-Match xG"] = None
    _write_matches_csv(tmp_path, rows)
    _write_cartola_round(tmp_path)

    with pytest.raises(ValueError, match="missing or non-numeric xG values.*Home Team Pre-Match xG"):
        load_footystats_feature_rows(
            season=SEASON,
            project_root=tmp_path,
            footystats_dir=Path("data/footystats"),
            league_slug=LEAGUE_SLUG,
            evaluation_scope="historical_candidate",
            current_year=None,
            footystats_mode="ppg_xg",
        )
```

Update `_match_row()` in the same test file:

```python
def _match_row(
    *,
    week: int,
    home: str = "Flamengo",
    away: str = "Palmeiras",
    home_ppg: float = 1.0,
    away_ppg: float = 2.0,
    home_xg: float = 1.1,
    away_xg: float = 0.9,
    status: str = "complete",
) -> dict[str, object]:
    return {
        "Game Week": week,
        "home_team_name": home,
        "away_team_name": away,
        "Pre-Match PPG (Home)": home_ppg,
        "Pre-Match PPG (Away)": away_ppg,
        "Home Team Pre-Match xG": home_xg,
        "Away Team Pre-Match xG": away_xg,
        "status": status,
        "home_team_goal_count": 3,
        "team_a_xg": 2.4,
        "team_b_xg": 0.6,
    }
```

- [ ] **Step 3: Run tests to verify they fail**

Run:

```bash
uv run --frozen pytest src/tests/backtesting/test_footystats_features.py -q
```

Expected: failures because `load_footystats_feature_rows`, xG constants, and xG validation do not exist yet.

- [ ] **Step 4: Implement mode-aware loader constants and result type**

In `src/cartola/backtesting/footystats_features.py`, replace the current source/result constants near the top with:

```python
REQUIRED_MATCH_IDENTITY_COLUMNS: tuple[str, ...] = (
    "Game Week",
    "home_team_name",
    "away_team_name",
    "status",
)
PPG_SOURCE_COLUMNS: tuple[str, ...] = ("Pre-Match PPG (Home)", "Pre-Match PPG (Away)")
FOOTYSTATS_XG_SOURCE_COLUMNS: tuple[str, ...] = ("Home Team Pre-Match xG", "Away Team Pre-Match xG")
REQUIRED_MATCH_COLUMNS: tuple[str, ...] = (*REQUIRED_MATCH_IDENTITY_COLUMNS, *PPG_SOURCE_COLUMNS)
SOURCE_COLUMNS_BY_MODE: dict[str, tuple[str, ...]] = {
    "ppg": REQUIRED_MATCH_COLUMNS,
    "ppg_xg": (*REQUIRED_MATCH_COLUMNS, *FOOTYSTATS_XG_SOURCE_COLUMNS),
}
PPG_FEATURE_COLUMNS: tuple[str, ...] = (
    "footystats_team_pre_match_ppg",
    "footystats_opponent_pre_match_ppg",
    "footystats_ppg_diff",
)
XG_FEATURE_COLUMNS: tuple[str, ...] = (
    "footystats_team_pre_match_xg",
    "footystats_opponent_pre_match_xg",
    "footystats_xg_diff",
)
FEATURE_COLUMNS_BY_MODE: dict[str, tuple[str, ...]] = {
    "ppg": PPG_FEATURE_COLUMNS,
    "ppg_xg": (*PPG_FEATURE_COLUMNS, *XG_FEATURE_COLUMNS),
}
RESULT_COLUMNS_BY_MODE: dict[str, tuple[str, ...]] = {
    "ppg": ("rodada", "id_clube", "opponent_id_clube", "is_home_footystats", *PPG_FEATURE_COLUMNS),
    "ppg_xg": ("rodada", "id_clube", "opponent_id_clube", "is_home_footystats", *PPG_FEATURE_COLUMNS, *XG_FEATURE_COLUMNS),
}
```

Rename `FootyStatsPPGLoadResult` to `FootyStatsFeatureLoadResult` and add a compatibility alias:

```python
@dataclass(frozen=True)
class FootyStatsFeatureLoadResult:
    rows: pd.DataFrame
    source_path: Path
    source_sha256: str
    diagnostics: FootyStatsJoinDiagnostics
    footystats_mode: str
    feature_columns: tuple[str, ...]


FootyStatsPPGLoadResult = FootyStatsFeatureLoadResult
```

- [ ] **Step 5: Implement safe source loading and xG validation**

Add helper functions:

```python
def _source_columns_for_mode(footystats_mode: str) -> tuple[str, ...]:
    if footystats_mode not in SOURCE_COLUMNS_BY_MODE:
        raise ValueError(f"Unsupported footystats_mode: {footystats_mode!r}")
    return SOURCE_COLUMNS_BY_MODE[footystats_mode]


def _feature_columns_for_mode(footystats_mode: str) -> tuple[str, ...]:
    if footystats_mode not in FEATURE_COLUMNS_BY_MODE:
        raise ValueError(f"Unsupported footystats_mode: {footystats_mode!r}")
    return FEATURE_COLUMNS_BY_MODE[footystats_mode]


def _read_source_frame(source_path: Path, required_columns: tuple[str, ...]) -> pd.DataFrame:
    header = pd.read_csv(source_path, nrows=0)
    _require_columns(header, required_columns)
    return pd.read_csv(source_path, usecols=list(required_columns))
```

Change `_require_columns()` signature:

```python
def _require_columns(df: pd.DataFrame, required_columns: tuple[str, ...]) -> None:
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise ValueError(f"FootyStats matches file missing required columns: {', '.join(missing_columns)}")
```

Add xG validator:

```python
def _validated_xg(df: pd.DataFrame, column: str) -> pd.Series:
    values = pd.to_numeric(df[column], errors="coerce")
    if bool(values.isna().any()):
        raise ValueError(f"FootyStats matches file has missing or non-numeric xG values in {column}")
    return values.astype(float)
```

- [ ] **Step 6: Implement `load_footystats_feature_rows()` and keep PPG wrapper**

Replace `load_footystats_ppg_rows()` with a generic implementation and wrapper:

```python
def load_footystats_feature_rows(
    *,
    season: int,
    project_root: Path,
    footystats_dir: Path,
    league_slug: str,
    evaluation_scope: str,
    current_year: int | None,
    footystats_mode: str,
) -> FootyStatsFeatureLoadResult:
    source_path = _source_path(
        project_root=project_root,
        footystats_dir=footystats_dir,
        league_slug=league_slug,
        season=season,
    )
    _validate_source_filename(source_path, season=season, league_slug=league_slug)

    if evaluation_scope not in {"historical_candidate", "live_current"}:
        raise ValueError(f"Unsupported FootyStats evaluation_scope: {evaluation_scope}")
    if evaluation_scope == "live_current":
        resolved_current_year = current_year if current_year is not None else datetime.now(UTC).year
        if season != resolved_current_year:
            raise ValueError(f"live_current requires season {season} to equal current_year {resolved_current_year}")

    source_columns = _source_columns_for_mode(footystats_mode)
    df = _read_source_frame(source_path, source_columns)

    game_weeks = _validated_game_weeks(df)
    home_ppg = _validated_ppg(df, "Pre-Match PPG (Home)")
    away_ppg = _validated_ppg(df, "Pre-Match PPG (Away)")
    home_xg = _validated_xg(df, "Home Team Pre-Match xG") if footystats_mode == "ppg_xg" else None
    away_xg = _validated_xg(df, "Away Team Pre-Match xG") if footystats_mode == "ppg_xg" else None
    statuses = _validated_statuses(df, evaluation_scope)
    _validate_team_names_present(df)

    if evaluation_scope == "historical_candidate":
        if any(status != "complete" for status in statuses):
            raise ValueError("historical_candidate requires all statuses to be complete")
        if sorted(set(game_weeks.astype(int).tolist())) != list(range(1, 39)):
            raise ValueError("historical_candidate requires exact game-week coverage 1..38")

    team_names = _team_names(df)
    comparison = compare_teams_to_cartola(season=season, footystats_team_names=team_names, project_root=project_root)
    _validate_team_mapping(comparison)

    rows = _build_feature_rows(
        df,
        game_weeks,
        home_ppg,
        away_ppg,
        comparison.mapped_teams,
        footystats_mode=footystats_mode,
        home_xg=home_xg,
        away_xg=away_xg,
    )
    _reject_duplicate_join_keys(rows)

    return FootyStatsFeatureLoadResult(
        rows=rows,
        source_path=source_path,
        source_sha256=_sha256_file(source_path),
        diagnostics=FootyStatsJoinDiagnostics(),
        footystats_mode=footystats_mode,
        feature_columns=_feature_columns_for_mode(footystats_mode),
    )


def load_footystats_ppg_rows(
    *,
    season: int,
    project_root: Path,
    footystats_dir: Path,
    league_slug: str,
    evaluation_scope: str,
    current_year: int | None,
) -> FootyStatsFeatureLoadResult:
    return load_footystats_feature_rows(
        season=season,
        project_root=project_root,
        footystats_dir=footystats_dir,
        league_slug=league_slug,
        evaluation_scope=evaluation_scope,
        current_year=current_year,
        footystats_mode="ppg",
    )
```

- [ ] **Step 7: Update row building for optional xG**

Change `_build_feature_rows()` signature:

```python
def _build_feature_rows(
    df: pd.DataFrame,
    game_weeks: pd.Series,
    home_ppg: pd.Series,
    away_ppg: pd.Series,
    mapped_teams: dict[str, int],
    *,
    footystats_mode: str,
    home_xg: pd.Series | None,
    away_xg: pd.Series | None,
) -> pd.DataFrame:
```

Inside the loop, build each record through local dictionaries:

```python
home_record = {
    "rodada": rodada,
    "id_clube": home_id,
    "opponent_id_clube": away_id,
    "is_home_footystats": 1,
    "footystats_team_pre_match_ppg": home_value,
    "footystats_opponent_pre_match_ppg": away_value,
    "footystats_ppg_diff": home_value - away_value,
}
away_record = {
    "rodada": rodada,
    "id_clube": away_id,
    "opponent_id_clube": home_id,
    "is_home_footystats": 0,
    "footystats_team_pre_match_ppg": away_value,
    "footystats_opponent_pre_match_ppg": home_value,
    "footystats_ppg_diff": away_value - home_value,
}
if footystats_mode == "ppg_xg":
    if home_xg is None or away_xg is None:
        raise ValueError("ppg_xg requires pre-match xG values")
    home_xg_value = float(home_xg.loc[index])
    away_xg_value = float(away_xg.loc[index])
    home_record.update(
        {
            "footystats_team_pre_match_xg": home_xg_value,
            "footystats_opponent_pre_match_xg": away_xg_value,
            "footystats_xg_diff": home_xg_value - away_xg_value,
        }
    )
    away_record.update(
        {
            "footystats_team_pre_match_xg": away_xg_value,
            "footystats_opponent_pre_match_xg": home_xg_value,
            "footystats_xg_diff": away_xg_value - home_xg_value,
        }
    )
rows.append(home_record)
rows.append(away_record)
```

Return:

```python
return pd.DataFrame(rows, columns=pd.Index(RESULT_COLUMNS_BY_MODE[footystats_mode]))
```

- [ ] **Step 8: Run focused tests**

Run:

```bash
uv run --frozen pytest src/tests/backtesting/test_footystats_features.py -q
```

Expected: all FootyStats feature tests pass, including existing PPG tests.

- [ ] **Step 9: Commit**

```bash
git add src/cartola/backtesting/footystats_features.py src/tests/backtesting/test_footystats_features.py
git commit -m "feat: load footystats xg features safely"
```

---

### Task 3: Merge Mode-Aware FootyStats Features Into Prediction And Training Frames

**Files:**
- Modify: `src/cartola/backtesting/features.py`
- Modify: `src/cartola/backtesting/footystats_features.py`
- Test: `src/tests/backtesting/test_features.py`

- [ ] **Step 1: Write failing frame merge tests**

In `src/tests/backtesting/test_features.py`, extend `_footystats_rows()` with xG rows by adding these keys to every row:

```python
"footystats_team_pre_match_xg": 1.0,
"footystats_opponent_pre_match_xg": 0.8,
"footystats_xg_diff": 0.2,
```

Use different values for round 3 club `10` and club `20`:

```python
{
    "rodada": 3,
    "id_clube": 10,
    "opponent_id_clube": 20,
    "is_home_footystats": 1,
    "footystats_team_pre_match_ppg": 1.5,
    "footystats_opponent_pre_match_ppg": 1.0,
    "footystats_ppg_diff": 0.5,
    "footystats_team_pre_match_xg": 1.4,
    "footystats_opponent_pre_match_xg": 0.7,
    "footystats_xg_diff": 0.7,
},
{
    "rodada": 3,
    "id_clube": 20,
    "opponent_id_clube": 10,
    "is_home_footystats": 0,
    "footystats_team_pre_match_ppg": 1.0,
    "footystats_opponent_pre_match_ppg": 1.5,
    "footystats_ppg_diff": -0.5,
    "footystats_team_pre_match_xg": 0.7,
    "footystats_opponent_pre_match_xg": 1.4,
    "footystats_xg_diff": -0.7,
},
```

Add test:

```python
def test_prediction_frame_merges_footystats_xg_rows_when_present() -> None:
    frame = build_prediction_frame(_season_df(), target_round=3, footystats_rows=_footystats_rows())

    club_10_player = frame.loc[frame["id_clube"] == 10].iloc[0]
    club_20_player = frame.loc[frame["id_clube"] == 20].iloc[0]
    assert club_10_player["footystats_team_pre_match_xg"] == 1.4
    assert club_10_player["footystats_opponent_pre_match_xg"] == 0.7
    assert club_10_player["footystats_xg_diff"] == 0.7
    assert club_20_player["footystats_team_pre_match_xg"] == 0.7
    assert club_20_player["footystats_opponent_pre_match_xg"] == 1.4
    assert club_20_player["footystats_xg_diff"] == -0.7
    assert "opponent_id_clube" not in frame.columns
```

Add test:

```python
def test_training_frame_merges_footystats_xg_rows_for_historical_rounds() -> None:
    frame = build_training_frame(_season_df(), target_round=4, footystats_rows=_footystats_rows())

    round_3 = frame[frame["rodada"] == 3].sort_values("id_clube").reset_index(drop=True)
    assert round_3["footystats_team_pre_match_xg"].tolist() == [1.4, 0.7]
    assert round_3["footystats_opponent_pre_match_xg"].tolist() == [0.7, 1.4]
    assert round_3["footystats_xg_diff"].tolist() == [0.7, -0.7]
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
uv run --frozen pytest src/tests/backtesting/test_features.py::test_prediction_frame_merges_footystats_xg_rows_when_present src/tests/backtesting/test_features.py::test_training_frame_merges_footystats_xg_rows_for_historical_rounds -q
```

Expected: failure because the merge only carries PPG columns.

- [ ] **Step 3: Implement generic feature merge**

In `src/cartola/backtesting/footystats_features.py`, add:

```python
ALL_FEATURE_COLUMNS: tuple[str, ...] = (*PPG_FEATURE_COLUMNS, *XG_FEATURE_COLUMNS)
```

Rename `merge_footystats_ppg()` to:

```python
def merge_footystats_features(
    frame: pd.DataFrame,
    footystats_rows: pd.DataFrame | None,
    *,
    target_round: int,
) -> pd.DataFrame:
```

Inside it, replace:

```python
feature_rows = round_rows[["id_clube", *PPG_FEATURE_COLUMNS]]
```

with:

```python
feature_columns = [column for column in ALL_FEATURE_COLUMNS if column in round_rows.columns]
feature_rows = round_rows[["id_clube", *feature_columns]]
```

Keep compatibility wrapper:

```python
def merge_footystats_ppg(
    frame: pd.DataFrame,
    footystats_rows: pd.DataFrame | None,
    *,
    target_round: int,
) -> pd.DataFrame:
    return merge_footystats_features(frame, footystats_rows, target_round=target_round)
```

In `src/cartola/backtesting/features.py`, change import:

```python
from cartola.backtesting.footystats_features import merge_footystats_features
```

and change:

```python
return merge_footystats_ppg(frame, footystats_rows, target_round=target_round)
```

to:

```python
return merge_footystats_features(frame, footystats_rows, target_round=target_round)
```

- [ ] **Step 4: Run focused tests**

Run:

```bash
uv run --frozen pytest src/tests/backtesting/test_features.py src/tests/backtesting/test_footystats_features.py -q
```

Expected: both files pass.

- [ ] **Step 5: Commit**

```bash
git add src/cartola/backtesting/features.py src/cartola/backtesting/footystats_features.py src/tests/backtesting/test_features.py
git commit -m "feat: merge footystats xg features into frames"
```

---

### Task 4: Wire `ppg_xg` Through Runner Metadata And CLI

**Files:**
- Modify: `src/cartola/backtesting/runner.py`
- Modify: `src/cartola/backtesting/cli.py`
- Test: `src/tests/backtesting/test_runner.py`
- Test: `src/tests/backtesting/test_cli.py`

- [ ] **Step 1: Write failing runner and CLI tests**

In `src/tests/backtesting/test_runner.py`, add:

```python
def test_run_backtest_ppg_xg_passes_mode_and_records_feature_columns(tmp_path, monkeypatch):
    source_path = tmp_path / "data/footystats/brazil-serie-a-matches-2025-to-2025-stats.csv"
    rows = _tiny_footystats_rows(range(1, 6))
    rows["footystats_team_pre_match_xg"] = 1.2
    rows["footystats_opponent_pre_match_xg"] = 0.8
    rows["footystats_xg_diff"] = 0.4
    calls: list[dict[str, object]] = []

    def fake_load_footystats_feature_rows(**kwargs: object) -> FootyStatsPPGLoadResult:
        calls.append(kwargs)
        return FootyStatsPPGLoadResult(
            rows=rows,
            source_path=source_path,
            source_sha256="fake-sha",
            diagnostics=FootyStatsJoinDiagnostics(),
            footystats_mode="ppg_xg",
            feature_columns=(
                "footystats_team_pre_match_ppg",
                "footystats_opponent_pre_match_ppg",
                "footystats_ppg_diff",
                "footystats_team_pre_match_xg",
                "footystats_opponent_pre_match_xg",
                "footystats_xg_diff",
            ),
        )

    monkeypatch.setattr(
        "cartola.backtesting.runner.load_footystats_feature_rows",
        fake_load_footystats_feature_rows,
    )

    config = BacktestConfig(
        project_root=tmp_path,
        start_round=5,
        budget=100,
        footystats_mode="ppg_xg",
    )
    result = run_backtest(config)

    assert calls[0]["footystats_mode"] == "ppg_xg"
    assert result.metadata.footystats_feature_columns == [
        "footystats_team_pre_match_ppg",
        "footystats_opponent_pre_match_ppg",
        "footystats_ppg_diff",
        "footystats_team_pre_match_xg",
        "footystats_opponent_pre_match_xg",
        "footystats_xg_diff",
    ]
    assert "footystats_xg_diff" in result.player_predictions.columns
```

In `src/tests/backtesting/test_cli.py`, add:

```python
def test_parse_args_accepts_ppg_xg_footystats_mode() -> None:
    args = parse_args(["--footystats-mode", "ppg_xg"])

    assert args.footystats_mode == "ppg_xg"
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
uv run --frozen pytest src/tests/backtesting/test_runner.py::test_run_backtest_ppg_xg_passes_mode_and_records_feature_columns src/tests/backtesting/test_cli.py::test_parse_args_accepts_ppg_xg_footystats_mode -q
```

Expected: failure because runner still calls `load_footystats_ppg_rows()` and CLI choices exclude `ppg_xg`.

- [ ] **Step 3: Update runner imports and resolver**

In `src/cartola/backtesting/runner.py`, change imports:

```python
from cartola.backtesting.footystats_features import (
    FootyStatsFeatureLoadResult,
    FootyStatsJoinDiagnostics,
    build_footystats_join_diagnostics,
    load_footystats_feature_rows,
)
```

Update type annotation:

```python
def _resolve_footystats(config: BacktestConfig) -> FootyStatsFeatureLoadResult | None:
```

Change the loader call:

```python
return load_footystats_feature_rows(
    season=config.season,
    project_root=config.project_root,
    footystats_dir=config.footystats_dir,
    league_slug=config.footystats_league_slug,
    evaluation_scope=config.footystats_evaluation_scope,
    current_year=config.current_year,
    footystats_mode=config.footystats_mode,
)
```

Change metadata feature-column assignment:

```python
footystats_feature_columns=(
    list(resolved_footystats.feature_columns) if resolved_footystats is not None else []
),
```

- [ ] **Step 4: Update CLI choices**

In `src/cartola/backtesting/cli.py`, change:

```python
parser.add_argument("--footystats-mode", choices=("none", "ppg"), default="none")
```

to:

```python
parser.add_argument("--footystats-mode", choices=("none", "ppg", "ppg_xg"), default="none")
```

- [ ] **Step 5: Run focused tests**

Run:

```bash
uv run --frozen pytest src/tests/backtesting/test_runner.py src/tests/backtesting/test_cli.py -q
```

Expected: tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/cartola/backtesting/runner.py src/cartola/backtesting/cli.py src/tests/backtesting/test_runner.py src/tests/backtesting/test_cli.py
git commit -m "feat: wire ppg_xg through runner and cli"
```

---

### Task 5: Generalize FootyStats Ablation Runner And Report Schema

**Files:**
- Modify: `src/cartola/backtesting/footystats_ablation.py`
- Add: `scripts/run_footystats_ablation.py`
- Modify: `scripts/run_footystats_ppg_ablation.py`
- Test: `src/tests/backtesting/test_footystats_ablation.py`

- [ ] **Step 1: Write failing ablation config and script tests**

In `src/tests/backtesting/test_footystats_ablation.py`, update default config test expectations:

```python
def test_config_from_default_args() -> None:
    config = ablation.config_from_args(ablation.parse_args([]))

    assert config.seasons == (2023, 2024, 2025)
    assert config.start_round == 5
    assert config.budget == 100.0
    assert config.project_root == Path(".")
    assert config.output_root == Path("data/08_reporting/backtests/footystats_ablation")
    assert config.footystats_league_slug == "brazil-serie-a"
    assert config.control_footystats_mode == "none"
    assert config.treatment_footystats_mode == "ppg"
    assert config.force is False
```

Add:

```python
def test_parse_args_accepts_control_and_treatment_modes() -> None:
    config = ablation.config_from_args(
        ablation.parse_args(
            [
                "--control-footystats-mode",
                "ppg",
                "--treatment-footystats-mode",
                "ppg_xg",
                "--output-root",
                "data/08_reporting/backtests/footystats_xg_ablation",
            ]
        )
    )

    assert config.control_footystats_mode == "ppg"
    assert config.treatment_footystats_mode == "ppg_xg"
    assert config.output_root == Path("data/08_reporting/backtests/footystats_xg_ablation")
```

Add:

```python
def test_resolve_output_root_rejects_ablation_name_outside_backtests_tree(tmp_path: Path) -> None:
    config = ablation.FootyStatsAblationConfig(
        project_root=tmp_path,
        output_root=Path("src/footystats_xg_ablation"),
        force=True,
    )

    with pytest.raises(ValueError, match="data/08_reporting/backtests"):
        ablation.resolve_output_root(config)
```

Add:

```python
def test_generic_script_imports_main_from_footystats_ablation() -> None:
    script_path = Path(__file__).resolve().parents[3] / "scripts" / "run_footystats_ablation.py"
    spec = importlib.util.spec_from_file_location("run_footystats_ablation", script_path)
    assert spec is not None
    assert spec.loader is not None
    script = importlib.util.module_from_spec(spec)

    spec.loader.exec_module(script)

    assert script.main is ablation.main
```

Add:

```python
def test_parse_args_rejects_same_control_and_treatment_modes() -> None:
    with pytest.raises(SystemExit):
        ablation.parse_args(["--control-footystats-mode", "ppg", "--treatment-footystats-mode", "ppg"])
```

Add:

```python
def test_run_footystats_ablation_rejects_same_control_and_treatment_modes(tmp_path: Path) -> None:
    config = ablation.FootyStatsAblationConfig(
        project_root=tmp_path,
        control_footystats_mode="ppg",
        treatment_footystats_mode="ppg",
        force=True,
    )

    with pytest.raises(ValueError, match="control and treatment FootyStats modes must differ"):
        ablation.run_footystats_ablation(config)
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
uv run --frozen pytest src/tests/backtesting/test_footystats_ablation.py::test_parse_args_accepts_control_and_treatment_modes src/tests/backtesting/test_footystats_ablation.py::test_generic_script_imports_main_from_footystats_ablation -q
```

Expected: failure because the new args and script do not exist.

- [ ] **Step 3: Rename dataclasses to generic names with compatibility aliases**

In `src/cartola/backtesting/footystats_ablation.py`, rename:

```python
FootyStatsPPGAblationConfig -> FootyStatsAblationConfig
FootyStatsPPGAblationResult -> FootyStatsAblationResult
run_footystats_ppg_ablation -> run_footystats_ablation
```

Add compatibility aliases after the dataclass definitions:

```python
FootyStatsPPGAblationConfig = FootyStatsAblationConfig
FootyStatsPPGAblationResult = FootyStatsAblationResult
```

Add compatibility wrapper near the runner:

```python
def run_footystats_ppg_ablation(config: FootyStatsAblationConfig) -> FootyStatsAblationResult:
    return run_footystats_ablation(config)
```

Compatibility rule: `scripts/run_footystats_ppg_ablation.py` preserves invocation compatibility only. Report filenames are intentionally generic after Task 7 (`footystats_ablation.csv/json`) even when the wrapper is used.

- [ ] **Step 4: Add control/treatment modes to config and parser**

Update dataclass:

```python
@dataclass(frozen=True)
class FootyStatsAblationConfig:
    seasons: tuple[int, ...] = DEFAULT_SEASONS
    start_round: int = 5
    budget: float = 100.0
    project_root: Path = Path(".")
    output_root: Path = DEFAULT_OUTPUT_ROOT
    footystats_league_slug: str = DEFAULT_LEAGUE_SLUG
    control_footystats_mode: FootyStatsMode = "none"
    treatment_footystats_mode: FootyStatsMode = "ppg"
    current_year: int | None = None
    force: bool = False
```

Update parser:

```python
parser.add_argument("--control-footystats-mode", choices=("none", "ppg", "ppg_xg"), default="none")
parser.add_argument("--treatment-footystats-mode", choices=("none", "ppg", "ppg_xg"), default="ppg")
```

After `parser.parse_args(argv)`, validate modes:

```python
args = parser.parse_args(argv)
if args.control_footystats_mode == args.treatment_footystats_mode:
    parser.error("control and treatment FootyStats modes must differ")
return args
```

Update `config_from_args()` to populate both modes.

Add shared runtime validation:

```python
def validate_ablation_config(config: FootyStatsAblationConfig) -> None:
    if config.control_footystats_mode == config.treatment_footystats_mode:
        raise ValueError("control and treatment FootyStats modes must differ")
```

Call it at the top of `run_footystats_ablation()` before resolving or preparing output roots:

```python
def run_footystats_ablation(config: FootyStatsAblationConfig) -> FootyStatsAblationResult:
    validate_ablation_config(config)
    resolved_output_root = resolve_output_root(config)
    ...
```

- [ ] **Step 5: Generalize output-root validation without weakening deletion safety**

Keep these existing constraints:

- relative `output_root` resolves under `project_root`;
- absolute `output_root` must also resolve under `project_root`;
- protected paths such as the project root, `data`, `data/08_reporting`, `data/08_reporting/backtests`, and normal season output directories are rejected.

Add a required parent-directory constraint:

```python
backtests_root = (project_root / "data" / "08_reporting" / "backtests").resolve()
if not _is_relative_to(resolved_output_root, backtests_root):
    raise ValueError(f"output_root must resolve inside {backtests_root}")
```

Replace:

```python
if resolved_output_root.name != "footystats_ablation":
    raise ValueError("output_root final directory name must be exactly 'footystats_ablation'")
```

with:

```python
if not (resolved_output_root.name.startswith("footystats") and resolved_output_root.name.endswith("ablation")):
    raise ValueError("output_root final directory name must start with 'footystats' and end with 'ablation'")
```

This allows report roots such as `data/08_reporting/backtests/footystats_ablation` and `data/08_reporting/backtests/footystats_xg_ablation`, but rejects destructive paths such as `src/footystats_xg_ablation --force`.

Update existing test `test_resolve_output_root_requires_footystats_ablation_directory_name` expected message to:

```python
with pytest.raises(ValueError, match="start with 'footystats' and end with 'ablation'"):
```

Add:

```python
def test_resolve_output_root_allows_treatment_specific_ablation_directory(tmp_path: Path) -> None:
    config = ablation.FootyStatsAblationConfig(
        project_root=tmp_path,
        output_root=Path("data/08_reporting/backtests/footystats_xg_ablation"),
    )

    assert ablation.resolve_output_root(config) == (
        tmp_path / "data" / "08_reporting" / "backtests" / "footystats_xg_ablation"
    ).resolve()
```

Update every existing ablation test that currently passes `output_root=Path("reports/footystats_ablation")` or uses `tmp_path / "reports" / "footystats_ablation"` so it uses the reporting subtree instead:

```python
output_root=Path("data/08_reporting/backtests/footystats_ablation")
```

and:

```python
output_root = tmp_path / "data" / "08_reporting" / "backtests" / "footystats_ablation"
```

Tests that intentionally validate rejected paths should keep invalid roots such as `../footystats_ablation`, `/tmp/footystats_ablation`, and `src/footystats_xg_ablation`.

- [ ] **Step 6: Add generic script**

Create `scripts/run_footystats_ablation.py`:

```python
from __future__ import annotations

from cartola.backtesting.footystats_ablation import main

if __name__ == "__main__":
    raise SystemExit(main())
```

Keep `scripts/run_footystats_ppg_ablation.py` as:

```python
from __future__ import annotations

from cartola.backtesting.footystats_ablation import main

if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 7: Run focused tests**

Run:

```bash
uv run --frozen pytest src/tests/backtesting/test_footystats_ablation.py -q
```

Expected: remaining failures will point to run orchestration still being hardcoded to `none -> ppg`; fix in Task 6.

- [ ] **Step 8: Commit**

```bash
git add src/cartola/backtesting/footystats_ablation.py scripts/run_footystats_ablation.py scripts/run_footystats_ppg_ablation.py src/tests/backtesting/test_footystats_ablation.py
git commit -m "feat: generalize footystats ablation config"
```

---

### Task 6: Make Ablation Eligibility, Backtest Configs, And Metadata Mode-Aware

**Files:**
- Modify: `src/cartola/backtesting/footystats_ablation.py`
- Test: `src/tests/backtesting/test_footystats_ablation.py`

- [ ] **Step 1: Write failing orchestration tests**

Add test:

```python
def test_build_backtest_config_supports_ppg_xg_mode(tmp_path: Path) -> None:
    resolved_output_root = tmp_path / "data" / "08_reporting" / "backtests" / "footystats_xg_ablation"
    config = ablation.FootyStatsAblationConfig(
        project_root=tmp_path,
        seasons=(2025,),
        control_footystats_mode="ppg",
        treatment_footystats_mode="ppg_xg",
    )

    treatment = ablation.build_backtest_config(config, 2025, "ppg_xg", resolved_output_root)

    assert treatment.output_root == resolved_output_root / "runs" / "2025" / "footystats_mode=ppg_xg"
    assert treatment.output_path == resolved_output_root / "runs" / "2025" / "footystats_mode=ppg_xg" / "2025"
    assert treatment.fixture_mode == "none"
    assert treatment.footystats_mode == "ppg_xg"
```

Add test:

```python
def test_ablation_validates_control_and_treatment_sources_for_ppg_to_ppg_xg(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loaded_modes: list[str] = []
    run_modes: list[str] = []

    def fake_load_feature_rows(**kwargs: object):
        mode = str(kwargs["footystats_mode"])
        loaded_modes.append(mode)
        return SimpleNamespace(
            rows=pd.DataFrame(),
            source_path=tmp_path / f"{mode}.csv",
            source_sha256=f"{mode}-sha",
            diagnostics=SimpleNamespace(),
            footystats_mode=mode,
            feature_columns=(),
        )

    def fake_run_backtest(config):
        run_modes.append(config.footystats_mode)
        _write_backtest_outputs(config.output_path)

    monkeypatch.setattr(ablation, "load_footystats_feature_rows", fake_load_feature_rows)
    monkeypatch.setattr(ablation, "run_backtest", fake_run_backtest)

    config = ablation.FootyStatsAblationConfig(
        project_root=tmp_path,
        output_root=Path("data/08_reporting/backtests/footystats_xg_ablation"),
        seasons=(2025,),
        control_footystats_mode="ppg",
        treatment_footystats_mode="ppg_xg",
        force=True,
    )

    result = ablation.run_footystats_ablation(config)

    assert loaded_modes == ["ppg", "ppg_xg"]
    assert run_modes == ["ppg", "ppg_xg"]
    record = result.seasons[0]
    assert record.control_source_path == str(tmp_path / "ppg.csv")
    assert record.control_source_sha256 == "ppg-sha"
    assert record.treatment_source_path == str(tmp_path / "ppg_xg.csv")
    assert record.treatment_source_sha256 == "ppg_xg-sha"
    assert record.metrics_comparable is True
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
uv run --frozen pytest src/tests/backtesting/test_footystats_ablation.py::test_build_backtest_config_supports_ppg_xg_mode src/tests/backtesting/test_footystats_ablation.py::test_ablation_validates_control_and_treatment_sources_for_ppg_to_ppg_xg -q
```

Expected: failure because `build_backtest_config()` only supports `none|ppg` and eligibility only validates treatment PPG.

- [ ] **Step 3: Update `SeasonAblationRecord` source metadata**

Add fields:

```python
control_source_path: str | None = None
control_source_sha256: str | None = None
treatment_source_path: str | None = None
treatment_source_sha256: str | None = None
```

Keep the existing treatment fields if they already exist, and add control fields before them.

- [ ] **Step 4: Update `build_backtest_config()`**

Change mode validation:

```python
if mode not in ("none", "ppg", "ppg_xg"):
    raise ValueError(f"Unsupported footystats mode: {mode!r}")
```

The output path line stays:

```python
output_root=resolved_output_root / "runs" / str(season) / f"footystats_mode={footystats_mode}",
```

- [ ] **Step 5: Replace `_load_eligibility()` with mode-aware source validation**

Replace:

```python
def _load_eligibility(config: FootyStatsPPGAblationConfig, season: int) -> FootyStatsPPGLoadResult:
```

with:

```python
def _load_eligibility(
    config: FootyStatsAblationConfig,
    season: int,
    mode: str,
) -> object | None:
    if mode == "none":
        return None
    return load_footystats_feature_rows(
        season=season,
        project_root=config.project_root.resolve(),
        footystats_dir=Path("data/footystats"),
        league_slug=config.footystats_league_slug,
        evaluation_scope="historical_candidate",
        current_year=config.resolved_current_year,
        footystats_mode=mode,
    )
```

Import the generic loader:

```python
from cartola.backtesting.footystats_features import load_footystats_feature_rows
```

- [ ] **Step 6: Update run loop**

In `run_footystats_ablation()`, build configs with configured modes:

```python
control_config = build_backtest_config(
    config,
    season=season,
    mode=config.control_footystats_mode,
    resolved_output_root=resolved_output_root,
)
treatment_config = build_backtest_config(
    config,
    season=season,
    mode=config.treatment_footystats_mode,
    resolved_output_root=resolved_output_root,
)
```

Replace eligibility block with side-specific validation so the report states whether control or treatment eligibility failed:

```python
try:
    control_eligibility = _load_eligibility(config, season, config.control_footystats_mode)
except Exception as exc:
    record.error_stage = "control_eligibility"
    record.error_message = str(exc)
    record.errors.append(_error("control_eligibility", exc))
    records.append(record)
    continue

try:
    treatment_eligibility = _load_eligibility(config, season, config.treatment_footystats_mode)
except Exception as exc:
    record.error_stage = "treatment_eligibility"
    record.error_message = str(exc)
    record.errors.append(_error("treatment_eligibility", exc))
    records.append(record)
    continue

record.season_status = "candidate"
if control_eligibility is not None:
    record.control_source_path = str(control_eligibility.source_path)
    record.control_source_sha256 = control_eligibility.source_sha256
if treatment_eligibility is not None:
    record.treatment_source_path = str(treatment_eligibility.source_path)
    record.treatment_source_sha256 = treatment_eligibility.source_sha256
```

- [ ] **Step 7: Run focused tests**

Run:

```bash
uv run --frozen pytest src/tests/backtesting/test_footystats_ablation.py -q
```

Expected: all ablation tests pass or fail only on report filename expectations fixed in Task 7.

- [ ] **Step 8: Commit**

```bash
git add src/cartola/backtesting/footystats_ablation.py src/tests/backtesting/test_footystats_ablation.py
git commit -m "feat: support mode-aware footystats ablations"
```

---

### Task 7: Emit Generic Report Filenames And Mode Metadata

**Files:**
- Modify: `src/cartola/backtesting/footystats_ablation.py`
- Test: `src/tests/backtesting/test_footystats_ablation.py`

- [ ] **Step 1: Write failing report schema tests**

Update or add:

```python
def test_run_footystats_ablation_writes_generic_csv_and_json(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_load_feature_rows(**kwargs: object):
        mode = str(kwargs["footystats_mode"])
        return SimpleNamespace(
            rows=pd.DataFrame(),
            source_path=tmp_path / f"{mode}.csv",
            source_sha256=f"{mode}-sha",
            diagnostics=SimpleNamespace(),
            footystats_mode=mode,
            feature_columns=(),
        )

    def fake_run_backtest(config):
        _write_backtest_outputs(config.output_path)

    monkeypatch.setattr(ablation, "load_footystats_feature_rows", fake_load_feature_rows)
    monkeypatch.setattr(ablation, "run_backtest", fake_run_backtest)

    config = ablation.FootyStatsAblationConfig(
        project_root=tmp_path,
        output_root=Path("data/08_reporting/backtests/footystats_xg_ablation"),
        seasons=(2025,),
        control_footystats_mode="ppg",
        treatment_footystats_mode="ppg_xg",
        force=True,
    )
    result = ablation.run_footystats_ablation(config)
    ablation.write_reports(result)

    output_root = tmp_path / "data" / "08_reporting" / "backtests" / "footystats_xg_ablation"
    assert (output_root / "footystats_ablation.csv").exists()
    assert (output_root / "footystats_ablation.json").exists()
    assert not (output_root / "ppg_ablation.csv").exists()
    assert not (output_root / "ppg_ablation.json").exists()

    report = json.loads((output_root / "footystats_ablation.json").read_text())
    assert report["config"]["control_footystats_mode"] == "ppg"
    assert report["config"]["treatment_footystats_mode"] == "ppg_xg"
    assert report["seasons"][0]["control_source_path"] == str(tmp_path / "ppg.csv")
    assert report["seasons"][0]["treatment_source_path"] == str(tmp_path / "ppg_xg.csv")
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
uv run --frozen pytest src/tests/backtesting/test_footystats_ablation.py::test_run_footystats_ablation_writes_generic_csv_and_json -q
```

Expected: failure because reports are still named `ppg_ablation.*`.

- [ ] **Step 3: Add mode and source columns to CSV schema**

In `CSV_COLUMNS`, add after `metric_status`:

```python
"control_footystats_mode",
"treatment_footystats_mode",
```

Add after diagnostics paths or output paths:

```python
"control_source_path",
"control_source_sha256",
"treatment_source_path",
"treatment_source_sha256",
```

Add fields to `SeasonAblationRecord`:

```python
control_footystats_mode: str | None = None
treatment_footystats_mode: str | None = None
```

When creating each record:

```python
control_footystats_mode=config.control_footystats_mode,
treatment_footystats_mode=config.treatment_footystats_mode,
```

For aggregate record:

```python
control_footystats_mode=None,
treatment_footystats_mode=None,
```

- [ ] **Step 4: Update `_config_json()`**

Change hardcoded values:

```python
"control_footystats_mode": "none",
"treatment_footystats_mode": "ppg",
```

to:

```python
"control_footystats_mode": config.control_footystats_mode,
"treatment_footystats_mode": config.treatment_footystats_mode,
```

- [ ] **Step 5: Rename report files**

In `write_reports()`, replace:

```python
csv_path = result.resolved_output_root / "ppg_ablation.csv"
json_path = result.resolved_output_root / "ppg_ablation.json"
csv_tmp = result.resolved_output_root / ".ppg_ablation.csv.tmp"
json_tmp = result.resolved_output_root / ".ppg_ablation.json.tmp"
csv_backup = result.resolved_output_root / ".ppg_ablation.csv.bak"
json_backup = result.resolved_output_root / ".ppg_ablation.json.bak"
```

with:

```python
csv_path = result.resolved_output_root / "footystats_ablation.csv"
json_path = result.resolved_output_root / "footystats_ablation.json"
csv_tmp = result.resolved_output_root / ".footystats_ablation.csv.tmp"
json_tmp = result.resolved_output_root / ".footystats_ablation.json.tmp"
csv_backup = result.resolved_output_root / ".footystats_ablation.csv.bak"
json_backup = result.resolved_output_root / ".footystats_ablation.json.bak"
```

- [ ] **Step 6: Run ablation tests**

Run:

```bash
uv run --frozen pytest src/tests/backtesting/test_footystats_ablation.py -q
```

Expected: all tests pass.

- [ ] **Step 7: Commit**

```bash
git add src/cartola/backtesting/footystats_ablation.py src/tests/backtesting/test_footystats_ablation.py
git commit -m "feat: emit generic footystats ablation reports"
```

---

### Task 8: Add xG Ablation Measurement And Decision Metadata

**Files:**
- Modify: `src/cartola/backtesting/footystats_ablation.py`
- Test: `src/tests/backtesting/test_footystats_ablation.py`

- [ ] **Step 1: Write failing success-evaluation tests**

Add helper:

```python
def _record_for_decision(season: int, delta: float, r2_delta: float = 0.0, corr_delta: float = 0.0):
    record = ablation.SeasonAblationRecord(
        season=season,
        season_status="candidate",
        metrics_comparable=True,
        control_status="ok",
        treatment_status="ok",
        metric_status="ok",
    )
    record.control_baseline_avg_points = 50.0
    record.treatment_baseline_avg_points = 50.0
    record.baseline_avg_points = 50.0
    record.baseline_avg_points_equal = True
    record.control_rf_avg_points = 55.0
    record.treatment_rf_avg_points = 55.0 + delta
    record.rf_avg_points_delta = delta
    record.control_player_r2 = 0.01
    record.treatment_player_r2 = 0.01 + r2_delta
    record.player_r2_delta = r2_delta
    record.control_player_corr = 0.20
    record.treatment_player_corr = 0.20 + corr_delta
    record.player_corr_delta = corr_delta
    record.rf_minus_baseline_control = 5.0
    record.rf_minus_baseline_treatment = 5.0 + delta
    return record
```

Add:

```python
def test_ablation_decision_accepts_positive_xg_over_successful_comparable_seasons() -> None:
    records = [
        _record_for_decision(2023, 1.0, 0.01, 0.01),
        _record_for_decision(2024, 0.5, 0.0, 0.0),
        _record_for_decision(2025, -0.2, 0.0, 0.0),
    ]
    aggregate = ablation.build_aggregate_record(records)

    decision = ablation.build_ablation_decision(records, aggregate)

    assert decision == {
        "minimum_successful_comparable_seasons": 2,
        "successful_comparable_season_count": 3,
        "positive_rf_delta_season_count": 2,
        "aggregate_rf_avg_points_delta_positive": True,
        "aggregate_player_r2_delta_non_negative": True,
        "aggregate_player_corr_delta_non_negative": True,
        "keep_treatment": True,
    }


def test_ablation_decision_rejects_when_fewer_than_two_comparable_seasons_succeed() -> None:
    records = [_record_for_decision(2025, 2.0, 0.1, 0.1)]
    aggregate = ablation.build_aggregate_record(records)

    decision = ablation.build_ablation_decision(records, aggregate)

    assert decision["successful_comparable_season_count"] == 1
    assert decision["keep_treatment"] is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
uv run --frozen pytest src/tests/backtesting/test_footystats_ablation.py::test_ablation_decision_accepts_positive_xg_over_successful_comparable_seasons src/tests/backtesting/test_footystats_ablation.py::test_ablation_decision_rejects_when_fewer_than_two_comparable_seasons_succeed -q
```

Expected: failure because `build_ablation_decision()` does not exist.

- [ ] **Step 3: Implement decision helper**

Add:

```python
def build_ablation_decision(
    records: list[SeasonAblationRecord],
    aggregate: SeasonAblationRecord,
) -> dict[str, object]:
    included = [record for record in records if _is_included_in_aggregate(record)]
    positive_rf_delta_count = sum(
        1 for record in included if record.rf_avg_points_delta is not None and record.rf_avg_points_delta > 0
    )
    aggregate_rf_positive = aggregate.rf_avg_points_delta is not None and aggregate.rf_avg_points_delta > 0
    aggregate_r2_non_negative = aggregate.player_r2_delta is not None and aggregate.player_r2_delta >= 0
    aggregate_corr_non_negative = aggregate.player_corr_delta is not None and aggregate.player_corr_delta >= 0
    keep_treatment = (
        len(included) >= 2
        and positive_rf_delta_count >= 2
        and aggregate_rf_positive
        and aggregate_r2_non_negative
        and aggregate_corr_non_negative
    )
    return {
        "minimum_successful_comparable_seasons": 2,
        "successful_comparable_season_count": len(included),
        "positive_rf_delta_season_count": positive_rf_delta_count,
        "aggregate_rf_avg_points_delta_positive": aggregate_rf_positive,
        "aggregate_player_r2_delta_non_negative": aggregate_r2_non_negative,
        "aggregate_player_corr_delta_non_negative": aggregate_corr_non_negative,
        "keep_treatment": keep_treatment,
    }
```

In `write_reports()`, add to `report`:

```python
"decision": build_ablation_decision(result.seasons, result.aggregate),
```

- [ ] **Step 4: Run tests**

Run:

```bash
uv run --frozen pytest src/tests/backtesting/test_footystats_ablation.py -q
```

Expected: all ablation tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/cartola/backtesting/footystats_ablation.py src/tests/backtesting/test_footystats_ablation.py
git commit -m "feat: add footystats ablation decision metadata"
```

---

### Task 9: Update Documentation And Roadmap Commands

**Files:**
- Modify: `README.md`
- Modify: `roadmap.md`

- [ ] **Step 1: Update README commands**

Replace the multi-season PPG command with generic PPG command:

```bash
uv run --frozen python scripts/run_footystats_ablation.py \
  --seasons 2023,2024,2025 \
  --start-round 5 \
  --budget 100 \
  --current-year 2026 \
  --control-footystats-mode none \
  --treatment-footystats-mode ppg \
  --output-root data/08_reporting/backtests/footystats_ablation \
  --force
```

Add xG-over-PPG command:

```bash
uv run --frozen python scripts/run_footystats_ablation.py \
  --seasons 2023,2024,2025 \
  --start-round 5 \
  --budget 100 \
  --current-year 2026 \
  --control-footystats-mode ppg \
  --treatment-footystats-mode ppg_xg \
  --output-root data/08_reporting/backtests/footystats_xg_ablation \
  --force
```

Update report file list:

```text
data/08_reporting/backtests/footystats_ablation/footystats_ablation.csv
data/08_reporting/backtests/footystats_ablation/footystats_ablation.json
data/08_reporting/backtests/footystats_xg_ablation/footystats_ablation.csv
data/08_reporting/backtests/footystats_xg_ablation/footystats_ablation.json
```

- [ ] **Step 2: Update roadmap**

In `roadmap.md`, add delivered/active milestone bullets:

```markdown
- FootyStats pre-match xG feature pack:
  - `footystats_mode=ppg_xg`,
  - adds team/opponent/diff pre-match xG features on top of PPG,
  - strict safe-column loader excludes post-match `team_a_xg` and `team_b_xg`,
  - generic FootyStats ablation runner compares arbitrary control/treatment modes without fixture context.
```

In the roadmap section, mark the next measurement as:

```markdown
1. Run and review the xG-over-PPG ablation:
   - control: `ppg`,
   - treatment: `ppg_xg`,
   - seasons: `2023`, `2024`, `2025`,
   - success: positive aggregate RF delta, non-negative R²/correlation deltas, and at least two comparable seasons with positive RF delta.
```

- [ ] **Step 3: Run markdown-adjacent checks through full gate later**

No separate markdown linter exists. This task is verified by the final full gate in Task 11.

- [ ] **Step 4: Commit**

```bash
git add README.md roadmap.md
git commit -m "docs: document footystats xg ablation workflow"
```

---

### Task 10: Run Real Ablations And Record Results

**Files:**
- Generated reports under `data/08_reporting/backtests/footystats_ablation/`
- Generated reports under `data/08_reporting/backtests/footystats_xg_ablation/`
- Modify: `roadmap.md`

- [ ] **Step 1: Verify 2023 PPG comparability before measurement**

The current branch is expected to include the earlier fix for the 2023 round-18 malformed `id_clube=1` rows. Verify that before interpreting the xG measurement as a three-season result.

Run:

```bash
uv run --frozen python - <<'PY'
from pathlib import Path
from cartola.backtesting.config import BacktestConfig
from cartola.backtesting.runner import run_backtest

config = BacktestConfig(
    season=2023,
    start_round=5,
    budget=100,
    project_root=Path("."),
    output_root=Path("data/08_reporting/backtests/footystats_2023_ppg_smoke"),
    fixture_mode="none",
    footystats_mode="ppg",
    footystats_evaluation_scope="historical_candidate",
    footystats_league_slug="brazil-serie-a",
    current_year=2026,
)
result = run_backtest(config)
print("missing", result.metadata.footystats_missing_join_keys_by_round)
print("duplicates", result.metadata.footystats_duplicate_join_keys_by_round)
print(result.summary[["strategy", "average_actual_points"]].to_string(index=False))
PY
```

Expected on the current branch:

```text
missing {}
duplicates {}
```

If this fails with `id_clube=1` in round `18`, stop and restore the earlier malformed-club identity fix before continuing. Do not continue to xG measurement while 2023 PPG comparability is broken.

- [ ] **Step 2: Run PPG baseline ablation with generic runner**

Run:

```bash
uv run --frozen python scripts/run_footystats_ablation.py \
  --seasons 2023,2024,2025 \
  --start-round 5 \
  --budget 100 \
  --current-year 2026 \
  --control-footystats-mode none \
  --treatment-footystats-mode ppg \
  --output-root data/08_reporting/backtests/footystats_ablation \
  --force
```

Expected: exit code `0`. The current branch should print `comparable seasons: 3`. If it prints fewer than `3`, inspect `footystats_ablation.json` and record the excluded seasons before making any generalization claim.

- [ ] **Step 3: Run xG-over-PPG ablation**

Run:

```bash
uv run --frozen python scripts/run_footystats_ablation.py \
  --seasons 2023,2024,2025 \
  --start-round 5 \
  --budget 100 \
  --current-year 2026 \
  --control-footystats-mode ppg \
  --treatment-footystats-mode ppg_xg \
  --output-root data/08_reporting/backtests/footystats_xg_ablation \
  --force
```

Expected: exit code `0`; at least `2` successful comparable seasons are required by the decision rule. The current target is `3` because 2023 comparability should already be restored.

- [ ] **Step 4: Extract xG report summary**

Run:

```bash
uv run --frozen python - <<'PY'
import json
from pathlib import Path

path = Path("data/08_reporting/backtests/footystats_xg_ablation/footystats_ablation.json")
report = json.loads(path.read_text())
print("included", report["aggregate"]["included_seasons"])
print("decision", report["decision"])
print("aggregate metrics")
for key, value in report["aggregate"]["metrics"].items():
    if key.endswith("_delta") or key.startswith("control_rf") or key.startswith("treatment_rf"):
        print(key, value)
print("season deltas")
for season in report["seasons"]:
    print(
        season["season"],
        season["control_status"],
        season["treatment_status"],
        season["metric_status"],
        season["rf_avg_points_delta"],
        season["player_r2_delta"],
        season["player_corr_delta"],
    )
PY
```

Expected: printed included seasons, decision object, aggregate deltas, and per-season deltas.

- [ ] **Step 5: Update roadmap with xG result**

In `roadmap.md`, add a short result block under Current Interpretation:

```markdown
The xG-over-PPG ablation result is:

- included seasons: `...`;
- aggregate RF average points delta: `...`;
- aggregate player R² delta: `...`;
- aggregate player correlation delta: `...`;
- decision: `keep_treatment=true|false`.
```

Use the actual numbers from Step 3.
If fewer than `3` seasons are included, explicitly list excluded seasons and do not say the result generalizes across all three candidate seasons.

- [ ] **Step 6: Commit result summary**

Do not commit generated report files unless the repository already tracks that output directory. Commit only documentation updates unless `git status --short` shows report files already tracked.

```bash
git add roadmap.md
git commit -m "docs: record footystats xg ablation result"
```

---

### Task 11: Full Verification

**Files:**
- All touched code and docs.

- [ ] **Step 1: Run focused tests**

Run:

```bash
uv run --frozen pytest \
  src/tests/backtesting/test_footystats_features.py \
  src/tests/backtesting/test_features.py \
  src/tests/backtesting/test_runner.py \
  src/tests/backtesting/test_cli.py \
  src/tests/backtesting/test_footystats_ablation.py \
  -q
```

Expected: all focused tests pass.

- [ ] **Step 2: Run full gate**

Run:

```bash
uv run --frozen scripts/pyrepo-check --all
```

Expected:

```text
All checks passed!
...
passed
```

The exact pytest count may be higher than `277` after new tests.

- [ ] **Step 3: Inspect git status**

Run:

```bash
git status --short --branch
```

Expected: only intentional code/docs changes are staged or unstaged. Generated ignored report files should not appear.

- [ ] **Step 4: Commit final cleanup if needed**

If Task 11 required small fixes, commit them:

```bash
git add <changed files>
git commit -m "test: verify footystats xg ablation workflow"
```

---

## Self-Review

- Spec coverage:
  - Existing PPG ablation report-contract fixes are preserved in Task 0.
  - Generic report naming is covered in Tasks 5 and 7.
  - Output-root deletion safety is covered in Task 5 with the required `data/08_reporting/backtests/` parent.
  - Mode-aware loader and strict safe `usecols` are covered in Task 2.
  - `ppg` regression behavior is covered in Task 2.
  - xG feature columns and frame merge are covered in Tasks 1 and 3.
  - Runner/CLI mode support is covered in Task 4.
  - Historical comparable ablation, 2023 pre-measurement verification, and success decision are covered in Tasks 8 and 10.
  - Documentation is covered in Task 9.

- Placeholder scan:
  - No placeholder tasks are left. Each task has concrete files, code snippets, commands, and expected outputs.

- Type consistency:
  - `ppg_xg` is the only new mode name.
  - Loader returns `FootyStatsFeatureLoadResult`; `FootyStatsPPGLoadResult` remains an alias for compatibility.
  - Generic ablation uses `FootyStatsAblationConfig`; PPG names remain aliases/wrappers only.
