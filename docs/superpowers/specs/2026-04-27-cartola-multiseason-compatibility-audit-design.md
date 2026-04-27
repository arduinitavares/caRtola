# Cartola Multi-Season Compatibility Audit Design

## Goal

Build an audit-only tool that runs the existing offline backtesting pipeline across every available season and reports compatibility issues clearly.

The purpose is to answer:

> Which historical seasons can the current loader, feature builder, and no-fixture backtester handle, and where do they fail?

This milestone does not fix older seasons, generate fixtures, train multi-season models, or compare partial current-season metrics against complete historical seasons.

## Scope

In scope:

- Discover season directories under `data/01_raw/`.
- Include complete historical seasons and the current partial 2026 season.
- Run staged compatibility checks using `fixture_mode="none"`.
- Write isolated audit outputs under `data/08_reporting/backtests/compatibility/`.
- Produce CSV and JSON reports with one row/object per detected season.
- Capture short CSV error messages and full JSON error details.

Out of scope:

- Schema fixes for failing seasons.
- Fixture reconstruction for older seasons.
- Strict historical fixture backfill.
- Multi-season model training.
- Model comparison or new predictors.
- Automatic retries or migration logic.

## Configuration

The audit uses explicit defaults:

```text
start_round = 5
complete_round_threshold = 38
expected_complete_rounds = 38
fixture_mode = none
```

The current calendar year is detected from the runtime date. For this milestone, the current year is 2026.

## Season Discovery

The audit scans:

```text
data/01_raw/
```

Include a directory only when:

- its directory name is numeric, such as `2025`;
- it contains at least one `rodada-*.csv` file.

Ignore:

- `fixtures/`;
- `fixtures_strict/`;
- `fixtures_snapshots/`;
- reporting directories;
- non-numeric directories;
- numeric directories with no `rodada-*.csv` files.

Round metadata is computed from filenames, not loaded data:

```text
round_file_count = number of rodada-*.csv files
min_round = minimum parsed round number
max_round = maximum parsed round number
```

Round filenames must match the existing project convention:

```text
rodada-<round>.csv
```

If a numeric season directory has malformed round filenames, the season receives a report row with:

```text
load_status = failed
feature_status = skipped
backtest_status = skipped
error_stage = discovery
```

## Season Classification

Each season gets a `season_status`:

```text
if season == current_year and max_round < complete_round_threshold:
    season_status = partial_current
elif round_file_count == expected_complete_rounds and max_round == expected_complete_rounds:
    season_status = complete_historical
else:
    season_status = irregular_historical
```

`metrics_comparable` is:

```text
true  only for complete_historical
false for partial_current and irregular_historical
```

Examples:

- `2025` with 38 round files and `max_round=38`: `complete_historical`.
- `2026` with 13 round files and `max_round=13`: `partial_current`.
- `2022` with 39 round files: `irregular_historical`.

For `partial_current`, add:

```text
notes = partial current season; metrics are smoke-test only
```

For `irregular_historical`, add:

```text
notes = historical season has unusual round file count
```

## Audit Stages

Every detected season runs stages independently. A failure in one season must not abort the whole audit.

Status values are:

```text
ok
failed
skipped
not_applicable
```

### Stage 1: Load

Run:

```python
load_season_data(season)
```

If loading succeeds:

```text
load_status = ok
```

If loading fails:

```text
load_status = failed
feature_status = skipped
backtest_status = skipped
error_stage = load
```

### Stage 2: Feature

Feature compatibility means every eligible target round can build the same frames the normal no-fixture backtest path needs.

For every target round `r` where:

```text
start_round <= r <= max_round
```

attempt:

```python
build_training_frame(
    season_df,
    r,
    playable_statuses=config.playable_statuses,
    fixtures=None,
)
build_prediction_frame(
    season_df,
    r,
    fixtures=None,
)
```

If all eligible rounds pass:

```text
feature_status = ok
```

If any target round fails:

```text
feature_status = failed
backtest_status = skipped
error_stage = feature
```

If `max_round < start_round`:

```text
feature_status = not_applicable
backtest_status = skipped
evaluated_rounds = 0
```

### Stage 3: Backtest Smoke

Run the normal backtester with:

```text
fixture_mode = none
start_round = 5
budget = 100
```

The backtest must use an isolated output path:

```text
data/08_reporting/backtests/compatibility/runs/{season}/
```

It must not overwrite normal experiment outputs:

```text
data/08_reporting/backtests/{season}/
```

If backtest succeeds:

```text
backtest_status = ok
```

If backtest fails:

```text
backtest_status = failed
error_stage = backtest
```

Metric columns are null unless `backtest_status="ok"`.

## Evaluation Round Metadata

For each season:

```text
evaluated_rounds = count of target rounds r where start_round <= r <= max_round
first_evaluated_round = start_round if evaluated_rounds > 0 else null
last_evaluated_round = max_round if evaluated_rounds > 0 else null
```

These values are based on filename-derived `max_round`.

## Fixture Reporting

The audit does not use fixtures.

Every report row includes:

```text
fixture_mode = none
fixture_status = not_applicable
```

Missing exploratory or strict fixture coverage must not be treated as an audit failure in this milestone.

## Output Files

Write:

```text
data/08_reporting/backtests/compatibility/season_compatibility.csv
data/08_reporting/backtests/compatibility/season_compatibility.json
```

Per-season isolated backtest outputs, when produced:

```text
data/08_reporting/backtests/compatibility/runs/{season}/
```

The CSV is optimized for quick scanning. The JSON preserves richer error details.

## CSV Schema

Required columns:

```text
season
season_status
metrics_comparable
round_file_count
min_round
max_round
start_round
evaluated_rounds
first_evaluated_round
last_evaluated_round
fixture_mode
fixture_status
load_status
feature_status
backtest_status
error_stage
error_type
error_message
baseline_avg_points
random_forest_avg_points
price_avg_points
notes
```

Rules:

- `error_message` is a short one-line message suitable for CSV inspection.
- `error_message` must be truncated to 300 characters.
- Metric columns are null unless `backtest_status="ok"`.
- For successful complete historical seasons, `metrics_comparable=true`.
- For `partial_current` and `irregular_historical`, `metrics_comparable=false`.

## JSON Schema

The JSON file contains an object with:

```text
generated_at_utc
project_root
config
seasons
```

`config` includes:

```text
start_round
complete_round_threshold
expected_complete_rounds
fixture_mode
```

Each entry in `seasons` includes all CSV fields plus:

```text
error_detail
```

`error_detail` includes:

```text
stage
exception_type
message
traceback
target_round
```

`traceback` may be null when no error occurred.

## Metrics

When `backtest_status="ok"`, read the returned `summary` dataframe from `run_backtest`.

Populate:

```text
baseline_avg_points
random_forest_avg_points
price_avg_points
```

The value for each strategy is the average actual points per evaluated round.

For `partial_current` seasons such as 2026, metrics are recorded but `metrics_comparable=false`.

## CLI

Add a script with this stable local invocation:

```bash
uv run python scripts/audit_backtest_compatibility.py
```

Supported options:

```text
--project-root PATH
--start-round INT
--complete-round-threshold INT
--expected-complete-rounds INT
--output-root PATH
```

Defaults:

```text
--project-root .
--start-round 5
--complete-round-threshold 38
--expected-complete-rounds 38
--output-root data/08_reporting/backtests/compatibility
```

## Error Handling

The audit must catch exceptions per season and continue to the next season.

Error-stage precedence:

```text
discovery
load
feature
backtest
```

Only the first failing stage is recorded as `error_stage`.

If a stage fails, later dependent stages are marked `skipped`.

## Testing Requirements

Add tests for:

- season discovery includes numeric season directories with `rodada-*.csv`;
- season discovery ignores non-season directories;
- round metadata is parsed from filenames;
- season classification for complete historical, partial current, and irregular historical seasons;
- load failure produces one report row and skips later stages;
- feature failure records the target round and skips backtest;
- feature check covers every eligible target round from `start_round..max_round`;
- backtest output path is isolated under compatibility output root;
- 2026 partial season has `metrics_comparable=false`;
- metrics are null unless backtest succeeds;
- CSV error messages are truncated;
- JSON preserves full error details.

Existing quality gate must remain clean:

```bash
uv run --frozen scripts/pyrepo-check --all
```

## Success Criteria

- The audit command completes across all detected seasons without aborting globally.
- Every detected season has exactly one CSV row and one JSON season object.
- Normal backtest report directories are not overwritten.
- 2026 is included and clearly marked as `partial_current`.
- 2022 or any unusual historical season is marked `irregular_historical`, not silently treated as comparable.
- The report is sufficient to choose the next compatibility fixes.
- The full quality gate passes.
