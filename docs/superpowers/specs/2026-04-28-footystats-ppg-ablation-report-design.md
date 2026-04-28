# FootyStats PPG Ablation Report Design

## Goal

Build a narrow measurement tool that answers one question:

> Does adding leakage-safe FootyStats pre-match PPG features improve RandomForest performance versus the same no-FootyStats control on complete historical candidate seasons?

This is an ablation report, not a generic backtest launcher. It must isolate the marginal value of `footystats_mode=ppg`.

## Scope

In scope:

- Run paired control/treatment backtests for candidate historical seasons.
- Default seasons: `2023`, `2024`, `2025`.
- Control: `footystats_mode="none"`.
- Treatment: `footystats_mode="ppg"`.
- `fixture_mode` is fixed to `"none"` for both runs.
- Write isolated backtest outputs under an ablation-specific root.
- Write a summary CSV and an authoritative JSON audit artifact.
- Continue when one season fails and report the failure.

Out of scope:

- Fixture context, including exploratory or strict fixtures.
- xG, odds, goal environment, cards, corners, or other FootyStats columns.
- Live/current-season gameplay.
- Pooled multi-season model training.
- Fixing older Cartola seasons.

## CLI Contract

Command:

```bash
uv run --frozen python scripts/run_footystats_ppg_ablation.py \
  --seasons 2023,2024,2025 \
  --start-round 5 \
  --budget 100 \
  --current-year 2026
```

Defaults:

- `--seasons`: `2023,2024,2025`
- `--start-round`: `5`
- `--budget`: `100.0`
- `--current-year`: runtime UTC year if omitted, but tests must pass it explicitly
- `--project-root`: `.`
- `--output-root`: `data/08_reporting/backtests/footystats_ablation`
- `--footystats-league-slug`: `brazil-serie-a`
- `--force`: absent by default

Season parsing:

- parse `--seasons` as comma-separated positive integers;
- reject empty entries;
- reject duplicates;
- preserve input order in per-season rows;
- use the same input order in JSON config.

No `--fixture-mode` argument exists in v1. The runner always uses `fixture_mode="none"`.

`2026` is excluded by default. A future smoke option can be added later, but v1 only reports comparable historical seasons.

## Eligibility

Before running either control or treatment for a season, validate the season with the same runtime rules required by the FootyStats PPG loader in `historical_candidate` mode:

- selected source file is the requested season and `matches` table;
- required safe columns are present;
- exact `Game Week` coverage is `1..38`;
- all fixture statuses are `complete`;
- team names map cleanly and bidirectionally to Cartola club IDs;
- no duplicate normalized `(rodada, id_clube)` rows;
- PPG values are present and numeric.

The ablation must not trust stale compatibility audit CSV/JSON output.

If eligibility fails, skip both control and treatment for that season:

- `control_status="skipped"`
- `treatment_status="skipped"`
- `error_stage="eligibility"`
- `metrics_comparable=false`

## Output Path Validation

Path validation runs before any eligibility check or backtest run, regardless of `--force`.

Resolve paths as follows:

- resolve `project_root` to an absolute path with symlinks followed;
- if `output_root` is relative, resolve it under `project_root`;
- if `output_root` is absolute, use it as-is;
- require resolved `output_root` to be equal to or inside resolved `project_root`;
- require resolved `output_root.name == "footystats_ablation"`;
- reject resolved `output_root` if it is equal to:
  - `project_root`
  - `project_root/data`
  - `project_root/data/08_reporting`
  - `project_root/data/08_reporting/backtests`
  - `project_root/data/08_reporting/backtests/{season}` for any requested season

Before running any backtest, compute each `BacktestConfig.output_path` and reject if it equals:

```text
{project_root}/data/08_reporting/backtests/{season}
```

for any requested season.

## Backtest Configuration

For each eligible season, run two backtests with identical non-FootyStats settings:

Control:

```python
BacktestConfig(
    season=season,
    start_round=start_round,
    budget=budget,
    project_root=project_root,
    output_root=output_root / "runs" / str(season) / "footystats_mode=none",
    fixture_mode="none",
    footystats_mode="none",
    footystats_evaluation_scope="historical_candidate",
    footystats_league_slug=league_slug,
    current_year=current_year,
)
```

Treatment:

```python
BacktestConfig(
    season=season,
    start_round=start_round,
    budget=budget,
    project_root=project_root,
    output_root=output_root / "runs" / str(season) / "footystats_mode=ppg",
    fixture_mode="none",
    footystats_mode="ppg",
    footystats_evaluation_scope="historical_candidate",
    footystats_league_slug=league_slug,
    current_year=current_year,
)
```

The nested final output directory is intentional. `BacktestConfig.output_path` appends `/{season}`, so the physical outputs are:

```text
{output_root}/runs/{season}/footystats_mode=none/{season}/
{output_root}/runs/{season}/footystats_mode=ppg/{season}/
```

The ablation runner must never write to:

```text
data/08_reporting/backtests/{season}/
```

## Output Root And Force Semantics

Default report root:

```text
data/08_reporting/backtests/footystats_ablation/
```

Reports:

```text
data/08_reporting/backtests/footystats_ablation/ppg_ablation.csv
data/08_reporting/backtests/footystats_ablation/ppg_ablation.json
```

If `output_root` exists and `--force` is not passed, fail before running any season.

If `--force` is passed:

- run the same output path validation described above;
- delete only the resolved ablation output root before running.

This keeps force useful without making it a broad recursive-delete footgun.

## CSV Schema

One row per season plus one aggregate row.

Columns:

- `season`
- `row_type`: `season | aggregate`
- `season_status`: `candidate | failed | aggregate`
- `metrics_comparable`
- `control_status`: `ok | failed | skipped | not_applicable`
- `treatment_status`: `ok | failed | skipped | not_applicable`
- `control_output_path`
- `treatment_output_path`
- `control_baseline_avg_points`
- `treatment_baseline_avg_points`
- `baseline_avg_points`
- `baseline_avg_points_equal`
- `control_rf_avg_points`
- `treatment_rf_avg_points`
- `rf_avg_points_delta`
- `control_player_r2`
- `treatment_player_r2`
- `player_r2_delta`
- `control_player_corr`
- `treatment_player_corr`
- `player_corr_delta`
- `rf_minus_baseline_control`
- `rf_minus_baseline_treatment`
- `error_stage`
- `error_message`

Metric fields are nullable:

- control metric fields are null unless `control_status="ok"`;
- treatment metric fields are null unless `treatment_status="ok"`;
- delta fields are null unless both statuses are `ok` and `metrics_comparable=true`;
- aggregate metric fields are null if no comparable successful season exists.

CSV serialization:

- null values are written as empty cells;
- `error_message` is truncated to 500 characters;
- full error messages and tracebacks live in JSON only.

`baseline_avg_points` is written only when both baseline values exist and are equal within a small tolerance (`1e-9`). If they differ, keep both control/treatment baseline fields, set `baseline_avg_points` null, set `baseline_avg_points_equal=false`, and mark `error_stage="metric_extraction"` because non-FootyStats baseline drift invalidates the paired comparison.

Aggregate row:

- `season="aggregate"`
- `row_type="aggregate"`
- `season_status="aggregate"`
- `control_status="not_applicable"`
- `treatment_status="not_applicable"`
- `metrics_comparable=true` only if at least one comparable successful season exists
- path and error fields null
- metric values are simple unweighted means over season rows where:
  - `metrics_comparable=true`
  - `control_status="ok"`
  - `treatment_status="ok"`

## JSON Schema

JSON is required and is the authoritative artifact.

Top-level fields:

- `config`
- `seasons`
- `aggregate`
- `generated_at_utc`

`config` includes:

- `project_root`
- `output_root`
- `resolved_project_root`
- `resolved_output_root`
- `seasons`
- `start_round`
- `budget`
- `current_year`
- `resolved_current_year`
- `fixture_mode`: always `"none"`
- `control_footystats_mode`: `"none"`
- `treatment_footystats_mode`: `"ppg"`
- `footystats_evaluation_scope`: `"historical_candidate"`
- `footystats_league_slug`
- `force`

Each `seasons[]` item includes:

- all CSV row fields;
- `control_config`;
- `treatment_config`;
- `treatment_source_path`;
- `treatment_source_sha256`;
- `control_summary_path`;
- `treatment_summary_path`;
- `control_diagnostics_path`;
- `treatment_diagnostics_path`;
- `errors`: list of objects with:
  - `stage`
  - `type`
  - `message`
  - `traceback`

`aggregate` includes:

- included seasons;
- excluded seasons with reason;
- aggregate metric values;
- aggregation method: `"unweighted_mean_across_successful_comparable_seasons"`.

## Metric Extraction Rules

Read metrics from each run's `summary.csv` and `diagnostics.csv`.

Required summary strategy rows:

- `baseline`
- `random_forest`

Required diagnostics rows:

- `section == "prediction"`
- `strategy == "random_forest"`
- `position == "all"`
- `metric == "player_r2"`
- `metric == "player_correlation"`

If a run exits successfully but a required row is missing or duplicated, mark that run failed with:

- `error_stage="metric_extraction"`
- metric fields for that run null

Do not invent zeros.

## Error Handling

The ablation continues across seasons.

Stages:

- `eligibility`
- `control_backtest`
- `treatment_backtest`
- `metric_extraction`
- `report_write`

CSV and JSON reports are written atomically:

- write to temporary files in the same output directory;
- rename temporary files to `ppg_ablation.csv` and `ppg_ablation.json` only after the write succeeds;
- if report writing fails, exit non-zero without leaving a final-looking partial report.

`generated_at_utc` is an ISO-8601 UTC string ending in `Z`.

If report writing fails, the command exits non-zero. Because the authoritative JSON may not exist in that case, report-write failure details are emitted to stderr/log output only. Otherwise, the command exits zero if at least one season completes both runs and report files are written; it exits non-zero if every season fails before producing comparable metrics.

## Tests

Add tests for:

- CLI defaults parse to seasons `2023,2024,2025`.
- `fixture_mode` is fixed to `none` in both generated `BacktestConfig` objects.
- output roots are exactly `{output_root}/runs/{season}/footystats_mode={mode}`.
- no output path equals `data/08_reporting/backtests/{season}`.
- eligibility failure skips both control and treatment.
- failed treatment/control records nullable metrics and error details.
- metric extraction fails on missing RF or baseline rows.
- baseline mismatch marks metric extraction failure.
- aggregate includes only rows where `metrics_comparable=true` and both statuses are `ok`.
- existing output root fails without `--force`.
- `--force` rejects unsafe paths and removes only a safe ablation root.
- output path validation rejects unsafe paths even without `--force`.
- absolute `--output-root` is accepted only when it resolves inside `project_root`.
- CLI season parsing rejects empty values and duplicates while preserving input order.
- JSON contains per-run configs, source hash, output paths, errors, and aggregate inputs.
- CSV writes empty cells for nulls and truncates `error_message` to 500 characters.
- report writes are atomic and do not leave final-looking partial files on failure.

## Acceptance Criteria

- `fixture_mode="none"` for every run.
- Runtime eligibility validation happens before both runs.
- Output path validation happens before any eligibility check or backtest run.
- `2023`, `2024`, and `2025` produce season rows or clear failures.
- Aggregate excludes failed and non-comparable rows.
- Normal backtest output directories are untouched.
- CSV and JSON reports are written.
- Quality gate passes:

```bash
uv run --frozen scripts/pyrepo-check --all
```
