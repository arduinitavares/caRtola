# Backtest Rich CLI Output Design

## Goal

Make `python -m cartola.backtesting.cli` explain its result directly in the terminal with the same Rich-based quality already used by `scripts/recommend_squad.py`, `scripts/run_live_round.py`, and `scripts/capture_strict_round_fixture.py`.

The backtest command currently prints only:

```text
WARNING: Exploratory fixture mode uses reconstructed fixture data and is not strict no-leakage.
Backtest complete: season=2025 output=data/08_reporting/backtests/perf_one_jobs_4/2025
```

That is operationally weak because the user cannot see the winning strategy, score deltas, runtime, parallelism settings, or report location without opening CSV/JSON files manually.

## User Pain

Backtests are now fast enough to run repeatedly with `--jobs`, but the terminal output does not summarize the run. This creates repeated friction:

- users do not know whether `random_forest`, `baseline`, or `price` won;
- performance tests require opening `run_metadata.json` to confirm `jobs`, backend, effective workers, and wall-clock time;
- warnings are plain text and do not visually separate expected warnings from successful completion;
- output style is inconsistent with the live and recommendation commands.

The command should show a useful run summary immediately after completion.

## Scope

Add Rich terminal output to `src/cartola/backtesting/cli.py`.

Do not change:

- `run_backtest()` behavior;
- report file contents;
- CSV schemas;
- JSON metadata schema;
- optimizer logic;
- model logic;
- scoring contract;
- fixture behavior;
- parallelism behavior;
- CLI arguments.

This is display-only.

## Existing Patterns

The repository already depends on Rich in `pyproject.toml`.

Existing commands use:

- `Console`;
- `Panel`;
- `Table`;
- green success panels;
- red failure panels;
- compact two-column detail tables.

The backtest CLI should follow the same style rather than inventing a new terminal UI.

Reference files:

- `scripts/recommend_squad.py`;
- `scripts/run_live_round.py`;
- `scripts/capture_strict_round_fixture.py`;
- `src/cartola/backtesting/cli.py`.

## Considered Approaches

### Approach A: Minimal Rich Wrapper

Replace the final plain print with one green panel and one strategy table.

Pros:

- fastest to implement;
- better than current output.

Cons:

- does not expose runtime/parallelism metadata;
- still requires opening metadata for performance benchmarking;
- does not solve the user's immediate `--jobs` comparison pain.

### Approach B: Rich Summary Plus Run Details

Print:

- success panel;
- warning panel when warnings exist;
- strategy results table from `summary.csv` data already present in the `BacktestResult`;
- run details table from `BacktestMetadata`.

Pros:

- directly answers “what happened?” and “how fast did it run?”;
- consistent with existing Rich command style;
- no report schema change;
- small implementation surface.

Cons:

- adds display formatting code to the CLI module.

### Approach C: Separate Report Viewer Command

Add a new command that reads a finished backtest directory and renders a Rich summary.

Pros:

- reusable for old runs;
- separates report viewing from backtest execution.

Cons:

- does not fix the default backtest UX;
- adds a new public command and more documentation surface;
- more work than needed for v1.

## Decision

Use Approach B.

The backtest CLI should print a rich, immediate summary at the end of every successful run. A separate report viewer is out of scope for v1 and requires its own design if demand appears.

## Terminal Output Contract

On success, print in this order:

1. Warning panel, only if `result.metadata.warnings` is non-empty.
2. Green success panel titled `Backtest complete`.
3. Strategy results table.
4. Run details table.

### Warning Panel

Warnings should be grouped into one yellow panel titled `Backtest warnings`.

Example:

```text
╭──────────────────────── Backtest warnings ────────────────────────╮
│ Exploratory fixture mode uses reconstructed fixture data and is    │
│ not strict no-leakage.                                             │
╰───────────────────────────────────────────────────────────────────╯
```

Do not print `WARNING:` plain text lines in the success path.

### Success Panel

The success panel should include the core run identity:

```text
season=2025  start_round=5  output=data/08_reporting/backtests/perf_one_jobs_4/2025
```

Title:

```text
Backtest complete
```

Border style:

```text
green
```

### Strategy Results Table

The table should render one row per `summary.csv` strategy.

Columns:

- `Strategy`;
- `Rounds`;
- `Total actual`;
- `Avg actual`;
- `Total predicted`;
- `Vs price`.

Values come from `result.summary`:

- `strategy`;
- `rounds`;
- `total_actual_points`;
- `average_actual_points`;
- `total_predicted_points`;
- `actual_points_delta_vs_price`.

Formatting:

- integer rounds with no decimal places;
- point values with two decimal places;
- `Vs price` with a leading sign for positive and negative values;
- missing values shown as `n/a`.

Sort order should preserve `result.summary` row order. Do not add ranking logic in the display layer.

### Run Details Table

The table should include:

- `Output`;
- `Fixture mode`;
- `FootyStats mode`;
- `Matchup context mode`;
- `Jobs requested`;
- `Workers effective`;
- `Parallel backend`;
- `Model n_jobs`;
- `Prediction frames built`;
- `Wall clock seconds`;
- `Scoring contract`.

Values come from `config`, `result.metadata`, and `config.output_path`.

Formatting:

- `Wall clock seconds` with two decimal places;
- missing values shown as `n/a`;
- paths rendered as project-root-relative paths when possible, matching existing CLI output style.

## Error Output

The current backtest CLI lets exceptions propagate. This spec does not require changing operational failure handling.

Do not add broad exception catching in this change. A failure-handling redesign should be separate because it affects debugging and exit behavior.

## Test Requirements

Add or update tests in `src/tests/backtesting/test_cli.py`.

Required assertions:

- successful CLI output contains `Backtest complete`;
- successful CLI output contains strategy names from the summary, including `random_forest`;
- successful CLI output contains `Fixture mode`, `FootyStats mode`, `Matchup context mode`, `Jobs requested`, `Workers effective`, `Parallel backend`, and `Wall clock seconds`;
- warnings render as `Backtest warnings`;
- the success path no longer prints plain `WARNING:`;
- the command still returns exit code `0`;
- CSV/JSON report writing remains driven by `run_backtest()` and is not duplicated in the CLI tests.

Tests may mock `run_backtest()` and inspect captured stdout. They do not need to execute a full backtest.

## Non-Goals

Do not add:

- a new report viewer command;
- a `--plain` output mode;
- a `--json` output mode;
- interactive terminal behavior;
- progress bars;
- per-round result display;
- selected-player display;
- output comparison across multiple runs;
- automatic benchmark matrix support.

The backtest CLI should remain a single-run command.

## Risks

### Rich Markup In Captured Tests

Rich may wrap or format output differently based on terminal width. Tests should assert stable text fragments, not exact full-table drawings.

### Overloading The CLI Module

`src/cartola/backtesting/cli.py` is small today. Adding display helpers there is acceptable for this v1. If a future change expands the output substantially, that change should first move display code to a dedicated module such as `src/cartola/backtesting/cli_output.py`.

Do not create that module in v1 unless the implementation becomes hard to read.

### Warning Semantics

Warnings are still warnings from `BacktestMetadata`. This change only changes their presentation. It must not suppress or reinterpret warnings.

## Acceptance Criteria

A user running:

```bash
uv run --frozen python -m cartola.backtesting.cli \
  --season 2025 \
  --start-round 5 \
  --budget 100 \
  --fixture-mode exploratory \
  --footystats-mode ppg \
  --matchup-context-mode cartola_matchup_v1 \
  --current-year 2026 \
  --jobs 12 \
  --output-root data/08_reporting/backtests/perf_one_jobs_12
```

should immediately see:

- the exploratory fixture warning in a visible warning panel;
- the output path;
- which strategies ran;
- total and average actual points by strategy;
- delta versus price by strategy;
- `jobs=12` and effective worker information;
- backend and model worker information;
- runtime from metadata.

The existing report files should still be written exactly as before.
