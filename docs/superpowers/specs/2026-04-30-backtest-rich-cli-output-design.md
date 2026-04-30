# Backtest Rich CLI Output Design

## Goal

Make `python -m cartola.backtesting.cli` explain its result directly in the terminal and generate one consolidated performance chart, using the same Rich-based terminal quality already used by `scripts/recommend_squad.py`, `scripts/run_live_round.py`, and `scripts/capture_strict_round_fixture.py`.

The backtest command currently prints only:

```text
WARNING: Exploratory fixture mode uses reconstructed fixture data and is not strict no-leakage.
Backtest complete: season=2025 output=data/08_reporting/backtests/perf_one_jobs_4/2025
```

That is operationally weak because the user cannot see the winning strategy, score deltas, runtime, parallelism settings, or report location without opening CSV/JSON files manually.

## User Pain

Backtests are now fast enough to run repeatedly with `--jobs`, but the terminal output does not summarize the run. This creates repeated friction:

- users do not know whether `random_forest`, `baseline`, or `price` won;
- users cannot quickly see how strategy performance evolved round by round;
- users cannot see which formation `random_forest` chose each round without opening `round_results.csv`;
- performance tests require opening `run_metadata.json` to confirm `jobs`, backend, effective workers, and wall-clock time;
- warnings are plain text and do not visually separate expected warnings from successful completion;
- output style is inconsistent with the live and recommendation commands.

The command should show a useful run summary immediately after completion.

## Scope

Add Rich terminal output to `src/cartola/backtesting/cli.py`.

Generate one consolidated chart from `result.round_results`.

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

This is display/report-only.

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
- run details table from `BacktestMetadata`;
- one consolidated performance chart path after chart generation.

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

The command should also generate one consolidated interactive Plotly HTML chart at:

```text
<output_path>/charts/strategy_performance_by_round.html
```

HTML is the v1 chart format because the user wants to inspect, zoom, hover, and interact with round-by-round performance. Use Plotly's HTML export. Do not add Matplotlib, Kaleido, browser automation, or static image export dependencies for this feature.

`plotly` must be a direct project dependency in `pyproject.toml`; do not rely on a transitive dependency already present in `uv.lock`.

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
- `Performance chart`;
- `Scoring contract`.

Values come from `config`, `result.metadata`, and `config.output_path`.

Formatting:

- `Wall clock seconds` with two decimal places;
- missing values shown as `n/a`;
- paths rendered as project-root-relative paths when possible, matching existing CLI output style.

## Chart Output Contract

Generate one interactive Plotly HTML chart after `run_backtest()` returns and before printing the run details table.

Path:

```text
<config.output_path>/charts/strategy_performance_by_round.html
```

The chart should be generated from `result.round_results` only. It must not read the CSV file back from disk.

The chart should be a standalone HTML file with Plotly interactivity. It should work when opened from disk without network access, so the implementation must embed Plotly.js in the output file with `include_plotlyjs=True`.

The figure should contain three vertically stacked subplots:

1. Cumulative actual points by strategy.
2. Per-round actual points by strategy.
3. `random_forest` formation used by round.

Use `actual_points` for strategy performance. In the current scoring contract, this is already the captain-aware total and includes the tecnico.

### Strategy Panels

Render one interactive line trace per strategy:

- `random_forest`;
- `baseline`;
- `price`.

If a strategy is missing from a run, omit that line instead of failing.

The cumulative panel should compute cumulative sums by strategy ordered by `rodada`.

The per-round panel should plot `actual_points` directly.

Hover data should include at least:

- round;
- strategy;
- actual points;
- cumulative points for the cumulative panel.

### Formation Panel

Show only the `random_forest` formation by round.

Reason: the chart's main purpose is to inspect the model strategy. Showing formations for every strategy makes the chart noisy. `round_results.csv` remains the source for all strategy/formation combinations.

Formation display contract:

- x-axis is `rodada`;
- y-axis is categorical formation label;
- one marker trace across rounds;
- hover data includes round, formation, actual points, captain name when present, and budget used when present;
- text labels may show the formation when they do not make the chart cluttered;
- the panel title must make clear that formations are for `random_forest`.

### Styling

Use a restrained chart style:

- white or near-white background;
- dark axis labels;
- clear legend;
- stable color mapping for strategies;
- stable output dimensions, e.g. 1200 by 900;
- shared x-axis behavior where practical;
- no decorative gradients or unrelated artwork.

### Empty Or Incomplete Data

If `result.round_results` is empty, still create the `charts/` directory but skip chart generation and print `n/a` for `Performance chart`.

If `result.round_results` lacks required columns for charting, fail loudly with a clear `ValueError`. Required columns:

- `rodada`;
- `strategy`;
- `actual_points`;
- `formation`.

Do not silently create a partial or misleading chart.

## Error Output

The current backtest CLI lets exceptions propagate. This spec does not require changing operational failure handling.

Do not add broad exception catching in this change. A failure-handling redesign should be separate because it affects debugging and exit behavior.

## Test Requirements

Add or update tests in `src/tests/backtesting/test_cli.py`.

Required assertions:

- successful CLI output contains `Backtest complete`;
- successful CLI output contains strategy names from the summary, including `random_forest`;
- successful CLI output contains `Fixture mode`, `FootyStats mode`, `Matchup context mode`, `Jobs requested`, `Workers effective`, `Parallel backend`, and `Wall clock seconds`;
- successful CLI output contains `Performance chart`;
- chart generation writes `charts/strategy_performance_by_round.html` for non-empty round results;
- generated chart contains visible references to all present strategies and `random_forest` formations;
- generated chart embeds Plotly.js so it is usable offline from disk;
- empty `round_results` does not fail and prints `Performance chart` as `n/a`;
- missing chart columns fail with `ValueError`;
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
- PNG/SVG export.

The backtest CLI should remain a single-run command.

## Patrimonio Scope

This feature must not change budget semantics.

The current backtest remains fixed-budget:

```text
each round budget = --budget
```

Do not add evolving patrimonio simulation here. That future feature requires its own data audit and design because it changes model evaluation and comparability. The chart may show `budget_used` in a later revision, but v1 should focus on score performance and formation choices.

## Risks

### Rich Markup In Captured Tests

Rich may wrap or format output differently based on terminal width. Tests should assert stable text fragments, not exact full-table drawings.

### Overloading The CLI Module

`src/cartola/backtesting/cli.py` is small today. Adding display helpers there is acceptable for this v1. If a future change expands the output substantially, that change should first move display code to a dedicated module such as `src/cartola/backtesting/cli_output.py`.

Do not create that module in v1 unless the implementation becomes hard to read.

### Warning Semantics

Warnings are still warnings from `BacktestMetadata`. This change only changes their presentation. It must not suppress or reinterpret warnings.

### Chart Complexity

Plotly HTML is appropriate for this v1 because the user explicitly wants interactive chart inspection. Keep the figure simple: three subplots, stable traces, and useful hover data. Static PNG/SVG export remains out of scope.

### Dependency Boundary

Adding Plotly as a direct dependency is acceptable. Adding Kaleido or browser automation for static export is not part of this feature.

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
- runtime from metadata;
- a chart path under `charts/strategy_performance_by_round.html`;
- one interactive chart with cumulative strategy performance, per-round strategy performance, and `random_forest` formations by round.

The existing report files should still be written exactly as before.
