# Cartola Offline Backtesting Design

Date: 2026-04-26

## Purpose

Build a Python-first offline backtesting pipeline that answers one question:

> If this algorithm had been used during a complete historical Cartola season, how would it have performed round by round?

Version 1 targets the 2025 season with a fixed per-round budget. It uses expanding-window walk-forward validation: before round N, train only on rounds earlier than N; at round N, predict available players, optimize a squad, then compare selected players against actual points from round N.

The goal is to prove the prediction and optimization loop before adding live market behavior, evolving wealth, external availability checks, or multi-season training.

## Scope

In scope for v1:

- Standalone Python module under `src/cartola/backtesting/`.
- Single-season backtest with `season=2025` as the default parameter.
- Fixed budget per round, defaulting to 100 cartoletas.
- Walk-forward simulation starting at round 5.
- Historical data loaded from `data/01_raw/{season}/rodada-*.csv`.
- Availability from the historical Cartola status field only.
- Baseline strategy plus one tree-based prediction model.
- Integer linear programming squad optimizer.
- CSV outputs under `data/08_reporting/backtests/{season}/`.
- Unit tests for data loading, feature time boundaries, optimization constraints, and metrics.

Out of scope for v1:

- Live Cartola API prediction workflow.
- Google/search/news availability agent.
- Evolving wealth and player price portfolio simulation.
- Multi-season training and evaluation.
- Web UI, dashboards, or automation.
- Kedro pipeline integration.

## Architecture

Create a focused package:

```text
src/cartola/backtesting/
  __init__.py
  config.py
  data.py
  features.py
  models.py
  optimizer.py
  metrics.py
  runner.py
  cli.py
```

Responsibilities:

- `config.py`: typed configuration objects for season, start round, budget, allowed formations, model, output path, and playable statuses.
- `data.py`: load raw round CSVs, normalize column names, map position/status IDs, and return one season dataframe.
- `features.py`: create leakage-safe features for each prediction round using only rows with `rodada < target_round`.
- `models.py`: provide a baseline predictor and a tree model behind a common interface.
- `optimizer.py`: choose the squad using integer linear programming.
- `metrics.py`: compute per-round and season-level performance.
- `runner.py`: orchestrate the walk-forward loop.
- `cli.py`: expose a command such as `python -m cartola.backtesting.cli --season 2025 --budget 100`.

This stays outside Kedro for v1 so the backtest loop is easy to run, debug, and test. The module should not depend on notebooks or R scripts.

## Data Model

The loader reads all `rodada-*.csv` files for the configured season and normalizes the fields needed by the backtester:

- `id_atleta`
- `apelido`
- `slug`
- `id_clube`
- `nome_clube`
- `posicao`
- `status`
- `rodada`
- `preco`
- `pontuacao`
- `media`
- `num_jogos`
- `variacao`
- scout columns such as `G`, `A`, `DS`, `SG`, `CA`, `FC`, `FS`, `FF`, `FD`, `FT`, `I`, `GS`, `DE`, `DP`, `CV`, `PP`, `PS`, `PC`, `GC`

Raw 2025 data stores Cartola-style names such as `atletas.status_id`, `atletas.posicao_id`, and `atletas.pontos_num`. The backtester should reuse the existing naming intent from `conf/base/parameters.yml` where practical, but v1 may implement a local explicit mapping to avoid introducing Kedro runtime dependencies.

Status IDs must be mapped to readable statuses:

- `2`: `Duvida`
- `3`: `Suspenso`
- `5`: `Contundido`
- `6`: `Nulo`
- `7`: `Provavel`

Position IDs must be mapped to Cartola position codes:

- `1`: `gol`
- `2`: `lat`
- `3`: `zag`
- `4`: `mei`
- `5`: `ata`
- `6`: `tec`

The loader should normalize accented status values to ASCII internally. The playable status whitelist for v1 is `{"Provavel"}`.

## Walk-Forward Flow

For each target round from `start_round` through the final available round:

1. Build the training set from rows where `rodada < target_round`.
2. Fit the predictor on the training set.
3. Build the candidate set from rows where `rodada == target_round`.
4. Filter candidates to playable players: `status in playable_statuses`.
5. Build features for target-round candidates using only historical rows from earlier rounds.
6. Predict target-round points for each candidate.
7. Run the optimizer with predicted points, player prices, positions, and budget.
8. Join the selected squad back to actual `pontuacao` from the target round.
9. Record selected players, predicted points, actual points, budget used, and constraint metadata.
10. Run the benchmark strategy for the same round and budget.

The feature builder must make the time boundary explicit in its API, for example:

```python
build_training_frame(season_df, target_round)
build_prediction_frame(season_df, target_round)
```

Tests must cover that no row from `target_round` or later contributes to rolling/cumulative features used for predictions.

## Features

V1 should prioritize simple, interpretable, leakage-safe features:

- Position code.
- Current target-round price.
- Current target-round season average from Cartola (`media`), treated as an available market field.
- Number of games played before the target round.
- Prior cumulative mean of points.
- Prior rolling means over 3 and 5 appearances for points.
- Prior cumulative and rolling means for key scouts.
- Prior price and price variation summaries.
- Club ID as categorical or encoded numeric feature.
- Round number.

The initial feature set should avoid opponent/team-strength features unless the required match data is already cleanly available for 2025. Opponent context can be added after the base backtest works.

Rows without sufficient prior history should remain eligible when possible. The feature builder can fill missing prior means with position-level priors computed only from past rows.

## Models

V1 includes two predictors:

- Baseline: predict by prior player mean, falling back to prior position mean.
- Tree model: use scikit-learn `RandomForestRegressor` as the default first model.

The baseline is required because it gives a sanity check for the feature pipeline and optimizer. The tree model is the first candidate algorithm to beat it.

Model evaluation inside the backtest should focus on squad performance, not only player-level RMSE. Player-level prediction metrics can be recorded as supporting diagnostics.

## Optimizer

Use integer linear programming through PuLP or OR-Tools. PuLP is preferred for v1 if it keeps dependencies lighter and the API simple.

Decision variable:

- `x_player` is 1 if the player is selected, else 0.

Objective:

- Maximize sum of predicted points for selected players.

Constraints:

- Total price must be less than or equal to fixed budget.
- Exactly one goalkeeper (`gol`).
- Exactly one coach (`tec`).
- Valid outfield formation. V1 should support a configurable list of formations, with `4-3-3` as the default: 1 `gol`, 2 `lat`, 2 `zag`, 3 `mei`, 3 `ata`, and 1 `tec`.
- At most one row per player ID.

The optimizer should return both the selected squad and metadata:

- Solver status.
- Budget used.
- Total predicted points.
- Total actual points after scoring.
- Formation used.
- Number of selected players.

If no feasible squad exists for a round, the runner should record the failure and continue to the next round.

## Benchmarks

V1 should include at least one naive benchmark and preferably two:

- Prior-mean benchmark: rank candidates by prior player mean and use the same optimizer.
- Price benchmark: rank candidates by current price and use the same optimizer.

Using the same optimizer keeps formation and budget constraints identical, so differences mostly reflect player scoring quality.

## Outputs

Write outputs under:

```text
data/08_reporting/backtests/{season}/
```

Files:

- `round_results.csv`: one row per strategy per round with predicted points, actual points, budget used, solver status, and selected count.
- `selected_players.csv`: one row per selected player per strategy per round.
- `player_predictions.csv`: all candidate predictions per round for diagnostics.
- `summary.csv`: season-level totals, averages, benchmark deltas, and simple prediction diagnostics.

All outputs should be deterministic for a fixed random seed.

## Error Handling

- Missing season directory: raise a clear error with the expected path.
- Missing round files: skip only if explicitly configured; default should fail clearly.
- Unknown status or position IDs: fail with the offending values.
- Empty candidate pool after availability filtering: record round failure and continue.
- Infeasible optimizer result: record solver status and continue.
- Missing optional scout columns: fill with zero if the scout does not exist in that season.

## Testing Strategy

Add focused tests under `src/tests/backtesting/`:

- Loader normalizes a small fixture with raw Cartola column names.
- Status and position mappings are correct.
- Feature generation excludes target and future rounds.
- Baseline predictor handles players with and without history.
- Optimizer respects budget, positions, and selected-player count.
- Runner produces one result row per round and strategy on a tiny synthetic season.
- Metrics compute cumulative advantage against benchmarks correctly.

Use synthetic fixtures for unit tests so tests are fast and independent of the full 2025 dataset.

## Success Criteria

V1 is complete when:

- A command can run the 2025 fixed-budget walk-forward backtest from round 5 onward.
- Outputs are written to `data/08_reporting/backtests/2025/`.
- The selected squads are legal under budget and formation constraints.
- No target-round or future-round actual performance leaks into training features.
- The summary reports model strategy performance against at least one benchmark.
- Tests cover the critical time-boundary and optimizer constraints.

## Future Extensions

After v1 is stable:

- Add multi-season evaluation.
- Add opponent and team-strength features.
- Add XGBoost or LightGBM.
- Add evolving wealth and player valuation simulation.
- Add live API prediction mode for the current season.
- Add external news/search availability checks only as a supplement to Cartola API status.
- Wrap the module in Kedro pipelines if it becomes part of the main project data workflow.
