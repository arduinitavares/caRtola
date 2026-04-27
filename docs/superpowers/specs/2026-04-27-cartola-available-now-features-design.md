# Cartola Available-Now Feature Pack Design

Date: 2026-04-27

## Purpose

Improve the offline Cartola walk-forward backtest by adding four leakage-safe features that can be computed from the existing 2025 player round CSVs. This milestone keeps the same model, optimizer, data source, and public feature-builder API so the before/after comparison isolates feature-engineering signal.

Success is measured by the 2025 diagnostics:

- Random Forest `player_r2 > 0.05`.
- Random Forest beats the baseline by at least 2 average points per round.
- All tests and `scripts/pyrepo-check --all` pass.

If the Random Forest still ties or loses to the baseline, that is an informative result: the existing player-only data is likely exhausted, and the next milestone should add fixture context.

## Scope

In scope:

- Add `prior_points_weighted3`.
- Add `prior_appearance_rate`.
- Add `prior_points_std`.
- Add `club_points_roll3`.
- Add focused unit tests for each feature.
- Rerun the 2025 backtest and inspect diagnostics.

Out of scope:

- Fixture data.
- Home/away or opponent-strength features.
- New model classes.
- DNP classifier or `p_play` adjustment.
- Weighted scout features.
- `prior_positive_scout_ratio`.
- Optimizer changes.
- Public API changes to `build_prediction_frame` or `build_training_frame`.

## Data Flow

`build_prediction_frame(season_df, target_round)` keeps its current public signature. Internally it builds two histories:

- `played_history`: existing `_played_history(season_df, target_round)`, containing only rows where `rodada < target_round` and `entrou_em_campo == True`.
- `all_history`: `season_df[season_df["rodada"] < target_round]`, retaining DNP rows.

It passes both to `_add_prior_features(candidates, played_history, all_history)`.

`build_training_frame` is unchanged. It already calls `build_prediction_frame` for each historical target round, so it automatically receives the new features with the same walk-forward boundary.

## Feature Definitions

### `prior_points_weighted3`

For each player, compute a weighted mean of the last three played appearances before the target round. Use weights `[0.2, 0.3, 0.5]` from oldest to newest. If fewer than three appearances exist, use the matching suffix of the weights and normalize by the sum of used weights.

Default: fill missing values with `prior_points_mean`.

### `prior_appearance_rate`

For each player, compute:

```text
played prior rows / all prior rows
```

The denominator includes DNP rows, so this must use `all_history`, not `played_history`.

Default: fill missing values with `1.0` for players with no prior rows.

### `prior_points_std`

For each player, compute the standard deviation of `pontuacao` over prior played appearances.

Default: fill missing values with `0.0`, covering players with fewer than two prior appearances.

### `club_points_roll3`

For each club, compute total `pontuacao` per club per prior round using only played rows. Then compute a rolling mean over the last three club-round totals with `rolling(3, min_periods=1)`. The feature value for target round N is the latest rolling value available for that club from rounds `< N`.

Default: fill missing values with the global mean club-round total from `played_history`; if history is empty, use `0.0`.

## Implementation Shape

All production changes stay in `src/cartola/backtesting/features.py`:

- Add the four columns to `FEATURE_COLUMNS`.
- Add the four numeric columns to `NUMERIC_PRIOR_COLUMNS`.
- Add `prior_points_weighted3` and `prior_points_std` to `_player_history_features`.
- Add `_weighted_recent_mean(values, weights=(0.2, 0.3, 0.5))`.
- Add `_appearance_history_features(all_history)`.
- Add `_club_history_features(played_history)`.
- Merge appearance and club features inside `_add_prior_features`.
- Add fill defaults inside `_add_prior_features`.

No existing feature names are renamed. The change is additive.

## Leakage Rules

- Every new feature uses only rows with `rodada < target_round`.
- `prior_points_weighted3`, `prior_points_std`, and `club_points_roll3` use post-round values only from completed prior rounds.
- `prior_appearance_rate` uses prior `entrou_em_campo` only.
- No target-round `pontuacao`, raw post-round `preco`, `variacao`, target-round scouts, `media`, or `num_jogos` may enter `FEATURE_COLUMNS`.

## Tests

Add tests in `src/tests/backtesting/test_features.py`:

- `test_prior_points_weighted3_weights_recent_rounds_higher`
- `test_prior_appearance_rate_counts_dnp_correctly`
- `test_prior_points_std_measures_volatility`
- `test_club_points_roll3_captures_team_form`

The tests should use small synthetic season frames and assert exact feature values where possible.

## Verification

Run:

```bash
uv run --frozen pytest src/tests/backtesting/ -v
uv run --frozen --no-dev python -m cartola.backtesting.cli --season 2025 --start-round 5 --budget 100
grep -E 'player_r2|player_correlation' data/08_reporting/backtests/2025/diagnostics.csv | grep ',all,'
uv run --frozen scripts/pyrepo-check --all
```

Commit message:

```text
feat: add weighted form, appearance rate, volatility, and club form features
```
