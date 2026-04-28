# FootyStats PPG Integration Design

## Purpose

Add the first FootyStats-derived model features using only the two required pre-match-safe fields already proven by the compatibility audit:

- `Pre-Match PPG (Home)`
- `Pre-Match PPG (Away)`

This milestone should answer one question: does adding pre-match team strength context from FootyStats improve Cartola predictions beyond the current fixture-context baseline?

## Core Distinction: Historical vs Live

`2026` is both partial and important.

- In historical evaluation, `2026` is `partial_current`; its metrics are not directly comparable to complete seasons.
- In live squad selection, `2026` is the production season; current-year FootyStats pre-match rows should be usable for upcoming-round recommendations.

The implementation must keep those meanings separate. Comparable backtests should use complete candidate seasons (`2023`, `2024`, `2025`). Live/current prediction can use `2026` FootyStats rows for target-round context if the required pre-match fields and team mapping are available.

## Scope

In scope:

- Load FootyStats match files for a season.
- Normalize each match into two club-side rows:
  - home club row
  - away club row
- Join club-side FootyStats rows to Cartola prediction/training frames by `season`, `rodada`, and `id_clube`.
- Add only PPG features in this milestone:
  - `footystats_team_pre_match_ppg`
  - `footystats_opponent_pre_match_ppg`
  - `footystats_ppg_diff`
- Run ablation backtests against current baselines.
- Preserve live/current-season usage for `2026` without marking it comparable to complete seasons.

Out of scope:

- Pre-match xG.
- Odds-derived probabilities.
- Goal environment, corners, cards.
- DNP probability modeling.
- Model architecture changes.
- Strict historical proof that FootyStats values were archived before each historical Cartola deadline.

## Data Contract

The loader reads only `matches` files accepted by the FootyStats compatibility audit.

Required input columns:

- `Game Week`
- `home_team_name`
- `away_team_name`
- `Pre-Match PPG (Home)`
- `Pre-Match PPG (Away)`

The loader must not read or expose post-match outcome columns such as goals, xG after match, cards, shots, corners, or possession.

Each FootyStats match becomes two rows:

| Column | Home row | Away row |
|---|---|---|
| `rodada` | `Game Week` | `Game Week` |
| `id_clube` | mapped home club ID | mapped away club ID |
| `opponent_id_clube` | mapped away club ID | mapped home club ID |
| `is_home_footystats` | `1` | `0` |
| `footystats_team_pre_match_ppg` | `Pre-Match PPG (Home)` | `Pre-Match PPG (Away)` |
| `footystats_opponent_pre_match_ppg` | `Pre-Match PPG (Away)` | `Pre-Match PPG (Home)` |
| `footystats_ppg_diff` | home PPG - away PPG | away PPG - home PPG |

The club mapping should reuse the audit's normalization and comparison helpers so the feature path and audit path agree.

## Backtest Behavior

Backtest feature integration should be explicit. Add a config option rather than silently enabling FootyStats features when files exist.

Recommended mode:

```python
footystats_mode: Literal["none", "ppg"] = "none"
```

Behavior:

- `none`: current behavior; no FootyStats features loaded.
- `ppg`: load FootyStats PPG club-side rows and merge them into training/prediction frames.

When `ppg` mode is enabled for comparable historical evaluation:

- Use complete candidate seasons only for cross-season claims.
- For a single-season 2025 benchmark, use the 2025 FootyStats match file and current Cartola season data as an exploratory feature ablation.
- If a target-round club has no FootyStats row, fail the run with a clear error. Do not silently fill with neutral values; missing context would make the ablation misleading.

## Live Behavior

Live/current season use is allowed for `2026` because the goal is actual squad selection, not complete-season metric comparison.

For live prediction:

- Use target-round FootyStats rows even if fixture `status` is `incomplete`, because that means the match is not finished yet.
- Do not use rows with missing required PPG values.
- Do not use post-match outcome columns.
- If a target-round fixture is missing, duplicated, unmapped, or missing PPG, fail and report the exact club/round problem.

This keeps live output operational while still preventing silent leakage or fallback-driven recommendations.

## Feature Columns

Add these to the model feature list only when `footystats_mode="ppg"`:

- `footystats_team_pre_match_ppg`
- `footystats_opponent_pre_match_ppg`
- `footystats_ppg_diff`

Do not add `opponent_id_clube` as a model feature. It is a join/debug column only.

## Validation

Tests must cover:

- Loading a FootyStats match file into two club-side rows.
- Home and away PPG assignment.
- Opponent PPG assignment.
- PPG diff sign.
- Team-name mapping reuse.
- Rejection of missing required PPG columns.
- Rejection of duplicate club rows for one `season + rodada + id_clube`.
- Prediction-frame merge for a target round.
- Training-frame merge through walk-forward frame construction.
- `footystats_mode="none"` preserves current behavior.
- `footystats_mode="ppg"` fails when required target-round rows are missing.

## Measurement

Run at least these comparisons:

1. Current no-FootyStats baseline:

```bash
uv run --frozen python -m cartola.backtesting.cli --season 2025 --start-round 5 --budget 100 --fixture-mode exploratory --footystats-mode none
```

2. PPG feature ablation:

```bash
uv run --frozen python -m cartola.backtesting.cli --season 2025 --start-round 5 --budget 100 --fixture-mode exploratory --footystats-mode ppg
```

Success signal:

- RF average squad points improves over the current fixture-context RF.
- Player-level `r2` and correlation do not regress materially.

Failure signal:

- If PPG features do not improve the model, keep the loader but do not proceed to broader FootyStats features until the ablation explains why.

## Implementation Notes

Prefer a new focused module for FootyStats feature loading rather than expanding `footystats_audit.py`.

Suggested module:

```text
src/cartola/backtesting/footystats_features.py
```

The audit module remains responsible for compatibility reporting. The feature module is responsible for audited, model-ready joins.

The feature module may import stable helpers from the audit module, such as `normalize_team_name` and `compare_teams_to_cartola`, but it should not write reports.
