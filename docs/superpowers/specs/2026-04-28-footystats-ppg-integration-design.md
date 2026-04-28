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
- Add explicit feature-column resolution so `footystats_mode="none"` keeps the existing model contract unchanged.
- Write enough run metadata to distinguish no-FootyStats and PPG runs after the fact.

Out of scope:

- Pre-match xG.
- Odds-derived probabilities.
- Goal environment, corners, cards.
- DNP probability modeling.
- Model architecture changes.
- Strict historical proof that FootyStats values were archived before each historical Cartola deadline.

## Data Contract

The loader reads only `matches` files accepted by the FootyStats compatibility audit.

Required input columns for PPG feature rows:

- `Game Week`
- `home_team_name`
- `away_team_name`
- `Pre-Match PPG (Home)`
- `Pre-Match PPG (Away)`

Required status column for mode validation:

- `status`

The loader must not read or expose post-match outcome columns such as goals, xG after match, cards, shots, corners, or possession.

File selection:

- Default league slug is `brazil-serie-a`.
- The default matches file path is selected by exact season and league slug:
  `data/footystats/{league_slug}-matches-{season}-to-{season}-stats.csv`.
- If the exact file is missing, fail with a clear `FileNotFoundError`.
- If a future CLI supports overriding the matches file path, the loader must still validate that the parsed filename season and league slug match the requested season and league slug.
- Do not rely on stale audit report files. The feature loader must independently validate required columns, team mapping, PPG values, and join uniqueness for the selected source file.

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

Validation rules:

- `Game Week` must parse as positive integers.
- `Game Week` is treated as Cartola `rodada`; there is no offset or fuzzy matching. If FootyStats round coverage does not match the Cartola candidate clubs for a target round, the frame build must fail for that round.
- Required PPG columns must exist and must be non-null for every row used by a training or prediction frame.
- Historical candidate evaluation requires `status == "complete"` for every match row in the selected season file.
- Each normalized match must produce exactly one row for each `(rodada, id_clube)`.
- Duplicate `(rodada, id_clube)` rows are fatal.
- During frame merge, validate `many_to_one`.
- Before fitting or predicting a target round, every candidate `id_clube` must have exactly one FootyStats row for that `rodada`.
- Extra FootyStats rows for clubs with no Cartola candidates are allowed but should be counted in run metadata.
- Cartola clubs with no candidate players in a target round do not require FootyStats rows for that frame.

## Backtest Behavior

Backtest feature integration should be explicit. Add a config option rather than silently enabling FootyStats features when files exist.

Recommended mode:

```python
footystats_mode: Literal["none", "ppg"] = "none"
```

Add an explicit evaluation scope:

```python
footystats_evaluation_scope: Literal["historical_candidate", "live_current"] = "historical_candidate"
```

Behavior:

- `none`: current behavior; no FootyStats features loaded.
- `ppg`: load FootyStats PPG club-side rows and merge them into training/prediction frames.

When `ppg` mode is enabled for comparable historical evaluation:

- The selected season's FootyStats audit classification must be `candidate`.
- A partial/current season such as `2026` must fail in `historical_candidate` scope, even if the loader can technically join rows.
- For a single-season 2025 benchmark, use the 2025 FootyStats match file and current Cartola season data as an exploratory feature ablation.
- If a target-round club has no FootyStats row, fail the run with a clear error. Do not silently fill with neutral values; missing context would make the ablation misleading.

The backtest output path must isolate ablation runs. The measurement commands must not use the default `data/08_reporting/backtests/{season}/` path for both runs.

`run_metadata.json` must include:

- `footystats_mode`
- `footystats_evaluation_scope`
- FootyStats league slug
- FootyStats matches source path
- FootyStats matches source SHA-256
- FootyStats feature columns used
- Missing join keys by round
- Duplicate join keys by round
- Extra FootyStats club rows by round

## Live Behavior

Live/current season use is allowed for `2026` because the goal is actual squad selection, not complete-season metric comparison.

For live prediction:

- Use target-round FootyStats rows even if fixture `status` is `incomplete`, because that means the match is not finished yet.
- `live_current` scope must be explicit.
- `live_current` scope is only valid when `season` equals the configured current year.
- `live_current` must reject target-round rows whose `status` is neither `complete` nor `incomplete`.
- Do not use rows with missing required PPG values.
- Do not use post-match outcome columns.
- If a target-round fixture is missing, duplicated, unmapped, suspended, or missing PPG, fail and report the exact club/round problem.

This keeps live output operational while still preventing silent leakage or fallback-driven recommendations.

## Feature Columns

Add these to the model feature list only when `footystats_mode="ppg"`:

- `footystats_team_pre_match_ppg`
- `footystats_opponent_pre_match_ppg`
- `footystats_ppg_diff`

Do not add `opponent_id_clube` as a model feature. It is a join/debug column only.

`FEATURE_COLUMNS` remains the base list. Add a resolver:

```python
FOOTYSTATS_PPG_FEATURE_COLUMNS: list[str] = [
    "footystats_team_pre_match_ppg",
    "footystats_opponent_pre_match_ppg",
    "footystats_ppg_diff",
]


def feature_columns_for_config(config: BacktestConfig) -> list[str]:
    if config.footystats_mode == "none":
        return FEATURE_COLUMNS
    if config.footystats_mode == "ppg":
        return [*FEATURE_COLUMNS, *FOOTYSTATS_PPG_FEATURE_COLUMNS]
    raise ValueError(f"Unsupported footystats_mode={config.footystats_mode!r}")
```

`RandomForestPointPredictor` must receive the resolved feature list explicitly. It must not import and use the global `FEATURE_COLUMNS` directly for fitting/prediction after this change.

Required model contract:

```python
RandomForestPointPredictor(
    random_seed=config.random_seed,
    feature_columns=feature_columns_for_config(config),
)
```

Tests must prove:

- `footystats_mode="none"` excludes FootyStats columns.
- `footystats_mode="ppg"` includes the three PPG columns.
- The RF model fits/predicts with the explicitly supplied columns.
- Existing `none` mode output frames do not need FootyStats columns.

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
- `footystats_mode="ppg"` fails in `historical_candidate` scope for a partial/current season.
- `footystats_mode="ppg"` can load current-year rows only under explicit `live_current` scope.
- Missing PPG values fail before model fitting.
- Duplicate `(rodada, id_clube)` joins fail before model fitting.
- Run metadata records FootyStats mode, source path, source hash, and feature columns.

## Measurement

Run at least these comparisons:

1. Current no-FootyStats baseline:

```bash
uv run --frozen python -m cartola.backtesting.cli \
  --season 2025 \
  --start-round 5 \
  --budget 100 \
  --fixture-mode exploratory \
  --footystats-mode none \
  --output-root data/08_reporting/backtests/footystats_none
```

2. PPG feature ablation:

```bash
uv run --frozen python -m cartola.backtesting.cli \
  --season 2025 \
  --start-round 5 \
  --budget 100 \
  --fixture-mode exploratory \
  --footystats-mode ppg \
  --footystats-evaluation-scope historical_candidate \
  --output-root data/08_reporting/backtests/footystats_ppg
```

The resulting report directories must be separate:

```text
data/08_reporting/backtests/footystats_none/2025/
data/08_reporting/backtests/footystats_ppg/2025/
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

Do not implement broad xG/odds support in this milestone. The dynamic feature-column resolver and metadata requirements are part of this milestone because they are prerequisites for adding any optional feature set safely.
