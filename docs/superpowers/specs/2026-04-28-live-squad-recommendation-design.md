# Live Squad Recommendation Design

## Purpose

Add a dedicated single-round squad recommendation workflow for actual Cartola gameplay.

The first deployment target is the current 2026 season using the strongest current no-fixture feature pack:

- `fixture_mode="none"`
- `footystats_mode="ppg"`
- `strategy="random_forest"`

This workflow is not a replacement for the multi-round backtest runner. It answers a different question:

> Given the data visible before a target round, what squad should I pick now?

It should also support leakage-safe historical replay for one round so we can ask:

> If we had run this command before round N, what would it have selected and how did it score?

## Scope

In scope:

- A dedicated single-round recommendation command:
  `scripts/recommend_squad.py`.
- A recommendation module:
  `src/cartola/backtesting/recommendation.py`.
- Required explicit `--target-round`.
- Two modes:
  - `live`
  - `replay`
- Fixed `fixture_mode="none"` for v1.
- Default `footystats_mode="ppg"`.
- Optional `footystats_mode="none"` for debugging.
- Optional `footystats_mode="ppg_xg"` for research, but not recommended by default.
- RandomForest recommendation strategy.
- Baseline and price scores in candidate diagnostics for comparison.
- Output CSV/JSON artifacts under a recommendation-specific reporting tree.
- Metadata proving which data slice, features, and source files were used.

Out of scope:

- Strict fixture integration.
- Exploratory fixture integration.
- Downloading live Cartola market data.
- Downloading live FootyStats files.
- DNP probability modeling.
- Model architecture comparison.
- Multi-round metrics.
- Wealth/patrimônio simulation.
- Auto-submitting a Cartola squad.

## Command Shape

Live recommendation:

```bash
uv run --frozen python scripts/recommend_squad.py \
  --season 2026 \
  --target-round 14 \
  --mode live \
  --budget 100 \
  --footystats-mode ppg \
  --current-year 2026
```

Historical replay:

```bash
uv run --frozen python scripts/recommend_squad.py \
  --season 2026 \
  --target-round 10 \
  --mode replay \
  --budget 100 \
  --footystats-mode ppg \
  --current-year 2026
```

CLI options:

- `--season`: integer season.
- `--target-round`: required positive integer.
- `--mode`: `live | replay`.
- `--budget`: default `100.0`.
- `--project-root`: default `.`.
- `--output-root`: default `data/08_reporting/recommendations`.
- `--footystats-mode`: `none | ppg | ppg_xg`, default `ppg`.
- `--footystats-league-slug`: default `brazil-serie-a`.
- `--footystats-dir`: default `data/footystats`.
- `--current-year`: optional integer, but examples and tests should pass it explicitly.
- `--allow-finalized-live-data`: default false.

Do not expose `--fixture-mode` in v1. The command always behaves as `fixture_mode="none"`.

Do not expose `--strategy` in v1. The optimized recommendation is always based on `random_forest_score`.

## Core Data Boundary

The recommendation workflow may load local raw season files, but future rounds must be cut off immediately after loading.

Required sequence:

```python
season_df = load_season_data(season, project_root=project_root)
visible_season_df = season_df[season_df["rodada"] <= target_round].copy()
```

Every downstream operation must receive only `visible_season_df`:

- mode validation
- finalized-data checks
- training-frame construction
- prediction-frame construction
- FootyStats join diagnostics
- model fitting
- model prediction
- optimization
- output generation

For `target_round=N`:

- training rows use only `rodada < N`;
- candidate rows use only `rodada == N`;
- no `rodada > N` rows may enter validation, feature building, model fitting, prediction, optimization, or outputs.

This applies even when later rounds exist locally.

If `visible_season_df` has no rows for `target_round`, fail with:

```text
Target round N not found in season YYYY data.
```

If no prior rounds exist before `target_round`, fail with:

```text
No training history exists before target round N.
```

## Live vs Replay Semantics

### Live Mode

Live mode is for an open or upcoming Cartola round.

Live mode must not expose target-round actual outcomes in outputs. The command must treat target-round actual/evaluation fields as forbidden output data:

- `pontuacao`
- `actual_points`
- `entrou_em_campo`
- scout columns from `DEFAULT_SCOUT_COLUMNS`

Before prediction, live mode checks whether target-round candidate rows look finalized:

- any target-round `pontuacao` is non-null;
- any target-round `entrou_em_campo` is true;
- any target-round scout column is non-null.

If finalized data is detected and `--allow-finalized-live-data` is absent, fail with a clear error.

If finalized data is detected and `--allow-finalized-live-data` is present:

- the command may run;
- `run_metadata.json` records `finalized_live_data_detected=true`;
- `run_metadata.json` records `allow_finalized_live_data=true`;
- live outputs still suppress all actual/evaluation fields.

`--allow-finalized-live-data` is a debugging escape hatch, not permission to leak actuals into live outputs.

### Replay Mode

Replay mode is for a target round that has already finished.

Replay mode uses the same pre-round prediction path:

1. slice to `rodada <= target_round`;
2. train from `< target_round`;
3. predict candidates from `== target_round`;
4. optimize using `random_forest_score`.

Only after optimization may replay mode attach target-round actuals for the selected squad.

Replay outputs may include:

- selected-player `pontuacao`;
- selected-player `entrou_em_campo`;
- selected-player scout columns;
- squad `actual_points`.

Replay must not use target-round actuals before prediction or optimization.

## FootyStats Recommendation Rules

Recommendation-specific FootyStats loading must not reuse historical-candidate validation blindly.

### Mode `footystats_mode="none"`

- Do not load FootyStats rows.
- Use base feature columns only.
- Useful for debugging recommendation plumbing.

### Mode `footystats_mode="ppg"`

Use the existing PPG features:

- `footystats_team_pre_match_ppg`
- `footystats_opponent_pre_match_ppg`
- `footystats_ppg_diff`

### Mode `footystats_mode="ppg_xg"`

Use PPG plus experimental xG features:

- `footystats_team_pre_match_xg`
- `footystats_opponent_pre_match_xg`
- `footystats_xg_diff`

`ppg_xg` remains experimental because multi-season ablation showed player-fit gains but worse squad points overall.

### Current Season Live Scope

When `season == current_year`, recommendation mode uses current-season FootyStats semantics:

- allow partial season files;
- allow incomplete future weeks;
- require only the rows needed for candidate clubs in rounds used by training and target-round prediction;
- reject missing or duplicate `(rodada, id_clube)` rows for those needed clubs;
- reject missing required feature values for those needed rows;
- reject target-round rows whose match status is neither `complete` nor `incomplete`.

For live target rounds, incomplete target-round FootyStats rows are valid because matches have not finished.

Future weeks must not be used as evidence for accepting or rejecting a recommendation run.

### Historical Replay Scope

For non-current historical seasons, replay should use `historical_candidate` validation:

- exact game-week coverage `1..38`;
- complete match statuses;
- clean team mapping;
- required mode-specific pre-match columns present;
- no duplicate join keys.

For current-year replay, the command should use current-season semantics rather than full historical-candidate validation, because the current season can be partial.

## Recommendation Algorithm

The recommendation workflow reuses existing model and optimizer pieces.

High-level flow:

1. Build `visible_season_df`.
2. Resolve FootyStats rows for the selected mode and recommendation scope.
3. Build training frame:
   `build_training_frame(visible_season_df, target_round, playable_statuses=("Provavel",), fixtures=None, footystats_rows=...)`.
4. Build prediction frame:
   `build_prediction_frame(visible_season_df, target_round, fixtures=None, footystats_rows=...)`.
5. Filter candidates to playable statuses.
6. Fit:
   - `BaselinePredictor`
   - `RandomForestPointPredictor(feature_columns=feature_columns_for_config(config_like_object))`
7. Score candidates:
   - `baseline_score`
   - `random_forest_score`
   - `price_score`
8. Optimize one squad:
   `optimize_squad(scored_candidates, score_column="random_forest_score", config=config_like_object)`.
9. Fail if optimizer status is not `Optimal`.
10. Write outputs.

The recommendation module can use a small recommendation config dataclass, but it must either:

- expose the same fields needed by `feature_columns_for_config` and `optimize_squad`; or
- create a `BacktestConfig` internally with `fixture_mode="none"` and the selected FootyStats settings.

The implementation must not call `run_backtest`.

## Output Directory

Default output path:

```text
data/08_reporting/recommendations/{season}/round-{target_round}/{mode}/
```

Examples:

```text
data/08_reporting/recommendations/2026/round-14/live/
data/08_reporting/recommendations/2026/round-10/replay/
```

The command may overwrite files in the exact target output directory. It must not write to:

```text
data/08_reporting/backtests/
```

## Output Files

### `recommended_squad.csv`

Selected players only.

Required columns in both modes:

- `rodada`
- `id_atleta`
- `apelido`
- `id_clube`
- `nome_clube`
- `posicao`
- `status`
- `preco_pre_rodada`
- `baseline_score`
- `random_forest_score`
- `price_score`
- `predicted_points`

Replay-only additional columns:

- `pontuacao`
- `entrou_em_campo`
- scout columns from `DEFAULT_SCOUT_COLUMNS`

Live mode must not include replay-only columns.

### `candidate_predictions.csv`

Playable target-round candidates.

Required columns in both modes:

- `rodada`
- `id_atleta`
- `apelido`
- `id_clube`
- `nome_clube`
- `posicao`
- `status`
- `preco_pre_rodada`
- `baseline_score`
- `random_forest_score`
- `price_score`

If `footystats_mode` is enabled, include the active FootyStats feature columns.

Live mode must not include:

- `pontuacao`
- `actual_points`
- `entrou_em_campo`
- scout columns from `DEFAULT_SCOUT_COLUMNS`

Replay mode may include actual/evaluation columns for diagnostics.

### `recommendation_summary.json`

Required fields:

```json
{
  "season": 2026,
  "target_round": 14,
  "mode": "live",
  "strategy": "random_forest",
  "formation": "4-3-3",
  "budget": 100.0,
  "optimizer_status": "Optimal",
  "selected_count": 12,
  "budget_used": 99.25,
  "predicted_points": 61.4,
  "actual_points": null,
  "output_directory": "data/08_reporting/recommendations/2026/round-14/live"
}
```

In replay mode, `actual_points` is the selected squad's realized target-round total when available.

In live mode, `actual_points` must be `null`.

### `run_metadata.json`

Required fields:

- `season`
- `target_round`
- `mode`
- `current_year`
- `training_rounds`
- `candidate_round`
- `visible_max_round`
- `fixture_mode`
- `footystats_mode`
- `footystats_evaluation_scope`
- `footystats_league_slug`
- `footystats_matches_source_path`
- `footystats_matches_source_sha256`
- `feature_columns`
- `playable_statuses`
- `formation`
- `budget`
- `random_seed`
- `finalized_live_data_detected`
- `allow_finalized_live_data`
- `optimizer_status`
- `warnings`
- `generated_at_utc`

`generated_at_utc` must be ISO-8601 UTC ending with `Z`.

## Error Handling

Fail loudly for:

- missing target round;
- no training history before target round;
- no playable target-round candidates;
- unsupported recommendation mode;
- unsupported FootyStats mode;
- live finalized data without `--allow-finalized-live-data`;
- missing FootyStats rows for needed clubs;
- duplicate FootyStats rows for needed clubs;
- missing required FootyStats feature values;
- invalid current-year live scope;
- optimizer status other than `Optimal`.

Errors should include season, target round, mode, and the failing key where relevant.

## Testing Requirements

Add focused tests for:

- CLI requires `--target-round`.
- CLI does not expose `--fixture-mode`.
- Future rows greater than target round do not enter training or prediction.
- Replay training uses only rounds `< target_round`.
- Live mode rejects finalized target-round rows by default.
- Live mode with `--allow-finalized-live-data` runs but suppresses actual/evaluation columns.
- Replay mode writes selected actual points only after optimization.
- Current-year live FootyStats validation ignores incomplete future weeks.
- Missing target-round FootyStats club rows fail.
- Duplicate target-round FootyStats club rows fail.
- Output files are written under `data/08_reporting/recommendations/{season}/round-{target_round}/{mode}/`.
- `recommendation_summary.json` has `actual_points=null` in live mode.
- `run_metadata.json` records training rounds, feature columns, source hash, and finalized-data flags.

## Acceptance Criteria

- `uv run --frozen python scripts/recommend_squad.py --season 2026 --target-round N --mode live --budget 100 --footystats-mode ppg --current-year 2026` writes a playable recommendation or fails with a data-quality error.
- `uv run --frozen python scripts/recommend_squad.py --season 2026 --target-round N --mode replay --budget 100 --footystats-mode ppg --current-year 2026` writes a replay recommendation and actual-point evaluation when target-round actuals exist.
- Live outputs never include target-round actual/evaluation columns.
- Replay predictions and optimization remain leakage-safe even when future round files exist locally.
- Existing backtest and ablation commands are unchanged.
- `uv run --frozen scripts/pyrepo-check --all` passes.
