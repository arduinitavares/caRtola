We now have a solid offline Cartola research/backtesting platform, not yet a “live auto-scaler” product.

**Delivered**
- Python 3.13 + `uv` project setup.
- GitHub Actions quality workflow with repo-local checks.
- `scripts/pyrepo-check --all`: Ruff, ty, Bandit, pytest.
- Walk-forward backtesting pipeline for Cartola.
- Fixed-budget squad optimization with standard Cartola 2026 scoring.
- Baseline, price, and RandomForest strategies.
- ILP optimizer searches all official Cartola formations.
- Captain-aware optimizer:
  - selects one non-tecnico captain inside the MILP;
  - applies the `1.5x` captain multiplier to round-level predicted and actual totals;
  - keeps selected-player `predicted_points` as the raw per-athlete model score;
  - reports EV, safe, and upside captain-policy diagnostics on the same selected squad.
- Diagnostics CSVs: prediction quality, selection quality, random valid squad comparison, DNP rates, R²/correlation/MAE.
- Leakage fixes:
  - uses `preco_pre_rodada`, not post-round `preco`;
  - scout features use per-round deltas, not cumulative scouts.
- Available-now feature pack:
  - weighted recent points,
  - appearance rate,
  - volatility,
  - club recent form.
- 2025 reconstructed fixture files for exploratory analysis.
- Strict no-leakage fixture infrastructure:
  - pre-lock Cartola snapshot capture,
  - strict canonical fixture generation,
  - manifest/hash/timing/path validation,
  - `fixture_mode`: `none`, `exploratory`, `strict`,
  - `run_metadata.json`,
  - strict alignment policy `fail | exclude_round`.
- Multi-season compatibility audit:
  - discovers every local raw season under `data/01_raw/`,
  - classifies complete, irregular, and current partial seasons,
  - runs loader, feature-frame, and no-fixture backtest smoke checks per season,
  - writes isolated reports under `data/08_reporting/backtests/compatibility/`,
  - keeps normal backtest outputs untouched.
- FootyStats compatibility audit:
  - audits local `data/footystats/` Brazil Serie A files,
  - validates filename/year/table shape,
  - validates safe pre-match columns,
  - checks team-name mapping against Cartola club IDs,
  - classifies `2023`, `2024`, and `2025` as integration candidates,
  - classifies `2026` as partial/current-season context.
- First FootyStats model integration:
  - `footystats_mode`: `none | ppg | ppg_xg`,
  - dynamic feature-column resolver so FootyStats columns do not affect default runs,
  - leakage-safe pre-match PPG and xG loader,
  - many-to-one join validation by `(rodada, id_clube)`,
  - ignores malformed Cartola rows without a real club identity when validating FootyStats join keys,
  - source file path/hash and join diagnostics in `run_metadata.json`,
  - `live_current` scope rejected by the historical backtest runner until a live workflow exists.
- Multi-season FootyStats ablation report:
  - compares paired control/treatment FootyStats modes for candidate seasons,
  - default comparison remains `footystats_mode=none` vs `footystats_mode=ppg`,
  - supports `footystats_mode=ppg` vs `footystats_mode=ppg_xg`,
  - writes CSV/JSON summaries and isolated per-season/mode runs under `data/08_reporting/backtests/footystats_ablation/`.
- Single-round squad recommendation workflow:
  - `scripts/recommend_squad.py` for `live` and `replay` modes,
  - hard data boundary at `rodada <= target_round`,
  - RF training uses only rounds `< target_round`,
  - `fixture_mode=none` fixed for v1 recommendations,
  - `footystats_mode=ppg` available for current-year/live usage,
  - replay mode can evaluate actual points after optimization,
  - live mode suppresses actual/scout output columns and rejects finalized target-round data unless explicitly allowed.
- Live market round capture:
  - `scripts/capture_market_round.py` writes the open market round CSV for live recommendations,
  - validates current-year/open-market scope,
  - sanitizes target-round outcome fields,
  - publishes CSV and `.capture.json` with safe overwrite rules.
- One-command live round workflow:
  - `scripts/run_live_round.py` captures or validates the open market round,
  - defaults to `capture_policy=fresh`,
  - uses the captured `rodada_atual` as the recommendation target,
  - archives every recommendation under `runs/run_started_at=...`,
  - links recommendation metadata back to the capture CSV/hash/metadata.
- Matchup fixture coverage audit:
  - `scripts/audit_matchup_fixture_coverage.py` checks whether requested seasons have fixture context for every played club-round,
  - prefers strict fixture CSVs with valid manifests and falls back to exploratory fixture CSVs,
  - reports missing, duplicate, and extra `(rodada, id_clube)` fixture-context keys,
  - writes `matchup_fixture_coverage.csv/json` under `data/08_reporting/fixtures/`,
  - currently reports `ready_for_matchup_context` for `2023`, `2024`, and `2025`.
- First Cartola matchup-context integration:
  - `matchup_context_mode`: `none | cartola_matchup_v1`,
  - kept separate from `footystats_mode`,
  - requires `fixture_mode=exploratory` or `fixture_mode=strict`,
  - adds narrow roll5 Cartola matchup features only when explicitly enabled:
    `matchup_is_home`,
    `matchup_opponent_allowed_points_roll5`,
    `matchup_opponent_allowed_position_points_roll5`,
    `matchup_club_position_points_roll5`,
    `matchup_opponent_allowed_position_count`,
    `matchup_club_position_count`,
  - excludes raw opponent IDs from model features,
  - records matchup mode and feature columns in `run_metadata.json`.
- Backtest performance engine:
  - builds per-round prediction frames once per run with an in-memory `RoundFrameStore`,
  - exposes `--jobs` for target-round parallelism,
  - uses thread-based workers with parent-owned aggregation/writes,
  - forces RF `n_jobs=1` when `--jobs > 1` to avoid nested parallelism,
  - records cache, worker, backend, thread-env, and wall-clock metadata in `run_metadata.json`,
  - keeps report semantics and scoring unchanged.
- Backtest output UX:
  - Rich terminal summary for `python -m cartola.backtesting.cli`,
  - warnings, strategy summary, run details, output path, fixture/FootyStats/matchup modes, jobs, effective workers, backend, model `n_jobs`, prediction-frame count, and wall-clock seconds,
  - standalone interactive Plotly chart at `charts/strategy_performance_by_round.html`,
  - chart traces for cumulative actual points, per-round actual points, and RandomForest formation markers.
- Controlled model/feature experiment runner:
  - `scripts/run_model_experiments.py`,
  - fixed v1 sklearn model registry: `random_forest`, `extra_trees`, `hist_gradient_boosting`, and `ridge`,
  - fixed feature packs: `ppg`, `ppg_xg`, `ppg_matchup`, and `ppg_xg_matchup`,
  - production-parity group for no-fixture comparisons,
  - matchup-research group for exploratory fixture/matchup-context comparisons,
  - private experiment-only primary model strategy support without exposing `--model-id` in the normal backtest CLI,
  - sequential child backtests; `--jobs` only controls per-child target-round parallelism,
  - source/candidate/solver comparability signatures that fail closed before ranking,
  - aggregate ranked summary by model/feature config,
  - prediction metrics, calibration deciles, per-season summary, metadata, comparability report, Markdown report, and HTML report artifacts.
- Experiment observability:
  - durable SQLite experiment index under `data/08_reporting/experiments/experiment_index.sqlite`,
  - optional tracker adapter boundary for future MLflow/local tracking,
  - best-effort tracking warnings that do not change experiment success semantics,
  - artifact-pointer policy so large child CSVs remain in the report tree instead of being duplicated into trackers.
- Production model selection for live recommendations:
  - `scripts/recommend_squad.py` and `scripts/run_live_round.py` now accept `--model-id`,
  - live/replay recommendations can use any model in the controlled registry,
  - current best no-fixture candidate can be deployed directly instead of being experiment-only.
- Constrained Ridge alpha tuning runner:
  - `scripts/run_ridge_tuning.py`,
  - fixed, predeclared Ridge alpha matrix: `0.01`, `0.03`, `0.1`, `0.3`, `1.0`, `3.0`, `10.0`, `30.0`, `100.0`, and `300.0`,
  - evaluates both `ppg` and `ppg_xg` for every alpha so xG is not advantaged only because it received tuning attention,
  - uses private experiment-only Ridge `model_params` plumbing while keeping normal backtest and live defaults unchanged,
  - screen stage ranks the full fixed matrix and final stage reruns the primary incumbent, secondary control, and top challengers,
  - writes `ranked_summary.csv`, `prediction_metrics.csv`, `calibration_deciles.csv`, `comparability_report.json`, `promotion_report.json`, `tuning_generation_manifest.json`, Markdown, and HTML reports under `data/08_reporting/experiments/model_tuning/`,
  - refuses promotion when final reruns are skipped, comparison controls drift, comparability fails, required metrics are missing, or lift is below the practical threshold.
- Standard scoring metadata:
  - `scoring_contract_version=cartola_standard_2026_v1`,
  - `captain_scoring_enabled=True`,
  - `captain_multiplier=1.5`,
  - `formation_search=all_official_formations`,
  - report readers reject old/mixed-contract backtest outputs.

**Current Interpretation**
The 2025 fixture-context result showed the first meaningful model lift: RF beat baseline and crossed the `player_r2 > 0.05` threshold, but that 2025 fixture data is still **exploratory reconstruction**, not strict historical proof. The strict system is now built for future/live capture, but we still need actual pre-lock snapshots to run strict evaluations.

The multi-season audit shows the current pipeline is compatible with recent seasons but not all historical data yet:

- `2023`, `2024`, `2025`: load, feature checks, and no-fixture backtests pass.
- `2026`: load, feature checks, and no-fixture backtest pass as a partial current-season smoke test; metrics are not comparable to complete seasons yet.
- `2022`: marked irregular because the raw round layout is unusual.
- `2018`, `2019`, `2020`: structurally complete, but currently fail at load time and need schema compatibility work before they can expand the training/evaluation history.

The first production-parity model/feature experiment is complete for `2023`,
`2024`, and `2025` with `fixture_mode=none`, `start_round=5`, and
`budget=100`.

Best aggregate result:

- `ridge + ppg_xg`;
- total actual points: `6515.24`;
- current baseline `random_forest + ppg`: `6029.86`;
- aggregate lift: `+485.38`;
- average lift: `+4.76` points per round;
- improved seasons: `3 / 3`;
- promotion status: `passes_v1_guardrails`.

Second result:

- `ridge + ppg`;
- total actual points: `6382.02`;
- aggregate lift: `+352.16`;
- improved seasons: `3 / 3`;
- promotion status: `passes_v1_guardrails`.

Interpretation: for no-fixture production, `ridge + ppg_xg` is now the best
candidate profile. The result does **not** prove that matchup context is useful,
because this group intentionally used `fixture_mode=none`. The next
prediction-quality bet is the matchup-research group plus a constrained
hyperparameter/model-spec experiment around the winning sklearn families.

The constrained Ridge tuning runner is now implemented and smoke-tested on a
single 2025 screen-only run. That smoke run validates the real backtest path,
artifact writing, fixed candidate matrix, and `--skip-final-rerun` promotion
guard. It is **not** promotion evidence: the full `2023,2024,2025` run with
final reruns still needs to be executed before changing the live model profile.

The FootyStats compatibility audit is now implemented and the current `data/footystats/` files are Brazil Serie A seasons, not sample EPL data. The audit result is:

- `2023`, `2024`, `2025`: `integration_status=candidate`.
- `2026`: `integration_status=partial_current` because the file contains incomplete/suspended fixtures.
- Team mapping is clean for all audited seasons.
- Required safe columns are present: `Pre-Match PPG (Home)` and `Pre-Match PPG (Away)`.
- Optional safe columns include pre-match xG, odds, goal environment, corners, and cards fields.

The first leakage-safe FootyStats feature integration is complete. On the 2025 fixture-context benchmark, `footystats_mode=ppg` improved RF from:

- `60.0406` to `61.1415` average points per round,
- `0.054011` to `0.063308` player R²,
- `0.268741` to `0.277921` player correlation.

That was a useful but still modest one-season lift. The multi-season ablation report is now implemented as a no-fixture paired comparison, first used for control-vs-PPG and now generalized for PPG-vs-PPG+xG.

The 2023 join gap has been investigated. Root cause: two Cartola 2023 round-18 coach rows have `status=Nulo`, missing `nome_clube`, and placeholder `id_clube=1`. They are not real club identities and should not require FootyStats match rows. The join validation now ignores rows without a real club identity.

After the fix, the no-fixture multi-season ablation is comparable for `2023`, `2024`, and `2025`:

- included seasons: `2023`, `2024`, `2025`;
- aggregate RF average points delta: `+3.3427`;
- aggregate player R² delta: `+0.0209`;
- aggregate player correlation delta: `+0.0271`;
- control RF minus baseline: `+0.7601`;
- treatment RF minus baseline: `+4.1028`.

Per-season RF average points deltas:

- `2023`: `+5.9279`;
- `2024`: `+1.5612`;
- `2025`: `+2.5391`.

Interpretation: keep FootyStats pre-match PPG. It generalizes across the currently comparable candidate seasons and is now the strongest no-fixture feature addition.

The older RF-only xG-over-PPG ablation is also implemented and comparable for `2023`, `2024`, and `2025`, but it should **not** by itself promote xG to the default feature pack:

- aggregate RF average points delta: `-0.6777`;
- aggregate player R² delta: `+0.00445`;
- aggregate player correlation delta: `+0.00468`.

Per-season RF average points deltas for `ppg -> ppg_xg`:

- `2023`: `-0.7500`;
- `2024`: `+2.0521`;
- `2025`: `-3.3353`.

Interpretation: pre-match xG slightly improved player-level fit metrics but hurt
RF squad selection in aggregate. The later production-parity experiment changed
the model context: `ridge + ppg_xg` beat `ridge + ppg` overall. Therefore xG is
not a universal default for every model, but it is part of the current best
no-fixture production candidate.

Important distinction:

- For historical comparison, `2026` is `partial_current` and should not be compared directly against complete seasons.
- For actual gameplay, `2026` is the live production season. Current-year FootyStats pre-match rows are useful for generating real squad recommendations, as long as only pre-deadline/pre-match-safe fields enter the model and missing target-round fixture context fails loudly.

The first recommendation workflow is now implemented. It is intentionally narrower than
the backtest runner: it generates one target-round squad, does not use fixtures yet,
and writes a replay/live audit trail under `data/08_reporting/recommendations/`.

The optimizer now matches the standard Cartola 2026 lineup contract: 11 players
plus one tecnico, with one of the 11 players marked as captain. The captain is
not an extra slot and the tecnico cannot be captain. For v1 there is no public
legacy scoring mode, no fixed-formation public config, and no configurable captain
multiplier.

The matchup fixture coverage audit is now implemented. Current result for
`2023,2024,2025` is `ready_for_matchup_context`:

- `2023`: complete historical season with full exploratory fixture coverage.
- `2024`: complete historical season with full exploratory fixture coverage.
- `2025`: complete historical season with full exploratory fixture coverage.

Interpretation: the data gate is clear for a proper multi-season matchup-context
experiment. The new `cartola_matchup_v1` path has been smoke-tested with
`footystats_mode=ppg` and `fixture_mode=exploratory` for all three seasons, but
we still need the controlled model/feature experiment report before making a
product recommendation.

The backtest runner now uses the standard Cartola 2026 scoring contract:

- 11 players plus 1 tecnico are selected;
- the tecnico is included in budget, predicted totals, and actual totals;
- one non-tecnico selected player is captain;
- round-level `predicted_points` and `actual_points` include the captain multiplier;
- selected-player `predicted_points` remains the raw per-athlete model score.

The backtest budget is still fixed per round. A run with `--budget 100` means
every round is optimized independently with at most C$ 100. It does **not** yet
simulate patrimonio growth from previous rounds.

Official Globo/ge documentation confirms that patrimonio changes through
selected asset price movement, not directly through total lineup points. The
exact price-variation formula is not publicly documented; official guidance
uses qualitative rules and the PRO "Minimo Para Valorizar" concept. Therefore,
future patrimonio simulation should replay official pre/post market prices or
official variation fields instead of reverse-engineering a hidden formula.

**How To Run Now**
No fixture context:

```bash
uv run --frozen python -m cartola.backtesting.cli --season 2025 --start-round 5 --budget 100 --fixture-mode none
```

Exploratory 2025 fixture context:

```bash
uv run --frozen python -m cartola.backtesting.cli --season 2025 --start-round 5 --budget 100 --fixture-mode exploratory
```

FootyStats PPG plus Cartola matchup-context v1:

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
  --output-root data/08_reporting/backtests/matchup_context_single
```

Production-parity model/feature experiment:

```bash
uv run --frozen python scripts/run_model_experiments.py \
  --group production-parity \
  --seasons 2023,2024,2025 \
  --start-round 5 \
  --budget 100 \
  --current-year 2026 \
  --jobs 12
```

Matchup-research model/feature experiment:

```bash
uv run --frozen python scripts/run_model_experiments.py \
  --group matchup-research \
  --seasons 2023,2024,2025 \
  --start-round 5 \
  --budget 100 \
  --current-year 2026 \
  --jobs 12
```

Constrained Ridge alpha tuning experiment:

```bash
uv run --frozen python scripts/run_ridge_tuning.py \
  --seasons 2023,2024,2025 \
  --start-round 5 \
  --budget 100 \
  --current-year 2026 \
  --jobs 12
```

Fast implementation smoke for the tuning runner:

```bash
uv run --frozen python scripts/run_ridge_tuning.py \
  --seasons 2025 \
  --start-round 5 \
  --budget 100 \
  --current-year 2026 \
  --jobs 4 \
  --skip-final-rerun
```

The experiment runner writes outputs under:

```text
data/08_reporting/experiments/model_feature/<experiment_id>/
```

The Ridge tuning runner writes outputs under:

```text
data/08_reporting/experiments/model_tuning/<experiment_id>/
```

Start with `ranked_summary.csv`, `per_season_summary.csv`,
`prediction_metrics.csv`, `calibration_deciles.csv`,
`comparability_report.json`, and `experiment_metadata.json`.
For Ridge tuning, also inspect `promotion_report.json` before changing any
production recommendation default.

Single-season no-fixture FootyStats PPG backtest:

```bash
uv run --frozen python -m cartola.backtesting.cli \
  --season 2025 \
  --start-round 5 \
  --budget 100 \
  --fixture-mode none \
  --footystats-mode ppg \
  --footystats-evaluation-scope historical_candidate \
  --footystats-league-slug brazil-serie-a \
  --current-year 2026 \
  --output-root data/08_reporting/backtests/footystats_ppg_single
```

Strict mode only works once strict fixture snapshots/manifests exist:

```bash
uv run --frozen python -m cartola.backtesting.cli --season 2026 --fixture-mode strict
```

Multi-season compatibility audit:

```bash
uv run --frozen python scripts/audit_backtest_compatibility.py --current-year 2026
```

FootyStats compatibility audit:

```bash
uv run --frozen python scripts/audit_footystats_compatibility.py --current-year 2026
```

Matchup fixture coverage audit:

```bash
uv run --frozen python scripts/audit_matchup_fixture_coverage.py \
  --seasons 2023,2024,2025 \
  --current-year 2026
```

Multi-season FootyStats PPG ablation report:

```bash
uv run --frozen python scripts/run_footystats_ablation.py \
  --seasons 2023,2024,2025 \
  --start-round 5 \
  --budget 100 \
  --current-year 2026 \
  --control-footystats-mode none \
  --treatment-footystats-mode ppg \
  --force
```

Multi-season FootyStats xG-over-PPG ablation report:

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

Capture the open market round for the current production season:

```bash
uv run --frozen python scripts/capture_market_round.py \
  --season 2026 \
  --auto \
  --current-year 2026
```

Live squad recommendation for the current production season:

```bash
uv run --frozen python scripts/run_live_round.py \
  --season 2026 \
  --budget 100 \
  --model-id ridge \
  --footystats-mode ppg_xg \
  --current-year 2026
```

Manual two-step live recommendation:

```bash
uv run --frozen python scripts/recommend_squad.py \
  --season 2026 \
  --target-round 14 \
  --mode live \
  --budget 100 \
  --model-id ridge \
  --footystats-mode ppg_xg \
  --current-year 2026
```

Replay a completed current-season round without looking past that round:

```bash
uv run --frozen python scripts/recommend_squad.py \
  --season 2026 \
  --target-round 10 \
  --mode replay \
  --budget 100 \
  --footystats-mode ppg \
  --current-year 2026
```

Quality gate:

```bash
uv run --frozen scripts/pyrepo-check --all
```

**Roadmap**
1. Treat `ridge + ppg_xg + fixture_mode=none` as the current best no-fixture live candidate.
   - Keep `random_forest + ppg` available as the historical baseline.
   - Continue to record the chosen model id and feature mode in every recommendation output.
   - Do not claim matchup-context production value from this result.
2. Run the matchup-research model/feature experiment.
   - Seasons: `2023`, `2024`, `2025`.
   - Group: `matchup-research`.
   - Purpose: decide whether `cartola_matchup_v1` is worth promoting into the next strict-fixture integration design.
   - Baseline: `random_forest + ppg + fixture_mode=exploratory + matchup_context_mode=none`.
   - Treat this as research evidence only, not strict live proof.
3. Run the constrained Ridge tuning experiment before adding external model libraries.
   - Spec: `docs/superpowers/specs/2026-05-01-constrained-ridge-tuning-design.md`.
   - Implementation exists; the remaining step is a full `2023,2024,2025` execution with final reruns enabled.
   - Use the generated `promotion_report.json` as the authority for whether a tuned candidate can replace `ridge + ppg_xg + alpha=1.0`.
   - Keep `ridge + ppg_xg + alpha=1.0` as the live no-fixture profile unless the full tuning run clears final-rerun reproducibility, exact comparability, null-metric, and practical-lift gates.
   - Defer RandomForest/ExtraTrees tuning until calibration wrappers are designed; the first production-parity result suggests tree overprediction is structural, not just a small hyperparameter miss.
   - Defer HGB, Optuna, XGBoost, LightGBM, and CatBoost until the Ridge tuning baseline is measured.
4. Interpret experiment outputs before building broader modeling features.
   - Start with `ranked_summary.csv`.
   - Check `per_season_summary.csv` for season robustness.
   - Check `prediction_metrics.csv` and `calibration_deciles.csv` to understand whether squad lift came from better ranking or just noisy optimization.
   - Check `comparability_report.json`; do not trust any ranking if comparability failed.
5. Make one model/feature decision after matchup and constrained tuning:
   - keep `ridge + ppg_xg` as the no-fixture live profile;
   - switch to a tuned sklearn variant only if it clears the same guardrails;
   - write a strict fixture integration spec if matchup context wins;
   - reject matchup context for now and focus on calibration/model diagnostics.
6. Use `scripts/run_live_round.py` for each 2026 open round and inspect `recommended_squad.csv`, `candidate_predictions.csv`, `run_metadata.json`, and `live_workflow_metadata.json` before making lineup decisions.
7. Capture strict pre-lock fixture snapshots every live round with `scripts/capture_strict_round_fixture.py`.
   - Manual v1 command captures snapshot evidence and generates strict `fixtures_strict` CSV/manifest.
   - Future step: integrate strict fixtures into live recommendations as an explicit opt-in mode after several successful live captures.
8. Audit patrimonio data before changing budget semantics.
   - Verify historical raw data contains reliable pre-round price and post-round price or official variation fields.
   - Verify tecnico rows have the same market fields.
   - Verify DNP/no-play behavior preserves or changes price as expected.
   - Verify whether enough information exists to replay official patrimonio without reverse-engineering Cartola's hidden valuation formula.
9. Add simulated patrimonio only after the audit passes.
   - Add `budget_mode=fixed|simulated_patrimonio`.
   - Keep `fixed` as the current controlled-comparison mode.
   - In `simulated_patrimonio`, start from `--budget`, optimize round N with current patrimonio, then update patrimonio from selected players' and tecnico's official post-round market values.
   - Persist `budget_available`, `budget_used`, `unspent_cash`, `patrimonio_after_round`, and `patrimonio_delta`.
   - Do not apply the captain multiplier to patrimonio unless an official source proves that Cartola does.
10. Defer wider matchup features until v1 is measured:
   - home/away split priors,
   - shorter roll3 variants,
   - odds/goal-environment fields,
   - or DNP probability modeling if selection reliability becomes the bigger live-game bottleneck.
11. Add DNP probability modeling if needed:
    - predict `p_play`,
    - use `expected_points = predicted_points * p_play`.
12. Defer external model libraries until the constrained sklearn tuning pass is measured.
    - XGBoost is technically compatible through `XGBRegressor` and its scikit-learn API.
    - Possible later candidates: XGBoost, CatBoost, or LightGBM.
    - Add one external model family at a time, with fixed specs and dependency/runtime tracking.
    - Do not start broad grid search over external libraries before the Ridge/RF tuning baseline is established.

**Backfill / Robustness Track**
These items are useful, but they are no longer the next prediction-quality bottleneck:

1. Fix historical loader compatibility for structurally complete failing seasons:
   - inspect 2018, 2019, and 2020 load errors from the compatibility audit JSON,
   - add schema normalization only where needed,
   - rerun the audit until those seasons reach `load_status=ok`.
2. Decide how to handle irregular historical seasons:
   - inspect the 2022 round layout,
   - document whether it should be normalized, excluded, or handled with season-specific rules.
