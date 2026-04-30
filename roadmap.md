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

The next prediction-quality bet is now a measured **Cartola matchup-context ablation** on top of FootyStats PPG. The feature path exists and runs on `2023`, `2024`, and `2025`, but it should not be promoted until a paired control/treatment report proves that it improves squad points, not only player-level fit.

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

The xG-over-PPG ablation is also implemented and comparable for `2023`, `2024`, and `2025`, but it should **not** be promoted to the default feature pack:

- aggregate RF average points delta: `-0.6777`;
- aggregate player R² delta: `+0.00445`;
- aggregate player correlation delta: `+0.00468`.

Per-season RF average points deltas for `ppg -> ppg_xg`:

- `2023`: `-0.7500`;
- `2024`: `+2.0521`;
- `2025`: `-3.3353`.

Interpretation: pre-match xG slightly improves player-level fit metrics but hurts squad selection in aggregate and only improves RF average points in one of three seasons. Keep `footystats_mode=ppg_xg` available as an experimental/research mode, but keep `footystats_mode=ppg` as the current recommended no-fixture FootyStats mode.

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
ablation. The new `cartola_matchup_v1` path has been smoke-tested with
`footystats_mode=ppg` and `fixture_mode=exploratory` for all three seasons, but
we still need the paired ablation report before making a product recommendation.

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
  --footystats-mode ppg \
  --current-year 2026
```

Manual two-step live recommendation:

```bash
uv run --frozen python scripts/recommend_squad.py \
  --season 2026 \
  --target-round 14 \
  --mode live \
  --budget 100 \
  --footystats-mode ppg \
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
1. Finish backtest output UX.
   - Add Rich terminal output to `python -m cartola.backtesting.cli`.
   - Show warnings, strategy summary, run metadata, output path, fixture/FootyStats/matchup modes, jobs, effective workers, backend, model `n_jobs`, prediction frame count, and wall-clock seconds.
   - Generate one consolidated interactive Plotly chart at `charts/strategy_performance_by_round.html`.
   - Chart layout: cumulative actual points by strategy, per-round actual points by strategy, and random_forest formation markers by round.
   - The chart should support hover/zoom from a standalone HTML file.
   - Keep this display/report-only: no scoring, optimizer, feature, or CSV/JSON schema changes.
2. Use `scripts/run_live_round.py` for the next 2026 open round and inspect `recommended_squad.csv`, `candidate_predictions.csv`, `run_metadata.json`, and `live_workflow_metadata.json` before making lineup decisions.
3. Keep PPG as the recommended no-fixture FootyStats feature pack; do not enable xG by default.
4. Capture strict pre-lock fixture snapshots every live round with `scripts/capture_strict_round_fixture.py`.
   - Manual v1 command captures snapshot evidence and generates strict `fixtures_strict` CSV/manifest.
   - Future step: integrate strict fixtures into live recommendations as an explicit opt-in mode after several successful live captures.
5. Build a paired matchup-context ablation report:
   - control: `footystats_mode=ppg`, `fixture_mode=exploratory`, `matchup_context_mode=none`;
   - treatment: `footystats_mode=ppg`, `fixture_mode=exploratory`, `matchup_context_mode=cartola_matchup_v1`;
   - seasons: `2023`, `2024`, `2025`;
   - require identical candidate pools and baseline/price results across arms.
6. Decide whether to promote `cartola_matchup_v1` only after the paired report shows positive squad-point lift across the complete historical gate.
7. Audit patrimonio data before changing budget semantics.
   - Verify historical raw data contains reliable pre-round price and post-round price or official variation fields.
   - Verify tecnico rows have the same market fields.
   - Verify DNP/no-play behavior preserves or changes price as expected.
   - Verify whether enough information exists to replay official patrimonio without reverse-engineering Cartola's hidden valuation formula.
8. Add simulated patrimonio only after the audit passes.
   - Add `budget_mode=fixed|simulated_patrimonio`.
   - Keep `fixed` as the current controlled-comparison mode.
   - In `simulated_patrimonio`, start from `--budget`, optimize round N with current patrimonio, then update patrimonio from selected players' and tecnico's official post-round market values.
   - Persist `budget_available`, `budget_used`, `unspent_cash`, `patrimonio_after_round`, and `patrimonio_delta`.
   - Do not apply the captain multiplier to patrimonio unless an official source proves that Cartola does.
9. Defer wider matchup features until v1 is measured:
   - home/away split priors,
   - shorter roll3 variants,
   - odds/goal-environment fields,
   - or DNP probability modeling if selection reliability becomes the bigger live-game bottleneck.
10. Add DNP probability modeling if needed:
    - predict `p_play`,
    - use `expected_points = predicted_points * p_play`.
11. Add model comparison only after features improve:
    - HistGradientBoosting,
    - GradientBoosting,
    - maybe XGBoost/CatBoost later.

**Backfill / Robustness Track**
These items are useful, but they are no longer the next prediction-quality bottleneck:

1. Fix historical loader compatibility for structurally complete failing seasons:
   - inspect 2018, 2019, and 2020 load errors from the compatibility audit JSON,
   - add schema normalization only where needed,
   - rerun the audit until those seasons reach `load_status=ok`.
2. Decide how to handle irregular historical seasons:
   - inspect the 2022 round layout,
   - document whether it should be normalized, excluded, or handled with season-specific rules.
