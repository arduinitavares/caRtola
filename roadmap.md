We now have a solid offline Cartola research/backtesting platform, not yet a “live auto-scaler” product.

**Delivered**
- Python 3.13 + `uv` project setup.
- GitHub Actions quality workflow with repo-local checks.
- `scripts/pyrepo-check --all`: Ruff, ty, Bandit, pytest.
- Walk-forward backtesting pipeline for Cartola.
- Fixed-budget squad optimization.
- Baseline, price, and RandomForest strategies.
- ILP optimizer for Cartola formations.
- Diagnostics CSVs: prediction quality, selection quality, random valid squad comparison, DNP rates, R²/correlation/MAE.
- Leakage fixes:
  - uses `preco_pre_rodada`, not post-round `preco`;
  - scout features use per-round deltas, not cumulative scouts.
- Available-now feature pack:
  - weighted recent points,
  - appearance rate,
  - volatility,
  - club recent form.
- Fixture context features:
  - `is_home`,
  - `opponent_club_points_roll3`.
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

**Current Interpretation**
The 2025 fixture-context result showed the first meaningful model lift: RF beat baseline and crossed the `player_r2 > 0.05` threshold, but that 2025 fixture data is still **exploratory reconstruction**, not strict historical proof. The strict system is now built for future/live capture, but we still need actual pre-lock snapshots to run strict evaluations.

The multi-season audit shows the current pipeline is compatible with recent seasons but not all historical data yet:

- `2023`, `2024`, `2025`: load, feature checks, and no-fixture backtests pass.
- `2026`: load, feature checks, and no-fixture backtest pass as a partial current-season smoke test; metrics are not comparable to complete seasons yet.
- `2022`: marked irregular because the raw round layout is unusual.
- `2018`, `2019`, `2020`: structurally complete, but currently fail at load time and need schema compatibility work before they can expand the training/evaluation history.

The next prediction-quality bet is still **FootyStats-style match context**, not forcing older Cartola seasons to load first. This is tabular modeling, so more old rows are not automatically better if the historical schema is degraded or partially reconstructed. FootyStats data is more likely to add independent signal if we can source Brasileirão seasons with pre-match-safe match/team fields.

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

**How To Run Now**
No fixture context:

```bash
uv run --frozen python -m cartola.backtesting.cli --season 2025 --start-round 5 --budget 100 --fixture-mode none
```

Exploratory 2025 fixture context:

```bash
uv run --frozen python -m cartola.backtesting.cli --season 2025 --start-round 5 --budget 100 --fixture-mode exploratory
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

Live squad recommendation for the current production season:

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
1. Use the new recommendation command for the next 2026 open round and inspect `recommended_squad.csv`, `candidate_predictions.csv`, and `run_metadata.json` before making lineup decisions.
2. Keep PPG as the recommended no-fixture FootyStats feature pack; do not enable xG by default.
3. Start capturing strict 2026 pre-lock Cartola fixture snapshots every round.
4. Generate strict fixtures from those snapshots and integrate strict/current fixture context into the recommendation workflow once snapshots exist.
5. Decide the next narrow feature bet:
   - odds/goal-environment features as a separate ablation over PPG, not stacked blindly after xG;
   - or DNP probability modeling if selection reliability is the bigger live-game bottleneck.
6. Add higher-signal Cartola fixture features:
   - opponent defensive weakness,
   - points conceded by opponent and position,
   - home/away split priors.
7. Add DNP probability modeling:
   - predict `p_play`,
   - use `expected_points = predicted_points * p_play`.
8. Add model comparison only after features improve:
   - HistGradientBoosting,
   - GradientBoosting,
   - maybe XGBoost/CatBoost later.
9. Add live data acquisition around the recommendation command:
   - fetch open market data,
   - verify pre-lock FootyStats/current fixture inputs,
   - archive recommendation outputs per round.
10. Add evolving patrimônio/wealth simulation after prediction quality is trustworthy.

**Backfill / Robustness Track**
These items are useful, but they are no longer the next prediction-quality bottleneck:

1. Fix historical loader compatibility for structurally complete failing seasons:
   - inspect 2018, 2019, and 2020 load errors from the compatibility audit JSON,
   - add schema normalization only where needed,
   - rerun the audit until those seasons reach `load_status=ok`.
2. Decide how to handle irregular historical seasons:
   - inspect the 2022 round layout,
   - document whether it should be normalized, excluded, or handled with season-specific rules.
