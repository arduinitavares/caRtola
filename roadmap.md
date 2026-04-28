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
  - `footystats_mode`: `none | ppg`,
  - dynamic feature-column resolver so FootyStats columns do not affect default runs,
  - leakage-safe pre-match PPG loader,
  - many-to-one join validation by `(rodada, id_clube)`,
  - source file path/hash and join diagnostics in `run_metadata.json`,
  - `live_current` scope rejected by the historical backtest runner until a live workflow exists.

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

The first leakage-safe FootyStats feature integration is now complete. On the 2025 fixture-context benchmark, `footystats_mode=ppg` improved RF from:

- `60.0406` to `61.1415` average points per round,
- `0.054011` to `0.063308` player R²,
- `0.268741` to `0.277921` player correlation.

That is a useful but still modest one-season lift. Before adding more FootyStats columns, we need to validate whether the PPG signal is stable across all complete candidate seasons (`2023`, `2024`, `2025`).

Important distinction:

- For historical comparison, `2026` is `partial_current` and should not be compared directly against complete seasons.
- For actual gameplay, `2026` is the live production season. Current-year FootyStats pre-match rows are useful for generating real squad recommendations, as long as only pre-deadline/pre-match-safe fields enter the model and missing target-round fixture context fails loudly.

**How To Run Now**
No fixture context:

```bash
uv run --frozen python -m cartola.backtesting.cli --season 2025 --start-round 5 --budget 100 --fixture-mode none
```

Exploratory 2025 fixture context:

```bash
uv run --frozen python -m cartola.backtesting.cli --season 2025 --start-round 5 --budget 100 --fixture-mode exploratory
```

FootyStats PPG ablation, isolated control:

```bash
uv run --frozen python -m cartola.backtesting.cli \
  --season 2025 \
  --start-round 5 \
  --budget 100 \
  --fixture-mode exploratory \
  --footystats-mode none \
  --output-root data/08_reporting/backtests/footystats_none
```

FootyStats PPG ablation, isolated treatment:

```bash
uv run --frozen python -m cartola.backtesting.cli \
  --season 2025 \
  --start-round 5 \
  --budget 100 \
  --fixture-mode exploratory \
  --footystats-mode ppg \
  --footystats-evaluation-scope historical_candidate \
  --footystats-league-slug brazil-serie-a \
  --current-year 2026 \
  --output-root data/08_reporting/backtests/footystats_ppg
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

Quality gate:

```bash
uv run --frozen scripts/pyrepo-check --all
```

**Roadmap**
1. Run a multi-season FootyStats PPG ablation report:
   - seasons: `2023`, `2024`, `2025`,
   - modes: `footystats_mode=none` vs `footystats_mode=ppg`,
   - isolated output roots per season/mode,
   - summary table with RF average points, baseline average, RF-vs-baseline delta, player R², player correlation, and PPG-vs-control delta,
   - mark `2026` only as a partial/current-season smoke test, not comparable historical evidence.
2. Decide whether PPG remains in the model based on multi-season stability:
   - keep if it improves most complete candidate seasons or improves the aggregate without degrading a season badly,
   - revise or drop if the 2025 lift does not generalize.
3. Add broader FootyStats match-context features after the multi-season PPG check:
   - pre-match xG where available,
   - odds-derived win/draw/loss probabilities where available,
   - pre-match goal environment features.
4. Run broader FootyStats ablation backtests:
   - baseline fixture context only,
   - PPG-only FootyStats features,
   - PPG + xG,
   - PPG + xG + odds/goal environment.
5. Start capturing strict 2026 pre-lock Cartola fixture snapshots every round.
6. Generate strict fixtures from those snapshots and run strict backtests as data accumulates.
7. Add higher-signal Cartola fixture features:
   - opponent defensive weakness,
   - points conceded by opponent and position,
   - home/away split priors.
8. Add DNP probability modeling:
   - predict `p_play`,
   - use `expected_points = predicted_points * p_play`.
9. Add model comparison only after features improve:
   - HistGradientBoosting,
   - GradientBoosting,
   - maybe XGBoost/CatBoost later.
10. Add live selection workflow:
   - read open market data,
   - use strict/current fixture snapshot,
   - output recommended squad.
11. Add evolving patrimônio/wealth simulation after prediction quality is trustworthy.

**Backfill / Robustness Track**
These items are useful, but they are no longer the next prediction-quality bottleneck:

1. Fix historical loader compatibility for structurally complete failing seasons:
   - inspect 2018, 2019, and 2020 load errors from the compatibility audit JSON,
   - add schema normalization only where needed,
   - rerun the audit until those seasons reach `load_status=ok`.
2. Decide how to handle irregular historical seasons:
   - inspect the 2022 round layout,
   - document whether it should be normalized, excluded, or handled with season-specific rules.
