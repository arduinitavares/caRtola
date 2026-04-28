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

**Current Interpretation**
The 2025 fixture-context result showed the first meaningful model lift: RF beat baseline and crossed the `player_r2 > 0.05` threshold, but that 2025 fixture data is still **exploratory reconstruction**, not strict historical proof. The strict system is now built for future/live capture, but we still need actual pre-lock snapshots to run strict evaluations.

The multi-season audit shows the current pipeline is compatible with recent seasons but not all historical data yet:

- `2023`, `2024`, `2025`: load, feature checks, and no-fixture backtests pass.
- `2026`: load, feature checks, and no-fixture backtest pass as a partial current-season smoke test; metrics are not comparable to complete seasons yet.
- `2022`: marked irregular because the raw round layout is unusual.
- `2018`, `2019`, `2020`: structurally complete, but currently fail at load time and need schema compatibility work before they can expand the training/evaluation history.

The next prediction-quality bet is **FootyStats-style match context**, not forcing older Cartola seasons to load first. This is tabular modeling, so more old rows are not automatically better if the historical schema is degraded or partially reconstructed. FootyStats data is more likely to add independent signal if we can source Brasileirão seasons with pre-match-safe match/team fields.

The current `data/footystats/` files appear to be England Premier League 2018-2019 examples, useful for schema inspection but not joinable to Cartola. The next step is an audit that determines whether we have or can add Brazil Serie A / Brasileirão FootyStats files and which columns are safe for walk-forward modeling.

**How To Run Now**
No fixture context:

```bash
uv run --frozen python -m cartola.backtesting.cli --season 2025 --start-round 5 --budget 100 --fixture-mode none
```

Exploratory 2025 fixture context:

```bash
uv run --frozen python -m cartola.backtesting.cli --season 2025 --start-round 5 --budget 100 --fixture-mode exploratory
```

Strict mode only works once strict fixture snapshots/manifests exist:

```bash
uv run --frozen python -m cartola.backtesting.cli --season 2026 --fixture-mode strict
```

Multi-season compatibility audit:

```bash
uv run --frozen python scripts/audit_backtest_compatibility.py --current-year 2026
```

Quality gate:

```bash
uv run --frozen scripts/pyrepo-check --all
```

**Roadmap**
1. Audit FootyStats data for Cartola usefulness:
   - discover all files under `data/footystats/`,
   - classify league/season coverage and identify whether Brazil Serie A / Brasileirão data exists,
   - inspect match, team, and player schemas,
   - classify columns as `pre_match_safe`, `post_match_outcome`, or `unsafe_unknown`,
   - report whether each season can be joined to Cartola by round and club.
2. If Brasileirão FootyStats data is available, design the integration:
   - map FootyStats team names to Cartola `id_clube`,
   - align FootyStats game weeks to Cartola `rodada`,
   - define leakage-safe feature columns,
   - validate that only pre-match-safe fields enter the model.
3. Add the first FootyStats match-context features:
   - home and away pre-match PPG,
   - pre-match xG where available,
   - odds-derived win/draw/loss probabilities where available,
   - pre-match goal environment features.
4. Start capturing strict 2026 pre-lock Cartola fixture snapshots every round.
5. Generate strict fixtures from those snapshots and run strict backtests as data accumulates.
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
9. Add live selection workflow:
   - read open market data,
   - use strict/current fixture snapshot,
   - output recommended squad.
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
