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

**Current Interpretation**
The 2025 fixture-context result showed the first meaningful model lift: RF beat baseline and crossed the `player_r2 > 0.05` threshold, but that 2025 fixture data is still **exploratory reconstruction**, not strict historical proof. The strict system is now built for future/live capture, but we still need actual pre-lock snapshots to run strict evaluations.

**How To Run Now**
No fixture context:

```bash
uv run python -m cartola.backtesting.cli --season 2025 --start-round 5 --budget 100 --fixture-mode none
```

Exploratory 2025 fixture context:

```bash
uv run python -m cartola.backtesting.cli --season 2025 --start-round 5 --budget 100 --fixture-mode exploratory
```

Strict mode only works once strict fixture snapshots/manifests exist:

```bash
uv run python -m cartola.backtesting.cli --season 2026 --fixture-mode strict
```

**Roadmap**
1. Start capturing strict 2026 pre-lock fixture snapshots every round.
2. Generate strict fixtures from those snapshots and run strict backtests as data accumulates.
3. Add higher-signal fixture features:
   - opponent defensive weakness,
   - points conceded by opponent and position,
   - home/away split priors.
4. Add DNP probability modeling:
   - predict `p_play`,
   - use `expected_points = predicted_points * p_play`.
5. Add model comparison only after features improve:
   - HistGradientBoosting,
   - GradientBoosting,
   - maybe XGBoost/CatBoost later.
6. Add live selection workflow:
   - read open market data,
   - use strict/current fixture snapshot,
   - output recommended squad.
7. Add evolving patrimônio/wealth simulation after prediction quality is trustworthy.