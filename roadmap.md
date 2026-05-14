We now have a solid offline Cartola research/backtesting platform, not yet a “live auto-scaler” product.

**Delivered**
- Python 3.13 + `uv` project setup.
- GitHub Actions quality workflow with repo-local checks.
- `scripts/pyrepo-check --all`: Ruff, ty, Bandit, pytest.
- Walk-forward backtesting pipeline for Cartola.
- Moving-budget squad optimization with standard Cartola 2026 scoring.
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
  - defaults to `fixture_mode=none`,
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
- Strict/live matchup recommendation integration:
  - `scripts/recommend_squad.py` and `scripts/run_live_round.py` now accept `--fixture-mode none|strict` and `--matchup-context-mode none|cartola_matchup_v1`,
  - live matchup context is opt-in and requires `fixture_mode=strict`,
  - strict fixture CSV/manifests are loaded from `data/01_raw/fixtures_strict/{season}/`,
  - missing strict fixture evidence raises instead of falling back to no-fixture or exploratory reconstruction,
  - live target-round fixture coverage is validated against candidate clubs, while historical training rounds keep strict played-club alignment checks,
  - recommendation metadata records fixture mode, matchup mode, strict manifest paths/hashes, generator versions, and feature columns,
  - CLI defaults are now the promoted no-fixture live profile:
    `xgboost_depth2_l2_heavy`, `ppg_xg`, `fixture_mode=none`, and
    `matchup_context_mode=none`.
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
  - historical multi-round optimization is sequential because each strategy has a stateful moving-budget path,
  - `--budget` now means initial budget for historical backtests and experiments,
  - `--jobs` remains accepted for compatibility/metadata, but target-round workers are disabled under moving-budget semantics,
  - records budget policy, initial/final/min budget, drawdown, constrained rounds, cache, backend, thread-env, and wall-clock metadata in `run_metadata.json`,
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
  - sequential child backtests; per-child target-round optimization is sequential under moving-budget semantics,
  - source/candidate/solver comparability signatures that fail closed before ranking,
  - budget policy is part of experiment metadata/indexing so old fixed-budget artifacts are not comparable to moving-budget runs,
  - aggregate ranked summary by model/feature config,
  - prediction metrics, calibration deciles, per-season summary, metadata, comparability report, Markdown report, and self-contained Plotly HTML report artifacts.
- Experiment observability:
  - durable SQLite experiment index under `data/08_reporting/experiments/experiment_index.sqlite`,
  - optional tracker adapter boundary for future MLflow/local tracking,
  - best-effort tracking warnings that do not change experiment success semantics,
  - artifact-pointer policy so large child CSVs remain in the report tree instead of being duplicated into trackers.
- Experiment report UX:
  - model-feature experiments now generate real offline Plotly dashboards for `squad_performance_comparison.html` and `calibration_plots.html`,
  - report charts use persisted CSV/JSON artifacts as source of truth and do not rerun backtests,
  - configuration identity includes `model_id`, `feature_pack`, and `fixture_mode` so exploratory and no-fixture results do not collapse into one label,
  - missing report artifacts or required columns render explicit incomplete-report pages instead of blank placeholders.
- Production model selection for live recommendations:
  - `scripts/recommend_squad.py` and `scripts/run_live_round.py` now accept `--model-id`,
  - live/replay recommendations can use any model in the controlled registry,
  - the live no-fixture default has been promoted to
    `xgboost_depth2_l2_heavy + ppg_xg` after a 2021-2025 moving-budget
    production-parity rerun passed the v1 promotion guardrails.
- Constrained Ridge alpha tuning runner:
  - `scripts/run_ridge_tuning.py`,
  - fixed, predeclared Ridge alpha matrix: `0.01`, `0.03`, `0.1`, `0.3`, `1.0`, `3.0`, `10.0`, `30.0`, `100.0`, and `300.0`,
  - evaluates both `ppg` and `ppg_xg` for every alpha so xG is not advantaged only because it received tuning attention,
  - uses private experiment-only Ridge `model_params` plumbing while keeping normal backtest and live defaults unchanged,
  - screen stage ranks the full fixed matrix and final stage reruns the primary incumbent, secondary control, and top challengers,
  - writes `ranked_summary.csv`, `prediction_metrics.csv`, `calibration_deciles.csv`, `comparability_report.json`, `promotion_report.json`, `tuning_generation_manifest.json`, Markdown, and HTML reports under `data/08_reporting/experiments/model_tuning/`,
  - refuses promotion when final reruns are skipped, comparison controls drift, comparability fails, required metrics are missing, or lift is below the practical threshold.
- Fixed XGBoost candidate exploration:
  - adds `xgboost_conservative`, `xgboost_balanced`, and `xgboost_capacity` to the controlled model registry,
  - keeps XGBoost out of the default `production-parity` and `matchup-research` matrices so historical comparisons do not silently change,
  - adds an explicit `xgboost-research` group that evaluates only the fixed XGBoost candidates on `ppg_xg` and `ppg_xg_matchup`,
  - uses the `XGBRegressor` scikit-learn API with `tree_method=hist`, `objective=reg:squarederror`, fixed regularization settings, and per-child `n_jobs` control,
  - lazy-loads the native XGBoost runtime so normal workflows remain import-safe when the local OpenMP runtime is missing.
- Fixed XGBoost sensitivity generation:
  - adds `xgboost-sensitivity-v2` as a frozen local sensitivity matrix around the `xgboost_conservative` winner,
  - evaluates only `ppg_xg_matchup` so the generation answers whether the conservative matchup win is a robust region or an isolated spike,
  - includes `ridge` and `xgboost_conservative` controls in the same matrix,
  - adds local variants for depth-1 stumps, slower/faster depth-2 learning, more trees, heavier `min_child_weight`, stronger subsampling, stronger L2, L1/gamma pruning, and a regularized depth-3 check,
  - keeps Optuna deferred until this fixed generation proves the local XGBoost region is season-stable.
- Oracle knowledge discovery:
  - `scripts/run_oracle_knowledge_discovery.py` analyzes completed experiment artifacts without rerunning backtests or changing promotion fields,
  - reads persisted `round_results.csv`, `selected_players.csv`, `player_predictions.csv`, child metadata, and parent experiment metadata as source of truth,
  - builds model-candidate oracle comparisons using the same selected candidate universe and moving-budget path recorded by the source run,
  - normalizes equivalent duplicate candidate rows before oracle optimization and fails loudly on conflicting duplicate rows,
  - writes `oracle_round_results.csv`, `oracle_selected_players.csv`, `model_vs_oracle_recall.csv`, `oracle_player_profiles.csv`, `profile_gap_summary.csv`, `invalid_oracle_rows.csv`, metadata, and `oracle_knowledge_discovery.html`,
  - reports deterministic profile gaps for home share, opponent-overlap concentration, same-club concentration, favorite proxy, predicted-rank position, and top-5 predicted-rank share,
  - treats all oracle outputs as hindsight research only; oracle-derived findings must not change live defaults or promotion decisions without a frozen validation rerun.
- Policy Simulation V1:
  - `scripts/run_policy_simulation.py` replays frozen optimizer-policy variants from persisted experiment artifacts,
  - supports H001 opponent-overlap policies, H002 goalkeeper-conflict policies, and H003 clean-sheet defensive-stack policies,
  - keeps model predictions and candidate pools fixed while varying only optimizer policy constraints,
  - uses independent moving-budget paths per policy variant,
  - writes policy summaries, selected players, round results, comparability metadata, invalid-row diagnostics, and an HTML report under `data/08_reporting/policy_simulations/`,
  - verifies exploratory fixture identity against persisted source fixture file hashes when source experiments record them,
  - treats fixture-dependent policies as fail-closed unless missing fixture clubs are verified no-fixture/no-scoring candidates in the source artifact,
  - treats unverified fixture identity as `diagnostic_only`; policy simulation remains research evidence only and must not change live defaults without frozen validation.
- H004 residual diagnostic:
  - `scripts/run_h004_residual_diagnostic.py` tests whether attack-vs-defense
    mismatch context is visible in model residuals before building a new feature
    pack,
  - reads persisted `xgboost_depth2_slow + ppg_xg_matchup` experiment artifacts
    instead of rerunning models,
  - writes residual correlations, quintile spreads, top-actual recall, selected
    residual profiles, DNP context profiles, decision JSON, and HTML under
    `data/08_reporting/hypotheses/`,
  - treats the output as model-signal research only; a pass authorizes a frozen
    feature-pack experiment plan, not a live default or optimizer-policy change.
- EBM feature diagnostic runner:
  - `scripts/run_ebm_feature_diagnostic.py` reads persisted experiment artifacts
    and fits discovery-only InterpretML EBMs across whole-season folds,
  - writes source context, predictive metrics, feature importance, feature shape,
    interaction, candidate hypothesis, invalid-row, decision JSON, and HTML
    artifacts under `data/08_reporting/ebm_diagnostics/`,
  - keeps `discovery_only=true` outputs out of promotion/index/default paths,
  - produced one residual candidate around
    `matchup_opponent_allowed_position_count` for the current
    `xgboost_depth2_slow + ppg_xg_matchup` source run.
- H005 count-aware matchup reliability research:
  - adds the `ppg_xg_matchup_h005` feature pack and deterministic H005 mechanism
    audit/decision scripts,
  - tests position-normalized matchup-count reliability without manual point
    shrinkage,
  - preserves H005 as research-only and fail-closed when source identity,
    mechanism support, comparability, budget, ranking, or calibration gates fail.
- M006 fixed-blend diagnostic:
  - `scripts/run_fixed_blend_diagnostic.py` replays fixed XGBoost/Ridge blends
    from completed production-parity artifacts without training a learned
    stacker,
  - validates source control reproduction, component metadata, candidate
    identity, moving-budget comparability, and no-fixture/no-matchup context,
  - writes `fixed_blend_manifest.json`, `blend_complementarity.csv`,
    `blend_round_results.csv`, `blend_selected_players.csv`,
    `blend_per_season_summary.csv`, `blend_ranked_summary.csv`,
    `blend_decision.json`, `invalid_rows.csv`, and `fixed_blend_report.html`,
  - produced a weak-positive research lead for `xgb80_ridge20` but no
    promotion candidate.
- 2020 raw-season compatibility work:
  - accepts accented normalized status strings such as `Provável`,
  - infers `entrou_em_campo` from cumulative `num_jogos` when older round CSVs
    do not provide explicit entry flags,
  - ignores numeric Cartola club-name placeholders when auditing FootyStats
    team mapping.
- Standard scoring metadata:
  - `scoring_contract_version=cartola_standard_2026_v1`,
  - `captain_scoring_enabled=True`,
  - `captain_multiplier=1.5`,
  - `formation_search=all_official_formations`,
  - report readers reject old/mixed-contract backtest outputs.

**Current Interpretation**
The 2025 fixture-context result showed the first meaningful model lift: RF beat baseline and crossed the `player_r2 > 0.05` threshold, but that 2025 fixture data is still **exploratory reconstruction**, not strict historical proof. The strict system is now built for future/live capture, but we still need actual pre-lock snapshots to run strict evaluations.

The multi-season audit shows the current pipeline is compatible with recent seasons but not all historical data yet:

- `2021`, `2022`, `2023`, `2024`, `2025`: load, feature checks, and no-fixture backtests pass.
- `2026`: load, feature checks, and no-fixture backtest pass as a partial current-season smoke test; metrics are not comparable to complete seasons yet.
- `2021`: legacy `Mercado_*.txt` Latin-1 JSON market files are now read directly; the loader skips `Mercado_1.txt` as the opening snapshot and treats `Mercado_2.txt` through `Mercado_39.txt` as rounds `1..38`. FootyStats 2021 is classified as an integration candidate, and a bounded `ppg_xg` no-fixture smoke backtest passes.
- `2022`: now ignores the `rodada-0.csv` market snapshot during season loading and compatibility discovery; FootyStats 2022 is classified as an integration candidate, and a bounded `ppg_xg` no-fixture smoke backtest passes.
- `2018`, `2019`, `2020`: structurally complete, but currently fail at load time and need schema compatibility work before they can expand the training/evaluation history.

Historical backtests and model experiments now use moving-budget semantics.
For these runs, `--budget 100` means initial budget only: every strategy starts
from 100, optimizes each target round under its current budget, then updates its
next-round budget from the selected squad's official historical `variacao`.
This means old fixed-budget backtest/experiment evidence is non-comparable and
must be rerun before making promotion decisions.

Policy-simulation evidence now has three frozen, fixture-verified rejections:

- H001 opponent-overlap exposure: best variant `soft_overlap_penalty_low` gained
  `+187.91` total points but regressed 2025 by `-57.38`, so it is rejected.
- H002 goalkeeper-vs-opponent-attack exposure: best variant
  `gk_vs_opponent_captain_soft` gained `+46.19` total points but improved only
  `1 / 5` seasons, entirely from 2022, so it is rejected.
- H003 clean-sheet defensive-stack exposure: best variant
  `home_cs_pair_bonus_025` gained only `+19.70` total points, improved `2 / 5`
  seasons, regressed 2025 by `-63.24`, changed `54.1%` of evaluated rounds, and
  worsened the final moving-budget path by `-6.12`, so it is rejected.

Interpretation: broad overlap penalties and narrow goalkeeper-conflict penalties
are not stable enough to become optimizer policies. A simple clean-sheet stack
bonus is also too blunt: it changes many squads but does not produce stable
season-level value. The next policy hypothesis should use stronger pre-match
football signal with direct attack-vs-defense mismatch evidence, preferably as
candidate scoring/context features before adding another optimizer constraint.

H004 attack-vs-defense mismatch is being tested as model-signal research, not
an optimizer policy. Phase 1 residual diagnostics read persisted
`xgboost_depth2_slow + ppg_xg_matchup` artifacts and decide whether residuals
correlate with pre-match xG/home/position matchup context strongly enough to
justify a frozen feature-pack experiment.

Latest H004 residual diagnostic:

- output:
  `data/08_reporting/hypotheses/h004_residual_diagnostic_started_at=20260508T182202655139Z`;
- diagnostic status: `passes`;
- passed families: `C`;
- Family C passed seasons: `2021`, `2022`, `2023`, and `2025`;
- Family A passed only `2022`;
- Family B passed no seasons;
- fixture identity status: `unverified`.

Interpretation: H004 Phase 1 is strong enough to proceed to a Phase 2
`ppg_xg_matchup_h004` feature-pack plan. The signal is specifically that actual
top scorers were often poorly ranked despite favorable context, not that simple
residual correlations already prove an attack/defense feature. Because fixture
identity is still unverified, this remains research evidence and must not
change live defaults without a frozen validation run.

H004 Phase 2 feature-pack experiment is complete and rejected:

- output:
  `data/08_reporting/experiments/model_feature/group=h004-attack-defense-mismatch__started_at=20260508T191417000002Z__matrix=fc77bcf76f40`;
- decision artifact:
  `h004_phase2_decision.json`;
- control:
  `xgboost_depth2_slow + ppg_xg_matchup`;
- challenger:
  `xgboost_depth2_slow + ppg_xg_matchup_h004`;
- fixture identity status: `verified`;
- candidate signature status: `ok`;
- aggregate actual-points delta: `-90.60`;
- improved seasons: `2 / 5`;
- season deltas:
  `2021=-80.24`, `2022=+110.38`, `2023=+101.26`, `2024=-38.73`,
  `2025=-183.27`;
- failed gates:
  aggregate lift, improved seasons, worst-season delta, 2025 delta, and
  top-two positive season concentration.

Interpretation: the H004 feature pack improved the moving-budget path and
helped 2022/2023, but it failed the actual-points gates and badly regressed the
most recent complete season. Do not promote or keep iterating on
`ppg_xg_matchup_h004`; preserve it as negative evidence. The next feature
hypothesis should start from a fresh residual diagnostic or a materially
different football signal, not a direct H004-family tweak.

EBM and H005 reliability research are complete enough to close this research
branch:

- EBM source diagnostic:
  `data/08_reporting/ebm_diagnostics/ebm_diagnostic_started_at=20260511T004620197204Z`;
- EBM status: `diagnostic_complete`;
- only residual-target lead:
  `matchup_opponent_allowed_position_count`;
- H005 mechanism audit:
  `data/08_reporting/hypotheses/h005_mechanism_audit_started_at=20260513T115234772087Z`;
- H005 feature experiment:
  `data/08_reporting/experiments/model_feature/group=h005-count-aware-matchup-shrinkage__started_at=20260513T115300476279Z__matrix=ede90899ae80`;
- H005 decision status: `invalid`;
- mechanism audit status: `invalid` because of a strict duplicate
  `row_identity_mismatch` on `2022`, round `5`, athlete `68923`;
- recomputed count match status: `ok`;
- if the duplicate identity check is ignored, mechanism support is still only
  `mixed_or_weak`;
- H005 aggregate actual-points delta: `-164.14`;
- H005 2025 actual-points delta: `-186.80`;
- comparability, fixture identity, candidate signature, and optimizer status:
  `ok`.

Interpretation: H005 ran through the experiment matrix, but it did not produce
valid or useful evidence. The normalized matchup-count reliability feature pack
should not be promoted, and it should not be followed by another direct
count-reliability tweak. The broader lesson is that the EBM lead was too weak
once translated into a frozen feature pack. Future feature hypotheses should
come from a fresh diagnostic or a materially different signal family.

M006 fixed-blend evidence is now complete for the first frozen blend matrix:

- output:
  `data/08_reporting/blend_diagnostics/fixed_blend_started_at=20260514T000703093758Z`;
- source experiment:
  `data/08_reporting/experiments/model_feature/group=production-parity__started_at=20260513T165550180815Z__matrix=9064290978ca`;
- source valid: `true`;
- control:
  `xgboost_depth2_l2_heavy + ppg_xg`;
- tested blends:
  `xgb90_ridge10`, `xgb80_ridge20`, and `xgb70_ridge30`;
- `xgb80_ridge20` status: `weak_positive_research_lead`;
- `xgb80_ridge20` aggregate actual-points delta: `+226.38`;
- `xgb80_ridge20` improved seasons: `4 / 5`;
- `xgb80_ridge20` 2025 delta: `+91.31`;
- `xgb80_ridge20` disaster rounds under 45 delta: `-7`;
- `xgb80_ridge20` selected calibration slope: `1.0426`;
- `xgb80_ridge20` top-50 Spearman delta: `-0.0012`;
- `xgb80_ridge20` was not a `candidate_blend` because budget/drawdown
  downside gates were not strong enough;
- `xgb90_ridge10` status: `inconclusive`, aggregate delta `+12.99`;
- `xgb70_ridge30` status: `rejected`, aggregate delta `+119.09`, but with a
  large 2022 regression of `-103.77`.

Interpretation: fixed blending is not dead. The `80% XGBoost / 20% Ridge`
profile is the first post-H005 research lead with meaningful aggregate lift,
season support, calibration, and disaster-round improvement. It is not yet a
live default because the moving-budget path still shows enough downside risk to
require a focused validation generation.

The no-fixture live default promotion matrix is complete for `2021`, `2022`,
`2023`, `2024`, and `2025` with `fixture_mode=none`, moving-budget semantics,
`start_round=5`, and `budget=100`.

Promoted default:

- model/profile: `xgboost_depth2_l2_heavy + ppg_xg`;
- artifact:
  `data/08_reporting/experiments/model_feature/group=production-parity__started_at=20260513T135649009077Z__matrix=e90eb0b9b692`;
- rank: `1`;
- total actual points: `9945.02`;
- baseline `random_forest + ppg`: `8975.77`;
- aggregate lift: `+969.25`;
- average lift: `+5.70` points per round;
- improved seasons: `5 / 5`;
- worst season average delta: `+0.67`;
- selected calibration slope: `0.8843`;
- top-50 Spearman delta: `+0.0111`;
- promotion status: `passes_v1_guardrails`.

Interpretation: `xgboost_depth2_l2_heavy + ppg_xg + fixture_mode=none` is now
the promoted historical moving-budget live default. `random_forest + ppg`
remains useful as a baseline comparator only; it should not be used as the live
default unless explicitly requested for a control/replay run. This no-fixture
promotion does not promote exploratory or strict matchup-context defaults.

The first fixed-budget production-parity model/feature experiment is complete for `2023`,
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

Interpretation: for no-fixture production, `ridge + ppg_xg` was the best
fixed-budget candidate profile. It is now superseded as promotion evidence until
rerun under moving budget. The result does **not** prove that matchup context is
useful, because this group intentionally used `fixture_mode=none`.

The first fixed-budget matchup-research model/feature experiment is complete for `2023`,
`2024`, and `2025` with `fixture_mode=exploratory`, `start_round=5`, and
`budget=100`. Comparability status is `ok`.

Best aggregate matchup result:

- `ridge + ppg_xg_matchup`;
- total actual points: `6649.11`;
- exploratory baseline `random_forest + ppg`: `6029.86`;
- aggregate lift: `+619.25`;
- average lift: `+6.07` points per round;
- improved seasons: `3 / 3`;
- promotion status: `passes_v1_guardrails`.

Second matchup result:

- `ridge + ppg_matchup`;
- total actual points: `6633.94`;
- aggregate lift: `+604.08`;
- improved seasons: `3 / 3`;
- promotion status: `passes_v1_guardrails`.

Interpretation: `cartola_matchup_v1` was useful research signal in fixed-budget
exploratory evidence. For Ridge,
`ppg_xg_matchup` beat `ppg_xg` by `+133.87` total points, about `+1.31` per
round. This is strong enough to justify a strict/live matchup integration spec,
but it is now superseded until rerun under moving budget and remains
**exploratory fixture evidence**, not strict no-leakage live proof.

The full constrained Ridge tuning run is complete for `2023`, `2024`, and
`2025` with final reruns enabled. `promotion_report.json` recommends
`keep_incumbent`: no tuned alpha cleared the practical lift threshold over
`ridge + ppg_xg + alpha=1.0`. Keep `ridge + ppg_xg + alpha=1.0` as the live
no-fixture profile.

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
not a universal default for every model. Its fixed-budget Ridge evidence is
superseded until rerun under moving budget.

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

The historical backtest budget is now moving per strategy. A run with
`--budget 100` means each strategy starts with C$ 100. After each completed
target round, the selected players' and tecnico's official historical
`variacao` is summed into that strategy's next-round budget. The budget update
happens only after selection and scoring, and a selected historical asset with
missing `variacao` invalidates the run instead of being treated as zero.

**How To Run Now**
For historical backtests and experiments below, `--budget 100` is the initial
moving budget. Old fixed-budget reports should not be compared against new
outputs unless they have been rerun under `budget_policy=moving`.

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

Fixed XGBoost candidate exploration:

```bash
uv run --frozen python scripts/run_model_experiments.py \
  --group xgboost-research \
  --seasons 2023,2024,2025 \
  --start-round 5 \
  --budget 100 \
  --current-year 2026 \
  --jobs 12 \
  --profile-runtime
```

Fixed XGBoost sensitivity v2 experiment:

```bash
uv run --frozen python scripts/run_model_experiments.py \
  --group xgboost-sensitivity-v2 \
  --seasons 2023,2024,2025 \
  --start-round 5 \
  --budget 100 \
  --current-year 2026 \
  --jobs 12 \
  --profile-runtime
```

M006 fixed XGBoost/Ridge blend diagnostic:

```bash
uv run --frozen python scripts/run_fixed_blend_diagnostic.py \
  --experiment-path data/08_reporting/experiments/model_feature/group=production-parity__started_at=20260513T165550180815Z__matrix=9064290978ca \
  --seasons 2021,2022,2023,2024,2025 \
  --feature-pack ppg_xg \
  --control-model xgboost_depth2_l2_heavy \
  --blend xgb90_ridge10=xgboost_depth2_l2_heavy:0.9,ridge:0.1 \
  --blend xgb80_ridge20=xgboost_depth2_l2_heavy:0.8,ridge:0.2 \
  --blend xgb70_ridge30=xgboost_depth2_l2_heavy:0.7,ridge:0.3 \
  --initial-budget 100 \
  --current-year 2026
```

Oracle knowledge discovery for a completed experiment:

```bash
uv run --frozen python scripts/run_oracle_knowledge_discovery.py \
  --experiment-path data/08_reporting/experiments/model_feature/<experiment_id> \
  --current-year 2026
```

H004 residual diagnostic for the current XGBoost sensitivity control:

```bash
uv run --frozen python scripts/run_h004_residual_diagnostic.py \
  --experiment-path data/08_reporting/experiments/model_feature/group=xgboost-sensitivity-v2__started_at=20260507T231127138806Z__matrix=f019652c883d \
  --seasons 2021,2022,2023,2024,2025
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

Live squad recommendation for the current no-fixture production profile:

```bash
uv run --frozen python scripts/run_live_round.py \
  --season 2026 \
  --budget 100 \
  --current-year 2026
```

Manual two-step live recommendation:

```bash
uv run --frozen python scripts/recommend_squad.py \
  --season 2026 \
  --target-round 14 \
  --mode live \
  --budget 100 \
  --current-year 2026
```

Opt-in strict matchup live recommendation after strict fixture evidence exists for the open round:

```bash
uv run --frozen python scripts/run_live_round.py \
  --season 2026 \
  --budget 100 \
  --model-id xgboost_depth2_slow \
  --footystats-mode ppg_xg \
  --fixture-mode strict \
  --matchup-context-mode cartola_matchup_v1 \
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

**Next Milestone: M006b Blend Validation And 2020 Expansion**

Goal: decide whether the `xgb80_ridge20` weak-positive lead is stable enough to
become a promotion candidate or should remain research-only.

Why now: M006 is the first recent research milestone to produce meaningful
positive evidence after H001-H005 were rejected. The best blend gained
`+226.38` aggregate points, improved `4 / 5` seasons, improved 2025 by
`+91.31`, reduced disaster rounds, and kept selected-player calibration inside
the guardrail. The remaining blocker is budget/downside risk, not point signal.

Scope:

- Run a focused validation generation for only `xgb80_ridge20` versus the
  promoted XGBoost control.
- Include `2020` if the compatibility audit confirms the new loader fixes make
  the season comparable; otherwise run a documented 2020 diagnostic-only slice.
- Report budget risk separately from point lift:
  final budget delta, min-budget delta, max drawdown delta, budget-constrained
  rounds, and per-season path plots.
- Add explicit 2020 source-compatibility status to the decision artifact.
- Add progress logging to the fixed-blend diagnostic so long MILP replay runs
  are observable.
- Do not build learned stacking or RF gating until `xgb80_ridge20` clears this
  focused validation.

Acceptance:

- M006b writes a deterministic decision artifact with statuses:
  `candidate_blend`, `weak_positive_research_lead`, `inconclusive`, `rejected`,
  or `diagnostic_only_2020`.
- A candidate requires positive aggregate lift, at least `4 / 6` improved
  seasons if 2020 is valid or `4 / 5` if not, no 2025 regression, no severe
  budget-path regression, and no increase in disaster rounds.
- The decision artifact includes a plain recommendation:
  promote blend, keep XGBoost default and monitor blend, or reject blend.
- `xgboost_depth2_l2_heavy + ppg_xg` remains the live default unless M006b
  clears the frozen promotion gates.

**Following Milestone: Live-Lineup Risk Guardrails**

Goal: make live squad recommendations operationally safer after the current
model-selection question is closed.

Scope:

- Add a live risk summary artifact for `run_live_round.py` and
  `recommend_squad.py` outputs.
- Surface market-capture age, capture policy, selected-player status mix,
  low-appearance/low-history flags, and selected-player DNP/null-risk context.
- Add CLI/report warnings that are visible before a human trusts the squad.
- Keep this as a guardrail/reporting milestone; do not change optimizer
  objective or promotion policy in the same milestone.

**Roadmap**
1. Build M006b blend validation.
   - Focus only on `xgb80_ridge20`.
   - Include 2020 only after compatibility is verified, or mark it
     diagnostic-only.
   - Treat budget/drawdown as the primary promotion blocker to resolve.
   - Do not build learned stackers, RF gating, or AutoML before M006b closes.
2. Build the live-lineup risk guardrails milestone after M006b.
   - The milestone improves recommendation trust, not model score quality.
   - It must not silently block all live recommendations; it should make risk
     explicit and fail only on deliberately configured hard thresholds.
3. Keep live model selection pinned to the promoted moving-budget default.
   - Current default: `xgboost_depth2_l2_heavy + ppg_xg + fixture_mode=none`.
   - Keep `random_forest + ppg` available as the historical baseline, not as a
     live recommendation default.
   - Continue to record the chosen model id and feature mode in every
     recommendation output.
   - Any future default change needs a fresh frozen moving-budget promotion
     artifact with the same comparability guardrails.
   - For single-round live recommendations, `--budget` still means the caller-provided current available budget for that one open round.
   - Strict matchup mode is available only as an opt-in research/live candidate until it survives real strict pre-lock rounds.
4. Use strict matchup mode in the next open round only after capturing strict fixture evidence.
   - Capture strict fixture snapshots before generating the recommendation.
   - First candidate to test: `xgboost_depth2_slow + ppg_xg + cartola_matchup_v1`, because it beat Ridge in all three exploratory seasons and fixed the 2025 regression seen in the faster aggregate winner.
   - Inspect `run_metadata.json` for fixture manifest paths/hashes and `candidate_predictions.csv` before trusting the squad.
   - Do not fall back to exploratory fixtures in live mode.
5. Treat the older fixed XGBoost sensitivity result as superseded by the
   no-fixture production-parity promotion rerun.
   - XGBoost is now allowed in live no-fixture defaults only for the promoted
     profile `xgboost_depth2_l2_heavy + ppg_xg`.
   - Matchup-context XGBoost profiles remain separate research/live candidates
     until strict pre-lock fixture evidence supports them.
   - Treat each fixed candidate list as a frozen generation. Any change to the XGBoost specs should become a new experiment group/generation, not an informal rerun.
   - On macOS, XGBoost requires the native OpenMP runtime (`libomp`) to be installed before the XGBoost children can run.
   - Defer RandomForest/ExtraTrees tuning until calibration wrappers are designed; tree models still fail calibration guardrails.
   - Defer HGB tuning despite the runtime fix; it is now operationally usable, but it did not beat Ridge.
   - Defer Optuna until the fixed XGBoost sensitivity pass shows a season-stable local region, not a single lucky aggregate winner.
   - Future Optuna work should use a seeded `TPESampler`, SQLite storage, a frozen search space, explicit trial manifests, and final full-backtest reruns for finalists; pruned or predictive-only trials must never be promotion-eligible.
   - Defer LightGBM and CatBoost until the promoted XGBoost default has enough
     live monitoring evidence to justify another model-family generation.
6. Use generated Plotly experiment reports as the standard review surface.
   - Start with `squad_performance_comparison.html` and `calibration_plots.html`.
   - Use `ranked_summary.csv`, `per_season_summary.csv`, `prediction_metrics.csv`, `calibration_deciles.csv`, and `comparability_report.json` as source-of-truth artifacts.
   - Do not trust any ranking if comparability failed.
7. Use `scripts/run_live_round.py` for each 2026 open round and inspect `recommended_squad.csv`, `candidate_predictions.csv`, `run_metadata.json`, and `live_workflow_metadata.json` before making lineup decisions.
8. Capture strict pre-lock fixture snapshots every live round with `scripts/capture_strict_round_fixture.py`.
   - Manual v1 command captures snapshot evidence and generates strict `fixtures_strict` CSV/manifest.
   - These snapshots are now consumed by opt-in strict matchup live recommendations.
9. Harden moving-budget evidence before any future default replacement.
   - Rerun the relevant candidate and baseline controls with moving budget.
   - Compare only artifacts with `budget_policy=moving`; treat missing `budget_policy` as old fixed-budget evidence.
   - Inspect `initial_budget`, `final_budget`, `total_budget_delta`, `min_budget`, `max_budget_drawdown`, and `budget_constrained_rounds` alongside points.
   - Validate that selected player and tecnico rows always preserve finite historical `variacao` in completed rounds.
10. Keep budget modeling realistic but deliberately simple.
   - Historical multi-round backtests use official historical `variacao`; no hidden Cartola price formula is reverse-engineered.
   - Budget updates are strategy-specific and happen only after selection/scoring.
   - Do not reintroduce a normal fixed-budget mode.
   - Live open-round recommendations remain single-round until completed-round replay data exists to update the budget path.
11. Defer wider matchup features until strict/live matchup v1 has real-round evidence:
   - home/away split priors,
   - shorter roll3 variants,
   - odds/goal-environment fields,
   - or DNP probability modeling if selection reliability becomes the bigger live-game bottleneck.
12. Add DNP probability modeling if needed:
    - predict `p_play`,
    - use `expected_points = predicted_points * p_play`.
13. Keep adaptive hyperparameter search behind the promoted fixed-candidate baseline.
    - XGBoost is now the live no-fixture default, but future tuning should still use frozen generations with full-backtest reruns.
    - Possible later candidates: CatBoost or LightGBM, one family at a time.
    - Optuna remains a future search engine, not v1 implementation scope.
    - Do not start broad grid search over external libraries before fixed candidates, strict/live matchup integration, and live reliability guardrails are understood.
14. Use oracle knowledge discovery as a diagnostic surface, not a model-selection surface.
    - Start with `oracle_knowledge_discovery.html`, `profile_gap_summary.csv`, and `model_vs_oracle_recall.csv`.
    - Useful questions: whether oracle squads and model squads differ in home/away exposure, opponent-overlap exposure, same-club concentration, favorite proxy, and predicted-rank recall.
    - Do not treat a single profile gap as a validated policy. Convert it into a frozen hypothesis, then validate through the normal walk-forward experiment workflow.
    - If oracle discovery becomes a regular workflow, optimize runtime by reducing repeated solver calls, streaming artifact writes, or caching per-round oracle inputs.

**Backfill / Robustness Track**
These items are useful, but they are no longer the next prediction-quality bottleneck:

1. Fix historical loader compatibility for structurally complete failing seasons:
   - rerun the compatibility audit after the 2020 loader fixes,
   - inspect remaining 2019 and 2018 load errors from the compatibility audit JSON,
   - add schema normalization only where needed,
   - rerun the audit until eligible structurally complete seasons reach
     `load_status=ok`.
2. Re-evaluate model/feature experiments with 2021 and 2022 included as frozen generations:
   - start with no-fixture `production-parity` over `2021,2022,2023,2024,2025`,
   - treat this as a new experiment generation, not a replacement for the original `2023-2025` evidence,
   - oracle discovery now normalizes equivalent duplicate `(rodada, id_atleta)` candidate rows and fails on conflicting duplicates; inspect `invalid_oracle_rows.csv` before using oracle diagnostics,
   - compare 2021/2022 calibration against 2023-2025 before making live/default promotion claims.
