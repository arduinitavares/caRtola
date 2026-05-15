# Repository Instructions

<!-- context7 -->
Use Context7 MCP to fetch current documentation whenever the user asks about a library, framework, SDK, API, CLI tool, or cloud service, even well-known ones like React, Next.js, Prisma, Express, Tailwind, Django, or Spring Boot. This includes API syntax, configuration, version migration, library-specific debugging, setup instructions, and CLI tool usage. Use it even when you think you know the answer because training data may not reflect recent changes. Prefer this over web search for library docs.

Do not use Context7 for refactoring, writing scripts from scratch, debugging business logic, code review, or general programming concepts.

Steps:

1. Always start with `resolve-library-id` using the library name and the user's question, unless the user provides an exact library ID in `/org/project` format.
2. Pick the best match by exact name match, description relevance, code snippet count, source reputation, and benchmark score. If results do not look right, try alternate names or queries. Use version-specific IDs when the user mentions a version.
3. Run `query-docs` with the selected library ID and the user's full question.
4. Answer using the fetched docs.
<!-- context7 -->

## Project Shape

- This is a Python/Kedro project managed with `uv`; use Python `3.13.12` from `.python-version`.
- Main Python package: `src/cartola`. Tests live in `src/tests`.
- Operational scripts live in `scripts/`. Generated reports and model outputs are written under `data/08_reporting/`.
- Do not commit secrets or local machine config from `conf/local`.

## Setup And Quality

- Install local dev dependencies with `uv sync --dev`.
- Reproduce the GitHub Actions quality gate with `uv sync --locked --dev` and `uv run --frozen scripts/pyrepo-check --all`.
- `scripts/pyrepo-check` supports targeted checks: `ruff`, `ty`, `bandit`, and `pytest`.
- Annotation presence is enforced by Ruff `ANN` rules; run only that gate with `uv run --frozen ruff check src/cartola src/tests scripts --select ANN`.
- Run tests directly with `uv run --frozen pytest` when a narrower pytest workflow is useful.
- Use `make clean` to remove Python caches, build artifacts, coverage fragments, `references/`, and `results/`.

## Backtesting And Audits

- Offline no-fixture backtest:
  `uv run --frozen python -m cartola.backtesting.cli --season 2025 --start-round 5 --budget 100 --fixture-mode none`
- Exploratory fixture backtest for reconstructed 2025 fixtures:
  `uv run --frozen python -m cartola.backtesting.cli --season 2025 --start-round 5 --budget 100 --fixture-mode exploratory`
- Historical backtests use moving-budget semantics: `--budget` is the initial budget, each strategy updates its own next-round budget from selected historical `variacao`, and old fixed-budget reports are non-comparable unless rerun.
- Moving-budget historical backtests process target rounds sequentially; `--jobs` remains accepted for metadata/model plumbing but does not create target-round workers.
- Backtests write `budget_policy=moving`, budget-path metadata, budget summary fields, and an interactive Plotly chart at `charts/strategy_performance_by_round.html`.
- Historical fixture import from TheSportsDB writes `data/01_raw/fixtures/{season}/partidas-*.csv` and `data/08_reporting/fixtures/{season}/round_alignment.csv`; if imports report unmapped teams, patch only `data/01_raw/fixtures/club_mapping.csv` and rerun:
  `uv run --frozen python scripts/import_fixture_schedule.py --season 2022 --first-round 1 --last-round 38`
  `uv run --frozen python scripts/import_fixture_schedule.py --season 2021 --first-round 1 --last-round 38`
- Matchup fixture coverage audit:
  `uv run --frozen python scripts/audit_matchup_fixture_coverage.py --seasons 2021,2022,2023,2024,2025 --current-year 2026`
- Exploratory Cartola matchup-context backtest:
  `uv run --frozen python -m cartola.backtesting.cli --season 2025 --start-round 5 --budget 100 --fixture-mode exploratory --footystats-mode ppg --matchup-context-mode cartola_matchup_v1 --current-year 2026 --jobs 12 --output-root data/08_reporting/backtests/matchup_context_single`
- Season compatibility audit:
  `uv run --frozen python scripts/audit_backtest_compatibility.py --current-year 2026`
- Focused season compatibility audit:
  `uv run --frozen python scripts/audit_backtest_compatibility.py --seasons 2020 --current-year 2026`
- The compatibility loader now skips opening snapshots such as `rodada-0.csv` and 2021 `Mercado_1.txt`; legacy `Mercado_*.txt` files are read as Latin-1 JSON when no `rodada-*.csv` files exist.
- FootyStats compatibility audit:
  `uv run --frozen python scripts/audit_footystats_compatibility.py --current-year 2026`
- For no-fixture live recommendations, the promoted historical moving-budget default is `xgboost_depth2_l2_heavy + ppg_xg + fixture_mode=none`. Promotion artifact: `data/08_reporting/experiments/model_feature/group=production-parity__started_at=20260513T135649009077Z__matrix=e90eb0b9b692`, rank 1, `promotion_eligible=true`, `promotion_reason=passes_v1_guardrails`, aggregate lift `+969.25` over `random_forest + ppg`, improved `5 / 5` seasons. Keep `random_forest + ppg` available only as the historical baseline comparator and do not infer matchup-context production value from this no-fixture result.
- Expanded 2020-2025 no-fixture evidence now makes `ridge + ppg_xg` the points leader, but M008 did not promote it because budget-risk gates failed. Keep the XGBoost default until live budget/risk guardrails are implemented and a fresh decision clears promotion.
- Backtests and recommendations use the `cartola_standard_2026_v1` scoring contract: all official formations are searched, a non-tecnico captain is selected with a `1.5x` multiplier, and report totals should use the captain-aware point fields.
- `matchup_context_mode=cartola_matchup_v1` is separate from `footystats_mode`; live recommendations support it only with `--fixture-mode strict` and keep it out of defaults.

## Model Experiment Workflow

- Production-parity experiment:
  `uv run --frozen python scripts/run_model_experiments.py --group production-parity --seasons 2023,2024,2025 --start-round 5 --budget 100 --current-year 2026 --jobs 12`
- Matchup-research experiment:
  `uv run --frozen python scripts/run_model_experiments.py --group matchup-research --seasons 2023,2024,2025 --start-round 5 --budget 100 --current-year 2026 --jobs 12`
- XGBoost fixed-candidate experiment:
  `uv run --frozen python scripts/run_model_experiments.py --group xgboost-research --seasons 2023,2024,2025 --start-round 5 --budget 100 --current-year 2026 --jobs 12 --profile-runtime`
- XGBoost sensitivity v2 experiment:
  `uv run --frozen python scripts/run_model_experiments.py --group xgboost-sensitivity-v2 --seasons 2023,2024,2025 --start-round 5 --budget 100 --current-year 2026 --jobs 12 --profile-runtime`
- H004 attack-defense mismatch experiment:
  `uv run --frozen python scripts/run_model_experiments.py --group h004-attack-defense-mismatch --seasons 2021,2022,2023,2024,2025 --start-round 5 --budget 100 --current-year 2026 --jobs 12 --profile-runtime`
- Experiment outputs are written under `data/08_reporting/experiments/model_feature/<experiment_id>/`.
- Experiment runs are indexed in SQLite at `data/08_reporting/experiments/experiment_index.sqlite`.
- Use `--models` and `--exclude-models` for targeted experiment slices; filters must leave at least one model.
- Successful experiments generate `squad_performance_comparison.html` and `calibration_plots.html` from persisted CSV/JSON artifacts.
- Compare experiment rows only within the same `budget_policy`; missing policy means old fixed-budget evidence and must not be mixed with moving-budget rankings.
- Treat 2021/2022 experiment expansion as a new generation, for example:
  `uv run --frozen python scripts/run_model_experiments.py --group production-parity --seasons 2021,2022,2023,2024,2025 --start-round 5 --budget 100 --current-year 2026 --jobs 12`
- After the 2021-2025 matchup fixture coverage audit passes, rerun `matchup-research`, `xgboost-research`, and `xgboost-sensitivity-v2` with `--seasons 2021,2022,2023,2024,2025`; keep those separate from older 2023-2025-only evidence.
- For long experiment runs, copy `.env.example` to `.env` to cap native OpenMP/BLAS threads; `scripts/run_model_experiments.py` loads it before runtime imports without overriding shell env values.
- Optional MLflow tracking:
  add `--tracker mlflow --mlflow-tracking-uri file:///tmp/cartola-mlruns`; tracking warnings are reported but do not change experiment success semantics.
- Experiment `--jobs` is passed to each child backtest, but moving-budget child backtests still optimize target rounds sequentially; experiment children are not run concurrently.
- Use `--profile-runtime` for runtime investigations; child backtest metadata records per-round fit/predict/optimizer timings.
- Normal backtests do not expose `--model-id`; model selection for backtest research stays private to experiment and tuning runners.
- XGBoost is lazy-loaded and may require the native OpenMP runtime on macOS (`brew install libomp`) before XGBoost children can run.

## Policy Simulation Workflow

- Artifact-backed H001 opponent-overlap policy replay for a completed model-feature experiment:
  `uv run --frozen python scripts/run_policy_simulation.py --experiment-path data/08_reporting/experiments/model_feature/group=xgboost-sensitivity-v2__started_at=20260507T231127138806Z__matrix=f019652c883d --hypothesis-id H001 --policy-set opponent-overlap-v1 --models xgboost_depth2_slow --feature-packs ppg_xg_matchup --seasons 2021,2022,2023,2024,2025 --current-year 2026`
- Artifact-backed H002 goalkeeper-conflict policy replay for a completed model-feature experiment:
  `uv run --frozen python scripts/run_policy_simulation.py --experiment-path data/08_reporting/experiments/model_feature/group=xgboost-sensitivity-v2__started_at=20260507T231127138806Z__matrix=f019652c883d --hypothesis-id H002 --policy-set gk-conflict-v1 --models xgboost_depth2_slow --feature-packs ppg_xg_matchup --seasons 2021,2022,2023,2024,2025 --current-year 2026`
- Artifact-backed H003 clean-sheet defensive-stack policy replay for a completed model-feature experiment:
  `uv run --frozen python scripts/run_policy_simulation.py --experiment-path data/08_reporting/experiments/model_feature/group=xgboost-sensitivity-v2__started_at=20260507T231127138806Z__matrix=f019652c883d --hypothesis-id H003 --policy-set clean-sheet-stack-v1 --models xgboost_depth2_slow --feature-packs ppg_xg_matchup --seasons 2021,2022,2023,2024,2025 --current-year 2026`
- Policy simulation outputs are written under `data/08_reporting/policy_simulations/policy_simulation_started_at=.../` and include `policy_simulation_manifest.json`, policy summary CSVs, selected-player and round-result CSVs, `policy_comparability_report.json`, and `policy_simulation_report.html`.
- Treat policy simulation as research-only: it replays policy variants from persisted experiment artifacts, keeps model predictions and candidate pools fixed, and gives each policy an independent moving-budget path.
- Exploratory fixture backtests record fixture source paths and SHA-256 hashes in `run_metadata.json`; policy simulation verifies those hashes when present.
- Verified fixture identity produces `ok` comparability; unverified or mismatched fixture identity produces `diagnostic_only`, which is not promotion evidence.
- H003 `clean-sheet-stack-v1` latest verified run was rejected; do not promote defensive-stack optimizer bonuses without a new frozen validation run.
- Do not change live defaults from policy simulation output without a frozen validation run.

## Oracle Discovery Workflow

- Artifact-backed oracle knowledge discovery for completed experiment runs:
  `uv run --frozen python scripts/run_oracle_knowledge_discovery.py --experiment-path data/08_reporting/experiments/model_feature/<experiment_id> --current-year 2026`
- Oracle discovery outputs are written under `data/08_reporting/oracle_discovery/oracle_discovery_started_at=.../` and include `oracle_round_results.csv`, `oracle_selected_players.csv`, `oracle_captain_profiles.csv`, `model_vs_oracle_recall.csv`, `profile_gap_summary.csv`, `invalid_oracle_rows.csv`, `oracle_discovery_metadata.json`, and `oracle_knowledge_discovery.html`.
- Treat oracle discovery as hindsight research only: it reads persisted source experiment artifacts, does not make promotion decisions, and must not change live defaults, experiment rankings, or experiment index promotion fields.

## H004 Attack-Defense Research

- Artifact-backed H004 residual diagnostic for the current XGBoost sensitivity control:
  `uv run --frozen python scripts/run_h004_residual_diagnostic.py --experiment-path data/08_reporting/experiments/model_feature/group=xgboost-sensitivity-v2__started_at=20260507T231127138806Z__matrix=f019652c883d --seasons 2021,2022,2023,2024,2025`
- H004 residual diagnostic outputs are written under `data/08_reporting/hypotheses/h004_residual_diagnostic_started_at=.../` and include `h004_residual_correlations.csv`, `h004_residual_quintiles.csv`, `h004_top_actual_recall.csv`, `h004_selected_residual_profile.csv`, `h004_dnp_context_profile.csv`, `h004_diagnostic_decision.json`, and `h004_residual_diagnostic.html`.
- H004 Phase 2 compares `xgboost_depth2_slow + ppg_xg_matchup` against `xgboost_depth2_slow + ppg_xg_matchup_h004` in the same experiment matrix; do not compare the H004 challenger against old control artifacts.
- Build the deterministic H004 Phase 2 decision artifact after a completed H004 experiment:
  `uv run --frozen python scripts/run_h004_feature_decision.py --experiment-path data/08_reporting/experiments/model_feature/group=h004-attack-defense-mismatch__started_at=20260508T191417000002Z__matrix=fc77bcf76f40`
- H004 Phase 2 writes `h004_phase2_decision.json`; the latest verified run rejected `ppg_xg_matchup_h004` after aggregate and 2025 actual-points regressions, so preserve it as negative research evidence rather than a live default or direct follow-up tweak.

## EBM Feature Diagnostic Workflow

- Artifact-backed EBM diagnostic for the current XGBoost sensitivity matchup source:
  `uv run --frozen python scripts/run_ebm_feature_diagnostic.py --experiment-path data/08_reporting/experiments/model_feature/group=xgboost-sensitivity-v2__started_at=20260507T231127138806Z__matrix=f019652c883d --model-id xgboost_depth2_slow --feature-pack ppg_xg_matchup --seasons 2021,2022,2023,2024,2025 --fixture-mode exploratory --current-year 2026 --profile-runtime`
- EBM diagnostic outputs are written under `data/08_reporting/ebm_diagnostics/ebm_diagnostic_started_at=.../` and include `ebm_diagnostic_manifest.json`, `ebm_diagnostic_decision.json`, `source_context.csv`, `predictive_metrics.csv`, `feature_importance_by_fold.csv`, `feature_shape_summary.csv`, `pairwise_interactions.csv`, `candidate_hypotheses.csv`, `invalid_ebm_rows.csv`, `invalid_diagnostic_report.csv`, and `ebm_feature_diagnostic.html`.
- Treat EBM as discovery-only: outputs include `discovery_only=true`, use whole-season validation folds with inner validation disabled, read persisted source experiment artifacts, and must not update live defaults, promotion fields, or experiment rankings.
- The latest completed EBM diagnostic for `xgboost_depth2_slow + ppg_xg_matchup` produced one source-residual candidate around `matchup_opponent_allowed_position_count`; use it only to freeze follow-up hypotheses such as H005, not as direct production evidence.
- InterpretML `interpret` is a project dependency for this diagnostic; if dependency/runtime validation fails, inspect `invalid_diagnostic_report.csv` before rerunning.

## H005 Count-Aware Matchup Reliability Research

- H005 commands require an active checkout that includes `scripts/run_h005_mechanism_audit.py` and `scripts/run_h005_feature_decision.py`; do not run them from an older checkout that only has the H005 docs/specs.
- Source-anchored H005 mechanism audit:
  `uv run --frozen python scripts/run_h005_mechanism_audit.py --experiment-path data/08_reporting/experiments/model_feature/group=xgboost-sensitivity-v2__started_at=20260507T231127138806Z__matrix=f019652c883d --seasons 2021,2022,2023,2024,2025 --model-id xgboost_depth2_slow --feature-pack ppg_xg_matchup`
- H005 feature experiment:
  `uv run --frozen python scripts/run_model_experiments.py --group h005-count-aware-matchup-shrinkage --seasons 2021,2022,2023,2024,2025 --start-round 5 --budget 100 --current-year 2026 --jobs 12 --profile-runtime`
- H005 feature decision:
  `uv run --frozen python scripts/run_h005_feature_decision.py --experiment-path data/08_reporting/experiments/model_feature/<h005-experiment-id> --audit-decision-path data/08_reporting/hypotheses/<h005-audit-id>/h005_mechanism_audit_decision.json`
- Latest H005 run:
  `data/08_reporting/experiments/model_feature/group=h005-count-aware-matchup-shrinkage__started_at=20260513T115300476279Z__matrix=ede90899ae80`; decision status `invalid`, mechanism audit status `invalid`, recomputed count match `ok`, aggregate actual-points delta `-164.14`, 2025 delta `-186.80`.
- H005 is research-only and should not be promoted or directly iterated as another count-reliability tweak. The latest run failed strict audit identity checks and was materially negative even though comparability, fixture identity, candidate signature, and optimizer status were ok.

## M006 Fixed-Blend Diagnostic Workflow

- Artifact-backed M006 fixed-blend diagnostic for the current production-parity source:
  `uv run --frozen python scripts/run_fixed_blend_diagnostic.py --experiment-path data/08_reporting/experiments/model_feature/group=production-parity__started_at=20260513T165550180815Z__matrix=9064290978ca --seasons 2021,2022,2023,2024,2025 --feature-pack ppg_xg --control-model xgboost_depth2_l2_heavy --blend xgb90_ridge10=xgboost_depth2_l2_heavy:0.9,ridge:0.1 --blend xgb80_ridge20=xgboost_depth2_l2_heavy:0.8,ridge:0.2 --blend xgb70_ridge30=xgboost_depth2_l2_heavy:0.7,ridge:0.3 --initial-budget 100 --current-year 2026`
- Focused M006b validation for the `xgb80_ridge20` lead:
  `uv run --frozen python scripts/run_fixed_blend_diagnostic.py --experiment-path data/08_reporting/experiments/model_feature/group=production-parity__started_at=20260513T165550180815Z__matrix=9064290978ca --seasons 2021,2022,2023,2024,2025 --promotion-seasons 2021,2022,2023,2024,2025 --feature-pack ppg_xg --control-model xgboost_depth2_l2_heavy --blend xgb80_ridge20=xgboost_depth2_l2_heavy:0.8,ridge:0.2 --initial-budget 100 --current-year 2026`
- M006 outputs are written under `data/08_reporting/blend_diagnostics/fixed_blend_started_at=.../` and include `fixed_blend_manifest.json`, `blend_complementarity.csv`, `blend_round_results.csv`, `blend_selected_players.csv`, `blend_per_season_summary.csv`, `blend_ranked_summary.csv`, `blend_decision.json`, `invalid_rows.csv`, `blend_budget_paths.html`, and `fixed_blend_report.html`.
- M006 is fixed-blend only: no learned stacker, RF gating, or AutoML. A `candidate_blend` decision is research evidence that still requires explicit promotion before changing live defaults.
- Latest M006 run:
  `data/08_reporting/blend_diagnostics/fixed_blend_started_at=20260514T000703093758Z`; source valid, `candidate_count=0`, best status `weak_positive_research_lead` for `xgb80_ridge20`, aggregate actual-points delta `+226.38`, improved `4 / 5` seasons, 2025 delta `+91.31`. Treat it as a bounded research lead, not a live-default change.
- Latest M006b validation:
  `data/08_reporting/blend_diagnostics/fixed_blend_started_at=20260514T125438183260Z`; source valid, `candidate_count=0`, decision `rejected` for `xgb80_ridge20`. The blend kept the point lead (`+226.38`, `4 / 5` improved seasons, 2025 `+91.31`) but failed the stricter budget-risk gate: worst final-budget delta `-13.73`, worst max-drawdown delta `+10.16`, and `+4` aggregate budget-constrained rounds. Keep `xgboost_depth2_l2_heavy + ppg_xg` as the live default.
- 2020 is now usable for no-fixture production-parity evidence after the focused compatibility audit and full 2020-2025 production-parity rerun completed. Older M006b blend artifacts remain 2021-2025-only unless rerun from a 2020-inclusive source experiment.

## M008 Ridge Promotion Decision Workflow

- Artifact-backed Ridge promotion decision for a completed production-parity experiment:
  `uv run --frozen python scripts/run_ridge_promotion_decision.py --experiment-path data/08_reporting/experiments/model_feature/group=production-parity__started_at=20260514T174628080193Z__matrix=af65eb4223e9 --candidate-model ridge --candidate-feature-pack ppg_xg --control-model xgboost_depth2_l2_heavy --control-feature-pack ppg_xg --baseline-model random_forest --baseline-feature-pack ppg --promotion-seasons 2020,2021,2022,2023,2024,2025`
- M008 reads only `ranked_summary.csv`, `per_season_summary.csv`, `prediction_metrics.csv`, and `comparability_report.json`, then writes `ridge_promotion_decision.json` and `ridge_promotion_decision.md` beside the source experiment.
- Latest M008 decision:
  `data/08_reporting/experiments/model_feature/group=production-parity__started_at=20260514T174628080193Z__matrix=af65eb4223e9/ridge_promotion_decision.json`; status `candidate_requires_budget_guardrail`; recommendation `keep_xgboost_default_until_live_budget_risk_guardrails`.
- M008 result: `ridge + ppg_xg` beats `xgboost_depth2_l2_heavy + ppg_xg` by `+450.82` points over 2020-2025 and passes calibration by exception, but fails budget gates with worst min budget `68.89`, max drawdown delta `+20.59`, and `+3` budget-constrained rounds versus control.
- Do not promote Ridge from M008 until a live budget/risk guardrail milestone is implemented and a fresh decision artifact clears the balanced promotion gates.

## M009 XGBoost Optuna Tuning Workflow

- Bounded Optuna tuning around the current no-fixture XGBoost profile:
  `uv run --frozen python scripts/run_xgboost_optuna_tuning.py --source-experiment-path data/08_reporting/experiments/model_feature/group=production-parity__started_at=20260514T174628080193Z__matrix=af65eb4223e9 --seasons 2020,2021,2022,2023,2024,2025 --n-trials 40 --current-year 2026 --control-model xgboost_depth2_l2_heavy --control-feature-pack ppg_xg --feature-pack ppg_xg --jobs 12 --study-seed 123`
- M009 uses Optuna `TPESampler` with a frozen XGBoost search space: `max_depth` 1-3, `n_estimators` 100-600, `learning_rate` 0.01-0.08, `min_child_weight` 1-20, `subsample`/`colsample_bytree` 0.65-1.0, `reg_lambda` 5-200, `reg_alpha` 0-20, and `gamma` 0-10. Do not add `scale_pos_weight`; this is regression, not class-imbalance classification.
- The tuning objective is balanced: actual-points lift versus the current XGBoost control minus penalties for budget floor failure, excess drawdown, budget-constrained rounds, 2025 regression, and selected-player calibration drift.
- M009 outputs are written under `data/08_reporting/experiments/model_tuning/xgboost_optuna_tuning_started_at=.../` unless `--output-root` is provided, and include `xgboost_optuna_tuning.json`, `xgboost_optuna_tuning.md`, `optuna_trials.csv`, `best_candidate_config.json`, and per-trial backtest artifacts.
- Latest M009 smoke check:
  `data/08_reporting/experiments/model_tuning/xgboost_optuna_tuning_smoke_m009`; one 2020-only trial completed and wrote all expected artifacts. It is a runner validation only, not promotion evidence.
- Optuna proposes candidates only. Promotion still requires rerunning top configs through a frozen full production-parity-style validation and balanced decision gates before any live default change.

## Ridge Tuning Workflow

- Constrained Ridge alpha tuning:
  `uv run --frozen python scripts/run_ridge_tuning.py --seasons 2023,2024,2025 --start-round 5 --budget 100 --current-year 2026 --jobs 12`
- Ridge tuning outputs are written under `data/08_reporting/experiments/model_tuning/<experiment_id>/`.
- The fixed matrix tests Ridge alphas `0.01`, `0.03`, `0.1`, `0.3`, `1.0`, `3.0`, `10.0`, `30.0`, `100.0`, and `300.0` across `ppg` and `ppg_xg`.
- Use `--skip-final-rerun` only for smoke checks; it marks candidates non-promotable and is not promotion evidence.
- Inspect `promotion_report.json` before changing the live no-fixture profile; fixed-budget tuning evidence is superseded unless the full tuning run is rerun under moving budget and clears the promotion gates.

## Live Recommendation Workflow

- Preferred one-command live workflow:
  `uv run --frozen python scripts/run_live_round.py --season 2026 --budget 100 --current-year 2026`
- Opt-in strict matchup live workflow, only after strict fixture capture exists for the open round:
  `uv run --frozen python scripts/run_live_round.py --season 2026 --budget 100 --model-id xgboost_depth2_slow --footystats-mode ppg_xg --fixture-mode strict --matchup-context-mode cartola_matchup_v1 --current-year 2026`
- `scripts/run_live_round.py` defaults to `--capture-policy fresh`; use `missing` to reuse a valid live capture when present, or `skip` to require one without fetching `atletas/mercado`.
- For live and single-round replay recommendations, `--budget` means current available budget for that one round; moving-budget updates apply only to historical multi-round workflows.
- One-command live recommendation outputs are archived under `data/08_reporting/recommendations/{season}/round-{target_round}/live/runs/run_started_at=.../`.
- `scripts/run_live_round.py` and `scripts/recommend_squad.py` support all controlled model IDs plus `--fixture-mode none|strict` and `--matchup-context-mode none|cartola_matchup_v1`; CLI defaults are `xgboost_depth2_l2_heavy`, `ppg_xg`, `fixture_mode=none`, and `matchup_context_mode=none`.
- Capture the open market round before a live recommendation:
  `uv run --frozen python scripts/capture_market_round.py --season 2026 --auto --current-year 2026`
- Generate a live squad recommendation:
  `uv run --frozen python scripts/recommend_squad.py --season 2026 --target-round 14 --mode live --budget 100 --current-year 2026`
- Replay a completed current-season round:
  `uv run --frozen python scripts/recommend_squad.py --season 2026 --target-round 10 --mode replay --budget 100 --footystats-mode ppg --current-year 2026`
- Recommendation outputs are written under `data/08_reporting/recommendations/{season}/round-{target_round}/{mode}/`.
- `recommended_squad.csv` keeps per-player `predicted_points` raw; use `predicted_points_with_captain` and `actual_points_with_captain` for captain-adjusted totals when present.
- Replay recommendation summaries include oracle comparison fields when candidate `pontuacao` is complete: `oracle_actual_points`, `oracle_gap`, `oracle_capture_rate`, and `oracle_optimizer_status`; live recommendations leave these fields null.

## Strict Fixture Capture

- Capture strict pre-lock fixture evidence for the open market round:
  `uv run --frozen python scripts/capture_strict_round_fixture.py --season 2026 --auto --current-year 2026`
- The command writes raw evidence under `data/01_raw/fixtures_snapshots/{season}/` and canonical strict fixture CSV/manifests under `data/01_raw/fixtures_strict/{season}/`.
- Strict matchup live recommendations require these canonical strict fixture CSV/manifests and fail loudly when they are missing.

## Cautions

- `--fixture-mode strict` requires pre-lock snapshot/manifests under `data/01_raw/fixtures_strict/{season}/`; do not claim strict no-leakage fixture evaluation without those files.
- Next milestone: run the full M009 2020-2025 bounded Optuna search, then rerun top candidates through frozen production-parity and balanced promotion gates. Until then, `Provavel` is not confirmed-lineup evidence, so inspect capture age, selected low-sample players, and current budget exposure before trusting live squads.
- CI updates raw data with `uv run --frozen --no-dev python src/cartola/download_data.py` and then `src/cartola/update_readme.py`.
- TODO: Verify the Docker workflow before relying on `make docker`; `Dockerfile` still references Poetry and Python 3.10 while the current project setup uses `uv` and Python 3.13.12.
