# Cartola Champion Squad Selector And Sports Prediction Platform

- Schema: agileforge.spec.v1
- Artifact id: SPEC.cartola-champion-squad-selector
- Status: accepted
- Version: 1.0
- Created: 2026-05-16
- Updated: 2026-05-21
- Markdown profile: agileforge.spec_markdown.v1

## Summary

Define an authority-ready private Cartola FC decision engine whose accepted scope is one legal pre-lock squad recommendation, reproducible recommendation artifacts, leakage-safe evaluation, evidence-gated live defaults, and explicit exclusions for real-money betting, public paid advice, and authenticated squad submission.

## Problem Statement

Cartola managers must choose a squad before market lock with incomplete information about availability, form, fixtures, prices, captain upside, and opponent context. Manual selection is slow and inconsistent, while naive models can win historical backtests but fail live by leaking future data, ignoring budget risk, overfitting one season, or selecting unavailable players. The repository already contains research and operational machinery, but project management needs a structured product-level specification that makes live Cartola squad selection the primary objective and treats betting predictions and subscription advice as future gated products.

## Controlled Terms

### captain-aware points

- Scope: domain
- Definition: Cartola points after applying the official captain multiplier to one non-tecnico selected player.

### live recommendation

- Scope: domain
- Definition: A squad recommendation generated before market lock for a target Cartola round using only pre-lock information.

### moving budget

- Scope: domain
- Definition: Backtest semantics where each strategy carries its own next-round budget from selected historical price variation.

### paper trading

- Scope: domain
- Definition: A future betting-research mode that records forecast probabilities and settlement outcomes without placing wagers or selling paid betting advice.

### strict fixture evidence

- Scope: domain
- Definition: Canonical pre-lock fixture CSV and manifest evidence required before strict matchup context can be used for live advice.

## Items

### ASSUMPTION.betting-research-later - Betting starts as research

- Type: ASSUMPTION
- Status: accepted
- Level: -
- Verification: -
- Tags: betting, future

Statement:

Future betting predictions start as paper-trading research and are not public advice or wager execution.

Acceptance:

- None

### ASSUMPTION.current-default-profile - Current live default profile

- Type: ASSUMPTION
- Status: accepted
- Level: -
- Verification: -
- Tags: default, model

Statement:

The current live default remains xgboost_depth2_l2_heavy with ppg_xg, fixture_mode none, and matchup_context_mode none until a later accepted promotion decision changes it.

Acceptance:

- None

### ASSUMPTION.operator-first - Operator-first product

- Type: ASSUMPTION
- Status: accepted
- Level: -
- Verification: -
- Tags: mvp

Statement:

The first accepted product is for the operator's own Cartola gameplay, not for public subscription distribution.

Acceptance:

- None

### CONSTRAINT.before-market-lock - Live decisions before market lock

- Type: CONSTRAINT
- Status: accepted
- Level: MUST
- Verification: inspection
- Tags: leakage, live

Statement:

Live Cartola decisions must be generated before market lock from available pre-lock information.

Acceptance:

- Live recommendation metadata records market status and capture timestamp.
- Recommendation generation fails when live mode cannot establish a pre-lock evidence boundary.

### CONSTRAINT.compliance-approval - Compliance approval for paid betting-adjacent advice

- Type: CONSTRAINT
- Status: accepted
- Level: MUST_NOT
- Verification: manual-review
- Tags: betting, compliance

Statement:

The system must not publish paid betting-adjacent advice before compliance approval.

Acceptance:

- Paid betting-related publishing surfaces remain disabled until legal review, responsible-gambling controls, jurisdiction rules, advertising disclosures, privacy requirements, and customer-support requirements are documented in accepted specs.
- No paid betting-related advice artifact is delivered to a subscriber or public user before those accepted specs exist.

### CONSTRAINT.fixed-budget-non-comparable - Fixed-budget reports are non-comparable

- Type: CONSTRAINT
- Status: accepted
- Level: MUST
- Verification: inspection
- Tags: backtest, comparability

Statement:

Old fixed-budget reports must remain historical artifacts and must not be mixed with moving-budget promotion evidence.

Acceptance:

- Promotion decisions identify budget policy for compared runs.
- Runs with missing or fixed budget policy are not ranked against moving-budget evidence as equivalent.

### CONSTRAINT.future-public-product-specs - Separate specs for public products

- Type: CONSTRAINT
- Status: accepted
- Level: MUST_NOT
- Verification: manual-review
- Tags: future, governance

Statement:

The system must not enable future real submission, subscription billing or delivery, or betting-adjacent public advice without a separate accepted governing spec.

Acceptance:

- Real Cartola account mutation remains disabled unless a separate accepted submission spec is cited.
- Subscription billing and entitlement features remain absent or disabled unless a separate accepted subscription spec is cited.
- Betting-adjacent public advice remains absent or disabled unless separate accepted product, compliance, privacy, support, and disclosure specs are cited.

### CONSTRAINT.no-guarantee-claims - No guarantee claims

- Type: CONSTRAINT
- Status: accepted
- Level: MUST_NOT
- Verification: inspection
- Tags: claims, compliance, forbidden-capability

Statement:

The system must not publish or generate user-facing claims that guarantee Cartola wins, championships, or betting profit.

Acceptance:

- User-facing recommendation, report, subscription, and advice surfaces do not contain claims that guarantee a Cartola round win.
- User-facing recommendation, report, subscription, and advice surfaces do not contain claims that guarantee a Cartola championship or league win.
- User-facing recommendation, report, subscription, and advice surfaces do not contain claims that guarantee betting profit.

### CONSTRAINT.no-real-money-betting - No real-money betting capability

- Type: CONSTRAINT
- Status: accepted
- Level: MUST_NOT
- Verification: inspection
- Tags: betting, forbidden-capability, scope

Statement:

The system must not provide real-money betting, bookmaker account integration, wager placement, or bankroll automation in this accepted scope.

Acceptance:

- No accepted interface places a wager.
- No accepted interface connects to a bookmaker account.
- No accepted interface automates bankroll allocation or staking.
- Future odds data, when proposed, is limited to paper-trading research until separate accepted specs exist.

### DATA.future-advice-record - Future advice publication record

- Type: DATA
- Status: proposed
- Level: SHOULD
- Verification: inspection
- Tags: data, future, subscription

Statement:

Future public advice must retain an immutable record of what was shown, when it was shown, and which model or version produced it.

Acceptance:

- A future advice record includes user-visible advice text, delivery timestamp, model version, confidence or probability, disclosure text, and delivery channel.

### DATA.market-capture - Live market capture artifact

- Type: DATA
- Status: accepted
- Level: MUST
- Verification: inspection
- Tags: capture, data

Statement:

A live market capture must identify the source market data used for a live recommendation.

Acceptance:

- The capture manifest records season, current_year, target_round, captured_at_utc, status_endpoint, status_final_url, status_http_status, status_response_sha256, market_endpoint, market_final_url, market_http_status, market_response_sha256, status_mercado, csv_path, and csv_sha256.
- The capture CSV path follows data/01_raw/{season}/rodada-{round}.csv for default live captures.
- The capture artifact is linked from the live recommendation metadata.

### DATA.promotion-decision - Promotion decision artifact

- Type: DATA
- Status: accepted
- Level: MUST
- Verification: inspection
- Tags: data, promotion

Statement:

A promotion decision artifact must record why a model, feature pack, policy, tuning candidate, or blend is accepted or rejected for live default use.

Acceptance:

- The artifact records candidate identity, control identity, comparison seasons, budget policy, points delta, budget-risk checks, calibration checks when applicable, comparability status, and final decision.

### DATA.recommendation-output-files - Recommendation output files

- Type: DATA
- Status: accepted
- Level: MUST
- Verification: integration-test
- Tags: artifacts, data, recommendation

Statement:

Each recommendation run must write the canonical recommendation output file set.

Acceptance:

- Each recommendation run writes recommended_squad.csv.
- Each recommendation run writes candidate_predictions.csv.
- Each recommendation run writes recommendation_summary.json.
- Each recommendation run writes run_metadata.json.

### DATA.recommended-squad - Recommended squad artifact

- Type: DATA
- Status: accepted
- Level: MUST
- Verification: inspection
- Tags: data, recommendation

Statement:

A recommended squad artifact must preserve enough information to review, reproduce, and compare a selected live squad.

Acceptance:

- recommended_squad.csv records selected player identity, club, position, price, raw predicted_points, is_captain, captain_policy_ev, captain_policy_safe, and captain_policy_upside.
- recommendation_summary.json records season, target_round, mode, strategy, formation, budget, budget_used, optimizer_status, selected_count, predicted_points_base, captain_bonus_predicted, predicted_points_with_captain, captain_id, captain_name, and output_directory.
- run_metadata.json records model_id, footystats_mode, fixture_mode, matchup_context_mode, feature_columns, training_rounds, candidate_round, generated_at_utc, finalized_live_data_detected, allow_finalized_live_data, optimizer_status, and warnings.

### DATA.submission-plan - Submission plan artifacts

- Type: DATA
- Status: accepted
- Level: MUST
- Verification: integration-test
- Tags: data, safety, submission

Statement:

A Phase 1 submission attempt must persist a reviewable plan and a non-submission result.

Acceptance:

- submission_plan.json records plan_status, phase, recommendation_path, payload, payload_sha256, source_artifact_hashes, target_round, season, formation, selected_count, captain_id, captain_name, model_id, footystats_mode, fixture_mode, matchup_context_mode, scoring_contract_version, and validation_report.
- submission_result.json records submission_status as plan_only, would_submit as false, submitted_at_utc as null, http_status as null, auth_token_present as false, auth_token_source as not_required, and payload_sha256.

### DECISION.focus-private-cartola-first - Focus first on personal Cartola squad selection

- Type: DECISION
- Status: accepted
- Level: -
- Verification: -
- Tags: mvp, scope

Statement:

The selected MVP direction is private Cartola squad selection because it directly supports the operator's current gameplay and uses existing repository strengths.

Rationale:

The alternatives of treating the fork as a generic research clone, focusing first on betting predictions, or selling subscriptions before validation do not match the current operator goal or risk profile.

Acceptance:

- None

### DECISION.keep-real-submit-disabled - Keep real Cartola submit disabled

- Type: DECISION
- Status: accepted
- Level: -
- Verification: -
- Tags: safety, submission

Statement:

Real authenticated Cartola submission remains disabled until a separate save and read-back contract is accepted.

Rationale:

This avoids unverified account mutation while still allowing the operator to review a sanitized submission plan and submit manually.

Acceptance:

- None

### GOAL.evidence-gated-promotion - Evidence-gated promotion

- Type: GOAL
- Status: accepted
- Level: -
- Verification: -
- Tags: governance, promotion

Statement:

Model, feature, optimizer, policy, and tuning changes should become live defaults only after frozen comparable evidence passes documented decision gates.

Acceptance:

- None

### GOAL.future-prediction-platform - Future prediction platform

- Type: GOAL
- Status: proposed
- Level: -
- Verification: -
- Tags: betting-research, future

Statement:

The prediction stack may later expand to football match-event forecasting, including match winner, goals, and cards, after separate validation and data contracts exist.

Acceptance:

- None

### GOAL.future-subscription-advice - Future subscription advice

- Type: GOAL
- Status: proposed
- Level: -
- Verification: -
- Tags: future, subscription

Statement:

A public paid advice product may be considered only after separate specs define performance evidence, compliance, disclosures, billing, privacy, support, and customer safeguards.

Acceptance:

- None

### GOAL.leakage-safe-evidence - Leakage-safe evidence

- Type: GOAL
- Status: accepted
- Level: -
- Verification: -
- Tags: evidence, leakage

Statement:

Historical, replay, and live evidence should maintain explicit data-boundary metadata so future information is not used for pre-event recommendations.

Acceptance:

- None

### GOAL.live-score-lift - Live score lift

- Type: GOAL
- Status: accepted
- Level: -
- Verification: -
- Tags: live, metrics

Statement:

Recommendation quality should be judged by live actual captain-aware points, budget path, captain contribution, DNP exposure, and baseline deltas rather than by historical model metrics alone.

Acceptance:

- None

### GOAL.private-cartola-edge - Private Cartola edge

- Type: GOAL
- Status: accepted
- Level: -
- Verification: -
- Tags: cartola, mvp

Statement:

The initial product should help the operator make better weekly and season-long Cartola decisions than manual selection and practical baselines.

Acceptance:

- None

### INTERFACE.backtest-command - Backtest command

- Type: INTERFACE
- Status: accepted
- Level: MUST
- Verification: integration-test
- Tags: backtest, cli

Statement:

The backtesting command must support historical moving-budget evaluation by season and start round.

Acceptance:

- Backtest runs accept season, start round, budget, and fixture mode inputs.
- Backtest metadata records moving-budget policy and evidence boundaries.

### INTERFACE.capture-market-command - Market capture command

- Type: INTERFACE
- Status: accepted
- Level: MUST
- Verification: integration-test
- Tags: capture, cli, live

Statement:

The market capture command must provide an explicit interface for capturing the open Cartola market round.

Acceptance:

- scripts/capture_market_round.py accepts --season and either --target-round or --auto.
- scripts/capture_market_round.py accepts --current-year, --project-root, and --force inputs.
- A successful capture writes a round CSV and companion capture manifest for the captured target round.

### INTERFACE.cartola-api - Cartola market API integration

- Type: INTERFACE
- Status: accepted
- Level: MAY
- Verification: integration-test
- Tags: cartola, integration

Statement:

The system may use Cartola public market, status, and athlete APIs for live capture and recommendation context.

Acceptance:

- When Cartola public API data is captured, the capture manifest records source identity and capture timestamp.

### INTERFACE.future-odds-source - Future odds source integration

- Type: INTERFACE
- Status: proposed
- Level: MAY
- Verification: inspection
- Tags: future, odds

Statement:

Future betting research may use legally available odds feeds or manually captured odds snapshots.

Acceptance:

- Future odds data records source, timestamp, market identity, offered odds, and legal availability assumptions.
- Odds data is used for paper-trading research only until later compliance specs are accepted.

### INTERFACE.live-round-command - Live round command

- Type: INTERFACE
- Status: accepted
- Level: MUST
- Verification: integration-test
- Tags: cli, live

Statement:

The live round command must provide the preferred one-command interface for private live recommendations.

Acceptance:

- scripts/run_live_round.py accepts --season, --budget, --current-year, --model-id, --footystats-mode, --fixture-mode, --matchup-context-mode, and --capture-policy inputs.
- scripts/run_live_round.py rejects a user-supplied --target-round input because the target round comes from the open market capture.
- A successful one-command live run writes outputs under data/08_reporting/recommendations/{season}/round-{target_round}/live/runs/run_started_at=...

### INTERFACE.replay-recommendation-command - Replay recommendation command

- Type: INTERFACE
- Status: accepted
- Level: MUST
- Verification: integration-test
- Tags: cli, replay

Statement:

The recommendation command must support completed-round replay for current-season review.

Acceptance:

- scripts/recommend_squad.py accepts --season, --target-round, --mode live|replay, --budget, --current-year, --model-id, --footystats-mode, --fixture-mode, and --matchup-context-mode inputs.
- Replay outputs include oracle comparison fields when complete target-round actual points are available.

### INTERFACE.submit-plan-command - Submission plan command

- Type: INTERFACE
- Status: accepted
- Level: MUST
- Verification: system-test
- Tags: cli, safety, submission

Statement:

The submission command must expose Phase 1 plan generation without allowing real authenticated submission.

Acceptance:

- scripts/submit_recommended_squad.py accepts --recommendation-path for Phase 1 plan generation.
- scripts/submit_recommended_squad.py accepts --submission-plan only for reserved future replay behavior and does not use it for Phase 1 real submission.
- When --confirm-submit is present, the command exits with CONTRACT_UNVERIFIED before loading .env or reading CARTOLA_GLB_TOKEN.

### NON_GOAL.downstream-project-tasks - Implementation tasks excluded

- Type: NON_GOAL
- Status: accepted
- Level: -
- Verification: -
- Tags: spec-boundary

Statement:

This spec does not create tickets, implementation tasks, payment systems, project generators, code changes, or downstream AgileForge workflow items.

Acceptance:

- None

### NON_GOAL.guaranteed-wins - Guaranteed wins excluded

- Type: NON_GOAL
- Status: accepted
- Level: -
- Verification: -
- Tags: claims

Statement:

Guaranteed Cartola round wins, championship wins, and betting profit claims are outside the accepted scope.

Acceptance:

- None

### NON_GOAL.paid-betting-advice-now - Paid betting advice excluded

- Type: NON_GOAL
- Status: accepted
- Level: -
- Verification: -
- Tags: compliance, subscription

Statement:

Paid betting-adjacent advice is not part of the accepted scope until legal, advertising, responsible-gambling, and prediction-performance requirements are accepted in later specs.

Acceptance:

- None

### NON_GOAL.real-money-betting - Real-money betting excluded

- Type: NON_GOAL
- Status: accepted
- Level: -
- Verification: -
- Tags: betting, scope

Statement:

Real-money betting, bookmaker account integration, wager placement, and bankroll automation are outside this spec.

Acceptance:

- None

### NON_GOAL.research-as-production - Research output is not production advice

- Type: NON_GOAL
- Status: accepted
- Level: -
- Verification: -
- Tags: production, research

Statement:

Research-only artifacts, oracle hindsight, exploratory fixture evidence, and one-off experiment wins are not accepted production recommendation sources.

Acceptance:

- None

### OPEN_QUESTION.agileforge-spec-update - AgileForge spec update command

- Type: OPEN_QUESTION
- Status: proposed
- Level: -
- Verification: -
- Tags: agileforge, workflow

Statement:

The AgileForge command for future spec updates after project creation is unresolved.

Rationale:

This affects how this living spec stays synchronized with project-management state.

Acceptance:

- None

### OPEN_QUESTION.first-betting-markets - First betting research markets

- Type: OPEN_QUESTION
- Status: proposed
- Level: -
- Verification: -
- Tags: betting, future

Statement:

The first future betting research markets are unresolved: match winner, over or under goals, player cards, team cards, or another market.

Rationale:

This decision changes data acquisition and forecast schema for future paper trading.

Acceptance:

- None

### OPEN_QUESTION.live-validation-window - Live validation window

- Type: OPEN_QUESTION
- Status: proposed
- Level: -
- Verification: -
- Tags: metrics, validation

Statement:

The live sample size or validation window needed before declaring the selector meaningfully better than manual selection or baseline is unresolved.

Rationale:

This decision changes promotion and confidence thresholds.

Acceptance:

- None

### OPEN_QUESTION.primary-cartola-objective - Primary Cartola competition objective

- Type: OPEN_QUESTION
- Status: proposed
- Level: -
- Verification: -
- Tags: objective

Statement:

The most important Cartola objective is unresolved: weekly high score, private league rank, global rank, season-long consistency, or another objective.

Rationale:

This decision changes objective weighting between upside, floor, and budget preservation.

Acceptance:

- None

### OPEN_QUESTION.real-submit-contract - Verified real-submit contract

- Type: OPEN_QUESTION
- Status: proposed
- Level: -
- Verification: -
- Tags: submission

Statement:

The save and read-back contract required before real authenticated Cartola submission is unresolved.

Rationale:

This blocks authenticated squad submission.

Acceptance:

- None

### OPEN_QUESTION.risk-thresholds - Accepted DNP and budget risk thresholds

- Type: OPEN_QUESTION
- Status: proposed
- Level: -
- Verification: -
- Tags: optimizer, risk

Statement:

The acceptable DNP, budget drawdown, and captain-risk thresholds for the operator's style of play are unresolved.

Rationale:

This decision changes optimizer risk policy and promotion gates.

Acceptance:

- None

### OPEN_QUESTION.subscription-jurisdictions - Subscription jurisdictions and languages

- Type: OPEN_QUESTION
- Status: proposed
- Level: -
- Verification: -
- Tags: future, subscription

Statement:

The first jurisdictions and languages for future subscription advice are unresolved.

Rationale:

This decision changes legal, payment, disclosure, localization, and responsible-gambling requirements.

Acceptance:

- None

### QUALITY.live-runtime - Live recommendation runtime

- Type: QUALITY
- Status: accepted
- Level: SHOULD
- Verification: monitoring
- Tags: operations, performance

Statement:

Private live recommendations should complete with enough time for manual review before market lock.

Acceptance:

- For the private MVP, p95 live recommendation runtime is under 5 minutes from command start to completed recommendation artifact on the operator's local machine.
- Long experiment workflows document native thread caps when needed to avoid local machine oversubscription.

### QUALITY.localization-accessibility - Readable outputs and localization

- Type: QUALITY
- Status: accepted
- Level: SHOULD
- Verification: manual-review
- Tags: accessibility, localization

Statement:

Machine-readable artifacts must remain canonical, while human-facing outputs should be readable and use appropriate Cartola terminology.

Acceptance:

- CLI output remains readable in terminals and CI logs.
- CSV and JSON artifacts are treated as canonical machine-readable outputs.
- Future user-facing advice supports Portuguese labels and Cartola terminology.

### QUALITY.observability - Outcome observability

- Type: QUALITY
- Status: accepted
- Level: SHOULD
- Verification: inspection
- Tags: metrics, observability

Statement:

Dashboards and reports should separate live production evidence from historical research evidence.

Acceptance:

- Live performance reports track actual points, baseline deltas, captain contribution, DNP count, budget delta, and recommendation usage by round.
- Research reports distinguish historical backtest, replay, paper-trading, and live production evidence.

### QUALITY.recommendation-traceability - Recommendation traceability

- Type: QUALITY
- Status: accepted
- Level: MUST
- Verification: inspection
- Tags: reliability, traceability

Statement:

Every live recommendation must write enough metadata to reproduce source data identity, capture timing, configuration, model identity, and output paths.

Acceptance:

- Recommendation metadata links to market capture identity, model profile, feature pack, fixture mode, matchup mode, budget, selected squad artifact, and output directory.
- Recommendation run directories are unique and do not overwrite prior live recommendations.

### QUALITY.security-secrets - Protect secrets and account data

- Type: QUALITY
- Status: accepted
- Level: MUST
- Verification: inspection
- Tags: privacy, security

Statement:

The repository and runtime artifacts must not expose secrets, local machine configuration, account tokens, subscriber payment data, or authenticated API payloads.

Acceptance:

- Committed files exclude CARTOLA_GLB_TOKEN, bookmaker credentials, subscriber payment data, and authenticated API payloads.
- Committed files exclude local machine configuration from conf/local except placeholder files intentionally kept for directory structure.
- Phase 1 submission planning does not read CARTOLA_GLB_TOKEN.
- Generated submission_result.json records auth_token_present as false and auth_token_source as not_required.

### QUALITY.utc-timestamps - UTC timestamps

- Type: QUALITY
- Status: accepted
- Level: MUST
- Verification: inspection
- Tags: data, time

Statement:

Persisted metadata timestamps that identify run, capture, delivery, or settlement events must use UTC ISO-8601 format.

Acceptance:

- Run, capture, delivery, and settlement event metadata use UTC ISO-8601 timestamps.

### REQ.baseline-comparison - Compare against practical baselines

- Type: REQ
- Status: accepted
- Level: MUST
- Verification: inspection
- Tags: baseline, metrics

Statement:

Recommendation artifacts must preserve baseline scores and replay oracle fields needed to compare selected squads against practical alternatives.

Acceptance:

- recommended_squad.csv includes baseline_score and price_score columns for selected players.
- candidate_predictions.csv includes baseline_score and price_score columns for candidate players.
- Replay recommendation_summary.json includes oracle_actual_points, oracle_gap, oracle_capture_rate, and oracle_optimizer_status when complete target-round actual points are available.

### REQ.betting-paper-trading - Track future betting forecasts as probabilities

- Type: REQ
- Status: proposed
- Level: SHOULD
- Verification: analysis
- Tags: betting, future

Statement:

Future betting predictions must be tracked as calibrated probability forecasts, not guaranteed picks.

Acceptance:

- Each future forecast records event, market, timestamp, data boundary, predicted probability, fair odds, offered odds when available, and realized outcome after settlement.
- Paper-trading reports include calibration, Brier score or log loss, expected-value assumptions, closing-line comparison when odds are available, and drawdown simulation.

### REQ.budget-bound - Stay within available budget

- Type: REQ
- Status: accepted
- Level: MUST
- Verification: system-test
- Tags: budget, optimizer

Statement:

The live squad must stay within the operator's current available Cartola budget.

Acceptance:

- For every live recommendation, recommendation_summary.json records budget_used &lt;= budget.
- For every live recommendation, recommended_squad.csv contains selected players whose summed preco_pre_rodada equals the recorded budget_used.
- When the available budget is missing, the live recommendation workflow does not infer a private account balance and requires an explicit operator-provided budget.

### REQ.captain-aware-optimization - Optimize captain-aware points

- Type: REQ
- Status: accepted
- Level: MUST
- Verification: system-test
- Tags: captain, scoring

Statement:

The optimizer must select recommendations using captain-aware predicted Cartola points while preserving raw per-player predicted points.

Acceptance:

- recommendation_summary.json records predicted_points_base, captain_bonus_predicted, and predicted_points_with_captain.
- recommendation_summary.json records captain_multiplier as 1.5 through the scoring contract fields.
- Round-level predicted totals use predicted_points_with_captain.
- Selected-player predicted_points remains the raw per-athlete prediction before captain multiplier.
- recommended_squad.csv contains exactly one is_captain=true row, and that row is not a tecnico.

### REQ.default-promotion-gate - Gate live default changes

- Type: REQ
- Status: accepted
- Level: MUST
- Verification: inspection
- Tags: governance, promotion

Statement:

The current live default must change only when a frozen decision artifact passes the accepted promotion gates.

Acceptance:

- Every live default change cites a frozen decision artifact.
- The cited decision artifact records candidate model_id, candidate feature pack or mode, control identity, points delta, budget-risk checks, DNP or availability checks when applicable, calibration checks when applicable, comparability status, and final decision.
- A one-off experiment win, oracle hindsight run, or exploratory fixture-only run cannot change the live default.

### REQ.exploratory-evidence-labeling - Label exploratory fixture evidence

- Type: REQ
- Status: accepted
- Level: MUST
- Verification: inspection
- Tags: fixtures, research

Statement:

Exploratory fixture evidence must remain labeled as research-only for strict no-leakage claims.

Acceptance:

- Metadata distinguishes exploratory fixture mode from strict fixture mode.
- Exploratory-only runs cannot justify strict no-leakage production claims.

### REQ.legal-roster - Satisfy Cartola roster rules

- Type: REQ
- Status: accepted
- Level: MUST
- Verification: system-test
- Tags: cartola, optimizer

Statement:

The selected live squad must satisfy Cartola roster rules.

Acceptance:

- The selected squad contains exactly 12 rows.
- The selected squad contains exactly one tecnico and 11 non-tecnico players.
- The selected squad contains exactly one non-tecnico captain.
- The selected squad uses one official Cartola formation.

### REQ.live-squad-recommendation - Recommend one live squad

- Type: REQ
- Status: accepted
- Level: MUST
- Verification: acceptance-test
- Tags: cartola, live

Statement:

The system must recommend exactly one live Cartola squad for the operator before market lock for the target round.

Acceptance:

- Given the Cartola market is open and a target round is available, when a live recommendation run completes, then the output identifies one selected squad, one formation, one captain, predicted total, budget used, target round, and recommendation metadata.
- Given the same completed live run, when the recommendation artifacts are inspected, then there is exactly one primary recommended squad marked as the operator-facing recommendation.

### REQ.moving-budget-backtests - Use moving-budget backtests

- Type: REQ
- Status: accepted
- Level: MUST
- Verification: system-test
- Tags: backtest, budget

Statement:

Historical backtests must use moving-budget semantics for strategy comparison.

Acceptance:

- Each strategy records budget_policy as moving.
- Each strategy records budget before and after each round, selected-player price variation, final budget, minimum budget, budget-constrained rounds, and maximum drawdown.
- Reports do not compare fixed-budget and moving-budget evidence as equivalent.

### REQ.no-finalized-live-outcomes - Suppress finalized target-round outcomes

- Type: REQ
- Status: accepted
- Level: MUST
- Verification: system-test
- Tags: leakage, live

Statement:

Live recommendations must not include finalized target-round actual points or scout outcomes.

Acceptance:

- Live recommended_squad.csv omits target-round pontuacao and target-round scout outcome columns.
- Live candidate_predictions.csv omits target-round pontuacao and target-round scout outcome columns.
- If finalized target-round data is present during live recommendation generation, the workflow fails unless an explicit debug-only mode is selected.

### REQ.post-round-review - Track completed-round outcome quality

- Type: REQ
- Status: accepted
- Level: MUST
- Verification: acceptance-test
- Tags: metrics, review

Statement:

The system must support post-round review for each completed live recommendation.

Acceptance:

- A post-round review links to the exact recommendation artifact it evaluates.
- The review records actual_points_with_captain, captain_bonus_actual, captain result, DNP count, budget delta, oracle_gap when available, and baseline comparisons.

### REQ.prelock-market-capture - Capture or validate pre-lock market data

- Type: REQ
- Status: accepted
- Level: MUST
- Verification: integration-test
- Tags: capture, live

Statement:

The live recommendation workflow must capture or validate pre-lock market data before producing live advice.

Acceptance:

- A live run records capture path, capture hash, capture timestamp, target round, market status, and capture policy in run_metadata.json or the live workflow metadata.
- When the market is unavailable or closed, live mode fails before squad generation unless the operator explicitly runs replay mode.

### REQ.real-submit-disabled - Disable real authenticated submission

- Type: REQ
- Status: accepted
- Level: MUST_NOT
- Verification: system-test
- Tags: security, submission

Statement:

The system must not perform real authenticated Cartola squad submission until a separate Phase 2 submission spec is accepted.

Acceptance:

- Any --confirm-submit invocation exits with CONTRACT_UNVERIFIED.
- The disabled real-submit path exits before reading account tokens.
- The disabled real-submit path exits before constructing an authenticated POST request.

### REQ.repo-quality-gate - Keep repository quality gates reproducible

- Type: REQ
- Status: accepted
- Level: MUST
- Verification: system-test
- Tags: quality, repository

Statement:

Repository quality checks must remain reproducible with the documented uv workflow.

Acceptance:

- uv sync --locked --dev completes in a clean checkout.
- uv run --frozen pyrepo-check --all reproduces the expected quality gate.

### REQ.research-not-direct-production - Keep research diagnostics out of direct production advice

- Type: REQ
- Status: accepted
- Level: MUST
- Verification: inspection
- Tags: production, research

Statement:

Research diagnostics must not directly change production advice.

Acceptance:

- Oracle discovery, EBM diagnostics, policy simulation, and hypothesis diagnostics record research-only status.
- Research-only artifacts require frozen validation before promotion to live defaults.

### REQ.strict-fixture-fail-closed - Fail closed for strict fixture mode

- Type: REQ
- Status: accepted
- Level: MUST
- Verification: integration-test
- Tags: fixtures, leakage

Statement:

Strict fixture and matchup context must fail before recommendation when required strict evidence is missing or invalid.

Acceptance:

- fixture_mode strict requires canonical strict fixture CSV and manifest files.
- Missing or invalid strict fixture evidence fails before recommendation generation.
- Strict live matchup context cannot run from exploratory fixture evidence alone.

### REQ.submission-plan-only - Generate submission plan without real submit

- Type: REQ
- Status: accepted
- Level: MUST
- Verification: system-test
- Tags: safety, submission

Statement:

The operator must be able to review a sanitized Cartola squad submission plan without real authenticated submission.

Acceptance:

- Submission planning writes submission_plan.json beside the recommendation run.
- Submission planning writes submission_result.json indicating that real submit is disabled.
- Submission planning does not mutate a Cartola account.

### REQ.subscription-disclosures - Disclose limitations for future subscription advice

- Type: REQ
- Status: proposed
- Level: SHOULD
- Verification: manual-review
- Tags: disclosure, future, subscription

Statement:

Future subscription advice must disclose limitations, confidence, material conflicts, and historical-performance context.

Acceptance:

- Future public advice includes timestamp, model version, confidence or probability, historical performance window, and non-guarantee language.
- Future public advice discloses material connections or conflicts when they affect user interpretation.

### REQ.walk-forward-boundary - Prevent future-round leakage

- Type: REQ
- Status: accepted
- Level: MUST
- Verification: system-test
- Tags: backtest, leakage

Statement:

Historical and replay evaluation must train only on information available before the evaluated target round.

Acceptance:

- For target round N, training uses only rounds earlier than N.
- Candidate optimization for target round N uses only round N candidate data.
- Backtest and replay metadata records the evidence boundary used for each target round.

### RISK.overfit-research - Research overfit

- Type: RISK
- Status: accepted
- Level: -
- Verification: -
- Tags: promotion, research

Statement:

A historical variant can outperform in backtests while harming live score, budget, or DNP exposure if promotion gates are weak.

Acceptance:

- None

### RISK.public-advice-harm - Public advice harm

- Type: RISK
- Status: accepted
- Level: -
- Verification: -
- Tags: compliance, future

Statement:

Future paid advice can mislead users or create legal exposure if performance, limitations, conflicts, and jurisdiction restrictions are not clear.

Acceptance:

- None

### RISK.stale-market-data - Stale market data

- Type: RISK
- Status: accepted
- Level: -
- Verification: -
- Tags: availability, live

Statement:

Player status can change after capture and before market lock, so late manual review remains necessary until stronger availability controls are validated.

Acceptance:

- None

## Relations

- CONSTRAINT.before-market-lock constrains REQ.live-squad-recommendation
- CONSTRAINT.future-public-product-specs constrains NON_GOAL.paid-betting-advice-now
- CONSTRAINT.no-guarantee-claims clarifies NON_GOAL.guaranteed-wins
- CONSTRAINT.no-real-money-betting clarifies NON_GOAL.real-money-betting
- DATA.recommendation-output-files implements REQ.live-squad-recommendation
- DATA.submission-plan implements REQ.submission-plan-only
- INTERFACE.capture-market-command implements REQ.prelock-market-capture
- INTERFACE.live-round-command implements REQ.live-squad-recommendation
- INTERFACE.submit-plan-command implements REQ.submission-plan-only
- QUALITY.recommendation-traceability tracks REQ.post-round-review
- REQ.betting-paper-trading depends_on CONSTRAINT.compliance-approval
- REQ.default-promotion-gate satisfies GOAL.evidence-gated-promotion
- REQ.live-squad-recommendation satisfies GOAL.private-cartola-edge
- REQ.post-round-review verifies GOAL.live-score-lift
- REQ.real-submit-disabled implements DECISION.keep-real-submit-disabled
- REQ.subscription-disclosures depends_on CONSTRAINT.future-public-product-specs
- REQ.walk-forward-boundary satisfies GOAL.leakage-safe-evidence

<!-- agileforge-review-notes:start -->
<!-- agileforge-review-notes:end -->
