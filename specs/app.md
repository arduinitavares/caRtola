# Technical Specification: Cartola Champion Squad Selector And Sports Prediction Platform

**Status:** Changed  
**Version:** 0.2  
**Created:** 2026-05-16  
**Last Updated:** 2026-05-16  
**Owner:** caRtola maintainer/operator  
**Reviewers:** Unknown  

## 1. Summary

This project uses the forked caRtola repository as the technical foundation for a personal Cartola FC decision engine whose first mission is to select the highest-scoring weekly squad and maximize the operator's chance of winning Cartola leagues. The near-term product is not a generic research archive; it is a disciplined fantasy-football advantage system with leakage-safe backtests, live recommendations, budget-aware squad optimization, and post-round learning loops. Future product lines may expand into football match-event prediction and paid subscription advice, but those tracks require separate validation, compliance, responsible-gambling, and customer-protection gates before public launch.

## 2. Problem Statement

Cartola managers must choose a squad before market lock with incomplete information about player availability, form, fixtures, price movement, captain upside, and opponent context. Manual selection is slow and inconsistent, and naive models can win historical backtests while failing live because they leak future data, ignore budget risk, overfit one season, or choose unavailable players.

The current repository has useful research and operational machinery, but the project-level spec should reflect the actual owner goal:

- win Cartola rounds and leagues through the best possible squad selector;
- measure recommendations against live outcomes and strong baselines;
- keep research-only ideas out of production defaults until they clear evidence gates;
- later reuse the prediction stack for football betting markets such as match winner, goals, cards, and similar outcomes;
- eventually sell advice as a subscription only when accuracy, compliance, disclosures, and customer safeguards are ready.

Confirmed facts:

- The project is forked from caRtola and uses existing historical Cartola data, scripts, feature specs, and backtesting modules as reference implementation.
- The active Python package is `src/cartola`; operational scripts live under `scripts/`.
- Backtests and experiments currently use moving-budget semantics and the `cartola_standard_2026_v1` scoring contract.
- Live recommendations default to `xgboost_depth2_l2_heavy + ppg_xg`, `fixture_mode=none`, and `matchup_context_mode=none`.
- Real authenticated Cartola squad submission is disabled in the current submission-planning phase.

Key assumptions:

- The initial product is for the operator's own Cartola gameplay, not a public subscription.
- Betting predictions are a future research and product track, not part of the initial Cartola squad-selection MVP.
- Public paid advice must avoid guarantees of profit, championship outcomes, or betting success.
- Any betting-adjacent product must pass jurisdiction-specific legal review before launch.

## 3. Goals And Non-Goals

### Goals

- Build and operate the best practical Cartola squad selector for weekly live decisions.
- Maximize live Cartola scoring, not only historical model metrics.
- Track whether recommendations outperform strong baselines across points, budget path, captain choice, DNP exposure, and league-relevant outcomes.
- Preserve strict no-leakage evaluation rules for all historical and replay evidence.
- Promote model, feature, optimizer, or policy changes only from frozen comparable artifacts with explicit decision gates.
- Maintain a clear product roadmap from private Cartola assistant to future public advice product.
- Define compliance and consumer-protection gates for future betting predictions and subscriptions.

### Non-Goals

- Do not guarantee that the system will win every round, win a championship, or produce betting profit.
- Do not enable real-money betting, bookmaker account integration, wager placement, or bankroll automation.
- Do not sell paid betting advice until legal, advertising, responsible-gambling, and prediction-performance gates are met.
- Do not expose research-only artifacts as user-facing recommendations.
- Do not change live defaults from one-off experiment wins, oracle hindsight, or exploratory fixture evidence.
- Do not implement tickets, code, project generators, payment systems, or downstream AgileForge tasks in this spec.
- Do not treat the original caRtola repository's historical/data-science goals as the final product goal.

## 4. Users And Stakeholders

- **Primary user:** the operator playing Cartola and trying to maximize weekly and season-long score.
- **Secondary users:** future trusted testers who compare recommendations against their own Cartola decisions.
- **Future subscribers:** paying users who may receive fantasy-football or betting-adjacent advice after public-product gates are satisfied.
- **Internal stakeholders:** maintainer/operator, model researcher, data engineer, future compliance/legal reviewer, future customer-support operator.
- **External systems:** Cartola market/status and athlete APIs, historical Cartola data, FootyStats local files, fixture sources, optional betting-market odds sources, payment/subscription provider, marketing channels, and AgileForge project management.

## 5. Current State

The repository is a Python `uv` project targeting Python `3.13.12`, with main package code in `src/cartola`, tests in `src/tests`, and operational scripts in `scripts/`. Kedro remains part of the inherited project shape, while current decision workflows live primarily in `cartola.backtesting` and script entrypoints.

Current useful capabilities include:

- Historical season loading and compatibility audits across local Cartola raw seasons.
- Walk-forward moving-budget backtests.
- Captain-aware optimizer that searches official formations, selects one non-tecnico captain, and applies the `1.5x` captain multiplier.
- Fixture modes: `none`, `exploratory`, and `strict` for historical contexts; strict live fixture support requires pre-lock evidence.
- FootyStats modes: `none`, `ppg`, and `ppg_xg`.
- Optional `cartola_matchup_v1` matchup context for controlled research and strict live opt-in.
- Model-feature experiments, promotion decisions, policy simulations, oracle discovery, EBM diagnostics, hypothesis diagnostics, fixed-blend diagnostics, and XGBoost tuning.
- One-command live recommendation through `scripts/run_live_round.py`.
- Phase 1 submission planning through `scripts/submit_recommended_squad.py`, with real submit intentionally disabled.

The current project weakness is product focus: many research tools exist, but the governing spec must make live Cartola victory the primary objective and treat betting/subscription ideas as staged future products with stronger gates.

Primary reference files:

- `README.md`
- `AGENTS.md`
- `roadmap.md`
- `pyproject.toml`
- `docs/superpowers/specs/`
- `scripts/run_live_round.py`
- `scripts/recommend_squad.py`
- `scripts/submit_recommended_squad.py`
- `src/cartola/backtesting/`

## 6. Proposed Specification

### 6.1 Product Phases

| Phase | Product Stage | Scope | Exit Criteria |
| --- | --- | --- | --- |
| P1 | Private Cartola champion assistant | Live squad recommendation, backtesting, budget tracking, captain selection, manual review, and post-round learning for the operator's own gameplay. | Recommendations are generated for live rounds with auditable metadata and tracked against baselines for at least one meaningful live sample. |
| P2 | Advanced Cartola edge engine | Adds strictly validated availability, fixture, risk, portfolio, and lineup-diversification signals when evidence shows live lift. | New signals beat the current default under frozen comparable validation and do not increase unacceptable budget/DNP risk. |
| P3 | Football match-event prediction research | Predicts match winner, goals, cards, and related markets using historical match/event data and odds where legally available. | Paper-trading reports show calibrated probabilities, closing-line comparison, and no use of leaked post-event information. |
| P4 | Private beta advice product | Gives selected fantasy-football advice to trusted users with clear disclaimers and no betting execution. | Users receive timestamped advice, performance reports, and transparent limitation disclosures. |
| P5 | Paid subscription advice product | Public subscription for fantasy advice and, only if approved, betting-adjacent analysis. | Legal, payment, privacy, marketing, responsible-gambling, and customer-support requirements are accepted in separate specs. |

### 6.2 Functional Requirements

| ID | Requirement | Acceptance Criteria | Priority |
| --- | --- | --- | --- |
| FR-001 | The system must recommend one live Cartola squad for the operator before market lock. | A live run outputs exactly one selected squad, formation, captain, predicted total, budget used, and recommendation metadata for the captured target round. | Must |
| FR-002 | The live squad must satisfy Cartola roster rules. | The selected squad contains exactly 12 rows, exactly one tecnico, 11 non-tecnico players, one non-tecnico captain, and one official formation. | Must |
| FR-003 | The live squad must stay within the operator's current budget. | `budget_used <= budget` in recommendation summary and selected-player artifacts. | Must |
| FR-004 | The system must optimize for expected captain-aware Cartola points. | Round-level predicted totals use `predicted_points_with_captain`; selected-player `predicted_points` remains raw per-athlete score. | Must |
| FR-005 | The system must capture or validate pre-lock market data before live recommendation. | `scripts/run_live_round.py` records capture path, capture hash, capture timestamp, target round, market status, and capture policy. | Must |
| FR-006 | Live recommendations must suppress finalized target-round outcomes. | Live outputs omit target-round actual points and scout outcomes; finalized data causes failure unless an explicit debug flag is set. | Must |
| FR-007 | Historical and replay evaluation must prevent future-round leakage. | For target round `N`, training uses only rounds `< N`; candidate optimization uses only round `N`; metadata records the evidence boundary. | Must |
| FR-008 | Historical backtests must use moving-budget semantics. | Each strategy records `budget_policy=moving`, budget before/after each round, selected-player `variacao`, final budget, min budget, constrained rounds, and max drawdown. | Must |
| FR-009 | The current live default must remain evidence-governed. | Default profile changes only when a frozen decision artifact passes stated points, budget, DNP, calibration, and comparability gates. | Must |
| FR-010 | The recommendation must track live outcome quality after each completed round. | Post-round review records actual captain-aware points, captain result, DNP count, budget delta, oracle gap when available, and comparison to baseline/default alternatives. | Must |
| FR-011 | The system must compare recommendations against strong practical baselines. | Reports include comparisons to at least current default, price/budget baseline, and prior promoted baseline where applicable. | Must |
| FR-012 | Strict fixture and matchup context must fail closed when evidence is missing. | `fixture_mode=strict` requires canonical strict fixture CSV and manifest files; missing or invalid evidence fails before recommendation. | Must |
| FR-013 | Exploratory fixture evidence must remain research-only for strict claims. | Metadata distinguishes `exploratory` from `strict`; exploratory-only runs cannot justify strict no-leakage claims. | Must |
| FR-014 | Research diagnostics must not directly change production advice. | Oracle discovery, EBM diagnostics, policy simulation, and hypothesis diagnostics record research-only status and require frozen validation before promotion. | Must |
| FR-015 | The operator must be able to review a submission plan without real submit. | `scripts/submit_recommended_squad.py --recommendation-path ...` writes sanitized `submission_plan.json` and `submission_result.json`. | Must |
| FR-016 | Real authenticated Cartola submission must remain disabled until a separate Phase 2 submission spec is accepted. | Any `--confirm-submit` invocation exits with `CONTRACT_UNVERIFIED` before reading tokens or constructing a POST request. | Must |
| FR-017 | Future betting predictions must be tracked as calibrated probability forecasts, not guaranteed picks. | Each prediction records event, market, timestamp, input data boundary, predicted probability, fair odds, offered odds when used, and realized outcome after settlement. | Should |
| FR-018 | Betting prediction research must run in paper-trading mode before any monetized use. | Reports include calibration, Brier/log loss, expected value assumptions, closing-line comparison when odds are available, and drawdown simulation. | Should |
| FR-019 | Subscription advice must disclose limitations and conflicts. | Public advice pages or messages show timestamp, model/version, confidence/probability, historical performance window, and clear non-guarantee language. | Should |
| FR-020 | The system must not publish paid betting-adjacent advice before compliance approval. | Paid betting-related features stay disabled until legal/compliance review, responsible-gambling controls, jurisdiction rules, and advertising disclosures are documented in accepted specs. | Must |
| FR-021 | Repository quality gates must remain reproducible. | `uv sync --locked --dev` and `uv run --frozen pyrepo-check --all` reproduce the GitHub Actions quality gate. | Must |

### 6.3 User Scenarios

```gherkin
Scenario: Choose the best live Cartola squad before market lock
  Given the Cartola market is open for the current round
  When the operator runs the live recommendation workflow with the current budget
  Then the system captures or validates the live market data
  And recommends one legal squad, captain, formation, predicted total, and budget usage
```

```gherkin
Scenario: Learn from a completed Cartola round
  Given a live recommendation was generated before market lock
  And the target round has completed
  When the operator runs or records a post-round review
  Then the system compares recommended score, actual score, captain performance, DNP exposure, budget delta, and baseline alternatives
```

```gherkin
Scenario: Prevent a research artifact from becoming public advice
  Given a research run finds a high-scoring historical variant
  When the variant lacks a frozen promotion decision with comparability and budget-risk checks
  Then the system must keep the current live default unchanged
```

```gherkin
Scenario: Research a future betting market without taking bets
  Given historical match and event data exists for a supported competition
  When the system produces match-winner, goals, or card forecasts
  Then each forecast is stored as a timestamped probability with data-boundary metadata
  And performance is evaluated after settlement in paper-trading reports
```

```gherkin
Scenario: Publish subscription advice only after product gates
  Given a paid advice product is proposed
  When compliance, disclosure, performance, support, and billing requirements are not accepted
  Then the system must not sell or distribute paid betting-adjacent advice
```

### 6.4 Data And State

Core entities:

- **Live market capture:** pre-lock Cartola market CSV plus `.capture.json` manifest containing season, round, capture timestamp, market status, CSV hash, and source identity.
- **Player prediction:** per-athlete target-round prediction with model ID, feature pack, raw predicted points, availability indicators, price, club, position, and data-boundary metadata.
- **Recommended squad:** selected legal squad with formation, captain, predicted points with captain, budget used, optimizer status, and selected-player rows.
- **Live outcome review:** post-round record linking a live recommendation to actual points, captain result, DNP exposure, budget delta, oracle gap, and baseline comparisons.
- **Promotion decision:** frozen artifact that determines whether a model, feature pack, policy, tuning candidate, or blend can become a live default.
- **Betting forecast:** future-stage record of a match-event probability, fair odds, available odds if captured, timestamp, data inputs, settlement result, and calibration metrics.
- **Advice publication:** future-stage timestamped recommendation shown to a user or subscriber with model version, confidence, limitations, and compliance disclosures.
- **Subscriber account:** future-stage customer identity, entitlement, billing status, delivery preferences, and consent records.

Important invariants:

- Live advice must be generated only from information available before market lock or event start.
- Historical results can be used for review and training, but not for pre-event recommendation generation.
- Fantasy recommendation confidence must be expressed as expected value or probability, not certainty.
- Betting forecasts must remain paper-trading outputs until compliance and monetization specs are accepted.
- Subscription advice must retain a record of what was shown, when it was shown, and which model/version produced it.

### 6.5 Interfaces And Integrations

Primary local commands:

```bash
uv run --frozen python scripts/run_live_round.py --season 2026 --budget 100 --current-year 2026
uv run --frozen python scripts/recommend_squad.py --season 2026 --target-round 10 --mode replay --budget 100 --current-year 2026
uv run --frozen python -m cartola.backtesting.cli --season 2025 --start-round 5 --budget 100 --fixture-mode none
uv run --frozen python scripts/run_model_experiments.py --group production-parity --seasons 2021,2022,2023,2024,2025 --start-round 5 --budget 100 --current-year 2026
uv run --frozen python scripts/submit_recommended_squad.py --recommendation-path data/08_reporting/recommendations/2026/round-16/live/runs/run_started_at=...
```

Current integrations:

- Cartola public market/status and athlete APIs for live capture and recommendation context.
- Local historical Cartola raw data under `data/01_raw/`.
- TheSportsDB and fixture snapshot files for fixture context.
- FootyStats local CSV files for pre-match PPG and xG features.
- XGBoost and related Python ML libraries.
- AgileForge CLI for project management using this spec file.

Future integrations:

- Legally available odds feeds or manually captured odds snapshots for betting-market research.
- Payment/subscription provider for future paid advice.
- Email, messaging, web, or mobile delivery channels for future subscribers.
- Identity, consent, age-gating, and jurisdiction controls if betting-adjacent advice is sold.

Regulatory and advertising references that shape future scope:

- Brazil's Ministry of Finance fixed-odds betting guidance states that fixed-odds betting in Brazil requires prior authorization from the Secretariat of Prizes and Betting, and nationally authorized betting sites use the `.bet.br` domain: <https://www.gov.br/fazenda/pt-br/composicao/orgaos/secretaria-de-premios-e-apostas/apostas-de-quota-fixa/apostas-de-quota-fixa>
- FTC endorsement and testimonial guidance says advertising and paid recommendations must avoid deceptive claims and require clear disclosures when material connections affect how consumers evaluate a recommendation: <https://www.ftc.gov/business-guidance/advertising-marketing/endorsements-influencers-reviews>

### 6.6 Error Handling And Edge Cases

| Case | Required Behavior | User/System Impact |
| --- | --- | --- |
| Live market unavailable or closed | Live recommendation fails before squad generation unless explicitly running replay mode. | Prevents stale or invalid live advice. |
| Missing current budget | Operator must provide current usable budget; system does not infer private account balance. | Prevents illegal squad cost. |
| Player status changes after capture | Recommendation metadata exposes capture time and age; operator reviews late changes before manual submission. | Makes stale advice visible. |
| Selected player does not play | Post-round review records DNP count and impact. | Supports future availability/risk modeling. |
| Model has high historical points but weak live/budget evidence | Promotion is blocked until balanced guardrails pass. | Protects live performance. |
| Strict fixture evidence missing | Strict mode fails before recommendation. | Prevents false no-leakage claims. |
| Betting forecast has no odds snapshot | Forecast can be evaluated for calibration but not expected-value or closing-line performance. | Keeps research claims honest. |
| Paid advice user asks for guaranteed profit | Product must refuse guarantee language and show limitation/risk disclosure. | Reduces deceptive marketing risk. |
| User is in unsupported betting jurisdiction | Betting-adjacent paid advice is unavailable until jurisdiction rules are accepted. | Reduces legal risk. |
| Real Cartola submit requested before Phase 2 | Command fails with `CONTRACT_UNVERIFIED` before auth or POST handling. | Prevents unverified account mutation. |

## 7. Quality Attributes

### Security And Privacy

- The system must not commit secrets, account tokens, local machine config, betting account details, subscriber payment data, or authenticated API payloads.
- Phase 1 Cartola submission planning must not read `CARTOLA_GLB_TOKEN`.
- Future authenticated Cartola submission must verify account/team identity before POST and must not serialize tokens.
- Future subscription features must store only necessary user data and must document retention, deletion, and access controls before launch.
- Future betting-adjacent features must not store bookmaker credentials or place wagers.

### Performance And Scale

- Private live recommendations should complete with enough time for manual review before market lock; a target of p95 under 5 minutes from command start to recommendation artifact is acceptable for the private MVP.
- Historical moving-budget backtests prioritize correctness over target-round parallelism.
- Long experiments should cap native ML threads to avoid local machine oversubscription.
- Future subscription delivery must define separate latency targets before public launch.

### Reliability And Operations

- Every live recommendation must write enough metadata to reproduce source data identity, capture timing, configuration, model identity, and output paths.
- Every post-round review must link back to the exact recommendation artifact it evaluates.
- Recommendation run directories must be unique and must not overwrite prior live recommendations.
- Public advice, if launched, must keep immutable records of advice text, delivery time, model version, and user-visible disclosures.

### Observability

- Live performance dashboards should track actual points, baseline deltas, captain contribution, DNP count, budget delta, and recommendation usage by round.
- Research dashboards should separate historical backtest, replay, paper-trading, and live production evidence.
- Subscription dashboards, if launched, must distinguish model performance from customer acquisition, retention, refunds, and support metrics.

### Accessibility And Localization

- CLI output should remain readable in terminals and CI logs.
- CSV/JSON artifacts are canonical machine-readable outputs; HTML reports are reviewer aids.
- User-facing advice should support Portuguese labels and Cartola terminology.
- Timestamps in persisted metadata must use UTC ISO-8601 when they identify run, capture, delivery, or settlement events.

## 8. Alternatives Considered

| Option | Pros | Cons | Decision |
| --- | --- | --- | --- |
| Treat the fork as a generic caRtola research clone | Preserves inherited repository purpose | Does not align with the owner's goal of winning Cartola and later monetizing advice | Rejected |
| Focus first on betting predictions | Larger future monetization path | Requires new data, legal review, odds evaluation, and responsible-gambling controls before it can be trusted or sold | Rejected for MVP |
| Focus first on personal Cartola squad selection | Directly supports the operator's current gameplay and uses existing repo strengths | Monetization comes later | Chosen |
| Sell subscriptions before live validation | Faster revenue experiment | High risk of misleading claims, refunds, user harm, and legal/advertising issues | Rejected |
| Promote every backtest winner automatically | Speeds experimentation | Overfits history and can damage live score or budget path | Rejected |
| Keep real Cartola submission disabled | Avoids unverified account mutation | Operator must still manually submit the lineup | Chosen until a separate submit contract is accepted |

## 9. Dependencies And Constraints

- **Dependencies:** Python 3.13.12, `uv`, Kedro, pandas, scikit-learn, XGBoost, PuLP, Plotly, Rich, Optuna, Cartola public APIs, historical Cartola data, FootyStats data, fixture sources, optional odds data, future payment and messaging providers, AgileForge CLI.
- **Constraints:** Live decisions must happen before market lock; historical evidence must avoid leakage; moving-budget artifacts are not comparable to fixed-budget artifacts; strict fixture evaluation requires pre-lock fixture evidence; real submission and paid betting advice require separate accepted specs.
- **Compliance constraints:** Betting and paid advice may trigger jurisdiction-specific gambling, advertising, consumer-protection, tax, privacy, age-gating, and responsible-gambling requirements. This spec does not approve launch of those features.
- **Assumptions:** The current live default remains `xgboost_depth2_l2_heavy + ppg_xg` until a fresh promotion decision clears guardrails. Betting research starts as paper trading only. Subscription features start with transparent fantasy advice before any betting-adjacent claims.

## 10. Rollout, Migration, And Compatibility

This is a living project-level spec and does not migrate data by itself.

Rollout stages:

- **Private MVP:** Use the system for the operator's own Cartola lineup decisions and collect live performance evidence.
- **Trusted tester stage:** Share advice with a small group only after advice artifacts include timestamps, model versions, and limitation disclosures.
- **Betting research stage:** Add football match-event forecasts in paper-trading mode, without wagers or paid betting advice.
- **Subscription beta:** Launch only after a separate subscription spec defines billing, entitlements, privacy, support, refunds, and user-facing claims.
- **Betting-adjacent public advice:** Launch only after separate legal/compliance and responsible-gambling specs are accepted.

Compatibility rules:

- `specs/app.md` is the current product-level spec for AgileForge-facing project management.
- Existing feature-level specs under `docs/superpowers/specs/` remain supporting references unless contradicted by this product-level spec or a later accepted spec.
- Old fixed-budget reports remain historical artifacts and must not be mixed with moving-budget promotion evidence.
- Any future real submission, subscription, or betting product must have a separate accepted spec before implementation.

## 11. Success Metrics

| Metric | Target | Measurement Source |
| --- | --- | --- |
| Live Cartola score lift | Recommended squad beats the configured baseline in aggregate actual captain-aware points over a reviewed live sample | Live outcome review reports |
| Captain contribution | Captain choice adds positive aggregate points versus no-captain or baseline-captain comparator over a reviewed sample | Selected-player and round review artifacts |
| DNP control | DNP count and DNP point loss stay within accepted thresholds set before each live validation period | Live outcome review reports |
| Budget health | Final/min budget and max drawdown remain within accepted thresholds versus current default and baseline | Budget-path reports |
| Recommendation traceability | 100% of live recommendations link capture hash, model profile, budget, selected squad, and post-round review | Recommendation and review metadata |
| Promotion discipline | 100% of default changes cite a frozen decision artifact with passed guardrails | Promotion decision files |
| Betting research calibration | Future event forecasts report Brier score, log loss, calibration curve, and settlement status before any monetization | Paper-trading reports |
| Subscription readiness | Public paid advice launches only after accepted specs cover compliance, privacy, disclosures, billing, and support | Accepted product/compliance specs |

## 12. Open Questions

| Question | Impact | Owner | Status |
| --- | --- | --- | --- |
| Which Cartola competition objective matters most: weekly high score, private league rank, global rank, or season-long consistency? | Determines objective weighting between upside, floor, and budget preservation. | caRtola maintainer/operator | Open |
| What live validation window is enough before declaring the selector meaningfully better than manual selection or baseline? | Determines promotion and confidence thresholds. | caRtola maintainer/operator | Open |
| What DNP, budget drawdown, and captain-risk thresholds are acceptable for the operator's style of play? | Determines optimizer risk policy and promotion gates. | caRtola maintainer/operator | Open |
| Which betting markets are first research targets: match winner, over/under goals, player cards, team cards, or another market? | Determines data acquisition and forecast schema for P3. | caRtola maintainer/operator | Open |
| Which jurisdictions and languages would a future subscription serve first? | Determines legal, payment, disclosure, localization, and responsible-gambling requirements. | caRtola maintainer/operator | Open |
| What verified Cartola save/read-back contract will be accepted for real submission Phase 2? | Blocks authenticated squad submission. | caRtola maintainer/operator | Open |
| What AgileForge command should be used for future spec updates after project creation? | Determines how this living spec remains synchronized with project management state. | caRtola maintainer/operator | Open |

## 13. Revision History

| Date | Version | Change | Author |
| --- | --- | --- | --- |
| 2026-05-16 | 0.1 | Initial AgileForge-facing project-level technical spec. | Codex |
| 2026-05-16 | 0.2 | Reframed project around the owner's goal: winning Cartola squad selection first, future betting prediction and subscription tracks gated by validation and compliance. | Codex |
