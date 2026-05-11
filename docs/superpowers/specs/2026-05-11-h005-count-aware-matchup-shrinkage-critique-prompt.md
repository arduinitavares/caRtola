# H005 External Critique Prompt

Use this prompt with another LLM reviewer before implementing H005.

```text
You are reviewing a frozen hypothesis design for a Cartola fantasy football
prediction/backtesting project. Be skeptical. Do not approve the design unless
the statistical, leakage, implementation, and decision-contract details are
strong enough.

CONTEXT

The project predicts Cartola player points and then builds squads with a MILP
optimizer under real game constraints: budget, official formations, positions,
club limits, and a non-coach captain multiplier. Historical experiments now use
sequential moving-budget semantics, so each strategy's selected players affect
the next round's budget through historical price variation.

The team wants data-driven football knowledge discovery, but several plausible
manual hypotheses have already failed once tested through moving-budget,
captain-aware backtests:

- H001 opponent-overlap optimizer policy: rejected.
- H002 goalkeeper-conflict optimizer policy: rejected.
- H003 clean-sheet defensive-stack optimizer policy: rejected.
- H004 attack-vs-defense feature pack: rejected after improving some seasons
  but regressing 2025 and failing aggregate gates.

Broad AutoML was rejected because the effective sample size is closer to about
170 target rounds / 5 seasons than to the raw player-row count. Instead, the
team built a discovery-only EBM diagnostic runner using InterpretML EBMs on
persisted experiment artifacts. It cannot promote models or change live
defaults. It can only nominate human-review leads.

SOURCE DISCOVERY EVIDENCE

The relevant EBM diagnostic run was:

data/08_reporting/ebm_diagnostics/ebm_diagnostic_started_at=20260511T004620197204Z

Source experiment context:

- model: xgboost_depth2_slow
- feature pack: ppg_xg_matchup
- seasons: 2021,2022,2023,2024,2025
- fixture mode: exploratory
- current year: 2026

The EBM diagnostic completed successfully. It did not produce a better model:

- 2023 source MAE 2.966 vs residual-corrected EBM MAE 3.096
- 2024 source MAE 2.807 vs residual-corrected EBM MAE 2.857
- 2025 source MAE 3.071 vs residual-corrected EBM MAE 3.087

Therefore, EBM is being used only for hypothesis generation, not as a model.

The strongest useful residual lead was:

- target_type: source_residual
- feature: matchup_opponent_allowed_position_count
- candidate_hypothesis_flag: true
- validation seasons with signal: 2023, 2024, 2025
- direction summary: inverted_u
- strongest positive residual evidence around count bins near 16.5-20.5

Important: this count is not itself a football skill signal. It may be a
reliability/data-support signal for an existing opponent-position matchup
feature.

PROPOSED H005

H005 tests one conservative feature transform:

feature_pack = ppg_xg_matchup_h005

Control:

xgboost_depth2_slow + ppg_xg_matchup

Challenger:

xgboost_depth2_slow + ppg_xg_matchup_h005

H005 adds exactly three columns:

- h005_opponent_position_reliability
- h005_opponent_allowed_position_points_shrunk
- h005_opponent_allowed_position_delta_shrunk

Inputs are existing pre-round matchup features already present in
ppg_xg_matchup:

opponent_position_allowed =
  matchup_opponent_allowed_position_points_roll5

opponent_all_allowed =
  matchup_opponent_allowed_points_roll5

opponent_position_count =
  matchup_opponent_allowed_position_count

Frozen formula:

count_nonnegative = max(opponent_position_count, 0)

h005_opponent_position_reliability =
  min(count_nonnegative / 20.0, 1.0)

h005_opponent_allowed_position_points_shrunk =
  h005_opponent_position_reliability * opponent_position_allowed
  + (1.0 - h005_opponent_position_reliability) * opponent_all_allowed

h005_opponent_allowed_position_delta_shrunk =
  h005_opponent_allowed_position_points_shrunk - opponent_all_allowed

For posicao == "tec", all H005 columns are set to 0.0.

H005 reuses existing matchup fallback semantics before applying the transform:

- missing opponent allowed position points fall back to opponent all-position
  allowed points, then position prior, then global prior;
- missing opponent all-position allowed points fall back to global played
  points prior;
- missing count becomes 0;
- nonfinite H005 outputs invalidate the run.

No new data sources, no optimizer rules, no threshold variants, no position
specific variants, no AutoML search, and no direct live-default changes are
allowed in H005 v1.

EXPERIMENT DESIGN

Run one frozen model-feature experiment group:

group = h005-count-aware-matchup-shrinkage

Matrix contains exactly:

- control: xgboost_depth2_slow + ppg_xg_matchup
- challenger: xgboost_depth2_slow + ppg_xg_matchup_h005

Command:

uv run --frozen python scripts/run_model_experiments.py \
  --group h005-count-aware-matchup-shrinkage \
  --seasons 2021,2022,2023,2024,2025 \
  --start-round 5 \
  --budget 100 \
  --current-year 2026 \
  --jobs 12 \
  --profile-runtime

Required comparability:

- same seasons, start round, moving-budget policy, initial budget, scoring
  contract, fixture mode, FootyStats mode, and matchup context mode;
- control and challenger tested side by side in the same experiment matrix;
- candidate-pool signatures match for every season and target round;
- signatures exclude model score columns and H005-added feature columns;
- raw Cartola source identity, FootyStats source identity, fixture source
  paths, and fixture hashes are recorded;
- exploratory fixture evidence is research evidence only.

ACCEPTANCE GATES

H005 becomes only a candidate research profile if all gates pass:

- comparability_report.status == "ok";
- exploratory fixture identity is verified; unverified fixture identity makes
  the result diagnostic_only;
- candidate-pool signatures match control for every season and target round;
- no additional non-optimal, infeasible, skipped, or budget-constrained rounds
  versus control;
- aggregate total actual point delta is at least +85;
- at least 4/5 seasons improve;
- worst season delta is no worse than -20;
- 2025 delta is no worse than -10;
- final budget delta is nonnegative in aggregate;
- no season final-budget delta is worse than -2;
- top-50 Spearman delta is nonnegative in at least 4/5 seasons;
- selected-player calibration slope remains between 0.8 and 1.2 for every
  season with at least 30 selected-player rows and non-constant predictions;
- top-two-season positive lift concentration is less than 70%.

Top-two-season positive lift concentration:

positive_season_delta = max(season_delta, 0)
concentration =
  sum(two_largest_positive_season_delta) / sum(all_positive_season_delta)

If total positive season delta is 0, H005 fails the concentration gate.

REQUESTED REVIEW

Critically assess this H005 design. Do not accept it blindly. Identify whether
it is ready for implementation, needs revision, or should be abandoned.

Please evaluate:

1. High-severity correctness issues.
2. Statistical and leakage risks.
3. Whether the EBM residual lead actually supports this shrinkage hypothesis.
4. Whether the formula is defensible or too arbitrary.
5. Whether keeping the original raw matchup feature plus the shrunk feature
   creates redundancy or lets XGBoost ignore the intended correction.
6. Whether the saturation count 20.0 is justified, under-specified, or an
   overfit threshold.
7. Whether a better alternative exists, such as:
   - model interaction with count instead of explicit shrinkage;
   - replacing the raw matchup feature with the shrunk feature;
   - using reliability-weighted residualization;
   - using monotonic constraints;
   - running position-specific diagnostics first;
   - stopping because the source model likely already learns the count effect.
8. Whether the acceptance gates are too strict, too loose, or misaligned with
   the hypothesis.
9. Hidden assumptions and missing definitions.
10. Implementation risks likely to produce misleading results.

Return the assessment in this structure:

1. Critical Issues (High Severity Only)
2. Hidden Assumptions
3. Gaps And Missing Definitions
4. Statistical / Leakage Risks
5. Alternative Designs
6. Regression And Implementation Risks
7. Actionable Fixes, Prioritized
8. Final Verdict

For the final verdict, include:

- Clarity score from 1-10
- Correctness confidence from 1-10
- Production readiness from 1-10
- Risk level: Low, Medium, or High
- One of: approve, revise before implementation, abandon
```
