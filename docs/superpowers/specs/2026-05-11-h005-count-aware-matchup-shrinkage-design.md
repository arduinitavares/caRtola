# H005 Count-Aware Matchup Shrinkage Design

## Summary

H005 tests whether the current matchup model over-trusts low-sample
opponent-position matchup features.

The EBM diagnostic did not produce a better replacement model. It did produce
one stable human-review lead on source-model residuals:

```text
target_type=source_residual
feature=matchup_opponent_allowed_position_count
candidate_hypothesis_flag=true
validation_seasons_with_signal=2023,2024,2025
direction_summary=inverted_u
```

H005 turns that lead into one frozen model-feature hypothesis:

```text
feature_pack = ppg_xg_matchup_h005
```

The hypothesis is simple: when the sample count behind
`matchup_opponent_allowed_position_points_roll5` is small, the feature should be
shrunk toward a broader opponent-all-position prior. When the sample count is
large enough, the model can trust the position-specific value more.

H005 is not an optimizer policy, not an AutoML result, and not a live-default
change. It must be tested only through the existing sequential moving-budget
experiment workflow against the current control:

```text
xgboost_depth2_slow + ppg_xg_matchup
```

## Motivation

The user wanted a data-driven way to discover football knowledge instead of
hand-writing fragile squad rules. H001-H003 tested direct optimizer policies and
were rejected. H004 tested attack-vs-defense feature interactions and was
rejected after moving-budget validation.

The EBM diagnostic was created to generate narrower hypotheses from existing
artifact-backed predictions. The completed diagnostic run:

```text
data/08_reporting/ebm_diagnostics/ebm_diagnostic_started_at=20260511T004620197204Z
```

finished with:

```text
diagnostic_status=diagnostic_complete
candidate_count=3
```

Only one candidate cleared the residual-target candidate flag:

```text
matchup_opponent_allowed_position_count
```

This count is not a direct football-quality signal. It is a reliability signal:
it tells us how much recent historical evidence supports the
opponent-position-specific allowed-points estimate.

## Evidence Boundary

The EBM diagnostic is discovery-only evidence. It can justify writing H005, but
it cannot promote a model or feature pack.

Key observed diagnostic facts:

- EBM raw/residual predictions did not beat the source XGBoost model on MAE.
- The source model remained better or roughly equal on actual-points MAE:
  - 2023 source MAE `2.966`, residual-corrected MAE `3.096`;
  - 2024 source MAE `2.807`, residual-corrected MAE `2.857`;
  - 2025 source MAE `3.071`, residual-corrected MAE `3.087`.
- The useful EBM output is therefore the residual shape, not the EBM model.
- The residual shape repeated across validation seasons `2023`, `2024`, and
  `2025`.
- The strongest positive residual bins were around counts near `16.5-20.5`.

H005 uses this evidence only to freeze one conservative transform. It does not
copy the exact EBM curve, because that would overfit a noisy diagnostic.

## Source Control And Comparability

H005 must be tested side-by-side with its control in the same experiment matrix.
Do not compare an H005 challenger against old control artifacts.

Required shared conditions:

- seasons: `2021,2022,2023,2024,2025`;
- start round: `5`;
- budget policy: `moving`;
- initial budget: `100`;
- scoring contract: `cartola_standard_2026_v1`;
- fixture mode: `exploratory` for this historical research generation;
- FootyStats mode: `ppg_xg`;
- matchup context mode: `cartola_matchup_v1`;
- model control: `xgboost_depth2_slow + ppg_xg_matchup`;
- challenger: `xgboost_depth2_slow + ppg_xg_matchup_h005`;
- candidate-pool signatures must match between control and challenger for every
  season and target round;
- candidate-pool signatures must exclude model score columns and H005-added
  feature columns;
- raw Cartola source identity, FootyStats source identity, fixture source paths,
  and fixture hashes must be recorded.

Exploratory fixture evidence is research evidence. It is not live-default
promotion evidence by itself.

## Frozen Hypothesis

H005 v1 adds exactly three feature columns:

```text
h005_opponent_position_reliability
h005_opponent_allowed_position_points_shrunk
h005_opponent_allowed_position_delta_shrunk
```

No threshold variants, alternative saturation counts, interaction columns,
position-specific variants, or optimizer policies are part of H005 v1.

### Inputs

H005 uses only existing pre-round matchup features already produced by
`ppg_xg_matchup`:

```text
opponent_position_allowed =
  matchup_opponent_allowed_position_points_roll5

opponent_all_allowed =
  matchup_opponent_allowed_points_roll5

opponent_position_count =
  matchup_opponent_allowed_position_count
```

These fields are computed from played history with `rodada < target_round` and
the target-round fixture context. H005 must not read target-round outcomes.

### Formula

First normalize the count:

```text
count_nonnegative = max(opponent_position_count, 0)
```

Then compute one frozen reliability weight:

```text
h005_opponent_position_reliability =
  min(count_nonnegative / 20.0, 1.0)
```

The saturation count `20.0` is frozen for H005 v1 because the EBM residual lead
showed its strongest positive evidence around counts near `16.5-20.5`. This is
not tunable inside H005.

Then shrink the position-specific allowed-points estimate toward the broader
opponent all-position estimate:

```text
h005_opponent_allowed_position_points_shrunk =
  h005_opponent_position_reliability
    * opponent_position_allowed
  + (1.0 - h005_opponent_position_reliability)
    * opponent_all_allowed
```

Finally expose the shrunk position-specific delta:

```text
h005_opponent_allowed_position_delta_shrunk =
  h005_opponent_allowed_position_points_shrunk
  - opponent_all_allowed
```

### `tec` Handling

For `posicao == "tec"`, set all H005 columns to `0.0`.

Coach scoring is structurally different from player scoring. H005 is a
player-position matchup reliability hypothesis.

### Missingness And Fallbacks

H005 must not introduce NaN or infinite values.

It reuses the existing matchup v1 fallback semantics before applying H005:

- missing opponent allowed position points fall back to opponent all-position
  allowed points, then position prior, then global prior;
- missing opponent all-position allowed points fall back to the global played
  points prior;
- missing count becomes `0`;
- nonfinite H005 outputs invalidate the run.

H005 does not add new data sources.

## Experiment Design

Add one research group:

```text
group = h005-count-aware-matchup-shrinkage
```

The matrix contains exactly:

- control: `xgboost_depth2_slow + ppg_xg_matchup`;
- challenger: `xgboost_depth2_slow + ppg_xg_matchup_h005`.

Run:

```bash
uv run --frozen python scripts/run_model_experiments.py \
  --group h005-count-aware-matchup-shrinkage \
  --seasons 2021,2022,2023,2024,2025 \
  --start-round 5 \
  --budget 100 \
  --current-year 2026 \
  --jobs 12 \
  --profile-runtime
```

The normal experiment runner should write the standard model-feature artifacts,
ranked summaries, comparability reports, prediction metrics, selected-player
artifacts, and charts. H005 should not require a custom backtest runner.

After the experiment completes, write a deterministic decision artifact:

```text
h005_feature_decision.json
```

The decision artifact may be produced by a small H005 decision script or by
extending an existing hypothesis decision helper, but it must read persisted
experiment outputs only.

## Acceptance Gates

H005 becomes a candidate research profile only if all gates pass:

- `comparability_report.status == "ok"`;
- exploratory fixture identity is verified; unverified fixture identity makes
  the result `diagnostic_only`;
- candidate-pool signatures match control for every season and target round;
- no additional non-optimal, infeasible, skipped, or budget-constrained rounds
  versus control;
- aggregate total actual point delta is at least `+85`;
- at least `4 / 5` seasons improve;
- worst season delta is no worse than `-20`;
- 2025 delta is no worse than `-10`;
- final budget delta is nonnegative in aggregate;
- no season final-budget delta is worse than `-2`;
- top-50 Spearman delta is nonnegative in at least `4 / 5` seasons;
- selected-player calibration slope remains between `0.8` and `1.2` for every
  season with at least `30` selected-player rows and non-constant predictions;
- top-two-season positive lift concentration is less than `70%`.

Top-two-season positive lift concentration is:

```text
positive_season_delta = max(season_delta, 0)
concentration =
  sum(two_largest_positive_season_delta) / sum(all_positive_season_delta)
```

If total positive season delta is `0`, H005 fails the concentration gate.

If any gate fails, H005 remains rejected or diagnostic-only.

## Output Artifacts

The H005 experiment writes under:

```text
data/08_reporting/experiments/model_feature/group=h005-count-aware-matchup-shrinkage__started_at=.../
```

Required standard artifacts:

- parent `experiment_metadata.json`;
- parent `ranked_summary.csv`;
- parent `comparability_report.json`;
- child `run_metadata.json`;
- child `player_predictions.csv`;
- child `round_results.csv`;
- child `selected_players.csv`;
- HTML comparison charts.

Required H005 decision artifact:

```text
h005_feature_decision.json
```

Required fields:

- `hypothesis_id`: `H005`;
- `decision_status`: `candidate_research_profile`, `rejected`,
  `diagnostic_only`, or `invalid`;
- `control_strategy`;
- `challenger_strategy`;
- aggregate actual-point delta;
- per-season actual-point deltas;
- final-budget deltas;
- optimizer/comparability status;
- candidate-pool signature status;
- fixture identity status;
- ranking/calibration deltas;
- concentration calculation;
- failed gates;
- source EBM diagnostic path.

## Tests

### Feature Tests

- Feature registry includes `ppg_xg_matchup_h005`.
- Base `ppg_xg_matchup` columns remain unchanged.
- H005 columns are added only for `ppg_xg_matchup_h005`.
- H005 formulas match the frozen definitions exactly.
- Counts below `0` are treated as `0`.
- Counts at `0`, `10`, `20`, and `30` produce reliability weights
  `0.0`, `0.5`, `1.0`, and `1.0`.
- `tec` rows receive zero H005 values.
- H005 outputs are finite in early rounds and low-sample positions.
- H005 does not change candidate identity or optimizer eligibility columns.

### Experiment/Decision Tests

- The H005 experiment group contains exactly the control and challenger rows.
- The decision script rejects missing control or missing challenger artifacts.
- The decision script rejects mismatched candidate-pool signatures.
- The decision script labels unverified fixture identity as `diagnostic_only`,
  not candidate evidence.
- The decision script applies all acceptance gates deterministically.
- The decision script rejects zero-positive-lift concentration.

## Risks

- The count feature may be a data-availability proxy rather than a football
  signal.
- The current XGBoost may already learn enough from the raw count column, making
  shrinkage redundant.
- The EBM residual shape is an exploratory lead from five seasons, not proof.
- Shrinkage may improve row metrics but still fail squad optimization.
- The saturation count `20.0` is plausible but not tuned; H005 intentionally
  avoids tuning it to reduce overfitting risk.
- Exploratory fixture evidence cannot become a live default without a separate
  strict/live validation path.

## Non-Goals

- No optimizer constraints or bonuses.
- No new model family.
- No AutoML search.
- No alternate count thresholds.
- No generated lag-feature factory.
- No direct live default changes.
- No promotion without the moving-budget acceptance gates.

## Final Decision Rule

1. If external review finds a blocker, revise this spec before implementation.
2. If implementation produces invalid comparability, stop and fix artifacts.
3. If the frozen experiment fails any acceptance gate, record H005 as rejected.
4. If all gates pass, H005 becomes a candidate research profile only.
5. Live defaults remain unchanged until a separate promotion protocol is
   explicitly approved.
