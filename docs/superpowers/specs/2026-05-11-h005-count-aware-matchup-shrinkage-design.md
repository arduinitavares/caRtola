# H005 Count-Aware Matchup Reliability Design

## Summary

H005 tests whether the model benefits from a position-normalized reliability
signal for opponent-position matchup features.

This is a revision of the original H005 shrinkage design. External review found
that the EBM diagnostic supports the claim that count matters, but does not
prove the original manual shrinkage formula.

Accepted review corrections:

- The EBM evidence identifies residual structure on
  `matchup_opponent_allowed_position_count`; it does not prove that
  `matchup_opponent_allowed_position_points_roll5` should be shrunk toward
  `matchup_opponent_allowed_points_roll5`.
- A global saturation count of `20.0` is position-biased because the count is a
  player-observation count, not a round count.
- The original design was ambiguous about whether raw matchup features remain
  in the challenger pack.
- H005 needs a source-anchored mechanism audit before the feature experiment is
  treated as candidate evidence.
- The mechanism audit cannot be purely artifact-only because the source
  `player_predictions.csv` starts at target round `5`, while expected-count
  features need rounds `1-4` history for the first target round.
- The denominator must use the actual available historical opponent-match
  opportunity window, not a fixed five-match assumption and not only
  opponent-position rows with nonzero observations.

Rejected or qualified review corrections:

- `matchup_opponent_allowed_position_points_roll5` and
  `matchup_opponent_allowed_points_roll5` are not different units. The code
  computes both as per-player rolling means. However, the all-position mean is
  still a poor shrinkage target because positions have different baselines.

H005 v1 no longer performs manual points shrinkage. It adds a normalized
reliability signal and lets the existing XGBoost model decide whether and how to
use it.

```text
feature_pack = ppg_xg_matchup_h005
```

H005 is not an optimizer policy, not an AutoML result, and not a live-default
change. It must be tested only through the existing sequential moving-budget
experiment workflow against the current control:

```text
xgboost_depth2_slow + ppg_xg_matchup
```

## Source Evidence

The completed EBM diagnostic run:

```text
data/08_reporting/ebm_diagnostics/ebm_diagnostic_started_at=20260511T004620197204Z
```

finished with:

```text
diagnostic_status=diagnostic_complete
candidate_count=3
```

The only residual-target human-review candidate was:

```text
target_type=source_residual
feature=matchup_opponent_allowed_position_count
candidate_hypothesis_flag=true
validation_seasons_with_signal=2023,2024,2025
direction_summary=inverted_u
```

EBM was not a better replacement model:

- 2023 source MAE `2.966`, residual-corrected EBM MAE `3.096`;
- 2024 source MAE `2.807`, residual-corrected EBM MAE `2.857`;
- 2025 source MAE `3.071`, residual-corrected EBM MAE `3.087`.

Therefore H005 uses the EBM output only to freeze one hypothesis: matchup
sample support may matter in a way the current model does not use well enough.

The EBM shape was inverted-U. H005 v1 deliberately does not copy that curve.
The curve is an exploratory diagnostic from the same seasons that will be used
for the first frozen research experiment, so copying it would overfit. H005
uses a conservative normalized count representation instead.

## Count Semantics

`matchup_opponent_allowed_position_count` is not a number of historical rounds.
It is the sum of player observations for the opponent and position over the
last five opponent matches used by the rolling matchup feature.

That means the same raw count has different meaning by position. A count of
`5` can be close to full recent support for `gol`, but low support for `mei`.
H005 must not use a single global denominator such as `20.0`.

## Frozen Hypothesis

H005 v1 adds exactly three feature columns:

```text
h005_opponent_position_available_match_count_roll5
h005_opponent_position_expected_count_roll5
h005_opponent_position_count_ratio
```

It does not add shrunk points, delta-shrunk points, threshold variants,
position-specific hand-tuned constants, clipped reliability-gap features,
optimizer policies, or new data sources.

The `ppg_xg_matchup_h005` pack includes all columns from `ppg_xg_matchup` plus
the three H005 columns. It is feature augmentation, not forced replacement. A
passing experiment would prove only that the H005 feature pack helped, not that
manual shrinkage was validated.

### Inputs

H005 uses only existing cutoff-safe training history and existing matchup
features:

```text
opponent_position_count =
  matchup_opponent_allowed_position_count

played_history =
  rows with rodada < target_round and entrou_em_campo == true
```

H005 must not read target-round outcomes.

### Expected Count Formula

For each target round, compute the cutoff-safe expected observation count by
position from played history:

```text
team_round_position_count =
  count_distinct(id_atleta)
  grouped by rodada, id_clube, posicao

position_players_per_team_round_prior =
  mean(team_round_position_count)
  grouped by posicao

available_opponent_match_count_roll5 =
  count_distinct(rodada)
  from the same recent opponent-match opportunity window used to compute
  opponent allowed roll5 features, including opponent matches where the
  candidate position has zero observations

h005_opponent_position_available_match_count_roll5 =
  available_opponent_match_count_roll5

h005_opponent_position_expected_count_roll5 =
  max(
    h005_opponent_position_available_match_count_roll5
      * position_players_per_team_round_prior,
    1.0
  )
```

This makes the denominator position-normalized and season/round cutoff-safe.
It adapts to historical formation mix without introducing hand-tuned constants
for `gol`, `lat`, `zag`, `mei`, or `ata`.

The available-match multiplier matters in early rounds and incomplete fixture
windows. For target round `5`, at most four prior opponent matches can exist,
so the expected count must not assume a full five-match window.

Available matches are opponent match opportunities, not only observed
opponent-position rows. If an opponent has five prior matches and a candidate
position appears in only two of those matches, the available-match count is
`5`, not `2`. The missing three position observations are part of the
reliability signal.

The expected count approximates the actual observation count. The actual
`matchup_opponent_allowed_position_count` sums per-round player observations;
the expected count multiplies available opponent matches by the league-wide
position player-count prior. These are close but not identical because of
substitutions, rotation, and formation variation.

If a position has no played-history prior for a target round, use the global
non-coach `position_players_per_team_round_prior` mean. If that is unavailable,
use `1.0`.

### Zero-Count Densification

The position player-count prior must include zero-count team-position
combinations. A naive groupby over observed rows is not enough, because it drops
positions that a team did not field in a round and artificially inflates the
mean.

For each historical played round, build the Cartesian product of:

```text
active non-coach clubs in that round
standard non-coach positions: gol, lat, zag, mei, ata
```

Then left-join observed distinct player counts and fill missing combinations
with `0` before computing `position_players_per_team_round_prior`.

A club is active in a historical round if at least one non-coach player with
`entrou_em_campo == true` appears for that club in that round. Clubs that did
not play because of postponed or missing fixtures must not be injected into the
Cartesian product for that round.

### Granularity And Season Scope

Expected count is computed once per `(season, target_round, posicao)` from that
season's played history only, then merged onto the candidate frame by `posicao`.
It must never pool future seasons or other seasons into a target season's
cutoff history.

### Reliability Formula

First normalize the raw count:

```text
count_nonnegative = max(opponent_position_count, 0)
```

Then compute:

```text
h005_opponent_position_count_ratio =
  count_nonnegative
  / h005_opponent_position_expected_count_roll5
```

`h005_opponent_position_count_ratio` is intentionally not clipped above `1.0`
so XGBoost can see unusually high support. H005 v1 deliberately does not add a
clipped reliability-gap feature because it is a deterministic transform of the
ratio and would add dimensionality without new information for a tree model.

### `tec` Handling

For `posicao == "tec"`, set all H005 columns to `0.0`.

Coach scoring is structurally different from player scoring. H005 is a
player-position matchup reliability hypothesis.

### Missingness And Finiteness

H005 must not introduce NaN or infinite values.

- Missing `matchup_opponent_allowed_position_count` becomes `0`.
- Missing expected count uses the fallback order defined above.
- Nonfinite H005 outputs invalidate the run.

## Architecture: Expected Count Injection Point

H005 requires `played_history`, fixtures, and `target_round`. Existing H004
augmentation only receives the already-built candidate frame, so H005 cannot use
the same frame-only helper shape.

Implementation must use one of these explicit paths:

1. Preferred: add an H005-specific augmentation branch in
   `build_prediction_frame` after matchup and FootyStats features are merged,
   passing `frame`, `played_history`, `fixtures`, and `target_round` into the
   H005 helper.
2. Acceptable: compute the H005 available-match and expected-count helper
   columns inside the matchup context builder where `played_history`, fixtures,
   and `target_round` are already available, then expose them only when
   `feature_augmentation_mode == "h005_matchup_reliability_v1"`.

The implementation must not approximate expected count from the target-round
candidate frame alone. That frame does not contain the prior rounds needed for
round `5` and would make the audit and backtest inconsistent.

The available opponent-match count must be derived from the same opponent
fixture window as the existing opponent allowed roll5 features. The
implementation may add an optional `round_count_column` or companion helper
around `_roll5_last`, but it must count opponent match opportunities before the
position join so zero-position rounds remain in the denominator.

## Phase 0 Mechanism Audit

Before the feature experiment is interpreted as candidate evidence, write a
source-anchored H005 mechanism audit.

The audit reads persisted source experiment predictions and residuals, but it
may also read validated raw season data and source fixture files to recompute
H005 features. It is not purely artifact-only because source
`player_predictions.csv` starts at target round `5` and does not contain the
rounds `1-4` history needed for the first target round.

To prevent source drift, the audit must recompute the existing
`matchup_opponent_allowed_position_count` for every source prediction row from
the selected raw season data and fixture context. If recomputed counts do not
match the persisted source artifact exactly for all valid rows, the audit is
`invalid`.

The audit manifest must record:

- raw season data paths used;
- computed raw season data SHA-256 hashes;
- source fixture paths and SHA-256 hashes;
- recomputed-vs-persisted matchup count match status;
- source prediction artifact paths;
- source prediction artifact SHA-256 hashes.

Required output:

```text
h005_mechanism_audit.csv
h005_mechanism_audit_decision.json
```

The audit must compute, by season, position, and reliability-ratio bin:

- row count;
- round count;
- source residual mean;
- source overprediction rate;
- mean `matchup_opponent_allowed_position_count`;
- mean `h005_opponent_position_available_match_count_roll5`;
- mean `h005_opponent_position_expected_count_roll5`;
- mean `h005_opponent_position_count_ratio`;
- mean `matchup_opponent_allowed_position_points_roll5`;
- mean `matchup_opponent_allowed_points_roll5`;
- mean position-vs-all allowed-points delta.

The audit must also write the same residual support summary by raw count bins so
we can compare the normalized ratio against the original raw count evidence.
If normalized ratio bins are weaker or less position-balanced than raw count
bins, the audit decision cannot be `supports_reliability_hypothesis`.

Reliability-ratio bins:

```text
0
(0, 0.5]
(0.5, 0.8]
(0.8, 1.0]
(1.0, 1.5]
> 1.5
```

Raw-count comparison bins:

```text
0
(0, 5]
(5, 10]
(10, 20]
(20, 30]
> 30
```

Audit decision statuses:

- `supports_reliability_hypothesis`;
- `mixed_or_weak`;
- `invalid`.

The audit supports the hypothesis only if:

- all required source artifacts validate;
- at least four non-coach positions have at least `500` rows total;
- low-reliability bins are exactly `0`, `(0, 0.5]`, and `(0.5, 0.8]`;
- normal-reliability bins are exactly `(0.8, 1.0]` and `(1.0, 1.5]`;
- at least `3 / 5` seasons have at least `500` low-reliability rows and at
  least `20` low-reliability target rounds;
- at least `3 / 5` seasons have at least `500` normal-reliability rows and at
  least `20` normal-reliability target rounds;
- `low_minus_normal_source_residual_mean` has the same sign in at least `3`
  supported seasons;
- median absolute `low_minus_normal_source_residual_mean` across supported
  seasons is at least `0.10` points;
- normalized-ratio low-reliability support includes at least four non-coach
  positions with at least `100` rows each;
- raw-count low-support bins must be evaluated with the same four-position,
  `100`-row support criterion before comparing position balance;
- no single season contributes more than `40%` of supported low-reliability
  rows.

If the audit is `mixed_or_weak`, the feature experiment may still be run for
diagnostics, but it cannot produce `candidate_research_profile`.

## Experiment Design

Add one research group:

```text
group = h005-count-aware-matchup-shrinkage
```

The group name stays stable even though the revised v1 is reliability-only. The
decision artifacts must record:

```text
h005_design_revision = reliability_v1
manual_points_shrinkage = false
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

The decision artifact must read persisted experiment outputs only.

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

## Decision Statuses

`h005_feature_decision.json` must use one of:

- `candidate_research_profile`;
- `weak_positive_research_lead`;
- `inconclusive`;
- `rejected`;
- `diagnostic_only`;
- `invalid`.

`candidate_research_profile` requires all candidate gates below.

`weak_positive_research_lead` is allowed when the feature shows a stable,
non-damaging positive result but misses the full candidate threshold. It cannot
change live defaults and cannot be promoted directly. It can only justify a
future frozen H006/H007 design.

`inconclusive` means the result is inside the noise band and should not drive a
new implementation immediately.

## Acceptance Gates

H005 becomes `candidate_research_profile` only if all gates pass:

- Phase 0 audit status is `supports_reliability_hypothesis`;
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

`weak_positive_research_lead` requires:

- Phase 0 audit status is `supports_reliability_hypothesis`;
- all comparability, fixture, signature, optimizer, and budget-integrity gates
  pass;
- aggregate total actual point delta is at least `+40`;
- at least `3 / 5` seasons improve;
- worst season delta is no worse than `-20`;
- 2025 delta is no worse than `-10`;
- top-two-season positive lift concentration is less than `75%`.

`inconclusive` applies when:

- comparability is valid;
- aggregate total actual point delta is between `-20` and `+40`;
- no season delta is worse than `-20`;
- 2025 delta is no worse than `-10`;
- no additional non-optimal, infeasible, skipped, or budget-constrained rounds
  versus control;
- no season final-budget delta is worse than `-2`.

All other valid comparable outcomes are `rejected`.

Top-two-season positive lift concentration is:

```text
positive_season_delta = max(season_delta, 0)
concentration =
  sum(two_largest_positive_season_delta) / sum(all_positive_season_delta)
```

If total positive season delta is `0`, H005 fails the concentration gate.

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

Required H005 artifacts:

```text
h005_mechanism_audit.csv
h005_mechanism_audit_decision.json
h005_feature_decision.json
```

Required decision fields:

- `hypothesis_id`: `H005`;
- `h005_design_revision`: `reliability_v1`;
- `manual_points_shrinkage`: `false`;
- `decision_status`;
- `mechanism_audit_status`;
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
- Expected count is computed from `rodada < target_round` only.
- Expected count is position-normalized.
- Available opponent-match count uses the same recent opponent-match
  opportunity window as the opponent allowed roll5 features, including
  position-zero rounds.
- Early rounds use the actual available prior opponent-match count, not a fixed
  five-match denominator.
- Position priors include zero-count team-round-position combinations before
  taking the mean.
- Counts below `0` are treated as `0`.
- `h005_opponent_position_count_ratio` is not clipped above `1.0`.
- `tec` rows receive zero H005 values.
- H005 outputs are finite in early rounds and low-sample positions.
- H005 does not change candidate identity or optimizer eligibility columns.
- The H005 augmentation path has access to `played_history`, fixtures, and
  `target_round`; it does not compute expected counts from target-round
  candidate rows alone.

### Audit/Decision Tests

- The mechanism audit rejects missing source artifacts.
- The mechanism audit rejects recomputed existing matchup-count mismatches.
- The mechanism audit writes all required bins even when a bin has zero rows.
- The mechanism audit computes residuals from persisted source predictions.
- The mechanism audit uses validated raw season history for rounds before the
  first source prediction round.
- The mechanism audit records raw data hashes and source prediction hashes.
- The mechanism audit compares normalized-ratio bins against raw count bins.
- The H005 experiment group contains exactly the control and challenger rows.
- The decision script rejects missing control or missing challenger artifacts.
- The decision script rejects mismatched candidate-pool signatures.
- The decision script labels unverified fixture identity as `diagnostic_only`,
  not candidate evidence.
- The decision script applies candidate, weak-positive, inconclusive, rejected,
  diagnostic-only, and invalid statuses deterministically.
- The decision script rejects zero-positive-lift concentration.

## Risks

- The count feature may be a data-availability proxy rather than a football
  signal.
- The current XGBoost may already learn enough from the raw count column and
  one-hot `posicao`, making the normalized reliability features redundant.
- The EBM residual shape is an exploratory lead from the same five seasons used
  for the first H005 experiment, not independent proof.
- The mechanism audit uses validated raw season data plus source artifacts, so
  source-drift checks are mandatory; without a recomputed-count match, the audit
  is invalid.
- Normalized reliability may improve row metrics but still fail squad
  optimization.
- A weak positive moving-budget result can still come from budget-path luck.
- Exploratory fixture evidence cannot become a live default without a separate
  strict/live validation path.

## Non-Goals

- No manual points shrinkage in H005 v1.
- No optimizer constraints or bonuses.
- No new model family.
- No AutoML search.
- No hand-tuned position denominators.
- No generated lag-feature factory.
- No direct live default changes.
- No promotion without the moving-budget acceptance gates.

## Final Decision Rule

1. If external review finds a blocker, revise this spec before implementation.
2. If Phase 0 mechanism audit is invalid, stop and fix artifacts.
3. If Phase 0 mechanism audit is mixed or weak, any later experiment is
   diagnostic-only.
4. If implementation produces invalid comparability, stop and fix artifacts.
5. If the frozen experiment fails acceptance gates, record H005 as rejected,
   inconclusive, or weak positive according to the deterministic status rules.
6. If all candidate gates pass, H005 becomes a candidate research profile only.
7. Live defaults remain unchanged until a separate promotion protocol is
   explicitly approved.
