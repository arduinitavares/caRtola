# Live Recommendation Artifact Contract

## Scope

This contract defines the canonical artifact set for one live pre-lock recommendation run. It is anchored to the current `scripts/run_live_round.py` and `cartola.backtesting.live_workflow` execution path.

## Run Directory

Live recommendation output MUST be written under:

```text
data/08_reporting/recommendations/{season}/round-{target_round}/live/runs/run_started_at={YYYYMMDDTHHMMSSffffffZ}/
```

The `run_started_at` token MUST be generated from the workflow start timestamp in UTC. The token MUST be a single path segment and MUST NOT contain `/`, `\`, `.`, `..`, or an empty value.

If the target run directory already exists, the workflow MUST fail before writing recommendation artifacts. A successful run MUST NOT overwrite an existing run directory.

The live workflow MUST reserve the target run directory before recommendation output begins by creating the final run directory with parent creation enabled and existing-directory reuse disabled. A directory collision MUST raise a clear archive-collision error before recommendation generation starts.

## Required Files

A successful live run MUST write:

- `recommended_squad.csv`
- `candidate_predictions.csv`
- `recommendation_summary.json`
- `run_metadata.json`
- `live_workflow_metadata.json`
- `risk_audit.json`

The live workflow metadata MUST link the recommendation output path and the capture metadata path.

## Risk Audit Artifact

The `risk_audit.json` artifact is advisory evidence for manual operator review. It MUST NOT reject, alter, or rescore the recommended squad.

The artifact MUST use schema version `cartola.risk_audit.v1` and include:

- `generated_timestamp`: UTC timestamp for the audit.
- `advisory_only`: `true`.
- `budget`: explicit operator-provided budget.
- `budget_used`: selected squad budget usage.
- `budget_utilization_pct`: `budget_used / budget * 100`.
- `captain_risk_policy`: active captain policy label.
- `captain_risk_score`: captain prior-performance volatility scalar.
- `dnp_risk`: one row per selected player with player id, name, position, pre-lock status, risk flag, and reason.
- `overall_risk_level`: `low`, `medium`, or `high`.
- `warnings`: advisory warning messages.

If any selected player has a pre-lock status other than an accepted probable status, `overall_risk_level` MUST be `high` and `warnings` MUST identify the affected player names.

## Promotion Gate Evidence Contract

Live-default promotion decisions MUST cite a frozen decision artifact before
changing the current live default. The promotion evidence contract is anchored
to accepted authority `REQ.default-promotion-gate` and
`DATA.promotion-decision`.

The cited frozen decision artifact MUST include these mandatory fields:

| Field | Expected Source Artifact |
| --- | --- |
| `decision_artifact_id` | Frozen promotion decision artifact metadata |
| `decision_artifact_sha256` | Frozen promotion decision artifact bytes |
| `generated_at_utc` | Frozen promotion decision artifact metadata |
| `candidate_model_id` | Experiment manifest or promotion decision input |
| `candidate_feature_pack_or_mode` | Experiment manifest, run metadata, or promotion decision input |
| `control_model_id` | Promotion decision input |
| `control_feature_pack_or_mode` | Promotion decision input |
| `comparison_seasons` | Ranked/per-season experiment summaries or promotion decision input |
| `budget_policy` | Experiment run metadata and ranked/per-season summaries |
| `points_delta` | Ranked/per-season experiment summaries and promotion decision calculation |
| `budget_risk_checks` | Budget path summary, drawdown checks, and budget-constrained round checks |
| `dnp_or_availability_checks` | Selected-player status evidence when applicable |
| `calibration_checks` | Calibration report or explicit `not_applicable` decision field |
| `comparability_status` | Comparability report or promotion decision validation result |
| `final_decision` | Frozen promotion decision artifact |
| `final_decision_reason` | Frozen promotion decision artifact |
| `authority_refs` | Accepted authority IDs cited by the decision artifact |
| `source_artifact_refs` | Paths and SHA-256 hashes for every source artifact used by the decision |

The decision artifact MUST state whether optional checks are applicable. When a
check is not applicable, the artifact MUST record `not_applicable` with a short
reason instead of omitting the field.

The decision artifact MUST reject promotion evidence when comparability is not
verified. Exploratory fixture-only evidence, oracle hindsight output, and
one-off experiment wins MUST remain research-only and MUST NOT change the live
default.

## Recommended Squad CSV

`recommended_squad.csv` MUST contain exactly the selected squad rows for the live target round. For the current Cartola scoring contract this means 12 selected rows, including one captain row.

The live CSV MUST include these columns:

- `rodada`
- `id_atleta`
- `apelido`
- `id_clube`
- `nome_clube`
- `posicao`
- `status`
- `preco_pre_rodada`
- `baseline_score`
- `price_score`
- `{model_id}_score`
- `predicted_points`
- `is_captain`
- `captain_policy_ev`
- `captain_policy_safe`
- `captain_policy_upside`

The live CSV MUST NOT include replay-only finalized columns such as `pontuacao`, `entrou_em_campo`, or scout result columns.

Exactly one row MUST have `is_captain=true`. The selected squad summary MUST report `selected_count=12`.

## Candidate Predictions CSV

`candidate_predictions.csv` MUST include the live candidate pool for the target round and MUST include the base candidate columns, `{model_id}_score`, and active feature-context columns. It MUST NOT include `is_captain`, captain policy flags, `pontuacao`, `entrou_em_campo`, or replay-only scout result columns in live mode.

## Recommendation Summary JSON

`recommendation_summary.json` MUST include:

- `season`
- `target_round`
- `mode`
- `strategy`
- `formation`
- `budget`
- `budget_used`
- `optimizer_status`
- `selected_count`
- `predicted_points`
- `predicted_points_base`
- `captain_bonus_predicted`
- `predicted_points_with_captain`
- `captain_id`
- `captain_name`
- `captain_position`
- `captain_club`
- `captain_policy_diagnostics`
- `output_directory`
- scoring contract fields from `contract_fields()`

In live mode, actual-score and oracle fields MUST be null unless the workflow is explicitly replaying completed data in a non-live mode.

## Run Metadata JSON

`run_metadata.json` MUST include:

- `season`
- `target_round`
- `mode`
- `current_year`
- `training_rounds`
- `candidate_round`
- `visible_max_round`
- `fixture_mode`
- `matchup_context_mode`
- `fixture_source_directory`
- `fixture_manifest_paths`
- `fixture_manifest_sha256`
- `fixture_generator_versions`
- `model_id`
- `footystats_mode`
- `footystats_evaluation_scope`
- `footystats_league_slug`
- `footystats_matches_source_path`
- `footystats_matches_source_sha256`
- `feature_columns`
- `playable_statuses`
- `formation`
- `allowed_formations`
- `captain_policy_definitions`
- `captain_policy_diagnostics`
- `budget`
- `random_seed`
- `finalized_live_data_detected`
- `finalized_live_data_evidence`
- `allow_finalized_live_data`
- `live_workflow`
- `optimizer_status`
- `warnings`
- `generated_at_utc`
- scoring contract fields from `contract_fields()`

`finalized_live_data_evidence` MUST include `pontuacao_non_zero_count`, `entrou_em_campo_true_count`, and `non_zero_scout_count`.

## Live Workflow Metadata JSON

`live_workflow_metadata.json` MUST include:

- `workflow_version`
- `run_started_at_utc`
- `capture_policy`
- `season`
- `current_year`
- `target_round`
- `budget`
- `model_id`
- `fixture_mode`
- `matchup_context_mode`
- `footystats_mode`
- `footystats_league_slug`
- `capture_csv_path`
- `capture_metadata_path`
- `capture_csv_sha256`
- `capture_captured_at_utc`
- `capture_age_seconds`
- `capture_status_mercado`
- `capture_deadline_timestamp`
- `capture_deadline_parse_status`
- `recommendation_output_path`
- `recommendation_summary_path`
- `recommendation_metadata_path`
- `recommended_squad_path`
- `candidate_predictions_path`
- selected squad summary fields
- captain summary fields
- `budget_used`
- `finalized_live_data_detected`
- `finalized_live_data_evidence`
- `allow_finalized_live_data`
- `status`
- `error_stage`
- `error_type`
- `error_message`

On success, `status` MUST be `ok`. On recommendation-stage failure, the workflow MAY write failed workflow metadata, but recommendation artifacts MUST NOT be written.

## Leakage And Override Rules

Live recommendation generation MUST inspect target-round candidate data for finalized evidence before writing recommendation artifacts. If any of `pontuacao_non_zero_count`, `entrou_em_campo_true_count`, or `non_zero_scout_count` is greater than zero, `finalized_live_data_detected` MUST be true.

When finalized data is detected and `allow_finalized_live_data` is false, the workflow MUST fail before writing recommendation artifacts.

When finalized data is detected and `allow_finalized_live_data` is true, the workflow MAY complete, but it MUST record the evidence in metadata and surface a CLI warning.
