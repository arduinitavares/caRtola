# One-Command Live Round Workflow Design

## Problem

The live gameplay flow now works, but it still requires multiple commands and manual coordination:

1. capture the open Cartola market round;
2. confirm the captured round number;
3. run live recommendation with the same round;
4. inspect output files.

That is acceptable for testing, but it is too easy to make an operational mistake before a real lineup decision: use stale market data, pass the wrong `target_round`, overwrite the previous recommendation output, or lose the link between a recommendation and the market capture that produced it.

## Goal

Add a single live-round orchestration command that captures or validates the open market round, runs the live recommendation for that exact round, archives the result in a unique output directory, and writes metadata linking the recommendation to the captured raw data.

The first version improves weekly operational safety. It does not add new model features, new strategies, fixture context, DNP modeling, or Cartola squad submission.

## Command

Default live workflow:

```bash
uv run --frozen python scripts/run_live_round.py \
  --season 2026 \
  --budget 100 \
  --footystats-mode ppg \
  --current-year 2026
```

CLI options:

- `--season`: required positive integer.
- `--budget`: float, default `100.0`.
- `--project-root`: path, default `.`.
- `--output-root`: path, default `data/08_reporting/recommendations`.
- `--footystats-mode`: `none | ppg | ppg_xg`, default `ppg`.
- `--footystats-league-slug`: default `brazil-serie-a`.
- `--footystats-dir`: default `data/footystats`.
- `--current-year`: optional positive integer; tests and examples must pass it explicitly.
- `--capture-policy`: `fresh | missing | skip`, default `fresh`.
- `--allow-finalized-live-data`: forwarded to recommendation, default false.

Do not expose `--target-round` in v1. The target round is determined from the current open Cartola market status or from validated capture metadata for that active round.

Do not expose `--fixture-mode` in v1. Recommendations remain `fixture_mode="none"`.

## Capture Policies

The wrapper supports three capture policies.

### `fresh`

Default live-game behavior.

Rules:

- Delegate current market status and athlete fetching to the existing safe capture primitive.
- Capture the active open market round by calling the existing safe capture primitive with:
  - `auto=True`
  - `force=True`
- This may only replace a previous valid live capture with compatible `.capture.json`.
- It must not overwrite arbitrary historical or manual raw files.
- The recommendation target round must be the `target_round` returned by the capture result.

This policy favors current data over preserving an earlier live capture. It is the correct default because prices and statuses can change before lock.

### `missing`

Use an existing valid capture if one already exists for the active open round; otherwise capture without force.

Rules:

- Fetch current Cartola market status to determine `rodada_atual` and confirm the market is open.
- If `data/01_raw/{season}/rodada-{rodada_atual}.csv` and matching `.capture.json` form a valid previous live capture, reuse them.
- Print the reused capture's `captured_at_utc`, capture age, metadata path, and CSV hash.
- If no valid capture exists, call the capture primitive with `auto=True` and `force=False`.
- If an invalid or non-live-capture CSV exists for the active round, fail before recommendation.

This policy is useful when the user wants to avoid refreshing the market snapshot, but it must make staleness visible.

### `skip`

Do not capture market athlete data; require a valid existing live capture for the active open round.

Rules:

- Fetch current Cartola market status to determine `rodada_atual` and confirm the market is open.
- Do not call the market capture primitive and do not request `atletas/mercado`.
- Require a valid existing `rodada-{rodada_atual}.csv` plus `.capture.json`.
- Fail before recommendation if the capture is missing, invalid, stale for another round, or not a live capture.

This policy is useful for rerunning the recommendation model against an already captured market snapshot.

## Output Archive

The wrapper must not overwrite prior recommendation outputs.

Each run writes to a unique archived output directory under the existing recommendation tree:

```text
data/08_reporting/recommendations/{season}/round-{target_round}/live/runs/run_started_at={YYYYMMDDTHHMMSSffffffZ}/
```

The timestamp is UTC and is generated once at the beginning of the wrapper run. The directory must not already exist. If it exists, the command fails rather than reusing or deleting it.

The existing `recommend_squad.py` command keeps its current deterministic output path for direct use:

```text
data/08_reporting/recommendations/{season}/round-{target_round}/live/
```

To support archived wrapper runs without path hacks, the recommendation module must accept an optional run directory suffix. When present, `RecommendationConfig.output_path` appends:

```text
runs/run_started_at={timestamp}
```

after the existing `live` mode directory.

## Data Flow

The wrapper sequence is:

1. Resolve `current_year` and validate `season == current_year`.
2. Generate `run_started_at_utc`.
3. Apply the selected capture policy.
4. Determine `target_round` from the capture result or validated active-round capture metadata.
5. Build a live `RecommendationConfig` with:
   - `mode="live"`
   - `target_round` from step 4
   - `fixture_mode="none"` implicitly through the recommendation workflow
   - `output_run_id="run_started_at={timestamp}"`
6. Run recommendation.
7. Write workflow metadata linking capture inputs and recommendation outputs.
8. Print a concise terminal summary.

No separate guessed or user-provided round may be used for recommendation in v1.

## Metadata

The wrapper writes an authoritative workflow metadata file in the archived output directory:

```text
live_workflow_metadata.json
```

It records:

- `workflow_version`
- `run_started_at_utc`
- `capture_policy`
- `season`
- `current_year`
- `target_round`
- `budget`
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
- `selected_count`
- `predicted_points`
- `budget_used`
- `status`: `ok | failed`
- `error_stage`: nullable string
- `error_type`: nullable string
- `error_message`: nullable string

The recommendation `run_metadata.json` must also include a `live_workflow` object with the capture policy, capture CSV path, capture metadata path, capture CSV SHA-256, and wrapper output path. This prevents a copied recommendation folder from losing the capture link.

## Failure Behavior

Failure must be explicit and stage-aware.

If capture or capture validation fails:

- do not run recommendation;
- print the failed stage and message;
- no archived recommendation output is required.

If capture succeeds but recommendation fails:

- create the archive output directory;
- write `live_workflow_metadata.json` with `status="failed"` and `error_stage="recommendation"`;
- include the capture CSV path, capture metadata path, and CSV hash;
- print that capture succeeded but recommendation failed.

If metadata writing fails after recommendation succeeds:

- return a non-zero exit code;
- leave the recommendation outputs in place;
- print the output path and explain that workflow metadata failed.

## Terminal Output

On success, print a compact Rich summary:

- capture policy;
- target round;
- capture timestamp and age;
- selected player count;
- predicted points;
- budget used;
- recommendation output path;
- capture metadata path.

For `missing` and `skip`, capture age must be visible because those modes can reuse stale market data by design.

## FootyStats Boundary

The wrapper does not download or refresh FootyStats files.

FootyStats validation remains inside the recommendation path:

- `footystats_mode=ppg` is the default;
- `ppg_xg` remains experimental;
- missing target-round FootyStats rows fail through the existing recommendation validation;
- stale FootyStats files are not solved by this wrapper in v1.

The terminal summary must make it clear which `footystats_mode` was used.

## Safety Rules

- The wrapper must not write under `data/08_reporting/backtests/`.
- Recommendation archive directories are append-only in v1.
- The wrapper must not overwrite prior archived recommendation outputs.
- Raw market capture overwrites are delegated only to the safe capture primitive.
- `fresh` must mean "latest safe live capture", not "overwrite anything".
- `missing` and `skip` must validate existing capture metadata and CSV hash before recommendation.

## Tests

Add focused tests for:

- default CLI arguments use `capture_policy="fresh"` and `footystats_mode="ppg"`;
- `fresh` calls capture with `auto=True` and `force=True`;
- recommendation uses the `target_round` returned by capture;
- `missing` reuses a valid active-round capture and prints capture age;
- `missing` captures without force when no active-round capture exists;
- `missing` fails on an invalid existing active-round CSV;
- `skip` validates an existing capture and does not call market capture;
- `skip` fails when no valid active-round capture exists;
- every successful run writes to a unique archived path;
- archive path collisions fail instead of overwriting;
- workflow metadata links capture CSV, capture metadata, CSV hash, recommendation metadata, and squad output;
- recommendation `run_metadata.json` includes the `live_workflow` link;
- capture success plus recommendation failure writes failed workflow metadata with capture details;
- output-root validation rejects paths under `data/08_reporting/backtests/`.

## Out Of Scope

- automatic scheduling;
- push notifications;
- Cartola squad submission;
- downloading FootyStats files;
- strict fixture integration;
- exploratory fixture integration;
- DNP probability modeling;
- new model families;
- patrimônio simulation.
