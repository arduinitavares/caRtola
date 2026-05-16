# Cartola Artifact Squad Submission Design

## Problem

The live workflow can generate a recommendation, but submitting the lineup still
requires manually copying players into Cartola. That is slow before market lock
and creates operational risk: the user can transpose a player, forget the
captain, submit a different formation, or submit a stale recommendation.

The system should submit a squad automatically only after the user explicitly
chooses one generated recommendation artifact. This keeps the model/recommend
workflow separate from authenticated write access and lets the user compare
multiple recommendation runs before committing one.

## Goal

Add a v1 command that converts one existing live recommendation artifact into a
reviewable Cartola submission plan, then submits that exact plan only after a
second explicit payload-hash confirmation.

The command must not generate a new squad. It reads the selected artifact,
validates it against the current open Cartola market, builds the Cartola payload,
writes a sanitized submission plan, and submits only from that plan. Real
authenticated POST support is blocked until the exact Cartola save request,
success response, error response, and read-back behavior are verified from a
controlled browser capture, official documentation, or a disposable account.

## Non-Goals

- No one-command recommend-and-submit workflow.
- No automatic selection among multiple recommendation runs.
- No browser automation fallback.
- No support for replay or historical artifacts.
- No password-based Globo login flow.
- No token storage in committed files, reports, command arguments, or logs.
- No promotion of exploratory models or model changes.

## Command

Plan/dry-run default:

```bash
uv run --frozen python scripts/submit_recommended_squad.py \
  --recommendation-path data/08_reporting/recommendations/2026/round-16/live/runs/run_started_at=...
```

Confirmed submit from a reviewed plan:

```bash
uv run --frozen python scripts/submit_recommended_squad.py \
  --submission-plan data/08_reporting/recommendations/2026/round-16/live/runs/run_started_at=.../submission_attempts/attempt_started_at=.../submission_plan.json \
  --confirm-payload-sha256 <payload_sha256> \
  --confirm-submit
```

CLI options:

- `--recommendation-path`: path to one recommendation run directory. Mutually
  exclusive with `--submission-plan`.
- `--submission-plan`: path to a previously generated sanitized submission plan.
  Mutually exclusive with `--recommendation-path`.
- `--project-root`: default `.`.
- `--timeout-seconds`: default `30.0`.
- `--confirm-submit`: opt-in real POST. Valid only with `--submission-plan`.
- `--confirm-payload-sha256`: required with `--confirm-submit`; must match the
  reviewed plan payload hash exactly.
- `--allow-non-approved-model`: opt-in override for exploratory artifacts.
- `--override-reason`: required when `--allow-non-approved-model` is used with a
  confirmed submit.

No `--token` CLI flag is allowed. Tokens in command arguments are visible in
shell history and process listings.

## Authentication

The command loads `.env` from `project_root` with `python-dotenv` and
`override=False`, then reads:

```bash
CARTOLA_GLB_TOKEN=...
```

The token can also be supplied by the shell environment. If both exist, the
already-exported shell variable wins because `.env` loading uses
`override=False`.

`CARTOLA_GLB_TOKEN` is required only when `--confirm-submit` is present. Plan
generation does not require a token unless an authenticated preflight endpoint is
later verified and enabled.

The command must never serialize the token. Audit output may record only:

- `auth_token_present`: boolean;
- `auth_token_source`: `process_env`, `project_dotenv`, `missing`, or
  `not_required`;
- `auth_header_name`: `X-GLB-Token`.

## Inputs

Plan generation reads these files from the recommendation run directory:

- `recommended_squad.csv`;
- `recommendation_summary.json`;
- `run_metadata.json`;
- `live_workflow_metadata.json`, when present.

The recommendation path must resolve inside `project_root` and match the
canonical live recommendation shape:

```text
data/08_reporting/recommendations/<season>/round-<round>/live/runs/run_started_at=*
```

The command must reject any path whose resolved components include `backtests`,
`experiments`, `policy_simulations`, `blend_diagnostics`, `oracle_discovery`,
`ebm_diagnostics`, or any non-recommendation reporting root.

Plan generation writes SHA-256 hashes for every source artifact it reads. If a
future live workflow writes a generation manifest, the submitter must verify it;
until then, confirmed submit must verify that the source files are unchanged
from the reviewed plan.

Confirmed submit reads only `submission_plan.json` plus the unchanged source
artifacts referenced by that plan.

## Artifact Validation

The artifact and current market must pass all checks before plan generation and
again immediately before confirmed submit:

- `run_metadata.mode == "live"`;
- resolved path matches the canonical live recommendation shape;
- artifact season equals `mercado/status.temporada`;
- artifact target round equals `mercado/status.rodada_atual`;
- market status is open by the explicit market-open predicate;
- recommendation contains exactly `12` selected rows;
- selected rows contain exactly `1` `tec`;
- selected rows contain exactly `11` non-tecnico players;
- selected position counts exactly match the selected formation from `/esquemas`;
- exactly one row has `is_captain == true`;
- captain row is not `tec`;
- all selected `id_atleta` values are finite integers;
- selected athlete IDs are unique;
- every selected athlete ID exists in the current market snapshot;
- current market name, position, club, price, and status match the artifact
  values after explicit normalization rules: athlete ID exact, position ID/name
  exact, club ID exact when available, status ID/name exact when available, and
  price unchanged within `0.01`;
- all selected current-market statuses satisfy the playable status allowlist by
  status ID/name, initially `Provavel`;
- `budget_used <= budget` using artifact summary values;
- account budget is verified when a safe authenticated read endpoint is
  available; otherwise record `account_budget_verified=false`;
- source artifact hashes match the reviewed submission plan on confirmed submit;
- artifact target model is the approved live default unless
  `--allow-non-approved-model` is present.

The market-open predicate is:

- `status_mercado == 1`;
- `game_over` is not true;
- artifact season equals `temporada`;
- artifact target round equals `rodada_atual`;
- `fechamento.timestamp` is present and still in the future by at least the
  configured safety margin, default `120` seconds.

The v1 approved live default is:

- `model_id = xgboost_depth2_l2_heavy`;
- `footystats_mode = ppg_xg`;
- `fixture_mode = none`;
- `matchup_context_mode = none`;
- `scoring_contract_version = cartola_standard_2026_v1`.

This approval profile prevents accidental submission of Ridge, blend, Optuna,
fixture, or matchup-context research artifacts before they have explicit
promotion evidence. If the override is used for a real submit, the attempt must
record the override reason and require the payload-hash confirmation.

## Cartola API Reads

Before submission, call:

```text
GET https://api.cartola.globo.com/mercado/status
GET https://api.cartola.globo.com/esquemas
GET https://api.cartola.globo.com/atletas/mercado
```

`/mercado/status` validates that the market is open and that the artifact round
is the active round.

`/esquemas` provides the current formation ID mapping. V1 must not hardcode an
unverified formation ID table as the only source of truth. It may include a
static fallback table for tests, but runtime submission should prefer the API
mapping and verify that the returned position counts match the local selected
formation.

As of the design date, the public `/esquemas` endpoint returns these labels and
IDs:

| Formation | esquema_id |
|---|---:|
| `3-4-3` | 1 |
| `3-5-2` | 2 |
| `4-3-3` | 3 |
| `4-4-2` | 4 |
| `4-5-1` | 5 |
| `5-3-2` | 6 |
| `5-4-1` | 7 |

If the artifact formation is missing from `/esquemas`, fail before POST.

`/atletas/mercado` provides the current athlete snapshot. V1 must use it to
validate every selected athlete by ID and reject drift in position, status,
availability, club, name, or price. The command must not repair drift by
refilling or replacing players.

Immediately before confirmed submit, re-fetch `/mercado/status` and
`/atletas/mercado`. If the market closed, the round changed, or any selected
athlete drifted after plan generation, fail before POST.

## Submission Payload

The provisional v1 payload uses the Cartola client shape used by existing API
clients:

```json
{
  "esquema": 3,
  "atletas": [123, 456, 789],
  "capitao": 123
}
```

Rules:

- `esquema` is the integer ID for the artifact formation.
- `atletas` contains all `12` selected athlete IDs, including the técnico.
- `capitao` is the selected non-tecnico captain athlete ID.
- Do not send a top-level `tecnico` field in v1.

The command should build the payload from the artifact only. It must not
re-optimize, refill missing positions, change captain, change formation, or
fetch a different squad from live market data.

This payload shape is not promotion-grade until verified against the real save
endpoint. Real submit must remain disabled or fail closed until the implementation
has a captured contract fixture covering request body, accepted status code,
success response schema, error response schema, and read-back behavior.

## Submission API Call

Real submit uses this endpoint only after the save contract is verified:

```text
POST https://api.cartola.globo.com/auth/time/salvar
Header: X-GLB-Token: <token>
Content-Type: application/json
```

The legacy host `https://api.cartolafc.globo.com` must not be the default. A
future fallback may be added only if the current host fails with a documented
compatibility error.

The command treats non-2xx HTTP status, invalid JSON, a 2xx response with an
error payload, or a response that does not clearly confirm a saved lineup as a
failed submission. Failed submissions write a sanitized audit file and exit
nonzero.

After a successful-looking POST, the command must perform a safe authenticated
read-back through a verified endpoint. `submission_status="submitted"` is
allowed only after the response and read-back criteria pass. If no read-back
endpoint has been verified, real submit remains disabled and the command must
fail closed before POST.

## Output Artifacts

Plan generation and real submit write under a unique attempt directory beside
the recommendation run:

```text
submission_attempts/attempt_started_at=<timestamp>/
  submission_plan.json
  submission_result.json
```

`submission_plan.json` contains the sanitized payload, payload hash, source
artifact hashes, current-market validation report, and human-readable summary:

```json
{
  "plan_status": "ready_for_review",
  "payload": {
    "esquema": 3,
    "atletas": [123, 456],
    "capitao": 123
  },
  "payload_sha256": "...",
  "recommendation_path": "...",
  "source_artifact_hashes": {
    "recommended_squad.csv": "...",
    "recommendation_summary.json": "...",
    "run_metadata.json": "..."
  },
  "target_round": 16,
  "formation": "4-3-3",
  "selected_count": 12,
  "captain_id": 123,
  "captain_name": "Player",
  "model_id": "xgboost_depth2_l2_heavy",
  "footystats_mode": "ppg_xg",
  "fixture_mode": "none",
  "matchup_context_mode": "none"
}
```

`submission_result.json` contains execution status:

```json
{
  "submission_status": "plan_only",
  "would_submit": false,
  "submitted_at_utc": null,
  "http_status": null,
  "auth_token_present": false,
  "auth_token_source": "not_required"
}
```

For real submit:

```json
{
  "submission_status": "submitted",
  "would_submit": false,
  "submitted_at_utc": "2026-05-16T13:00:00Z",
  "http_status": 200,
  "auth_token_present": true,
  "auth_token_source": "project_dotenv",
  "payload_sha256": "..."
}
```

The raw response body may be written only after removing token-like fields. The
request headers must never be serialized.

Attempt directories must be unique so repeated dry-runs and submits do not
overwrite evidence. Audit files should be created with restrictive permissions
where the platform supports it.

## Failure Handling

These failures exit nonzero before POST:

- missing recommendation path;
- missing submission plan for confirmed submit;
- payload hash confirmation missing or mismatched;
- path outside `project_root`;
- path outside canonical live recommendation tree;
- missing required artifact files;
- source artifact hash mismatch;
- invalid JSON or CSV schema;
- live mode missing or false;
- market closed;
- market round mismatch;
- market season mismatch;
- market closes or round changes between plan generation and submit;
- selected athlete missing from current market;
- selected athlete current-market status, position, club, name, or price drift;
- unsupported or unmapped formation;
- malformed squad size or positions;
- selected positions do not match `/esquemas` formation counts;
- duplicate athlete IDs;
- missing or invalid captain;
- captain is técnico;
- non-approved model without override;
- missing token with `--confirm-submit`.

These failures happen after POST and still write `submission_result.json`:

- non-2xx response;
- timeout;
- invalid response JSON;
- 2xx response with error payload;
- response JSON does not confirm saved lineup;
- read-back does not match submitted payload.

There is no automatic retry after `401`, `403`, `409`, `422`, timeout,
non-JSON response, or round mismatch. A failed confirmed submit must never retry
into a changed market state or next round.

## Testing

Unit tests:

- build payload from a valid recommendation fixture;
- payload contains 12 selected IDs including técnico;
- payload contains one non-tecnico captain;
- formation ID is resolved from `/esquemas`;
- missing formation mapping fails;
- selected position counts must match formation counts;
- market closed fails before POST;
- market round mismatch fails before POST;
- market closes between plan and submit fails before POST;
- current-market athlete drift fails before POST;
- source artifact hash mismatch fails before POST;
- malformed squad size fails before POST;
- duplicate athlete IDs fail before POST;
- captain missing fails before POST;
- captain técnico fails before POST;
- non-approved model fails without `--allow-non-approved-model`;
- token is required only for confirmed submit;
- token is never present in output payload/result JSON;
- token-like strings are redacted from HTTP errors.

CLI tests:

- missing recommendation path exits nonzero;
- plan generation writes `submission_plan.json` and result without calling POST;
- confirmed submit requires `--submission-plan`, `--confirm-submit`, and
  `--confirm-payload-sha256`;
- confirmed submit calls POST with `X-GLB-Token` only after revalidation;
- API failure writes failed result and exits nonzero;
- `.env` is loaded from explicit project-root path with `override=False`;
- symlink and path traversal attempts outside project root fail.

Integration-style mocked tests:

- mock `/mercado/status`, `/esquemas`, `/atletas/mercado`, and
  `/auth/time/salvar`;
- verify no POST occurs when any validation fails;
- verify one POST occurs for a valid confirmed submission;
- verify read-back mismatch prevents `submission_status="submitted"`;

## Operational Workflow

Generate one or more recommendations:

```bash
uv run --frozen python scripts/run_live_round.py \
  --season 2026 \
  --budget 92.98 \
  --current-year 2026
```

Inspect the preferred run directory.

Generate and review a submission plan:

```bash
uv run --frozen python scripts/submit_recommended_squad.py \
  --recommendation-path data/08_reporting/recommendations/2026/round-16/live/runs/run_started_at=...
```

Submit only after reviewing the plan and copying its payload hash:

```bash
uv run --frozen python scripts/submit_recommended_squad.py \
  --submission-plan data/08_reporting/recommendations/2026/round-16/live/runs/run_started_at=.../submission_attempts/attempt_started_at=.../submission_plan.json \
  --confirm-payload-sha256 <payload_sha256> \
  --confirm-submit
```

## Implementation Notes

Recommended module boundary:

- `src/cartola/backtesting/squad_submission.py`
  - artifact loading;
  - validation;
  - payload building;
  - submission plan building;
  - source artifact hashing;
  - API client functions;
  - audit writing.
- `scripts/submit_recommended_squad.py`
  - CLI parsing;
  - `.env` bootstrap;
  - console output;
  - exit codes.

Use dependency injection for HTTP functions in tests. Do not call the real
authenticated endpoint in automated tests.

## Acceptance Criteria

- A valid artifact plan writes sanitized `submission_plan.json` and
  `submission_result.json` under a unique attempt directory.
- A valid confirmed submit sends exactly one authenticated POST only after the
  payload hash is confirmed and all source/current-market checks are re-run.
- Any failed validation prevents POST.
- No token is printed or written.
- The command can submit only approved live-default artifacts unless explicitly
  overridden.
- Real submit remains fail-closed until the save API contract is verified by a
  captured request/response or disposable-account test fixture.
- The implementation is covered by focused unit and CLI tests.
- `uv run --frozen pyrepo-check --all` passes.
