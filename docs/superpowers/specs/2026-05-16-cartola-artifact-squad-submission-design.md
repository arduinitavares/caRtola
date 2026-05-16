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
reviewable Cartola submission plan. Real authenticated submission is a separate
Phase 2 capability and must not be implemented in Phase 1.

The command must not generate a new squad. It reads the selected artifact,
validates it against the current open Cartola market, builds the provisional
Cartola payload, writes a sanitized submission plan, and hard-disables
`--confirm-submit` with `CONTRACT_UNVERIFIED` until Phase 2 lands.

## Non-Goals

- No one-command recommend-and-submit workflow.
- No automatic selection among multiple recommendation runs.
- No browser automation fallback.
- No support for replay or historical artifacts.
- No password-based Globo login flow.
- No token storage in committed files, reports, command arguments, or logs.
- No promotion of exploratory models or model changes.

## Delivery Phases

Phase 1 is the only approved implementation scope for this spec:

- generate a sanitized submission plan from one reviewed live artifact;
- validate public market status, formation, current athlete drift, artifact
  hashes, payload shape, and approved-profile metadata;
- compute a canonical payload hash;
- write unique attempt audit artifacts;
- parse future submit flags only to fail safely;
- any invocation with `--confirm-submit` must exit nonzero with
  `CONTRACT_UNVERIFIED` before loading `.env`, reading `CARTOLA_GLB_TOKEN`,
  constructing an authenticated HTTP client, or constructing any POST request.

Phase 1 may include mocked tests for the future submit CLI shape, but those tests
must assert that no token is read and no POST-capable code path is reached.

Phase 2 requires a separate spec and commit before any real POST is enabled.
That later spec must add:

- a verified save request fixture from a controlled browser capture, official
  documentation, or disposable account;
- accepted success status codes, success schema, and error schema;
- a verified authenticated preflight endpoint for team/account identity, current
  budget, and current lineup context;
- a verified authenticated read-back endpoint for saved lineup verification;
- nonsecret expected identity config, for example
  `CARTOLA_EXPECTED_TEAM_ID`;
- an explicit code gate that can be reviewed in diff before enabling POST;
- tests for wrong-team token, budget mismatch, save response validation, and
  read-back mismatch using sanitized captured fixtures.

## Command

Plan/dry-run default:

```bash
uv run --frozen python scripts/submit_recommended_squad.py \
  --recommendation-path data/08_reporting/recommendations/2026/round-16/live/runs/run_started_at=...
```

Future Phase 2 confirmed submit from a reviewed plan:

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
  In Phase 1 this always fails with `CONTRACT_UNVERIFIED`.
- `--confirm-payload-sha256`: required with `--confirm-submit`; must match the
  reviewed plan payload hash exactly.
- `--allow-non-approved-model`: opt-in override for exploratory artifacts.
- `--override-reason`: required when `--allow-non-approved-model` is used with a
  confirmed submit.

No `--token` CLI flag is allowed. Tokens in command arguments are visible in
shell history and process listings.

## Authentication

In Phase 2, the command loads `.env` from `project_root` with `python-dotenv`
and `override=False`, then reads:

```bash
CARTOLA_GLB_TOKEN=...
```

The token can also be supplied by the shell environment. If both exist, the
already-exported shell variable wins because `.env` loading uses
`override=False`.

Phase 1 does not load `.env` or read `CARTOLA_GLB_TOKEN`, even when
`--confirm-submit` is provided, because submit fails before auth handling.

In Phase 2, `CARTOLA_GLB_TOKEN` is required only when `--confirm-submit` is
present. Plan generation does not require a token unless an authenticated
preflight endpoint is later verified and enabled.

Phase 2 real submit must also require nonsecret expected identity config such as:

```bash
CARTOLA_EXPECTED_TEAM_ID=...
```

The authenticated preflight response must match the expected team ID before POST.
If the expected identity config is missing or mismatched, fail before POST.

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
- account budget is informational during Phase 1 plan generation and must be
  recorded as `account_budget_verified=false`;
- Phase 2 real submit requires authenticated account budget verification; if
  `account_budget_verified=false`, fail before POST;
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

Before Phase 1 plan generation, call:

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

`/atletas/mercado` provides the current athlete snapshot. Phase 1 must use it to
validate every selected athlete by ID and reject drift in position, status,
availability, club, name, or price. The command must not repair drift by
refilling or replacing players.

For Phase 2, immediately before confirmed submit, re-fetch `/mercado/status` and
`/atletas/mercado`. If the market closed, the round changed, or any selected
athlete drifted after plan generation, fail before POST.

Ignore row-level athlete `rodada_id` in `/atletas/mercado`; public athlete rows
may carry a previous rodada while `/mercado/status.rodada_atual` identifies the
open market round.

Current-market field mapping:

| Check | Artifact field | Market field | Normalization |
|---|---|---|---|
| Athlete ID | `id_atleta` | `atleta_id` | integer exact |
| Nickname | `apelido` | `apelido` | trimmed string exact |
| Position | `posicao` | `posicoes[posicao_id].abreviacao` | lower-case exact |
| Position ID | optional artifact `posicao_id` | `posicao_id` | integer exact when artifact field exists |
| Club | `id_clube` | `clube_id` | integer exact when artifact field exists |
| Price | `preco_pre_rodada` | `preco_num` | numeric, absolute delta <= `0.01` |
| Status | `status` / optional `status_id` | `status_id`, `status.nome` | prefer `status_id == 7`; fallback accent-stripped name equals `provavel` |

If an artifact lacks an exact comparable field, the plan should record that
field as `not_comparable` and continue only for Phase 1. Phase 2 real submit
must require the fields needed for identity, position, club, price, and status
checks to be comparable.

## Submission Payload

The provisional Phase 1 payload uses the Cartola client shape used by existing
API clients:

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

Payload hash rules:

- normalize the payload to JSON with UTF-8 encoding;
- use `sort_keys=True`;
- use compact separators `(",", ":")`;
- preserve `atletas` in the exact order that would be submitted;
- normalize every athlete ID, captain ID, and `esquema` to JSON integers;
- compute SHA-256 over the resulting byte string.

## Submission API Call

Phase 2 real submit uses this endpoint only after the save contract is verified:

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

Before any Phase 2 POST, the command must call a verified authenticated preflight
endpoint and require:

- authenticated team ID equals `CARTOLA_EXPECTED_TEAM_ID`;
- current account budget is present and `>= payload_budget_used`;
- current lineup context belongs to the same season and target round.

After a successful-looking POST, the command must perform a safe authenticated
read-back through a verified endpoint. `submission_status="submitted"` is
allowed only after the response and read-back criteria pass. If no read-back
endpoint has been verified, real submit remains disabled and the command must
fail closed before POST.

## Output Artifacts

Plan generation writes under a unique attempt directory beside the recommendation
run:

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

Phase 2 submit-from-plan must not overwrite the original plan attempt. It should
create a new unique submit attempt directory that references the reviewed plan:

```text
submission_attempts/submit_started_at=<timestamp>/
  submission_result.json
```

The submit result must include:

- `source_submission_plan`;
- `source_plan_payload_sha256`;
- `confirmed_payload_sha256`;
- fresh source artifact hashes;
- fresh market status snapshot hash;
- authenticated preflight identity status;
- account budget verification status.

The raw response body may be written only after removing token-like fields. The
request headers must never be serialized. Redaction must also cover account
emails, cookies, session identifiers, and raw account names if those appear in
authenticated responses.

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
- `CONTRACT_UNVERIFIED` in Phase 1 for any `--confirm-submit` invocation.
- missing token with `--confirm-submit` in Phase 2.

These Phase 2 failures happen after POST and still write `submission_result.json`:

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

Phase 1 required unit tests:

- build payload from a valid recommendation fixture;
- canonical payload hash is stable for sorted keys and compact UTF-8 JSON;
- payload hash preserves `atletas` submitted order;
- payload contains 12 selected IDs including técnico;
- payload contains one non-tecnico captain;
- formation ID is resolved from `/esquemas`;
- missing formation mapping fails;
- selected position counts must match formation counts;
- status validation prefers `status_id == 7` and falls back to accent-stripped
  `Provável` / `Provavel`;
- athlete row-level `rodada_id` is ignored;
- market closed fails before POST;
- market round mismatch fails before POST;
- current-market athlete drift fails before POST;
- source artifact hash mismatch fails before POST;
- malformed squad size fails before POST;
- duplicate athlete IDs fail before POST;
- captain missing fails before POST;
- captain técnico fails before POST;
- non-approved model fails without `--allow-non-approved-model`;
- Phase 1 `--confirm-submit` fails with `CONTRACT_UNVERIFIED` before token read;
- token is never present in output payload/result JSON;
- token-like strings are redacted from HTTP errors.

Phase 1 required CLI tests:

- missing recommendation path exits nonzero;
- plan generation writes `submission_plan.json` and result without calling POST;
- confirmed submit with `--confirm-submit` exits with `CONTRACT_UNVERIFIED`
  before validating `--submission-plan` payload hash or reading auth config;
- Phase 1 confirmed submit exits with `CONTRACT_UNVERIFIED` and never calls POST;
- API failure writes failed result and exits nonzero;
- `.env` is not loaded in Phase 1;
- symlink and path traversal attempts outside project root fail.

Phase 1 required integration-style mocked tests:

- mock `/mercado/status`, `/esquemas`, and `/atletas/mercado`;
- verify no POST occurs when any validation fails;
- Phase 1 verifies no POST occurs even with `--confirm-submit`;

Future Phase 2 tests, not part of this implementation plan:

- confirmed submit requires `--submission-plan`, `--confirm-submit`, and
  `--confirm-payload-sha256`;
- confirmed submit calls POST with `X-GLB-Token` only after revalidation,
  authenticated identity preflight, and account budget check;
- `--allow-non-approved-model` must be present for both plan generation and
  confirmed submit when the artifact is not approved;
- `.env` is loaded from explicit project-root path with `override=False`;
- mock `/auth/time/salvar`, authenticated preflight, and read-back endpoints;
- Phase 2 verifies one POST occurs for a valid confirmed submission;
- Phase 2 wrong-team token preflight fails before POST;
- Phase 2 account-budget mismatch fails before POST;
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

Phase 1 submit attempts fail closed:

```bash
uv run --frozen python scripts/submit_recommended_squad.py \
  --submission-plan data/08_reporting/recommendations/2026/round-16/live/runs/run_started_at=.../submission_attempts/attempt_started_at=.../submission_plan.json \
  --confirm-payload-sha256 <payload_sha256> \
  --confirm-submit
```

Expected Phase 1 result: nonzero exit with `CONTRACT_UNVERIFIED`, before token
read or POST setup.

In Phase 2, submit only after reviewing the plan and copying its payload hash:

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
  - canonical payload hashing;
  - submission plan building;
  - source artifact hashing;
  - API client functions;
  - audit writing.
- `scripts/submit_recommended_squad.py`
  - CLI parsing;
  - Phase 1 no-auth bootstrap; Phase 2 `.env` bootstrap only after
    `CONTRACT_UNVERIFIED` is removed by a separate spec;
  - console output;
  - exit codes.

Use dependency injection for HTTP functions in tests. Do not call the real
authenticated endpoint in automated tests.

## Acceptance Criteria

- A valid artifact plan writes sanitized `submission_plan.json` and
  `submission_result.json` under a unique attempt directory.
- Phase 1 `--confirm-submit` exits nonzero with `CONTRACT_UNVERIFIED` before
  loading `.env`, reading `CARTOLA_GLB_TOKEN`, constructing a POST request, or
  touching authenticated API code.
- Phase 2 is not part of this implementation plan.
- A future Phase 2 confirmed submit may send exactly one authenticated POST only
  after the payload hash is confirmed, all source/current-market checks are
  re-run, authenticated identity preflight matches `CARTOLA_EXPECTED_TEAM_ID`,
  and account budget is verified.
- Any failed validation prevents POST.
- No token is printed or written.
- The command can submit only approved live-default artifacts unless explicitly
  overridden.
- Real submit remains fail-closed until the save API contract is verified by a
  captured request/response or disposable-account test fixture.
- The implementation is covered by focused unit and CLI tests.
- `uv run --frozen pyrepo-check --all` passes.
