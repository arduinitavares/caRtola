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

Add a v1 command that submits one existing live recommendation artifact to
Cartola with strong validation, dry-run default behavior, and an audit trail.

The command must not generate a new squad. It reads the selected artifact,
validates it against the current open Cartola market, builds the Cartola payload,
and submits only when `--confirm-submit` is present.

## Non-Goals

- No one-command recommend-and-submit workflow.
- No automatic selection among multiple recommendation runs.
- No browser automation fallback.
- No support for replay or historical artifacts.
- No password-based Globo login flow.
- No token storage in committed files, reports, command arguments, or logs.
- No promotion of exploratory models or model changes.

## Command

Dry-run default:

```bash
uv run --frozen python scripts/submit_recommended_squad.py \
  --recommendation-path data/08_reporting/recommendations/2026/round-16/live/runs/run_started_at=...
```

Real submission:

```bash
uv run --frozen python scripts/submit_recommended_squad.py \
  --recommendation-path data/08_reporting/recommendations/2026/round-16/live/runs/run_started_at=... \
  --confirm-submit
```

CLI options:

- `--recommendation-path`: required path to one recommendation run directory.
- `--project-root`: default `.`.
- `--timeout-seconds`: default `30.0`.
- `--confirm-submit`: opt-in real POST. Omitted means dry-run.
- `--allow-non-approved-model`: opt-in override for exploratory artifacts.

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

`CARTOLA_GLB_TOKEN` is required only when `--confirm-submit` is present. Dry-run
does not require a token unless a future validation path needs an authenticated
read.

The command must never serialize the token. Audit output may record only:

- `auth_token_present`: boolean;
- `auth_token_source`: `environment`, `.env`, `missing`, or `not_required`;
- `auth_header_name`: `X-GLB-Token`.

## Inputs

The submitter reads these files from the recommendation run directory:

- `recommended_squad.csv`;
- `recommendation_summary.json`;
- `run_metadata.json`;
- `live_workflow_metadata.json`, when present.

The path must resolve inside `project_root`. The command must reject paths under
`data/08_reporting/backtests` because backtest outputs are not live submission
artifacts.

## Artifact Validation

The artifact must pass all checks before any POST:

- recommendation mode is `live`;
- target season equals the current Cartola market season;
- target round equals `mercado/status.rodada_atual`;
- market status is open (`status_mercado == 1`);
- recommendation contains exactly `12` selected rows;
- selected rows contain exactly `1` `tec`;
- selected rows contain exactly `11` non-tecnico players;
- exactly one row has `is_captain == true`;
- captain row is not `tec`;
- all selected `id_atleta` values are finite integers;
- selected athlete IDs are unique;
- all selected `status` values are in the recommendation's playable status
  policy, initially `Provavel`;
- `budget_used <= budget` using artifact summary values;
- artifact target model is the approved live default unless
  `--allow-non-approved-model` is present.

The v1 approved live default is:

- `model_id = xgboost_depth2_l2_heavy`;
- `footystats_mode = ppg_xg`;
- `fixture_mode = none`;
- `matchup_context_mode = none`;
- `scoring_contract_version = cartola_standard_2026_v1`.

This gate prevents accidental submission of Ridge, blend, Optuna, fixture, or
matchup-context research artifacts before they have explicit promotion evidence.

## Cartola API Reads

Before submission, call:

```text
GET https://api.cartola.globo.com/mercado/status
GET https://api.cartola.globo.com/esquemas
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

## Submission Payload

The v1 payload uses the Cartola client shape used by existing API clients:

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

## Submission API Call

Real submit uses:

```text
POST https://api.cartola.globo.com/auth/time/salvar
Header: X-GLB-Token: <token>
Content-Type: application/json
```

The legacy host `https://api.cartolafc.globo.com` must not be the default. A
future fallback may be added only if the current host fails with a documented
compatibility error.

The command treats non-2xx HTTP status, invalid JSON, or a response that does
not clearly confirm a saved lineup as a failed submission. Failed submissions
write an audit file and exit nonzero.

## Output Artifacts

Dry-run and real submit write beside the recommendation run:

```text
cartola_submission_payload.json
cartola_submission_result.json
```

`cartola_submission_payload.json` contains the sanitized payload and validation
context:

```json
{
  "payload": {
    "esquema": 3,
    "atletas": [123, 456],
    "capitao": 123
  },
  "payload_sha256": "...",
  "recommendation_path": "...",
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

`cartola_submission_result.json` contains execution status:

```json
{
  "submission_status": "dry_run",
  "would_submit": true,
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
  "auth_token_source": ".env",
  "payload_sha256": "..."
}
```

The raw response body may be written only after removing token-like fields. The
request headers must never be serialized.

## Failure Handling

These failures exit nonzero before POST:

- missing recommendation path;
- path outside `project_root`;
- missing required artifact files;
- invalid JSON or CSV schema;
- live mode missing or false;
- market closed;
- market round mismatch;
- market season mismatch;
- unsupported or unmapped formation;
- malformed squad size or positions;
- duplicate athlete IDs;
- missing or invalid captain;
- captain is técnico;
- non-approved model without override;
- missing token with `--confirm-submit`.

These failures happen after POST and still write `cartola_submission_result.json`:

- non-2xx response;
- timeout;
- invalid response JSON;
- response JSON does not confirm saved lineup.

## Testing

Unit tests:

- build payload from a valid recommendation fixture;
- payload contains 12 selected IDs including técnico;
- payload contains one non-tecnico captain;
- formation ID is resolved from `/esquemas`;
- missing formation mapping fails;
- market closed fails before POST;
- market round mismatch fails before POST;
- malformed squad size fails before POST;
- duplicate athlete IDs fail before POST;
- captain missing fails before POST;
- captain técnico fails before POST;
- non-approved model fails without `--allow-non-approved-model`;
- token is required only for confirmed submit;
- token is never present in output payload/result JSON.

CLI tests:

- missing recommendation path exits nonzero;
- dry-run writes payload and result without calling POST;
- `--confirm-submit` calls POST with `X-GLB-Token`;
- API failure writes failed result and exits nonzero;
- `.env` is loaded with `override=False`.

Integration-style mocked tests:

- mock `/mercado/status`, `/esquemas`, and `/auth/time/salvar`;
- verify no POST occurs when any validation fails;
- verify one POST occurs for a valid confirmed submission.

## Operational Workflow

Generate one or more recommendations:

```bash
uv run --frozen python scripts/run_live_round.py \
  --season 2026 \
  --budget 92.98 \
  --current-year 2026
```

Inspect the preferred run directory.

Dry-run the submission:

```bash
uv run --frozen python scripts/submit_recommended_squad.py \
  --recommendation-path data/08_reporting/recommendations/2026/round-16/live/runs/run_started_at=...
```

Submit only after reviewing the dry-run:

```bash
uv run --frozen python scripts/submit_recommended_squad.py \
  --recommendation-path data/08_reporting/recommendations/2026/round-16/live/runs/run_started_at=... \
  --confirm-submit
```

## Implementation Notes

Recommended module boundary:

- `src/cartola/backtesting/squad_submission.py`
  - artifact loading;
  - validation;
  - payload building;
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

- A valid artifact dry-run writes sanitized payload and result files.
- A valid artifact confirmed submit sends exactly one authenticated POST.
- Any failed validation prevents POST.
- No token is printed or written.
- The command can submit only approved live-default artifacts unless explicitly
  overridden.
- The implementation is covered by focused unit and CLI tests.
- `uv run --frozen pyrepo-check --all` passes.
