# Strict Round Fixture Capture Command Design

## Goal

Create a dedicated manual command, `scripts/capture_strict_round_fixture.py`, that captures strict pre-lock Cartola fixture evidence for the current production season and immediately generates the canonical strict fixture CSV plus manifest for the same captured snapshot.

This is an operational provenance command. It does not change `run_live_round.py`, `recommend_squad.py`, model features, or live recommendation defaults. Live recommendations continue to use `fixture_mode=none` until strict fixture context is explicitly integrated later.

## Non-Goals

- Do not integrate strict fixtures into live squad recommendation yet.
- Do not schedule cron, launchd, reminders, or other automation.
- Do not overwrite raw fixture snapshot directories.
- Do not support non-Cartola fixture sources in v1.
- Do not write a wrapper-specific metadata file. Existing `capture.json` and strict fixture manifests remain the audit trail.

## CLI Contract

Primary live usage:

```bash
uv run --frozen python scripts/capture_strict_round_fixture.py \
  --season 2026 \
  --auto \
  --current-year 2026
```

Arguments:

- `--season`: required positive integer.
- `--auto`: detect the active open Cartola round from `mercado/status`.
- `--round N`: explicit target round for manual/test use.
- `--current-year`: optional positive integer. If omitted, resolve with UTC runtime year.
- `--source cartola_api`: optional, defaults to `cartola_api`; no other values are accepted.
- `--project-root`: optional, defaults to `Path(".")`.
- `--force-generate`: optional. Allows overwriting only the generated strict fixture CSV and manifest by passing `force=True` to `generate_strict_fixture(...)`.

`--auto` and `--round` are mutually exclusive, and exactly one must be provided. Passing both or neither fails during CLI parsing.

The command fails before network capture if `season != resolved_current_year`.

## Source And Round Resolution

For `--auto`, the command resolves the active round from the Cartola market-status payload using shared parser/helper code from strict fixture snapshot capture or an equally strict helper in the same module. The wrapper must not introduce looser duplicate parsing.

- endpoint: `https://api.cartola.globo.com/mercado/status`;
- required field: `rodada_atual`;
- required field: `status_mercado`;
- `rodada_atual` must parse as a positive integer;
- `status_mercado` must parse as an integer and must equal `1`.

The active-round fetch is advisory for choosing the round number. `capture_cartola_snapshot(...)` still fetches its own fixture and deadline payloads and must validate that the deadline payload matches the requested `season` and `round_number`. If Cartola changes active round between the advisory fetch and snapshot capture, the command may fail after the advisory step, but it must not generate a strict fixture from mismatched evidence. When the advisory `--auto` fetch succeeds but later snapshot capture fails, the error output includes the advisory `round_number` that was attempted.

For `--round N`, the command passes `round_number=N` to `capture_cartola_snapshot(...)`. Either the wrapper or snapshot capture must fail when the deadline payload has `rodada_atual != round_number` or `status_mercado != 1`. V1 implementation must tighten `capture_cartola_snapshot(...)`/`cartola_deadline_at(...)` so both `--auto` and explicit `--round` share the same authoritative open-market validation.

## Data Flow

1. Parse CLI arguments.
2. Resolve `current_year` from `--current-year` or UTC runtime year.
3. Fail before network capture unless `season == current_year`.
4. Resolve `round_number` from either `--auto` or `--round`.
5. Call:

   ```python
   capture_cartola_snapshot(
       project_root=project_root,
       season=season,
       round_number=round_number,
       source="cartola_api",
   )
   ```

6. Capture writes raw strict evidence under:

   ```text
   data/01_raw/fixtures_snapshots/{season}/rodada-{round_number}/captured_at=.../
   ```

   with:

   ```text
   capture.json
   fixtures.json
   deadline.json
   ```

7. Call:

   ```python
   generate_strict_fixture(
       project_root=project_root,
       season=season,
       round_number=round_number,
       source="cartola_api",
       captured_at=capture_result.captured_at_utc,
       force=force_generate,
   )
   ```

   Passing `captured_at=capture_result.captured_at_utc` is required. Generation must bind to the exact snapshot just captured, not implicitly select the latest valid snapshot for that round.

8. Generation writes and validates:

   ```text
   data/01_raw/fixtures_strict/{season}/partidas-{round_number}.csv
   data/01_raw/fixtures_strict/{season}/partidas-{round_number}.manifest.json
   ```

## Force And Overwrite Rules

`--force-generate` only affects the canonical strict fixture and manifest target pair. It maps directly to `generate_strict_fixture(force=True)`.

It does not:

- overwrite raw `fixtures_snapshots/.../captured_at=.../` directories;
- delete existing snapshots;
- bypass strict snapshot validation;
- bypass manifest validation.

If a raw snapshot directory already exists for the same captured timestamp, `capture_cartola_snapshot(...)` raises `FileExistsError`. The wrapper does not recover by overwriting it.

If capture succeeds but strict fixture generation fails because a fixture or manifest already exists and `--force-generate` was not passed, the raw snapshot is retained and the terminal error includes the retained `capture_dir`.

## Output

All printed artifact paths are project-root-relative when they resolve under `project_root`. For example, print `data/01_raw/fixtures_strict/2026/partidas-14.csv`, not an absolute path. If a path unexpectedly cannot be made relative to `project_root`, print the resolved absolute path.

`Captured at UTC` and `Deadline at UTC` are printed as ISO-8601 UTC strings ending in `Z`, with no `+00:00` offset.

On success, print stable labels:

- `Strict fixture capture complete`
- `Season`
- `Round`
- `Snapshot directory`
- `Strict fixture`
- `Manifest`
- `Captured at UTC`
- `Deadline at UTC`

On capture failure, print a clean failure message and do not attempt generation.

On generation failure after capture succeeds, print a clean failure message that includes:

- retained snapshot directory;
- captured-at UTC;
- deadline-at UTC;
- original exception type;
- original exception message.

The command may use Rich formatting, but tests assert stable labels and values rather than box-drawing characters.

## Error Handling

CLI parse failures use argparse's normal parser behavior. Runtime operational failures return exit code `1` without traceback. `main()` catches expected operational exceptions and prints `error: {message}` to stderr.

Parse-time failures include:

- unsupported source;
- missing or conflicting `--auto`/`--round`;

Runtime operational failures include:

- wrong current year;
- Cartola market not open;
- active round parse failure;
- capture failure from `ValueError`, `FileExistsError`, `FileNotFoundError`, JSON decode errors, or `requests.RequestException`;
- existing generated strict fixture without `--force-generate`;
- manifest or hash validation failure.

Unexpected exceptions may still propagate during development. The v1 command handles known operational errors cleanly.

## Tests

Add CLI/module tests for:

- `--auto` and `--round` are mutually exclusive.
- exactly one of `--auto` or `--round` is required.
- wrong current year fails before capture/status fetch.
- `--source` rejects non-`cartola_api` at parse time.
- `--auto` resolves active round from market status.
- explicit `--round` fails when the deadline/status payload reports `status_mercado != 1`.
- explicit `--round` passes that round to `capture_cartola_snapshot(...)`.
- capture success triggers generation.
- generation receives `captured_at` from the just-returned capture result.
- `--force-generate` maps only to `generate_strict_fixture(force=True)`.
- existing strict fixture plus no `--force-generate` returns clean error, keeps the raw snapshot, and reports the snapshot path.
- terminal success output includes snapshot directory, strict fixture path, manifest path, `captured_at_utc`, and `deadline_at_utc`.
- printed paths are project-root-relative.
- printed UTC timestamps end in `Z`.

## Acceptance Criteria

- A user can run one command before market lock to capture strict fixture evidence and generate a strict fixture CSV/manifest for the active round.
- The generated manifest proves the canonical strict fixture came from the snapshot captured by that command.
- Raw snapshot evidence is preserved when generation fails.
- The command does not affect live recommendation output or model behavior.
- Full repo checks pass with:

  ```bash
  uv run --frozen scripts/pyrepo-check --all
  ```

## Future Work

After this command has been used successfully for several live rounds:

1. Add optional strict fixture capture/generation to the live round workflow as an explicit mode, not a default side effect.
2. Integrate strict fixture context into live recommendations.
3. Add automation or reminders only after the manual command has proven reliable.
