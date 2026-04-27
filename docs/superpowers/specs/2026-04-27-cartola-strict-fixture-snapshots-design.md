# Cartola Strict Fixture Snapshot Design

## Purpose

Build a production-forward, no-leakage fixture data path for Cartola backtesting and future live use.

The current 2025 fixture files under `data/01_raw/fixtures/2025/` are useful exploratory reconstruction data, but they are not strict no-leakage data because they were aligned using same-round `entrou_em_campo` outcomes. This milestone does not try to prove 2025 retroactively. It defines and implements the durable rule for future fixture features:

> Fixture features may only be strict when they come from timestamped pre-lock snapshots whose provenance is validated before the backtest uses them.

Historical 2025 provenance audit is a separate follow-up milestone.

## Scope

In scope:

- Add strict fixture snapshot storage.
- Add strict canonical fixture storage with sidecar manifests.
- Capture Cartola API fixture and deadline payload snapshots before lock.
- Generate strict canonical `partidas-{round}.csv` files only from raw snapshots and club mapping.
- Add manifest validation that proves source, hashes, identity, and pre-deadline timing.
- Add backtest modes: `none`, `exploratory`, and `strict`.
- Make `fixture_mode="none"` the safe default.
- Add required `run_metadata.json` output for backtests.
- Preserve current 2025 reconstructed fixtures as explicitly exploratory.

Out of scope:

- Proving 2025 fixtures are strict.
- Backfilling 2025 from archived pre-lock evidence.
- Generic multi-source adapter architecture.
- Odds, Elo, opponent defensive weakness, or new model features.
- Live lineup/news snapshots.

## Fixture Modes

Backtest config gets:

```python
fixture_mode: Literal["none", "exploratory", "strict"] = "none"
strict_alignment_policy: Literal["fail", "exclude_round"] = "fail"
```

Modes:

- `none`: do not load fixture files. This is the default. The feature builder keeps schema-compatible neutral fixture columns (`is_home=0`, `opponent_club_points_roll3=global_club_points_prior`) so existing models still receive a stable feature matrix, but those columns carry no fixture signal.
- `exploratory`: load reconstructed fixtures from `data/01_raw/fixtures/{season}/partidas-*.csv`. Users must opt in explicitly. The CLI must print a warning and `run_metadata.json` must include a warning that exploratory fixtures are not strict no-leakage data.
- `strict`: load only manifest-validated fixtures from `data/01_raw/fixtures_strict/{season}/`.

Strict mode is the production-grade path. Exploratory mode is a research path.

## Strict Manifest Contract

A canonical fixture file is strict-valid only when it has a sidecar manifest that proves:

- the raw fixture snapshot exists and its hash matches
- the raw deadline snapshot exists and its hash matches
- the capture metadata exists and its hash matches
- the club mapping exists and its hash matches
- the canonical fixture CSV hash matches
- timestamps parse as timezone-aware datetimes with UTC offset `+00:00` or `Z`; naive datetimes and non-UTC offsets are invalid
- `captured_at_utc < deadline_at_utc`
- identity fields match the requested load context
- all manifest paths resolve under `project_root` after following symlinks

Equality is invalid. If `captured_at_utc == deadline_at_utc`, the fixture is not strict-valid.

Required manifest schema:

```json
{
  "mode": "strict",
  "season": 2026,
  "rodada": 12,
  "source": "cartola_api",
  "capture_metadata_path": "data/01_raw/fixtures_snapshots/2026/rodada-12/captured_at=2026-06-01T18-00-00Z/capture.json",
  "capture_metadata_sha256": "...",
  "fixture_snapshot_path": "data/01_raw/fixtures_snapshots/2026/rodada-12/captured_at=2026-06-01T18-00-00Z/fixtures.json",
  "fixture_snapshot_sha256": "...",
  "deadline_snapshot_path": "data/01_raw/fixtures_snapshots/2026/rodada-12/captured_at=2026-06-01T18-00-00Z/deadline.json",
  "deadline_snapshot_sha256": "...",
  "captured_at_utc": "2026-06-01T18:00:00Z",
  "deadline_at_utc": "2026-06-01T21:59:00Z",
  "deadline_source": "cartola_api_market_status",
  "fixture_endpoint": "https://api.cartola.globo.com/partidas/12",
  "fixture_final_url": "https://api.cartola.globo.com/partidas/12",
  "deadline_endpoint": "https://api.cartola.globo.com/mercado/status",
  "deadline_final_url": "https://api.cartola.globo.com/mercado/status",
  "generator_version": "fixture_snapshot_v1",
  "club_mapping_path": "data/01_raw/fixtures/club_mapping.csv",
  "club_mapping_sha256": "...",
  "canonical_fixture_path": "data/01_raw/fixtures_strict/2026/partidas-12.csv",
  "canonical_fixture_sha256": "..."
}
```

Identity validation must verify:

- manifest `season` matches the requested season
- manifest `rodada` matches the fixture filename and requested round
- manifest `source` matches the requested source
- manifest `canonical_fixture_path` points to the file being loaded
- each manifest path is resolved with symlinks followed, and the resolved absolute path must be equal to or inside the resolved absolute `project_root`

`captured_at_utc` is the local system timestamp, in UTC, taken after both fixture and deadline response bodies are fully received and before or while the raw snapshot files are written. It must not come from the remote source payload.

## Storage Layout

```text
data/01_raw/fixtures_snapshots/
  2026/
    rodada-12/
      captured_at=2026-06-01T18-00-00Z/
        capture.json
        fixtures.json
        deadline.json

data/01_raw/fixtures_strict/
  2026/
    partidas-12.csv
    partidas-12.manifest.json

data/01_raw/fixtures/
  2025/
    partidas-12.csv
```

`fixtures_snapshots` stores raw evidence. Snapshot directories are append-only and atomically written:

1. create `.tmp-{uuid}` under the round snapshot directory
2. write `capture.json`, `fixtures.json`, and `deadline.json`
3. close and flush files
4. rename `.tmp-{uuid}` to `captured_at=...`

A failed capture must not leave a `captured_at=...` directory. Stale `.tmp-*` directories may be cleaned or ignored, but they are never valid snapshots.

`capture.json` records local capture metadata:

```json
{
  "capture_started_at_utc": "2026-06-01T17:59:58Z",
  "captured_at_utc": "2026-06-01T18:00:00Z",
  "fixture_http_date_utc": "2026-06-01T18:00:00Z",
  "deadline_http_date_utc": "2026-06-01T18:00:00Z",
  "clock_skew_tolerance_seconds": 300,
  "max_observed_clock_skew_seconds": 1.2,
  "source": "cartola_api",
  "season": 2026,
  "rodada": 12,
  "capture_version": "fixture_capture_v1"
}
```

`fixtures_strict` stores derived canonical fixtures plus manifests. These files are hash-verified before use. Regeneration must either write to a temp path and atomically replace the file, or require `--force` if the target canonical file or manifest already exists.

## Data Flow

For v1, `cartola_api` is the only supported strict source.

1. Capture command fetches the fixture payload and deadline payload before lock.
2. Capture command writes `capture.json`, `fixtures.json`, and `deadline.json` atomically.
3. Generator selects the latest strict-valid snapshot before deadline, unless an explicit `captured_at` is provided.
4. Generator reads only snapshot files plus club mapping.
5. Generator writes canonical `partidas-{round}.csv` and `partidas-{round}.manifest.json` atomically.
6. Strict loader validates manifest fields, hashes, identity, and timing before returning fixtures.
7. Backtest uses strict loader only when `fixture_mode="strict"`.

Multiple snapshots rule:

```text
Use the latest strict-valid snapshot before deadline, unless the user passes an explicit captured_at value.
```

Strict-valid snapshot means:

- `capture.json`, `fixtures.json`, and `deadline.json` exist in the same capture directory
- timestamps parse as timezone-aware datetimes with UTC offset `+00:00` or `Z`
- `captured_at_utc < deadline_at_utc`
- source payload can be mapped to Cartola club IDs

If multiple snapshot directories parse to the same `captured_at_utc`, snapshot discovery fails and the user must provide an explicit capture path or delete the duplicate. A capture command must fail if its target `captured_at=...` directory already exists.

## Cartola API Source Contract

V1 strict capture supports only `source="cartola_api"`.

Fixture endpoint:

```text
GET https://api.cartola.globo.com/partidas/{round_number}
```

Legacy `https://api.cartolafc.globo.com/partidas/{round_number}` may redirect to this endpoint, but the manifest should record the final resolved URL.

Required fixture response fields:

- top-level `rodada`
- top-level `partidas`
- per-partida `clube_casa_id`
- per-partida `clube_visitante_id`
- per-partida `partida_data`
- per-partida `timestamp`
- per-partida `valida`

Fixture extraction rule:

- top-level `rodada` must equal requested `round_number`
- each canonical fixture row is generated from one `partidas[]` item where `valida is true`
- `id_clube_home = clube_casa_id`
- `id_clube_away = clube_visitante_id`
- `data = date(partida_data)`
- score/result fields such as `placar_oficial_mandante` and `placar_oficial_visitante` must be ignored

Deadline endpoint:

```text
GET https://api.cartola.globo.com/mercado/status
```

Legacy `https://api.cartolafc.globo.com/mercado/status` may redirect to this endpoint, but the manifest should record the final resolved URL.

Required deadline response fields:

- `temporada`
- `rodada_atual`
- `status_mercado`
- `fechamento.timestamp`
- `fechamento.ano`
- `fechamento.mes`
- `fechamento.dia`
- `fechamento.hora`
- `fechamento.minuto`

Deadline extraction rule:

- `temporada` must equal requested `season`
- `rodada_atual` must equal requested `round_number`
- `deadline_at_utc` is parsed from `fechamento.timestamp` as Unix epoch seconds
- the component fields under `fechamento` are retained as supporting evidence and may be cross-checked, but they are not the primary timestamp source
- if `fechamento.timestamp` is missing, non-numeric, or unparsable, capture fails

The capture command fetches both endpoints, records local UTC time after both response bodies are fully received, writes both payloads into the same `.tmp-{uuid}` directory, and fails if that timestamp is not strictly before the parsed deadline.

Tests must use frozen Cartola fixture and deadline payloads so endpoint parsing is deterministic and does not depend on live API availability.

## Clock Evidence

Strict capture relies on local system time plus HTTP response evidence.

The HTTP `Date` header from both Cartola responses is required for strict capture. Header timestamps must parse as timezone-aware datetimes with UTC offset `+00:00` or `Z`; naive datetimes and non-UTC offsets are invalid.

Strict capture fails when:

- either response is missing an HTTP `Date` header
- either HTTP `Date` header cannot be parsed as UTC
- `abs(captured_at_utc - fixture_http_date_utc) > clock_skew_tolerance_seconds`
- `abs(captured_at_utc - deadline_http_date_utc) > clock_skew_tolerance_seconds`

Default tolerance is 300 seconds. If clock skew is unknown or exceeds tolerance, the snapshot cannot be strict-valid.

## Structural No-Outcome Rule

The strict generator must not inspect Cartola same-round outcomes.

Strict generator API accepts only:

```python
project_root
season
round_number
source
captured_at  # optional
force        # optional
```

The strict generator must not accept `season_df`.

The strict generator module must not import:

- `load_season_data`
- `played_club_set`
- `run_backtest`
- backtest result modules
- metrics or diagnostics modules

The generator must not read:

- `season_df`
- `entrou_em_campo`
- `pontuacao`
- selected players
- optimizer outputs
- diagnostics
- any same-round outcome columns

This is the core code boundary that prevents recreating the current 2025 leakage pattern.

## CLI

New commands:

```bash
uv run python scripts/capture_fixture_snapshot.py --season 2026 --round 12 --source cartola_api
uv run python scripts/generate_strict_fixtures.py --season 2026 --round 12 --source cartola_api
```

The CLI may expose `--round` for user ergonomics, but internal Python APIs should use `round_number` to avoid shadowing the Python built-in.

Backtest CLI arguments:

```bash
--fixture-mode none|exploratory|strict
--strict-alignment-policy fail|exclude_round
```

Examples:

```bash
# default: no fixture features
uv run python -m cartola.backtesting.cli --season 2025

# exploratory 2025 reconstruction
uv run python -m cartola.backtesting.cli --season 2025 --fixture-mode exploratory

# strict production path
uv run python -m cartola.backtesting.cli --season 2026 --fixture-mode strict

# strict with visible round exclusion
uv run python -m cartola.backtesting.cli --season 2026 --fixture-mode strict --strict-alignment-policy exclude_round
```

## Strict Backtest Validation Scope

For a backtest with `start_round=N` and `max_round=M`, strict mode must validate fixture provenance and alignment for every fixture-feature target round that could enter the model:

```text
rounds 1..M
```

Reason: `build_training_frame` creates historical training examples by calling `build_prediction_frame` for prior rounds. If fixture features are enabled, invalid fixture context in a historical target round can enter model training.

If implementation later optimizes training to start from a later minimum round, the validation scope may be narrowed explicitly. V1 keeps the rule simple: validate all rounds present in the season data up to `max_round`.

If a strict canonical fixture file or manifest is missing for any round in `1..M`, the round is invalid. With `strict_alignment_policy="fail"`, the backtest fails before the model loop. With `strict_alignment_policy="exclude_round"`, the missing round is removed from the season dataframe before training and prediction and is recorded in `run_metadata.json`.

## Alignment Policy

Strict fixtures are not rewritten from post-round outcomes.

After a round closes, the pipeline may compare strict fixture files to actual Cartola data for diagnostics:

- missing clubs
- extra clubs
- postponed matches
- clubs with no entrants

That diagnostic report must not modify strict fixture files or manifests.

Policy:

```text
strict_alignment_policy = fail | exclude_round
```

Default: `fail`.

If `fail`, any invalid alignment fails the backtest.

If `exclude_round`, invalid rounds are removed from the season dataframe before any training or prediction frames are built. This simple rule avoids mixed feature histories. Excluded rounds must be visible in `run_metadata.json` and should be represented in reporting with a status such as `FixtureAlignmentExcluded`.

## Required Backtest Metadata

Every backtest run writes:

```text
data/08_reporting/backtests/{season}/run_metadata.json
```

Fields:

```json
{
  "season": 2026,
  "start_round": 5,
  "max_round": 38,
  "fixture_mode": "strict",
  "strict_alignment_policy": "fail",
  "fixture_source_directory": "data/01_raw/fixtures_strict/2026",
  "fixture_manifest_paths": ["..."],
  "fixture_manifest_sha256": {
    "partidas-12.manifest.json": "..."
  },
  "generator_versions": ["fixture_snapshot_v1"],
  "excluded_rounds": [],
  "warnings": []
}
```

Strict metadata must record exact fixture manifest paths used for every validated round, not only rounds with candidate players.

For `fixture_mode="none"`, metadata uses empty fixture provenance fields:

```json
{
  "fixture_mode": "none",
  "fixture_source_directory": null,
  "fixture_manifest_paths": [],
  "fixture_manifest_sha256": {},
  "generator_versions": [],
  "excluded_rounds": [],
  "warnings": []
}
```

For exploratory mode, `warnings` must include a clear message that fixture files may be reconstructed from post-round evidence and are not strict no-leakage data. The CLI must also print the warning to stdout or stderr.

Example exploratory metadata:

```json
{
  "fixture_mode": "exploratory",
  "fixture_source_directory": "data/01_raw/fixtures/2025",
  "fixture_manifest_paths": [],
  "fixture_manifest_sha256": {},
  "generator_versions": [],
  "excluded_rounds": [],
  "warnings": [
    "Exploratory fixture mode is not strict no-leakage data; files may be reconstructed from post-round evidence."
  ]
}
```

## Error Handling

- Capture fails if fixture payload or deadline payload cannot be fetched.
- Capture fails if either Cartola payload is missing required response fields.
- Capture fails if the deadline payload `temporada` or `rodada_atual` does not match the requested season and round.
- Capture fails if `deadline_at_utc` cannot be extracted from the Cartola deadline payload.
- Capture fails if the HTTP `Date` header is missing, non-UTC, or outside the clock-skew tolerance for either response.
- Capture fails if local `captured_at_utc` is not before `deadline_at_utc`.
- Capture failure before completion leaves no valid `captured_at=...` directory.
- Generation fails if no strict-valid snapshot exists before deadline.
- Generation fails if multiple snapshot directories have the same parsed `captured_at_utc`.
- Generation fails if `fixtures.json`, `deadline.json`, and `capture.json` do not come from the same capture directory.
- Generation fails if source teams cannot be mapped to Cartola IDs.
- Generation refuses overwrite unless `--force` is passed.
- Strict loading fails if any required manifest field is missing.
- Strict loading fails if any manifest hash mismatches the file content.
- Strict loading fails if timestamps are not timezone-aware UTC values.
- Strict loading fails if `mode != "strict"`.
- Strict loading fails if `captured_at_utc >= deadline_at_utc`.
- Strict loading fails if manifest identity fields do not match the requested season, round number, source, fixture filename, or `project_root`.
- Strict loading fails if any manifest path resolves outside `project_root` after following symlinks.
- Strict loading treats missing canonical fixture files or manifests in the required validation range as invalid strict rounds.
- Strict backtest fails if fixture alignment is invalid and `strict_alignment_policy="fail"`.
- Strict backtest excludes invalid rounds before frame construction if `strict_alignment_policy="exclude_round"`.

## Testing

Required tests:

1. Snapshot capture writes `capture.json`, `fixtures.json`, and `deadline.json` into `.tmp-{uuid}` and renames to `captured_at=...` on success.
2. Capture failure before completion leaves no valid `captured_at=...` directory.
3. Stale `.tmp-*` directories are ignored by strict snapshot discovery.
4. Frozen Cartola fixture payload parsing extracts only `valida is true` partidas into canonical rows.
5. Frozen Cartola deadline payload parsing extracts `deadline_at_utc` from `fechamento.timestamp`.
6. Capture rejects fixture payloads whose top-level `rodada` does not match `round_number`.
7. Capture rejects deadline payloads whose `temporada` or `rodada_atual` does not match the request.
8. Capture rejects missing HTTP `Date` headers.
9. Capture rejects HTTP `Date` headers with clock skew above tolerance.
10. Manifest validation rejects missing fields.
11. Manifest validation rejects bad fixture snapshot hash.
12. Manifest validation rejects bad deadline snapshot hash.
13. Manifest validation rejects bad capture metadata hash.
14. Manifest validation rejects non-UTC timestamps.
15. Manifest validation rejects equality at deadline.
16. Manifest validation rejects post-deadline capture.
17. Manifest validation rejects `mode != "strict"`.
18. Manifest validation rejects season mismatch.
19. Manifest validation rejects rodada mismatch.
20. Manifest validation rejects source mismatch.
21. Manifest validation rejects `canonical_fixture_path` pointing somewhere other than the loaded file.
22. Manifest validation rejects paths outside `project_root`, including symlink traversal.
23. Generator chooses latest strict-valid snapshot before deadline.
24. Generator rejects duplicate parsed `captured_at_utc` snapshot ties unless an explicit capture is provided.
25. Generator supports explicit `captured_at`.
26. Generator refuses if `capture.json`, `fixtures.json`, and `deadline.json` are not from the same capture directory.
27. Generator refuses overwrite unless `--force`.
28. Generator module import and signature checks confirm it does not depend on season data or backtest outcome modules.
29. Strict loader rejects canonical fixtures without manifest.
30. Strict loader rejects edited canonical CSV after manifest hash mismatch.
31. Strict loader rejects missing strict fixture files or manifests in the required validation range.
32. Runner default `fixture_mode="none"` does not load exploratory files accidentally and emits schema-compatible neutral fixture columns.
33. `fixture_mode="exploratory"` loads current 2025 fixtures only when explicitly requested and writes warning to stdout or stderr and `run_metadata.json`.
34. `run_metadata.json` for `none` uses empty fixture provenance fields.
35. `fixture_mode="strict"` validates all fixture-feature target rounds used by training and prediction.
36. Strict backtest metadata records exact fixture manifest paths used for every validated round.
37. `strict_alignment_policy="fail"` raises on misalignment.
38. `strict_alignment_policy="exclude_round"` removes invalid rounds from the season dataframe before training and prediction and records them in metadata.
39. Existing 2025 exploratory backtest remains runnable only when explicitly requested.
40. Full quality gate remains `uv run --frozen scripts/pyrepo-check --all`.

## Follow-Up Milestone B: Historical 2025 Audit

After Milestone A, investigate whether strict 2025 fixture provenance exists.

Possible outcomes:

- If archived pre-lock fixture and deadline evidence exists, backfill 2025 into `fixtures_snapshots` and `fixtures_strict`.
- If it does not exist, keep 2025 under exploratory labeling.

Milestone B must not block the production-forward strict capture path.
