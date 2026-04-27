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

- `none`: do not load fixture features. This is the default.
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
- timestamps are timezone-aware UTC values
- `captured_at_utc < deadline_at_utc`
- identity fields match the requested load context
- all manifest paths stay under `project_root`

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
- manifest paths resolve under `project_root`

`captured_at_utc` is the local system timestamp, in UTC, taken after the response body is fully received and before or while the raw snapshot is written. It must not come from the remote source payload.

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
  "captured_at_utc": "2026-06-01T18:00:00Z",
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
- timestamps are parseable timezone-aware UTC values
- `captured_at_utc < deadline_at_utc`
- source payload can be mapped to Cartola club IDs

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

For exploratory mode, `warnings` must include a clear message that fixture files may be reconstructed from post-round evidence and are not strict no-leakage data. The CLI must also print the warning to stdout or stderr.

## Error Handling

- Capture fails if fixture payload or deadline payload cannot be fetched.
- Capture fails if `deadline_at_utc` cannot be extracted from the Cartola deadline payload.
- Capture fails if local `captured_at_utc` is not before `deadline_at_utc`.
- Capture failure before completion leaves no valid `captured_at=...` directory.
- Generation fails if no strict-valid snapshot exists before deadline.
- Generation fails if `fixtures.json`, `deadline.json`, and `capture.json` do not come from the same capture directory.
- Generation fails if source teams cannot be mapped to Cartola IDs.
- Generation refuses overwrite unless `--force` is passed.
- Strict loading fails if any required manifest field is missing.
- Strict loading fails if any manifest hash mismatches the file content.
- Strict loading fails if timestamps are not timezone-aware UTC values.
- Strict loading fails if `mode != "strict"`.
- Strict loading fails if `captured_at_utc >= deadline_at_utc`.
- Strict loading fails if manifest identity fields do not match the requested season, round, source, fixture filename, or `project_root`.
- Strict backtest fails if fixture alignment is invalid and `strict_alignment_policy="fail"`.
- Strict backtest excludes invalid rounds before frame construction if `strict_alignment_policy="exclude_round"`.

## Testing

Required tests:

1. Snapshot capture writes `capture.json`, `fixtures.json`, and `deadline.json` into `.tmp-{uuid}` and renames to `captured_at=...` on success.
2. Capture failure before completion leaves no valid `captured_at=...` directory.
3. Stale `.tmp-*` directories are ignored by strict snapshot discovery.
4. Manifest validation rejects missing fields.
5. Manifest validation rejects bad fixture snapshot hash.
6. Manifest validation rejects bad deadline snapshot hash.
7. Manifest validation rejects bad capture metadata hash.
8. Manifest validation rejects non-UTC timestamps.
9. Manifest validation rejects equality at deadline.
10. Manifest validation rejects post-deadline capture.
11. Manifest validation rejects `mode != "strict"`.
12. Manifest validation rejects season mismatch.
13. Manifest validation rejects rodada mismatch.
14. Manifest validation rejects source mismatch.
15. Manifest validation rejects `canonical_fixture_path` pointing somewhere other than the loaded file.
16. Manifest validation rejects paths outside `project_root`.
17. Generator chooses latest strict-valid snapshot before deadline.
18. Generator supports explicit `captured_at`.
19. Generator refuses if `capture.json`, `fixtures.json`, and `deadline.json` are not from the same capture directory.
20. Generator refuses overwrite unless `--force`.
21. Generator module import and signature checks confirm it does not depend on season data or backtest outcome modules.
22. Strict loader rejects canonical fixtures without manifest.
23. Strict loader rejects edited canonical CSV after manifest hash mismatch.
24. Runner default `fixture_mode="none"` does not load exploratory files accidentally.
25. `fixture_mode="exploratory"` loads current 2025 fixtures only when explicitly requested and writes warning to stdout or stderr and `run_metadata.json`.
26. `fixture_mode="strict"` validates all fixture-feature target rounds used by training and prediction.
27. Strict backtest metadata records exact fixture manifest paths used for every validated round.
28. `strict_alignment_policy="fail"` raises on misalignment.
29. `strict_alignment_policy="exclude_round"` removes invalid rounds from the season dataframe before training and prediction and records them in metadata.
30. Existing 2025 exploratory backtest remains runnable only when explicitly requested.
31. Full quality gate remains `uv run --frozen scripts/pyrepo-check --all`.

## Follow-Up Milestone B: Historical 2025 Audit

After Milestone A, investigate whether strict 2025 fixture provenance exists.

Possible outcomes:

- If archived pre-lock fixture and deadline evidence exists, backfill 2025 into `fixtures_snapshots` and `fixtures_strict`.
- If it does not exist, keep 2025 under exploratory labeling.

Milestone B must not block the production-forward strict capture path.
