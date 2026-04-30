# Matchup Fixture Coverage Audit Design

## Goal

Create a dedicated audit command that proves whether seasons can support leakage-safe home/away and opponent matchup features before any model, optimizer, or live recommendation changes are made.

The command answers one gate question:

```text
Do the selected seasons have exactly one usable fixture context row for every Cartola club-round that must receive matchup features?
```

Primary usage:

```bash
uv run --frozen python scripts/audit_matchup_fixture_coverage.py \
  --seasons 2023,2024,2025 \
  --current-year 2026
```

## Non-Goals

- Do not add `matchup_context_mode`.
- Do not add new model features.
- Do not change RandomForest settings.
- Do not change optimizer behavior.
- Do not change live or replay recommendation behavior.
- Do not import or fetch missing fixtures in this command.
- Do not claim strict no-leakage provenance for reconstructed historical fixture files.

## Scope

The audit reads existing local season data and existing local fixture data. It produces reports under:

```text
data/08_reporting/fixtures/matchup_fixture_coverage.csv
data/08_reporting/fixtures/matchup_fixture_coverage.json
```

It checks each requested season independently. Historical complete seasons are comparable only when all fixture coverage checks pass. A current-year partial season is allowed as `partial_current`, but it is not comparable.

## Fixture Sources

The audit supports two local fixture source classes:

- `strict`: canonical strict fixture files under `data/01_raw/fixtures_strict/{season}/partidas-{round}.csv`, validated against their `.manifest.json`.
- `exploratory`: canonical reconstructed fixture files under `data/01_raw/fixtures/{season}/partidas-{round}.csv`.

Source priority for a round is:

1. Use strict fixture if a strict CSV exists and its manifest validates.
2. Otherwise use exploratory fixture if it exists.
3. Otherwise mark the round missing fixture source.

Comparability means the selected local fixture source fully covers the season's Cartola club-round keys. It does not mean the source is strict live provenance unless the source is `strict`.

## Club-Round Universe

For each season and round, the expected club set is built from normalized Cartola season data:

```text
played clubs = clubs with at least one row where rodada == N and entrou_em_campo == True
```

This matches the existing fixture alignment logic and avoids forcing fixture coverage for postponed/non-Cartola matches that were not represented by players entering the field.

If a round has no played clubs, the round is auditable but contributes zero expected club-round keys. Historical seasons should still expose such rounds in the report.

## Fixture Context Contract

For every selected fixture match:

```text
home club row:
  rodada = N
  id_clube = id_clube_home
  opponent_id_clube = id_clube_away
  is_home = 1

away club row:
  rodada = N
  id_clube = id_clube_away
  opponent_id_clube = id_clube_home
  is_home = 0
```

Each context row must include:

- `season`
- `rodada`
- `id_clube`
- `opponent_id_clube`
- `is_home`
- `fixture_source`
- `source_file`
- `source_manifest`
- `source_sha256`
- `source_manifest_sha256`

`source_manifest` and `source_manifest_sha256` are null for exploratory fixtures.

The fixture row may provide club IDs only. Club names are optional in raw fixture files today, so the audit report should expose `opponent_nome_clube` when it can be resolved from Cartola season data, and null otherwise. Missing opponent names must not fail coverage when IDs are valid.

## Required Checks

For each `season + rodada + id_clube` expected key:

- exactly one fixture context row must exist;
- the context row must have a non-null `opponent_id_clube`;
- `opponent_id_clube` must not equal `id_clube`;
- `is_home` must be exactly `0` or `1`;
- `fixture_source` must be `strict` or `exploratory`;
- `source_file` must identify an existing fixture CSV;
- strict rows must include an existing manifest path and valid manifest hash;
- exploratory rows must record null manifest fields.

Season failure conditions:

- missing context rows;
- duplicate context rows;
- extra context rows for clubs not in the expected played club set;
- invalid strict manifest;
- malformed fixture file;
- fixture source file missing for any round with expected played clubs;
- historical season not complete over expected rounds `1..38`.

## Season Classification

Default config:

- `expected_complete_rounds = 38`
- `complete_round_threshold = 38`

Classification:

- `complete_historical`: requested season has exact rounds `1..38` and is not a partial current season.
- `partial_current`: season equals `current_year` and max detected round is below `complete_round_threshold`.
- `irregular_historical`: season is not partial current and does not have exact rounds `1..38`.

Only `complete_historical` seasons can be comparable. A complete season is comparable only when fixture coverage status is `ok`.

## Report Contract

CSV is one row per season with stable columns:

```text
season
season_status
metrics_comparable
fixture_status
round_file_count
min_round
max_round
detected_rounds
expected_club_round_count
fixture_context_row_count
missing_context_count
duplicate_context_count
extra_context_count
strict_context_count
exploratory_context_count
fixture_sources
error_stage
error_type
error_message
notes
```

JSON includes:

- `generated_at_utc`
- `project_root`
- `config`
- `decision`
- `seasons`

Each season object includes the CSV summary fields plus:

- `rounds`: per-round counts and source summary;
- `missing_context_keys`;
- `duplicate_context_keys`;
- `extra_context_keys`;
- `source_files`;
- `source_manifests`;
- `error_detail` with traceback for unexpected per-season failures.

The `decision` object evaluates the requested seasons:

```text
ready_for_matchup_context:
  all of 2023, 2024, 2025 are complete_historical and comparable

exploratory_only:
  2025 is complete_historical and comparable, but not all core seasons 2023, 2024, and 2025 are comparable

coverage_blocked:
  2025 is not comparable, or no requested complete historical season is comparable
```

This decision is advisory. The command exits `0` when reports are written successfully, even if seasons fail coverage. Invalid CLI arguments or unrecoverable report-writing failures still return non-zero through normal Python/argparse behavior.

## CLI Contract

Arguments:

- `--seasons`: comma-separated positive integer list, default `2023,2024,2025`.
- `--current-year`: optional positive integer. If omitted, resolve from UTC runtime year.
- `--project-root`: optional path, default `Path(".")`.
- `--output-root`: optional path, default `data/08_reporting/fixtures`.
- `--expected-complete-rounds`: optional positive integer, default `38`.
- `--complete-round-threshold`: optional positive integer, default `38`.

Output on success:

```text
Matchup fixture coverage audit complete
CSV: data/08_reporting/fixtures/matchup_fixture_coverage.csv
JSON: data/08_reporting/fixtures/matchup_fixture_coverage.json
Decision: ready_for_matchup_context
```

## Acceptance Criteria

- The command writes both reports at the expected paths.
- The report records missing, duplicate, and extra fixture context keys.
- Historical seasons are not comparable unless complete and coverage-clean.
- Current partial seasons are classified as `partial_current` and not comparable.
- Strict fixture rows validate manifest integrity before they count.
- Exploratory fixture rows clearly identify their source and null manifest fields.
- Existing model, optimizer, backtest, and recommendation behavior remains unchanged.

## Next Gate

If `2023`, `2024`, and `2025` are all `complete_historical` and comparable, implement:

```text
matchup_context_mode=cartola_matchup_v1
```

If only `2025` passes, matchup context remains exploratory only.

If coverage is weak, import or fix fixture coverage before feature work.
