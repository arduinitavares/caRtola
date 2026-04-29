# Live Market Round Capture Design

## Problem

The live recommendation command needs a raw Cartola CSV for the next open market round, for example `data/01_raw/2026/rodada-14.csv`. The local 2026 data currently stops at round 13, so live recommendations for round 14 fail before prediction.

The old downloader is not a valid live-round capture primitive because it names the output file from athlete-level `rodada_id`. During the open market for round 14, Cartola's market status reports `rodada_atual=14`, while athlete rows can still carry `rodada_id=13`. For live play, the file must represent the active market round.

## Goal

Add a deterministic capture command that creates the missing current-round raw CSV from the open Cartola market, so the user can run live recommendations before lineup lock.

The initial scope is operational live gameplay only. It is not a strict historical provenance system and does not replace strict fixture snapshots.

## CLI

Manual capture:

```bash
uv run --frozen python scripts/capture_market_round.py \
  --season 2026 \
  --target-round 14 \
  --current-year 2026
```

Auto-detected capture:

```bash
uv run --frozen python scripts/capture_market_round.py \
  --season 2026 \
  --auto \
  --current-year 2026
```

Rules:

- `--target-round` and `--auto` are mutually exclusive.
- `--project-root` defaults to `.`.
- `--current-year` defaults to the runtime UTC year. Tests must pass it explicitly.
- The command refuses every capture where `season != current_year`.
- `--auto` reads `mercado/status.rodada_atual` and uses that as the target round.
- The command refuses to overwrite `data/01_raw/{season}/rodada-{target_round}.csv` unless `--force` is passed and the existing CSV is a previous live capture with a valid matching `.capture.json`.
- The command prints the written CSV path, market status, athlete count, deadline parse status, and deadline timestamp value or `null`.

## Data Sources

The command uses:

- `https://api.cartola.globo.com/mercado/status`
- `https://api.cartola.globo.com/atletas/mercado`

The target round comes from `mercado/status.rodada_atual`, not from `atletas[].rodada_id`.

The command requires `status_mercado == 1` for normal live capture. If the market is not open, it fails with a message that includes the reported `rodada_atual` and `status_mercado`.

HTTP behavior:

- Each request uses a finite timeout.
- Non-200 responses fail.
- Invalid JSON responses fail.
- The command records HTTP status codes, final URLs, response SHA-256 hashes, and endpoint URLs in metadata.
- Redirects are allowed only through the HTTP client default behavior; the final URL is recorded.

Required status payload fields:

- `rodada_atual`: positive integer.
- `status_mercado`: integer.
- `fechamento.timestamp`: optional integer Unix timestamp. If missing or invalid, metadata records `deadline_timestamp=null` and `deadline_parse_status` as `missing` or `invalid`; otherwise `deadline_parse_status="ok"`.

Required market payload fields:

- top-level `atletas`: non-empty list.
- top-level `clubes`: object keyed by club id.
- for every athlete row: `atleta_id`, `apelido`, `clube_id`, `posicao_id`, `status_id`, `preco_num`, `media_num`, and `jogos_num`.
- optional athlete fields preserved when present: `slug`, `nome`, `foto`, `apelido_abreviado`, and `minimo_para_valorizar`.
- each athlete `clube_id` must have a matching `clubes` entry with `id` and `nome`; missing club mappings fail before writing.

## CSV Contract

The output CSV must be compatible with `load_season_data()` and the existing recommendation pipeline.

The CSV must contain the loader-compatible columns listed in Appendix A. Missing required columns fail before publishing. Extra passthrough columns are allowed only when they are already present in the Cartola market payload and do not conflict with the sanitized live columns.

For every athlete row:

- `atletas.rodada_id` is set to the target round.
- `atletas.pontos_num` is set to `0.0`.
- `atletas.entrou_em_campo` is set to `False`.
- `atletas.variacao_num` is set to `0.0`, so `preco_pre_rodada == preco` after normalization.
- all scout columns in `DEFAULT_SCOUT_COLUMNS` are present and set to `0`.
- current market fields required by Appendix A are preserved.

This creates a live candidate snapshot. It must not include finalized same-round outcome fields.

The command overwrites `atletas.pontos_num`, `atletas.entrou_em_campo`, `atletas.variacao_num`, and all scout columns regardless of values returned by the API.

## Output Files

The command writes:

```text
data/01_raw/{season}/rodada-{target_round}.csv
data/01_raw/{season}/rodada-{target_round}.capture.json
```

The JSON metadata is operational, not strict provenance. It records:

- `capture_version`
- `season`
- `current_year`
- `target_round`
- `captured_at_utc`
- `status_endpoint`
- `status_final_url`
- `status_http_status`
- `status_response_sha256`
- `market_endpoint`
- `market_final_url`
- `market_http_status`
- `market_response_sha256`
- `rodada_atual`
- `status_mercado`
- `deadline_timestamp`
- `deadline_parse_status`
- `athlete_count`
- `csv_path`
- `csv_sha256`

Metadata is useful for operational debugging. It is not strict no-leakage proof and is not used to make historical evaluation claims.

## Publication Safety

The command publishes into `data/01_raw/{season}/`, which is canonical model input. It must therefore use validate-before-publish semantics.

Publication rules:

- Build the CSV and metadata in a temporary location under the same season directory, for example `.tmp-market-capture-{uuid}`.
- Validate the temporary CSV with the same loader path used by recommendation, at minimum `load_round_file(temp_csv)`.
- Validate that every loaded row has `rodada == target_round`.
- Validate that the temporary metadata references the intended final CSV path and contains the computed CSV SHA-256.
- A final `rodada-{target_round}.csv` must never become visible without a matching final `.capture.json`.
- For first-time publication, write and validate both temp files, rename metadata into place, then rename the CSV into place.
- If the command crashes before the final CSV rename, recommendation must not see a partial final CSV.
- Stale `.tmp-market-capture-*` directories are ignored by loaders and may be cleaned by a later run.

Overwrite rules:

- Without `--force`, any existing final CSV or final metadata fails the command.
- With `--force`, the existing final CSV may be replaced only when a matching `.capture.json` exists and validates:
  - metadata `capture_version` matches this capture tool family;
  - metadata `season` and `target_round` match the requested capture;
  - metadata `csv_path` points to the existing final CSV;
  - metadata `csv_sha256` matches the existing final CSV.
- `--force` must refuse to overwrite raw files that do not have valid matching live-capture metadata.
- For forced replacement, move the previous final CSV and metadata out of the final names before publishing the new pair. If publication fails, restore the previous pair when possible; otherwise leave no final CSV rather than a mismatched or partial CSV.

## Integration With Recommendation

After capture, the existing live command works without additional flags:

```bash
uv run --frozen python scripts/recommend_squad.py \
  --season 2026 \
  --target-round 14 \
  --mode live \
  --budget 100 \
  --footystats-mode ppg \
  --current-year 2026
```

The recommendation command remains prediction-only. It does not silently fetch live data.

## Auto Mode Boundary

`--auto` captures the current open market round reported by `mercado/status`.

Rules:

- `--auto` without `--force` follows the same overwrite rules as manual capture: any existing final CSV or final metadata fails the command.
- `--auto --force` fetches a fresh market response and may replace only a previous valid live capture with compatible metadata.
- `--auto` does not schedule itself, send notifications, or run recommendations in v1.

A later command can wrap capture plus recommendation, but this design keeps the base capture operation explicit and testable.

## Error Handling

The command fails when:

- the Cartola API response is not valid JSON;
- either request returns a non-200 status;
- `rodada_atual` is missing or not a positive integer;
- `status_mercado != 1`;
- `season != current_year`;
- `--target-round` does not match `rodada_atual`;
- the market payload has no athlete rows;
- required athlete, club, position, or status payload fields are missing;
- an athlete `clube_id` has no matching club payload;
- the generated CSV fails `load_round_file()`;
- the destination CSV already exists and `--force` was not passed;
- `--force` is passed for a destination CSV that is not a previous valid live capture.

## Tests

Add focused tests for:

- manual capture writes `rodada-{target_round}.csv` using `mercado/status.rodada_atual`;
- wrong-season capture is refused;
- athlete-level stale `rodada_id` is replaced by the target round;
- points, variation, scouts, and `entrou_em_campo` are sanitized for live mode;
- existing CSV is not overwritten without `--force`;
- `--force` refuses a raw CSV with no matching capture metadata;
- `--force` allows replacement of a previous valid live capture;
- `--auto` captures the reported active round;
- market-closed status fails clearly;
- target-round mismatch fails clearly;
- club mapping gaps fail before writing;
- interrupted or failed temporary writes do not leave a final visible CSV;
- final CSV is not published without metadata;
- written CSV loads through `load_season_data()` and can be used by `run_recommendation()` in live mode.

## Acceptance Criteria

- Running the capture command during an open market creates the next-round CSV.
- The resulting CSV can drive `scripts/recommend_squad.py --mode live`.
- The command does not overwrite existing raw round data unless explicitly forced.
- The old downloader is no longer the documented path for live recommendations.

## Appendix A: Required Output CSV Columns

The generated CSV must include these loader-compatible raw columns:

- `atletas.rodada_id`
- `atletas.status_id`
- `atletas.posicao_id`
- `atletas.atleta_id`
- `atletas.apelido`
- `atletas.slug`
- `atletas.clube_id`
- `atletas.clube.id.full.name`
- `atletas.preco_num`
- `atletas.pontos_num`
- `atletas.media_num`
- `atletas.jogos_num`
- `atletas.variacao_num`
- `atletas.entrou_em_campo`
- `atletas.minimo_para_valorizar`
- `atletas.apelido_abreviado`
- `atletas.nome`
- `atletas.foto`

The generated CSV must also include every scout column from `DEFAULT_SCOUT_COLUMNS`:

- `G`
- `A`
- `DS`
- `SG`
- `CA`
- `FC`
- `FS`
- `FF`
- `FD`
- `FT`
- `I`
- `GS`
- `DE`
- `DP`
- `V`
- `CV`
- `PP`
- `PS`
- `PC`
- `GC`
