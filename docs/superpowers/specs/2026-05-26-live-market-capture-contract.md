# Live Market Capture Contract

## Scope

This document defines the current pre-lock Cartola market capture contract used
by `scripts/capture_market_round.py` and `cartola.backtesting.market_capture`.

The capture artifact is the canonical raw evidence input for live recommendation
runs. Live recommendation orchestration must validate this artifact before squad
generation.

## Storage Layout

Default live capture files are stored under the project root:

- CSV: `data/01_raw/{season}/rodada-{target_round}.csv`
- metadata JSON: `data/01_raw/{season}/rodada-{target_round}.capture.json`

The metadata file must describe the CSV beside it. The `csv_path` value must
resolve to exactly the final CSV path for the same `{season}` and
`{target_round}`.

Capture writes must use a temporary directory under
`data/01_raw/{season}/.tmp-market-capture-*` and publish the CSV/metadata pair
atomically as a matched artifact set. Temporary capture directories must be
removed after successful or failed publication.

## Required Metadata Fields

The capture JSON must contain these fields:

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

`capture_version` must be `market_capture_v1`.

`status_endpoint` must be the Cartola market-status endpoint:
`https://api.cartola.globo.com/mercado/status`.

`market_endpoint` must be the Cartola market-athletes endpoint:
`https://api.cartola.globo.com/atletas/mercado`.

`captured_at_utc` must be UTC ISO-8601 text ending in `Z`.

`status_mercado` must be `1` for a valid live pre-lock capture.

## CSV Contract

The CSV must contain the player attributes required for live squad selection as
available from the Cartola market API.

The current raw output includes these required identity and selection columns:

- `atletas.rodada_id`
- `atletas.status_id`
- `atletas.posicao_id`
- `atletas.atleta_id`
- `atletas.apelido`
- `atletas.clube_id`
- `atletas.preco_num`
- `atletas.media_num`
- `atletas.jogos_num`

The raw output may include optional descriptive fields, valuation fields, and
scout columns when provided by the API.

All rows in a capture CSV must belong to the captured `target_round`.

## Integrity Checks

The capture metadata must record SHA-256 hashes for both source API responses
and the final CSV:

- `status_response_sha256`
- `market_response_sha256`
- `csv_sha256`

`csv_sha256` must equal the SHA-256 hash of the bytes in
`data/01_raw/{season}/rodada-{target_round}.csv`.

`load_valid_live_capture()` is the shared validator for live recommendation
consumers. It must reject a capture when:

- either the CSV or metadata file is missing;
- metadata is not JSON object data;
- `capture_version` does not match `market_capture_v1`;
- `season` or `target_round` does not match the requested capture;
- `csv_path` does not point to the expected final CSV;
- `csv_sha256` does not match the actual CSV bytes;
- `captured_at_utc` is not UTC ISO-8601 with trailing `Z`;
- `status_mercado` is not `1`;
- `deadline_timestamp` cannot be parsed when present.

## Overwrite Safety

`--force` may replace an existing live capture only when the existing CSV and
metadata validate as a previous valid live capture for the same season and
target round.

`--force` must not overwrite arbitrary historical, manual, partial, or
unverified raw files.

Without `--force`, capture publication must fail before replacing any existing
CSV or metadata file.
