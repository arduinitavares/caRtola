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
  --target-round 14
```

Auto-detected capture:

```bash
uv run --frozen python scripts/capture_market_round.py \
  --season 2026 \
  --auto
```

Rules:

- `--target-round` and `--auto` are mutually exclusive.
- `--auto` reads `mercado/status.rodada_atual` and uses that as the target round.
- The command refuses to overwrite `data/01_raw/{season}/rodada-{target_round}.csv` unless `--force` is passed.
- The command prints the written CSV path, market status, athlete count, and deadline timestamp when available.

## Data Sources

The command uses:

- `https://api.cartola.globo.com/mercado/status`
- `https://api.cartola.globo.com/atletas/mercado`

The target round comes from `mercado/status.rodada_atual`, not from `atletas[].rodada_id`.

The command requires `status_mercado == 1` for normal live capture. If the market is not open, it fails with a message that includes the reported `rodada_atual` and `status_mercado`.

## CSV Contract

The output CSV must be compatible with `load_season_data()` and the existing recommendation pipeline.

For every athlete row:

- `atletas.rodada_id` is set to the target round.
- `atletas.pontos_num` is set to `0.0`.
- `atletas.entrou_em_campo` is set to `False`.
- `atletas.variacao_num` is set to `0.0`, so `preco_pre_rodada == preco` after normalization.
- scout columns are present and set to `0`.
- current market fields such as athlete id, nickname, club id, club name, position, status, price, average, and games are preserved.

This creates a live candidate snapshot. It must not include finalized same-round outcome fields.

## Output Files

The command writes:

```text
data/01_raw/{season}/rodada-{target_round}.csv
data/01_raw/{season}/rodada-{target_round}.capture.json
```

The JSON metadata is operational, not strict provenance. It records:

- `season`
- `target_round`
- `captured_at_utc`
- `status_endpoint`
- `market_endpoint`
- `rodada_atual`
- `status_mercado`
- `deadline_timestamp` when present
- `athlete_count`
- `csv_path`

## Integration With Recommendation

After capture, the existing live command should work without additional flags:

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

`--auto` only captures the current open market round if the CSV is missing. It does not schedule itself, send notifications, or run recommendations in v1.

A later command can wrap capture plus recommendation, but this design keeps the base capture operation explicit and testable.

## Error Handling

The command fails when:

- the Cartola API response is not valid JSON;
- `rodada_atual` is missing or not a positive integer;
- `status_mercado != 1`;
- `--target-round` does not match `rodada_atual`;
- the market payload has no athlete rows;
- required athlete, club, position, or status payload fields are missing;
- the destination CSV already exists and `--force` was not passed.

## Tests

Add focused tests for:

- manual capture writes `rodada-{target_round}.csv` using `mercado/status.rodada_atual`;
- athlete-level stale `rodada_id` is replaced by the target round;
- points, variation, scouts, and `entrou_em_campo` are sanitized for live mode;
- existing CSV is not overwritten without `--force`;
- `--auto` captures the reported active round;
- market-closed status fails clearly;
- target-round mismatch fails clearly;
- written CSV loads through `load_season_data()` and can be used by `run_recommendation()` in live mode.

## Acceptance Criteria

- Running the capture command during an open market creates the next-round CSV.
- The resulting CSV can drive `scripts/recommend_squad.py --mode live`.
- The command does not overwrite existing raw round data unless explicitly forced.
- The old downloader is no longer the documented path for live recommendations.
