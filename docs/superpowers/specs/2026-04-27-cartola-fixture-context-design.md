# Cartola Fixture Context Feature Design

Date: 2026-04-27

## Purpose

Add fixture context to the offline Cartola backtest so the model can distinguish the same player in different match contexts. The previous player-only feature milestone improved Random Forest squad performance, but missed the target thresholds:

- Random Forest average points: `57.7541`
- Baseline average points: `56.3197`
- Random Forest player-level `r2`: `-0.0072517923`

This makes match context the next feature source to test, especially home/away status and opponent strength.

## Scope

In scope:

- Add a committed fixture data layout under `data/01_raw/fixtures/`.
- Add a committed club-name mapping file for external fixture sources.
- Import 2025 Brasileirão fixture data into canonical per-round Cartola fixture CSVs.
- Validate fixture round alignment against Cartola player round data.
- Add a fixture loader for the backtesting pipeline.
- Add two leakage-safe fixture features:
  - `is_home`
  - `opponent_club_points_roll3`
- Rerun the 2025 backtest and inspect diagnostics.

Out of scope:

- `opponent_id` as a model feature.
- Opponent defensive weakness, meaning points scored by players facing the opponent.
- Betting odds features.
- Elo or table-position features.
- Live 2026 fixture capture.
- Model architecture changes.
- Optimizer changes.

## Data Layout

Fixture files live under:

```text
data/01_raw/fixtures/
  club_mapping.csv
  2025/
    partidas-1.csv
    partidas-2.csv
    ...
    partidas-38.csv
```

`club_mapping.csv` is a manually curated, committed artifact. It maps external source names to Cartola club IDs:

```csv
external_name,id_clube
Flamengo RJ,262
Botafogo RJ,263
Corinthians,264
```

The exact external names must match the selected source. The loader never fetches or infers this mapping at runtime.

Each `partidas-{round}.csv` file uses this canonical schema:

```csv
rodada,id_clube_home,id_clube_away,data
1,283,2305,2025-03-29
1,356,266,2025-03-29
```

Required columns:

- `rodada`
- `id_clube_home`
- `id_clube_away`
- `data`

Additional columns are out of scope for this milestone; fixture features must use only the required columns.

## Source Strategy

The verified source for 2025 match data is TheSportsDB's Brazilian Serie A round endpoint:

```text
https://www.thesportsdb.com/api/v1/json/{api_key}/eventsround.php?id=4351&r={round}&s=2025
```

Local verification showed that this endpoint returns one official Brasileirão round at a time with `intRound`, `dateEvent`, `strHomeTeam`, and `strAwayTeam`. The public test key `3` is enough for this import, and the script also supports `THESPORTSDB_API_KEY`.

Football-Data's `https://www.football-data.co.uk/new/BRA.csv` remains useful for match results, but it is not used for this milestone because it lacks an explicit round column.

The import step must:

1. Fetch TheSportsDB events for rounds 1 through 38.
2. Map `strHomeTeam` and `strAwayTeam` through committed `club_mapping.csv`.
3. Load normalized Cartola player data for the season.
4. Keep only official-round matches where both clubs are in the Cartola `entrou_em_campo == True` club set for that same round.
5. Write one canonical `partidas-{round}.csv` per round.
6. Produce a validation report comparing fixture club sets with clubs that actually entered the field in each Cartola round.

If any club that entered the field in Cartola is missing from the generated fixture file for that round, the import must fail rather than silently creating unreliable fixture files.

## Round Alignment Validation

Cartola rounds can differ from simple chronological Brasileirão matchdays because of postponed matches. The validation step is a required data-quality gate.

For each round:

1. Build the fixture club set from `partidas-{round}.csv`:

```text
fixture_clubs = set(id_clube_home) union set(id_clube_away)
```

2. Build the Cartola played club set from normalized player data:

```text
played_clubs = clubs with at least one row where rodada == N and entrou_em_campo == True
```

3. Report:

- `rodada`
- `fixture_club_count`
- `played_club_count`
- `missing_from_fixtures`
- `extra_in_fixtures`
- `is_valid`

Validation output should be written under:

```text
data/08_reporting/fixtures/2025/round_alignment.csv
```

A round is valid when both differences are empty. If real Cartola data has incomplete player rows for a round, the validation report should make the mismatch visible; implementation should not hide it.

## Fixture Loader

Add the fixture loader in `src/cartola/backtesting/data.py`, following the existing season-data loader pattern.

API:

```python
load_fixtures(season: int, project_root: str | Path = ".") -> pd.DataFrame
```

Return columns:

- `rodada`
- `id_clube_home`
- `id_clube_away`
- `data`

Loader behavior:

- Read `data/01_raw/fixtures/{season}/partidas-*.csv`.
- Validate required columns.
- Convert `rodada`, `id_clube_home`, and `id_clube_away` to numeric.
- Convert `data` to datetime/date-compatible values.
- Reject duplicate club appearances within the same round.
- Reject rows where `id_clube_home == id_clube_away`.
- Return a single normalized fixture dataframe.

## Feature Definitions

### `is_home`

Binary feature for the candidate player's club in the target round:

```text
1 if id_clube == id_clube_home for that round
0 if id_clube == id_clube_away for that round
```

If the player's club is absent from fixture data for the target round, fill with `0`.

### `opponent_club_points_roll3`

For each candidate player, find the opponent club for the target round using fixture data. Then compute the opponent club's rolling mean of total Cartola player points over the last three prior rounds.

This measures opponent offensive strength, not defensive weakness.

Computation:

1. Use fixture data only to determine `opponent_id` for the target round.
2. Use existing prior `played_history` to compute club-round point totals.
3. Reuse the same rolling logic as `club_points_roll3`.
4. Join the opponent club's rolling value onto candidates.

Default: fill missing values with the same global club-points prior used for `club_points_roll3`.

## Join-Only Fields

`opponent_id` is an intermediate join key only. It must not be added to `FEATURE_COLUMNS` or `NUMERIC_PRIOR_COLUMNS`.

Reason: the current model pipeline has no categorical encoding for this ID. Treating a 20-value club identifier as numeric would introduce meaningless ordinal structure.

## Backtesting Data Flow

Update the backtest runner so fixture data is loaded once and passed into feature construction.

API:

```python
build_prediction_frame(
    season_df: pd.DataFrame,
    target_round: int,
    fixtures: pd.DataFrame | None = None,
) -> pd.DataFrame
```

`fixtures` should default to `None` so existing tests and fixture-less workflows remain valid. When fixtures are absent:

- `is_home = 0`
- `opponent_club_points_roll3 = global_club_points_prior`

`build_training_frame` should accept and pass through the same optional fixture dataframe. It should continue to call `build_prediction_frame` for each historical target round, preserving the walk-forward boundary.

## Leakage Rules

- Fixture schedule fields (`home`, `away`, date) are pre-round information.
- `is_home` is safe for target round N because fixture assignment is known before kickoff.
- `opponent_club_points_roll3` must use only club points from rounds `< N`.
- No target-round match result, score, odds movement, lineup confirmation, or player points may be used.
- `opponent_id` must not enter the feature matrix.

## Tests

Add focused tests for:

- Loading valid fixture files.
- Rejecting malformed fixture files.
- Rejecting duplicate club appearances in a round.
- Building `is_home` for home and away players.
- Keeping `opponent_id` out of `FEATURE_COLUMNS`.
- Computing `opponent_club_points_roll3` from prior rounds only.
- Fixture-less fallback behavior.
- Round-alignment validation report.

## Verification

Run:

```bash
uv run --frozen pytest src/tests/backtesting/ -v
uv run --frozen --no-dev python -m cartola.backtesting.cli --season 2025 --start-round 5 --budget 100
grep -E 'player_r2|player_correlation' data/08_reporting/backtests/2025/diagnostics.csv | grep ',all,'
uv run --frozen scripts/pyrepo-check --all
```

Success criteria remain:

- Random Forest `player_r2 > 0.05`.
- Random Forest beats the baseline by at least 2 average points per round.

If fixture features still miss these thresholds, the next model work should be position-specific predictors or an explicit DNP probability adjustment.
