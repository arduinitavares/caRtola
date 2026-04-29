# Live Squad Recommendation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a single-round Cartola squad recommendation command for live 2026 gameplay and leakage-safe historical replay.

**Architecture:** Add a dedicated recommendation module that reuses the existing loaders, feature builders, RF model, baseline model, and ILP optimizer without calling `run_backtest`. The module slices raw Cartola data to `rodada <= target_round` immediately, uses target-sliced FootyStats rows, writes explicit live/replay output allowlists, and stores recommendation metadata under `data/08_reporting/recommendations/`.

**Tech Stack:** Python 3.13, pandas, scikit-learn RandomForest, PuLP optimizer, pytest, `uv run --frozen scripts/pyrepo-check --all`.

---

## File Structure

- Create `src/cartola/backtesting/recommendation.py`
  - Owns recommendation config/result dataclasses, data visibility helpers, finalized-live-data detection, orchestration, output writing, and metadata serialization.
- Create `scripts/recommend_squad.py`
  - Thin CLI wrapper around `run_recommendation()`.
- Create `src/tests/backtesting/test_recommendation.py`
  - Unit/integration tests for leakage boundaries, live/replay behavior, output allowlists, and metadata.
- Create `src/tests/backtesting/test_recommend_squad_cli.py`
  - CLI parsing and main wiring tests.
- Modify `src/cartola/backtesting/footystats_features.py`
  - Add target-sliced recommendation loader using the same safe-column `usecols` pattern.
- Modify `src/tests/backtesting/test_footystats_features.py`
  - Add target-sliced FootyStats recommendation loader tests.
- Modify `README.md`
  - Add the live/replay recommendation commands.
- Modify `roadmap.md`
  - Mark the live recommendation milestone as active or delivered after implementation.

---

### Task 1: Add Recommendation Test Scaffolding

**Files:**
- Create: `src/tests/backtesting/test_recommendation.py`

- [ ] **Step 1: Create shared test helpers**

Add this file with helpers only. These helpers intentionally create enough players for the existing `4-3-3` formation.

```python
from __future__ import annotations

import json
import pandas as pd

from cartola.backtesting.config import DEFAULT_SCOUT_COLUMNS


def _round_frame(
    round_number: int,
    *,
    finalized: bool = True,
    zero_filled_scouts: bool = False,
    points_offset: float = 0.0,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    player_id = 1
    for posicao, count in {"gol": 2, "lat": 3, "zag": 3, "mei": 4, "ata": 4, "tec": 2}.items():
        for offset in range(count):
            row: dict[str, object] = {
                "id_atleta": player_id,
                "apelido": f"{posicao}-{offset}",
                "slug": f"{posicao}-{offset}",
                "id_clube": player_id,
                "nome_clube": f"Club {player_id}",
                "posicao": posicao,
                "status": "Provavel",
                "rodada": round_number,
                "preco": 5.0,
                "preco_pre_rodada": 5.0,
                "pontuacao": float(round_number + offset + points_offset) if finalized else 0.0,
                "media": float(round_number + offset),
                "num_jogos": round_number - 1,
                "variacao": 0.0,
                "entrou_em_campo": finalized,
            }
            for scout in DEFAULT_SCOUT_COLUMNS:
                row[scout] = 0 if zero_filled_scouts else (1 if finalized and scout == "DS" else 0)
            rows.append(row)
            player_id += 1
    return pd.DataFrame(rows)


def _season_frame(rounds: range, *, target_round: int | None = None, live_target: bool = False) -> pd.DataFrame:
    frames = []
    for round_number in rounds:
        frames.append(
            _round_frame(
                round_number,
                finalized=not (live_target and target_round == round_number),
                zero_filled_scouts=live_target and target_round == round_number,
            )
        )
    return pd.concat(frames, ignore_index=True)


def _footystats_rows(rounds: range, clubs: range = range(1, 19)) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for round_number in rounds:
        for club_id in clubs:
            opponent_id = club_id + 1 if club_id % 2 == 1 else club_id - 1
            team_ppg = float(round_number) + club_id / 100.0
            opponent_ppg = float(round_number) + opponent_id / 100.0
            rows.append(
                {
                    "rodada": round_number,
                    "id_clube": club_id,
                    "opponent_id_clube": opponent_id,
                    "is_home_footystats": int(club_id % 2 == 1),
                    "footystats_team_pre_match_ppg": team_ppg,
                    "footystats_opponent_pre_match_ppg": opponent_ppg,
                    "footystats_ppg_diff": team_ppg - opponent_ppg,
                }
            )
    return pd.DataFrame(rows)
```

- [ ] **Step 2: Run test discovery to verify the empty helper file imports**

Run:

```bash
uv run --frozen pytest src/tests/backtesting/test_recommendation.py -q
```

Expected: `no tests ran` or `1 warning` with no import error.

- [ ] **Step 3: Commit**

```bash
git add src/tests/backtesting/test_recommendation.py
git commit -m "test: add recommendation test helpers"
```

---

### Task 2: Add Recommendation Config, Visibility, and Finalized-Data Helpers

**Files:**
- Create: `src/cartola/backtesting/recommendation.py`
- Modify: `src/tests/backtesting/test_recommendation.py`

- [ ] **Step 1: Write failing helper tests**

Append these tests to `src/tests/backtesting/test_recommendation.py`:

```python
from cartola.backtesting.recommendation import (
    RecommendationConfig,
    _finalized_live_data_evidence,
    _validate_mode_scope,
    _visible_season_frame,
)
import pytest


def test_visible_season_frame_excludes_future_rounds() -> None:
    season_df = _season_frame(range(1, 6), target_round=3, live_target=True)

    visible = _visible_season_frame(season_df, target_round=3)

    assert sorted(visible["rodada"].unique().tolist()) == [1, 2, 3]
    assert 4 not in visible["rodada"].unique()
    assert 5 not in visible["rodada"].unique()


def test_live_mode_requires_current_year() -> None:
    config = RecommendationConfig(season=2025, target_round=10, mode="live", current_year=2026)

    with pytest.raises(ValueError, match="live mode requires season 2025 to equal current_year 2026"):
        _validate_mode_scope(config)


def test_replay_mode_allows_historical_season() -> None:
    config = RecommendationConfig(season=2025, target_round=10, mode="replay", current_year=2026)

    _validate_mode_scope(config)


def test_finalized_evidence_ignores_zero_filled_live_rows() -> None:
    target = _round_frame(14, finalized=False, zero_filled_scouts=True)

    evidence = _finalized_live_data_evidence(target)

    assert evidence == {
        "pontuacao_non_zero_count": 0,
        "entrou_em_campo_true_count": 0,
        "non_zero_scout_count": 0,
    }


def test_finalized_evidence_detects_played_rows_and_non_zero_scouts() -> None:
    target = _round_frame(14, finalized=True)

    evidence = _finalized_live_data_evidence(target)

    assert evidence["pontuacao_non_zero_count"] > 0
    assert evidence["entrou_em_campo_true_count"] > 0
    assert evidence["non_zero_scout_count"] > 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
uv run --frozen pytest src/tests/backtesting/test_recommendation.py -q
```

Expected: fail with `ModuleNotFoundError: No module named 'cartola.backtesting.recommendation'`.

- [ ] **Step 3: Implement config and helper functions**

Create `src/cartola/backtesting/recommendation.py`:

```python
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, Mapping

import pandas as pd

from cartola.backtesting.config import (
    DEFAULT_FORMATIONS,
    DEFAULT_SCOUT_COLUMNS,
    FootyStatsMode,
)

RecommendationMode = Literal["live", "replay"]


@dataclass(frozen=True)
class RecommendationConfig:
    season: int
    target_round: int
    mode: RecommendationMode
    budget: float = 100.0
    playable_statuses: tuple[str, ...] = ("Provavel",)
    formation_name: str = "4-3-3"
    random_seed: int = 123
    project_root: Path = Path(".")
    output_root: Path = Path("data/08_reporting/recommendations")
    footystats_mode: FootyStatsMode = "ppg"
    footystats_league_slug: str = "brazil-serie-a"
    footystats_dir: Path = Path("data/footystats")
    current_year: int | None = None
    allow_finalized_live_data: bool = False
    scout_columns: tuple[str, ...] = DEFAULT_SCOUT_COLUMNS
    formations: Mapping[str, Mapping[str, int]] = field(default_factory=lambda: DEFAULT_FORMATIONS)

    @property
    def output_path(self) -> Path:
        return self.project_root / self.output_root / str(self.season) / f"round-{self.target_round}" / self.mode

    @property
    def selected_formation(self) -> Mapping[str, int]:
        if self.formation_name not in self.formations:
            raise ValueError(f"Unknown formation {self.formation_name!r}. Available: {sorted(self.formations)}")
        return self.formations[self.formation_name]


def _resolved_current_year(config: RecommendationConfig) -> int:
    return config.current_year if config.current_year is not None else datetime.now(UTC).year


def _validate_mode_scope(config: RecommendationConfig) -> None:
    if config.mode not in {"live", "replay"}:
        raise ValueError(f"Unsupported recommendation mode: {config.mode!r}")
    if config.target_round <= 0:
        raise ValueError("target_round must be a positive integer")
    if config.mode == "live":
        current_year = _resolved_current_year(config)
        if config.season != current_year:
            raise ValueError(
                f"live mode requires season {config.season} to equal current_year {current_year}"
            )


def _visible_season_frame(season_df: pd.DataFrame, *, target_round: int) -> pd.DataFrame:
    rodada = pd.to_numeric(season_df["rodada"], errors="raise").astype(int)
    return season_df.loc[rodada.le(target_round)].copy()


def _finalized_live_data_evidence(target_frame: pd.DataFrame) -> dict[str, int]:
    pontuacao = pd.to_numeric(target_frame.get("pontuacao", pd.Series(dtype=float)), errors="coerce")
    pontuacao_non_zero_count = int(pontuacao.fillna(0.0).ne(0.0).sum())

    entrou = target_frame.get("entrou_em_campo", pd.Series(dtype=bool))
    entrou_true_count = int(entrou.fillna(False).astype(bool).sum())

    non_zero_scout_count = 0
    for scout in DEFAULT_SCOUT_COLUMNS:
        if scout in target_frame.columns:
            values = pd.to_numeric(target_frame[scout], errors="coerce").fillna(0.0)
            non_zero_scout_count += int(values.ne(0.0).sum())

    return {
        "pontuacao_non_zero_count": pontuacao_non_zero_count,
        "entrou_em_campo_true_count": entrou_true_count,
        "non_zero_scout_count": non_zero_scout_count,
    }
```

- [ ] **Step 4: Run helper tests**

Run:

```bash
uv run --frozen pytest src/tests/backtesting/test_recommendation.py -q
```

Expected: all tests in `test_recommendation.py` pass.

- [ ] **Step 5: Commit**

```bash
git add src/cartola/backtesting/recommendation.py src/tests/backtesting/test_recommendation.py
git commit -m "feat: add recommendation config and data visibility helpers"
```

---

### Task 3: Add Target-Sliced FootyStats Loader for Recommendations

**Files:**
- Modify: `src/cartola/backtesting/footystats_features.py`
- Modify: `src/tests/backtesting/test_footystats_features.py`

- [ ] **Step 1: Write failing FootyStats recommendation tests**

Append these tests to `src/tests/backtesting/test_footystats_features.py`:

```python
from cartola.backtesting.footystats_features import load_footystats_feature_rows_for_recommendation


def test_load_footystats_recommendation_rows_ignores_future_bad_rows(tmp_path: Path) -> None:
    rows = [_match_row(week=week, status="complete") for week in range(1, 4)]
    future = _match_row(week=4, status="suspended")
    future["Pre-Match PPG (Home)"] = None
    rows.append(future)
    _write_matches_csv(tmp_path, rows)
    _write_cartola_round(tmp_path)
    required_keys = pd.DataFrame(
        [
            {"rodada": rodada, "id_clube": club_id}
            for rodada in range(1, 4)
            for club_id in (262, 275)
        ]
    )

    result = load_footystats_feature_rows_for_recommendation(
        season=SEASON,
        project_root=tmp_path,
        footystats_dir=Path("data/footystats"),
        league_slug=LEAGUE_SLUG,
        current_year=2026,
        target_round=3,
        footystats_mode="ppg",
        require_complete_status=True,
        required_keys=required_keys,
    )

    assert sorted(result.rows["rodada"].unique().tolist()) == [1, 2, 3]


def test_load_footystats_recommendation_rows_rejects_missing_required_key(tmp_path: Path) -> None:
    _write_matches_csv(tmp_path, [_match_row(week=1, status="complete")])
    _write_cartola_round(tmp_path)
    required_keys = pd.DataFrame(
        [
            {"rodada": 1, "id_clube": 262},
            {"rodada": 1, "id_clube": 999},
        ]
    )

    with pytest.raises(ValueError, match="missing FootyStats recommendation rows"):
        load_footystats_feature_rows_for_recommendation(
            season=SEASON,
            project_root=tmp_path,
            footystats_dir=Path("data/footystats"),
            league_slug=LEAGUE_SLUG,
            current_year=2026,
            target_round=1,
            footystats_mode="ppg",
            require_complete_status=False,
            required_keys=required_keys,
        )


def test_load_footystats_recommendation_rows_rejects_duplicate_required_key(tmp_path: Path) -> None:
    rows = [_match_row(week=1, status="complete"), _match_row(week=1, status="complete")]
    _write_matches_csv(tmp_path, rows)
    _write_cartola_round(tmp_path)
    required_keys = pd.DataFrame([{"rodada": 1, "id_clube": 262}])

    with pytest.raises(ValueError, match="duplicate FootyStats recommendation rows"):
        load_footystats_feature_rows_for_recommendation(
            season=SEASON,
            project_root=tmp_path,
            footystats_dir=Path("data/footystats"),
            league_slug=LEAGUE_SLUG,
            current_year=2026,
            target_round=1,
            footystats_mode="ppg",
            require_complete_status=False,
            required_keys=required_keys,
        )
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
uv run --frozen pytest src/tests/backtesting/test_footystats_features.py::test_load_footystats_recommendation_rows_ignores_future_bad_rows src/tests/backtesting/test_footystats_features.py::test_load_footystats_recommendation_rows_rejects_missing_required_key src/tests/backtesting/test_footystats_features.py::test_load_footystats_recommendation_rows_rejects_duplicate_required_key -q
```

Expected: fail with import error for `load_footystats_feature_rows_for_recommendation`.

- [ ] **Step 3: Implement the target-sliced loader**

Modify `src/cartola/backtesting/footystats_features.py`.

Add this public function after `load_footystats_feature_rows()`:

```python
def load_footystats_feature_rows_for_recommendation(
    *,
    season: int,
    project_root: Path,
    footystats_dir: Path,
    league_slug: str,
    current_year: int | None,
    target_round: int,
    footystats_mode: str,
    require_complete_status: bool,
    required_keys: pd.DataFrame,
) -> FootyStatsPPGLoadResult:
    if footystats_mode not in SOURCE_COLUMNS_BY_MODE:
        raise ValueError(f"Unsupported footystats_mode: {footystats_mode}")
    if target_round <= 0:
        raise ValueError("target_round must be a positive integer")

    source_path = _source_path(
        project_root=project_root,
        footystats_dir=footystats_dir,
        league_slug=league_slug,
        season=season,
    )
    _validate_source_filename(source_path, season=season, league_slug=league_slug)

    required_columns = SOURCE_COLUMNS_BY_MODE[footystats_mode]
    df = _read_source_frame(source_path, required_columns)
    game_weeks = _validated_game_weeks(df)
    visible_mask = game_weeks.le(target_round)
    df = df.loc[visible_mask].copy()
    game_weeks = game_weeks.loc[visible_mask]

    home_ppg = _validated_ppg(df, "Pre-Match PPG (Home)")
    away_ppg = _validated_ppg(df, "Pre-Match PPG (Away)")
    home_xg = _validated_xg(df, "Home Team Pre-Match xG") if footystats_mode == "ppg_xg" else None
    away_xg = _validated_xg(df, "Away Team Pre-Match xG") if footystats_mode == "ppg_xg" else None
    statuses = _validated_statuses(df, "live_current")
    if require_complete_status and any(status != "complete" for status in statuses):
        raise ValueError("replay recommendation requires visible FootyStats statuses to be complete")
    _validate_team_names_present(df)

    team_names = _team_names(df)
    comparison = compare_teams_to_cartola(season=season, footystats_team_names=team_names, project_root=project_root)
    _validate_team_mapping(comparison, require_all_cartola_teams=False)

    rows = _build_feature_rows(
        df,
        game_weeks,
        home_ppg,
        away_ppg,
        comparison.mapped_teams,
        footystats_mode=footystats_mode,
        home_xg=home_xg,
        away_xg=away_xg,
    )
    _validate_required_recommendation_keys(rows, required_keys)

    return FootyStatsPPGLoadResult(
        rows=rows,
        source_path=source_path,
        source_sha256=_sha256_file(source_path),
        diagnostics=FootyStatsJoinDiagnostics(),
        footystats_mode=footystats_mode,
        feature_columns=FEATURE_COLUMNS_BY_MODE[footystats_mode],
    )
```

Change `_validate_team_mapping` to accept the new flag:

```python
def _validate_team_mapping(comparison, *, require_all_cartola_teams: bool = True) -> None:
    failures: list[str] = []
    if comparison.unmapped_footystats_teams:
        failures.append(f"unmapped FootyStats teams: {', '.join(comparison.unmapped_footystats_teams)}")
    if require_all_cartola_teams and comparison.missing_cartola_teams:
        failures.append(f"missing Cartola teams: {', '.join(comparison.missing_cartola_teams)}")
    if comparison.duplicate_mapped_cartola_teams:
        failures.append(f"duplicate mapped Cartola teams: {', '.join(comparison.duplicate_mapped_cartola_teams)}")
    if failures:
        raise ValueError("FootyStats team mapping failed: " + "; ".join(failures))
```

Add this helper near `_reject_duplicate_join_keys()`:

```python
def _validate_required_recommendation_keys(rows: pd.DataFrame, required_keys: pd.DataFrame) -> None:
    if required_keys.empty:
        return

    required = _unique_key_frame(required_keys)
    available = _unique_key_frame(rows)

    missing = required.merge(available, on=["rodada", "id_clube"], how="left", indicator=True)
    missing = missing[missing["_merge"].eq("left_only")][["rodada", "id_clube"]]
    if not missing.empty:
        raise ValueError(
            "missing FootyStats recommendation rows: "
            f"{_group_key_records_by_round(missing)}"
        )

    duplicate_counts = _duplicate_count_frame(rows)
    duplicates = duplicate_counts.merge(required, on=["rodada", "id_clube"], how="inner")
    duplicates = duplicates[duplicates["count"].gt(1)]
    if not duplicates.empty:
        raise ValueError(
            "duplicate FootyStats recommendation rows: "
            f"{_group_key_records_by_round(duplicates, include_count=True)}"
        )
```

- [ ] **Step 4: Run focused FootyStats tests**

Run:

```bash
uv run --frozen pytest src/tests/backtesting/test_footystats_features.py -q
```

Expected: all FootyStats feature tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/cartola/backtesting/footystats_features.py src/tests/backtesting/test_footystats_features.py
git commit -m "feat: add target-sliced footystats recommendation loader"
```

---

### Task 4: Implement Core Recommendation Orchestration

**Files:**
- Modify: `src/cartola/backtesting/recommendation.py`
- Modify: `src/tests/backtesting/test_recommendation.py`

- [ ] **Step 1: Write failing orchestration tests**

Append these tests to `src/tests/backtesting/test_recommendation.py`:

```python
from cartola.backtesting.recommendation import run_recommendation


def test_run_recommendation_ignores_future_cartola_rows(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    season_df = _season_frame(range(1, 6), target_round=3, live_target=True)
    load_calls: list[dict[str, object]] = []

    def fake_load_footystats(**kwargs: object):
        load_calls.append(kwargs)
        from cartola.backtesting.footystats_features import FootyStatsJoinDiagnostics, FootyStatsPPGLoadResult

        return FootyStatsPPGLoadResult(
            rows=_footystats_rows(range(1, 4)),
            source_path=tmp_path / "data/footystats/source.csv",
            source_sha256="sha",
            diagnostics=FootyStatsJoinDiagnostics(),
        )

    monkeypatch.setattr("cartola.backtesting.recommendation.load_season_data", lambda *a, **k: season_df)
    monkeypatch.setattr(
        "cartola.backtesting.recommendation.load_footystats_feature_rows_for_recommendation",
        fake_load_footystats,
    )
    config = RecommendationConfig(
        season=2026,
        target_round=3,
        mode="live",
        project_root=tmp_path,
        current_year=2026,
    )

    result = run_recommendation(config)

    assert result.metadata["visible_max_round"] == 3
    assert result.metadata["training_rounds"] == [1, 2]
    required_keys = load_calls[0]["required_keys"]
    assert int(required_keys["rodada"].max()) == 3
    assert result.candidate_predictions["rodada"].eq(3).all()


def test_run_recommendation_replay_reports_actual_points(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    season_df = _season_frame(range(1, 4))

    def fake_load_footystats(**kwargs: object):
        from cartola.backtesting.footystats_features import FootyStatsJoinDiagnostics, FootyStatsPPGLoadResult

        return FootyStatsPPGLoadResult(
            rows=_footystats_rows(range(1, 4)),
            source_path=tmp_path / "data/footystats/source.csv",
            source_sha256="sha",
            diagnostics=FootyStatsJoinDiagnostics(),
        )

    monkeypatch.setattr("cartola.backtesting.recommendation.load_season_data", lambda *a, **k: season_df)
    monkeypatch.setattr(
        "cartola.backtesting.recommendation.load_footystats_feature_rows_for_recommendation",
        fake_load_footystats,
    )
    config = RecommendationConfig(
        season=2026,
        target_round=3,
        mode="replay",
        project_root=tmp_path,
        current_year=2026,
    )

    result = run_recommendation(config)

    assert result.summary["actual_points"] is not None
    assert "pontuacao" in result.recommended_squad.columns
    assert result.summary["optimizer_status"] == "Optimal"


def test_run_recommendation_live_suppresses_actual_columns_when_finalized_allowed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    season_df = _season_frame(range(1, 4))

    def fake_load_footystats(**kwargs: object):
        from cartola.backtesting.footystats_features import FootyStatsJoinDiagnostics, FootyStatsPPGLoadResult

        return FootyStatsPPGLoadResult(
            rows=_footystats_rows(range(1, 4)),
            source_path=tmp_path / "data/footystats/source.csv",
            source_sha256="sha",
            diagnostics=FootyStatsJoinDiagnostics(),
        )

    monkeypatch.setattr("cartola.backtesting.recommendation.load_season_data", lambda *a, **k: season_df)
    monkeypatch.setattr(
        "cartola.backtesting.recommendation.load_footystats_feature_rows_for_recommendation",
        fake_load_footystats,
    )
    config = RecommendationConfig(
        season=2026,
        target_round=3,
        mode="live",
        project_root=tmp_path,
        current_year=2026,
        allow_finalized_live_data=True,
    )

    result = run_recommendation(config)

    assert result.summary["actual_points"] is None
    assert "pontuacao" not in result.recommended_squad.columns
    assert "entrou_em_campo" not in result.candidate_predictions.columns
    assert result.metadata["finalized_live_data_detected"] is True
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
uv run --frozen pytest src/tests/backtesting/test_recommendation.py -q
```

Expected: fail because `run_recommendation` and result dataclasses do not exist.

- [ ] **Step 3: Implement orchestration dataclasses and helpers**

Add imports to `src/cartola/backtesting/recommendation.py`:

```python
import json

from cartola.backtesting.config import MARKET_OPEN_PRICE_COLUMN, BacktestConfig
from cartola.backtesting.data import load_season_data
from cartola.backtesting.features import (
    FOOTYSTATS_PPG_FEATURE_COLUMNS,
    FOOTYSTATS_XG_FEATURE_COLUMNS,
    build_prediction_frame,
    build_training_frame,
    feature_columns_for_config,
)
from cartola.backtesting.footystats_features import (
    FootyStatsPPGLoadResult,
    build_footystats_join_diagnostics,
    load_footystats_feature_rows_for_recommendation,
)
from cartola.backtesting.models import BaselinePredictor, RandomForestPointPredictor
from cartola.backtesting.optimizer import optimize_squad
```

Add result dataclass:

```python
@dataclass(frozen=True)
class RecommendationResult:
    recommended_squad: pd.DataFrame
    candidate_predictions: pd.DataFrame
    summary: dict[str, object]
    metadata: dict[str, object]
```

Add helpers:

```python
def _backtest_config(config: RecommendationConfig) -> BacktestConfig:
    return BacktestConfig(
        season=config.season,
        start_round=config.target_round,
        budget=config.budget,
        playable_statuses=config.playable_statuses,
        formation_name=config.formation_name,
        random_seed=config.random_seed,
        project_root=config.project_root,
        output_root=Path("data/08_reporting/backtests"),
        fixture_mode="none",
        footystats_mode=config.footystats_mode,
        footystats_evaluation_scope=_footystats_scope(config),
        footystats_league_slug=config.footystats_league_slug,
        footystats_dir=config.footystats_dir,
        current_year=config.current_year,
        scout_columns=config.scout_columns,
        formations=config.formations,
    )


def _footystats_scope(config: RecommendationConfig) -> str:
    if config.footystats_mode == "none":
        return "historical_candidate"
    if config.season == _resolved_current_year(config):
        return "live_current"
    return "historical_candidate"


def _real_club_keys(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=pd.Index(["rodada", "id_clube"]))
    source = frame.copy()
    if "nome_clube" in source.columns:
        has_name = source["nome_clube"].notna() & source["nome_clube"].map(lambda value: str(value).strip() != "")
        source = source.loc[has_name]
    keys = source[["rodada", "id_clube"]].dropna().drop_duplicates().copy()
    keys["rodada"] = pd.to_numeric(keys["rodada"], errors="raise").astype(int)
    keys["id_clube"] = pd.to_numeric(keys["id_clube"], errors="raise").astype(int)
    return keys.sort_values(["rodada", "id_clube"]).reset_index(drop=True)


def _load_recommendation_footystats(
    config: RecommendationConfig,
    visible_season_df: pd.DataFrame,
) -> FootyStatsPPGLoadResult | None:
    if config.footystats_mode == "none":
        return None
    return load_footystats_feature_rows_for_recommendation(
        season=config.season,
        project_root=config.project_root,
        footystats_dir=config.footystats_dir,
        league_slug=config.footystats_league_slug,
        current_year=config.current_year,
        target_round=config.target_round,
        footystats_mode=config.footystats_mode,
        require_complete_status=config.season != _resolved_current_year(config),
        required_keys=_real_club_keys(visible_season_df),
    )
```

Add output allowlist helpers:

```python
BASE_OUTPUT_COLUMNS = [
    "rodada",
    "id_atleta",
    "apelido",
    "id_clube",
    "nome_clube",
    "posicao",
    "status",
    MARKET_OPEN_PRICE_COLUMN,
    "baseline_score",
    "random_forest_score",
    "price_score",
]


def _active_footystats_columns(config: RecommendationConfig) -> list[str]:
    if config.footystats_mode == "none":
        return []
    if config.footystats_mode == "ppg":
        return list(FOOTYSTATS_PPG_FEATURE_COLUMNS)
    if config.footystats_mode == "ppg_xg":
        return [*FOOTYSTATS_PPG_FEATURE_COLUMNS, *FOOTYSTATS_XG_FEATURE_COLUMNS]
    raise ValueError(f"Unsupported footystats_mode: {config.footystats_mode!r}")


def _select_columns(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    return frame[[column for column in columns if column in frame.columns]].copy()
```

Add `run_recommendation()`:

```python
def run_recommendation(config: RecommendationConfig) -> RecommendationResult:
    _validate_mode_scope(config)
    season_df = load_season_data(config.season, project_root=config.project_root)
    visible = _visible_season_frame(season_df, target_round=config.target_round)
    target = visible[visible["rodada"].eq(config.target_round)].copy()
    if target.empty:
        raise ValueError(f"Target round {config.target_round} not found in season {config.season} data.")
    if visible[visible["rodada"].lt(config.target_round)].empty:
        raise ValueError(f"No training history exists before target round {config.target_round}.")

    finalized_evidence = _finalized_live_data_evidence(target)
    finalized_detected = any(value > 0 for value in finalized_evidence.values())
    if config.mode == "live" and finalized_detected and not config.allow_finalized_live_data:
        raise ValueError(
            "live mode target-round data appears finalized: "
            f"season={config.season} target_round={config.target_round} evidence={finalized_evidence}"
        )

    footystats = _load_recommendation_footystats(config, visible)
    footystats_rows = footystats.rows if footystats is not None else None
    diagnostics = build_footystats_join_diagnostics(visible, footystats_rows) if footystats_rows is not None else None
    if diagnostics is not None and diagnostics.missing_join_keys_by_round:
        raise ValueError(f"FootyStats recommendation missing join keys: {diagnostics.missing_join_keys_by_round}")
    if diagnostics is not None and diagnostics.duplicate_join_keys_by_round:
        raise ValueError(f"FootyStats recommendation duplicate join keys: {diagnostics.duplicate_join_keys_by_round}")

    backtest_config = _backtest_config(config)
    training = build_training_frame(
        visible,
        config.target_round,
        playable_statuses=config.playable_statuses,
        fixtures=None,
        footystats_rows=footystats_rows,
    )
    candidates = build_prediction_frame(visible, config.target_round, fixtures=None, footystats_rows=footystats_rows)
    candidates = candidates[candidates["status"].isin(config.playable_statuses)].copy()
    if training.empty:
        raise ValueError(f"No training rows remain before target round {config.target_round}.")
    if candidates.empty:
        raise ValueError(f"No playable target-round candidates for round {config.target_round}.")

    feature_columns = feature_columns_for_config(backtest_config)
    scored = candidates.copy()
    baseline_model = BaselinePredictor().fit(training)
    forest_model = RandomForestPointPredictor(
        random_seed=config.random_seed,
        feature_columns=feature_columns,
    ).fit(training)
    scored["baseline_score"] = baseline_model.predict(scored)
    scored["random_forest_score"] = forest_model.predict(scored)
    scored["price_score"] = scored[MARKET_OPEN_PRICE_COLUMN].astype(float)

    optimized = optimize_squad(scored.assign(predicted_points=scored["random_forest_score"]), "random_forest_score", backtest_config)
    if optimized.status != "Optimal":
        raise ValueError(f"Recommendation optimizer failed: status={optimized.status}")

    selected = optimized.selected.copy()
    selected["predicted_points"] = selected["random_forest_score"]
    actual_points = optimized.actual_points if config.mode == "replay" else None

    selected_columns = [*BASE_OUTPUT_COLUMNS, "predicted_points"]
    candidate_columns = [*BASE_OUTPUT_COLUMNS, *_active_footystats_columns(config)]
    if config.mode == "replay":
        replay_columns = ["pontuacao", "entrou_em_campo", *config.scout_columns]
        selected_columns = [*selected_columns, *replay_columns]
        candidate_columns = [*candidate_columns, *replay_columns]

    recommended_squad = _select_columns(selected, selected_columns)
    candidate_predictions = _select_columns(scored, candidate_columns)
    summary = {
        "season": config.season,
        "target_round": config.target_round,
        "mode": config.mode,
        "strategy": "random_forest",
        "formation": config.formation_name,
        "budget": float(config.budget),
        "optimizer_status": optimized.status,
        "selected_count": int(optimized.selected_count),
        "budget_used": float(optimized.budget_used),
        "predicted_points": float(optimized.predicted_points),
        "actual_points": None if actual_points is None else float(actual_points),
        "output_directory": str(config.output_path),
    }
    metadata = _build_metadata(
        config=config,
        visible=visible,
        feature_columns=feature_columns,
        footystats=footystats,
        finalized_detected=finalized_detected,
        finalized_evidence=finalized_evidence,
        optimizer_status=optimized.status,
    )

    _write_recommendation_outputs(config, recommended_squad, candidate_predictions, summary, metadata)
    return RecommendationResult(
        recommended_squad=recommended_squad,
        candidate_predictions=candidate_predictions,
        summary=summary,
        metadata=metadata,
    )
```

- [ ] **Step 4: Add metadata and writing helpers**

Add below `run_recommendation()`:

```python
def _utc_now_z() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _build_metadata(
    *,
    config: RecommendationConfig,
    visible: pd.DataFrame,
    feature_columns: list[str],
    footystats: FootyStatsPPGLoadResult | None,
    finalized_detected: bool,
    finalized_evidence: dict[str, int],
    optimizer_status: str,
) -> dict[str, object]:
    training_rounds = sorted(
        int(round_number)
        for round_number in visible.loc[visible["rodada"].lt(config.target_round), "rodada"].dropna().unique()
    )
    return {
        "season": config.season,
        "target_round": config.target_round,
        "mode": config.mode,
        "current_year": _resolved_current_year(config),
        "training_rounds": training_rounds,
        "candidate_round": config.target_round,
        "visible_max_round": int(pd.to_numeric(visible["rodada"], errors="raise").max()),
        "fixture_mode": "none",
        "footystats_mode": config.footystats_mode,
        "footystats_evaluation_scope": _footystats_scope(config),
        "footystats_league_slug": config.footystats_league_slug,
        "footystats_matches_source_path": str(footystats.source_path) if footystats is not None else None,
        "footystats_matches_source_sha256": footystats.source_sha256 if footystats is not None else None,
        "feature_columns": feature_columns,
        "playable_statuses": list(config.playable_statuses),
        "formation": config.formation_name,
        "budget": float(config.budget),
        "random_seed": config.random_seed,
        "finalized_live_data_detected": finalized_detected,
        "finalized_live_data_evidence": finalized_evidence,
        "allow_finalized_live_data": config.allow_finalized_live_data,
        "optimizer_status": optimizer_status,
        "warnings": [],
        "generated_at_utc": _utc_now_z(),
    }


def _write_recommendation_outputs(
    config: RecommendationConfig,
    recommended_squad: pd.DataFrame,
    candidate_predictions: pd.DataFrame,
    summary: dict[str, object],
    metadata: dict[str, object],
) -> None:
    output_path = config.output_path
    output_path.mkdir(parents=True, exist_ok=True)
    recommended_squad.to_csv(output_path / "recommended_squad.csv", index=False)
    candidate_predictions.to_csv(output_path / "candidate_predictions.csv", index=False)
    (output_path / "recommendation_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_path / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
```

- [ ] **Step 5: Run recommendation tests**

Run:

```bash
uv run --frozen pytest src/tests/backtesting/test_recommendation.py -q
```

Expected: all recommendation tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/cartola/backtesting/recommendation.py src/tests/backtesting/test_recommendation.py
git commit -m "feat: add single-round recommendation orchestration"
```

---

### Task 5: Add Output File Assertions and Live/Replays Edge Tests

**Files:**
- Modify: `src/tests/backtesting/test_recommendation.py`
- Modify: `src/cartola/backtesting/recommendation.py`

- [ ] **Step 1: Add tests for output files and live finalized rejection**

Append:

```python
def test_live_mode_rejects_finalized_target_round_without_escape_hatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    season_df = _season_frame(range(1, 4))
    monkeypatch.setattr("cartola.backtesting.recommendation.load_season_data", lambda *a, **k: season_df)
    config = RecommendationConfig(
        season=2026,
        target_round=3,
        mode="live",
        project_root=tmp_path,
        current_year=2026,
    )

    with pytest.raises(ValueError, match="appears finalized"):
        run_recommendation(config)


def test_run_recommendation_writes_expected_output_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    season_df = _season_frame(range(1, 4), target_round=3, live_target=True)

    def fake_load_footystats(**kwargs: object):
        from cartola.backtesting.footystats_features import FootyStatsJoinDiagnostics, FootyStatsPPGLoadResult

        return FootyStatsPPGLoadResult(
            rows=_footystats_rows(range(1, 4)),
            source_path=tmp_path / "data/footystats/source.csv",
            source_sha256="sha",
            diagnostics=FootyStatsJoinDiagnostics(),
        )

    monkeypatch.setattr("cartola.backtesting.recommendation.load_season_data", lambda *a, **k: season_df)
    monkeypatch.setattr(
        "cartola.backtesting.recommendation.load_footystats_feature_rows_for_recommendation",
        fake_load_footystats,
    )
    config = RecommendationConfig(
        season=2026,
        target_round=3,
        mode="live",
        project_root=tmp_path,
        current_year=2026,
    )

    run_recommendation(config)

    output_path = tmp_path / "data/08_reporting/recommendations/2026/round-3/live"
    assert (output_path / "recommended_squad.csv").exists()
    assert (output_path / "candidate_predictions.csv").exists()
    assert (output_path / "recommendation_summary.json").exists()
    assert (output_path / "run_metadata.json").exists()
    summary = json.loads((output_path / "recommendation_summary.json").read_text(encoding="utf-8"))
    metadata = json.loads((output_path / "run_metadata.json").read_text(encoding="utf-8"))
    assert summary["actual_points"] is None
    assert metadata["training_rounds"] == [1, 2]
    assert metadata["footystats_matches_source_sha256"] == "sha"
```

- [ ] **Step 2: Run tests**

Run:

```bash
uv run --frozen pytest src/tests/backtesting/test_recommendation.py -q
```

Expected: if the implementation from Task 4 is complete, tests pass. If `test_live_mode_rejects_finalized_target_round_without_escape_hatch` fails because FootyStats is loaded first, move finalized-data validation before FootyStats loading.

- [ ] **Step 3: Commit**

```bash
git add src/cartola/backtesting/recommendation.py src/tests/backtesting/test_recommendation.py
git commit -m "test: cover recommendation outputs and finalized live rejection"
```

---

### Task 6: Add Recommendation CLI

**Files:**
- Create: `scripts/recommend_squad.py`
- Create: `src/tests/backtesting/test_recommend_squad_cli.py`

- [ ] **Step 1: Write failing CLI tests**

Create `src/tests/backtesting/test_recommend_squad_cli.py`:

```python
from __future__ import annotations

from pathlib import Path

import pytest

from scripts.recommend_squad import main, parse_args
from cartola.backtesting.recommendation import RecommendationConfig, RecommendationResult


def test_parse_args_requires_target_round() -> None:
    with pytest.raises(SystemExit):
        parse_args(["--season", "2026", "--mode", "live"])


def test_parse_args_has_no_fixture_mode() -> None:
    with pytest.raises(SystemExit):
        parse_args(["--season", "2026", "--target-round", "14", "--fixture-mode", "none"])


def test_parse_args_builds_live_defaults() -> None:
    args = parse_args(["--season", "2026", "--target-round", "14", "--mode", "live", "--current-year", "2026"])

    assert args.season == 2026
    assert args.target_round == 14
    assert args.mode == "live"
    assert args.budget == 100.0
    assert args.footystats_mode == "ppg"
    assert args.output_root == Path("data/08_reporting/recommendations")


def test_main_builds_recommendation_config(monkeypatch, tmp_path: Path, capsys) -> None:
    observed: list[RecommendationConfig] = []

    def fake_run_recommendation(config: RecommendationConfig) -> RecommendationResult:
        observed.append(config)
        return RecommendationResult(
            recommended_squad=None,
            candidate_predictions=None,
            summary={"predicted_points": 42.0, "output_directory": str(config.output_path)},
            metadata={},
        )

    monkeypatch.setattr("scripts.recommend_squad.run_recommendation", fake_run_recommendation)

    exit_code = main(
        [
            "--season",
            "2026",
            "--target-round",
            "14",
            "--mode",
            "live",
            "--project-root",
            str(tmp_path),
            "--current-year",
            "2026",
        ]
    )

    assert exit_code == 0
    assert observed == [
        RecommendationConfig(
            season=2026,
            target_round=14,
            mode="live",
            project_root=tmp_path,
            current_year=2026,
        )
    ]
    assert "Recommendation complete" in capsys.readouterr().out
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
uv run --frozen pytest src/tests/backtesting/test_recommend_squad_cli.py -q
```

Expected: fail because `scripts/recommend_squad.py` does not exist.

- [ ] **Step 3: Implement CLI script**

Create `scripts/recommend_squad.py`:

```python
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from cartola.backtesting.recommendation import RecommendationConfig, run_recommendation


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a single-round Cartola squad recommendation.")
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--target-round", type=_positive_int, required=True)
    parser.add_argument("--mode", choices=("live", "replay"), required=True)
    parser.add_argument("--budget", type=float, default=100.0)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--output-root", type=Path, default=Path("data/08_reporting/recommendations"))
    parser.add_argument("--footystats-mode", choices=("none", "ppg", "ppg_xg"), default="ppg")
    parser.add_argument("--footystats-league-slug", default="brazil-serie-a")
    parser.add_argument("--footystats-dir", type=Path, default=Path("data/footystats"))
    parser.add_argument("--current-year", type=int, default=None)
    parser.add_argument("--allow-finalized-live-data", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config = RecommendationConfig(
        season=args.season,
        target_round=args.target_round,
        mode=args.mode,
        budget=args.budget,
        project_root=args.project_root,
        output_root=args.output_root,
        footystats_mode=args.footystats_mode,
        footystats_league_slug=args.footystats_league_slug,
        footystats_dir=args.footystats_dir,
        current_year=args.current_year,
        allow_finalized_live_data=args.allow_finalized_live_data,
    )

    result = run_recommendation(config)
    print(
        "Recommendation complete: "
        f"season={config.season} target_round={config.target_round} "
        f"mode={config.mode} predicted_points={result.summary['predicted_points']} "
        f"output={config.output_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run CLI tests**

Run:

```bash
uv run --frozen pytest src/tests/backtesting/test_recommend_squad_cli.py -q
```

Expected: all CLI tests pass.

- [ ] **Step 5: Commit**

```bash
git add scripts/recommend_squad.py src/tests/backtesting/test_recommend_squad_cli.py
git commit -m "feat: add squad recommendation cli"
```

---

### Task 7: Run Real 2026 Replay Smoke Test

**Files:**
- No code edits expected.

- [ ] **Step 1: Run replay for an available 2026 round**

Use round 10 because local data currently includes at least rounds 1-13.

```bash
uv run --frozen python scripts/recommend_squad.py \
  --season 2026 \
  --target-round 10 \
  --mode replay \
  --budget 100 \
  --footystats-mode ppg \
  --current-year 2026
```

Expected: command writes:

```text
data/08_reporting/recommendations/2026/round-10/replay/recommended_squad.csv
data/08_reporting/recommendations/2026/round-10/replay/candidate_predictions.csv
data/08_reporting/recommendations/2026/round-10/replay/recommendation_summary.json
data/08_reporting/recommendations/2026/round-10/replay/run_metadata.json
```

- [ ] **Step 2: Inspect summary and metadata**

Run:

```bash
python - <<'PY'
import json
from pathlib import Path

root = Path("data/08_reporting/recommendations/2026/round-10/replay")
summary = json.loads((root / "recommendation_summary.json").read_text())
metadata = json.loads((root / "run_metadata.json").read_text())
print(summary)
print({
    "training_rounds": metadata["training_rounds"],
    "footystats_mode": metadata["footystats_mode"],
    "visible_max_round": metadata["visible_max_round"],
})
PY
```

Expected:

- `summary["optimizer_status"] == "Optimal"`;
- `summary["actual_points"]` is not `None`;
- `metadata["visible_max_round"] == 10`;
- `metadata["training_rounds"] == [1, 2, 3, 4, 5, 6, 7, 8, 9]`.

- [ ] **Step 3: Do not commit generated recommendation reports**

Run:

```bash
git status --short
```

Expected: generated files under `data/08_reporting/recommendations/` are ignored or absent from git status. If they appear, add this line to `.gitignore` and commit it:

```gitignore
data/08_reporting/recommendations/
```

Commit only if `.gitignore` changes:

```bash
git add .gitignore
git commit -m "chore: ignore recommendation reports"
```

---

### Task 8: Document Live and Replay Commands

**Files:**
- Modify: `README.md`
- Modify: `roadmap.md`

- [ ] **Step 1: Update README command section**

Add this section near the existing backtest commands:

```markdown
### Single-Round Squad Recommendation

Live/current recommendation:

```bash
uv run --frozen python scripts/recommend_squad.py \
  --season 2026 \
  --target-round 14 \
  --mode live \
  --budget 100 \
  --footystats-mode ppg \
  --current-year 2026
```

Leakage-safe historical replay:

```bash
uv run --frozen python scripts/recommend_squad.py \
  --season 2026 \
  --target-round 10 \
  --mode replay \
  --budget 100 \
  --footystats-mode ppg \
  --current-year 2026
```

Live mode writes recommendations without actual-score fields. Replay mode uses the same pre-round prediction path and then reports realized selected-squad points when target-round actuals exist.
```

- [ ] **Step 2: Update roadmap delivered/current section**

Add under Delivered after the FootyStats ablation bullets:

```markdown
- Single-round squad recommendation workflow:
  - `scripts/recommend_squad.py`;
  - explicit `--target-round`;
  - `live` and `replay` modes;
  - target-round data slicing to avoid future-round leakage;
  - live output allowlists that suppress actual/evaluation columns;
  - replay actual-point reporting after optimization.
```

Add under Current Interpretation:

```markdown
The next operational step is to run the recommendation command for each 2026 target round before lineup lock, using `footystats_mode=ppg` and `fixture_mode=none` until strict fixture snapshots are available.
```

- [ ] **Step 3: Commit docs**

```bash
git add README.md roadmap.md
git commit -m "docs: document squad recommendation workflow"
```

---

### Task 9: Full Verification

**Files:**
- No planned code edits.

- [ ] **Step 1: Run focused tests**

Run:

```bash
uv run --frozen pytest \
  src/tests/backtesting/test_recommendation.py \
  src/tests/backtesting/test_recommend_squad_cli.py \
  src/tests/backtesting/test_footystats_features.py \
  -q
```

Expected: all selected tests pass.

- [ ] **Step 2: Run full quality gate**

Run:

```bash
uv run --frozen scripts/pyrepo-check --all
```

Expected:

- Ruff passes.
- ty passes.
- Bandit reports no issues.
- pytest passes.

- [ ] **Step 3: Commit any final fixes**

If Step 2 required fixes:

```bash
git status --short
git add src/cartola/backtesting/recommendation.py src/cartola/backtesting/footystats_features.py scripts/recommend_squad.py src/tests/backtesting/test_recommendation.py src/tests/backtesting/test_recommend_squad_cli.py src/tests/backtesting/test_footystats_features.py README.md roadmap.md .gitignore
git commit -m "fix: stabilize squad recommendation workflow"
```

If Step 2 passed without changes, do not create an empty commit.

---

## Self-Review Notes

- Spec coverage:
  - Target-round visible data boundary: Tasks 2 and 4.
  - Live current-year rule: Task 2.
  - Zero-filled scout handling: Task 2.
  - FootyStats `Game Week <= target_round` slicing: Task 3.
  - Missing/duplicate FootyStats key failures: Task 3 and Task 4.
  - Live/replay output allowlists: Task 4 and Task 5.
  - CLI and output path: Task 6.
  - Real 2026 smoke run: Task 7.
  - Documentation: Task 8.
- No implementation task calls `run_backtest`.
- The plan keeps fixture mode fixed to `none` by not exposing any fixture option in `scripts/recommend_squad.py`.
