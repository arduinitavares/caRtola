# Cartola Offline Backtesting Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Python offline backtesting pipeline that runs a fixed-budget, walk-forward Cartola squad simulation for the 2025 season.

**Architecture:** Add a standalone `cartola.backtesting` package outside Kedro. The package loads and normalizes historical round CSVs, builds leakage-safe features, predicts player points with a baseline and RandomForest model, optimizes legal squads with PuLP, and writes per-round CSV outputs.

**Tech Stack:** Python 3.10, pandas, scikit-learn `RandomForestRegressor`, PuLP, pytest, Poetry.

---

## File Structure

- Create `src/cartola/backtesting/__init__.py`: package marker and public exports.
- Create `src/cartola/backtesting/config.py`: dataclasses and constants for budget, season, statuses, scouts, positions, formations, and output paths.
- Create `src/cartola/backtesting/data.py`: raw CSV loading, unnamed index-column cleanup, column normalization, status/position mapping, per-file scout filling, season concatenation.
- Create `src/cartola/backtesting/features.py`: leakage-safe training and prediction feature frames.
- Create `src/cartola/backtesting/models.py`: baseline predictor and RandomForest predictor behind a common `fit`/`predict` interface.
- Create `src/cartola/backtesting/optimizer.py`: PuLP integer-programming squad optimizer.
- Create `src/cartola/backtesting/metrics.py`: round and season summary helpers.
- Create `src/cartola/backtesting/runner.py`: walk-forward orchestration and CSV output writing.
- Create `src/cartola/backtesting/cli.py`: command line entry point.
- Modify `pyproject.toml`: add `scikit-learn` and `pulp` dependencies.
- Create tests under `src/tests/backtesting/`.

## Task 1: Dependencies, Package Skeleton, And Config

**Files:**
- Modify: `pyproject.toml`
- Create: `src/cartola/backtesting/__init__.py`
- Create: `src/cartola/backtesting/config.py`
- Test: `src/tests/backtesting/test_config.py`

- [ ] **Step 1: Add dependencies**

Run:

```bash
poetry add scikit-learn pulp
```

Expected: `pyproject.toml` and `poetry.lock` update with `scikit-learn` and `pulp`.

- [ ] **Step 2: Write the failing config tests**

Create `src/tests/backtesting/test_config.py`:

```python
from pathlib import Path

from cartola.backtesting.config import (
    BacktestConfig,
    DEFAULT_SCOUT_COLUMNS,
    POSITION_ID_TO_CODE,
    STATUS_ID_TO_NAME,
)


def test_default_config_matches_v1_scope():
    config = BacktestConfig()

    assert config.season == 2025
    assert config.start_round == 5
    assert config.budget == 100.0
    assert config.playable_statuses == ("Provavel",)
    assert config.formation_name == "4-3-3"
    assert config.output_path == Path("data/08_reporting/backtests/2025")


def test_default_mappings_cover_cartola_values():
    assert STATUS_ID_TO_NAME[7] == "Provavel"
    assert STATUS_ID_TO_NAME[2] == "Duvida"
    assert POSITION_ID_TO_CODE[1] == "gol"
    assert POSITION_ID_TO_CODE[2] == "lat"
    assert POSITION_ID_TO_CODE[3] == "zag"
    assert POSITION_ID_TO_CODE[4] == "mei"
    assert POSITION_ID_TO_CODE[5] == "ata"
    assert POSITION_ID_TO_CODE[6] == "tec"


def test_default_scout_columns_include_v():
    assert "V" in DEFAULT_SCOUT_COLUMNS
    assert {"G", "A", "DS", "SG", "CA", "FC"}.issubset(DEFAULT_SCOUT_COLUMNS)
```

- [ ] **Step 3: Run config tests to verify they fail**

Run:

```bash
poetry run pytest src/tests/backtesting/test_config.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'cartola.backtesting'`.

- [ ] **Step 4: Create package skeleton**

Create `src/cartola/backtesting/__init__.py`:

```python
"""Offline Cartola backtesting tools."""

from cartola.backtesting.config import BacktestConfig

__all__ = ["BacktestConfig"]
```

- [ ] **Step 5: Implement config**

Create `src/cartola/backtesting/config.py`:

```python
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

DEFAULT_SCOUT_COLUMNS: tuple[str, ...] = (
    "G",
    "A",
    "DS",
    "SG",
    "CA",
    "FC",
    "FS",
    "FF",
    "FD",
    "FT",
    "I",
    "GS",
    "DE",
    "DP",
    "V",
    "CV",
    "PP",
    "PS",
    "PC",
    "GC",
)

STATUS_ID_TO_NAME: Mapping[int, str] = {
    2: "Duvida",
    3: "Suspenso",
    5: "Contundido",
    6: "Nulo",
    7: "Provavel",
}

POSITION_ID_TO_CODE: Mapping[int, str] = {
    1: "gol",
    2: "lat",
    3: "zag",
    4: "mei",
    5: "ata",
    6: "tec",
}

DEFAULT_FORMATIONS: Mapping[str, Mapping[str, int]] = {
    "4-3-3": {
        "gol": 1,
        "lat": 2,
        "zag": 2,
        "mei": 3,
        "ata": 3,
        "tec": 1,
    }
}


@dataclass(frozen=True)
class BacktestConfig:
    season: int = 2025
    start_round: int = 5
    budget: float = 100.0
    playable_statuses: tuple[str, ...] = ("Provavel",)
    formation_name: str = "4-3-3"
    random_seed: int = 123
    project_root: Path = Path(".")
    output_root: Path = Path("data/08_reporting/backtests")
    scout_columns: tuple[str, ...] = DEFAULT_SCOUT_COLUMNS
    formations: Mapping[str, Mapping[str, int]] = field(default_factory=lambda: DEFAULT_FORMATIONS)

    @property
    def raw_season_path(self) -> Path:
        return self.project_root / "data" / "01_raw" / str(self.season)

    @property
    def output_path(self) -> Path:
        return self.project_root / self.output_root / str(self.season)

    @property
    def selected_formation(self) -> Mapping[str, int]:
        if self.formation_name not in self.formations:
            raise ValueError(f"Unknown formation {self.formation_name!r}. Available: {sorted(self.formations)}")
        return self.formations[self.formation_name]
```

- [ ] **Step 6: Run config tests to verify they pass**

Run:

```bash
poetry run pytest src/tests/backtesting/test_config.py -v
```

Expected: PASS.

- [ ] **Step 7: Commit**

Run:

```bash
git add pyproject.toml poetry.lock src/cartola/backtesting src/tests/backtesting/test_config.py
git commit -m "feat: add backtesting config"
```

## Task 2: Data Loader And Normalization

**Files:**
- Create: `src/cartola/backtesting/data.py`
- Test: `src/tests/backtesting/test_data.py`

- [ ] **Step 1: Write failing data loader tests**

Create `src/tests/backtesting/test_data.py`:

```python
from pathlib import Path

import pandas as pd
import pytest

from cartola.backtesting.data import load_round_file, load_season_data, normalize_round_frame


def test_normalize_round_frame_drops_index_maps_columns_and_fills_scouts():
    raw = pd.DataFrame(
        {
            "Unnamed: 0": [0],
            "atletas.rodada_id": [1],
            "atletas.status_id": [7],
            "atletas.posicao_id": [5],
            "atletas.atleta_id": [10],
            "atletas.apelido": ["Player"],
            "atletas.slug": ["player"],
            "atletas.clube_id": [100],
            "atletas.clube.id.full.name": ["Club"],
            "atletas.preco_num": [12.3],
            "atletas.pontos_num": [4.5],
            "atletas.media_num": [4.5],
            "atletas.jogos_num": [1],
            "atletas.variacao_num": [0.2],
            "atletas.entrou_em_campo": [True],
        }
    )

    normalized = normalize_round_frame(raw, source=Path("rodada-1.csv"))

    assert "Unnamed: 0" not in normalized.columns
    assert normalized.loc[0, "status"] == "Provavel"
    assert normalized.loc[0, "posicao"] == "ata"
    assert normalized.loc[0, "id_atleta"] == 10
    assert normalized.loc[0, "V"] == 0
    assert normalized.loc[0, "G"] == 0
    assert bool(normalized.loc[0, "entrou_em_campo"]) is True


def test_normalize_round_frame_rejects_unknown_status():
    raw = pd.DataFrame(
        {
            "atletas.rodada_id": [1],
            "atletas.status_id": [99],
            "atletas.posicao_id": [5],
            "atletas.atleta_id": [10],
            "atletas.apelido": ["Player"],
            "atletas.clube_id": [100],
            "atletas.preco_num": [12.3],
            "atletas.pontos_num": [4.5],
            "atletas.media_num": [4.5],
            "atletas.jogos_num": [1],
            "atletas.variacao_num": [0.2],
        }
    )

    with pytest.raises(ValueError, match="Unknown status_id values"):
        normalize_round_frame(raw, source=Path("rodada-1.csv"))


def test_load_season_data_orders_rounds_numerically(tmp_path):
    season_dir = tmp_path / "data" / "01_raw" / "2025"
    season_dir.mkdir(parents=True)
    base = {
        "atletas.status_id": [7],
        "atletas.posicao_id": [5],
        "atletas.atleta_id": [10],
        "atletas.apelido": ["Player"],
        "atletas.clube_id": [100],
        "atletas.preco_num": [12.3],
        "atletas.pontos_num": [4.5],
        "atletas.media_num": [4.5],
        "atletas.jogos_num": [1],
        "atletas.variacao_num": [0.2],
    }
    pd.DataFrame({**base, "atletas.rodada_id": [10]}).to_csv(season_dir / "rodada-10.csv", index=False)
    pd.DataFrame({**base, "atletas.rodada_id": [2]}).to_csv(season_dir / "rodada-2.csv", index=False)

    loaded = load_season_data(2025, project_root=tmp_path)

    assert loaded["rodada"].tolist() == [2, 10]
```

- [ ] **Step 2: Run data tests to verify they fail**

Run:

```bash
poetry run pytest src/tests/backtesting/test_data.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'cartola.backtesting.data'`.

- [ ] **Step 3: Implement data loader**

Create `src/cartola/backtesting/data.py`:

```python
from __future__ import annotations

import re
import unicodedata
from pathlib import Path

import pandas as pd

from cartola.backtesting.config import DEFAULT_SCOUT_COLUMNS, POSITION_ID_TO_CODE, STATUS_ID_TO_NAME

COLUMN_MAP = {
    "atletas.atleta_id": "id_atleta",
    "atletas.apelido": "apelido",
    "atletas.slug": "slug",
    "atletas.clube_id": "id_clube",
    "atletas.clube.id.full.name": "nome_clube",
    "atletas.posicao_id": "posicao",
    "atletas.status_id": "status",
    "atletas.rodada_id": "rodada",
    "atletas.preco_num": "preco",
    "atletas.pontos_num": "pontuacao",
    "atletas.media_num": "media",
    "atletas.jogos_num": "num_jogos",
    "atletas.variacao_num": "variacao",
    "atletas.entrou_em_campo": "entrou_em_campo",
}

REQUIRED_COLUMNS = (
    "id_atleta",
    "apelido",
    "id_clube",
    "posicao",
    "status",
    "rodada",
    "preco",
    "pontuacao",
    "media",
    "num_jogos",
    "variacao",
)


def normalize_status(value: object) -> str:
    if pd.isna(value):
        return "Nulo"
    if isinstance(value, str):
        normalized = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")
        return normalized
    return STATUS_ID_TO_NAME[int(value)]


def normalize_position(value: object) -> str:
    if isinstance(value, str) and value in set(POSITION_ID_TO_CODE.values()):
        return value
    return POSITION_ID_TO_CODE[int(value)]


def _drop_saved_index_columns(df: pd.DataFrame) -> pd.DataFrame:
    columns_to_drop = [col for col in df.columns if str(col).startswith("Unnamed:") or str(col).strip() == ""]
    return df.drop(columns=columns_to_drop)


def normalize_round_frame(
    df: pd.DataFrame,
    source: Path,
    scout_columns: tuple[str, ...] = DEFAULT_SCOUT_COLUMNS,
) -> pd.DataFrame:
    normalized = _drop_saved_index_columns(df).rename(columns=COLUMN_MAP).copy()

    missing_required = [col for col in REQUIRED_COLUMNS if col not in normalized.columns]
    if missing_required:
        raise ValueError(f"{source}: missing required columns {missing_required}")

    unknown_statuses = sorted(set(normalized["status"].dropna().astype(int)) - set(STATUS_ID_TO_NAME))
    if unknown_statuses:
        raise ValueError(f"{source}: Unknown status_id values {unknown_statuses}")

    unknown_positions = sorted(set(normalized["posicao"].dropna().astype(int)) - set(POSITION_ID_TO_CODE))
    if unknown_positions:
        raise ValueError(f"{source}: Unknown posicao_id values {unknown_positions}")

    normalized["status"] = normalized["status"].map(normalize_status)
    normalized["posicao"] = normalized["posicao"].map(normalize_position)

    if "slug" not in normalized.columns:
        normalized["slug"] = pd.NA
    if "nome_clube" not in normalized.columns:
        normalized["nome_clube"] = pd.NA
    if "entrou_em_campo" not in normalized.columns:
        normalized["entrou_em_campo"] = normalized["pontuacao"].notna()

    for scout in scout_columns:
        if scout not in normalized.columns:
            normalized[scout] = 0
        normalized[scout] = normalized[scout].fillna(0)

    for numeric_col in ("id_atleta", "id_clube", "rodada", "preco", "pontuacao", "media", "num_jogos", "variacao"):
        normalized[numeric_col] = pd.to_numeric(normalized[numeric_col], errors="coerce")

    normalized["entrou_em_campo"] = normalized["entrou_em_campo"].fillna(False).astype(bool)
    return normalized


def _round_number(path: Path) -> int:
    match = re.search(r"rodada-(\d+)\.csv$", path.name)
    if not match:
        raise ValueError(f"Unexpected round filename: {path.name}")
    return int(match.group(1))


def load_round_file(path: Path, scout_columns: tuple[str, ...] = DEFAULT_SCOUT_COLUMNS) -> pd.DataFrame:
    return normalize_round_frame(pd.read_csv(path), source=path, scout_columns=scout_columns)


def load_season_data(
    season: int,
    project_root: Path = Path("."),
    scout_columns: tuple[str, ...] = DEFAULT_SCOUT_COLUMNS,
) -> pd.DataFrame:
    season_dir = project_root / "data" / "01_raw" / str(season)
    if not season_dir.exists():
        raise FileNotFoundError(f"Missing season directory: {season_dir}")

    files = sorted(season_dir.glob("rodada-*.csv"), key=_round_number)
    if not files:
        raise FileNotFoundError(f"No rodada-*.csv files found in {season_dir}")

    frames = [load_round_file(path, scout_columns=scout_columns) for path in files]
    return pd.concat(frames, ignore_index=True).sort_values(["rodada", "id_atleta"], ignore_index=True)
```

- [ ] **Step 4: Run data tests to verify they pass**

Run:

```bash
poetry run pytest src/tests/backtesting/test_data.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

Run:

```bash
git add src/cartola/backtesting/data.py src/tests/backtesting/test_data.py
git commit -m "feat: add backtesting data loader"
```

## Task 3: Leakage-Safe Feature Builder And Baseline Predictor

**Files:**
- Create: `src/cartola/backtesting/features.py`
- Create: `src/cartola/backtesting/models.py`
- Test: `src/tests/backtesting/test_features.py`
- Test: `src/tests/backtesting/test_models.py`

- [ ] **Step 1: Write failing feature tests**

Create `src/tests/backtesting/test_features.py`:

```python
import pandas as pd

from cartola.backtesting.features import FEATURE_COLUMNS, build_prediction_frame, build_training_frame


def _season_df():
    return pd.DataFrame(
        [
            {"rodada": 1, "id_atleta": 1, "apelido": "A", "posicao": "ata", "status": "Provavel", "preco": 10, "pontuacao": 2, "media": 2, "num_jogos": 1, "variacao": 0, "id_clube": 10, "entrou_em_campo": True, "G": 0, "A": 0, "DS": 1, "V": 0},
            {"rodada": 2, "id_atleta": 1, "apelido": "A", "posicao": "ata", "status": "Provavel", "preco": 11, "pontuacao": 8, "media": 5, "num_jogos": 2, "variacao": 1, "id_clube": 10, "entrou_em_campo": True, "G": 1, "A": 0, "DS": 0, "V": 0},
            {"rodada": 3, "id_atleta": 1, "apelido": "A", "posicao": "ata", "status": "Provavel", "preco": 12, "pontuacao": 100, "media": 36.7, "num_jogos": 3, "variacao": 1, "id_clube": 10, "entrou_em_campo": True, "G": 5, "A": 5, "DS": 0, "V": 0},
            {"rodada": 1, "id_atleta": 2, "apelido": "B", "posicao": "mei", "status": "Provavel", "preco": 8, "pontuacao": 4, "media": 4, "num_jogos": 1, "variacao": 0, "id_clube": 20, "entrou_em_campo": True, "G": 0, "A": 1, "DS": 1, "V": 1},
            {"rodada": 2, "id_atleta": 2, "apelido": "B", "posicao": "mei", "status": "Provavel", "preco": 9, "pontuacao": 6, "media": 5, "num_jogos": 2, "variacao": 1, "id_clube": 20, "entrou_em_campo": True, "G": 0, "A": 0, "DS": 2, "V": 1},
            {"rodada": 3, "id_atleta": 2, "apelido": "B", "posicao": "mei", "status": "Provavel", "preco": 10, "pontuacao": 7, "media": 5.7, "num_jogos": 3, "variacao": 1, "id_clube": 20, "entrou_em_campo": True, "G": 0, "A": 1, "DS": 2, "V": 2},
        ]
    )


def test_prediction_features_use_only_prior_rounds():
    frame = build_prediction_frame(_season_df(), target_round=3)
    player = frame.loc[frame["id_atleta"] == 1].iloc[0]

    assert player["prior_points_mean"] == 5
    assert player["prior_points_roll3"] == 5
    assert player["prior_G_mean"] == 0.5
    assert player["pontuacao"] == 100


def test_training_frame_excludes_target_round_from_feature_history():
    frame = build_training_frame(_season_df(), target_round=3)
    player_round_2 = frame[(frame["id_atleta"] == 1) & (frame["rodada"] == 2)].iloc[0]

    assert player_round_2["prior_points_mean"] == 2
    assert player_round_2["target"] == 8


def test_feature_columns_exist_in_prediction_frame():
    frame = build_prediction_frame(_season_df(), target_round=3)

    for column in FEATURE_COLUMNS:
        assert column in frame.columns
```

- [ ] **Step 2: Write failing baseline model tests**

Create `src/tests/backtesting/test_models.py`:

```python
import pandas as pd

from cartola.backtesting.models import BaselinePredictor


def test_baseline_predictor_uses_prior_player_mean_with_position_fallback():
    train = pd.DataFrame(
        {
            "id_atleta": [1, 2],
            "posicao": ["ata", "mei"],
            "target": [6.0, 4.0],
            "prior_points_mean": [5.0, 3.0],
        }
    )
    predict = pd.DataFrame(
        {
            "id_atleta": [1, 3],
            "posicao": ["ata", "mei"],
            "prior_points_mean": [5.0, None],
        }
    )

    model = BaselinePredictor().fit(train)
    predictions = model.predict(predict)

    assert predictions.tolist() == [5.0, 4.0]
```

- [ ] **Step 3: Run feature and model tests to verify they fail**

Run:

```bash
poetry run pytest src/tests/backtesting/test_features.py src/tests/backtesting/test_models.py -v
```

Expected: FAIL with missing `features` and `models` modules.

- [ ] **Step 4: Implement features**

Create `src/cartola/backtesting/features.py`:

```python
from __future__ import annotations

import pandas as pd

from cartola.backtesting.config import DEFAULT_SCOUT_COLUMNS

MARKET_COLUMNS = [
    "id_atleta",
    "apelido",
    "slug",
    "id_clube",
    "nome_clube",
    "posicao",
    "status",
    "rodada",
    "preco",
    "pontuacao",
    "media",
    "num_jogos",
    "variacao",
    "entrou_em_campo",
]

FEATURE_COLUMNS = [
    "preco",
    "media",
    "num_jogos",
    "prior_appearances",
    "prior_points_mean",
    "prior_points_roll3",
    "prior_points_roll5",
    "prior_price_mean",
    "prior_variation_mean",
    "id_clube",
    "rodada",
    "posicao",
] + [f"prior_{scout}_mean" for scout in DEFAULT_SCOUT_COLUMNS]


def _played_history(df: pd.DataFrame, target_round: int) -> pd.DataFrame:
    history = df[df["rodada"] < target_round].copy()
    if "entrou_em_campo" in history.columns:
        history = history[history["entrou_em_campo"]]
    return history


def _last_n_mean(values: pd.Series, n: int) -> float:
    return float(values.tail(n).mean())


def _player_history_features(history: pd.DataFrame) -> pd.DataFrame:
    if history.empty:
        return pd.DataFrame(columns=["id_atleta"])

    ordered = history.sort_values(["id_atleta", "rodada"])
    grouped = ordered.groupby("id_atleta", as_index=False)
    features = grouped.agg(
        prior_appearances=("rodada", "count"),
        prior_points_mean=("pontuacao", "mean"),
        prior_price_mean=("preco", "mean"),
        prior_variation_mean=("variacao", "mean"),
    )
    roll3 = (
        ordered.groupby("id_atleta")["pontuacao"]
        .apply(lambda values: _last_n_mean(values, 3))
        .reset_index(name="prior_points_roll3")
    )
    roll5 = (
        ordered.groupby("id_atleta")["pontuacao"]
        .apply(lambda values: _last_n_mean(values, 5))
        .reset_index(name="prior_points_roll5")
    )
    features = features.merge(roll3, on="id_atleta", how="left").merge(roll5, on="id_atleta", how="left")

    scout_means = ordered.groupby("id_atleta", as_index=False).agg(
        **{f"prior_{scout}_mean": (scout, "mean") for scout in DEFAULT_SCOUT_COLUMNS if scout in ordered.columns}
    )
    return features.merge(scout_means, on="id_atleta", how="left")


def _position_priors(history: pd.DataFrame) -> pd.DataFrame:
    if history.empty:
        return pd.DataFrame(columns=["posicao", "position_points_prior"])
    return history.groupby("posicao", as_index=False).agg(position_points_prior=("pontuacao", "mean"))


def build_prediction_frame(season_df: pd.DataFrame, target_round: int) -> pd.DataFrame:
    candidates = season_df[season_df["rodada"] == target_round].copy()
    history = _played_history(season_df, target_round)
    features = _player_history_features(history)
    position_priors = _position_priors(history)

    frame = candidates.merge(features, on="id_atleta", how="left").merge(position_priors, on="posicao", how="left")
    global_prior = float(history["pontuacao"].mean()) if not history.empty else 0.0
    frame["position_points_prior"] = frame["position_points_prior"].fillna(global_prior)
    frame["prior_points_mean"] = frame["prior_points_mean"].fillna(frame["position_points_prior"])
    frame["prior_points_roll3"] = frame["prior_points_roll3"].fillna(frame["prior_points_mean"])
    frame["prior_points_roll5"] = frame["prior_points_roll5"].fillna(frame["prior_points_mean"])
    frame["prior_appearances"] = frame["prior_appearances"].fillna(0)
    frame["prior_price_mean"] = frame["prior_price_mean"].fillna(frame["preco"])
    frame["prior_variation_mean"] = frame["prior_variation_mean"].fillna(0)

    for scout in DEFAULT_SCOUT_COLUMNS:
        column = f"prior_{scout}_mean"
        if column not in frame.columns:
            frame[column] = 0.0
        frame[column] = frame[column].fillna(0.0)

    return frame


def build_training_frame(season_df: pd.DataFrame, target_round: int) -> pd.DataFrame:
    frames = []
    for round_number in sorted(round_ for round_ in season_df["rodada"].dropna().unique() if 1 < round_ < target_round):
        prediction_frame = build_prediction_frame(season_df, int(round_number))
        prediction_frame["target"] = prediction_frame["pontuacao"]
        frames.append(prediction_frame)
    if not frames:
        return pd.DataFrame(columns=MARKET_COLUMNS + FEATURE_COLUMNS + ["target"])
    return pd.concat(frames, ignore_index=True)
```

- [ ] **Step 5: Implement baseline and RandomForest model wrapper**

Create `src/cartola/backtesting/models.py`:

```python
from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from cartola.backtesting.features import FEATURE_COLUMNS


class BaselinePredictor:
    def __init__(self) -> None:
        self.player_means_: pd.Series | None = None
        self.position_means_: pd.Series | None = None
        self.global_mean_: float = 0.0

    def fit(self, frame: pd.DataFrame) -> "BaselinePredictor":
        self.player_means_ = frame.groupby("id_atleta")["target"].mean()
        self.position_means_ = frame.groupby("posicao")["target"].mean()
        self.global_mean_ = float(frame["target"].mean()) if not frame.empty else 0.0
        return self

    def predict(self, frame: pd.DataFrame) -> pd.Series:
        if self.player_means_ is None or self.position_means_ is None:
            raise RuntimeError("BaselinePredictor must be fitted before predict().")
        predictions = frame["id_atleta"].map(self.player_means_)
        position_fallback = frame["posicao"].map(self.position_means_).fillna(self.global_mean_)
        predictions = predictions.fillna(frame.get("prior_points_mean")).fillna(position_fallback)
        return predictions.astype(float)


class RandomForestPointPredictor:
    def __init__(self, random_seed: int = 123) -> None:
        numeric_features = [column for column in FEATURE_COLUMNS if column != "posicao"]
        categorical_features = ["posicao"]
        self.pipeline = Pipeline(
            steps=[
                (
                    "preprocess",
                    ColumnTransformer(
                        transformers=[
                            ("numeric", SimpleImputer(strategy="median"), numeric_features),
                            ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_features),
                        ]
                    ),
                ),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=200,
                        min_samples_leaf=3,
                        random_state=random_seed,
                        n_jobs=-1,
                    ),
                ),
            ]
        )

    def fit(self, frame: pd.DataFrame) -> "RandomForestPointPredictor":
        self.pipeline.fit(frame[FEATURE_COLUMNS], frame["target"])
        return self

    def predict(self, frame: pd.DataFrame) -> pd.Series:
        return pd.Series(self.pipeline.predict(frame[FEATURE_COLUMNS]), index=frame.index, dtype=float)
```

- [ ] **Step 6: Run feature and model tests to verify they pass**

Run:

```bash
poetry run pytest src/tests/backtesting/test_features.py src/tests/backtesting/test_models.py -v
```

Expected: PASS.

- [ ] **Step 7: Commit**

Run:

```bash
git add src/cartola/backtesting/features.py src/cartola/backtesting/models.py src/tests/backtesting/test_features.py src/tests/backtesting/test_models.py
git commit -m "feat: add backtesting features and models"
```

## Task 4: Squad Optimizer

**Files:**
- Create: `src/cartola/backtesting/optimizer.py`
- Test: `src/tests/backtesting/test_optimizer.py`

- [ ] **Step 1: Write failing optimizer tests**

Create `src/tests/backtesting/test_optimizer.py`:

```python
import pandas as pd

from cartola.backtesting.config import BacktestConfig
from cartola.backtesting.optimizer import optimize_squad


def _candidates():
    rows = []
    player_id = 1
    for pos, count in {"gol": 2, "lat": 3, "zag": 3, "mei": 4, "ata": 4, "tec": 2}.items():
        for offset in range(count):
            rows.append(
                {
                    "id_atleta": player_id,
                    "apelido": f"{pos}-{offset}",
                    "posicao": pos,
                    "preco": 5.0 + offset,
                    "predicted_points": 10.0 - offset,
                    "pontuacao": 7.0 - offset,
                }
            )
            player_id += 1
    return pd.DataFrame(rows)


def test_optimizer_selects_legal_433_squad_under_budget():
    result = optimize_squad(_candidates(), score_column="predicted_points", config=BacktestConfig(budget=80))

    assert result.status == "Optimal"
    assert result.selected_count == 12
    assert result.selected.groupby("posicao").size().to_dict() == {
        "ata": 3,
        "gol": 1,
        "lat": 2,
        "mei": 3,
        "tec": 1,
        "zag": 2,
    }
    assert result.budget_used <= 80


def test_optimizer_reports_infeasible_budget():
    result = optimize_squad(_candidates(), score_column="predicted_points", config=BacktestConfig(budget=1))

    assert result.status == "Infeasible"
    assert result.selected.empty
```

- [ ] **Step 2: Run optimizer tests to verify they fail**

Run:

```bash
poetry run pytest src/tests/backtesting/test_optimizer.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'cartola.backtesting.optimizer'`.

- [ ] **Step 3: Implement optimizer**

Create `src/cartola/backtesting/optimizer.py`:

```python
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import pulp

from cartola.backtesting.config import BacktestConfig


@dataclass(frozen=True)
class OptimizerResult:
    selected: pd.DataFrame
    status: str
    budget_used: float
    predicted_points: float
    actual_points: float
    formation_name: str
    selected_count: int


def optimize_squad(candidates: pd.DataFrame, score_column: str, config: BacktestConfig) -> OptimizerResult:
    if candidates.empty:
        return OptimizerResult(pd.DataFrame(), "Empty", 0.0, 0.0, 0.0, config.formation_name, 0)

    frame = candidates.drop_duplicates("id_atleta").reset_index(drop=True).copy()
    problem = pulp.LpProblem("cartola_squad", pulp.LpMaximize)
    variables = {idx: pulp.LpVariable(f"player_{idx}", cat="Binary") for idx in frame.index}

    problem += pulp.lpSum(float(frame.loc[idx, score_column]) * variables[idx] for idx in frame.index)
    problem += pulp.lpSum(float(frame.loc[idx, "preco"]) * variables[idx] for idx in frame.index) <= config.budget

    formation = config.selected_formation
    for position, count in formation.items():
        problem += (
            pulp.lpSum(variables[idx] for idx in frame.index if frame.loc[idx, "posicao"] == position) == count,
            f"position_{position}",
        )

    status_code = problem.solve(pulp.PULP_CBC_CMD(msg=False))
    status = pulp.LpStatus[status_code]
    if status != "Optimal":
        return OptimizerResult(pd.DataFrame(), status, 0.0, 0.0, 0.0, config.formation_name, 0)

    selected_indices = [idx for idx in frame.index if pulp.value(variables[idx]) == 1]
    selected = frame.loc[selected_indices].copy()
    budget_used = float(selected["preco"].sum())
    predicted_points = float(selected[score_column].sum())
    actual_points = float(selected["pontuacao"].sum()) if "pontuacao" in selected.columns else 0.0

    return OptimizerResult(
        selected=selected,
        status=status,
        budget_used=budget_used,
        predicted_points=predicted_points,
        actual_points=actual_points,
        formation_name=config.formation_name,
        selected_count=len(selected),
    )
```

- [ ] **Step 4: Run optimizer tests to verify they pass**

Run:

```bash
poetry run pytest src/tests/backtesting/test_optimizer.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

Run:

```bash
git add src/cartola/backtesting/optimizer.py src/tests/backtesting/test_optimizer.py
git commit -m "feat: add Cartola squad optimizer"
```

## Task 5: Metrics And Output Records

**Files:**
- Create: `src/cartola/backtesting/metrics.py`
- Test: `src/tests/backtesting/test_metrics.py`

- [ ] **Step 1: Write failing metrics tests**

Create `src/tests/backtesting/test_metrics.py`:

```python
import pandas as pd

from cartola.backtesting.metrics import build_summary


def test_build_summary_computes_strategy_totals_and_benchmark_delta():
    round_results = pd.DataFrame(
        [
            {"strategy": "model", "rodada": 5, "actual_points": 50.0, "predicted_points": 55.0, "solver_status": "Optimal"},
            {"strategy": "model", "rodada": 6, "actual_points": 60.0, "predicted_points": 58.0, "solver_status": "Optimal"},
            {"strategy": "price", "rodada": 5, "actual_points": 45.0, "predicted_points": 45.0, "solver_status": "Optimal"},
            {"strategy": "price", "rodada": 6, "actual_points": 50.0, "predicted_points": 50.0, "solver_status": "Optimal"},
        ]
    )

    summary = build_summary(round_results, benchmark_strategy="price")
    model = summary[summary["strategy"] == "model"].iloc[0]

    assert model["rounds"] == 2
    assert model["total_actual_points"] == 110.0
    assert model["average_actual_points"] == 55.0
    assert model["actual_points_delta_vs_price"] == 15.0
```

- [ ] **Step 2: Run metrics tests to verify they fail**

Run:

```bash
poetry run pytest src/tests/backtesting/test_metrics.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'cartola.backtesting.metrics'`.

- [ ] **Step 3: Implement metrics**

Create `src/cartola/backtesting/metrics.py`:

```python
from __future__ import annotations

import pandas as pd


def build_summary(round_results: pd.DataFrame, benchmark_strategy: str = "price") -> pd.DataFrame:
    if round_results.empty:
        return pd.DataFrame(
            columns=[
                "strategy",
                "rounds",
                "total_actual_points",
                "average_actual_points",
                "total_predicted_points",
                f"actual_points_delta_vs_{benchmark_strategy}",
            ]
        )

    successful = round_results[round_results["solver_status"] == "Optimal"].copy()
    grouped = successful.groupby("strategy", as_index=False).agg(
        rounds=("rodada", "nunique"),
        total_actual_points=("actual_points", "sum"),
        average_actual_points=("actual_points", "mean"),
        total_predicted_points=("predicted_points", "sum"),
    )

    benchmark_rows = grouped[grouped["strategy"] == benchmark_strategy]
    benchmark_total = float(benchmark_rows["total_actual_points"].iloc[0]) if not benchmark_rows.empty else 0.0
    grouped[f"actual_points_delta_vs_{benchmark_strategy}"] = grouped["total_actual_points"] - benchmark_total
    return grouped.sort_values("total_actual_points", ascending=False, ignore_index=True)
```

- [ ] **Step 4: Run metrics tests to verify they pass**

Run:

```bash
poetry run pytest src/tests/backtesting/test_metrics.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

Run:

```bash
git add src/cartola/backtesting/metrics.py src/tests/backtesting/test_metrics.py
git commit -m "feat: add backtesting metrics"
```

## Task 6: Walk-Forward Runner

**Files:**
- Create: `src/cartola/backtesting/runner.py`
- Test: `src/tests/backtesting/test_runner.py`

- [ ] **Step 1: Write failing runner test**

Create `src/tests/backtesting/test_runner.py`:

```python
import pandas as pd

from cartola.backtesting.config import BacktestConfig
from cartola.backtesting.runner import run_backtest


def _tiny_round(round_number: int) -> pd.DataFrame:
    rows = []
    player_id = 1
    for pos, count in {"gol": 2, "lat": 3, "zag": 3, "mei": 4, "ata": 4, "tec": 2}.items():
        for offset in range(count):
            rows.append(
                {
                    "id_atleta": player_id,
                    "apelido": f"{pos}-{offset}",
                    "slug": f"{pos}-{offset}",
                    "id_clube": player_id,
                    "nome_clube": "Club",
                    "posicao": pos,
                    "status": "Provavel",
                    "rodada": round_number,
                    "preco": 5.0,
                    "pontuacao": float(10 - offset + round_number),
                    "media": float(5 + offset),
                    "num_jogos": round_number,
                    "variacao": 0.0,
                    "entrou_em_campo": True,
                    "G": 0,
                    "A": 0,
                    "DS": 0,
                    "V": 0,
                }
            )
            player_id += 1
    return pd.DataFrame(rows)


def test_run_backtest_writes_round_players_predictions_and_summary(tmp_path):
    season_df = pd.concat([_tiny_round(round_number) for round_number in range(1, 6)], ignore_index=True)
    config = BacktestConfig(project_root=tmp_path, start_round=5, budget=100)

    result = run_backtest(config, season_df=season_df)

    assert set(result.round_results["strategy"]) == {"baseline", "random_forest", "price"}
    assert (tmp_path / "data/08_reporting/backtests/2025/round_results.csv").exists()
    assert (tmp_path / "data/08_reporting/backtests/2025/selected_players.csv").exists()
    assert (tmp_path / "data/08_reporting/backtests/2025/player_predictions.csv").exists()
    assert (tmp_path / "data/08_reporting/backtests/2025/summary.csv").exists()
```

- [ ] **Step 2: Run runner test to verify it fails**

Run:

```bash
poetry run pytest src/tests/backtesting/test_runner.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'cartola.backtesting.runner'`.

- [ ] **Step 3: Implement runner**

Create `src/cartola/backtesting/runner.py`:

```python
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from cartola.backtesting.config import BacktestConfig
from cartola.backtesting.data import load_season_data
from cartola.backtesting.features import build_prediction_frame, build_training_frame
from cartola.backtesting.metrics import build_summary
from cartola.backtesting.models import BaselinePredictor, RandomForestPointPredictor
from cartola.backtesting.optimizer import OptimizerResult, optimize_squad


@dataclass(frozen=True)
class BacktestResult:
    round_results: pd.DataFrame
    selected_players: pd.DataFrame
    player_predictions: pd.DataFrame
    summary: pd.DataFrame


def _round_result_record(round_number: int, strategy: str, result: OptimizerResult) -> dict[str, object]:
    return {
        "rodada": round_number,
        "strategy": strategy,
        "solver_status": result.status,
        "formation": result.formation_name,
        "selected_count": result.selected_count,
        "budget_used": result.budget_used,
        "predicted_points": result.predicted_points,
        "actual_points": result.actual_points,
    }


def _selected_players_frame(round_number: int, strategy: str, result: OptimizerResult) -> pd.DataFrame:
    if result.selected.empty:
        return pd.DataFrame()
    selected = result.selected.copy()
    selected["rodada"] = round_number
    selected["strategy"] = strategy
    return selected


def run_backtest(config: BacktestConfig, season_df: pd.DataFrame | None = None) -> BacktestResult:
    data = season_df.copy() if season_df is not None else load_season_data(config.season, project_root=config.project_root)
    max_round = int(data["rodada"].max())
    round_records: list[dict[str, object]] = []
    selected_frames: list[pd.DataFrame] = []
    prediction_frames: list[pd.DataFrame] = []

    for round_number in range(config.start_round, max_round + 1):
        training = build_training_frame(data, round_number)
        candidates = build_prediction_frame(data, round_number)
        candidates = candidates[candidates["status"].isin(config.playable_statuses)].copy()
        if candidates.empty or training.empty:
            continue

        baseline = BaselinePredictor().fit(training)
        random_forest = RandomForestPointPredictor(random_seed=config.random_seed).fit(training)

        candidates["baseline_score"] = baseline.predict(candidates)
        candidates["random_forest_score"] = random_forest.predict(candidates)
        candidates["price_score"] = candidates["preco"].astype(float)

        prediction_export = candidates.copy()
        prediction_export["target_round"] = round_number
        prediction_frames.append(prediction_export)

        strategies = {
            "baseline": "baseline_score",
            "random_forest": "random_forest_score",
            "price": "price_score",
        }
        for strategy, score_column in strategies.items():
            strategy_candidates = candidates.rename(columns={score_column: "predicted_points"}).copy()
            result = optimize_squad(strategy_candidates, score_column="predicted_points", config=config)
            round_records.append(_round_result_record(round_number, strategy, result))
            selected = _selected_players_frame(round_number, strategy, result)
            if not selected.empty:
                selected_frames.append(selected)

    round_results = pd.DataFrame(round_records)
    selected_players = pd.concat(selected_frames, ignore_index=True) if selected_frames else pd.DataFrame()
    player_predictions = pd.concat(prediction_frames, ignore_index=True) if prediction_frames else pd.DataFrame()
    summary = build_summary(round_results, benchmark_strategy="price")

    config.output_path.mkdir(parents=True, exist_ok=True)
    round_results.to_csv(config.output_path / "round_results.csv", index=False)
    selected_players.to_csv(config.output_path / "selected_players.csv", index=False)
    player_predictions.to_csv(config.output_path / "player_predictions.csv", index=False)
    summary.to_csv(config.output_path / "summary.csv", index=False)

    return BacktestResult(
        round_results=round_results,
        selected_players=selected_players,
        player_predictions=player_predictions,
        summary=summary,
    )
```

- [ ] **Step 4: Run runner test to verify it passes**

Run:

```bash
poetry run pytest src/tests/backtesting/test_runner.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

Run:

```bash
git add src/cartola/backtesting/runner.py src/tests/backtesting/test_runner.py
git commit -m "feat: add walk-forward backtest runner"
```

## Task 7: CLI And End-To-End 2025 Backtest

**Files:**
- Create: `src/cartola/backtesting/cli.py`
- Test: `src/tests/backtesting/test_cli.py`
- Modify: `README.md`

- [ ] **Step 1: Write failing CLI test**

Create `src/tests/backtesting/test_cli.py`:

```python
from cartola.backtesting.cli import parse_args


def test_parse_args_accepts_v1_options():
    args = parse_args(["--season", "2025", "--start-round", "5", "--budget", "100"])

    assert args.season == 2025
    assert args.start_round == 5
    assert args.budget == 100.0
```

- [ ] **Step 2: Run CLI test to verify it fails**

Run:

```bash
poetry run pytest src/tests/backtesting/test_cli.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'cartola.backtesting.cli'`.

- [ ] **Step 3: Implement CLI**

Create `src/cartola/backtesting/cli.py`:

```python
from __future__ import annotations

import argparse
from pathlib import Path

from cartola.backtesting.config import BacktestConfig
from cartola.backtesting.runner import run_backtest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run an offline Cartola walk-forward backtest.")
    parser.add_argument("--season", type=int, default=2025)
    parser.add_argument("--start-round", type=int, default=5)
    parser.add_argument("--budget", type=float, default=100.0)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    config = BacktestConfig(
        season=args.season,
        start_round=args.start_round,
        budget=args.budget,
        project_root=args.project_root,
    )
    result = run_backtest(config)
    print(f"Wrote {len(result.round_results)} strategy-round results to {config.output_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run CLI test to verify it passes**

Run:

```bash
poetry run pytest src/tests/backtesting/test_cli.py -v
```

Expected: PASS.

- [ ] **Step 5: Add README usage section**

Modify `README.md` after the "Dados" section:

````markdown
## Backtesting Python

Run the offline fixed-budget walk-forward backtest for the 2025 season:

```bash
poetry run python -m cartola.backtesting.cli --season 2025 --start-round 5 --budget 100
```

Outputs are written to `data/08_reporting/backtests/2025/`:

- `round_results.csv`
- `selected_players.csv`
- `player_predictions.csv`
- `summary.csv`
````

- [ ] **Step 6: Run all backtesting tests**

Run:

```bash
poetry run pytest src/tests/backtesting -v
```

Expected: PASS.

- [ ] **Step 7: Run the 2025 backtest**

Run:

```bash
poetry run python -m cartola.backtesting.cli --season 2025 --start-round 5 --budget 100
```

Expected: command completes and writes:

```text
data/08_reporting/backtests/2025/round_results.csv
data/08_reporting/backtests/2025/selected_players.csv
data/08_reporting/backtests/2025/player_predictions.csv
data/08_reporting/backtests/2025/summary.csv
```

- [ ] **Step 8: Inspect output sanity**

Run:

```bash
poetry run python - <<'PY'
import csv
from pathlib import Path

base = Path("data/08_reporting/backtests/2025")
for name in ["round_results.csv", "selected_players.csv", "player_predictions.csv", "summary.csv"]:
    path = base / name
    with path.open() as f:
        rows = list(csv.reader(f))
    print(name, len(rows) - 1)
PY
```

Expected: each file reports at least one data row; `round_results.csv` should include strategies `baseline`, `random_forest`, and `price`.

- [ ] **Step 9: Commit**

Run:

```bash
git add README.md src/cartola/backtesting/cli.py src/tests/backtesting/test_cli.py
git commit -m "feat: add backtesting CLI"
```

## Task 8: Final Verification

**Files:**
- Verify all changed files.

- [ ] **Step 1: Run full test suite**

Run:

```bash
poetry run pytest -v
```

Expected: PASS.

- [ ] **Step 2: Run formatting check**

Run:

```bash
uv run --frozen scripts/pyrepo-check ruff
```

Expected: PASS.

- [ ] **Step 3: Check git status**

Run:

```bash
git status --short
```

Expected: clean working tree.

- [ ] **Step 4: Record final result**

Run:

```bash
python - <<'PY'
import csv
from pathlib import Path

summary = Path("data/08_reporting/backtests/2025/summary.csv")
with summary.open() as f:
    for row in csv.DictReader(f):
        print(row["strategy"], row["total_actual_points"], row["actual_points_delta_vs_price"])
PY
```

Expected: prints one line per strategy with season total and delta against the price benchmark.

## Self-Review

- Spec coverage: covered standalone package, 2025 season default, fixed budget, walk-forward from round 5, data loading from `data/01_raw/{season}`, playable status filtering, baseline and RandomForest, PuLP optimizer, CSV outputs, and tests.
- Data quirks: covered per-round missing scouts, `V`, `entrou_em_campo`, and unnamed index columns.
- Type consistency: `BacktestConfig`, `build_prediction_frame`, `build_training_frame`, `BaselinePredictor`, `RandomForestPointPredictor`, `optimize_squad`, `build_summary`, and `run_backtest` signatures are consistent across tasks.
- Library APIs: scikit-learn usage follows current `RandomForestRegressor.fit/predict`, `ColumnTransformer`, `OneHotEncoder(handle_unknown="ignore")`; PuLP usage follows `LpProblem`, `LpMaximize`, binary `LpVariable`, `lpSum`, `solve`, and `LpStatus`.
