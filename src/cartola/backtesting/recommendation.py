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
            raise ValueError(f"live mode requires season {config.season} to equal current_year {current_year}")


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
