from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Mapping

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

MARKET_OPEN_PRICE_COLUMN = "preco_pre_rodada"

FixtureMode = Literal["none", "exploratory", "strict"]
StrictAlignmentPolicy = Literal["fail", "exclude_round"]
FootyStatsMode = Literal["none", "ppg"]
FootyStatsEvaluationScope = Literal["historical_candidate", "live_current"]

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
    fixture_mode: FixtureMode = "none"
    strict_alignment_policy: StrictAlignmentPolicy = "fail"
    footystats_mode: FootyStatsMode = "none"
    footystats_evaluation_scope: FootyStatsEvaluationScope = "historical_candidate"
    footystats_league_slug: str = "brazil-serie-a"
    footystats_dir: Path = Path("data/footystats")
    current_year: int | None = None
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
