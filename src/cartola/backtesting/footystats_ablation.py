from __future__ import annotations

import argparse
import shutil
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Sequence, cast

from cartola.backtesting.config import BacktestConfig, FootyStatsMode

DEFAULT_SEASONS = (2023, 2024, 2025)
DEFAULT_OUTPUT_ROOT = Path("data/08_reporting/backtests/footystats_ablation")
DEFAULT_LEAGUE_SLUG = "brazil-serie-a"


@dataclass(frozen=True)
class FootyStatsPPGAblationConfig:
    seasons: tuple[int, ...] = DEFAULT_SEASONS
    start_round: int = 5
    budget: float = 100.0
    project_root: Path = Path(".")
    output_root: Path = DEFAULT_OUTPUT_ROOT
    footystats_league_slug: str = DEFAULT_LEAGUE_SLUG
    current_year: int | None = None
    force: bool = False

    @property
    def resolved_current_year(self) -> int:
        if self.current_year is not None:
            return self.current_year
        return datetime.now(UTC).year


def parse_seasons(value: str) -> tuple[int, ...]:
    raw_parts = value.split(",")
    if any(part.strip() == "" for part in raw_parts):
        raise ValueError("seasons must not contain empty entries")

    seasons: list[int] = []
    seen: set[int] = set()
    for part in raw_parts:
        try:
            season = int(part)
        except ValueError as exc:
            raise ValueError(f"Invalid season: {part}") from exc
        if season <= 0:
            raise ValueError("seasons must be positive integers")
        if season in seen:
            raise ValueError(f"duplicate season: {season}")
        seasons.append(season)
        seen.add(season)
    return tuple(seasons)


def parse_seasons_arg(value: str) -> tuple[int, ...]:
    try:
        return parse_seasons(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the FootyStats PPG ablation report.")
    parser.add_argument("--seasons", type=parse_seasons_arg, default=DEFAULT_SEASONS)
    parser.add_argument("--start-round", type=int, default=5)
    parser.add_argument("--budget", type=float, default=100.0)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--footystats-league-slug", default=DEFAULT_LEAGUE_SLUG)
    parser.add_argument("--current-year", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def config_from_args(args: argparse.Namespace) -> FootyStatsPPGAblationConfig:
    return FootyStatsPPGAblationConfig(
        seasons=args.seasons,
        start_round=args.start_round,
        budget=args.budget,
        project_root=args.project_root,
        output_root=args.output_root,
        footystats_league_slug=args.footystats_league_slug,
        current_year=args.current_year,
        force=args.force,
    )


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _validate_output_root(config: FootyStatsPPGAblationConfig, output_root: Path) -> Path:
    project_root = config.project_root.resolve()
    resolved_output_root = (output_root if output_root.is_absolute() else project_root / output_root).resolve()

    if not _is_relative_to(resolved_output_root, project_root):
        raise ValueError(f"output_root must resolve inside project_root: {project_root}")

    normal_backtests_root = project_root / "data" / "08_reporting" / "backtests"
    protected_paths = {
        project_root,
        project_root / "data",
        project_root / "data" / "08_reporting",
        normal_backtests_root,
        *(normal_backtests_root / str(season) for season in config.seasons),
    }
    if resolved_output_root in protected_paths:
        raise ValueError(f"output_root resolves to a protected path: {resolved_output_root}")

    if resolved_output_root.name != "footystats_ablation":
        raise ValueError("output_root final directory name must be exactly 'footystats_ablation'")

    return resolved_output_root


def resolve_output_root(config: FootyStatsPPGAblationConfig) -> Path:
    return _validate_output_root(config, config.output_root)


def build_backtest_config(
    config: FootyStatsPPGAblationConfig,
    season: int,
    mode: str,
    resolved_output_root: Path,
) -> BacktestConfig:
    resolved_output_root = _validate_output_root(config, resolved_output_root)

    if mode not in ("none", "ppg"):
        raise ValueError(f"Unsupported footystats mode: {mode!r}")
    footystats_mode = cast(FootyStatsMode, mode)

    backtest_config = BacktestConfig(
        season=season,
        start_round=config.start_round,
        budget=config.budget,
        project_root=config.project_root.resolve(),
        output_root=resolved_output_root / "runs" / str(season) / f"footystats_mode={footystats_mode}",
        fixture_mode="none",
        footystats_mode=footystats_mode,
        footystats_evaluation_scope="historical_candidate",
        footystats_league_slug=config.footystats_league_slug,
        current_year=config.resolved_current_year,
    )

    normal_backtest_output_path = backtest_config.project_root / "data" / "08_reporting" / "backtests" / str(season)
    if backtest_config.output_path.resolve() == normal_backtest_output_path.resolve():
        raise ValueError(f"Refusing to use normal backtest output path: {normal_backtest_output_path}")

    return backtest_config


def prepare_output_root(config: FootyStatsPPGAblationConfig, resolved_output_root: Path) -> None:
    resolved_output_root = _validate_output_root(config, resolved_output_root)

    if resolved_output_root.exists():
        if not config.force:
            raise FileExistsError(f"output_root already exists: {resolved_output_root}")
        shutil.rmtree(resolved_output_root)
    resolved_output_root.mkdir(parents=True, exist_ok=True)


def main(argv: Sequence[str] | None = None) -> int:
    config = config_from_args(parse_args(argv))
    print(
        "FootyStats PPG ablation report is not implemented yet "
        f"for seasons {', '.join(str(season) for season in config.seasons)}."
    )
    return 2
