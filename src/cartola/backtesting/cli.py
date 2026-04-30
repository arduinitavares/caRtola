from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from cartola.backtesting.config import BacktestConfig
from cartola.backtesting.runner import run_backtest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run an offline Cartola backtest.")
    parser.add_argument("--season", type=int, default=2025)
    parser.add_argument("--start-round", type=int, default=5)
    parser.add_argument("--budget", type=float, default=100.0)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--output-root", type=Path, default=Path("data/08_reporting/backtests"))
    parser.add_argument("--fixture-mode", choices=("none", "exploratory", "strict"), default="none")
    parser.add_argument("--strict-alignment-policy", choices=("fail", "exclude_round"), default="fail")
    parser.add_argument("--matchup-context-mode", choices=("none", "cartola_matchup_v1"), default="none")
    parser.add_argument("--footystats-mode", choices=("none", "ppg", "ppg_xg"), default="none")
    parser.add_argument(
        "--footystats-evaluation-scope",
        choices=("historical_candidate", "live_current"),
        default="historical_candidate",
    )
    parser.add_argument("--footystats-league-slug", default="brazil-serie-a")
    parser.add_argument("--footystats-dir", type=Path, default=Path("data/footystats"))
    parser.add_argument("--current-year", type=int, default=None)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config = BacktestConfig(
        season=args.season,
        start_round=args.start_round,
        budget=args.budget,
        project_root=args.project_root,
        output_root=args.output_root,
        fixture_mode=args.fixture_mode,
        strict_alignment_policy=args.strict_alignment_policy,
        matchup_context_mode=args.matchup_context_mode,
        footystats_mode=args.footystats_mode,
        footystats_evaluation_scope=args.footystats_evaluation_scope,
        footystats_league_slug=args.footystats_league_slug,
        footystats_dir=args.footystats_dir,
        current_year=args.current_year,
    )

    result = run_backtest(config)
    for warning in result.metadata.warnings:
        print(f"WARNING: {warning}")
    print(f"Backtest complete: season={config.season} output={config.output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
