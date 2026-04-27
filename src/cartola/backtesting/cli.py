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
    parser.add_argument("--fixture-mode", choices=("none", "exploratory", "strict"), default="none")
    parser.add_argument("--strict-alignment-policy", choices=("fail", "exclude_round"), default="fail")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config = BacktestConfig(
        season=args.season,
        start_round=args.start_round,
        budget=args.budget,
        project_root=args.project_root,
        fixture_mode=args.fixture_mode,
        strict_alignment_policy=args.strict_alignment_policy,
    )

    result = run_backtest(config)
    for warning in result.metadata.warnings:
        print(f"WARNING: {warning}")
    print(f"Backtest complete: season={config.season} output={config.output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
