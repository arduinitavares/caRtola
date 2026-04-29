#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from cartola.backtesting.market_capture import MarketCaptureConfig, capture_market_round


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Capture the open Cartola market round for live recommendations.")
    parser.add_argument("--season", type=_positive_int, required=True)
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--target-round", type=_positive_int, default=None)
    target.add_argument("--auto", action="store_true")
    parser.add_argument("--current-year", type=_positive_int, default=None)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = capture_market_round(
        MarketCaptureConfig(
            season=args.season,
            target_round=args.target_round,
            auto=args.auto,
            force=args.force,
            current_year=args.current_year,
            project_root=args.project_root,
        )
    )
    print(f"Captured live market round: {result.csv_path}")
    print(
        "metadata="
        f"{result.metadata_path} "
        f"target_round={result.target_round} "
        f"athletes={result.athlete_count} "
        f"status_mercado={result.status_mercado} "
        f"deadline_parse_status={result.deadline_parse_status} "
        f"deadline_timestamp={result.deadline_timestamp}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
