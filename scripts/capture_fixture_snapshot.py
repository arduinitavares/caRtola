#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from cartola.backtesting.fixture_snapshots import capture_cartola_snapshot


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Capture strict pre-lock Cartola fixture snapshot.")
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--round", dest="round_number", type=int, required=True)
    parser.add_argument("--source", choices=("cartola_api",), default="cartola_api")
    parser.add_argument("--project-root", type=Path, default=Path("."))
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = capture_cartola_snapshot(
        project_root=args.project_root,
        season=args.season,
        round_number=args.round_number,
        source=args.source,
    )
    print(f"Captured fixture snapshot: {result.capture_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
