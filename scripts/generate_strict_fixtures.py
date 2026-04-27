#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from cartola.backtesting.strict_fixtures import generate_strict_fixture


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate canonical strict fixture CSVs from captured snapshots.")
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--round", dest="round_number", type=int, required=True)
    parser.add_argument("--source", choices=("cartola_api",), default="cartola_api")
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--captured-at", default=None)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = generate_strict_fixture(
        project_root=args.project_root,
        season=args.season,
        round_number=args.round_number,
        source=args.source,
        captured_at=args.captured_at,
        force=args.force,
    )
    print(f"Generated strict fixture: {result.fixture_path}")
    print(f"Validated manifest: {result.manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
