#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

import requests

from cartola.backtesting.strict_round_fixture_capture import (
    StrictFixtureCaptureError,
    StrictFixtureGenerationError,
    StrictRoundFixtureCaptureConfig,
    format_project_path,
    format_utc_z,
    run_strict_round_fixture_capture,
)


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("value must be a positive integer") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Capture a strict round fixture snapshot and generate strict fixtures.")
    parser.add_argument("--season", type=_positive_int, required=True)
    selector = parser.add_mutually_exclusive_group(required=True)
    selector.add_argument("--auto", action="store_true")
    selector.add_argument("--round", dest="round_number", type=_positive_int, default=None)
    parser.add_argument("--current-year", type=_positive_int, default=None)
    parser.add_argument("--source", choices=["cartola_api"], default="cartola_api")
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--force-generate", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config = StrictRoundFixtureCaptureConfig(
        season=args.season,
        round_number=args.round_number,
        auto=args.auto,
        current_year=args.current_year,
        source=args.source,
        project_root=args.project_root,
        force_generate=args.force_generate,
    )

    try:
        result = run_strict_round_fixture_capture(config)
    except StrictFixtureGenerationError as exc:
        capture_result = exc.capture_result
        print("error: strict fixture generation failed after snapshot capture", file=sys.stderr)
        print(
            f"Retained snapshot directory: {format_project_path(args.project_root, capture_result.capture_dir)}",
            file=sys.stderr,
        )
        print(f"Captured at UTC: {format_utc_z(capture_result.captured_at_utc)}", file=sys.stderr)
        print(f"Deadline at UTC: {format_utc_z(capture_result.deadline_at_utc)}", file=sys.stderr)
        print(f"Original error: {type(exc.original).__name__}: {exc.original}", file=sys.stderr)
        return 1
    except StrictFixtureCaptureError as exc:
        print(f"error: strict fixture capture failed for round {exc.round_number}", file=sys.stderr)
        print(f"Original error: {type(exc.original).__name__}: {exc.original}", file=sys.stderr)
        return 1
    except (ValueError, FileExistsError, FileNotFoundError, json.JSONDecodeError, requests.RequestException) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print("Strict fixture capture complete")
    print(f"Season: {result.season}")
    print(f"Round: {result.round_number}")
    print(f"Snapshot directory: {format_project_path(args.project_root, result.capture_dir)}")
    print(f"Strict fixture: {format_project_path(args.project_root, result.fixture_path)}")
    print(f"Manifest: {format_project_path(args.project_root, result.manifest_path)}")
    print(f"Captured at UTC: {format_utc_z(result.captured_at_utc)}")
    print(f"Deadline at UTC: {format_utc_z(result.deadline_at_utc)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
