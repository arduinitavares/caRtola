#!/usr/bin/env python
from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path

from rich.console import Console
from rich.panel import Panel

build_oracle_discovery_report: Callable[..., object] | None = None


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Cartola oracle knowledge discovery report.")
    parser.add_argument("--experiment-path", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("data/08_reporting/oracle_discovery"))
    parser.add_argument("--current-year", type=int, default=None)
    return parser.parse_args(argv)


def _timestamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")


def _load_runtime_dependencies() -> None:
    global build_oracle_discovery_report

    if build_oracle_discovery_report is None:
        from cartola.backtesting.oracle_discovery import (
            build_oracle_discovery_report as imported_build_oracle_discovery_report,
        )

        build_oracle_discovery_report = imported_build_oracle_discovery_report


def _print_error(console: Console, error: Exception) -> None:
    console.print(Panel(str(error), title="Oracle discovery failed", border_style="red"))


def _print_success(console: Console, *, output_path: Path) -> None:
    console.print(
        Panel(
            f"output_path={output_path}",
            title="Oracle discovery complete",
            border_style="green",
        )
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    stdout = Console()
    stderr = Console(stderr=True)
    output_path = args.output_root / f"oracle_discovery_started_at={_timestamp()}"

    try:
        _load_runtime_dependencies()
        if build_oracle_discovery_report is None:
            raise RuntimeError("Oracle discovery runtime dependencies were not loaded.")
        build_oracle_discovery_report(experiment_path=args.experiment_path, output_path=output_path)
    except Exception as error:
        _print_error(stderr, error)
        return 1
    _print_success(stdout, output_path=output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
