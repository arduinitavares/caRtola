#!/usr/bin/env python
from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from rich.console import Console

run_policy_simulation: Callable[[argparse.Namespace, Console], Any] | None = None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Cartola policy simulation replay artifacts.")
    parser.add_argument("--experiment-path", type=Path, required=True)
    parser.add_argument("--hypothesis-id", required=True)
    parser.add_argument("--policy-set", required=True)
    parser.add_argument("--models", required=True, help="Comma-separated model IDs to include.")
    parser.add_argument("--feature-packs", required=True, help="Comma-separated feature pack IDs to include.")
    parser.add_argument("--seasons", default="2021,2022,2023,2024,2025")
    parser.add_argument("--current-year", type=int, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("data/08_reporting/policy_simulations"))
    parser.add_argument(
        "--allow-incomplete-report",
        action="store_true",
        help="Write diagnostic report artifacts even when replay invalid rows are present.",
    )
    return parser.parse_args(argv)


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _bootstrap_dotenv(project_root: Path | None = None) -> bool:
    resolved_project_root = _project_root() if project_root is None else project_root
    dotenv_path = resolved_project_root.expanduser() / ".env"
    if not dotenv_path.is_file():
        return False
    load_dotenv(dotenv_path=dotenv_path, override=False)
    return True


def _load_runtime_dependencies() -> None:
    global run_policy_simulation

    if run_policy_simulation is None:
        from cartola.backtesting.policy_simulation import run_policy_simulation as imported_run_policy_simulation

        run_policy_simulation = imported_run_policy_simulation


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    _bootstrap_dotenv()
    _load_runtime_dependencies()
    if run_policy_simulation is None:
        raise RuntimeError("Policy simulation runtime dependencies were not loaded.")

    console = Console()
    run_policy_simulation(args, console)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
