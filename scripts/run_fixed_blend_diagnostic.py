#!/usr/bin/env python
from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

run_fixed_blend_diagnostic: Callable[..., Path] | None = None
parse_blend_specs: Callable[[tuple[str, ...]], Any] | None = None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Cartola M006 fixed-blend diagnostic artifacts.")
    parser.add_argument("--experiment-path", type=Path, required=True)
    parser.add_argument("--seasons", default="2021,2022,2023,2024,2025")
    parser.add_argument("--feature-pack", default="ppg_xg")
    parser.add_argument("--control-model", default="xgboost_depth2_l2_heavy")
    parser.add_argument("--blend", action="append", required=True)
    parser.add_argument("--initial-budget", type=float, default=100.0)
    parser.add_argument("--current-year", type=int, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("data/08_reporting/blend_diagnostics"))
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
    global parse_blend_specs, run_fixed_blend_diagnostic

    if run_fixed_blend_diagnostic is None or parse_blend_specs is None:
        from cartola.backtesting.fixed_blend_diagnostic import (
            parse_blend_specs as imported_parse_blend_specs,
        )
        from cartola.backtesting.fixed_blend_diagnostic import (
            run_fixed_blend_diagnostic as imported_run_fixed_blend_diagnostic,
        )

        parse_blend_specs = imported_parse_blend_specs
        run_fixed_blend_diagnostic = imported_run_fixed_blend_diagnostic


def _parse_seasons(raw: str) -> tuple[int, ...]:
    seasons = tuple(int(value.strip()) for value in raw.split(",") if value.strip())
    if not seasons:
        raise ValueError("At least one season is required.")
    if len(set(seasons)) != len(seasons):
        raise ValueError("Duplicate seasons are not allowed.")
    return seasons


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    _bootstrap_dotenv()
    _load_runtime_dependencies()
    if run_fixed_blend_diagnostic is None or parse_blend_specs is None:
        raise RuntimeError("Fixed blend diagnostic runtime dependencies were not loaded.")

    output_path = run_fixed_blend_diagnostic(
        experiment_path=args.experiment_path,
        seasons=_parse_seasons(args.seasons),
        feature_pack=args.feature_pack,
        control_model=args.control_model,
        blend_specs=parse_blend_specs(tuple(args.blend)),
        initial_budget=args.initial_budget,
        current_year=args.current_year,
        output_root=args.output_root,
    )
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
