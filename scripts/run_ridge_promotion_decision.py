#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

from rich.console import Console
from rich.panel import Panel

from cartola.backtesting.ridge_promotion_decision import RidgePromotionDecisionError, write_ridge_promotion_decision


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the M008 Ridge PPG_XG balanced promotion decision.")
    parser.add_argument("--experiment-path", type=Path, required=True)
    parser.add_argument("--candidate-model", required=True)
    parser.add_argument("--candidate-feature-pack", required=True)
    parser.add_argument("--control-model", required=True)
    parser.add_argument("--control-feature-pack", required=True)
    parser.add_argument("--baseline-model", required=True)
    parser.add_argument("--baseline-feature-pack", required=True)
    parser.add_argument("--promotion-seasons", required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    console = Console()
    stderr = Console(stderr=True)
    try:
        output_path = write_ridge_promotion_decision(
            experiment_path=args.experiment_path,
            candidate_model=args.candidate_model,
            candidate_feature_pack=args.candidate_feature_pack,
            control_model=args.control_model,
            control_feature_pack=args.control_feature_pack,
            baseline_model=args.baseline_model,
            baseline_feature_pack=args.baseline_feature_pack,
            promotion_seasons=_parse_seasons(args.promotion_seasons),
        )
    except RidgePromotionDecisionError as error:
        stderr.print(Panel(str(error), title="Ridge promotion decision failed", border_style="red"))
        return 1
    console.print(f"Ridge promotion decision written: {output_path}")
    return 0


def _parse_seasons(raw: str) -> tuple[int, ...]:
    seasons = tuple(int(value.strip()) for value in raw.split(",") if value.strip())
    if not seasons:
        raise RidgePromotionDecisionError("At least one promotion season is required.")
    if len(set(seasons)) != len(seasons):
        raise RidgePromotionDecisionError("Duplicate promotion seasons are not allowed.")
    return seasons


if __name__ == "__main__":
    raise SystemExit(main())
