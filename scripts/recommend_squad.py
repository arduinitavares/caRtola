from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from cartola.backtesting.recommendation import RecommendationConfig, run_recommendation


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a single-round Cartola squad recommendation.")
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--target-round", type=_positive_int, required=True)
    parser.add_argument("--mode", choices=("live", "replay"), required=True)
    parser.add_argument("--budget", type=float, default=100.0)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--output-root", type=Path, default=Path("data/08_reporting/recommendations"))
    parser.add_argument("--footystats-mode", choices=("none", "ppg", "ppg_xg"), default="ppg")
    parser.add_argument("--footystats-league-slug", default="brazil-serie-a")
    parser.add_argument("--footystats-dir", type=Path, default=Path("data/footystats"))
    parser.add_argument("--current-year", type=int, default=None)
    parser.add_argument("--allow-finalized-live-data", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config = RecommendationConfig(
        season=args.season,
        target_round=args.target_round,
        mode=args.mode,
        budget=args.budget,
        project_root=args.project_root,
        output_root=args.output_root,
        footystats_mode=args.footystats_mode,
        footystats_league_slug=args.footystats_league_slug,
        footystats_dir=args.footystats_dir,
        current_year=args.current_year,
        allow_finalized_live_data=args.allow_finalized_live_data,
    )

    result = run_recommendation(config)
    print(
        "Recommendation complete: "
        f"season={config.season} target_round={config.target_round} "
        f"mode={config.mode} predicted_points={result.summary['predicted_points']} "
        f"output={config.output_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
