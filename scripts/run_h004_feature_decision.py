from __future__ import annotations

import argparse
from pathlib import Path

from rich.console import Console

from cartola.backtesting.h004_feature_decision import write_h004_phase2_decision

DEFAULT_PHASE1_DECISION = Path(
    "data/08_reporting/hypotheses/"
    "h004_residual_diagnostic_started_at=20260508T182202655139Z/"
    "h004_diagnostic_decision.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the deterministic H004 Phase 2 decision artifact.")
    parser.add_argument("--experiment-path", required=True, type=Path)
    parser.add_argument("--phase1-decision-path", type=Path, default=DEFAULT_PHASE1_DECISION)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = write_h004_phase2_decision(
        experiment_path=args.experiment_path,
        phase1_decision_path=args.phase1_decision_path,
    )
    console = Console()
    console.print(f"H004 Phase 2 decision written: {output_path}")


if __name__ == "__main__":
    main()
