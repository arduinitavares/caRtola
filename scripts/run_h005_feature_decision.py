from __future__ import annotations

import argparse
from pathlib import Path

from rich.console import Console

from cartola.backtesting.h005_feature_decision import write_h005_feature_decision


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the deterministic H005 feature decision artifact.")
    parser.add_argument("--experiment-path", required=True, type=Path)
    parser.add_argument("--audit-decision-path", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = write_h005_feature_decision(
        experiment_path=args.experiment_path,
        audit_decision_path=args.audit_decision_path,
    )
    console = Console()
    console.print(f"H005 feature decision written: {output_path}")


if __name__ == "__main__":
    main()
