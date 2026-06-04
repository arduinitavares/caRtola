#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

from rich.console import Console

from cartola.backtesting.promotion_gate_evidence import (
    PromotionGateEvidenceError,
    scan_promotion_gate_evidence,
    write_markdown_report,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify frozen promotion decision artifacts against the evidence contract.")
    parser.add_argument("--root", type=Path, default=Path("data/08_reporting"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/08_reporting/governance/promotion_gate_evidence_verification.md"),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    console = Console()
    stderr = Console(stderr=True)
    try:
        checks = scan_promotion_gate_evidence(args.root, project_root=Path.cwd())
        output_path = write_markdown_report(checks, args.output)
    except PromotionGateEvidenceError as error:
        stderr.print(f"Promotion gate evidence verification failed: {error}")
        return 1
    fail_count = sum(1 for check in checks if check.status == "fail")
    console.print(f"Promotion gate evidence verification written: {output_path}")
    console.print(f"Artifacts scanned: {len(checks)}; failing artifacts: {fail_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
