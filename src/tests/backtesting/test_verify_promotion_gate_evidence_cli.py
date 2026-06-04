from __future__ import annotations

import importlib.util
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "verify_promotion_gate_evidence.py"
SPEC = importlib.util.spec_from_file_location("verify_promotion_gate_evidence", SCRIPT_PATH)
assert SPEC is not None
assert SPEC.loader is not None
cli = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(cli)


def test_parse_args_uses_default_paths() -> None:
    args = cli.parse_args([])

    assert args.root == Path("data/08_reporting")
    assert args.output == Path("data/08_reporting/governance/promotion_gate_evidence_verification.md")


def test_main_writes_report(tmp_path: Path) -> None:
    root = tmp_path / "reporting"
    artifact = root / "candidate_decision.json"
    artifact.parent.mkdir(parents=True)
    artifact.write_text(
        json.dumps(
            {
                "decision_status": "rejected",
                "candidate_strategy": {"model_id": "ridge", "feature_pack": "ppg_xg"},
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    output = tmp_path / "promotion_gate_evidence_verification.md"

    exit_code = cli.main(["--root", str(root), "--output", str(output)])

    assert exit_code == 0
    assert output.is_file()
    assert "candidate_decision.json" in output.read_text(encoding="utf-8")
