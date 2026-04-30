from __future__ import annotations

import importlib.util
from pathlib import Path

from cartola.backtesting import matchup_fixture_audit as audit

SCRIPT_PATH = Path(__file__).resolve().parents[3] / "scripts" / "audit_matchup_fixture_coverage.py"
SPEC = importlib.util.spec_from_file_location("audit_matchup_fixture_coverage", SCRIPT_PATH)
assert SPEC is not None
assert SPEC.loader is not None
cli = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(cli)


def test_script_delegates_to_module_main() -> None:
    assert cli.main is audit.main
