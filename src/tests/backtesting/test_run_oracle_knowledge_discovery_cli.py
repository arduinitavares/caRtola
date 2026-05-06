from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import NoReturn

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_oracle_knowledge_discovery import main, parse_args  # noqa: E402


def test_importing_cli_does_not_import_oracle_module() -> None:
    code = "\n".join(
        [
            "import importlib",
            "import sys",
            "importlib.import_module('scripts.run_oracle_knowledge_discovery')",
            "print('cartola.backtesting.oracle_discovery' in sys.modules)",
        ]
    )

    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.stdout.strip() == "False"


def test_parse_args_defaults() -> None:
    args = parse_args(["--experiment-path", "exp", "--current-year", "2026"])

    assert args.experiment_path == Path("exp")
    assert args.current_year == 2026
    assert args.output_root == Path("data/08_reporting/oracle_discovery")
    assert not hasattr(args, "allow_incomplete")


def test_parse_args_current_year_is_optional_workflow_compatibility() -> None:
    args = parse_args(["--experiment-path", "exp"])

    assert args.current_year is None


def test_parse_args_rejects_allow_incomplete() -> None:
    with pytest.raises(SystemExit):
        parse_args(["--experiment-path", "exp", "--current-year", "2026", "--allow-incomplete"])


def test_main_calls_report_builder(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import scripts.run_oracle_knowledge_discovery as cli

    observed: dict[str, object] = {}

    def fake_builder(**kwargs: object) -> None:
        observed.update(kwargs)

    monkeypatch.setattr(cli, "build_oracle_discovery_report", fake_builder)

    exit_code = main(
        [
            "--experiment-path",
            str(tmp_path / "exp"),
            "--output-root",
            str(tmp_path / "oracle"),
            "--current-year",
            "2026",
        ]
    )

    assert exit_code == 0
    assert observed["experiment_path"] == tmp_path / "exp"
    assert str(observed["output_path"]).startswith(str(tmp_path / "oracle"))
    assert "oracle_discovery_started_at=" in str(observed["output_path"])


def test_main_reports_builder_failure_without_traceback(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    import scripts.run_oracle_knowledge_discovery as cli

    def fake_builder(**_kwargs: object) -> NoReturn:
        raise RuntimeError("oracle failed")

    monkeypatch.setattr(cli, "build_oracle_discovery_report", fake_builder)

    exit_code = main(
        [
            "--experiment-path",
            str(tmp_path / "exp"),
            "--current-year",
            "2026",
        ]
    )

    assert exit_code == 1
    captured = capsys.readouterr()
    assert "oracle failed" in captured.err
    assert "Traceback" not in captured.err
