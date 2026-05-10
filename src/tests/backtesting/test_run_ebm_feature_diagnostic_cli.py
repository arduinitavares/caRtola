from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _load_cli_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "run_ebm_feature_diagnostic",
        PROJECT_ROOT / "scripts" / "run_ebm_feature_diagnostic.py",
    )
    if spec is None or spec.loader is None:
        raise AssertionError("Expected EBM feature diagnostic CLI module spec")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


cli = _load_cli_module()


def test_parse_args_defaults() -> None:
    args = cli.parse_args(
        [
            "--experiment-path",
            "data/08_reporting/experiments/model_feature/example",
            "--model-id",
            "xgboost_depth2_l2_heavy",
            "--feature-pack",
            "ppg_xg_matchup",
            "--seasons",
            "2021,2022,2023,2024,2025",
            "--current-year",
            "2026",
        ]
    )

    assert args.experiment_path == Path("data/08_reporting/experiments/model_feature/example")
    assert args.output_root == Path("data/08_reporting/ebm_diagnostics")
    assert args.model_id == "xgboost_depth2_l2_heavy"
    assert args.feature_pack == "ppg_xg_matchup"
    assert args.seasons == (2021, 2022, 2023, 2024, 2025)
    assert args.current_year == 2026
    assert args.fixture_mode == "exploratory"


def test_parse_args_rejects_duplicate_seasons(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc_info:
        cli.parse_args(
            [
                "--experiment-path",
                "data/08_reporting/experiments/model_feature/example",
                "--model-id",
                "xgboost_depth2_l2_heavy",
                "--feature-pack",
                "ppg_xg_matchup",
                "--seasons",
                "2021,2022,2022",
                "--current-year",
                "2026",
            ]
        )

    assert exc_info.value.code == 2
    captured = capsys.readouterr()
    assert "Duplicate seasons" in captured.err
    assert "Traceback" not in captured.err
