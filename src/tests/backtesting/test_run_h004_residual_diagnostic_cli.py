from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _load_cli_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "run_h004_residual_diagnostic",
        PROJECT_ROOT / "scripts" / "run_h004_residual_diagnostic.py",
    )
    if spec is None or spec.loader is None:
        raise AssertionError("Expected H004 diagnostic CLI module spec")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


cli = _load_cli_module()


def test_parse_args_accepts_h004_source_options() -> None:
    args = cli.parse_args(
        [
            "--experiment-path",
            "data/experiment",
            "--seasons",
            "2021,2022",
            "--output-root",
            "data/out",
        ]
    )

    assert args.experiment_path == Path("data/experiment")
    assert args.seasons == "2021,2022"
    assert args.output_root == Path("data/out")
    assert args.model_id == "xgboost_depth2_slow"
    assert args.feature_pack == "ppg_xg_matchup"


def test_parse_seasons_rejects_duplicates() -> None:
    try:
        cli._parse_seasons("2021,2021")
    except ValueError as exc:
        assert "Duplicate seasons" in str(exc)
    else:
        raise AssertionError("Expected duplicate season validation failure")
