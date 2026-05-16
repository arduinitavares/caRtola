from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any, cast

import pytest

from cartola.backtesting.squad_submission import (
    CONTRACT_UNVERIFIED,
    ContractUnverifiedError,
    SquadSubmissionResult,
    SubmissionConfig,
)

SCRIPT_PATH = Path(__file__).resolve().parents[3] / "scripts" / "submit_recommended_squad.py"
SPEC = importlib.util.spec_from_file_location("submit_recommended_squad", SCRIPT_PATH)
assert SPEC is not None
assert SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(_MODULE)
submit_recommended_squad = cast("Any", _MODULE)


def test_parse_args_accepts_recommendation_path() -> None:
    args = submit_recommended_squad.parse_args(["--recommendation-path", "path"])

    assert args.recommendation_path == Path("path")
    assert args.submission_plan is None
    assert args.project_root == Path(".")
    assert args.confirm_submit is False


def test_parse_args_rejects_both_recommendation_and_plan() -> None:
    with pytest.raises(SystemExit):
        submit_recommended_squad.parse_args(
            [
                "--recommendation-path",
                "run",
                "--submission-plan",
                "submission_plan.json",
            ]
        )


def test_main_prints_plan_summary(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed_configs: list[SubmissionConfig] = []

    def fake_run_submission(config: SubmissionConfig) -> SquadSubmissionResult:
        observed_configs.append(config)
        return SquadSubmissionResult(
            attempt_directory=tmp_path / "attempt",
            submission_plan_path=tmp_path / "attempt" / "submission_plan.json",
            submission_result_path=tmp_path / "attempt" / "submission_result.json",
            payload_sha256="abc123",
            status="plan_only",
        )

    monkeypatch.setattr(submit_recommended_squad, "run_submission", fake_run_submission)

    exit_code = submit_recommended_squad.main(
        ["--recommendation-path", "run", "--project-root", str(tmp_path)]
    )

    assert exit_code == 0
    assert observed_configs == [SubmissionConfig(project_root=tmp_path, recommendation_path=Path("run"))]
    captured = capsys.readouterr()
    assert "Submission plan ready" in captured.out
    assert "abc123" in captured.out


def test_main_contract_unverified_does_not_load_dotenv(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def fake_run_submission(config: SubmissionConfig) -> SquadSubmissionResult:
        calls.append("run_submission")
        raise ContractUnverifiedError(CONTRACT_UNVERIFIED)

    def fail_load_dotenv(*args: object, **kwargs: object) -> None:
        calls.append("load_dotenv")
        raise AssertionError("load_dotenv must not be called")

    monkeypatch.setattr(submit_recommended_squad, "run_submission", fake_run_submission)
    monkeypatch.setattr(submit_recommended_squad, "load_dotenv", fail_load_dotenv, raising=False)

    exit_code = submit_recommended_squad.main(
        [
            "--submission-plan",
            "attempt/submission_plan.json",
            "--confirm-payload-sha256",
            "abc123",
            "--confirm-submit",
            "--project-root",
            str(tmp_path),
        ]
    )

    assert exit_code == 1
    assert calls == ["run_submission"]
    captured = capsys.readouterr()
    assert CONTRACT_UNVERIFIED in captured.err
