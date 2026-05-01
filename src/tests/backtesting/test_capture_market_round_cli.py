from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from cartola.backtesting.market_capture import MarketCaptureConfig, MarketCaptureResult

SCRIPT_PATH = Path(__file__).resolve().parents[3] / "scripts" / "capture_market_round.py"
SPEC = importlib.util.spec_from_file_location("capture_market_round", SCRIPT_PATH)
assert SPEC is not None
assert SPEC.loader is not None
cli = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(cli)


def test_parse_args_requires_target_or_auto() -> None:
    with pytest.raises(SystemExit):
        cli.parse_args(["--season", "2026", "--current-year", "2026"])


def test_parse_args_rejects_target_and_auto_together() -> None:
    with pytest.raises(SystemExit):
        cli.parse_args(["--season", "2026", "--target-round", "14", "--auto", "--current-year", "2026"])


def test_main_prints_capture_summary(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    def fake_capture(config: MarketCaptureConfig) -> MarketCaptureResult:
        assert config.season == 2026
        assert config.target_round == 14
        assert config.current_year == 2026
        assert config.force is False
        return MarketCaptureResult(
            csv_path=Path("data/01_raw/2026/rodada-14.csv"),
            metadata_path=Path("data/01_raw/2026/rodada-14.capture.json"),
            target_round=14,
            athlete_count=747,
            status_mercado=1,
            deadline_timestamp=1777748340,
            deadline_parse_status="ok",
        )

    monkeypatch.setattr(cli, "capture_market_round", fake_capture)

    exit_code = cli.main(["--season", "2026", "--target-round", "14", "--current-year", "2026"])

    assert exit_code == 0
    captured = capsys.readouterr()
    assert "Captured live market round: data/01_raw/2026/rodada-14.csv" in captured.out
    assert "athletes=747" in captured.out
    assert "status_mercado=1" in captured.out
    assert "deadline_timestamp=1777748340" in captured.out
