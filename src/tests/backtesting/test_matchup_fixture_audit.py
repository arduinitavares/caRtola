from __future__ import annotations

from pathlib import Path

import pytest

from cartola.backtesting import matchup_fixture_audit as audit


def test_parse_seasons_accepts_comma_separated_positive_unique_values() -> None:
    assert audit.parse_seasons("2023,2024,2025") == (2023, 2024, 2025)


@pytest.mark.parametrize("value", ["", "2023,", "2023,2023", "0", "-2024", "abc"])
def test_parse_seasons_rejects_invalid_values(value: str) -> None:
    with pytest.raises(ValueError):
        audit.parse_seasons(value)


def test_parse_args_defaults_to_requested_historical_gate() -> None:
    args = audit.parse_args(["--current-year", "2026"])

    assert args.seasons == (2023, 2024, 2025)
    assert args.current_year == 2026
    assert args.project_root == Path(".")
    assert args.output_root == Path("data/08_reporting/fixtures")
    assert args.expected_complete_rounds == 38
    assert args.complete_round_threshold == 38


def test_config_from_args_preserves_values() -> None:
    args = audit.parse_args(
        [
            "--seasons",
            "2024,2025",
            "--current-year",
            "2026",
            "--project-root",
            "/tmp/cartola",
            "--output-root",
            "/tmp/reports",
            "--expected-complete-rounds",
            "30",
            "--complete-round-threshold",
            "20",
        ]
    )

    config = audit.config_from_args(args)

    assert config.seasons == (2024, 2025)
    assert config.current_year == 2026
    assert config.project_root == Path("/tmp/cartola")
    assert config.output_root == Path("/tmp/reports")
    assert config.expected_complete_rounds == 30
    assert config.complete_round_threshold == 20
