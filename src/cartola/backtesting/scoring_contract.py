from __future__ import annotations

import json
from pathlib import Path
from typing import Any

SCORING_CONTRACT_VERSION = "cartola_standard_2026_v1"
CAPTAIN_SCORING_ENABLED = True
CAPTAIN_MULTIPLIER = 1.5
FORMATION_SEARCH = "all_official_formations"


def contract_fields() -> dict[str, object]:
    return {
        "scoring_contract_version": SCORING_CONTRACT_VERSION,
        "captain_scoring_enabled": CAPTAIN_SCORING_ENABLED,
        "captain_multiplier": CAPTAIN_MULTIPLIER,
        "formation_search": FORMATION_SEARCH,
    }


def validate_contract_mapping(mapping: dict[str, Any]) -> dict[str, object]:
    expected = contract_fields()
    for key, expected_value in expected.items():
        if key not in mapping:
            raise ValueError(f"Missing scoring contract field: {key}")
        if mapping[key] != expected_value:
            raise ValueError(
                f"Unsupported scoring contract field {key}: expected {expected_value!r}, got {mapping[key]!r}"
            )
    return expected


def validate_report_contract(output_path: Path) -> dict[str, object]:
    metadata_path = output_path / "run_metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing run_metadata.json beside report outputs: {metadata_path}")
    try:
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in run_metadata.json: {metadata_path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"run_metadata.json must contain an object: {metadata_path}")
    return validate_contract_mapping(payload)
