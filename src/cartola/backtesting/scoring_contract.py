from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

SCORING_CONTRACT_VERSION = "cartola_standard_2026_v1"
CAPTAIN_SCORING_ENABLED = True
CAPTAIN_MULTIPLIER = 1.5
FORMATION_SEARCH = "all_official_formations"


def _finite_numeric_series(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        raise ValueError(f"Missing required numeric column: {column}")
    numeric = pd.to_numeric(frame[column], errors="coerce")
    if not np.isfinite(numeric).all():
        raise ValueError(f"Column must contain only finite numeric values: {column}")
    return numeric.astype(float)


def _selected_captain_row(selected: pd.DataFrame) -> pd.Series:
    if "is_captain" not in selected.columns:
        captain_count = 0
        raise ValueError(f"Selected squad must contain exactly one captain, got {captain_count}")
    captain_mask = selected["is_captain"].eq(True)
    captain_count = int(captain_mask.sum())
    if captain_count != 1:
        raise ValueError(f"Selected squad must contain exactly one captain, got {captain_count}")
    return selected.loc[captain_mask].iloc[0]


def actual_scores_with_captain(
    selected: pd.DataFrame,
    *,
    actual_column: str = "pontuacao",
) -> dict[str, float]:
    actual_scores = _finite_numeric_series(selected, actual_column)
    captain = _selected_captain_row(selected)
    actual_points_base = float(actual_scores.sum())
    captain_actual = float(actual_scores.loc[captain.name])
    captain_bonus_actual = (CAPTAIN_MULTIPLIER - 1.0) * captain_actual
    return {
        "actual_points_base": actual_points_base,
        "captain_bonus_actual": captain_bonus_actual,
        "actual_points_with_captain": actual_points_base + captain_bonus_actual,
    }


def captain_policy_diagnostics(
    selected: pd.DataFrame,
    *,
    predicted_column: str,
    actual_column: str | None = None,
) -> list[dict[str, object]]:
    candidates = selected.loc[selected["posicao"].ne("tec")].copy()
    if candidates.empty:
        raise ValueError("Selected squad has no non-tecnico captain candidates")

    predicted_candidates = _finite_numeric_series(candidates, predicted_column)
    predicted_selected = _finite_numeric_series(selected, predicted_column)

    if "prior_points_std" in candidates.columns:
        prior_std = pd.to_numeric(candidates["prior_points_std"], errors="coerce")
        prior_std = prior_std.where(np.isfinite(prior_std), 0.0).fillna(0.0)
    else:
        prior_std = pd.Series(0.0, index=candidates.index)

    candidate_scores = candidates.copy()
    candidate_scores["_predicted_points"] = predicted_candidates
    candidate_scores["_prior_points_std"] = prior_std.astype(float)

    actual_scores: pd.Series | None = None
    actual_points_base: float | None = None
    if actual_column is not None:
        actual_scores = _finite_numeric_series(selected, actual_column)
        actual_points_base = float(actual_scores.sum())

    predicted_points_base = float(predicted_selected.sum())
    policies = {
        "ev": candidate_scores["_predicted_points"],
        "safe": candidate_scores["_predicted_points"] - candidate_scores["_prior_points_std"],
        "upside": candidate_scores["_predicted_points"] + candidate_scores["_prior_points_std"],
    }

    records: list[dict[str, object]] = []
    ev_actual_total: float | None = None
    for policy, policy_scores in policies.items():
        scored = candidate_scores.assign(_policy_score=policy_scores)
        captain = scored.sort_values(
            by=["_policy_score", "_predicted_points", "id_atleta"],
            ascending=[False, False, True],
            kind="mergesort",
        ).iloc[0]

        captain_predicted = float(captain["_predicted_points"])
        predicted_captain_bonus = (CAPTAIN_MULTIPLIER - 1.0) * captain_predicted
        predicted_points_with_policy = predicted_points_base + predicted_captain_bonus

        actual_captain_points: float | None = None
        actual_captain_bonus: float | None = None
        actual_points_with_policy: float | None = None
        if actual_scores is not None and actual_points_base is not None:
            actual_captain_points = float(actual_scores.loc[captain.name])
            actual_captain_bonus = (CAPTAIN_MULTIPLIER - 1.0) * actual_captain_points
            actual_points_with_policy = actual_points_base + actual_captain_bonus
            if policy == "ev":
                ev_actual_total = actual_points_with_policy

        actual_delta_vs_ev = (
            None
            if actual_points_with_policy is None or ev_actual_total is None
            else actual_points_with_policy - ev_actual_total
        )

        records.append(
            {
                "policy": policy,
                "captain_id": int(captain["id_atleta"]),
                "captain_name": str(captain["apelido"]),
                "captain_position": str(captain["posicao"]),
                "captain_club": str(captain["nome_clube"]),
                "captain_predicted_points": captain_predicted,
                "predicted_captain_bonus": predicted_captain_bonus,
                "predicted_points_with_policy": predicted_points_with_policy,
                "actual_captain_points": actual_captain_points,
                "actual_captain_bonus": actual_captain_bonus,
                "actual_points_with_policy": actual_points_with_policy,
                "actual_delta_vs_ev": actual_delta_vs_ev,
            }
        )

    return records


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
