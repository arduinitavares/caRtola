from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Iterable

import pandas as pd

_FIXTURE_COLUMNS: tuple[str, ...] = ("rodada", "id_clube_home", "id_clube_away")


@dataclass(frozen=True)
class OptimizerPolicy:
    policy_variant: str
    overlap_penalty: float = 0.0
    max_overlap_assets: int | None = None


@dataclass(frozen=True)
class OptimizerPolicySet:
    policy_set_id: str
    policies: tuple[OptimizerPolicy, ...]


NO_POLICY = OptimizerPolicy(policy_variant="no_policy")

_OPPONENT_OVERLAP_V1 = OptimizerPolicySet(
    policy_set_id="opponent-overlap-v1",
    policies=(
        NO_POLICY,
        OptimizerPolicy(policy_variant="soft_overlap_penalty_low", overlap_penalty=0.15),
        OptimizerPolicy(policy_variant="soft_overlap_penalty_medium", overlap_penalty=0.35),
        OptimizerPolicy(policy_variant="hard_max_overlap_3", max_overlap_assets=3),
        OptimizerPolicy(policy_variant="hard_max_overlap_2", max_overlap_assets=2),
    ),
)


def get_policy_set(policy_set_id: str) -> OptimizerPolicySet:
    if policy_set_id == _OPPONENT_OVERLAP_V1.policy_set_id:
        return _OPPONENT_OVERLAP_V1
    raise ValueError(f"Unknown policy set: {policy_set_id}")


class FixtureCoverageError(ValueError):
    pass


class DuplicateCandidateError(ValueError):
    pass


def fixture_signature(fixtures: pd.DataFrame) -> str:
    missing = _missing_columns(fixtures, _FIXTURE_COLUMNS)
    if missing:
        raise ValueError(f"Missing fixture signature columns: {', '.join(missing)}")

    records = (
        fixtures.loc[:, list(_FIXTURE_COLUMNS)]
        .astype({column: int for column in _FIXTURE_COLUMNS})
        .sort_values(list(_FIXTURE_COLUMNS), kind="mergesort")
        .to_dict("records")
    )
    payload = json.dumps(records, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def validate_fixture_coverage(
    fixtures: pd.DataFrame,
    *,
    candidate_club_ids: Iterable[int],
    round_number: int,
) -> None:
    missing = _missing_columns(fixtures, _FIXTURE_COLUMNS)
    if missing:
        raise FixtureCoverageError(f"Missing fixture coverage columns: {', '.join(missing)}")

    round_fixtures = fixtures.loc[fixtures["rodada"].astype(int).eq(int(round_number)), list(_FIXTURE_COLUMNS)]
    club_counts: dict[int, int] = {}
    for row in round_fixtures.to_dict("records"):
        for column in ("id_clube_home", "id_clube_away"):
            club_id = int(row[column])
            club_counts[club_id] = club_counts.get(club_id, 0) + 1

    duplicated_clubs = sorted(club_id for club_id, fixture_count in club_counts.items() if fixture_count > 1)
    if duplicated_clubs:
        raise FixtureCoverageError(
            f"Club appears in more than one fixture for round {round_number}: {duplicated_clubs}"
        )

    missing_clubs = sorted(int(club_id) for club_id in candidate_club_ids if int(club_id) not in club_counts)
    if missing_clubs:
        raise FixtureCoverageError(
            f"Round {round_number} has missing fixture coverage for candidate clubs: {missing_clubs}"
        )


def normalize_policy_candidates(candidates: pd.DataFrame, *, score_column: str) -> pd.DataFrame:
    critical_columns = ("rodada", "id_atleta", "id_clube", "posicao", "preco_pre_rodada", score_column)
    missing = _missing_columns(candidates, critical_columns)
    if missing:
        raise DuplicateCandidateError(f"Missing duplicate-normalization columns: {', '.join(missing)}")
    if candidates.empty:
        return candidates.iloc[0:0].copy()

    kept_rows: list[pd.Series] = []
    for key, group in candidates.groupby(["rodada", "id_atleta"], dropna=False, sort=False):
        critical_values = group.loc[:, list(critical_columns)].drop_duplicates()
        if len(critical_values) > 1:
            raise DuplicateCandidateError(f"Conflicting duplicate candidate rows for {key}")

        richest_index = group.notna().sum(axis=1).sort_values(ascending=False, kind="mergesort").index[0]
        kept_rows.append(group.loc[richest_index])

    return (
        pd.DataFrame(kept_rows)
        .sort_values(["rodada", "id_atleta", "id_clube", "posicao"], kind="mergesort")
        .reset_index(drop=True)
    )


def _missing_columns(frame: pd.DataFrame, required_columns: tuple[str, ...]) -> list[str]:
    return [column for column in required_columns if column not in frame.columns]
