import pandas as pd
import pytest

from cartola.backtesting.optimizer_policies import (
    DuplicateCandidateError,
    FixtureCoverageError,
    fixture_signature,
    get_policy_set,
    normalize_policy_candidates,
    validate_fixture_coverage,
)


def test_opponent_overlap_v1_policy_set_is_frozen() -> None:
    policy_set = get_policy_set("opponent-overlap-v1")

    assert [policy.policy_variant for policy in policy_set.policies] == [
        "no_policy",
        "soft_overlap_penalty_low",
        "soft_overlap_penalty_medium",
        "hard_max_overlap_3",
        "hard_max_overlap_2",
    ]
    assert policy_set.policies[0].overlap_penalty == 0.0
    assert policy_set.policies[1].overlap_penalty == 0.15
    assert policy_set.policies[2].overlap_penalty == 0.35
    assert policy_set.policies[3].max_overlap_assets == 3
    assert policy_set.policies[4].max_overlap_assets == 2


def test_fixture_signature_is_order_stable() -> None:
    left = pd.DataFrame(
        [
            {"rodada": 2, "id_clube_home": 30, "id_clube_away": 40},
            {"rodada": 1, "id_clube_home": 10, "id_clube_away": 20},
        ]
    )
    right = left.iloc[[1, 0]].reset_index(drop=True)

    assert fixture_signature(left) == fixture_signature(right)


def test_fixture_coverage_rejects_duplicate_club_in_round() -> None:
    fixtures = pd.DataFrame(
        [
            {"rodada": 5, "id_clube_home": 1, "id_clube_away": 2},
            {"rodada": 5, "id_clube_home": 1, "id_clube_away": 3},
        ]
    )

    with pytest.raises(FixtureCoverageError, match="more than one fixture"):
        validate_fixture_coverage(fixtures, candidate_club_ids={1, 2, 3}, round_number=5)


def test_fixture_coverage_rejects_missing_candidate_club() -> None:
    fixtures = pd.DataFrame([{"rodada": 5, "id_clube_home": 1, "id_clube_away": 2}])

    with pytest.raises(FixtureCoverageError, match="missing fixture coverage"):
        validate_fixture_coverage(fixtures, candidate_club_ids={1, 2, 3}, round_number=5)


def test_normalize_policy_candidates_keeps_richest_equivalent_duplicate() -> None:
    rows = pd.DataFrame(
        [
            {
                "rodada": 5,
                "id_atleta": 10,
                "id_clube": 1,
                "posicao": "ata",
                "preco_pre_rodada": 8.0,
                "model_score": 4.0,
                "apelido": None,
            },
            {
                "rodada": 5,
                "id_atleta": 10,
                "id_clube": 1,
                "posicao": "ata",
                "preco_pre_rodada": 8.0,
                "model_score": 4.0,
                "apelido": "A10",
            },
        ]
    )

    normalized = normalize_policy_candidates(rows, score_column="model_score")

    assert len(normalized) == 1
    assert normalized.iloc[0]["apelido"] == "A10"


def test_normalize_policy_candidates_rejects_conflicting_duplicate() -> None:
    rows = pd.DataFrame(
        [
            {
                "rodada": 5,
                "id_atleta": 10,
                "id_clube": 1,
                "posicao": "ata",
                "preco_pre_rodada": 8.0,
                "model_score": 4.0,
            },
            {
                "rodada": 5,
                "id_atleta": 10,
                "id_clube": 1,
                "posicao": "ata",
                "preco_pre_rodada": 9.0,
                "model_score": 4.0,
            },
        ]
    )

    with pytest.raises(DuplicateCandidateError, match="Conflicting duplicate candidate"):
        normalize_policy_candidates(rows, score_column="model_score")
