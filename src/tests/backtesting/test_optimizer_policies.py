import hashlib
import json

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


def test_get_policy_set_rejects_unknown_policy_set_id() -> None:
    with pytest.raises(ValueError, match="Unknown policy set"):
        get_policy_set("missing-policy-set")


def test_fixture_signature_is_order_stable() -> None:
    left = pd.DataFrame(
        [
            {"rodada": 2, "id_clube_home": 30, "id_clube_away": 40},
            {"rodada": 1, "id_clube_home": 10, "id_clube_away": 20},
        ]
    )
    right = left.iloc[[1, 0]].reset_index(drop=True)

    assert fixture_signature(left) == fixture_signature(right)


def test_fixture_signature_rejects_missing_required_columns() -> None:
    fixtures = pd.DataFrame([{"rodada": 1, "id_clube_home": 10}])

    with pytest.raises(ValueError, match="Missing fixture signature columns"):
        fixture_signature(fixtures)


def test_fixture_signature_uses_canonical_json_sha256_with_integer_values() -> None:
    fixtures = pd.DataFrame(
        [
            {"rodada": 2, "id_clube_home": 30, "id_clube_away": 40},
            {"rodada": 1, "id_clube_home": 10, "id_clube_away": 20},
        ]
    )
    canonical_records = [
        {"rodada": 1, "id_clube_home": 10, "id_clube_away": 20},
        {"rodada": 2, "id_clube_home": 30, "id_clube_away": 40},
    ]
    canonical_json = json.dumps(canonical_records, sort_keys=True, separators=(",", ":")).encode("utf-8")
    expected = hashlib.sha256(canonical_json).hexdigest()

    assert fixture_signature(fixtures) == expected


def test_fixture_coverage_rejects_missing_fixture_columns() -> None:
    fixtures = pd.DataFrame([{"rodada": 5, "id_clube_home": 1}])

    with pytest.raises(FixtureCoverageError, match="Missing fixture coverage columns"):
        validate_fixture_coverage(fixtures, candidate_club_ids={1}, round_number=5)


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


def test_normalize_policy_candidates_rejects_missing_critical_columns() -> None:
    rows = pd.DataFrame(
        [
            {
                "rodada": 5,
                "id_atleta": 10,
                "id_clube": 1,
                "posicao": "ata",
                "preco_pre_rodada": 8.0,
            }
        ]
    )

    with pytest.raises(DuplicateCandidateError, match="Missing duplicate-normalization columns"):
        normalize_policy_candidates(rows, score_column="model_score")


def test_normalize_policy_candidates_returns_empty_copy_for_empty_input() -> None:
    rows = pd.DataFrame(
        columns=["rodada", "id_atleta", "id_clube", "posicao", "preco_pre_rodada", "model_score", "apelido"]
    )

    normalized = normalize_policy_candidates(rows, score_column="model_score")

    assert normalized.empty
    assert normalized is not rows
    assert normalized.columns.tolist() == rows.columns.tolist()


def test_normalize_policy_candidates_returns_empty_copy_for_truly_empty_input() -> None:
    rows = pd.DataFrame()

    normalized = normalize_policy_candidates(rows, score_column="model_score")

    assert normalized.empty
    assert normalized is not rows
    assert normalized.columns.tolist() == []


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


def test_normalize_policy_candidates_sorts_by_round_player_club_and_position() -> None:
    rows = pd.DataFrame(
        [
            {
                "rodada": 6,
                "id_atleta": 1,
                "id_clube": 1,
                "posicao": "ata",
                "preco_pre_rodada": 8.0,
                "model_score": 4.0,
            },
            {
                "rodada": 5,
                "id_atleta": 2,
                "id_clube": 1,
                "posicao": "mei",
                "preco_pre_rodada": 7.0,
                "model_score": 3.0,
            },
            {
                "rodada": 5,
                "id_atleta": 3,
                "id_clube": 2,
                "posicao": "zag",
                "preco_pre_rodada": 6.0,
                "model_score": 2.0,
            },
            {
                "rodada": 5,
                "id_atleta": 1,
                "id_clube": 1,
                "posicao": "ata",
                "preco_pre_rodada": 5.0,
                "model_score": 1.0,
            },
        ]
    )

    normalized = normalize_policy_candidates(rows, score_column="model_score")

    assert normalized.loc[:, ["rodada", "id_atleta", "id_clube", "posicao"]].to_dict("records") == [
        {"rodada": 5, "id_atleta": 1, "id_clube": 1, "posicao": "ata"},
        {"rodada": 5, "id_atleta": 2, "id_clube": 1, "posicao": "mei"},
        {"rodada": 5, "id_atleta": 3, "id_clube": 2, "posicao": "zag"},
        {"rodada": 6, "id_atleta": 1, "id_clube": 1, "posicao": "ata"},
    ]


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
