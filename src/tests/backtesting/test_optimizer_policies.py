import hashlib
import json

import pandas as pd
import pytest

from cartola.backtesting.optimizer_policies import (
    DuplicateCandidateError,
    FixtureCoverageError,
    count_opponent_overlap,
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


def test_gk_conflict_v1_policy_set_is_frozen() -> None:
    policy_set = get_policy_set("gk-conflict-v1")

    assert [policy.policy_variant for policy in policy_set.policies] == [
        "no_policy",
        "gk_vs_selected_ata_soft_low",
        "gk_vs_selected_ata_soft_medium",
        "gk_vs_opponent_captain_soft",
        "gk_vs_opponent_attack_hard",
    ]
    assert policy_set.policies[1].gk_opponent_attack_penalty == 0.5
    assert policy_set.policies[1].gk_opponent_attack_positions == ("ata",)
    assert policy_set.policies[2].gk_opponent_attack_penalty == 1.0
    assert policy_set.policies[2].gk_opponent_attack_positions == ("ata",)
    assert policy_set.policies[3].gk_opponent_captain_penalty == 2.0
    assert policy_set.policies[3].gk_opponent_captain_positions == ("ata", "mei")
    assert policy_set.policies[4].max_gk_opponent_attack_pairs == 0
    assert policy_set.policies[4].gk_opponent_attack_positions == ("ata",)


def test_clean_sheet_stack_v1_policy_set_is_frozen() -> None:
    policy_set = get_policy_set("clean-sheet-stack-v1")

    assert [policy.policy_variant for policy in policy_set.policies] == [
        "no_policy",
        "home_cs_pair_bonus_025",
        "home_cs_pair_bonus_050",
        "home_cs_pair_bonus_075",
        "home_cs_pair_bonus_100",
    ]
    assert [policy.clean_sheet_pair_bonus for policy in policy_set.policies] == [
        0.0,
        0.25,
        0.50,
        0.75,
        1.00,
    ]
    for policy in policy_set.policies[1:]:
        assert policy.clean_sheet_pair_anchor_position == "gol"
        assert policy.clean_sheet_pair_partner_positions == ("lat", "zag")
        assert policy.clean_sheet_pair_min_ppg_diff == 0.75
        assert policy.clean_sheet_pair_min_xg_diff == 0.20
        assert policy.clean_sheet_pair_home_only is True
        assert policy.max_clean_sheet_pair_bonuses == 1


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


def test_fixture_signature_rejects_non_integral_fixture_values() -> None:
    fixtures = pd.DataFrame([{"rodada": 5.9, "id_clube_home": 1, "id_clube_away": 2}])

    with pytest.raises(ValueError, match="whole-number"):
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


def test_fixture_coverage_rejects_non_integral_fixture_values() -> None:
    fixtures = pd.DataFrame([{"rodada": 5, "id_clube_home": 1.8, "id_clube_away": 2}])

    with pytest.raises(FixtureCoverageError, match="whole-number"):
        validate_fixture_coverage(fixtures, candidate_club_ids={1, 2}, round_number=5)


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
        columns=pd.Index(
            ["rodada", "id_atleta", "id_clube", "posicao", "preco_pre_rodada", "model_score", "apelido"]
        )
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


def test_normalize_policy_candidates_uses_positional_richest_row_with_duplicate_indexes() -> None:
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
        ],
        index=pd.Index([0, 0]),
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


def test_count_opponent_overlap_counts_assets_and_matches_with_both_sides_selected() -> None:
    selected = pd.DataFrame(
        [
            {"id_atleta": 1, "id_clube": 10, "posicao": "gol"},
            {"id_atleta": 2, "id_clube": 10, "posicao": "tec"},
            {"id_atleta": 3, "id_clube": 20, "posicao": "ata"},
            {"id_atleta": 4, "id_clube": 30, "posicao": "mei"},
        ]
    )
    fixtures = pd.DataFrame(
        [
            {"rodada": 5, "id_clube_home": 10, "id_clube_away": 20},
            {"rodada": 5, "id_clube_home": 30, "id_clube_away": 40},
        ]
    )

    counts = count_opponent_overlap(selected, fixtures)

    assert counts.opponent_overlap_asset_count == 3
    assert counts.opponent_overlap_match_count == 1


def test_count_opponent_overlap_returns_zero_for_one_sided_fixture_selection() -> None:
    selected = pd.DataFrame(
        [
            {"id_atleta": 1, "id_clube": 10, "posicao": "gol"},
            {"id_atleta": 2, "id_clube": 10, "posicao": "tec"},
        ]
    )
    fixtures = pd.DataFrame([{"rodada": 5, "id_clube_home": 10, "id_clube_away": 20}])

    counts = count_opponent_overlap(selected, fixtures)

    assert counts.opponent_overlap_asset_count == 0
    assert counts.opponent_overlap_match_count == 0
