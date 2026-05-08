import inspect

import pandas as pd
import pulp
import pytest

import cartola.backtesting.optimizer as optimizer
from cartola.backtesting.config import DEFAULT_FORMATIONS, BacktestConfig
from cartola.backtesting.optimizer import optimize_squad
from cartola.backtesting.optimizer_policies import NO_POLICY, OptimizerPolicy
from cartola.backtesting.scoring_contract import CAPTAIN_MULTIPLIER, SCORING_CONTRACT_VERSION


def _row(player_id: int, posicao: str, score: float, price: float = 5.0) -> dict[str, object]:
    return {
        "id_atleta": player_id,
        "apelido": f"{posicao}-{player_id}",
        "posicao": posicao,
        "clube": f"club-{posicao}",
        "preco": 999.0,
        "preco_pre_rodada": price,
        "predicted_points": score,
        "pontuacao": score - 1.0,
    }


def _candidates() -> pd.DataFrame:
    rows = []
    player_id = 1
    scores = {
        "gol": [8.0, 7.0],
        "lat": [5.0, 4.8, 4.6],
        "zag": [7.5, 7.0, 6.5, 6.0],
        "mei": [18.0, 17.0, 16.0, 15.0, 14.0, 4.0],
        "ata": [11.0, 10.0, 9.0, 3.0],
        "tec": [40.0, 6.0],
    }
    for posicao, position_scores in scores.items():
        for score in position_scores:
            rows.append(_row(player_id, posicao, score))
            player_id += 1
    return pd.DataFrame(rows)


def _policy_row(player_id: int, posicao: str, score: float, club_id: int) -> dict[str, object]:
    row = _row(player_id, posicao, score=score, price=1.0)
    row["id_clube"] = club_id
    row["clube"] = f"club-{club_id}"
    row["score"] = score
    return row


def _policy_candidates() -> pd.DataFrame:
    return pd.DataFrame(
        [
            _policy_row(1, "gol", 12.0, 1),
            _policy_row(2, "gol", 6.0, 10),
            _policy_row(3, "lat", 11.0, 2),
            _policy_row(4, "lat", 5.0, 11),
            _policy_row(5, "lat", 4.5, 12),
            _policy_row(6, "zag", 10.0, 1),
            _policy_row(7, "zag", 9.8, 2),
            _policy_row(8, "zag", 5.5, 13),
            _policy_row(9, "zag", 5.4, 14),
            _policy_row(10, "zag", 5.3, 15),
            _policy_row(11, "mei", 9.6, 1),
            _policy_row(12, "mei", 9.4, 2),
            _policy_row(13, "mei", 5.2, 16),
            _policy_row(14, "mei", 5.1, 17),
            _policy_row(15, "mei", 5.0, 18),
            _policy_row(16, "mei", 4.9, 19),
            _policy_row(17, "ata", 9.2, 1),
            _policy_row(18, "ata", 9.0, 2),
            _policy_row(19, "ata", 4.8, 20),
            _policy_row(20, "ata", 4.7, 21),
            _policy_row(21, "tec", 8.8, 2),
            _policy_row(22, "tec", 4.6, 22),
        ]
    )


def _policy_fixtures() -> pd.DataFrame:
    return pd.DataFrame([{"rodada": 5, "id_clube_home": 1, "id_clube_away": 2}])


def _gk_conflict_candidates() -> pd.DataFrame:
    rows = [
        _policy_row(1, "gol", 10.0, 1),
        _policy_row(2, "gol", 9.8, 3),
        _policy_row(3, "lat", 5.0, 4),
        _policy_row(4, "lat", 4.9, 5),
        _policy_row(5, "zag", 5.0, 6),
        _policy_row(6, "zag", 4.9, 7),
        _policy_row(7, "zag", 4.8, 8),
        _policy_row(8, "mei", 7.0, 9),
        _policy_row(9, "mei", 6.9, 10),
        _policy_row(10, "mei", 6.8, 11),
        _policy_row(11, "mei", 6.7, 12),
        _policy_row(12, "mei", 6.6, 13),
        _policy_row(20, "ata", 9.0, 2),
        _policy_row(21, "ata", 8.8, 2),
        _policy_row(22, "ata", 8.6, 14),
        _policy_row(30, "tec", 5.0, 15),
    ]
    return pd.DataFrame(rows)


def _clean_sheet_stack_candidates(
    matchup_is_home: int = 1,
    footystats_ppg_diff: float = 0.80,
    footystats_xg_diff: float = 0.25,
) -> pd.DataFrame:
    rows = [
        _policy_row(1, "gol", 9.0, 1),
        _policy_row(2, "gol", 9.1, 2),
        _policy_row(3, "zag", 8.0, 1),
        _policy_row(4, "zag", 7.9, 3),
        _policy_row(5, "zag", 7.8, 4),
        _policy_row(6, "lat", 7.7, 5),
        _policy_row(7, "lat", 7.6, 6),
        _policy_row(8, "mei", 20.0, 7),
        _policy_row(9, "mei", 19.0, 8),
        _policy_row(10, "mei", 18.0, 9),
        _policy_row(11, "mei", 17.0, 10),
        _policy_row(12, "mei", 16.0, 11),
        _policy_row(13, "ata", 15.0, 12),
        _policy_row(14, "ata", 14.0, 13),
        _policy_row(15, "ata", 13.0, 14),
        _policy_row(16, "tec", 12.0, 15),
    ]
    candidates = pd.DataFrame(rows)
    candidates["matchup_is_home"] = 0
    candidates["footystats_ppg_diff"] = 0.0
    candidates["footystats_xg_diff"] = 0.0
    eligible_club = candidates["id_clube"].eq(1)
    candidates.loc[eligible_club, "matchup_is_home"] = matchup_is_home
    candidates.loc[eligible_club, "footystats_ppg_diff"] = footystats_ppg_diff
    candidates.loc[eligible_club, "footystats_xg_diff"] = footystats_xg_diff
    return candidates


def _multi_clean_sheet_stack_candidates() -> pd.DataFrame:
    rows = [
        _policy_row(1, "gol", 9.0, 15),
        _policy_row(2, "gol", 8.9, 16),
        _policy_row(3, "lat", 8.0, 1),
        _policy_row(4, "lat", 7.9, 2),
        _policy_row(5, "lat", 7.8, 3),
        _policy_row(6, "lat", 7.7, 4),
        _policy_row(7, "zag", 8.0, 1),
        _policy_row(17, "zag", 7.9, 2),
        _policy_row(18, "zag", 7.8, 5),
        _policy_row(19, "zag", 7.7, 6),
        _policy_row(8, "mei", 20.0, 6),
        _policy_row(9, "mei", 19.0, 7),
        _policy_row(10, "mei", 18.0, 8),
        _policy_row(11, "mei", 17.0, 9),
        _policy_row(12, "mei", 16.0, 10),
        _policy_row(13, "ata", 15.0, 11),
        _policy_row(14, "ata", 14.0, 12),
        _policy_row(15, "ata", 13.0, 13),
        _policy_row(16, "tec", 12.0, 14),
    ]
    candidates = pd.DataFrame(rows)
    candidates["matchup_is_home"] = 1
    candidates["footystats_ppg_diff"] = 0.80
    candidates["footystats_xg_diff"] = 0.25
    return candidates


def test_optimizer_searches_all_formations_and_returns_captain_aware_scores() -> None:
    result = optimize_squad(_candidates(), score_column="predicted_points", config=BacktestConfig(budget=80))

    assert result.status == "Optimal"
    assert result.selected_count == 12
    assert result.formation_name == "3-5-2"
    assert result.selected.groupby("posicao").size().to_dict() == {
        "ata": 2,
        "gol": 1,
        "mei": 5,
        "tec": 1,
        "zag": 3,
    }
    assert result.selected["is_captain"].sum() == 1
    captain = result.selected.loc[result.selected["is_captain"]].iloc[0]
    assert captain["posicao"] != "tec"
    assert result.captain_id == captain["id_atleta"]
    assert result.captain_name == captain["apelido"]
    assert result.captain_position == captain["posicao"]
    assert result.captain_club == captain["clube"]
    assert result.captain_predicted_points == pytest.approx(captain["predicted_points"])
    assert result.scoring_contract_version == SCORING_CONTRACT_VERSION
    assert result.captain_multiplier == CAPTAIN_MULTIPLIER

    expected_base = float(result.selected["predicted_points"].sum())
    expected_bonus = (CAPTAIN_MULTIPLIER - 1.0) * float(captain["predicted_points"])
    assert result.predicted_points_base == pytest.approx(expected_base)
    assert result.captain_bonus_predicted == pytest.approx(expected_bonus)
    assert result.predicted_points_with_captain == pytest.approx(expected_base + expected_bonus)
    assert result.predicted_points == pytest.approx(result.predicted_points_with_captain)

    assert len(result.formation_scores) == len(DEFAULT_FORMATIONS)
    assert {score["formation"] for score in result.formation_scores} == set(DEFAULT_FORMATIONS)
    chosen_score = next(score for score in result.formation_scores if score["formation"] == result.formation_name)
    assert chosen_score["solver_status"] == "Optimal"
    assert chosen_score["captain_id"] == result.captain_id
    assert result.captain_policy_diagnostics == []


def test_hard_overlap_cap_forces_different_squad() -> None:
    candidates = _policy_candidates()
    fixtures = _policy_fixtures()
    config = BacktestConfig(season=2025, start_round=5, budget=100)

    no_policy = optimize_squad(candidates, "score", config, policy=None, fixtures_for_round=fixtures)
    capped = optimize_squad(
        candidates,
        "score",
        config,
        policy=OptimizerPolicy("hard_test", max_overlap_assets=2),
        fixtures_for_round=fixtures,
    )

    assert no_policy.status == "Optimal"
    assert capped.status == "Optimal"
    assert no_policy.opponent_overlap_asset_count > 2
    assert capped.opponent_overlap_asset_count <= 2
    assert no_policy.selected["id_atleta"].astype(int).tolist() != capped.selected["id_atleta"].astype(int).tolist()


def test_no_policy_selection_is_unchanged_with_fixture_context() -> None:
    candidates = _policy_candidates()
    fixtures = _policy_fixtures()
    config = BacktestConfig(season=2025, start_round=5, budget=100)

    baseline = optimize_squad(candidates, "score", config)
    with_none_policy = optimize_squad(candidates, "score", config, policy=None, fixtures_for_round=fixtures)
    with_no_policy = optimize_squad(candidates, "score", config, policy=NO_POLICY, fixtures_for_round=fixtures)

    assert with_none_policy.policy_variant == "no_policy"
    assert with_no_policy.policy_variant == "no_policy"
    assert baseline.selected["id_atleta"].astype(int).tolist() == with_none_policy.selected["id_atleta"].astype(int).tolist()
    assert baseline.selected["id_atleta"].astype(int).tolist() == with_no_policy.selected["id_atleta"].astype(int).tolist()
    assert baseline.captain_id == with_none_policy.captain_id == with_no_policy.captain_id
    assert baseline.predicted_points_with_captain == pytest.approx(with_none_policy.predicted_points_with_captain)
    assert baseline.predicted_points_with_captain == pytest.approx(with_no_policy.predicted_points_with_captain)
    assert baseline.clean_sheet_pair_count == 0
    assert baseline.clean_sheet_pair_bonus_applied == 0.0
    assert with_none_policy.clean_sheet_pair_count == 0
    assert with_none_policy.clean_sheet_pair_bonus_applied == 0.0
    assert with_no_policy.clean_sheet_pair_count == 0
    assert with_no_policy.clean_sheet_pair_bonus_applied == 0.0


def test_soft_overlap_penalty_changes_selection_only_when_large_enough() -> None:
    candidates = _policy_candidates()
    fixtures = _policy_fixtures()
    config = BacktestConfig(season=2025, start_round=5, budget=100)

    no_policy = optimize_squad(candidates, "score", config, fixtures_for_round=fixtures)
    low_penalty = optimize_squad(
        candidates,
        "score",
        config,
        policy=OptimizerPolicy("soft_low_test", overlap_penalty=0.01),
        fixtures_for_round=fixtures,
    )
    high_penalty = optimize_squad(
        candidates,
        "score",
        config,
        policy=OptimizerPolicy("soft_high_test", overlap_penalty=20.0),
        fixtures_for_round=fixtures,
    )

    no_policy_ids = no_policy.selected["id_atleta"].astype(int).tolist()
    assert low_penalty.selected["id_atleta"].astype(int).tolist() == no_policy_ids
    assert high_penalty.selected["id_atleta"].astype(int).tolist() != no_policy_ids
    assert high_penalty.opponent_overlap_asset_count < no_policy.opponent_overlap_asset_count


def test_gk_opponent_attack_soft_penalty_can_remove_conflicting_gk_pick() -> None:
    candidates = _gk_conflict_candidates()
    fixtures = pd.DataFrame([{"rodada": 5, "id_clube_home": 1, "id_clube_away": 2}])
    config = BacktestConfig(season=2025, start_round=5, budget=100)

    baseline = optimize_squad(
        candidates,
        "score",
        config,
        policy=OptimizerPolicy("gk_conflict_count_test", gk_opponent_attack_positions=("ata",)),
        fixtures_for_round=fixtures,
    )
    penalized = optimize_squad(
        candidates,
        "score",
        config,
        policy=OptimizerPolicy(
            "gk_conflict_test",
            gk_opponent_attack_penalty=10.0,
            gk_opponent_attack_positions=("ata",),
        ),
        fixtures_for_round=fixtures,
    )

    assert baseline.status == "Optimal"
    assert penalized.status == "Optimal"
    assert baseline.gk_opponent_attack_pair_count > 0
    assert penalized.gk_opponent_attack_pair_count == 0
    assert int(baseline.selected.loc[baseline.selected["posicao"].eq("gol"), "id_clube"].iloc[0]) == 1
    assert int(penalized.selected.loc[penalized.selected["posicao"].eq("gol"), "id_clube"].iloc[0]) == 3


def test_gk_opponent_attack_hard_cap_blocks_conflicting_ata_pair() -> None:
    candidates = _gk_conflict_candidates()
    fixtures = pd.DataFrame([{"rodada": 5, "id_clube_home": 1, "id_clube_away": 2}])
    config = BacktestConfig(season=2025, start_round=5, budget=100)

    result = optimize_squad(
        candidates,
        "score",
        config,
        policy=OptimizerPolicy(
            "gk_conflict_hard_test",
            max_gk_opponent_attack_pairs=0,
            gk_opponent_attack_positions=("ata",),
        ),
        fixtures_for_round=fixtures,
    )

    assert result.status == "Optimal"
    assert result.gk_opponent_attack_pair_count == 0


def test_gk_opponent_captain_penalty_targets_attacking_midfielder_captain_only() -> None:
    candidates = _gk_conflict_candidates()
    candidates.loc[candidates["id_atleta"].eq(21), ["posicao", "score"]] = ["mei", 16.0]
    fixtures = pd.DataFrame([{"rodada": 5, "id_clube_home": 1, "id_clube_away": 2}])
    config = BacktestConfig(season=2025, start_round=5, budget=100)

    baseline = optimize_squad(
        candidates,
        "score",
        config,
        policy=OptimizerPolicy("gk_captain_count_test", gk_opponent_captain_positions=("ata", "mei")),
        fixtures_for_round=fixtures,
    )
    penalized = optimize_squad(
        candidates,
        "score",
        config,
        policy=OptimizerPolicy(
            "gk_captain_conflict_test",
            gk_opponent_captain_penalty=20.0,
            gk_opponent_captain_positions=("ata", "mei"),
        ),
        fixtures_for_round=fixtures,
    )

    assert baseline.status == "Optimal"
    assert penalized.status == "Optimal"
    assert baseline.gk_opponent_captain_count == 1
    assert penalized.gk_opponent_captain_count == 0
    assert int(baseline.selected.loc[baseline.selected["posicao"].eq("gol"), "id_clube"].iloc[0]) == 1
    assert int(penalized.selected.loc[penalized.selected["posicao"].eq("gol"), "id_clube"].iloc[0]) == 3


def test_clean_sheet_pair_bonus_can_select_eligible_goalkeeper_defender_pair_without_fixtures() -> None:
    candidates = _clean_sheet_stack_candidates()
    config = BacktestConfig(season=2025, start_round=5, budget=100)

    baseline = optimize_squad(candidates, "score", config)
    stacked = optimize_squad(
        candidates,
        "score",
        config,
        policy=OptimizerPolicy(
            "clean_sheet_pair_test",
            clean_sheet_pair_bonus=0.25,
            max_clean_sheet_pair_bonuses=1,
        ),
    )

    assert baseline.status == "Optimal"
    assert stacked.status == "Optimal"
    assert int(baseline.selected.loc[baseline.selected["posicao"].eq("gol"), "id_clube"].iloc[0]) == 2
    assert int(stacked.selected.loc[stacked.selected["posicao"].eq("gol"), "id_clube"].iloc[0]) == 1
    assert stacked.clean_sheet_pair_count == 1
    assert stacked.clean_sheet_pair_bonus_applied == pytest.approx(0.25)


@pytest.mark.parametrize(
    ("matchup_is_home", "footystats_ppg_diff", "footystats_xg_diff"),
    [
        (0, 0.80, 0.25),
        (1, 0.74, 0.25),
        (1, 0.80, 0.19),
    ],
)
def test_clean_sheet_pair_bonus_does_not_apply_when_context_is_ineligible(
    matchup_is_home: int,
    footystats_ppg_diff: float,
    footystats_xg_diff: float,
) -> None:
    candidates = _clean_sheet_stack_candidates(
        matchup_is_home=matchup_is_home,
        footystats_ppg_diff=footystats_ppg_diff,
        footystats_xg_diff=footystats_xg_diff,
    )
    config = BacktestConfig(season=2025, start_round=5, budget=100)

    result = optimize_squad(
        candidates,
        "score",
        config,
        policy=OptimizerPolicy(
            "clean_sheet_pair_test",
            clean_sheet_pair_bonus=1.0,
            max_clean_sheet_pair_bonuses=1,
        ),
    )

    assert result.status == "Optimal"
    assert int(result.selected.loc[result.selected["posicao"].eq("gol"), "id_clube"].iloc[0]) == 2
    assert result.clean_sheet_pair_count == 0
    assert result.clean_sheet_pair_bonus_applied == 0.0


@pytest.mark.parametrize(
    ("column", "conflicting_value"),
    [
        ("matchup_is_home", 0),
        ("footystats_ppg_diff", 0.70),
        ("footystats_xg_diff", 0.15),
    ],
)
def test_clean_sheet_pair_bonus_rejects_conflicting_same_club_context(
    column: str,
    conflicting_value: object,
) -> None:
    candidates = _clean_sheet_stack_candidates()
    candidates.loc[candidates["id_atleta"].eq(3), column] = conflicting_value
    config = BacktestConfig(season=2025, start_round=5, budget=100)

    with pytest.raises(ValueError, match=column):
        optimize_squad(
            candidates,
            "score",
            config,
            policy=OptimizerPolicy(
                "clean_sheet_pair_test",
                clean_sheet_pair_bonus=0.25,
                max_clean_sheet_pair_bonuses=1,
            ),
        )


def test_clean_sheet_pair_bonus_rejects_non_binary_home_context() -> None:
    candidates = _clean_sheet_stack_candidates(matchup_is_home=2)
    config = BacktestConfig(season=2025, start_round=5, budget=100)

    with pytest.raises(ValueError, match="matchup_is_home"):
        optimize_squad(
            candidates,
            "score",
            config,
            policy=OptimizerPolicy(
                "clean_sheet_pair_test",
                clean_sheet_pair_bonus=0.25,
                max_clean_sheet_pair_bonuses=1,
            ),
        )


def test_clean_sheet_pair_bonus_cap_limits_multi_club_pairs() -> None:
    candidates = _multi_clean_sheet_stack_candidates()
    config = BacktestConfig(season=2025, start_round=5, budget=100)

    uncapped = optimize_squad(
        candidates,
        "score",
        config,
        policy=OptimizerPolicy(
            "clean_sheet_pair_uncapped_test",
            clean_sheet_pair_bonus=10.0,
            clean_sheet_pair_anchor_position="zag",
            clean_sheet_pair_partner_positions=("lat",),
        ),
    )
    capped = optimize_squad(
        candidates,
        "score",
        config,
        policy=OptimizerPolicy(
            "clean_sheet_pair_capped_test",
            clean_sheet_pair_bonus=10.0,
            clean_sheet_pair_anchor_position="zag",
            clean_sheet_pair_partner_positions=("lat",),
            max_clean_sheet_pair_bonuses=1,
        ),
    )

    assert uncapped.status == "Optimal"
    assert capped.status == "Optimal"
    assert uncapped.clean_sheet_pair_count == 2
    assert capped.clean_sheet_pair_count == 1
    assert capped.clean_sheet_pair_bonus_applied == pytest.approx(10.0)


def test_clean_sheet_pair_bonus_does_not_apply_without_selected_anchor_partner_pair() -> None:
    candidates = _clean_sheet_stack_candidates()
    candidates.loc[candidates["id_atleta"].eq(3), "id_clube"] = 16
    config = BacktestConfig(season=2025, start_round=5, budget=100)

    result = optimize_squad(
        candidates,
        "score",
        config,
        policy=OptimizerPolicy(
            "clean_sheet_pair_no_cheating_test",
            clean_sheet_pair_bonus=100.0,
            max_clean_sheet_pair_bonuses=1,
        ),
    )

    assert result.status == "Optimal"
    assert result.clean_sheet_pair_count == 0
    assert result.clean_sheet_pair_bonus_applied == 0.0


def test_optimizer_never_captains_unselected_phantom_player_or_tecnico() -> None:
    candidates = _candidates()
    phantom = _row(999, "ata", score=1000.0, price=500.0)
    candidates = pd.concat([candidates, pd.DataFrame([phantom])], ignore_index=True)

    result = optimize_squad(candidates, score_column="predicted_points", config=BacktestConfig(budget=80))

    assert result.status == "Optimal"
    assert result.captain_id != 999
    assert 999 not in set(result.selected["id_atleta"])
    assert result.captain_position != "tec"
    assert result.captain_predicted_points == pytest.approx(18.0)


def test_optimizer_rejects_high_scoring_unexpected_position_from_selected_roster() -> None:
    candidates = _candidates()
    unexpected = _row(999, "ban", score=1000.0, price=0.1)
    candidates = pd.concat([candidates, pd.DataFrame([unexpected])], ignore_index=True)

    result = optimize_squad(candidates, score_column="predicted_points", config=BacktestConfig(budget=80))

    assert result.status == "Optimal"
    assert result.selected_count == 12
    assert 999 not in set(result.selected["id_atleta"])
    assert "ban" not in set(result.selected["posicao"])
    assert sum(result.selected.groupby("posicao").size().to_dict().values()) == 12


def test_optimizer_reports_per_formation_scores_for_infeasible_and_chosen_formations() -> None:
    candidates = _candidates().loc[lambda frame: frame["posicao"] != "lat"].copy()

    result = optimize_squad(candidates, score_column="predicted_points", config=BacktestConfig(budget=80))

    assert result.status == "Optimal"
    assert result.formation_name in {"3-4-3", "3-5-2"}
    assert len(result.formation_scores) == len(DEFAULT_FORMATIONS)
    assert any(score["solver_status"] == "Infeasible" for score in result.formation_scores)
    assert any(
        score["formation"] == result.formation_name and score["solver_status"] == "Optimal"
        for score in result.formation_scores
    )


def test_optimizer_breaks_exact_ties_with_lower_player_and_captain_ids() -> None:
    rows = []
    player_id = 1
    for posicao, count in {
        "gol": 2,
        "lat": 2,
        "zag": 4,
        "mei": 6,
        "ata": 4,
        "tec": 2,
    }.items():
        for _ in range(count):
            rows.append(_row(player_id, posicao, score=5.0, price=5.0))
            player_id += 1
    candidates = pd.DataFrame(rows).sample(frac=1, random_state=7).reset_index(drop=True)

    results = [
        optimize_squad(candidates, score_column="predicted_points", config=BacktestConfig(budget=80))
        for _ in range(3)
    ]

    selected_ids = [tuple(result.selected["id_atleta"].astype(int)) for result in results]
    captain_ids = [result.captain_id for result in results]
    assert len(set(selected_ids)) == 1
    assert len(set(captain_ids)) == 1
    assert selected_ids[0] == (1, 5, 6, 7, 9, 10, 11, 12, 15, 16, 17, 19)
    assert captain_ids[0] == 1


def test_tie_break_objective_penalizes_captain_variables_separately() -> None:
    player_rows = pd.DataFrame({"id_atleta": [10, 20]})
    selected_vars = {
        0: pulp.LpVariable("selected_0", cat=pulp.LpBinary),
        1: pulp.LpVariable("selected_1", cat=pulp.LpBinary),
    }
    captain_vars = {
        0: pulp.LpVariable("captain_0", cat=pulp.LpBinary),
        1: pulp.LpVariable("captain_1", cat=pulp.LpBinary),
    }

    expression = optimizer._tie_break_objective(player_rows, selected_vars, captain_vars)
    coefficients = dict(expression.items())

    assert coefficients[selected_vars[0]] == pytest.approx(-10.0)
    assert coefficients[selected_vars[1]] == pytest.approx(-20.0)
    assert coefficients[captain_vars[0]] < 0.0
    assert coefficients[captain_vars[1]] < coefficients[captain_vars[0]]


def test_optimizer_reports_all_formations_without_exposing_fixed_formation_api() -> None:
    candidates = _candidates().loc[lambda frame: frame["posicao"] != "lat"].copy()

    result = optimize_squad(candidates, score_column="predicted_points", config=BacktestConfig(budget=80))

    assert result.status == "Optimal"
    assert len(result.formation_scores) == len(DEFAULT_FORMATIONS)
    assert {score["formation"] for score in result.formation_scores} == set(DEFAULT_FORMATIONS)
    assert any(score["solver_status"] == "Infeasible" for score in result.formation_scores)
    assert "formation_name" not in inspect.signature(optimize_squad).parameters


def test_optimizer_returns_empty_result_for_empty_candidates() -> None:
    result = optimize_squad(pd.DataFrame(), score_column="predicted_points", config=BacktestConfig())

    assert result.status == "Empty"
    assert result.selected.empty
    assert result.selected_count == 0
    assert result.budget_used == 0
    assert result.predicted_points == 0
    assert result.predicted_points_base == 0
    assert result.captain_bonus_predicted == 0
    assert result.predicted_points_with_captain == 0
    assert result.formation_name == ""
    assert result.captain_id is None
    assert result.scoring_contract_version == SCORING_CONTRACT_VERSION
    assert result.captain_multiplier == CAPTAIN_MULTIPLIER
    assert result.clean_sheet_pair_count == 0
    assert result.clean_sheet_pair_bonus_applied == 0.0


def test_optimizer_reports_infeasible_budget_with_formation_scores() -> None:
    result = optimize_squad(_candidates(), score_column="predicted_points", config=BacktestConfig(budget=1))

    assert result.status == "Infeasible"
    assert result.selected.empty
    assert result.selected_count == 0
    assert len(result.formation_scores) == len(DEFAULT_FORMATIONS)
    assert {score["solver_status"] for score in result.formation_scores} == {"Infeasible"}
    assert all(score["infeasibility_reason"] for score in result.formation_scores)


def test_optimizer_deduplicates_candidates_by_player_before_solving() -> None:
    duplicate = _candidates().iloc[[0]].copy()
    duplicate["preco"] = 0.1
    duplicate["preco_pre_rodada"] = 0.1
    duplicate["predicted_points"] = 1000.0
    candidates = pd.concat([_candidates(), duplicate], ignore_index=True)

    result = optimize_squad(candidates, score_column="predicted_points", config=BacktestConfig(budget=80))

    assert result.status == "Optimal"
    assert result.selected_count == 12
    assert result.selected["id_atleta"].is_unique
    assert (
        result.selected.loc[result.selected["id_atleta"] == duplicate.iloc[0]["id_atleta"], "preco_pre_rodada"].iloc[0]
        == 5.0
    )
    assert result.selected.loc[result.selected["id_atleta"] == duplicate.iloc[0]["id_atleta"], "predicted_points"].iloc[
        0
    ] == pytest.approx(8.0)


def test_optimizer_uses_pre_round_price_for_budget() -> None:
    candidates = _candidates()
    candidates["preco"] = 100.0
    candidates["preco_pre_rodada"] = 5.0

    result = optimize_squad(candidates, score_column="predicted_points", config=BacktestConfig(budget=80))

    assert result.status == "Optimal"
    assert result.selected_count == 12
    assert result.budget_used == 60.0


@pytest.mark.parametrize("column", ["id_atleta", "apelido", "posicao", "preco_pre_rodada", "predicted_points"])
def test_optimizer_rejects_missing_required_columns(column: str) -> None:
    candidates = _candidates().drop(columns=[column])

    with pytest.raises(ValueError, match=column):
        optimize_squad(candidates, score_column="predicted_points", config=BacktestConfig(budget=80))


@pytest.mark.parametrize("column", ["preco_pre_rodada", "predicted_points"])
def test_optimizer_rejects_non_numeric_solver_columns(column: str) -> None:
    candidates = _candidates()
    candidates[column] = candidates[column].astype(object)
    candidates.loc[0, column] = "bad"

    with pytest.raises(ValueError, match=column):
        optimize_squad(candidates, score_column="predicted_points", config=BacktestConfig(budget=80))
