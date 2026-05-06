import pandas as pd
import pytest

from cartola.backtesting.budgeting import (
    BUDGET_CONSTRAINT_TOLERANCE,
    BUDGET_POLICY_FIXED,
    BUDGET_POLICY_MOVING,
    BudgetState,
    advance_budget,
    initial_budget_state,
    normalize_budget_policy,
)


def test_initial_budget_state_sets_current_peak_and_minimum() -> None:
    state = initial_budget_state(100.0)

    assert state.current_budget == 100.0
    assert state.peak_budget == 100.0
    assert state.min_budget == 100.0
    assert state.max_drawdown == 0.0


@pytest.mark.parametrize("initial_budget", [0, -1, float("nan"), float("inf")])
def test_initial_budget_state_rejects_non_positive_or_non_finite_budget(initial_budget: float) -> None:
    with pytest.raises(ValueError, match="initial budget must be a positive finite value"):
        initial_budget_state(initial_budget)


def test_advance_budget_sums_selected_variacao_including_tecnico() -> None:
    selected = pd.DataFrame(
        {
            "id_atleta": [1, 2, 3],
            "posicao": ["gol", "ata", "tec"],
            "variacao": [-1.5, 2.0, -0.2],
        }
    )
    state = initial_budget_state(100.0)

    update = advance_budget(state, selected, budget_used=96.0)

    assert update.budget_before_round == 100.0
    assert update.budget_delta == pytest.approx(0.3)
    assert update.budget_after_round == pytest.approx(100.3)
    assert update.budget_remaining == pytest.approx(4.0)
    assert update.next_state.current_budget == pytest.approx(100.3)
    assert update.next_state.peak_budget == pytest.approx(100.3)
    assert update.next_state.min_budget == pytest.approx(100.0)
    assert update.next_state.max_drawdown == pytest.approx(0.0)


def test_advance_budget_tracks_drawdown_from_peak() -> None:
    state = BudgetState(current_budget=105.0, peak_budget=110.0, min_budget=100.0, max_drawdown=2.0)
    selected = pd.DataFrame({"variacao": [-8.0]})

    update = advance_budget(state, selected, budget_used=90.0)

    assert update.budget_after_round == 97.0
    assert update.budget_peak == 110.0
    assert update.budget_drawdown == 13.0
    assert update.next_state.min_budget == 97.0
    assert update.next_state.max_drawdown == 13.0


def test_advance_budget_preserves_zero_or_negative_budget_path() -> None:
    selected = pd.DataFrame({"variacao": [-105.0]})

    update = advance_budget(initial_budget_state(100.0), selected, budget_used=99.0)

    assert update.budget_after_round == -5.0
    assert update.next_state.current_budget == -5.0
    assert update.next_state.min_budget == -5.0


def test_advance_budget_fails_when_selected_variacao_column_is_missing() -> None:
    selected = pd.DataFrame({"id_atleta": [1]})

    with pytest.raises(ValueError, match="Selected squad is missing required variacao column"):
        advance_budget(initial_budget_state(100.0), selected, budget_used=10.0)


@pytest.mark.parametrize("bad_value", [None, float("nan"), float("inf"), "bad"])
def test_advance_budget_fails_when_selected_variacao_is_invalid(bad_value: object) -> None:
    selected = pd.DataFrame({"id_atleta": [1], "variacao": [bad_value]})

    with pytest.raises(ValueError, match="Selected squad variacao must contain finite numeric values"):
        advance_budget(initial_budget_state(100.0), selected, budget_used=10.0)


def test_advance_budget_allows_empty_selected_squad_without_budget_change() -> None:
    selected = pd.DataFrame({"variacao": pd.Series(dtype=float)})

    update = advance_budget(initial_budget_state(100.0), selected, budget_used=0.0)

    assert update.budget_before_round == 100.0
    assert update.budget_delta == 0.0
    assert update.budget_after_round == 100.0
    assert update.budget_remaining == 100.0
    assert update.is_budget_constrained is False


def test_budget_constrained_flag_uses_tolerance() -> None:
    selected = pd.DataFrame({"variacao": [0.0]})

    constrained = advance_budget(
        initial_budget_state(100.0),
        selected,
        budget_used=100.0 - (BUDGET_CONSTRAINT_TOLERANCE / 2),
    )
    not_constrained = advance_budget(
        initial_budget_state(100.0),
        selected,
        budget_used=100.0 - (BUDGET_CONSTRAINT_TOLERANCE * 2),
    )

    assert constrained.is_budget_constrained is True
    assert not_constrained.is_budget_constrained is False
    assert constrained.budget_remaining == pytest.approx(BUDGET_CONSTRAINT_TOLERANCE / 2)


def test_normalize_budget_policy_treats_missing_as_fixed() -> None:
    assert normalize_budget_policy(None) == BUDGET_POLICY_FIXED
    assert normalize_budget_policy("") == BUDGET_POLICY_FIXED
    assert normalize_budget_policy(BUDGET_POLICY_MOVING) == BUDGET_POLICY_MOVING
    assert normalize_budget_policy(BUDGET_POLICY_FIXED) == BUDGET_POLICY_FIXED


def test_normalize_budget_policy_rejects_unknown_policy() -> None:
    with pytest.raises(ValueError, match="Unknown budget policy"):
        normalize_budget_policy("legacy")
