from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

BUDGET_POLICY_MOVING = "moving"
BUDGET_POLICY_FIXED = "fixed"
BUDGET_CONSTRAINT_TOLERANCE = 1e-6


@dataclass(frozen=True)
class BudgetState:
    current_budget: float
    peak_budget: float
    min_budget: float
    max_drawdown: float


@dataclass(frozen=True)
class BudgetRoundUpdate:
    budget_before_round: float
    budget_after_round: float
    budget_delta: float
    budget_remaining: float
    budget_peak: float
    budget_drawdown: float
    is_budget_constrained: bool
    next_state: BudgetState


def initial_budget_state(initial_budget: float) -> BudgetState:
    budget = float(initial_budget)
    if not np.isfinite(budget) or budget <= 0:
        raise ValueError("initial budget must be a positive finite value")
    return BudgetState(
        current_budget=budget,
        peak_budget=budget,
        min_budget=budget,
        max_drawdown=0.0,
    )


def normalize_budget_policy(value: object) -> str:
    if value is None or pd.isna(value):
        return BUDGET_POLICY_FIXED
    policy = str(value).strip()
    if not policy:
        return BUDGET_POLICY_FIXED
    if policy not in {BUDGET_POLICY_MOVING, BUDGET_POLICY_FIXED}:
        raise ValueError(f"Unknown budget policy: {policy}")
    return policy


def advance_budget(state: BudgetState, selected: pd.DataFrame, *, budget_used: float) -> BudgetRoundUpdate:
    budget_before = float(state.current_budget)
    used = float(budget_used)
    if not np.isfinite(used) or used < 0:
        raise ValueError("budget_used must be a non-negative finite value")

    budget_delta = _selected_variation_sum(selected)
    budget_after = budget_before + budget_delta
    budget_remaining = budget_before - used
    budget_peak = max(float(state.peak_budget), budget_after)
    min_budget = min(float(state.min_budget), budget_before, budget_after)
    budget_drawdown = budget_peak - budget_after
    max_drawdown = max(float(state.max_drawdown), budget_drawdown)
    next_state = BudgetState(
        current_budget=budget_after,
        peak_budget=budget_peak,
        min_budget=min_budget,
        max_drawdown=max_drawdown,
    )
    return BudgetRoundUpdate(
        budget_before_round=budget_before,
        budget_after_round=budget_after,
        budget_delta=budget_delta,
        budget_remaining=budget_remaining,
        budget_peak=budget_peak,
        budget_drawdown=budget_drawdown,
        is_budget_constrained=budget_remaining <= BUDGET_CONSTRAINT_TOLERANCE,
        next_state=next_state,
    )


def _selected_variation_sum(selected: pd.DataFrame) -> float:
    if selected.empty:
        return 0.0
    if "variacao" not in selected.columns:
        raise ValueError("Selected squad is missing required variacao column")
    try:
        values = pd.to_numeric(selected["variacao"], errors="raise").astype(float)
    except (TypeError, ValueError) as exc:
        raise ValueError("Selected squad variacao must contain finite numeric values") from exc
    if values.isna().any() or not np.isfinite(values).all():
        raise ValueError("Selected squad variacao must contain finite numeric values")
    return float(values.sum())
