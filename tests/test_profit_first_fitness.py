"""The market GA must optimise for money, not for being right.

Context (2026-08-17): after 1347 generations the champion sat at 55.6%
accuracy with profit factor 0.981 -- accurate and losing money. Two causes:

  * curriculum_fitness weighted accuracy 20x profit (500 * accuracy vs
    25 * profit_factor), so the search was never really pointed at profit.
  * evaluate_genome broke out of the walk-forward loop when a candidate
    missed the non-economic prescreen floors, recording its optimistic
    fold-1 profit as the verdict. Re-running the 12 best such candidates on
    full folds collapsed PF 1.23-1.38 to 0.79-0.98 -- every single one.

These tests pin the corrected objective.
"""
import pytest

from scripts.market_evolution_service import (
    PROFIT_CONTINUE_FLOOR,
    curriculum_fitness,
)


def fitness(pf, accuracy, coverage, expectancy, *, folds=3, observations=180,
            mcc=0.15, margin=0.01):
    return curriculum_fitness(
        fold_count=folds, min_accuracy=accuracy,
        min_balanced=accuracy - 0.01, min_mcc=mcc, min_margin=margin,
        min_coverage=coverage, min_observations=observations,
        min_expectancy=expectancy, min_profit=pf, max_ece=0.12,
        max_drawdown=0.8, conditional_ghost_pass=False,
        conditional_ghost_accuracy=0.0,
    )


def test_profit_beats_accuracy():
    """A 70%-accurate money-loser must lose to a 56%-accurate earner."""
    loser = fitness(0.95, 0.70, 0.70, -0.0010)
    earner = fitness(1.20, 0.56, 0.55, 0.0025)
    assert earner > loser


def test_profit_beats_coverage():
    """Trading more often is worthless if the trades lose money."""
    wide_loser = fitness(0.97, 0.55, 0.85, -0.0004)
    selective_earner = fitness(1.18, 0.62, 0.45, 0.0022)
    assert selective_earner > wide_loser


def test_the_real_champion_loses_to_a_real_earner():
    """The live champion (PF 0.981) must rank below an actual earner."""
    champion = fitness(0.981, 0.556, 0.621, -0.00005)
    earner = fitness(1.15, 0.60, 0.50, 0.0020)
    assert earner > champion


@pytest.mark.parametrize("lower,higher", [
    (1.00, 1.05), (1.05, 1.10), (1.10, 1.20), (1.20, 1.30),
])
def test_fitness_is_monotonic_in_profit(lower, higher):
    """More profit must always score higher -- no dead zones."""
    assert (fitness(higher, 0.58, 0.55, (higher - 1) * 0.01)
            > fitness(lower, 0.58, 0.55, (lower - 1) * 0.01))


def test_single_fold_luck_cannot_beat_measured_profit():
    """The exact failure that wasted 1347 generations.

    A fold-1 PF of 1.377 (which measured 0.936 across three folds) must not
    outrank a smaller profit that actually survived the walk-forward.
    """
    fold1_luck = fitness(1.377, 0.675, 0.419, 0.0037, folds=1, observations=110)
    fully_measured = fitness(1.120, 0.590, 0.520, 0.0015, folds=3, observations=170)
    assert fully_measured > fold1_luck


def test_losses_are_punished_harder_than_gains_are_rewarded():
    """Capital preservation: losing must cost more than winning pays."""
    breakeven = fitness(1.00, 0.55, 0.60, 0.0)
    gain = fitness(1.10, 0.55, 0.60, 0.001)
    loss = fitness(0.90, 0.55, 0.60, -0.001)
    assert breakeven > loss
    assert (breakeven - loss) > (gain - breakeven)


def test_completing_the_walk_forward_is_never_penalised():
    """A candidate must not gain by being abandoned early."""
    for pf in (0.95, 1.00, 1.15, 1.30):
        partial = fitness(pf, 0.60, 0.55, 0.001, folds=1)
        complete = fitness(pf, 0.60, 0.55, 0.001, folds=3)
        assert complete > partial, f"early exit outranked full folds at PF {pf}"


def test_profit_continue_floor_requires_at_least_breakeven():
    """Profitable candidates earn more folds; losers still exit early."""
    assert PROFIT_CONTINUE_FLOOR >= 1.0
