"""The stated objective must govern the code, not just describe it.

market_evolution_service declares OBJECTIVE_PROFIT_FACTOR and
OBJECTIVE_COST_BPS at the top of the module. These tests fail if any
threshold, default or fitness weight quietly contradicts them -- which is
how the search spent 1347 generations optimising accuracy while losing
money, on a cost lower than gas alone.
"""
import inspect

from scripts import market_evolution_service as mes


def test_objective_is_declared():
    """The goal must exist as a value the code can read, not a comment."""
    assert isinstance(mes.OBJECTIVE_PROFIT_FACTOR, float)
    assert mes.OBJECTIVE_PROFIT_FACTOR > 1.0, "the objective must be profit"
    assert isinstance(mes.OBJECTIVE_COST_BPS, float)
    assert mes.OBJECTIVE_COST_BPS > 0, "trading is never free"


def test_evaluation_cost_default_is_not_below_true_cost():
    """Rule 2: measuring below real cost manufactures profit.

    The old default was 20 bps while GAS_ROUNDTRIP_FEE_RATIO alone is 25.
    """
    signature = inspect.signature(mes.main) if hasattr(mes, "main") else None
    assert signature is not None
    source = inspect.getsource(mes.main)
    assert "default=OBJECTIVE_COST_BPS" in source, (
        "--cost-bps must default to the objective's true cost"
    )


def test_success_target_honours_the_objective():
    """No 'working target' may declare success below the objective."""
    assert mes.WORKING_TARGET["profit_factor"] >= mes.OBJECTIVE_PROFIT_FACTOR


def test_admission_floor_requires_real_profit():
    """The hard floor must demand profit, not merely break-even."""
    assert mes.FLOOR["profit_factor"] > 1.0


def test_profit_outranks_every_tie_breaker():
    """Rule 1, enforced numerically rather than asserted in a comment."""
    def score(pf, accuracy, coverage, mcc, expectancy):
        return mes.curriculum_fitness(
            fold_count=3, min_accuracy=accuracy, min_balanced=accuracy - 0.01,
            min_mcc=mcc, min_margin=0.01, min_coverage=coverage,
            min_observations=180, min_expectancy=expectancy, min_profit=pf,
            max_ece=0.12, max_drawdown=0.8, conditional_ghost_pass=False,
            conditional_ghost_accuracy=0.0,
        )

    earner = score(1.15, 0.55, 0.45, 0.05, 0.0018)
    # A rival that wins on EVERY tie-breaker but loses money must still rank
    # below the earner.
    better_at_everything_else = score(0.95, 0.75, 0.90, 0.40, -0.0010)
    assert earner > better_at_everything_else


def test_unmeasured_profit_cannot_outrank_measured_profit():
    """Rule 2, enforced numerically.

    A single-fold PF of 1.377 measured 0.936 across three folds, so it must
    never outrank a smaller profit that survived the full walk-forward.
    """
    def score(pf, folds):
        return mes.curriculum_fitness(
            fold_count=folds, min_accuracy=0.60, min_balanced=0.59,
            min_mcc=0.20, min_margin=0.01, min_coverage=0.50,
            min_observations=150, min_expectancy=0.002, min_profit=pf,
            max_ece=0.12, max_drawdown=0.8, conditional_ghost_pass=False,
            conditional_ghost_accuracy=0.0,
        )

    assert score(1.12, 3) > score(1.377, 1)


def test_the_objective_is_stated_in_the_source():
    """A future reader must find the goal before the thresholds."""
    source = inspect.getsource(mes)
    header = source[:source.index("FLOOR = {")]
    assert "THE OBJECTIVE" in header
    assert "OBJECTIVE_PROFIT_FACTOR" in header
