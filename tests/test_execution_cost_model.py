"""A round-trip fee may only be charged when a position actually changes.

metrics() previously did `pnl = predicted * realized - cost_bps/10_000` on
EVERY observation. `predicted` is a direction (+1/-1/0), so continuing to
hold the same direction was billed a full entry+exit. At a 12-period horizon
that overstates cost by the average holding length.

Measured 2026-08-17 on champion cf9b7c8b2e7675fd at the real 25 bps cost:
per-bar charging gave PF ~0.948; charging only real position changes gives
PF 1.1210 with positive expectancy. The gross edge (PF ~1.18 before fees)
was always there -- the accounting was wrong.
"""
import numpy as np

from scripts.market_evolution_service import position_turnover
from scripts.market_signal_audit import metrics


def rows(asset_times):
    return [{"asset": a, "timestamp": t} for a, t in asset_times]


def test_flipping_every_bar_is_charged_every_bar():
    """No discount when the model genuinely trades every period."""
    r = rows([("A", i) for i in range(6)])
    turnover = position_turnover(r, np.array([1, -1, 1, -1, 1, -1]))
    assert turnover.sum() == 6.0


def test_holding_is_charged_once():
    """Entering once and holding costs one round trip, not six."""
    r = rows([("A", i) for i in range(6)])
    turnover = position_turnover(r, np.array([1, 1, 1, 1, 1, 1]))
    assert turnover.sum() == 1.0
    assert turnover[0] == 1.0


def test_each_switch_costs_a_round_trip():
    r = rows([("A", i) for i in range(6)])
    turnover = position_turnover(r, np.array([1, 1, -1, -1, 1, 1]))
    assert turnover.sum() == 3.0


def test_assets_are_accounted_separately():
    """Holding two assets means two entries, not one."""
    r = rows([("A", 0), ("B", 0), ("A", 1), ("B", 1)])
    turnover = position_turnover(r, np.array([1, 1, 1, 1]))
    assert turnover.sum() == 2.0


def test_flat_is_never_free():
    """A zero direction is not a held position, so it earns no discount."""
    r = rows([("A", i) for i in range(6)])
    turnover = position_turnover(r, np.zeros(6, dtype=int))
    assert turnover.sum() == 6.0


def test_turnover_is_order_independent_of_input_sequence():
    """Rows arrive unordered; turnover must follow time within each asset."""
    forward = rows([("A", 0), ("A", 1), ("A", 2)])
    shuffled = rows([("A", 2), ("A", 0), ("A", 1)])
    direction = np.array([1, 1, 1])
    assert position_turnover(forward, direction).sum() == 1.0
    assert position_turnover(shuffled, direction).sum() == 1.0


def test_metrics_without_turnover_stays_conservative():
    """Omitting turnover must fall back to per-bar charging.

    That can only understate profit, never invent it.
    """
    predicted = np.array([1, 1, 1, 1])
    realized = np.array([0.01, 0.01, 0.01, 0.01])
    actual = np.array([1, 1, 1, 1])
    prob = np.array([0.7, 0.7, 0.7, 0.7])
    per_bar = metrics(actual, predicted, prob, realized, 25.0)
    held = metrics(actual, predicted, prob, realized, 25.0,
                   turnover=np.array([1.0, 0.0, 0.0, 0.0]))
    assert held["net_expectancy"] > per_bar["net_expectancy"]


def test_turnover_cannot_manufacture_profit_when_flipping():
    """If the model trades every bar, the corrected model must agree."""
    predicted = np.array([1, -1, 1, -1])
    realized = np.array([0.01, -0.01, 0.01, -0.01])
    actual = np.array([1, -1, 1, -1])
    prob = np.array([0.7, 0.7, 0.7, 0.7])
    per_bar = metrics(actual, predicted, prob, realized, 25.0)
    corrected = metrics(actual, predicted, prob, realized, 25.0,
                        turnover=np.ones(4))
    assert corrected["net_expectancy"] == per_bar["net_expectancy"]
