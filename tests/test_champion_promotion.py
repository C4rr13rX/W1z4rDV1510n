"""A better earner must never be suppressed silently.

Context (2026-08-18): champion_replacement_allowed enforced max_ece and
max_drawdown as a strict Pareto bound against the incumbent. Genome
701a3dfda8afcf6e beat the champion on EVERY economic metric -- profit factor
1.1965 vs 1.0567, expectancy 3x, better accuracy, MCC, coverage and margin --
and was refused the title because its calibration error was 0.178 vs 0.087.
A second genome was blocked identically. Nothing in the logs said so; the only
symptom was a champion that quietly stopped improving.

The objective states profit decides and calibration/drawdown are
tie-breakers, so they are now absolute safety ceilings rather than vetoes a
more profitable genome can never clear.
"""
import math

import pytest

from scripts.market_evolution_service import (
    Genome,
    champion_replacement_allowed,
    champion_replacement_blockers,
)


def make(genome_id: str, *, fitness: float, pf: float, ece: float,
         drawdown: float, accuracy: float = 0.56, coverage: float = 0.70,
         expectancy: float = 0.002, mcc: float = 0.11, folds: int = 3) -> Genome:
    return Genome(
        genome_id=genome_id,
        features=["r2"],
        learning_rate=0.1, max_iter=100, max_leaf_nodes=31,
        min_samples_leaf=20, l2_regularization=0.0,
        confidence_quantile=0.25, binding_threshold=0.5,
        concept_threshold=5, presentations=1,
        fitness=fitness,
        result={
            "evaluated_folds": folds,
            "requested_folds": 3,
            "summary": {
                "min_profit_factor": pf, "min_accuracy": accuracy,
                "min_balanced_accuracy": accuracy - 0.01, "min_mcc": mcc,
                "min_baseline_margin": 0.01, "min_coverage": coverage,
                "min_expectancy": expectancy, "max_ece": ece,
                "max_drawdown": drawdown,
            },
        },
    )


INCUMBENT = make("incumbent", fitness=1669.4, pf=1.0567, ece=0.0867,
                 drawdown=0.7819, accuracy=0.5159, coverage=0.6508,
                 expectancy=0.00068, mcc=0.0444)


def test_the_real_suppressed_genome_is_now_promotable():
    """The exact case that stalled the search."""
    challenger = make("701a3dfda8afcf6e", fitness=2644.4, pf=1.1965,
                      ece=0.1783, drawdown=0.8437, accuracy=0.5610,
                      coverage=0.7540, expectancy=0.00205, mcc=0.1148)
    assert champion_replacement_allowed(challenger, INCUMBENT)
    assert champion_replacement_blockers(challenger, INCUMBENT) == []


def test_worse_calibration_alone_cannot_veto_more_profit():
    """Calibration is a tie-breaker, not a profit veto."""
    challenger = make("c", fitness=2000.0, pf=1.25, ece=0.20, drawdown=0.80)
    assert champion_replacement_allowed(challenger, INCUMBENT)


def test_safety_ceilings_still_refuse_reckless_candidates():
    """No amount of profit buys through an absolute risk ceiling."""
    reckless = make("r", fitness=9999.0, pf=3.0, ece=0.40, drawdown=1.9)
    assert not champion_replacement_allowed(reckless, INCUMBENT)
    reasons = {b.get("metric") for b in champion_replacement_blockers(reckless, INCUMBENT)}
    assert reasons == {"max_ece", "max_drawdown"}


def test_economic_regressions_are_still_refused():
    """Profit may not be bought by giving up the other economics."""
    regressed = make("g", fitness=5000.0, pf=1.30, ece=0.10, drawdown=0.75,
                     coverage=0.20, expectancy=-0.001)
    assert not champion_replacement_allowed(regressed, INCUMBENT)
    metrics = {b.get("metric") for b in champion_replacement_blockers(regressed, INCUMBENT)}
    assert "min_coverage" in metrics
    assert "min_expectancy" in metrics


def test_incomplete_walk_forward_is_refused():
    """Single-fold luck must never take the championship."""
    thin = make("t", fitness=9000.0, pf=1.38, ece=0.10, drawdown=0.70, folds=1)
    assert not champion_replacement_allowed(thin, INCUMBENT)
    reasons = {b.get("reason") for b in champion_replacement_blockers(thin, INCUMBENT)}
    assert "incomplete_walk_forward" in reasons


def test_lower_fitness_is_refused():
    weaker = make("w", fitness=100.0, pf=1.30, ece=0.09, drawdown=0.70)
    assert not champion_replacement_allowed(weaker, INCUMBENT)


def test_blockers_name_the_offending_metric_and_values():
    """Diagnosis must be readable, not a bare False."""
    reckless = make("r", fitness=9999.0, pf=3.0, ece=0.40, drawdown=0.70)
    blockers = champion_replacement_blockers(reckless, INCUMBENT)
    ece = next(b for b in blockers if b.get("metric") == "max_ece")
    assert ece["reason"] == "safety_ceiling"
    assert ece["candidate"] == pytest.approx(0.40)
    assert ece["ceiling"] == pytest.approx(0.25)

def test_a_coverage_regression_cannot_veto_a_better_earner():
    """Coverage is not profit, and gating on it cost real money.

    Measured on this run's event log: 40 candidates that out-earned the
    incumbent were suppressed, 23 of them blocked ONLY by a diagnostic
    regression, with min_coverage the single most common blocker (26 of 84).
    Genome df3a00203d29 earned PF 1.2132 against an incumbent's 1.1682 and was
    refused for trading less often (coverage 0.48 vs 0.80).

    Across 1069 full-fold genomes in the same run corr(coverage, expectancy)
    is -0.003: how often a genome trades says essentially nothing about what
    it earns.
    """
    incumbent = make("incumbent", fitness=2450.0, pf=1.1682, ece=0.09,
                     drawdown=0.78, coverage=0.8024, expectancy=0.0018)
    better = make("df3a00203d29", fitness=2734.1, pf=1.2132, ece=0.09,
                  drawdown=0.78, coverage=0.4799, expectancy=0.00215)
    assert champion_replacement_allowed(better, incumbent)


def test_a_profit_regression_still_blocks():
    """Profit and expectancy remain hard vetoes -- this is profit-FIRST, not
    profit-only-when-convenient."""
    incumbent = make("incumbent", fitness=2450.0, pf=1.20, ece=0.09,
                     drawdown=0.78, expectancy=0.0020)
    poorer = make("poorer", fitness=9999.0, pf=1.05, ece=0.01,
                  drawdown=0.10, expectancy=0.0020)
    assert not champion_replacement_allowed(poorer, incumbent)

    losing = make("losing", fitness=9999.0, pf=1.25, ece=0.01,
                  drawdown=0.10, expectancy=0.0001)
    assert not champion_replacement_allowed(losing, incumbent)


def test_a_diagnostic_regression_is_reported_even_though_it_does_not_block():
    """A promotion that trades accuracy for profit must stay auditable."""
    incumbent = make("incumbent", fitness=2450.0, pf=1.1682, ece=0.09,
                     drawdown=0.78, coverage=0.8024, expectancy=0.0018)
    better = make("better", fitness=2734.1, pf=1.2132, ece=0.09,
                  drawdown=0.78, coverage=0.4799, expectancy=0.00215)
    blockers = champion_replacement_blockers(better, incumbent)
    observed = [b for b in blockers if b["reason"] == "observed_regression"]
    assert any(b["metric"] == "min_coverage" for b in observed)
    # ...but nothing that actually blocks the promotion.
    assert not [b for b in blockers if b["reason"] == "regression"]

