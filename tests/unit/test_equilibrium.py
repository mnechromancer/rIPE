"""
Tests for GAME-004: Equilibrium Solver
"""

import numpy as np
from ipe.core.games.equilibrium import EquilibriumSolver
from ipe.core.games.nash_solver import iterative_best_response, stability_analysis


def test_nash_equilibrium():
    game = EquilibriumSolver(players=2, strategies=["A", "B"], constraints=None)
    game.compute_payoff_matrix()
    eq = game.compute_nash_equilibrium()
    assert isinstance(eq, list)
    assert "A" in eq or "B" in eq


def test_ess_detection():
    game = EquilibriumSolver(players=2, strategies=["A", "B"], constraints=None)
    ess = game.detect_ess()
    assert isinstance(ess, list)
    assert set(ess) == set(["A", "B"])


def test_invasion_fitness():
    game = EquilibriumSolver(players=2, strategies=["A", "B"], constraints=None)
    game.compute_payoff_matrix()
    fit = game.compute_invasion_fitness("A", "B")
    assert isinstance(fit, float)


def test_iterative_best_response():
    matrix = np.eye(2)
    strategies = ["A", "B"]
    eq = iterative_best_response(matrix, strategies)
    assert eq == strategies


def test_stability_analysis():
    matrix = np.eye(2)
    assert stability_analysis(matrix) is True
