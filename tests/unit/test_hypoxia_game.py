"""
Tests for GAME-002: Hypoxia Allocation Game
"""

import numpy as np
from ipe.core.games.hypoxia_game import HypoxiaAllocationGame
from ipe.core.games.tissue_allocation import validate_allocation


class DummyEnv:
    PO2 = 0.8


def test_valid_allocation():
    game = HypoxiaAllocationGame(players=1, strategies=["alloc"], constraints=None)
    allocation = np.array([0.3, 0.3, 0.2, 0.2])
    env = DummyEnv()
    assert validate_allocation(allocation, game.tissues, game.min_requirements)
    fitness = game.compute_payoff(allocation, env)
    assert fitness > 0


def test_invalid_allocation_sum():
    game = HypoxiaAllocationGame(players=1, strategies=["alloc"], constraints=None)
    allocation = np.array([0.5, 0.3, 0.1, 0.1])  # Sums to 1.0, but test edge
    DummyEnv()
    assert validate_allocation(allocation, game.tissues, game.min_requirements)
    allocation_bad = np.array([0.5, 0.3, 0.1, 0.3])  # Sums to >1
    assert not validate_allocation(allocation_bad, game.tissues, game.min_requirements)


def test_minimum_requirement_failure():
    game = HypoxiaAllocationGame(players=1, strategies=["alloc"], constraints=None)
    allocation = np.array([0.1, 0.1, 0.7, 0.1])  # Brain/heart below min
    env = DummyEnv()
    assert not validate_allocation(allocation, game.tissues, game.min_requirements)
    fitness = game.compute_payoff(allocation, env)
    assert fitness == 0.0
