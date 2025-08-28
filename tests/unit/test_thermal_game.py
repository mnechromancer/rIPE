"""
Tests for GAME-003: Thermogenesis Trade-off Game
"""

import pytest
import numpy as np
from ipe.core.games.thermal_game import ThermalGame


class DummyEnv:
    temperature = 5.0
    PO2 = 0.9


def test_valid_thermogenesis():
    game = ThermalGame(
        players=1,
        strategies=["shivering", "non_shivering"],
        constraints={"environment_temp": 5.0},
    )
    env = DummyEnv()
    fitness = game.compute_payoff(0.6, 0.4, env)
    assert isinstance(fitness, float)
    assert fitness < 10  # Should be reasonable


def test_invalid_thermogenesis_sum():
    game = ThermalGame(
        players=1,
        strategies=["shivering", "non_shivering"],
        constraints={"environment_temp": 5.0},
    )
    env = DummyEnv()
    with pytest.raises(ValueError):
        game.compute_payoff(0.7, 0.4, env)


def test_environmental_response():
    game = ThermalGame(
        players=1,
        strategies=["shivering", "non_shivering"],
        constraints={"environment_temp": 5.0},
    )
    env = DummyEnv()
    fitness_cold = game.compute_payoff(0.5, 0.5, env)
    env.temperature = 30.0
    fitness_warm = game.compute_payoff(0.5, 0.5, env)
    assert fitness_cold < fitness_warm  # More heat needed in cold
