"""
Tests for GAME-001: Base Game Specification System
"""

import pytest
import numpy as np
from ipe.core.games.base import GameSpecification

class DummyGame(GameSpecification):
    def compute_payoff_matrix(self) -> np.ndarray:
        return np.array([[1, 0], [0, 1]])
    def validate_strategies(self, strategies):
        return True

def test_game_specification_abstract():
    game = DummyGame(players=2, strategies=["A", "B"])
    assert game.players == 2
    assert game.strategies == ["A", "B"]
    assert game.validate_strategies(["A", "B"]) is True
    matrix = game.compute_payoff_matrix()
    assert isinstance(matrix, np.ndarray)
    assert matrix.shape == (2, 2)

def test_game_serialization():
    game = DummyGame(players=2, strategies=["A", "B"])
    game.payoff_matrix = np.array([[1, 0], [0, 1]])
    serialized = game.serialize()
    assert "players" in serialized
    assert "strategies" in serialized
    assert "payoff_matrix" in serialized
