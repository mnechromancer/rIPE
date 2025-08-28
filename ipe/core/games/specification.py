"""
GAME-001: Game Specification Implementations

This module provides concrete implementations and helpers for game specifications.
"""

from .base import GameSpecification
from typing import List
import numpy as np


class SymmetricGame(GameSpecification):
    """
    Implementation for a symmetric game.
    """

    def compute_payoff_matrix(self) -> np.ndarray:
        """
        Compute a symmetric payoff matrix (example: identity matrix).
        Returns:
            np.ndarray: Payoff matrix.
        """
        n = len(self.strategies)
        self.payoff_matrix = np.eye(n)
        return self.payoff_matrix

    def validate_strategies(self, strategies: List[str]) -> bool:
        """
        Validate that all strategies are in allowed set and meet constraints.
        Returns:
            bool: True if valid, False otherwise.
        """
        allowed = set(self.strategies)
        for s in strategies:
            if s not in allowed:
                return False
        # Example constraint: no duplicate strategies
        if len(set(strategies)) != len(strategies):
            return False
        # Additional constraints can be checked here
        return True


class AsymmetricGame(GameSpecification):
    """
    Implementation for an asymmetric game.
    """

    def compute_payoff_matrix(self) -> np.ndarray:
        """
        Compute an asymmetric payoff matrix (example: random values).
        Returns:
            np.ndarray: Payoff matrix.
        """
        n = len(self.strategies)
        self.payoff_matrix = np.random.rand(n, n)
        return self.payoff_matrix

    def validate_strategies(self, strategies: List[str]) -> bool:
        """
        Validate strategies for asymmetric game (example: must match player count).
        Returns:
            bool: True if valid, False otherwise.
        """
        if len(strategies) != self.players:
            return False
        allowed = set(self.strategies)
        for s in strategies:
            if s not in allowed:
                return False
        return True
