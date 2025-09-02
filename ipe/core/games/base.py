"""
GAME-001: Base Game Specification System

This module implements the abstract base class for game specifications.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import numpy as np
import json


class GameSpecification(ABC):
    """
    Abstract base class for game specifications.
    Supports symmetric and asymmetric games.
    """

    players: int
    strategies: List[str]
    constraints: Dict[str, Any]
    payoff_matrix: Optional[np.ndarray]

    def __init__(
        self,
        players: int,
        strategies: List[str],
        constraints: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Initialize a game specification.
        Args:
            players: Number of players in the game.
            strategies: List of available strategies.
            constraints: Optional dictionary of strategy constraints.
        """
        self.players = players
        self.strategies = strategies
        self.constraints = constraints or {}
        self.payoff_matrix = None

    @abstractmethod
    def compute_payoff_matrix(self) -> np.ndarray:
        """
        Compute the payoff matrix for the game.
        Returns:
            np.ndarray: Payoff matrix of shape (n_strategies, n_strategies) or
                appropriate for game type.
        """

    @abstractmethod
    def validate_strategies(self, strategies: List[str]) -> bool:
        """
        Validate strategies against constraints.
        Args:
            strategies: List of strategies to validate.
        Returns:
            bool: True if valid, False otherwise.
        """

    def serialize(self) -> str:
        """
        Serialize the game specification to JSON.
        Returns:
            str: JSON string representing the game specification.
        """
        data = {
            "players": self.players,
            "strategies": self.strategies,
            "constraints": self.constraints,
            "payoff_matrix": (
                self.payoff_matrix.tolist() if self.payoff_matrix is not None else None
            ),
        }
        return json.dumps(data)

    @staticmethod
    def json_schema() -> Dict[str, Any]:
        """
        Return the JSON schema for game definitions.
        Returns:
            Dict[str, Any]: JSON schema dictionary.
        """
        return {
            "type": "object",
            "properties": {
                "players": {"type": "integer"},
                "strategies": {"type": "array", "items": {"type": "string"}},
                "constraints": {"type": "object"},
                "payoff_matrix": {"type": "array"},
            },
            "required": ["players", "strategies"],
        }
