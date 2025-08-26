"""
GAME-003: Thermogenesis Trade-off Game

Implements shivering vs non-shivering thermogenesis, O2 cost, and heat balance.
"""

from ipe.core.games.base import GameSpecification
from typing import Any, Dict, List
import numpy as np

class ThermalGame(GameSpecification):
    def __init__(self, players: int, strategies: List[str], constraints: Dict[str, Any] = None):
        super().__init__(players, strategies, constraints)
        self.environment_temp = constraints.get('environment_temp', 20.0) if constraints else 20.0

    def compute_payoff(self, shivering: float, non_shivering: float, environment: Any) -> float:
        """
        Calculate fitness based on thermogenesis trade-off and environment.
        Args:
            shivering: Fraction of shivering thermogenesis (0-1)
            non_shivering: Fraction of non-shivering thermogenesis (0-1)
            environment: Dict or object with temperature and O2
        Returns:
            float: Fitness value
        """
        if not np.isclose(shivering + non_shivering, 1.0):
            raise ValueError("Thermogenesis fractions must sum to 1")
        temp = environment.get('temperature', self.environment_temp) if isinstance(environment, dict) else getattr(environment, 'temperature', self.environment_temp)
        po2 = environment.get('PO2', 1.0) if isinstance(environment, dict) else getattr(environment, 'PO2', 1.0)
        # Example: O2 cost is higher for shivering
        o2_cost = shivering * 2.0 + non_shivering * 1.0
        # Heat balance: more heat needed at lower temp
        heat_needed = max(0.0, 37.0 - temp)
        # Fitness: heat produced minus O2 cost, scaled by PO2
        heat_produced = shivering * 10.0 + non_shivering * 8.0
        fitness = (heat_produced - o2_cost) * po2 - heat_needed
        return float(fitness)

    def compute_payoff_matrix(self) -> np.ndarray:
        """
        Dummy implementation for abstract method. Returns identity matrix.
        """
        n = len(self.strategies)
        self.payoff_matrix = np.eye(n)
        return self.payoff_matrix

    def validate_strategies(self, strategies: List[str]) -> bool:
        """
        Dummy implementation for abstract method. Checks if all strategies are valid.
        """
        allowed = set(self.strategies)
        for s in strategies:
            if s not in allowed:
                return False
        return True
