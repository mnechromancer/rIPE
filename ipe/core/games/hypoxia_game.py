"""
GAME-002: Hypoxia Allocation Game

Implements multi-tissue O2 allocation optimization and fitness calculation.
"""

from ipe.core.games.base import GameSpecification
from typing import List, Dict, Any
import numpy as np


class HypoxiaAllocationGame(GameSpecification):
    tissues: List[str] = ["brain", "heart", "muscle", "brown_fat"]
    min_requirements: Dict[str, float] = {
        "brain": 0.2,
        "heart": 0.2,
        "muscle": 0.1,
        "brown_fat": 0.05,
    }

    def compute_payoff(self, allocation: np.ndarray, environment: Any) -> float:
        """
        Calculate fitness from O2 allocation and environment.
        Args:
            allocation: np.ndarray of tissue allocations (sums to 1)
            environment: PhysiologicalState or dict with PO2
        Returns:
            float: Integrated fitness value
        """
        if not np.isclose(np.sum(allocation), 1.0):
            raise ValueError("Allocation must sum to 1")
        for i, tissue in enumerate(self.tissues):
            if allocation[i] < self.min_requirements[tissue]:
                return 0.0  # Fails minimum requirement
        # Example: fitness is sum of allocation * PO2
        po2 = (
            environment.get("PO2", 1.0)
            if isinstance(environment, dict)
            else getattr(environment, "PO2", 1.0)
        )
        tissue_perf = allocation * po2
        return float(np.sum(tissue_perf))

    def compute_payoff_matrix(self) -> np.ndarray:
        """
        Dummy implementation for abstract method. Returns identity matrix.
        """
        n = len(self.tissues)
        self.payoff_matrix = np.eye(n)
        return self.payoff_matrix

    def validate_strategies(self, strategies: List[str]) -> bool:
        """
        Dummy implementation for abstract method. Checks if all strategies are 
        valid tissues.
        """
        allowed = set(self.tissues)
        for s in strategies:
            if s not in allowed:
                return False
        return True
