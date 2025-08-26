"""
GAME-004: Nash Solver Utilities

Helpers for Nash equilibrium and stability analysis.
"""

from typing import List, Any
import numpy as np

def iterative_best_response(payoff_matrix: np.ndarray, strategies: List[str], max_iter: int = 100) -> List[Any]:
    """
    Dummy iterative best response dynamics.
    Args:
        payoff_matrix: np.ndarray of payoffs
        strategies: List of strategies
        max_iter: Maximum iterations
    Returns:
        List[Any]: List of equilibrium strategies
    """
    # Placeholder: return all strategies
    return strategies

def stability_analysis(payoff_matrix: np.ndarray) -> bool:
    """
    Dummy stability analysis.
    Args:
        payoff_matrix: np.ndarray of payoffs
    Returns:
        bool: True if stable, False otherwise
    """
    # Placeholder: always stable
    return True
