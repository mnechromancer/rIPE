"""
GAME-002: Tissue Allocation Utilities

Helpers for tissue-specific allocation and validation.
"""

from typing import List, Dict
import numpy as np


def validate_allocation(
    allocation: np.ndarray, tissues: List[str], min_requirements: Dict[str, float]
) -> bool:
    """
    Validate allocation vector against tissue minimums.
    Args:
        allocation: np.ndarray of allocations
        tissues: List of tissue names
        min_requirements: Dict of minimum required allocation per tissue
    Returns:
        bool: True if valid, False otherwise
    """
    if not np.isclose(np.sum(allocation), 1.0):
        return False
    for i, tissue in enumerate(tissues):
        if allocation[i] < min_requirements[tissue]:
            return False
    return True
