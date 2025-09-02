"""
State Space Management System

Implements efficient state storage, spatial indexing, and neighbor search for
physiological states.
"""

from typing import List, Optional, Dict
import numpy as np
from scipy.spatial import KDTree
from ipe.core.physiology.state import PhysiologicalState


class StateSpace:
    """
    Efficient storage and spatial indexing for physiological states.
    """

    def __init__(self, dimensions: Dict[str, int]):
        self.dimensions = dimensions
        self.states: List[PhysiologicalState] = []
        self.index: Optional[KDTree] = None
        self._cache = {}

    def add_state(self, state: PhysiologicalState) -> int:
        """Add state and return ID"""
        self.states.append(state)
        self.index = None  # Invalidate index for lazy rebuild
        return len(self.states) - 1

    def build_index(self):
        """Build KDTree index lazily"""
        if not self.states:
            return
        arr = np.array([self._state_to_vector(s) for s in self.states])
        self.index = KDTree(arr)

    def find_neighbors(
        self, state: PhysiologicalState, radius: float
    ) -> List[PhysiologicalState]:
        """Find states within radius using KDTree"""
        if self.index is None:
            self.build_index()
        arr = np.array([self._state_to_vector(state)])
        idxs = self.index.query_ball_point(arr, r=radius)[0]
        return [self.states[i] for i in idxs]

    def _state_to_vector(self, state: PhysiologicalState) -> np.ndarray:
        """Convert state to vector for KDTree"""
        return np.array(
            [
                state.po2,
                state.temperature,
                state.altitude,
                state.heart_mass,
                state.hematocrit,
            ]
        )

    # Extend with PCA/UMAP and reachability calculations as needed
