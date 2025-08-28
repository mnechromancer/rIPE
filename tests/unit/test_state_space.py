"""
Tests for State Space Management System

Ensures spatial indexing, neighbor search, and dimensionality reduction
work as expected.
"""

import numpy as np
from ipe.core.physiology.state import PhysiologicalState
from ipe.core.state.space import StateSpace
from ipe.core.state.indexing import pca_reduce


class TestStateSpace:
    def test_add_and_find_neighbors(self):
        space = StateSpace(dimensions={})
        s1 = PhysiologicalState(po2=15.0, temperature=25.0, altitude=1000.0)
        s2 = PhysiologicalState(po2=16.0, temperature=25.0, altitude=1000.0)
        space.add_state(s1)
        space.add_state(s2)
        neighbors = space.find_neighbors(s1, radius=2.0)
        assert s1 in neighbors
        assert s2 in neighbors

    def test_pca_reduce(self):
        arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        reduced = pca_reduce(arr, n_components=2)
        assert reduced.shape[1] == 2

    # TODO: Add tests for reachability and caching
