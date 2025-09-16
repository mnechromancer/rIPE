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

    def test_caching_behavior(self):
        """Test that StateSpace properly manages its cache"""
        space = StateSpace(dimensions={})
        s1 = PhysiologicalState(po2=15.0, temperature=25.0, altitude=1000.0)
        s2 = PhysiologicalState(po2=16.0, temperature=25.0, altitude=1000.0)

        # Add states and verify index is rebuilt
        space.add_state(s1)
        assert space.index is None  # Index should be None until built

        # First neighbor search should build index
        space.find_neighbors(s1, radius=2.0)
        assert space.index is not None

        # Adding new state should invalidate index
        space.add_state(s2)
        assert space.index is None

    def test_reachability_analysis(self):
        """Test state reachability within the space"""
        space = StateSpace(dimensions={})

        # Create states at different distances
        center = PhysiologicalState(po2=15.0, temperature=25.0, altitude=1000.0)
        close = PhysiologicalState(po2=15.5, temperature=25.0, altitude=1000.0)
        far = PhysiologicalState(po2=20.0, temperature=30.0, altitude=2000.0)

        space.add_state(center)
        space.add_state(close)
        space.add_state(far)

        # Test different radius values for reachability
        # Distance from center to close is ~0.5, center to far is ~1000
        small_radius_neighbors = space.find_neighbors(center, radius=1.0)
        large_radius_neighbors = space.find_neighbors(center, radius=1500.0)

        # Close state should be reachable with small radius
        assert center in small_radius_neighbors
        assert close in small_radius_neighbors
        assert far not in small_radius_neighbors

        # All states should be reachable with large radius
        assert len(large_radius_neighbors) == 3
        assert center in large_radius_neighbors
        assert close in large_radius_neighbors
        assert far in large_radius_neighbors
