"""
Spatial Indexing Utilities for State Space

Provides helper functions for dimensionality reduction and caching.
"""

import numpy as np
from sklearn.decomposition import PCA

# UMAP can be added if available


def pca_reduce(states: np.ndarray, n_components: int = 2) -> np.ndarray:
    """Reduce dimensionality of state vectors using PCA"""
    pca = PCA(n_components=n_components)
    return pca.fit_transform(states)


# Extend with UMAP support and caching utilities as needed
