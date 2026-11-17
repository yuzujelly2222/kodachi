"""Approximate nearest neighbor search with random projection trees.

kodachi provides an Annoy-style index that builds a forest of random
hyperplane trees and searches it for the vectors with the highest cosine
similarity to a query.

Example:
    >>> import numpy as np
    >>> from kodachi import Index
    >>> data = np.random.default_rng(0).normal(size=(1000, 16))
    >>> index = Index(dim=16, n_trees=10, seed=42)
    >>> index.build_index(data)
    >>> index.search(data[0], k=5)
    array([...])
"""

from ._index import FORMAT_VERSION, Index

__version__ = "1.0.0"
__all__ = ["FORMAT_VERSION", "Index"]
