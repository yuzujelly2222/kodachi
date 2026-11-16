"""Approximate nearest neighbor search with random projection trees.

This module provides an Annoy-style index that builds a forest of random
hyperplane trees and searches it for the vectors with the highest cosine
similarity to a query.

Example:
    >>> import numpy as np
    >>> data = np.random.default_rng(0).normal(size=(1000, 16))
    >>> index = Index(dim=16, n_trees=10, seed=42)
    >>> index.build_index(data)
    >>> index.search(data[0], k=5)
    array([...])
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import heapq
import itertools


@dataclass(slots=True)
class _Node:
    """A node of a random projection tree.

    An internal node splits space by the hyperplane ``normal · x + offset = 0``.
    A leaf node stores the indices of the data points that fall into it.

    Attributes:
        normal: Unit normal vector of the splitting hyperplane. ``None`` for leaves.
        offset: Bias term of the splitting hyperplane.
        left: Child containing points with ``normal · x + offset < 0``.
        right: Child containing points with ``normal · x + offset > 0``.
        indices: Indices of the data points in this leaf. ``None`` for internal nodes.
    """

    normal: np.ndarray | None = None
    offset: float = 0.0
    left: _Node | None = None
    right: _Node | None = None
    indices: np.ndarray | None = None

    @property
    def is_leaf(self) -> bool:
        """bool: Whether this node is a leaf."""
        return self.indices is not None


class _Helper:
    """Stateless numerical helpers used by :class:`Index`."""

    @staticmethod
    def make_offset_and_normal(point_1: np.ndarray, point_2: np.ndarray) -> tuple[np.ndarray, float]:
        """Computes the hyperplane that equidistantly separates two points.

        Args:
            point_1: First point, shape ``(dim,)``.
            point_2: Second point, shape ``(dim,)``.

        Returns:
            A tuple ``(normal, offset)`` where ``normal`` is the unit vector
            pointing from ``point_2`` to ``point_1`` and ``offset`` places the
            hyperplane at their midpoint. If the points are identical,
            ``normal`` is the zero vector.
        """
        normal = point_1 - point_2
        norm = np.linalg.norm(normal)
        if norm > 0:
            normal = normal / norm
        offset = float(np.dot(-normal, (point_1 + point_2) / 2))
        return normal, offset

    @staticmethod
    def normalize_rows(a: np.ndarray) -> np.ndarray:
        """L2-normalizes each row of a 2-D array.

        Rows with zero norm are left unchanged.

        Args:
            a: Array of shape ``(n, dim)``.

        Returns:
            A new array of shape ``(n, dim)`` whose non-zero rows have unit length.
        """
        norms = np.linalg.norm(a, axis=1, keepdims=True)
        norms[norms == 0] = 1
        return a / norms


class Index:
    """Approximate nearest neighbor index based on cosine similarity.

    The index builds ``n_trees`` random projection trees over L2-normalized
    data. Searching descends every tree toward the query and then explores
    the most promising unvisited branches until enough candidates are
    collected, finally ranking the candidates by exact cosine similarity.

    Attributes:
        dim: Dimensionality of the vectors.
        n_trees: Number of trees in the forest.
        max_leaf_size: Maximum number of points stored in a single leaf.
    """

    def __init__(self, dim: int, n_trees: int = 10, max_leaf_size: int = 32, seed: int | None = None):
        """Initializes an empty index.

        Args:
            dim: Dimensionality of the vectors to be indexed.
            n_trees: Number of trees to build. More trees improve recall at
                the cost of memory and build time.
            max_leaf_size: Maximum number of points in a leaf node.
            seed: Seed for the random number generator. Use a fixed value
                for reproducible trees.
        """
        self.dim = dim
        self.n_trees = n_trees
        self.max_leaf_size = max_leaf_size
        self._rng = np.random.default_rng(seed)
        self._roots: list[_Node] = []
        self._data: np.ndarray | None = None

    def build_index(self, data: np.ndarray) -> None:
        """Builds the forest from the given data.

        Any previously built trees are discarded. The data is copied, cast to
        ``float32`` and L2-normalized before indexing.

        Args:
            data: Array-like of shape ``(n_points, dim)``.

        Raises:
            ValueError: If ``data`` is not 2-D or its second dimension does
                not equal ``dim``.
        """
        data = np.asarray(data, dtype=np.float32)
        if data.ndim != 2 or data.shape[1] != self.dim:
            raise ValueError(f"Expected data of shape (n_points, {self.dim}), but got {data.shape}.")
        self._data = _Helper.normalize_rows(data)
        self._roots = []
        all_idx = np.arange(data.shape[0])
        for _ in range(self.n_trees):
            self._roots.append(self._make_tree(all_idx))

    def _make_tree(self, idx: np.ndarray) -> _Node:
        """Recursively builds a tree over a subset of the data.

        Args:
            idx: Indices of the data points to place in this subtree.

        Returns:
            The root node of the built subtree.
        """
        node = _Node()
        if len(idx) <= self.max_leaf_size:
            node.indices = idx
            return node

        point_1, point_2 = self._rng.choice(idx, size=2, replace=False)
        normal, offset = _Helper.make_offset_and_normal(self._data[point_1], self._data[point_2])
        node.normal = normal
        node.offset = offset
        s = self._data[idx] @ normal + offset
        left_mask = 0 > s
        on_mask = 0 == s
        right_mask = 0 < s

        on = self._rng.permutation(idx[on_mask])
        half = len(on) // 2
        left = np.concatenate((idx[left_mask], on[half:]))
        right = np.concatenate((idx[right_mask], on[:half]))

        node.left = self._make_tree(left)
        node.right = self._make_tree(right)
        return node

    def search(self, query: np.ndarray, k: int = 10, search_k: int | None = None) -> np.ndarray:
        """Finds the approximate ``k`` nearest neighbors of a query.

        Args:
            query: Query vector of shape ``(dim,)``.
            k: Number of neighbors to return.
            search_k: Number of candidate points to collect before ranking.
                Larger values improve recall at the cost of speed. Defaults
                to ``k * n_trees * 10``. If it is at least the number of
                indexed points, an exact brute-force search is performed
                instead of traversing the trees.

        Returns:
            Array of up to ``k`` data indices, sorted by descending cosine
            similarity to ``query``.

        Raises:
            RuntimeError: If :meth:`build_index` has not been called.
            ValueError: If ``query`` does not have shape ``(dim,)``.
        """
        if self._data is None:
            raise RuntimeError("Index has not been built. Call build_index() first.")
        query = np.asarray(query, dtype=np.float32)
        if query.shape != (self.dim,):
            raise ValueError(f"Expected query of shape ({self.dim},), but got {query.shape}.")
        query = _Helper.normalize_rows(query[None, :])[0]
        if search_k is None:
            search_k = k * self.n_trees * 10

        if search_k >= len(self._data):
            cand = np.arange(len(self._data))
        else:
            cand = self._collect_candidates(query, search_k)

        scores = self._data[cand] @ query
        order = np.argsort(-scores)[:k]
        return cand[order]

    def _collect_candidates(self, query: np.ndarray, search_k: int) -> np.ndarray:
        """Collects candidate points by best-first traversal of the forest.

        Args:
            query: Normalized query vector of shape ``(dim,)``.
            search_k: Number of candidates to collect before stopping.

        Returns:
            Array of unique data indices of the collected candidates.
        """
        heap: list[tuple[float, int, _Node]] = []
        counter = itertools.count()
        candidates: set[int] = set()

        for node in self._roots:
            leaf = self.down_node(heap, counter, node, query, np.inf)
            candidates.update(leaf.indices.tolist())

        while heap and len(candidates) < search_k:
            neg_priority, _, node = heapq.heappop(heap)
            leaf = self.down_node(heap, counter, node, query, -neg_priority)
            candidates.update(leaf.indices.tolist())

        return np.fromiter(candidates, dtype=np.int64)

    def down_node(self, heap: list[tuple[float, int, _Node]], counter: itertools.count,
                  node: _Node, query: np.ndarray, space: float) -> _Node:
        """Descends from a node to a leaf, deferring the other branches.

        At each internal node the query follows the side of the hyperplane it
        lies on, and the opposite branch is pushed onto ``heap`` with a
        priority based on the query's margin to the hyperplane.

        Args:
            heap: Priority queue of deferred branches, stored as
                ``(negated_priority, tie_breaker, node)`` tuples. Modified in place.
            counter: Monotonic counter used as a tie-breaker in ``heap``.
            node: Node to start descending from.
            query: Normalized query vector of shape ``(dim,)``.
            space: Smallest margin accumulated along the path to ``node``.

        Returns:
            The leaf node reached by the descent.
        """
        while not node.is_leaf:
            s = float(query @ node.normal + node.offset)
            if s <= 0:
                near, far = node.left, node.right
            else:
                near, far = node.right, node.left
            heapq.heappush(heap, (-min(space, -abs(s)), next(counter), far))
            space = min(space, abs(s))
            node = near
        return node


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n, dim, k = 20000, 32, 10
    data = rng.normal(size=(n, dim))
    queries = rng.normal(size=(100, dim))

    index = Index(dim=dim, n_trees=10, seed=42)
    index.build_index(data)

    data_n = _Helper.normalize_rows(data)
    queries_n = _Helper.normalize_rows(queries)
    for search_k in (None, 1000, 5000):
        hits = 0
        for q, qn in zip(queries, queries_n):
            truth = set(np.argsort(-(data_n @ qn))[:k].tolist())
            got = set(index.search(q, k=k, search_k=search_k).tolist())
            hits += len(truth & got)
        print(f"search_k={search_k}: recall@{k} = {hits / (len(queries) * k):.3f}")
