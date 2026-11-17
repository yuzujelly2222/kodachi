"""Implementation of :class:`kodachi.Index`.

This module is internal. Import the public API from :mod:`kodachi` instead.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Literal, overload
import numpy as np
import heapq
import itertools
import json
import os
import zipfile

FORMAT_VERSION = 1
"""int: Version of the on-disk format written by :meth:`Index.save`."""


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

    @staticmethod
    def write_npy(zf: zipfile.ZipFile, name: str, arr: np.ndarray) -> None:
        """Writes an array into a zip archive as a ``.npy`` member.

        Args:
            zf: Zip archive opened for writing.
            name: Member name inside the archive.
            arr: Array to write. Object arrays are not allowed.
        """
        with zf.open(name, "w", force_zip64=True) as f:
            np.save(f, arr, allow_pickle=False)

    @staticmethod
    def read_npy(zf: zipfile.ZipFile, name: str) -> np.ndarray:
        """Reads a ``.npy`` member from a zip archive.

        Args:
            zf: Zip archive opened for reading.
            name: Member name inside the archive.

        Returns:
            The stored array.
        """
        with zf.open(name) as f:
            return np.load(f, allow_pickle=False)

    @staticmethod
    def check_positive_int(name: str, value: object) -> int:
        """Validates that a value is an integer of at least 1.

        Args:
            name: Parameter name used in error messages.
            value: Value to validate. Python and NumPy integers are accepted;
                ``bool`` is rejected.

        Returns:
            ``value`` as a Python ``int``.

        Raises:
            TypeError: If ``value`` is not an integer.
            ValueError: If ``value`` is less than 1.
        """
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise TypeError(f"{name} must be an int, but got {type(value).__name__}.")
        if value < 1:
            raise ValueError(f"{name} must be at least 1, but got {value}.")
        return int(value)


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

        Raises:
            TypeError: If ``dim``, ``n_trees`` or ``max_leaf_size`` is not an int.
            ValueError: If ``dim``, ``n_trees`` or ``max_leaf_size`` is less than 1.
        """
        self.dim = _Helper.check_positive_int("dim", dim)
        self.n_trees = _Helper.check_positive_int("n_trees", n_trees)
        self.max_leaf_size = _Helper.check_positive_int("max_leaf_size", max_leaf_size)
        self._rng = np.random.default_rng(seed)
        self._roots: list[_Node] = []
        self._data: np.ndarray | None = None

    def __len__(self) -> int:
        """Returns the number of indexed points, or 0 if the index is not built."""
        return 0 if self._data is None else len(self._data)

    def __repr__(self) -> str:
        """Returns a summary such as ``Index(dim=64, n_trees=20, max_leaf_size=32, n_points=10000)``."""
        n_points = "not built" if self._data is None else len(self._data)
        return (f"Index(dim={self.dim}, n_trees={self.n_trees}, "
                f"max_leaf_size={self.max_leaf_size}, n_points={n_points})")

    def build_index(self, data: np.ndarray) -> None:
        """Builds the forest from the given data.

        Any previously built trees are discarded. The data is copied, cast to
        ``float32`` and L2-normalized before indexing.

        Args:
            data: Array-like of shape ``(n_points, dim)``.

        Raises:
            ValueError: If ``data`` is not 2-D, its second dimension does not
                equal ``dim``, it has no rows, or it contains NaN or infinity.
        """
        data = np.asarray(data, dtype=np.float32)
        if data.ndim != 2 or data.shape[1] != self.dim:
            raise ValueError(f"Expected data of shape (n_points, {self.dim}), but got {data.shape}.")
        if data.shape[0] == 0:
            raise ValueError("data must contain at least one point.")
        if not np.isfinite(data).all():
            raise ValueError("data must not contain NaN or infinity.")
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

    @overload
    def search(self, query: np.ndarray, k: int = ..., search_k: int | None = ...,
               return_scores: Literal[False] = ...) -> np.ndarray: ...

    @overload
    def search(self, query: np.ndarray, k: int = ..., search_k: int | None = ...,
               *, return_scores: Literal[True]) -> tuple[np.ndarray, np.ndarray]: ...

    def search(self, query: np.ndarray, k: int = 10, search_k: int | None = None,
               return_scores: bool = False) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """Finds the approximate ``k`` nearest neighbors of a query.

        Args:
            query: Query vector of shape ``(dim,)``.
            k: Number of neighbors to return.
            search_k: Number of candidate points to collect before ranking.
                Larger values improve recall at the cost of speed. Defaults
                to ``k * n_trees * 10``. If it is at least the number of
                indexed points, an exact brute-force search is performed
                instead of traversing the trees.
            return_scores: If ``True``, also return the cosine similarity of
                each neighbor.

        Returns:
            Array of up to ``k`` data indices (``int64``), sorted by
            descending cosine similarity to ``query``. If ``return_scores``
            is ``True``, a tuple ``(indices, scores)`` where ``scores`` holds
            the matching cosine similarities (``float32``, from -1 to 1).

        Raises:
            RuntimeError: If :meth:`build_index` has not been called.
            TypeError: If ``k`` or ``search_k`` is not an int.
            ValueError: If ``query`` does not have shape ``(dim,)`` or contains
                NaN or infinity, or if ``k`` or ``search_k`` is less than 1.
        """
        if self._data is None:
            raise RuntimeError("Index has not been built. Call build_index() first.")
        k = _Helper.check_positive_int("k", k)
        if search_k is not None:
            search_k = _Helper.check_positive_int("search_k", search_k)
        query = np.asarray(query, dtype=np.float32)
        if query.shape != (self.dim,):
            raise ValueError(f"Expected query of shape ({self.dim},), but got {query.shape}.")
        if not np.isfinite(query).all():
            raise ValueError("query must not contain NaN or infinity.")
        query = _Helper.normalize_rows(query[None, :])[0]
        if search_k is None:
            search_k = k * self.n_trees * 10

        if search_k >= len(self._data):
            cand = np.arange(len(self._data), dtype=np.int64)
        else:
            cand = self._collect_candidates(query, search_k)

        scores = self._data[cand] @ query
        order = np.argsort(-scores, kind="stable")[:k]
        if return_scores:
            return cand[order], scores[order]
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
            leaf = self._down_node(heap, counter, node, query, np.inf)
            candidates.update(leaf.indices.tolist())

        while heap and len(candidates) < search_k:
            neg_priority, _, node = heapq.heappop(heap)
            leaf = self._down_node(heap, counter, node, query, -neg_priority)
            candidates.update(leaf.indices.tolist())

        return np.fromiter(candidates, dtype=np.int64)

    def _down_node(self, heap: list[tuple[float, int, _Node]], counter: itertools.count,
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

    def save(self, path: str | os.PathLike) -> None:
        """Saves the index to a file.

        The file is an uncompressed zip archive containing ``settings.json``,
        the normalized data as ``data.npy`` and each tree flattened into
        ``.npy`` arrays under ``tree_<t>/``. The conventional extension is
        ``.kodachi``. An existing file at ``path`` is overwritten.

        Args:
            path: Destination file path, e.g. ``"index.kodachi"``.

        Raises:
            RuntimeError: If :meth:`build_index` has not been called.
        """
        if self._data is None:
            raise RuntimeError("Index has not been built. Call build_index() first.")
        settings = {
            "format_version": FORMAT_VERSION,
            "dim": self.dim,
            "n_trees": len(self._roots),
            "max_leaf_size": self.max_leaf_size,
        }
        with zipfile.ZipFile(path, "w", zipfile.ZIP_STORED) as zf:
            zf.writestr("settings.json", json.dumps(settings, indent=2))
            _Helper.write_npy(zf, "data.npy", self._data)
            for t, root in enumerate(self._roots):
                for name, arr in self._flatten_tree(root).items():
                    _Helper.write_npy(zf, f"tree_{t}/{name}.npy", arr)

    @classmethod
    def load(cls, path: str | os.PathLike, seed: int | None = None) -> Index:
        """Loads an index saved with :meth:`save`.

        The loaded index returns the same search results as the saved one.

        Args:
            path: Path of a file written by :meth:`save`.
            seed: Seed for the random number generator, used only if
                :meth:`build_index` is called again on the loaded index.

        Returns:
            The restored index, ready for :meth:`search`.

        Raises:
            FileNotFoundError: If ``path`` does not exist.
            ValueError: If the file is not a valid kodachi index or was
                written in an unsupported format version.
        """
        try:
            with zipfile.ZipFile(path, "r") as zf:
                settings = json.loads(zf.read("settings.json"))
                version = settings.get("format_version")
                if version != FORMAT_VERSION:
                    raise ValueError(
                        f"Unsupported format version {version!r}; "
                        f"this version of kodachi supports {FORMAT_VERSION}."
                    )
                index = cls(dim=settings["dim"], n_trees=settings["n_trees"],
                            max_leaf_size=settings["max_leaf_size"], seed=seed)
                data = _Helper.read_npy(zf, "data.npy")
                if data.ndim != 2 or data.shape[1] != index.dim:
                    raise ValueError(
                        f"Stored data has shape {data.shape}, which does not match dim={index.dim}."
                    )
                index._data = data
                names = ("normals", "offsets", "left", "right",
                         "leaf_start", "leaf_end", "leaf_indices")
                for t in range(settings["n_trees"]):
                    arrays = {name: _Helper.read_npy(zf, f"tree_{t}/{name}.npy") for name in names}
                    index._roots.append(cls._unflatten_tree(arrays))
        except (zipfile.BadZipFile, KeyError, TypeError, json.JSONDecodeError) as e:
            raise ValueError(f"{os.fspath(path)!r} is not a valid kodachi index file.") from e
        return index

    @staticmethod
    def _flatten_tree(root: _Node) -> dict[str, np.ndarray]:
        """Converts a tree into flat arrays for serialization.

        Nodes are numbered in depth-first order with the root as node 0.

        Args:
            root: Root node of the tree.

        Returns:
            A dict of arrays, each indexed by node number:

            - ``normals``: ``(n_nodes, dim)`` hyperplane normals (zeros for leaves).
            - ``offsets``: ``(n_nodes,)`` hyperplane offsets.
            - ``left`` / ``right``: child node numbers, ``-1`` for leaves.
            - ``leaf_start`` / ``leaf_end``: slice of ``leaf_indices`` owned
              by each leaf, ``-1`` for internal nodes.
            - ``leaf_indices``: concatenated data indices of all leaves.
        """
        nodes: list[_Node] = []
        ids: dict[int, int] = {}
        stack = [root]
        while stack:
            node = stack.pop()
            ids[id(node)] = len(nodes)
            nodes.append(node)
            if not node.is_leaf:
                stack.append(node.right)
                stack.append(node.left)

        n = len(nodes)
        dim = next((nd.normal.shape[0] for nd in nodes if not nd.is_leaf), 0)
        normals = np.zeros((n, dim), dtype=np.float32)
        offsets = np.zeros(n, dtype=np.float32)
        left = np.full(n, -1, dtype=np.int32)
        right = np.full(n, -1, dtype=np.int32)
        leaf_start = np.full(n, -1, dtype=np.int64)
        leaf_end = np.full(n, -1, dtype=np.int64)
        leaf_parts: list[np.ndarray] = []
        pos = 0
        for i, node in enumerate(nodes):
            if node.is_leaf:
                leaf_start[i] = pos
                pos += len(node.indices)
                leaf_end[i] = pos
                leaf_parts.append(node.indices)
            else:
                normals[i] = node.normal
                offsets[i] = node.offset
                left[i] = ids[id(node.left)]
                right[i] = ids[id(node.right)]
        leaf_indices = (np.concatenate(leaf_parts) if leaf_parts
                        else np.empty(0)).astype(np.int64)
        return {"normals": normals, "offsets": offsets, "left": left, "right": right,
                "leaf_start": leaf_start, "leaf_end": leaf_end, "leaf_indices": leaf_indices}

    @staticmethod
    def _unflatten_tree(a: dict[str, np.ndarray]) -> _Node:
        """Rebuilds a tree from the arrays produced by :meth:`_flatten_tree`.

        Args:
            a: Dict of flat arrays as returned by :meth:`_flatten_tree`.

        Returns:
            The root node of the rebuilt tree.
        """
        n = len(a["left"])
        nodes = [_Node() for _ in range(n)]
        for i, node in enumerate(nodes):
            if a["left"][i] == -1:
                node.indices = a["leaf_indices"][a["leaf_start"][i]:a["leaf_end"][i]]
            else:
                node.normal = a["normals"][i]
                node.offset = float(a["offsets"][i])
                node.left = nodes[a["left"][i]]
                node.right = nodes[a["right"][i]]
        return nodes[0]


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
