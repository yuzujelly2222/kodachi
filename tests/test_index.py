import numpy as np
import pytest

from kodachi import Index
from kodachi.main import _Helper


DIM = 16


def brute_force(data: np.ndarray, query: np.ndarray, k: int) -> np.ndarray:
    data_n = _Helper.normalize_rows(np.asarray(data, dtype=np.float32))
    query_n = _Helper.normalize_rows(np.asarray(query, dtype=np.float32)[None, :])[0]
    return np.argsort(-(data_n @ query_n))[:k]


@pytest.fixture
def data() -> np.ndarray:
    return np.random.default_rng(0).normal(size=(2000, DIM))


@pytest.fixture
def index(data: np.ndarray) -> Index:
    idx = Index(dim=DIM, n_trees=10, max_leaf_size=16, seed=42)
    idx.build_index(data)
    return idx


def collect_leaves(node, leaves):
    if node.is_leaf:
        leaves.append(node)
    else:
        collect_leaves(node.left, leaves)
        collect_leaves(node.right, leaves)
    return leaves


class TestValidation:
    def test_search_before_build_raises(self):
        with pytest.raises(RuntimeError, match="build_index"):
            Index(dim=DIM).search(np.zeros(DIM))

    @pytest.mark.parametrize("shape", [(10,), (10, DIM + 1), (2, 5, DIM)])
    def test_build_with_wrong_shape_raises(self, shape):
        with pytest.raises(ValueError, match="Expected data of shape"):
            Index(dim=DIM).build_index(np.zeros(shape))

    @pytest.mark.parametrize("shape", [(DIM + 1,), (1, DIM)])
    def test_search_with_wrong_shape_raises(self, index, shape):
        with pytest.raises(ValueError, match="Expected query of shape"):
            index.search(np.zeros(shape))


class TestTreeStructure:
    def test_each_tree_partitions_all_points(self, index, data):
        for root in index._roots:
            leaves = collect_leaves(root, [])
            all_idx = np.concatenate([leaf.indices for leaf in leaves])
            assert len(all_idx) == len(data)
            assert set(all_idx.tolist()) == set(range(len(data)))

    def test_leaf_size_is_bounded(self, index):
        for root in index._roots:
            for leaf in collect_leaves(root, []):
                assert len(leaf.indices) <= index.max_leaf_size

    def test_number_of_trees(self, index):
        assert len(index._roots) == index.n_trees

    def test_rebuild_replaces_trees(self, index, data):
        index.build_index(data[:100])
        assert len(index._roots) == index.n_trees
        assert len(index._data) == 100

    def test_duplicate_points_terminate(self):
        data = np.ones((200, DIM))
        idx = Index(dim=DIM, n_trees=3, max_leaf_size=8, seed=0)
        idx.build_index(data)
        assert len(idx.search(np.ones(DIM), k=5)) == 5


class TestSearch:
    def test_returns_k_unique_indices(self, index, data):
        result = index.search(data[0], k=10)
        assert result.shape == (10,)
        assert len(set(result.tolist())) == 10
        assert np.all((0 <= result) & (result < len(data)))

    def test_results_sorted_by_similarity(self, index, data):
        query = np.random.default_rng(1).normal(size=DIM)
        result = index.search(query, k=20)
        q = query / np.linalg.norm(query)
        scores = index._data[result] @ q
        assert np.all(np.diff(scores) <= 1e-6)

    def test_finds_itself(self, index, data):
        for i in range(0, len(data), 200):
            assert index.search(data[i], k=1)[0] == i

    def test_large_search_k_is_exact(self, index, data):
        rng = np.random.default_rng(2)
        for _ in range(10):
            query = rng.normal(size=DIM)
            got = index.search(query, k=10, search_k=len(data))
            np.testing.assert_array_equal(got, brute_force(data, query, 10))

    def test_k_larger_than_data(self):
        data = np.random.default_rng(0).normal(size=(5, DIM))
        idx = Index(dim=DIM, seed=0)
        idx.build_index(data)
        result = idx.search(data[0], k=10)
        assert sorted(result.tolist()) == list(range(5))

    def test_scale_invariant(self, index, data):
        np.testing.assert_array_equal(
            index.search(data[3], k=10), index.search(data[3] * 100, k=10)
        )

    def test_zero_query_does_not_crash(self, index):
        assert len(index.search(np.zeros(DIM), k=5)) == 5

    def test_seed_is_reproducible(self, data):
        query = np.random.default_rng(3).normal(size=DIM)
        results = []
        for _ in range(2):
            idx = Index(dim=DIM, n_trees=5, seed=123)
            idx.build_index(data)
            results.append(idx.search(query, k=10, search_k=200))
        np.testing.assert_array_equal(results[0], results[1])

    def test_recall_improves_with_search_k(self, index, data):
        queries = np.random.default_rng(4).normal(size=(50, DIM))
        k = 10

        def recall(search_k):
            hits = 0
            for q in queries:
                truth = set(brute_force(data, q, k).tolist())
                hits += len(truth & set(index.search(q, k=k, search_k=search_k).tolist()))
            return hits / (len(queries) * k)

        low, high = recall(100), recall(1000)
        assert high >= low
        assert high >= 0.8
