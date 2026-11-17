import json
import zipfile

import numpy as np
import pytest

import kodachi
from kodachi import FORMAT_VERSION, Index
from kodachi._index import _Helper


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

    @pytest.mark.parametrize("name", ["dim", "n_trees", "max_leaf_size"])
    @pytest.mark.parametrize("value", [0, -1])
    def test_init_rejects_non_positive(self, name, value):
        kwargs = {"dim": DIM, name: value}
        with pytest.raises(ValueError, match=f"{name} must be at least 1"):
            Index(**kwargs)

    @pytest.mark.parametrize("name", ["dim", "n_trees", "max_leaf_size"])
    @pytest.mark.parametrize("value", [1.5, "3", True, None])
    def test_init_rejects_non_int(self, name, value):
        kwargs = {"dim": DIM, name: value}
        with pytest.raises(TypeError, match=f"{name} must be an int"):
            Index(**kwargs)

    def test_init_accepts_numpy_int(self):
        idx = Index(dim=np.int64(DIM), n_trees=np.int32(3))
        assert idx.dim == DIM and type(idx.dim) is int
        assert idx.n_trees == 3

    def test_build_with_empty_data_raises(self):
        with pytest.raises(ValueError, match="at least one point"):
            Index(dim=DIM).build_index(np.zeros((0, DIM)))

    @pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
    def test_build_with_non_finite_raises(self, data, bad):
        data = data.copy()
        data[3, 2] = bad
        with pytest.raises(ValueError, match="NaN or infinity"):
            Index(dim=DIM).build_index(data)

    @pytest.mark.parametrize("bad", [np.nan, np.inf])
    def test_search_with_non_finite_query_raises(self, index, bad):
        query = np.ones(DIM)
        query[0] = bad
        with pytest.raises(ValueError, match="NaN or infinity"):
            index.search(query)

    @pytest.mark.parametrize("name", ["k", "search_k"])
    @pytest.mark.parametrize("value", [0, -5])
    def test_search_rejects_non_positive(self, index, name, value):
        with pytest.raises(ValueError, match=f"{name} must be at least 1"):
            index.search(np.ones(DIM), **{name: value})

    @pytest.mark.parametrize("name", ["k", "search_k"])
    def test_search_rejects_non_int(self, index, name):
        with pytest.raises(TypeError, match=f"{name} must be an int"):
            index.search(np.ones(DIM), **{name: 2.0})


class TestPublicApi:
    def test_exports(self):
        assert kodachi.__all__ == ["FORMAT_VERSION", "Index"]
        assert kodachi.__version__ == "1.0.0"

    def test_down_node_is_private(self):
        assert not hasattr(Index, "down_node")

    def test_len(self, index, data):
        assert len(Index(dim=DIM)) == 0
        assert len(index) == len(data)

    def test_repr(self, index, data):
        assert repr(Index(dim=DIM, n_trees=3, max_leaf_size=8)) == (
            f"Index(dim={DIM}, n_trees=3, max_leaf_size=8, n_points=not built)"
        )
        assert repr(index) == (
            f"Index(dim={DIM}, n_trees=10, max_leaf_size=16, n_points={len(data)})"
        )


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
    def test_return_scores(self, index, data):
        query = np.random.default_rng(6).normal(size=DIM)
        ids, scores = index.search(query, k=10, return_scores=True)
        np.testing.assert_array_equal(ids, index.search(query, k=10))
        assert ids.dtype == np.int64
        assert scores.dtype == np.float32
        assert scores.shape == (10,)
        q = query / np.linalg.norm(query)
        np.testing.assert_allclose(scores, index._data[ids] @ q, rtol=1e-5, atol=1e-6)
        assert np.all(np.diff(scores) <= 0)
        assert np.all((-1.0001 <= scores) & (scores <= 1.0001))

    def test_return_scores_self_is_one(self, index, data):
        ids, scores = index.search(data[7], k=1, return_scores=True)
        assert ids[0] == 7
        assert scores[0] == pytest.approx(1.0, abs=1e-5)

    def test_indices_dtype_is_int64_for_brute_force(self, index, data):
        assert index.search(data[0], k=5, search_k=len(data)).dtype == np.int64

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


class TestSaveLoad:
    def test_roundtrip_gives_same_results(self, index, data, tmp_path):
        path = tmp_path / "index.kodachi"
        index.save(path)
        loaded = Index.load(path)

        assert loaded.dim == index.dim
        assert loaded.n_trees == index.n_trees
        assert loaded.max_leaf_size == index.max_leaf_size
        np.testing.assert_array_equal(loaded._data, index._data)

        rng = np.random.default_rng(5)
        for _ in range(20):
            query = rng.normal(size=DIM)
            for search_k in (None, 50, 500):
                np.testing.assert_array_equal(
                    loaded.search(query, k=10, search_k=search_k),
                    index.search(query, k=10, search_k=search_k),
                )

    def test_roundtrip_preserves_tree_structure(self, index, tmp_path):
        path = tmp_path / "index.kodachi"
        index.save(path)
        loaded = Index.load(path)
        for a, b in zip(index._roots, loaded._roots):
            leaves_a = [leaf.indices.tolist() for leaf in collect_leaves(a, [])]
            leaves_b = [leaf.indices.tolist() for leaf in collect_leaves(b, [])]
            assert leaves_a == leaves_b

    def test_accepts_str_path(self, index, tmp_path):
        path = str(tmp_path / "index.kodachi")
        index.save(path)
        assert Index.load(path).search(np.ones(DIM), k=3).shape == (3,)

    def test_file_is_uncompressed_zip(self, index, tmp_path):
        path = tmp_path / "index.kodachi"
        index.save(path)
        with zipfile.ZipFile(path) as zf:
            names = set(zf.namelist())
            settings = json.loads(zf.read("settings.json"))
            assert all(info.compress_type == zipfile.ZIP_STORED for info in zf.infolist())
        assert {"settings.json", "data.npy", "tree_0/normals.npy"} <= names
        assert settings == {
            "format_version": FORMAT_VERSION,
            "dim": DIM,
            "n_trees": index.n_trees,
            "max_leaf_size": index.max_leaf_size,
        }

    def test_single_leaf_trees(self, tmp_path):
        data = np.random.default_rng(0).normal(size=(5, DIM))
        idx = Index(dim=DIM, n_trees=3, seed=0)
        idx.build_index(data)
        path = tmp_path / "small.kodachi"
        idx.save(path)
        loaded = Index.load(path)
        np.testing.assert_array_equal(loaded.search(data[0], k=5), idx.search(data[0], k=5))

    def test_loaded_index_can_be_rebuilt(self, index, data, tmp_path):
        path = tmp_path / "index.kodachi"
        index.save(path)
        loaded = Index.load(path, seed=1)
        loaded.build_index(data[:100])
        assert len(loaded._data) == 100

    def test_save_overwrites(self, index, data, tmp_path):
        path = tmp_path / "index.kodachi"
        index.save(path)
        small = Index(dim=DIM, n_trees=2, seed=0)
        small.build_index(data[:50])
        small.save(path)
        loaded = Index.load(path)
        assert loaded.n_trees == 2
        assert len(loaded._data) == 50

    def test_save_before_build_raises(self, tmp_path):
        with pytest.raises(RuntimeError, match="build_index"):
            Index(dim=DIM).save(tmp_path / "index.kodachi")

    def test_load_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            Index.load(tmp_path / "missing.kodachi")

    def test_load_non_zip_raises(self, tmp_path):
        path = tmp_path / "broken.kodachi"
        path.write_bytes(b"not a zip file")
        with pytest.raises(ValueError, match="not a valid kodachi index"):
            Index.load(path)

    def test_load_zip_without_members_raises(self, tmp_path):
        path = tmp_path / "empty.kodachi"
        with zipfile.ZipFile(path, "w") as zf:
            zf.writestr("hello.txt", "hi")
        with pytest.raises(ValueError, match="not a valid kodachi index"):
            Index.load(path)

    def test_load_unsupported_version_raises(self, index, tmp_path):
        path = tmp_path / "index.kodachi"
        index.save(path)
        patched = tmp_path / "future.kodachi"
        with zipfile.ZipFile(path) as src, zipfile.ZipFile(patched, "w") as dst:
            for info in src.infolist():
                content = src.read(info.filename)
                if info.filename == "settings.json":
                    settings = json.loads(content)
                    settings["format_version"] = FORMAT_VERSION + 1
                    content = json.dumps(settings)
                dst.writestr(info, content)
        with pytest.raises(ValueError, match="Unsupported format version"):
            Index.load(patched)
