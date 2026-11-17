# User guide

## Parameters

| Parameter | Default | Description |
|---|---|---|
| `dim` | (required) | Dimensionality of the vectors |
| `n_trees` | `10` | Number of trees. More trees improve recall at the cost of memory and build time |
| `max_leaf_size` | `32` | Maximum number of points in a leaf |
| `seed` | `None` | Random seed. Fix it to get the same trees every time |
| `k` | `10` | Number of neighbors to return |
| `search_k` | `k * n_trees * 10` | Number of candidates collected before ranking. Larger values improve recall at the cost of speed. If it is at least the number of indexed points, an exact brute-force search is performed instead of traversing the trees |
| `return_scores` | `False` | If `True`, {meth}`~kodachi.Index.search` returns `(indices, scores)` with the cosine similarity of each neighbor |

Both data and queries are L2-normalized internally, so vector magnitude does
not affect the results.

## Getting similarity scores

Pass `return_scores=True` to get the cosine similarity of each neighbor, for
example to drop results below a threshold:

```python
ids, scores = index.search(query, k=10, return_scores=True)
ids = ids[scores >= 0.8]
```

`scores` is a `float32` array in descending order, with values from -1 to 1.

## Inspecting an index

```python
len(index)   # number of indexed points (0 before build_index)
index        # Index(dim=64, n_trees=20, max_leaf_size=32, n_points=10000)
```

## Input validation

| Situation | Exception |
|---|---|
| `dim`, `n_trees`, `max_leaf_size`, `k` or `search_k` is not an int | {class}`TypeError` |
| any of them is less than 1 | {class}`ValueError` |
| `data` is not `(n_points, dim)`, is empty, or contains NaN / infinity | {class}`ValueError` |
| `query` is not `(dim,)` or contains NaN / infinity | {class}`ValueError` |
| `search` or `save` is called before `build_index` | {class}`RuntimeError` |
| `load` gets a path that does not exist | {class}`FileNotFoundError` |
| `load` gets a file that is not a kodachi index, or an unsupported format version | {class}`ValueError` |

NumPy integers such as `np.int64(10)` are accepted wherever an int is expected.
If `k` exceeds the number of points, all points are returned in similarity order.

## Saving and loading

Save a built index with {meth}`~kodachi.Index.save` and restore it with
{meth}`~kodachi.Index.load` without rebuilding the trees. A loaded index
returns exactly the same results as the saved one.

```python
index.save("index.kodachi")
index = Index.load("index.kodachi")
```

### The `.kodachi` file format

A `.kodachi` file is an uncompressed zip archive:

| File | Contents |
|---|---|
| `settings.json` | `format_version`, `dim`, `n_trees`, `max_leaf_size` |
| `data.npy` | Normalized data, `float32`, shape `(n_points, dim)` |
| `tree_<t>/normals.npy` | Hyperplane normals per node, shape `(n_nodes, dim)` (zeros for leaves) |
| `tree_<t>/offsets.npy` | Hyperplane offsets per node, shape `(n_nodes,)` |
| `tree_<t>/left.npy`, `right.npy` | Child node ids, `-1` for leaves |
| `tree_<t>/leaf_start.npy`, `leaf_end.npy` | Slice of `leaf_indices` owned by each leaf, `-1` for internal nodes |
| `tree_<t>/leaf_indices.npy` | Data indices of all leaves, concatenated |

Nodes are numbered in depth-first order with the root as node 0. Since the
file is plain zip and `.npy`, it can be inspected with {mod}`zipfile` and
{func}`numpy.load` even without kodachi. Arrays are loaded with
`allow_pickle=False`, so loading a file never executes arbitrary code.

{meth}`~kodachi.Index.load` raises {class}`ValueError` when the
`format_version` differs from {data}`kodachi.FORMAT_VERSION`.

## How it works

### Building

1. Pick two random points and split the space with the hyperplane equidistant
   from them.
2. Points lying exactly on the hyperplane are split randomly between the two
   sides.
3. Repeat 1–2 on each side until it holds at most `max_leaf_size` points.

This is done `n_trees` times, producing a forest of trees that each split the
space slightly differently. Data is normalized beforehand, so splitting by
Euclidean distance gives the same ranking as cosine similarity.

### Searching

1. In each tree, descend straight to the leaf on the query's side and collect
   its points. Branches not taken are pushed onto a priority queue.
2. Until `search_k` candidates are collected, explore the deferred branches in
   order of how close the query was to their hyperplane. The closer the query
   is to a hyperplane, the more likely the other side also holds near
   neighbors.
3. Compute the cosine similarity between the query and all candidates at once
   and return the top `k`.

## Complexity and tuning

Let `n` be the number of points, `d` the dimensionality, and `T` the number of
trees.

- **Build:** roughly `O(T · n · d · log n)`
- **Search:** `O(T · d · log n)` to descend the trees, plus `O(search_k · d)`
  to rank candidates
- **Memory:** the data itself (`n · d`), plus per tree about
  `n / max_leaf_size` internal nodes, each holding a `d`-dimensional normal
  vector

Measured on 20,000 random points with 32 dimensions (recall@10):

| n_trees | search_k | recall@10 | ms/query |
|---|---|---|---|
| 10 | 2,000 | 0.76 | 1.7 |
| 20 | 4,000 | 0.95 | 3.5 |
| 50 | 5,000 | 0.98 | 5.0 |
| brute force | — | 1.00 | 0.6 |

At this scale, NumPy brute force is faster. Random data is the hardest case
for nearest neighbor search, and the dataset is small. Trees pay off with
hundreds of thousands of points or more, and with structured data such as real
embedding vectors.

**Recommended tuning:** start with `n_trees` around 20–50, then measure on
your own data the smallest `search_k` that meets your target recall.
