# kodachi

English | [日本語](https://github.com/yuzujelly2222/kodachi/blob/main/README.ja.md)

A modern, pure-Python ANN library built on a forest of random projection trees.

## What is this?

kodachi is an approximate nearest neighbor (ANN) search library that uses a forest of random projection trees, written in pure Python with NumPy as its only dependency.

Graph-based methods such as HNSW (Hierarchical Navigable Small World) dominate modern ANN and are faster. Random projection forests, on the other hand, generally use less memory and are simpler to understand.

kodachi favors ease of installation (no C extensions, works anywhere) and readable code over raw speed.

## Installation

```bash
pip install kodachi
```

Requires Python 3.10+ and NumPy.

## Usage

```python
import numpy as np
from kodachi import Index

data = np.random.default_rng(0).normal(size=(10000, 64))

index = Index(dim=64, n_trees=20, seed=42)
index.build_index(data)

neighbors = index.search(data[0], k=10)  # data indices sorted by cosine similarity
```

- `build_index(data)` takes an array of shape `(n_points, dim)` and builds the trees. Calling it again discards the previous trees.
- `search(query, k, search_k)` takes a query of shape `(dim,)` and returns up to `k` data indices, sorted by descending cosine similarity.
- Both data and queries are L2-normalized internally, so vector magnitude does not affect the results.

## Parameters

| Parameter | Default | Description |
|---|---|---|
| `dim` | (required) | Dimensionality of the vectors |
| `n_trees` | `10` | Number of trees. More trees improve recall at the cost of memory and build time |
| `max_leaf_size` | `32` | Maximum number of points in a leaf |
| `seed` | `None` | Random seed. Fix it to get the same trees every time |
| `k` | `10` | Number of neighbors to return |
| `search_k` | `k * n_trees * 10` | Number of candidates collected before ranking. Larger values improve recall at the cost of speed. If it is at least the number of indexed points, an exact brute-force search is performed instead of traversing the trees |

## How it works

**Building**
1. Pick two random points and split the space with the hyperplane equidistant from them.
2. Points lying exactly on the hyperplane are split randomly between the two sides.
3. Repeat 1–2 on each side until it holds at most `max_leaf_size` points.

This is done `n_trees` times, producing a forest of trees that each split the space slightly differently. Data is normalized beforehand, so splitting by Euclidean distance gives the same ranking as cosine similarity.

**Searching**
1. In each tree, descend straight to the leaf on the query's side and collect its points. Branches not taken are pushed onto a priority queue.
2. Until `search_k` candidates are collected, explore the deferred branches in order of how close the query was to their hyperplane. The closer the query is to a hyperplane, the more likely the other side also holds near neighbors.
3. Compute the cosine similarity between the query and all candidates at once and return the top `k`.

## Complexity and tuning

Let `n` be the number of points, `d` the dimensionality, and `T` the number of trees.

- **Build:** roughly `O(T · n · d · log n)`
- **Search:** `O(T · d · log n)` to descend the trees, plus `O(search_k · d)` to rank candidates
- **Memory:** the data itself (`n · d`), plus per tree about `n / max_leaf_size` internal nodes, each holding a `d`-dimensional normal vector

Measured on 20,000 random points with 32 dimensions (recall@10):

| n_trees | search_k | recall@10 | ms/query |
|---|---|---|---|
| 10 | 2,000 | 0.76 | 1.7 |
| 20 | 4,000 | 0.95 | 3.5 |
| 50 | 5,000 | 0.98 | 5.0 |
| brute force | — | 1.00 | 0.6 |

Note that at this scale, NumPy brute force is faster. Random data is the hardest case for nearest neighbor search, and the dataset is small. Trees pay off with hundreds of thousands of points or more, and with structured data such as real embedding vectors.

**Recommended tuning:** start with `n_trees` around 20–50, then measure on your own data the smallest `search_k` that meets your target recall.

## Development

```bash
pip install -e ".[dev]"
pytest
```

### Building the docs

```bash
pip install -e ".[docs]"
sphinx-build docs docs/_build/html
```
