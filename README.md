[English](https://github.com/yuzujelly2222/kodachi/blob/main/README.md) | [日本語](https://github.com/yuzujelly2222/kodachi/blob/main/README.ja.md)

# kodachi

A modern, pure-Python ANN library built on a forest of random projection trees.

[![PyPI](https://img.shields.io/pypi/v/kodachi)](https://pypi.org/project/kodachi/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/yuzujelly2222/kodachi/blob/main/LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)

> 詳しい日本語の解説は [README.ja.md](https://github.com/yuzujelly2222/kodachi/blob/main/README.ja.md) にあります。

## What is this

kodachi is an approximate nearest neighbor (ANN) search library built on a forest of
random projection trees, written in pure Python with NumPy as its only dependency.

Graph-based methods such as HNSW (Hierarchical Navigable Small World) dominate modern
ANN and are faster. Random projection forests, on the other hand, generally use less
memory and are simpler to understand. kodachi favors ease of installation (no C
extensions, works anywhere) and readable code over raw speed.

Input is a plain numeric array of shape `(n_points, dim)`, so the vectors can come from
anywhere: text or image embeddings, feature vectors, user/item vectors for
recommendation — anything you want to search by cosine similarity.

The whole implementation fits in a single file, `kodachi/_index.py` (about 500 lines including docstrings),
so it also works as a readable reference for the algorithm.

## Install

```bash
pip install kodachi
```

Depends only on `numpy`. Requires Python 3.10+.

## Usage

```python
import numpy as np
from kodachi import Index

# Vectors to index (e.g. 10,000 vectors with 64 dimensions)
data = np.random.default_rng(0).normal(size=(10000, 64))

index = Index(dim=64, n_trees=20, seed=42)
index.build_index(data)

# Indices of the top 10 neighbors, sorted by cosine similarity
neighbors = index.search(data[0], k=10)
print(neighbors)  # -> [0 ...] (the query itself ranks first)
```

### Similarity search over embeddings

```python
texts = ["I like apples", "I like oranges", "I commute by train", ...]

embeddings = embed(texts)          # any embedding model, shape (len(texts), dim)

index = Index(dim=embeddings.shape[1], n_trees=30, seed=0)
index.build_index(embeddings)

ids = index.search(embed(["I like fruit"])[0], k=3)
print([texts[i] for i in ids])
```

### Trading accuracy for speed

Raising `search_k` (the number of candidates collected before ranking) improves recall
at the cost of speed.

```python
index.search(query, k=10)                      # default: search_k = k * n_trees * 10
index.search(query, k=10, search_k=5000)       # more candidates, higher recall
index.search(query, k=10, search_k=len(data))  # >= n_points: exact brute-force search
```

### Getting similarity scores

Pass `return_scores=True` to also get the cosine similarity of each neighbor (from -1
to 1, in descending order), e.g. to drop results below a threshold:

```python
ids, scores = index.search(query, k=10, return_scores=True)
ids = ids[scores >= 0.8]   # keep only neighbors with similarity >= 0.8
```

### Inspecting an index

```python
len(index)   # number of indexed points (0 before build_index)
index        # Index(dim=64, n_trees=20, max_leaf_size=32, n_points=10000)
```

### Saving and loading

Save a built index to a `.kodachi` file and load it later without rebuilding the trees.
A loaded index returns exactly the same results as the saved one.

```python
index.save("index.kodachi")

from kodachi import Index
index = Index.load("index.kodachi")
neighbors = index.search(query, k=10)
```

A `.kodachi` file is an uncompressed zip archive containing:

| File | Contents |
|---|---|
| `settings.json` | Format version (`format_version`), `dim`, `n_trees`, `max_leaf_size` |
| `data.npy` | Normalized data (`float32`) |
| `tree_<t>/*.npy` | Tree `t` flattened into per-node arrays (normals, offsets, child ids, leaf indices) |

Since it is just zip and `.npy`, it can be read with `zipfile` and `numpy.load` even
without kodachi. Arrays are loaded with `allow_pickle=False`, so loading a file never
executes arbitrary code.

## How it works

**Building** (`build_index` -> `_make_tree`)

1. **Normalize** all data to unit length, so splitting by Euclidean distance gives the
   same ranking as cosine similarity.
2. **Split** by picking two random points and cutting the space with the hyperplane
   equidistant from them. Points exactly on the hyperplane are split evenly.
3. **Recurse** on each side until it holds at most `max_leaf_size` (default 32) points.
4. Repeat `n_trees` times to get a forest of trees that each split the space differently.

**Searching** (`search` -> `_collect_candidates` -> `_down_node`)

1. In each tree, descend straight to the leaf on the query's side. Branches not taken
   are pushed onto a priority queue.
2. Until `search_k` candidates are collected, explore deferred branches in order of how
   close the query was to their hyperplane.
3. Rank all candidates by exact cosine similarity and return the top `k`.

```
data  -> normalize -> recursively split by random bisecting hyperplanes -> n_trees trees

query -> normalize -> descend each tree to a leaf (stage 1)
      -> explore branches closest to their hyperplane first (stage 2, up to search_k)
      -> return the top k candidates by cosine similarity
```

Let `n` be the number of points, `d` the dimensionality, and `T` the number of trees.
Building takes roughly `O(T · n · d · log n)`; searching takes `O(T · d · log n)` plus
`O(search_k · d)` to rank candidates.

On 20,000 random 32-dimensional points, `n_trees=20, search_k=4000` reaches recall@10 of
0.95 at 3.5 ms/query — but NumPy brute force takes only 0.6 ms at this scale. Trees pay
off with hundreds of thousands of points or more and with structured data such as real
embeddings. Start with `n_trees` around 20–50 and tune `search_k` on your own data. See
[README.ja.md](https://github.com/yuzujelly2222/kodachi/blob/main/README.ja.md) for the
full benchmark table.

## API

The public API is `Index` and `FORMAT_VERSION`, imported from the `kodachi` package.
The `kodachi._index` module and everything prefixed with `_` (`_Node`, `_Helper`,
`_make_tree`, `_collect_candidates`, `_down_node`) are internal. From 1.0.0 on, the
public API follows semantic versioning.

| Member | Description |
|---|---|
| `Index(dim, n_trees=10, max_leaf_size=32, seed=None)` | Create an empty index. Fix `seed` for reproducible trees |
| `build_index(data)` | Build the forest from an array of shape `(n_points, dim)`. Calling it again rebuilds from scratch |
| `search(query, k=10, search_k=None, return_scores=False)` | Return up to `k` data indices (`int64`) for a `(dim,)` query, sorted by descending cosine similarity. With `return_scores=True`, return `(indices, scores)` |
| `save(path)` | Save the index to a `.kodachi` file (uncompressed zip), overwriting any existing file |
| `Index.load(path, seed=None)` | Class method that restores an index saved with `save` |
| `len(index)` | Number of indexed points (0 before `build_index`) |
| `repr(index)` | Summary like `Index(dim=..., n_trees=..., max_leaf_size=..., n_points=...)` |
| `FORMAT_VERSION` | Version of the `.kodachi` file format written by `save` (currently `1`) |

Invalid input raises:

- `TypeError` if `dim`, `n_trees`, `max_leaf_size`, `k` or `search_k` is not an int
  (NumPy integers are accepted; `bool` is not)
- `ValueError` if any of them is less than 1
- `ValueError` if `build_index` gets an array that is not 2-D, whose width is not `dim`,
  that is empty, or that contains NaN or infinity
- `ValueError` if `search` gets a query whose shape is not `(dim,)` or that contains NaN or infinity
- `RuntimeError` if `search` or `save` is called before `build_index`
- `FileNotFoundError` if `load` gets a path that does not exist
- `ValueError` if `load` gets a file that is not a kodachi index or uses an unsupported format version

If `k` exceeds the number of points, all points are returned in similarity order.

## Documentation & tests

A Sphinx-generated API reference lives in `docs/` (source in `docs_src/`), published via
GitHub Pages at <https://yuzujelly2222.github.io/kodachi/>. To rebuild it locally:

```bash
pip install -e ".[docs]"
sphinx-build -b html -d docs_src/_build/doctrees docs_src docs
```

A pytest suite covering the index lives in `tests/`:

```bash
pip install -e ".[dev]"
pytest
```

## License & credits

Distributed under the **MIT License** — see
[LICENSE](https://github.com/yuzujelly2222/kodachi/blob/main/LICENSE).

The algorithm is an independent Python implementation inspired by Spotify's
[Annoy](https://github.com/spotify/annoy) by Erik Bernhardsson: a forest of trees split
by random bisecting hyperplanes, searched with a priority queue. No Annoy code is included.
