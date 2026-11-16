# kodachi

A modern, pure-Python ANN library built on a forest of random projection trees.

kodachi is an approximate nearest neighbor (ANN) search library that uses a
forest of random projection trees, written in pure Python with NumPy as its
only dependency. It favors ease of installation (no C extensions, works
anywhere) and readable code over raw speed.

## Installation

```bash
pip install kodachi
```

Requires Python 3.10+ and NumPy.

## Quickstart

```python
import numpy as np
from kodachi import Index

data = np.random.default_rng(0).normal(size=(10000, 64))

index = Index(dim=64, n_trees=20, seed=42)
index.build_index(data)

neighbors = index.search(data[0], k=10)  # data indices sorted by cosine similarity
```

```{toctree}
:maxdepth: 2
:caption: Contents

guide
api
```
