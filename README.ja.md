[English](https://github.com/yuzujelly2222/kodachi/blob/main/README.md) | [日本語](https://github.com/yuzujelly2222/kodachi/blob/main/README.ja.md)

# kodachi

ランダム射影の森を使ったモダンなANNライブラリのPythonネイティブ実装

[![PyPI](https://img.shields.io/pypi/v/kodachi)](https://pypi.org/project/kodachi/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/yuzujelly2222/kodachi/blob/main/LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)

## これは何か

kodachi は、ランダム射影木の森による近似最近傍探索（ANN）を、NumPy だけに依存する純粋な Python で実装したライブラリです。
最近の ANN では HNSW（階層的航行可能スモールワールド）のようなグラフ型の手法が主流で、速度ではそちらが優れています。一方で、ランダム射影の森は一般にメモリ使用量が少なく、仕組みもシンプルです。
kodachi は速度よりも、C 拡張なしでどこでもすぐ入る手軽さと、コードの読みやすさを重視しています。

`build_index()` / `search()` に渡すのは `(点の数, 次元数)` の数値配列だけなので、ベクトルの出所は問いません。
文章や画像の埋め込みベクトル、特徴量、推薦用のユーザー・アイテムベクトルなど、コサイン類似度で
「似ているもの」を探したい場面であればそのまま使えます。

本体は `kodachi/_index.py` の1ファイル（docstring 込みで約500行）にまとまっており、アルゴリズムを読んで学ぶ教材としても使えます。

## インストール

```bash
pip install kodachi
```

依存パッケージは `numpy` のみです。Python 3.10 以上が必要です。

## 使い方

```python
import numpy as np
from kodachi import Index

# インデックスに登録するベクトル（例: 10,000 個の 64 次元ベクトル）
data = np.random.default_rng(0).normal(size=(10000, 64))

index = Index(dim=64, n_trees=20, seed=42)
index.build_index(data)

# クエリに近い上位 10 件のデータ番号（コサイン類似度の高い順）
neighbors = index.search(data[0], k=10)
print(neighbors)  # -> [0 ...]（自分自身が 1 位）
```

### 埋め込みベクトルで類似検索する例

ベクトルの出所は問わないため、任意のモデルで作った埋め込みをそのまま渡すだけで使えます。

```python
texts = ["りんごが好き", "みかんが好き", "電車で通勤する", ...]

# 埋め込みの作り方は用途に応じて用意する（sentence-transformers など）
embeddings = embed(texts)          # 形は (len(texts), dim)

index = Index(dim=embeddings.shape[1], n_trees=30, seed=0)
index.build_index(embeddings)

ids = index.search(embed(["果物が好き"])[0], k=3)
print([texts[i] for i in ids])
```

### 精度と速度の調整

`search_k`（順位付けの前に集める候補の数）を増やすと recall が上がり、その分遅くなります。

```python
index.search(query, k=10)                  # 既定: search_k = k * n_trees * 10
index.search(query, k=10, search_k=5000)   # 候補を増やして精度を上げる
index.search(query, k=10, search_k=len(data))  # データ数以上なら総当たりで正確に計算
```

### 類似度も受け取る

`return_scores=True` を付けると、データ番号と一緒にコサイン類似度（-1〜1、高い順）も返します。
しきい値で足切りしたいときに使えます。

```python
ids, scores = index.search(query, k=10, return_scores=True)
ids = ids[scores >= 0.8]   # 類似度 0.8 以上だけ残す
```

### インデックスの中身を確認する

```python
len(index)   # 登録されている点の数（build_index 前は 0）
index        # Index(dim=64, n_trees=20, max_leaf_size=32, n_points=10000)
```

### 保存と読み込み

作ったインデックスは `.kodachi` ファイルに保存して、あとで木を作り直さずに読み込めます。
読み込んだインデックスは、保存前とまったく同じ検索結果を返します。

```python
index.save("index.kodachi")

from kodachi import Index
index = Index.load("index.kodachi")
neighbors = index.search(query, k=10)
```

`.kodachi` ファイルの中身は無圧縮の zip で、次のファイルが入っています。

| ファイル | 内容 |
|---|---|
| `settings.json` | 保存形式のバージョン（`format_version`）、`dim`、`n_trees`、`max_leaf_size` |
| `data.npy` | 正規化済みのデータ（`float32`） |
| `tree_<t>/*.npy` | `t` 本目の木をノード番号順の配列にしたもの（法線、オフセット、子ノード番号、葉の点の番号） |

中身は zip と `.npy` だけなので、kodachi がなくても `zipfile` と `numpy.load` で読めます。
読み込みは `allow_pickle=False` で行うため、ファイルから任意のコードが実行されることはありません。

## アルゴリズム解説

kodachi はデータをランダムな超平面で何度も 2 つに分けた木を複数本作り、検索時は
クエリが入る葉と、その近くの葉だけを調べることで計算を減らします。

### 木を作る（`build_index` → `_make_tree`）

1. **正規化** — 全データを長さ 1 に正規化する。正規化済みならユークリッド距離で分けても、
   順位がコサイン類似度と一致する。
2. **分割** — 点の集合からランダムに 2 点を選び、その 2 点からちょうど等距離にある超平面
   （`_Helper.make_offset_and_normal`）で空間を 2 つに分ける。面のちょうど上にある点は左右に半分ずつ振り分ける。
3. **再帰** — 点の数が `max_leaf_size`（既定 32）以下になるまで、左右それぞれで 2 を繰り返す。
4. これを `n_trees` 回行い、少しずつ違う分け方をした木の森を作る。

### 検索する（`search` → `_collect_candidates` → `_down_node`）

1. **第1段階** — 各木でクエリがある側にまっすぐ降りて、葉を 1 枚ずつ集める。
   通らなかった側の枝は優先度付きキューに後回しとして入れておく。
2. **第2段階** — 候補が `search_k` 個たまるまで、後回しにした枝を「クエリが超平面にどれだけ近かったか」の順に探す。
   超平面に近いほど、反対側にも近い点がある可能性が高いため。
3. **順位付け** — 集めた候補とクエリのコサイン類似度をまとめて計算し、上位 `k` 個を返す。

```
データ → 正規化 → ランダムな2点の垂直二等分面で再帰的に分割 → n_trees 本の木

クエリ → 正規化 → 各木で葉まで降りる（第1段階）
       → 超平面に近かった枝から優先的に探索（第2段階、search_k 個まで）
       → 候補とのコサイン類似度で上位 k 件を返す
```

### 計算量と目安

`n` を点の数、`d` を次元数、`T` を木の本数とします。

- **構築:** およそ `O(T · n · d · log n)`
- **検索:** 木を降りる部分が `O(T · d · log n)`、候補の順位付けが `O(search_k · d)`
- **メモリ:** データ本体の `n · d` に加えて、木 1 本あたり内部ノード（およそ `n / max_leaf_size` 個）× `d` の法線ベクトル

20,000 点 × 32 次元のランダムなデータでの測定結果:

| n_trees | search_k | recall@10 | ms/クエリ |
|---|---|---|---|
| 10 | 2,000 | 0.76 | 1.7 |
| 20 | 4,000 | 0.95 | 3.5 |
| 50 | 5,000 | 0.98 | 5.0 |
| 総当たり | — | 1.00 | 0.6 |

この規模では NumPy による総当たりのほうが速いことに注意してください。ランダムなデータは近傍探索にとって
最も難しいケースで、さらに点の数も少ないためです。木を使う効果が出るのは、点の数が数十万以上で、
実際の埋め込みベクトルのように偏りのあるデータです。

`n_trees` は 20〜50 くらいから始めて、実際に使うデータで「目標の recall を満たす最小の `search_k`」を測って決めるのがおすすめです。

## API

### `Index`

公開 API は `kodachi` パッケージから import できる `Index` と `FORMAT_VERSION` だけです。
実装の入った `kodachi._index` モジュールや、`_Node`（木のノード）、`_Helper`（幾何計算）、
`_make_tree`、`_collect_candidates`、`_down_node` は内部実装用のため `_` を付けて非公開にしています。
1.0.0 以降、公開 API は互換性を壊さないように保ちます（セマンティック バージョニング）。

| メンバ | 説明 |
|---|---|
| `Index(dim, n_trees=10, max_leaf_size=32, seed=None)` | 空のインデックスを作る。`seed` を固定すると毎回同じ木ができる |
| `build_index(data)` | `(点の数, dim)` の配列から木を作る。もう一度呼ぶと前の木は捨てて作り直す |
| `search(query, k=10, search_k=None, return_scores=False)` | `(dim,)` のクエリに対し、コサイン類似度の高い順に最大 `k` 個のデータ番号（`int64`）を返す。`return_scores=True` なら `(番号, 類似度)` のタプルを返す |
| `save(path)` | インデックスを `.kodachi` ファイル（無圧縮の zip）に保存する。同名のファイルは上書きする |
| `Index.load(path, seed=None)` | `save` で保存したファイルからインデックスを復元するクラスメソッド |
| `len(index)` | 登録されている点の数（`build_index` 前は 0） |
| `repr(index)` | `Index(dim=..., n_trees=..., max_leaf_size=..., n_points=...)` 形式の要約 |

### その他

- `FORMAT_VERSION` — `save` が書き出す `.kodachi` ファイルの保存形式のバージョン（現在は `1`）

### バリデーション

不正な入力は次の例外で弾きます。

- `dim`、`n_trees`、`max_leaf_size`、`k`、`search_k` が整数でない → `TypeError`（`np.int64` などの NumPy の整数は受け付ける。`True` / `False` は不可）
- 上記の値が 1 未満 → `ValueError`
- `build_index` に 2 次元でない配列、列数が `dim` と違う配列、空の配列、NaN や無限大を含む配列を渡した → `ValueError`
- `search` に形が `(dim,)` でないクエリ、または NaN や無限大を含むクエリを渡した → `ValueError`
- `build_index` を呼ぶ前に `search` または `save` を呼んだ → `RuntimeError`
- `load` に存在しないパスを渡した → `FileNotFoundError`
- `load` に kodachi のインデックスでないファイル、または対応していない保存形式のバージョンのファイルを渡した → `ValueError`

`k` がデータ数より大きい場合はエラーにせず、全データを類似度順に返します。

## ドキュメント・テスト

Sphinx で生成した API リファレンスが `docs/` にあり、GitHub Pages として
<https://yuzujelly2222.github.io/kodachi/> で公開しています
（`docs_src/` が Sphinx のソース、`docs/` がビルド済みの公開用 HTML です）。ソースから再生成する場合:

```bash
pip install -e ".[docs]"
sphinx-build -b html -d docs_src/_build/doctrees docs_src docs
```

`tests/` には pytest ベースのテストスイートがあります。

```bash
pip install -e ".[dev]"
pytest
```

## ライセンスと出典

本リポジトリのコードは MIT License（[LICENSE](https://github.com/yuzujelly2222/kodachi/blob/main/LICENSE)）で公開しています。

アルゴリズムは Spotify の [Annoy](https://github.com/spotify/annoy)（Erik Bernhardsson 作）で使われている、
ランダムな 2 点の垂直二等分面で分割する木の森と、優先度付きキューによる探索の考え方を参考に、
独自に Python で実装したものです。Annoy のコードは含んでいません。
