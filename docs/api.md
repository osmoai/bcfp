# BCFP API

BCFP generates molecular fingerprints — **ECFP** (atom-centered) and **BCFP** (bond-centered) — with
three interchangeable hash backends and Sort&Slice feature selection + out-of-vocabulary (OOV)
handling. The C++ core is fast and **self-contained** (xxHash is header-only; BLAKE3 is vendored), so
RDKit is the only external build dependency.

| layer | entry point | use it for |
|---|---|---|
| **sklearn-style** | `BCFPEnhanced` | `fit` / `transform` / `fit_transform` with Sort&Slice + OOV (recommended) |
| **low-level** | `FingerprintGenerator` | `generate_sparse` + static `sortslice_fit` / `sortslice_transform` / `fold_sparse_to_dense` |
| **persistence** | `FingerprintModel`, `save_sortslice_vocab` / `load_sortslice_vocab` | save/restore a fitted vocabulary for inference |

## Install / build

See [BUILD.md](../BUILD.md):

```bash
conda env create -f environment.yml && conda activate bcfp
pip install -e .
```

## Hashes and fingerprint types

- **Hashes** (`hash_func`): `rdkit_native` (RDKit's Morgan hash), `xxhash` (XXH3-128, header-only),
  `blake3` (vendored portable C). All three support both ECFP and BCFP.
- **Types** (`fp_type`): `ecfp` (atom-centered Morgan) or `bcfp` (bond-centered).

## `BCFPEnhanced` — scikit-learn style (recommended)

```python
from bcfp import BCFPEnhanced

# Sort&Slice to top-512 keys + an OOV bucket for unseen test bits
gen = BCFPEnhanced(fp_type='ecfp', radius=2, n_bits=2048, top_k=512, include_oov=True)
X_train = gen.fit_transform(train_smiles)   # (n_train, 513): 512 selected + 1 OOV
X_test  = gen.transform(test_smiles)         # applies the fitted vocabulary + OOV bucket

# Baseline (no selection): fold to n_bits, no fit required
base = BCFPEnhanced(fp_type='ecfp', radius=2, n_bits=2048, top_k=None)
X = base.transform(smiles)                   # (n, 2048)
```

- `top_k=None` → plain folding to `n_bits` (no vocabulary; `transform` works without `fit`).
- `top_k=K` → Sort&Slice selects the top-K most frequent training keys (`fit` required before
  `transform`). `include_oov=True` appends one column counting test-time keys outside the vocabulary.
- Fitted state: `gen.vocab_`, `gen.key2col_`, `gen.selected_indices_`.

## `FingerprintGenerator` — low-level

```python
from bcfp import FingerprintGenerator
from rdkit import Chem
import numpy as np

gen = FingerprintGenerator('rdkit_native', 'ecfp', radius=2, use_counts=True, n_bits=2048)
sparse = [gen.generate_sparse(Chem.MolFromSmiles(s)) for s in smiles]   # list of {key: count}

# (a) fold to a fixed width
X = np.array([gen.fold_sparse_to_dense(sp) for sp in sparse])

# (b) Sort&Slice (static helpers)
vocab, key2col = FingerprintGenerator.sortslice_fit(sparse, train_idx, top_k=512, sort_by='df', min_df=2)
X = FingerprintGenerator.sortslice_transform(sparse, all_idx, key2col, use_counts=True, add_oov_bucket=True)
```

## Parameter reference

| parameter | default | meaning |
|---|---|---|
| `fp_type` | `ecfp` | `ecfp` (atom-centered) or `bcfp` (bond-centered). |
| `radius` | 2 | Morgan radius. |
| `n_bits` | 2048 | Folded width in baseline mode (`top_k=None`). |
| `hash_func` | `rdkit_native` | `rdkit_native` / `xxhash` / `blake3`. |
| `top_k` | None | Sort&Slice vocabulary size; None = plain folding. |
| `include_oov` | False | Append an out-of-vocabulary bucket column (with `top_k`). |
| `use_counts` | True | Count occurrences (else binary presence). |
| `use_chirality` | True | Include chirality in the fingerprint. |
| `sort_by` | `df` | Rank keys by document frequency (`df`) or total frequency (`tf`). |
| `min_df` | 2 | Minimum document frequency for a key to be selectable. |

## Persistence

```python
from bcfp import save_sortslice_vocab, load_sortslice_vocab
save_sortslice_vocab(gen.key2col_, 'vocab.json')
key2col = load_sortslice_vocab('vocab.json')
```
`FingerprintModel` (see `bcfp.model_persistence`) bundles a fitted generator + vocabulary for
end-to-end save/load.
