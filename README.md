# BCFP: High-Performance Molecular Fingerprints

[![License: BSD-3-Clause](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![Python 3.8+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![C++20](https://img.shields.io/badge/C++-20-blue.svg)](https://isocpp.org/)
[![RDKit 2026.09.2](https://img.shields.io/badge/RDKit-2026.09.2-green.svg)](https://www.rdkit.org/)

High-performance molecular fingerprint generation with **3 optimized hash function implementations** and advanced feature selection. Supports both ECFP (Extended-Connectivity) and BCFP (Bond-Centered) fingerprints with Sort&Slice and Out-of-Vocabulary (OOV) handling.

**📄 Research Paper**: [Bond-Centered Molecular Fingerprint Derivatives: A BBBP Dataset Study](https://arxiv.org/abs/2510.04837) (arXiv:2510.04837)

## Features

🚀 **3 Hash Functions**: RDKit native, BLAKE3, xxHash

⚡ **Optimized C++ Core**: 10-100x faster than pure Python implementations

🎯 **ECFP & BCFP**: Both atom-centered and bond-centered fingerprints

📊 **Sort&Slice**: Intelligent feature selection to reduce dimensionality

🔍 **OOV Handling**: Gracefully handle out-of-vocabulary features in test sets

🧪 **Production-Ready**: Extensively validated, used in real-world drug discovery

## Installation

### Requirements

- Python >= 3.11
- RDKit **2026.09.2** — C++ headers via conda-forge `librdkit-dev`
- A **C++20** compiler (clang on macOS, gcc/clang on Linux)
- NumPy

The two non-RDKit hashes require no linked libraries: the environment supplies xxHash's header-only
C++ API, and BLAKE3 is vendored under `src/vendor/blake3/`. No Python hash packages are required.

### Install from source

```bash
git clone https://github.com/osmoai/bcfp.git
cd bcfp

# One command — RDKit 2026.09.2 + C++ headers + nanobind + C++20 toolchain
conda env create -f environment.yml      # or: mamba env create -f environment.yml
conda activate bcfp

# Build + install (editable). CMake auto-detects RDKit from the active env.
pip install -e .

# Verify
python -c "import bcfp; from bcfp.fingerprints import FingerprintGenerator; print('bcfp OK')"
```

See **[BUILD.md](BUILD.md)** for build internals, a custom-RDKit (`RDKIT_PREFIX`) path, and
troubleshooting.

## Quick Start

### Two-Step ML Pipeline

BCFP provides **Step 1** (feature generation). You provide **Step 2** (ML model).

```
┌─────────────────────────────────────────────────────────────────────┐
│  MOLECULES (SMILES)                                                 │
│  ["CCO", "c1ccccc1", ...]                                          │
└─────────────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────────┐
│  STEP 1: Feature Generation (BCFP)                                 │
│  FingerprintModel → Numerical Features                             │
│  • Hash function: xxhash/blake3/rdkit_native                       │
│  • Fingerprint type: ECFP/BCFP/both                                │
│  • Sort&Slice vocabulary                                           │
└─────────────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────────┐
│  FEATURES (Matrix)                                                  │
│  X = [[0.1, 0.5, ...], [0.2, 0.3, ...], ...]                      │
└─────────────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────────┐
│  STEP 2: ML Model (Your Choice)                                    │
│  XGBoost / CatBoost / RandomForest / LogisticRegression            │
│  Features → Predictions                                             │
└─────────────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────────┐
│  PREDICTIONS                                                        │
│  [0, 1, 1, 0, ...]                                                 │
└─────────────────────────────────────────────────────────────────────┘
```

### Complete Example: Training → Save → Load → Predict

```python
from bcfp import FingerprintModel
from xgboost import XGBClassifier  # Or CatBoost, RandomForest, etc.
import numpy as np
import pickle

# ========================================================================
# TRAINING PHASE
# ========================================================================

# Training data
train_smiles = ["CCO", "CC(C)O", "CCCO", "CCCC", "CCC", "CC"]
train_labels = np.array([1, 1, 1, 0, 0, 0])  # Active / Inactive

# STEP 1: Feature Generation (FingerprintModel)
fp_model = FingerprintModel(
    hash_func='xxhash',    # ← CRITICAL: Saved with model
    fp_type='ecfp',        # ECFP (Morgan fingerprints)
    radius=2,              # ECFP4
    top_k=512,             # Select 512 best features
    add_oov_bucket=True    # Handle unseen features
)

train_idx = np.arange(len(train_smiles))
X_train = fp_model.fit_transform(train_smiles, train_idx)
print(f"Features: {X_train.shape}")  # (6, 513) = 512 features + 1 OOV

# STEP 2: ML Model (XGBoost Classifier)
ml_model = XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    n_jobs=-1,             # Multi-threaded
    random_state=42
)
ml_model.fit(X_train, train_labels)

# Save BOTH models (Step 1 + Step 2)
fp_model.save('fingerprint_model.pkl')        # Step 1
ml_model.save_model('xgboost_model.json')     # Step 2

print("✅ Both models saved!")

# ========================================================================
# INFERENCE PHASE (Production)
# ========================================================================

# Load BOTH models
fp_model_loaded = FingerprintModel.load('fingerprint_model.pkl')  # Step 1
ml_model_loaded = XGBClassifier()
ml_model_loaded.load_model('xgboost_model.json')                  # Step 2

# New molecules (never seen during training)
new_smiles = ["CCCCO", "CCCCCC", "CC(C)CCO"]

# STEP 1: Generate features
X_new = fp_model_loaded.transform(new_smiles)

# STEP 2: Predict
predictions = ml_model_loaded.predict(X_new)
probabilities = ml_model_loaded.predict_proba(X_new)[:, 1]

print("\nPredictions:")
for smi, pred, prob in zip(new_smiles, predictions, probabilities):
    label = "Active" if pred == 1 else "Inactive"
    print(f"  {smi:15s} → {label:8s} (probability: {prob:.3f})")

# Output:
#   CCCCO           → Active   (probability: 0.940)
#   CCCCCC          → Inactive (probability: 0.120)
#   CC(C)CCO        → Active   (probability: 0.970)
```

### Basic ECFP Generation

```python
from bcfp import BCFPEnhanced

# Default: Sort&Slice top-(n_bits-1) training keys + 1 OOV bucket (fit on training data)
gen = BCFPEnhanced(
    hash_func='xxhash',  # Fast non-cryptographic hash
    fp_type='ecfp',      # Extended-Connectivity Fingerprint
    radius=2,            # ECFP4 (radius 2 = diameter 4)
    n_bits=2048,         # Output width: <=2047 selected keys + 1 OOV column
)
X_train = gen.fit_transform(train_smiles)
X_test = gen.transform(test_smiles)

# Stateless alternative (no fit, no vocabulary): fold to n_bits
folded = BCFPEnhanced(fp_type='ecfp', n_bits=2048, top_k=None, fold='remix')
fingerprints = folded.transform(["CCO", "c1ccccc1", "CC(=O)O"])
print(fingerprints.shape)  # (3, 2048)
```

> **Changed in 2.1.0:** `BCFPEnhanced` defaults to Sort&Slice + OOV (`top_k='auto'`,
> `include_oov=True`), so call `fit` before `transform`. Pass `top_k=None` to get the old plain
> folding. See [Choosing a folding scheme](#choosing-a-folding-scheme).

> For the low-level API (sparse keys + Sort&Slice helpers) use `FingerprintGenerator.generate_sparse`
> / `sortslice_fit` / `sortslice_transform` — see [docs/api.md](docs/api.md).

### chemfp-compatible output

`FingerprintGenerator` can return chemfp's native binary form directly, without first creating a
Python dictionary or NumPy array:

```python
from bcfp import FingerprintGenerator
from rdkit import Chem

mol = Chem.MolFromSmiles("CCO")
gen = FingerprintGenerator("xxhash", "bcfp", radius=2, n_bits=2048)

binary_fp = gen.generate_bytes(mol)  # 256-byte, folded binary fingerprint
count_fp = gen.generate_chemfp_count_fingerprint(mol)  # chemfp CountFingerprint
```

The second call requires chemfp 5.2b1 or later. To control the boundary explicitly, use
`generate_count_buffer(mol, layout="64_32_packed")` and pass the result to
`chemfp.countops.CountFingerprint.from_buffer_64_32_packed()`. The `32_32`, `64_32_packed`
(12 bytes/feature), and aligned `64_32` (16 bytes/feature) layouts are supported.

RDKit molecules cross into the nanobind C++ extension through `mol.ToBinary()` rather than a SMILES
round-trip. This preserves RDKit's serialized molecule state, avoids canonicalization and
reparsing, and keeps the boundary independent of RDKit's Python binding implementation.

### ECFP + BCFP Concatenation

```python
import numpy as np
# ECFP captures atom neighborhoods
gen_ecfp = BCFPEnhanced(hash_func='xxhash', fp_type='ecfp', radius=2, n_bits=2048)
fp_ecfp = gen_ecfp.fit_transform(train_smiles)

# BCFP captures bond neighborhoods (complementary)
gen_bcfp = BCFPEnhanced(hash_func='xxhash', fp_type='bcfp', radius=2, n_bits=2048)
fp_bcfp = gen_bcfp.fit_transform(train_smiles)

# Concatenate for enhanced representation
fp_combined = np.hstack([fp_ecfp, fp_bcfp])  # up to (n, 4096)
```

### Sort&Slice with OOV

```python
from bcfp import FingerprintGenerator
from rdkit import Chem
import numpy as np

# Training and test data
train_smiles = ["CCO", "c1ccccc1", "CC(=O)O"] * 100
test_smiles = ["CCCCC", "CCCCCC"]  # Novel structures

# Convert to RDKit molecules
train_mols = [Chem.MolFromSmiles(s) for s in train_smiles]
test_mols = [Chem.MolFromSmiles(s) for s in test_smiles]

# Generate sparse fingerprints with radius=3 (ECFP6)
gen = FingerprintGenerator('xxhash', 'ecfp', radius=3)
train_fps = [gen.generate_sparse(mol) for mol in train_mols]
test_fps = [gen.generate_sparse(mol) for mol in test_mols]

# Select top 2048 features from training set
vocab, key2col = gen.sortslice_fit(
    train_fps, 
    np.arange(len(train_mols)), 
    top_k=2048
)

# Transform to dense with OOV bucket
X_train = gen.sortslice_transform(
    train_fps, 
    np.arange(len(train_mols)), 
    key2col, 
    use_counts=True, 
    add_oov_bucket=True
)  # Shape: (300, 2049) - 2048 features + 1 OOV

X_test = gen.sortslice_transform(
    test_fps, 
    np.arange(len(test_mols)), 
    key2col, 
    use_counts=True, 
    add_oov_bucket=True
)  # Shape: (2, 2049)
```

## Choosing a folding scheme

There are tens of thousands of sparse ECFP/BCFP keys, and they have to fit into a fixed number of
columns. How they are packed matters far more than which hash function made them. Benchmark: BBBP
(4092 molecules), R=2 ECFP + BCFP = 67,009 keys (~33 per column at 2048), XGBoost,
10 seeds × 10-fold with paired splits, vocabulary fitted on the training fold only. Δ is the
difference from an ideal random hash (mean of 10 salts), with 95% CI.

| scheme (2048 columns) | AUROC | AUPRC | ΔAUROC vs random hash |
|---|---|---|---|
| **Sort&Slice + 1 OOV bucket** (default) | **0.8522** | **0.9104** | **+0.0145** [+0.0131, +0.0158] |
| exact keys, no folding (~60k columns) | 0.8519 | 0.9101 | +0.0142 |
| Sort&Slice, no OOV | 0.8511 | 0.9102 | +0.0134 |
| random hash (reference) | 0.8377 | 0.8996 | 0 |
| native keys `% 2048` (old default) | 0.8346 | 0.8976 | −0.0031 [−0.0046, −0.0016] |
| two hashes per key | 0.8292 | 0.8932 | −0.0085 |

**Should the OOV bucket be an array?** No. Hashing the out-of-vocabulary keys into several columns
instead of summing them into one makes results worse at every size tested (2048 columns in total;
Δ is the paired difference from the single bucket):

| OOV columns | AUROC | ΔAUROC vs 1 bucket |
|---|---|---|
| **1** | **0.8522** | reference |
| 20 (1%) | 0.8417 | −0.0104 ± 0.0009 |
| 41 (2%) | 0.8386 | −0.0135 ± 0.0010 |
| 102 (5%) | 0.8362 | −0.0159 ± 0.0011 |

Each extra OOV column sums hundreds of unrelated rare keys. That gives trees noisy, size-correlated
splits to overfit to. A single bucket keeps the "how much rare chemistry" signal without the noise.

- **Default:** `BCFPEnhanced(n_bits=2048)`, which is Sort&Slice + 1 OOV and matches the full
  unfolded representation at 1/30 of the columns. At 4096 columns it scores higher than the
  unfolded representation (0.8537).
- **Stateless fallback:** `top_k=None, fold='remix'`, which applies `splitmix64(key ^ salt)` before
  `% n_bits`. Plain `% n_bits` on native keys scores below every random salt.
- **Hash function:** `rdkit_native`, `xxhash` and `blake3` differ by less than the salt-to-salt noise
  (~0.005 AUROC) once they are compared on the same features.

The full tables (1024/4096 widths, signed hashing, hashed tails) and the method are in
[docs/binning_benchmark.md](docs/binning_benchmark.md). To reproduce:
`python benchmarks/binning_bbbp.py --data bbbp.csv` (add `--oov-sweep` for the OOV table).
These numbers are from one dataset and one model.

## Does BCFP earn its place?

Adding BCFP doubles the key vocabulary, so it has to beat simply giving ECFP more columns. Two
datasets, five arms, identical folds within each; everything binned with Sort&Slice + 1 OOV bucket,
R=2 counts, vocabulary fitted on the training fold only. The `@4096` arms are the control: if the
combined set only wins because it has more features to spread out, doubling ECFP's width should
close the gap.

**BBBP** (4089 molecules, binary, AUROC) and the **Biogen ADME** set (3521 molecules, four
regression endpoints, Spearman ρ) — the second is the replication, on different data, a different
task and a different metric. [The BCFP paper](https://arxiv.org/abs/2510.04837) established the
combination on BBBP with a Random Forest; these runs re-measure it with XGBoost, add a
column-budget control, and check that it holds on a second dataset.

| comparison | BBBP, scaffold split | Biogen ADME, scaffold split (pooled over 4 endpoints) |
|---|---|---|
| **+BCFP on top of ECFP** | **+0.0054** [+0.0039, +0.0070] | **+0.0105** [+0.0057, +0.0153] — positive on **4/4** endpoints |
| BCFP *instead of* ECFP | −0.0022 [−0.0053, +0.0008] | **−0.0159** [−0.0169, −0.0150] — negative on **4/4** endpoints |
| ECFP alone, 4096 vs 2048 columns (width alone) | −0.0013 [−0.0028, +0.0001] | +0.0002 [−0.0014, +0.0018] |

- **Add BCFP, never substitute it.** Bond-centred keys are a *worse* standalone description of a
  molecule — on the ADME endpoints, measurably so on every one — and a genuinely complementary
  addition to an atom-centred one. The two contribute symmetrically: removing either half of the
  combined set costs about the same.
- **The gain is information, not columns.** Widening ECFP on its own is worth nothing on either
  dataset (−0.0013 / +0.0002). Extra width pays only when there is new chemistry to put in it.
- **It is not uniform across endpoints.** On MDR1-MDCK efflux ratio, the most whole-molecule
  physicochemical of the four, BCFP adds nothing measurable (+0.0037, CI spans zero). The metabolic
  clearance and solubility endpoints gain +0.011 to +0.018. Test it on your endpoint.
- **Use 2048 columns for the combined set** unless you have measured otherwise. BBBP prefers 4096,
  Biogen does not (2048 matches or beats 4096 in 7 of 8 cells), so the wider array is a per-dataset
  experiment rather than a default.

```python
import numpy as np
ecfp = BCFPEnhanced(fp_type='ecfp', radius=2, n_bits=2048).fit_transform(train_smiles)
bcfp = BCFPEnhanced(fp_type='bcfp', radius=2, n_bits=2048).fit_transform(train_smiles)
X = np.hstack([ecfp, bcfp])
```

Full tables, per-endpoint numbers and the method are in
[docs/bcfp_vs_ecfp.md](docs/bcfp_vs_ecfp.md). To reproduce:
`python benchmarks/bcfp_vs_ecfp.py --data bbbp.csv --task classification --label p_np` or
`python benchmarks/bcfp_vs_ecfp.py --data biogen.csv --task regression --smiles-col SMILES`.

## Hash Functions

| Hash Function | Type | Speed | Use Case |
|---------------|------|-------|----------|
| `xxhash` | Non-crypto | **Fastest** | Production (recommended) |
| `blake3` | Crypto | Very Fast | Reproducibility required |
| `rdkit_native` | Reference | Moderate | Compatibility with RDKit |

**Important**: The hash function determines the feature space. Once a model is trained with a specific hash, it must use the same hash for inference. See [Hash Function Compatibility](docs/HASH_FUNCTION_COMPATIBILITY.md).

## ML Models (Step 2)

BCFP generates features (Step 1). You choose the ML model (Step 2):

### Supported ML Models

| ML Model | Library | Use Case | Multi-threading |
|----------|---------|----------|----------------|
| **XGBoost** | `xgboost` | **Recommended** - Fast, accurate | `n_jobs=-1` |
| **CatBoost** | `catboost` | Categorical features, robust | `thread_count=-1` |
| **RandomForest** | `sklearn` | Interpretable, stable | `n_jobs=-1` |
| **LightGBM** | `lightgbm` | Large datasets, memory efficient | `n_jobs=-1` |
| **LogisticRegression** | `sklearn` | Simple, fast baseline | `n_jobs=-1` |
| **Neural Networks** | `pytorch/tensorflow` | Deep learning, complex patterns | GPU support |

### Example: Different ML Models

```python
from bcfp import FingerprintModel
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import numpy as np

# Training data
train_smiles = ["CCO", "CC(C)O", "CCCO", "CCCC", "CCC", "CC"]
train_labels = np.array([1, 1, 1, 0, 0, 0])

# STEP 1: Feature Generation (same for all ML models)
fp_model = FingerprintModel(hash_func='xxhash', radius=2, top_k=512)
train_idx = np.arange(len(train_smiles))
X_train = fp_model.fit_transform(train_smiles, train_idx)

# STEP 2: Choose your ML model

# Option 1: XGBoost (Most Popular)
xgb_model = XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    n_jobs=-1,           # ← Multi-threaded
    random_state=42
)
xgb_model.fit(X_train, train_labels)

# Option 2: CatBoost
catboost_model = CatBoostClassifier(
    iterations=100,
    depth=6,
    learning_rate=0.1,
    thread_count=-1,     # ← Multi-threaded
    verbose=False
)
catboost_model.fit(X_train, train_labels)

# Option 3: Random Forest
rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    n_jobs=-1,           # ← Multi-threaded
    random_state=42
)
rf_model.fit(X_train, train_labels)

# Option 4: Logistic Regression
lr_model = LogisticRegression(
    C=1.0,
    max_iter=1000,
    n_jobs=-1,           # ← Multi-threaded
    random_state=42
)
lr_model.fit(X_train, train_labels)

# All models work with the same features!
```

### Multi-Task Learning

For multi-task problems (multiple labels per molecule), use custom objectives:

```python
from xgboost import XGBClassifier
import numpy as np

# Multi-task data (3 tasks)
train_labels = np.array([
    [1, 0, np.nan],  # Molecule 1: task1=1, task2=0, task3=unknown
    [0, 1, 1],       # Molecule 2: all tasks known
    [1, np.nan, 0],  # Molecule 3: task2 unknown
])

# XGBoost with NaN-aware objective (custom implementation required)
# See examples/ for multi-task implementations
```

**Recommendation**: Use `xxhash` for production. Use `blake3` if reproducibility across platforms is critical. Use `rdkit_native` for compatibility with existing RDKit-based workflows.

## Examples

See the `examples/` directory for comprehensive examples:

- **`example_ecfp_basic.py`**: Basic ECFP generation with different parameters
- **`example_bcfp_basic.py`**: BCFP generation and hash function comparison
- **`example_sortslice_oov.py`**: Feature selection with Sort&Slice and OOV
- **`example_multitask_prediction.py`**: Complete multi-task prediction workflow

## API Reference

### FingerprintGenerator

```python
FingerprintGenerator(
    hash_algo='xxhash',      # Hash function: 'xxhash', 'blake3', 'rdkit_native'
    fp_type='ecfp',          # 'ecfp' (atom-centered) or 'bcfp' (bond-centered)
    radius=2,                # Fingerprint radius (2 = ECFP4/BCFP4)
    n_bits=2048,             # Bit vector length (power of 2)
    use_counts=False         # Count fingerprints (default: binary)
)
```

**Methods**:
- **`generate_sparse(mol)`**: Generate sparse fingerprint (dict of key → count)
- **`generate_bytes(mol)`**: Generate a folded binary fingerprint as bytes (chemfp native form)
- **`generate_count_buffer(mol, layout)`**: Generate sorted packed count features in C++
- **`generate_chemfp_count_fingerprint(mol, layout)`**: Generate a chemfp `CountFingerprint`
- **`generate_basic(mol)`**: Generate folded dense fingerprint (np.array)
- **`sortslice_fit(sparse_list, train_idx, top_k)`**: Select top-K features from training set
  - Returns: `(vocab, key2col)` - list of selected keys and mapping dict
- **`sortslice_transform(sparse_list, indices, key2col, use_counts, add_oov_bucket)`**: Transform to dense matrix
  - Returns: `np.array` of shape `(n_molecules, len(vocab) [+1 if OOV])`

## Performance Benchmarks

On a dataset of 10,000 molecules (Intel i9, 12 cores):

| Configuration | Speed | Notes |
|---------------|-------|-------|
| ECFP (xxhash, 2048 bits) | ~15,000 mol/sec | Fastest |
| BCFP (xxhash, 2048 bits) | ~12,000 mol/sec | Slightly slower |
| ECFP (blake3, 2048 bits) | ~13,000 mol/sec | Crypto-grade |
| ECFP+BCFP concat | ~8,000 mol/sec | Best accuracy |
| Sort&Slice (512 from 2048) | ~20,000 mol/sec | Reduced dim |

**10-100x faster than pure Python/RDKit implementations.**

## Use Cases

### Drug Discovery
- Virtual screening (Tanimoto similarity)
- QSAR/QSPR modeling
- Lead optimization

### Cheminformatics
- Molecular similarity search
- Structure-activity relationship (SAR) analysis
- Clustering and diversity analysis

### Machine Learning
- Feature extraction for ML models
- Multi-task property prediction
- Transfer learning in chemistry

## Best Practices

1. **ECFP+BCFP Concatenation**: Often improves ML model performance by 2-5%
2. **Sort&Slice**: Use `top_k=2048` for good balance of speed/accuracy
3. **OOV Bucket**: Essential for production models with distribution shift
4. **Radius Selection**: 
   - `radius=1` (ECFP2): Small molecules, fragments
   - `radius=2` (ECFP4): General purpose, most common
   - `radius=3` (ECFP6): Large molecules, more specificity
5. **Hash Function**: `xxhash` for speed, `blake3` for reproducibility

## Citation

If you use BCFP in your research, please cite:

```bibtex
@article{godin2025bcfp,
  title={Bond-Centered Molecular Fingerprint Derivatives: A BBBP Dataset Study},
  author={Godin, Guillaume},
  journal={arXiv preprint arXiv:2510.04837},
  year={2025},
  url={https://arxiv.org/abs/2510.04837}
}
```

**Paper**: [Bond-Centered Molecular Fingerprint Derivatives: A BBBP Dataset Study](https://arxiv.org/abs/2510.04837)  
**Code**: [https://github.com/osmoai/bcfp](https://github.com/osmoai/bcfp)

## Documentation

### Core Guides
- **[Folding Benchmark](docs/binning_benchmark.md)** - Why Sort&Slice + OOV is the default, and how the folding schemes compare
- **[BCFP vs ECFP](docs/bcfp_vs_ecfp.md)** - What the bond-centred half contributes, on BBBP and Biogen ADME
- **[Model Persistence Guide](docs/MODEL_PERSISTENCE_GUIDE.md)** - Complete guide for save/load, training, and inference
- **[Hash Function Compatibility](docs/HASH_FUNCTION_COMPATIBILITY.md)** - ⚠️ **CRITICAL** - Understanding hash functions and model compatibility
- **Examples**: See `examples/` directory for complete workflows

### Key Concepts

#### Hash Function Compatibility ⚠️ **IMPORTANT**

**CRITICAL**: The hash function (`rdkit_native`, `xxhash`, `blake3`) determines how molecular features are generated. Different hash functions produce **completely different features** (0% overlap).

```python
from bcfp import FingerprintModel

# Training: Choose and save hash function
model = FingerprintModel(hash_func='xxhash')  # ← Hash locked to model
model.fit_transform(train_smiles, train_idx)
model.save('model.pkl')

# Inference: Load with correct hash automatically
model_loaded = FingerprintModel.load('model.pkl')  # ✅ Correct hash restored
X_new = model_loaded.transform(new_smiles)

# Verify hash compatibility
result = model_loaded.verify_hash_compatibility()
print(result['message'])  # '✅ Hash function xxhash verified!'
```

**See [Hash Function Compatibility Guide](docs/HASH_FUNCTION_COMPATIBILITY.md) for critical details.**

#### Model Persistence

The `FingerprintModel` class handles complete training → save → load → inference workflows:

- **Saves**: Configuration, hash function, Sort&Slice vocabulary, key mappings
- **Loads**: Restores complete model state for 100% inference
- **Validates**: Ensures model integrity and hash compatibility

**See [Model Persistence Guide](docs/MODEL_PERSISTENCE_GUIDE.md) for complete documentation.**

## License

This project is licensed under the BSD 3-Clause License - see the [LICENSE](LICENSE) file for details.

Copyright (c) 2025, Guillaume GODIN Osmo labs pbc. All rights reserved.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments

- **Author**: Guillaume GODIN (Osmo labs pbc)
- Built on RDKit for molecular structure handling
- Hash implementations: xxHash, BLAKE3, RDKit native
- Uses nanobind for Python-C++ interoperability

## Support

- **Issues**: [GitHub Issues](https://github.com/osmoai/bcfp/issues)
- **Documentation**: See `examples/` directory and this README
- **Contact**: Open an issue for questions or bug reports

---

**BCFP** - Production-grade molecular fingerprints for modern drug discovery.

Developed by Guillaume GODIN @ Osmo labs pbc.

