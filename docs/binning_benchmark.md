# Folding benchmark: Sort&Slice + OOV vs hash folding

This benchmark asks how 67k sparse ECFP/BCFP keys should be turned into a fixed-width feature vector. It explains why `BCFPEnhanced` now defaults to **Sort&Slice + one OOV bucket**, and why `fold="remix"` exists.

## Setup

- **Data:** BBBP, 4092 parseable molecules (`smiles`, `p_np`).
- **Features:** R=2 count ECFP (`rdkit_native`, 21,702 keys) + BCFP (`rdkit_native`, 45,307 keys) = **67,009 exact keys**, ~103 per molecule. ECFP and BCFP keys are tagged so they never merge. At 2048 columns that is ~33 keys per column.
- **Protocol:** 10 seeds × `StratifiedKFold(10)` = 100 paired splits. XGBoost `hist`, 300 trees, lr 0.1, depth 6, subsample/colsample 0.8, no early stopping. Every vocabulary is fitted on the training fold only.
- **Null:** 10 random salts of an ideal hash, `splitmix64(key_id ^ salt) % W`. Their spread is the "which hash function" noise.
- **Script:** [`benchmarks/binning_bbbp.py`](../benchmarks/binning_bbbp.py) (`--data bbbp.csv`; the full grid takes ~24 min on a 14-core laptop; `--oov-sweep` ~4 min).

## Results at 2048 columns

**Rows:** folding scheme. **Columns:** mean over the 100 splits (and over draws, where a scheme has several). **Δ:** paired per-split difference from the random-hash reference row, with 95% CI. AUROC/AUPRC are on a 0–1 scale.

| scheme | what goes in the columns | AUROC | AUPRC | ΔAUROC vs random hash |
|---|---|---|---|---|
| **Sort&Slice + 1 OOV bucket** (new default) | top 2047 training keys + 1 column summing all other keys | **0.8522** | **0.9104** | **+0.0145** [+0.0131, +0.0158] |
| exact keys, no folding (= MPHF) | all ~60k training keys, one column each | 0.8519 | 0.9101 | +0.0142 [+0.0131, +0.0153] |
| Sort&Slice | top 2048 training keys, rest dropped | 0.8511 | 0.9102 | +0.0134 [+0.0122, +0.0145] |
| signed hash | ±count into 1 random bin | 0.8390 | 0.9001 | +0.0013 [+0.0007, +0.0019] |
| **random hash (reference, 10 salts)** | `splitmix64(key ^ salt) % 2048` | 0.8377 | 0.8996 | 0 (SD over salts 0.0016) |
| top 1792 + 256 hashed tail bins | frequent keys exact, rare keys hashed | 0.8353 | 0.8987 | −0.0024 [−0.0031, −0.0016] |
| native keys `% 2048` (old default) | ECFP and BCFP keys `% W` in one space | 0.8346 | 0.8976 | −0.0031 [−0.0046, −0.0016] |
| native keys, ECFP/BCFP halves | `% 1024` each, concatenated | 0.8336 | 0.8966 | −0.0041 [−0.0054, −0.0027] |
| two hashes per key | count added to 2 random bins | 0.8292 | 0.8932 | −0.0085 [−0.0092, −0.0078] |

Same picture at the other widths. **ΔAUROC of Sort&Slice + OOV vs the random-hash mean:**

| width | random hash AUROC | Sort&Slice + OOV AUROC | Δ |
|---|---|---|---|
| 1024 | 0.8296 | 0.8455 | +0.0159 |
| 2048 | 0.8377 | 0.8522 | +0.0145 |
| 4096 | 0.8434 | **0.8537** | +0.0103 (beats exact keys, 0.8519) |

## One OOV bucket, or an OOV array?

A natural extension is to spread the out-of-vocabulary keys over a small hashed array, for example 5% of the width, instead of summing them into one column. **Tested at 2048: every extra OOV column makes results worse.**

**Rows:** OOV columns (the vocabulary shrinks by the same number so the width stays 2048). **Δ:** paired difference from the single bucket, mean over 5 hash salts × 100 splits, with 95% CI.

| OOV columns | AUROC | AUPRC | ΔAUROC vs 1 bucket | ΔAUPRC vs 1 bucket |
|---|---|---|---|---|
| **1 (default)** | **0.8522** | **0.9104** | reference | reference |
| 20 (1%) | 0.8417 | 0.9038 | −0.0104 ± 0.0009 | −0.0066 ± 0.0007 |
| 41 (2%) | 0.8386 | 0.9011 | −0.0135 ± 0.0010 | −0.0092 ± 0.0008 |
| 102 (5%) | 0.8362 | 0.8995 | −0.0159 ± 0.0011 | −0.0108 ± 0.0009 |
| 256 (12.5%) | 0.8353 | 0.8987 | −0.0169 | −0.0117 |

With 102 OOV columns, the rare keys carry ~30% of each molecule's counts and land in ~33 columns per molecule. Each column is a sum of hundreds of unrelated rare keys. Gradient-boosted trees split on these noisy, size-correlated columns and overfit. One bucket keeps the useful "how much uncommon chemistry is there" signal without that noise. (With a single hashed OOV column, the code reproduces `ss_oov` exactly, which rules out an implementation artifact.)

## Hash choice and remixing

- The hash function itself is not where the gains are. Across random salts of the same features, the AUROC range at 2048 is about 0.005. The named hashes (`rdkit_native`, `xxhash`, `blake3`) all fall inside that spread when they are compared on the same features.
- **Native keys folded with plain `% W` sit below every random salt** (−0.003 AUROC at 2048). RDKit-style boost-combined keys are not uniform in their low bits. If you need a stateless fold with no vocabulary to store, use `fold="remix"`, which applies `splitmix64(key ^ fold_salt) % n_bits`. Keep ECFP and BCFP in one shared space rather than splitting the width into halves.

## Recommendations

1. Default: `BCFPEnhanced(n_bits=2048)` → Sort&Slice top-2047 + 1 OOV bucket (call `fit` on training data).
2. Stateless fallback: `BCFPEnhanced(top_k=None, fold="remix")`.
3. Avoid: hashed OOV arrays, multi-hash, and separate ECFP/BCFP halves when folding.

These results are from one dataset (BBBP) with one model (XGBoost). Re-run `benchmarks/binning_bbbp.py` on your own data before relying on the exact numbers.
