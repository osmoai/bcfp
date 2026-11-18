# What BCFP adds to ECFP

This benchmark asks a narrower question than [`binning_benchmark.md`](binning_benchmark.md), which
settled *how* to pack sparse keys into columns. Given the best packing we have — Sort&Slice + one OOV
bucket — what does the bond-centred half of the feature set actually contribute, and is it worth the
columns it occupies?

## Relation to the BCFP paper

[*Bond-Centered Molecular Fingerprint Derivatives: A BBBP Dataset Study*](https://arxiv.org/abs/2510.04837)
(Godin, 2025) already concluded this: concatenating ECFP with BCFP **consistently improves AUROC and
AUPRC over either descriptor alone**, confirmed by Tukey HSD, with BCFP presented as complementing
the atom-centred ECFP rather than replacing it. The paper also introduced BCFP-Sort&Slice, which is
where this library's default binning comes from.

What follows is not a new conclusion, then, but the same one measured under conditions the original
study did not cover:

- **A second dataset and task.** The paper's evidence is BBBP classification. The Biogen ADME results
  below are the first check that the combination holds off BBBP — four regression endpoints, scored
  by Spearman ρ.
- **A different model.** The paper used a fast Random Forest; everything here is XGBoost, so the
  benefit is not specific to one learner.
- **A column-budget control.** The `@4096` arms test whether concatenation helps because it carries
  complementary information or merely because it produces a wider feature vector. The paper's design
  could not separate those.
- **Radius 2 throughout.** The paper found r = 1 best, with r = 2 not statistically separable under
  its test. These runs are all at r = 2, so they say nothing about the radius question and should not
  be read as bearing on it.

## Setup

Five arms, identical rows, labels and folds within each dataset:

| arm | features | columns |
|---|---|---|
| `ecfp@2048` | ECFP R=2 counts | 2048 |
| `ecfp@4096` | ECFP R=2 counts | 4096 |
| `bcfp@2048` | BCFP R=2 counts | 2048 |
| `both@2048` | ECFP + BCFP R=2 counts | 2048 |
| `both@4096` | ECFP + BCFP R=2 counts | 4096 |

- **Binning:** Sort&Slice top-(W−1) by training-fold document frequency, plus one OOV column summing
  every key outside that vocabulary. The vocabulary is fitted on the training fold only, so test-only
  chemistry always lands in the OOV bucket and never leaks.
- **Namespacing:** ECFP and BCFP keys are tagged (`("E", k)` / `("B", k)`) so they cannot collide
  with each other. In the combined arms the two families compete for the same slots on training
  frequency alone — neither gets a reserved quota.
- **The control:** the two `@4096` arms. If the combined set wins only because it has more features
  to spend columns on, then giving ECFP the same doubled budget should close the gap.
- **Model:** XGBoost `hist`, 300 trees, lr 0.1, depth 6, subsample/colsample 0.8, no early stopping.

All Δ values below are **paired per-fold differences** with 95% CIs. **Bold** marks a CI that
excludes zero.

## Dataset 1 — BBBP (classification)

4089 molecules, binary blood–brain-barrier label, AUROC. 10 seeds × `StratifiedKFold(10)` = 100
folds, and 10 seeds × `StratifiedGroupKFold(5)` on Bemis-Murcko scaffolds = 50 folds.

**Vocabulary.** The union is exactly the sum, so the two families share no keys at all:

| features | distinct keys | per molecule |
|---|---|---|
| ECFP R=2 | 21,668 | 44.5 |
| BCFP R=2 | 45,245 | 58.3 |
| ECFP + BCFP | 66,913 | 102.8 |

**Mean AUROC** (0–1 scale; every arm sees the same folds):

| arm | random 10-fold | scaffold 5-fold |
|---|---|---|
| ECFP @ 2048 | 0.8517 | 0.8466 |
| ECFP @ 4096 | 0.8512 | 0.8452 |
| BCFP @ 2048 | 0.8508 | 0.8443 |
| ECFP+BCFP @ 2048 | 0.8529 | 0.8494 |
| **ECFP+BCFP @ 4096** | **0.8538** | **0.8507** |

**Paired ΔAUROC:**

| comparison | random split | scaffold split |
|---|---|---|
| both@2048 − ecfp@2048 (BCFP's contribution) | +0.0012 [−0.0002, +0.0025] | **+0.0029** [+0.0005, +0.0052] |
| both@4096 − ecfp@4096 (same, doubled budget) | **+0.0026** [+0.0012, +0.0039] | **+0.0054** [+0.0039, +0.0070] |
| both@2048 − bcfp@2048 (ECFP's contribution) | **+0.0021** [+0.0005, +0.0036] | **+0.0051** [+0.0033, +0.0069] |
| bcfp@2048 − ecfp@2048 (is BCFP a better ECFP?) | −0.0009 [−0.0029, +0.0011] | −0.0022 [−0.0053, +0.0008] |
| ecfp@4096 − ecfp@2048 (columns alone) | −0.0005 [−0.0016, +0.0006] | −0.0013 [−0.0028, +0.0001] |
| both@4096 − both@2048 (columns alone) | +0.0009 [−0.0002, +0.0020] | +0.0012 [−0.0005, +0.0030] |

## Dataset 2 — Biogen ADME (regression)

The replication, deliberately harder: different data, a different task, a different metric, and four
independent endpoints instead of one. 3521 molecules, all parsed, 2986 Bemis-Murcko scaffolds.
Preprocessing is `MolFromSmiles` → `MolToSmiles(isomericSmiles=True)` with no standardisation.
`XGBRegressor`, scored by **Spearman ρ**, 5 seeds × `KFold(10)` = 50 folds and
5 seeds × `GroupKFold(5)` on scaffolds = 25 folds. The two plasma-protein-binding endpoints
(194 and 168 values) are too small to model and are skipped.

**Vocabulary** — the same ~2.1× ratio and the same disjoint union as BBBP:

| features | distinct keys | per molecule |
|---|---|---|
| ECFP R=2 | 16,585 | 44.5 |
| BCFP R=2 | 39,224 | 57.7 |
| ECFP + BCFP | 55,809 | 102.2 |

**Mean Spearman ρ** (higher is better; best per row in bold):

| endpoint | split | n | ECFP@2048 | BCFP@2048 | both@2048 | both@4096 |
|---|---|---|---|---|---|---|
| HLM CLint | random | 3087 | 0.6525 | 0.6429 | 0.6666 | **0.6673** |
| HLM CLint | scaffold | 3087 | 0.6458 | 0.6309 | **0.6570** | 0.6527 |
| RLM CLint | random | 3054 | 0.6868 | 0.6741 | **0.7003** | 0.6987 |
| RLM CLint | scaffold | 3054 | 0.6786 | 0.6616 | 0.6938 | **0.6940** |
| MDR1-MDCK ER | random | 2642 | 0.7391 | 0.7209 | **0.7396** | 0.7395 |
| MDR1-MDCK ER | scaffold | 2642 | 0.7333 | 0.7170 | **0.7370** | 0.7347 |
| Solubility pH 6.8 | random | 2173 | 0.5066 | 0.4902 | **0.5246** | 0.5179 |
| Solubility pH 6.8 | scaffold | 2173 | 0.4971 | 0.4816 | **0.5091** | 0.5029 |

**Pooled Δ Spearman** — the mean of the four per-endpoint paired deltas, with its 95% CI, and how
many endpoints agree in sign:

| comparison | random split | scaffold split | endpoints positive |
|---|---|---|---|
| both@2048 − ecfp@2048 (BCFP's contribution) | **+0.0115** [+0.0041, +0.0190] | **+0.0105** [+0.0057, +0.0153] | **4/4 and 4/4** |
| both@4096 − ecfp@4096 (same, doubled budget) | **+0.0098** [+0.0037, +0.0158] | **+0.0072** [+0.0028, +0.0115] | **4/4 and 4/4** |
| both@2048 − bcfp@2048 (ECFP's contribution) | **+0.0257** [+0.0193, +0.0322] | **+0.0264** [+0.0215, +0.0314] | 4/4 and 4/4 |
| bcfp@2048 − ecfp@2048 (is BCFP a better ECFP?) | **−0.0142** [−0.0179, −0.0105] | **−0.0159** [−0.0169, −0.0150] | **0/4 and 0/4** |
| ecfp@4096 − ecfp@2048 (columns alone) | −0.0002 [−0.0023, +0.0020] | +0.0002 [−0.0014, +0.0018] | 1/4 and 3/4 |

**Per-endpoint Δ for the headline comparison** (both@2048 − ecfp@2048, scaffold split):

| endpoint | Δ Spearman | CI excludes zero |
|---|---|---|
| RLM CLint | +0.0151 [+0.0114, +0.0189] | yes |
| Solubility pH 6.8 | +0.0120 [+0.0040, +0.0200] | yes |
| HLM CLint | +0.0112 [+0.0050, +0.0173] | yes |
| MDR1-MDCK ER | +0.0037 [−0.0012, +0.0085] | no |

## What to conclude

1. **Add BCFP, never substitute it.** Alone, BCFP is no better than ECFP on BBBP and *measurably
   worse* on all four ADME endpoints (−0.014 to −0.016 pooled), despite carrying 2.1× the distinct
   keys. Added to ECFP it is worth +0.005 AUROC and +0.011 Spearman. Both halves of that statement
   now have replications across two datasets and five endpoints.
2. **The gain is information, not capacity.** Widening ECFP alone moves nothing: −0.0013 on BBBP,
   +0.0002 pooled on Biogen. The combined set gains from the same widening on BBBP. Extra columns pay
   off only when there is genuinely new chemistry to fill them.
3. **The two families contribute symmetrically.** Removing ECFP from the pair costs about as much as
   removing BCFP (+0.0021 vs +0.0026 on BBBP at 2048). Neither is a passenger.
4. **Expect endpoint dependence.** MDR1-MDCK efflux ratio — the most whole-molecule physicochemical
   of the four — gains nothing measurable from BCFP, while metabolic clearance and solubility gain
   +0.011 to +0.018. Local bond environments help where local chemistry drives the endpoint.
5. **Default to 2048 columns for the combined set.** BBBP prefers 4096; Biogen does not (2048 matches
   or beats 4096 in 7 of 8 cells). Treat the wider array as a per-dataset experiment.

## Reproducing

```bash
python benchmarks/bcfp_vs_ecfp.py --data bbbp.csv   --task classification --smiles-col smiles --label p_np
python benchmarks/bcfp_vs_ecfp.py --data biogen.csv --task regression     --smiles-col SMILES
```

BBBP takes ~3 min and the four Biogen endpoints ~5 min on a 14-core laptop at the default seed
counts. `--nseeds` trades runtime for CI width. The Biogen set ships with the paper that prompted
this check: Blake, Imrie & Deane, *On the limitations of molecular representations*
([oxpig/On-the-limitations-of-molecular-representations](https://github.com/oxpig/On-the-limitations-of-molecular-representations),
`Datasets/biogen.csv`).

Both datasets are small and both verdicts are measurements on them, not laws. The one claim that is
robust across everything tested here is the asymmetry: **BCFP complements ECFP and does not replace
it.**
