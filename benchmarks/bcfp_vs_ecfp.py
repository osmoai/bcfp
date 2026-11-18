#!/usr/bin/env python3
"""What does BCFP add on top of ECFP, and is it worth the columns?

Five arms, identical rows, labels and folds. Everything is binned with Sort&Slice top-(W-1) by
training-fold document frequency plus one OOV column summing every key outside that vocabulary, so
the comparison runs on the packing scheme this library defaults to:

    ecfp@2048   ECFP R=2 counts                2048 columns
    ecfp@4096   ECFP R=2 counts                4096 columns  <- column-budget control
    bcfp@2048   BCFP R=2 counts                2048 columns
    both@2048   ECFP + BCFP R=2 counts         2048 columns  <- same budget as ecfp@2048
    both@4096   ECFP + BCFP R=2 counts         4096 columns  <- same budget as ecfp@4096

ECFP and BCFP keys are namespaced so they cannot collide with each other; in the combined arms the
two families compete for the same slots on training frequency alone, with no reserved quota. The
@4096 arms are the control: if the combined set wins only because it has more features to spread
over columns, doubling ECFP's width should close the gap.

Classification is scored by AUROC, regression by Spearman rho (with RMSE recorded). Every dataset is
run under both a random k-fold and a Bemis-Murcko scaffold-grouped split.

Usage
-----
    python bcfp_vs_ecfp.py --data bbbp.csv   --task classification --smiles-col smiles --label p_np
    python bcfp_vs_ecfp.py --data biogen.csv --task regression     --smiles-col SMILES

With --task regression and no --label, every sufficiently populated numeric column is treated as an
endpoint and the per-endpoint deltas are pooled. Results are written to --out (CSV).

See docs/bcfp_vs_ecfp.md for the published tables.
"""
import argparse
import os
from collections import Counter

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold
from scipy import sparse as sp
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, KFold, StratifiedGroupKFold, StratifiedKFold

RDLogger.DisableLog("rdApp.*")

try:
    import _bcfp
except ImportError as exc:  # pragma: no cover
    raise SystemExit("this benchmark needs the compiled _bcfp extension: pip install -e .") from exc

ARMS = [("ecfp", 2048), ("ecfp", 4096), ("bcfp", 2048), ("both", 2048), ("both", 4096)]
PAIRS = [
    ("both@2048", "ecfp@2048", "BCFP's contribution"),
    ("both@4096", "ecfp@4096", "BCFP's contribution, doubled budget"),
    ("both@2048", "bcfp@2048", "ECFP's contribution"),
    ("bcfp@2048", "ecfp@2048", "is BCFP a better ECFP?"),
    ("ecfp@4096", "ecfp@2048", "columns alone"),
    ("both@4096", "both@2048", "columns alone"),
]


def sparse_keys(mols, kind, radius):
    """Sparse count dicts, with ECFP and BCFP keys namespaced so they never merge."""
    out = []
    for mol in mols:
        feats = {}
        if kind in ("ecfp", "both"):
            feats.update({("E", k): c for k, c in
                          dict(_bcfp.get_morgan_fingerprint(mol, radius=radius)).items()})
        if kind in ("bcfp", "both"):
            feats.update({("B", k): c for k, c in
                          dict(_bcfp.get_bond_morgan_fingerprint_native(mol, radius=radius)).items()})
        out.append(feats)
    return out


def ss_oov_matrix(sparse_list, train_idx, width):
    """Sort&Slice top-(width-1) training keys, everything else summed into the last column."""
    doc_freq = Counter()
    for i in train_idx:
        doc_freq.update(sparse_list[i].keys())
    vocab = {k: j for j, (k, _) in enumerate(
        sorted(doc_freq.items(), key=lambda kv: (-kv[1], repr(kv[0])))[:width - 1])}
    rows, cols, vals = [], [], []
    for i, feats in enumerate(sparse_list):
        oov = 0
        for key, count in feats.items():
            col = vocab.get(key)
            if col is None:
                oov += count
            else:
                rows.append(i); cols.append(col); vals.append(count)
        rows.append(i); cols.append(width - 1); vals.append(oov)
    return sp.csr_matrix((np.array(vals, np.float32), (rows, cols)),
                         shape=(len(sparse_list), width))


def run_arm(endpoint, kind, width, split_name, feats, y, splits, task):
    import xgboost as xgb
    out = []
    for seed, fold, tr, te in splits:
        X = ss_oov_matrix(feats, tr, width)
        common = dict(n_estimators=300, learning_rate=0.1, max_depth=6, subsample=0.8,
                      colsample_bytree=0.8, tree_method="hist", n_jobs=2,
                      random_state=seed, verbosity=0)
        if task == "classification":
            model = xgb.XGBClassifier(**common)
            model.fit(X[tr], y[tr])
            pred = model.predict_proba(X[te])[:, 1]
            score = roc_auc_score(y[te], pred)
        else:
            model = xgb.XGBRegressor(**common)
            model.fit(X[tr], y[tr])
            pred = model.predict(X[te])
            score = spearmanr(y[te], pred).statistic
        out.append(dict(endpoint=endpoint, arm=f"{kind}@{width}", split=split_name,
                        seed=seed, fold=fold, score=float(score),
                        rmse=float(np.sqrt(np.mean((y[te] - pred) ** 2)))))
    return out


def make_splits(y, groups, nseeds, task):
    """Random k-fold and scaffold-grouped k-fold, stratified when the task is classification."""
    random_splits, scaffold_splits = [], []
    for seed in range(nseeds):
        if task == "classification":
            kf = StratifiedKFold(10, shuffle=True, random_state=seed).split(np.zeros(len(y)), y)
            gkf = StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(
                np.zeros(len(y)), y, groups=groups)
            random_splits += [(seed, f, tr, te) for f, (tr, te) in enumerate(kf)]
            scaffold_splits += [(seed, f, tr, te) for f, (tr, te) in enumerate(gkf)]
        else:
            kf = KFold(10, shuffle=True, random_state=seed).split(y)
            random_splits += [(seed, f, tr, te) for f, (tr, te) in enumerate(kf)]
            # GroupKFold is deterministic, so shuffle the rows to vary folds across seeds
            perm = np.random.RandomState(seed).permutation(len(y))
            inv = np.argsort(perm)
            gkf = GroupKFold(5).split(y[perm], groups=groups[perm])
            scaffold_splits += [(seed, f, inv[tr], inv[te]) for f, (tr, te) in enumerate(gkf)]
    return random_splits, scaffold_splits


def report(out, metric_name):
    print("\nmean " + metric_name + ":")
    print(out.pivot_table(index=["endpoint", "split"], columns="arm", values="score")
          .round(4).to_string())

    endpoints = list(dict.fromkeys(out.endpoint))
    print(f"\npaired deltas (same folds), {metric_name}, 95% CI; * = CI excludes zero:")
    pooled = {(a, b): [] for a, b, _ in PAIRS}
    for endpoint in endpoints:
        for split_name in ("random", "scaffold"):
            table = out[(out.endpoint == endpoint) & (out.split == split_name)].pivot_table(
                index=["seed", "fold"], columns="arm", values="score")
            for a, b, _ in PAIRS:
                d = (table[a] - table[b]).dropna()
                mu = d.mean()
                se = d.std(ddof=1) / np.sqrt(len(d))
                star = "*" if abs(mu) > 1.96 * se else " "
                print(f"  {endpoint:16s} {split_name:9s} {a:10s} - {b:10s}: {mu:+.4f} "
                      f"[{mu - 1.96 * se:+.4f}, {mu + 1.96 * se:+.4f}] {star} n={len(d)}")
                pooled[(a, b)].append((split_name, mu))

    if len(endpoints) > 1:
        print(f"\npooled across the {len(endpoints)} endpoints (mean of per-endpoint deltas):")
        for (a, b), vals in pooled.items():
            for split_name in ("random", "scaffold"):
                v = [m for s, m in vals if s == split_name]
                mu = np.mean(v)
                se = np.std(v, ddof=1) / np.sqrt(len(v))
                print(f"  {split_name:9s} {a:10s} - {b:10s}: {mu:+.4f} "
                      f"[{mu - 1.96 * se:+.4f}, {mu + 1.96 * se:+.4f}] "
                      f"({sum(1 for x in v if x > 0)}/{len(v)} endpoints positive)")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="CSV with a SMILES column and one or more labels")
    ap.add_argument("--task", choices=("classification", "regression"), default="classification")
    ap.add_argument("--smiles-col", default="smiles")
    ap.add_argument("--label", default=None,
                    help="label column; for regression, omit to use every numeric column")
    ap.add_argument("--min-values", type=int, default=500,
                    help="skip a regression endpoint with fewer measurements (default 500)")
    ap.add_argument("--exclude", nargs="*", default=(),
                    help="column names to never treat as endpoints")
    ap.add_argument("--radius", type=int, default=2)
    ap.add_argument("--nseeds", type=int, default=5)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--out", default="bcfp_vs_ecfp_results.csv")
    args = ap.parse_args()

    raw = pd.read_csv(args.data)
    if args.smiles_col not in raw.columns:
        raise SystemExit(f"no column {args.smiles_col!r}; found {list(raw.columns)}")
    n_total = len(raw)

    mols, canon, keep = [], [], []
    for i, smi in enumerate(raw[args.smiles_col]):
        mol = Chem.MolFromSmiles(str(smi))
        if mol is None:
            continue
        keep.append(i); mols.append(mol)
        canon.append(Chem.MolToSmiles(mol, isomericSmiles=True))
    raw = raw.iloc[keep].reset_index(drop=True)
    print(f"{len(raw)} molecules parsed of {n_total}")

    feats = {k: sparse_keys(mols, k, args.radius) for k in ("ecfp", "bcfp", "both")}
    for kind, vals in feats.items():
        print(f"{kind}: {len(set().union(*[set(d) for d in vals]))} distinct keys, "
              f"{np.mean([len(d) for d in vals]):.1f} per molecule")

    scaffolds = np.array([MurckoScaffold.MurckoScaffoldSmiles(smiles=s, includeChirality=False)
                          for s in canon])
    print(f"{len(set(scaffolds))} Bemis-Murcko scaffolds")

    if args.label:
        endpoints = {args.label: args.label}
    elif args.task == "regression":
        # Float columns only: integer columns are almost always identifiers, not measurements.
        candidates = [c for c in raw.select_dtypes("float").columns if c not in args.exclude]
        endpoints = {c: c for c in candidates if raw[c].notna().sum() >= args.min_values}
        skipped = [f"{c} (n={int(raw[c].notna().sum())})"
                   for c in candidates if c not in endpoints]
        if skipped:
            print(f"skipping endpoints below --min-values {args.min_values}: {', '.join(skipped)}")
        if not endpoints:
            raise SystemExit(f"no float column has >= {args.min_values} values; "
                             f"pass --label explicitly")
        print(f"endpoints: {', '.join(endpoints)}")
    else:
        raise SystemExit("--label is required for classification")

    jobs = []
    for name, col in endpoints.items():
        idx = np.where(raw[col].notna().values)[0]
        y = raw[col].values[idx].astype(np.float64)
        if args.task == "classification":
            y = y.astype(int)
        per_arm = {k: [feats[k][i] for i in idx] for k in feats}
        random_splits, scaffold_splits = make_splits(y, scaffolds[idx], args.nseeds, args.task)
        print(f"{name}: n={len(y)}, {len(set(scaffolds[idx]))} scaffolds")
        for kind, width in ARMS:
            for split_name, splits in (("random", random_splits), ("scaffold", scaffold_splits)):
                jobs.append((name, kind, width, split_name, per_arm[kind], y, splits, args.task))

    res = Parallel(n_jobs=args.jobs, verbose=5)(delayed(run_arm)(*j) for j in jobs)
    out = pd.DataFrame([r for rs in res for r in rs])
    out.to_csv(args.out, index=False)
    report(out, "AUROC" if args.task == "classification" else "Spearman rho")
    print(f"\nwrote {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
