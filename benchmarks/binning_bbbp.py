#!/usr/bin/env python3
"""Folding-scheme benchmark: R=2 ECFP+BCFP on BBBP (heavy load: ~67k keys). See docs/binning_benchmark.md.

Every config is X_full @ P, where X_full (N x n_keys) holds the exact count keys
(RDKit-native ECFP R=2 + _bcfp native BCFP R=2, tagged so they never merge) and
P (n_keys x W) is the binning. P is either fixed or fitted on the training fold only.

Configs per width W:
  salt_i         splitmix64(id ^ salt_i) % W                       (null, 10 salts)
  native_sum     ECFP key % W + BCFP key % W in one W space         (current practice)
  native_concat  ECFP key % (W/2) | BCFP key % (W/2)
  sortslice      top-W train document-frequency keys, rest dropped
  ss_oov         top-(W-1) + 1 bucket summing all other keys
  hybrid_T       top-(W-T) exact + every other key hashed into T bins (T=W/8; 5 salts; --oov-sweep: T=1%/2%/5% of W)
  twohash        each key adds its count to 2 salted bins (5 salts)
  signed         +-count into 1 salted bin, sign from a second hash (5 salts)
plus exact_train (all keys seen in train, = MPHF), width-free.
Protocol as salt_null.py: 10 seeds x 10-fold, XGB hist 300 trees lr 0.1, no early stopping.
"""
import os, time, pickle, argparse
import numpy as np, pandas as pd
from scipy import sparse
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score
from joblib import Parallel, delayed



def mix(x, salt):
    """vectorised splitmix64 of uint64 array x ^ salt"""
    with np.errstate(over="ignore"):
        z = (x ^ np.uint64(salt)) + np.uint64(0x9E3779B97F4A7C15)
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        return z ^ (z >> np.uint64(31))


def build(data, cache):
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")
    import _bcfp
    df = pd.read_csv(data)  # columns: smiles, p_np
    ms = [Chem.MolFromSmiles(s) for s in df.smiles]
    keep = [i for i, m in enumerate(ms) if m is not None]
    ms = [ms[i] for i in keep]
    y = df.p_np.values[keep]
    ids, raw, src = {}, [], []
    rows, cols, vals = [], [], []
    for i, m in enumerate(ms):
        for tag, d in ((0, _bcfp.get_morgan_fingerprint(m, radius=2)),
                       (1, _bcfp.get_bond_morgan_fingerprint_native(m, radius=2))):
            for k, c in dict(d).items():
                j = ids.get((tag, k))
                if j is None:
                    j = ids[(tag, k)] = len(ids); raw.append(k); src.append(tag)
                rows.append(i); cols.append(j); vals.append(c)
    X = sparse.csr_matrix((np.array(vals, np.float32), (rows, cols)), shape=(len(ms), len(ids)))
    pickle.dump(dict(X=X, y=y, raw=np.array(raw, np.uint64), src=np.array(src, np.int8)), open(cache, "wb"))


def P_from(cols, W, weights=None, rows=None):
    n = len(cols) if rows is None else None
    rows = np.arange(len(cols)) if rows is None else rows
    w = np.ones(len(cols), np.float32) if weights is None else weights.astype(np.float32)
    return sparse.csr_matrix((w, (rows, cols)), shape=(rows.max() + 1 if n is None else n, W))


def fixed_P(name, W, raw, src, n, salt):
    idx = np.arange(n, dtype=np.uint64)
    if name == "salt":
        return P_from((mix(idx, salt) % np.uint64(W)).astype(np.int64), W)
    if name == "native_sum":
        return P_from((raw % np.uint64(W)).astype(np.int64), W)
    if name == "native_concat":
        h = W // 2
        return P_from((raw % np.uint64(h)).astype(np.int64) + h * src.astype(np.int64), W)
    if name == "twohash":
        c1 = (mix(idx, salt) % np.uint64(W)).astype(np.int64)
        c2 = (mix(idx, salt ^ 0xABCDEF1234567) % np.uint64(W)).astype(np.int64)
        P = P_from(np.r_[c1, c2], W, rows=np.r_[np.arange(n), np.arange(n)])
        P.sum_duplicates(); return P
    if name == "signed":
        c = (mix(idx, salt) % np.uint64(W)).astype(np.int64)
        s = np.where((mix(idx, salt ^ 0x5151) >> np.uint64(63)) == 1, -1.0, 1.0)
        return P_from(c, W, weights=s)
    raise ValueError(name)


def fitted_P(name, W, n, df, salt, frac=0.125):
    order = np.lexsort((np.arange(n), -df))          # by DF desc, stable
    seen = df > 0
    if name == "exact_train":
        keep = np.where(seen)[0]
        cols = np.full(n, -1); cols[keep] = np.arange(len(keep)); Wd = len(keep)
    elif name == "sortslice":
        top = order[:W]; top = top[seen[top]]
        cols = np.full(n, -1); cols[top] = np.arange(len(top)); Wd = W
    elif name == "ss_oov":
        top = order[:W - 1]; top = top[seen[top]]
        cols = np.full(n, W - 1); cols[top] = np.arange(len(top)); Wd = W
    elif name == "hybrid":
        T = max(1, int(round(W * frac))); top = order[:W - T]; top = top[seen[top]]
        idx = np.arange(n, dtype=np.uint64)
        cols = (W - T + (mix(idx, salt) % np.uint64(T))).astype(np.int64)
        cols[top] = np.arange(len(top)); Wd = W
    else:
        raise ValueError(name)
    m = cols >= 0
    return sparse.csr_matrix((np.ones(m.sum(), np.float32), (np.where(m)[0], cols[m])), shape=(n, Wd))


def fit_eval(Xtr, ytr, Xte, yte, seed):
    import xgboost as xgb
    m = xgb.XGBClassifier(n_estimators=300, learning_rate=0.1, max_depth=6, subsample=0.8,
                          colsample_bytree=0.8, tree_method="hist", n_jobs=2,
                          random_state=seed, verbosity=0)
    m.fit(Xtr, ytr)
    p = m.predict_proba(Xte)[:, 1]
    return roc_auc_score(yte, p), average_precision_score(yte, p)


def run(cfg, X, y, raw, src, splits):
    name, W, salt, label = cfg[:4]
    frac = cfg[4] if len(cfg) > 4 else 0.125
    n = X.shape[1]
    P = None if name in ("exact_train", "sortslice", "ss_oov", "hybrid") else fixed_P(name, W, raw, src, n, salt)
    out = []
    for seed, fold, tr, te in splits:
        if P is None:
            df = np.asarray((X[tr] > 0).sum(0)).ravel()
            Pk = fitted_P(name, W, n, df, salt, frac)
        else:
            Pk = P
        Z = (X @ Pk).tocsr()
        a, p = fit_eval(Z[tr], y[tr], Z[te], y[te], seed)
        out.append(dict(config=label, method=name, width=W, seed=seed, fold=fold, auroc=a, auprc=p))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="BBBP csv with columns smiles,p_np")
    ap.add_argument("--workdir", default=".", help="where the key cache and results are written")
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--nseeds", type=int, default=10)
    ap.add_argument("--time-only", action="store_true")
    ap.add_argument("--oov-sweep", action="store_true", help="W=2048: 1 OOV bucket vs hashed OOV arrays of 1/2/5%% of W")
    a = ap.parse_args()
    cache = os.path.join(a.workdir, "keys_r2_ecfp_bcfp.pkl")
    if not os.path.exists(cache):
        build(a.data, cache)
    D = pickle.load(open(cache, "rb"))
    X, y, raw, src = D["X"], D["y"], D["raw"], D["src"]
    print("molecules", X.shape[0], "keys", X.shape[1], "ECFP", int((src == 0).sum()), "BCFP", int((src == 1).sum()),
          "nnz/mol", X.nnz / X.shape[0])
    splits = [(s, f, tr, te) for s in range(a.nseeds)
              for f, (tr, te) in enumerate(StratifiedKFold(10, shuffle=True, random_state=s).split(np.zeros(len(y)), y))]
    S = [int(mix(np.array([1000 + i], np.uint64), 7)[0]) for i in range(10)]
    cfgs = [("exact_train", 0, 0, "exact_train")]
    for W in (1024, 2048, 4096):
        cfgs += [("salt", W, S[i], f"salt{i}") for i in range(10)]
        cfgs += [("native_sum", W, 0, "native_sum"), ("native_concat", W, 0, "native_concat"),
                 ("sortslice", W, 0, "sortslice"), ("ss_oov", W, 0, "ss_oov")]
        for m in ("hybrid", "twohash", "signed"):
            cfgs += [(m, W, S[i], f"{m}{i}") for i in range(5)]
    if a.oov_sweep:
        cfgs = [("ss_oov", 2048, 0, "ss_oov")]
        cfgs += [("hybrid", 2048, S[i], f"oov{int(f*100)}pct_{i}", f) for f in (0.01, 0.02, 0.05) for i in range(5)]
        res = Parallel(n_jobs=a.jobs, verbose=5)(delayed(run)(c, X, y, raw, src, splits) for c in cfgs)
        pd.DataFrame([r for rs in res for r in rs]).to_csv(os.path.join(a.workdir, "oov_sweep_results.csv"), index=False)
        return
    if a.time_only:
        t = time.time(); run(("salt", 2048, S[0], "t"), X, y, raw, src, splits[:3])
        t2 = time.time(); run(("hybrid", 2048, S[0], "t"), X, y, raw, src, splits[:3])
        print("sec/fit salt", (t2 - t) / 3, "hybrid", (time.time() - t2) / 3, "configs", len(cfgs)); return
    t0 = time.time()
    res = Parallel(n_jobs=a.jobs, verbose=5)(delayed(run)(c, X, y, raw, src, splits) for c in cfgs)
    pd.DataFrame([r for rs in res for r in rs]).to_csv(os.path.join(a.workdir, "binning_r2_results.csv"), index=False)
    print("done", (time.time() - t0) / 60, "min")


if __name__ == "__main__":
    main()
