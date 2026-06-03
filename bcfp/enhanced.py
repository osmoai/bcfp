"""bcfp.enhanced — scikit-learn-style fingerprint transformer.

`BCFPEnhanced` wraps the low-level :class:`bcfp.fingerprints.FingerprintGenerator`
(`generate_sparse` + `sortslice_fit`/`sortslice_transform`) behind the familiar
``fit`` / ``transform`` / ``fit_transform`` API, with optional Sort&Slice feature selection
(`top_k`) and an out-of-vocabulary bucket (`include_oov`).

    inference:  SMILES ──transform──▶ dense fingerprint matrix
    selection:  fit() learns the top-k vocabulary on training molecules; transform() applies it
                (unseen bits fall into the OOV bucket when include_oov=True)

Example
-------
>>> from bcfp import BCFPEnhanced
>>> gen = BCFPEnhanced(fp_type='ecfp', radius=2, n_bits=2048, top_k=512, include_oov=True)
>>> X_train = gen.fit_transform(train_smiles)   # (n_train, 513)
>>> X_test  = gen.transform(test_smiles)        # uses the fitted vocabulary + OOV bucket
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np
from rdkit import Chem

from .fingerprints import FingerprintGenerator

__all__ = ["BCFPEnhanced"]


class BCFPEnhanced:
    """ECFP/BCFP fingerprints with optional Sort&Slice selection and OOV handling.

    Parameters
    ----------
    fp_type : {'ecfp', 'bcfp'}
        Atom-centered (ECFP) or bond-centered (BCFP) fingerprint.
    radius : int
        Morgan radius.
    n_bits : int
        Folded width when ``top_k`` is None (baseline mode).
    hash_func : {'rdkit_native', 'xxhash', 'blake3'}
        Hash backend (xxhash is header-only; blake3 is vendored — both built in).
    top_k : int or None
        If set, select the top-k most frequent training keys (Sort&Slice). If None, fold to
        ``n_bits`` and do no vocabulary selection.
    include_oov : bool
        If True (and ``top_k`` set), append one out-of-vocabulary bucket column that counts
        test-time bits not in the fitted vocabulary.
    use_counts, use_chirality : bool
        Forwarded to the fingerprint generator.
    sort_by : {'df', 'tf'}
        Rank keys by document frequency or total frequency during Sort&Slice.
    min_df : int
        Minimum document frequency for a key to be eligible.

    Attributes
    ----------
    vocab_ , key2col_ , selected_indices_
        The fitted Sort&Slice vocabulary (None in baseline mode).
    """

    def __init__(
        self,
        fp_type: str = "ecfp",
        radius: int = 2,
        n_bits: int = 2048,
        hash_func: str = "rdkit_native",
        top_k: Optional[int] = None,
        include_oov: bool = False,
        use_counts: bool = True,
        use_chirality: bool = True,
        sort_by: str = "df",
        min_df: int = 2,
    ):
        self.gen = FingerprintGenerator(
            hash_func, fp_type, radius=radius,
            use_counts=use_counts, use_chirality=use_chirality, n_bits=n_bits,
        )
        self.fp_type = fp_type
        self.radius = radius
        self.n_bits = n_bits
        self.hash_func = hash_func
        self.top_k = top_k
        self.include_oov = include_oov
        self.use_counts = use_counts
        self.sort_by = sort_by
        self.min_df = min_df
        self.vocab_ = None
        self.key2col_ = None
        self.selected_indices_ = None
        self._fitted = False

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _check_smiles(smiles: Sequence[str]) -> list:
        if isinstance(smiles, str):
            raise TypeError("smiles must be a sequence of SMILES strings, not a single str")
        smiles = list(smiles)
        if not smiles:
            raise ValueError("smiles is empty")
        return smiles

    def _sparse(self, smiles: Sequence[str]) -> List[dict]:
        # generate_sparse handles mol is None (invalid SMILES) -> {} gracefully
        return [self.gen.generate_sparse(Chem.MolFromSmiles(s)) for s in smiles]

    # ----------------------------------------------------------------------- fit
    def fit(self, smiles: Sequence[str], y=None) -> "BCFPEnhanced":
        """Learn the Sort&Slice vocabulary on training molecules (no-op in baseline mode)."""
        smiles = self._check_smiles(smiles)
        if self.top_k is not None:
            sparse = self._sparse(smiles)
            self.vocab_, self.key2col_ = FingerprintGenerator.sortslice_fit(
                sparse, np.arange(len(sparse)), top_k=self.top_k,
                sort_by=self.sort_by, min_df=self.min_df,
            )
            self.selected_indices_ = self.vocab_
        self._fitted = True
        return self

    # ----------------------------------------------------------------- transform
    def transform(self, smiles: Sequence[str]) -> np.ndarray:
        """SMILES → dense fingerprint matrix (folded, or Sort&Slice + optional OOV)."""
        if self.top_k is not None and not self._fitted:
            raise RuntimeError("call fit() (or fit_transform()) before transform() when top_k is set")
        smiles = self._check_smiles(smiles)
        sparse = self._sparse(smiles)
        if self.top_k is None:
            return np.asarray([self.gen.fold_sparse_to_dense(sp) for sp in sparse])
        return FingerprintGenerator.sortslice_transform(
            sparse, np.arange(len(sparse)), self.key2col_,
            use_counts=self.use_counts, add_oov_bucket=self.include_oov,
        )

    def fit_transform(self, smiles: Sequence[str], y=None) -> np.ndarray:
        """Convenience: ``fit(smiles)`` then ``transform(smiles)``."""
        return self.fit(smiles).transform(smiles)
