"""
BCFP - Bond-Centered Fingerprints for Molecular Machine Learning

A high-performance Python library for generating molecular fingerprints with:
- Multiple hash functions (rdkit_native, xxhash, blake3)
- ECFP (Morgan/Atom-centered) and BCFP (Bond-centered) fingerprints
- Sort&Slice feature selection with OOV handling
- C++ implementation for maximum speed

Two API layers:
    * BCFPEnhanced   — scikit-learn-style fit/transform with Sort&Slice + OOV (recommended);
    * FingerprintGenerator — low-level: generate_sparse() + static sortslice_fit/sortslice_transform.

Example (sklearn-style):
    >>> from bcfp import BCFPEnhanced
    >>> gen = BCFPEnhanced(fp_type='ecfp', radius=2, n_bits=2048, top_k=512, include_oov=True)
    >>> X_train = gen.fit_transform(train_smiles)   # learn top-512 vocab, dense (n, 513)
    >>> X_test  = gen.transform(test_smiles)         # apply fitted vocab + OOV bucket

Example (low-level):
    >>> from bcfp import FingerprintGenerator
    >>> from rdkit import Chem
    >>> gen = FingerprintGenerator('rdkit_native', 'ecfp', radius=2)
    >>> sparse = [gen.generate_sparse(Chem.MolFromSmiles(s)) for s in smiles]
    >>> X = np.array([gen.fold_sparse_to_dense(s) for s in sparse])  # folded to n_bits
"""

__version__ = '2.0.0'
__author__ = 'Guillaume Godin'

from .fingerprints import (
    FingerprintGenerator,
    save_sortslice_vocab,
    load_sortslice_vocab,
    combine_fingerprints
)
from .enhanced import BCFPEnhanced
from .model_persistence import FingerprintModel
from . import utils

__all__ = [
    'BCFPEnhanced',
    'FingerprintGenerator',
    'FingerprintModel',
    'save_sortslice_vocab',
    'load_sortslice_vocab',
    'combine_fingerprints',
    'utils',
]

