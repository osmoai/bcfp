"""
BCFP - Bond-Centered Fingerprints for Molecular Machine Learning

A high-performance Python library for generating molecular fingerprints with:
- 3 hash functions (rdkit_native, xxhash, blake3)
- ECFP (Morgan/Atom-centered) and BCFP (Bond-centered) fingerprints
- Sort&Slice feature selection with OOV handling
- C++ implementation for maximum speed

Example:
    >>> from bcfp import FingerprintGenerator
    >>> from rdkit import Chem
    >>> 
    >>> # Create generator with XXHash (fastest)
    >>> gen = FingerprintGenerator('xxhash', 'ecfp', radius=2)
    >>> 
    >>> # Generate fingerprints
    >>> mol = Chem.MolFromSmiles('CCO')
    >>> fp = gen.generate_sparse(mol)  # Sparse dict
    >>> X = gen.generate_basic(mol)  # Dense folded vector
    >>> 
    >>> # Sort&Slice with OOV for feature selection
    >>> smiles = ['CCO', 'c1ccccc1', 'CC(=O)O']
    >>> sparse_fps = [gen.generate_sparse(Chem.MolFromSmiles(s)) for s in smiles]
    >>> vocab, key2col = gen.sortslice_fit(sparse_fps, list(range(len(smiles))), top_k=128, include_oov=True)
    >>> X_train = gen.sortslice_transform(sparse_fps, list(range(len(smiles))), key2col, include_oov=True)
"""

__version__ = '2.0.0'
__author__ = 'Guillaume Osmond'

from .fingerprints import FingerprintGenerator
from . import utils

__all__ = [
    'FingerprintGenerator',
    'utils',
]
