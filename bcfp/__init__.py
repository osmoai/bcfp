"""
BCFP - Bond-Centered Fingerprints for Molecular Machine Learning

A high-performance Python library for generating molecular fingerprints with:
- Multiple hash functions (rdkit_native, xxhash, blake3)
- ECFP (Morgan/Atom-centered) and BCFP (Bond-centered) fingerprints
- Sort&Slice feature selection with OOV handling
- C++ implementation for maximum speed

Example:
    >>> from bcfp import FingerprintGenerator
    >>> 
    >>> # Generate ECFP fingerprints (batch processing)
    >>> gen = FingerprintGenerator('rdkit_native', 'ecfp', radius=2)
    >>> smiles = ['CCO', 'c1ccccc1', 'CC(=O)O']
    >>> X = gen.transform(smiles)  # Shape: (3, 2048)
    >>> 
    >>> # Sort&Slice with OOV handling
    >>> X_train = gen.fit_transform(train_smiles, top_k=512, include_oov=True)
    >>> X_test = gen.transform(test_smiles)  # Uses fitted vocabulary
"""

__version__ = '2.0.0'
__author__ = 'Guillaume Godin'

from .fingerprints import (
    FingerprintGenerator,
    save_sortslice_vocab,
    load_sortslice_vocab,
    combine_fingerprints
)
from .model_persistence import FingerprintModel
from . import utils

__all__ = [
    'FingerprintGenerator',
    'FingerprintModel',
    'save_sortslice_vocab',
    'load_sortslice_vocab',
    'combine_fingerprints',
    'utils',
]

