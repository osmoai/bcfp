"""
Basic tests for BCFP functionality
"""

import pytest
import numpy as np
from bcfp import FingerprintGenerator


def test_import():
    """Test that the package imports correctly."""
    from bcfp import FingerprintGenerator
    assert FingerprintGenerator is not None


def test_ecfp_generation():
    """Test basic ECFP generation."""
    smiles = ["CC", "CCC", "CCCC"]
    gen = FingerprintGenerator('xxhash', 'ecfp', radius=2, n_bits=2048)
    fingerprints = gen.transform(smiles)
    
    assert fingerprints.shape == (3, 2048)
    assert fingerprints.dtype in [np.int32, np.int64, np.uint8]
    assert np.all(fingerprints >= 0)


def test_bcfp_generation():
    """Test basic BCFP generation."""
    smiles = ["CC", "CCC", "CCCC"]
    gen = FingerprintGenerator('xxhash', 'bcfp', radius=2, n_bits=2048)
    fingerprints = gen.transform(smiles)
    
    assert fingerprints.shape == (3, 2048)
    assert fingerprints.dtype in [np.int32, np.int64, np.uint8]
    assert np.all(fingerprints >= 0)


def test_multiple_hash_functions():
    """Test that multiple hash functions work."""
    smiles = ["CCO"]
    hash_funcs = ['xxhash', 'blake3', 'rdkit_native']
    
    for hash_func in hash_funcs:
        try:
            gen = FingerprintGenerator(hash_func, 'ecfp', radius=2, n_bits=1024)
            fp = gen.transform(smiles)
            assert fp.shape == (1, 1024)
        except Exception as e:
            # Some hash functions might not be available
            pytest.skip(f"{hash_func} not available: {str(e)}")


def test_sortslice():
    """Test Sort&Slice feature selection."""
    smiles = ["CC", "CCC", "CCCC"] * 10
    
    gen = BCFPEnhanced(
        fp_type='ecfp',
        radius=2,
        n_bits=2048,
        top_k=512,
        include_oov=False
    )
    
    gen.fit(smiles)
    fingerprints = gen.transform(smiles)
    
    assert fingerprints.shape == (30, 512)


def test_sortslice_with_oov():
    """Test Sort&Slice with OOV bucket."""
    train_smiles = ["CC", "CCC", "CCCC"] * 10
    test_smiles = ["CCCCC", "CCCCCC"]
    
    gen = BCFPEnhanced(
        fp_type='ecfp',
        radius=2,
        n_bits=2048,
        top_k=512,
        include_oov=True
    )
    
    gen.fit(train_smiles)
    train_fp = gen.transform(train_smiles)
    test_fp = gen.transform(test_smiles)
    
    assert train_fp.shape == (30, 513)  # 512 + 1 OOV
    assert test_fp.shape == (2, 513)


def test_consistency():
    """Test that fingerprints are consistent across multiple calls."""
    smiles = ["CCO", "c1ccccc1"]
    gen = FingerprintGenerator('xxhash', 'ecfp', radius=2, n_bits=1024)
    
    fp1 = gen.transform(smiles)
    fp2 = gen.transform(smiles)
    
    np.testing.assert_array_equal(fp1, fp2)


def test_different_radii():
    """Test that different radii produce different fingerprints."""
    smiles = ["c1ccccc1"]
    
    gen_r1 = FingerprintGenerator('xxhash', 'ecfp', radius=1, n_bits=1024)
    gen_r2 = FingerprintGenerator('xxhash', 'ecfp', radius=2, n_bits=1024)
    gen_r3 = FingerprintGenerator('xxhash', 'ecfp', radius=3, n_bits=1024)
    
    fp_r1 = gen_r1.transform(smiles)
    fp_r2 = gen_r2.transform(smiles)
    fp_r3 = gen_r3.transform(smiles)
    
    # Different radii should produce different fingerprints
    assert not np.array_equal(fp_r1, fp_r2)
    assert not np.array_equal(fp_r2, fp_r3)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

