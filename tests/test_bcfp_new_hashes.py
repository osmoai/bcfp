"""
Test BCFP with XXHash and Blake3 hash functions
"""
import pytest
import numpy as np
from rdkit import Chem
from bcfp import FingerprintGenerator


def test_bcfp_xxhash_basic():
    """Test BCFP with XXHash generates fingerprints"""
    gen = FingerprintGenerator('xxhash', 'bcfp', radius=2)
    mol = Chem.MolFromSmiles('CCO')
    fp = gen.generate_sparse(mol)
    
    assert isinstance(fp, dict), "Should return dict"
    assert len(fp) > 0, "Should have features"
    assert all(isinstance(k, int) for k in fp.keys()), "Keys should be integers"
    assert all(isinstance(v, int) for v in fp.values()), "Values should be integers"


def test_bcfp_blake3_basic():
    """Test BCFP with Blake3 generates fingerprints"""
    gen = FingerprintGenerator('blake3', 'bcfp', radius=2)
    mol = Chem.MolFromSmiles('CCO')
    fp = gen.generate_sparse(mol)
    
    assert isinstance(fp, dict), "Should return dict"
    assert len(fp) > 0, "Should have features"
    assert all(isinstance(k, int) for k in fp.keys()), "Keys should be integers"
    assert all(isinstance(v, int) for v in fp.values()), "Values should be integers"


def test_bcfp_xxhash_different_molecules():
    """Test BCFP XXHash produces different fingerprints for different molecules"""
    gen = FingerprintGenerator('xxhash', 'bcfp', radius=2)
    
    mol1 = Chem.MolFromSmiles('CCO')
    mol2 = Chem.MolFromSmiles('c1ccccc1')
    
    fp1 = gen.generate_sparse(mol1)
    fp2 = gen.generate_sparse(mol2)
    
    assert fp1 != fp2, "Different molecules should have different fingerprints"


def test_bcfp_blake3_different_molecules():
    """Test BCFP Blake3 produces different fingerprints for different molecules"""
    gen = FingerprintGenerator('blake3', 'bcfp', radius=2)
    
    mol1 = Chem.MolFromSmiles('CCO')
    mol2 = Chem.MolFromSmiles('c1ccccc1')
    
    fp1 = gen.generate_sparse(mol1)
    fp2 = gen.generate_sparse(mol2)
    
    assert fp1 != fp2, "Different molecules should have different fingerprints"


def test_bcfp_xxhash_radius_effect():
    """Test that radius affects BCFP XXHash fingerprints"""
    mol = Chem.MolFromSmiles('c1ccccc1')
    
    gen_r1 = FingerprintGenerator('xxhash', 'bcfp', radius=1)
    gen_r2 = FingerprintGenerator('xxhash', 'bcfp', radius=2)
    gen_r3 = FingerprintGenerator('xxhash', 'bcfp', radius=3)
    
    fp_r1 = gen_r1.generate_sparse(mol)
    fp_r2 = gen_r2.generate_sparse(mol)
    fp_r3 = gen_r3.generate_sparse(mol)
    
    # Different radii should produce different fingerprints
    assert fp_r1 != fp_r2, "radius=1 and radius=2 should differ"
    assert fp_r2 != fp_r3, "radius=2 and radius=3 should differ"
    
    # Larger radius generally means more features
    assert len(fp_r3) >= len(fp_r2), "radius=3 should have >= features than radius=2"


def test_bcfp_blake3_radius_effect():
    """Test that radius affects BCFP Blake3 fingerprints"""
    mol = Chem.MolFromSmiles('c1ccccc1')
    
    gen_r1 = FingerprintGenerator('blake3', 'bcfp', radius=1)
    gen_r2 = FingerprintGenerator('blake3', 'bcfp', radius=2)
    gen_r3 = FingerprintGenerator('blake3', 'bcfp', radius=3)
    
    fp_r1 = gen_r1.generate_sparse(mol)
    fp_r2 = gen_r2.generate_sparse(mol)
    fp_r3 = gen_r3.generate_sparse(mol)
    
    # Different radii should produce different fingerprints
    assert fp_r1 != fp_r2, "radius=1 and radius=2 should differ"
    assert fp_r2 != fp_r3, "radius=2 and radius=3 should differ"


def test_bcfp_xxhash_consistency():
    """Test BCFP XXHash produces consistent results"""
    gen = FingerprintGenerator('xxhash', 'bcfp', radius=2)
    mol = Chem.MolFromSmiles('CCO')
    
    fp1 = gen.generate_sparse(mol)
    fp2 = gen.generate_sparse(mol)
    
    assert fp1 == fp2, "Same molecule should produce identical fingerprints"


def test_bcfp_blake3_consistency():
    """Test BCFP Blake3 produces consistent results"""
    gen = FingerprintGenerator('blake3', 'bcfp', radius=2)
    mol = Chem.MolFromSmiles('CCO')
    
    fp1 = gen.generate_sparse(mol)
    fp2 = gen.generate_sparse(mol)
    
    assert fp1 == fp2, "Same molecule should produce identical fingerprints"


def test_bcfp_hash_functions_produce_different_hashes():
    """Test that different hash functions produce different feature IDs"""
    mol = Chem.MolFromSmiles('CCO')
    
    gen_native = FingerprintGenerator('rdkit_native', 'bcfp', radius=2)
    gen_xxhash = FingerprintGenerator('xxhash', 'bcfp', radius=2)
    gen_blake3 = FingerprintGenerator('blake3', 'bcfp', radius=2)
    
    fp_native = gen_native.generate_sparse(mol)
    fp_xxhash = gen_xxhash.generate_sparse(mol)
    fp_blake3 = gen_blake3.generate_sparse(mol)
    
    # All should have same number of features (same molecule structure)
    assert len(fp_native) == len(fp_xxhash) == len(fp_blake3), \
        "All hash functions should detect same number of features"
    
    # But feature IDs should be different (different hash functions)
    assert set(fp_native.keys()) != set(fp_xxhash.keys()), \
        "Native and XXHash should have different feature IDs"
    assert set(fp_native.keys()) != set(fp_blake3.keys()), \
        "Native and Blake3 should have different feature IDs"
    assert set(fp_xxhash.keys()) != set(fp_blake3.keys()), \
        "XXHash and Blake3 should have different feature IDs"


def test_bcfp_xxhash_with_sortslice():
    """Test BCFP XXHash works with Sort&Slice pipeline"""
    gen = FingerprintGenerator('xxhash', 'bcfp', radius=2)
    
    smiles = ["CCO", "c1ccccc1", "CC(=O)O"] * 10
    mols = [Chem.MolFromSmiles(s) for s in smiles]
    fps = [gen.generate_sparse(mol) for mol in mols]
    
    train_idx = np.arange(len(mols))
    vocab, key2col = FingerprintGenerator.sortslice_fit(fps, train_idx, top_k=10)
    
    X = FingerprintGenerator.sortslice_transform(
        fps, train_idx, key2col, use_counts=True, add_oov_bucket=True
    )
    
    assert X.shape == (30, 11), "Should have 30 samples × (10 features + 1 OOV)"
    assert np.all(X >= 0), "All counts should be non-negative"


def test_bcfp_blake3_with_sortslice():
    """Test BCFP Blake3 works with Sort&Slice pipeline"""
    gen = FingerprintGenerator('blake3', 'bcfp', radius=2)
    
    smiles = ["CCO", "c1ccccc1", "CC(=O)O"] * 10
    mols = [Chem.MolFromSmiles(s) for s in smiles]
    fps = [gen.generate_sparse(mol) for mol in mols]
    
    train_idx = np.arange(len(mols))
    vocab, key2col = FingerprintGenerator.sortslice_fit(fps, train_idx, top_k=10)
    
    X = FingerprintGenerator.sortslice_transform(
        fps, train_idx, key2col, use_counts=True, add_oov_bucket=True
    )
    
    assert X.shape == (30, 11), "Should have 30 samples × (10 features + 1 OOV)"
    assert np.all(X >= 0), "All counts should be non-negative"


def test_bcfp_all_three_hashes_complete_pipeline():
    """Test complete pipeline with all 3 hash functions for BCFP"""
    smiles = ["CCO", "c1ccccc1", "CC(=O)O", "CCC", "CCCC"]
    mols = [Chem.MolFromSmiles(s) for s in smiles]
    
    for hash_func in ['rdkit_native', 'xxhash', 'blake3']:
        gen = FingerprintGenerator(hash_func, 'bcfp', radius=2)
        
        # Generate sparse fingerprints
        fps = [gen.generate_sparse(mol) for mol in mols]
        
        # Sort&Slice
        train_idx = np.arange(len(mols))
        vocab, key2col = gen.sortslice_fit(fps, train_idx, top_k=5)
        
        # Transform
        X = gen.sortslice_transform(
            fps, train_idx, key2col, use_counts=True, add_oov_bucket=True
        )
        
        assert X.shape[0] == len(mols), f"{hash_func}: Wrong number of samples"
        assert X.shape[1] <= 6, f"{hash_func}: Should have <= 5 features + 1 OOV"
        assert X.shape[1] >= 2, f"{hash_func}: Should have at least 1 feature + 1 OOV"
        assert np.all(X >= 0), f"{hash_func}: All values should be non-negative"


if __name__ == "__main__":
    # Run all tests
    pytest.main([__file__, "-v"])

