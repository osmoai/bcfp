"""
Test hash function compatibility and verification.

This test ensures that:
1. Hash function is correctly saved and loaded
2. Hash verification works correctly
3. Model comparison detects hash mismatches
4. get_model_info() returns correct hash information

Author: Guillaume GODIN - Osmo labs pbc
"""

import pytest
import numpy as np
import tempfile
import os
from bcfp import FingerprintModel


def test_hash_func_saved_and_loaded():
    """Test that hash function is correctly saved and loaded."""
    train_smiles = ['CCO', 'CC(C)O', 'CCCO', 'CCCC']
    train_idx = np.arange(len(train_smiles))
    
    # Create and fit model with xxhash
    model = FingerprintModel(hash_func='xxhash', radius=2, top_k=10)
    model.fit(train_smiles, train_idx)
    
    # Save to temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as f:
        model.save(f.name)
        temp_path = f.name
    
    try:
        # Load model
        model_loaded = FingerprintModel.load(temp_path, validate=False)
        
        # Verify hash function is preserved
        assert model_loaded.hash_func == 'xxhash', "Hash function not preserved!"
        assert model_loaded.radius == 2, "Radius not preserved!"
        assert model_loaded.fp_type == 'ecfp', "FP type not preserved!"
        
        print("✅ Hash function correctly saved and loaded")
        
    finally:
        os.remove(temp_path)


def test_hash_verification():
    """Test hash compatibility verification."""
    train_smiles = ['CCO', 'CC(C)O', 'CCCO', 'CCCC']
    train_idx = np.arange(len(train_smiles))
    
    # Test all three hash functions
    for hash_func in ['rdkit_native', 'xxhash', 'blake3']:
        model = FingerprintModel(hash_func=hash_func, radius=2, top_k=10)
        model.fit(train_smiles, train_idx)
        
        # Verify hash compatibility
        result = model.verify_hash_compatibility()
        
        assert result['status'] == 'OK', f"Verification failed for {hash_func}"
        assert result['hash_func'] == hash_func, "Hash mismatch in result"
        assert 'n_features_generated' in result, "Missing feature count"
        assert result['vocab_size'] > 0, "Invalid vocab size"
        
        print(f"✅ Hash verification passed for {hash_func}")


def test_get_model_info():
    """Test get_model_info() returns correct hash information."""
    train_smiles = ['CCO', 'CC(C)O', 'CCCO', 'CCCC']
    train_idx = np.arange(len(train_smiles))
    
    model = FingerprintModel(hash_func='blake3', radius=3, top_k=20)
    model.fit(train_smiles, train_idx)
    
    info = model.get_model_info()
    
    assert info['hash_func'] == 'blake3', "Hash function not in info"
    assert info['fp_type'] == 'ecfp', "FP type not in info"
    assert info['radius'] == 3, "Radius not in info"
    assert info['is_fitted'] == True, "Fitted state not in info"
    assert info['n_features'] > 0, "Feature count missing"
    assert 'bcfp_version' in info, "Version missing"
    
    print(f"✅ get_model_info() returns correct information")
    print(f"   Hash: {info['hash_func']}")
    print(f"   Features: {info['n_features']}")


def test_compare_models_same_hash():
    """Test model comparison with same hash function."""
    train_smiles = ['CCO', 'CC(C)O', 'CCCO', 'CCCC']
    train_idx = np.arange(len(train_smiles))
    
    # Create two models with SAME hash
    model1 = FingerprintModel(hash_func='xxhash', radius=2, top_k=10)
    model1.fit(train_smiles, train_idx)
    
    model2 = FingerprintModel(hash_func='xxhash', radius=2, top_k=20)
    model2.fit(train_smiles, train_idx)
    
    # Compare
    comparison = FingerprintModel.compare_models(model1, model2)
    
    assert comparison['hash_func_match'] == True, "Hash should match"
    assert comparison['status'] == 'COMPATIBLE', "Models should be compatible"
    assert comparison['hash_func_1'] == 'xxhash', "Hash func 1 incorrect"
    assert comparison['hash_func_2'] == 'xxhash', "Hash func 2 incorrect"
    
    print(f"✅ Model comparison: {comparison['message']}")


def test_compare_models_different_hash():
    """Test model comparison with different hash functions."""
    train_smiles = ['CCO', 'CC(C)O', 'CCCO', 'CCCC']
    train_idx = np.arange(len(train_smiles))
    
    # Create two models with DIFFERENT hashes
    model1 = FingerprintModel(hash_func='xxhash', radius=2, top_k=10)
    model1.fit(train_smiles, train_idx)
    
    model2 = FingerprintModel(hash_func='blake3', radius=2, top_k=10)
    model2.fit(train_smiles, train_idx)
    
    # Compare
    comparison = FingerprintModel.compare_models(model1, model2)
    
    assert comparison['hash_func_match'] == False, "Hash should NOT match"
    assert comparison['status'] == 'INCOMPATIBLE', "Models should be INCOMPATIBLE"
    assert comparison['hash_func_1'] == 'xxhash', "Hash func 1 incorrect"
    assert comparison['hash_func_2'] == 'blake3', "Hash func 2 incorrect"
    assert 'hash_func' in comparison['message'], "Message should mention hash mismatch"
    
    print(f"✅ Model comparison detected incompatibility: {comparison['message']}")


def test_different_hashes_produce_different_features():
    """CRITICAL TEST: Verify different hashes produce different features."""
    train_smiles = ['CCO', 'CC(C)O', 'CCCO', 'CCCC', 'CCC', 'CC']
    train_idx = np.arange(len(train_smiles))
    
    # Train with xxhash
    model_xxhash = FingerprintModel(hash_func='xxhash', radius=2, top_k=50)
    X_xxhash = model_xxhash.fit_transform(train_smiles, train_idx)
    
    # Train with blake3
    model_blake3 = FingerprintModel(hash_func='blake3', radius=2, top_k=50)
    X_blake3 = model_blake3.fit_transform(train_smiles, train_idx)
    
    # Train with rdkit_native
    model_native = FingerprintModel(hash_func='rdkit_native', radius=2, top_k=50)
    X_native = model_native.fit_transform(train_smiles, train_idx)
    
    # Verify they produce different vocabularies (different feature spaces)
    # The vocabularies should be different because hash functions produce different keys
    vocab_xxhash = set(model_xxhash.vocab_)
    vocab_blake3 = set(model_blake3.vocab_)
    vocab_native = set(model_native.vocab_)
    
    # Check that vocabularies are different
    xxhash_blake3_overlap = len(vocab_xxhash & vocab_blake3) / min(len(vocab_xxhash), len(vocab_blake3))
    xxhash_native_overlap = len(vocab_xxhash & vocab_native) / min(len(vocab_xxhash), len(vocab_native))
    blake3_native_overlap = len(vocab_blake3 & vocab_native) / min(len(vocab_blake3), len(vocab_native))
    
    # Expect minimal overlap (hash functions produce different keys)
    # In practice, there should be almost no overlap, but we allow up to 10% due to collisions
    assert xxhash_blake3_overlap < 0.1, f"XXHash and Blake3 have {xxhash_blake3_overlap*100:.1f}% overlap - too high!"
    assert xxhash_native_overlap < 0.1, f"XXHash and Native have {xxhash_native_overlap*100:.1f}% overlap - too high!"
    assert blake3_native_overlap < 0.1, f"Blake3 and Native have {blake3_native_overlap*100:.1f}% overlap - too high!"
    
    print(f"✅ Different hashes produce different feature spaces:")
    print(f"   XXHash vocab:  {len(vocab_xxhash)} keys, shape: {X_xxhash.shape}")
    print(f"   Blake3 vocab:  {len(vocab_blake3)} keys, shape: {X_blake3.shape}")
    print(f"   Native vocab:  {len(vocab_native)} keys, shape: {X_native.shape}")
    print(f"   Overlap XXHash-Blake3: {xxhash_blake3_overlap*100:.1f}%")
    print(f"   Overlap XXHash-Native: {xxhash_native_overlap*100:.1f}%")
    print(f"   Overlap Blake3-Native: {blake3_native_overlap*100:.1f}%")


def test_save_load_preserves_hash():
    """Test that save/load cycle preserves hash function perfectly."""
    train_smiles = ['CCO', 'CC(C)O', 'CCCO', 'CCCC']
    train_idx = np.arange(len(train_smiles))
    test_smiles = ['CCCCO', 'CCC(C)C']
    
    for hash_func in ['rdkit_native', 'xxhash', 'blake3']:
        # Train model
        model_original = FingerprintModel(hash_func=hash_func, radius=2, top_k=20)
        X_train_original = model_original.fit_transform(train_smiles, train_idx)
        X_test_original = model_original.transform(test_smiles)
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as f:
            model_original.save(f.name)
            temp_path = f.name
        
        try:
            # Load model
            model_loaded = FingerprintModel.load(temp_path, validate=False)
            
            # Transform same test data
            X_test_loaded = model_loaded.transform(test_smiles)
            
            # Verify EXACT match
            assert np.array_equal(X_test_original, X_test_loaded), \
                f"Features don't match after load for {hash_func}!"
            
            print(f"✅ Save/load preserves features for {hash_func}")
            
        finally:
            os.remove(temp_path)


if __name__ == '__main__':
    print("=" * 80)
    print("HASH FUNCTION COMPATIBILITY TESTS")
    print("=" * 80)
    
    test_hash_func_saved_and_loaded()
    test_hash_verification()
    test_get_model_info()
    test_compare_models_same_hash()
    test_compare_models_different_hash()
    test_different_hashes_produce_different_features()
    test_save_load_preserves_hash()
    
    print("\n" + "=" * 80)
    print("🎉 ALL HASH COMPATIBILITY TESTS PASSED!")
    print("=" * 80)

