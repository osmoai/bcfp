"""
Test save_sortslice_vocab() and load_sortslice_vocab() functionality
"""
import pytest
import numpy as np
import os
import tempfile
from rdkit import Chem
from bcfp import FingerprintGenerator, save_sortslice_vocab, load_sortslice_vocab


def test_save_load_sortslice_vocab():
    """Test saving and loading Sort&Slice vocabulary"""
    
    # Training data
    train_smiles = ["CCO", "c1ccccc1", "CC(=O)O", "CCCC", "CCC"] * 20
    train_mols = [Chem.MolFromSmiles(s) for s in train_smiles]
    
    # Test data (includes novel structures)
    test_smiles = ["CCCCC", "CCCCCC", "CCO"]
    test_mols = [Chem.MolFromSmiles(s) for s in test_smiles]
    
    # Generate fingerprints
    gen = FingerprintGenerator('xxhash', 'ecfp', radius=2)
    train_fps = [gen.generate_sparse(mol) for mol in train_mols]
    test_fps = [gen.generate_sparse(mol) for mol in test_mols]
    
    # Fit Sort&Slice on training data
    vocab_orig, key2col_orig = FingerprintGenerator.sortslice_fit(
        train_fps,
        np.arange(len(train_mols)),
        top_k=50
    )
    
    # Transform with original vocab
    X_train_orig = FingerprintGenerator.sortslice_transform(
        train_fps,
        np.arange(len(train_mols)),
        key2col_orig,
        use_counts=True,
        add_oov_bucket=True
    )
    
    X_test_orig = FingerprintGenerator.sortslice_transform(
        test_fps,
        np.arange(len(test_mols)),
        key2col_orig,
        use_counts=True,
        add_oov_bucket=True
    )
    
    # Save vocabulary
    with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.pkl') as f:
        vocab_file = f.name
    
    try:
        save_sortslice_vocab(vocab_orig, key2col_orig, vocab_file)
        
        # Load vocabulary
        vocab_loaded, key2col_loaded = load_sortslice_vocab(vocab_file)
        
        # Verify vocab matches
        assert len(vocab_loaded) == len(vocab_orig), "Vocab length mismatch"
        assert vocab_loaded == vocab_orig, "Vocab content mismatch"
        assert key2col_loaded == key2col_orig, "key2col mapping mismatch"
        
        # Transform with loaded vocab
        X_train_loaded = FingerprintGenerator.sortslice_transform(
            train_fps,
            np.arange(len(train_mols)),
            key2col_loaded,
            use_counts=True,
            add_oov_bucket=True
        )
        
        X_test_loaded = FingerprintGenerator.sortslice_transform(
            test_fps,
            np.arange(len(test_mols)),
            key2col_loaded,
            use_counts=True,
            add_oov_bucket=True
        )
        
        # Verify features match exactly
        assert np.allclose(X_train_orig, X_train_loaded), "Train features mismatch"
        assert np.allclose(X_test_orig, X_test_loaded), "Test features mismatch"
        
        print("✅ Save/Load test passed!")
        print(f"   Vocab size: {len(vocab_loaded)}")
        print(f"   Train shape: {X_train_loaded.shape}")
        print(f"   Test shape: {X_test_loaded.shape}")
        
    finally:
        # Clean up
        if os.path.exists(vocab_file):
            os.remove(vocab_file)


def test_save_load_with_oov():
    """Test that OOV bucket works correctly after load"""
    
    # Training data (small molecules)
    train_smiles = ["C", "CC", "CCC"] * 10
    train_mols = [Chem.MolFromSmiles(s) for s in train_smiles]
    
    # Test data (larger molecules with many OOV features)
    test_smiles = ["C" * 10, "C" * 12]
    test_mols = [Chem.MolFromSmiles(s) for s in test_smiles]
    
    # Generate fingerprints
    gen = FingerprintGenerator('xxhash', 'ecfp', radius=3)
    train_fps = [gen.generate_sparse(mol) for mol in train_mols]
    test_fps = [gen.generate_sparse(mol) for mol in test_mols]
    
    # Fit Sort&Slice with small top_k
    vocab, key2col = FingerprintGenerator.sortslice_fit(
        train_fps,
        np.arange(len(train_mols)),
        top_k=10  # Very small to force OOV
    )
    
    # Transform test data
    X_test_orig = FingerprintGenerator.sortslice_transform(
        test_fps,
        np.arange(len(test_mols)),
        key2col,
        use_counts=True,
        add_oov_bucket=True
    )
    
    # Save and load
    with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.pkl') as f:
        vocab_file = f.name
    
    try:
        save_sortslice_vocab(vocab, key2col, vocab_file)
        vocab_loaded, key2col_loaded = load_sortslice_vocab(vocab_file)
        
        # Transform with loaded vocab
        X_test_loaded = FingerprintGenerator.sortslice_transform(
            test_fps,
            np.arange(len(test_mols)),
            key2col_loaded,
            use_counts=True,
            add_oov_bucket=True
        )
        
        # Verify OOV column is non-zero (last column)
        assert X_test_orig[:, -1].sum() > 0, "OOV bucket should be non-zero"
        assert np.allclose(X_test_orig, X_test_loaded), "Features mismatch after load"
        
        print("✅ OOV test passed!")
        print(f"   OOV counts: {X_test_loaded[:, -1]}")
        
    finally:
        if os.path.exists(vocab_file):
            os.remove(vocab_file)


if __name__ == "__main__":
    test_save_load_sortslice_vocab()
    test_save_load_with_oov()
    print("\n🎉 All save/load tests passed!")

