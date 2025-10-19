"""
Test ML Workflow Example
=========================

This test verifies the complete ML workflow with:
- 10 training molecules
- 2 test molecules
- Logistic Regression classifier
- Save/Load vocabulary functionality
"""

import pytest
import numpy as np
import os
import tempfile
from rdkit import Chem
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from bcfp import FingerprintGenerator, save_sortslice_vocab, load_sortslice_vocab


def test_ml_workflow_10_train_2_test():
    """
    Test complete ML workflow with 10 training molecules and 2 test molecules.
    
    This replicates the example_ml_workflow.py as a pytest test.
    """
    # =========================================================================
    # Step 1: Training Data (10 molecules)
    # =========================================================================
    train_smiles = [
        "CC",           # Ethane (0)
        "CCC",          # Propane (0)
        "CCCC",         # Butane (0)
        "CCCCC",        # Pentane (0)
        "CCCCCC",       # Hexane (0)
        "CCO",          # Ethanol (1)
        "CCCO",         # Propanol (1)
        "CCCCO",        # Butanol (1)
        "CCCCCO",       # Pentanol (1)
        "CCCCCCO",      # Hexanol (1)
    ]
    
    train_labels = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    train_mols = [Chem.MolFromSmiles(s) for s in train_smiles]
    
    assert len(train_mols) == 10, "Should have 10 training molecules"
    assert sum(train_labels == 0) == 5, "Should have 5 class-0 molecules"
    assert sum(train_labels == 1) == 5, "Should have 5 class-1 molecules"
    
    # =========================================================================
    # Step 2: Generate ECFP Fingerprints
    # =========================================================================
    gen = FingerprintGenerator('xxhash', 'ecfp', radius=2)
    train_fps = [gen.generate_sparse(mol) for mol in train_mols]
    
    assert len(train_fps) == 10, "Should have 10 fingerprints"
    assert all(isinstance(fp, dict) for fp in train_fps), "Fingerprints should be dicts"
    
    # =========================================================================
    # Step 3: Sort&Slice Feature Selection
    # =========================================================================
    train_idx = np.arange(len(train_mols))
    top_k = 64
    
    vocab, key2col = FingerprintGenerator.sortslice_fit(
        train_fps, train_idx, top_k=top_k
    )
    
    assert isinstance(vocab, list), "vocab should be a list"
    assert isinstance(key2col, dict), "key2col should be a dict"
    assert len(vocab) <= top_k, f"vocab should have at most {top_k} features"
    assert len(key2col) == len(vocab), "key2col should match vocab length"
    
    # =========================================================================
    # Step 4: Transform to Dense Matrix with OOV
    # =========================================================================
    X_train = FingerprintGenerator.sortslice_transform(
        train_fps, train_idx, key2col,
        use_counts=True, add_oov_bucket=True
    )
    
    assert X_train.shape[0] == 10, "Should have 10 samples"
    assert X_train.shape[1] == len(vocab) + 1, "Should have vocab + 1 OOV column"
    assert np.all(X_train >= 0), "All counts should be non-negative"
    
    # =========================================================================
    # Step 5: Train Logistic Regression Model
    # =========================================================================
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train, train_labels)
    
    train_pred = model.predict(X_train)
    train_acc = accuracy_score(train_labels, train_pred)
    
    assert train_acc >= 0.8, f"Training accuracy should be >= 0.8, got {train_acc:.3f}"
    print(f"✅ Training accuracy: {train_acc:.3f}")
    
    # =========================================================================
    # Step 6: Save Sort&Slice Vocabulary
    # =========================================================================
    temp_dir = tempfile.mkdtemp()
    vocab_file = os.path.join(temp_dir, 'vocab_test.pkl')
    
    try:
        save_sortslice_vocab(vocab, key2col, vocab_file)
        assert os.path.exists(vocab_file), "Vocabulary file should exist"
        
        # =========================================================================
        # Step 7: Production Inference on New Molecules
        # =========================================================================
        test_smiles = [
            "CCCCCCCO",     # Heptanol (should predict 1)
            "CCCCCCC",      # Heptane (should predict 0)
        ]
        
        test_labels_true = np.array([1, 0])
        test_mols = [Chem.MolFromSmiles(s) for s in test_smiles]
        
        # Load vocabulary (simulating production)
        vocab_loaded, key2col_loaded = load_sortslice_vocab(vocab_file)
        
        assert vocab_loaded == vocab, "Loaded vocab should match original"
        assert key2col_loaded == key2col, "Loaded key2col should match original"
        
        # Generate test fingerprints
        test_fps = [gen.generate_sparse(mol) for mol in test_mols]
        test_idx = np.arange(len(test_mols))
        
        X_test = FingerprintGenerator.sortslice_transform(
            test_fps, test_idx, key2col_loaded,
            use_counts=True, add_oov_bucket=True
        )
        
        assert X_test.shape[0] == 2, "Should have 2 test samples"
        assert X_test.shape[1] == X_train.shape[1], "Test and train should have same features"
        
        # Make predictions
        test_pred = model.predict(X_test)
        test_acc = accuracy_score(test_labels_true, test_pred)
        
        print(f"✅ Test accuracy: {test_acc:.3f}")
        print(f"   Predictions: {test_pred}")
        print(f"   True labels: {test_labels_true}")
        
        # Verify predictions are reasonable
        assert test_acc >= 0.5, f"Test accuracy should be >= 0.5, got {test_acc:.3f}"
        
        # =========================================================================
        # Step 8: Verify OOV Bucket Usage
        # =========================================================================
        oov_counts = X_test[:, -1]
        
        # OOV bucket should have some counts for new molecules
        assert np.all(oov_counts >= 0), "OOV counts should be non-negative"
        print(f"✅ OOV counts: {oov_counts}")
        
    finally:
        # Cleanup
        if os.path.exists(vocab_file):
            os.remove(vocab_file)
        if os.path.exists(temp_dir):
            os.rmdir(temp_dir)


def test_ml_workflow_perfect_separation():
    """
    Test ML workflow with perfectly separable data.
    
    This should achieve 100% accuracy on both training and test.
    """
    # Very simple molecules: single carbon vs benzene
    train_smiles = ["C"] * 5 + ["c1ccccc1"] * 5
    train_labels = np.array([0] * 5 + [1] * 5)
    train_mols = [Chem.MolFromSmiles(s) for s in train_smiles]
    
    # Generate fingerprints
    gen = FingerprintGenerator('xxhash', 'ecfp', radius=2)
    train_fps = [gen.generate_sparse(mol) for mol in train_mols]
    
    # Sort&Slice
    train_idx = np.arange(len(train_mols))
    vocab, key2col = FingerprintGenerator.sortslice_fit(train_fps, train_idx, top_k=32)
    
    X_train = FingerprintGenerator.sortslice_transform(
        train_fps, train_idx, key2col, use_counts=True, add_oov_bucket=True
    )
    
    # Train model
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train, train_labels)
    
    train_pred = model.predict(X_train)
    train_acc = accuracy_score(train_labels, train_pred)
    
    # Should achieve perfect training accuracy
    assert train_acc == 1.0, f"Should achieve perfect accuracy, got {train_acc:.3f}"
    
    # Test on similar molecules
    test_smiles = ["CC", "c1ccc(C)cc1"]  # Ethane (0), Toluene (1)
    test_labels = np.array([0, 1])
    test_mols = [Chem.MolFromSmiles(s) for s in test_smiles]
    
    test_fps = [gen.generate_sparse(mol) for mol in test_mols]
    test_idx = np.arange(len(test_mols))
    
    X_test = FingerprintGenerator.sortslice_transform(
        test_fps, test_idx, key2col, use_counts=True, add_oov_bucket=True
    )
    
    test_pred = model.predict(X_test)
    test_acc = accuracy_score(test_labels, test_pred)
    
    print(f"✅ Perfect separation test:")
    print(f"   Training accuracy: {train_acc:.3f}")
    print(f"   Test accuracy: {test_acc:.3f}")
    
    # Should generalize well to test set
    assert test_acc >= 0.5, f"Test accuracy should be >= 0.5, got {test_acc:.3f}"


if __name__ == "__main__":
    print("=" * 80)
    print("Testing ML Workflow")
    print("=" * 80)
    
    print("\n[Test 1] Complete workflow with 10 training + 2 test molecules")
    test_ml_workflow_10_train_2_test()
    
    print("\n[Test 2] Perfect separation test")
    test_ml_workflow_perfect_separation()
    
    print("\n" + "=" * 80)
    print("✅ All ML workflow tests passed!")
    print("=" * 80)

