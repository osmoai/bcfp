"""
Complete ML Workflow Example with BCFP
========================================

This example demonstrates:
1. Generate ECFP fingerprints with Sort&Slice + OOV
2. Train a Logistic Regression model
3. Save the Sort&Slice vocabulary
4. Apply to new molecules in production

Note: This example uses ECFP only for simplicity. For ECFP+BCFP concatenation,
      see example_sortslice_oov.py
"""

import numpy as np
from rdkit import Chem
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
import tempfile
import os

from bcfp import FingerprintGenerator, save_sortslice_vocab, load_sortslice_vocab


def main():
    print("=" * 80)
    print("BCFP ML Workflow: Training and Inference Example")
    print("=" * 80)
    
    # =========================================================================
    # Step 1: Training Data (10 molecules)
    # =========================================================================
    print("\n[Step 1] Prepare Training Data")
    print("-" * 80)
    
    # Small alkanes (label 0) and alcohols (label 1)
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
    
    print(f"Training set: {len(train_smiles)} molecules")
    print(f"  Class 0 (alkanes):  {sum(train_labels == 0)} molecules")
    print(f"  Class 1 (alcohols): {sum(train_labels == 1)} molecules")
    
    # Convert to RDKit molecules
    train_mols = [Chem.MolFromSmiles(s) for s in train_smiles]
    print(f"✅ Converted to RDKit molecules")
    
    # =========================================================================
    # Step 2: Generate ECFP Fingerprints
    # =========================================================================
    print("\n[Step 2] Generate Fingerprints (ECFP)")
    print("-" * 80)
    
    # ECFP (atom-centered) with xxhash for speed
    gen_ecfp = FingerprintGenerator('xxhash', 'ecfp', radius=2)
    train_fps_ecfp = [gen_ecfp.generate_sparse(mol) for mol in train_mols]
    print(f"✅ ECFP fingerprints generated (radius=2, xxhash)")
    
    # =========================================================================
    # Step 3: Sort&Slice Feature Selection
    # =========================================================================
    print("\n[Step 3] Sort&Slice Feature Selection")
    print("-" * 80)
    
    train_idx = np.arange(len(train_mols))
    top_k = 64  # Select top 64 features
    
    # ECFP Sort&Slice
    vocab_ecfp, key2col_ecfp = FingerprintGenerator.sortslice_fit(
        train_fps_ecfp, train_idx, top_k=top_k
    )
    print(f"✅ ECFP vocabulary: {len(vocab_ecfp)} features selected")
    
    # =========================================================================
    # Step 4: Transform to Dense Matrix with OOV
    # =========================================================================
    print("\n[Step 4] Transform to Dense Matrix (with OOV bucket)")
    print("-" * 80)
    
    # ECFP transform
    X_train = FingerprintGenerator.sortslice_transform(
        train_fps_ecfp, train_idx, key2col_ecfp,
        use_counts=True, add_oov_bucket=True
    )
    print(f"✅ ECFP training matrix: {X_train.shape} ({top_k} features + 1 OOV)")
    
    # =========================================================================
    # Step 5: Train Logistic Regression Model
    # =========================================================================
    print("\n[Step 5] Train Logistic Regression Model")
    print("-" * 80)
    
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train, train_labels)
    
    # Training performance
    train_pred = model.predict(X_train)
    train_proba = model.predict_proba(X_train)[:, 1]
    train_acc = accuracy_score(train_labels, train_pred)
    train_auc = roc_auc_score(train_labels, train_proba)
    
    print(f"✅ Model trained successfully")
    print(f"   Training Accuracy: {train_acc:.3f}")
    print(f"   Training AUC:      {train_auc:.3f}")
    
    # =========================================================================
    # Step 6: Save Sort&Slice Vocabulary
    # =========================================================================
    print("\n[Step 6] Save Sort&Slice Vocabulary")
    print("-" * 80)
    
    # Create temporary directory for this example
    temp_dir = tempfile.mkdtemp()
    vocab_file = os.path.join(temp_dir, 'vocab_ecfp.pkl')
    
    save_sortslice_vocab(vocab_ecfp, key2col_ecfp, vocab_file)
    print(f"   Vocabulary saved to: {vocab_file}")
    
    # =========================================================================
    # Step 7: Production Inference on New Molecules
    # =========================================================================
    print("\n[Step 7] Production Inference on New Molecules")
    print("-" * 80)
    
    # Two new molecules (unseen during training)
    test_smiles = [
        "CCCCCCCO",     # Heptanol (should predict 1 - alcohol)
        "CCCCCCC",      # Heptane (should predict 0 - alkane)
    ]
    
    test_labels_true = np.array([1, 0])  # Ground truth for validation
    
    print(f"Test set: {len(test_smiles)} molecules")
    for i, smi in enumerate(test_smiles):
        print(f"  {i+1}. {smi} (true label: {test_labels_true[i]})")
    
    # Convert to RDKit molecules
    test_mols = [Chem.MolFromSmiles(s) for s in test_smiles]
    
    # Load vocabulary (simulating production environment)
    print("\n   Loading vocabulary...")
    vocab_loaded, key2col_loaded = load_sortslice_vocab(vocab_file)
    
    # Generate fingerprints
    test_fps = [gen_ecfp.generate_sparse(mol) for mol in test_mols]
    
    # Transform using loaded vocabulary
    test_idx = np.arange(len(test_mols))
    
    X_test = FingerprintGenerator.sortslice_transform(
        test_fps, test_idx, key2col_loaded,
        use_counts=True, add_oov_bucket=True
    )
    print(f"   ✅ Test matrix: {X_test.shape}")
    
    # Make predictions
    test_pred = model.predict(X_test)
    test_proba = model.predict_proba(X_test)[:, 1]
    
    print("\n   Predictions:")
    for i, (smi, true_label, pred_label, proba) in enumerate(
        zip(test_smiles, test_labels_true, test_pred, test_proba)
    ):
        correct = "✅" if pred_label == true_label else "❌"
        print(f"   {i+1}. {smi:15s} → Pred: {pred_label} (P={proba:.3f}) | True: {true_label} {correct}")
    
    # Test performance
    test_acc = accuracy_score(test_labels_true, test_pred)
    test_auc = roc_auc_score(test_labels_true, test_proba)
    
    print(f"\n   Test Accuracy: {test_acc:.3f}")
    print(f"   Test AUC:      {test_auc:.3f}")
    
    # =========================================================================
    # Step 8: Verify OOV Bucket Usage
    # =========================================================================
    print("\n[Step 8] Verify OOV Bucket Usage")
    print("-" * 80)
    
    oov_counts = X_test[:, -1]
    print(f"   OOV counts: {oov_counts}")
    
    if np.any(oov_counts > 0):
        print(f"   ✅ OOV bucket captured unseen features from new molecules")
    else:
        print(f"   ℹ️  No OOV features (all features seen in training)")
    
    # Cleanup
    os.remove(vocab_file)
    os.rmdir(temp_dir)
    
    print("\n" + "=" * 80)
    print("✅ Complete ML Workflow Finished Successfully!")
    print("=" * 80)
    print("\nKey Takeaways:")
    print("  1. ECFP fingerprints provide rich molecular representation")
    print("  2. Sort&Slice reduces dimensionality while preserving information")
    print("  3. OOV bucket handles unseen features in production")
    print("  4. Save/Load vocabularies enable production deployment")
    print("  5. Simple workflow: Fit → Save → Load → Transform → Predict")
    
    return {
        'train_acc': train_acc,
        'train_auc': train_auc,
        'test_acc': test_acc,
        'test_auc': test_auc,
        'X_train_shape': X_train.shape,
        'X_test_shape': X_test.shape,
    }


if __name__ == "__main__":
    results = main()

