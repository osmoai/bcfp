"""
Complete Training and Inference Workflow with Model Persistence

This example demonstrates:
1. Training: Fit model on training data and save for production
2. Inference: Load saved model and predict on new molecules

Author: Guillaume GODIN - Osmo labs pbc
License: BSD-3-Clause
"""

import numpy as np
from rdkit import Chem
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score
import pickle


def example_1_basic_workflow():
    """Example 1: Basic train → save → load → inference workflow."""
    print("=" * 80)
    print("EXAMPLE 1: Basic Training and Inference Workflow")
    print("=" * 80)
    
    from bcfp import FingerprintModel
    
    # ========================================================================
    # STEP 1: TRAINING PHASE (done once, save for production)
    # ========================================================================
    print("\n📚 TRAINING PHASE")
    print("-" * 80)
    
    # Training data
    train_smiles = [
        'CCO', 'CC(C)O', 'CCCO', 'CC(C)CO',  # Alcohols (active)
        'CCCC', 'CCC', 'CC', 'CCCCC'         # Alkanes (inactive)
    ]
    train_labels = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    
    print(f"Training molecules: {len(train_smiles)}")
    print(f"  Active: {sum(train_labels)}")
    print(f"  Inactive: {len(train_labels) - sum(train_labels)}")
    
    # Create and fit fingerprint model
    fp_model = FingerprintModel(
        hash_func='xxhash',
        fp_type='ecfp',
        radius=2,
        top_k=512,
        add_oov_bucket=True
    )
    
    train_idx = np.arange(len(train_smiles))
    X_train = fp_model.fit_transform(train_smiles, train_idx)
    
    print(f"\nFingerprint matrix: {X_train.shape}")
    print(f"  Features: {X_train.shape[1] - 1} + 1 OOV")
    
    # Train ML model
    ml_model = RandomForestClassifier(n_estimators=100, random_state=42)
    ml_model.fit(X_train, train_labels)
    
    train_pred = ml_model.predict(X_train)
    train_acc = accuracy_score(train_labels, train_pred)
    print(f"\nTraining accuracy: {train_acc:.3f}")
    
    # Save both models
    fp_model.save('fingerprint_model.pkl')
    with open('ml_model.pkl', 'wb') as f:
        pickle.dump(ml_model, f)
    
    print("\n✅ Models saved to disk!")
    
    # ========================================================================
    # STEP 2: INFERENCE PHASE (production, load models and predict)
    # ========================================================================
    print("\n🔮 INFERENCE PHASE (Production)")
    print("-" * 80)
    
    # New molecules (never seen during training)
    new_smiles = [
        'CCCCO',      # New alcohol → should predict active
        'CCCCCC',     # New alkane → should predict inactive
        'CC(C)CCO',   # New alcohol → should predict active
        'CCC(C)C'     # New alkane → should predict inactive
    ]
    
    print(f"New molecules: {len(new_smiles)}")
    for smi in new_smiles:
        print(f"  {smi}")
    
    # Load saved models
    fp_model_loaded = FingerprintModel.load('fingerprint_model.pkl')
    with open('ml_model.pkl', 'rb') as f:
        ml_model_loaded = pickle.load(f)
    
    # Generate fingerprints for new molecules
    X_new = fp_model_loaded.transform(new_smiles)
    print(f"\nFingerprint matrix: {X_new.shape}")
    
    # Predict
    predictions = ml_model_loaded.predict(X_new)
    probabilities = ml_model_loaded.predict_proba(X_new)[:, 1]
    
    print("\n📊 Predictions:")
    for i, smi in enumerate(new_smiles):
        pred_label = "Active" if predictions[i] == 1 else "Inactive"
        prob = probabilities[i]
        print(f"  {smi:15s} → {pred_label:8s} (prob: {prob:.3f})")
    
    print("\n✅ Inference complete!")


def example_2_realistic_workflow():
    """Example 2: Realistic workflow with train/test split."""
    print("\n\n" + "=" * 80)
    print("EXAMPLE 2: Realistic Workflow with Train/Test Split")
    print("=" * 80)
    
    from bcfp import FingerprintModel
    from sklearn.model_selection import train_test_split
    
    # ========================================================================
    # DATASET: BBBP-like molecules
    # ========================================================================
    print("\n📚 Dataset Preparation")
    print("-" * 80)
    
    # Simulated BBBP molecules
    all_smiles = [
        # BBB+ (penetrant)
        'CC(C)Cc1ccc(cc1)C(C)C(=O)O',
        'CN1C=NC2=C1C(=O)N(C(=O)N2C)C',
        'CC(C)NCC(COc1ccccc1)O',
        'COc1ccc2c(c1)c(CC(=O)O)c(C)n2C(=O)c3ccc(cc3)Cl',
        'c1ccc(cc1)CCNC(=O)c2ccccc2',
        # BBB- (non-penetrant)
        'CC(C)(C)NCC(c1cc(c(c(c1)O)O)O)O',
        'CN(C)CCC=C1c2ccccc2CCc3c1cccc3',
        'COc1cc2c(cc1OC)C(=O)C(CC2)Cc3ccc(c(c3)OC)OC',
        'c1ccc2c(c1)C(=O)c3ccccc3C2=O',
        'CCN(CC)CCNC(=O)c1ccc(cc1N)S(=O)(=O)N',
    ]
    all_labels = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
    
    # Split train/test
    train_smi, test_smi, y_train, y_test = train_test_split(
        all_smiles, all_labels, test_size=0.3, random_state=42, stratify=all_labels
    )
    
    print(f"Total molecules: {len(all_smiles)}")
    print(f"Training set: {len(train_smi)}")
    print(f"Test set: {len(test_smi)}")
    
    # ========================================================================
    # TRAINING PHASE
    # ========================================================================
    print("\n📚 Training Phase")
    print("-" * 80)
    
    # Create fingerprint model with E/BCFP concatenation
    fp_model = FingerprintModel(
        hash_func='xxhash',
        fp_type='both',  # ECFP + BCFP concatenation
        radius=3,
        top_k=2048,
        sort_by='df',
        min_df=1,
        add_oov_bucket=True
    )
    
    # Fit on training data
    train_idx = np.arange(len(train_smi))
    X_train = fp_model.fit_transform(train_smi, train_idx)
    
    print(f"Training fingerprints: {X_train.shape}")
    print(f"  Sparsity: {np.mean(X_train == 0) * 100:.1f}%")
    
    # Train classifier
    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    clf.fit(X_train, y_train)
    
    # Evaluate on training set
    train_pred = clf.predict(X_train)
    train_proba = clf.predict_proba(X_train)[:, 1]
    train_acc = accuracy_score(y_train, train_pred)
    train_auc = roc_auc_score(y_train, train_proba)
    
    print(f"\nTraining performance:")
    print(f"  Accuracy: {train_acc:.3f}")
    print(f"  ROC-AUC: {train_auc:.3f}")
    
    # Save models
    fp_model.save('bbbp_fingerprint_model.pkl')
    with open('bbbp_ml_model.pkl', 'wb') as f:
        pickle.dump(clf, f)
    
    print("\n✅ Models saved!")
    
    # ========================================================================
    # INFERENCE PHASE (Test set evaluation)
    # ========================================================================
    print("\n🔮 Inference Phase (Test Set)")
    print("-" * 80)
    
    # Load models (simulating production environment)
    fp_model_loaded = FingerprintModel.load('bbbp_fingerprint_model.pkl')
    with open('bbbp_ml_model.pkl', 'rb') as f:
        clf_loaded = pickle.load(f)
    
    # Transform test set
    X_test = fp_model_loaded.transform(test_smi)
    
    print(f"Test fingerprints: {X_test.shape}")
    print(f"  Sparsity: {np.mean(X_test == 0) * 100:.1f}%")
    
    # Check OOV column
    oov_col = X_test[:, -1]
    print(f"  OOV features: mean={np.mean(oov_col):.1f}, max={np.max(oov_col):.0f}")
    
    # Predict
    test_pred = clf_loaded.predict(X_test)
    test_proba = clf_loaded.predict_proba(X_test)[:, 1]
    test_acc = accuracy_score(y_test, test_pred)
    test_auc = roc_auc_score(y_test, test_proba)
    
    print(f"\nTest performance:")
    print(f"  Accuracy: {test_acc:.3f}")
    print(f"  ROC-AUC: {test_auc:.3f}")
    
    print("\n📊 Test Set Predictions:")
    for i, smi in enumerate(test_smi):
        true_label = "BBB+" if y_test[i] == 1 else "BBB-"
        pred_label = "BBB+" if test_pred[i] == 1 else "BBB-"
        correct = "✅" if y_test[i] == test_pred[i] else "❌"
        print(f"  {correct} {smi:45s} True: {true_label} Pred: {pred_label} (p={test_proba[i]:.3f})")
    
    print("\n✅ Inference complete!")


def example_3_production_api():
    """Example 3: Production API for single molecule inference."""
    print("\n\n" + "=" * 80)
    print("EXAMPLE 3: Production API (Single Molecule Inference)")
    print("=" * 80)
    
    from bcfp import FingerprintModel
    
    # Simulate production environment: models already trained and saved
    # For this example, we'll use the models from example 1
    
    print("\n🔮 Production Inference API")
    print("-" * 80)
    
    # Load pre-trained models once at startup
    print("Loading models...")
    fp_model = FingerprintModel.load('fingerprint_model.pkl')
    with open('ml_model.pkl', 'rb') as f:
        ml_model = pickle.load(f)
    print("✅ Models loaded!\n")
    
    def predict_molecule(smiles: str) -> dict:
        """
        Production API: Predict for a single molecule.
        
        Args:
            smiles: SMILES string
            
        Returns:
            dict with prediction and probability
        """
        # Generate fingerprint
        X = fp_model.transform([smiles])
        
        # Predict
        pred = ml_model.predict(X)[0]
        proba = ml_model.predict_proba(X)[0, 1]
        
        return {
            'smiles': smiles,
            'prediction': 'Active' if pred == 1 else 'Inactive',
            'probability': float(proba),
            'confidence': 'High' if abs(proba - 0.5) > 0.3 else 'Low'
        }
    
    # Test API with various molecules
    test_molecules = [
        'CCO',           # Ethanol
        'c1ccccc1',      # Benzene
        'CC(=O)O',       # Acetic acid
        'CCCCCCCCCCCC',  # Dodecane
    ]
    
    print("API Predictions:")
    for smi in test_molecules:
        result = predict_molecule(smi)
        print(f"  {result['smiles']:20s} → {result['prediction']:8s} "
              f"(p={result['probability']:.3f}, {result['confidence']} confidence)")
    
    print("\n✅ API ready for production!")


if __name__ == '__main__':
    # Run all examples
    example_1_basic_workflow()
    example_2_realistic_workflow()
    example_3_production_api()
    
    print("\n\n" + "=" * 80)
    print("🎉 ALL EXAMPLES COMPLETE!")
    print("=" * 80)
    print("\nKey Takeaways:")
    print("  1. Use FingerprintModel for complete train/inference workflow")
    print("  2. Save models after training for production use")
    print("  3. Load models once at startup for efficient inference")
    print("  4. The Sort&Slice vocabulary ensures consistent features")
    print("  5. OOV bucket handles unseen features gracefully")
    print("=" * 80)

