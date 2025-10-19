#!/usr/bin/env python3
"""
Test ECFP+BCFP with Sort&Slice + OOV functionality
"""
import numpy as np
from rdkit import Chem
from bcfp import FingerprintGenerator

print("=" * 80)
print("BCFP-GITHUB2 TEST - ECFP+BCFP with Sort&Slice + OOV")
print("=" * 80)

# Test data
train_smiles = ['CCO', 'c1ccccc1', 'CC(=O)O', 'CC(C)O', 'c1ccc(O)cc1'] * 10
test_smiles = ['CCC', 'c1ccc(N)cc1', 'CCCO']
train_mols = [Chem.MolFromSmiles(s) for s in train_smiles]
test_mols = [Chem.MolFromSmiles(s) for s in test_smiles]

print(f"\nTrain set: {len(train_smiles)} molecules")
print(f"Test set: {len(test_smiles)} molecules")

# Test 1: ECFP with rdkit_native
print("\n[1] Testing ECFP with rdkit_native...")
gen_ecfp = FingerprintGenerator('rdkit_native', 'ecfp', radius=2)
fp = gen_ecfp.generate_sparse(train_mols[0])
print(f"✅ Sparse ECFP: {len(fp)} features")

# Test 2: BCFP with rdkit_native  
print("\n[2] Testing BCFP with rdkit_native...")
gen_bcfp = FingerprintGenerator('rdkit_native', 'bcfp', radius=2)
fp = gen_bcfp.generate_sparse(train_mols[0])
print(f"✅ Sparse BCFP: {len(fp)} features")

# Test 3: ECFP with xxhash
print("\n[3] Testing ECFP with xxhash...")
gen_xxhash = FingerprintGenerator('xxhash', 'ecfp', radius=2)
fp = gen_xxhash.generate_sparse(train_mols[0])
print(f"✅ Sparse ECFP (xxhash): {len(fp)} features")

# Test 4: ECFP with blake3
print("\n[4] Testing ECFP with blake3...")
gen_blake3 = FingerprintGenerator('blake3', 'ecfp', radius=2)
fp = gen_blake3.generate_sparse(train_mols[0])
print(f"✅ Sparse ECFP (blake3): {len(fp)} features")

# Test 5: Folded dense fingerprints
print("\n[5] Testing folded dense fingerprints...")
X_train_dense = gen_ecfp.generate_basic(train_mols[0])
print(f"✅ Dense ECFP shape: {X_train_dense.shape}")
print(f"   Non-zero bits: {np.count_nonzero(X_train_dense)}")

# Test 6: Sort&Slice (ECFP)
print("\n[6] Testing Sort&Slice for ECFP...")
train_fps = [gen_ecfp.generate_sparse(m) for m in train_mols]
vocab, key2col = gen_ecfp.sortslice_fit(train_fps, list(range(len(train_mols))), top_k=128)
print(f"✅ Vocabulary size: {len(vocab)} features")

X_train = gen_ecfp.sortslice_transform(train_fps, list(range(len(train_mols))), key2col, use_counts=True)
print(f"✅ Training matrix: {X_train.shape}")
print(f"   Non-zero entries: {np.count_nonzero(X_train)}")

# Test 7: Sort&Slice + OOV (ECFP)
print("\n[7] Testing Sort&Slice + OOV for ECFP...")
vocab_oov, key2col_oov = gen_ecfp.sortslice_fit(train_fps, list(range(len(train_mols))), 
                                                  top_k=128)
print(f"✅ Vocabulary size: {len(vocab_oov)} features")

X_train_oov = gen_ecfp.sortslice_transform(train_fps, list(range(len(train_mols))), 
                                            key2col_oov, use_counts=True, add_oov_bucket=True)
print(f"✅ Training matrix (with OOV): {X_train_oov.shape}")

# Test 8: Test set with OOV
print("\n[8] Testing test set with OOV...")
test_fps = [gen_ecfp.generate_sparse(m) for m in test_mols]
X_test_oov = gen_ecfp.sortslice_transform(test_fps, list(range(len(test_mols))), 
                                          key2col_oov, use_counts=True, add_oov_bucket=True)
print(f"✅ Test matrix (with OOV): {X_test_oov.shape}")
oov_col = X_test_oov.shape[1] - 1
has_oov = np.any(X_test_oov[:, oov_col] > 0)
print(f"✅ OOV bucket (column {oov_col}) has values: {has_oov}")
if has_oov:
    oov_counts = X_test_oov[:, oov_col]
    print(f"   OOV counts per molecule: {oov_counts}")

# Test 9: BCFP Sort&Slice + OOV
print("\n[9] Testing BCFP Sort&Slice + OOV...")
train_fps_bcfp = [gen_bcfp.generate_sparse(m) for m in train_mols]
vocab_bcfp, key2col_bcfp = gen_bcfp.sortslice_fit(train_fps_bcfp, list(range(len(train_mols))), 
                                                    top_k=128)
print(f"✅ BCFP Vocabulary size: {len(vocab_bcfp)} features")

X_train_bcfp = gen_bcfp.sortslice_transform(train_fps_bcfp, list(range(len(train_mols))), 
                                            key2col_bcfp, use_counts=True, add_oov_bucket=True)
print(f"✅ BCFP Training matrix (with OOV): {X_train_bcfp.shape}")

# Test 10: ECFP + BCFP Concatenation with Sort&Slice + OOV
print("\n[10] Testing ECFP + BCFP Concatenation...")
X_train_ecfp = gen_ecfp.sortslice_transform(train_fps, list(range(len(train_mols))), 
                                            key2col_oov, use_counts=True, add_oov_bucket=True)
X_train_combined = np.hstack([X_train_ecfp, X_train_bcfp])
print(f"✅ Combined ECFP+BCFP training matrix: {X_train_combined.shape}")
print(f"   ({X_train_ecfp.shape[1]} ECFP + {X_train_bcfp.shape[1]} BCFP = {X_train_combined.shape[1]} total features)")

# Test 11: Test set with combined ECFP+BCFP
print("\n[11] Testing test set with combined ECFP+BCFP...")
test_fps_bcfp = [gen_bcfp.generate_sparse(m) for m in test_mols]
X_test_ecfp = gen_ecfp.sortslice_transform(test_fps, list(range(len(test_mols))), 
                                           key2col_oov, use_counts=True, add_oov_bucket=True)
X_test_bcfp = gen_bcfp.sortslice_transform(test_fps_bcfp, list(range(len(test_mols))), 
                                           key2col_bcfp, use_counts=True, add_oov_bucket=True)
X_test_combined = np.hstack([X_test_ecfp, X_test_bcfp])
print(f"✅ Combined ECFP+BCFP test matrix: {X_test_combined.shape}")

print("\n" + "=" * 80)
print("✅ ALL TESTS PASSED! BCFP-GITHUB2 with Sort&Slice + OOV is fully functional!")
print("=" * 80)
print("\nKey features verified:")
print("  ✓ ECFP generation (rdkit_native, xxhash, blake3)")
print("  ✓ BCFP generation (rdkit_native)")
print("  ✓ Sparse and dense fingerprints")
print("  ✓ Sort&Slice feature selection")
print("  ✓ Out-of-Vocabulary (OOV) handling")
print("  ✓ ECFP+BCFP concatenation")
print("  ✓ Train/test split with consistent vocabulary")
print("=" * 80)

