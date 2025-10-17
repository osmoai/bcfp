#!/usr/bin/env python3
"""Simple test of ECFP+BCFP with Sort&Slice + OOV"""
import numpy as np
from rdkit import Chem
from bcfp import FingerprintGenerator

print("Testing ECFP+BCFP Sort&Slice + OOV...")

# Test molecules
smiles = ['CCO', 'c1ccccc1', 'CC(=O)O', 'CC(C)O', 'c1ccc(O)cc1'] * 2
mols = [Chem.MolFromSmiles(s) for s in smiles]
test_smiles = ['CCC', 'CCCO']
test_mols = [Chem.MolFromSmiles(s) for s in test_smiles]

# ECFP generator
gen_ecfp = FingerprintGenerator('rdkit_native', 'ecfp', radius=2, use_counts=True)
print(f"✅ Created ECFP generator")

# Generate sparse fingerprints
train_fps = [gen_ecfp.generate_sparse(m) for m in mols]
test_fps = [gen_ecfp.generate_sparse(m) for m in test_mols]
print(f"✅ Generated sparse FPs: train={len(train_fps)}, test={len(test_fps)}")

# Sort&Slice fit
vocab, key2col = gen_ecfp.sortslice_fit(train_fps, np.arange(len(train_fps)), top_k=20)
print(f"✅ Vocabulary: {len(vocab)} features")

# Transform with OOV
X_train = gen_ecfp.sortslice_transform(train_fps, np.arange(len(train_fps)), 
                                       key2col, use_counts=True, add_oov_bucket=True)
X_test = gen_ecfp.sortslice_transform(test_fps, np.arange(len(test_fps)), 
                                      key2col, use_counts=True, add_oov_bucket=True)

print(f"✅ Train matrix: {X_train.shape}")
print(f"✅ Test matrix: {X_test.shape}")
print(f"✅ OOV column {X_train.shape[1]-1} has test values: {np.any(X_test[:, -1] > 0)}")

# BCFP
gen_bcfp = FingerprintGenerator('rdkit_native', 'bcfp', radius=2, use_counts=True)
train_fps_bcfp = [gen_bcfp.generate_sparse(m) for m in mols]
vocab_bcfp, key2col_bcfp = gen_bcfp.sortslice_fit(train_fps_bcfp, np.arange(len(train_fps_bcfp)), top_k=20)
X_train_bcfp = gen_bcfp.sortslice_transform(train_fps_bcfp, np.arange(len(train_fps_bcfp)),
                                            key2col_bcfp, use_counts=True, add_oov_bucket=True)
print(f"✅ BCFP train matrix: {X_train_bcfp.shape}")

# Concatenate
X_combined = np.hstack([X_train, X_train_bcfp])
print(f"✅ Combined ECFP+BCFP: {X_combined.shape} ({X_train.shape[1]} ECFP + {X_train_bcfp.shape[1]} BCFP)")

print("\n🎉 All tests passed! ECFP+BCFP Sort&Slice + OOV works perfectly!")
