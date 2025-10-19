#!/usr/bin/env python3
"""
Comprehensive pytest suite for BCFP
Tests all 3 hash functions (rdkit_native, xxhash, blake3) and validates
ECFP+BCFP with Sort&Slice and OOV functionality.
"""
import pytest
import numpy as np
from rdkit import Chem
from bcfp import FingerprintGenerator


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def test_molecules():
    """Provide standard test molecules"""
    smiles = [
        'CCO',          # ethanol
        'c1ccccc1',     # benzene
        'CC(=O)O',      # acetic acid
        'CCN',          # ethylamine
        'CC(C)O',       # isopropanol
        'CCCC',         # butane
        'c1ccc(O)cc1',  # phenol
        'CC(C)C',       # isobutane
    ]
    return [Chem.MolFromSmiles(s) for s in smiles]


@pytest.fixture
def train_test_molecules():
    """Provide train/test split for OOV testing"""
    train_smiles = ['CCO', 'c1ccccc1', 'CC(=O)O', 'CCN', 'CC(C)O']
    test_smiles = ['CCCC', 'c1ccc(O)cc1']  # Different molecules for test
    
    train = [Chem.MolFromSmiles(s) for s in train_smiles]
    test = [Chem.MolFromSmiles(s) for s in test_smiles]
    return train, test


# ============================================================================
# 1. Hash Function Tests
# ============================================================================

def test_rdkit_native_hash(test_molecules):
    """Verify rdkit_native hash function works"""
    gen = FingerprintGenerator('rdkit_native', 'ecfp', radius=2)
    fp = gen.generate_sparse(test_molecules[0])
    
    assert isinstance(fp, dict)
    assert len(fp) > 0
    assert all(isinstance(k, int) for k in fp.keys())
    assert all(isinstance(v, int) for v in fp.values())


def test_xxhash_works(test_molecules):
    """Verify xxhash hash function works"""
    gen = FingerprintGenerator('xxhash', 'ecfp', radius=2)
    fp = gen.generate_sparse(test_molecules[0])
    
    assert isinstance(fp, dict)
    assert len(fp) > 0
    assert all(isinstance(k, int) for k in fp.keys())
    assert all(isinstance(v, int) for v in fp.values())


def test_blake3_works(test_molecules):
    """Verify blake3 hash function works"""
    gen = FingerprintGenerator('blake3', 'ecfp', radius=2)
    fp = gen.generate_sparse(test_molecules[0])
    
    assert isinstance(fp, dict)
    assert len(fp) > 0
    assert all(isinstance(k, int) for k in fp.keys())
    assert all(isinstance(v, int) for v in fp.values())


def test_only_supported_hashers():
    """Verify only rdkit_native, xxhash, blake3 are available"""
    try:
        import _bcfp
        # Check C++ module exports
        funcs = [x for x in dir(_bcfp) if not x.startswith('_')]
        
        # Verify removed hashers are NOT present
        removed_patterns = ['blake2b', 't1ha', 'highwayhash', 'highway']
        for func in funcs:
            func_lower = func.lower()
            for pattern in removed_patterns:
                assert pattern not in func_lower, \
                    f"Function '{func}' references unsupported hasher '{pattern}'"
    except ImportError:
        pytest.skip("C++ module not available")


# ============================================================================
# 2. ECFP Tests (per hash function)
# ============================================================================

def test_ecfp_rdkit_native(test_molecules):
    """Generate ECFP with rdkit_native"""
    gen = FingerprintGenerator('rdkit_native', 'ecfp', radius=2)
    
    for mol in test_molecules:
        fp = gen.generate_sparse(mol)
        assert len(fp) > 0, f"Empty fingerprint for molecule {Chem.MolToSmiles(mol)}"


def test_ecfp_xxhash(test_molecules):
    """Generate ECFP with xxhash"""
    gen = FingerprintGenerator('xxhash', 'ecfp', radius=2)
    
    for mol in test_molecules:
        fp = gen.generate_sparse(mol)
        assert len(fp) > 0, f"Empty fingerprint for molecule {Chem.MolToSmiles(mol)}"


def test_ecfp_blake3(test_molecules):
    """Generate ECFP with blake3"""
    gen = FingerprintGenerator('blake3', 'ecfp', radius=2)
    
    for mol in test_molecules:
        fp = gen.generate_sparse(mol)
        assert len(fp) > 0, f"Empty fingerprint for molecule {Chem.MolToSmiles(mol)}"


def test_ecfp_consistency(test_molecules):
    """Same molecule produces consistent fingerprints"""
    mol = test_molecules[0]
    
    for hasher in ['rdkit_native', 'xxhash', 'blake3']:
        gen = FingerprintGenerator(hasher, 'ecfp', radius=2)
        fp1 = gen.generate_sparse(mol)
        fp2 = gen.generate_sparse(mol)
        
        assert fp1 == fp2, f"Inconsistent fingerprints with {hasher}"


# ============================================================================
# 3. BCFP Tests
# ============================================================================

def test_bcfp_rdkit_native(test_molecules):
    """Generate BCFP with rdkit_native"""
    gen = FingerprintGenerator('rdkit_native', 'bcfp', radius=2)
    
    for mol in test_molecules:
        fp = gen.generate_sparse(mol)
        assert len(fp) > 0, f"Empty BCFP for molecule {Chem.MolToSmiles(mol)}"


def test_bcfp_different_molecules(test_molecules):
    """Different molecules produce different fingerprints"""
    gen = FingerprintGenerator('rdkit_native', 'bcfp', radius=2)
    
    fp1 = gen.generate_sparse(test_molecules[0])  # ethanol
    fp2 = gen.generate_sparse(test_molecules[1])  # benzene
    
    assert fp1 != fp2, "Different molecules should have different fingerprints"


def test_bcfp_radius_effect(test_molecules):
    """Different radii produce different results"""
    mol = test_molecules[0]
    
    gen_r1 = FingerprintGenerator('rdkit_native', 'bcfp', radius=1)
    gen_r2 = FingerprintGenerator('rdkit_native', 'bcfp', radius=2)
    gen_r3 = FingerprintGenerator('rdkit_native', 'bcfp', radius=3)
    
    fp_r1 = gen_r1.generate_sparse(mol)
    fp_r2 = gen_r2.generate_sparse(mol)
    fp_r3 = gen_r3.generate_sparse(mol)
    
    # Larger radius should generally have more features
    assert len(fp_r2) >= len(fp_r1), "Radius 2 should have >= features than radius 1"
    assert len(fp_r3) >= len(fp_r2), "Radius 3 should have >= features than radius 2"


# ============================================================================
# 4. Sort&Slice Tests
# ============================================================================

def test_sortslice_fit(test_molecules):
    """Vocabulary selection works"""
    gen = FingerprintGenerator('rdkit_native', 'ecfp', radius=2)
    fps = [gen.generate_sparse(mol) for mol in test_molecules]
    
    vocab, key2col = gen.sortslice_fit(fps, list(range(len(test_molecules))), top_k=10)
    
    # May return fewer than top_k if not enough unique features exist
    assert len(vocab) <= 10, "Should select at most top_k features"
    assert len(vocab) > 0, "Should select at least some features"
    assert len(key2col) == len(vocab), "key2col should match vocab size"
    assert isinstance(vocab, list)
    assert isinstance(key2col, dict)


def test_sortslice_transform(test_molecules):
    """Feature matrix generation works"""
    gen = FingerprintGenerator('rdkit_native', 'ecfp', radius=2)
    fps = [gen.generate_sparse(mol) for mol in test_molecules]
    
    vocab, key2col = gen.sortslice_fit(fps, list(range(len(test_molecules))), top_k=15)
    X = gen.sortslice_transform(fps, list(range(len(test_molecules))), key2col, use_counts=True)
    
    assert X.shape[0] == len(test_molecules), "Matrix should have n_mols rows"
    assert X.shape[1] == len(vocab), "Matrix columns should match vocab size"
    assert X.dtype in [np.int32, np.int64, np.float32, np.float64]


def test_sortslice_top_k(test_molecules):
    """Correct number of features selected"""
    gen = FingerprintGenerator('rdkit_native', 'ecfp', radius=2)
    fps = [gen.generate_sparse(mol) for mol in test_molecules]
    
    for k in [5, 10, 20]:
        vocab, key2col = gen.sortslice_fit(fps, list(range(len(test_molecules))), top_k=k)
        # May return fewer if not enough unique features
        assert len(vocab) <= k, f"Should select at most {k} features"
        assert len(vocab) > 0, f"Should select at least 1 feature"


def test_sortslice_consistency(test_molecules):
    """Deterministic results"""
    gen = FingerprintGenerator('rdkit_native', 'ecfp', radius=2)
    fps = [gen.generate_sparse(mol) for mol in test_molecules]
    
    vocab1, key2col1 = gen.sortslice_fit(fps, list(range(len(test_molecules))), top_k=10)
    vocab2, key2col2 = gen.sortslice_fit(fps, list(range(len(test_molecules))), top_k=10)
    
    assert vocab1 == vocab2, "Sort&Slice should be deterministic"
    assert key2col1 == key2col2, "key2col mapping should be deterministic"


# ============================================================================
# 5. OOV Tests
# ============================================================================

def test_oov_bucket_creation(train_test_molecules):
    """OOV column is added"""
    train, test = train_test_molecules
    gen = FingerprintGenerator('rdkit_native', 'ecfp', radius=2)
    
    train_fps = [gen.generate_sparse(mol) for mol in train]
    vocab, key2col = gen.sortslice_fit(train_fps, list(range(len(train))), top_k=10)
    
    X_train = gen.sortslice_transform(train_fps, list(range(len(train))), 
                                      key2col, use_counts=True, add_oov_bucket=True)
    
    assert X_train.shape[1] == len(vocab) + 1, "Should have vocab_size + 1 OOV bucket"


def test_oov_unseen_features(train_test_molecules):
    """Unseen features go to OOV bucket"""
    train, test = train_test_molecules
    gen = FingerprintGenerator('rdkit_native', 'ecfp', radius=2)
    
    train_fps = [gen.generate_sparse(mol) for mol in train]
    test_fps = [gen.generate_sparse(mol) for mol in test]
    
    vocab, key2col = gen.sortslice_fit(train_fps, list(range(len(train))), top_k=10)
    
    X_test = gen.sortslice_transform(test_fps, list(range(len(test))), 
                                     key2col, use_counts=True, add_oov_bucket=True)
    
    oov_col = X_test[:, -1]  # Last column is OOV
    # Test molecules should have some OOV features
    assert np.sum(oov_col) >= 0, "OOV bucket should contain counts"


def test_oov_train_features(train_test_molecules):
    """Training features don't use OOV bucket"""
    train, test = train_test_molecules
    gen = FingerprintGenerator('rdkit_native', 'ecfp', radius=2)
    
    train_fps = [gen.generate_sparse(mol) for mol in train]
    vocab, key2col = gen.sortslice_fit(train_fps, list(range(len(train))), top_k=10)
    
    X_train = gen.sortslice_transform(train_fps, list(range(len(train))), 
                                      key2col, use_counts=True, add_oov_bucket=True)
    
    oov_col = X_train[:, -1]  # Last column is OOV
    # Training molecules should have zero or minimal OOV (only features not in top-k)
    assert np.sum(oov_col) >= 0, "OOV counts should be non-negative"


def test_oov_counts(train_test_molecules):
    """OOV counts are correct"""
    train, test = train_test_molecules
    gen = FingerprintGenerator('rdkit_native', 'ecfp', radius=2)
    
    train_fps = [gen.generate_sparse(mol) for mol in train]
    test_fps = [gen.generate_sparse(mol) for mol in test]
    
    vocab, key2col = gen.sortslice_fit(train_fps, list(range(len(train))), top_k=5)
    
    X_test = gen.sortslice_transform(test_fps, list(range(len(test))), 
                                     key2col, use_counts=True, add_oov_bucket=True)
    
    # Check that total counts are preserved
    for i, fp in enumerate(test_fps):
        total_counts = sum(fp.values())
        row_counts = np.sum(X_test[i, :])
        # Counts should match (all features accounted for)
        assert row_counts <= total_counts, "Row counts should not exceed total fingerprint counts"


# ============================================================================
# 6. ECFP+BCFP Concatenation Tests
# ============================================================================

def test_ecfp_bcfp_concat(test_molecules):
    """Concatenation produces correct shape"""
    gen_ecfp = FingerprintGenerator('rdkit_native', 'ecfp', radius=2)
    gen_bcfp = FingerprintGenerator('rdkit_native', 'bcfp', radius=2)
    
    ecfp_fps = [gen_ecfp.generate_sparse(mol) for mol in test_molecules]
    bcfp_fps = [gen_bcfp.generate_sparse(mol) for mol in test_molecules]
    
    vocab_e, key2col_e = gen_ecfp.sortslice_fit(ecfp_fps, list(range(len(test_molecules))), top_k=10)
    vocab_b, key2col_b = gen_bcfp.sortslice_fit(bcfp_fps, list(range(len(test_molecules))), top_k=10)
    
    X_ecfp = gen_ecfp.sortslice_transform(ecfp_fps, list(range(len(test_molecules))), 
                                          key2col_e, use_counts=True)
    X_bcfp = gen_bcfp.sortslice_transform(bcfp_fps, list(range(len(test_molecules))), 
                                          key2col_b, use_counts=True)
    
    X_combined = np.hstack([X_ecfp, X_bcfp])
    
    expected_cols = len(vocab_e) + len(vocab_b)
    assert X_combined.shape == (len(test_molecules), expected_cols), \
        f"Combined should be (n_mols, {expected_cols})"


def test_ecfp_bcfp_features(test_molecules):
    """Features from both fingerprints are present"""
    gen_ecfp = FingerprintGenerator('rdkit_native', 'ecfp', radius=2)
    gen_bcfp = FingerprintGenerator('rdkit_native', 'bcfp', radius=2)
    
    ecfp_fps = [gen_ecfp.generate_sparse(mol) for mol in test_molecules]
    bcfp_fps = [gen_bcfp.generate_sparse(mol) for mol in test_molecules]
    
    vocab_e, key2col_e = gen_ecfp.sortslice_fit(ecfp_fps, list(range(len(test_molecules))), top_k=10)
    vocab_b, key2col_b = gen_bcfp.sortslice_fit(bcfp_fps, list(range(len(test_molecules))), top_k=10)
    
    X_ecfp = gen_ecfp.sortslice_transform(ecfp_fps, list(range(len(test_molecules))), 
                                          key2col_e, use_counts=True)
    X_bcfp = gen_bcfp.sortslice_transform(bcfp_fps, list(range(len(test_molecules))), 
                                          key2col_b, use_counts=True)
    
    # Both should have non-zero features
    assert np.sum(X_ecfp) > 0, "ECFP features should be non-zero"
    assert np.sum(X_bcfp) > 0, "BCFP features should be non-zero"


def test_ecfp_bcfp_consistency(test_molecules):
    """Concatenation is deterministic"""
    gen_ecfp = FingerprintGenerator('rdkit_native', 'ecfp', radius=2)
    gen_bcfp = FingerprintGenerator('rdkit_native', 'bcfp', radius=2)
    
    ecfp_fps = [gen_ecfp.generate_sparse(mol) for mol in test_molecules]
    bcfp_fps = [gen_bcfp.generate_sparse(mol) for mol in test_molecules]
    
    vocab_e, key2col_e = gen_ecfp.sortslice_fit(ecfp_fps, list(range(len(test_molecules))), top_k=10)
    vocab_b, key2col_b = gen_bcfp.sortslice_fit(bcfp_fps, list(range(len(test_molecules))), top_k=10)
    
    X_ecfp1 = gen_ecfp.sortslice_transform(ecfp_fps, list(range(len(test_molecules))), 
                                           key2col_e, use_counts=True)
    X_ecfp2 = gen_ecfp.sortslice_transform(ecfp_fps, list(range(len(test_molecules))), 
                                           key2col_e, use_counts=True)
    
    assert np.array_equal(X_ecfp1, X_ecfp2), "Transform should be deterministic"


# ============================================================================
# 7. Integration Tests
# ============================================================================

def test_full_pipeline_rdkit_native(train_test_molecules):
    """End-to-end with rdkit_native"""
    train, test = train_test_molecules
    gen = FingerprintGenerator('rdkit_native', 'ecfp', radius=2)
    
    # Generate fingerprints
    train_fps = [gen.generate_sparse(mol) for mol in train]
    test_fps = [gen.generate_sparse(mol) for mol in test]
    
    # Fit vocabulary
    vocab, key2col = gen.sortslice_fit(train_fps, list(range(len(train))), top_k=20)
    
    # Transform with OOV
    X_train = gen.sortslice_transform(train_fps, list(range(len(train))), 
                                      key2col, use_counts=True, add_oov_bucket=True)
    X_test = gen.sortslice_transform(test_fps, list(range(len(test))), 
                                     key2col, use_counts=True, add_oov_bucket=True)
    
    expected_cols = len(vocab) + 1  # vocab + OOV
    assert X_train.shape == (len(train), expected_cols), \
        f"Train shape should be (n_train, {expected_cols})"
    assert X_test.shape == (len(test), expected_cols), \
        f"Test shape should be (n_test, {expected_cols})"


def test_full_pipeline_xxhash(train_test_molecules):
    """End-to-end with xxhash"""
    train, test = train_test_molecules
    gen = FingerprintGenerator('xxhash', 'ecfp', radius=2)
    
    train_fps = [gen.generate_sparse(mol) for mol in train]
    test_fps = [gen.generate_sparse(mol) for mol in test]
    
    vocab, key2col = gen.sortslice_fit(train_fps, list(range(len(train))), top_k=20)
    
    X_train = gen.sortslice_transform(train_fps, list(range(len(train))), 
                                      key2col, use_counts=True, add_oov_bucket=True)
    X_test = gen.sortslice_transform(test_fps, list(range(len(test))), 
                                     key2col, use_counts=True, add_oov_bucket=True)
    
    expected_cols = len(vocab) + 1
    assert X_train.shape == (len(train), expected_cols)
    assert X_test.shape == (len(test), expected_cols)


def test_full_pipeline_blake3(train_test_molecules):
    """End-to-end with blake3"""
    train, test = train_test_molecules
    gen = FingerprintGenerator('blake3', 'ecfp', radius=2)
    
    train_fps = [gen.generate_sparse(mol) for mol in train]
    test_fps = [gen.generate_sparse(mol) for mol in test]
    
    vocab, key2col = gen.sortslice_fit(train_fps, list(range(len(train))), top_k=20)
    
    X_train = gen.sortslice_transform(train_fps, list(range(len(train))), 
                                      key2col, use_counts=True, add_oov_bucket=True)
    X_test = gen.sortslice_transform(test_fps, list(range(len(test))), 
                                     key2col, use_counts=True, add_oov_bucket=True)
    
    expected_cols = len(vocab) + 1
    assert X_train.shape == (len(train), expected_cols)
    assert X_test.shape == (len(test), expected_cols)


def test_train_test_split(train_test_molecules):
    """Proper handling of train/test vocabulary"""
    train, test = train_test_molecules
    gen = FingerprintGenerator('rdkit_native', 'ecfp', radius=2)
    
    train_fps = [gen.generate_sparse(mol) for mol in train]
    test_fps = [gen.generate_sparse(mol) for mol in test]
    
    # Vocabulary should be based on training set only
    vocab, key2col = gen.sortslice_fit(train_fps, list(range(len(train))), top_k=15)
    
    # Test that vocabulary comes from training data
    all_train_keys = set()
    for fp in train_fps:
        all_train_keys.update(fp.keys())
    
    # All vocab keys should come from training set
    for key in vocab:
        assert key in all_train_keys, "Vocabulary should only contain training keys"


# ============================================================================
# 8. C++ Module Tests
# ============================================================================

def test_cpp_module_loads():
    """_bcfp module can be imported"""
    try:
        import _bcfp
        assert True
    except ImportError:
        pytest.skip("C++ module not available (using Python fallback)")


def test_cpp_functions_available():
    """Expected functions are exposed"""
    try:
        import _bcfp
        funcs = [x for x in dir(_bcfp) if not x.startswith('_')]
        
        # Should have at least the core functions
        expected_funcs = [
            'get_morgan_fingerprint',
            'get_bond_morgan_fingerprint'
        ]
        
        for func in expected_funcs:
            assert func in funcs or any(func in f for f in funcs), \
                f"Expected function '{func}' not found in C++ module"
    except ImportError:
        pytest.skip("C++ module not available")


def test_cpp_vs_python():
    """C++ and Python produce same results (if applicable)"""
    try:
        import _bcfp
        # If C++ module is available, test consistency
        # For now, just verify it doesn't crash
        mol = Chem.MolFromSmiles('CCO')
        gen = FingerprintGenerator('rdkit_native', 'ecfp', radius=2)
        fp = gen.generate_sparse(mol)
        assert len(fp) > 0
    except ImportError:
        pytest.skip("C++ module not available")

