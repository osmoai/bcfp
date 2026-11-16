"""Tests for the folding options (modulo / remix) and the Sort&Slice + OOV default."""
import numpy as np
import pytest
from rdkit import Chem

pytest.importorskip("_bcfp")
from bcfp import BCFPEnhanced, FingerprintGenerator
from bcfp.fingerprints import remix_key

TRAIN = ["CCO", "CCC", "CCCC", "c1ccccc1", "CC(=O)O", "CCN", "CCCN", "c1ccc(O)cc1",
         "c1ccc(C)cc1", "C1CCCCC1", "CC(C)C", "CCCCO"] * 5
TEST = ["CCCCCCC", "CC(C)(C)C", "c1ccc(Cl)cc1", "CCO", "c1ccccc1"]


def test_remix_key_is_splitmix64_and_salt_dependent():
    # splitmix64(0) reference value of the standard generator's first output
    assert remix_key(0) == 0xE220A8397B1DCDAF
    assert remix_key(12345, salt=1) != remix_key(12345, salt=2)
    assert 0 <= remix_key(2**32 - 1) < 2**64


def test_modulo_is_still_the_default_fold():
    mol = Chem.MolFromSmiles("c1ccc(O)cc1")
    gen = FingerprintGenerator("rdkit_native", "ecfp", radius=2, n_bits=256)
    sparse = gen.generate_sparse(mol)
    expected = np.zeros(256, dtype=np.int32)
    for k, c in sparse.items():
        expected[k % 256] += c
    assert np.array_equal(gen.generate_basic(mol), expected)


@pytest.mark.parametrize("fp_type", ["ecfp", "bcfp"])
def test_remix_fold_preserves_counts_and_moves_bins(fp_type):
    mol = Chem.MolFromSmiles("CC(=O)Nc1ccc(O)cc1")
    mod = FingerprintGenerator("rdkit_native", fp_type, radius=2, n_bits=128)
    rmx = FingerprintGenerator("rdkit_native", fp_type, radius=2, n_bits=128, fold="remix")
    a, b = mod.generate_basic(mol), rmx.generate_basic(mol)
    assert a.sum() == b.sum() == sum(mod.generate_sparse(mol).values())
    assert not np.array_equal(a, b)


def test_invalid_fold_rejected():
    with pytest.raises(ValueError):
        FingerprintGenerator("rdkit_native", "ecfp", fold="xor")


def test_default_is_sortslice_with_oov_at_n_bits_width():
    gen = BCFPEnhanced(fp_type="ecfp", radius=2, n_bits=64, min_df=1)
    assert gen.top_k == 63 and gen.include_oov
    Xtr = gen.fit_transform(TRAIN)
    Xte = gen.transform(TEST)
    assert Xtr.shape[1] == Xte.shape[1] == len(gen.key2col_) + 1 <= 64
    # OOV bucket = total count minus counts that hit the vocabulary
    total = [sum(gen.gen.generate_sparse(Chem.MolFromSmiles(s)).values()) for s in TEST]
    assert np.array_equal(Xte.sum(1), total)


def test_default_requires_fit():
    with pytest.raises(RuntimeError):
        BCFPEnhanced().transform(TEST)


def test_auto_without_oov_keeps_n_bits_keys():
    assert BCFPEnhanced(n_bits=128, include_oov=False).top_k == 128


def test_topk_none_folds_with_remix_without_fit():
    X = BCFPEnhanced(n_bits=256, top_k=None, fold="remix").transform(TEST)
    assert X.shape == (len(TEST), 256) and X.sum() > 0
