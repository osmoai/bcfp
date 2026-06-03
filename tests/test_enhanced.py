"""Tests for the sklearn-style BCFPEnhanced wrapper (fit/transform + Sort&Slice + OOV)."""
import numpy as np
import pytest

pytest.importorskip("_bcfp")
from bcfp import BCFPEnhanced

TRAIN = ["CCO", "CCC", "CCCC", "c1ccccc1", "CC(=O)O", "CCN", "CCCN", "c1ccc(O)cc1",
         "c1ccc(C)cc1", "C1CCCCC1", "CC(C)C", "CCCCO"] * 5
TEST = ["CCCCCCC", "CC(C)(C)C", "c1ccc(Cl)cc1", "CCO", "c1ccccc1"]


def test_baseline_transform_no_fit_needed():
    # top_k=None -> plain folding to n_bits, no vocabulary, no fit required
    gen = BCFPEnhanced(fp_type="ecfp", radius=2, n_bits=1024, top_k=None)
    X = gen.transform(TRAIN)
    assert X.shape == (len(TRAIN), 1024)


def test_sortslice_fit_transform_shapes():
    gen = BCFPEnhanced(fp_type="ecfp", radius=2, n_bits=2048, top_k=64, include_oov=False)
    Xtr = gen.fit_transform(TRAIN)
    Xte = gen.transform(TEST)
    assert Xtr.shape[1] == Xte.shape[1] <= 64          # <=64 (min_df filter may select fewer)
    assert Xtr.shape[0] == len(TRAIN) and Xte.shape[0] == len(TEST)
    assert gen.selected_indices_ is not None


def test_oov_bucket_adds_column_and_counts_unseen():
    gen = BCFPEnhanced(fp_type="ecfp", radius=2, top_k=32, include_oov=True).fit(TRAIN)
    Xtr = gen.transform(TRAIN)
    Xte = gen.transform(TEST)
    base = len(gen.key2col_)
    assert Xtr.shape[1] == base + 1 and Xte.shape[1] == base + 1   # one OOV column
    # novel test molecules should land some mass in the OOV bucket; a known train molecule none
    assert Xte[:, -1].sum() > 0


def test_transform_before_fit_raises_when_topk_set():
    with pytest.raises(RuntimeError):
        BCFPEnhanced(fp_type="ecfp", top_k=64).transform(TEST)


def test_single_str_rejected():
    with pytest.raises(TypeError):
        BCFPEnhanced(top_k=None).transform("CCO")


@pytest.mark.parametrize("hash_func", ["rdkit_native", "xxhash", "blake3"])
def test_all_three_hashes(hash_func):
    gen = BCFPEnhanced(fp_type="ecfp", radius=2, n_bits=512, top_k=None, hash_func=hash_func)
    X = gen.transform(TRAIN[:6])
    assert X.shape == (6, 512) and X.sum() > 0


@pytest.mark.parametrize("fp_type", ["ecfp", "bcfp"])
def test_ecfp_and_bcfp(fp_type):
    gen = BCFPEnhanced(fp_type=fp_type, radius=2, n_bits=512, top_k=None)
    assert gen.transform(TRAIN[:6]).shape == (6, 512)
