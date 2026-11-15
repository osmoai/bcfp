import struct
import sys
from types import ModuleType, SimpleNamespace

import pytest
from rdkit import Chem

from bcfp import FingerprintGenerator


@pytest.fixture
def mol():
    return Chem.MolFromSmiles("CC(O)c1ccccc1")


def test_cpp_accepts_rdkit_molecule_binary(mol):
    cpp = pytest.importorskip("_bcfp")

    from_mol = cpp.get_bond_morgan_fingerprint_native(mol)
    from_binary = cpp.get_bond_morgan_fingerprint_native(mol.ToBinary())

    assert from_binary == from_mol


def test_cpp_transition_does_not_use_smiles(monkeypatch, mol):
    cpp = pytest.importorskip("_bcfp")

    def fail_if_called(*args, **kwargs):
        raise AssertionError("MolToSmiles must not be used by the C++ transition")

    monkeypatch.setattr(Chem, "MolToSmiles", fail_if_called)

    assert cpp.get_bond_morgan_fingerprint_xxhash(mol)


@pytest.mark.parametrize("fp_type", ["ecfp", "bcfp"])
@pytest.mark.parametrize("hash_func", ["rdkit_native", "xxhash", "blake3"])
def test_generate_bytes_matches_sparse_feature_folding(mol, fp_type, hash_func):
    gen = FingerprintGenerator(hash_func, fp_type, n_bits=257)

    sparse = gen.generate_sparse(mol)
    fingerprint = gen.generate_bytes(mol)
    on_bits = {
        byte_index * 8 + bit_index
        for byte_index, value in enumerate(fingerprint)
        for bit_index in range(8)
        if value & (1 << bit_index)
    }

    assert len(fingerprint) == 33
    assert on_bits == {feature % 257 for feature in sparse}


@pytest.mark.parametrize(
    ("layout", "record_format", "record_size"),
    [
        ("32_32", "<II", 8),
        ("64_32_packed", "<QI", 12),
        ("64_32", "<QI4x", 16),
    ],
)
def test_count_buffers_match_sparse_features(mol, layout, record_format, record_size):
    gen = FingerprintGenerator("xxhash", "bcfp", use_counts=True)

    sparse = gen.generate_sparse(mol)
    buffer = gen.generate_count_buffer(mol, layout)

    assert len(buffer) == len(sparse) * record_size
    assert list(struct.iter_unpack(record_format, buffer)) == sorted(sparse.items())


@pytest.mark.parametrize("fp_type", ["ecfp", "bcfp"])
@pytest.mark.parametrize("hash_func", ["rdkit_native", "xxhash", "blake3"])
def test_count_buffer_presence_mode_has_unit_counts(mol, fp_type, hash_func):
    gen = FingerprintGenerator(hash_func, fp_type, use_counts=False)

    records = list(struct.iter_unpack("<QI", gen.generate_count_buffer(mol)))

    assert records
    assert {count for _, count in records} == {1}


def test_generate_chemfp_count_fingerprint_uses_matching_constructor(monkeypatch, mol):
    received = {}

    class FakeCountFingerprint:
        @staticmethod
        def from_buffer_64_32_packed(buffer):
            received["buffer"] = buffer
            return "count-fingerprint"

    chemfp = ModuleType("chemfp")
    chemfp.countops = SimpleNamespace(CountFingerprint=FakeCountFingerprint)
    monkeypatch.setitem(sys.modules, "chemfp", chemfp)

    gen = FingerprintGenerator("blake3", "bcfp")

    assert gen.generate_chemfp_count_fingerprint(mol) == "count-fingerprint"
    assert received["buffer"] == gen.generate_count_buffer(mol)


def test_generate_chemfp_count_fingerprint_live(mol):
    pytest.importorskip("chemfp")
    gen = FingerprintGenerator("blake3", "bcfp")

    chemfp_fp = gen.generate_chemfp_count_fingerprint(mol)

    assert list(chemfp_fp) == sorted(gen.generate_sparse(mol).items())


def test_empty_and_invalid_count_buffers():
    gen = FingerprintGenerator("xxhash", "bcfp", n_bits=17)

    assert gen.generate_bytes(None) == b"\0\0\0"
    assert gen.generate_count_buffer(None) == b""
    with pytest.raises(ValueError, match="layout"):
        gen.generate_count_buffer(Chem.MolFromSmiles("CC"), "bad")
