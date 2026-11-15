#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include "bcfp.h"
#include <rdkit/GraphMol/MolPickler.h>

#include <algorithm>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace nb = nanobind;

namespace {

using SparseFingerprint = std::unordered_map<uint32_t, int>;

std::unique_ptr<RDKit::ROMol> MolFromPython(const nb::object& mol_obj) {
    try {
        nb::bytes binary = nb::isinstance<nb::bytes>(mol_obj)
            ? nb::borrow<nb::bytes>(mol_obj)
            : nb::bytes(mol_obj.attr("ToBinary")());
        std::string pickle(
            static_cast<const char*>(binary.data()), binary.size()
        );
        auto mol = std::make_unique<RDKit::ROMol>();
        RDKit::MolPickler::molFromPickle(pickle, *mol);
        return mol;
    } catch (const std::exception& e) {
        throw std::runtime_error(
            "Failed to convert molecule from RDKit binary: " + std::string(e.what())
        );
    }
}

nb::dict FingerprintToDict(const SparseFingerprint& features) {
    nb::dict result;
    for (const auto& [key, count] : features) {
        result[nb::int_(key)] = count;
    }
    return result;
}

void AppendLittleEndian(std::string& output, uint64_t value, int num_bytes) {
    for (int i = 0; i < num_bytes; ++i) {
        output.push_back(static_cast<char>((value >> (i * 8)) & 0xff));
    }
}

std::vector<std::pair<uint32_t, uint32_t>> SortedFeatures(
    const SparseFingerprint& features
) {
    std::vector<std::pair<uint32_t, uint32_t>> sorted;
    sorted.reserve(features.size());
    for (const auto& [key, count] : features) {
        if (count <= 0) {
            throw std::runtime_error("Fingerprint counts must be positive");
        }
        sorted.emplace_back(key, static_cast<uint32_t>(count));
    }
    std::sort(sorted.begin(), sorted.end());
    return sorted;
}

nb::bytes FingerprintToCountBuffer(
    const SparseFingerprint& features,
    const std::string& layout
) {
    const auto sorted = SortedFeatures(features);
    std::string output;

    if (layout == "32_32") {
        output.reserve(sorted.size() * 8);
        for (const auto& [key, count] : sorted) {
            AppendLittleEndian(output, key, 4);
            AppendLittleEndian(output, count, 4);
        }
    } else if (layout == "64_32_packed") {
        output.reserve(sorted.size() * 12);
        for (const auto& [key, count] : sorted) {
            AppendLittleEndian(output, key, 8);
            AppendLittleEndian(output, count, 4);
        }
    } else if (layout == "64_32") {
        output.reserve(sorted.size() * 16);
        for (const auto& [key, count] : sorted) {
            AppendLittleEndian(output, key, 8);
            AppendLittleEndian(output, count, 4);
            AppendLittleEndian(output, 0, 4);
        }
    } else {
        throw std::invalid_argument(
            "count layout must be '32_32', '64_32_packed', or '64_32'"
        );
    }
    return nb::bytes(output.data(), output.size());
}

nb::bytes FingerprintToBytes(const SparseFingerprint& features, int n_bits) {
    if (n_bits <= 0) {
        throw std::invalid_argument("nBits must be positive");
    }
    std::string output((n_bits + 7) / 8, '\0');
    for (const auto& [key, count] : features) {
        if (count <= 0) {
            continue;
        }
        const auto bit = key % static_cast<uint32_t>(n_bits);
        const auto byte_index = bit / 8;
        const auto byte = static_cast<unsigned char>(output[byte_index]);
        output[byte_index] = static_cast<char>(byte | (1u << (bit % 8)));
    }
    return nb::bytes(output.data(), output.size());
}

nb::object FormatFingerprint(
    const SparseFingerprint& features,
    const std::string& return_type,
    int n_bits
) {
    if (return_type == "dict") {
        return FingerprintToDict(features);
    }
    if (return_type == "bytes") {
        return FingerprintToBytes(features, n_bits);
    }
    return FingerprintToCountBuffer(features, return_type);
}

template <typename Generator>
nb::object GenerateFingerprint(
    const nb::object& mol_obj,
    const std::string& return_type,
    int n_bits,
    Generator&& generator
) {
    auto mol = MolFromPython(mol_obj);
    SparseFingerprint features;
    {
        nb::gil_scoped_release release;
        features = generator(mol.get());
    }
    return FormatFingerprint(features, return_type, n_bits);
}

}  // namespace

// Wrapper for ECFP (Morgan) fingerprint
std::vector<int> GetMorganFingerprintAsBitVectPython(
    nb::object mol_obj,
    int radius = 2,
    int nBits = 2048,
    bool useChirality = true
) {
    auto mol = MolFromPython(mol_obj);
    nb::gil_scoped_release release;
    auto result = bcfp::GetMorganFingerprintAsBitVect(
        mol.get(), radius, nBits, useChirality
    );
    return result;
}

// Wrapper for ECFP (Morgan) sparse fingerprint
nb::object GetMorganFingerprintPython(
    nb::object mol_obj,
    int radius = 2,
    bool useCounts = true,
    bool useChirality = true,
    const std::string& returnType = "dict",
    int nBits = 2048
) {
    return GenerateFingerprint(
        mol_obj, returnType, nBits, [=](const RDKit::ROMol* mol) {
            return bcfp::GetMorganFingerprint(mol, radius, useCounts, useChirality);
        }
    );
}

// Wrapper for ECFP XXHash sparse fingerprint (RDKit atom invariants + XXHash)
nb::object GetMorganFingerprintXXHashPython(
    nb::object mol_obj,
    int radius = 2,
    bool useCounts = true,
    bool useChirality = true,
    const std::string& returnType = "dict",
    int nBits = 2048
) {
    return GenerateFingerprint(
        mol_obj, returnType, nBits, [=](const RDKit::ROMol* mol) {
            return bcfp::GetMorganFingerprintXXHash(
                mol, radius, useCounts, useChirality
            );
        }
    );
}

// Wrapper for ECFP XXHash bit vector (RDKit atom invariants + XXHash)
std::vector<int> GetMorganFingerprintXXHashAsBitVectPython(
    nb::object mol_obj,
    int radius = 2,
    int nBits = 2048,
    bool useChirality = true
) {
    auto mol = MolFromPython(mol_obj);
    nb::gil_scoped_release release;
    auto result = bcfp::GetMorganFingerprintXXHashAsBitVect(
        mol.get(), radius, nBits, useChirality
    );
    return result;
}

// Wrapper for ECFP Blake3 sparse fingerprint (RDKit atom invariants + Blake3)
// NOTE: Blake3 C library not available via Homebrew - Python version available
// Wrapper for ECFP Blake3 sparse fingerprint (RDKit atom invariants + Blake3 hash)
nb::object GetMorganFingerprintBlake3Python(
    nb::object mol_obj,
    int radius = 2,
    bool useCounts = true,
    bool useChirality = true,
    const std::string& returnType = "dict",
    int nBits = 2048
) {
    return GenerateFingerprint(
        mol_obj, returnType, nBits, [=](const RDKit::ROMol* mol) {
            return bcfp::GetMorganFingerprintBlake3(
                mol, radius, useCounts, useChirality
            );
        }
    );
}

// Wrapper for ECFP Blake3 bit vector (RDKit atom invariants + Blake3)
std::vector<int> GetMorganFingerprintBlake3AsBitVectPython(
    nb::object mol_obj,
    int radius = 2,
    int nBits = 2048,
    bool useChirality = true
) {
    auto mol = MolFromPython(mol_obj);
    nb::gil_scoped_release release;
    auto result = bcfp::GetMorganFingerprintBlake3AsBitVect(
        mol.get(), radius, nBits, useChirality
    );
    return result;
}

// ============================================================================
// BCFP Native (Native RDKit-style hashing) - Python wrapper
// ============================================================================

// BCFP Native (Native) wrapper
nb::object GetBondMorganFingerprintNativePython(
    nb::object mol_obj,
    int radius = 2,
    bool useCounts = true,
    bool includeSharedAtomInvariants = true,
    bool includeEndpointAtoms = true,
    bool oriented = false,
    bool useChirality = true,
    const std::string& returnType = "dict",
    int nBits = 2048
) {
    return GenerateFingerprint(
        mol_obj, returnType, nBits, [=](const RDKit::ROMol* mol) {
            return bcfp::GetBondMorganFingerprintNative(
                mol, radius, useCounts, includeSharedAtomInvariants,
                includeEndpointAtoms, oriented, useChirality, nullptr
            );
        }
    );
}

// ============================================================================
// BCFP with alternative hash functions - Python wrappers
// ============================================================================

// BCFP XXHash wrappers
nb::object GetBondMorganFingerprintXXHashPython(
    nb::object mol_obj,
    int radius = 2,
    bool useCounts = true,
    bool includeSharedAtomInvariants = true,
    bool includeEndpointAtoms = true,
    bool oriented = false,
    bool useChirality = true,
    const std::string& returnType = "dict",
    int nBits = 2048
) {
    return GenerateFingerprint(
        mol_obj, returnType, nBits, [=](const RDKit::ROMol* mol) {
            return bcfp::GetBondMorganFingerprintXXHash(
                mol, radius, useCounts, includeSharedAtomInvariants,
                includeEndpointAtoms, oriented, useChirality, nullptr
            );
        }
    );
}

// BCFP Blake3 wrappers
nb::object GetBondMorganFingerprintBlake3Python(
    nb::object mol_obj,
    int radius = 2,
    bool useCounts = true,
    bool includeSharedAtomInvariants = true,
    bool includeEndpointAtoms = true,
    bool oriented = false,
    bool useChirality = true,
    const std::string& returnType = "dict",
    int nBits = 2048
) {
    return GenerateFingerprint(
        mol_obj, returnType, nBits, [=](const RDKit::ROMol* mol) {
            return bcfp::GetBondMorganFingerprintBlake3(
                mol, radius, useCounts, includeSharedAtomInvariants,
                includeEndpointAtoms, oriented, useChirality, nullptr
            );
        }
    );
}

NB_MODULE(_bcfp, m) {
    m.doc() = "Pure C++ BCFP/ECFP Implementation with Sort&Slice - Real C++ for maximum speed (2-3x faster)";
    
    // ============================================================================
    // ECFP (Morgan Fingerprint) Functions
    // ============================================================================
    
    // ECFP Native (RDKit's built-in implementation)
    m.def("get_morgan_fingerprint", &GetMorganFingerprintPython,
          "Get Morgan (ECFP) Fingerprint as sparse dict - RDKit Native",
          nb::arg("mol"),
          nb::arg("radius") = 2,
          nb::arg("useCounts") = true,
          nb::arg("useChirality") = true,
          nb::arg("returnType") = "dict",
          nb::arg("nBits") = 2048
    );
    
    m.def("get_morgan_fingerprint_as_bit_vect", &GetMorganFingerprintAsBitVectPython,
          "Get Morgan Fingerprint as Bit Vector - RDKit Native",
          nb::arg("mol"),
          nb::arg("radius") = 2,
          nb::arg("nBits") = 2048,
          nb::arg("useChirality") = true
    );
    
    // ECFP XXHash (hash-specific invariants)
    m.def("get_morgan_fingerprint_xxhash", &GetMorganFingerprintXXHashPython,
          "Get Morgan (ECFP) with XXHash (sparse dict) - Hash-specific invariants!",
          nb::arg("mol"),
          nb::arg("radius") = 2,
          nb::arg("useCounts") = true,
          nb::arg("useChirality") = true,
          nb::arg("returnType") = "dict",
          nb::arg("nBits") = 2048
    );
    
    m.def("get_morgan_fingerprint_xxhash_as_bit_vect", &GetMorganFingerprintXXHashAsBitVectPython,
          "Get Morgan with XXHash as Bit Vector - Hash-specific invariants!",
          nb::arg("mol"),
          nb::arg("radius") = 2,
          nb::arg("nBits") = 2048,
          nb::arg("useChirality") = true
    );
    
    // ECFP Blake3 (hash-specific invariants)
    m.def("get_morgan_fingerprint_blake3", &GetMorganFingerprintBlake3Python,
          "Get Morgan (ECFP) with Blake3 (sparse dict) - Hash-specific invariants!",
          nb::arg("mol"),
          nb::arg("radius") = 2,
          nb::arg("useCounts") = true,
          nb::arg("useChirality") = true,
          nb::arg("returnType") = "dict",
          nb::arg("nBits") = 2048
    );
    
    m.def("get_morgan_fingerprint_blake3_as_bit_vect", &GetMorganFingerprintBlake3AsBitVectPython,
          "Get Morgan with Blake3 as Bit Vector - Hash-specific invariants!",
          nb::arg("mol"),
          nb::arg("radius") = 2,
          nb::arg("nBits") = 2048,
          nb::arg("useChirality") = true
    );
    
    // ============================================================================
    // BCFP (Bond-Centered Morgan Fingerprint) Functions
    // ============================================================================
    
    // BCFP Native (Native RDKit-style hashing) - Hash-specific!
    m.def("get_bond_morgan_fingerprint_native", &GetBondMorganFingerprintNativePython,
          "Get BCFP with RDKit Native hashing (sparse dict) - Hash-specific invariants!",
          nb::arg("mol"),
          nb::arg("radius") = 2,
          nb::arg("useCounts") = true,
          nb::arg("includeSharedAtomInvariants") = true,
          nb::arg("includeEndpointAtoms") = true,
          nb::arg("oriented") = false,
          nb::arg("useChirality") = true,
          nb::arg("returnType") = "dict",
          nb::arg("nBits") = 2048
    );
    
    // BCFP XXHash - Hash-specific!
    m.def("get_bond_morgan_fingerprint_xxhash", &GetBondMorganFingerprintXXHashPython,
          "Get BCFP with XXHash (sparse dict) - Ultra-fast with hash-specific invariants!",
          nb::arg("mol"),
          nb::arg("radius") = 2,
          nb::arg("useCounts") = true,
          nb::arg("includeSharedAtomInvariants") = true,
          nb::arg("includeEndpointAtoms") = true,
          nb::arg("oriented") = false,
          nb::arg("useChirality") = true,
          nb::arg("returnType") = "dict",
          nb::arg("nBits") = 2048
    );
    
    // BCFP Blake3 - Hash-specific!
    m.def("get_bond_morgan_fingerprint_blake3", &GetBondMorganFingerprintBlake3Python,
          "Get BCFP with Blake3 (sparse dict) - Modern crypto hash with hash-specific invariants!",
          nb::arg("mol"),
          nb::arg("radius") = 2,
          nb::arg("useCounts") = true,
          nb::arg("includeSharedAtomInvariants") = true,
          nb::arg("includeEndpointAtoms") = true,
          nb::arg("oriented") = false,
          nb::arg("useChirality") = true,
          nb::arg("returnType") = "dict",
          nb::arg("nBits") = 2048
    );
}
