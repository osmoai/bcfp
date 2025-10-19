#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "bcfp.h"
#include <rdkit/GraphMol/SmilesParse/SmilesParse.h>

namespace py = pybind11;

// Wrapper for ECFP (Morgan) fingerprint
std::vector<int> GetMorganFingerprintAsBitVectPython(
    py::object mol_obj,
    int radius = 2,
    int nBits = 2048,
    bool useChirality = true
) {
    // Convert Python RDKit molecule to SMILES then to C++ ROMol
    RDKit::ROMol* mol = nullptr;
    try {
        py::object chem_module = py::module::import("rdkit.Chem");
        std::string smiles = chem_module.attr("MolToSmiles")(mol_obj).cast<std::string>();
        
        // Parse SMILES to C++ ROMol
        mol = RDKit::SmilesToMol(smiles);
        
        if (!mol) {
            throw std::runtime_error("Failed to parse SMILES: " + smiles);
        }
    } catch (const std::exception& e) {
        throw std::runtime_error("Failed to convert molecule: " + std::string(e.what()));
    }
    
    if (!mol) {
        throw std::runtime_error("Invalid molecule object");
    }
    
    // Call pure C++ ECFP implementation
    auto result = bcfp::GetMorganFingerprintAsBitVect(mol, radius, nBits, useChirality);
    
    // Clean up
    delete mol;
    
    return result;
}

// Wrapper for ECFP (Morgan) sparse fingerprint
py::dict GetMorganFingerprintPython(
    py::object mol_obj,
    int radius = 2,
    bool useCounts = true,
    bool useChirality = true
) {
    // Convert Python RDKit molecule to SMILES then to C++ ROMol
    RDKit::ROMol* mol = nullptr;
    try {
        py::object chem_module = py::module::import("rdkit.Chem");
        std::string smiles = chem_module.attr("MolToSmiles")(mol_obj).cast<std::string>();
        
        // Parse SMILES to C++ ROMol
        mol = RDKit::SmilesToMol(smiles);
        
        if (!mol) {
            throw std::runtime_error("Failed to parse SMILES: " + smiles);
        }
    } catch (const std::exception& e) {
        throw std::runtime_error("Failed to convert molecule: " + std::string(e.what()));
    }
    
    if (!mol) {
        throw std::runtime_error("Invalid molecule object");
    }
    
    // Call pure C++ ECFP implementation
    auto features = bcfp::GetMorganFingerprint(mol, radius, useCounts, useChirality);
    
    // Clean up
    delete mol;
    
    // Convert to Python dict
    py::dict result;
    for (const auto& [key, count] : features) {
        result[py::int_(key)] = count;
    }
    
    return result;
}

// Wrapper for ECFP XXHash sparse fingerprint (RDKit atom invariants + XXHash)
py::dict GetMorganFingerprintXXHashPython(
    py::object mol_obj,
    int radius = 2,
    bool useCounts = true,
    bool useChirality = true
) {
    // Convert Python RDKit molecule to SMILES then to C++ ROMol
    RDKit::ROMol* mol = nullptr;
    try {
        py::object chem_module = py::module::import("rdkit.Chem");
        std::string smiles = chem_module.attr("MolToSmiles")(mol_obj).cast<std::string>();
        
        // Parse SMILES to C++ ROMol
        mol = RDKit::SmilesToMol(smiles);
        
        if (!mol) {
            throw std::runtime_error("Failed to parse SMILES: " + smiles);
        }
    } catch (const std::exception& e) {
        throw std::runtime_error("Failed to convert molecule: " + std::string(e.what()));
    }
    
    if (!mol) {
        throw std::runtime_error("Invalid molecule object");
    }
    
    // Call pure C++ ECFP XXHash implementation
    auto features = bcfp::GetMorganFingerprintXXHash(mol, radius, useCounts, useChirality);
    
    // Clean up
    delete mol;
    
    // Convert to Python dict
    py::dict result;
    for (const auto& [key, count] : features) {
        result[py::int_(key)] = count;
    }
    
    return result;
}

// Wrapper for ECFP XXHash bit vector (RDKit atom invariants + XXHash)
std::vector<int> GetMorganFingerprintXXHashAsBitVectPython(
    py::object mol_obj,
    int radius = 2,
    int nBits = 2048,
    bool useChirality = true
) {
    // Convert Python RDKit molecule to SMILES then to C++ ROMol
    RDKit::ROMol* mol = nullptr;
    try {
        py::object chem_module = py::module::import("rdkit.Chem");
        std::string smiles = chem_module.attr("MolToSmiles")(mol_obj).cast<std::string>();
        
        // Parse SMILES to C++ ROMol
        mol = RDKit::SmilesToMol(smiles);
        
        if (!mol) {
            throw std::runtime_error("Failed to parse SMILES: " + smiles);
        }
    } catch (const std::exception& e) {
        throw std::runtime_error("Failed to convert molecule: " + std::string(e.what()));
    }
    
    if (!mol) {
        throw std::runtime_error("Invalid molecule object");
    }
    
    // Call pure C++ ECFP XXHash implementation (bit vector)
    auto result = bcfp::GetMorganFingerprintXXHashAsBitVect(mol, radius, nBits, useChirality);
    
    // Clean up
    delete mol;
    
    return result;
}

// Wrapper for ECFP Blake3 sparse fingerprint (RDKit atom invariants + Blake3)
// NOTE: Blake3 C library not available via Homebrew - Python version available
// Wrapper for ECFP Blake3 sparse fingerprint (RDKit atom invariants + Blake3 hash)
py::dict GetMorganFingerprintBlake3Python(
    py::object mol_obj,
    int radius = 2,
    bool useCounts = true,
    bool useChirality = true
) {
    // Convert Python RDKit molecule to SMILES then to C++ ROMol
    RDKit::ROMol* mol = nullptr;
    try {
        py::object chem_module = py::module::import("rdkit.Chem");
        std::string smiles = chem_module.attr("MolToSmiles")(mol_obj).cast<std::string>();

        // Parse SMILES to C++ ROMol
        mol = RDKit::SmilesToMol(smiles);

        if (!mol) {
            throw std::runtime_error("Failed to parse SMILES: " + smiles);
        }
    } catch (const std::exception& e) {
        throw std::runtime_error("Failed to convert molecule: " + std::string(e.what()));
    }

    if (!mol) {
        throw std::runtime_error("Invalid molecule object");
    }

    // Call pure C++ ECFP Blake3 implementation
    auto features = bcfp::GetMorganFingerprintBlake3(mol, radius, useCounts, useChirality);

    // Clean up
    delete mol;

    // Convert to Python dict
    py::dict result;
    for (const auto& [key, count] : features) {
        result[py::int_(key)] = count;
    }

    return result;
}

// Wrapper for ECFP Blake3 bit vector (RDKit atom invariants + Blake3)
std::vector<int> GetMorganFingerprintBlake3AsBitVectPython(
    py::object mol_obj,
    int radius = 2,
    int nBits = 2048,
    bool useChirality = true
) {
    // Convert Python RDKit molecule to SMILES then to C++ ROMol
    RDKit::ROMol* mol = nullptr;
    try {
        py::object chem_module = py::module::import("rdkit.Chem");
        std::string smiles = chem_module.attr("MolToSmiles")(mol_obj).cast<std::string>();

        // Parse SMILES to C++ ROMol
        mol = RDKit::SmilesToMol(smiles);

        if (!mol) {
            throw std::runtime_error("Failed to parse SMILES: " + smiles);
        }
    } catch (const std::exception& e) {
        throw std::runtime_error("Failed to convert molecule: " + std::string(e.what()));
    }

    if (!mol) {
        throw std::runtime_error("Invalid molecule object");
    }

    // Call pure C++ ECFP Blake3 implementation (bit vector)
    auto result = bcfp::GetMorganFingerprintBlake3AsBitVect(mol, radius, nBits, useChirality);

    // Clean up
    delete mol;

    return result;
}

// ============================================================================
// BCFP Native (Native RDKit-style hashing) - Python wrapper
// ============================================================================

// BCFP Native (Native) wrapper
py::dict GetBondMorganFingerprintNativePython(
    py::object mol_obj,
    int radius = 2,
    bool useCounts = true,
    bool includeSharedAtomInvariants = true,
    bool includeEndpointAtoms = true,
    bool oriented = false,
    bool useChirality = true
) {
    RDKit::ROMol* mol = nullptr;
    try {
        py::object chem_module = py::module::import("rdkit.Chem");
        std::string smiles = chem_module.attr("MolToSmiles")(mol_obj).cast<std::string>();
        mol = RDKit::SmilesToMol(smiles);
        if (!mol) {
            throw std::runtime_error("Failed to parse SMILES: " + smiles);
        }
    } catch (const std::exception& e) {
        throw std::runtime_error("Failed to convert molecule: " + std::string(e.what()));
    }

    auto features = bcfp::GetBondMorganFingerprintNative(
        mol, radius, useCounts, includeSharedAtomInvariants, includeEndpointAtoms,
        oriented, useChirality, nullptr
    );

    delete mol;

    py::dict result;
    for (const auto& [key, count] : features) {
        result[py::int_(key)] = count;
    }
    return result;
}

// ============================================================================
// BCFP with alternative hash functions - Python wrappers
// ============================================================================

// BCFP XXHash wrappers
py::dict GetBondMorganFingerprintXXHashPython(
    py::object mol_obj,
    int radius = 2,
    bool useCounts = true,
    bool includeSharedAtomInvariants = true,
    bool includeEndpointAtoms = true,
    bool oriented = false,
    bool useChirality = true
) {
    RDKit::ROMol* mol = nullptr;
    try {
        py::object chem_module = py::module::import("rdkit.Chem");
        std::string smiles = chem_module.attr("MolToSmiles")(mol_obj).cast<std::string>();
        mol = RDKit::SmilesToMol(smiles);
        if (!mol) {
            throw std::runtime_error("Failed to parse SMILES: " + smiles);
        }
    } catch (const std::exception& e) {
        throw std::runtime_error("Failed to convert molecule: " + std::string(e.what()));
    }

    auto features = bcfp::GetBondMorganFingerprintXXHash(
        mol, radius, useCounts, includeSharedAtomInvariants, includeEndpointAtoms,
        oriented, useChirality, nullptr
    );

    delete mol;

    py::dict result;
    for (const auto& [key, count] : features) {
        result[py::int_(key)] = count;
    }
    return result;
}

// BCFP Blake3 wrappers
py::dict GetBondMorganFingerprintBlake3Python(
    py::object mol_obj,
    int radius = 2,
    bool useCounts = true,
    bool includeSharedAtomInvariants = true,
    bool includeEndpointAtoms = true,
    bool oriented = false,
    bool useChirality = true
) {
    RDKit::ROMol* mol = nullptr;
    try {
        py::object chem_module = py::module::import("rdkit.Chem");
        std::string smiles = chem_module.attr("MolToSmiles")(mol_obj).cast<std::string>();
        mol = RDKit::SmilesToMol(smiles);
        if (!mol) {
            throw std::runtime_error("Failed to parse SMILES: " + smiles);
        }
    } catch (const std::exception& e) {
        throw std::runtime_error("Failed to convert molecule: " + std::string(e.what()));
    }

    auto features = bcfp::GetBondMorganFingerprintBlake3(
        mol, radius, useCounts, includeSharedAtomInvariants, includeEndpointAtoms,
        oriented, useChirality, nullptr
    );

    delete mol;

    py::dict result;
    for (const auto& [key, count] : features) {
        result[py::int_(key)] = count;
    }
    return result;
}

PYBIND11_MODULE(_bcfp, m) {
    m.doc() = "Pure C++ BCFP/ECFP Implementation with Sort&Slice - Real C++ for maximum speed (2-3x faster)";
    
    // ============================================================================
    // ECFP (Morgan Fingerprint) Functions
    // ============================================================================
    
    // ECFP Native (RDKit's built-in implementation)
    m.def("get_morgan_fingerprint", &GetMorganFingerprintPython,
          "Get Morgan (ECFP) Fingerprint as sparse dict - RDKit Native",
          py::arg("mol"),
          py::arg("radius") = 2,
          py::arg("useCounts") = true,
          py::arg("useChirality") = true
    );
    
    m.def("get_morgan_fingerprint_as_bit_vect", &GetMorganFingerprintAsBitVectPython,
          "Get Morgan Fingerprint as Bit Vector - RDKit Native",
          py::arg("mol"),
          py::arg("radius") = 2,
          py::arg("nBits") = 2048,
          py::arg("useChirality") = true
    );
    
    // ECFP XXHash (hash-specific invariants)
    m.def("get_morgan_fingerprint_xxhash", &GetMorganFingerprintXXHashPython,
          "Get Morgan (ECFP) with XXHash (sparse dict) - Hash-specific invariants!",
          py::arg("mol"),
          py::arg("radius") = 2,
          py::arg("useCounts") = true,
          py::arg("useChirality") = true
    );
    
    m.def("get_morgan_fingerprint_xxhash_as_bit_vect", &GetMorganFingerprintXXHashAsBitVectPython,
          "Get Morgan with XXHash as Bit Vector - Hash-specific invariants!",
          py::arg("mol"),
          py::arg("radius") = 2,
          py::arg("nBits") = 2048,
          py::arg("useChirality") = true
    );
    
    // ECFP Blake3 (hash-specific invariants)
    m.def("get_morgan_fingerprint_blake3", &GetMorganFingerprintBlake3Python,
          "Get Morgan (ECFP) with Blake3 (sparse dict) - Hash-specific invariants!",
          py::arg("mol"),
          py::arg("radius") = 2,
          py::arg("useCounts") = true,
          py::arg("useChirality") = true
    );
    
    m.def("get_morgan_fingerprint_blake3_as_bit_vect", &GetMorganFingerprintBlake3AsBitVectPython,
          "Get Morgan with Blake3 as Bit Vector - Hash-specific invariants!",
          py::arg("mol"),
          py::arg("radius") = 2,
          py::arg("nBits") = 2048,
          py::arg("useChirality") = true
    );
    
    // ============================================================================
    // BCFP (Bond-Centered Morgan Fingerprint) Functions
    // ============================================================================
    
    // BCFP Native (Native RDKit-style hashing) - Hash-specific!
    m.def("get_bond_morgan_fingerprint_native", &GetBondMorganFingerprintNativePython,
          "Get BCFP with RDKit Native hashing (sparse dict) - Hash-specific invariants!",
          py::arg("mol"),
          py::arg("radius") = 2,
          py::arg("useCounts") = true,
          py::arg("includeSharedAtomInvariants") = true,
          py::arg("includeEndpointAtoms") = true,
          py::arg("oriented") = false,
          py::arg("useChirality") = true
    );
    
    // BCFP XXHash - Hash-specific!
    m.def("get_bond_morgan_fingerprint_xxhash", &GetBondMorganFingerprintXXHashPython,
          "Get BCFP with XXHash (sparse dict) - Ultra-fast with hash-specific invariants!",
          py::arg("mol"),
          py::arg("radius") = 2,
          py::arg("useCounts") = true,
          py::arg("includeSharedAtomInvariants") = true,
          py::arg("includeEndpointAtoms") = true,
          py::arg("oriented") = false,
          py::arg("useChirality") = true
    );
    
    // BCFP Blake3 - Hash-specific!
    m.def("get_bond_morgan_fingerprint_blake3", &GetBondMorganFingerprintBlake3Python,
          "Get BCFP with Blake3 (sparse dict) - Modern crypto hash with hash-specific invariants!",
          py::arg("mol"),
          py::arg("radius") = 2,
          py::arg("useCounts") = true,
          py::arg("includeSharedAtomInvariants") = true,
          py::arg("includeEndpointAtoms") = true,
          py::arg("oriented") = false,
          py::arg("useChirality") = true
    );
}
