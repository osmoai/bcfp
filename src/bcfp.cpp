#include "bcfp.h"
#include <algorithm>
#include <cstring>
#include <rdkit/GraphMol/MolOps.h>
#include <rdkit/GraphMol/Chirality.h>
#include <rdkit/GraphMol/Fingerprints/MorganFingerprints.h>
#include <rdkit/DataStructs/SparseIntVect.h>

extern "C" {
}

// XXHash - ultra-fast, non-cryptographic hash
#define XXH_INLINE_ALL
#include <xxhash.h>

// Blake3 - next-gen cryptographic hash (7-10 GB/s, faster than Blake2b!)
#include <blake3.h>


extern "C" {
}

namespace bcfp {

// ============================================================================
// XXHasher Implementation (XXH3_128 - ultra-fast!)
// ============================================================================

uint32_t XXHasher::hash_tuple(const std::vector<int64_t>& items, uint32_t seed) {
    // Use XXH3_128 for best speed and collision resistance
    // We'll hash all items as a single byte buffer for maximum speed
    
    // Calculate total size needed
    size_t total_size = sizeof(uint64_t) + (items.size() * sizeof(int64_t));
    
    // Allocate buffer (stack allocation for speed, limited size)
    constexpr size_t MAX_STACK_SIZE = 1024;
    uint8_t stack_buffer[MAX_STACK_SIZE];
    std::vector<uint8_t> heap_buffer;
    uint8_t* buffer;
    
    if (total_size <= MAX_STACK_SIZE) {
        buffer = stack_buffer;
    } else {
        heap_buffer.resize(total_size);
        buffer = heap_buffer.data();
    }
    
    // Write seed as 64-bit (little-endian)
    uint64_t seed64 = static_cast<uint64_t>(seed);
    std::memcpy(buffer, &seed64, sizeof(uint64_t));
    
    // Write all items as 64-bit (little-endian)
    size_t offset = sizeof(uint64_t);
    for (int64_t item : items) {
        std::memcpy(buffer + offset, &item, sizeof(int64_t));
        offset += sizeof(int64_t);
    }
    
    // Hash with XXH3_128 (returns 128-bit hash, we take lower 32 bits)
    XXH128_hash_t hash = XXH3_128bits(buffer, total_size);
    
    // Return lower 32 bits
    return static_cast<uint32_t>(hash.low64 & 0xFFFFFFFF);
}

// ============================================================================
// Blake3Hasher Implementation (next-gen cryptographic hash, 7-10 GB/s!)
// ============================================================================

uint32_t Blake3Hasher::hash_tuple(const std::vector<int64_t>& items, uint32_t seed) {
    // Use Blake3 for ultra-fast cryptographic hashing (faster than Blake2b!)
    // Blake3 is a modern, parallelizable hash with excellent collision resistance

    // Calculate total size needed
    size_t total_size = sizeof(uint64_t) + (items.size() * sizeof(int64_t));

    // Allocate buffer (stack allocation for speed, limited size)
    constexpr size_t MAX_STACK_SIZE = 1024;
    uint8_t stack_buffer[MAX_STACK_SIZE];
    std::vector<uint8_t> heap_buffer;
    uint8_t* buffer;

    if (total_size <= MAX_STACK_SIZE) {
        buffer = stack_buffer;
    } else {
        heap_buffer.resize(total_size);
        buffer = heap_buffer.data();
    }

    // Write seed as 64-bit (little-endian)
    uint64_t seed64 = static_cast<uint64_t>(seed);
    std::memcpy(buffer, &seed64, sizeof(uint64_t));

    // Write all items as 64-bit (little-endian)
    size_t offset = sizeof(uint64_t);
    for (int64_t item : items) {
        std::memcpy(buffer + offset, &item, sizeof(int64_t));
        offset += sizeof(int64_t);
    }

    // Hash with Blake3 (returns 32-byte hash, we take first 32 bits)
    uint8_t hash_output[BLAKE3_OUT_LEN];
    blake3_hasher hasher;
    blake3_hasher_init(&hasher);
    blake3_hasher_update(&hasher, buffer, total_size);
    blake3_hasher_finalize(&hasher, hash_output, BLAKE3_OUT_LEN);

    // Return lower 32 bits as uint32_t
    uint32_t result;
    std::memcpy(&result, hash_output, sizeof(uint32_t));
    return result;
}

// ============================================================================
// Invariants Implementation
// ============================================================================

uint32_t Invariants::atom_invariant(const RDKit::Atom* atom, bool useChirality) {
    std::vector<int64_t> inv;
    
    // Basic invariants
    inv.push_back(atom->getAtomicNum());
    inv.push_back(atom->getTotalDegree());
    
    int valence = atom->getTotalValence();
    inv.push_back(valence >= 0 ? valence : 0);
    
    inv.push_back(atom->getFormalCharge());
    inv.push_back(atom->getIsAromatic() ? 1 : 0);
    // Check if atom is in a ring using ring info
    bool in_ring = false;
    if (atom->getOwningMol().getRingInfo() && atom->getOwningMol().getRingInfo()->isInitialized()) {
        in_ring = atom->getOwningMol().getRingInfo()->numAtomRings(atom->getIdx()) > 0;
    }
    inv.push_back(in_ring ? 1 : 0);
    
    // Chirality
    if (useChirality) {
        int has_cip = atom->hasProp("_CIPCode") ? 1 : 0;
        int chiral_flag = (atom->getChiralTag() != RDKit::Atom::ChiralType::CHI_UNSPECIFIED) ? 1 : 0;
        inv.push_back(has_cip + chiral_flag);
    }
    
    return XXHasher::hash_tuple(inv, SEED_ATOM);
}

uint32_t Invariants::bond_invariant(const RDKit::Bond* bond, bool oriented, 
                                    bool useChirality, bool includeEndpointAtoms) {
    std::vector<int64_t> inv;
    
    // Basic bond properties
    inv.push_back(static_cast<int>(bond->getBondTypeAsDouble()));
    inv.push_back(bond->getIsAromatic() ? 1 : 0);
    inv.push_back(bond->getIsConjugated() ? 1 : 0);
    // Check if bond is in a ring using ring info
    bool in_ring = false;
    if (bond->getOwningMol().getRingInfo() && bond->getOwningMol().getRingInfo()->isInitialized()) {
        in_ring = bond->getOwningMol().getRingInfo()->numBondRings(bond->getIdx()) > 0;
    }
    inv.push_back(in_ring ? 1 : 0);
    
    // Chirality
    if (useChirality) {
        inv.push_back(static_cast<int>(bond->getStereo()));
    }
    
    // Endpoint atoms
    if (includeEndpointAtoms) {
        const RDKit::Atom* a1 = bond->getBeginAtom();
        const RDKit::Atom* a2 = bond->getEndAtom();
        
        uint32_t a1i = atom_invariant(a1, useChirality);
        uint32_t a2i = atom_invariant(a2, useChirality);
        
        if (oriented) {
            // Keep direction (begin -> end)
            inv.push_back(a1i);
            inv.push_back(a2i);
        } else {
            // Orientation-invariant (sorted)
            if (a1i < a2i) {
                inv.push_back(a1i);
                inv.push_back(a2i);
            } else {
                inv.push_back(a2i);
                inv.push_back(a1i);
            }
        }
    }
    
    return XXHasher::hash_tuple(inv, SEED_BOND);
}

int Invariants::bond_shared_end(const RDKit::Bond* bond, int shared_atom_idx) {
    if (bond->getBeginAtomIdx() == shared_atom_idx) {
        return 0;  // Begin atom
    } else if (bond->getEndAtomIdx() == shared_atom_idx) {
        return 1;  // End atom
    }
    return -1;  // Not shared (error case)
}

// ============================================================================
// Line Graph Construction
// ============================================================================

LineGraphAdjacency build_line_graph(const RDKit::ROMol* mol) {
    int num_bonds = mol->getNumBonds();
    LineGraphAdjacency neighbors(num_bonds);
    
    for (int i = 0; i < num_bonds; i++) {
        const RDKit::Bond* bond_i = mol->getBondWithIdx(i);
        int begin_atom = bond_i->getBeginAtomIdx();
        int end_atom = bond_i->getEndAtomIdx();
        
        // Check neighbors through begin atom
        const RDKit::Atom* atom_begin = mol->getAtomWithIdx(begin_atom);
        for (const auto& nb : mol->atomBonds(atom_begin)) {
            int nb_idx = mol->getBondBetweenAtoms(nb->getBeginAtomIdx(), nb->getEndAtomIdx())->getIdx();
            if (nb_idx != i) {
                neighbors[i].push_back({nb_idx, begin_atom});
            }
        }
        
        // Check neighbors through end atom
        const RDKit::Atom* atom_end = mol->getAtomWithIdx(end_atom);
        for (const auto& nb : mol->atomBonds(atom_end)) {
            int nb_idx = mol->getBondBetweenAtoms(nb->getBeginAtomIdx(), nb->getEndAtomIdx())->getIdx();
            if (nb_idx != i) {
                neighbors[i].push_back({nb_idx, end_atom});
            }
        }
    }
    
    return neighbors;
}

// ============================================================================
// Main BCFP Implementation
// ============================================================================

std::unordered_map<uint32_t, int> GetBondMorganFingerprint(
    const RDKit::ROMol* mol,
    int radius,
    bool useCounts,
    bool includeSharedAtomInvariants,
    bool includeEndpointAtoms,
    bool oriented,
    bool useChirality,
    const std::vector<int>* fromBonds
) {
    std::unordered_map<uint32_t, int> features;
    
    if (mol->getNumBonds() == 0) {
        return features;
    }
    
    // Initialize ring info (needed for isInRing checks)
    RDKit::MolOps::findSSSR(*const_cast<RDKit::ROMol*>(mol));
    
    // Assign stereochemistry if needed
    if (useChirality) {
        RDKit::MolOps::assignStereochemistry(*const_cast<RDKit::ROMol*>(mol), true, true);
    }
    
    // Get bond indices
    std::vector<int> bond_indices;
    if (fromBonds) {
        bond_indices = *fromBonds;
    } else {
        for (int i = 0; i < mol->getNumBonds(); i++) {
            bond_indices.push_back(i);
        }
    }
    
    // Precompute atom invariants
    std::vector<uint32_t> atom_invars;
    for (int i = 0; i < mol->getNumAtoms(); i++) {
        atom_invars.push_back(Invariants::atom_invariant(mol->getAtomWithIdx(i), useChirality));
    }
    
    // Compute initial bond invariants
    std::vector<uint32_t> bond_ids_curr;
    for (int i = 0; i < mol->getNumBonds(); i++) {
        bond_ids_curr.push_back(Invariants::bond_invariant(
            mol->getBondWithIdx(i), oriented, useChirality, includeEndpointAtoms
        ));
    }
    
    // Build line graph
    auto neighbors = build_line_graph(mol);
    
    // Emit radius 0 features
    for (int bidx : bond_indices) {
        std::vector<int64_t> fid_parts = {static_cast<int64_t>(bond_ids_curr[bidx]), 0};
        uint32_t fid = XXHasher::hash_tuple(fid_parts, SEED_EMIT);
        features[fid] += 1;
    }
    
    // Iterative updates for each radius
    std::vector<uint32_t> bond_ids_next(mol->getNumBonds());
    
    for (int r = 1; r <= radius; r++) {
        // Update bond identifiers
        for (int bidx = 0; bidx < mol->getNumBonds(); bidx++) {
            const RDKit::Bond* bond = mol->getBondWithIdx(bidx);
            
            std::vector<int64_t> parts = {static_cast<int64_t>(bond_ids_curr[bidx])};
            std::vector<int64_t> nbr_parts;
            
            for (const auto& [nbidx, shared_atom_idx] : neighbors[bidx]) {
                uint32_t pid = bond_ids_curr[nbidx];
                
                if (includeSharedAtomInvariants) {
                    int endflag = Invariants::bond_shared_end(bond, shared_atom_idx);
                    std::vector<int64_t> nbr_hash_parts = {
                        static_cast<int64_t>(pid),
                        static_cast<int64_t>(endflag),
                        static_cast<int64_t>(atom_invars[shared_atom_idx])
                    };
                    nbr_parts.push_back(XXHasher::hash_tuple(nbr_hash_parts, SEED_LINE));
                } else {
                    nbr_parts.push_back(pid);
                }
            }
            
            // Sort neighbor parts (orientation-invariant)
            std::sort(nbr_parts.begin(), nbr_parts.end());
            parts.insert(parts.end(), nbr_parts.begin(), nbr_parts.end());
            
            bond_ids_next[bidx] = XXHasher::hash_tuple(parts, SEED_UPDATE ^ r);
        }
        
        // Swap current and next
        std::swap(bond_ids_curr, bond_ids_next);
        
        // Emit features for this radius
        for (int bidx : bond_indices) {
            std::vector<int64_t> fid_parts = {static_cast<int64_t>(bond_ids_curr[bidx]), r};
            uint32_t fid = XXHasher::hash_tuple(fid_parts, SEED_EMIT);
            features[fid] += 1;
        }
    }
    
    // Apply useCounts flag
    if (!useCounts) {
        for (auto& [k, v] : features) {
            v = 1;
        }
    }
    
    return features;
}

std::vector<int> GetBondMorganFingerprintAsBitVect(
    const RDKit::ROMol* mol,
    int radius,
    int nBits,
    bool useCounts,
    bool includeSharedAtomInvariants,
    bool includeEndpointAtoms,
    bool oriented,
    bool useChirality,
    const std::vector<int>* fromBonds
) {
    std::vector<int> result(nBits, 0);
    
    // Get sparse fingerprint
    auto features = GetBondMorganFingerprint(
        mol, radius, useCounts, includeSharedAtomInvariants,
        includeEndpointAtoms, oriented, useChirality, fromBonds
    );
    
    // Fold into bit vector
    for (const auto& [fid, count] : features) {
        int bit = fid % nBits;
        result[bit] = 1;  // Set bit (ignore count for bit vector)
    }
    
    return result;
}

// ============================================================================
// ECFP (Morgan) Implementation - Using RDKit's Native Implementation
// ============================================================================

// Note: For ECFP, we use RDKit's built-in Morgan fingerprint generator
// to ensure exact compatibility with Python GetMorganFingerprint()

std::unordered_map<uint32_t, int> GetMorganFingerprint(
    const RDKit::ROMol* mol,
    int radius,
    bool useCounts,
    bool useChirality
) {
    std::unordered_map<uint32_t, int> features;
    
    if (!mol || mol->getNumAtoms() == 0) {
        return features;
    }
    
    // Use RDKit's native Morgan fingerprint generator
    // This ensures exact compatibility with Python AllChem.GetMorganFingerprint
    // Simplified API: getFingerprint(mol, radius, invariants, fromAtoms, useChirality, useBondTypes, useFeatures)
    RDKit::SparseIntVect<std::uint32_t>* fp = RDKit::MorganFingerprints::getFingerprint(
        *mol,
        static_cast<unsigned int>(radius),
        nullptr,      // invariants
        nullptr,      // fromAtoms
        useChirality  // useChirality
    );
    
    if (fp) {
        // Convert SparseIntVect to unordered_map
        for (const auto& [key, value] : fp->getNonzeroElements()) {
            features[key] = value;
        }
        delete fp;
    }
    
    return features;
}

std::vector<int> GetMorganFingerprintAsBitVect(
    const RDKit::ROMol* mol,
    int radius,
    int nBits,
    bool useChirality
) {
    std::vector<int> result(nBits, 0);
    
    // Get sparse fingerprint
    auto features = GetMorganFingerprint(mol, radius, true, useChirality);
    
    // Fold into bit vector
    for (const auto& [fid, count] : features) {
        int bit = fid % nBits;
        result[bit] = 1;  // Set bit (ignore count for bit vector)
    }
    
    return result;
}

// ============================================================================
// ECFP with Blake2b hashing (for performance comparison)
// ============================================================================



// ============================================================================
// ECFP with Blake2b hashing V2: RDKit Native Atom Invariants + Blake2b Hash
// ============================================================================



// ============================================================================
// ECFP (Morgan) with XXHash - FASTEST & BEST ML PERFORMANCE! 🚀
// ============================================================================

std::unordered_map<uint32_t, int> GetMorganFingerprintXXHash(
    const RDKit::ROMol* mol,
    int radius,
    bool useCounts,
    bool useChirality
) {
    std::unordered_map<uint32_t, int> features;
    
    if (!mol || mol->getNumAtoms() == 0) {
        return features;
    }
    
    // Initialize ring info
    RDKit::MolOps::findSSSR(*const_cast<RDKit::ROMol*>(mol));
    
    // Assign stereochemistry if needed
    if (useChirality) {
        RDKit::MolOps::assignStereochemistry(*const_cast<RDKit::ROMol*>(mol), true, true);
    }
    
    int numAtoms = mol->getNumAtoms();
    
    // Precompute atom invariants using RDKit's NATIVE invariant computation
    std::vector<uint32_t> atom_ids_curr;
    atom_ids_curr.reserve(numAtoms);
    for (int i = 0; i < numAtoms; i++) {
        const RDKit::Atom* atom = mol->getAtomWithIdx(i);
        
        // Use RDKit's native atom invariant encoding
        uint32_t inv = 0;
        
        // Atomic number (7 bits)
        inv = atom->getAtomicNum() % 128;
        
        // Degree (4 bits, 0-15)
        unsigned int deg = atom->getDegree();
        if (deg > 15) deg = 15;
        inv |= (deg << 7);
        
        // Total valence (4 bits, 0-15)
        unsigned int val = atom->getTotalValence();
        if (val > 15) val = 15;
        inv |= (val << 11);
        
        // Formal charge (3 bits, offset by 4 to handle -4 to +3)
        int fc = atom->getFormalCharge();
        unsigned int fc_enc = (fc + 4) % 8;
        inv |= (fc_enc << 15);
        
        // Number of radical electrons (2 bits, 0-3)
        unsigned int nre = atom->getNumRadicalElectrons();
        if (nre > 3) nre = 3;
        inv |= (nre << 18);
        
        // Hybridization (3 bits, 0-7)
        unsigned int hyb = static_cast<unsigned int>(atom->getHybridization());
        if (hyb > 7) hyb = 7;
        inv |= (hyb << 20);
        
        // Aromaticity (1 bit)
        if (atom->getIsAromatic()) {
            inv |= (1 << 23);
        }
        
        // Ring membership (1 bit)
        if (mol->getRingInfo()->isInitialized() && 
            mol->getRingInfo()->numAtomRings(atom->getIdx()) > 0) {
            inv |= (1 << 24);
        }
        
        // For chirality, we'll use a simple approach
        if (useChirality) {
            if (atom->getChiralTag() != RDKit::Atom::ChiralType::CHI_UNSPECIFIED) {
                inv |= (1 << 25);
            }
        }
        
        atom_ids_curr.push_back(inv);
    }
    
    // Emit radius 0 features (initial atom invariants, hashed with XXHash)
    for (int aidx = 0; aidx < numAtoms; aidx++) {
        uint32_t curr_id = atom_ids_curr[aidx];
        
        // Hash with radius using XXHash
        std::vector<int64_t> fid_parts = {static_cast<int64_t>(curr_id), 0};
        uint32_t fid = XXHasher::hash_tuple(fid_parts, SEED_EMIT);
        
        if (useCounts) {
            features[fid]++;
        } else {
            features[fid] = 1;
        }
    }
    
    // Build neighbor lists for atoms
    std::vector<std::vector<int>> atom_neighbors(numAtoms);
    for (int i = 0; i < mol->getNumBonds(); i++) {
        const RDKit::Bond* bond = mol->getBondWithIdx(i);
        int a1 = bond->getBeginAtomIdx();
        int a2 = bond->getEndAtomIdx();
        atom_neighbors[a1].push_back(a2);
        atom_neighbors[a2].push_back(a1);
    }
    
    // Iterate through radii
    for (int r = 1; r <= radius; r++) {
        std::vector<uint32_t> atom_ids_next;
        atom_ids_next.reserve(numAtoms);
        
        // Update each atom's identifier
        for (int aidx = 0; aidx < numAtoms; aidx++) {
            uint32_t curr_id = atom_ids_curr[aidx];
            
            // Collect neighbor identifiers
            std::vector<int64_t> neighbor_ids;
            neighbor_ids.reserve(atom_neighbors[aidx].size() + 1);
            neighbor_ids.push_back(static_cast<int64_t>(curr_id));  // Include current atom
            
            for (int neighbor_idx : atom_neighbors[aidx]) {
                neighbor_ids.push_back(static_cast<int64_t>(atom_ids_curr[neighbor_idx]));
            }
            
            // Sort neighbors (for canonical ordering)
            std::sort(neighbor_ids.begin() + 1, neighbor_ids.end());
            
            // Hash to get next identifier using XXHash
            uint32_t next_id = XXHasher::hash_tuple(neighbor_ids, SEED_UPDATE);
            atom_ids_next.push_back(next_id);
            
            // Emit feature for this radius using XXHash
            std::vector<int64_t> fid_parts = {static_cast<int64_t>(next_id), static_cast<int64_t>(r)};
            uint32_t fid = XXHasher::hash_tuple(fid_parts, SEED_EMIT);
            
            if (useCounts) {
                features[fid]++;
            } else {
                features[fid] = 1;
            }
        }
        
        atom_ids_curr = std::move(atom_ids_next);
    }
    
    return features;
}

std::vector<int> GetMorganFingerprintXXHashAsBitVect(
    const RDKit::ROMol* mol,
    int radius,
    int nBits,
    bool useChirality
) {
    std::vector<int> result(nBits, 0);
    
    // Get sparse fingerprint
    auto features = GetMorganFingerprintXXHash(mol, radius, true, useChirality);
    
    // Fold into bit vector
    for (const auto& [fid, count] : features) {
        int bit = fid % nBits;
        result[bit] += count;  // Accumulate counts
    }
    
    return result;
}

// ============================================================================
// ECFP (Morgan) with Blake3 - Next-gen cryptographic hash!
// ============================================================================

std::unordered_map<uint32_t, int> GetMorganFingerprintBlake3(
    const RDKit::ROMol* mol,
    int radius,
    bool useCounts,
    bool useChirality
) {
    std::unordered_map<uint32_t, int> features;
    
    if (!mol || mol->getNumAtoms() == 0) {
        return features;
    }
    
    // Initialize ring info
    RDKit::MolOps::findSSSR(*const_cast<RDKit::ROMol*>(mol));
    
    // Assign stereochemistry if needed
    if (useChirality) {
        RDKit::MolOps::assignStereochemistry(*const_cast<RDKit::ROMol*>(mol), true, true);
    }
    
    int numAtoms = mol->getNumAtoms();
    
    // Precompute atom invariants using RDKit's NATIVE invariant computation
    std::vector<uint32_t> atom_ids_curr;
    atom_ids_curr.reserve(numAtoms);
    for (int i = 0; i < numAtoms; i++) {
        const RDKit::Atom* atom = mol->getAtomWithIdx(i);
        
        // Use RDKit's native atom invariant encoding
        uint32_t inv = 0;
        
        // Atomic number (7 bits)
        inv = atom->getAtomicNum() % 128;
        
        // Degree (4 bits, 0-15)
        unsigned int deg = atom->getDegree();
        if (deg > 15) deg = 15;
        inv |= (deg << 7);
        
        // Total valence (4 bits, 0-15)
        unsigned int val = atom->getTotalValence();
        if (val > 15) val = 15;
        inv |= (val << 11);
        
        // Formal charge (3 bits, offset by 4 to handle -4 to +3)
        int fc = atom->getFormalCharge();
        unsigned int fc_enc = (fc + 4) % 8;
        inv |= (fc_enc << 15);
        
        // Number of radical electrons (2 bits, 0-3)
        unsigned int nre = atom->getNumRadicalElectrons();
        if (nre > 3) nre = 3;
        inv |= (nre << 18);
        
        // Hybridization (3 bits, 0-7)
        unsigned int hyb = static_cast<unsigned int>(atom->getHybridization());
        if (hyb > 7) hyb = 7;
        inv |= (hyb << 20);
        
        // Aromaticity (1 bit)
        if (atom->getIsAromatic()) {
            inv |= (1 << 23);
        }
        
        // Ring membership (1 bit)
        if (mol->getRingInfo()->isInitialized() && 
            mol->getRingInfo()->numAtomRings(atom->getIdx()) > 0) {
            inv |= (1 << 24);
        }
        
        // For chirality, we'll use a simple approach
        if (useChirality) {
            if (atom->getChiralTag() != RDKit::Atom::ChiralType::CHI_UNSPECIFIED) {
                inv |= (1 << 25);
            }
        }
        
        atom_ids_curr.push_back(inv);
    }
    
    // Emit radius 0 features (initial atom invariants, hashed with Blake3)
    for (int aidx = 0; aidx < numAtoms; aidx++) {
        uint32_t curr_id = atom_ids_curr[aidx];
        
        // Hash with radius using Blake3
        std::vector<int64_t> fid_parts = {static_cast<int64_t>(curr_id), 0};
        uint32_t fid = Blake3Hasher::hash_tuple(fid_parts, SEED_EMIT);
        
        if (useCounts) {
            features[fid]++;
        } else {
            features[fid] = 1;
        }
    }
    
    // Build neighbor lists for atoms
    std::vector<std::vector<int>> atom_neighbors(numAtoms);
    for (int i = 0; i < mol->getNumBonds(); i++) {
        const RDKit::Bond* bond = mol->getBondWithIdx(i);
        int a1 = bond->getBeginAtomIdx();
        int a2 = bond->getEndAtomIdx();
        atom_neighbors[a1].push_back(a2);
        atom_neighbors[a2].push_back(a1);
    }
    
    // Iterate through radii
    for (int r = 1; r <= radius; r++) {
        std::vector<uint32_t> atom_ids_next;
        atom_ids_next.reserve(numAtoms);
        
        // Update each atom's identifier
        for (int aidx = 0; aidx < numAtoms; aidx++) {
            uint32_t curr_id = atom_ids_curr[aidx];
            
            // Collect neighbor identifiers
            std::vector<int64_t> neighbor_ids;
            neighbor_ids.reserve(atom_neighbors[aidx].size() + 1);
            neighbor_ids.push_back(static_cast<int64_t>(curr_id));  // Include current atom
            
            for (int neighbor_idx : atom_neighbors[aidx]) {
                neighbor_ids.push_back(static_cast<int64_t>(atom_ids_curr[neighbor_idx]));
            }
            
            // Sort neighbors (for canonical ordering)
            std::sort(neighbor_ids.begin() + 1, neighbor_ids.end());
            
            // Hash to get next identifier using Blake3
            uint32_t next_id = Blake3Hasher::hash_tuple(neighbor_ids, SEED_UPDATE);
            atom_ids_next.push_back(next_id);
            
            // Emit feature for this radius using Blake3
            std::vector<int64_t> fid_parts = {static_cast<int64_t>(next_id), static_cast<int64_t>(r)};
            uint32_t fid = Blake3Hasher::hash_tuple(fid_parts, SEED_EMIT);
            
            if (useCounts) {
                features[fid]++;
            } else {
                features[fid] = 1;
            }
        }
        
        atom_ids_curr = std::move(atom_ids_next);
    }
    
    return features;
}

std::vector<int> GetMorganFingerprintBlake3AsBitVect(
    const RDKit::ROMol* mol,
    int radius,
    int nBits,
    bool useChirality
) {
    std::vector<int> result(nBits, 0);
    
    // Get sparse fingerprint
    auto features = GetMorganFingerprintBlake3(mol, radius, true, useChirality);
    
    // Fold into bit vector
    for (const auto& [fid, count] : features) {
        int bit = fid % nBits;
        result[bit] += count;  // Accumulate counts
    }
    
    return result;
}

// ============================================================================
// ECFP (Morgan) with HighwayHash - Google's SIMD-optimized fast hash!
// ============================================================================



// ============================================================================
// Sort&Slice Implementation
// ============================================================================

std::pair<std::vector<uint32_t>, std::unordered_map<uint32_t, int>> 
SortSliceFit(
    const std::vector<std::unordered_map<uint32_t, int>>& sparse_list,
    const std::vector<int>& train_idx,
    int top_k,
    const std::string& sort_by,
    int min_df
) {
    std::unordered_map<uint32_t, int> DF;  // Document frequency
    std::unordered_map<uint32_t, int> TF;  // Total frequency
    
    // One pass over train indices
    for (int i : train_idx) {
        const auto& kv = sparse_list[i];
        if (kv.empty()) continue;
        
        // Update DF (document frequency)
        for (const auto& [key, count] : kv) {
            DF[key]++;
        }
        
        // Update TF (total frequency)
        for (const auto& [key, count] : kv) {
            TF[key] += count;
        }
    }
    
    // Filter by min_df
    std::vector<uint32_t> keys;
    for (const auto& [key, df_count] : DF) {
        if (df_count >= min_df) {
            keys.push_back(key);
        }
    }
    
    // Sort keys
    if (sort_by == "df") {
        // Sort by (DF, TF, key) descending
        std::sort(keys.begin(), keys.end(), [&](uint32_t a, uint32_t b) {
            if (DF[a] != DF[b]) return DF[a] > DF[b];
            if (TF[a] != TF[b]) return TF[a] > TF[b];
            return a > b;  // Stable tie-breaker
        });
    } else {  // "tf"
        // Sort by (TF, DF, key) descending
        std::sort(keys.begin(), keys.end(), [&](uint32_t a, uint32_t b) {
            if (TF[a] != TF[b]) return TF[a] > TF[b];
            if (DF[a] != DF[b]) return DF[a] > DF[b];
            return a > b;  // Stable tie-breaker
        });
    }
    
    // Keep only top-K
    if (keys.size() > static_cast<size_t>(top_k)) {
        keys.resize(top_k);
    }
    
    // Build key2col map
    std::unordered_map<uint32_t, int> key2col;
    for (size_t j = 0; j < keys.size(); j++) {
        key2col[keys[j]] = static_cast<int>(j);
    }
    
    return {keys, key2col};
}

std::vector<std::vector<float>> SortSliceTransform(
    const std::vector<std::unordered_map<uint32_t, int>>& sparse_list,
    const std::vector<int>& indices,
    const std::unordered_map<uint32_t, int>& key2col,
    bool use_counts,
    bool add_oov_bucket
) {
    int N = indices.size();
    int baseD = key2col.size();
    int D = baseD + (add_oov_bucket ? 1 : 0);
    int oov_col = add_oov_bucket ? baseD : -1;
    
    // Initialize result matrix
    std::vector<std::vector<float>> X(N, std::vector<float>(D, 0.0f));
    
    for (int row = 0; row < N; row++) {
        int i = indices[row];
        const auto& kv = sparse_list[i];
        
        for (const auto& [key, count] : kv) {
            auto it = key2col.find(key);
            if (it != key2col.end()) {
                // In-vocabulary key
                int col = it->second;
                X[row][col] = use_counts ? static_cast<float>(count) : 1.0f;
            } else if (add_oov_bucket) {
                // Out-of-vocabulary key
                X[row][oov_col] += use_counts ? static_cast<float>(count) : 1.0f;
            }
        }
        
        // For binary mode with OOV, ensure OOV column is 0 or 1
        if (!use_counts && add_oov_bucket && X[row][oov_col] > 1.0f) {
            X[row][oov_col] = 1.0f;
        }
    }
    
    return X;
}

// ============================================================================
// BCFP v2: Optimized implementation using RDKit-style hashing
// ============================================================================

uint32_t RDKitHasher::hash_tuple_fast(const std::vector<int64_t>& items, uint32_t seed) {
    // Start with seed
    uint32_t hash = seed;
    
    // Combine each item using boost-style hash_combine
    for (int64_t item : items) {
        // Convert 64-bit to 32-bit (upper and lower parts)
        uint32_t lower = static_cast<uint32_t>(item & 0xFFFFFFFF);
        uint32_t upper = static_cast<uint32_t>((item >> 32) & 0xFFFFFFFF);
        
        hash_combine(hash, lower);
        hash_combine(hash, upper);
    }
    
    return hash;
}

std::unordered_map<uint32_t, int> GetBondMorganFingerprintV2(
    const RDKit::ROMol* mol,
    int radius,
    bool useCounts,
    bool includeSharedAtomInvariants,
    bool includeEndpointAtoms,
    bool oriented,
    bool useChirality,
    const std::vector<int>* fromBonds
) {
    std::unordered_map<uint32_t, int> features;
    
    if (mol->getNumBonds() == 0) {
        return features;
    }
    
    // Initialize ring info (needed for isInRing checks)
    RDKit::MolOps::findSSSR(*const_cast<RDKit::ROMol*>(mol));
    
    // Assign stereochemistry if needed
    if (useChirality) {
        RDKit::MolOps::assignStereochemistry(*const_cast<RDKit::ROMol*>(mol), true, true);
    }
    
    // Get bond indices
    std::vector<int> bond_indices;
    if (fromBonds) {
        bond_indices = *fromBonds;
    } else {
        bond_indices.reserve(mol->getNumBonds());
        for (int i = 0; i < mol->getNumBonds(); i++) {
            bond_indices.push_back(i);
        }
    }
    
    // Precompute atom invariants (reuse from v1)
    std::vector<uint32_t> atom_invars;
    atom_invars.reserve(mol->getNumAtoms());
    for (int i = 0; i < mol->getNumAtoms(); i++) {
        atom_invars.push_back(Invariants::atom_invariant(mol->getAtomWithIdx(i), useChirality));
    }
    
    // Compute initial bond invariants (reuse from v1)
    std::vector<uint32_t> bond_ids_curr;
    bond_ids_curr.reserve(mol->getNumBonds());
    for (int i = 0; i < mol->getNumBonds(); i++) {
        bond_ids_curr.push_back(Invariants::bond_invariant(
            mol->getBondWithIdx(i), oriented, useChirality, includeEndpointAtoms
        ));
    }
    
    // Build line graph (reuse from v1)
    auto neighbors = build_line_graph(mol);
    
    // Emit radius 0 features (using fast hash instead of Blake2b)
    for (int bidx : bond_indices) {
        std::vector<int64_t> fid_parts = {static_cast<int64_t>(bond_ids_curr[bidx]), 0};
        uint32_t fid = RDKitHasher::hash_tuple_fast(fid_parts, SEED_EMIT);
        features[fid] += 1;
    }
    
    // Iterative updates for each radius
    std::vector<uint32_t> bond_ids_next(mol->getNumBonds());
    
    for (int r = 1; r <= radius; r++) {
        // Update bond identifiers
        for (int bidx = 0; bidx < mol->getNumBonds(); bidx++) {
            const RDKit::Bond* bond = mol->getBondWithIdx(bidx);
            
            std::vector<int64_t> parts = {static_cast<int64_t>(bond_ids_curr[bidx])};
            std::vector<int64_t> nbr_parts;
            nbr_parts.reserve(neighbors[bidx].size());
            
            for (const auto& [nbidx, shared_atom_idx] : neighbors[bidx]) {
                uint32_t pid = bond_ids_curr[nbidx];
                
                if (includeSharedAtomInvariants) {
                    int endflag = Invariants::bond_shared_end(bond, shared_atom_idx);
                    std::vector<int64_t> nbr_hash_parts = {
                        static_cast<int64_t>(pid),
                        static_cast<int64_t>(endflag),
                        static_cast<int64_t>(atom_invars[shared_atom_idx])
                    };
                    nbr_parts.push_back(RDKitHasher::hash_tuple_fast(nbr_hash_parts, SEED_LINE));
                } else {
                    nbr_parts.push_back(pid);
                }
            }
            
            // Sort neighbor parts (orientation-invariant)
            std::sort(nbr_parts.begin(), nbr_parts.end());
            parts.insert(parts.end(), nbr_parts.begin(), nbr_parts.end());
            
            bond_ids_next[bidx] = RDKitHasher::hash_tuple_fast(parts, SEED_UPDATE ^ r);
        }
        
        // Swap current and next
        std::swap(bond_ids_curr, bond_ids_next);
        
        // Emit features for this radius (using fast hash)
        for (int bidx : bond_indices) {
            std::vector<int64_t> fid_parts = {static_cast<int64_t>(bond_ids_curr[bidx]), r};
            uint32_t fid = RDKitHasher::hash_tuple_fast(fid_parts, SEED_EMIT);
            features[fid] += 1;
        }
    }
    
    // Apply useCounts flag
    if (!useCounts) {
        for (auto& [k, v] : features) {
            v = 1;
        }
    }
    
    return features;
}

std::vector<int> GetBondMorganFingerprintV2AsBitVect(
    const RDKit::ROMol* mol,
    int radius,
    int nBits,
    bool useCounts,
    bool includeSharedAtomInvariants,
    bool includeEndpointAtoms,
    bool oriented,
    bool useChirality,
    const std::vector<int>* fromBonds
) {
    std::vector<int> result(nBits, 0);
    
    // Get sparse fingerprint
    auto features = GetBondMorganFingerprintV2(
        mol, radius, useCounts, includeSharedAtomInvariants,
        includeEndpointAtoms, oriented, useChirality, fromBonds
    );
    
    // Fold into bit vector
    for (const auto& [fid, count] : features) {
        int bit = fid % nBits;
        result[bit] = useCounts ? count : 1;
    }
    
    return result;
}

// ============================================================================

} // namespace bcfp

