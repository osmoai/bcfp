"""
Utility functions for BCFP fingerprint generation.
"""

import numpy as np
from rdkit import Chem
from typing import List, Union


def smiles_to_mol(smiles: str):
    """
    Convert SMILES string to RDKit molecule.
    
    Args:
        smiles: SMILES string
        
    Returns:
        RDKit molecule object or None if invalid
    """
    return Chem.MolFromSmiles(smiles)


def load_molecules(smiles_list: List[str], verbose: bool = False):
    """
    Load a list of SMILES strings into RDKit molecules.
    
    Args:
        smiles_list: List of SMILES strings
        verbose: Print progress
        
    Returns:
        List of RDKit molecule objects (None for invalid SMILES)
    """
    molecules = []
    n_invalid = 0
    
    for i, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi)
        molecules.append(mol)
        
        if mol is None:
            n_invalid += 1
        
        if verbose and (i + 1) % 1000 == 0:
            print(f"Loaded {i + 1}/{len(smiles_list)} molecules ({n_invalid} invalid)")
    
    if verbose:
        print(f"✅ Loaded {len(molecules)} molecules ({n_invalid} invalid)")
    
    return molecules


def ensure_molecule(mol_or_smiles: Union[str, object]):
    """
    Ensure input is an RDKit molecule object.
    
    Args:
        mol_or_smiles: Either SMILES string or RDKit molecule
        
    Returns:
        RDKit molecule object or None if invalid
    """
    if isinstance(mol_or_smiles, str):
        return Chem.MolFromSmiles(mol_or_smiles)
    return mol_or_smiles


def batch_generate(generator, molecules: List, show_progress: bool = False):
    """
    Generate fingerprints for a batch of molecules.
    
    Args:
        generator: FingerprintGenerator instance
        molecules: List of RDKit molecules or SMILES strings
        show_progress: Show progress bar (requires tqdm)
        
    Returns:
        numpy array of fingerprints
    """
    if show_progress:
        try:
            from tqdm import tqdm
            molecules = tqdm(molecules, desc="Generating fingerprints")
        except ImportError:
            print("Warning: tqdm not available, progress bar disabled")
    
    fingerprints = []
    for mol in molecules:
        mol = ensure_molecule(mol)
        if mol is not None:
            fp = generator.generate_basic(mol)
            fingerprints.append(fp)
        else:
            # Add zero vector for invalid molecules
            fingerprints.append(np.zeros(generator.n_bits, dtype=np.int32))
    
    return np.array(fingerprints)


__all__ = [
    'smiles_to_mol',
    'load_molecules',
    'ensure_molecule',
    'batch_generate',
]

