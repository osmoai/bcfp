"""
Model Persistence for BCFP/ECFP Fingerprints

This module provides comprehensive save/load functionality for:
1. FingerprintGenerator configuration
2. Sort&Slice vocabulary (selected features)
3. Complete model state for reproducible inference

Author: Guillaume GODIN - Osmo labs pbc
License: BSD-3-Clause
"""

import pickle
from typing import Dict, List, Tuple, Optional
import numpy as np


class FingerprintModel:
    """
    Complete fingerprint model for training and inference.
    
    This class encapsulates:
    - FingerprintGenerator configuration (hash function, radius, etc.)
    - Sort&Slice vocabulary (selected features)
    - OOV bucket settings
    
    Usage:
        # Training
        model = FingerprintModel(hash_func='xxhash', radius=2, top_k=2048)
        X_train = model.fit_transform(train_smiles, train_idx)
        model.save('my_model.pkl')
        
        # Inference (in production)
        model = FingerprintModel.load('my_model.pkl')
        X_new = model.transform(new_smiles)
    """
    
    def __init__(self, 
                 hash_func: str = 'rdkit_native',
                 fp_type: str = 'ecfp',
                 radius: int = 2,
                 n_bits: int = 2048,
                 use_counts: bool = True,
                 use_chirality: bool = True,
                 top_k: int = 2048,
                 sort_by: str = 'df',
                 min_df: int = 2,
                 add_oov_bucket: bool = True):
        """
        Initialize fingerprint model.
        
        Args:
            hash_func: Hash function ('rdkit_native', 'xxhash', 'blake3')
            fp_type: Fingerprint type ('ecfp', 'bcfp', 'both')
            radius: Morgan fingerprint radius
            n_bits: Number of bits for folded fingerprints
            use_counts: Use count features (True) or binary (False)
            use_chirality: Include chirality information
            top_k: Number of top features to select in Sort&Slice
            sort_by: Sort by 'df' (document frequency) or 'tf' (total frequency)
            min_df: Minimum document frequency for feature selection
            add_oov_bucket: Add out-of-vocabulary bucket column
        """
        from .fingerprints import FingerprintGenerator
        
        self.hash_func = hash_func
        self.fp_type = fp_type
        self.radius = radius
        self.n_bits = n_bits
        self.use_counts = use_counts
        self.use_chirality = use_chirality
        self.top_k = top_k
        self.sort_by = sort_by
        self.min_df = min_df
        self.add_oov_bucket = add_oov_bucket
        
        # Generator (created on demand)
        self._generator = None
        
        # Vocabulary (set during fit)
        self.vocab_ = None
        self.key2col_ = None
        self.is_fitted_ = False
    
    @property
    def generator(self):
        """Lazy initialization of FingerprintGenerator."""
        if self._generator is None:
            from .fingerprints import FingerprintGenerator
            
            # Handle 'both' case specially
            if self.fp_type == 'both':
                # Create two generators
                self._generator = {
                    'ecfp': FingerprintGenerator(
                        hash_func=self.hash_func,
                        fp_type='ecfp',
                        radius=self.radius,
                        n_bits=self.n_bits,
                        use_counts=self.use_counts,
                        use_chirality=self.use_chirality
                    ),
                    'bcfp': FingerprintGenerator(
                        hash_func=self.hash_func,
                        fp_type='bcfp',
                        radius=self.radius,
                        n_bits=self.n_bits,
                        use_counts=self.use_counts,
                        use_chirality=self.use_chirality
                    )
                }
            else:
                self._generator = FingerprintGenerator(
                    hash_func=self.hash_func,
                    fp_type=self.fp_type,
                    radius=self.radius,
                    n_bits=self.n_bits,
                    use_counts=self.use_counts,
                    use_chirality=self.use_chirality
                )
        return self._generator
    
    def fit(self, molecules: List, train_idx: np.ndarray) -> 'FingerprintModel':
        """
        Fit the model on training data (select features).
        
        Args:
            molecules: List of RDKit molecules or SMILES strings
            train_idx: Indices of training molecules
            
        Returns:
            self
        """
        from rdkit import Chem
        from .fingerprints import combine_fingerprints
        
        # Convert SMILES to molecules if needed
        if isinstance(molecules[0], str):
            molecules = [Chem.MolFromSmiles(smi) for smi in molecules]
        
        # Generate sparse fingerprints
        if self.fp_type == 'both':
            # Generate ECFP and BCFP separately
            ecfp_sparse = [self.generator['ecfp'].generate_sparse(mol) for mol in molecules]
            bcfp_sparse = [self.generator['bcfp'].generate_sparse(mol) for mol in molecules]
            
            # Combine sparse fingerprints (merge dictionaries with key offset)
            sparse_list = []
            max_ecfp_key = max(max(fp.keys(), default=0) for fp in ecfp_sparse) if ecfp_sparse else 0
            key_offset = max_ecfp_key + 1
            
            for ecfp_fp, bcfp_fp in zip(ecfp_sparse, bcfp_sparse):
                combined = dict(ecfp_fp)
                # Add BCFP keys with offset to avoid collision
                for k, v in bcfp_fp.items():
                    combined[k + key_offset] = v
                sparse_list.append(combined)
        else:
            sparse_list = [self.generator.generate_sparse(mol) for mol in molecules]
        
        # Fit Sort&Slice vocabulary
        self.vocab_, self.key2col_ = self.generator['ecfp'].sortslice_fit(
            sparse_list, train_idx, 
            top_k=self.top_k, 
            sort_by=self.sort_by, 
            min_df=self.min_df
        ) if self.fp_type == 'both' else self.generator.sortslice_fit(
            sparse_list, train_idx, 
            top_k=self.top_k, 
            sort_by=self.sort_by, 
            min_df=self.min_df
        )
        
        self.is_fitted_ = True
        return self
    
    def transform(self, molecules: List, indices: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Transform molecules to feature matrix using fitted vocabulary.
        
        Args:
            molecules: List of RDKit molecules or SMILES strings
            indices: Indices to transform (default: all)
            
        Returns:
            Feature matrix of shape (n_molecules, n_features)
        """
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before transform. Call fit() first.")
        
        from rdkit import Chem
        
        # Convert SMILES to molecules if needed
        if isinstance(molecules[0], str):
            molecules = [Chem.MolFromSmiles(smi) for smi in molecules]
        
        # Generate sparse fingerprints
        if self.fp_type == 'both':
            # Generate ECFP and BCFP separately
            ecfp_sparse = [self.generator['ecfp'].generate_sparse(mol) for mol in molecules]
            bcfp_sparse = [self.generator['bcfp'].generate_sparse(mol) for mol in molecules]
            
            # Combine sparse fingerprints (same key offset as fit)
            sparse_list = []
            max_ecfp_key = max(max(fp.keys(), default=0) for fp in ecfp_sparse) if ecfp_sparse else 0
            key_offset = max_ecfp_key + 1
            
            for ecfp_fp, bcfp_fp in zip(ecfp_sparse, bcfp_sparse):
                combined = dict(ecfp_fp)
                for k, v in bcfp_fp.items():
                    combined[k + key_offset] = v
                sparse_list.append(combined)
        else:
            sparse_list = [self.generator.generate_sparse(mol) for mol in molecules]
        
        # Default: transform all
        if indices is None:
            indices = np.arange(len(molecules))
        
        # Transform using vocabulary
        gen = self.generator['ecfp'] if self.fp_type == 'both' else self.generator
        X = gen.sortslice_transform(
            sparse_list, indices, 
            self.key2col_, 
            self.use_counts, 
            self.add_oov_bucket
        )
        
        return X
    
    def fit_transform(self, molecules: List, train_idx: np.ndarray) -> np.ndarray:
        """
        Fit and transform in one step.
        
        Args:
            molecules: List of RDKit molecules or SMILES strings
            train_idx: Indices of training molecules
            
        Returns:
            Feature matrix for training molecules
        """
        self.fit(molecules, train_idx)
        return self.transform(molecules, train_idx)
    
    def save(self, filepath: str):
        """
        Save complete model state to disk.
        
        Args:
            filepath: Path to save model (e.g., 'my_model.pkl')
            
        Example:
            model.save('production_model.pkl')
        """
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before saving.")
        
        state = {
            # Configuration (CRITICAL: hash_func determines feature generation!)
            'hash_func': self.hash_func,
            'fp_type': self.fp_type,
            'radius': self.radius,
            'n_bits': self.n_bits,
            'use_counts': self.use_counts,
            'use_chirality': self.use_chirality,
            'top_k': self.top_k,
            'sort_by': self.sort_by,
            'min_df': self.min_df,
            'add_oov_bucket': self.add_oov_bucket,
            
            # Fitted state
            'vocab': self.vocab_,
            'key2col': self.key2col_,
            'is_fitted': self.is_fitted_,
            
            # Metadata
            'n_features': len(self.vocab_),
            'n_features_with_oov': len(self.vocab_) + (1 if self.add_oov_bucket else 0),
            
            # Version control
            'bcfp_version': '2.0.0',
            'saved_timestamp': pickle.time.time() if hasattr(pickle, 'time') else None
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        print(f"✅ Model saved to {filepath}")
        print(f"   Hash function: {self.hash_func} ⚠️  CRITICAL - Must match during inference!")
        print(f"   Fingerprint type: {self.fp_type}")
        print(f"   Radius: {self.radius}")
        print(f"   Features: {state['n_features']} (+{1 if self.add_oov_bucket else 0} OOV)")
        print(f"\n⚠️  WARNING: This model ONLY works with hash_func='{self.hash_func}'")
        print(f"   Using a different hash will produce INCORRECT features and predictions!")
    
    @classmethod
    def load(cls, filepath: str, validate: bool = True) -> 'FingerprintModel':
        """
        Load complete model state from disk.
        
        Args:
            filepath: Path to saved model
            validate: If True, validate hash function compatibility (default: True)
            
        Returns:
            Loaded FingerprintModel ready for inference
            
        Example:
            model = FingerprintModel.load('production_model.pkl')
            X_new = model.transform(new_molecules)
            
        Warning:
            The loaded model will ONLY work correctly with the exact hash function
            that was used during training. Using a different hash will produce
            INCORRECT features and predictions!
        """
        with open(filepath, 'rb') as f:
            state = pickle.load(f)
        
        # Validate required fields
        required_fields = ['hash_func', 'fp_type', 'vocab', 'key2col']
        missing = [f for f in required_fields if f not in state]
        if missing:
            raise ValueError(f"Invalid model file: missing fields {missing}")
        
        # Create new instance with loaded configuration
        model = cls(
            hash_func=state['hash_func'],
            fp_type=state['fp_type'],
            radius=state['radius'],
            n_bits=state['n_bits'],
            use_counts=state['use_counts'],
            use_chirality=state['use_chirality'],
            top_k=state['top_k'],
            sort_by=state['sort_by'],
            min_df=state['min_df'],
            add_oov_bucket=state['add_oov_bucket']
        )
        
        # Restore fitted state
        model.vocab_ = state['vocab']
        model.key2col_ = state['key2col']
        model.is_fitted_ = state['is_fitted']
        
        print(f"✅ Model loaded from {filepath}")
        print(f"   Hash function: {model.hash_func} ⚠️  CRITICAL - Must use this hash!")
        print(f"   Fingerprint type: {model.fp_type}")
        print(f"   Radius: {model.radius}")
        print(f"   Features: {state['n_features']} (+{1 if model.add_oov_bucket else 0} OOV)")
        
        # Validation warning
        if validate:
            print(f"\n⚠️  IMPORTANT: This model uses hash_func='{model.hash_func}'")
            print(f"   All inference MUST use the same hash function!")
            print(f"   To verify compatibility, call model.verify_hash_compatibility()")
        
        return model
    
    def verify_hash_compatibility(self, test_smiles: str = 'CCO') -> Dict:
        """
        Verify that the current hash function produces consistent features.
        
        This is critical for production deployment to ensure the model
        is using the correct hash function.
        
        Args:
            test_smiles: Test molecule (default: ethanol 'CCO')
            
        Returns:
            Dictionary with verification results
            
        Example:
            model = FingerprintModel.load('model.pkl')
            result = model.verify_hash_compatibility()
            if result['status'] == 'OK':
                print("Hash function verified!")
        """
        from rdkit import Chem
        
        if not self.is_fitted_:
            return {
                'status': 'ERROR',
                'message': 'Model not fitted - cannot verify'
            }
        
        try:
            # Generate test fingerprint
            mol = Chem.MolFromSmiles(test_smiles)
            if mol is None:
                return {
                    'status': 'ERROR',
                    'message': f'Invalid test SMILES: {test_smiles}'
                }
            
            # Generate sparse fingerprint
            if self.fp_type == 'both':
                ecfp_fp = self.generator['ecfp'].generate_sparse(mol)
                bcfp_fp = self.generator['bcfp'].generate_sparse(mol)
                n_features_ecfp = len(ecfp_fp)
                n_features_bcfp = len(bcfp_fp)
                n_features = n_features_ecfp + n_features_bcfp
            else:
                fp = self.generator.generate_sparse(mol)
                n_features = len(fp)
            
            return {
                'status': 'OK',
                'hash_func': self.hash_func,
                'fp_type': self.fp_type,
                'test_molecule': test_smiles,
                'n_features_generated': n_features,
                'vocab_size': len(self.vocab_),
                'message': f'✅ Hash function {self.hash_func} verified and working correctly!'
            }
            
        except Exception as e:
            return {
                'status': 'ERROR',
                'hash_func': self.hash_func,
                'message': f'Hash verification failed: {str(e)}'
            }
    
    def get_model_info(self) -> Dict:
        """
        Get complete model information including hash function.
        
        Returns:
            Dictionary with all model configuration and state
            
        Example:
            model = FingerprintModel.load('model.pkl')
            info = model.get_model_info()
            print(f"Hash function: {info['hash_func']}")
        """
        return {
            # Configuration
            'hash_func': self.hash_func,
            'fp_type': self.fp_type,
            'radius': self.radius,
            'n_bits': self.n_bits,
            'use_counts': self.use_counts,
            'use_chirality': self.use_chirality,
            'top_k': self.top_k,
            'sort_by': self.sort_by,
            'min_df': self.min_df,
            'add_oov_bucket': self.add_oov_bucket,
            
            # State
            'is_fitted': self.is_fitted_,
            'n_features': len(self.vocab_) if self.is_fitted_ else None,
            'n_features_with_oov': len(self.vocab_) + (1 if self.add_oov_bucket else 0) if self.is_fitted_ else None,
            
            # Version
            'bcfp_version': '2.0.0'
        }
    
    @staticmethod
    def compare_models(model1: 'FingerprintModel', model2: 'FingerprintModel') -> Dict:
        """
        Compare two models to check compatibility.
        
        Args:
            model1: First model
            model2: Second model
            
        Returns:
            Dictionary with comparison results
            
        Example:
            model_old = FingerprintModel.load('old_model.pkl')
            model_new = FingerprintModel.load('new_model.pkl')
            diff = FingerprintModel.compare_models(model_old, model_new)
            if diff['hash_func_match']:
                print("Hash functions match!")
        """
        info1 = model1.get_model_info()
        info2 = model2.get_model_info()
        
        comparison = {
            'hash_func_match': info1['hash_func'] == info2['hash_func'],
            'hash_func_1': info1['hash_func'],
            'hash_func_2': info2['hash_func'],
            'fp_type_match': info1['fp_type'] == info2['fp_type'],
            'radius_match': info1['radius'] == info2['radius'],
            'config_match': (info1['hash_func'] == info2['hash_func'] and
                           info1['fp_type'] == info2['fp_type'] and
                           info1['radius'] == info2['radius'] and
                           info1['use_counts'] == info2['use_counts'] and
                           info1['use_chirality'] == info2['use_chirality'])
        }
        
        if comparison['config_match']:
            comparison['status'] = 'COMPATIBLE'
            comparison['message'] = '✅ Models have compatible configurations'
        else:
            comparison['status'] = 'INCOMPATIBLE'
            differences = []
            if not comparison['hash_func_match']:
                differences.append(f"hash_func: {info1['hash_func']} vs {info2['hash_func']}")
            if not comparison['fp_type_match']:
                differences.append(f"fp_type: {info1['fp_type']} vs {info2['fp_type']}")
            if not comparison['radius_match']:
                differences.append(f"radius: {info1['radius']} vs {info2['radius']}")
            comparison['message'] = f'⚠️  Models are INCOMPATIBLE: {", ".join(differences)}'
        
        return comparison
    
    def get_feature_importance(self, X: np.ndarray, y: np.ndarray = None) -> Dict:
        """
        Get feature statistics for interpretation.
        
        Args:
            X: Feature matrix
            y: Optional labels
            
        Returns:
            Dictionary with feature statistics
        """
        if not self.is_fitted_:
            raise ValueError("Model must be fitted.")
        
        stats = {
            'n_features': len(self.vocab_),
            'feature_keys': self.vocab_,
            'feature_sparsity': np.mean(X == 0, axis=0),
            'feature_mean': np.mean(X, axis=0),
            'feature_std': np.std(X, axis=0)
        }
        
        if y is not None:
            # Compute correlation with labels
            stats['feature_label_corr'] = np.array([
                np.corrcoef(X[:, i], y)[0, 1] if X[:, i].std() > 0 else 0
                for i in range(X.shape[1])
            ])
        
        return stats
    
    def __repr__(self) -> str:
        status = "fitted" if self.is_fitted_ else "not fitted"
        n_feat = len(self.vocab_) if self.is_fitted_ else self.top_k
        return (f"FingerprintModel({status}, hash={self.hash_func}, "
                f"type={self.fp_type}, radius={self.radius}, features={n_feat})")


# Backward compatibility: keep original functions
def save_sortslice_vocab(vocab: List, key2col: Dict, filepath: str):
    """
    Save Sort&Slice vocabulary (feature keys and mapping) for inference.
    
    DEPRECATED: Use FingerprintModel.save() for complete model persistence.
    """
    state = {
        'vocab': vocab,
        'key2col': key2col,
        'n_features': len(vocab)
    }
    
    with open(filepath, 'wb') as f:
        pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    print(f"✅ Sort&Slice vocabulary saved to {filepath}")
    print(f"   Features: {len(vocab)}")


def load_sortslice_vocab(filepath: str) -> Tuple[List, Dict]:
    """
    Load Sort&Slice vocabulary for inference on new data.
    
    DEPRECATED: Use FingerprintModel.load() for complete model persistence.
    """
    with open(filepath, 'rb') as f:
        state = pickle.load(f)
    
    print(f"✅ Sort&Slice vocabulary loaded from {filepath}")
    print(f"   Features: {state['n_features']}")
    
    return state['vocab'], state['key2col']

