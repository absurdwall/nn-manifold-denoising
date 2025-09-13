"""
Dataset properties and management classes.
Based on the structure from manifold_gen2.py but cleaned and organized.
"""

import os
import numpy as np
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime


@dataclass
class DatasetProperties:
    """
    Properties of a single dataset.
    """
    # Core generation parameters
    k: int                    # Number of groups
    N: int                    # Points per group  
    d: int                    # Intrinsic dimension
    D: int                    # Ambient dimension
    kernel_smoothness: float  # RBF kernel length scale
    noise_sigma: float        # Noise level
    base_type: str           # Base manifold type
    
    # Experiment metadata
    dataset_num: int         # Dataset number in experiment
    experiment_id: str       # Experiment identifier (e.g., 'tb25091301')
    random_seed: int         # Random seed used
    
    # Processing options
    train_ratio: float = 0.8     # Training data fraction
    centralize: bool = True      # Whether data was centralized
    rescale: bool = True         # Whether data was rescaled
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DatasetProperties':
        """Create from dictionary."""
        return cls(**data)
    
    @property
    def total_points(self) -> int:
        """Total number of points in dataset."""
        return self.k * self.N
    
    @property
    def dataset_id(self) -> str:
        """Unique dataset identifier."""
        return f"{self.experiment_id}_dataset{self.dataset_num}"


@dataclass 
class NormalizationInfo:
    """
    Information about data normalization applied to a dataset.
    """
    data_mean: np.ndarray         # Mean vector (if centralized)
    normalization_factor: float  # Scaling factor (if rescaled)
    centralize_applied: bool      # Whether centralization was applied
    rescale_applied: bool         # Whether rescaling was applied
    original_diameter: float      # Diameter before normalization
    final_diameter: float         # Diameter after normalization
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        from ..utils.data_utils import handle_types
        return handle_types(asdict(self))
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'NormalizationInfo':
        """Create from dictionary."""
        # Convert data_mean back to numpy array if it's a list
        if 'data_mean' in data and isinstance(data['data_mean'], list):
            data['data_mean'] = np.array(data['data_mean'])
        return cls(**data)


class Dataset:
    """
    A single generated dataset with all associated data and metadata.
    """
    
    def __init__(self, properties: DatasetProperties):
        self.properties = properties
        self.normalization_info: Optional[NormalizationInfo] = None
        
        # Data arrays (set by generator)
        self.raw_embedded: Optional[np.ndarray] = None      # Before normalization
        self.clean_embedded: Optional[np.ndarray] = None    # After normalization  
        self.noisy_embedded: Optional[np.ndarray] = None    # With noise added
        self.intrinsic_coords: Optional[np.ndarray] = None  # Intrinsic coordinates
        
        # Train/test splits (set when requested)
        self.train_test_data: Optional[Dict[str, np.ndarray]] = None
        
        # Metadata
        self.generation_time: Optional[float] = None
        self.created_at: str = datetime.now().isoformat()
        
    def set_data(self, raw_embedded: np.ndarray, clean_embedded: np.ndarray,
                 intrinsic_coords: np.ndarray, normalization_info: NormalizationInfo,
                 noisy_embedded: Optional[np.ndarray] = None, generation_time: Optional[float] = None):
        """Set the generated data arrays."""
        self.raw_embedded = raw_embedded
        self.clean_embedded = clean_embedded
        self.intrinsic_coords = intrinsic_coords
        self.normalization_info = normalization_info
        self.noisy_embedded = noisy_embedded
        self.generation_time = generation_time
        
    def create_train_test_split(self, random_seed: Optional[int] = None) -> Dict[str, np.ndarray]:
        """Create train/test split with noise as labels."""
        if self.clean_embedded is None or self.noisy_embedded is None:
            raise ValueError("Both clean and noisy data must be set before creating train/test split")
        
        from ..utils.data_utils import create_train_test_split
        
        seed = random_seed if random_seed is not None else self.properties.random_seed
        self.train_test_data = create_train_test_split(
            self.clean_embedded, self.noisy_embedded,
            self.properties.train_ratio, seed
        )
        return self.train_test_data
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the dataset."""
        summary = {
            'dataset_id': self.properties.dataset_id,
            'properties': self.properties.to_dict(),
            'created_at': self.created_at,
            'generation_time': self.generation_time,
            'data_shapes': {},
            'has_train_test_split': self.train_test_data is not None
        }
        
        # Add shape information
        if self.raw_embedded is not None:
            summary['data_shapes']['raw_embedded'] = list(self.raw_embedded.shape)
        if self.clean_embedded is not None:
            summary['data_shapes']['clean_embedded'] = list(self.clean_embedded.shape)
        if self.noisy_embedded is not None:
            summary['data_shapes']['noisy_embedded'] = list(self.noisy_embedded.shape)
        if self.intrinsic_coords is not None:
            summary['data_shapes']['intrinsic_coords'] = list(self.intrinsic_coords.shape)
            
        # Add normalization info
        if self.normalization_info is not None:
            summary['normalization_info'] = self.normalization_info.to_dict()
            
        return summary
    
    def save_to_directory(self, output_dir: str) -> Dict[str, str]:
        """
        Save dataset to directory with standardized file structure.
        
        Returns:
            Dictionary mapping data type to saved file path
        """
        os.makedirs(output_dir, exist_ok=True)
        
        prefix = os.path.join(output_dir, f"dataset{self.properties.dataset_num}")
        saved_files = {}
        
        # Save data arrays
        if self.raw_embedded is not None:
            path = f"{prefix}_raw.npy"
            np.save(path, self.raw_embedded)
            saved_files['raw_embedded'] = path
            
        if self.clean_embedded is not None:
            path = f"{prefix}_clean.npy"
            np.save(path, self.clean_embedded)
            saved_files['clean_embedded'] = path
            
        if self.noisy_embedded is not None:
            path = f"{prefix}_noisy.npy"
            np.save(path, self.noisy_embedded)
            saved_files['noisy_embedded'] = path
            
        if self.intrinsic_coords is not None:
            path = f"{prefix}_intrinsic.npy"
            np.save(path, self.intrinsic_coords)
            saved_files['intrinsic_coords'] = path
            
        # Save normalization info as separate arrays for easy loading
        if self.normalization_info is not None:
            if self.normalization_info.centralize_applied:
                path = f"{prefix}_data_mean.npy"
                np.save(path, self.normalization_info.data_mean)
                saved_files['data_mean'] = path
                
        # Save metadata
        metadata = {
            'properties': self.properties.to_dict(),
            'summary': self.get_summary(),
            'files_created': saved_files
        }
        
        from ..utils.data_utils import save_json
        metadata_path = f"{prefix}_metadata.json"
        save_json(metadata, metadata_path)
        saved_files['metadata'] = metadata_path
        
        # Save train/test split if it exists
        if self.train_test_data is not None:
            split_files = {}
            for split_name, split_data in self.train_test_data.items():
                path = f"{prefix}_{split_name}.npy"
                np.save(path, split_data)
                split_files[split_name] = path
            saved_files.update(split_files)
            
        return saved_files
    
    @classmethod
    def load_from_directory(cls, dataset_path: str) -> 'Dataset':
        """Load dataset from directory."""
        # Load metadata first
        from ..utils.data_utils import load_json
        metadata = load_json(f"{dataset_path}_metadata.json")
        
        # Create dataset with properties
        properties = DatasetProperties.from_dict(metadata['properties'])
        dataset = cls(properties)
        
        # Load data arrays
        prefix = dataset_path
        
        if os.path.exists(f"{prefix}_raw.npy"):
            dataset.raw_embedded = np.load(f"{prefix}_raw.npy")
        if os.path.exists(f"{prefix}_clean.npy"):
            dataset.clean_embedded = np.load(f"{prefix}_clean.npy")
        if os.path.exists(f"{prefix}_noisy.npy"):
            dataset.noisy_embedded = np.load(f"{prefix}_noisy.npy")
        if os.path.exists(f"{prefix}_intrinsic.npy"):
            dataset.intrinsic_coords = np.load(f"{prefix}_intrinsic.npy")
            
        # Load normalization info
        if 'normalization_info' in metadata['summary']:
            dataset.normalization_info = NormalizationInfo.from_dict(
                metadata['summary']['normalization_info']
            )
            
        # Load train/test split if it exists
        train_test_files = ['train_data', 'train_labels', 'test_data', 'test_labels', 
                           'train_clean', 'test_clean']
        if all(os.path.exists(f"{prefix}_{name}.npy") for name in train_test_files):
            dataset.train_test_data = {}
            for name in train_test_files:
                dataset.train_test_data[name] = np.load(f"{prefix}_{name}.npy")
                
        return dataset
