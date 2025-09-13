"""
Utility functions for data handling and type management.
Shared across the manifold denoising project.
"""

import numpy as np
import json
from typing import Any, Dict, List, Union


def handle_types(obj: Any) -> Any:
    """
    Convert numpy types to Python native types for JSON serialization.
    
    Args:
        obj: Object that may contain numpy types
        
    Returns:
        Object with numpy types converted to Python native types
    """
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: handle_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [handle_types(item) for item in obj]
    else:
        return obj


def save_json(data: Dict, filepath: str, indent: int = 2) -> None:
    """
    Save dictionary to JSON file with proper type handling.
    
    Args:
        data: Dictionary to save
        filepath: Output file path
        indent: JSON indentation level
    """
    processed_data = handle_types(data)
    with open(filepath, 'w') as f:
        json.dump(processed_data, f, indent=indent)


def load_json(filepath: str) -> Dict:
    """
    Load JSON file to dictionary.
    
    Args:
        filepath: Input file path
        
    Returns:
        Dictionary loaded from JSON
    """
    with open(filepath, 'r') as f:
        return json.load(f)


def compute_diameter(coords: np.ndarray, method: str = 'mean_centered') -> float:
    """
    Compute diameter of point cloud.
    
    Args:
        coords: (N, D) array of coordinates
        method: Method to use ('mean_centered' or 'pairwise_max')
        
    Returns:
        Estimated diameter
    """
    if method == 'mean_centered':
        # Faster method: 2 * max distance from mean
        mean_point = np.mean(coords, axis=0)
        centered_coords = coords - mean_point
        norms = np.linalg.norm(centered_coords, axis=1)
        return 2.0 * np.max(norms)
    elif method == 'pairwise_max':
        # Exact method: maximum pairwise distance (slower)
        from sklearn.metrics import pairwise_distances
        dists = pairwise_distances(coords, n_jobs=-1)
        return float(np.max(dists))
    else:
        raise ValueError(f"Unknown method: {method}")


def create_train_test_split(clean_data: np.ndarray, noisy_data: np.ndarray, 
                           train_ratio: float = 0.8, random_seed: int = 42) -> Dict[str, np.ndarray]:
    """
    Create train/test split with proper shuffling.
    
    Args:
        clean_data: (N, D) clean coordinates
        noisy_data: (N, D) noisy coordinates  
        train_ratio: Fraction for training
        random_seed: Random seed for reproducibility
        
    Returns:
        Dictionary with train/test data and labels (noise)
    """
    np.random.seed(random_seed)
    n_total = len(clean_data)
    n_train = int(n_total * train_ratio)
    
    # Create shuffled indices
    indices = np.arange(n_total)
    np.random.shuffle(indices)
    
    train_indices = indices[:n_train]
    test_indices = indices[n_train:]
    
    # Split data
    train_clean = clean_data[train_indices]
    train_noisy = noisy_data[train_indices]
    test_clean = clean_data[test_indices]
    test_noisy = noisy_data[test_indices]
    
    # Compute noise labels (noise = noisy - clean)
    train_noise = train_noisy - train_clean
    test_noise = test_noisy - test_clean
    
    return {
        'train_data': train_noisy,      # Input: noisy data
        'train_labels': train_noise,    # Target: pure noise
        'test_data': test_noisy,        # Input: noisy data
        'test_labels': test_noise,      # Target: pure noise
        'train_clean': train_clean,     # Reference: clean data
        'test_clean': test_clean        # Reference: clean data
    }
