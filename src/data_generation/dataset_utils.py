"""
Dataset utilities for loading and creating manifold datasets.

This module provides utilities for loading generated datasets and converting them
into PyTorch-compatible formats for training.
"""

import os
import json
import torch
import numpy as np
from torch.utils.data import TensorDataset
from .dataset_classes import Dataset, DatasetProperties


def load_dataset_metadata(dataset_path):
    """Load dataset metadata from file."""
    metadata_path = f"{dataset_path}_metadata.json"
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    return metadata


def create_manifold_dataset(dataset_path):
    """
    Create PyTorch datasets from saved manifold data.
    
    Args:
        dataset_path: Path to the dataset files (without extension)
        
    Returns:
        tuple: (train_dataset, test_dataset, metadata)
    """
    # Load metadata
    metadata = load_dataset_metadata(dataset_path)
    
    # Load the Dataset object
    dataset = Dataset.load(dataset_path)
    
    # Convert to PyTorch tensors
    # Use clean_embedded as input and noisy_embedded as target for denoising
    input_data = torch.FloatTensor(dataset.noisy_embedded)
    target_data = torch.FloatTensor(dataset.clean_embedded)
    
    # Create train/test split if not already done
    if hasattr(dataset, 'train_test_data') and dataset.train_test_data:
        # Use pre-split data
        train_input = torch.FloatTensor(dataset.train_test_data['train_noisy'])
        train_target = torch.FloatTensor(dataset.train_test_data['train_clean'])
        test_input = torch.FloatTensor(dataset.train_test_data['test_noisy'])
        test_target = torch.FloatTensor(dataset.train_test_data['test_clean'])
    else:
        # Create 80/20 split
        n_total = len(input_data)
        n_train = int(0.8 * n_total)
        
        # Random split
        indices = torch.randperm(n_total)
        train_indices = indices[:n_train]
        test_indices = indices[n_train:]
        
        train_input = input_data[train_indices]
        train_target = target_data[train_indices]
        test_input = input_data[test_indices]
        test_target = target_data[test_indices]
    
    # Create PyTorch datasets
    train_dataset = TensorDataset(train_input, train_target)
    test_dataset = TensorDataset(test_input, test_target)
    
    return train_dataset, test_dataset, metadata


def create_simple_dataset(dataset_path):
    """
    Create a simple dataset where input equals output (for identity mapping).
    
    Args:
        dataset_path: Path to the dataset files
        
    Returns:
        tuple: (train_dataset, test_dataset, metadata)
    """
    # Load metadata
    metadata = load_dataset_metadata(dataset_path)
    
    # Load the Dataset object
    dataset = Dataset.load(dataset_path)
    
    # For simple identity mapping, use clean data as both input and target
    data = torch.FloatTensor(dataset.clean_embedded)
    
    # Create train/test split
    n_total = len(data)
    n_train = int(0.8 * n_total)
    
    indices = torch.randperm(n_total)
    train_indices = indices[:n_train]
    test_indices = indices[n_train:]
    
    train_data = data[train_indices]
    test_data = data[test_indices]
    
    # Create datasets where input == target
    train_dataset = TensorDataset(train_data, train_data)
    test_dataset = TensorDataset(test_data, test_data)
    
    return train_dataset, test_dataset, metadata


def get_dataset_info(dataset_path):
    """Get basic information about a dataset."""
    metadata = load_dataset_metadata(dataset_path)
    dataset = Dataset.load(dataset_path)
    
    info = {
        'path': dataset_path,
        'metadata': metadata,
        'n_samples': len(dataset.clean_embedded),
        'ambient_dim': dataset.clean_embedded.shape[1],
        'intrinsic_dim': metadata.get('d', 'unknown'),
        'noise_level': metadata.get('noise_std', 'unknown'),
        'kernel_smoothness': metadata.get('kernel_smoothness', 'unknown')
    }
    
    return info
