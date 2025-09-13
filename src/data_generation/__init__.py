"""
Data Generation Module

This module handles the generation of synthetic manifold datasets.
Step 1 of the pipeline: Generate datasets with various geometric properties.

Key components:
- Manifold generators using Gaussian Process approach
- Dataset properties and management classes
- Normalization and noise addition utilities
- Systematic dataset generation for experiments

Main classes:
- DatasetGenerator: Main interface for generating dataset collections
- Dataset: Individual dataset with all associated data
- DatasetProperties: Properties and metadata for datasets
"""

from .dataset_generator import DatasetGenerator
from .dataset_classes import Dataset, DatasetProperties, NormalizationInfo
from .manifold_generator import (
    generate_manifold_embedding,
    generate_intrinsic_coordinates,
    apply_normalization,
    add_noise,
    rbf_kernel
)

__all__ = [
    'DatasetGenerator',
    'Dataset', 
    'DatasetProperties',
    'NormalizationInfo',
    'generate_manifold_embedding',
    'generate_intrinsic_coordinates', 
    'apply_normalization',
    'add_noise',
    'rbf_kernel'
]
