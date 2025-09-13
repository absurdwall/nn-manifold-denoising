"""
Dataset Generator - Main interface for generating manifold datasets.

This is the clean, organized version of step1_generate_datasets_working.py
that uses the modular components for dataset generation.
"""

import os
import time
import numpy as np
from datetime import datetime
from typing import List, Dict, Optional, Any
from itertools import product

from .manifold_generator import generate_manifold_embedding, apply_normalization, add_noise
from .dataset_classes import Dataset, DatasetProperties, NormalizationInfo
from ..utils.data_utils import save_json


class DatasetGenerator:
    """
    Main class for generating collections of manifold datasets.
    """
    
    def __init__(self, experiment_id: Optional[str] = None, base_output_dir: str = "data"):
        """
        Initialize dataset generator.
        
        Args:
            experiment_id: Unique experiment identifier (auto-generated if None)
            base_output_dir: Base directory for saving datasets
        """
        if experiment_id is None:
            # Auto-generate experiment ID based on current date/time
            now = datetime.now()
            experiment_id = f"tb{now.year-2000:02d}{now.month:02d}{now.day:02d}{now.hour:02d}"
        
        self.experiment_id = experiment_id
        self.base_output_dir = base_output_dir
        self.output_dir = os.path.join(base_output_dir, experiment_id)
        
        # Find unique output directory
        counter = 1
        while os.path.exists(self.output_dir):
            self.output_dir = os.path.join(base_output_dir, f"{experiment_id}_{counter:02d}")
            counter += 1
        
        self.datasets: List[Dataset] = []
        self.generation_summary: Dict[str, Any] = {}
        
    def generate_parameter_grid(self, k_values: List[int], N: int, ds: List[int], 
                              Ds: List[int], kernel_smoothness: List[float],
                              noise_sigma: float, base_type: str = 'unit_ball',
                              train_ratio: float = 0.8, centralize: bool = True,
                              rescale: bool = True, random_seed: int = 42) -> List[DatasetProperties]:
        """
        Generate a grid of dataset properties for systematic experiments.
        
        Args:
            k_values: List of number of groups
            N: Points per group
            ds: List of intrinsic dimensions
            Ds: List of ambient dimensions  
            kernel_smoothness: List of kernel smoothness values
            noise_sigma: Noise level
            base_type: Base manifold type
            train_ratio: Training data fraction
            centralize: Whether to centralize data
            rescale: Whether to rescale data
            random_seed: Base random seed
            
        Returns:
            List of DatasetProperties for each parameter combination
        """
        properties_list = []
        dataset_num = 0
        
        for d, D, k, sigma in product(ds, Ds, k_values, kernel_smoothness):
            props = DatasetProperties(
                k=k, N=N, d=d, D=D,
                kernel_smoothness=sigma,
                noise_sigma=noise_sigma,
                base_type=base_type,
                dataset_num=dataset_num,
                experiment_id=self.experiment_id,
                random_seed=random_seed + dataset_num,
                train_ratio=train_ratio,
                centralize=centralize,
                rescale=rescale
            )
            properties_list.append(props)
            dataset_num += 1
            
        return properties_list
    
    def generate_single_dataset(self, properties: DatasetProperties, 
                               create_train_test: bool = True,
                               verbose: bool = True) -> Dataset:
        """
        Generate a single dataset with given properties.
        
        Args:
            properties: Dataset properties
            create_train_test: Whether to create train/test split
            verbose: Whether to print progress
            
        Returns:
            Generated Dataset object
        """
        if verbose:
            print(f"Generating dataset {properties.dataset_num}: "
                  f"d={properties.d}, D={properties.D}, k={properties.k}, "
                  f"N={properties.N}, σ={properties.kernel_smoothness}")
        
        start_time = time.time()
        
        # 1. Generate raw manifold embedding
        raw_embedded, intrinsic_coords = generate_manifold_embedding(
            k=properties.k,
            N=properties.N,
            d=properties.d,
            D=properties.D,
            kernel_smoothness=properties.kernel_smoothness,
            base_type=properties.base_type,
            random_seed=properties.random_seed,
            verbose=False  # Suppress detailed output for cleaner logs
        )
        
        if verbose:
            print(f"  ✓ Raw generation: {raw_embedded.shape}")
        
        # 2. Apply normalization
        clean_embedded, norm_info_dict = apply_normalization(
            raw_embedded,
            centralize=properties.centralize,
            rescale=properties.rescale,
            verbose=verbose
        )
        
        # Convert normalization info to proper dataclass
        normalization_info = NormalizationInfo(**norm_info_dict)
        
        # 3. Add noise to create noisy version
        noisy_embedded = None
        if properties.noise_sigma > 0:
            noisy_embedded = add_noise(
                clean_embedded,
                properties.noise_sigma,
                properties.random_seed + 1000  # Different seed for noise
            )
            if verbose:
                print(f"  ✓ Added noise (σ={properties.noise_sigma}): {noisy_embedded.shape}")
        
        generation_time = time.time() - start_time
        
        # 4. Create dataset object
        dataset = Dataset(properties)
        dataset.set_data(
            raw_embedded=raw_embedded,
            clean_embedded=clean_embedded,
            intrinsic_coords=intrinsic_coords,
            normalization_info=normalization_info,
            noisy_embedded=noisy_embedded,
            generation_time=generation_time
        )
        
        # 5. Create train/test split if requested
        if create_train_test and noisy_embedded is not None:
            dataset.create_train_test_split()
            if verbose:
                split_info = dataset.train_test_data
                print(f"  ✓ Train/test split: {split_info['train_data'].shape}/{split_info['test_data'].shape}")
        
        if verbose:
            print(f"  ✓ Dataset {properties.dataset_num} completed in {generation_time:.2f}s")
        
        return dataset
    
    def generate_datasets(self, properties_list: List[DatasetProperties],
                         create_train_test: bool = True,
                         save_individual: bool = True,
                         verbose: bool = True) -> List[Dataset]:
        """
        Generate multiple datasets from properties list.
        
        Args:
            properties_list: List of dataset properties
            create_train_test: Whether to create train/test splits
            save_individual: Whether to save each dataset individually
            verbose: Whether to print progress
            
        Returns:
            List of generated Dataset objects
        """
        os.makedirs(self.output_dir, exist_ok=True)
        
        if verbose:
            print(f"Dataset Generation Experiment: {self.experiment_id}")
            print("=" * 60)
            print(f"Output directory: {self.output_dir}")
            print(f"Total datasets: {len(properties_list)}")
        
        # Save experiment settings
        experiment_settings = {
            'experiment_id': self.experiment_id,
            'total_datasets': len(properties_list),
            'created_at': datetime.now().isoformat(),
            'output_directory': self.output_dir,
            'dataset_properties': [props.to_dict() for props in properties_list]
        }
        save_json(experiment_settings, os.path.join(self.output_dir, 'experiment_settings.json'))
        
        # Generate datasets
        successful_datasets = []
        failed_datasets = []
        start_time = time.time()
        
        for i, properties in enumerate(properties_list):
            if verbose:
                print(f"\n[{i+1}/{len(properties_list)}] ", end="")
            
            try:
                dataset = self.generate_single_dataset(
                    properties, create_train_test, verbose
                )
                successful_datasets.append(dataset)
                
                # Save individual dataset if requested
                if save_individual:
                    saved_files = dataset.save_to_directory(self.output_dir)
                    if verbose:
                        print(f"  ✓ Saved {len(saved_files)} files")
                        
            except Exception as e:
                error_info = {
                    'dataset_num': properties.dataset_num,
                    'error': str(e),
                    'properties': properties.to_dict()
                }
                failed_datasets.append(error_info)
                if verbose:
                    print(f"  ✗ Error: {e}")
        
        total_time = time.time() - start_time
        
        # Create generation summary
        self.generation_summary = {
            'experiment_id': self.experiment_id,
            'total_datasets': len(properties_list),
            'successful_datasets': len(successful_datasets),
            'failed_datasets': len(failed_datasets),
            'total_time': total_time,
            'average_time_per_dataset': total_time / len(successful_datasets) if successful_datasets else 0,
            'successful_dataset_nums': [d.properties.dataset_num for d in successful_datasets],
            'failed_dataset_info': failed_datasets,
            'output_directory': self.output_dir
        }
        
        # Save summary
        save_json(self.generation_summary, os.path.join(self.output_dir, 'generation_summary.json'))
        
        if verbose:
            print("=" * 60)
            print("Generation Summary:")
            print(f"  Total time: {total_time:.2f}s ({total_time/60:.2f}m)")
            print(f"  Successful: {len(successful_datasets)}/{len(properties_list)}")
            print(f"  Failed: {len(failed_datasets)}/{len(properties_list)}")
            if successful_datasets:
                avg_time = total_time / len(successful_datasets)
                print(f"  Average per dataset: {avg_time:.2f}s")
            print(f"  Results saved to: {self.output_dir}")
        
        self.datasets = successful_datasets
        return successful_datasets
    
    def get_experiment_summary(self) -> Dict[str, Any]:
        """Get comprehensive experiment summary."""
        if not self.generation_summary:
            return {"error": "No generation summary available. Run generate_datasets() first."}
        
        summary = self.generation_summary.copy()
        
        # Add dataset statistics if datasets were generated
        if self.datasets:
            # Collect parameter statistics
            params_stats = {
                'd_values': list(set(d.properties.d for d in self.datasets)),
                'D_values': list(set(d.properties.D for d in self.datasets)),
                'k_values': list(set(d.properties.k for d in self.datasets)),
                'kernel_smoothness_values': list(set(d.properties.kernel_smoothness for d in self.datasets)),
                'base_types': list(set(d.properties.base_type for d in self.datasets))
            }
            summary['parameter_statistics'] = params_stats
            
            # Add data shape statistics
            shape_examples = {}
            if self.datasets:
                example_dataset = self.datasets[0]
                if example_dataset.clean_embedded is not None:
                    shape_examples['embedded_shape_example'] = list(example_dataset.clean_embedded.shape)
                if example_dataset.intrinsic_coords is not None:
                    shape_examples['intrinsic_shape_example'] = list(example_dataset.intrinsic_coords.shape)
            summary['shape_examples'] = shape_examples
            
        return summary
