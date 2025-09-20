#!/usr/bin/env python3
"""
Curvature vs Kernel Smoothness Analysis using Modular Framework

This script implements the new modular approach for curvature estimation
with proper neighbor selection strategies.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Add src to path
script_dir = Path(__file__).parent
project_root = script_dir.parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

# Import our new modular functions
from geometric_analysis import (
    compute_manifold_curvature,
    compute_curvature_vs_parameter
)


def load_dataset_with_metadata(data_dir: Path, dataset_idx: int) -> Dict:
    """Load dataset with its metadata."""
    dataset_dir = data_dir / f"dataset{dataset_idx}"
    
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")
    
    # Try different file patterns
    embedded_file = dataset_dir / f"dataset{dataset_idx}_clean.npy"
    intrinsic_file = dataset_dir / f"dataset{dataset_idx}_intrinsic.npy"
    metadata_file = dataset_dir / f"dataset{dataset_idx}_metadata.json"
    
    if not embedded_file.exists():
        raise FileNotFoundError(f"Embedded data not found: {embedded_file}")
    if not intrinsic_file.exists():
        raise FileNotFoundError(f"Intrinsic data not found: {intrinsic_file}")
    if not metadata_file.exists():
        raise FileNotFoundError(f"Metadata not found: {metadata_file}")
    
    # Load data
    embedded_coords = np.load(embedded_file)
    intrinsic_coords = np.load(intrinsic_file)
    
    # Load metadata
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
    
    # Extract kernel_smoothness from metadata structure
    kernel_smoothness = metadata.get('properties', {}).get('kernel_smoothness', 'unknown')
    
    return {
        'embedded_coords': embedded_coords,
        'intrinsic_coords': intrinsic_coords,
        'metadata': metadata,
        'kernel_smoothness': kernel_smoothness,
        'dataset_idx': dataset_idx
    }


def analyze_curvature_vs_kernel_smoothness(
    data_dir: str,
    dataset_indices: List[int],
    curvature_method: str = "mean",
    neighbor_strategy: str = "k_nearest",
    k_neighbors: Optional[int] = None,
    radius_values: Optional[List[float]] = None,
    n_sample_points: int = 1000,
    output_dir: str = None,
    random_state: int = 42
) -> Dict:
    """
    Analyze curvature vs kernel smoothness using the modular framework.
    
    Args:
        data_dir: Directory containing the datasets
        dataset_indices: List of dataset indices to analyze
        curvature_method: Method for curvature estimation ("mean" or "pca")
        neighbor_strategy: Strategy for neighbor selection ("k_nearest" or "radius")
        k_neighbors: Number of neighbors (for k_nearest strategy) or list of k values
        radius_values: List of radius values (for radius strategy)
        n_sample_points: Number of points to sample from each dataset
        output_dir: Directory to save results
        random_state: Random seed
        
    Returns:
        Dictionary with analysis results
    """
    data_path = Path(data_dir)
    
    print(f"Analyzing curvature vs kernel smoothness...")
    print(f"Data directory: {data_path}")
    print(f"Dataset indices: {dataset_indices}")
    print(f"Curvature method: {curvature_method}")
    print(f"Neighbor strategy: {neighbor_strategy}")
    print(f"Sample points per dataset: {n_sample_points}")
    
    # Load datasets
    print("\\nLoading datasets...")
    datasets = []
    kernel_smoothness_values = []
    
    for idx in dataset_indices:
        try:
            dataset = load_dataset_with_metadata(data_path, idx)
            datasets.append(dataset)
            kernel_smoothness_values.append(dataset['kernel_smoothness'])
            print(f"  Dataset {idx}: kernel_smoothness = {dataset['kernel_smoothness']}")
        except Exception as e:
            print(f"  Failed to load dataset {idx}: {e}")
            continue
    
    if len(datasets) == 0:
        raise ValueError("No datasets were successfully loaded")
    
    # Determine analysis parameters based on strategy
    if neighbor_strategy == "k_nearest":
        if k_neighbors is None:
            k_neighbors = [10, 20, 30]
        elif not isinstance(k_neighbors, list):
            k_neighbors = [k_neighbors]
        analysis_params = k_neighbors
        param_name = "k_neighbors"
    elif neighbor_strategy == "radius":
        if radius_values is None:
            radius_values = [0.1, 0.2, 0.3]
        analysis_params = radius_values
        param_name = "radius"
    else:
        raise ValueError(f"Unknown neighbor strategy: {neighbor_strategy}")
    
    # Run analysis for each parameter value
    results = {}
    
    for param_val in analysis_params:
        print(f"\\nAnalyzing with {param_name} = {param_val}...")
        
        # Prepare computation kwargs
        computation_kwargs = {
            'n_sample_points': n_sample_points,
            'random_state': random_state
        }
        
        if neighbor_strategy == "k_nearest":
            computation_kwargs['k_neighbors'] = param_val
        elif neighbor_strategy == "radius":
            computation_kwargs['radius'] = param_val
            computation_kwargs['k_neighbors'] = None  # No limit on neighbors within radius
        
        # Run analysis
        param_results = compute_curvature_vs_parameter(
            datasets=datasets,
            parameter_name="kernel_smoothness",
            parameter_values=kernel_smoothness_values,
            curvature_method=curvature_method,
            neighbor_strategy=neighbor_strategy,
            **computation_kwargs
        )
        
        results[param_val] = param_results
    
    # Compile summary
    summary_data = []
    for param_val, param_results in results.items():
        for i, (kernel_smooth, stats) in enumerate(zip(
            param_results["parameter_values"], 
            param_results["curvature_statistics"]
        )):
            dataset_idx = datasets[i]['dataset_idx']
            
            row = {
                'dataset': f'dataset{dataset_idx}',
                'kernel_smoothness': kernel_smooth,
                param_name: param_val,
                'curvature_mean': stats['mean'],
                'curvature_std': stats['std'],
                'curvature_median': stats['median'],
                'num_valid': stats['num_valid'],
                'num_total': stats['num_total'],
                'success_rate': stats['success_rate']
            }
            summary_data.append(row)
    
    summary_df = pd.DataFrame(summary_data)
    
    # Save results if output directory specified
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save detailed results
        results_file = output_path / f"curvature_vs_kernel_smoothness_{neighbor_strategy}_{curvature_method}.json"
        with open(results_file, 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            json_results = {}
            for param_val, param_results in results.items():
                json_results[str(param_val)] = {
                    "parameter_name": param_results["parameter_name"],
                    "parameter_values": param_results["parameter_values"],
                    "curvature_statistics": param_results["curvature_statistics"],
                    "metadata": param_results["metadata"]
                }
            json.dump(json_results, f, indent=2)
        
        # Save summary CSV
        summary_file = output_path / f"curvature_vs_kernel_smoothness_{neighbor_strategy}_{curvature_method}_summary.csv"
        summary_df.to_csv(summary_file, index=False)
        
        print(f"\\nResults saved to {output_path}")
    
    return {
        'detailed_results': results,
        'summary_df': summary_df,
        'datasets': datasets,
        'analysis_metadata': {
            'curvature_method': curvature_method,
            'neighbor_strategy': neighbor_strategy,
            'n_sample_points': n_sample_points,
            'random_state': random_state
        }
    }


def plot_curvature_vs_kernel_smoothness(
    results: Dict,
    output_dir: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 8)
) -> None:
    """
    Create plots for curvature vs kernel smoothness analysis.
    
    Args:
        results: Results dictionary from analyze_curvature_vs_kernel_smoothness
        output_dir: Directory to save plots
        figsize: Figure size
    """
    summary_df = results['summary_df']
    analysis_meta = results['analysis_metadata']
    
    neighbor_strategy = analysis_meta['neighbor_strategy']
    curvature_method = analysis_meta['curvature_method']
    
    # Determine parameter name
    param_name = 'k_neighbors' if neighbor_strategy == 'k_nearest' else 'radius'
    param_values = sorted(summary_df[param_name].unique())
    
    # Create plots
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle(f'Curvature vs Kernel Smoothness\\n({curvature_method.title()} Curvature, {neighbor_strategy.replace("_", " ").title()} Strategy)', 
                 fontsize=14, fontweight='bold')
    
    # Define colors for different parameter values
    colors = plt.cm.viridis(np.linspace(0, 1, len(param_values)))
    
    for i, param_val in enumerate(param_values):
        data = summary_df[summary_df[param_name] == param_val]
        color = colors[i]
        label = f'{param_name}={param_val}'
        
        # Plot 1: Mean curvature vs kernel smoothness
        axes[0, 0].semilogx(data['kernel_smoothness'], data['curvature_mean'], 
                           'o-', color=color, label=label, markersize=6)
        axes[0, 0].set_xlabel('Kernel Smoothness')
        axes[0, 0].set_ylabel('Mean Curvature')
        axes[0, 0].set_title('Mean Curvature vs Kernel Smoothness')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].legend()
        
        # Plot 2: Standard deviation
        axes[0, 1].semilogx(data['kernel_smoothness'], data['curvature_std'],
                           'o-', color=color, label=label, markersize=6)
        axes[0, 1].set_xlabel('Kernel Smoothness')
        axes[0, 1].set_ylabel('Curvature Std Dev')
        axes[0, 1].set_title('Curvature Std Dev vs Kernel Smoothness')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].legend()
        
        # Plot 3: Success rate
        axes[1, 0].semilogx(data['kernel_smoothness'], data['success_rate'] * 100,
                           'o-', color=color, label=label, markersize=6)
        axes[1, 0].set_xlabel('Kernel Smoothness')
        axes[1, 0].set_ylabel('Success Rate (%)')
        axes[1, 0].set_title('Estimation Success Rate vs Kernel Smoothness')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].legend()
        
        # Plot 4: Number of valid estimates
        axes[1, 1].semilogx(data['kernel_smoothness'], data['num_valid'],
                           'o-', color=color, label=label, markersize=6)
        axes[1, 1].set_xlabel('Kernel Smoothness')
        axes[1, 1].set_ylabel('Number of Valid Estimates')
        axes[1, 1].set_title('Valid Estimates vs Kernel Smoothness')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].legend()
    
    plt.tight_layout()
    
    # Save plot if output directory specified
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        plot_file = output_path / f"curvature_vs_kernel_smoothness_{neighbor_strategy}_{curvature_method}.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {plot_file}")
    
    plt.show()


def main():
    """Main function to run the analysis."""
    # Configuration
    data_dir = "/home/tim/python_projects/nn_manifold_denoising/data/data_250914_0100"
    output_dir = "/home/tim/python_projects/nn_manifold_denoising/results/curvature_modular_analysis"
    
    # Analysis parameters
    dataset_indices = [0, 1, 2, 3, 4]  # First 5 datasets
    n_sample_points = 1000
    random_state = 42
    
    # Test both strategies
    strategies = [
        {
            'neighbor_strategy': 'k_nearest',
            'k_neighbors': [10, 20, 30],
            'curvature_method': 'mean'
        },
        {
            'neighbor_strategy': 'k_nearest', 
            'k_neighbors': [10, 20, 30],
            'curvature_method': 'pca'
        }
    ]
    
    # Run analyses
    for i, strategy in enumerate(strategies):
        print(f"\\n{'='*60}")
        print(f"Running Analysis {i+1}/{len(strategies)}")
        print(f"Strategy: {strategy}")
        print(f"{'='*60}")
        
        try:
            # Run analysis
            results = analyze_curvature_vs_kernel_smoothness(
                data_dir=data_dir,
                dataset_indices=dataset_indices,
                n_sample_points=n_sample_points,
                output_dir=output_dir,
                random_state=random_state,
                **strategy
            )
            
            # Create plots
            plot_curvature_vs_kernel_smoothness(
                results=results,
                output_dir=output_dir
            )
            
            print(f"\\nAnalysis {i+1} completed successfully!")
            
        except Exception as e:
            print(f"\\nError in analysis {i+1}: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
