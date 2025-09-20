#!/usr/bin/env python3
"""
Curvature vs Kernel Smoothness with Fixed Radius - Modular Version

This script implements the radius-based analysis using the modular framework,
ensuring we get results comparable to the previous fixed-radius analysis.
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

# Import our modular functions
from geometric_analysis import compute_manifold_curvature


def load_dataset_with_metadata(data_dir: Path, dataset_idx: int) -> Dict:
    """Load dataset with its metadata."""
    dataset_dir = data_dir / f"dataset{dataset_idx}"
    
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")
    
    # Load files
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


def analyze_curvature_vs_kernel_smoothness_radius(
    data_dir: str,
    dataset_indices: List[int] = [0, 1, 2, 3, 4],
    radius_values: List[float] = [0.05, 0.1, 0.15, 0.2, 0.25],
    n_sample_points: int = 1000,
    output_dir: str = None,
    random_state: int = 42
) -> Dict:
    """
    Analyze curvature vs kernel smoothness using fixed radius neighbors.
    
    Args:
        data_dir: Directory containing the datasets
        dataset_indices: List of dataset indices to analyze
        radius_values: List of radius values for neighbor selection
        n_sample_points: Number of points to sample from each dataset
        output_dir: Directory to save results
        random_state: Random seed
        
    Returns:
        Dictionary with analysis results
    """
    data_path = Path(data_dir)
    
    print(f"Analyzing curvature vs kernel smoothness with fixed radius...")
    print(f"Data directory: {data_path}")
    print(f"Dataset indices: {dataset_indices}")
    print(f"Radius values: {radius_values}")
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
    
    # Analyze for each radius value
    all_results = {}
    summary_data = []
    
    for radius in radius_values:
        print(f"\\nAnalyzing with radius = {radius}...")
        
        radius_results = {
            "radius": radius,
            "datasets": [],
            "curvature_statistics": []
        }
        
        for i, dataset in enumerate(datasets):
            dataset_idx = dataset['dataset_idx']
            kernel_smooth = dataset['kernel_smoothness']
            
            print(f"  Processing dataset {dataset_idx} (kernel_smoothness={kernel_smooth})...")
            
            # Test both curvature methods
            for method in ["mean", "pca"]:
                try:
                    result = compute_manifold_curvature(
                        embedded_coords=dataset["embedded_coords"],
                        intrinsic_coords=dataset["intrinsic_coords"],
                        curvature_method=method,
                        neighbor_strategy="radius",
                        radius=radius,
                        n_sample_points=n_sample_points,
                        random_state=random_state
                    )
                    
                    stats = result["statistics"]
                    
                    # Count average neighbors from diagnostics
                    neighbor_counts = []
                    for diag in result["diagnostics"]:
                        if "num_neighbors" in diag and diag["num_neighbors"] > 0:
                            neighbor_counts.append(diag["num_neighbors"])
                    
                    avg_neighbors = np.mean(neighbor_counts) if neighbor_counts else 0
                    
                    # Add to summary
                    row = {
                        'dataset': f'dataset{dataset_idx}',
                        'kernel_smoothness': kernel_smooth,
                        'radius': radius,
                        'method': method,
                        'avg_neighbors': avg_neighbors,
                        'curvature_mean': stats['mean'],
                        'curvature_std': stats['std'],
                        'curvature_median': stats['median'],
                        'num_valid': stats['num_valid'],
                        'num_total': stats['num_total'],
                        'success_rate': stats['success_rate']
                    }
                    summary_data.append(row)
                    
                    # Debug output for mean curvature method
                    if method == "mean" and stats['success_rate'] > 0:
                        valid_curvatures = result["curvature_values"][np.isfinite(result["curvature_values"])]
                        if len(valid_curvatures) > 0:
                            print(f"    {method.upper()}: mean={stats['mean']:.4f}, "
                                  f"range=[{np.min(valid_curvatures):.4f}, {np.max(valid_curvatures):.4f}], "
                                  f"success={stats['success_rate']:.1%}, "
                                  f"avg_neighbors={avg_neighbors:.1f}")
                        else:
                            print(f"    {method.upper()}: No valid curvatures!")
                    else:
                        print(f"    {method.upper()}: mean={stats['mean']:.4f}, "
                              f"success={stats['success_rate']:.1%}, "
                              f"avg_neighbors={avg_neighbors:.1f}")
                    
                except Exception as e:
                    print(f"    {method.upper()}: Failed - {e}")
                    
                    # Add failed entry
                    row = {
                        'dataset': f'dataset{dataset_idx}',
                        'kernel_smoothness': kernel_smooth,
                        'radius': radius,
                        'method': method,
                        'avg_neighbors': 0,
                        'curvature_mean': np.nan,
                        'curvature_std': np.nan,
                        'curvature_median': np.nan,
                        'num_valid': 0,
                        'num_total': n_sample_points,
                        'success_rate': 0.0
                    }
                    summary_data.append(row)
        
        all_results[radius] = radius_results
    
    # Create summary DataFrame
    summary_df = pd.DataFrame(summary_data)
    
    # Save results if output directory specified
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save summary CSV
        summary_file = output_path / "curvature_vs_kernel_smoothness_radius_modular.csv"
        summary_df.to_csv(summary_file, index=False)
        
        # Save detailed results  
        results_file = output_path / "curvature_vs_kernel_smoothness_radius_modular.json"
        # Convert for JSON serialization
        json_results = {
            "summary_data": summary_data,
            "metadata": {
                "radius_values": radius_values,
                "dataset_indices": dataset_indices,
                "n_sample_points": n_sample_points,
                "random_state": random_state
            }
        }
        with open(results_file, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        print(f"\\nResults saved to {output_path}")
    
    return {
        'summary_df': summary_df,
        'all_results': all_results,
        'datasets': datasets
    }


def plot_curvature_vs_kernel_smoothness_radius(
    results: Dict,
    output_dir: Optional[str] = None,
    figsize: Tuple[int, int] = (15, 10)
) -> None:
    """
    Create plots for curvature vs kernel smoothness with different radius values.
    
    Args:
        results: Results dictionary from analyze_curvature_vs_kernel_smoothness_radius
        output_dir: Directory to save plots
        figsize: Figure size
    """
    summary_df = results['summary_df']
    
    # Create separate plots for each method
    methods = summary_df['method'].unique()
    
    for method in methods:
        method_data = summary_df[summary_df['method'] == method]
        radius_values = sorted(method_data['radius'].unique())
        
        # Create plots
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle(f'Curvature vs Kernel Smoothness ({method.upper()} Method, Radius Strategy)', 
                     fontsize=14, fontweight='bold')
        
        # Define colors for different radius values
        colors = plt.cm.viridis(np.linspace(0, 1, len(radius_values)))
        
        for i, radius in enumerate(radius_values):
            data = method_data[method_data['radius'] == radius]
            # Filter out failed cases
            data = data[data['success_rate'] > 0]
            
            if len(data) == 0:
                continue
                
            color = colors[i]
            label = f'r={radius}'
            
            # Plot 1: Mean curvature vs kernel smoothness
            axes[0, 0].semilogx(data['kernel_smoothness'], data['curvature_mean'], 
                               'o-', color=color, label=label, markersize=6)
            axes[0, 0].set_xlabel('Kernel Smoothness')
            axes[0, 0].set_ylabel('Mean Curvature')
            axes[0, 0].set_title('Mean Curvature vs Kernel Smoothness')
            axes[0, 0].grid(True, alpha=0.3)
            axes[0, 0].legend()
            
            # Plot 2: Success rate
            axes[0, 1].semilogx(data['kernel_smoothness'], data['success_rate'] * 100,
                               'o-', color=color, label=label, markersize=6)
            axes[0, 1].set_xlabel('Kernel Smoothness')
            axes[0, 1].set_ylabel('Success Rate (%)')
            axes[0, 1].set_title('Success Rate vs Kernel Smoothness')
            axes[0, 1].grid(True, alpha=0.3)
            axes[0, 1].legend()
            
            # Plot 3: Average neighbors
            axes[1, 0].semilogx(data['kernel_smoothness'], data['avg_neighbors'],
                               'o-', color=color, label=label, markersize=6)
            axes[1, 0].set_xlabel('Kernel Smoothness')
            axes[1, 0].set_ylabel('Average Neighbors')
            axes[1, 0].set_title('Average Neighbors vs Kernel Smoothness')
            axes[1, 0].grid(True, alpha=0.3)
            axes[1, 0].legend()
            
            # Plot 4: Standard deviation
            axes[1, 1].semilogx(data['kernel_smoothness'], data['curvature_std'],
                               'o-', color=color, label=label, markersize=6)
            axes[1, 1].set_xlabel('Kernel Smoothness')
            axes[1, 1].set_ylabel('Curvature Std Dev')
            axes[1, 1].set_title('Curvature Std Dev vs Kernel Smoothness')
            axes[1, 1].grid(True, alpha=0.3)
            axes[1, 1].legend()
        
        plt.tight_layout()
        
        # Save plot if output directory specified
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            plot_file = output_path / f"curvature_vs_kernel_smoothness_radius_{method}.png"
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {plot_file}")
        
        plt.show()


def main():
    """Main function to run the radius-based analysis."""
    # Configuration
    data_dir = "/home/tim/python_projects/nn_manifold_denoising/data/data_250914_0100"
    output_dir = "/home/tim/python_projects/nn_manifold_denoising/results/curvature_radius_modular"
    
    # Analysis parameters
    dataset_indices = [0, 1, 2, 3, 4]  # First 5 datasets
    radius_values = [0.05, 0.1, 0.15, 0.2, 0.25]  # Same as previous analysis
    n_sample_points = 1000
    random_state = 42
    
    print("Curvature vs Kernel Smoothness Analysis - Fixed Radius Strategy")
    print("=" * 70)
    
    try:
        # Run analysis
        results = analyze_curvature_vs_kernel_smoothness_radius(
            data_dir=data_dir,
            dataset_indices=dataset_indices,
            radius_values=radius_values,
            n_sample_points=n_sample_points,
            output_dir=output_dir,
            random_state=random_state
        )
        
        # Create plots
        plot_curvature_vs_kernel_smoothness_radius(
            results=results,
            output_dir=output_dir
        )
        
        print("\\nAnalysis completed successfully!")
        
        # Print summary
        summary_df = results['summary_df']
        print("\\nSummary:")
        for method in summary_df['method'].unique():
            method_data = summary_df[summary_df['method'] == method]
            successful = method_data[method_data['success_rate'] > 0]
            print(f"  {method.upper()} method: {len(successful)}/{len(method_data)} successful configurations")
        
    except Exception as e:
        print(f"\\nError in analysis: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
