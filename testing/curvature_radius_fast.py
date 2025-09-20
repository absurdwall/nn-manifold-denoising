#!/usr/bin/env python3
"""
Fast Curvature vs Kernel Smoothness Analysis

Optimized version that focuses on successful configurations only.
For each radius r, plot how curvature scales with kernel smoothness parameter.
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

from geometric_analysis import compute_manifold_curvature


def load_dataset_with_metadata(data_dir: Path, dataset_idx: int) -> Dict:
    """Load dataset with its metadata."""
    dataset_dir = data_dir / f"dataset{dataset_idx}"
    
    # Load files
    embedded_file = dataset_dir / f"dataset{dataset_idx}_clean.npy"
    intrinsic_file = dataset_dir / f"dataset{dataset_idx}_intrinsic.npy"
    metadata_file = dataset_dir / f"dataset{dataset_idx}_metadata.json"
    
    # Load data
    embedded_coords = np.load(embedded_file)
    intrinsic_coords = np.load(intrinsic_file)
    
    # Load metadata
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
    
    # Extract kernel_smoothness 
    kernel_smoothness = metadata.get('properties', {}).get('kernel_smoothness', 'unknown')
    
    return {
        'embedded_coords': embedded_coords,
        'intrinsic_coords': intrinsic_coords,
        'metadata': metadata,
        'kernel_smoothness': kernel_smoothness
    }


def main():
    """Run focused radius analysis with optimizations."""
    
    # Configuration
    data_dir = Path("data/data_250914_0100")
    output_dir = Path("figs_curvature_split/radius_focused")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Analysis parameters - only successful configurations
    radius_values = [0.2, 0.25]  # Skip 0.15 which has low success on dataset 0
    dataset_indices = [1, 2, 3, 4]  # Skip dataset 0 which has low success rates
    sample_size = 100  # Further reduced for speed
    
    print("Fast Curvature vs Kernel Smoothness Analysis")
    print("=" * 50)
    print(f"Analyzing datasets {dataset_indices}")
    print(f"Using radius values: {radius_values}")
    print(f"Sample points per dataset: {sample_size}")
    
    # Load datasets
    print("\nLoading datasets...")
    datasets = {}
    for dataset_idx in dataset_indices:
        datasets[dataset_idx] = load_dataset_with_metadata(data_dir, dataset_idx)
        kappa = datasets[dataset_idx]['kernel_smoothness']
        print(f"  Dataset {dataset_idx}: kernel_smoothness = {kappa}")
    
    # Results storage
    results = {}
    
    # Analyze each radius
    for radius in radius_values:
        print(f"\n--- Analyzing radius = {radius} ---")
        results[radius] = {}
        
        for dataset_idx in dataset_indices:
            dataset = datasets[dataset_idx]
            kappa = dataset['kernel_smoothness']
            
            print(f"  Dataset {dataset_idx} (κ={kappa})...", end=" ")
            
            try:
                # Compute curvature
                result = compute_manifold_curvature(
                    embedded_coords=dataset['embedded_coords'],
                    intrinsic_coords=dataset['intrinsic_coords'],
                    neighbor_strategy='radius',
                    radius=radius,
                    curvature_method='pca',
                    n_sample_points=sample_size,
                    random_state=42
                )
                
                # Store results
                curvature_values = result['curvature_values']
                valid_mask = ~np.isnan(curvature_values)
                
                if valid_mask.sum() > 0:
                    mean_curvature = np.mean(curvature_values[valid_mask])
                    success_rate = valid_mask.sum() / len(curvature_values) * 100
                    avg_neighbors = np.mean(result['neighbor_counts'])
                    
                    results[radius][dataset_idx] = {
                        'mean_curvature': mean_curvature,
                        'success_rate': success_rate,
                        'avg_neighbors': avg_neighbors,
                        'kernel_smoothness': kappa
                    }
                    
                    print(f"✓ curvature={mean_curvature:.4f}, success={success_rate:.1f}%, neighbors={avg_neighbors:.1f}")
                else:
                    print("✗ No valid curvature values")
                    
            except Exception as e:
                print(f"✗ Error: {e}")
    
    # Create plots
    print(f"\nCreating plots...")
    
    # Setup figure with subplots for each radius
    fig, axes = plt.subplots(1, len(radius_values), figsize=(5*len(radius_values), 4))
    if len(radius_values) == 1:
        axes = [axes]
    
    # Plot for each radius
    for i, radius in enumerate(radius_values):
        ax = axes[i]
        
        # Extract data for this radius
        if radius in results and len(results[radius]) > 0:
            radius_data = results[radius]
            
            kappa_values = []
            curvature_values = []
            
            for dataset_idx in sorted(radius_data.keys()):
                data = radius_data[dataset_idx]
                kappa_values.append(data['kernel_smoothness'])
                curvature_values.append(data['mean_curvature'])
            
            kappa_values = np.array(kappa_values)
            curvature_values = np.array(curvature_values)
            
            # Plot with loglog scale
            ax.loglog(kappa_values, curvature_values, 'o-', linewidth=2, markersize=8, label=f'r={radius}')
            ax.grid(True, alpha=0.3)
            ax.set_xlabel('Kernel Smoothness (κ)')
            ax.set_ylabel('Mean Curvature')
            ax.set_title(f'Radius = {radius}')
            
            # Add point labels
            for j, (kappa, curv) in enumerate(zip(kappa_values, curvature_values)):
                dataset_idx = sorted(radius_data.keys())[j]
                ax.annotate(f'D{dataset_idx}', (kappa, curv), xytext=(5, 5), 
                           textcoords='offset points', fontsize=8)
    
    plt.tight_layout()
    
    # Save plot
    plot_file = output_dir / "curvature_vs_kernel_smoothness_radius_fast.png"
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    print(f"Saved plot: {plot_file}")
    
    # Create combined plot - all radius on same axes
    plt.figure(figsize=(8, 6))
    colors = ['blue', 'red', 'green', 'orange']
    
    for i, radius in enumerate(radius_values):
        if radius in results and len(results[radius]) > 0:
            radius_data = results[radius]
            
            kappa_values = []
            curvature_values = []
            
            for dataset_idx in sorted(radius_data.keys()):
                data = radius_data[dataset_idx]
                kappa_values.append(data['kernel_smoothness'])
                curvature_values.append(data['mean_curvature'])
            
            kappa_values = np.array(kappa_values)
            curvature_values = np.array(curvature_values)
            
            # Plot with loglog scale
            plt.loglog(kappa_values, curvature_values, 'o-', 
                      linewidth=2, markersize=8, color=colors[i % len(colors)],
                      label=f'r={radius}')
    
    plt.grid(True, alpha=0.3)
    plt.xlabel('Kernel Smoothness (κ)')
    plt.ylabel('Mean Curvature')
    plt.title('Curvature vs Kernel Smoothness - All Radius Values')
    plt.legend()
    
    # Save combined plot
    combined_plot_file = output_dir / "curvature_vs_kernel_smoothness_all_radius.png"
    plt.savefig(combined_plot_file, dpi=150, bbox_inches='tight')
    print(f"Saved combined plot: {combined_plot_file}")
    
    # Save results to JSON
    results_file = output_dir / "curvature_radius_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved results: {results_file}")
    
    print("\nAnalysis complete!")
    plt.show()


if __name__ == "__main__":
    main()
