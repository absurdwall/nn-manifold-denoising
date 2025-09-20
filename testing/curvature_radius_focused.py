#!/usr/bin/env python3
"""
Curvature vs Kernel Smoothness with Fixed Radius - Focused Analysis

This script creates the specific experiment you requested:
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
        'kernel_smoothness': kernel_smoothness,
        'dataset_idx': dataset_idx
    }


def main():
    """Run focused curvature vs kernel smoothness analysis."""
    
    # Configuration
    data_dir = "/home/tim/python_projects/nn_manifold_denoising/data/data_250914_0100"
    output_dir = "/home/tim/python_projects/nn_manifold_denoising/results/curvature_radius_focused"
    
    # Analysis parameters
    dataset_indices = [0, 1, 2, 3, 4]  # First 5 datasets
    radius_values = [0.15, 0.2, 0.25]  # Focus on radii that work well
    SAMPLE_SIZE = 200  # Number of points to sample for analysis
    random_state = 42
    
    print("Curvature vs Kernel Smoothness - Focused Radius Analysis")
    print("=" * 60)
    print(f"Analyzing datasets {dataset_indices}")
    print(f"Using radius values: {radius_values}")
    print(f"Sample points per dataset: {SAMPLE_SIZE}")
    
    # Load datasets
    print("\\nLoading datasets...")
    datasets = []
    kernel_smoothness_values = []
    
    for idx in dataset_indices:
        try:
            dataset = load_dataset_with_metadata(Path(data_dir), idx)
            datasets.append(dataset)
            kernel_smoothness_values.append(dataset['kernel_smoothness'])
            print(f"  Dataset {idx}: kernel_smoothness = {dataset['kernel_smoothness']}")
        except Exception as e:
            print(f"  Failed to load dataset {idx}: {e}")
            continue
    
    if len(datasets) == 0:
        raise ValueError("No datasets were successfully loaded")
    
    # Store results for plotting
    results_data = []
    
    # Analyze each radius
    for radius in radius_values:
        print(f"\\n--- Analyzing radius = {radius} ---")
        
        radius_curvatures = []
        radius_kernel_values = []
        
        for i, dataset in enumerate(datasets):
            dataset_idx = dataset['dataset_idx']
            kernel_smooth = dataset['kernel_smoothness']
            
            print(f"  Dataset {dataset_idx} (κ={kernel_smooth})...", end=" ")
            
            try:
                # Use PCA method since it's more robust
                result = compute_manifold_curvature(
                    embedded_coords=dataset["embedded_coords"],
                    intrinsic_coords=dataset["intrinsic_coords"],
                    curvature_method="pca",
                    neighbor_strategy="radius",
                    radius=radius,
                    n_sample_points=SAMPLE_SIZE,
                    random_state=random_state
                )
                
                stats = result["statistics"]
                
                # Count neighbors
                neighbor_counts = []
                for diag in result["diagnostics"]:
                    if "num_neighbors" in diag and diag["num_neighbors"] > 0:
                        neighbor_counts.append(diag["num_neighbors"])
                
                avg_neighbors = np.mean(neighbor_counts) if neighbor_counts else 0
                
                if stats['success_rate'] > 0.5:  # At least 50% success
                    radius_curvatures.append(stats['mean'])
                    radius_kernel_values.append(kernel_smooth)
                    
                    print(f"✓ curvature={stats['mean']:.4f}, "
                          f"success={stats['success_rate']:.1%}, "
                          f"neighbors={avg_neighbors:.1f}")
                    
                    # Store for detailed analysis
                    results_data.append({
                        'radius': radius,
                        'kernel_smoothness': kernel_smooth,
                        'curvature_mean': stats['mean'],
                        'curvature_std': stats['std'],
                        'success_rate': stats['success_rate'],
                        'avg_neighbors': avg_neighbors,
                        'dataset_idx': dataset_idx
                    })
                else:
                    print(f"✗ Low success rate: {stats['success_rate']:.1%}")
                    
            except Exception as e:
                print(f"✗ Error: {e}")
    
    # Create the requested plot: for each radius, curvature vs kernel smoothness
    if results_data:
        results_df = pd.DataFrame(results_data)
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save results
        results_df.to_csv(output_path / "curvature_vs_kernel_smoothness_radius_focused.csv", index=False)
        
        # Create the plot
        plt.figure(figsize=(12, 8))
        
        # Plot each radius as a separate line
        colors = ['blue', 'red', 'green', 'orange', 'purple']
        
        for i, radius in enumerate(radius_values):
            radius_data = results_df[results_df['radius'] == radius]
            
            if len(radius_data) > 0:
                # Sort by kernel smoothness for proper line plotting
                radius_data = radius_data.sort_values('kernel_smoothness')
                
                plt.loglog(radius_data['kernel_smoothness'], 
                          radius_data['curvature_mean'],
                          'o-', color=colors[i % len(colors)], 
                          label=f'r = {radius}',
                          markersize=8, linewidth=2)
        
        plt.xlabel('Kernel Smoothness', fontsize=12)
        plt.ylabel('Mean Curvature (PCA Method)', fontsize=12)
        plt.title('Curvature vs Kernel Smoothness for Different Radius Values', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=11)
        
        # Save plot
        plot_file = output_path / "curvature_vs_kernel_smoothness_radius_focused.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"\\nPlot saved to: {plot_file}")
        
        # Also create individual subplots for each radius
        fig, axes = plt.subplots(1, len(radius_values), figsize=(15, 5), sharey=True)
        if len(radius_values) == 1:
            axes = [axes]
            
        for i, radius in enumerate(radius_values):
            radius_data = results_df[results_df['radius'] == radius]
            
            if len(radius_data) > 0:
                radius_data = radius_data.sort_values('kernel_smoothness')
                
                axes[i].loglog(radius_data['kernel_smoothness'], 
                              radius_data['curvature_mean'],
                              'o-', color=colors[i % len(colors)], 
                              markersize=8, linewidth=2)
                
                axes[i].set_xlabel('Kernel Smoothness')
                axes[i].set_title(f'Radius = {radius}')
                axes[i].grid(True, alpha=0.3)
        
        axes[0].set_ylabel('Mean Curvature')
        plt.suptitle('Curvature vs Kernel Smoothness - Individual Radius Plots', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        subplot_file = output_path / "curvature_vs_kernel_smoothness_radius_subplots.png"
        plt.savefig(subplot_file, dpi=300, bbox_inches='tight')
        print(f"Subplots saved to: {subplot_file}")
        
        plt.show()
        
        # Print summary
        print(f"\\nAnalysis Summary:")
        print(f"Total successful configurations: {len(results_data)}")
        for radius in radius_values:
            radius_data = results_df[results_df['radius'] == radius]
            print(f"  Radius {radius}: {len(radius_data)} datasets successful")
        
        print(f"\\nResults saved to: {output_path}")
        
    else:
        print("\\nNo successful configurations found!")


if __name__ == "__main__":
    main()
