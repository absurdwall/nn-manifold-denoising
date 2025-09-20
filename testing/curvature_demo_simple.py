#!/usr/bin/env python3
"""
Simple demonstration of curvature vs kernel smoothness for radius analysis.
"""

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add src to path
script_dir = Path(__file__).parent
project_root = script_dir.parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

from geometric_analysis import compute_manifold_curvature


def main():
    """Run a simple demonstration with minimal computation."""
    
    # Configuration
    data_dir = Path("data/data_250914_0100")
    output_dir = Path("figs_curvature_split/radius_demo")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Simple Curvature vs Kernel Smoothness Demonstration")
    print("=" * 55)
    
    # Load just a few datasets for quick analysis
    dataset_info = [
        (1, 0.1),   # Dataset 1: κ=0.1
        (2, 1.0),   # Dataset 2: κ=1.0
        (3, 10.0),  # Dataset 3: κ=10.0
        (4, 100.0)  # Dataset 4: κ=100.0
    ]
    
    # Test with a couple of radius values
    radius_values = [0.2, 0.25]
    
    # Very small sample for speed
    sample_size = 50
    
    results = {}
    
    for radius in radius_values:
        print(f"\nRadius = {radius}")
        results[radius] = {}
        
        for dataset_idx, kappa in dataset_info:
            print(f"  Loading dataset {dataset_idx} (κ={kappa})...", end=" ")
            
            try:
                # Load dataset
                dataset_dir = data_dir / f"dataset{dataset_idx}"
                embedded_file = dataset_dir / f"dataset{dataset_idx}_clean.npy"
                intrinsic_file = dataset_dir / f"dataset{dataset_idx}_intrinsic.npy"
                
                embedded_coords = np.load(embedded_file)
                intrinsic_coords = np.load(intrinsic_file)
                
                # Quick computation with small sample
                result = compute_manifold_curvature(
                    embedded_coords=embedded_coords,
                    intrinsic_coords=intrinsic_coords,
                    neighbor_strategy='radius',
                    radius=radius,
                    curvature_method='pca',
                    n_sample_points=sample_size,
                    random_state=42
                )
                
                # Extract statistics
                stats = result['statistics']
                mean_curvature = stats['mean']
                success_rate = stats['success_rate'] * 100
                
                results[radius][dataset_idx] = {
                    'kappa': kappa,
                    'mean_curvature': mean_curvature,
                    'success_rate': success_rate
                }
                
                print(f"curvature={mean_curvature:.4f}, success={success_rate:.1f}%")
                
            except Exception as e:
                print(f"Error: {e}")
    
    # Create plot
    print("\nCreating plot...")
    
    plt.figure(figsize=(10, 6))
    
    colors = ['blue', 'red']
    markers = ['o', 's']
    
    for i, radius in enumerate(radius_values):
        if radius in results and len(results[radius]) > 0:
            
            kappa_values = []
            curvature_values = []
            
            for dataset_idx in sorted(results[radius].keys()):
                data = results[radius][dataset_idx]
                if not np.isnan(data['mean_curvature']) and data['success_rate'] > 50:
                    kappa_values.append(data['kappa'])
                    curvature_values.append(data['mean_curvature'])
            
            if len(kappa_values) > 0:
                kappa_values = np.array(kappa_values)
                curvature_values = np.array(curvature_values)
                
                plt.loglog(kappa_values, curvature_values, 
                          marker=markers[i], color=colors[i], linewidth=2, 
                          markersize=8, label=f'r={radius}')
    
    plt.grid(True, alpha=0.3)
    plt.xlabel('Kernel Smoothness (κ)')
    plt.ylabel('Mean Curvature')
    plt.title('Curvature vs Kernel Smoothness - Radius Analysis')
    plt.legend()
    
    # Save plot
    plot_file = output_dir / "curvature_vs_kernel_smoothness_demo.png"
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    print(f"Saved plot: {plot_file}")
    
    # Save results
    results_file = output_dir / "demo_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved results: {results_file}")
    
    # Print summary
    print("\nResults Summary:")
    for radius in radius_values:
        if radius in results:
            print(f"\nRadius {radius}:")
            for dataset_idx in sorted(results[radius].keys()):
                data = results[radius][dataset_idx]
                print(f"  Dataset {dataset_idx}: κ={data['kappa']:5.1f}, "
                      f"curvature={data['mean_curvature']:.4f}, "
                      f"success={data['success_rate']:.1f}%")
    
    plt.show()


if __name__ == "__main__":
    main()
