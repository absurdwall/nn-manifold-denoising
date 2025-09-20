#!/usr/bin/env python3
"""
Demo: Modular Curvature Analysis Framework

This script demonstrates the new modular approach for curvature estimation
with controlled neighbor selection, addressing the issues from the previous analysis.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add src to path
script_dir = Path(__file__).parent
project_root = script_dir.parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

from geometric_analysis import (
    compute_manifold_curvature,
    estimate_mean_curvature_at_point,
    estimate_pca_curvature_at_point
)


def generate_test_manifold(n_points=500, noise_level=0.1):
    """Generate a simple 2D manifold embedded in 3D for testing."""
    # Generate intrinsic coordinates on a 2D grid
    u = np.random.uniform(-1, 1, n_points)
    v = np.random.uniform(-1, 1, n_points)
    intrinsic_coords = np.column_stack([u, v])
    
    # Embed in 3D as a curved surface (paraboloid)
    x = u
    y = v
    z = 0.5 * (u**2 + v**2)  # Paraboloid shape
    
    # Add noise
    noise = np.random.normal(0, noise_level, (n_points, 3))
    embedded_coords = np.column_stack([x, y, z]) + noise
    
    return intrinsic_coords, embedded_coords


def demo_single_point_estimation():
    """Demonstrate point-wise curvature estimation."""
    print("=== Demo: Single Point Curvature Estimation ===")
    
    # Generate test data
    intrinsic_coords, embedded_coords = generate_test_manifold(n_points=200)
    
    # Pick a test point
    test_idx = 100
    test_point_intrinsic = intrinsic_coords[test_idx]
    test_point_embedded = embedded_coords[test_idx]
    
    # Get k nearest neighbors
    from sklearn.neighbors import NearestNeighbors
    nbrs = NearestNeighbors(n_neighbors=21).fit(embedded_coords)
    _, indices = nbrs.kneighbors(test_point_embedded.reshape(1, -1))
    neighbor_indices = indices[0][1:]  # Exclude the point itself
    
    neighbor_points_embedded = embedded_coords[neighbor_indices]
    neighbor_points_intrinsic = intrinsic_coords[neighbor_indices]
    
    print(f"Test point intrinsic: {test_point_intrinsic}")
    print(f"Test point embedded: {test_point_embedded}")
    print(f"Number of neighbors: {len(neighbor_points_embedded)}")
    
    # Estimate curvature using different methods
    
    # Method 1: Mean curvature (with intrinsic coordinates)
    mean_curv, mean_info = estimate_mean_curvature_at_point(
        test_point_embedded, neighbor_points_embedded,
        test_point_intrinsic, neighbor_points_intrinsic
    )
    print(f"\\nMean curvature: {mean_curv:.4f}")
    print(f"Mean curvature info: {mean_info}")
    
    # Method 2: PCA-based curvature (no intrinsic needed)
    pca_curv, pca_info = estimate_pca_curvature_at_point(
        test_point_embedded, neighbor_points_embedded
    )
    print(f"\\nPCA curvature: {pca_curv:.4f}")
    print(f"PCA curvature info: {pca_info}")


def demo_manifold_analysis():
    """Demonstrate full manifold curvature analysis."""
    print("\\n\\n=== Demo: Full Manifold Analysis ===")
    
    # Generate test data
    intrinsic_coords, embedded_coords = generate_test_manifold(n_points=1000)
    
    print(f"Dataset shape - Intrinsic: {intrinsic_coords.shape}, Embedded: {embedded_coords.shape}")
    
    # Analysis 1: k-nearest neighbors with different k values
    print("\\n--- K-Nearest Neighbors Analysis ---")
    for k in [10, 20, 30]:
        result = compute_manifold_curvature(
            embedded_coords=embedded_coords,
            intrinsic_coords=intrinsic_coords,
            curvature_method="mean",
            neighbor_strategy="k_nearest",
            k_neighbors=k,
            n_sample_points=200,
            random_state=42
        )
        
        stats = result["statistics"]
        print(f"k={k:2d}: Mean curvature = {stats['mean']:.4f} ± {stats['std']:.4f}, "
              f"Success rate = {stats['success_rate']:.1%}")
    
    # Analysis 2: Fixed radius neighbors
    print("\\n--- Fixed Radius Analysis ---")
    for radius in [0.2, 0.3, 0.4]:
        result = compute_manifold_curvature(
            embedded_coords=embedded_coords,
            intrinsic_coords=intrinsic_coords,
            curvature_method="mean",
            neighbor_strategy="radius",
            radius=radius,
            n_sample_points=200,
            random_state=42
        )
        
        stats = result["statistics"]
        print(f"r={radius:.1f}: Mean curvature = {stats['mean']:.4f} ± {stats['std']:.4f}, "
              f"Success rate = {stats['success_rate']:.1%}")
    
    # Analysis 3: Compare methods
    print("\\n--- Method Comparison ---")
    for method in ["mean", "pca"]:
        result = compute_manifold_curvature(
            embedded_coords=embedded_coords,
            intrinsic_coords=intrinsic_coords,
            curvature_method=method,
            neighbor_strategy="k_nearest",
            k_neighbors=20,
            n_sample_points=200,
            random_state=42
        )
        
        stats = result["statistics"]
        print(f"{method.upper():4s}: Mean curvature = {stats['mean']:.4f} ± {stats['std']:.4f}, "
              f"Success rate = {stats['success_rate']:.1%}")


def demo_neighbor_consistency():
    """Demonstrate the importance of consistent neighbor selection."""
    print("\\n\\n=== Demo: Neighbor Selection Consistency ===")
    
    # Generate test data with varying density
    intrinsic_coords, embedded_coords = generate_test_manifold(n_points=1000)
    
    print("Comparing curvature estimation with different neighbor strategies...")
    print("This addresses the issue from the previous analysis where varying")
    print("numbers of neighbors made results inconsistent.")
    
    # Sample a subset for analysis
    n_sample = 100
    
    # Strategy 1: K-nearest (consistent number of neighbors)
    result_k = compute_manifold_curvature(
        embedded_coords=embedded_coords,
        intrinsic_coords=intrinsic_coords,
        curvature_method="pca",
        neighbor_strategy="k_nearest",
        k_neighbors=15,
        n_sample_points=n_sample,
        random_state=42
    )
    
    # Strategy 2: Fixed radius (varying number of neighbors)
    result_r = compute_manifold_curvature(
        embedded_coords=embedded_coords,
        intrinsic_coords=intrinsic_coords,
        curvature_method="pca",
        neighbor_strategy="radius",
        radius=0.3,
        n_sample_points=n_sample,
        random_state=42
    )
    
    print(f"\\nK-nearest (k=15):")
    print(f"  Mean curvature: {result_k['statistics']['mean']:.4f}")
    print(f"  Std curvature: {result_k['statistics']['std']:.4f}")
    print(f"  Success rate: {result_k['statistics']['success_rate']:.1%}")
    
    print(f"\\nFixed radius (r=0.3):")
    print(f"  Mean curvature: {result_r['statistics']['mean']:.4f}")
    print(f"  Std curvature: {result_r['statistics']['std']:.4f}")
    print(f"  Success rate: {result_r['statistics']['success_rate']:.1%}")
    
    # Analyze neighbor count distribution for radius method
    neighbor_counts = []
    for i, diag in enumerate(result_r["diagnostics"]):
        if "num_neighbors" in diag:
            neighbor_counts.append(diag["num_neighbors"])
    
    if neighbor_counts:
        print(f"\\nRadius method neighbor count distribution:")
        print(f"  Min neighbors: {min(neighbor_counts)}")
        print(f"  Max neighbors: {max(neighbor_counts)}")
        print(f"  Mean neighbors: {np.mean(neighbor_counts):.1f}")
        print(f"  Std neighbors: {np.std(neighbor_counts):.1f}")
        print("\\nThis variation in neighbor count can affect curvature estimation consistency!")


def main():
    """Run all demonstrations."""
    print("Modular Curvature Analysis Framework Demo")
    print("=" * 50)
    print()
    print("This demo showcases the new modular approach that addresses")
    print("the issues identified in your previous analysis:")
    print("1. Controlled neighbor selection (k-nearest vs radius)")
    print("2. Consistent number of neighbors across points")
    print("3. Modular design separating point-wise vs manifold-level analysis")
    print("4. Clear separation of curvature estimation methods")
    
    try:
        demo_single_point_estimation()
        demo_manifold_analysis()
        demo_neighbor_consistency()
        
        print("\\n\\n" + "=" * 50)
        print("Demo completed successfully!")
        print("\\nKey improvements of the modular framework:")
        print("- Point-wise functions: estimate curvature for single points")
        print("- Manifold functions: handle sampling and neighbor selection")
        print("- Consistent neighbor counts improve result reliability")
        print("- Clean separation allows easy testing of different approaches")
        
    except Exception as e:
        print(f"\\nError in demo: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
