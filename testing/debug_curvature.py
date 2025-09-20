#!/usr/bin/env python3
"""
Debug script to compare old vs new curvature implementations
"""

import numpy as np
import sys
from pathlib import Path

# Add src to path
script_dir = Path(__file__).parent
project_root = script_dir.parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

from geometric_analysis import (
    estimate_mean_curvature,
    estimate_mean_curvature_at_point
)

def debug_curvature_comparison():
    """Compare old and new curvature implementations."""
    
    # Load test dataset
    data_path = Path("data/data_250914_0100/dataset0")
    embedded_coords = np.load(data_path / "dataset0_clean.npy")
    intrinsic_coords = np.load(data_path / "dataset0_intrinsic.npy")
    
    print("Dataset shapes:")
    print(f"  Embedded: {embedded_coords.shape}")
    print(f"  Intrinsic: {intrinsic_coords.shape}")
    
    # Take a small subset for testing
    n_test = 100
    embedded_subset = embedded_coords[:n_test]
    intrinsic_subset = intrinsic_coords[:n_test]
    
    print(f"\\nTesting with {n_test} points...")
    
    # Test 1: Old implementation (whole dataset approach)
    print("\\n=== OLD IMPLEMENTATION ===")
    try:
        old_curvatures = estimate_mean_curvature(
            intrinsic_coords=intrinsic_subset,
            embedded_coords=embedded_subset,
            k_neighbors=20,
            max_points=50,  # Only process 50 points
            random_state=42
        )
        
        valid_old = old_curvatures[np.isfinite(old_curvatures)]
        print(f"Old method - Valid curvatures: {len(valid_old)}/{len(old_curvatures)}")
        if len(valid_old) > 0:
            print(f"  Mean: {np.mean(valid_old):.4f}")
            print(f"  Std: {np.std(valid_old):.4f}")
            print(f"  Range: [{np.min(valid_old):.4f}, {np.max(valid_old):.4f}]")
        
    except Exception as e:
        print(f"Old method failed: {e}")
        old_curvatures = None
    
    # Test 2: New implementation (point-wise approach)
    print("\\n=== NEW IMPLEMENTATION ===")
    
    # Manually test a few points with the new approach
    from sklearn.neighbors import NearestNeighbors
    
    k_neighbors = 20
    nbrs = NearestNeighbors(n_neighbors=k_neighbors + 1).fit(embedded_subset)
    
    new_curvatures = []
    test_indices = [10, 20, 30, 40, 50]  # Test a few points
    
    for i, test_idx in enumerate(test_indices):
        print(f"\\nTesting point {i+1}/{len(test_indices)} (index {test_idx})...")
        
        # Get neighbors
        distances, indices = nbrs.kneighbors(embedded_subset[test_idx:test_idx+1])
        neighbor_indices = indices[0][1:]  # Exclude the point itself
        
        target_embedded = embedded_subset[test_idx]
        target_intrinsic = intrinsic_subset[test_idx]
        neighbor_embedded = embedded_subset[neighbor_indices]
        neighbor_intrinsic = intrinsic_subset[neighbor_indices]
        
        print(f"  Target point - Embedded: {target_embedded[:3]}, Intrinsic: {target_intrinsic}")
        print(f"  Neighbors: {len(neighbor_embedded)} points")
        
        # Test new method
        try:
            curvature, info = estimate_mean_curvature_at_point(
                point=target_embedded,
                neighbors=neighbor_embedded,
                intrinsic_point=target_intrinsic,
                intrinsic_neighbors=neighbor_intrinsic
            )
            
            print(f"  New method result: {curvature:.4f}")
            print(f"  Info: {info}")
            
            if np.isfinite(curvature):
                new_curvatures.append(curvature)
                
        except Exception as e:
            print(f"  New method failed: {e}")
    
    print(f"\\n=== COMPARISON ===")
    print(f"New method - Valid curvatures: {len(new_curvatures)}")
    if len(new_curvatures) > 0:
        print(f"  Mean: {np.mean(new_curvatures):.4f}")
        print(f"  Std: {np.std(new_curvatures):.4f}")
        print(f"  Range: [{np.min(new_curvatures):.4f}, {np.max(new_curvatures):.4f}]")
    
    # Check data scaling
    print(f"\\n=== DATA SCALING ===")
    print(f"Embedded coords:")
    print(f"  Mean magnitude: {np.mean(np.linalg.norm(embedded_subset, axis=1)):.4f}")
    print(f"  Std magnitude: {np.std(np.linalg.norm(embedded_subset, axis=1)):.4f}")
    print(f"Intrinsic coords:")
    print(f"  Mean magnitude: {np.mean(np.linalg.norm(intrinsic_subset, axis=1)):.4f}")
    print(f"  Std magnitude: {np.std(np.linalg.norm(intrinsic_subset, axis=1)):.4f}")


if __name__ == "__main__":
    debug_curvature_comparison()
