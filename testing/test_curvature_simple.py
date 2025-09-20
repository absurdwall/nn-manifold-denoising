#!/usr/bin/env python3
"""
Simple test script to debug curvature implementation
"""

import numpy as np
import sys
from pathlib import Path

# Add src to path
script_dir = Path(__file__).parent
project_root = script_dir.parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

print("Starting curvature test...")

# Create simple synthetic test data
np.random.seed(42)
n_points = 30
d = 2  # intrinsic dimension
D = 10  # embedded dimension

# Create a simple manifold: intrinsic coordinates mapped to embedded space
intrinsic = np.random.uniform(-1, 1, (n_points, d))
# Simple embedding: first d dimensions are intrinsic, rest are small noise
embedded = np.zeros((n_points, D))
embedded[:, :d] = intrinsic
embedded[:, d:] = 0.01 * np.random.randn(n_points, D-d)

print(f"Test data created: {n_points} points, {d}D -> {D}D")

try:
    from geometric_analysis.point_curvature import estimate_mean_curvature_at_point
    from geometric_analysis.curvature_estimation import estimate_mean_curvature
    print("✓ Imports successful")
    
    # Test single point with new implementation
    test_idx = 10
    k_neighbors = 8
    
    # Find neighbors
    from sklearn.neighbors import NearestNeighbors
    nbrs = NearestNeighbors(n_neighbors=k_neighbors+1).fit(embedded)
    distances, indices = nbrs.kneighbors(embedded[test_idx:test_idx+1])
    neighbor_indices = indices[0][1:]  # Remove self
    
    target_embedded = embedded[test_idx]
    target_intrinsic = intrinsic[test_idx]
    neighbor_embedded = embedded[neighbor_indices]
    neighbor_intrinsic = intrinsic[neighbor_indices]
    
    print(f"\\nTesting point {test_idx} with {k_neighbors} neighbors")
    print(f"Target intrinsic: {target_intrinsic}")
    print(f"Target embedded: {target_embedded[:3]}...")
    
    # New implementation
    curvature_new, info_new = estimate_mean_curvature_at_point(
        point=target_embedded,
        neighbors=neighbor_embedded,
        intrinsic_point=target_intrinsic,
        intrinsic_neighbors=neighbor_intrinsic
    )
    
    print(f"\\nNew implementation:")
    print(f"  Curvature: {curvature_new:.6f}")
    print(f"  Info: {info_new}")
    
    # Test old implementation for comparison 
    old_curvatures = estimate_mean_curvature(
        intrinsic_coords=intrinsic,
        embedded_coords=embedded,
        k_neighbors=k_neighbors,
        max_points=1,  # Just one point
        random_state=42
    )
    
    valid_old = old_curvatures[np.isfinite(old_curvatures)]
    print(f"\\nOld implementation:")
    print(f"  Valid curvatures: {len(valid_old)}")
    if len(valid_old) > 0:
        print(f"  Curvature: {valid_old[0]:.6f}")
        
        if curvature_new > 0 and valid_old[0] > 0:
            ratio = curvature_new / valid_old[0]
            print(f"\\nRatio (new/old): {ratio:.2f}")
    
    print("\\n✓ Test completed successfully")

except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()
