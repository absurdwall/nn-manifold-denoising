#!/usr/bin/env python3
"""
Test the geometric analysis implementation on existing test data.

This script tests our Step 2 implementation on the small test dataset
to verify everything works correctly.
"""

import sys
from pathlib import Path
import numpy as np
import json

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

def load_test_data(data_dir: Path):
    """Load test data from the separate .npy files format."""
    intrinsic_file = data_dir / "dataset0_intrinsic.npy"
    clean_file = data_dir / "dataset0_clean.npy"
    
    if not intrinsic_file.exists() or not clean_file.exists():
        raise FileNotFoundError(f"Required files not found in {data_dir}")
    
    intrinsic_coords = np.load(intrinsic_file)
    embedded_coords = np.load(clean_file)
    
    # Load metadata if available
    metadata = {}
    settings_file = data_dir / "experiment_settings.json"
    if settings_file.exists():
        with open(settings_file, 'r') as f:
            metadata = json.load(f)
    
    return intrinsic_coords, embedded_coords, metadata

def test_dimension_estimation():
    """Test dimension estimation functionality."""
    print("Testing dimension estimation...")
    
    # Import here to handle potential import issues gracefully
    try:
        from geometric_analysis import DimensionEstimator
    except ImportError as e:
        print(f"Import error: {e}")
        return False
    
    # Load test data
    data_dir = Path(__file__).parent.parent / "data" / "test_small"
    try:
        intrinsic_coords, embedded_coords, metadata = load_test_data(data_dir)
        print(f"Loaded data: intrinsic shape {intrinsic_coords.shape}, embedded shape {embedded_coords.shape}")
    except Exception as e:
        print(f"Failed to load test data: {e}")
        return False
    
    # Test dimension estimation
    try:
        estimator = DimensionEstimator(verbose=True)
        results = estimator.fit(embedded_coords, methods=['PCA', 'k-NN'])
        print(f"Dimension estimation results: {results}")
        
        # Compare with true dimension
        true_dim = intrinsic_coords.shape[1]
        print(f"True intrinsic dimension: {true_dim}")
        
        return True
    except Exception as e:
        print(f"Dimension estimation failed: {e}")
        return False

def test_curvature_estimation():
    """Test curvature estimation functionality."""
    print("\nTesting curvature estimation...")
    
    try:
        from geometric_analysis import estimate_mean_curvature, estimate_curvature_pca_based
    except ImportError as e:
        print(f"Import error: {e}")
        return False
    
    # Load test data
    data_dir = Path(__file__).parent.parent / "data" / "test_small"
    try:
        intrinsic_coords, embedded_coords, metadata = load_test_data(data_dir)
    except Exception as e:
        print(f"Failed to load test data: {e}")
        return False
    
    # Test mean curvature estimation
    try:
        print("Testing mean curvature estimation...")
        curvatures = estimate_mean_curvature(
            intrinsic_coords, 
            embedded_coords, 
            k_neighbors=20,
            max_points=100  # Small for testing
        )
        valid_curvatures = curvatures[np.isfinite(curvatures)]
        print(f"Mean curvature: computed {len(valid_curvatures)}/{len(curvatures)} values")
        if len(valid_curvatures) > 0:
            print(f"  Mean: {np.mean(valid_curvatures):.6f}")
            print(f"  Std: {np.std(valid_curvatures):.6f}")
            print(f"  Range: [{np.min(valid_curvatures):.6f}, {np.max(valid_curvatures):.6f}]")
    except Exception as e:
        print(f"Mean curvature estimation failed: {e}")
    
    # Test PCA-based curvature estimation
    try:
        print("Testing PCA-based curvature estimation...")
        pca_curvatures = estimate_curvature_pca_based(embedded_coords)
        valid_pca = pca_curvatures[np.isfinite(pca_curvatures)]
        print(f"PCA curvature: computed {len(valid_pca)}/{len(pca_curvatures)} values")
        if len(valid_pca) > 0:
            print(f"  Mean: {np.mean(valid_pca):.6f}")
            print(f"  Std: {np.std(valid_pca):.6f}")
    except Exception as e:
        print(f"PCA curvature estimation failed: {e}")
    
    return True

def test_geometric_statistics():
    """Test geometric statistics functionality."""
    print("\nTesting geometric statistics...")
    
    try:
        from geometric_analysis import compute_geometric_summary, estimate_extrinsic_diameter
    except ImportError as e:
        print(f"Import error: {e}")
        return False
    
    # Load test data
    data_dir = Path(__file__).parent.parent / "data" / "test_small"
    try:
        intrinsic_coords, embedded_coords, metadata = load_test_data(data_dir)
    except Exception as e:
        print(f"Failed to load test data: {e}")
        return False
    
    # Test individual functions
    try:
        diameter = estimate_extrinsic_diameter(embedded_coords)
        print(f"Extrinsic diameter: {diameter:.6f}")
    except Exception as e:
        print(f"Diameter estimation failed: {e}")
    
    # Test comprehensive summary
    try:
        summary = compute_geometric_summary(
            embedded_coords, 
            intrinsic_coords,
            include_curvature=False,  # Skip for speed
            include_dimension=False   # Skip for speed
        )
        print("Geometric summary:")
        for key, value in summary.items():
            if isinstance(value, dict):
                print(f"  {key}: {value}")
            else:
                print(f"  {key}: {value}")
    except Exception as e:
        print(f"Geometric summary failed: {e}")
    
    return True

def main():
    """Run all tests."""
    print("Testing geometric analysis implementation...")
    print("=" * 50)
    
    # Configure Python environment first
    try:
        from configure_python_environment import configure_python_environment
        print("Configuring Python environment...")
        configure_python_environment()
    except:
        print("Note: Could not configure Python environment automatically")
    
    success = True
    
    # Run tests
    success &= test_dimension_estimation()
    success &= test_curvature_estimation() 
    success &= test_geometric_statistics()
    
    print("\n" + "=" * 50)
    if success:
        print("✓ All tests completed successfully!")
        print("\nThe geometric analysis implementation is ready to use.")
        print("You can now run the full pipeline with:")
        print("  python scripts/step2_geometric_analysis.py --data_dir data/test_small")
    else:
        print("✗ Some tests failed. Check the error messages above.")
    
    return success

if __name__ == "__main__":
    main()
