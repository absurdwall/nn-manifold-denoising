#!/usr/bin/env python3
"""
Quick test of the geometric analysis on a single dataset.
"""

import sys
from pathlib import Path
import numpy as np
import json

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

def main():
    """Test on a single dataset."""
    print("Testing geometric analysis on dataset0...")
    
    # Load test data
    data_dir = Path(__file__).parent.parent / "data" / "data_250913_1749" / "dataset0"
    
    if not data_dir.exists():
        print(f"Dataset directory not found: {data_dir}")
        return False
    
    # Load files
    intrinsic_file = data_dir / "dataset0_intrinsic.npy"
    clean_file = data_dir / "dataset0_clean.npy"
    metadata_file = data_dir / "dataset0_metadata.json"
    
    if not intrinsic_file.exists() or not clean_file.exists():
        print("Required files not found")
        return False
    
    intrinsic_coords = np.load(intrinsic_file)
    embedded_coords = np.load(clean_file)
    
    print(f"Loaded data:")
    print(f"  Intrinsic shape: {intrinsic_coords.shape}")
    print(f"  Embedded shape: {embedded_coords.shape}")
    
    # Load metadata
    if metadata_file.exists():
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        print(f"  Metadata: {metadata}")
    
    # Test basic functionality
    try:
        from geometric_analysis import DimensionEstimator
        print("\n✓ Successfully imported DimensionEstimator")
        
        # Test dimension estimation
        estimator = DimensionEstimator(verbose=False)
        results = estimator.fit(embedded_coords[:500], methods=['PCA'])  # Small sample for speed
        print(f"✓ Dimension estimation result: {results}")
        
    except Exception as e:
        print(f"✗ Error in dimension estimation: {e}")
        return False
    
    try:
        from geometric_analysis import estimate_extrinsic_diameter
        diameter = estimate_extrinsic_diameter(embedded_coords[:200])  # Small sample
        print(f"✓ Extrinsic diameter: {diameter:.6f}")
    except Exception as e:
        print(f"✗ Error in diameter estimation: {e}")
    
    print("\n✓ Basic functionality test passed!")
    print("Ready to run full analysis.")
    return True

if __name__ == "__main__":
    main()
