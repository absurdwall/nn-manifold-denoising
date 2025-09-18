#!/usr/bin/env python3
"""
Quick test of the comprehensive analysis on a single dataset.
"""

import sys
from pathlib import Path
import numpy as np
import json

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

def main():
    """Test comprehensive analysis on a single dataset."""
    print("Testing comprehensive analysis on dataset0...")
    
    # Test data loading
    data_dir = Path(__file__).parent.parent / "data" / "data_250913_1749" / "dataset0"
    
    if not data_dir.exists():
        print(f"Dataset directory not found: {data_dir}")
        return False
    
    # Check what files we have
    files = list(data_dir.glob("*.npy"))
    print(f"Available .npy files: {[f.name for f in files]}")
    
    # Test loading all data types
    dataset_name = "dataset0"
    
    # Check for the three data types
    gp_file = data_dir / f"{dataset_name}_raw.npy"
    clean_file = data_dir / f"{dataset_name}_clean.npy"
    noisy_file = data_dir / f"{dataset_name}_noisy.npy"
    intrinsic_file = data_dir / f"{dataset_name}_intrinsic.npy"
    
    print(f"\nData file availability:")
    print(f"  GP data (raw): {'✓' if gp_file.exists() else '✗'}")
    print(f"  Clean data: {'✓' if clean_file.exists() else '✗'}")
    print(f"  Noisy data: {'✓' if noisy_file.exists() else '✗'}")
    print(f"  Intrinsic coords: {'✓' if intrinsic_file.exists() else '✗'}")
    
    if all(f.exists() for f in [gp_file, clean_file, noisy_file, intrinsic_file]):
        # Load and compare data shapes
        gp_data = np.load(gp_file)
        clean_data = np.load(clean_file)
        noisy_data = np.load(noisy_file)
        intrinsic_data = np.load(intrinsic_file)
        
        print(f"\nData shapes:")
        print(f"  GP data: {gp_data.shape}")
        print(f"  Clean data: {clean_data.shape}")
        print(f"  Noisy data: {noisy_data.shape}")
        print(f"  Intrinsic: {intrinsic_data.shape}")
        
        # Compare data differences
        print(f"\nData relationships:")
        print(f"  Clean vs GP (mean diff): {np.mean(np.abs(clean_data - gp_data)):.6f}")
        print(f"  Noisy vs Clean (mean diff): {np.mean(np.abs(noisy_data - clean_data)):.6f}")
        print(f"  Noisy vs GP (mean diff): {np.mean(np.abs(noisy_data - gp_data)):.6f}")
        
        # Test basic geometric analysis
        try:
            from geometric_analysis import DimensionEstimator, estimate_extrinsic_diameter
            
            print(f"\n✓ Successfully imported geometric analysis modules")
            
            # Quick dimension test on each data type
            estimator = DimensionEstimator(verbose=False)
            
            sample_size = 500  # Small sample for speed
            
            for name, data in [("GP", gp_data), ("Clean", clean_data), ("Noisy", noisy_data)]:
                sample = data[:sample_size] if len(data) > sample_size else data
                results = estimator.fit(sample, methods=['PCA'])
                diameter = estimate_extrinsic_diameter(sample[:200])
                
                print(f"  {name} data - PCA dim: {results.get('PCA', 'N/A')}, "
                      f"diameter: {diameter:.6f}")
            
            print(f"\n✓ Basic functionality test passed!")
            return True
            
        except Exception as e:
            print(f"✗ Error in geometric analysis: {e}")
            return False
    else:
        print("✗ Some required data files are missing")
        return False

if __name__ == "__main__":
    main()
