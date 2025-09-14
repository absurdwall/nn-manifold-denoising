#!/usr/bin/env python3
"""
Quick timing test to identify bottlenecks in our manifold generation.
"""

import sys
import os
import time
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

def test_timing():
    print("=" * 60)
    print("MANIFOLD GENERATION TIMING TEST")
    print("=" * 60)
    
    try:
        from data_generation.manifold_generator import generate_manifold_embedding
        print("✓ Import successful")
        
        # Test small case
        print("\nTest 1: Small case (k=2, N=100, d=2, D=10)")
        start = time.time()
        embedded, intrinsic = generate_manifold_embedding(
            k=2, N=100, d=2, D=10, kernel_smoothness=1.0, 
            base_type='unit_ball', verbose=True
        )
        elapsed = time.time() - start
        print(f"✓ Small case completed in {elapsed:.2f}s")
        
        # Test medium case  
        print("\nTest 2: Medium case (k=5, N=200, d=2, D=20)")
        start = time.time()
        embedded, intrinsic = generate_manifold_embedding(
            k=5, N=200, d=2, D=20, kernel_smoothness=1.0,
            base_type='unit_ball', verbose=True
        )
        elapsed = time.time() - start
        print(f"✓ Medium case completed in {elapsed:.2f}s")
        
        # Test your original large case
        print("\nTest 3: Large case (k=10, N=1000, d=2, D=100) - PARTIAL")
        start = time.time()
        embedded, intrinsic = generate_manifold_embedding(
            k=10, N=1000, d=2, D=100, kernel_smoothness=1.0,
            base_type='unit_ball', verbose=True
        )
        elapsed = time.time() - start
        print(f"✓ Large case completed in {elapsed:.2f}s")
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_timing()
