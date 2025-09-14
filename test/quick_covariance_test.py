#!/usr/bin/env python3
"""
Quick test runner for the covariance diagnostic tool.
TEMPORARY FILE - CAN BE DELETED SAFELY
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from diagnose_covariance_warnings import diagnose_manifold_generation

def quick_test():
    """Run a quick test case that should show the warning."""
    print("🚀 Quick Test: Parameters that typically show warnings")
    print("This uses k=3, N=500, D=20 - similar to your original case")
    print()
    
    diagnose_manifold_generation(
        k=3, N=500, d=2, D=20, 
        kernel_smoothness=1.0, 
        random_seed=42
    )

if __name__ == "__main__":
    quick_test()
