#!/usr/bin/env python3
"""
Diagnostic tool to investigate the RuntimeWarning about covariance matrices.

This file mimics the manifold generation process with detailed diagnostics
to understand when and why the "covariance is not symmetric positive-semidefinite" 
warning occurs.

TEMPORARY FILE - CAN BE DELETED SAFELY
"""

import sys
import os
import numpy as np
import time
import warnings
from typing import Tuple, Optional

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_generation.manifold_generator import (
    rbf_kernel, 
    generate_intrinsic_coordinates
)


def diagnose_manifold_generation(k: int, N: int, d: int, D: int, 
                                kernel_smoothness: float, base_type: str = 'unit_ball',
                                random_seed: Optional[int] = None) -> None:
    """
    Diagnostic version of manifold generation with detailed covariance analysis.
    """
    print(f"🔍 DIAGNOSTIC: Manifold Generation Analysis")
    print(f"Parameters: k={k}, N={N}, d={d}, D={D}, σ={kernel_smoothness}")
    print("=" * 70)
    
    if random_seed is not None:
        np.random.seed(random_seed)
    
    # 1. Generate intrinsic coordinates
    print("Step 1: Generating intrinsic coordinates...")
    group_intrinsic_coords = []
    for i in range(k):
        coords = generate_intrinsic_coordinates(
            N, d, base_type, 
            random_seed=(random_seed + i) if random_seed is not None else None
        )
        group_intrinsic_coords.append(coords)
        print(f"  Group {i}: shape {coords.shape}, mean norm: {np.mean(np.linalg.norm(coords, axis=1)):.4f}")
    
    # 2. Compute kernel matrices
    print("\nStep 2: Computing kernel matrices...")
    kernel_matrices = {}
    for i in range(k):
        for j in range(k):
            K_ij = rbf_kernel(
                group_intrinsic_coords[i], 
                group_intrinsic_coords[j], 
                kernel_smoothness
            )
            kernel_matrices[(i, j)] = K_ij
            
            # Analyze kernel matrix properties
            if i == j:
                eigenvals = np.linalg.eigvals(K_ij)
                min_eig = np.min(eigenvals)
                cond_num = np.linalg.cond(K_ij)
                print(f"  K[{i},{j}]: min_eigenval={min_eig:.2e}, cond={cond_num:.2e}")
    
    # 3. GP generation with detailed diagnostics
    print("\nStep 3: GP generation with diagnostics...")
    
    threshold = 1e-6
    function_vals = np.empty((k, D, N))
    mus = [[np.zeros(N) for _ in range(D)] for _ in range(k)]
    
    # Pre-compute kernel matrix inverses
    print("  Computing kernel matrix inverses...")
    Sigma_invs = []
    for i in range(k):
        K_ii = kernel_matrices[(i, i)]
        regularized = K_ii + np.eye(N) * threshold
        
        # Check regularized matrix
        eigenvals = np.linalg.eigvals(regularized)
        min_eig = np.min(eigenvals)
        is_pos_def = np.all(eigenvals > 0)
        
        Sigma_inv = np.linalg.inv(regularized)
        Sigma_invs.append(Sigma_inv)
        
        print(f"    Group {i} regularized K: min_eig={min_eig:.2e}, pos_def={is_pos_def}")
    
    # Track warnings during generation
    print("  Generating GP functions...")
    
    for i in range(k):
        print(f"\n  🔍 ANALYZING GROUP {i}")
        
        # Build conditional precision matrix
        print(f"    Building precision matrix...")
        Sigma_inv = (i + 1) * Sigma_invs[i]
        
        print(f"      Initial precision: eigenvals range [{np.min(np.linalg.eigvals(Sigma_inv)):.2e}, {np.max(np.linalg.eigvals(Sigma_inv)):.2e}]")
        
        for j in range(i):
            print(f"      Adding conditioning from group {j}...")
            K_jiK_ii_inv = kernel_matrices[(j, i)] @ Sigma_invs[i]
            Sigma_ji = kernel_matrices[(j, j)] - K_jiK_ii_inv @ kernel_matrices[(j, i)].T
            
            # Check Sigma_ji properties
            Sigma_ji_eigs = np.linalg.eigvals(Sigma_ji)
            Sigma_ji_min_eig = np.min(Sigma_ji_eigs)
            print(f"        Sigma_ji min_eig: {Sigma_ji_min_eig:.2e}")
            
            Sigma_ji_inv = np.linalg.inv(Sigma_ji + np.eye(N) * threshold)
            Sigma_inv += (K_jiK_ii_inv.T @ Sigma_ji_inv @ K_jiK_ii_inv)
            
            # Update means
            for dim_id in range(D):
                mus[i][dim_id] += (K_jiK_ii_inv.T @ Sigma_ji_inv @ function_vals[j][dim_id])
        
        # Convert precision to covariance
        print(f"      Converting precision to covariance...")
        final_precision = Sigma_inv + np.eye(N) * threshold
        
        # Detailed analysis of final precision matrix
        prec_eigenvals = np.linalg.eigvals(final_precision)
        prec_min_eig = np.min(prec_eigenvals)
        prec_max_eig = np.max(prec_eigenvals)
        prec_cond = np.linalg.cond(final_precision)
        is_symmetric = np.allclose(final_precision, final_precision.T)
        is_pos_def = np.all(prec_eigenvals > 0)
        
        print(f"      Final precision matrix:")
        print(f"        Eigenvalue range: [{prec_min_eig:.2e}, {prec_max_eig:.2e}]")
        print(f"        Condition number: {prec_cond:.2e}")
        print(f"        Is symmetric: {is_symmetric}")
        print(f"        Is positive definite: {is_pos_def}")
        
        # Compute covariance
        Sigmas_i = np.linalg.inv(final_precision)
        
        # Analyze resulting covariance matrix
        cov_eigenvals = np.linalg.eigvals(Sigmas_i)
        cov_min_eig = np.min(cov_eigenvals)
        cov_max_eig = np.max(cov_eigenvals)
        cov_cond = np.linalg.cond(Sigmas_i)
        cov_is_symmetric = np.allclose(Sigmas_i, Sigmas_i.T)
        cov_is_pos_def = np.all(cov_eigenvals > 0)
        
        print(f"      Resulting covariance matrix:")
        print(f"        Eigenvalue range: [{cov_min_eig:.2e}, {cov_max_eig:.2e}]")
        print(f"        Condition number: {cov_cond:.2e}")
        print(f"        Is symmetric: {cov_is_symmetric}")
        print(f"        Is positive definite: {cov_is_pos_def}")
        
        if not cov_is_pos_def:
            neg_eigs = np.sum(cov_eigenvals <= 0)
            zero_eigs = np.sum(np.abs(cov_eigenvals) < 1e-14)
            print(f"        ⚠️  Non-positive eigenvalues: {neg_eigs} (zeros: {zero_eigs})")
        
        # Capture warnings during sampling
        print(f"      Sampling {D} GP functions...")
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            for dim_id in range(D):
                mean_vec = Sigmas_i @ mus[i][dim_id]
                
                # Sample and check for warnings
                function_vals[i][dim_id] = np.random.multivariate_normal(mean_vec, Sigmas_i)
                
                # Report any warnings immediately
                if w:
                    for warning in w[-1:]:  # Only show latest warning
                        print(f"        ⚠️  WARNING on dim {dim_id}: {warning.message}")
                        print(f"             Category: {warning.category.__name__}")
                        print(f"             Line: {warning.lineno}")
            
            if not w:
                print(f"        ✅ No warnings during sampling")
            else:
                print(f"        ⚠️  Total warnings: {len(w)}")


def test_different_scenarios():
    """Test different parameter combinations to identify warning patterns."""
    
    scenarios = [
        # Small case
        {"name": "Small case", "k": 2, "N": 100, "d": 2, "D": 5, "sigma": 1.0},
        # Medium case
        {"name": "Medium case", "k": 3, "N": 200, "d": 2, "D": 10, "sigma": 1.0},
        # Different smoothness
        {"name": "Very smooth", "k": 2, "N": 100, "d": 2, "D": 5, "sigma": 10.0},
        {"name": "Less smooth", "k": 2, "N": 100, "d": 2, "D": 5, "sigma": 0.1},
        # Different group count
        {"name": "More groups", "k": 5, "N": 100, "d": 2, "D": 5, "sigma": 1.0},
    ]
    
    for scenario in scenarios:
        print("\n" + "="*80)
        print(f"SCENARIO: {scenario['name']}")
        print("="*80)
        
        try:
            diagnose_manifold_generation(
                k=scenario["k"],
                N=scenario["N"], 
                d=scenario["d"],
                D=scenario["D"],
                kernel_smoothness=scenario["sigma"],
                random_seed=42
            )
        except Exception as e:
            print(f"❌ Error in scenario '{scenario['name']}': {e}")
        
        print("\n" + "-"*50)
        input("Press Enter to continue to next scenario...")


if __name__ == "__main__":
    print("Covariance Matrix Diagnostic Tool")
    print("This tool will analyze the manifold generation process to identify")
    print("when and why the 'covariance is not symmetric positive-semidefinite' warning occurs.")
    print()
    
    choice = input("Choose mode:\n1. Single test case\n2. Multiple scenarios\nEnter choice (1 or 2): ")
    
    if choice == "1":
        # Single test case - use parameters that typically show the warning
        diagnose_manifold_generation(k=3, N=500, d=2, D=20, kernel_smoothness=1.0, random_seed=42)
    elif choice == "2":
        test_different_scenarios()
    else:
        print("Invalid choice. Running single test case...")
        diagnose_manifold_generation(k=3, N=500, d=2, D=20, kernel_smoothness=1.0, random_seed=42)
