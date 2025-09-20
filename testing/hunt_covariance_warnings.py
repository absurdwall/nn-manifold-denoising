#!/usr/bin/env python3
"""
Targeted diagnostic to reproduce the specific covariance warning.

Since the warning is intermittent, this tool will:
1. Test the exact parameters from your original runs
2. Try multiple random seeds to find problematic cases
3. Test edge cases that might trigger numerical issues

TEMPORARY FILE - CAN BE DELETED SAFELY
"""

import sys
import os
import numpy as np
import warnings
from typing import List, Dict, Any

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_generation.manifold_generator import generate_manifold_embedding


def capture_warnings_during_generation(k: int, N: int, d: int, D: int, 
                                      kernel_smoothness: float, 
                                      random_seed: int = None) -> Dict[str, Any]:
    """
    Run manifold generation and capture any warnings.
    
    Returns:
        Dictionary with warning info and generation results
    """
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        
        try:
            embedded, intrinsic = generate_manifold_embedding(
                k=k, N=N, d=d, D=D, 
                kernel_smoothness=kernel_smoothness,
                random_seed=random_seed,
                verbose=False  # Quiet mode to focus on warnings
            )
            
            success = True
            error = None
            
        except Exception as e:
            success = False
            error = str(e)
            embedded, intrinsic = None, None
        
        # Analyze warnings
        covariance_warnings = []
        other_warnings = []
        
        for warning in w:
            warning_info = {
                'message': str(warning.message),
                'category': warning.category.__name__,
                'filename': warning.filename,
                'lineno': warning.lineno
            }
            
            if 'covariance' in str(warning.message).lower() and 'symmetric' in str(warning.message).lower():
                covariance_warnings.append(warning_info)
            else:
                other_warnings.append(warning_info)
        
        return {
            'success': success,
            'error': error,
            'covariance_warnings': covariance_warnings,
            'other_warnings': other_warnings,
            'total_warnings': len(w),
            'shapes': (embedded.shape, intrinsic.shape) if success else None,
            'parameters': {'k': k, 'N': N, 'd': d, 'D': D, 'sigma': kernel_smoothness, 'seed': random_seed}
        }


def test_parameter_space():
    """
    Test a wide range of parameters and seeds to find problematic cases.
    """
    print("🔍 SYSTEMATIC SEARCH for Covariance Warnings")
    print("=" * 60)
    
    # Parameters similar to your original problematic case
    base_configs = [
        # Your typical case that showed warnings
        {'k': 10, 'N': 1000, 'd': 2, 'D': 100, 'sigma': 1.0},
        # Variations that might be more sensitive
        {'k': 10, 'N': 1000, 'd': 2, 'D': 50, 'sigma': 1.0},
        {'k': 5, 'N': 1000, 'd': 2, 'D': 100, 'sigma': 1.0},
        {'k': 10, 'N': 500, 'd': 2, 'D': 100, 'sigma': 1.0},
        # Different smoothness values
        {'k': 5, 'N': 500, 'd': 2, 'D': 50, 'sigma': 0.1},
        {'k': 5, 'N': 500, 'd': 2, 'D': 50, 'sigma': 10.0},
    ]
    
    warning_cases = []
    total_tests = 0
    
    for config in base_configs:
        print(f"\nTesting config: k={config['k']}, N={config['N']}, D={config['D']}, σ={config['sigma']}")
        
        # Test multiple random seeds for each config
        seeds_to_test = [42, 123, 456, 789, 999, 1337, 2023, 2024, None, None]  # Include None for random
        
        config_warnings = 0
        for i, seed in enumerate(seeds_to_test):
            total_tests += 1
            
            result = capture_warnings_during_generation(**config, random_seed=seed)
            
            if result['covariance_warnings']:
                warning_cases.append(result)
                config_warnings += 1
                print(f"  Seed {seed}: ⚠️  {len(result['covariance_warnings'])} covariance warnings")
                for warning in result['covariance_warnings']:
                    print(f"    - Line {warning['lineno']}: {warning['message']}")
            elif not result['success']:
                print(f"  Seed {seed}: ❌ Error: {result['error']}")
            else:
                print(f"  Seed {seed}: ✅ No warnings")
            
            # Don't spam too much for successful cases
            if i >= 2 and config_warnings == 0:
                print(f"  ... (testing {len(seeds_to_test) - i - 1} more seeds quietly)")
                # Test remaining seeds quietly
                for remaining_seed in seeds_to_test[i+1:]:
                    total_tests += 1
                    result = capture_warnings_during_generation(**config, random_seed=remaining_seed)
                    if result['covariance_warnings']:
                        warning_cases.append(result)
                        config_warnings += 1
                break
        
        print(f"  Config summary: {config_warnings} cases with warnings out of {len(seeds_to_test)} tests")
    
    # Summary
    print(f"\n" + "=" * 60)
    print(f"SEARCH SUMMARY")
    print(f"=" * 60)
    print(f"Total tests run: {total_tests}")
    print(f"Cases with covariance warnings: {len(warning_cases)}")
    print(f"Warning rate: {len(warning_cases)/total_tests*100:.1f}%")
    
    if warning_cases:
        print(f"\n📋 PROBLEMATIC CASES:")
        for i, case in enumerate(warning_cases[:5]):  # Show first 5
            params = case['parameters']
            print(f"  Case {i+1}: k={params['k']}, N={params['N']}, D={params['D']}, σ={params['sigma']}, seed={params['seed']}")
            print(f"    Warnings: {len(case['covariance_warnings'])}")
            for warning in case['covariance_warnings']:
                print(f"      Line {warning['lineno']}: {warning['message'][:80]}...")
        
        if len(warning_cases) > 5:
            print(f"  ... and {len(warning_cases) - 5} more cases")
        
        # Recommend a specific case for detailed analysis
        best_case = warning_cases[0]
        params = best_case['parameters']
        print(f"\n🎯 RECOMMENDED FOR DETAILED ANALYSIS:")
        print(f"Parameters: k={params['k']}, N={params['N']}, d={params['d']}, D={params['D']}, σ={params['sigma']}")
        print(f"Seed: {params['seed']}")
        print(f"Run this command to reproduce:")
        print(f"  python -c \"from test.diagnose_covariance_warnings import diagnose_manifold_generation; diagnose_manifold_generation({params['k']}, {params['N']}, {params['d']}, {params['D']}, {params['sigma']}, random_seed={params['seed']})\"")
        
    else:
        print(f"\n🤔 NO WARNINGS FOUND")
        print(f"The covariance warning might be:")
        print(f"  1. Very rare and requires specific conditions we haven't hit")
        print(f"  2. Dependent on system/environment factors")
        print(f"  3. Related to specific data patterns not captured in our tests")
        print(f"  4. A transient numerical issue that's hard to reproduce")
        
        print(f"\n💡 SUGGESTIONS:")
        print(f"  - Try running your actual dataset generation script that showed the warning")
        print(f"  - Check if the warning appears with your specific --quick or --full parameter sets")
        print(f"  - The warning might be harmless - numpy sometimes flags matrices that are")
        print(f"    technically positive definite but have eigenvalues very close to zero")


def test_edge_cases():
    """Test edge cases that might trigger numerical issues."""
    print("\n🧪 TESTING EDGE CASES")
    print("=" * 40)
    
    edge_cases = [
        # Very small regularization (more sensitive to numerical issues)
        {'name': 'Small regularization', 'k': 3, 'N': 100, 'd': 2, 'D': 10, 'sigma': 1.0},
        # Large condition numbers
        {'name': 'Very smooth kernel', 'k': 3, 'N': 200, 'd': 2, 'D': 10, 'sigma': 100.0},
        # Very rough kernel
        {'name': 'Very rough kernel', 'k': 3, 'N': 200, 'd': 2, 'D': 10, 'sigma': 0.01},
        # Many groups
        {'name': 'Many groups', 'k': 20, 'N': 100, 'd': 2, 'D': 10, 'sigma': 1.0},
    ]
    
    for case in edge_cases:
        print(f"\nTesting: {case['name']}")
        case_params = {k: v for k, v in case.items() if k != 'name'}
        
        # Test a few seeds
        for seed in [42, 123, 456]:
            result = capture_warnings_during_generation(**case_params, random_seed=seed)
            
            if result['covariance_warnings']:
                print(f"  Seed {seed}: ⚠️  Found covariance warning!")
                return result['parameters']  # Return first problematic case
            else:
                print(f"  Seed {seed}: ✅ OK")
    
    return None


if __name__ == "__main__":
    print("Targeted Covariance Warning Hunter")
    print("This tool will systematically search for the conditions that trigger")
    print("the 'covariance is not symmetric positive-semidefinite' warning.")
    print()
    
    # First try systematic parameter space search
    test_parameter_space()
    
    # Then try edge cases
    problematic_case = test_edge_cases()
    
    if problematic_case:
        print(f"\n🎯 Found problematic case in edge testing!")
        print(f"Parameters: {problematic_case}")
    else:
        print(f"\n🤷 No warnings found in edge cases either.")
        
    print(f"\n" + "=" * 60)
    print(f"If no warnings were found, the issue might be:")
    print(f"1. Very environment-specific (different numpy/scipy versions)")
    print(f"2. Dependent on exact timing or memory state")
    print(f"3. Related to your specific parameter combinations")
    print(f"4. A harmless numerical precision warning that can be ignored")
    print(f"=" * 60)
