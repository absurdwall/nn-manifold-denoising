#!/usr/bin/env python3
"""
Step 1: Generate Manifold Datasets

Generate synthetic manifold datasets with specified parameters.
This script provides a command-line interface for dataset generation.

Usage:
    python step1_generate_datasets.py --ds 2,4 --Ds 100,300 --sigma 0.1,1,10
    python step1_generate_datasets.py --quick  # Quick test with small parameters
    
Examples:
    # Generate datasets with multiple dimensions and smoothness values
    python step1_generate_datasets.py --ds 2,4,8 --Ds 100,300,1000 --sigma 0.01,0.1,1,10,100
    
    # Quick test with minimal parameters
    python step1_generate_datasets.py --quick
    
    # Custom experiment with specific parameters
    python step1_generate_datasets.py --k 20 --N 1000 --ds 4 --Ds 200 --sigma 1.0 --noise 0.01
"""

import argparse
import sys
import os
from typing import List

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from data_generation.dataset_generator import DatasetGenerator
except ImportError as e:
    print(f"Error importing DatasetGenerator: {e}")
    print("Make sure you're running from the correct directory and src/ is available.")
    print("Current working directory:", os.getcwd())
    print("Python path:", sys.path[:3])
    sys.exit(1)


def parse_list_arg(arg_str: str, dtype=float) -> List:
    """Parse comma-separated string into list of specified type."""
    if arg_str is None:
        return []
    return [dtype(x.strip()) for x in arg_str.split(',')]


def get_quick_parameters():
    """Get minimal parameters for quick testing."""
    return {
        'k_values': [5],
        'N': 500,
        'ds': [2, 4],
        'Ds': [50, 100],
        'kernel_smoothness': [0.1, 1.0],
        'noise_sigma': 0.01,
        'base_type': 'unit_ball',
        'train_ratio': 0.8,
        'centralize': True,
        'rescale': True,
        'apply_rotation': True,
        'random_seed': 42
    }


def get_full_parameters():
    """Get full parameter set for comprehensive experiments."""
    return {
        'k_values': [10],
        'N': 1000,
        'ds': [2, 4],
        'Ds': [100, 300],
        'kernel_smoothness': [0.01, 0.1, 1, 10, 100],
        'noise_sigma': 0.01,
        'base_type': 'unit_ball',
        'train_ratio': 0.8,
        'centralize': True,
        'rescale': True,
        'apply_rotation': True,
        'random_seed': 42
    }


def main():
    parser = argparse.ArgumentParser(
        description='Generate manifold datasets for neural network denoising experiments',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Parameter options
    parser.add_argument('--k', type=int, default=10,
                       help='Number of groups (default: 10)')
    parser.add_argument('--N', type=int, default=1000,
                       help='Points per group (default: 1000)')
    parser.add_argument('--ds', type=str, default='2,4',
                       help='Intrinsic dimensions (comma-separated, default: 2,4)')
    parser.add_argument('--Ds', type=str, default='100,300',
                       help='Ambient dimensions (comma-separated, default: 100,300)')
    parser.add_argument('--sigma', type=str, default='0.01,0.1,1,10,100',
                       help='Kernel smoothness values (comma-separated, default: 0.01,0.1,1,10,100)')
    parser.add_argument('--noise', type=float, default=0.01,
                       help='Noise level (default: 0.01)')
    
    # Processing options
    parser.add_argument('--base-type', default='unit_ball', choices=['unit_ball', 'rectangle', 'sphere'],
                       help='Base manifold type (default: unit_ball)')
    parser.add_argument('--train-ratio', type=float, default=0.8,
                       help='Training data fraction (default: 0.8)')
    parser.add_argument('--no-centralize', action='store_true',
                       help='Disable data centralization')
    parser.add_argument('--no-rescale', action='store_true',
                       help='Disable data rescaling')
    parser.add_argument('--no-rotation', action='store_true',
                       help='Disable random rotation')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')
    
    # Experiment options
    from datetime import datetime
    now = datetime.now()
    default_experiment_id = f"data_{now.year-2000:02d}{now.month:02d}{now.day:02d}_{now.hour:02d}{now.minute:02d}"
    parser.add_argument('--experiment-id', type=str, default=default_experiment_id,
                       help=f'Experiment identifier (default: {default_experiment_id})')
    parser.add_argument('--output-dir', default='data',
                       help='Output directory (default: data)')
    
    # Convenience options
    parser.add_argument('--quick', action='store_true',
                       help='Use quick test parameters (overrides other parameter settings)')
    parser.add_argument('--full', action='store_true',
                       help='Use full parameter set for comprehensive experiments')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would be generated without actually generating')
    
    args = parser.parse_args()
    
    # Determine parameters
    if args.quick:
        print("Using quick test parameters...")
        params = get_quick_parameters()
    elif args.full:
        print("Using full parameter set...")
        params = get_full_parameters()
    else:
        # Use command line arguments
        params = {
            'k_values': [args.k],
            'N': args.N,
            'ds': parse_list_arg(args.ds, int),
            'Ds': parse_list_arg(args.Ds, int),
            'kernel_smoothness': parse_list_arg(args.sigma, float),
            'noise_sigma': args.noise,
            'base_type': args.base_type,
            'train_ratio': args.train_ratio,
            'centralize': not args.no_centralize,
            'rescale': not args.no_rescale,
            'apply_rotation': not args.no_rotation,
            'random_seed': args.seed
        }
    
    print("Neural Network Manifold Denoising - Step 1: Dataset Generation")
    print("=" * 70)
    
    # Initialize generator
    generator = DatasetGenerator(
        experiment_id=args.experiment_id,
        base_output_dir=args.output_dir
    )
    
    print(f"Experiment ID: {generator.experiment_id}")
    print(f"Output directory: {generator.output_dir}")
    
    # Generate parameter grid
    properties_list = generator.generate_parameter_grid(**params)
    
    print(f"\nDataset Configuration:")
    print(f"  k (groups): {params['k_values']}")
    print(f"  N (points per group): {params['N']}")
    print(f"  d (intrinsic dims): {params['ds']}")
    print(f"  D (ambient dims): {params['Ds']}")
    print(f"  σ (kernel smoothness): {params['kernel_smoothness']}")
    print(f"  noise_σ: {params['noise_sigma']}")
    print(f"  base_type: {params['base_type']}")
    print(f"  Processing: centralize={params['centralize']}, rescale={params['rescale']}, rotation={params['apply_rotation']}")
    
    print(f"\nTotal combinations: {len(params['ds'])} d × {len(params['Ds'])} D × {len(params['k_values'])} k × {len(params['kernel_smoothness'])} σ = {len(properties_list)} datasets")
    
    if args.dry_run:
        print("\nDry run - showing what would be generated:")
        for i, props in enumerate(properties_list):
            print(f"  Dataset {i:2d}: d={props.d}, D={props.D:3d}, k={props.k}, N={props.N}, σ={props.kernel_smoothness:6.2f}")
        print(f"\nWould generate {len(properties_list)} datasets in {generator.output_dir}")
        return
    
    # # Confirm generation
    # response = input(f"\nGenerate {len(properties_list)} datasets? [y/N]: ")
    # if response.lower() not in ['y', 'yes']:
    #     print("Generation cancelled.")
    #     return
    
    # Generate datasets
    print("\nStarting dataset generation...")
    datasets = generator.generate_datasets(
        properties_list,
        create_train_test=True,
        save_individual=True,
        verbose=True
    )
    
    # Show summary
    summary = generator.get_experiment_summary()
    print("\n" + "=" * 70)
    print("STEP 1 COMPLETE: Dataset Generation")
    print("=" * 70)
    print(f"✅ Successfully generated {summary['successful_datasets']}/{len(properties_list)} datasets")
    print(f"📁 Location: {summary['output_directory']}")
    print(f"⏱️  Total time: {summary['total_time']:.2f}s ({summary['total_time']/60:.2f}m)")
    
    if summary['failed_datasets'] > 0:
        print(f"❌ Failed: {summary['failed_datasets']} datasets")
        print("Check the generation_summary.json for error details.")
    
    if datasets:
        # Show example of first dataset
        first_dataset = datasets[0]
        props = first_dataset.properties
        print(f"\nExample dataset (dataset 0):")
        print(f"  Parameters: d={props.d}, D={props.D}, k={props.k}, N={props.N}, σ={props.kernel_smoothness}")
        if first_dataset.clean_embedded is not None:
            print(f"  Clean shape: {first_dataset.clean_embedded.shape}")
        if first_dataset.noisy_embedded is not None:
            print(f"  Noisy shape: {first_dataset.noisy_embedded.shape}")
        if first_dataset.train_test_data is not None:
            train_shape = first_dataset.train_test_data['train_data'].shape
            test_shape = first_dataset.train_test_data['test_data'].shape
            print(f"  Train/test: {train_shape}/{test_shape}")
    
    print(f"\n🎉 Step 1 complete! Ready for Step 2: Geometric Analysis")
    print(f"Next: python step2_geometric_analysis.py --experiment-id {generator.experiment_id}")


if __name__ == "__main__":
    main()
