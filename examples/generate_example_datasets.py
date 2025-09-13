#!/usr/bin/env python3
"""
Example script demonstrating dataset generation.

This shows how to use the clean, organized data generation system.
"""

import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_generation import DatasetGenerator


def generate_example_datasets():
    """Generate a small set of example datasets."""
    
    print("Neural Network Manifold Denoising - Dataset Generation Example")
    print("=" * 60)
    
    # Initialize generator (will auto-create experiment ID)
    generator = DatasetGenerator(base_output_dir="../data")
    
    print(f"Experiment ID: {generator.experiment_id}")
    print(f"Output directory: {generator.output_dir}")
    
    # Define parameter grid for systematic experiment
    properties_list = generator.generate_parameter_grid(
        k_values=[20],              # Number of groups
        N=1000,                     # Points per group
        ds=[2, 4],                  # Intrinsic dimensions to test
        Ds=[50, 100],               # Ambient dimensions to test
        kernel_smoothness=[0.5, 1.0], # Kernel smoothness values
        noise_sigma=0.01,           # Noise level
        base_type='unit_ball',      # Base manifold type
        train_ratio=0.8,            # Training data fraction
        centralize=True,            # Center data at origin
        rescale=True,               # Rescale to unit radius
        random_seed=42              # For reproducibility
    )
    
    print(f"Generated {len(properties_list)} dataset configurations")
    print("Parameter combinations:")
    for i, props in enumerate(properties_list):
        print(f"  Dataset {i}: d={props.d}, D={props.D}, σ={props.kernel_smoothness}")
    
    # Generate all datasets
    datasets = generator.generate_datasets(
        properties_list,
        create_train_test=True,     # Create train/test splits
        save_individual=True,       # Save each dataset
        verbose=True                # Print progress
    )
    
    # Show summary
    summary = generator.get_experiment_summary()
    print("\n" + "=" * 60)
    print("EXPERIMENT COMPLETE")
    print("=" * 60)
    print(f"✅ Successfully generated {summary['successful_datasets']} datasets")
    print(f"📁 Location: {summary['output_directory']}")
    print(f"⏱️  Total time: {summary['total_time']:.2f}s")
    
    if datasets:
        # Show details of first dataset
        first_dataset = datasets[0]
        print(f"\nFirst dataset example:")
        print(f"  Dataset ID: {first_dataset.properties.dataset_id}")
        print(f"  Properties: d={first_dataset.properties.d}, D={first_dataset.properties.D}")
        print(f"  Shapes:")
        if first_dataset.clean_embedded is not None:
            print(f"    Clean embedded: {first_dataset.clean_embedded.shape}")
        if first_dataset.noisy_embedded is not None:
            print(f"    Noisy embedded: {first_dataset.noisy_embedded.shape}")
        if first_dataset.intrinsic_coords is not None:
            print(f"    Intrinsic coords: {first_dataset.intrinsic_coords.shape}")
        if first_dataset.train_test_data is not None:
            print(f"    Train data: {first_dataset.train_test_data['train_data'].shape}")
            print(f"    Test data: {first_dataset.train_test_data['test_data'].shape}")
    
    return generator, datasets


if __name__ == "__main__":
    # Generate example datasets
    generator, datasets = generate_example_datasets()
    
    print(f"\n🎉 Example complete! Check {generator.output_dir} for results.")
    print("\nNext steps:")
    print("1. Implement geometric analysis (Step 2)")  
    print("2. Create visualization tools (Step 3)")
    print("3. Design neural network training (Step 4)")
    print("4. Build analysis pipeline (Step 5)")
