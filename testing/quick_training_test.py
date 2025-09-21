#!/usr/bin/env python3
"""
Quick test of the full training pipeline with minimal configuration.
This creates dummy datasets and runs a small subset of experiments.
"""

import os
import sys
import torch
import numpy as np
import json
from torch.utils.data import TensorDataset, DataLoader

# Add src to path for imports  
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from network_training import (
    NetworkConfig, TrainingConfig, ExperimentTracker,
    create_network, train_model, evaluate_model, 
    create_train_val_split
)


def create_dummy_manifold_dataset(dataset_id=0, n_samples=1000):
    """Create a dummy manifold dataset for testing."""
    torch.manual_seed(42 + dataset_id)
    
    # Simple 2D manifold in 5D space
    intrinsic_dim = 2
    ambient_dim = 5
    
    # Generate points on a 2D surface
    t = torch.linspace(0, 2*np.pi, n_samples)
    s = torch.linspace(-1, 1, n_samples)
    t, s = torch.meshgrid(t, s, indexing='ij')
    t, s = t.flatten()[:n_samples], s.flatten()[:n_samples]
    
    # Manifold: torus-like surface
    x1 = (2 + s) * torch.cos(t)
    x2 = (2 + s) * torch.sin(t) 
    x3 = s
    x4 = 0.5 * torch.sin(2*t) * s
    x5 = 0.3 * torch.cos(3*t) * s
    
    x_clean = torch.stack([x1, x2, x3, x4, x5], dim=1)
    
    # Add noise
    x_noisy = x_clean + 0.05 * torch.randn_like(x_clean)
    
    # Create train/test split
    n_train = int(0.8 * n_samples)
    train_dataset = TensorDataset(x_noisy[:n_train], x_clean[:n_train])
    test_dataset = TensorDataset(x_noisy[n_train:], x_clean[n_train:])
    
    metadata = {
        'D': ambient_dim,
        'd': intrinsic_dim,
        'n_samples': n_samples,
        'noise_level': 0.05,
        'dataset_id': dataset_id
    }
    
    return train_dataset, test_dataset, metadata


def run_quick_training_test():
    """Run a quick test of the training pipeline."""
    print("Quick Training Pipeline Test")
    print("=" * 40)
    
    # Set random seed
    torch.manual_seed(42)
    np.random.seed(42)
    
    device = torch.device('cpu')  # Use CPU for quick test
    
    # Create dummy datasets
    datasets = []
    for i in range(2):  # Just 2 datasets
        train_data, test_data, metadata = create_dummy_manifold_dataset(i)
        datasets.append((train_data, test_data, metadata))
    
    # Define network configurations
    network_configs = [
        NetworkConfig(
            network_type="DeepFCNet",
            input_dim=5, output_dim=5,
            width=64, depth=2,
            init_scheme="standard",
            norm_type="batch"
        ),
        NetworkConfig(
            network_type="DeepFCNet", 
            input_dim=5, output_dim=5,
            width=64, depth=4,
            init_scheme="rezero",
            norm_type="batch"
        )
    ]
    
    # Define training configurations
    training_configs = [
        TrainingConfig(
            max_epochs=30,
            batch_size=64,
            learning_rate=1e-3,
            optimizer_name='adam',
            early_stop_patience=10
        )
    ]
    
    # Initialize experiment tracker
    tracker = ExperimentTracker()
    
    # Run experiments
    total_experiments = len(datasets) * len(network_configs) * len(training_configs)
    completed = 0
    
    print(f"Running {total_experiments} experiments...")
    
    for dataset_idx, (train_dataset, test_dataset, metadata) in enumerate(datasets):
        print(f"\nDataset {dataset_idx + 1}/{len(datasets)}")
        
        # Create data loaders
        N = len(train_dataset)
        train_subset, val_subset = create_train_val_split(train_dataset, N, 0.2)
        
        train_loader = DataLoader(train_subset, batch_size=64, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=64, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
        
        for net_idx, network_config in enumerate(network_configs):
            for train_idx, training_config in enumerate(training_configs):
                completed += 1
                experiment_id = f"dataset{dataset_idx}_net{net_idx}_train{train_idx}"
                
                print(f"  Experiment {completed}/{total_experiments}: {experiment_id}")
                
                try:
                    # Create and train model
                    model = create_network(network_config).to(device)
                    
                    metrics, best_state = train_model(
                        model, train_loader, val_loader, training_config,
                        device=device, verbose=False
                    )
                    
                    # Evaluate
                    test_loss = evaluate_model(model, test_loader, device)
                    
                    # Track experiment
                    tracker.add_experiment(
                        experiment_id=experiment_id,
                        network_config=network_config.to_dict(),
                        training_config=training_config.to_dict(),
                        metrics=metrics,
                        test_loss=test_loss,
                        dataset_info=metadata
                    )
                    
                    print(f"    ✓ Success: test_loss={test_loss:.4f}, epochs={metrics.epochs_completed}")
                    
                except Exception as e:
                    print(f"    ✗ Failed: {e}")
    
    # Print summary
    print("\n" + "=" * 40)
    print("EXPERIMENT SUMMARY")
    print("=" * 40)
    
    if tracker.experiments:
        best_experiments = tracker.get_best_experiments(metric='test_loss', top_k=3)
        
        print("Top 3 experiments by test loss:")
        for i, exp in enumerate(best_experiments, 1):
            print(f"{i}. {exp['experiment_id']}: {exp['test_loss']:.4f}")
        
        # Save results
        results_dir = "/home/tim/python_projects/nn_manifold_denoising/testing/quick_test_results"
        os.makedirs(results_dir, exist_ok=True)
        tracker.save_summary(os.path.join(results_dir, "experiment_summary.json"))
        print(f"\nResults saved to: {results_dir}")
    else:
        print("No successful experiments!")
    
    print("\n✓ Quick training test completed!")


if __name__ == "__main__":
    run_quick_training_test()
