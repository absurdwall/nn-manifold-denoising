#!/usr/bin/env python3
"""
Run specific experiments for validation.

Experiment 1: Single network configuration across all datasets
Experiment 2: Multiple network configurations on single dataset
"""

import os
import sys
import json
import argparse
import torch
import numpy as np
import pandas as pd
from itertools import product
from torch.utils.data import DataLoader, TensorDataset
import time

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from network_training import (
    NetworkConfig, TrainingConfig, ExperimentTracker,
    create_network, train_model, evaluate_model, 
    create_train_val_split
)


def load_dataset_from_directory(dataset_dir):
    """
    Load dataset from your format.
    
    Args:
        dataset_dir: Path to dataset directory containing .pt files
        
    Returns:
        train_dataset, test_dataset, metadata
    """
    # Load metadata
    dataset_name = os.path.basename(dataset_dir)
    metadata_file = os.path.join(dataset_dir, f"{dataset_name}_metadata.json")
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
    
    # Extract relevant properties
    properties = metadata['properties']
    dataset_metadata = {
        'D': properties['D'],
        'd': properties['d'],
        'noise_level': properties['noise_sigma'],
        'N': properties['N'],
        'dataset_id': dataset_name,
        'experiment_id': properties['experiment_id']
    }
    
    # Load data tensors
    train_data = torch.load(os.path.join(dataset_dir, f"{dataset_name}_train_data.pt"))
    train_clean = torch.load(os.path.join(dataset_dir, f"{dataset_name}_train_clean.pt"))
    test_data = torch.load(os.path.join(dataset_dir, f"{dataset_name}_test_data.pt"))
    test_clean = torch.load(os.path.join(dataset_dir, f"{dataset_name}_test_clean.pt"))
    
    # Create datasets (task: denoise noisy data to clean data)
    train_dataset = TensorDataset(train_data, train_clean)
    test_dataset = TensorDataset(test_data, test_clean)
    
    return train_dataset, test_dataset, dataset_metadata


def find_datasets_in_directory(data_dir):
    """Find all dataset directories in the given directory."""
    datasets = []
    
    if os.path.exists(data_dir):
        for item in os.listdir(data_dir):
            item_path = os.path.join(data_dir, item)
            if os.path.isdir(item_path) and item.startswith('dataset'):
                datasets.append(item_path)
    
    return sorted(datasets)


def run_experiment_1():
    """
    Experiment 1: Single network configuration across all datasets in data_250914_0100
    Network: FC, no norm/dropout, depth=2, width=400, lr=1e-3, batch_size=512
    """
    print("="*60)
    print("EXPERIMENT 1: Single Network, All Datasets")
    print("="*60)
    
    # Setup
    data_dir = "/home/tim/python_projects/nn_manifold_denoising/data/data_250914_0100"
    results_dir = "results/nn_train_exp1_250914_0100"
    
    # Create results directory with timestamp
    timestamp = time.strftime("%Y%m%d_%H%M")
    tb_num = f"tb{timestamp}"
    exp_results_dir = os.path.join(results_dir, tb_num)
    os.makedirs(exp_results_dir, exist_ok=True)
    
    # Find all datasets
    dataset_dirs = find_datasets_in_directory(data_dir)
    print(f"Found {len(dataset_dirs)} datasets")
    
    # Network configuration
    network_config = NetworkConfig(
        network_type="DeepFCNet",
        input_dim=None,  # Will be set from data
        output_dim=None,  # Will be set from data
        width=400,
        depth=2,
        activation="relu",
        norm_type=None,
        dropout=0.0,
        use_residual=True,
        init_scheme="standard"
    )
    
    # Training configuration
    training_config = TrainingConfig(
        max_epochs=2000,
        batch_size=512,
        learning_rate=1e-3,
        optimizer_name='adam',
        scheduler_mode='fixed',
        weight_decay=0.0,
        early_stop_patience=50,
        val_ratio=0.2
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize experiment tracker
    tracker = ExperimentTracker()
    
    # Run experiments
    results = []
    for i, dataset_dir in enumerate(dataset_dirs):
        dataset_name = os.path.basename(dataset_dir)
        print(f"\nDataset {i+1}/{len(dataset_dirs)}: {dataset_name}")
        
        try:
            # Load dataset
            train_dataset, test_dataset, metadata = load_dataset_from_directory(dataset_dir)
            
            # Set dimensions
            network_config.input_dim = metadata['D']
            network_config.output_dim = metadata['D']
            
            # Create model
            model = create_network(network_config).to(device)
            
            # Create data loaders
            N = len(train_dataset)
            train_subset, val_subset = create_train_val_split(train_dataset, N, training_config.val_ratio)
            
            train_loader = DataLoader(train_subset, batch_size=training_config.batch_size, shuffle=True)
            val_loader = DataLoader(val_subset, batch_size=training_config.batch_size, shuffle=False)
            test_loader = DataLoader(test_dataset, batch_size=training_config.batch_size, shuffle=False)
            
            # Train model
            print(f"  Training...")
            metrics, _ = train_model(model, train_loader, val_loader, training_config, device, verbose=False)
            
            # Test model
            test_loss = evaluate_model(model, test_loader, device)
            
            # Save results
            experiment_id = f"{dataset_name}_single_net"
            result = {
                'experiment_id': experiment_id,
                'dataset_name': dataset_name,
                'dataset_metadata': metadata,
                'network_config': network_config.to_dict(),
                'training_config': training_config.to_dict(),
                'test_loss': test_loss,
                'best_val_loss': metrics.best_val_loss,
                'epochs_completed': metrics.epochs_completed,
                'training_time': metrics.training_time
            }
            
            results.append(result)
            
            # Save individual result
            result_file = os.path.join(exp_results_dir, f"{experiment_id}.json")
            with open(result_file, 'w') as f:
                json.dump(result, f, indent=2)
            
            print(f"  ✓ Success: test_loss={test_loss:.6f}, epochs={metrics.epochs_completed}")
            
        except Exception as e:
            print(f"  ✗ Failed: {e}")
            import traceback
            traceback.print_exc()
    
    # Save summary
    summary_file = os.path.join(exp_results_dir, "experiment_summary.json")
    with open(summary_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save results to CSV for easy analysis
    if results:
        import pandas as pd
        csv_data = []
        for result in results:
            row = {
                'experiment_id': result['experiment_id'],
                'dataset_name': result['dataset_name'],
                'test_loss': result['test_loss'],
                'best_val_loss': result['best_val_loss'],
                'epochs_completed': result['epochs_completed'],
                'training_time': result['training_time'],
                # Dataset properties
                'D': result['dataset_metadata']['D'],
                'd': result['dataset_metadata']['d'],
                'noise_level': result['dataset_metadata']['noise_level'],
                'N': result['dataset_metadata']['N'],
                # Network properties
                'network_type': result['network_config']['network_type'],
                'width': result['network_config']['width'],
                'depth': result['network_config']['depth'],
                'activation': result['network_config']['activation'],
                'norm_type': result['network_config']['norm_type'],
                'use_residual': result['network_config']['use_residual'],
                'init_scheme': result['network_config']['init_scheme'],
                # Training properties
                'learning_rate': result['training_config']['learning_rate'],
                'optimizer_name': result['training_config']['optimizer_name'],
                'scheduler_mode': result['training_config']['scheduler_mode'],
                'batch_size': result['training_config']['batch_size'],
            }
            csv_data.append(row)
        
        df = pd.DataFrame(csv_data)
        csv_path = os.path.join(exp_results_dir, 'experiment1_results.csv')
        df.to_csv(csv_path, index=False)
        print(f"CSV results saved to: {csv_path}")
    
    print(f"\nExperiment 1 completed. Results saved to: {exp_results_dir}")
    return results


def run_experiment_2():
    """
    Experiment 2: Multiple network configurations on single dataset
    Vary: depth [1,2,4,8], width [100,200,400,800], lr [1e-3,1e-4,1e-5], batch_size=1024
    """
    print("="*60)
    print("EXPERIMENT 2: Multiple Networks, Single Dataset")
    print("="*60)
    
    # Setup
    data_dir = "/home/tim/python_projects/nn_manifold_denoising/data/data_250914_0100_01"
    results_dir = "results/nn_train_exp2_250914_0100_01"
    
    # Create results directory with timestamp
    timestamp = time.strftime("%Y%m%d_%H%M")
    tb_num = f"tb{timestamp}"
    exp_results_dir = os.path.join(results_dir, tb_num)
    os.makedirs(exp_results_dir, exist_ok=True)
    
    # Find dataset
    dataset_dirs = find_datasets_in_directory(data_dir)
    if not dataset_dirs:
        print("No datasets found!")
        return []
    
    dataset_dir = dataset_dirs[0]  # Should be only one
    dataset_name = os.path.basename(dataset_dir)
    print(f"Using dataset: {dataset_name}")
    
    # Load dataset once
    train_dataset, test_dataset, metadata = load_dataset_from_directory(dataset_dir)
    
    # Parameter grids
    depths = [1, 2, 4, 8]
    widths = [100, 200, 400, 800]
    learning_rates = [1e-3, 1e-4, 1e-5]
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Run experiments
    results = []
    total_experiments = len(depths) * len(widths) * len(learning_rates)
    completed = 0
    
    print(f"Running {total_experiments} experiments...")
    
    for depth in depths:
        for width in widths:
            for lr in learning_rates:
                completed += 1
                experiment_id = f"{dataset_name}_d{depth}_w{width}_lr{lr:.0e}"
                print(f"\nExperiment {completed}/{total_experiments}: {experiment_id}")
                
                try:
                    # Network configuration
                    network_config = NetworkConfig(
                        network_type="DeepFCNet",
                        input_dim=metadata['D'],
                        output_dim=metadata['D'],
                        width=width,
                        depth=depth,
                        activation="relu",
                        norm_type=None,
                        dropout=0.0,
                        use_residual=True,
                        init_scheme="standard"
                    )
                    
                    # Training configuration
                    training_config = TrainingConfig(
                        max_epochs=2000,
                        batch_size=1024,
                        learning_rate=lr,
                        optimizer_name='adam',
                        scheduler_mode='fixed',
                        weight_decay=0.0,
                        early_stop_patience=50,
                        val_ratio=0.2
                    )
                    
                    # Create model
                    model = create_network(network_config).to(device)
                    
                    # Create data loaders
                    N = len(train_dataset)
                    train_subset, val_subset = create_train_val_split(train_dataset, N, training_config.val_ratio)
                    
                    train_loader = DataLoader(train_subset, batch_size=training_config.batch_size, shuffle=True)
                    val_loader = DataLoader(val_subset, batch_size=training_config.batch_size, shuffle=False)
                    test_loader = DataLoader(test_dataset, batch_size=training_config.batch_size, shuffle=False)
                    
                    # Train model
                    metrics, _ = train_model(model, train_loader, val_loader, training_config, device, verbose=False)
                    
                    # Test model
                    test_loss = evaluate_model(model, test_loader, device)
                    
                    # Save results
                    result = {
                        'experiment_id': experiment_id,
                        'dataset_name': dataset_name,
                        'dataset_metadata': metadata,
                        'network_config': network_config.to_dict(),
                        'training_config': training_config.to_dict(),
                        'test_loss': test_loss,
                        'best_val_loss': metrics.best_val_loss,
                        'epochs_completed': metrics.epochs_completed,
                        'training_time': metrics.training_time,
                        'depth': depth,
                        'width': width,
                        'learning_rate': lr
                    }
                    
                    results.append(result)
                    
                    # Save individual result
                    result_file = os.path.join(exp_results_dir, f"{experiment_id}.json")
                    with open(result_file, 'w') as f:
                        json.dump(result, f, indent=2)
                    
                    print(f"  ✓ Success: test_loss={test_loss:.6f}, epochs={metrics.epochs_completed}")
                    
                except Exception as e:
                    print(f"  ✗ Failed: {e}")
                    import traceback
                    traceback.print_exc()
    
    # Save summary
    summary_file = os.path.join(exp_results_dir, "experiment_summary.json")
    with open(summary_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save results to CSV for easy analysis
    if results:
        import pandas as pd
        csv_data = []
        for result in results:
            row = {
                'experiment_id': result['experiment_id'],
                'dataset_name': result['dataset_name'],
                'test_loss': result['test_loss'],
                'best_val_loss': result['best_val_loss'],
                'epochs_completed': result['epochs_completed'],
                'training_time': result['training_time'],
                'depth': result['depth'],
                'width': result['width'],
                'learning_rate': result['learning_rate'],
                # Dataset properties
                'D': result['dataset_metadata']['D'],
                'd': result['dataset_metadata']['d'],
                'noise_level': result['dataset_metadata']['noise_level'],
                'N': result['dataset_metadata']['N'],
                # Network properties
                'network_type': result['network_config']['network_type'],
                'activation': result['network_config']['activation'],
                'norm_type': result['network_config']['norm_type'],
                'use_residual': result['network_config']['use_residual'],
                'init_scheme': result['network_config']['init_scheme'],
                # Training properties
                'optimizer_name': result['training_config']['optimizer_name'],
                'scheduler_mode': result['training_config']['scheduler_mode'],
                'batch_size': result['training_config']['batch_size'],
            }
            csv_data.append(row)
        
        df = pd.DataFrame(csv_data)
        csv_path = os.path.join(exp_results_dir, 'experiment2_results.csv')
        df.to_csv(csv_path, index=False)
        print(f"CSV results saved to: {csv_path}")
    
    print(f"\nExperiment 2 completed. Results saved to: {exp_results_dir}")
    return results


def main():
    parser = argparse.ArgumentParser(description='Run validation experiments')
    parser.add_argument('--exp', type=int, choices=[1, 2], default=1,
                       help='Experiment number to run (1 or 2)')
    parser.add_argument('--both', action='store_true',
                       help='Run both experiments sequentially')
    
    args = parser.parse_args()
    
    # Set random seed
    torch.manual_seed(42)
    np.random.seed(42)
    
    if args.both:
        print("Running both experiments...")
        results1 = run_experiment_1()
        results2 = run_experiment_2()
        print(f"\nBoth experiments completed!")
        print(f"Experiment 1: {len(results1)} results")
        print(f"Experiment 2: {len(results2)} results")
    elif args.exp == 1:
        results = run_experiment_1()
        print(f"Experiment 1 completed with {len(results)} results")
    elif args.exp == 2:
        results = run_experiment_2()
        print(f"Experiment 2 completed with {len(results)} results")


if __name__ == "__main__":
    main()
