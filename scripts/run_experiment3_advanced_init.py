#!/usr/bin/env python3
"""
Experiment 3: ReZero and Fixup Initialization Study

This experiment specifically tests ReZero and Fixup initialization schemes
against standard initialization on a single dataset, with various network configurations.
"""

import os
import sys
import json
import time
import argparse
import torch
import numpy as np
import pandas as pd
from datetime import datetime
from torch.utils.data import DataLoader, Subset

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_generation.dataset_utils import load_dataset_metadata, create_manifold_dataset
from network_training import (
    NetworkConfig, TrainingConfig, ExperimentTracker,
    create_network, train_model, evaluate_model, 
    create_train_val_split, apply_lr_scaling
)


def load_dataset_from_directory(dataset_dir):
    """Load dataset from a directory containing .pt files."""
    
    train_data_path = os.path.join(dataset_dir, 'train_data.pt')
    train_clean_path = os.path.join(dataset_dir, 'train_clean.pt')
    test_data_path = os.path.join(dataset_dir, 'test_data.pt')
    test_clean_path = os.path.join(dataset_dir, 'test_clean.pt')
    metadata_path = os.path.join(dataset_dir, 'metadata.json')
    
    # Load data
    train_data = torch.load(train_data_path)
    train_clean = torch.load(train_clean_path)
    test_data = torch.load(test_data_path)
    test_clean = torch.load(test_clean_path)
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    # Create datasets
    from torch.utils.data import TensorDataset
    train_dataset = TensorDataset(train_data, train_clean)
    test_dataset = TensorDataset(test_data, test_clean)
    
    return train_dataset, test_dataset, metadata


def run_experiment_3():
    """
    Experiment 3: Advanced initialization schemes (ReZero, Fixup) vs Standard
    Dataset: Single dataset (dataset0)
    Networks: Various depths with ReZero/Fixup initialization + residual connections
    """
    print("="*60)
    print("EXPERIMENT 3: Advanced Initialization Study")
    print("="*60)
    
    # Setup
    data_dir = "/home/tim/python_projects/nn_manifold_denoising/data/data_250914_0100_01"
    results_dir = "results/nn_train_exp3_advanced_init"
    
    # Create results directory with timestamp
    timestamp = time.strftime("%Y%m%d_%H%M")
    tb_num = f"tb{timestamp}"
    exp_results_dir = os.path.join(results_dir, tb_num)
    os.makedirs(exp_results_dir, exist_ok=True)
    
    # Find dataset
    dataset_dir = os.path.join(data_dir, "dataset0")
    if not os.path.exists(dataset_dir):
        print("Dataset not found!")
        return []
    
    print(f"Using dataset: {dataset_dir}")
    
    # Load dataset once
    train_dataset, test_dataset, metadata = load_dataset_from_directory(dataset_dir)
    
    # Parameter grids - Focus on deeper networks where init schemes matter
    depths = [1, 2, 4, 8, 16]  # Include deeper networks
    widths = [200, 400, 800]   # Focus on wider networks
    init_schemes = ['standard', 'rezero', 'fixup']  # Compare all three
    learning_rates = [1e-3, 1e-4]  # Two learning rates
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Run experiments
    results = []
    total_experiments = len(depths) * len(widths) * len(init_schemes) * len(learning_rates)
    completed = 0
    
    print(f"Running {total_experiments} experiments...")
    
    for depth in depths:
        for width in widths:
            for init_scheme in init_schemes:
                for lr in learning_rates:
                    completed += 1
                    
                    # Skip certain combinations
                    if depth == 1 and init_scheme in ['rezero', 'fixup']:
                        print(f"Skipping depth=1 with {init_scheme} (not beneficial)")
                        continue
                    
                    experiment_id = f"dataset0_d{depth}_w{width}_{init_scheme}_lr{lr:.0e}"
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
                            norm_type=None,  # Keep clean for init scheme comparison
                            dropout=0.0,
                            use_residual=True,  # Always use residual connections
                            init_scheme=init_scheme
                        )
                        
                        # Training configuration
                        training_config = TrainingConfig(
                            max_epochs=2000,
                            batch_size=1024,
                            learning_rate=lr,
                            optimizer_name='adam',
                            scheduler_mode='fixed',
                            weight_decay=0.0,
                            early_stop_patience=100,  # More patience for deeper networks
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
                        start_time = time.time()
                        metrics, _ = train_model(model, train_loader, val_loader, training_config, device, verbose=False)
                        training_time = time.time() - start_time
                        
                        # Test model
                        test_loss = evaluate_model(model, test_loader, device)
                        
                        # Save results
                        result = {
                            'experiment_id': experiment_id,
                            'dataset_name': 'dataset0',
                            'dataset_metadata': metadata,
                            'network_config': network_config.to_dict(),
                            'training_config': training_config.to_dict(),
                            'test_loss': test_loss,
                            'best_val_loss': metrics.best_val_loss,
                            'epochs_completed': metrics.epochs_completed,
                            'training_time': training_time,
                            'depth': depth,
                            'width': width,
                            'init_scheme': init_scheme,
                            'learning_rate': lr,
                            'metrics': metrics.to_dict()
                        }
                        
                        results.append(result)
                        
                        # Save individual result
                        result_file = os.path.join(exp_results_dir, f"{experiment_id}.json")
                        with open(result_file, 'w') as f:
                            json.dump(result, f, indent=2)
                        
                        print(f"  ✓ Success: test_loss={test_loss:.6f}, epochs={metrics.epochs_completed}, time={training_time:.1f}s")
                        
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
                'init_scheme': result['init_scheme'],
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
                # Training properties
                'optimizer_name': result['training_config']['optimizer_name'],
                'scheduler_mode': result['training_config']['scheduler_mode'],
                'batch_size': result['training_config']['batch_size'],
            }
            csv_data.append(row)
        
        df = pd.DataFrame(csv_data)
        csv_path = os.path.join(exp_results_dir, 'experiment3_results.csv')
        df.to_csv(csv_path, index=False)
        print(f"CSV results saved to: {csv_path}")
    
    print(f"\nExperiment 3 completed. Results saved to: {exp_results_dir}")
    print(f"Total successful experiments: {len(results)}")
    
    # Print summary statistics
    if results:
        df = pd.DataFrame([{
            'init_scheme': r['init_scheme'],
            'depth': r['depth'],
            'test_loss': r['test_loss']
        } for r in results])
        
        print("\n=== SUMMARY BY INITIALIZATION SCHEME ===")
        for init_scheme in df['init_scheme'].unique():
            subset = df[df['init_scheme'] == init_scheme]
            print(f"{init_scheme.upper()}: mean_loss={subset['test_loss'].mean():.6f}, "
                  f"std={subset['test_loss'].std():.6f}, "
                  f"min={subset['test_loss'].min():.6f}")
        
        print("\n=== BEST RESULTS BY DEPTH ===")
        for depth in sorted(df['depth'].unique()):
            subset = df[df['depth'] == depth]
            best_row = subset.loc[subset['test_loss'].idxmin()]
            print(f"Depth {depth}: best_loss={best_row['test_loss']:.6f} "
                  f"(init_scheme={best_row['init_scheme']})")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Run advanced initialization experiment')
    
    args = parser.parse_args()
    
    # Set random seed
    torch.manual_seed(42)
    np.random.seed(42)
    
    results = run_experiment_3()
    print(f"Experiment 3 completed with {len(results)} successful results")


if __name__ == "__main__":
    main()
