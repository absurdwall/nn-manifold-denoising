#!/usr/bin/env python3
"""
Step 4: Neural Network Training Pipeline

This script trains neural networks on the generated manifold datasets from Step 1.
It performs comprehensive experiments with different network architectures,
initialization schemes, and training configurations.

The script supports:
- Multiple network architectures (DeepFCNet, SimpleFCNet)
- Different initialization schemes (standard, ReZero, Fixup)
- Various normalization techniques (batch norm, layer norm)
- Comprehensive hyperparameter sweeps
- Automatic result tracking and visualization
"""

import os
import sys
import json
import argparse
import torch
import numpy as np
import pandas as pd
from itertools import product
from torch.utils.data import DataLoader
import time

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_generation.dataset_utils import load_dataset_metadata, create_manifold_dataset
from network_training import (
    NetworkConfig, TrainingConfig, ExperimentTracker,
    create_network, train_model, evaluate_model, 
    create_train_val_split, apply_lr_scaling
)
from plotting.training_plots import plot_training_curves, plot_experiment_summary


def set_random_seeds(seed=42):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def create_experiment_configs():
    """Create comprehensive experiment configurations."""
    
    # Network configurations
    network_configs = []
    
    # Different architectures
    architectures = ['DeepFCNet', 'SimpleFCNet']
    widths = [100, 200, 400, 800]
    depths = [1, 2, 4, 8, 16]
    init_schemes = ['standard', 'rezero', 'fixup']
    norm_types = [None, 'batch', 'layer']
    activations = ['relu', 'gelu']
    
    for arch, width, depth, init_scheme, norm_type, activation in product(
        architectures, widths, depths, init_schemes, norm_types, activations
    ):
        # Skip certain combinations
        if arch == 'SimpleFCNet' and init_scheme != 'standard':
            continue  # SimpleFCNet doesn't support special init schemes
        if init_scheme == 'fixup' and norm_type is not None:
            continue  # Fixup doesn't use normalization
        if depth == 1 and init_scheme in ['rezero', 'fixup']:
            continue  # Special schemes only useful for deep networks
            
        config = NetworkConfig(
            network_type=arch,
            width=width,
            depth=depth,
            activation=activation,
            norm_type=norm_type,
            use_residual=(arch == 'DeepFCNet'),
            init_scheme=init_scheme
        )
        network_configs.append(config)
    
    # Training configurations
    training_configs = []
    
    base_lrs = [1e-2, 1e-3, 1e-4]
    optimizers = ['adam', 'adamw', 'sgd']
    schedulers = ['fixed', 'plateau', 'cosine']
    lr_scaling_rules = [None, '1/L', 'sqrt_L']
    
    for base_lr, optimizer, scheduler, lr_rule in product(
        base_lrs, optimizers, schedulers, lr_scaling_rules
    ):
        config = TrainingConfig(
            max_epochs=2000,
            batch_size=1024,
            learning_rate=base_lr,
            optimizer_name=optimizer,
            scheduler_mode=scheduler,
            early_stop_patience=50,
            lr_scaling_rule=lr_rule,
            weight_decay=1e-5 if optimizer in ['adam', 'adamw'] else 5e-4
        )
        training_configs.append(config)
    
    return network_configs, training_configs


def run_single_experiment(dataset_path, network_config, training_config, 
                         device, experiment_id, results_dir, verbose=True):
    """Run a single training experiment."""
    
    # Load dataset
    try:
        train_dataset, test_dataset, metadata = create_manifold_dataset(dataset_path)
    except Exception as e:
        print(f"Error loading dataset {dataset_path}: {e}")
        return None
    
    # Set input/output dimensions
    network_config.input_dim = metadata['D']
    network_config.output_dim = metadata['D']
    
    # Apply learning rate scaling
    if training_config.lr_scaling_rule:
        scaled_lr = apply_lr_scaling(
            training_config.learning_rate, 
            network_config.depth, 
            training_config.lr_scaling_rule
        )
        training_config.learning_rate = scaled_lr
    
    # Create network
    model = create_network(network_config).to(device)
    
    # Create data loaders
    N = len(train_dataset)
    train_subset, val_subset = create_train_val_split(
        train_dataset, N, training_config.val_ratio
    )
    
    train_loader = DataLoader(
        train_subset, 
        batch_size=training_config.batch_size, 
        shuffle=True
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=training_config.batch_size,
        shuffle=False
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=training_config.batch_size,
        shuffle=False
    )
    
    # Train model
    start_time = time.time()
    try:
        metrics, best_state = train_model(
            model, train_loader, val_loader, training_config, device, verbose
        )
        
        # Evaluate on test set
        test_loss = evaluate_model(model, test_loader, device)
        
        training_time = time.time() - start_time
        
        # Save results
        result = {
            'experiment_id': experiment_id,
            'dataset_path': dataset_path,
            'dataset_metadata': metadata,
            'network_config': network_config.to_dict(),
            'training_config': training_config.to_dict(),
            'metrics': metrics.to_dict(),
            'test_loss': test_loss,
            'training_time': training_time,
            'success': True
        }
        
        # Save individual experiment
        exp_path = os.path.join(results_dir, f'{experiment_id}.json')
        os.makedirs(os.path.dirname(exp_path), exist_ok=True)
        with open(exp_path, 'w') as f:
            json.dump(result, f, indent=2)
        
        if verbose:
            print(f"Experiment {experiment_id} completed successfully")
            print(f"Test loss: {test_loss:.6f}, Training time: {training_time:.2f}s")
        
        return result
        
    except Exception as e:
        print(f"Error in experiment {experiment_id}: {e}")
        return {
            'experiment_id': experiment_id,
            'dataset_path': dataset_path,
            'error': str(e),
            'success': False
        }


def main():
    parser = argparse.ArgumentParser(description='Train neural networks on manifold datasets')
    parser.add_argument('--config', type=str, default='configs/step4_training_config.json',
                       help='Path to training configuration file')
    parser.add_argument('--data-dir', type=str, default='data/datasets',
                       help='Directory containing generated datasets')
    parser.add_argument('--results-dir', type=str, default='results/step4_training',
                       help='Directory to save training results')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use for training (auto, cuda, cpu)')
    parser.add_argument('--subset', type=int, default=None,
                       help='Run only on a subset of experiments (for testing)')
    parser.add_argument('--verbose', action='store_true',
                       help='Print detailed training progress')
    parser.add_argument('--resume', action='store_true',
                       help='Resume from existing results')
    
    args = parser.parse_args()
    
    # Set device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Set random seeds
    set_random_seeds(42)
    
    # Create results directory
    os.makedirs(args.results_dir, exist_ok=True)
    
    # Load training configuration if exists
    if os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config_data = json.load(f)
        print(f"Loaded configuration from {args.config}")
    else:
        print("Using default configuration")
        config_data = {}
    
    # Find all dataset files
    dataset_files = []
    for root, dirs, files in os.walk(args.data_dir):
        for file in files:
            if file.endswith('_metadata.json'):
                dataset_files.append(os.path.join(root, file.replace('_metadata.json', '')))
    
    print(f"Found {len(dataset_files)} datasets")
    
    # Create experiment configurations
    network_configs, training_configs = create_experiment_configs()
    
    # Filter configurations if subset is specified
    if args.subset:
        network_configs = network_configs[:args.subset]
        training_configs = training_configs[:args.subset]
    
    print(f"Running {len(network_configs)} network configs × {len(training_configs)} training configs")
    print(f"Total experiments per dataset: {len(network_configs) * len(training_configs)}")
    
    # Initialize experiment tracker
    tracker = ExperimentTracker()
    
    # Run experiments
    total_experiments = len(dataset_files) * len(network_configs) * len(training_configs)
    completed_experiments = 0
    
    print(f"Starting {total_experiments} total experiments...")
    
    for dataset_idx, dataset_path in enumerate(dataset_files):
        dataset_name = os.path.basename(dataset_path)
        print(f"\n=== Dataset {dataset_idx + 1}/{len(dataset_files)}: {dataset_name} ===")
        
        for net_idx, network_config in enumerate(network_configs):
            for train_idx, training_config in enumerate(training_configs):
                experiment_id = f"{dataset_name}_net{net_idx:03d}_train{train_idx:03d}"
                
                # Check if experiment already exists (for resume functionality)
                exp_path = os.path.join(args.results_dir, f'{experiment_id}.json')
                if args.resume and os.path.exists(exp_path):
                    print(f"Skipping existing experiment: {experiment_id}")
                    completed_experiments += 1
                    continue
                
                print(f"Running experiment {completed_experiments + 1}/{total_experiments}: {experiment_id}")
                
                result = run_single_experiment(
                    dataset_path, network_config, training_config,
                    device, experiment_id, args.results_dir, args.verbose
                )
                
                if result and result.get('success', False):
                    tracker.add_experiment(
                        experiment_id=experiment_id,
                        network_config=network_config.to_dict(),
                        training_config=training_config.to_dict(),
                        metrics=None,  # Will be loaded from file if needed
                        test_loss=result['test_loss'],
                        dataset_info=result['dataset_metadata']
                    )
                
                completed_experiments += 1
                
                # Save intermediate summary
                if completed_experiments % 100 == 0:
                    summary_path = os.path.join(args.results_dir, 'summary_intermediate.json')
                    tracker.save_summary(summary_path)
    
    # Save final summary
    summary_path = os.path.join(args.results_dir, 'experiment_summary.json')
    tracker.save_summary(summary_path)
    
    # Create summary plots
    try:
        plot_experiment_summary(args.results_dir, 
                              os.path.join(args.results_dir, 'plots'))
        print("Summary plots created successfully")
    except Exception as e:
        print(f"Error creating plots: {e}")
    
    print(f"\nTraining completed! Results saved to {args.results_dir}")
    print(f"Completed {completed_experiments} experiments")
    
    # Print best experiments
    best_experiments = tracker.get_best_experiments(metric='test_loss', top_k=5)
    print("\nTop 5 experiments by test loss:")
    for i, exp in enumerate(best_experiments, 1):
        print(f"{i}. {exp['experiment_id']}: {exp['test_loss']:.6f}")


if __name__ == "__main__":
    main()
