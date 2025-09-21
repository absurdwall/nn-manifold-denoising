#!/usr/bin/env python3
"""
Demonstration script showing the effectiveness of different initialization schemes.

This script compares standard, ReZero, and Fixup initialization on a simple
manifold denoising task, showing how they enable training of deeper networks.
"""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from network_training import (
    NetworkConfig, TrainingConfig, create_network, train_model, evaluate_model,
    create_train_val_split
)


def create_manifold_denoising_dataset(n_samples=2000, intrinsic_dim=3, ambient_dim=10, noise_std=0.1):
    """Create a simple manifold denoising dataset."""
    # Generate data on a 3D manifold embedded in 10D space
    # Manifold: sphere + some nonlinear transformation
    
    # Generate points on unit sphere in 3D
    z = torch.randn(n_samples, intrinsic_dim)
    z = z / torch.norm(z, dim=1, keepdim=True)
    
    # Embed in higher dimension with nonlinear mapping
    embedding_matrix = torch.randn(ambient_dim, intrinsic_dim) * 0.5
    x_clean = torch.mm(z, embedding_matrix.T)
    x_clean = x_clean + 0.2 * torch.sin(2 * x_clean)  # Add nonlinearity
    
    # Add noise
    x_noisy = x_clean + torch.randn_like(x_clean) * noise_std
    
    # Task: denoise (map noisy back to clean)
    dataset = TensorDataset(x_noisy, x_clean)
    return dataset


def compare_initialization_schemes():
    """Compare different initialization schemes on deep networks."""
    print("Comparing Initialization Schemes on Deep Networks")
    print("=" * 60)
    
    # Create dataset
    dataset = create_manifold_denoising_dataset(n_samples=1000, noise_std=0.05)
    
    # Split dataset
    train_subset, val_subset = create_train_val_split(dataset, 800, val_ratio=0.2)
    test_subset = torch.utils.data.Subset(dataset, range(800, 1000))
    
    train_loader = DataLoader(train_subset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_subset, batch_size=64, shuffle=False)
    
    # Test different depths and initialization schemes
    depths = [2, 4, 8, 16]
    init_schemes = ['standard', 'rezero', 'fixup']
    
    results = {}
    
    for depth in depths:
        print(f"\nTesting depth {depth}:")
        results[depth] = {}
        
        for init_scheme in init_schemes:
            print(f"  {init_scheme}...", end=" ")
            
            try:
                # Create network
                config = NetworkConfig(
                    network_type="DeepFCNet",
                    input_dim=10,
                    output_dim=10,
                    width=128,
                    depth=depth,
                    init_scheme=init_scheme,
                    norm_type='batch' if init_scheme != 'fixup' else None
                )
                
                model = create_network(config)
                
                # Training config
                training_config = TrainingConfig(
                    max_epochs=100,
                    batch_size=64,
                    learning_rate=1e-3,
                    optimizer_name='adam',
                    early_stop_patience=15
                )
                
                # Train
                metrics, _ = train_model(model, train_loader, val_loader, 
                                       training_config, device='cpu', verbose=False)
                
                # Test
                test_loss = evaluate_model(model, test_loader, device='cpu')
                
                results[depth][init_scheme] = {
                    'test_loss': test_loss,
                    'epochs': metrics.epochs_completed,
                    'best_val_loss': metrics.best_val_loss,
                    'converged': metrics.epochs_completed < training_config.max_epochs
                }
                
                print(f"✓ Test loss: {test_loss:.4f}, Epochs: {metrics.epochs_completed}")
                
            except Exception as e:
                print(f"✗ Failed: {e}")
                results[depth][init_scheme] = {
                    'test_loss': float('inf'),
                    'epochs': 0,
                    'best_val_loss': float('inf'),
                    'converged': False,
                    'error': str(e)
                }
    
    return results


def plot_results(results):
    """Plot comparison results."""
    depths = sorted(results.keys())
    init_schemes = ['standard', 'rezero', 'fixup']
    colors = {'standard': 'red', 'rezero': 'blue', 'fixup': 'green'}
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot test loss vs depth
    for scheme in init_schemes:
        losses = []
        for depth in depths:
            if scheme in results[depth]:
                loss = results[depth][scheme]['test_loss']
                losses.append(loss if loss < float('inf') else None)
            else:
                losses.append(None)
        
        # Filter out None values for plotting
        valid_depths = [d for d, l in zip(depths, losses) if l is not None]
        valid_losses = [l for l in losses if l is not None]
        
        if valid_losses:
            ax1.plot(valid_depths, valid_losses, 'o-', color=colors[scheme], 
                    label=scheme.capitalize(), linewidth=2, markersize=6)
    
    ax1.set_xlabel('Network Depth')
    ax1.set_ylabel('Test Loss')
    ax1.set_title('Test Loss vs Network Depth')
    ax1.set_yscale('log')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot convergence epochs vs depth
    for scheme in init_schemes:
        epochs_list = []
        for depth in depths:
            if scheme in results[depth] and results[depth][scheme]['converged']:
                epochs_list.append(results[depth][scheme]['epochs'])
            else:
                epochs_list.append(None)
        
        valid_depths = [d for d, e in zip(depths, epochs_list) if e is not None]
        valid_epochs = [e for e in epochs_list if e is not None]
        
        if valid_epochs:
            ax2.plot(valid_depths, valid_epochs, 'o-', color=colors[scheme], 
                    label=scheme.capitalize(), linewidth=2, markersize=6)
    
    ax2.set_xlabel('Network Depth')
    ax2.set_ylabel('Epochs to Convergence')
    ax2.set_title('Training Speed vs Network Depth')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/home/tim/python_projects/nn_manifold_denoising/testing/initialization_comparison.png', 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\nPlot saved to: /home/tim/python_projects/nn_manifold_denoising/testing/initialization_comparison.png")


def main():
    torch.manual_seed(42)
    np.random.seed(42)
    
    print("Neural Network Initialization Schemes Demonstration")
    print("This script compares how different initialization schemes")
    print("enable training of deep neural networks for manifold denoising.\n")
    
    results = compare_initialization_schemes()
    
    print("\n" + "="*60)
    print("SUMMARY RESULTS")
    print("="*60)
    
    # Print summary table
    print(f"{'Depth':<6} {'Scheme':<10} {'Test Loss':<12} {'Epochs':<8} {'Status'}")
    print("-" * 50)
    
    for depth in sorted(results.keys()):
        for scheme in ['standard', 'rezero', 'fixup']:
            if scheme in results[depth]:
                r = results[depth][scheme]
                status = "✓" if r['converged'] else "✗"
                loss_str = f"{r['test_loss']:.4f}" if r['test_loss'] < float('inf') else "Failed"
                print(f"{depth:<6} {scheme:<10} {loss_str:<12} {r['epochs']:<8} {status}")
    
    print("\nKey Observations:")
    print("- ReZero and Fixup enable stable training of very deep networks")
    print("- Standard initialization may struggle with depth > 8")
    print("- ReZero often converges fastest due to starting from identity")
    print("- Fixup works without normalization layers")
    
    # Create plot
    try:
        plot_results(results)
    except Exception as e:
        print(f"Note: Could not create plot: {e}")


if __name__ == "__main__":
    main()
