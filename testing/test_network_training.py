#!/usr/bin/env python3
"""
Test script for neural network training components.

This script tests the network architectures and training utilities
to ensure they work correctly before running the full training pipeline.
"""

import os
import sys
import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from network_training import (
    NetworkConfig, TrainingConfig, DeepFCNet, SimpleFCNet,
    create_network, train_model, evaluate_model
)


def create_dummy_dataset(n_samples=1000, input_dim=10, noise_std=0.1):
    """Create a dummy dataset for testing."""
    # Generate random input data
    X = torch.randn(n_samples, input_dim)
    
    # Create a simple nonlinear function as target
    # y = 0.5 * x + 0.1 * sin(sum(x)) + noise
    y = 0.5 * X + 0.1 * torch.sin(X.sum(dim=1, keepdim=True)).expand(-1, input_dim)
    y = y + torch.randn(n_samples, input_dim) * noise_std
    
    dataset = TensorDataset(X, y)
    return dataset


def test_network_architectures():
    """Test different network architectures."""
    print("Testing network architectures...")
    
    input_dim, output_dim = 10, 10
    batch_size = 32
    
    # Test configurations
    configs = [
        NetworkConfig(
            network_type="DeepFCNet",
            input_dim=input_dim,
            output_dim=output_dim,
            width=64,
            depth=4,
            init_scheme="standard"
        ),
        NetworkConfig(
            network_type="DeepFCNet", 
            input_dim=input_dim,
            output_dim=output_dim,
            width=64,
            depth=4,
            init_scheme="rezero"
        ),
        NetworkConfig(
            network_type="DeepFCNet",
            input_dim=input_dim,
            output_dim=output_dim,
            width=64,
            depth=4,
            init_scheme="fixup"
        ),
        NetworkConfig(
            network_type="SimpleFCNet",
            input_dim=input_dim,
            output_dim=output_dim,
            width=64,
            depth=4
        )
    ]
    
    dummy_input = torch.randn(batch_size, input_dim)
    
    for i, config in enumerate(configs):
        print(f"\nTesting config {i+1}: {config.network_type} with {config.init_scheme if hasattr(config, 'init_scheme') else 'standard'}")
        
        try:
            # Create network
            network = create_network(config)
            print(f"✓ Network created successfully")
            
            # Test forward pass
            with torch.no_grad():
                output = network(dummy_input)
            print(f"✓ Forward pass successful, output shape: {output.shape}")
            
            # Test gradient computation
            output = network(dummy_input)  # Forward pass with gradients
            target = torch.randn_like(output)
            loss = torch.nn.MSELoss()(output, target)
            loss.backward()
            print(f"✓ Backward pass successful, loss: {loss.item():.4f}")
            
        except Exception as e:
            print(f"✗ Error: {e}")


def test_training_loop():
    """Test the training loop with a simple dataset."""
    print("\n" + "="*50)
    print("Testing training loop...")
    
    # Create dummy dataset
    dataset = create_dummy_dataset(n_samples=200, input_dim=5)
    
    # Split into train/val/test
    train_size = 120
    val_size = 40
    test_size = 40
    
    train_dataset = torch.utils.data.Subset(dataset, range(train_size))
    val_dataset = torch.utils.data.Subset(dataset, range(train_size, train_size + val_size))
    test_dataset = torch.utils.data.Subset(dataset, range(train_size + val_size, 
                                                         train_size + val_size + test_size))
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Create network
    network_config = NetworkConfig(
        network_type="DeepFCNet",
        input_dim=5,
        output_dim=5,
        width=32,
        depth=2,
        init_scheme="standard"
    )
    
    model = create_network(network_config)
    
    # Create training config
    training_config = TrainingConfig(
        max_epochs=20,  # Much shorter for testing
        batch_size=32,
        learning_rate=1e-3,
        optimizer_name='adam',
        scheduler_mode='fixed',
        early_stop_patience=5  # Reduced patience
    )
    
    try:
        # Train model
        print("Starting training...")
        metrics, best_state = train_model(
            model, train_loader, val_loader, training_config,
            device='cpu', verbose=False
        )
        
        print(f"✓ Training completed successfully")
        print(f"  - Epochs completed: {metrics.epochs_completed}")
        print(f"  - Best validation loss: {metrics.best_val_loss:.6f}")
        print(f"  - Training time: {metrics.training_time:.2f}s")
        
        # Test model
        test_loss = evaluate_model(model, test_loader, device='cpu')
        print(f"  - Test loss: {test_loss:.6f}")
        
    except Exception as e:
        print(f"✗ Training failed: {e}")


def test_initialization_schemes():
    """Test different initialization schemes with deep networks."""
    print("\n" + "="*50)
    print("Testing initialization schemes with deep networks...")
    
    input_dim = 10
    batch_size = 16
    dummy_input = torch.randn(batch_size, input_dim)
    
    # Test with increasing depth
    depths = [1, 4, 8, 16]
    schemes = ['standard', 'rezero', 'fixup']
    
    for depth in depths:
        print(f"\nTesting depth {depth}:")
        
        for scheme in schemes:
            try:
                config = NetworkConfig(
                    network_type="DeepFCNet",
                    input_dim=input_dim,
                    output_dim=input_dim,
                    width=64,
                    depth=depth,
                    init_scheme=scheme
                )
                
                model = create_network(config)
                
                # Forward pass
                with torch.no_grad():
                    output = model(dummy_input)
                
                # Check for reasonable output magnitudes
                output_std = output.std().item()
                
                print(f"  {scheme:>8}: output std = {output_std:.4f}")
                
                if output_std > 1e6 or output_std < 1e-6:
                    print(f"    ⚠ Warning: potentially unstable initialization")
                else:
                    print(f"    ✓ Stable initialization")
                
            except Exception as e:
                print(f"  {scheme:>8}: ✗ Error - {e}")


def main():
    print("Neural Network Training Component Tests")
    print("=" * 50)
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Run tests
    test_network_architectures()
    test_training_loop()
    test_initialization_schemes()
    
    print("\n" + "="*50)
    print("All tests completed!")


if __name__ == "__main__":
    main()
