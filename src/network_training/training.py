"""
Training utilities for neural network experiments.

This module provides functions for training neural networks on manifold
denoising tasks with various optimization strategies, learning rate scheduling,
and early stopping.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR, CosineAnnealingLR
import numpy as np
import time
import json
import os
from typing import Dict, List, Tuple, Optional, Any


class TrainingConfig:
    """Configuration class for training hyperparameters."""
    
    def __init__(self,
                 max_epochs=1000,
                 batch_size=256,
                 learning_rate=1e-3,
                 optimizer_name='adam',
                 scheduler_mode='fixed',
                 scheduler_patience=10,
                 scheduler_factor=0.1,
                 scheduler_step_size=50,
                 weight_decay=0.0,
                 momentum=0.9,
                 early_stop_patience=20,
                 val_ratio=0.2,
                 gradient_clip=None,
                 lr_scaling_rule=None):
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.optimizer_name = optimizer_name
        self.scheduler_mode = scheduler_mode
        self.scheduler_patience = scheduler_patience
        self.scheduler_factor = scheduler_factor
        self.scheduler_step_size = scheduler_step_size
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.early_stop_patience = early_stop_patience
        self.val_ratio = val_ratio
        self.gradient_clip = gradient_clip
        self.lr_scaling_rule = lr_scaling_rule
    
    def to_dict(self):
        return self.__dict__
    
    @classmethod
    def from_dict(cls, config_dict):
        return cls(**config_dict)


def get_optimizer(model, config: TrainingConfig):
    """Create optimizer based on configuration."""
    optimizer_name = config.optimizer_name.lower()
    lr = config.learning_rate
    weight_decay = config.weight_decay
    
    if optimizer_name == 'adam':
        return optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == 'adamw':
        return optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == 'sgd':
        return optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, 
                        momentum=config.momentum)
    elif optimizer_name == 'rmsprop':
        return optim.RMSprop(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")


def get_scheduler(optimizer, config: TrainingConfig):
    """Create learning rate scheduler based on configuration."""
    mode = config.scheduler_mode.lower()
    
    if mode == 'fixed':
        return None
    elif mode == 'plateau':
        return ReduceLROnPlateau(optimizer, 'min', 
                               patience=config.scheduler_patience,
                               factor=config.scheduler_factor)
    elif mode == 'step':
        return StepLR(optimizer, step_size=config.scheduler_step_size, 
                     gamma=config.scheduler_factor)
    elif mode == 'cosine':
        return CosineAnnealingLR(optimizer, T_max=config.max_epochs)
    else:
        raise ValueError(f"Unknown scheduler mode: {mode}")


def apply_lr_scaling(base_lr: float, network_depth: int, rule: str = None) -> float:
    """Apply learning rate scaling based on network depth."""
    if rule is None:
        return base_lr
    elif rule == '1/L':
        return base_lr / network_depth
    elif rule == '1/L2':
        return base_lr / (network_depth ** 2)
    elif rule == 'sqrt_L':
        return base_lr / np.sqrt(network_depth)
    else:
        return base_lr


def create_train_val_split(dataset, N: int, val_ratio: float = 0.2):
    """
    Split dataset into training and validation subsets.
    
    Args:
        dataset: The full dataset to split
        N: The number of samples to use
        val_ratio: The proportion of data to use as validation
        
    Returns:
        train_subset, val_subset: Training and validation subsets
    """
    train_subset = Subset(dataset, indices=range(N))
    val_split = int((1 - val_ratio) * N)
    val_subset = Subset(train_subset, indices=range(val_split, len(train_subset)))
    train_subset = Subset(train_subset, indices=range(val_split))
    return train_subset, val_subset


def move_to_device(data, device):
    """Move data to specified device."""
    if isinstance(data, (list, tuple)):
        return [move_to_device(x, device) for x in data]
    return data.to(device)


class TrainingMetrics:
    """Class to track training metrics."""
    
    def __init__(self):
        self.train_losses = []
        self.val_losses = []
        self.learning_rates = []
        self.epochs_completed = 0
        self.best_val_loss = float('inf')
        self.best_epoch = 0
        self.training_time = 0.0
    
    def update(self, train_loss: float, val_loss: float, lr: float):
        """Update metrics for current epoch."""
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        self.learning_rates.append(lr)
        self.epochs_completed += 1
        
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.best_epoch = self.epochs_completed
    
    def to_dict(self):
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'learning_rates': self.learning_rates,
            'epochs_completed': self.epochs_completed,
            'best_val_loss': self.best_val_loss,
            'best_epoch': self.best_epoch,
            'training_time': self.training_time
        }


def train_model(model, train_loader, val_loader, config: TrainingConfig, 
                device='cuda', verbose=True) -> Tuple[TrainingMetrics, dict]:
    """
    Train a neural network model.
    
    Args:
        model: The neural network model to train
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        config: Training configuration
        device: Device to train on ('cuda' or 'cpu')
        verbose: Whether to print training progress
        
    Returns:
        metrics: TrainingMetrics object with training history
        best_state: Best model state dict
    """
    criterion = nn.MSELoss()
    optimizer = get_optimizer(model, config)
    scheduler = get_scheduler(optimizer, config)
    
    metrics = TrainingMetrics()
    best_state = None
    no_improvement_epochs = 0
    
    start_time = time.time()
    
    for epoch in range(config.max_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = move_to_device((inputs, targets), device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            
            # Gradient clipping if specified
            if config.gradient_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
            
            optimizer.step()
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = move_to_device((inputs, targets), device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        
        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        
        # Update metrics
        metrics.update(avg_train_loss, avg_val_loss, current_lr)
        
        # Learning rate scheduling
        if scheduler is not None:
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(avg_val_loss)
            else:
                scheduler.step()
        
        # Early stopping and best model tracking
        if avg_val_loss < metrics.best_val_loss:
            best_state = model.state_dict().copy()
            no_improvement_epochs = 0
        else:
            no_improvement_epochs += 1
        
        if no_improvement_epochs >= config.early_stop_patience:
            if verbose:
                print(f"Early stopping at epoch {epoch + 1}")
            break
        
        # Verbose output
        if verbose and (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{config.max_epochs}, "
                  f"Train Loss: {avg_train_loss:.6f}, "
                  f"Val Loss: {avg_val_loss:.6f}, "
                  f"LR: {current_lr:.2e}")
    
    metrics.training_time = time.time() - start_time
    
    # Load best model state
    if best_state is not None:
        model.load_state_dict(best_state)
    
    return metrics, best_state


def evaluate_model(model, test_loader, device='cuda') -> float:
    """
    Evaluate model on test data.
    
    Args:
        model: The trained model
        test_loader: DataLoader for test data
        device: Device to evaluate on
        
    Returns:
        test_loss: Average test loss
    """
    criterion = nn.MSELoss()
    model.eval()
    test_loss = 0.0
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = move_to_device((inputs, targets), device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
    
    return test_loss / len(test_loader)


def save_training_results(metrics: TrainingMetrics, network_config: dict, 
                         training_config: dict, test_loss: float, 
                         save_path: str):
    """Save training results to JSON file."""
    results = {
        'network_config': network_config,
        'training_config': training_config,
        'metrics': metrics.to_dict(),
        'test_loss': test_loss
    }
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        json.dump(results, f, indent=2)


class ExperimentTracker:
    """Class to track multiple experiments and their results."""
    
    def __init__(self):
        self.experiments = []
    
    def add_experiment(self, experiment_id: str, network_config: dict, 
                      training_config: dict, metrics: TrainingMetrics, 
                      test_loss: float, dataset_info: dict = None):
        """Add an experiment result."""
        experiment = {
            'experiment_id': experiment_id,
            'network_config': network_config,
            'training_config': training_config,
            'metrics': metrics.to_dict(),
            'test_loss': test_loss,
            'dataset_info': dataset_info or {}
        }
        self.experiments.append(experiment)
    
    def save_summary(self, save_path: str):
        """Save experiment summary to JSON file."""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w') as f:
            json.dump(self.experiments, f, indent=2)
    
    def get_best_experiments(self, metric='test_loss', top_k=5):
        """Get top-k experiments by specified metric."""
        sorted_experiments = sorted(self.experiments, 
                                  key=lambda x: x[metric])
        return sorted_experiments[:top_k]
