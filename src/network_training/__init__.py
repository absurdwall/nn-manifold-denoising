"""
Network Training Module

This module handles neural network training for manifold denoising.
Step 4 of the pipeline: Train neural networks on generated datasets.

Key components:
- Network architectures
- Training loops
- Loss functions
- Model evaluation
- Training result storage
"""

from .networks import (
    DeepFCNet,
    SimpleFCNet,
    NetworkConfig,
    create_network,
    get_activation
)

from .training import (
    TrainingConfig,
    TrainingMetrics,
    ExperimentTracker,
    train_model,
    evaluate_model,
    create_train_val_split,
    save_training_results,
    get_optimizer,
    get_scheduler,
    apply_lr_scaling
)

__all__ = [
    # Networks
    'DeepFCNet',
    'SimpleFCNet', 
    'NetworkConfig',
    'create_network',
    'get_activation',
    
    # Training
    'TrainingConfig',
    'TrainingMetrics',
    'ExperimentTracker',
    'train_model',
    'evaluate_model',
    'create_train_val_split',
    'save_training_results',
    'get_optimizer',
    'get_scheduler',
    'apply_lr_scaling'
]
