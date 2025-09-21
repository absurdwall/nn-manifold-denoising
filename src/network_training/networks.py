"""
Neural network architectures for manifold denoising experiments.

This module implements various fully connected neural network architectures
with support for residual connections, normalization, dropout, and different
initialization schemes including ReZero and Fixup.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def get_activation(name):
    """Get activation function by name."""
    if isinstance(name, str):
        act = name.lower()
        if act == 'relu':
            return nn.ReLU
        elif act == 'gelu':
            return nn.GELU
        elif act == 'swish':
            return nn.SiLU  # SiLU is PyTorch's Swish
        elif act == 'mish':
            return nn.Mish
        elif act == 'leaky_relu':
            return nn.LeakyReLU
        elif act == 'tanh':
            return nn.Tanh
        elif act == 'sigmoid':
            return nn.Sigmoid
        else:
            raise ValueError(f"Unknown activation: {name}")
    return name  # Assume it's a class or callable


class ResidualBlock(nn.Module):
    """
    Width-preserving 2-layer residual block with optional normalization.
    
    Supports different initialization schemes:
    - standard: Normal Kaiming initialization
    - rezero: Learnable scalar gate initialized at 0
    - fixup: Careful weight scaling without normalization
    """
    
    def __init__(self, width, norm_type=None, activation='relu', dropout=0.0,
                 init_scheme="standard", total_blocks=None):
        super().__init__()
        Act = get_activation(activation)
        self.init_scheme = init_scheme.lower()
        self.total_blocks = total_blocks  # needed for Fixup scaling

        # Build the residual branch
        layers = [nn.Linear(width, width)]
        if norm_type == 'batch':
            layers.append(nn.BatchNorm1d(width))
        elif norm_type == 'layer':
            layers.append(nn.LayerNorm(width))
        
        layers.extend([Act(), nn.Linear(width, width)])
        
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
            
        self.branch = nn.Sequential(*layers)

        # ReZero: learnable gate α initialized at 0
        if self.init_scheme == "rezero":
            self.alpha = nn.Parameter(torch.zeros(1))

        # Mark the last linear layer for Fixup initialization
        if hasattr(self.branch[-1], 'weight'):
            self.branch[-1].is_last_in_block = True
        else:
            # If last layer is dropout, mark the linear layer before it
            for layer in reversed(self.branch):
                if hasattr(layer, 'weight'):
                    layer.is_last_in_block = True
                    break

    def forward(self, x):
        if self.init_scheme == "rezero":
            return x + self.alpha * self.branch(x)
        else:  # "standard" or "fixup"
            return x + self.branch(x)


class DeepFCNet(nn.Module):
    """
    Deep fully connected neural network with residual connections.
    
    Features:
    - Configurable depth and width
    - Optional residual connections with different initialization schemes
    - Optional normalization (batch norm, layer norm)
    - Optional dropout
    - Multiple activation functions
    """
    
    def __init__(self, input_dim, output_dim, width, depth,
                 norm_type=None, activation='relu', dropout=0.0,
                 init_scheme="standard", use_residual=True):
        super().__init__()
        self.init_scheme = init_scheme.lower()
        self.use_residual = use_residual
        self.depth = depth
        
        # Input projection
        self.input = nn.Linear(input_dim, width)
        
        # Hidden layers
        if use_residual and depth > 0:
            self.layers = nn.ModuleList([
                ResidualBlock(width, norm_type, activation, dropout,
                              init_scheme=self.init_scheme,
                              total_blocks=depth)
                for _ in range(depth)
            ])
        else:
            # Non-residual layers
            self.layers = nn.ModuleList()
            Act = get_activation(activation)
            for _ in range(depth):
                modules = [nn.Linear(width, width)]
                if norm_type == 'batch':
                    modules.append(nn.BatchNorm1d(width))
                elif norm_type == 'layer':
                    modules.append(nn.LayerNorm(width))
                modules.append(Act())
                if dropout > 0:
                    modules.append(nn.Dropout(dropout))
                self.layers.append(nn.Sequential(*modules))

        # Output projection
        self.output = nn.Linear(width, output_dim)
        
        # Initialize weights
        self._initialize_weights(activation, depth)

    def forward(self, x):
        """Forward pass through the deep residual network."""
        x = self.input(x)
        for layer in self.layers:
            x = layer(x)
        return self.output(x)

    def _initialize_weights(self, activation, depth):
        """Initialize network weights based on the chosen scheme."""
        nonlin = activation.lower() if isinstance(activation, str) else 'relu'

        # Standard Kaiming initialization for all layers
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity=nonlin)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # Special initialization for residual schemes
        if not self.use_residual:
            return
            
        if self.init_scheme == "rezero":
            # ReZero: α parameters already set to 0 in ResidualBlock
            pass
            
        elif self.init_scheme == "fixup":
            # Fixup initialization: careful scaling without normalization
            if depth > 0:
                # Scale factor: L^{-1/2} for m=2 (2-layer residual blocks)
                scale = depth ** (-0.5)
                
                for block in self.layers:
                    if hasattr(block, 'branch'):
                        # Find first and last linear layers in the block
                        first_linear = None
                        last_linear = None
                        
                        for layer in block.branch:
                            if isinstance(layer, nn.Linear):
                                if first_linear is None:
                                    first_linear = layer
                                last_linear = layer
                        
                        # Scale first layer weights and zero-init last layer
                        if first_linear is not None:
                            first_linear.weight.data.mul_(scale)
                        if last_linear is not None:
                            nn.init.zeros_(last_linear.weight)
                
                # Optional: zero-init output layer for classification-like tasks
                # nn.init.zeros_(self.output.weight)


class SimpleFCNet(nn.Module):
    """
    Simple fully connected network without residual connections.
    Useful as a baseline comparison.
    """
    
    def __init__(self, input_dim, output_dim, width, depth, 
                 activation='relu', dropout=0.0, norm_type=None):
        super().__init__()
        Act = get_activation(activation)
        
        layers = [nn.Linear(input_dim, width)]
        
        for _ in range(depth):
            if norm_type == 'batch':
                layers.append(nn.BatchNorm1d(width))
            elif norm_type == 'layer':
                layers.append(nn.LayerNorm(width))
            layers.append(Act())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            layers.append(nn.Linear(width, width))
        
        layers.append(nn.Linear(width, output_dim))
        
        self.network = nn.Sequential(*layers)
        self._initialize_weights(activation)
    
    def forward(self, x):
        return self.network(x)
    
    def _initialize_weights(self, activation):
        nonlin = activation.lower() if isinstance(activation, str) else 'relu'
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity=nonlin)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)


class NetworkConfig:
    """Configuration class for network hyperparameters."""
    
    def __init__(self, 
                 network_type="DeepFCNet",
                 input_dim=None,
                 output_dim=None, 
                 width=256,
                 depth=4,
                 activation='relu',
                 norm_type=None,
                 dropout=0.0,
                 use_residual=True,
                 init_scheme="standard"):
        self.network_type = network_type
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.width = width
        self.depth = depth
        self.activation = activation
        self.norm_type = norm_type
        self.dropout = dropout
        self.use_residual = use_residual
        self.init_scheme = init_scheme
    
    def to_dict(self):
        return self.__dict__
    
    @classmethod
    def from_dict(cls, config_dict):
        return cls(**config_dict)


def create_network(config):
    """Factory function to create networks from configuration."""
    if config.network_type == "DeepFCNet":
        return DeepFCNet(
            input_dim=config.input_dim,
            output_dim=config.output_dim,
            width=config.width,
            depth=config.depth,
            norm_type=config.norm_type,
            activation=config.activation,
            dropout=config.dropout,
            init_scheme=config.init_scheme,
            use_residual=config.use_residual
        )
    elif config.network_type == "SimpleFCNet":
        return SimpleFCNet(
            input_dim=config.input_dim,
            output_dim=config.output_dim,
            width=config.width,
            depth=config.depth,
            activation=config.activation,
            dropout=config.dropout,
            norm_type=config.norm_type
        )
    else:
        raise ValueError(f"Unknown network type: {config.network_type}")
