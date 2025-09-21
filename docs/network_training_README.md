# Neural Network Training for Manifold Denoising

This directory contains the implementation of neural network training for manifold denoising experiments. The implementation is based on the successful approaches from the original research codebase, with improvements in organization and extensibility.

## Overview

The neural network training pipeline consists of three main components:

1. **Network Architectures** (`src/network_training/networks.py`)
2. **Training Utilities** (`src/network_training/training.py`)  
3. **Training Script** (`scripts/step4_training.py`)

## Network Architectures

### DeepFCNet
The main architecture is a deep fully connected neural network with residual connections. Key features:

- **Configurable depth and width**: Support for networks from shallow (1 layer) to very deep (16+ layers)
- **Residual connections**: Skip connections to enable training of very deep networks
- **Multiple initialization schemes**: Standard, ReZero, and Fixup initialization
- **Normalization options**: Batch normalization, layer normalization, or none
- **Activation functions**: ReLU, GELU, Swish, Mish, etc.
- **Dropout**: Optional dropout for regularization

### Initialization Schemes

#### Standard Initialization
- Uses Kaiming (He) initialization for all layers
- Works well for shallow to moderately deep networks
- Can suffer from gradient explosion/vanishing in very deep networks

#### ReZero Initialization
Based on [Bachlechner et al. (2020)](https://arxiv.org/abs/2003.04887):
- Adds learnable scalar gates (α) initialized at 0
- Residual connection: `x_{l+1} = x_l + α_l * f(x_l)`
- Network starts as identity and gradually learns residuals
- Enables training of very deep networks without normalization

#### Fixup Initialization  
Based on [Zhang et al. (2019)](https://arxiv.org/abs/1901.09321):
- Carefully scales weights without normalization layers
- Scales first layer weights by `L^{-1/2}` where L is depth
- Zero-initializes last layer in each residual block
- Enables stable training without batch/layer normalization

### SimpleFCNet
A baseline fully connected network without residual connections for comparison.

## Training Features

### Optimizers
- **Adam**: Adaptive learning rate with momentum
- **AdamW**: Adam with decoupled weight decay
- **SGD**: Stochastic gradient descent with momentum
- **RMSprop**: Root mean square propagation

### Learning Rate Scheduling
- **Fixed**: Constant learning rate
- **Plateau**: Reduce on validation loss plateau
- **Step**: Step decay at fixed intervals
- **Cosine**: Cosine annealing schedule

### Advanced Features
- **Learning rate scaling**: Automatic LR scaling based on network depth (`1/L`, `1/L²`, `√L`)
- **Early stopping**: Stop training when validation loss stops improving
- **Gradient clipping**: Prevent gradient explosion
- **Automatic mixed precision**: Memory-efficient training (future feature)

## Usage

### Quick Start

```python
from network_training import NetworkConfig, TrainingConfig, create_network, train_model

# Define network
network_config = NetworkConfig(
    network_type="DeepFCNet",
    input_dim=10,
    output_dim=10, 
    width=256,
    depth=8,
    init_scheme="rezero",
    norm_type="batch"
)

# Define training  
training_config = TrainingConfig(
    max_epochs=1000,
    batch_size=512,
    learning_rate=1e-3,
    optimizer_name='adam',
    scheduler_mode='plateau'
)

# Create and train
model = create_network(network_config)
metrics, best_state = train_model(model, train_loader, val_loader, training_config)
```

### Running Full Training Pipeline

```bash
# Run comprehensive training sweep
python scripts/step4_training.py \
    --data-dir data/datasets \
    --results-dir results/step4_training \
    --config configs/step4_training_config.json

# Run subset for testing
python scripts/step4_training.py --subset 10 --verbose

# Resume interrupted training
python scripts/step4_training.py --resume
```

### Configuration File

Create `configs/step4_training_config.json` to customize experiments:

```json
{
  "network_configs": {
    "architectures": ["DeepFCNet"],
    "widths": [100, 200, 400],
    "depths": [1, 2, 4, 8],
    "init_schemes": ["standard", "rezero", "fixup"]
  },
  "training_configs": {
    "max_epochs": 2000,
    "base_learning_rates": [0.001, 0.0001],
    "optimizers": ["adam", "adamw"]
  }
}
```

## Testing

Test the implementation before running full experiments:

```bash
python testing/test_network_training.py
```

This will verify:
- Network architectures can be created and executed
- Training loop works correctly
- Initialization schemes are stable
- Forward/backward passes complete successfully

## Output Structure

Results are saved in the following structure:

```
results/step4_training/
├── experiment_summary.json          # Overall summary
├── summary_stats.json              # Statistical summary  
├── dataset1_net001_train001.json   # Individual experiments
├── dataset1_net001_train002.json
└── ...

plots/step4_training/
├── test_loss_vs_depth.png          # Analysis plots
├── test_loss_vs_width.png
├── training_time_vs_size.png
├── test_loss_heatmap.png
└── top_experiments_analysis.png
```

## Key Implementation Details

### Residual Block Architecture
```python
def forward(self, x):
    if self.init_scheme == "rezero":
        return x + self.alpha * self.branch(x)  # α starts at 0
    else:
        return x + self.branch(x)  # Standard residual
```

### Fixup Weight Scaling
```python
# Scale first layer weights by L^{-1/2}
scale = depth ** (-0.5)
first_linear.weight.data.mul_(scale)

# Zero-initialize last layer
nn.init.zeros_(last_linear.weight)
```

### Learning Rate Scaling Rules
```python
def apply_lr_scaling(base_lr, depth, rule):
    if rule == '1/L':
        return base_lr / depth
    elif rule == '1/L2': 
        return base_lr / (depth ** 2)
    elif rule == 'sqrt_L':
        return base_lr / sqrt(depth)
```

## Performance Considerations

### Memory Usage
- Batch size should be adjusted based on network size and GPU memory
- Deeper networks require more memory for gradient storage
- Use gradient checkpointing for very deep networks (future feature)

### Training Speed
- Larger batch sizes generally train faster but may require LR adjustment
- ReZero networks often converge faster than standard residual networks
- Fixup networks can train without normalization, reducing memory and computation

### Numerical Stability
- ReZero and Fixup schemes improve stability for deep networks
- Monitor gradient norms during training
- Use gradient clipping if gradients become too large

## Research Applications

This implementation enables investigation of:

1. **Depth vs Performance**: How network depth affects manifold denoising quality
2. **Initialization Impact**: Comparison of initialization schemes on convergence
3. **Architecture Ablations**: Effect of normalization, activation functions, etc.
4. **Scaling Laws**: How performance scales with network size and data
5. **Generalization**: How well networks trained on one manifold generalize to others

## References

- **ReZero**: Bachlechner, T., et al. "ReZero is All You Need: Fast Convergence at Large Depth." arXiv:2003.04887 (2020)
- **Fixup**: Zhang, H., et al. "Fixup Initialization: Residual Learning Without Normalization." arXiv:1901.09321 (2019)  
- **Deep Residual Networks**: He, K., et al. "Deep Residual Learning for Image Recognition." CVPR 2016
