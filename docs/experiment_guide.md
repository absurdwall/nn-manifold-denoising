# Neural Network Manifold Denoising - Experiment Configuration Guide

This guide explains how to set up and run custom experiments using the configurable experiment framework.

## Quick Start

1. **Run a predefined experiment:**
   ```bash
   python scripts/run_custom_experiment.py --config configs/experiment3_rezero.json --background --name my_rezero_test
   ```

2. **Monitor experiments:**
   ```bash
   python scripts/experiment_manager.py --status
   ```

3. **View results:**
   ```bash
   python scripts/experiment_manager.py --results
   ```

## Configuration File Structure

Experiment configurations are JSON files with three main sections:

### 1. Network Template (`network_template`)
Defines the neural network architecture:
```json
{
  "network_template": {
    "input_dim": 2,
    "hidden_layers": "{network_width}",  // Parameter placeholder
    "output_dim": 2,
    "activation": "tanh",
    "init_scheme": "{init_scheme}",      // Parameter placeholder
    "num_layers": "{network_depth}"      // Parameter placeholder
  }
}
```

### 2. Training Template (`training_template`)
Defines training parameters:
```json
{
  "training_template": {
    "n_epochs": 5000,
    "learning_rate": "{learning_rate}",   // Parameter placeholder
    "optimizer": "{optimizer}",           // Parameter placeholder
    "scheduler": "{scheduler}",           // Parameter placeholder
    "weight_decay": 1e-4
  }
}
```

### 3. Parameter Grid (`parameter_grid`)
Defines parameter sweeps:
```json
{
  "parameter_grid": {
    "dataset": ["circle", "swiss_roll"],
    "network_width": [100, 200, 400],
    "network_depth": [1, 2, 4, 8],
    "learning_rate": [1e-2, 1e-3, 1e-4],
    "optimizer": ["adam", "sgd"],
    "scheduler": ["cosine", "step"],
    "init_scheme": ["standard", "rezero"]
  }
}
```

## Parameter Placeholders

Use `{parameter_name}` in templates to reference parameter grid values:
- `{network_width}` → Values from `parameter_grid.network_width`
- `{learning_rate}` → Values from `parameter_grid.learning_rate`
- `{dataset}` → Values from `parameter_grid.dataset`

## Available Parameters

### Network Parameters:
- `network_width`: Hidden layer width (e.g., 100, 200, 400)
- `network_depth`: Number of layers (e.g., 1, 2, 4, 8)
- `init_scheme`: Initialization scheme ("standard", "rezero", "fixup")

### Training Parameters:
- `learning_rate`: Learning rate (e.g., 1e-2, 1e-3, 1e-4)
- `optimizer`: Optimizer type ("adam", "sgd", "rmsprop")
- `scheduler`: Learning rate scheduler ("cosine", "step", "none")

### Dataset Parameters:
- `dataset`: Dataset type ("circle", "swiss_roll", "s_curve")

## Example Configurations

### 1. ReZero Initialization Study (`experiment3_rezero.json`)
Tests ReZero initialization across different architectures:
```bash
python scripts/run_custom_experiment.py --config configs/experiment3_rezero.json --background --name rezero_study
```

### 2. Initialization Comparison (`experiment3_fixup.json`)
Compares standard vs Fixup initialization:
```bash
python scripts/run_custom_experiment.py --config configs/experiment3_fixup.json --background --name init_comparison
```

### 3. Comprehensive Analysis (`experiment3_comprehensive.json`)
Large-scale comparison across all parameters:
```bash
python scripts/run_custom_experiment.py --config configs/experiment3_comprehensive.json --background --name comprehensive_study
```

## Creating Custom Configurations

1. **Copy an existing configuration:**
   ```bash
   cp configs/experiment3_rezero.json configs/my_experiment.json
   ```

2. **Edit the parameter grid:**
   - Modify `parameter_grid` to include your desired parameter values
   - Add new parameters if needed

3. **Update templates:**
   - Modify `network_template` or `training_template` to use new parameters
   - Use `{parameter_name}` syntax for placeholders

4. **Run your experiment:**
   ```bash
   python scripts/run_custom_experiment.py --config configs/my_experiment.json --background --name my_study
   ```

## Running Experiments

### Foreground Execution
```bash
python scripts/run_custom_experiment.py --config configs/experiment3_rezero.json --name test_run
```

### Background Execution (Recommended)
```bash
python scripts/run_custom_experiment.py --config configs/experiment3_rezero.json --background --name background_run
```

### Specify Output Directory
```bash
python scripts/run_custom_experiment.py --config configs/experiment3_rezero.json --background --name custom_run --output results/my_results
```

## Monitoring Experiments

### List Running Sessions
```bash
python scripts/experiment_manager.py --list
```

### Check Experiment Status
```bash
python scripts/experiment_manager.py --status
```

### View Live Logs
```bash
python scripts/experiment_manager.py --logs SESSION_NAME
```

### Kill Running Experiment
```bash
python scripts/experiment_manager.py --kill SESSION_NAME
```

## Results Analysis

### View Results Summary
```bash
python scripts/experiment_manager.py --results
```

### Generate Enhanced Plots
```bash
python scripts/enhanced_plots.py
```

### Access Raw Data
- CSV files: `results/experiment_TIMESTAMP/*/experiment_results.csv`
- Log files: `results/experiment_TIMESTAMP/logs/`
- Plots: `plots/enhanced_validation/`

## Tips and Best Practices

1. **Start Small**: Test with a small parameter grid first
2. **Use Background**: Always use `--background` for long experiments
3. **Unique Names**: Use descriptive, unique names for each experiment
4. **Monitor Progress**: Check logs regularly with experiment_manager.py
5. **Save Configs**: Keep your custom configurations in version control

## Troubleshooting

### Experiment Won't Start
- Check configuration file syntax with a JSON validator
- Verify all parameter placeholders are defined in parameter_grid
- Ensure data directory exists

### Tmux Session Issues
- List sessions: `tmux list-sessions`
- Attach to session: `tmux attach -t SESSION_NAME`
- Kill all sessions: `tmux kill-server`

### Memory Issues
- Reduce batch size in network training
- Limit parameter grid size
- Monitor system resources during experiments

## Advanced Usage

### Custom Network Architectures
Modify `network_template` to support new architectures:
```json
{
  "network_template": {
    "architecture": "resnet",
    "blocks": "{num_blocks}",
    "channels": "{num_channels}"
  }
}
```

### Multi-GPU Support
Add GPU configuration to training template:
```json
{
  "training_template": {
    "device": "cuda:{gpu_id}",
    "distributed": true
  }
}
```

### Custom Metrics
Extend the experiment script to compute additional metrics and include them in CSV output.
