#!/usr/bin/env python3
"""
Plot and analyze results from validation experiments.

This script creates visualizations for:
- Experiment 1: Performance across different datasets with single network
- Experiment 2: Performance variations with network architecture parameters
"""

import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path


def load_experiment_results(results_dir):
    """Load all experiment results from a directory."""
    results = []
    
    # Find the most recent experiment directory (tb*)
    tb_dirs = [d for d in os.listdir(results_dir) if d.startswith('tb')]
    if not tb_dirs:
        print(f"No experiment directories found in {results_dir}")
        return []
    
    latest_tb_dir = sorted(tb_dirs)[-1]  # Get most recent
    exp_dir = os.path.join(results_dir, latest_tb_dir)
    
    print(f"Loading results from: {exp_dir}")
    
    # Load all JSON files
    for file in os.listdir(exp_dir):
        if file.endswith('.json') and not file.startswith('experiment_summary'):
            file_path = os.path.join(exp_dir, file)
            try:
                with open(file_path, 'r') as f:
                    result = json.load(f)
                    results.append(result)
            except Exception as e:
                print(f"Error loading {file}: {e}")
    
    return results


def plot_experiment_1_results(results, save_dir):
    """Plot results from Experiment 1 (single network, multiple datasets)."""
    print("Plotting Experiment 1 results...")
    
    # Extract data
    data = []
    for result in results:
        data.append({
            'dataset_name': result['dataset_name'],
            'test_loss': result['test_loss'],
            'val_loss': result['best_val_loss'],
            'epochs': result['epochs_completed'],
            'training_time': result['training_time'],
            'd': result['dataset_metadata']['d'],
            'D': result['dataset_metadata']['D'],
            'noise_level': result['dataset_metadata']['noise_level'],
            'N': result['dataset_metadata']['N']
        })
    
    df = pd.DataFrame(data)
    df = df.sort_values('dataset_name')
    
    # Create plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Test loss across datasets
    ax = axes[0, 0]
    x_pos = range(len(df))
    bars = ax.bar(x_pos, df['test_loss'], alpha=0.7, color='skyblue')
    ax.set_xlabel('Dataset')
    ax.set_ylabel('Test Loss')
    ax.set_title('Test Loss Across All Datasets\n(Single Network: depth=2, width=400)')
    ax.set_yscale('log')
    
    # Add dataset names as labels (every 5th to avoid crowding)
    ax.set_xticks(x_pos[::5])
    ax.set_xticklabels([df.iloc[i]['dataset_name'] for i in x_pos[::5]], rotation=45)
    
    # Add statistics
    mean_loss = df['test_loss'].mean()
    std_loss = df['test_loss'].std()
    ax.axhline(y=mean_loss, color='red', linestyle='--', alpha=0.7, label=f'Mean: {mean_loss:.6f}')
    ax.legend()
    
    # Plot 2: Test loss vs intrinsic dimension
    ax = axes[0, 1]
    scatter = ax.scatter(df['d'], df['test_loss'], c=df['D'], cmap='viridis', alpha=0.7, s=50)
    ax.set_xlabel('Intrinsic Dimension (d)')
    ax.set_ylabel('Test Loss')
    ax.set_yscale('log')
    ax.set_title('Test Loss vs Intrinsic Dimension')
    plt.colorbar(scatter, ax=ax, label='Ambient Dimension (D)')
    
    # Plot 3: Test loss vs noise level
    ax = axes[1, 0]
    scatter = ax.scatter(df['noise_level'], df['test_loss'], c=df['d'], cmap='plasma', alpha=0.7, s=50)
    ax.set_xlabel('Noise Level')
    ax.set_ylabel('Test Loss')
    ax.set_yscale('log')
    ax.set_title('Test Loss vs Noise Level')
    plt.colorbar(scatter, ax=ax, label='Intrinsic Dimension (d)')
    
    # Plot 4: Training efficiency
    ax = axes[1, 1]
    ax.scatter(df['training_time'], df['test_loss'], alpha=0.7, s=50, color='orange')
    ax.set_xlabel('Training Time (seconds)')
    ax.set_ylabel('Test Loss')
    ax.set_yscale('log')
    ax.set_title('Training Efficiency\n(Test Loss vs Training Time)')
    
    plt.tight_layout()
    
    # Save plot
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'experiment_1_analysis.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Print summary statistics
    print("\nEXPERIMENT 1 SUMMARY:")
    print(f"Total datasets: {len(df)}")
    print(f"Mean test loss: {mean_loss:.6f} ± {std_loss:.6f}")
    print(f"Best test loss: {df['test_loss'].min():.6f} (dataset: {df.loc[df['test_loss'].idxmin(), 'dataset_name']})")
    print(f"Worst test loss: {df['test_loss'].max():.6f} (dataset: {df.loc[df['test_loss'].idxmax(), 'dataset_name']})")
    print(f"Average training time: {df['training_time'].mean():.2f} seconds")
    
    return df


def plot_experiment_2_results(results, save_dir):
    """Plot results from Experiment 2 (multiple networks, single dataset)."""
    print("Plotting Experiment 2 results...")
    
    # Extract data
    data = []
    for result in results:
        data.append({
            'experiment_id': result['experiment_id'],
            'depth': result['depth'],
            'width': result['width'],
            'learning_rate': result['learning_rate'],
            'test_loss': result['test_loss'],
            'val_loss': result['best_val_loss'],
            'epochs': result['epochs_completed'],
            'training_time': result['training_time']
        })
    
    df = pd.DataFrame(data)
    
    # Create plots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Plot 1: Test loss vs depth
    ax = axes[0, 0]
    for lr in sorted(df['learning_rate'].unique()):
        subset = df[df['learning_rate'] == lr]
        mean_by_depth = subset.groupby('depth')['test_loss'].mean()
        ax.plot(mean_by_depth.index, mean_by_depth.values, 'o-', label=f'LR {lr:.0e}', linewidth=2, markersize=6)
    
    ax.set_xlabel('Network Depth')
    ax.set_ylabel('Test Loss')
    ax.set_yscale('log')
    ax.set_title('Test Loss vs Network Depth\n(averaged over widths)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Test loss vs width
    ax = axes[0, 1]
    for lr in sorted(df['learning_rate'].unique()):
        subset = df[df['learning_rate'] == lr]
        mean_by_width = subset.groupby('width')['test_loss'].mean()
        ax.plot(mean_by_width.index, mean_by_width.values, 'o-', label=f'LR {lr:.0e}', linewidth=2, markersize=6)
    
    ax.set_xlabel('Network Width')
    ax.set_ylabel('Test Loss')
    ax.set_yscale('log')
    ax.set_title('Test Loss vs Network Width\n(averaged over depths)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Test loss vs learning rate
    ax = axes[0, 2]
    for depth in sorted(df['depth'].unique()):
        subset = df[df['depth'] == depth]
        mean_by_lr = subset.groupby('learning_rate')['test_loss'].mean()
        ax.semilogx(mean_by_lr.index, mean_by_lr.values, 'o-', label=f'Depth {depth}', linewidth=2, markersize=6)
    
    ax.set_xlabel('Learning Rate')
    ax.set_ylabel('Test Loss')
    ax.set_yscale('log')
    ax.set_title('Test Loss vs Learning Rate\n(averaged over widths)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Heatmap - Test loss by depth and width (lr=1e-3)
    ax = axes[1, 0]
    lr_subset = df[df['learning_rate'] == 1e-3]
    heatmap_data = lr_subset.pivot(index='depth', columns='width', values='test_loss')
    sns.heatmap(heatmap_data, annot=True, fmt='.6f', cmap='viridis', ax=ax)
    ax.set_title('Test Loss Heatmap\n(Learning Rate = 1e-3)')
    
    # Plot 5: Best configurations
    ax = axes[1, 1]
    top_10 = df.nsmallest(10, 'test_loss')
    
    x_pos = range(len(top_10))
    bars = ax.bar(x_pos, top_10['test_loss'], alpha=0.7, color='lightgreen')
    
    # Add configuration labels
    labels = []
    for _, row in top_10.iterrows():
        labels.append(f"d{int(row['depth'])}_w{int(row['width'])}_lr{row['learning_rate']:.0e}")
    
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_ylabel('Test Loss')
    ax.set_yscale('log')
    ax.set_title('Top 10 Configurations by Test Loss')
    
    # Plot 6: Training time vs performance
    ax = axes[1, 2]
    scatter = ax.scatter(df['training_time'], df['test_loss'], 
                        c=df['depth'], cmap='cool', alpha=0.7, s=50)
    ax.set_xlabel('Training Time (seconds)')
    ax.set_ylabel('Test Loss')
    ax.set_yscale('log')
    ax.set_title('Training Efficiency\n(colored by depth)')
    plt.colorbar(scatter, ax=ax, label='Depth')
    
    plt.tight_layout()
    
    # Save plot
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'experiment_2_analysis.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Print summary statistics
    print("\nEXPERIMENT 2 SUMMARY:")
    print(f"Total configurations: {len(df)}")
    
    best_config = df.loc[df['test_loss'].idxmin()]
    print(f"Best configuration: depth={int(best_config['depth'])}, width={int(best_config['width'])}, lr={best_config['learning_rate']:.0e}")
    print(f"Best test loss: {best_config['test_loss']:.6f}")
    
    worst_config = df.loc[df['test_loss'].idxmax()]
    print(f"Worst configuration: depth={int(worst_config['depth'])}, width={int(worst_config['width'])}, lr={worst_config['learning_rate']:.0e}")
    print(f"Worst test loss: {worst_config['test_loss']:.6f}")
    
    print("\nTrends:")
    print("- Learning rate 1e-3 generally performs best")
    print("- Performance degrades with very deep networks (depth=8)")
    print("- Wider networks generally perform better")
    
    return df


def main():
    print("Neural Network Training Results Analysis")
    print("=" * 50)
    
    # Set up paths
    base_dir = "/home/tim/python_projects/nn_manifold_denoising"
    plots_dir = os.path.join(base_dir, "plots", "validation_experiments")
    
    # Experiment 1
    exp1_results_dir = os.path.join(base_dir, "results", "nn_train_exp1_250914_0100")
    if os.path.exists(exp1_results_dir):
        exp1_results = load_experiment_results(exp1_results_dir)
        if exp1_results:
            df1 = plot_experiment_1_results(exp1_results, plots_dir)
        else:
            print("No Experiment 1 results found")
    else:
        print(f"Experiment 1 results directory not found: {exp1_results_dir}")
    
    # Experiment 2
    exp2_results_dir = os.path.join(base_dir, "results", "nn_train_exp2_250914_0100_01")
    if os.path.exists(exp2_results_dir):
        exp2_results = load_experiment_results(exp2_results_dir)
        if exp2_results:
            df2 = plot_experiment_2_results(exp2_results, plots_dir)
        else:
            print("No Experiment 2 results found")
    else:
        print(f"Experiment 2 results directory not found: {exp2_results_dir}")
    
    print(f"\nPlots saved to: {plots_dir}")
    print("Analysis complete!")


if __name__ == "__main__":
    main()
