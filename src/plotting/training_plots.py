"""
Training visualization utilities.

This module provides functions for plotting training curves, experiment summaries,
and network performance analysis.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import json
import os
from typing import List, Dict, Any


def plot_training_curves(metrics_dict: Dict, save_path: str = None, 
                        title: str = "Training Curves"):
    """
    Plot training and validation loss curves.
    
    Args:
        metrics_dict: Dictionary containing 'train_losses' and 'val_losses'
        save_path: Path to save the plot
        title: Title for the plot
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    epochs = range(1, len(metrics_dict['train_losses']) + 1)
    
    # Linear scale
    ax1.plot(epochs, metrics_dict['train_losses'], label='Train Loss', alpha=0.8)
    ax1.plot(epochs, metrics_dict['val_losses'], label='Validation Loss', alpha=0.8)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title(f'{title} - Linear Scale')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Log scale
    ax2.plot(epochs, metrics_dict['train_losses'], label='Train Loss', alpha=0.8)
    ax2.plot(epochs, metrics_dict['val_losses'], label='Validation Loss', alpha=0.8)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss (log scale)')
    ax2.set_yscale('log')
    ax2.set_title(f'{title} - Log Scale')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_experiment_summary(results_dir: str, plots_dir: str):
    """
    Create summary plots from experiment results.
    
    Args:
        results_dir: Directory containing experiment JSON files
        plots_dir: Directory to save plots
    """
    os.makedirs(plots_dir, exist_ok=True)
    
    # Load all experiment results
    experiments = []
    for file in os.listdir(results_dir):
        if file.endswith('.json') and not file.startswith('summary'):
            try:
                with open(os.path.join(results_dir, file), 'r') as f:
                    data = json.load(f)
                    if data.get('success', False):
                        experiments.append(data)
            except Exception as e:
                print(f"Error loading {file}: {e}")
                continue
    
    if not experiments:
        print("No successful experiments found")
        return
    
    # Convert to DataFrame for easier analysis
    records = []
    for exp in experiments:
        record = {
            'experiment_id': exp['experiment_id'],
            'test_loss': exp['test_loss'],
            'best_val_loss': exp['metrics']['best_val_loss'],
            'epochs_completed': exp['metrics']['epochs_completed'],
            'training_time': exp['training_time'],
            # Network config
            'network_type': exp['network_config']['network_type'],
            'width': exp['network_config']['width'],
            'depth': exp['network_config']['depth'],
            'activation': exp['network_config']['activation'],
            'norm_type': exp['network_config'].get('norm_type', 'none'),
            'init_scheme': exp['network_config'].get('init_scheme', 'standard'),
            # Training config
            'optimizer': exp['training_config']['optimizer_name'],
            'learning_rate': exp['training_config']['learning_rate'],
            'scheduler': exp['training_config']['scheduler_mode'],
            # Dataset info
            'dataset_d': exp['dataset_metadata']['d'],
            'dataset_D': exp['dataset_metadata']['D'],
            'dataset_noise': exp['dataset_metadata'].get('noise_level', 0),
        }
        records.append(record)
    
    df = pd.DataFrame(records)
    
    # Plot 1: Test loss vs network depth
    plt.figure(figsize=(10, 6))
    for init_scheme in df['init_scheme'].unique():
        subset = df[df['init_scheme'] == init_scheme]
        plt.scatter(subset['depth'], subset['test_loss'], 
                   label=f'{init_scheme}', alpha=0.7)
    plt.xlabel('Network Depth')
    plt.ylabel('Test Loss')
    plt.yscale('log')
    plt.title('Test Loss vs Network Depth by Initialization Scheme')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(plots_dir, 'test_loss_vs_depth.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 2: Test loss vs network width
    plt.figure(figsize=(10, 6))
    for norm_type in df['norm_type'].unique():
        subset = df[df['norm_type'] == norm_type]
        plt.scatter(subset['width'], subset['test_loss'], 
                   label=f'{norm_type}', alpha=0.7)
    plt.xlabel('Network Width')
    plt.ylabel('Test Loss')
    plt.yscale('log')
    plt.title('Test Loss vs Network Width by Normalization Type')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(plots_dir, 'test_loss_vs_width.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 3: Training time vs network size
    df['network_size'] = df['width'] * df['depth']
    plt.figure(figsize=(10, 6))
    plt.scatter(df['network_size'], df['training_time'], alpha=0.7)
    plt.xlabel('Network Size (width × depth)')
    plt.ylabel('Training Time (seconds)')
    plt.title('Training Time vs Network Size')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(plots_dir, 'training_time_vs_size.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 4: Heatmap of test loss by width and depth
    pivot_table = df.pivot_table(values='test_loss', index='depth', 
                                columns='width', aggfunc='mean')
    plt.figure(figsize=(12, 8))
    sns.heatmap(pivot_table, annot=True, fmt='.3f', cmap='viridis')
    plt.title('Average Test Loss by Network Width and Depth')
    plt.savefig(os.path.join(plots_dir, 'test_loss_heatmap.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 5: Best experiments
    top_experiments = df.nsmallest(20, 'test_loss')
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # By initialization scheme
    init_counts = top_experiments['init_scheme'].value_counts()
    axes[0, 0].pie(init_counts.values, labels=init_counts.index, autopct='%1.1f%%')
    axes[0, 0].set_title('Top 20 Experiments: Initialization Schemes')
    
    # By normalization
    norm_counts = top_experiments['norm_type'].value_counts()
    axes[0, 1].pie(norm_counts.values, labels=norm_counts.index, autopct='%1.1f%%')
    axes[0, 1].set_title('Top 20 Experiments: Normalization Types')
    
    # By optimizer
    opt_counts = top_experiments['optimizer'].value_counts()
    axes[1, 0].pie(opt_counts.values, labels=opt_counts.index, autopct='%1.1f%%')
    axes[1, 0].set_title('Top 20 Experiments: Optimizers')
    
    # By activation
    act_counts = top_experiments['activation'].value_counts()
    axes[1, 1].pie(act_counts.values, labels=act_counts.index, autopct='%1.1f%%')
    axes[1, 1].set_title('Top 20 Experiments: Activations')
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'top_experiments_analysis.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save summary statistics
    summary_stats = {
        'total_experiments': len(df),
        'best_test_loss': float(df['test_loss'].min()),
        'worst_test_loss': float(df['test_loss'].max()),
        'median_test_loss': float(df['test_loss'].median()),
        'best_experiment': df.loc[df['test_loss'].idxmin()]['experiment_id'],
        'avg_training_time': float(df['training_time'].mean()),
        'total_training_time': float(df['training_time'].sum()),
    }
    
    with open(os.path.join(plots_dir, 'summary_stats.json'), 'w') as f:
        json.dump(summary_stats, f, indent=2)
    
    print(f"Created summary plots in {plots_dir}")
    print(f"Best experiment: {summary_stats['best_experiment']} "
          f"(test loss: {summary_stats['best_test_loss']:.6f})")


def plot_learning_curves_comparison(results_dir: str, experiment_ids: List[str], 
                                  plots_dir: str, title: str = "Learning Curves Comparison"):
    """
    Compare learning curves of multiple experiments.
    
    Args:
        results_dir: Directory containing experiment results
        experiment_ids: List of experiment IDs to compare
        plots_dir: Directory to save plots
        title: Title for the plot
    """
    os.makedirs(plots_dir, exist_ok=True)
    
    plt.figure(figsize=(12, 8))
    
    for exp_id in experiment_ids:
        exp_file = os.path.join(results_dir, f'{exp_id}.json')
        if not os.path.exists(exp_file):
            print(f"Warning: {exp_file} not found")
            continue
            
        with open(exp_file, 'r') as f:
            data = json.load(f)
        
        if not data.get('success', False):
            continue
            
        metrics = data['metrics']
        epochs = range(1, len(metrics['train_losses']) + 1)
        
        plt.plot(epochs, metrics['val_losses'], label=f'{exp_id}', alpha=0.8)
    
    plt.xlabel('Epoch')
    plt.ylabel('Validation Loss')
    plt.yscale('log')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    save_path = os.path.join(plots_dir, 'learning_curves_comparison.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Learning curves comparison saved to {save_path}")
