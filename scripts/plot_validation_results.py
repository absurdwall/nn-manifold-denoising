#!/usr/bin/env python3
"""
Plot validation experiment results.

This script generates plots from the results of validation experiments,
creating visualizations for both individual experiments and comparative analysis.
"""

import os
import sys
import json
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

def load_experiment_results(results_dir):
    """Load all experiment results from a directory."""
    results = []
    
    # Look for CSV files first
    csv_files = list(Path(results_dir).glob("**/experiment_results.csv"))
    if csv_files:
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)
                results.append(df)
                print(f"Loaded CSV: {csv_file}")
            except Exception as e:
                print(f"Error loading {csv_file}: {e}")
    
    # If no CSV files, look for JSON files
    if not results:
        json_files = list(Path(results_dir).glob("**/*.json"))
        if json_files:
            data_list = []
            for json_file in json_files:
                try:
                    with open(json_file, 'r') as f:
                        data = json.load(f)
                    if 'test_loss' in data:  # Only experiment results, not summaries
                        data_list.append(data)
                except Exception as e:
                    print(f"Error loading {json_file}: {e}")
            
            if data_list:
                df = pd.DataFrame(data_list)
                results.append(df)
                print(f"Loaded {len(data_list)} JSON experiments")
    
    if results:
        combined_df = pd.concat(results, ignore_index=True)
        print(f"Total experiments loaded: {len(combined_df)}")
        return combined_df
    else:
        print("No experiment results found!")
        return pd.DataFrame()


def plot_loss_vs_network_depth(df, save_dir):
    """Plot test loss vs network depth."""
    if 'network_depth' not in df.columns or 'test_loss' not in df.columns:
        print("Missing columns for depth plot")
        return
    
    plt.figure(figsize=(10, 6))
    
    # Group by depth and plot mean/std
    depth_stats = df.groupby('network_depth')['test_loss'].agg(['mean', 'std', 'count'])
    
    plt.errorbar(depth_stats.index, depth_stats['mean'], 
                yerr=depth_stats['std'], marker='o', capsize=5)
    
    plt.xlabel('Network Depth')
    plt.ylabel('Test Loss')
    plt.title('Test Loss vs Network Depth')
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    # Add count annotations
    for depth, stats in depth_stats.iterrows():
        plt.annotate(f'n={stats["count"]}', 
                    (depth, stats['mean']), 
                    textcoords="offset points", 
                    xytext=(0,10), ha='center')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'test_loss_vs_depth.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: test_loss_vs_depth.png")


def plot_loss_vs_network_width(df, save_dir):
    """Plot test loss vs network width."""
    if 'network_width' not in df.columns or 'test_loss' not in df.columns:
        print("Missing columns for width plot")
        return
    
    plt.figure(figsize=(10, 6))
    
    # Group by width and plot mean/std
    width_stats = df.groupby('network_width')['test_loss'].agg(['mean', 'std', 'count'])
    
    plt.errorbar(width_stats.index, width_stats['mean'], 
                yerr=width_stats['std'], marker='o', capsize=5)
    
    plt.xlabel('Network Width')
    plt.ylabel('Test Loss')
    plt.title('Test Loss vs Network Width')
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    # Add count annotations
    for width, stats in width_stats.iterrows():
        plt.annotate(f'n={stats["count"]}', 
                    (width, stats['mean']), 
                    textcoords="offset points", 
                    xytext=(0,10), ha='center')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'test_loss_vs_width.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: test_loss_vs_width.png")


def plot_loss_vs_learning_rate(df, save_dir):
    """Plot test loss vs learning rate."""
    if 'learning_rate' not in df.columns or 'test_loss' not in df.columns:
        print("Missing columns for learning rate plot")
        return
    
    plt.figure(figsize=(10, 6))
    
    # Group by learning rate and plot mean/std
    lr_stats = df.groupby('learning_rate')['test_loss'].agg(['mean', 'std', 'count'])
    
    plt.errorbar(lr_stats.index, lr_stats['mean'], 
                yerr=lr_stats['std'], marker='o', capsize=5)
    
    plt.xlabel('Learning Rate')
    plt.ylabel('Test Loss')
    plt.title('Test Loss vs Learning Rate')
    plt.grid(True, alpha=0.3)
    plt.xscale('log')
    plt.yscale('log')
    
    # Add count annotations
    for lr, stats in lr_stats.iterrows():
        plt.annotate(f'n={stats["count"]}', 
                    (lr, stats['mean']), 
                    textcoords="offset points", 
                    xytext=(0,10), ha='center')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'test_loss_vs_lr.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: test_loss_vs_lr.png")


def plot_initialization_comparison(df, save_dir):
    """Plot comparison of initialization schemes."""
    if 'init_scheme' not in df.columns or 'test_loss' not in df.columns:
        print("Missing columns for initialization plot")
        return
    
    plt.figure(figsize=(12, 6))
    
    # Box plot by initialization scheme
    init_schemes = df['init_scheme'].unique()
    test_losses = [df[df['init_scheme'] == scheme]['test_loss'].values for scheme in init_schemes]
    
    plt.boxplot(test_losses, labels=init_schemes)
    plt.ylabel('Test Loss')
    plt.xlabel('Initialization Scheme')
    plt.title('Test Loss Distribution by Initialization Scheme')
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    
    # Add sample counts
    for i, scheme in enumerate(init_schemes):
        count = len(df[df['init_scheme'] == scheme])
        plt.text(i+1, plt.ylim()[1]*0.8, f'n={count}', ha='center')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'initialization_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: initialization_comparison.png")


def plot_best_experiments(df, save_dir, top_k=10):
    """Plot the best experiments."""
    if 'test_loss' not in df.columns:
        print("Missing test_loss column")
        return
    
    # Sort by test loss and get top k
    best_df = df.nsmallest(top_k, 'test_loss')
    
    plt.figure(figsize=(12, 8))
    
    # Create labels for experiments
    labels = []
    for idx, row in best_df.iterrows():
        label_parts = []
        if 'network_depth' in row:
            label_parts.append(f"D={row['network_depth']}")
        if 'network_width' in row:
            label_parts.append(f"W={row['network_width']}")
        if 'learning_rate' in row:
            label_parts.append(f"LR={row['learning_rate']:.0e}")
        if 'init_scheme' in row:
            label_parts.append(f"Init={row['init_scheme']}")
        
        labels.append("\n".join(label_parts))
    
    plt.barh(range(len(best_df)), best_df['test_loss'])
    plt.yticks(range(len(best_df)), labels)
    plt.xlabel('Test Loss')
    plt.title(f'Top {top_k} Best Experiments')
    plt.xscale('log')
    plt.grid(True, alpha=0.3)
    
    # Add loss values as text
    for i, loss in enumerate(best_df['test_loss']):
        plt.text(loss, i, f'{loss:.2e}', va='center', ha='left', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'top_{top_k}_experiments.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: top_{top_k}_experiments.png")


def create_summary_table(df, save_dir):
    """Create a summary table of experiment results."""
    if df.empty:
        return
    
    summary_stats = []
    
    # Overall statistics
    summary_stats.append({
        'Metric': 'Total Experiments',
        'Value': len(df),
        'Description': 'Total number of experiments run'
    })
    
    if 'test_loss' in df.columns:
        summary_stats.extend([
            {
                'Metric': 'Best Test Loss',
                'Value': f"{df['test_loss'].min():.6f}",
                'Description': 'Lowest test loss achieved'
            },
            {
                'Metric': 'Median Test Loss',
                'Value': f"{df['test_loss'].median():.6f}",
                'Description': 'Median test loss across all experiments'
            },
            {
                'Metric': 'Mean Test Loss',
                'Value': f"{df['test_loss'].mean():.6f}",
                'Description': 'Mean test loss across all experiments'
            }
        ])
    
    # Save summary table
    summary_df = pd.DataFrame(summary_stats)
    summary_path = os.path.join(save_dir, 'experiment_summary.csv')
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved: experiment_summary.csv")
    
    return summary_df


def main():
    parser = argparse.ArgumentParser(description='Plot validation experiment results')
    parser.add_argument('--results-dir', type=str, required=True,
                       help='Directory containing experiment results')
    parser.add_argument('--output-dir', type=str, default='plots/validation_results',
                       help='Directory to save plots')
    parser.add_argument('--top-k', type=int, default=10,
                       help='Number of top experiments to show')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load experiment results
    print(f"Loading results from: {args.results_dir}")
    df = load_experiment_results(args.results_dir)
    
    if df.empty:
        print("No results to plot!")
        return
    
    print(f"Loaded {len(df)} experiments")
    print(f"Columns: {list(df.columns)}")
    
    # Create plots
    print("Creating plots...")
    
    plot_loss_vs_network_depth(df, args.output_dir)
    plot_loss_vs_network_width(df, args.output_dir)
    plot_loss_vs_learning_rate(df, args.output_dir)
    plot_initialization_comparison(df, args.output_dir)
    plot_best_experiments(df, args.output_dir, args.top_k)
    
    # Create summary
    summary_df = create_summary_table(df, args.output_dir)
    if summary_df is not None:
        print("\nExperiment Summary:")
        print(summary_df.to_string(index=False))
    
    print(f"\nAll plots saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
