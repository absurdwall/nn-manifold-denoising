#!/usr/bin/env python3
"""
Improved plotting script for validation experiments.

Features:
- Adds kernel_smoothness data from original metadata
- Creates line plots for experiment 1 (grouping by similar parameters)
- Better organization and styling
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import json
import glob

def load_and_enhance_exp1_data(csv_path):
    """Load experiment 1 data and enhance with kernel_smoothness from metadata."""
    df = pd.read_csv(csv_path)
    
    # Add kernel_smoothness from original metadata
    kernel_smoothness_list = []
    for _, row in df.iterrows():
        dataset_name = row['dataset_name']
        # Find the metadata file
        metadata_pattern = f"data/data_250914_0100_01/{dataset_name}/{dataset_name}_metadata.json"
        try:
            with open(metadata_pattern, 'r') as f:
                metadata = json.load(f)
                kernel_smoothness = metadata['properties']['kernel_smoothness']
                kernel_smoothness_list.append(kernel_smoothness)
        except:
            kernel_smoothness_list.append(None)
    
    df['kernel_smoothness'] = kernel_smoothness_list
    return df

def create_exp1_line_plots(df, output_dir):
    """Create line plots for experiment 1 with proper grouping."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Since all experiments use the same network config, we group by similar dataset properties
    # Plot 1: Test loss vs intrinsic dimension d (group by ambient dimension D)
    plt.figure(figsize=(12, 8))
    for D in sorted(df['D'].unique()):
        subset = df[df['D'] == D].sort_values('d')
        if not subset.empty:
            plt.plot(subset['d'], subset['test_loss'], 
                    marker='o', label=f'D={D}', linewidth=2, markersize=6)
    
    plt.xlabel('Intrinsic Dimension (d)')
    plt.ylabel('Test Loss')
    plt.title('Test Loss vs Intrinsic Dimension (grouped by ambient dimension)')
    plt.yscale('log')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'exp1_test_loss_vs_d_grouped.png'), 
                dpi=150, bbox_inches='tight')
    plt.close()
    
    # Plot 2: Test loss vs ambient dimension D (group by intrinsic dimension d)
    plt.figure(figsize=(12, 8))
    for d in sorted(df['d'].unique()):
        subset = df[df['d'] == d].sort_values('D')
        if not subset.empty:
            plt.plot(subset['D'], subset['test_loss'], 
                    marker='s', label=f'd={d}', linewidth=2, markersize=6)
    
    plt.xlabel('Ambient Dimension (D)')
    plt.ylabel('Test Loss')
    plt.title('Test Loss vs Ambient Dimension (grouped by intrinsic dimension)')
    plt.yscale('log')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'exp1_test_loss_vs_D_grouped.png'), 
                dpi=150, bbox_inches='tight')
    plt.close()
    
    # Plot 3: Test loss vs kernel smoothness (group by intrinsic dimension d)
    if 'kernel_smoothness' in df.columns and not df['kernel_smoothness'].isna().all():
        plt.figure(figsize=(12, 8))
        for d in sorted(df['d'].unique()):
            subset = df[df['d'] == d].sort_values('kernel_smoothness')
            if not subset.empty:
                plt.plot(subset['kernel_smoothness'], subset['test_loss'], 
                        marker='^', label=f'd={d}', linewidth=2, markersize=6)
        
        plt.xlabel('Kernel Smoothness')
        plt.ylabel('Test Loss')
        plt.title('Test Loss vs Kernel Smoothness (grouped by intrinsic dimension)')
        plt.xscale('log')
        plt.yscale('log')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(output_dir, 'exp1_test_loss_vs_kernel_smoothness_grouped.png'), 
                    dpi=150, bbox_inches='tight')
        plt.close()
    
    # Plot 4: Test loss vs sample size N (group by intrinsic dimension d)
    plt.figure(figsize=(12, 8))
    for d in sorted(df['d'].unique()):
        subset = df[df['d'] == d].sort_values('N')
        if not subset.empty:
            plt.plot(subset['N'], subset['test_loss'], 
                    marker='D', label=f'd={d}', linewidth=2, markersize=6)
    
    plt.xlabel('Sample Size (N)')
    plt.ylabel('Test Loss')
    plt.title('Test Loss vs Sample Size (grouped by intrinsic dimension)')
    plt.yscale('log')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'exp1_test_loss_vs_N_grouped.png'), 
                dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Experiment 1 enhanced plots saved to: {output_dir}")

def create_exp2_line_plots(df, output_dir):
    """Create line plots for experiment 2 (already implemented well)."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot 1: Test loss vs network width (group by depth and learning rate)
    plt.figure(figsize=(14, 8))
    colors = plt.cm.tab10(np.linspace(0, 1, len(df['depth'].unique())))
    markers = ['o', 's', '^']
    
    for i, depth in enumerate(sorted(df['depth'].unique())):
        for j, lr in enumerate(sorted(df['learning_rate'].unique())):
            mask = (df['depth'] == depth) & (df['learning_rate'] == lr)
            subset = df[mask].sort_values('width')
            if not subset.empty:
                plt.plot(subset['width'], subset['test_loss'], 
                        marker=markers[j % len(markers)], color=colors[i],
                        label=f'depth={depth}, lr={lr:.0e}', linewidth=2, markersize=6)
    
    plt.xlabel('Network Width (neurons)')
    plt.ylabel('Test Loss')
    plt.title('Test Loss vs Network Width')
    plt.yscale('log')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'exp2_test_loss_vs_width.png'), 
                dpi=150, bbox_inches='tight')
    plt.close()
    
    # Plot 2: Test loss vs network depth (group by width and learning rate)
    plt.figure(figsize=(14, 8))
    colors = plt.cm.tab10(np.linspace(0, 1, len(df['width'].unique())))
    
    for i, width in enumerate(sorted(df['width'].unique())):
        for j, lr in enumerate(sorted(df['learning_rate'].unique())):
            mask = (df['width'] == width) & (df['learning_rate'] == lr)
            subset = df[mask].sort_values('depth')
            if not subset.empty:
                plt.plot(subset['depth'], subset['test_loss'], 
                        marker=markers[j % len(markers)], color=colors[i],
                        label=f'width={width}, lr={lr:.0e}', linewidth=2, markersize=6)
    
    plt.xlabel('Network Depth (layers)')
    plt.ylabel('Test Loss')
    plt.title('Test Loss vs Network Depth')
    plt.yscale('log')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'exp2_test_loss_vs_depth.png'), 
                dpi=150, bbox_inches='tight')
    plt.close()
    
    # Plot 3: Test loss vs learning rate (group by width and depth)
    plt.figure(figsize=(14, 8))
    colors = plt.cm.tab10(np.linspace(0, 1, len(df['width'].unique())))
    
    for i, width in enumerate(sorted(df['width'].unique())):
        for j, depth in enumerate(sorted(df['depth'].unique())):
            mask = (df['width'] == width) & (df['depth'] == depth)
            subset = df[mask].sort_values('learning_rate')
            if not subset.empty:
                marker = markers[j % len(markers)]
                plt.plot(subset['learning_rate'], subset['test_loss'], 
                        marker=marker, color=colors[i],
                        label=f'width={width}, depth={depth}', linewidth=2, markersize=6)
    
    plt.xlabel('Learning Rate')
    plt.ylabel('Test Loss')
    plt.title('Test Loss vs Learning Rate')
    plt.xscale('log')
    plt.yscale('log')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'exp2_test_loss_vs_lr.png'), 
                dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Experiment 2 enhanced plots saved to: {output_dir}")

def main():
    # Enhanced plotting for experiment 1
    exp1_path = 'results/nn_train_exp1_250914_0100/tb20250921_2116/experiment1_results.csv'
    if os.path.exists(exp1_path):
        print("Creating enhanced experiment 1 plots...")
        df1 = load_and_enhance_exp1_data(exp1_path)
        create_exp1_line_plots(df1, 'plots/enhanced_validation/exp1')
    
    # Enhanced plotting for experiment 2
    exp2_path = 'results/nn_train_exp2_250914_0100_01/tb20250921_2124/experiment2_results.csv'
    if os.path.exists(exp2_path):
        print("Creating enhanced experiment 2 plots...")
        df2 = pd.read_csv(exp2_path)
        create_exp2_line_plots(df2, 'plots/enhanced_validation/exp2')

if __name__ == "__main__":
    main()
