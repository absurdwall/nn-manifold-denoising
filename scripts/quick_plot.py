#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

print("Starting plotting...")

# Create output directory
os.makedirs('plots/validation_results', exist_ok=True)

# Load experiment 1 data
exp1_path = 'results/nn_train_exp1_250914_0100/tb20250921_2116/experiment1_results.csv'
if os.path.exists(exp1_path):
    print(f"Loading experiment 1 data from {exp1_path}")
    df1 = pd.read_csv(exp1_path)
    print(f"Experiment 1 shape: {df1.shape}")
    print(f"Columns: {list(df1.columns)}")
    
    # Plot 1: Test loss vs intrinsic dimension d
    plt.figure(figsize=(10, 6))
    plt.scatter(df1['d'], df1['test_loss'], alpha=0.7, s=50)
    plt.xlabel('Intrinsic Dimension (d)')
    plt.ylabel('Test Loss')
    plt.title('Test Loss vs Intrinsic Dimension d')
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    plt.savefig('plots/validation_results/exp1_test_loss_vs_d.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Plot 2: Test loss vs ambient dimension D
    plt.figure(figsize=(10, 6))
    plt.scatter(df1['D'], df1['test_loss'], alpha=0.7, s=50)
    plt.xlabel('Ambient Dimension (D)')
    plt.ylabel('Test Loss')
    plt.title('Test Loss vs Ambient Dimension D')
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    plt.savefig('plots/validation_results/exp1_test_loss_vs_D.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Plot 3: Test loss vs sample size N
    plt.figure(figsize=(10, 6))
    plt.scatter(df1['N'], df1['test_loss'], alpha=0.7, s=50)
    plt.xlabel('Sample Size (N)')
    plt.ylabel('Test Loss')
    plt.title('Test Loss vs Sample Size N')
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    plt.savefig('plots/validation_results/exp1_test_loss_vs_N.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("Experiment 1 plots created")
else:
    print(f"Experiment 1 file not found: {exp1_path}")

# Load experiment 2 data
exp2_path = 'results/nn_train_exp2_250914_0100_01/tb20250921_2124/experiment2_results.csv'
if os.path.exists(exp2_path):
    print(f"Loading experiment 2 data from {exp2_path}")
    df2 = pd.read_csv(exp2_path)
    print(f"Experiment 2 shape: {df2.shape}")
    print(f"Columns: {list(df2.columns)}")
    
    # Plot 4: Test loss vs network width
    plt.figure(figsize=(12, 8))
    for depth in sorted(df2['depth'].unique()):
        for lr in sorted(df2['learning_rate'].unique()):
            mask = (df2['depth'] == depth) & (df2['learning_rate'] == lr)
            subset = df2[mask].sort_values('width')
            if not subset.empty:
                plt.plot(subset['width'], subset['test_loss'], 
                        marker='o', label=f'depth={depth}, lr={lr:.0e}', linewidth=2)
    
    plt.xlabel('Network Width (neurons)')
    plt.ylabel('Test Loss')
    plt.title('Test Loss vs Network Width')
    plt.yscale('log')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.savefig('plots/validation_results/exp2_test_loss_vs_width.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Plot 5: Test loss vs network depth
    plt.figure(figsize=(12, 8))
    for width in sorted(df2['width'].unique()):
        for lr in sorted(df2['learning_rate'].unique()):
            mask = (df2['width'] == width) & (df2['learning_rate'] == lr)
            subset = df2[mask].sort_values('depth')
            if not subset.empty:
                plt.plot(subset['depth'], subset['test_loss'], 
                        marker='s', label=f'width={width}, lr={lr:.0e}', linewidth=2)
    
    plt.xlabel('Network Depth (layers)')
    plt.ylabel('Test Loss')
    plt.title('Test Loss vs Network Depth')
    plt.yscale('log')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.savefig('plots/validation_results/exp2_test_loss_vs_depth.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Plot 6: Test loss vs learning rate
    plt.figure(figsize=(12, 8))
    for width in sorted(df2['width'].unique()):
        for depth in sorted(df2['depth'].unique()):
            mask = (df2['width'] == width) & (df2['depth'] == depth)
            subset = df2[mask].sort_values('learning_rate')
            if not subset.empty:
                plt.plot(subset['learning_rate'], subset['test_loss'], 
                        marker='^', label=f'width={width}, depth={depth}', linewidth=2)
    
    plt.xlabel('Learning Rate')
    plt.ylabel('Test Loss')
    plt.title('Test Loss vs Learning Rate')
    plt.xscale('log')
    plt.yscale('log')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.savefig('plots/validation_results/exp2_test_loss_vs_lr.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("Experiment 2 plots created")
else:
    print(f"Experiment 2 file not found: {exp2_path}")

print("All plots saved to plots/validation_results/")
