#!/usr/bin/env python3
"""
Step 3: Comprehensive Plotting Pipeline

This script creates plots focused on how geometric properties vary with dataset parameters,
especially kernel_smoothness (σ) parameter.

Key plots:
- kernel_smoothness vs diameter (with theoretical curves)
- kernel_smoothness vs estimated dimensionality
- kernel_smoothness vs curvature estimation (with theoretical curves)

Usage:
    python scripts/step3_plotting_pipeline.py [--results_dir RESULTS_DIR] [--output_dir OUTPUT_DIR]
"""

import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional
import warnings
import logging
from scipy.special import gamma, gammainc

# Set plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
warnings.filterwarnings('ignore', category=UserWarning)

# Suppress matplotlib font debugging output
import matplotlib
matplotlib.set_loglevel("WARNING")


def setup_logging(verbose: bool = False) -> logging.Logger:
    """Setup logging configuration."""
    level = logging.INFO  # Changed from DEBUG even when verbose
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    
    # Suppress matplotlib debug logging
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)
    
    return logging.getLogger(__name__)


def theoretical_expected_diameter(d: float, D: float, kernel_smoothness: float) -> float:
    """
    Compute theoretical expected diameter using corrected incomplete gamma function.
    
    Formula: Expected diameter ≈ 2 * sqrt(d * D * Γ(3/2, log(3))) / σ
    where Γ(3/2, log(3)) is the upper incomplete gamma function.
    """
    if kernel_smoothness <= 0:
        return np.nan
    
    # Upper incomplete gamma function: Γ(3/2, log(3))
    # scipy.special.gammainc(a, x) = γ(a,x)/Γ(a) = lower incomplete / complete
    # We want upper incomplete: Γ(a,x) = Γ(a) * (1 - gammainc(a,x))
    upper_incomplete_gamma = gamma(1.5) * (1 - gammainc(1.5, np.log(3)))
    
    # Expected radius
    expected_radius = np.sqrt(d * D * upper_incomplete_gamma) / kernel_smoothness
    
    # Expected diameter
    expected_diameter = 2 * expected_radius
    
    return expected_diameter


def theoretical_curvature_squared(d: float, kernel_smoothness: float) -> float:
    """
    Compute theoretical curvature squared using corrected gamma function.
    
    Formula: E[κ²] ≈ Γ(3/2, log(3)) / (d * σ²)
    """
    if kernel_smoothness <= 0 or d <= 0:
        return np.nan
    
    # Upper incomplete gamma function calculation
    upper_incomplete_gamma = gamma(1.5) * (1 - gammainc(1.5, np.log(3)))
    
    curvature_squared = upper_incomplete_gamma / (d * kernel_smoothness**2)
    return np.sqrt(curvature_squared)  # Return mean curvature, not squared


def load_comprehensive_results(results_dir: Path) -> pd.DataFrame:
    """Load comprehensive analysis results."""
    
    # Load the CSV summary file with all parameters
    csv_file = results_dir / "comprehensive_summary.csv"
    if not csv_file.exists():
        raise FileNotFoundError(f"Summary CSV not found: {csv_file}")
    
    df = pd.read_csv(csv_file)
    
    # Ensure required columns exist
    required_cols = ['kernel_smoothness', 'true_d', 'true_D', 'extrinsic_diameter', 
                    'mean_curvature', 'pca_curvature', 'data_type']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    return df


def create_kernel_smoothness_vs_diameter_plot(
    df: pd.DataFrame,
    output_dir: Path,
    logger: logging.Logger
) -> None:
    """Create kernel_smoothness vs diameter plot with theoretical curves."""
    
    logger.info("Creating kernel_smoothness vs diameter plot...")
    
    # Create separate plots for each data type
    data_types = df['data_type'].unique()
    data_types = [dt for dt in data_types if pd.notna(dt)]
    
    for data_type in data_types:
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Filter data for this data type
        df_type = df[df['data_type'] == data_type].copy()
        df_type = df_type.dropna(subset=['kernel_smoothness', 'extrinsic_diameter'])
        
        if len(df_type) == 0:
            logger.warning(f"No data for {data_type}")
            plt.close()
            continue
        
        # Group by other parameters to create different lines
        # Each line represents datasets with same (d, D, k) but different σ
        grouping_cols = ['true_d', 'true_D', 'true_k']
        grouped = df_type.groupby(grouping_cols)
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(grouped)))
        
        plotted_theory = set()  # Keep track of theoretical curves plotted
        
        for i, ((d, D, k), group) in enumerate(grouped):
            if pd.isna(d) or pd.isna(D):
                continue
            
            # Sort by kernel_smoothness for line plot
            group = group.sort_values('kernel_smoothness')
            
            if len(group) > 1:  # Only plot if we have multiple points
                # Plot empirical data
                ax.plot(group['kernel_smoothness'], group['extrinsic_diameter'], 
                       'o-', color=colors[i], 
                       label=f'd={int(d)}, D={int(D)}, k={int(k)} (empirical)',
                       linewidth=2, markersize=6)
                
                # Plot theoretical curve if not already plotted for this (d,D)
                theory_key = (d, D)
                if theory_key not in plotted_theory:
                    # Generate theoretical curve
                    sigma_range = np.logspace(
                        np.log10(group['kernel_smoothness'].min()),
                        np.log10(group['kernel_smoothness'].max()),
                        50
                    )
                    theoretical_diameters = [
                        theoretical_expected_diameter(d, D, sigma) 
                        for sigma in sigma_range
                    ]
                    
                    ax.plot(sigma_range, theoretical_diameters, 
                           '--', color=colors[i], alpha=0.7,
                           label=f'd={int(d)}, D={int(D)} (theoretical)')
                    plotted_theory.add(theory_key)
        
        ax.set_xlabel('Kernel Smoothness (σ)', fontsize=12)
        ax.set_ylabel('Extrinsic Diameter', fontsize=12)
        ax.set_title(f'Diameter vs Kernel Smoothness - {data_type.replace("_", " ").title()}', 
                    fontsize=14)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        save_path = output_dir / f'diameter_vs_kernel_smoothness_{data_type}.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"  ✓ Saved: {save_path}")
        plt.close()


def create_kernel_smoothness_vs_dimension_plot(
    df: pd.DataFrame,
    output_dir: Path,
    logger: logging.Logger
) -> None:
    """Create kernel_smoothness vs estimated dimensionality plots."""
    
    logger.info("Creating kernel_smoothness vs dimensionality plots...")
    
    # Dimension estimation methods
    dim_methods = ['dim_PCA', 'dim_kNN', 'dim_TwoNN', 'dim_MLE']
    method_names = ['PCA', 'k-NN', 'TwoNN', 'MLE']
    
    data_types = df['data_type'].unique()
    data_types = [dt for dt in data_types if pd.notna(dt)]
    
    for data_type in data_types:
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
        
        df_type = df[df['data_type'] == data_type].copy()
        
        for i, (method, method_name) in enumerate(zip(dim_methods, method_names)):
            ax = axes[i]
            
            # Filter valid data
            df_method = df_type.dropna(subset=['kernel_smoothness', method, 'true_d'])
            
            if len(df_method) == 0:
                ax.text(0.5, 0.5, f'No data for {method_name}', 
                       ha='center', va='center', transform=ax.transAxes)
                continue
            
            # Group by (d, D, k) for different lines
            grouping_cols = ['true_d', 'true_D', 'true_k']
            grouped = df_method.groupby(grouping_cols)
            
            colors = plt.cm.tab10(np.linspace(0, 1, len(grouped)))
            
            for j, ((d, D, k), group) in enumerate(grouped):
                if pd.isna(d) or pd.isna(D):
                    continue
                
                group = group.sort_values('kernel_smoothness')
                
                if len(group) > 1:
                    # Plot estimated dimensions
                    ax.plot(group['kernel_smoothness'], group[method], 
                           'o-', color=colors[j], 
                           label=f'd={int(d)}, D={int(D)}, k={int(k)}',
                           linewidth=2, markersize=6)
                    
                    # Plot true dimension as horizontal line
                    ax.axhline(y=d, color=colors[j], linestyle=':', alpha=0.7)
            
            ax.set_xlabel('Kernel Smoothness (σ)', fontsize=10)
            ax.set_ylabel(f'Estimated Dimension ({method_name})', fontsize=10)
            ax.set_title(f'{method_name} Dimension Estimation vs σ', fontsize=12)
            ax.set_xscale('log')
            ax.grid(True, alpha=0.3)
            
            if i == 0:  # Only show legend for first subplot
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        
        plt.suptitle(f'Dimension Estimation vs Kernel Smoothness - {data_type.replace("_", " ").title()}', 
                    fontsize=16)
        plt.tight_layout()
        save_path = output_dir / f'dimensions_vs_kernel_smoothness_{data_type}.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"  ✓ Saved: {save_path}")
        plt.close()


def create_kernel_smoothness_vs_curvature_plot(
    df: pd.DataFrame,
    output_dir: Path,
    logger: logging.Logger
) -> None:
    """Create kernel_smoothness vs curvature plots with theoretical curves."""
    
    logger.info("Creating kernel_smoothness vs curvature plots...")
    
    data_types = df['data_type'].unique()
    data_types = [dt for dt in data_types if pd.notna(dt)]
    
    for data_type in data_types:
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        df_type = df[df['data_type'] == data_type].copy()
        
        # Plot 1: Mean curvature
        ax = axes[0]
        df_curv = df_type.dropna(subset=['kernel_smoothness', 'mean_curvature'])
        
        if len(df_curv) > 0:
            grouping_cols = ['true_d', 'true_D', 'true_k']
            grouped = df_curv.groupby(grouping_cols)
            colors = plt.cm.tab10(np.linspace(0, 1, len(grouped)))
            
            plotted_theory = set()
            
            for i, ((d, D, k), group) in enumerate(grouped):
                if pd.isna(d):
                    continue
                
                group = group.sort_values('kernel_smoothness')
                
                if len(group) > 1:
                    # Plot empirical curvature
                    ax.plot(group['kernel_smoothness'], group['mean_curvature'], 
                           'o-', color=colors[i], 
                           label=f'd={int(d)}, D={int(D)}, k={int(k)} (empirical)',
                           linewidth=2, markersize=6)
                    
                    # Plot theoretical curve
                    if d not in plotted_theory:
                        sigma_range = np.logspace(
                            np.log10(group['kernel_smoothness'].min()),
                            np.log10(group['kernel_smoothness'].max()),
                            50
                        )
                        theoretical_curv = [
                            theoretical_curvature_squared(d, sigma) 
                            for sigma in sigma_range
                        ]
                        
                        ax.plot(sigma_range, theoretical_curv, 
                               '--', color=colors[i], alpha=0.7,
                               label=f'd={int(d)} (theoretical)')
                        plotted_theory.add(d)
        
        ax.set_xlabel('Kernel Smoothness (σ)', fontsize=12)
        ax.set_ylabel('Mean Curvature', fontsize=12)
        ax.set_title('Mean Curvature vs σ', fontsize=14)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        
        # Plot 2: PCA curvature
        ax = axes[1]
        df_pca = df_type.dropna(subset=['kernel_smoothness', 'pca_curvature'])
        
        if len(df_pca) > 0:
            grouped = df_pca.groupby(grouping_cols)
            colors = plt.cm.tab10(np.linspace(0, 1, len(grouped)))
            
            for i, ((d, D, k), group) in enumerate(grouped):
                if pd.isna(d):
                    continue
                
                group = group.sort_values('kernel_smoothness')
                
                if len(group) > 1:
                    ax.plot(group['kernel_smoothness'], group['pca_curvature'], 
                           'o-', color=colors[i], 
                           label=f'd={int(d)}, D={int(D)}, k={int(k)}',
                           linewidth=2, markersize=6)
        
        ax.set_xlabel('Kernel Smoothness (σ)', fontsize=12)
        ax.set_ylabel('PCA-based Curvature', fontsize=12)
        ax.set_title('PCA Curvature vs σ', fontsize=14)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        
        plt.suptitle(f'Curvature vs Kernel Smoothness - {data_type.replace("_", " ").title()}', 
                    fontsize=16)
        plt.tight_layout()
        save_path = output_dir / f'curvature_vs_kernel_smoothness_{data_type}.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"  ✓ Saved: {save_path}")
        plt.close()


def create_parameter_overview_plot(
    df: pd.DataFrame,
    output_dir: Path,
    logger: logging.Logger
) -> None:
    """Create overview plot showing parameter distributions."""
    
    logger.info("Creating parameter overview plot...")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    # Plot parameter distributions
    params = ['kernel_smoothness', 'true_d', 'true_D', 'true_k', 'noise_sigma']
    param_names = ['Kernel Smoothness (σ)', 'Intrinsic Dim (d)', 
                   'Embedding Dim (D)', 'GP parameter (k)', 'Noise Level']
    
    for i, (param, name) in enumerate(zip(params, param_names)):
        ax = axes[i]
        
        if param in df.columns:
            values = df[param].dropna()
            if len(values) > 0:
                ax.hist(values, bins=20, alpha=0.7, edgecolor='black')
                ax.set_xlabel(name)
                ax.set_ylabel('Frequency')
                ax.set_title(f'{name} Distribution')
                ax.grid(True, alpha=0.3)
        
    # Summary statistics in last subplot
    ax = axes[5]
    ax.axis('off')
    
    summary_text = f"""
    Dataset Summary:
    Total datasets: {len(df['dataset'].unique()) if 'dataset' in df.columns else 'N/A'}
    Data types: {len(df['data_type'].unique())}
    
    Parameter ranges:
    σ: {df['kernel_smoothness'].min():.3f} - {df['kernel_smoothness'].max():.3f}
    d: {df['true_d'].min():.0f} - {df['true_d'].max():.0f}
    D: {df['true_D'].min():.0f} - {df['true_D'].max():.0f}
    """
    
    ax.text(0.1, 0.7, summary_text, fontsize=12, verticalalignment='top',
           bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
    
    plt.suptitle('Dataset Parameter Overview', fontsize=16)
    plt.tight_layout()
    save_path = output_dir / 'parameter_overview.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    logger.info(f"  ✓ Saved: {save_path}")
    plt.close()


def main():
    """Main plotting pipeline."""
    parser = argparse.ArgumentParser(description="Step 3: Kernel Smoothness vs Geometry Plotting")
    parser.add_argument("--results_dir", type=Path,
                       default=Path("results/step2_comprehensive_test"),
                       help="Directory containing analysis results")
    parser.add_argument("--output_dir", type=Path,
                       default=Path("plots/step3_kernel_smoothness_analysis"),
                       help="Output directory for plots")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Setup
    logger = setup_logging(args.verbose)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("Starting Step 3: Kernel Smoothness vs Geometry Plotting Pipeline")
    logger.info(f"Results directory: {args.results_dir}")
    logger.info(f"Output directory: {args.output_dir}")
    
    # Load data
    try:
        df = load_comprehensive_results(args.results_dir)
        logger.info(f"Loaded {len(df)} analysis results")
        
        # Show basic statistics
        logger.info(f"Data types: {df['data_type'].unique()}")
        logger.info(f"Kernel smoothness range: {df['kernel_smoothness'].min():.4f} - {df['kernel_smoothness'].max():.4f}")
        logger.info(f"Dimensions: d={df['true_d'].unique()}, D={df['true_D'].unique()}")
        
    except Exception as e:
        logger.error(f"Failed to load results: {e}")
        return
    
    # Create plots
    try:
        logger.info("\n[1/4] Creating kernel_smoothness vs diameter plots...")
        create_kernel_smoothness_vs_diameter_plot(df, args.output_dir, logger)
        
        logger.info("\n[2/4] Creating kernel_smoothness vs dimension estimation plots...")
        create_kernel_smoothness_vs_dimension_plot(df, args.output_dir, logger)
        
        logger.info("\n[3/4] Creating kernel_smoothness vs curvature plots...")
        create_kernel_smoothness_vs_curvature_plot(df, args.output_dir, logger)
        
        logger.info("\n[4/4] Creating parameter overview...")
        create_parameter_overview_plot(df, args.output_dir, logger)
        
        logger.info(f"\n{'='*60}")
        logger.info("PLOTTING COMPLETE!")
        logger.info(f"All plots saved to: {args.output_dir}")
        logger.info(f"{'='*60}")
        
    except Exception as e:
        logger.error(f"Error during plotting: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
