#!/usr/bin/env python3
"""
Curvature Analysis Results Interpreter

This script analyzes the results from curvature_neighborhood_analysis.py
and provides insights about optimal neighborhood sizes and curvature patterns.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse


def analyze_curvature_stability(df: pd.DataFrame) -> dict:
    """Analyze stability of curvature estimates across neighborhood sizes."""
    results = {}
    
    for dataset in df['dataset'].unique():
        dataset_df = df[df['dataset'] == dataset]
        
        # Mean curvature stability
        mc_vals = dataset_df['mean_curvature_mean'].dropna()
        if len(mc_vals) > 1:
            mc_cv = mc_vals.std() / mc_vals.mean() if mc_vals.mean() > 0 else np.inf
            mc_trend = np.polyfit(dataset_df.loc[mc_vals.index, 'k_neighbors'], mc_vals, 1)[0]
        else:
            mc_cv = np.nan
            mc_trend = np.nan
        
        # PCA curvature stability
        pc_vals = dataset_df['pca_curvature_mean'].dropna()
        if len(pc_vals) > 1:
            pc_cv = pc_vals.std() / pc_vals.mean() if pc_vals.mean() > 0 else np.inf
            pc_trend = np.polyfit(dataset_df.loc[pc_vals.index, 'k_neighbors'], pc_vals, 1)[0]
        else:
            pc_cv = np.nan
            pc_trend = np.nan
        
        # Success rate analysis
        success_rates = dataset_df['mean_curvature_success_rate'].dropna()
        min_k_for_success = dataset_df[dataset_df['mean_curvature_success_rate'] > 0]['k_neighbors'].min() if len(success_rates[success_rates > 0]) > 0 else np.nan
        
        results[dataset] = {
            'mean_curvature_cv': mc_cv,
            'mean_curvature_trend': mc_trend,
            'pca_curvature_cv': pc_cv,
            'pca_curvature_trend': pc_trend,
            'min_k_for_success': min_k_for_success,
            'intrinsic_dim': dataset_df['intrinsic_dim'].iloc[0],
            'diameter': dataset_df['diameter'].iloc[0]
        }
    
    return results


def find_optimal_k_values(df: pd.DataFrame, stability_results: dict) -> dict:
    """Find optimal k values for each dataset based on stability and success rate."""
    optimal_k = {}
    
    for dataset in df['dataset'].unique():
        dataset_df = df[df['dataset'] == dataset]
        
        # Find k values with high success rate (>= 0.9)
        high_success = dataset_df[dataset_df['mean_curvature_success_rate'] >= 0.9]
        
        if len(high_success) > 0:
            # Among high success rates, find the one with most stable estimates
            if len(high_success) > 1:
                # Calculate local stability (difference between consecutive k values)
                mc_vals = high_success['mean_curvature_mean'].dropna()
                if len(mc_vals) > 1:
                    # Find k where curvature estimate is most stable (smallest relative change)
                    stability_scores = []
                    for i in range(1, len(mc_vals)):
                        rel_change = abs(mc_vals.iloc[i] - mc_vals.iloc[i-1]) / mc_vals.iloc[i-1]
                        stability_scores.append(rel_change)
                    
                    if stability_scores:
                        most_stable_idx = np.argmin(stability_scores) + 1  # +1 because we start from index 1
                        optimal_k_val = mc_vals.index[most_stable_idx]
                        optimal_k[dataset] = high_success.loc[optimal_k_val, 'k_neighbors']
                    else:
                        optimal_k[dataset] = high_success['k_neighbors'].iloc[0]
                else:
                    optimal_k[dataset] = high_success['k_neighbors'].iloc[0]
            else:
                optimal_k[dataset] = high_success['k_neighbors'].iloc[0]
        else:
            # No high success rate, choose PCA-based optimal k
            pc_vals = dataset_df['pca_curvature_mean'].dropna()
            if len(pc_vals) > 1:
                # Find k where PCA curvature is most stable
                pc_changes = []
                for i in range(1, len(pc_vals)):
                    rel_change = abs(pc_vals.iloc[i] - pc_vals.iloc[i-1]) / pc_vals.iloc[i-1]
                    pc_changes.append(rel_change)
                
                if pc_changes:
                    most_stable_idx = np.argmin(pc_changes) + 1
                    optimal_k_val = pc_vals.index[most_stable_idx]
                    optimal_k[dataset] = dataset_df.loc[optimal_k_val, 'k_neighbors']
                else:
                    optimal_k[dataset] = dataset_df['k_neighbors'].iloc[0]
            else:
                optimal_k[dataset] = dataset_df['k_neighbors'].iloc[0] if len(dataset_df) > 0 else np.nan
    
    return optimal_k


def create_analysis_report(df: pd.DataFrame, output_dir: Path) -> None:
    """Create comprehensive analysis report."""
    stability_results = analyze_curvature_stability(df)
    optimal_k_values = find_optimal_k_values(df, stability_results)
    
    report_path = output_dir / "curvature_analysis_report.txt"
    
    with open(report_path, 'w') as f:
        f.write("Curvature Neighborhood Analysis Report\n")
        f.write("=" * 40 + "\n\n")
        
        f.write("SUMMARY STATISTICS\n")
        f.write("-" * 20 + "\n")
        f.write(f"Total datasets analyzed: {df['dataset'].nunique()}\n")
        f.write(f"Neighborhood size range: {df['k_neighbors'].min()} - {df['k_neighbors'].max()}\n")
        f.write(f"Average diameter: {df.groupby('dataset')['diameter'].first().mean():.3f}\n\n")
        
        f.write("OPTIMAL NEIGHBORHOOD SIZES\n")
        f.write("-" * 30 + "\n")
        for dataset, k_opt in optimal_k_values.items():
            intrinsic_dim = stability_results[dataset]['intrinsic_dim']
            diameter = stability_results[dataset]['diameter']
            f.write(f"{dataset}: k={k_opt:.0f} (d={intrinsic_dim}, diameter={diameter:.2f})\n")
        
        f.write(f"\nAverage optimal k: {np.nanmean(list(optimal_k_values.values())):.1f}\n")
        f.write(f"Std optimal k: {np.nanstd(list(optimal_k_values.values())):.1f}\n\n")
        
        f.write("STABILITY ANALYSIS\n")
        f.write("-" * 20 + "\n")
        f.write("Coefficient of Variation (lower = more stable):\n")
        
        mc_cvs = [r['mean_curvature_cv'] for r in stability_results.values() if not np.isnan(r['mean_curvature_cv'])]
        pc_cvs = [r['pca_curvature_cv'] for r in stability_results.values() if not np.isnan(r['pca_curvature_cv'])]
        
        if mc_cvs:
            f.write(f"Mean Curvature CV: {np.mean(mc_cvs):.3f} ± {np.std(mc_cvs):.3f}\n")
        if pc_cvs:
            f.write(f"PCA Curvature CV: {np.mean(pc_cvs):.3f} ± {np.std(pc_cvs):.3f}\n")
        
        f.write("\nTREND ANALYSIS\n")
        f.write("-" * 15 + "\n")
        f.write("Slope of curvature vs k_neighbors (positive = increasing):\n")
        
        mc_trends = [r['mean_curvature_trend'] for r in stability_results.values() if not np.isnan(r['mean_curvature_trend'])]
        pc_trends = [r['pca_curvature_trend'] for r in stability_results.values() if not np.isnan(r['pca_curvature_trend'])]
        
        if mc_trends:
            f.write(f"Mean Curvature trend: {np.mean(mc_trends):.4f} ± {np.std(mc_trends):.4f}\n")
        if pc_trends:
            f.write(f"PCA Curvature trend: {np.mean(pc_trends):.4f} ± {np.std(pc_trends):.4f}\n")
        
        f.write("\nSUCCESS RATE ANALYSIS\n")
        f.write("-" * 22 + "\n")
        min_k_values = [r['min_k_for_success'] for r in stability_results.values() if not np.isnan(r['min_k_for_success'])]
        if min_k_values:
            f.write(f"Minimum k for successful estimation: {np.min(min_k_values):.0f}\n")
            f.write(f"Average minimum k: {np.mean(min_k_values):.1f}\n")
        
        f.write("\nRECOMMENDATIONS\n")
        f.write("-" * 15 + "\n")
        
        if optimal_k_values:
            avg_opt_k = np.nanmean(list(optimal_k_values.values()))
            f.write(f"1. Recommended default k_neighbors: {avg_opt_k:.0f}\n")
            
            # Relationship to intrinsic dimension
            dims = [stability_results[d]['intrinsic_dim'] for d in optimal_k_values.keys()]
            k_vals = [optimal_k_values[d] for d in optimal_k_values.keys()]
            if len(dims) > 1 and not any(np.isnan([k for k in k_vals])):
                correlation = np.corrcoef(dims, k_vals)[0, 1]
                f.write(f"2. Correlation with intrinsic dimension: {correlation:.3f}\n")
                
                # Rule of thumb
                k_per_dim = np.array(k_vals) / np.array(dims)
                avg_k_per_dim = np.nanmean(k_per_dim)
                f.write(f"3. Rule of thumb: k ≈ {avg_k_per_dim:.1f} × intrinsic_dimension\n")
        
        f.write("\nDETAILED RESULTS BY DATASET\n")
        f.write("-" * 30 + "\n")
        for dataset, results in stability_results.items():
            f.write(f"\n{dataset}:\n")
            f.write(f"  Intrinsic dimension: {results['intrinsic_dim']}\n")
            f.write(f"  Diameter: {results['diameter']:.3f}\n")
            f.write(f"  Optimal k: {optimal_k_values.get(dataset, 'N/A')}\n")
            f.write(f"  Mean curvature CV: {results['mean_curvature_cv']:.3f}\n")
            f.write(f"  PCA curvature CV: {results['pca_curvature_cv']:.3f}\n")
            f.write(f"  Min k for success: {results['min_k_for_success']}\n")
    
    print(f"Analysis report saved to: {report_path}")


def main():
    parser = argparse.ArgumentParser(description="Analyze curvature neighborhood results")
    parser.add_argument("results_dir", type=Path, help="Directory containing analysis results")
    
    args = parser.parse_args()
    
    # Load results
    csv_path = args.results_dir / "curvature_neighborhood_summary.csv"
    if not csv_path.exists():
        print(f"Error: Could not find {csv_path}")
        return
    
    df = pd.read_csv(csv_path)
    
    # Create analysis report
    create_analysis_report(df, args.results_dir)
    
    # Create additional plots
    plt.style.use('default')
    
    # Plot 1: Optimal k vs intrinsic dimension
    stability_results = analyze_curvature_stability(df)
    optimal_k_values = find_optimal_k_values(df, stability_results)
    
    dims = [stability_results[d]['intrinsic_dim'] for d in optimal_k_values.keys()]
    k_vals = [optimal_k_values[d] for d in optimal_k_values.keys() if not np.isnan(optimal_k_values[d])]
    dims = [dims[i] for i in range(len(dims)) if i < len(k_vals) and not np.isnan(k_vals[i])]
    
    if len(dims) > 1:
        plt.figure(figsize=(10, 6))
        plt.scatter(dims, k_vals, alpha=0.7, s=60)
        
        # Fit line
        if len(dims) > 2:
            z = np.polyfit(dims, k_vals, 1)
            p = np.poly1d(z)
            x_line = np.linspace(min(dims), max(dims), 100)
            plt.plot(x_line, p(x_line), "r--", alpha=0.8, label=f'Trend: k = {z[0]:.1f}×d + {z[1]:.1f}')
        
        plt.xlabel('Intrinsic Dimension')
        plt.ylabel('Optimal k_neighbors')
        plt.title('Optimal Neighborhood Size vs Intrinsic Dimension')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(args.results_dir / "plots" / "optimal_k_vs_dimension.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"Additional analysis plots saved to: {args.results_dir / 'plots'}")


if __name__ == "__main__":
    main()
