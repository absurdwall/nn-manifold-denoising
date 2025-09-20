#!/usr/bin/env python3
"""
Curvature Neighborhood Analysis

This script analyzes how curvature estimates vary with different neighborhood sizes
for clean manifold datasets. It focuses only on clean data and produces plots
showing curvature vs neighborhood size for each dataset.

Usage:
    python scripts/curvature_neighborhood_analysis.py [--data_dir DATA_DIR] [--output_dir OUTPUT_DIR]
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
from tqdm import tqdm

# Add src to path
import sys
sys.path.append(str(Path(__file__).parent.parent / "src"))

from geometric_analysis import (
    estimate_mean_curvature,
    estimate_curvature_pca_based,
    compute_curvature_statistics
)


def setup_logging(verbose: bool = False) -> logging.Logger:
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    return logging.getLogger(__name__)


def load_dataset_from_folder(dataset_dir: Path) -> Dict[str, Any]:
    """
    Load a dataset from the step1 output folder structure.
    
    Args:
        dataset_dir: Path to dataset folder (e.g., dataset0/)
        
    Returns:
        Dictionary containing intrinsic_coords, clean_coords, and metadata
    """
    # Load the required data files
    intrinsic_file = dataset_dir / f"{dataset_dir.name}_intrinsic.npy"
    clean_file = dataset_dir / f"{dataset_dir.name}_clean.npy"
    metadata_file = dataset_dir / f"{dataset_dir.name}_metadata.json"
    
    if not intrinsic_file.exists():
        raise FileNotFoundError(f"Intrinsic coords file not found: {intrinsic_file}")
    if not clean_file.exists():
        raise FileNotFoundError(f"Clean coords file not found: {clean_file}")
    
    # Load coordinate data
    intrinsic_coords = np.load(intrinsic_file)
    clean_coords = np.load(clean_file)
    
    # Load metadata if available
    metadata = {}
    if metadata_file.exists():
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
    
    return {
        "intrinsic_coords": intrinsic_coords,
        "clean_coords": clean_coords,
        "metadata": metadata
    }


def analyze_curvature_vs_neighborhood(
    intrinsic_coords: np.ndarray,
    clean_coords: np.ndarray,
    dataset_info: Dict[str, Any],
    k_neighbors_range: List[int],
    max_points: int = 1000,
    logger: logging.Logger = None
) -> Dict[str, Any]:
    """
    Analyze curvature estimates across different neighborhood sizes.
    
    Args:
        intrinsic_coords: (N, d) intrinsic coordinates
        clean_coords: (N, D) clean embedded coordinates
        dataset_info: Dataset metadata
        k_neighbors_range: List of neighborhood sizes to test
        max_points: Maximum points to use for curvature computation
        logger: Logger instance
        
    Returns:
        Dictionary containing results for all neighborhood sizes
    """
    if logger is None:
        logger = logging.getLogger(__name__)
        
    dataset_name = dataset_info.get('name', 'unknown')
    logger.info(f"Analyzing curvature vs neighborhood size for {dataset_name}")
    
    N, d = intrinsic_coords.shape
    _, D = clean_coords.shape
    
    # Subsample if needed for efficiency
    if N > max_points:
        indices = np.random.choice(N, size=max_points, replace=False)
        intrinsic_sample = intrinsic_coords[indices]
        clean_sample = clean_coords[indices]
        logger.info(f"Subsampled to {max_points} points for analysis")
    else:
        intrinsic_sample = intrinsic_coords
        clean_sample = clean_coords
    
    results = {
        "dataset_info": dataset_info,
        "basic_info": {
            "num_points_total": N,
            "num_points_analyzed": len(clean_sample),
            "intrinsic_dimension": d,
            "embedding_dimension": D,
            "diameter_estimate": np.max(np.linalg.norm(clean_sample - clean_sample.mean(axis=0), axis=1)) * 2
        },
        "neighborhood_analysis": {}
    }
    
    # Test each neighborhood size
    for k in tqdm(k_neighbors_range, desc=f"Testing k_neighbors for {dataset_name}", leave=False):
        if k >= len(clean_sample):
            logger.warning(f"Skipping k={k} (too large for dataset with {len(clean_sample)} points)")
            continue
            
        k_results = {"k_neighbors": k}
        
        # Mean curvature estimation
        try:
            logger.debug(f"Computing mean curvature with k={k}")
            mean_curvatures = estimate_mean_curvature(
                intrinsic_sample,
                clean_sample,
                k_neighbors=k,
                max_points=None,  # Already subsampled
                weight="gaussian",
                regularization=1e-10,
                fallback_on_failure=True
            )
            
            mean_curv_stats = compute_curvature_statistics(mean_curvatures)
            k_results["mean_curvature"] = mean_curv_stats
            k_results["mean_curvature_values"] = mean_curvatures.tolist()
            
            logger.debug(f"k={k}, mean curvature: mean={mean_curv_stats['mean']:.6f}, "
                        f"success_rate={mean_curv_stats['success_rate']:.2f}")
            
        except Exception as e:
            logger.warning(f"Mean curvature failed for k={k}: {e}")
            k_results["mean_curvature"] = {"error": str(e)}
        
        # PCA-based curvature estimation (with timeout protection)
        try:
            logger.debug(f"Computing PCA curvature with k={k}")
            
            # For very large k, subsample further to avoid computational issues
            if k > 40 and len(clean_sample) > 300:
                pca_sample_size = min(300, len(clean_sample))
                pca_indices = np.random.choice(len(clean_sample), size=pca_sample_size, replace=False)
                pca_sample = clean_sample[pca_indices]
                logger.debug(f"Subsampled to {pca_sample_size} points for PCA curvature with k={k}")
            else:
                pca_sample = clean_sample
            
            pca_curvatures = estimate_curvature_pca_based(
                pca_sample,
                k_neighbors=min(k, len(pca_sample) - 1),
                intrinsic_dim=d
            )
            
            pca_curv_stats = compute_curvature_statistics(pca_curvatures)
            k_results["pca_curvature"] = pca_curv_stats
            
            logger.debug(f"k={k}, PCA curvature: mean={pca_curv_stats['mean']:.6f}")
            
        except Exception as e:
            logger.warning(f"PCA curvature failed for k={k}: {e}")
            k_results["pca_curvature"] = {"error": str(e)}
        
        results["neighborhood_analysis"][k] = k_results
    
    return results


def create_curvature_plots(
    all_results: List[Dict[str, Any]],
    output_dir: Path,
    logger: logging.Logger
) -> None:
    """
    Create comprehensive plots showing curvature vs neighborhood size.
    
    Args:
        all_results: List of analysis results for all datasets
        output_dir: Directory to save plots
        logger: Logger instance
    """
    logger.info("Creating curvature vs neighborhood plots...")
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(exist_ok=True)
    
    # Extract data for plotting
    plot_data = []
    
    for result in all_results:
        if "error" in result:
            continue
            
        dataset_name = result["dataset_info"].get("name", "unknown")
        basic_info = result["basic_info"]
        
        for k, k_result in result["neighborhood_analysis"].items():
            row = {
                "dataset": dataset_name,
                "k": result["dataset_info"].get("k", "unknown"),
                "k_neighbors": k_result["k_neighbors"],
                "num_points": basic_info["num_points_analyzed"],
                "intrinsic_dim": basic_info["intrinsic_dimension"],
                "embedding_dim": basic_info["embedding_dimension"],
                "diameter": basic_info["diameter_estimate"]
            }
            
            # Mean curvature stats
            if "mean_curvature" in k_result and "error" not in k_result["mean_curvature"]:
                mc = k_result["mean_curvature"]
                row.update({
                    "mean_curvature_mean": mc["mean"],
                    "mean_curvature_std": mc["std"],
                    "mean_curvature_median": mc["median"],
                    "mean_curvature_success_rate": mc["success_rate"]
                })
            else:
                row.update({
                    "mean_curvature_mean": np.nan,
                    "mean_curvature_std": np.nan,
                    "mean_curvature_median": np.nan,
                    "mean_curvature_success_rate": 0.0
                })
            
            # PCA curvature stats
            if "pca_curvature" in k_result and "error" not in k_result["pca_curvature"]:
                pc = k_result["pca_curvature"]
                row.update({
                    "pca_curvature_mean": pc["mean"],
                    "pca_curvature_std": pc["std"],
                    "pca_curvature_median": pc["median"]
                })
            else:
                row.update({
                    "pca_curvature_mean": np.nan,
                    "pca_curvature_std": np.nan,
                    "pca_curvature_median": np.nan
                })
            
            plot_data.append(row)
    
    if not plot_data:
        logger.warning("No valid data for plotting")
        return
    
    df = pd.DataFrame(plot_data)
    
    # Set up plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Plot 1: Mean Curvature vs k_neighbors (separate plot for each dataset)
    datasets = df['dataset'].unique()
    
    for dataset in datasets:
        dataset_df = df[df['dataset'] == dataset]
        if len(dataset_df) == 0:
            continue
            
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Curvature Analysis vs Neighborhood Size - {dataset}', fontsize=16)
        
        # Mean curvature mean
        valid_data = dataset_df.dropna(subset=['mean_curvature_mean'])
        if len(valid_data) > 0:
            axes[0, 0].plot(valid_data['k_neighbors'], valid_data['mean_curvature_mean'], 
                           'o-', linewidth=2, markersize=6)
            axes[0, 0].set_xlabel('k_neighbors')
            axes[0, 0].set_ylabel('Mean Curvature (Mean)')
            axes[0, 0].set_title('Mean Curvature vs Neighborhood Size')
            axes[0, 0].grid(True, alpha=0.3)
        
        # Mean curvature success rate
        valid_data = dataset_df.dropna(subset=['mean_curvature_success_rate'])
        if len(valid_data) > 0:
            axes[0, 1].plot(valid_data['k_neighbors'], valid_data['mean_curvature_success_rate'], 
                           'o-', linewidth=2, markersize=6, color='orange')
            axes[0, 1].set_xlabel('k_neighbors')
            axes[0, 1].set_ylabel('Success Rate')
            axes[0, 1].set_title('Mean Curvature Success Rate')
            axes[0, 1].set_ylim(0, 1.1)
            axes[0, 1].grid(True, alpha=0.3)
        
        # PCA curvature mean
        valid_data = dataset_df.dropna(subset=['pca_curvature_mean'])
        if len(valid_data) > 0:
            axes[1, 0].plot(valid_data['k_neighbors'], valid_data['pca_curvature_mean'], 
                           'o-', linewidth=2, markersize=6, color='green')
            axes[1, 0].set_xlabel('k_neighbors')
            axes[1, 0].set_ylabel('PCA Curvature (Mean)')
            axes[1, 0].set_title('PCA-based Curvature vs Neighborhood Size')
            axes[1, 0].grid(True, alpha=0.3)
        
        # Comparison plot
        valid_mean = dataset_df.dropna(subset=['mean_curvature_mean'])
        valid_pca = dataset_df.dropna(subset=['pca_curvature_mean'])
        
        if len(valid_mean) > 0:
            axes[1, 1].plot(valid_mean['k_neighbors'], valid_mean['mean_curvature_mean'], 
                           'o-', linewidth=2, markersize=6, label='Mean Curvature', alpha=0.8)
        if len(valid_pca) > 0:
            axes[1, 1].plot(valid_pca['k_neighbors'], valid_pca['pca_curvature_mean'], 
                           'o-', linewidth=2, markersize=6, label='PCA Curvature', alpha=0.8)
        
        axes[1, 1].set_xlabel('k_neighbors')
        axes[1, 1].set_ylabel('Curvature Estimate')
        axes[1, 1].set_title('Method Comparison')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(plot_dir / f"curvature_vs_k_{dataset}.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    # Plot 2: Summary plot across all datasets
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Curvature vs Neighborhood Size - All Datasets', fontsize=16)
    
    # Mean curvature across datasets
    for dataset in datasets:
        dataset_df = df[df['dataset'] == dataset]
        valid_data = dataset_df.dropna(subset=['mean_curvature_mean'])
        if len(valid_data) > 0:
            axes[0, 0].plot(valid_data['k_neighbors'], valid_data['mean_curvature_mean'], 
                           'o-', label=f'{dataset}', alpha=0.7)
    
    axes[0, 0].set_xlabel('k_neighbors')
    axes[0, 0].set_ylabel('Mean Curvature (Mean)')
    axes[0, 0].set_title('Mean Curvature vs k_neighbors')
    axes[0, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Success rate across datasets
    for dataset in datasets:
        dataset_df = df[df['dataset'] == dataset]
        valid_data = dataset_df.dropna(subset=['mean_curvature_success_rate'])
        if len(valid_data) > 0:
            axes[0, 1].plot(valid_data['k_neighbors'], valid_data['mean_curvature_success_rate'], 
                           'o-', label=f'{dataset}', alpha=0.7)
    
    axes[0, 1].set_xlabel('k_neighbors')
    axes[0, 1].set_ylabel('Success Rate')
    axes[0, 1].set_title('Success Rate vs k_neighbors')
    axes[0, 1].set_ylim(0, 1.1)
    axes[0, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[0, 1].grid(True, alpha=0.3)
    
    # PCA curvature across datasets
    for dataset in datasets:
        dataset_df = df[df['dataset'] == dataset]
        valid_data = dataset_df.dropna(subset=['pca_curvature_mean'])
        if len(valid_data) > 0:
            axes[1, 0].plot(valid_data['k_neighbors'], valid_data['pca_curvature_mean'], 
                           'o-', label=f'{dataset}', alpha=0.7)
    
    axes[1, 0].set_xlabel('k_neighbors')
    axes[1, 0].set_ylabel('PCA Curvature (Mean)')
    axes[1, 0].set_title('PCA Curvature vs k_neighbors')
    axes[1, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Coefficient of variation (stability measure)
    cv_data = []
    for dataset in datasets:
        dataset_df = df[df['dataset'] == dataset]
        
        # Calculate CV for mean curvature
        valid_data = dataset_df.dropna(subset=['mean_curvature_mean', 'mean_curvature_std'])
        if len(valid_data) > 1:
            mean_vals = valid_data['mean_curvature_mean'].values
            if np.mean(mean_vals) > 1e-12:
                cv_mean = np.std(mean_vals) / np.mean(mean_vals)
                cv_data.append({"dataset": dataset, "method": "Mean Curvature", "cv": cv_mean})
        
        # Calculate CV for PCA curvature
        valid_data = dataset_df.dropna(subset=['pca_curvature_mean', 'pca_curvature_std'])
        if len(valid_data) > 1:
            pca_vals = valid_data['pca_curvature_mean'].values
            if np.mean(pca_vals) > 1e-12:
                cv_pca = np.std(pca_vals) / np.mean(pca_vals)
                cv_data.append({"dataset": dataset, "method": "PCA Curvature", "cv": cv_pca})
    
    if cv_data:
        cv_df = pd.DataFrame(cv_data)
        sns.barplot(data=cv_df, x='dataset', y='cv', hue='method', ax=axes[1, 1])
        axes[1, 1].set_title('Coefficient of Variation (Stability)')
        axes[1, 1].set_ylabel('CV (lower = more stable)')
        axes[1, 1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(plot_dir / "curvature_vs_k_summary.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Plots saved to {plot_dir}")


def save_results(
    all_results: List[Dict[str, Any]],
    output_dir: Path,
    logger: logging.Logger
) -> None:
    """Save analysis results to files."""
    logger.info("Saving results...")
    
    # Save detailed results as JSON
    with open(output_dir / "curvature_neighborhood_analysis.json", 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    # Create summary CSV
    summary_data = []
    for result in all_results:
        if "error" in result:
            continue
        
        dataset_name = result["dataset_info"].get("name", "unknown")
        basic_info = result["basic_info"]
        
        for k, k_result in result["neighborhood_analysis"].items():
            row = {
                "dataset": dataset_name,
                "k_param": result["dataset_info"].get("k", "unknown"),
                "k_neighbors": k_result["k_neighbors"],
                "num_points": basic_info["num_points_analyzed"],
                "intrinsic_dim": basic_info["intrinsic_dimension"],
                "diameter": basic_info["diameter_estimate"]
            }
            
            # Add curvature statistics
            if "mean_curvature" in k_result and "error" not in k_result["mean_curvature"]:
                mc = k_result["mean_curvature"]
                row.update({
                    "mean_curvature_mean": mc["mean"],
                    "mean_curvature_std": mc["std"],
                    "mean_curvature_success_rate": mc["success_rate"]
                })
            
            if "pca_curvature" in k_result and "error" not in k_result["pca_curvature"]:
                pc = k_result["pca_curvature"]
                row.update({
                    "pca_curvature_mean": pc["mean"],
                    "pca_curvature_std": pc["std"]
                })
            
            summary_data.append(row)
    
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(output_dir / "curvature_neighborhood_summary.csv", index=False)
        logger.info(f"Summary saved to {output_dir / 'curvature_neighborhood_summary.csv'}")


def main():
    """Main analysis pipeline."""
    parser = argparse.ArgumentParser(description="Curvature Neighborhood Analysis")
    parser.add_argument("--data_dir", type=Path, 
                       default=Path("data/data_250913_1749"),
                       help="Directory containing manifold data")
    parser.add_argument("--output_dir", type=Path,
                       default=Path("results/curvature_neighborhood_analysis"),
                       help="Output directory for results")
    parser.add_argument("--k_min", type=int, default=5,
                       help="Minimum neighborhood size")
    parser.add_argument("--k_max", type=int, default=100,
                       help="Maximum neighborhood size")
    parser.add_argument("--k_step", type=int, default=5,
                       help="Step size for neighborhood range")
    parser.add_argument("--max_points", type=int, default=1000,
                       help="Maximum points to analyze per dataset")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Setup
    logger = setup_logging(args.verbose)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate neighborhood size range
    k_neighbors_range = list(range(args.k_min, args.k_max + 1, args.k_step))
    
    logger.info("Starting Curvature Neighborhood Analysis")
    logger.info(f"Data directory: {args.data_dir}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"k_neighbors range: {k_neighbors_range}")
    logger.info(f"Max points per dataset: {args.max_points}")
    
    # Find all dataset directories
    dataset_dirs = [d for d in args.data_dir.iterdir() if d.is_dir() and d.name.startswith("dataset")]
    if not dataset_dirs:
        logger.error(f"No dataset directories found in {args.data_dir}")
        return
    
    dataset_dirs.sort()
    logger.info(f"Found {len(dataset_dirs)} dataset directories")
    
    # Analyze each dataset
    all_results = []
    
    for dataset_dir in tqdm(dataset_dirs, desc="Analyzing datasets"):
        try:
            # Load data
            data = load_dataset_from_folder(dataset_dir)
            
            dataset_info = {
                "name": dataset_dir.name,
                "dir_path": str(dataset_dir),
                **data.get("metadata", {})
            }
            
            # Perform neighborhood analysis
            result = analyze_curvature_vs_neighborhood(
                data["intrinsic_coords"],
                data["clean_coords"],
                dataset_info,
                k_neighbors_range,
                args.max_points,
                logger
            )
            
            all_results.append(result)
            
        except Exception as e:
            logger.error(f"Failed to analyze {dataset_dir}: {e}")
            all_results.append({
                "dataset_info": {"name": dataset_dir.name, "dir_path": str(dataset_dir)},
                "error": str(e)
            })
    
    # Save results and create plots
    save_results(all_results, args.output_dir, logger)
    create_curvature_plots(all_results, args.output_dir, logger)
    
    # Print summary
    successful_analyses = sum(1 for r in all_results if "error" not in r)
    logger.info(f"Analysis complete: {successful_analyses}/{len(all_results)} datasets analyzed successfully")
    logger.info(f"Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
