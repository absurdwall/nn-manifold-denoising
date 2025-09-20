#!/usr/bin/env python3
"""
Curvature vs Kernel Smoothness Analysis (Fixed Neighborhood Sizes)

This script analyzes how curvature estimates vary with kernel smoothness parameter (σ)
using fixed neighborhood sizes (geometric radius) rather than fixed k_neighbors.

The goal is to test whether using the same geometric neighborhood size gives more
consistent curvature vs kernel_smoothness relationships.

Usage:
    python analysis/curvature_analysis/curvature_vs_kernel_smoothness_fixed_radius.py
"""

import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional, Tuple
import warnings
import logging
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors

# Add src to path
import sys
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

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
    """Load a dataset from the step1 output folder structure."""
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


def get_neighbors_by_radius(
    coords: np.ndarray, 
    radius: float, 
    max_neighbors: int = 200
) -> List[np.ndarray]:
    """
    Get neighbors within a fixed radius for each point.
    
    Args:
        coords: (N, d) coordinate array
        radius: Fixed radius for neighborhood
        max_neighbors: Maximum number of neighbors to return per point
        
    Returns:
        List of neighbor indices for each point
    """
    N = len(coords)
    nbrs = NearestNeighbors(radius=radius, algorithm='auto').fit(coords)
    
    neighbor_lists = []
    for i in range(N):
        # Get all neighbors within radius
        distances, indices = nbrs.radius_neighbors(coords[i:i+1])
        neighbor_idx = indices[0]
        
        # Remove self from neighbors
        neighbor_idx = neighbor_idx[neighbor_idx != i]
        
        # Limit number of neighbors for efficiency
        if len(neighbor_idx) > max_neighbors:
            neighbor_idx = np.random.choice(neighbor_idx, size=max_neighbors, replace=False)
        
        neighbor_lists.append(neighbor_idx)
    
    return neighbor_lists


def estimate_curvature_fixed_radius(
    intrinsic_coords: np.ndarray,
    clean_coords: np.ndarray,
    radius: float,
    max_points: int = 300,
    logger: logging.Logger = None
) -> Dict[str, Any]:
    """
    Estimate curvature using fixed geometric radius neighborhoods.
    
    Args:
        intrinsic_coords: (N, d) intrinsic coordinates
        clean_coords: (N, D) clean embedded coordinates
        radius: Fixed radius for neighborhoods
        max_points: Maximum points to process
        logger: Logger instance
        
    Returns:
        Dictionary with curvature statistics
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    N, d = intrinsic_coords.shape
    _, D = clean_coords.shape
    
    # Subsample if needed
    if N > max_points:
        indices = np.random.choice(N, size=max_points, replace=False)
        intrinsic_sample = intrinsic_coords[indices]
        clean_sample = clean_coords[indices]
        logger.debug(f"Subsampled to {max_points} points for radius={radius}")
    else:
        intrinsic_sample = intrinsic_coords
        clean_sample = clean_coords
    
    # Get neighbors by radius in intrinsic space
    neighbor_lists = get_neighbors_by_radius(intrinsic_sample, radius)
    
    # Count actual neighborhood sizes
    neighbor_sizes = [len(neighbors) for neighbors in neighbor_lists]
    avg_neighbors = np.mean(neighbor_sizes)
    min_neighbors = np.min(neighbor_sizes)
    max_neighbors = np.max(neighbor_sizes)
    
    logger.debug(f"Radius {radius}: avg_neighbors={avg_neighbors:.1f}, "
                f"range={min_neighbors}-{max_neighbors}")
    
    # For mean curvature, we need to use a different approach since the original
    # function expects k_neighbors. We'll approximate by using the average k.
    avg_k = max(5, int(avg_neighbors))  # Ensure minimum k=5
    
    results = {
        "radius": radius,
        "avg_neighbors": avg_neighbors,
        "min_neighbors": min_neighbors,
        "max_neighbors": max_neighbors,
        "num_points": len(clean_sample)
    }
    
    # Mean curvature estimation (using average k as approximation)
    try:
        mean_curvatures = estimate_mean_curvature(
            intrinsic_sample,
            clean_sample,
            k_neighbors=avg_k,
            max_points=None,
            weight="gaussian",
            regularization=1e-10,
            fallback_on_failure=True
        )
        
        mean_curv_stats = compute_curvature_statistics(mean_curvatures)
        results["mean_curvature"] = mean_curv_stats
        
        logger.debug(f"Radius {radius}: mean curvature = {mean_curv_stats['mean']:.6f}, "
                    f"success_rate = {mean_curv_stats['success_rate']:.2f}")
        
    except Exception as e:
        logger.warning(f"Mean curvature failed for radius {radius}: {e}")
        results["mean_curvature"] = {"error": str(e)}
    
    # PCA-based curvature estimation (using average k)
    try:
        pca_curvatures = estimate_curvature_pca_based(
            clean_sample,
            k_neighbors=avg_k,
            intrinsic_dim=d
        )
        
        pca_curv_stats = compute_curvature_statistics(pca_curvatures)
        results["pca_curvature"] = pca_curv_stats
        
        logger.debug(f"Radius {radius}: PCA curvature = {pca_curv_stats['mean']:.6f}")
        
    except Exception as e:
        logger.warning(f"PCA curvature failed for radius {radius}: {e}")
        results["pca_curvature"] = {"error": str(e)}
    
    return results


def analyze_dataset_with_radius_neighborhoods(
    dataset_dir: Path,
    radius_list: List[float],
    max_points: int = 300,
    logger: logging.Logger = None
) -> Dict[str, Any]:
    """Analyze a single dataset with different radius neighborhoods."""
    if logger is None:
        logger = logging.getLogger(__name__)
    
    # Load dataset
    data = load_dataset_from_folder(dataset_dir)
    dataset_info = {
        "name": dataset_dir.name,
        "dir_path": str(dataset_dir),
        **data.get("metadata", {})
    }
    
    # Extract kernel_smoothness from nested properties if needed
    kernel_smoothness = dataset_info.get('kernel_smoothness', 'unknown')
    if kernel_smoothness == 'unknown' and 'properties' in dataset_info:
        kernel_smoothness = dataset_info['properties'].get('kernel_smoothness', 'unknown')
    
    dataset_info['kernel_smoothness'] = kernel_smoothness
    logger.info(f"Analyzing {dataset_dir.name} (σ={kernel_smoothness})")
    
    N, d = data["intrinsic_coords"].shape
    _, D = data["clean_coords"].shape
    
    # Calculate dataset diameter for reference
    clean_coords = data["clean_coords"]
    diameter = np.max(np.linalg.norm(clean_coords - clean_coords.mean(axis=0), axis=1)) * 2
    
    results = {
        "dataset_info": dataset_info,
        "basic_info": {
            "num_points_total": N,
            "intrinsic_dimension": d,
            "embedding_dimension": D,
            "diameter": diameter,
            "kernel_smoothness": kernel_smoothness
        },
        "radius_analysis": {}
    }
    
    # Test each radius
    for radius in tqdm(radius_list, desc=f"Testing radii for {dataset_dir.name}", leave=False):
        try:
            radius_results = estimate_curvature_fixed_radius(
                data["intrinsic_coords"],
                data["clean_coords"],
                radius,
                max_points,
                logger
            )
            results["radius_analysis"][radius] = radius_results
            
        except Exception as e:
            logger.error(f"Failed analysis for radius {radius}: {e}")
            results["radius_analysis"][radius] = {"error": str(e)}
    
    return results


def create_curvature_vs_kernel_smoothness_plots(
    all_results: List[Dict[str, Any]],
    radius_list: List[float],
    output_dir: Path,
    logger: logging.Logger
) -> None:
    """Create plots showing curvature vs kernel_smoothness for different radii."""
    logger.info("Creating curvature vs kernel_smoothness plots...")
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(exist_ok=True)
    
    # Extract data for plotting
    plot_data = []
    
    for result in all_results:
        if "error" in result:
            continue
        
        dataset_info = result["dataset_info"]
        basic_info = result["basic_info"]
        kernel_smoothness = basic_info.get("kernel_smoothness", np.nan)
        
        if pd.isna(kernel_smoothness) or kernel_smoothness == 'unknown':
            continue
        
        for radius, radius_result in result["radius_analysis"].items():
            if "error" in radius_result:
                continue
                
            row = {
                "dataset": dataset_info.get("name", "unknown"),
                "kernel_smoothness": kernel_smoothness,
                "radius": radius,
                "avg_neighbors": radius_result.get("avg_neighbors", np.nan),
                "diameter": basic_info["diameter"],
                "true_d": basic_info["intrinsic_dimension"],
                "true_D": basic_info["embedding_dimension"]
            }
            
            # Mean curvature
            if "mean_curvature" in radius_result and "error" not in radius_result["mean_curvature"]:
                mc = radius_result["mean_curvature"]
                row.update({
                    "mean_curvature_mean": mc["mean"],
                    "mean_curvature_std": mc["std"],
                    "mean_curvature_success_rate": mc["success_rate"]
                })
            else:
                row.update({
                    "mean_curvature_mean": np.nan,
                    "mean_curvature_std": np.nan,
                    "mean_curvature_success_rate": 0.0
                })
            
            # PCA curvature
            if "pca_curvature" in radius_result and "error" not in radius_result["pca_curvature"]:
                pc = radius_result["pca_curvature"]
                row.update({
                    "pca_curvature_mean": pc["mean"],
                    "pca_curvature_std": pc["std"]
                })
            else:
                row.update({
                    "pca_curvature_mean": np.nan,
                    "pca_curvature_std": np.nan
                })
            
            plot_data.append(row)
    
    if not plot_data:
        logger.warning("No valid data for plotting")
        return
    
    df = pd.DataFrame(plot_data)
    
    # Set up plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Main plot: Curvature vs Kernel Smoothness for different radii
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Curvature vs Kernel Smoothness (Fixed Radius Neighborhoods)', fontsize=16)
    
    # Plot 1: Mean curvature vs kernel_smoothness
    ax = axes[0, 0]
    colors = plt.cm.tab10(np.linspace(0, 1, len(radius_list)))
    
    for i, radius in enumerate(radius_list):
        radius_data = df[df['radius'] == radius]
        radius_data = radius_data.dropna(subset=['kernel_smoothness', 'mean_curvature_mean'])
        if len(radius_data) > 0:
            radius_data = radius_data.sort_values('kernel_smoothness')
            ax.plot(radius_data['kernel_smoothness'], radius_data['mean_curvature_mean'], 
                   'o-', color=colors[i], label=f'radius={radius:.3f}', 
                   linewidth=2, markersize=6)
    
    ax.set_xlabel('Kernel Smoothness (σ)')
    ax.set_ylabel('Mean Curvature')
    ax.set_title('Mean Curvature vs σ')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Success rate
    ax = axes[0, 1]
    for i, radius in enumerate(radius_list):
        radius_data = df[df['radius'] == radius]
        radius_data = radius_data.dropna(subset=['kernel_smoothness', 'mean_curvature_success_rate'])
        if len(radius_data) > 0:
            radius_data = radius_data.sort_values('kernel_smoothness')
            ax.plot(radius_data['kernel_smoothness'], radius_data['mean_curvature_success_rate'], 
                   'o-', color=colors[i], label=f'radius={radius:.3f}', 
                   linewidth=2, markersize=6)
    
    ax.set_xlabel('Kernel Smoothness (σ)')
    ax.set_ylabel('Success Rate')
    ax.set_title('Mean Curvature Success Rate vs σ')
    ax.set_xscale('log')
    ax.set_ylim(0, 1.1)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: PCA curvature vs kernel_smoothness
    ax = axes[1, 0]
    for i, radius in enumerate(radius_list):
        radius_data = df[df['radius'] == radius]
        radius_data = radius_data.dropna(subset=['kernel_smoothness', 'pca_curvature_mean'])
        if len(radius_data) > 0:
            radius_data = radius_data.sort_values('kernel_smoothness')
            ax.plot(radius_data['kernel_smoothness'], radius_data['pca_curvature_mean'], 
                   'o-', color=colors[i], label=f'radius={radius:.3f}', 
                   linewidth=2, markersize=6)
    
    ax.set_xlabel('Kernel Smoothness (σ)')
    ax.set_ylabel('PCA Curvature')
    ax.set_title('PCA Curvature vs σ')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Average neighborhood size vs kernel_smoothness
    ax = axes[1, 1]
    for i, radius in enumerate(radius_list):
        radius_data = df[df['radius'] == radius]
        radius_data = radius_data.dropna(subset=['kernel_smoothness', 'avg_neighbors'])
        if len(radius_data) > 0:
            radius_data = radius_data.sort_values('kernel_smoothness')
            ax.plot(radius_data['kernel_smoothness'], radius_data['avg_neighbors'], 
                   'o-', color=colors[i], label=f'radius={radius:.3f}', 
                   linewidth=2, markersize=6)
    
    ax.set_xlabel('Kernel Smoothness (σ)')
    ax.set_ylabel('Average # Neighbors')
    ax.set_title('Neighborhood Size vs σ (for fixed radius)')
    ax.set_xscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = plot_dir / "curvature_vs_kernel_smoothness_fixed_radius.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    logger.info(f"  ✓ Saved: {save_path}")
    plt.close()
    
    # Additional plot: Compare with theoretical curves if possible
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Curvature vs Kernel Smoothness - Comparison with Theory', fontsize=16)
    
    # Mean curvature with theoretical overlay
    ax = axes[0]
    for i, radius in enumerate(radius_list):
        radius_data = df[df['radius'] == radius]
        radius_data = radius_data.dropna(subset=['kernel_smoothness', 'mean_curvature_mean'])
        if len(radius_data) > 0:
            radius_data = radius_data.sort_values('kernel_smoothness')
            ax.plot(radius_data['kernel_smoothness'], radius_data['mean_curvature_mean'], 
                   'o-', color=colors[i], label=f'radius={radius:.3f} (empirical)', 
                   linewidth=2, markersize=6)
            
            # Add theoretical curve: κ ∝ 1/σ (simplified)
            if len(radius_data) > 1:
                sigma_range = radius_data['kernel_smoothness'].values
                # Use first point to normalize
                theoretical_curve = radius_data['mean_curvature_mean'].iloc[0] * (
                    radius_data['kernel_smoothness'].iloc[0] / sigma_range
                )
                ax.plot(sigma_range, theoretical_curve, '--', color=colors[i], alpha=0.7,
                       label=f'radius={radius:.3f} (∝ 1/σ)')
    
    ax.set_xlabel('Kernel Smoothness (σ)')
    ax.set_ylabel('Mean Curvature')
    ax.set_title('Mean Curvature vs σ (with theoretical curves)')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    
    # PCA curvature
    ax = axes[1]
    for i, radius in enumerate(radius_list):
        radius_data = df[df['radius'] == radius]
        radius_data = radius_data.dropna(subset=['kernel_smoothness', 'pca_curvature_mean'])
        if len(radius_data) > 0:
            radius_data = radius_data.sort_values('kernel_smoothness')
            ax.plot(radius_data['kernel_smoothness'], radius_data['pca_curvature_mean'], 
                   'o-', color=colors[i], label=f'radius={radius:.3f}', 
                   linewidth=2, markersize=6)
    
    ax.set_xlabel('Kernel Smoothness (σ)')
    ax.set_ylabel('PCA Curvature')
    ax.set_title('PCA Curvature vs σ')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = plot_dir / "curvature_vs_kernel_smoothness_with_theory.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    logger.info(f"  ✓ Saved: {save_path}")
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
    with open(output_dir / "curvature_vs_kernel_smoothness_fixed_radius.json", 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    # Create summary CSV
    summary_data = []
    for result in all_results:
        if "error" in result:
            continue
        
        dataset_info = result["dataset_info"]
        basic_info = result["basic_info"]
        
        for radius, radius_result in result["radius_analysis"].items():
            if "error" in radius_result:
                continue
            
            row = {
                "dataset": dataset_info.get("name", "unknown"),
                "kernel_smoothness": basic_info.get("kernel_smoothness", np.nan),
                "radius": radius,
                "avg_neighbors": radius_result.get("avg_neighbors", np.nan),
                "diameter": basic_info["diameter"],
                "true_d": basic_info["intrinsic_dimension"],
                "true_D": basic_info["embedding_dimension"]
            }
            
            # Add curvature statistics
            if "mean_curvature" in radius_result and "error" not in radius_result["mean_curvature"]:
                mc = radius_result["mean_curvature"]
                row.update({
                    "mean_curvature_mean": mc["mean"],
                    "mean_curvature_std": mc["std"],
                    "mean_curvature_success_rate": mc["success_rate"]
                })
            
            if "pca_curvature" in radius_result and "error" not in radius_result["pca_curvature"]:
                pc = radius_result["pca_curvature"]
                row.update({
                    "pca_curvature_mean": pc["mean"],
                    "pca_curvature_std": pc["std"]
                })
            
            summary_data.append(row)
    
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(output_dir / "curvature_vs_kernel_smoothness_fixed_radius_summary.csv", index=False)
        logger.info(f"Summary saved to {output_dir / 'curvature_vs_kernel_smoothness_fixed_radius_summary.csv'}")


def main():
    """Main analysis pipeline."""
    parser = argparse.ArgumentParser(description="Curvature vs Kernel Smoothness (Fixed Radius)")
    parser.add_argument("--data_dir", type=Path, 
                       default=Path("data/data_250914_0100"),
                       help="Directory containing manifold data")
    parser.add_argument("--output_dir", type=Path,
                       default=Path("results/curvature_vs_kernel_smoothness_fixed_radius"),
                       help="Output directory for results")
    parser.add_argument("--datasets", type=int, nargs='+', 
                       default=[0, 1, 2, 3, 4],
                       help="Dataset numbers to analyze")
    parser.add_argument("--radius_list", type=float, nargs='+', 
                       default=[0.05, 0.1, 0.15, 0.2, 0.25],
                       help="List of radius values for neighborhoods")
    parser.add_argument("--max_points", type=int, default=300,
                       help="Maximum points to analyze per dataset")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Setup
    logger = setup_logging(args.verbose)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("Starting Curvature vs Kernel Smoothness Analysis (Fixed Radius)")
    logger.info(f"Data directory: {args.data_dir}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Datasets to analyze: {args.datasets}")
    logger.info(f"Radius values: {args.radius_list}")
    logger.info(f"Max points per dataset: {args.max_points}")
    
    # Analyze specified datasets
    all_results = []
    
    for dataset_num in args.datasets:
        dataset_dir = args.data_dir / f"dataset{dataset_num}"
        if not dataset_dir.exists():
            logger.warning(f"Dataset directory not found: {dataset_dir}")
            continue
        
        try:
            result = analyze_dataset_with_radius_neighborhoods(
                dataset_dir,
                args.radius_list,
                args.max_points,
                logger
            )
            all_results.append(result)
            
        except Exception as e:
            logger.error(f"Failed to analyze {dataset_dir}: {e}")
            all_results.append({
                "dataset_info": {"name": f"dataset{dataset_num}", "dir_path": str(dataset_dir)},
                "error": str(e)
            })
    
    if not all_results:
        logger.error("No datasets were successfully analyzed")
        return
    
    # Save results and create plots
    save_results(all_results, args.output_dir, logger)
    create_curvature_vs_kernel_smoothness_plots(all_results, args.radius_list, args.output_dir, logger)
    
    # Print summary
    successful_analyses = sum(1 for r in all_results if "error" not in r)
    logger.info(f"Analysis complete: {successful_analyses}/{len(all_results)} datasets analyzed successfully")
    logger.info(f"Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
