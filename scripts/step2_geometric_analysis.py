#!/usr/bin/env python3
"""
Step 2: Geometric Analysis Pipeline

This script performs comprehensive geometric analysis on manifold datasets,
including dimension estimation, curvature analysis, and geometric statistics.

Usage:
    python scripts/step2_geometric_analysis.py [--config CONFIG_FILE] [--data_dir DATA_DIR] [--output_dir OUTPUT_DIR]
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
    DimensionEstimator,
    estimate_mean_curvature,
    estimate_curvature_pca_based,
    compute_curvature_statistics,
    compute_geometric_summary,
    estimate_extrinsic_diameter,
    analyze_point_density
)
# We'll load data directly since we know the format


def setup_logging(verbose: bool = False) -> logging.Logger:
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    return logging.getLogger(__name__)


def load_analysis_config(config_path: Optional[Path] = None) -> Dict[str, Any]:
    """
    Load analysis configuration.
    
    Args:
        config_path: Path to config file (uses default if None)
        
    Returns:
        Configuration dictionary
    """
    default_config = {
        "dimension_estimation": {
            "methods": ["PCA", "k-NN", "TwoNN", "MLE"],
            "max_points_for_analysis": 2000,
            "k_neighbors_range": [10, 20, 30]
        },
        "curvature_analysis": {
            "enable_mean_curvature": True,
            "enable_pca_curvature": True,
            "k_neighbors": 40,
            "max_points_for_curvature": 1000,
            "weight": "gaussian",
            "regularization": 1e-10
        },
        "geometric_statistics": {
            "compute_diameter": True,
            "compute_volume": True,
            "compute_density_analysis": True,
            "sample_size_for_diameter": 2000
        },
        "output": {
            "save_detailed_results": True,
            "create_plots": True,
            "plot_format": "png",
            "plot_dpi": 300
        }
    }
    
    if config_path and config_path.exists():
        with open(config_path, 'r') as f:
            user_config = json.load(f)
        # Merge configs (user config overrides defaults)
        for key, value in user_config.items():
            if isinstance(value, dict) and key in default_config:
                default_config[key].update(value)
            else:
                default_config[key] = value
    
    return default_config


def load_dataset_from_folder(dataset_dir: Path) -> Dict[str, Any]:
    """
    Load a dataset from the step1 output folder structure.
    
    Args:
        dataset_dir: Path to dataset folder (e.g., dataset0/)
        
    Returns:
        Dictionary containing intrinsic_coords, embedded_coords, and metadata
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
    embedded_coords = np.load(clean_file)
    
    # Load metadata if available
    metadata = {}
    if metadata_file.exists():
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
    
    return {
        "intrinsic_coords": intrinsic_coords,
        "embedded_coords": embedded_coords,
        "metadata": metadata
    }


def analyze_single_dataset(
    intrinsic_coords: np.ndarray,
    embedded_coords: np.ndarray,
    dataset_info: Dict[str, Any],
    config: Dict[str, Any],
    logger: logging.Logger
) -> Dict[str, Any]:
    """
    Perform complete geometric analysis on a single dataset.
    
    Args:
        intrinsic_coords: (N, d) intrinsic coordinates
        embedded_coords: (N, D) embedded coordinates
        dataset_info: Dataset metadata
        config: Analysis configuration
        logger: Logger instance
        
    Returns:
        Dictionary of analysis results
    """
    logger.info(f"Analyzing dataset: {dataset_info.get('name', 'unknown')}")
    
    N, d = intrinsic_coords.shape
    _, D = embedded_coords.shape
    
    results = {
        "dataset_info": dataset_info,
        "basic_info": {
            "num_points": N,
            "intrinsic_dimension": d,
            "embedding_dimension": D
        }
    }
    
    # 1. Dimension Estimation
    logger.info("Starting dimension estimation...")
    dim_config = config["dimension_estimation"]
    
    try:
        # Subsample for efficiency if needed
        max_points = dim_config["max_points_for_analysis"]
        if N > max_points:
            indices = np.random.choice(N, size=max_points, replace=False)
            sample_embedded = embedded_coords[indices]
            logger.info(f"Subsampled to {max_points} points for dimension estimation")
        else:
            sample_embedded = embedded_coords
        
        estimator = DimensionEstimator(verbose=False)
        dim_results = estimator.fit(
            sample_embedded, 
            methods=dim_config["methods"]
        )
        
        results["dimension_estimation"] = dim_results
        logger.info(f"Dimension estimates: {dim_results}")
        
    except Exception as e:
        logger.error(f"Dimension estimation failed: {e}")
        results["dimension_estimation"] = {"error": str(e)}
    
    # 2. Curvature Analysis
    curv_config = config["curvature_analysis"]
    
    if curv_config["enable_mean_curvature"]:
        logger.info("Computing mean curvature...")
        try:
            max_curv_points = curv_config["max_points_for_curvature"]
            
            curvatures = estimate_mean_curvature(
                intrinsic_coords,
                embedded_coords,
                k_neighbors=curv_config["k_neighbors"],
                max_points=max_curv_points,
                weight=curv_config["weight"],
                regularization=curv_config["regularization"]
            )
            
            curv_stats = compute_curvature_statistics(curvatures)
            results["mean_curvature"] = curv_stats
            results["mean_curvature_values"] = curvatures.tolist()
            
            logger.info(f"Mean curvature stats: mean={curv_stats['mean']:.6f}, "
                       f"std={curv_stats['std']:.6f}, success_rate={curv_stats['success_rate']:.2f}")
            
        except Exception as e:
            logger.error(f"Mean curvature estimation failed: {e}")
            results["mean_curvature"] = {"error": str(e)}
    
    if curv_config["enable_pca_curvature"]:
        logger.info("Computing PCA-based curvature...")
        try:
            pca_curvatures = estimate_curvature_pca_based(
                embedded_coords,
                k_neighbors=curv_config["k_neighbors"]
            )
            
            pca_curv_stats = compute_curvature_statistics(pca_curvatures)
            results["pca_curvature"] = pca_curv_stats
            
            logger.info(f"PCA curvature stats: mean={pca_curv_stats['mean']:.6f}, "
                       f"std={pca_curv_stats['std']:.6f}")
            
        except Exception as e:
            logger.error(f"PCA curvature estimation failed: {e}")
            results["pca_curvature"] = {"error": str(e)}
    
    # 3. Geometric Statistics
    logger.info("Computing geometric statistics...")
    geom_config = config["geometric_statistics"]
    
    try:
        geom_summary = compute_geometric_summary(
            embedded_coords,
            intrinsic_coords,
            include_curvature=False,  # Already computed above
            include_dimension=False   # Already computed above
        )
        results["geometric_statistics"] = geom_summary
        
        logger.info(f"Geometric stats: diameter={geom_summary['extrinsic_diameter']:.4f}, "
                   f"volume={geom_summary.get('estimated_volume', 'N/A')}")
        
    except Exception as e:
        logger.error(f"Geometric statistics computation failed: {e}")
        results["geometric_statistics"] = {"error": str(e)}
    
    # 4. Density Analysis
    if geom_config["compute_density_analysis"]:
        logger.info("Analyzing point density...")
        try:
            density_stats = analyze_point_density(embedded_coords)
            results["density_analysis"] = density_stats
            
            logger.info(f"Density analysis: ratio={density_stats['density_ratio']:.2f}, "
                       f"mean_knn_dist={density_stats['mean_knn_distance']:.4f}")
            
        except Exception as e:
            logger.error(f"Density analysis failed: {e}")
            results["density_analysis"] = {"error": str(e)}
    
    return results


def create_analysis_plots(
    all_results: List[Dict[str, Any]],
    output_dir: Path,
    config: Dict[str, Any],
    logger: logging.Logger
) -> None:
    """
    Create comprehensive analysis plots.
    
    Args:
        all_results: List of analysis results for all datasets
        output_dir: Directory to save plots
        config: Configuration dictionary
        logger: Logger instance
    """
    if not config["output"]["create_plots"]:
        return
    
    logger.info("Creating analysis plots...")
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(exist_ok=True)
    
    plot_format = config["output"]["plot_format"]
    dpi = config["output"]["plot_dpi"]
    
    # Extract data for plotting
    plot_data = []
    for result in all_results:
        if "error" in result:
            continue
            
        row = {
            "dataset": result["dataset_info"].get("name", "unknown"),
            "k": result["dataset_info"].get("k", None),
            "N": result["basic_info"]["num_points"],
            "d": result["basic_info"]["intrinsic_dimension"],
            "D": result["basic_info"]["embedding_dimension"]
        }
        
        # Dimension estimates
        if "dimension_estimation" in result and "error" not in result["dimension_estimation"]:
            dim_est = result["dimension_estimation"]
            for method, value in dim_est.items():
                if isinstance(value, (int, float)):
                    row[f"dim_est_{method}"] = value
        
        # Curvature statistics
        if "mean_curvature" in result and "error" not in result["mean_curvature"]:
            curv = result["mean_curvature"]
            row["mean_curvature_mean"] = curv.get("mean", np.nan)
            row["mean_curvature_std"] = curv.get("std", np.nan)
            row["curvature_success_rate"] = curv.get("success_rate", np.nan)
        
        if "pca_curvature" in result and "error" not in result["pca_curvature"]:
            pca_curv = result["pca_curvature"]
            row["pca_curvature_mean"] = pca_curv.get("mean", np.nan)
            row["pca_curvature_std"] = pca_curv.get("std", np.nan)
        
        # Geometric statistics
        if "geometric_statistics" in result and "error" not in result["geometric_statistics"]:
            geom = result["geometric_statistics"]
            row["extrinsic_diameter"] = geom.get("extrinsic_diameter", np.nan)
            row["estimated_volume"] = geom.get("estimated_volume", np.nan)
            row["reach_estimate"] = geom.get("reach_estimate", np.nan)
        
        # Density analysis
        if "density_analysis" in result and "error" not in result["density_analysis"]:
            density = result["density_analysis"]
            row["density_ratio"] = density.get("density_ratio", np.nan)
            row["mean_knn_distance"] = density.get("mean_knn_distance", np.nan)
        
        plot_data.append(row)
    
    if not plot_data:
        logger.warning("No valid data for plotting")
        return
    
    df = pd.DataFrame(plot_data)
    
    # Plot 1: Dimension estimation comparison
    if any(col.startswith("dim_est_") for col in df.columns):
        fig, ax = plt.subplots(figsize=(12, 8))
        
        dim_cols = [col for col in df.columns if col.startswith("dim_est_")]
        if dim_cols and "d" in df.columns:
            for col in dim_cols:
                method = col.replace("dim_est_", "")
                valid_data = df.dropna(subset=[col, "d"])
                if len(valid_data) > 0:
                    ax.scatter(valid_data["d"], valid_data[col], 
                             label=method, alpha=0.7, s=50)
            
            # Add perfect estimation line
            d_range = [df["d"].min(), df["d"].max()]
            ax.plot(d_range, d_range, 'k--', alpha=0.5, label="Perfect estimation")
            
            ax.set_xlabel("True Intrinsic Dimension")
            ax.set_ylabel("Estimated Dimension")
            ax.set_title("Dimension Estimation Accuracy")
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(plot_dir / f"dimension_estimation.{plot_format}", dpi=dpi)
            plt.close()
    
    # Plot 2: Curvature analysis
    if "mean_curvature_mean" in df.columns or "pca_curvature_mean" in df.columns:
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Mean curvature vs parameters
        if "mean_curvature_mean" in df.columns:
            valid_df = df.dropna(subset=["mean_curvature_mean"])
            if len(valid_df) > 0:
                axes[0].scatter(valid_df.get("k", range(len(valid_df))), 
                               valid_df["mean_curvature_mean"], alpha=0.7)
                axes[0].set_xlabel("Dataset Index (or k)")
                axes[0].set_ylabel("Mean Curvature")
                axes[0].set_title("Mean Curvature Estimates")
                axes[0].grid(True, alpha=0.3)
        
        # PCA curvature comparison
        if "pca_curvature_mean" in df.columns:
            valid_df = df.dropna(subset=["pca_curvature_mean"])
            if len(valid_df) > 0:
                axes[1].scatter(valid_df.get("k", range(len(valid_df))), 
                               valid_df["pca_curvature_mean"], alpha=0.7, color='orange')
                axes[1].set_xlabel("Dataset Index (or k)")
                axes[1].set_ylabel("PCA-based Curvature")
                axes[1].set_title("PCA-based Curvature Estimates")
                axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(plot_dir / f"curvature_analysis.{plot_format}", dpi=dpi)
        plt.close()
    
    # Plot 3: Geometric properties overview
    geom_cols = ["extrinsic_diameter", "estimated_volume", "reach_estimate"]
    available_geom_cols = [col for col in geom_cols if col in df.columns]
    
    if available_geom_cols:
        fig, axes = plt.subplots(1, len(available_geom_cols), 
                                figsize=(5*len(available_geom_cols), 5))
        if len(available_geom_cols) == 1:
            axes = [axes]
        
        for i, col in enumerate(available_geom_cols):
            valid_df = df.dropna(subset=[col])
            if len(valid_df) > 0:
                axes[i].hist(valid_df[col], bins=20, alpha=0.7, edgecolor='black')
                axes[i].set_xlabel(col.replace("_", " ").title())
                axes[i].set_ylabel("Frequency")
                axes[i].set_title(f"Distribution of {col.replace('_', ' ').title()}")
                axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(plot_dir / f"geometric_properties.{plot_format}", dpi=dpi)
        plt.close()
    
    logger.info(f"Plots saved to {plot_dir}")


def save_results(
    all_results: List[Dict[str, Any]],
    output_dir: Path,
    config: Dict[str, Any],
    logger: logging.Logger
) -> None:
    """
    Save analysis results to files.
    
    Args:
        all_results: List of analysis results
        output_dir: Output directory
        config: Configuration
        logger: Logger instance
    """
    logger.info("Saving results...")
    
    # Save detailed results as JSON
    if config["output"]["save_detailed_results"]:
        with open(output_dir / "detailed_results.json", 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
    
    # Save summary as CSV
    summary_data = []
    for result in all_results:
        if "error" in result:
            continue
        
        row = {
            "dataset": result["dataset_info"].get("name", "unknown"),
            "num_points": result["basic_info"]["num_points"],
            "intrinsic_dim": result["basic_info"]["intrinsic_dimension"],
            "embedding_dim": result["basic_info"]["embedding_dimension"]
        }
        
        # Add key metrics
        if "dimension_estimation" in result and "error" not in result["dimension_estimation"]:
            dim_est = result["dimension_estimation"]
            if "consensus" in dim_est:
                row["estimated_dimension"] = dim_est["consensus"]
        
        if "mean_curvature" in result and "error" not in result["mean_curvature"]:
            curv = result["mean_curvature"]
            row["mean_curvature"] = curv.get("mean", np.nan)
            row["curvature_success_rate"] = curv.get("success_rate", np.nan)
        
        if "geometric_statistics" in result and "error" not in result["geometric_statistics"]:
            geom = result["geometric_statistics"]
            row["extrinsic_diameter"] = geom.get("extrinsic_diameter", np.nan)
        
        summary_data.append(row)
    
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(output_dir / "analysis_summary.csv", index=False)
        logger.info(f"Summary saved to {output_dir / 'analysis_summary.csv'}")


def main():
    """Main analysis pipeline."""
    parser = argparse.ArgumentParser(description="Step 2: Geometric Analysis Pipeline")
    parser.add_argument("--config", type=Path, help="Configuration file path")
    parser.add_argument("--data_dir", type=Path, 
                       default=Path("data/data_250913_1749"),
                       help="Directory containing manifold data")
    parser.add_argument("--output_dir", type=Path,
                       default=Path("results/geometric_analysis"),
                       help="Output directory for results")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Setup
    logger = setup_logging(args.verbose)
    config = load_analysis_config(args.config)
    
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("Starting Step 2: Geometric Analysis Pipeline")
    logger.info(f"Data directory: {args.data_dir}")
    logger.info(f"Output directory: {args.output_dir}")
    
    # Find all dataset directories
    dataset_dirs = [d for d in args.data_dir.iterdir() if d.is_dir() and d.name.startswith("dataset")]
    if not dataset_dirs:
        logger.error(f"No dataset directories found in {args.data_dir}")
        return
    
    dataset_dirs.sort()  # Ensure consistent ordering
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
            
            # Perform analysis
            result = analyze_single_dataset(
                data["intrinsic_coords"],
                data["embedded_coords"],
                dataset_info,
                config,
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
    save_results(all_results, args.output_dir, config, logger)
    create_analysis_plots(all_results, args.output_dir, config, logger)
    
    # Print summary
    successful_analyses = sum(1 for r in all_results if "error" not in r)
    logger.info(f"Analysis complete: {successful_analyses}/{len(all_results)} datasets analyzed successfully")
    
    if config["output"]["save_detailed_results"]:
        logger.info(f"Detailed results saved to {args.output_dir}/detailed_results.json")


if __name__ == "__main__":
    main()
