#!/usr/bin/env python3
"""
Step 2: Comprehensive Geometric Analysis Pipeline

This script performs comprehensive geometric analysis on all three types of manifold data:
1. GP data (raw Gaussian Process generated data, before processing)
2. Clean data (after centralization, normalization, rotation)  
3. Noisy data (clean data + noise)

For each data type, it computes:
- Dimension estimation using multiple methods
- Curvature estimation using various approaches
- Geometric statistics (diameter, volume, etc.)

Usage:
    python scripts/step2_comprehensive_analysis.py [--config CONFIG_FILE] [--data_dir DATA_DIR] [--output_dir OUTPUT_DIR]
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
import time

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


def setup_logging(verbose: bool = False) -> logging.Logger:
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    return logging.getLogger(__name__)


def load_comprehensive_config(config_path: Optional[Path] = None) -> Dict[str, Any]:
    """Load comprehensive analysis configuration."""
    default_config = {
        "data_types": {
            "analyze_gp_data": True,
            "analyze_clean_data": True, 
            "analyze_noisy_data": True
        },
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
            "regularization": 1e-10,
            "fallback_on_failure": True,
            "min_neighbors": 10
        },
        "geometric_statistics": {
            "compute_diameter": True,
            "compute_volume": True,
            "compute_density_analysis": True,
            "sample_size_for_diameter": 2000
        },
        "output": {
            "save_detailed_results": True,
            "save_per_dataset": True,
            "save_summary_tables": True
        },
        "performance": {
            "max_datasets": None,
            "show_progress": True
        }
    }
    
    if config_path and config_path.exists():
        with open(config_path, 'r') as f:
            user_config = json.load(f)
        # Merge configs
        for key, value in user_config.items():
            if isinstance(value, dict) and key in default_config:
                default_config[key].update(value)
            else:
                default_config[key] = value
    
    return default_config


def load_all_dataset_types(dataset_dir: Path) -> Dict[str, Dict[str, Any]]:
    """
    Load all three types of data for a dataset.
    
    Returns:
        Dictionary with keys 'gp_data', 'clean_data', 'noisy_data'
    """
    dataset_name = dataset_dir.name
    
    # Load intrinsic coordinates (same for all data types)
    intrinsic_file = dataset_dir / f"{dataset_name}_intrinsic.npy"
    if not intrinsic_file.exists():
        raise FileNotFoundError(f"Intrinsic coords file not found: {intrinsic_file}")
    intrinsic_coords = np.load(intrinsic_file)
    
    # Load metadata
    metadata_file = dataset_dir / f"{dataset_name}_metadata.json"
    metadata = {}
    if metadata_file.exists():
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
    
    results = {}
    
    # 1. GP data (raw data before any processing)
    gp_file = dataset_dir / f"{dataset_name}_raw.npy"
    if gp_file.exists():
        gp_coords = np.load(gp_file)
        results['gp_data'] = {
            "intrinsic_coords": intrinsic_coords,
            "embedded_coords": gp_coords,
            "metadata": metadata,
            "data_type": "gp_data",
            "description": "Raw Gaussian Process generated data (before processing)"
        }
    
    # 2. Clean data (after centralization, normalization, rotation)
    clean_file = dataset_dir / f"{dataset_name}_clean.npy"
    if clean_file.exists():
        clean_coords = np.load(clean_file)
        results['clean_data'] = {
            "intrinsic_coords": intrinsic_coords,
            "embedded_coords": clean_coords,
            "metadata": metadata,
            "data_type": "clean_data",
            "description": "Processed data (centralized, normalized, rotated)"
        }
    
    # 3. Noisy data (clean data + noise)
    noisy_file = dataset_dir / f"{dataset_name}_noisy.npy"
    if noisy_file.exists():
        noisy_coords = np.load(noisy_file)
        results['noisy_data'] = {
            "intrinsic_coords": intrinsic_coords,
            "embedded_coords": noisy_coords,
            "metadata": metadata,
            "data_type": "noisy_data",
            "description": "Processed data with added noise"
        }
    
    return results


def analyze_single_data_type(
    intrinsic_coords: np.ndarray,
    embedded_coords: np.ndarray,
    data_type: str,
    config: Dict[str, Any],
    logger: logging.Logger
) -> Dict[str, Any]:
    """Perform complete analysis on a single data type."""
    
    N, d = intrinsic_coords.shape
    _, D = embedded_coords.shape
    
    results = {
        "data_type": data_type,
        "basic_info": {
            "num_points": N,
            "intrinsic_dimension": d,
            "embedding_dimension": D
        },
        "analysis_timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # 1. Dimension Estimation
    logger.info(f"  Computing dimension estimation for {data_type}...")
    dim_config = config["dimension_estimation"]
    
    try:
        max_points = dim_config["max_points_for_analysis"]
        if N > max_points:
            indices = np.random.choice(N, size=max_points, replace=False)
            sample_embedded = embedded_coords[indices]
        else:
            sample_embedded = embedded_coords
        
        estimator = DimensionEstimator(verbose=False)
        dim_results = estimator.fit(
            sample_embedded, 
            methods=dim_config["methods"]
        )
        results["dimension_estimation"] = dim_results
        logger.info(f"    ✓ Dimension estimates: {dim_results}")
        
    except Exception as e:
        logger.error(f"    ✗ Dimension estimation failed: {e}")
        results["dimension_estimation"] = {"error": str(e)}
    
    # 2. Curvature Analysis
    curv_config = config["curvature_analysis"]
    
    if curv_config["enable_mean_curvature"]:
        logger.info(f"  Computing mean curvature for {data_type}...")
        try:
            max_curv_points = curv_config["max_points_for_curvature"]
            
            curvatures = estimate_mean_curvature(
                intrinsic_coords,
                embedded_coords,
                k_neighbors=curv_config["k_neighbors"],
                max_points=max_curv_points,
                weight=curv_config["weight"],
                regularization=curv_config["regularization"],
                fallback_on_failure=curv_config.get("fallback_on_failure", True),
                min_neighbors=curv_config.get("min_neighbors", 10)
            )
            
            curv_stats = compute_curvature_statistics(curvatures)
            results["mean_curvature"] = curv_stats
            
            logger.info(f"    ✓ Mean curvature: mean={curv_stats['mean']:.6f}, "
                       f"success_rate={curv_stats['success_rate']:.2f}")
            
        except Exception as e:
            logger.error(f"    ✗ Mean curvature estimation failed: {e}")
            results["mean_curvature"] = {"error": str(e)}
    
    if curv_config["enable_pca_curvature"]:
        logger.info(f"  Computing PCA-based curvature for {data_type}...")
        try:
            pca_curvatures = estimate_curvature_pca_based(
                embedded_coords,
                k_neighbors=curv_config["k_neighbors"],
                intrinsic_dim=d
            )
            
            pca_curv_stats = compute_curvature_statistics(pca_curvatures)
            results["pca_curvature"] = pca_curv_stats
            
            logger.info(f"    ✓ PCA curvature: mean={pca_curv_stats['mean']:.6f}")
            
        except Exception as e:
            logger.error(f"    ✗ PCA curvature estimation failed: {e}")
            results["pca_curvature"] = {"error": str(e)}
    
    # 3. Geometric Statistics
    logger.info(f"  Computing geometric statistics for {data_type}...")
    try:
        geom_summary = compute_geometric_summary(
            embedded_coords,
            intrinsic_coords,
            include_curvature=False,  # Already computed above
            include_dimension=False   # Already computed above
        )
        results["geometric_statistics"] = geom_summary
        
        logger.info(f"    ✓ Diameter: {geom_summary['extrinsic_diameter']:.4f}")
        
    except Exception as e:
        logger.error(f"    ✗ Geometric statistics failed: {e}")
        results["geometric_statistics"] = {"error": str(e)}
    
    # 4. Density Analysis
    geom_config = config["geometric_statistics"]
    if geom_config["compute_density_analysis"]:
        logger.info(f"  Computing density analysis for {data_type}...")
        try:
            density_stats = analyze_point_density(embedded_coords)
            results["density_analysis"] = density_stats
            
            logger.info(f"    ✓ Density ratio: {density_stats['density_ratio']:.2f}")
            
        except Exception as e:
            logger.error(f"    ✗ Density analysis failed: {e}")
            results["density_analysis"] = {"error": str(e)}
    
    return results


def analyze_complete_dataset(
    dataset_dir: Path,
    config: Dict[str, Any], 
    logger: logging.Logger
) -> Dict[str, Any]:
    """Analyze all data types for a single dataset."""
    
    dataset_name = dataset_dir.name
    logger.info(f"\nAnalyzing dataset: {dataset_name}")
    
    try:
        # Load all data types
        all_data = load_all_dataset_types(dataset_dir)
        
        if not all_data:
            logger.error(f"  No data found for {dataset_name}")
            return {"error": "No data files found"}
        
        logger.info(f"  Found data types: {list(all_data.keys())}")
        
        # Analyze each data type
        results = {
            "dataset_name": dataset_name,
            "dataset_dir": str(dataset_dir),
            "analysis_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "data_analyses": {}
        }
        
        data_config = config["data_types"]
        
        for data_type, data_info in all_data.items():
            # Check if we should analyze this data type
            analyze_key = f"analyze_{data_type}"
            if not data_config.get(analyze_key, True):
                logger.info(f"  Skipping {data_type} (disabled in config)")
                continue
            
            logger.info(f"  Analyzing {data_type}...")
            
            try:
                analysis = analyze_single_data_type(
                    data_info["intrinsic_coords"],
                    data_info["embedded_coords"],
                    data_type,
                    config,
                    logger
                )
                
                # Add metadata info
                analysis["metadata"] = data_info["metadata"]
                analysis["description"] = data_info["description"]
                
                results["data_analyses"][data_type] = analysis
                
            except Exception as e:
                logger.error(f"  ✗ Failed to analyze {data_type}: {e}")
                results["data_analyses"][data_type] = {
                    "error": str(e),
                    "data_type": data_type
                }
        
        return results
        
    except Exception as e:
        logger.error(f"  ✗ Failed to load dataset {dataset_name}: {e}")
        return {"error": str(e), "dataset_name": dataset_name}


def save_comprehensive_results(
    all_results: List[Dict[str, Any]],
    output_dir: Path,
    config: Dict[str, Any],
    logger: logging.Logger
) -> None:
    """Save comprehensive analysis results."""
    
    logger.info("Saving comprehensive results...")
    
    output_config = config["output"]
    
    # Save detailed results
    if output_config["save_detailed_results"]:
        detailed_file = output_dir / "comprehensive_analysis_results.json"
        with open(detailed_file, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        logger.info(f"  ✓ Detailed results: {detailed_file}")
    
    # Save per-dataset results
    if output_config["save_per_dataset"]:
        dataset_dir = output_dir / "per_dataset"
        dataset_dir.mkdir(exist_ok=True)
        
        for result in all_results:
            if "error" in result:
                continue
            
            dataset_name = result["dataset_name"]
            dataset_file = dataset_dir / f"{dataset_name}_analysis.json"
            
            with open(dataset_file, 'w') as f:
                json.dump(result, f, indent=2, default=str)
        
        logger.info(f"  ✓ Per-dataset results: {dataset_dir}")
    
    # Create summary tables
    if output_config["save_summary_tables"]:
        create_summary_tables(all_results, output_dir, logger)


def create_summary_tables(
    all_results: List[Dict[str, Any]],
    output_dir: Path,
    logger: logging.Logger
) -> None:
    """Create summary tables for all analyses."""
    
    logger.info("Creating summary tables...")
    
    # Collect data for summary tables
    summary_data = []
    
    for result in all_results:
        if "error" in result:
            continue
        
        dataset_name = result["dataset_name"]
        
        for data_type, analysis in result.get("data_analyses", {}).items():
            if "error" in analysis:
                continue
            
            # Extract metadata
            metadata = analysis.get("metadata", {}).get("properties", {})
            
            # Basic info
            basic = analysis.get("basic_info", {})
            
            # Dimension estimates
            dim_est = analysis.get("dimension_estimation", {})
            
            # Curvature stats
            mean_curv = analysis.get("mean_curvature", {})
            pca_curv = analysis.get("pca_curvature", {})
            
            # Geometric stats
            geom = analysis.get("geometric_statistics", {})
            density = analysis.get("density_analysis", {})
            
            row = {
                "dataset": dataset_name,
                "data_type": data_type,
                
                # Dataset parameters (crucial for plotting)
                "true_d": metadata.get("d", np.nan),
                "true_D": metadata.get("D", np.nan),
                "true_k": metadata.get("k", np.nan),
                "kernel_smoothness": metadata.get("kernel_smoothness", np.nan),  # σ parameter
                "noise_sigma": metadata.get("noise_sigma", np.nan),
                "base_type": metadata.get("base_type", "unknown"),
                "num_points": basic.get("num_points", np.nan),
                
                # Normalization information (for theoretical comparisons)
                "normalization_factor": metadata.get("normalization_factor", np.nan),
                "original_diameter": metadata.get("original_diameter", np.nan),
                
                # Dimension estimates
                "dim_PCA": dim_est.get("PCA", np.nan),
                "dim_kNN": dim_est.get("k-NN", np.nan),
                "dim_TwoNN": dim_est.get("TwoNN", np.nan),
                "dim_MLE": dim_est.get("MLE", np.nan),
                
                # Curvature
                "mean_curvature": mean_curv.get("mean", np.nan),
                "curvature_success_rate": mean_curv.get("success_rate", np.nan),
                "pca_curvature": pca_curv.get("mean", np.nan),
                
                # Geometric properties
                "extrinsic_diameter": geom.get("extrinsic_diameter", np.nan),
                "estimated_volume": geom.get("estimated_volume", np.nan),
                "density_ratio": density.get("density_ratio", np.nan),
            }
            
            summary_data.append(row)
    
    if summary_data:
        # Save as CSV
        df = pd.DataFrame(summary_data)
        summary_file = output_dir / "comprehensive_summary.csv"
        df.to_csv(summary_file, index=False)
        logger.info(f"  ✓ Summary table: {summary_file}")
        
        # Save separate tables by data type
        for data_type in df["data_type"].unique():
            if pd.isna(data_type):
                continue
            subset = df[df["data_type"] == data_type]
            type_file = output_dir / f"summary_{data_type}.csv"
            subset.to_csv(type_file, index=False)
        
        logger.info(f"  ✓ Data type summaries saved")


def main():
    """Main comprehensive analysis pipeline."""
    parser = argparse.ArgumentParser(description="Step 2: Comprehensive Geometric Analysis")
    parser.add_argument("--config", type=Path, help="Configuration file path")
    parser.add_argument("--data_dir", type=Path, 
                       default=Path("data/data_250913_1749"),
                       help="Directory containing manifold data")
    parser.add_argument("--output_dir", type=Path,
                       default=Path("results/step2_comprehensive_analysis"),
                       help="Output directory for results")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Setup
    logger = setup_logging(args.verbose)
    config = load_comprehensive_config(args.config)
    
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("Starting Step 2: Comprehensive Geometric Analysis Pipeline")
    logger.info(f"Data directory: {args.data_dir}")
    logger.info(f"Output directory: {args.output_dir}")
    
    # Find all dataset directories
    dataset_dirs = [d for d in args.data_dir.iterdir() if d.is_dir() and d.name.startswith("dataset")]
    if not dataset_dirs:
        logger.error(f"No dataset directories found in {args.data_dir}")
        return
    
    dataset_dirs.sort()
    
    # Apply max_datasets limit if specified
    max_datasets = config["performance"].get("max_datasets")
    if max_datasets and max_datasets < len(dataset_dirs):
        dataset_dirs = dataset_dirs[:max_datasets]
        logger.info(f"Limited to first {max_datasets} datasets")
    
    logger.info(f"Found {len(dataset_dirs)} datasets to analyze")
    
    # Analyze each dataset
    all_results = []
    
    show_progress = config["performance"].get("show_progress", True)
    iterator = tqdm(dataset_dirs, desc="Analyzing datasets") if show_progress else dataset_dirs
    
    for dataset_dir in iterator:
        try:
            result = analyze_complete_dataset(dataset_dir, config, logger)
            all_results.append(result)
        except Exception as e:
            logger.error(f"Failed to analyze {dataset_dir}: {e}")
            all_results.append({
                "error": str(e),
                "dataset_name": dataset_dir.name
            })
    
    # Save results
    save_comprehensive_results(all_results, args.output_dir, config, logger)
    
    # Print summary
    successful = sum(1 for r in all_results if "error" not in r)
    logger.info(f"\nAnalysis complete: {successful}/{len(all_results)} datasets analyzed successfully")
    
    # Count analyses by data type
    total_analyses = 0
    successful_analyses = 0
    
    for result in all_results:
        if "error" not in result:
            data_analyses = result.get("data_analyses", {})
            for data_type, analysis in data_analyses.items():
                total_analyses += 1
                if "error" not in analysis:
                    successful_analyses += 1
    
    logger.info(f"Total data type analyses: {successful_analyses}/{total_analyses}")
    logger.info(f"Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
