"""
General manifold curvature analysis framework.

This module provides high-level functions for computing curvature statistics
across manifolds with different neighbor selection strategies.
"""

import numpy as np
from sklearn.neighbors import NearestNeighbors
from typing import Tuple, Optional, Dict, Callable, Union, List
import warnings
from .point_curvature import (
    estimate_mean_curvature_at_point,
    estimate_pca_curvature_at_point,
    estimate_gaussian_curvature_at_point
)


def get_k_nearest_neighbors(
    points: np.ndarray,
    target_indices: np.ndarray,
    k: int
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Get k nearest neighbors for specified target points.
    
    Args:
        points: (N, d) array of all points
        target_indices: (n,) indices of points to analyze
        k: Number of nearest neighbors
        
    Returns:
        neighbor_indices: List of (k,) arrays of neighbor indices for each target
        neighbor_points: List of (k, d) arrays of neighbor coordinates for each target
    """
    N = len(points)
    k_eff = min(k, N - 1)  # Can't have more neighbors than available points
    
    # Build nearest neighbors index
    nbrs = NearestNeighbors(n_neighbors=k_eff + 1, algorithm='auto').fit(points)
    
    neighbor_indices = []
    neighbor_points = []
    
    for idx in target_indices:
        # Get k+1 neighbors (includes the point itself)
        distances, indices = nbrs.kneighbors(points[idx:idx+1])
        
        # Remove the point itself (first neighbor)
        neigh_idx = indices[0][1:]
        neigh_points = points[neigh_idx]
        
        neighbor_indices.append(neigh_idx)
        neighbor_points.append(neigh_points)
    
    return neighbor_indices, neighbor_points


def get_radius_neighbors(
    points: np.ndarray,
    target_indices: np.ndarray,
    radius: float,
    k: Optional[int] = None,
    random_state: Optional[int] = None
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Get neighbors within a fixed radius, optionally sampling k of them.
    
    Args:
        points: (N, d) array of all points
        target_indices: (n,) indices of points to analyze
        radius: Maximum distance for neighbors
        k: Optional number of neighbors to sample from those within radius
        random_state: Random seed for sampling
        
    Returns:
        neighbor_indices: List of (≤k,) arrays of neighbor indices for each target
        neighbor_points: List of (≤k, d) arrays of neighbor coordinates for each target
    """
    # Build radius neighbors index
    nbrs = NearestNeighbors(radius=radius, algorithm='auto').fit(points)
    
    neighbor_indices = []
    neighbor_points = []
    
    rng = np.random.default_rng(random_state) if random_state is not None else None
    
    for idx in target_indices:
        # Get all neighbors within radius
        indices = nbrs.radius_neighbors(points[idx:idx+1], return_distance=False)[0]
        
        # Remove the point itself
        indices = indices[indices != idx]
        
        # Sample k neighbors if requested and if we have enough
        if k is not None and len(indices) > k:
            if rng is not None:
                indices = rng.choice(indices, size=k, replace=False)
            else:
                indices = np.random.choice(indices, size=k, replace=False)
        
        neigh_points = points[indices]
        
        neighbor_indices.append(indices)
        neighbor_points.append(neigh_points)
    
    return neighbor_indices, neighbor_points


def compute_manifold_curvature(
    embedded_coords: np.ndarray,
    intrinsic_coords: Optional[np.ndarray] = None,
    curvature_method: str = "mean",
    neighbor_strategy: str = "k_nearest",
    n_sample_points: Optional[int] = None,
    k_neighbors: Optional[int] = None,
    radius: Optional[float] = None,
    random_state: Optional[int] = 42,
    **curvature_kwargs
) -> Dict:
    """
    Compute curvature statistics for a manifold dataset.
    
    Args:
        embedded_coords: (N, D) array of embedded coordinates
        intrinsic_coords: (N, d) array of intrinsic coordinates (optional)
        curvature_method: Method for curvature estimation ("mean", "pca", "gaussian")
        neighbor_strategy: How to select neighbors ("k_nearest", "radius")
        n_sample_points: Number of points to analyze (None for all)
        k_neighbors: Number of neighbors (for k_nearest) or max neighbors (for radius)
        radius: Radius for neighbor selection (for radius strategy)
        random_state: Random seed for reproducibility
        **curvature_kwargs: Additional arguments for curvature estimation
        
    Returns:
        results: Dictionary containing curvature statistics and metadata
    """
    N, D = embedded_coords.shape
    d = intrinsic_coords.shape[1] if intrinsic_coords is not None else None
    
    # Validate inputs
    if neighbor_strategy == "k_nearest" and k_neighbors is None:
        raise ValueError("k_neighbors must be specified for k_nearest strategy")
    if neighbor_strategy == "radius" and radius is None:
        raise ValueError("radius must be specified for radius strategy")
    
    # Select curvature estimation function
    curvature_functions = {
        "mean": estimate_mean_curvature_at_point,
        "pca": estimate_pca_curvature_at_point,
        "gaussian": estimate_gaussian_curvature_at_point
    }
    
    if curvature_method not in curvature_functions:
        raise ValueError(f"Unknown curvature method: {curvature_method}")
    
    curvature_func = curvature_functions[curvature_method]
    
    # Select points to analyze
    rng = np.random.default_rng(random_state)
    if n_sample_points is None or n_sample_points >= N:
        target_indices = np.arange(N)
    else:
        target_indices = rng.choice(N, size=n_sample_points, replace=False)
    
    n_target = len(target_indices)
    
    # Get neighbors based on strategy
    if neighbor_strategy == "k_nearest":
        neighbor_indices, neighbor_coords = get_k_nearest_neighbors(
            embedded_coords, target_indices, k_neighbors
        )
        if intrinsic_coords is not None:
            _, intrinsic_neighbor_coords = get_k_nearest_neighbors(
                intrinsic_coords, target_indices, k_neighbors
            )
        else:
            intrinsic_neighbor_coords = [None] * n_target
            
    elif neighbor_strategy == "radius":
        neighbor_indices, neighbor_coords = get_radius_neighbors(
            embedded_coords, target_indices, radius, k_neighbors, random_state
        )
        if intrinsic_coords is not None:
            _, intrinsic_neighbor_coords = get_radius_neighbors(
                intrinsic_coords, target_indices, radius, k_neighbors, random_state
            )
        else:
            intrinsic_neighbor_coords = [None] * n_target
    else:
        raise ValueError(f"Unknown neighbor strategy: {neighbor_strategy}")
    
    # Compute curvatures
    curvatures = np.full(n_target, np.nan)
    diagnostics = []
    
    for i, idx in enumerate(target_indices):
        target_point = embedded_coords[idx]
        neighbors = neighbor_coords[i]
        
        # Skip if not enough neighbors
        if len(neighbors) == 0:
            diagnostics.append({"error": "no_neighbors", "index": idx})
            continue
        
        # Prepare intrinsic coordinates if available
        if intrinsic_coords is not None:
            intrinsic_point = intrinsic_coords[idx]
            intrinsic_neighbors = intrinsic_neighbor_coords[i]
        else:
            intrinsic_point = None
            intrinsic_neighbors = None
        
        # Estimate curvature at this point
        if curvature_method == "pca":
            # PCA method doesn't use intrinsic coordinates
            curvature, info = curvature_func(
                target_point, neighbors, **curvature_kwargs
            )
        else:
            # Other methods can use intrinsic coordinates
            curvature, info = curvature_func(
                target_point, neighbors, 
                intrinsic_point, intrinsic_neighbors,
                **curvature_kwargs
            )
        
        curvatures[i] = curvature
        info["index"] = idx
        info["num_neighbors"] = len(neighbors)
        diagnostics.append(info)
    
    # Compute statistics
    valid_curvatures = curvatures[np.isfinite(curvatures)]
    
    if len(valid_curvatures) == 0:
        stats = {
            "mean": np.nan,
            "std": np.nan,
            "median": np.nan,
            "min": np.nan,
            "max": np.nan,
            "num_valid": 0,
            "num_total": n_target,
            "success_rate": 0.0
        }
    else:
        stats = {
            "mean": float(np.mean(valid_curvatures)),
            "std": float(np.std(valid_curvatures)),
            "median": float(np.median(valid_curvatures)),
            "min": float(np.min(valid_curvatures)),
            "max": float(np.max(valid_curvatures)),
            "num_valid": int(len(valid_curvatures)),
            "num_total": int(n_target),
            "success_rate": float(len(valid_curvatures) / n_target)
        }
    
    # Compile results
    results = {
        "curvature_values": curvatures,
        "statistics": stats,
        "diagnostics": diagnostics,
        "metadata": {
            "curvature_method": curvature_method,
            "neighbor_strategy": neighbor_strategy,
            "n_sample_points": n_target,
            "k_neighbors": k_neighbors,
            "radius": radius,
            "embedded_shape": embedded_coords.shape,
            "intrinsic_shape": intrinsic_coords.shape if intrinsic_coords is not None else None,
            "target_indices": target_indices
        }
    }
    
    return results


def compute_curvature_vs_parameter(
    datasets: List[Dict],
    parameter_name: str,
    parameter_values: List,
    curvature_method: str = "mean",
    neighbor_strategy: str = "k_nearest",
    **computation_kwargs
) -> Dict:
    """
    Compute curvature across multiple datasets varying a parameter.
    
    Args:
        datasets: List of dataset dictionaries with 'embedded_coords' and optionally 'intrinsic_coords'
        parameter_name: Name of the parameter being varied
        parameter_values: Values of the parameter for each dataset
        curvature_method: Method for curvature estimation
        neighbor_strategy: How to select neighbors
        **computation_kwargs: Additional arguments for compute_manifold_curvature
        
    Returns:
        results: Dictionary with parameter values and corresponding curvature statistics
    """
    if len(datasets) != len(parameter_values):
        raise ValueError("Number of datasets must match number of parameter values")
    
    results = {
        "parameter_name": parameter_name,
        "parameter_values": parameter_values,
        "curvature_statistics": [],
        "all_results": [],
        "metadata": {
            "curvature_method": curvature_method,
            "neighbor_strategy": neighbor_strategy,
            "num_datasets": len(datasets)
        }
    }
    
    for i, (dataset, param_val) in enumerate(zip(datasets, parameter_values)):
        print(f"Processing dataset {i+1}/{len(datasets)} ({parameter_name}={param_val})...")
        
        # Compute curvature for this dataset
        curvature_result = compute_manifold_curvature(
            embedded_coords=dataset["embedded_coords"],
            intrinsic_coords=dataset.get("intrinsic_coords", None),
            curvature_method=curvature_method,
            neighbor_strategy=neighbor_strategy,
            **computation_kwargs
        )
        
        # Store results
        results["curvature_statistics"].append(curvature_result["statistics"])
        results["all_results"].append(curvature_result)
        
        # Print progress
        success_rate = curvature_result["statistics"]["success_rate"]
        mean_curv = curvature_result["statistics"]["mean"]
        print(f"  Success rate: {success_rate:.1%}, Mean curvature: {mean_curv:.4f}")
    
    return results
