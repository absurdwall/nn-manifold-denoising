"""
Geometric statistics and analysis for manifold datasets.

This module provides functions for computing various geometric quantities
and statistics on manifold data, including diameter estimation and other
geometric measures.
"""

import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from typing import Tuple, Optional, Dict, List, Union
import warnings


def estimate_extrinsic_diameter(
    embedded_coords: np.ndarray,
    method: str = "max_distance",
    sample_size: Optional[int] = None,
    random_state: Optional[int] = 42
) -> float:
    """
    Estimate the extrinsic diameter of the manifold.
    
    Args:
        embedded_coords: (N, D) array of embedded coordinates
        method: Method to use ("max_distance", "percentile", "sample_based")
        sample_size: Number of points to sample for efficiency (None = use all)
        random_state: Random seed for reproducibility
        
    Returns:
        Estimated extrinsic diameter
    """
    N, D = embedded_coords.shape
    
    if sample_size is not None and sample_size < N:
        rng = np.random.default_rng(random_state)
        indices = rng.choice(N, size=sample_size, replace=False)
        coords = embedded_coords[indices]
    else:
        coords = embedded_coords
    
    if method == "max_distance":
        # Compute all pairwise distances and find maximum
        from scipy.spatial.distance import pdist
        distances = pdist(coords)
        return float(np.max(distances))
    
    elif method == "percentile":
        # Use 95th percentile to be robust to outliers
        from scipy.spatial.distance import pdist
        distances = pdist(coords)
        return float(np.percentile(distances, 95))
    
    elif method == "sample_based":
        # Sample pairs of points to estimate diameter efficiently
        n_samples = min(10000, len(coords) * (len(coords) - 1) // 2)
        rng = np.random.default_rng(random_state)
        
        max_dist = 0.0
        for _ in range(n_samples):
            i, j = rng.choice(len(coords), size=2, replace=False)
            dist = np.linalg.norm(coords[i] - coords[j])
            max_dist = max(max_dist, dist)
        
        return float(max_dist)
    
    else:
        raise ValueError(f"Unknown method: {method}")


def estimate_intrinsic_diameter(
    intrinsic_coords: np.ndarray,
    method: str = "max_distance"
) -> float:
    """
    Estimate the intrinsic diameter using intrinsic coordinates.
    
    Args:
        intrinsic_coords: (N, d) array of intrinsic coordinates
        method: Method to use for estimation
        
    Returns:
        Estimated intrinsic diameter
    """
    if method == "max_distance":
        from scipy.spatial.distance import pdist
        distances = pdist(intrinsic_coords)
        return float(np.max(distances))
    else:
        raise ValueError(f"Unknown method: {method}")


def estimate_volume(
    intrinsic_coords: np.ndarray,
    method: str = "convex_hull"
) -> float:
    """
    Estimate the volume of the manifold using intrinsic coordinates.
    
    Args:
        intrinsic_coords: (N, d) array of intrinsic coordinates
        method: Method to use ("convex_hull", "kde_based")
        
    Returns:
        Estimated volume
    """
    N, d = intrinsic_coords.shape
    
    if method == "convex_hull":
        if d <= 3:
            try:
                from scipy.spatial import ConvexHull
                hull = ConvexHull(intrinsic_coords)
                return float(hull.volume)
            except Exception:
                # Fall back to bounding box volume
                ranges = np.max(intrinsic_coords, axis=0) - np.min(intrinsic_coords, axis=0)
                return float(np.prod(ranges))
        else:
            # High-dimensional case: use bounding box volume
            ranges = np.max(intrinsic_coords, axis=0) - np.min(intrinsic_coords, axis=0)
            return float(np.prod(ranges))
    
    elif method == "kde_based":
        # Use KDE to estimate density and integrate
        # This is more complex and approximate
        from sklearn.neighbors import KernelDensity
        
        # Use Scott's rule for bandwidth
        n = len(intrinsic_coords)
        bandwidth = n**(-1/(d+4))
        
        kde = KernelDensity(bandwidth=bandwidth)
        kde.fit(intrinsic_coords)
        
        # Estimate volume by sampling
        # This is very approximate
        ranges = np.max(intrinsic_coords, axis=0) - np.min(intrinsic_coords, axis=0)
        bbox_volume = np.prod(ranges)
        
        # Sample points in bounding box and estimate what fraction is on manifold
        n_samples = min(10000, n * 10)
        mins = np.min(intrinsic_coords, axis=0)
        maxs = np.max(intrinsic_coords, axis=0)
        
        rng = np.random.default_rng(42)
        sample_points = rng.uniform(
            low=mins, high=maxs, size=(n_samples, d)
        )
        
        log_densities = kde.score_samples(sample_points)
        # Points with reasonable density are "on" the manifold
        threshold = np.percentile(kde.score_samples(intrinsic_coords), 10)
        on_manifold = np.sum(log_densities > threshold)
        
        estimated_volume = bbox_volume * (on_manifold / n_samples)
        return float(estimated_volume)
    
    else:
        raise ValueError(f"Unknown method: {method}")


def compute_geodesic_distances(
    embedded_coords: np.ndarray,
    k_neighbors: int = 10,
    method: str = "dijkstra"
) -> np.ndarray:
    """
    Compute approximate geodesic distances using graph-based methods.
    
    Args:
        embedded_coords: (N, D) array of embedded coordinates
        k_neighbors: Number of neighbors for graph construction
        method: Method to use ("dijkstra", "floyd_warshall")
        
    Returns:
        (N, N) matrix of approximate geodesic distances
    """
    N = len(embedded_coords)
    
    # Build k-nearest neighbor graph
    nbrs = NearestNeighbors(n_neighbors=k_neighbors+1, algorithm='auto')
    nbrs.fit(embedded_coords)
    distances, indices = nbrs.kneighbors(embedded_coords)
    
    # Create weighted adjacency matrix
    from scipy.sparse import csr_matrix
    
    # Flatten the arrays for sparse matrix construction
    row_indices = []
    col_indices = []
    edge_weights = []
    
    for i in range(N):
        for j in range(1, len(indices[i])):  # Skip self (index 0)
            neighbor_idx = indices[i][j]
            distance = distances[i][j]
            
            row_indices.extend([i, neighbor_idx])
            col_indices.extend([neighbor_idx, i])
            edge_weights.extend([distance, distance])
    
    # Create sparse adjacency matrix
    adj_matrix = csr_matrix(
        (edge_weights, (row_indices, col_indices)),
        shape=(N, N)
    )
    
    # Compute shortest paths
    if method == "dijkstra":
        from scipy.sparse.csgraph import shortest_path
        geodesic_distances = shortest_path(
            adj_matrix, method='D', directed=False
        )
    elif method == "floyd_warshall":
        from scipy.sparse.csgraph import shortest_path
        geodesic_distances = shortest_path(
            adj_matrix, method='FW', directed=False
        )
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return geodesic_distances


def estimate_reach(
    embedded_coords: np.ndarray,
    intrinsic_coords: Optional[np.ndarray] = None,
    k_neighbors: int = 20
) -> float:
    """
    Estimate the reach of the manifold (minimum radius of curvature).
    
    Args:
        embedded_coords: (N, D) array of embedded coordinates
        intrinsic_coords: Optional (N, d) array of intrinsic coordinates
        k_neighbors: Number of neighbors for local analysis
        
    Returns:
        Estimated reach
    """
    N, D = embedded_coords.shape
    
    # Simple approach: use local PCA to estimate how "flat" the manifold is
    nbrs = NearestNeighbors(n_neighbors=k_neighbors, algorithm='auto')
    nbrs.fit(embedded_coords)
    _, indices = nbrs.kneighbors(embedded_coords)
    
    local_flatness = []
    
    for i in range(N):
        try:
            local_points = embedded_coords[indices[i]]
            centered = local_points - local_points[0]
            
            if len(centered) < 3:
                continue
            
            # PCA to get principal components
            pca = PCA()
            pca.fit(centered)
            
            # The ratio of smallest to largest eigenvalue indicates flatness
            eigenvals = pca.explained_variance_
            if len(eigenvals) > 1 and eigenvals[0] > 1e-12:
                flatness = eigenvals[-1] / eigenvals[0]
                local_flatness.append(flatness)
        
        except Exception:
            continue
    
    if len(local_flatness) == 0:
        return np.inf
    
    # Reach is inversely related to curvature
    # This is a very rough approximation
    mean_flatness = np.mean(local_flatness)
    if mean_flatness > 1e-12:
        reach_estimate = 1.0 / np.sqrt(mean_flatness)
    else:
        reach_estimate = np.inf
    
    return float(reach_estimate)


def compute_geometric_summary(
    embedded_coords: np.ndarray,
    intrinsic_coords: Optional[np.ndarray] = None,
    include_curvature: bool = True,
    include_dimension: bool = True
) -> Dict[str, Union[float, int, Dict]]:
    """
    Compute a comprehensive summary of geometric properties.
    
    Args:
        embedded_coords: (N, D) array of embedded coordinates
        intrinsic_coords: Optional (N, d) array of intrinsic coordinates
        include_curvature: Whether to compute curvature statistics
        include_dimension: Whether to estimate intrinsic dimension
        
    Returns:
        Dictionary containing geometric summary statistics
    """
    N, D = embedded_coords.shape
    
    summary = {
        "num_points": N,
        "embedding_dimension": D,
        "extrinsic_diameter": estimate_extrinsic_diameter(embedded_coords),
        "reach_estimate": estimate_reach(embedded_coords, intrinsic_coords)
    }
    
    if intrinsic_coords is not None:
        d = intrinsic_coords.shape[1]
        summary.update({
            "intrinsic_dimension": d,
            "intrinsic_diameter": estimate_intrinsic_diameter(intrinsic_coords),
            "estimated_volume": estimate_volume(intrinsic_coords)
        })
    
    if include_dimension and intrinsic_coords is None:
        from .dimension_estimation import DimensionEstimator
        estimator = DimensionEstimator(verbose=False)
        dim_results = estimator.fit(embedded_coords)
        summary["dimension_estimates"] = dim_results
        
        # Use consensus dimension if available
        if "consensus" in dim_results:
            summary["estimated_intrinsic_dimension"] = dim_results["consensus"]
    
    if include_curvature:
        from .curvature_estimation import estimate_mean_curvature, compute_curvature_statistics
        
        if intrinsic_coords is not None:
            # Can compute full curvature analysis
            curvatures = estimate_mean_curvature(
                intrinsic_coords, embedded_coords, max_points=min(1000, N)
            )
            summary["curvature_statistics"] = compute_curvature_statistics(curvatures)
        else:
            # Use PCA-based method
            from .curvature_estimation import estimate_curvature_pca_based
            curvatures = estimate_curvature_pca_based(
                embedded_coords, k_neighbors=20
            )
            summary["curvature_statistics_pca"] = compute_curvature_statistics(curvatures)
    
    return summary


def analyze_point_density(
    embedded_coords: np.ndarray,
    k_neighbors: int = 10
) -> Dict[str, float]:
    """
    Analyze the density distribution of points on the manifold.
    
    Args:
        embedded_coords: (N, D) array of embedded coordinates
        k_neighbors: Number of neighbors for density estimation
        
    Returns:
        Dictionary with density statistics
    """
    N = len(embedded_coords)
    
    # Use k-nearest neighbor distances as density proxy
    nbrs = NearestNeighbors(n_neighbors=k_neighbors+1, algorithm='auto')
    nbrs.fit(embedded_coords)
    distances, _ = nbrs.kneighbors(embedded_coords)
    
    # Use mean distance to k-th nearest neighbor
    knn_distances = distances[:, -1]  # k-th nearest neighbor distance
    
    # Density is inverse of volume of k-neighborhood
    # Volume of d-ball with radius r is proportional to r^d
    # So density ~ 1/r^d, but we don't know d exactly
    # Use simple inverse for now
    densities = 1.0 / (knn_distances + 1e-12)
    
    return {
        "mean_density": float(np.mean(densities)),
        "std_density": float(np.std(densities)),
        "min_density": float(np.min(densities)),
        "max_density": float(np.max(densities)),
        "density_ratio": float(np.max(densities) / np.min(densities)),
        "mean_knn_distance": float(np.mean(knn_distances)),
        "std_knn_distance": float(np.std(knn_distances))
    }
