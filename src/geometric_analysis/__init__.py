"""
Geometric analysis module for manifold datasets.

This module provides tools for:
- Intrinsic dimension estimation
- Curvature estimation (mean, Gaussian, Ricci)
- Geometric statistics (diameter, volume, reach)
- Comprehensive geometric analysis
"""

# Import functions only when the module is actually used
# This avoids import errors during initial setup

def __getattr__(name):
    """Lazy import to handle missing dependencies gracefully."""
    if name == 'DimensionEstimator':
        from .dimension_estimation import DimensionEstimator
        return DimensionEstimator
    
    elif name in ['estimate_mean_curvature', 'estimate_gaussian_curvature', 
                  'estimate_ricci_curvature', 'estimate_curvature_pca_based',
                  'compute_curvature_statistics']:
        from .curvature_estimation import (
            estimate_mean_curvature,
            estimate_gaussian_curvature, 
            estimate_ricci_curvature,
            estimate_curvature_pca_based,
            compute_curvature_statistics
        )
        return locals()[name]
    
    elif name in ['estimate_mean_curvature_at_point', 'estimate_pca_curvature_at_point', 
                  'estimate_gaussian_curvature_at_point']:
        from .point_curvature import (
            estimate_mean_curvature_at_point,
            estimate_pca_curvature_at_point,
            estimate_gaussian_curvature_at_point
        )
        return locals()[name]
    
    elif name in ['compute_manifold_curvature', 'compute_curvature_vs_parameter',
                  'get_k_nearest_neighbors', 'get_radius_neighbors']:
        from .manifold_curvature import (
            compute_manifold_curvature,
            compute_curvature_vs_parameter,
            get_k_nearest_neighbors,
            get_radius_neighbors
        )
        return locals()[name]
    
    elif name in ['estimate_extrinsic_diameter', 'estimate_intrinsic_diameter',
                  'estimate_volume', 'estimate_reach', 'compute_geodesic_distances',
                  'compute_geometric_summary', 'analyze_point_density']:
        from .geometric_statistics import (
            estimate_extrinsic_diameter,
            estimate_intrinsic_diameter,
            estimate_volume,
            estimate_reach,
            compute_geodesic_distances,
            compute_geometric_summary,
            analyze_point_density
        )
        return locals()[name]
    
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

__all__ = [
    # Dimension estimation
    'DimensionEstimator',
    
    # Legacy curvature estimation (whole dataset)
    'estimate_mean_curvature',
    'estimate_gaussian_curvature',
    'estimate_ricci_curvature', 
    'estimate_curvature_pca_based',
    'compute_curvature_statistics',
    
    # Point-wise curvature estimation
    'estimate_mean_curvature_at_point',
    'estimate_pca_curvature_at_point', 
    'estimate_gaussian_curvature_at_point',
    
    # Manifold-level curvature analysis
    'compute_manifold_curvature',
    'compute_curvature_vs_parameter',
    'get_k_nearest_neighbors',
    'get_radius_neighbors',
    
    # Geometric statistics
    'estimate_extrinsic_diameter',
    'estimate_intrinsic_diameter',
    'estimate_volume',
    'estimate_reach',
    'compute_geodesic_distances',
    'compute_geometric_summary',
    'analyze_point_density'
]
