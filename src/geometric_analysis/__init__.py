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
    
    # Curvature estimation
    'estimate_mean_curvature',
    'estimate_gaussian_curvature',
    'estimate_ricci_curvature', 
    'estimate_curvature_pca_based',
    'compute_curvature_statistics',
    
    # Geometric statistics
    'estimate_extrinsic_diameter',
    'estimate_intrinsic_diameter',
    'estimate_volume',
    'estimate_reach',
    'compute_geodesic_distances',
    'compute_geometric_summary',
    'analyze_point_density'
]
