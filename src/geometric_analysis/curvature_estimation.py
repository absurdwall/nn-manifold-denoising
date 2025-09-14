"""
Curvature estimation for manifold datasets.

This module provides multiple approaches to curvature estimation including:
- Mean curvature estimation (improved version)
- Gaussian curvature estimation
- Ricci curvature estimation
- Alternative curvature measures
"""

import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from typing import Tuple, Optional, Dict, Union
import warnings


def estimate_mean_curvature(
    intrinsic_coords: np.ndarray,
    embedded_coords: np.ndarray,
    k_neighbors: int = 40,
    max_points: Optional[int] = None,
    weight: str = "gaussian",
    bandwidth: Optional[float] = None,
    regularization: float = 1e-10,
    normalize_intrinsic: bool = True,
    return_details: bool = False,
    random_state: Optional[int] = 42,
) -> Union[np.ndarray, Tuple[np.ndarray, Dict]]:
    """
    Improved parameterization-invariant pointwise mean curvature magnitude estimation.
    
    This is based on the original implementation but with additional robustness improvements
    and better error handling.
    
    Args:
        intrinsic_coords: (N, d) array of intrinsic coordinates
        embedded_coords: (N, D) array of embedded coordinates
        k_neighbors: Number of neighbors for local fitting
        max_points: Maximum points to process (for speed)
        weight: Weighting scheme ("gaussian" or "uniform")
        bandwidth: Kernel bandwidth (auto-selected if None)
        regularization: Ridge regularization parameter
        normalize_intrinsic: Whether to normalize intrinsic coordinates locally
        return_details: Whether to return diagnostic information
        random_state: Random seed for reproducibility
        
    Returns:
        curvatures: (N,) array of mean curvature magnitudes
        details: Optional diagnostic dictionary (if return_details=True)
    """
    N, d = intrinsic_coords.shape
    _, D = embedded_coords.shape

    # Triangular indices for quadratic terms (j <= k)
    tri_pairs = [(j, k) for j in range(d) for k in range(j, d)]
    n_quad = len(tri_pairs)
    n_coeffs = 1 + d + n_quad  # constant + linear + quadratic

    # Choose which points to process
    rng = np.random.default_rng(random_state)
    indices_to_process = np.arange(N)
    if max_points is not None and max_points < N:
        indices_to_process = rng.choice(N, size=max_points, replace=False)

    # Nearest neighbors in intrinsic space
    k_eff = min(k_neighbors, N)
    nbrs = NearestNeighbors(n_neighbors=k_eff, algorithm='auto').fit(intrinsic_coords)
    _, all_indices = nbrs.kneighbors(intrinsic_coords)

    curvatures = np.full(N, np.nan)

    # Optional diagnostics
    diag = {
        "condition_A": np.full(N, np.nan),
        "rank_J": np.full(N, np.nan),
        "fail_reason": np.array([""] * N, dtype=object),
        "num_neighbors": np.full(N, np.nan),
    } if return_details else None

    def build_design(Z):
        """Build design matrix for quadratic regression."""
        k = Z.shape[0]
        A = np.zeros((k, n_coeffs))
        A[:, 0] = 1.0
        A[:, 1:1 + d] = Z
        col = 1 + d
        for (j, k_idx) in tri_pairs:
            A[:, col] = Z[:, j] * Z[:, k_idx]
            col += 1
        return A

    for idx in indices_to_process:
        try:
            neigh = all_indices[idx]
            X = intrinsic_coords[neigh]
            Y = embedded_coords[neigh]

            # Center at the base point
            Xc = X - X[0]
            Yc = Y - Y[0]

            # Need enough neighbors for quadratic fit
            if Xc.shape[0] < n_coeffs:
                if return_details: 
                    diag["fail_reason"][idx] = "too_few_neighbors"
                    diag["num_neighbors"][idx] = Xc.shape[0]
                continue

            # Local normalization for better conditioning
            if normalize_intrinsic:
                scale = np.linalg.norm(Xc, axis=0) / np.sqrt(max(1, Xc.shape[0]))
                scale[scale == 0.0] = 1.0
                Z = Xc / scale[None, :]
            else:
                scale = np.ones(d)
                Z = Xc

            # Design matrix
            A = build_design(Z)

            # Radial weights
            if weight == "gaussian":
                r2 = np.sum(Xc**2, axis=1)
                if bandwidth is None:
                    med_r = np.sqrt(np.median(r2)) if Xc.shape[0] > 1 else 1.0
                    h = med_r / max(1e-12, np.sqrt(2.0 * np.log(2.0)))
                else:
                    h = bandwidth
                w = np.exp(-r2 / max(1e-18, 2.0 * h * h))
            else:
                w = np.ones(Xc.shape[0])

            Wsqrt = np.sqrt(w)[:, None]
            Aw = Wsqrt * A
            Yw = Wsqrt * Yc

            # Weighted ridge regression
            AtWA = Aw.T @ Aw
            AtWA.flat[::AtWA.shape[0]+1] += regularization
            AtWY = Aw.T @ Yw

            # Solve for coefficients
            try:
                coeffs = np.linalg.solve(AtWA, AtWY)
            except np.linalg.LinAlgError:
                if return_details: diag["fail_reason"][idx] = "solve_failed"
                continue

            # Jacobian and correct for normalization
            J = coeffs[1:1 + d, :].T
            J = J / scale[None, :]

            # SVD for tangent space
            try:
                U, S, Vt = np.linalg.svd(J, full_matrices=False)
                rank = int(np.sum(S > 1e-12))
                if return_details:
                    diag["rank_J"][idx] = rank

                if rank < d:
                    if return_details: diag["fail_reason"][idx] = "rank_deficient_J"
                    continue

                tangent_basis = U[:, :d]
                normal_proj = np.eye(D) - tangent_basis @ tangent_basis.T

            except np.linalg.LinAlgError:
                if return_details: diag["fail_reason"][idx] = "svd_failed"
                continue

            # Build Hessian tensor
            H = np.zeros((D, d, d))
            col = 1 + d
            for (j, k_idx) in tri_pairs:
                c = coeffs[col, :]
                if j == k_idx:
                    H[:, j, k_idx] = 2.0 * c
                else:
                    H[:, j, k_idx] = c
                    H[:, k_idx, j] = c
                col += 1

            # Correct Hessian for normalization
            inv_scale = 1.0 / scale
            for j in range(d):
                for k_idx in range(d):
                    H[:, j, k_idx] *= inv_scale[j] * inv_scale[k_idx]

            # Project to normal space
            Hn = np.zeros_like(H)
            for j in range(d):
                for k_idx in range(d):
                    Hn[:, j, k_idx] = normal_proj @ H[:, j, k_idx]

            # Metric and inverse
            g = J.T @ J
            try:
                g_inv = np.linalg.pinv(g, rcond=1e-12)
            except np.linalg.LinAlgError:
                if return_details: diag["fail_reason"][idx] = "g_pinv_fail"
                continue

            # Mean curvature vector
            H_mean_vec = np.zeros(D)
            for j in range(d):
                for k_idx in range(d):
                    H_mean_vec += g_inv[j, k_idx] * Hn[:, j, k_idx]
            H_mean_vec /= d

            curvatures[idx] = np.linalg.norm(H_mean_vec)

            if return_details:
                svals = np.linalg.svd(A, full_matrices=False, compute_uv=False)
                diag["condition_A"][idx] = (svals[0] / svals[-1]) if svals[-1] > 0 else np.inf
                diag["num_neighbors"][idx] = Xc.shape[0]

        except Exception as e:
            if return_details: diag["fail_reason"][idx] = f"exception:{type(e).__name__}"
            continue

    if return_details:
        return curvatures, diag
    return curvatures


def estimate_gaussian_curvature(
    intrinsic_coords: np.ndarray,
    embedded_coords: np.ndarray,
    k_neighbors: int = 40,
    max_points: Optional[int] = None
) -> np.ndarray:
    """
    Estimate Gaussian curvature using local triangulation approach.
    
    This provides an alternative curvature measure that doesn't depend on 
    the mean curvature calculation.
    
    Args:
        intrinsic_coords: (N, d) array of intrinsic coordinates
        embedded_coords: (N, D) array of embedded coordinates  
        k_neighbors: Number of neighbors for local computation
        max_points: Maximum points to process
        
    Returns:
        gaussian_curvatures: (N,) array of Gaussian curvature estimates
    """
    N, d = intrinsic_coords.shape
    
    if d != 2:
        warnings.warn("Gaussian curvature estimation is most reliable for 2D manifolds")
    
    # For now, return placeholder - this is a more complex computation
    # Would require proper implementation of discrete Gaussian curvature
    gaussian_curvatures = np.full(N, np.nan)
    
    # TODO: Implement discrete Gaussian curvature using angle defect method
    # or other appropriate discrete differential geometry approach
    
    return gaussian_curvatures


def estimate_ricci_curvature(
    embedded_coords: np.ndarray,
    k_neighbors: int = 40,
    max_points: Optional[int] = None
) -> np.ndarray:
    """
    Estimate Ricci curvature using Ollivier-Ricci curvature on the point cloud.
    
    This provides a graph-based approach to curvature that doesn't require
    intrinsic coordinates.
    
    Args:
        embedded_coords: (N, D) array of embedded coordinates
        k_neighbors: Number of neighbors for graph construction  
        max_points: Maximum points to process
        
    Returns:
        ricci_curvatures: (N,) array of Ricci curvature estimates
    """
    N, D = embedded_coords.shape
    
    # For now, return placeholder - this requires implementation of
    # Ollivier-Ricci curvature computation
    ricci_curvatures = np.full(N, np.nan)
    
    # TODO: Implement Ollivier-Ricci curvature using optimal transport
    # or use existing library like NetworkX if appropriate
    
    return ricci_curvatures


def estimate_curvature_pca_based(
    embedded_coords: np.ndarray,
    k_neighbors: int = 40,
    intrinsic_dim: Optional[int] = None
) -> np.ndarray:
    """
    Alternative curvature estimation using PCA-based local analysis.
    
    This method doesn't require intrinsic coordinates and provides a
    simpler alternative to the full differential geometry approach.
    
    Args:
        embedded_coords: (N, D) array of embedded coordinates
        k_neighbors: Number of neighbors for local PCA
        intrinsic_dim: Expected intrinsic dimension (estimated if None)
        
    Returns:
        curvatures: (N,) array of curvature estimates
    """
    N, D = embedded_coords.shape
    
    # Estimate intrinsic dimension if not provided
    if intrinsic_dim is None:
        from .dimension_estimation import DimensionEstimator
        estimator = DimensionEstimator(verbose=False)
        results = estimator.fit(embedded_coords, methods=['PCA'])
        intrinsic_dim = int(np.round(results.get('PCA', 2)))
        intrinsic_dim = max(1, min(intrinsic_dim, D-1))
    
    # Nearest neighbors
    k_eff = min(k_neighbors, N)
    nbrs = NearestNeighbors(n_neighbors=k_eff, algorithm='auto').fit(embedded_coords)
    _, indices = nbrs.kneighbors(embedded_coords)
    
    curvatures = np.full(N, np.nan)
    
    for i in range(N):
        try:
            # Get local neighborhood
            local_points = embedded_coords[indices[i]]
            
            # Center the neighborhood
            centered = local_points - local_points[0]
            
            if len(centered) < intrinsic_dim + 1:
                continue
            
            # PCA to get local tangent space
            pca = PCA()
            pca.fit(centered)
            
            # Project to tangent space
            tangent_proj = pca.components_[:intrinsic_dim]
            projected = centered @ tangent_proj.T
            
            # Compute residuals (normal component)
            reconstructed = projected @ tangent_proj
            residuals = centered - reconstructed
            residual_norms = np.linalg.norm(residuals, axis=1)
            
            # Estimate curvature from residual pattern
            # Points further from center should have larger residuals for curved manifolds
            distances = np.linalg.norm(projected, axis=1)
            
            # Avoid division by zero
            valid_idx = distances > 1e-12
            if np.sum(valid_idx) < 3:
                continue
            
            # Fit quadratic relationship: residual ≈ c * distance^2
            # This gives a rough curvature estimate
            try:
                # Use robust fitting
                coeffs = np.polyfit(distances[valid_idx]**2, residual_norms[valid_idx], 1)
                curvature_est = abs(coeffs[0])  # Coefficient of distance^2 term
                curvatures[i] = curvature_est
            except:
                continue
                
        except Exception:
            continue
    
    return curvatures


def compute_curvature_statistics(curvatures: np.ndarray) -> Dict[str, float]:
    """
    Compute summary statistics for curvature values.
    
    Args:
        curvatures: Array of curvature values (may contain NaN)
        
    Returns:
        Dictionary of curvature statistics
    """
    valid_curvatures = curvatures[np.isfinite(curvatures)]
    
    if len(valid_curvatures) == 0:
        return {
            "mean": np.nan,
            "std": np.nan,
            "median": np.nan,
            "min": np.nan,
            "max": np.nan,
            "num_valid": 0,
            "num_total": len(curvatures),
            "success_rate": 0.0
        }
    
    return {
        "mean": float(np.mean(valid_curvatures)),
        "std": float(np.std(valid_curvatures)),
        "median": float(np.median(valid_curvatures)),
        "min": float(np.min(valid_curvatures)),
        "max": float(np.max(valid_curvatures)),
        "num_valid": int(len(valid_curvatures)),
        "num_total": int(len(curvatures)),
        "success_rate": float(len(valid_curvatures) / len(curvatures))
    }
