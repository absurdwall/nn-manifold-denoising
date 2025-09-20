"""
Core curvature estimation functions for individual points.

This module provides functions to estimate curvature at a single point
given its neighbors, without handling neighbor selection or sampling.
"""

import numpy as np
from sklearn.decomposition import PCA
from typing import Tuple, Optional, Dict
import warnings


def estimate_mean_curvature_at_point(
    point: np.ndarray,
    neighbors: np.ndarray,
    intrinsic_point: Optional[np.ndarray] = None,
    intrinsic_neighbors: Optional[np.ndarray] = None,
    weight: str = "gaussian",
    bandwidth: Optional[float] = None,
    regularization: float = 1e-10,
    normalize_intrinsic: bool = True
) -> Tuple[float, Dict]:
    """
    Estimate mean curvature at a single point using differential geometry.
    
    This function estimates curvature at one point given its neighbors,
    using the intrinsic-extrinsic coordinate relationship if available.
    
    Args:
        point: (D,) target point in embedded space
        neighbors: (k, D) neighbor points in embedded space
        intrinsic_point: (d,) target point in intrinsic space (optional)
        intrinsic_neighbors: (k, d) neighbor points in intrinsic space (optional)
        weight: Weighting scheme ("gaussian" or "uniform")
        bandwidth: Kernel bandwidth (auto-selected if None)
        regularization: Ridge regularization parameter
        normalize_intrinsic: Whether to normalize intrinsic coordinates locally
        
    Returns:
        curvature: Mean curvature magnitude at the point
        info: Dictionary with diagnostic information
    """
    
    # If no intrinsic coordinates provided, fall back to PCA-based method
    if intrinsic_point is None or intrinsic_neighbors is None:
        return estimate_pca_curvature_at_point(point, neighbors)
    
    # Combine point with neighbors for processing (same as original implementation)
    all_intrinsic = np.vstack([intrinsic_point[None, :], intrinsic_neighbors])
    all_embedded = np.vstack([point[None, :], neighbors])
    
    X = all_intrinsic
    Y = all_embedded
    d = X.shape[1]
    D = Y.shape[1]
    k = len(neighbors)
    
    # Center at the base point (same as original)
    Xc = X - X[0]
    Yc = Y - Y[0]
    
    # Triangular indices for quadratic terms (same as original)
    tri_pairs = [(j, k) for j in range(d) for k in range(j, d)]
    n_quad = len(tri_pairs)
    n_coeffs = 1 + d + n_quad  # constant + linear + quadratic
    
    # Need enough neighbors for quadratic fit
    if k + 1 < n_coeffs:
        return np.nan, {"error": "too_few_neighbors", "k": k, "n_coeffs": n_coeffs}
    
    try:
        # Local normalization for better conditioning (same as original)
        if normalize_intrinsic:
            scale = np.linalg.norm(Xc, axis=0) / np.sqrt(max(1, Xc.shape[0]))
            scale[scale == 0.0] = 1.0
            Z = Xc / scale[None, :]
        else:
            scale = np.ones(d)
            Z = Xc
        
        # Build design matrix for quadratic regression (same as original)
        def build_design(Z):
            k_pts = Z.shape[0]
            A = np.zeros((k_pts, n_coeffs))
            A[:, 0] = 1.0
            A[:, 1:1 + d] = Z
            col = 1 + d
            for (j, k_idx) in tri_pairs:
                A[:, col] = Z[:, j] * Z[:, k_idx]
                col += 1
            return A
        
        A = build_design(Z)
        
        # Radial weights (same as original)
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
        
        # Weighted ridge regression (same as original)
        AtWA = Aw.T @ Aw
        AtWA.flat[::AtWA.shape[0]+1] += regularization
        AtWY = Aw.T @ Yw
        
        # Solve for coefficients
        coeffs = np.linalg.solve(AtWA, AtWY)
        
        # Jacobian and correct for normalization (same as original)
        J = coeffs[1:1 + d, :].T
        J = J / scale[None, :]
        
        # SVD for tangent space (same as original)
        U, S, Vt = np.linalg.svd(J, full_matrices=False)
        rank = int(np.sum(S > 1e-12))
        
        if rank < d:
            return np.nan, {"error": "rank_deficient_J", "rank": rank, "d": d}
        
        tangent_basis = U[:, :d]
        normal_proj = np.eye(D) - tangent_basis @ tangent_basis.T
        
        # Build Hessian tensor (same as original)
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
        
        # Correct Hessian for normalization (same as original)
        inv_scale = 1.0 / scale
        for j in range(d):
            for k_idx in range(d):
                H[:, j, k_idx] *= inv_scale[j] * inv_scale[k_idx]
        
        # Project to normal space (same as original)
        Hn = np.zeros_like(H)
        for j in range(d):
            for k_idx in range(d):
                Hn[:, j, k_idx] = normal_proj @ H[:, j, k_idx]
        
        # Metric and inverse (same as original)
        g = J.T @ J
        g_inv = np.linalg.pinv(g, rcond=1e-12)
        
        # Mean curvature vector (same as original)
        H_mean_vec = np.zeros(D)
        for j in range(d):
            for k_idx in range(d):
                H_mean_vec += g_inv[j, k_idx] * Hn[:, j, k_idx]
        H_mean_vec /= d
        
        curvature = np.linalg.norm(H_mean_vec)
        
        info = {
            "success": True,
            "rank": rank,
            "condition_number": S[0] / S[-1] if S[-1] > 0 else np.inf,
            "num_neighbors": k
        }
        
        return curvature, info
        
    except Exception as e:
        return np.nan, {"error": f"exception_{type(e).__name__}", "k": k}


def estimate_pca_curvature_at_point(
    point: np.ndarray,
    neighbors: np.ndarray,
    intrinsic_dim: Optional[int] = None
) -> Tuple[float, Dict]:
    """
    Estimate curvature at a single point using PCA-based analysis.
    
    This method doesn't require intrinsic coordinates and provides a
    simpler alternative based on local deviation from linear subspace.
    
    Args:
        point: (D,) target point in embedded space
        neighbors: (k, D) neighbor points in embedded space
        intrinsic_dim: Expected intrinsic dimension (estimated if None)
        
    Returns:
        curvature: Curvature estimate at the point
        info: Dictionary with diagnostic information
    """
    
    # Combine point with neighbors
    all_points = np.vstack([point[None, :], neighbors])
    D = all_points.shape[1]
    k = len(neighbors)
    
    # Center the neighborhood at the target point
    centered = all_points - point[None, :]
    
    if intrinsic_dim is None:
        # Simple heuristic: use PCA to estimate intrinsic dimension
        pca_temp = PCA()
        pca_temp.fit(centered)
        explained_var_ratio = pca_temp.explained_variance_ratio_
        # Find dimension that captures 95% of variance
        cumsum = np.cumsum(explained_var_ratio)
        intrinsic_dim = int(np.argmax(cumsum >= 0.95) + 1)
        intrinsic_dim = max(1, min(intrinsic_dim, D-1))
    
    if k < intrinsic_dim + 1:
        return np.nan, {"error": "too_few_neighbors", "k": k, "intrinsic_dim": intrinsic_dim}
    
    try:
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
            return np.nan, {"error": "too_few_valid_distances", "valid_count": np.sum(valid_idx)}
        
        # Fit quadratic relationship: residual ≈ c * distance^2
        # This gives a rough curvature estimate
        coeffs = np.polyfit(distances[valid_idx]**2, residual_norms[valid_idx], 1)
        curvature_est = abs(coeffs[0])  # Coefficient of distance^2 term
        
        info = {
            "success": True,
            "intrinsic_dim": intrinsic_dim,
            "num_neighbors": k,
            "explained_variance_ratio": float(np.sum(pca.explained_variance_ratio_[:intrinsic_dim])),
            "valid_distances": int(np.sum(valid_idx))
        }
        
        return curvature_est, info
        
    except Exception as e:
        return np.nan, {"error": f"exception_{type(e).__name__}", "k": k}


def estimate_gaussian_curvature_at_point(
    point: np.ndarray,
    neighbors: np.ndarray,
    intrinsic_point: Optional[np.ndarray] = None,
    intrinsic_neighbors: Optional[np.ndarray] = None
) -> Tuple[float, Dict]:
    """
    Estimate Gaussian curvature at a single point.
    
    Note: This is a placeholder for future implementation.
    Gaussian curvature estimation is more complex and requires
    careful handling of the determinant of the shape operator.
    
    Args:
        point: (D,) target point in embedded space
        neighbors: (k, D) neighbor points in embedded space
        intrinsic_point: (d,) target point in intrinsic space (optional)
        intrinsic_neighbors: (k, d) neighbor points in intrinsic space (optional)
        
    Returns:
        curvature: Gaussian curvature estimate (currently NaN)
        info: Dictionary with diagnostic information
    """
    return np.nan, {"error": "not_implemented", "method": "gaussian_curvature"}
