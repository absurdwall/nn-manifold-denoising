"""
Core manifold generation functions.
Based on the working approach from test_corrected_manifold.py
"""

import numpy as np
import time
from typing import Tuple, Optional
from sklearn.metrics import pairwise_distances


def rbf_kernel(X1: np.ndarray, X2: Optional[np.ndarray] = None, 
               length_scale: float = 1.0) -> np.ndarray:
    """
    Compute RBF (Gaussian) kernel matrix between point sets.
    
    Args:
        X1: (N1, d) array of points
        X2: (N2, d) array of points (if None, use X1)
        length_scale: Kernel length scale parameter
        
    Returns:
        (N1, N2) kernel matrix
    """
    if X2 is None:
        X2 = X1
    
    sq_dists = np.square(pairwise_distances(X1, X2, n_jobs=-1))
    return np.exp(-0.5 * (1 / length_scale**2) * sq_dists)


def generate_intrinsic_coordinates(n: int, d: int, base_type: str = 'unit_ball', 
                                  random_seed: Optional[int] = None) -> np.ndarray:
    """
    Generate intrinsic coordinates on base manifold.
    
    Args:
        n: Number of points
        d: Intrinsic dimension
        base_type: Type of base manifold ('unit_ball', 'rectangle', 'sphere')
        random_seed: Random seed for reproducibility
        
    Returns:
        (n, d) array of intrinsic coordinates
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    
    if base_type == 'unit_ball':
        # Generate points uniformly in unit ball
        normal_samples = np.random.randn(n, d)
        norms = np.linalg.norm(normal_samples, axis=1, keepdims=True)
        unit_sphere = normal_samples / norms
        radii = np.random.random(size=(n, 1)) ** (1.0/d)
        return unit_sphere * radii
        
    elif base_type == 'rectangle':
        # Generate points uniformly in [-1, 1]^d
        return np.random.uniform(-1, 1, (n, d))
        
    elif base_type == 'sphere':
        # Generate points on unit sphere surface
        theta = np.random.uniform(0, 2 * np.pi, n)
        u = np.random.uniform(-1, 1, n)
        phi = np.arccos(u)
        sorted_indices = np.lexsort((phi, theta))
        x = np.sin(phi) * np.cos(theta)
        y = np.sin(phi) * np.sin(theta)
        z = np.cos(phi)
        coords = np.array([x[sorted_indices], y[sorted_indices], z[sorted_indices]]).T
        if d == 2:
            return coords[:, :2]  # Project to 2D if needed
        return coords
        
    else:
        raise ValueError(f"Unknown base_type: {base_type}")


def generate_manifold_embedding(k: int, N: int, d: int, D: int, 
                               kernel_smoothness: float, base_type: str = 'unit_ball',
                               random_seed: Optional[int] = None, 
                               verbose: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate manifold embedding using Gaussian Process approach.
    
    This implements the corrected approach from test_corrected_manifold.py that
    produces proper d-dimensional manifolds.
    
    Args:
        k: Number of groups
        N: Number of points per group
        d: Intrinsic dimension
        D: Ambient dimension
        kernel_smoothness: RBF kernel length scale
        base_type: Type of base manifold
        random_seed: Random seed for reproducibility
        verbose: Whether to print progress
        
    Returns:
        Tuple of (embedded_coords, intrinsic_coords)
        - embedded_coords: (k*N, D) embedded coordinates
        - intrinsic_coords: (k*N, d) intrinsic coordinates
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    
    if verbose:
        print(f"    Generating manifold: k={k}, N={N}, d={d}, D={D}, σ={kernel_smoothness}, base={base_type}")
    
    start_time = time.time()
    
    # 1. Generate intrinsic coordinates for each group
    coord_start = time.time()
    group_intrinsic_coords = []
    for i in range(k):
        coords = generate_intrinsic_coordinates(
            N, d, base_type, 
            random_seed=(random_seed + i) if random_seed is not None else None
        )
        group_intrinsic_coords.append(coords)
    
    if verbose:
        print(f"    ✓ Intrinsic coordinates: {time.time() - coord_start:.2f}s")
    
    if verbose:
        print(f"  Generated {k} groups of intrinsic coordinates")
    
    # 2. Compute RBF kernel matrices between all groups
    kernel_start = time.time()
    kernel_matrices = {}
    for i in range(k):
        for j in range(k):
            kernel_matrices[(i, j)] = rbf_kernel(
                group_intrinsic_coords[i], 
                group_intrinsic_coords[j], 
                kernel_smoothness
            )
    
    if verbose:
        print(f"    ✓ Kernel matrices ({k}x{k}): {time.time() - kernel_start:.2f}s")
    
    # 3. Generate GP functions using sequential conditioning
    gp_setup_start = time.time()
    threshold = 1e-6
    function_vals = np.empty((k, D, N))  # k groups, D functions, N points each
    mus = [[np.zeros(N) for _ in range(D)] for _ in range(k)]
    Sigma_invs = [np.linalg.inv(kernel_matrices[(i, i)] + np.eye(N) * threshold) 
                  for i in range(k)]
    
    if verbose:
        print(f"    ✓ GP setup (inversions): {time.time() - gp_setup_start:.2f}s")
        print(f"    Starting GP generation for {k} groups × {D} functions...")
    
    # Use original efficient approach - pre-compute all matrices
    threshold = 1e-6
    function_vals = np.empty((k, D, N))  # k groups, D functions, N points each
    mus = [[np.zeros(N) for _ in range(D)] for _ in range(k)]
    
    # Pre-compute kernel matrix inverses (much more efficient)
    Sigma_invs = [np.linalg.inv(kernel_matrices[(i, i)] + np.eye(N) * threshold) for i in range(k)]
    
    # Generate GP functions using the original efficient conditioning approach
    for i in range(k):
        group_start_time = time.time()
        
        # Build conditional precision matrix efficiently
        Sigma_inv = (i + 1) * Sigma_invs[i]
        for j in range(i):
            K_jiK_ii_inv = kernel_matrices[(j, i)] @ Sigma_invs[i]
            Sigma_ji = kernel_matrices[(j, j)] - K_jiK_ii_inv @ kernel_matrices[(j, i)].T
            Sigma_ji_inv = np.linalg.inv(Sigma_ji + np.eye(N) * threshold)
            Sigma_inv += (K_jiK_ii_inv.T @ Sigma_ji_inv @ K_jiK_ii_inv)
            for dim_id in range(D):
                mus[i][dim_id] += (K_jiK_ii_inv.T @ Sigma_ji_inv @ function_vals[j][dim_id])
        
        # Convert precision to covariance
        cov_start_time = time.time()
        Sigmas_i = np.linalg.inv(Sigma_inv + np.eye(N) * threshold)
        cov_time = time.time() - cov_start_time
        
        # Generate D independent function values for this group
        sample_start_time = time.time()
        for dim_id in range(D):
            try:
                function_vals[i][dim_id] = np.random.multivariate_normal(Sigmas_i @ mus[i][dim_id], Sigmas_i)
            except np.linalg.LinAlgError as e:
                if verbose:
                    print(f"        Warning: Sampling failed for group {i}, dim {dim_id}: {e}")
                # Fallback to independent sampling
                function_vals[i][dim_id] = np.random.multivariate_normal(
                    np.zeros(N), kernel_matrices[(i, i)] + np.eye(N) * 1e-3
                )
        
        sample_time = time.time() - sample_start_time
        group_total_time = time.time() - group_start_time
        
        if verbose:
            print(f"      Group {i}: cov={cov_time:.2f}s, sample={sample_time:.2f}s, total={group_total_time:.2f}s")
    
    total_time = time.time() - start_time
    if verbose:
        print(f"  ✓ Manifold generation completed in {total_time:.2f}s")
    
    # Reshape to final format: (k*N, D) for embedded, (k*N, d) for intrinsic
    embedded_coords = function_vals.transpose(0, 2, 1).reshape((k * N, D))  # (k*N, D)
    intrinsic_coords = np.concatenate(group_intrinsic_coords, axis=0)        # (k*N, d)
    
    if verbose:
        print(f"  Final shapes: intrinsic {intrinsic_coords.shape}, embedded {embedded_coords.shape}")
    
    return embedded_coords, intrinsic_coords


def generate_random_rotation(D: int, random_seed: Optional[int] = None) -> np.ndarray:
    """
    Generate a random orthogonal matrix for rotation.
    
    Args:
        D: Dimension of the rotation matrix
        random_seed: Random seed for reproducibility
        
    Returns:
        (D, D) orthogonal rotation matrix
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    
    # Generate random matrix and use QR decomposition to get orthogonal matrix
    A = np.random.randn(D, D)
    Q, R = np.linalg.qr(A)
    
    # Ensure proper rotation (det = 1) rather than reflection (det = -1)
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1
    
    return Q


def apply_normalization(coords: np.ndarray, centralize: bool = True, 
                       rescale: bool = True, apply_rotation: bool = True,
                       random_seed: Optional[int] = None, verbose: bool = True) -> Tuple[np.ndarray, dict]:
    """
    Apply centralization, rescaling, and random rotation to coordinates.
    
    Args:
        coords: (N, D) coordinates to normalize
        centralize: Whether to center data at origin
        rescale: Whether to rescale by radius
        apply_rotation: Whether to apply random rotation
        random_seed: Random seed for rotation
        verbose: Whether to print details
        
    Returns:
        Tuple of (normalized_coords, normalization_info)
    """
    normalization_info = {
        'data_mean': np.zeros(coords.shape[1]),
        'normalization_factor': 1.0,
        'rotation_matrix': np.eye(coords.shape[1]),
        'centralize_applied': centralize,
        'rescale_applied': rescale,
        'rotation_applied': apply_rotation,
        'original_diameter': None,
        'final_diameter': None
    }
    
    result_coords = coords.copy()
    
    # Step 1: Centralization
    if centralize:
        data_mean = np.mean(coords, axis=0)
        result_coords = coords - data_mean[np.newaxis, :]
        normalization_info['data_mean'] = data_mean
        if verbose:
            print(f"  ✓ Centralized (mean norm: {np.linalg.norm(data_mean):.6f})")
    
    # Compute diameter before rescaling
    norms = np.linalg.norm(result_coords, axis=1)
    diameter = 2.0 * np.max(norms)
    normalization_info['original_diameter'] = diameter
    
    # Step 2: Rescaling
    if rescale:
        normalization_factor = diameter / 2.0  # Normalize by radius
        result_coords = result_coords / normalization_factor
        normalization_info['normalization_factor'] = normalization_factor
        
        # Compute final diameter
        final_norms = np.linalg.norm(result_coords, axis=1)
        final_diameter = 2.0 * np.max(final_norms)
        normalization_info['final_diameter'] = final_diameter
        
        if verbose:
            print(f"  ✓ Rescaled (diameter: {diameter:.4f} → {final_diameter:.4f}, factor: {normalization_factor:.4f})")
    else:
        normalization_info['final_diameter'] = diameter
    
    # Step 3: Random rotation
    if apply_rotation:
        rotation_matrix = generate_random_rotation(
            coords.shape[1], 
            random_seed
        )
        result_coords = result_coords @ rotation_matrix.T
        normalization_info['rotation_matrix'] = rotation_matrix
        if verbose:
            print(f"  ✓ Applied random rotation (determinant: {np.linalg.det(rotation_matrix):.6f})")
    
    return result_coords, normalization_info


def add_noise(coords: np.ndarray, noise_sigma: float, 
              random_seed: Optional[int] = None) -> np.ndarray:
    """
    Add Gaussian noise to coordinates.
    
    Args:
        coords: (N, D) clean coordinates
        noise_sigma: Standard deviation of noise
        random_seed: Random seed for reproducibility
        
    Returns:
        (N, D) noisy coordinates
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    
    noise = np.random.normal(0, noise_sigma, coords.shape)
    return coords + noise
