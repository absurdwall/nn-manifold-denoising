"""
Intrinsic dimension estimation using multiple methods.

This module provides robust dimension estimation with improved implementations,
particularly addressing issues with the k-NN method that was giving consistently low estimates.
"""

import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from typing import Dict, Optional
import warnings

try:
    from skdim.id import TwoNN, MLE, MOM
    SKDIM_AVAILABLE = True
except ImportError:
    SKDIM_AVAILABLE = False
    warnings.warn("scikit-dimension not available. Some methods will be disabled.")


class DimensionEstimator:
    """
    Comprehensive intrinsic dimension estimator with multiple methods.
    
    Includes improved k-NN estimation and robust error handling.
    """
    
    def __init__(self, verbose: bool = False):
        """
        Initialize the dimension estimator.
        
        Args:
            verbose: Whether to print detailed information during estimation
        """
        self.verbose = verbose
        self.results = {}
        
        # Define available methods
        self.methods = {
            'PCA': self._estimate_pca,
            'k-NN': self._estimate_knn_improved,
            'k-NN-Levina': self._estimate_knn_levina_bickel,
        }
        
        # Add scikit-dimension methods if available
        if SKDIM_AVAILABLE:
            self.methods.update({
                'TwoNN': self._estimate_twonn,
                'MLE': self._estimate_mle,
                'MOM': self._estimate_mom,
            })
    
    def fit(self, data: np.ndarray, methods: Optional[list] = None) -> Dict[str, float]:
        """
        Estimate intrinsic dimension using multiple methods.
        
        Args:
            data: (N, D) array of embedded points
            methods: List of method names to use. If None, use all available methods.
            
        Returns:
            Dictionary mapping method names to dimension estimates
        """
        if self.verbose:
            print(f"Estimating intrinsic dimension for data shape {data.shape}")
        
        # Use all methods if none specified
        if methods is None:
            methods = list(self.methods.keys())
        
        self.results = {}
        
        for method_name in methods:
            if method_name not in self.methods:
                if self.verbose:
                    print(f"Warning: Method {method_name} not available")
                continue
                
            try:
                if self.verbose:
                    print(f"  Computing {method_name}...")
                
                estimate = self.methods[method_name](data)
                self.results[method_name] = float(estimate)
                
                if self.verbose:
                    print(f"    {method_name}: {estimate:.3f}")
                    
            except Exception as e:
                if self.verbose:
                    print(f"    {method_name}: Failed - {e}")
                self.results[method_name] = np.nan
        
        return self.results
    
    def _estimate_pca(self, data: np.ndarray, variance_threshold: float = 0.95) -> float:
        """
        PCA-based dimension estimation.
        
        Args:
            data: Input data
            variance_threshold: Cumulative variance threshold for dimension estimation
            
        Returns:
            Estimated dimension
        """
        pca = PCA()
        pca.fit(data)
        
        explained_variance_ratio = np.cumsum(pca.explained_variance_ratio_)
        intrinsic_dimension = np.argmax(explained_variance_ratio >= variance_threshold) + 1
        
        # Alternative: use eigenvalue gap
        eigenvals = pca.explained_variance_
        if len(eigenvals) > 1:
            gaps = eigenvals[:-1] / eigenvals[1:]  # Ratio of consecutive eigenvalues
            max_gap_idx = np.argmax(gaps)
            gap_dimension = max_gap_idx + 1
            
            # Use the smaller of the two estimates for conservativeness
            return min(intrinsic_dimension, gap_dimension)
        
        return intrinsic_dimension
    
    def _estimate_knn_improved(self, data: np.ndarray, k: int = 10, 
                              distance_ratios: bool = True) -> float:
        """
        Improved k-NN dimension estimation addressing the low estimate issue.
        
        This implementation uses multiple k values and improved distance ratio computation.
        
        Args:
            data: Input data
            k: Number of neighbors (will try multiple values around this)
            distance_ratios: Whether to use distance ratios (more robust)
            
        Returns:
            Estimated dimension
        """
        N, D = data.shape
        
        # Try multiple k values to get robust estimate
        k_values = [max(5, k//2), k, min(2*k, N//4)]
        estimates = []
        
        for k_val in k_values:
            k_val = min(k_val, N - 1)  # Ensure we don't exceed data size
            
            try:
                nbrs = NearestNeighbors(n_neighbors=k_val + 1).fit(data)
                distances, _ = nbrs.kneighbors(data)
                
                # Remove self-distance (first column is always 0)
                distances = distances[:, 1:]
                
                if distance_ratios:
                    # Use distance ratios (more stable)
                    # Estimate using the fact that for d-dimensional data,
                    # the ratio of k-th to 1st nearest neighbor distance follows a specific distribution
                    ratios = distances[:, -1] / distances[:, 0]  # k-th / 1st distance
                    
                    # Remove infinite and zero ratios
                    valid_ratios = ratios[np.isfinite(ratios) & (ratios > 0)]
                    
                    if len(valid_ratios) > 0:
                        # Use median ratio for robustness
                        median_ratio = np.median(valid_ratios)
                        
                        # Dimension estimate based on expected ratio growth
                        # For d-dimensional data: E[r_k/r_1] ≈ k^(1/d)
                        if median_ratio > 1:
                            dim_est = np.log(k_val) / np.log(median_ratio)
                            estimates.append(dim_est)
                else:
                    # Original Levina-Bickel approach with corrections
                    dim_est = self._levina_bickel_single_k(distances, k_val)
                    if np.isfinite(dim_est) and dim_est > 0:
                        estimates.append(dim_est)
                        
            except Exception:
                continue
        
        if estimates:
            # Use median of estimates for robustness
            return np.median(estimates)
        else:
            # Fallback to PCA if k-NN fails
            return self._estimate_pca(data)
    
    def _estimate_knn_levina_bickel(self, data: np.ndarray, k: int = 10) -> float:
        """
        Original Levina-Bickel k-NN estimator with improvements.
        
        Args:
            data: Input data
            k: Number of neighbors
            
        Returns:
            Estimated dimension
        """
        N = data.shape[0]
        k = min(k, N - 1)
        
        nbrs = NearestNeighbors(n_neighbors=k + 1).fit(data)
        distances, _ = nbrs.kneighbors(data)
        distances = distances[:, 1:]  # Remove self-distance
        
        return self._levina_bickel_single_k(distances, k)
    
    def _levina_bickel_single_k(self, distances: np.ndarray, k: int) -> float:
        """
        Single-k Levina-Bickel estimator.
        
        Args:
            distances: Distance matrix (N, k)
            k: Number of neighbors
            
        Returns:
            Dimension estimate
        """
        # Avoid division by zero
        distances = np.maximum(distances, 1e-12)
        
        # Compute log ratios
        log_ratios = np.log(distances[:, -1:] / distances[:, :-1])
        
        # Remove invalid values
        valid_ratios = log_ratios[np.isfinite(log_ratios)]
        
        if len(valid_ratios) == 0:
            return np.nan
        
        # Levina-Bickel estimator
        mean_log_ratio = np.mean(valid_ratios)
        
        if mean_log_ratio <= 0:
            return np.nan
            
        return (k - 1) / mean_log_ratio
    
    def _estimate_twonn(self, data: np.ndarray) -> float:
        """TwoNN estimator from scikit-dimension."""
        if not SKDIM_AVAILABLE:
            raise ImportError("scikit-dimension not available")
        
        model = TwoNN()
        return model.fit_transform(data)
    
    def _estimate_mle(self, data: np.ndarray) -> float:
        """MLE estimator from scikit-dimension."""
        if not SKDIM_AVAILABLE:
            raise ImportError("scikit-dimension not available")
        
        model = MLE()
        return model.fit_transform(data)
    
    def _estimate_mom(self, data: np.ndarray) -> float:
        """Method of Moments estimator from scikit-dimension."""
        if not SKDIM_AVAILABLE:
            raise ImportError("scikit-dimension not available")
        
        model = MOM()
        return model.fit_transform(data)
    
    def get_consensus_estimate(self, weights: Optional[Dict[str, float]] = None) -> float:
        """
        Get a consensus dimension estimate from multiple methods.
        
        Args:
            weights: Optional weights for different methods
            
        Returns:
            Weighted consensus estimate
        """
        if not self.results:
            raise ValueError("No results available. Run fit() first.")
        
        # Filter out failed estimates
        valid_results = {k: v for k, v in self.results.items() 
                        if np.isfinite(v) and v > 0}
        
        if not valid_results:
            return np.nan
        
        if weights is None:
            # Default weights: prefer TwoNN and MLE if available, then PCA
            weights = {
                'TwoNN': 0.3,
                'MLE': 0.3,
                'PCA': 0.2,
                'k-NN': 0.1,
                'k-NN-Levina': 0.05,
                'MOM': 0.05
            }
        
        # Normalize weights for available methods
        available_weights = {k: weights.get(k, 0.1) for k in valid_results.keys()}
        weight_sum = sum(available_weights.values())
        available_weights = {k: w/weight_sum for k, w in available_weights.items()}
        
        # Compute weighted average
        consensus = sum(available_weights[k] * valid_results[k] 
                       for k in valid_results.keys())
        
        return consensus
    
    def print_summary(self) -> None:
        """Print a summary of dimension estimation results."""
        if not self.results:
            print("No results available. Run fit() first.")
            return
        
        print("\nDimension Estimation Summary:")
        print("-" * 40)
        
        for method, estimate in self.results.items():
            if np.isfinite(estimate):
                print(f"{method:>12}: {estimate:6.3f}")
            else:
                print(f"{method:>12}: Failed")
        
        # Show consensus if we have valid results
        valid_results = {k: v for k, v in self.results.items() 
                        if np.isfinite(v) and v > 0}
        
        if valid_results:
            consensus = self.get_consensus_estimate()
            print("-" * 40)
            print(f"{'Consensus':>12}: {consensus:6.3f}")


# Utility functions for backward compatibility
def estimate_dimension_pca(data: np.ndarray, variance_threshold: float = 0.95) -> float:
    """Standalone PCA dimension estimation."""
    estimator = DimensionEstimator()
    return estimator._estimate_pca(data, variance_threshold)


def estimate_dimension_knn(data: np.ndarray, k: int = 10) -> float:
    """Standalone improved k-NN dimension estimation.""" 
    estimator = DimensionEstimator()
    return estimator._estimate_knn_improved(data, k)
