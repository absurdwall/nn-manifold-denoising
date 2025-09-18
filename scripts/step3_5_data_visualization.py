#!/usr/bin/env python3
"""
Step 3.5: Data Visualization Pipeline

This script creates comprehensive visualizations of the generated manifold datasets:
1. Point cloud visualizations with different colors for different k batches
2. Multiple 3D coordinate combinations for better understanding
3. Mesh generation using intrinsic coordinates and surface reconstruction
4. Open3D integration for advanced visualization

Usage:
    python scripts/step3_5_data_visualization.py [--data_dir DATA_DIR] [--output_dir OUTPUT_DIR]
"""

import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from typing import Dict, List, Any, Optional, Tuple
import warnings
import logging
import itertools

# Set plotting style
plt.style.use('seaborn-v0_8')
warnings.filterwarnings('ignore', category=UserWarning)
import matplotlib
matplotlib.set_loglevel("WARNING")

# Try to import Open3D for mesh generation
try:
    import open3d as o3d
    OPEN3D_AVAILABLE = True
except ImportError:
    OPEN3D_AVAILABLE = False
    print("Warning: Open3D not available. Mesh generation will be skipped.")


def setup_logging(verbose: bool = False) -> logging.Logger:
    """Setup logging configuration."""
    level = logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    
    # Suppress matplotlib debug logging
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)
    
    return logging.getLogger(__name__)


def load_dataset_files(dataset_dir: Path) -> Dict[str, Any]:
    """Load all files for a single dataset."""
    
    dataset_name = dataset_dir.name
    files = {}
    
    # Load metadata
    metadata_file = dataset_dir / f"{dataset_name}_metadata.json"
    if metadata_file.exists():
        with open(metadata_file, 'r') as f:
            files['metadata'] = json.load(f)
    
    # Load numpy arrays
    for data_type in ['raw', 'clean', 'noisy']:
        data_file = dataset_dir / f"{dataset_name}_{data_type}.npy"
        if data_file.exists():
            files[f'{data_type}_data'] = np.load(data_file)
    
    # Load intrinsic coordinates
    intrinsic_file = dataset_dir / f"{dataset_name}_intrinsic.npy"
    if intrinsic_file.exists():
        files['intrinsic'] = np.load(intrinsic_file)
    
    return files


def get_colorblind_friendly_colors(n: int) -> List[str]:
    """Get a colorblind-friendly color palette."""
    # Paul Tol's colorblind-friendly palette
    base_colors = [
        '#E69F00',  # Orange
        '#56B4E9',  # Sky Blue
        '#009E73',  # Bluish Green
        '#F0E442',  # Yellow
        '#0072B2',  # Blue
        '#D55E00',  # Vermillion
        '#CC79A7',  # Reddish Purple
        '#332288',  # Indigo
        '#117733',  # Green
        '#999933',  # Olive
        '#882255',  # Wine
        '#AA4499',  # Rose
        '#44AA99',  # Teal
        '#88CCEE',  # Light Blue
        '#DDCC77',  # Sand
        '#DDDDDD',  # Light Grey
        '#77AADD',  # Light Blue 2
        '#99DDFF',  # Very Light Blue
        '#44BB99',  # Mint
        '#BBCC33',  # Pear
    ]
    
    if n <= len(base_colors):
        return base_colors[:n]
    else:
        # If we need more colors, cycle through and add alpha variations
        colors = base_colors[:]
        while len(colors) < n:
            colors.extend(base_colors)
        return colors[:n]


def create_point_cloud_plots(
    data: np.ndarray,
    intrinsic: np.ndarray,
    metadata: Dict[str, Any],
    output_dir: Path,
    dataset_name: str,
    data_type: str,
    logger: logging.Logger,
    n_combinations: int = 10,
    max_vis_k: int = 5,
    max_vis_n: int = 100
) -> None:
    """Create multiple 3D point cloud visualizations with different coordinate combinations.
    
    Generates both full data plots and bounded (limited k,N) plots for clarity.
    """
    
    properties = metadata['properties']
    k = properties['k']
    N = properties['N']
    D = properties['D']
    d = properties['d']
    
    logger.info(f"Creating point cloud plots for {dataset_name} ({data_type})")
    logger.info(f"  Data shape: {data.shape}, k={k}, N={N}, D={D}, d={d}")
    
    # Get colors for all batches (for full plots)
    colors_full = get_colorblind_friendly_colors(k)
    
    # For bounded plots
    vis_k = min(k, max_vis_k)
    vis_n = min(N, max_vis_n)
    colors_bounded = get_colorblind_friendly_colors(vis_k)
    
    logger.info(f"  Will generate both full plots (k={k}, N={N}) and bounded plots (k={vis_k}, N={vis_n})")
    
    # Create directory for this dataset and data type
    plot_dir = output_dir / "point_clouds" / dataset_name / data_type
    plot_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate dimension combinations for 3D plotting
    if D >= 3:
        # Generate combinations of 3 dimensions
        all_dims = list(range(D))
        
        # Strategy: Take evenly spaced combinations + some random ones
        combinations = []
        
        # First few dimensions
        combinations.append((0, 1, 2))
        
        # Evenly spaced through the dimensions
        if n_combinations > 1:
            step = max(1, D // max(1, n_combinations - 2))
            for i in range(1, min(n_combinations - 1, D - 2)):
                dim1 = (i * step) % D
                dim2 = ((i * step) + max(1, step // 2)) % D
                dim3 = ((i * step) + step) % D
                if len(set([dim1, dim2, dim3])) == 3:  # Ensure all different
                    combinations.append((dim1, dim2, dim3))
        
        # Add last dimensions
        if D > 3 and n_combinations > 1:
            combinations.append((D-3, D-2, D-1))
        
        # Remove duplicates and limit to n_combinations
        combinations = list(set(combinations))[:n_combinations]
    else:
        # If D < 3, pad with zeros or use available dimensions
        if D == 1:
            combinations = [(0, 0, 0)]
        elif D == 2:
            combinations = [(0, 1, 0)]
        else:
            combinations = [(0, 1, 2)]
    
    logger.info(f"  Creating {len(combinations)} coordinate combinations × 2 versions (full + bounded)")
    
    # Create plots for each combination - BOTH full and bounded versions
    for idx, (dim1, dim2, dim3) in enumerate(combinations):
        
        # VERSION 1: FULL DATA PLOT
        logger.info(f"    Combination {idx+1}/{len(combinations)}: dims ({dim1},{dim2},{dim3}) - creating full version")
        
        # Create 3D point cloud plot - FULL VERSION
        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot each batch with different color - ALL BATCHES
        for batch_idx in range(k):
            start_idx = batch_idx * N
            end_idx = (batch_idx + 1) * N
            
            batch_data = data[start_idx:end_idx]
            
            ax.scatter(
                batch_data[:, dim1],
                batch_data[:, dim2], 
                batch_data[:, dim3],
                c=colors_full[batch_idx],
                label=f'Batch {batch_idx + 1}' if batch_idx < 10 else '',  # Limit legend entries
                alpha=0.6,
                s=8  # Smaller points for full data
            )
        
        ax.set_xlabel(f'Dimension {dim1}')
        ax.set_ylabel(f'Dimension {dim2}')
        ax.set_zlabel(f'Dimension {dim3}')
        ax.set_title(f'{dataset_name} - {data_type.title()} Data (Full)\n'
                    f'3D Point Cloud: Dimensions ({dim1}, {dim2}, {dim3})\n'
                    f'All {k} batches, {N} samples/batch, σ={properties.get("kernel_smoothness", "N/A")}',
                    fontsize=12, pad=20)
        
        # Position legend outside the plot (only show first 10 batches in legend)
        if k <= 10:
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        
        # Save FULL 3D plot
        plot_path_full_3d = plot_dir / f'3d_pointcloud_dims_{dim1}_{dim2}_{dim3}_FULL.png'
        plt.savefig(plot_path_full_3d, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create 2D projection plot - FULL VERSION
        fig, ax = plt.subplots(figsize=(10, 8))
        
        for batch_idx in range(k):
            start_idx = batch_idx * N
            end_idx = (batch_idx + 1) * N
            
            batch_data = data[start_idx:end_idx]
            
            ax.scatter(
                batch_data[:, dim1],
                batch_data[:, dim2],
                c=colors_full[batch_idx],
                label=f'Batch {batch_idx + 1}' if batch_idx < 10 else '',
                alpha=0.6,
                s=8
            )
        
        ax.set_xlabel(f'Dimension {dim1}')
        ax.set_ylabel(f'Dimension {dim2}')
        ax.set_title(f'{dataset_name} - {data_type.title()} Data (Full)\n'
                    f'2D Projection: Dimensions ({dim1}, {dim2})\n'
                    f'All {k} batches, {N} samples/batch, σ={properties.get("kernel_smoothness", "N/A")}',
                    fontsize=12, pad=15)
        if k <= 10:
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save FULL 2D plot
        plot_path_full_2d = plot_dir / f'2d_projection_dims_{dim1}_{dim2}_FULL.png'
        plt.savefig(plot_path_full_2d, dpi=300, bbox_inches='tight')
        plt.close()
        
        # VERSION 2: BOUNDED DATA PLOT
        logger.info(f"    Combination {idx+1}/{len(combinations)}: dims ({dim1},{dim2},{dim3}) - creating bounded version")
        
        # Sample data for bounded visualization
        bounded_data = []
        for batch_idx in range(vis_k):
            start_idx = batch_idx * N
            end_idx = (batch_idx + 1) * N
            batch_data = data[start_idx:end_idx]
            
            # Sample vis_n points from this batch
            if vis_n < N:
                sample_indices = np.random.choice(N, vis_n, replace=False)
                batch_sample = batch_data[sample_indices]
            else:
                batch_sample = batch_data[:vis_n]
            
            bounded_data.append(batch_sample)
        
        bounded_data = np.vstack(bounded_data)
        
        # Create 3D point cloud plot - BOUNDED VERSION
        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot each batch with different color - LIMITED BATCHES
        for batch_idx in range(vis_k):
            start_idx = batch_idx * vis_n
            end_idx = (batch_idx + 1) * vis_n
            
            batch_data = bounded_data[start_idx:end_idx]
            
            ax.scatter(
                batch_data[:, dim1],
                batch_data[:, dim2], 
                batch_data[:, dim3],
                c=colors_bounded[batch_idx],
                label=f'Batch {batch_idx + 1}',
                alpha=0.7,
                s=15  # Larger points for bounded data
            )
        
        ax.set_xlabel(f'Dimension {dim1}')
        ax.set_ylabel(f'Dimension {dim2}')
        ax.set_zlabel(f'Dimension {dim3}')
        ax.set_title(f'{dataset_name} - {data_type.title()} Data (Bounded)\n'
                    f'3D Point Cloud: Dimensions ({dim1}, {dim2}, {dim3})\n'
                    f'Showing {vis_k}/{k} batches, {vis_n}/{N} samples/batch, σ={properties.get("kernel_smoothness", "N/A")}',
                    fontsize=12, pad=20)
        
        # Position legend outside the plot
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        
        # Save BOUNDED 3D plot
        plot_path_bounded_3d = plot_dir / f'3d_pointcloud_dims_{dim1}_{dim2}_{dim3}_BOUNDED.png'
        plt.savefig(plot_path_bounded_3d, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create 2D projection plot - BOUNDED VERSION
        fig, ax = plt.subplots(figsize=(10, 8))
        
        for batch_idx in range(vis_k):
            start_idx = batch_idx * vis_n
            end_idx = (batch_idx + 1) * vis_n
            
            batch_data = bounded_data[start_idx:end_idx]
            
            ax.scatter(
                batch_data[:, dim1],
                batch_data[:, dim2],
                c=colors_bounded[batch_idx],
                label=f'Batch {batch_idx + 1}',
                alpha=0.7,
                s=15
            )
        
        ax.set_xlabel(f'Dimension {dim1}')
        ax.set_ylabel(f'Dimension {dim2}')
        ax.set_title(f'{dataset_name} - {data_type.title()} Data (Bounded)\n'
                    f'2D Projection: Dimensions ({dim1}, {dim2})\n'
                    f'Showing {vis_k}/{k} batches, {vis_n}/{N} samples/batch, σ={properties.get("kernel_smoothness", "N/A")}',
                    fontsize=12, pad=15)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save BOUNDED 2D plot
        plot_path_bounded_2d = plot_dir / f'2d_projection_dims_{dim1}_{dim2}_BOUNDED.png'
        plt.savefig(plot_path_bounded_2d, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"    ✓ Saved combination {idx+1}/{len(combinations)}: FULL + BOUNDED versions")


def create_mesh_visualization(
    data: np.ndarray,
    intrinsic: np.ndarray,
    metadata: Dict[str, Any],
    output_dir: Path,
    dataset_name: str,
    data_type: str,
    logger: logging.Logger,
    n_combinations: int = 10,
    max_mesh_k: int = 3,
    max_mesh_n: int = 200,
    save_ply: bool = False
) -> None:
    """Create mesh visualization using Open3D with proper surface reconstruction.
    
    Generates meshes for multiple coordinate combinations, similar to point cloud plots.
    """
    
    if not OPEN3D_AVAILABLE:
        logger.warning("Open3D not available, skipping mesh generation")
        return
    
    properties = metadata['properties']
    k = properties['k']
    N = properties['N']
    D = properties['D']
    
    # Apply limits for mesh generation (meshes need fewer points for good results)
    mesh_k = min(k, max_mesh_k)
    mesh_n = min(N, max_mesh_n)
    
    logger.info(f"Creating mesh visualization for {dataset_name} ({data_type})")
    logger.info(f"  Using {mesh_k}/{k} batches, {mesh_n}/{N} samples/batch for mesh generation")
    logger.info(f"  Will generate meshes for {n_combinations} coordinate combinations")
    
    # Create directory for meshes
    mesh_dir = output_dir / "meshes" / dataset_name / data_type
    mesh_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate the same dimension combinations as point clouds
    if D >= 3:
        # Generate combinations of 3 dimensions (same logic as point clouds)
        combinations = []
        
        # First few dimensions
        combinations.append((0, 1, 2))
        
        # Evenly spaced through the dimensions
        if n_combinations > 1:
            step = max(1, D // max(1, n_combinations - 2))
            for i in range(1, min(n_combinations - 1, D - 2)):
                dim1 = (i * step) % D
                dim2 = ((i * step) + max(1, step // 2)) % D
                dim3 = ((i * step) + step) % D
                if len(set([dim1, dim2, dim3])) == 3:  # Ensure all different
                    combinations.append((dim1, dim2, dim3))
        
        # Add last dimensions
        if D > 3 and n_combinations > 1:
            combinations.append((D-3, D-2, D-1))
        
        # Remove duplicates and limit to n_combinations
        combinations = list(set(combinations))[:n_combinations]
    else:
        # If D < 3, pad with zeros or use available dimensions
        if D == 1:
            combinations = [(0, 0, 0)]
        elif D == 2:
            combinations = [(0, 1, 0)]
        else:
            combinations = [(0, 1, 2)]
    
    # Suppress Open3D warnings
    import contextlib
    import sys
    from io import StringIO
    
    @contextlib.contextmanager
    def suppress_stdout_stderr():
        """Suppress Open3D warnings."""
        old_stdout, old_stderr = sys.stdout, sys.stderr
        try:
            sys.stdout = StringIO()
            sys.stderr = StringIO()
            yield
        finally:
            sys.stdout, sys.stderr = old_stdout, old_stderr
    
    # Sample data for mesh generation (same for all combinations)
    colors = get_colorblind_friendly_colors(mesh_k)
    rgb_colors = []
    for color_hex in colors:
        color_hex = color_hex.lstrip('#')
        rgb = tuple(int(color_hex[i:i+2], 16) / 255.0 for i in (0, 2, 4))
        rgb_colors.append(rgb)
    
    # Sample data for mesh
    mesh_data = []
    point_colors = []
    
    for batch_idx in range(mesh_k):
        start_idx = batch_idx * N
        end_idx = (batch_idx + 1) * N
        batch_data = data[start_idx:end_idx]
        
        # Sample points from this batch
        if mesh_n < N:
            sample_indices = np.random.choice(N, mesh_n, replace=False)
            batch_sample = batch_data[sample_indices]
        else:
            batch_sample = batch_data[:mesh_n]
        
        mesh_data.append(batch_sample)
        # Assign color to all points in this batch
        point_colors.extend([rgb_colors[batch_idx]] * len(batch_sample))
    
    mesh_data = np.vstack(mesh_data)
    logger.info(f"  Mesh data shape: {mesh_data.shape}")
    
    # Generate meshes for each coordinate combination
    for combo_idx, (dim1, dim2, dim3) in enumerate(combinations):
        logger.info(f"  Generating meshes for combination {combo_idx+1}/{len(combinations)}: dims ({dim1},{dim2},{dim3})")
        
        # Extract 3D coordinates for this combination
        points_3d = np.column_stack([
            mesh_data[:, dim1] if dim1 < mesh_data.shape[1] else np.zeros(mesh_data.shape[0]),
            mesh_data[:, dim2] if dim2 < mesh_data.shape[1] else np.zeros(mesh_data.shape[0]),
            mesh_data[:, dim3] if dim3 < mesh_data.shape[1] else np.zeros(mesh_data.shape[0])
        ])
        
        try:
            # Create point cloud for this coordinate combination
            with suppress_stdout_stderr():
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(points_3d)
                pcd.colors = o3d.utility.Vector3dVector(point_colors)
                
                # Optionally save point cloud PLY file
                if save_ply:
                    pcd_path = mesh_dir / f"pointcloud_{data_type}_dims_{dim1}_{dim2}_{dim3}.ply"
                    o3d.io.write_point_cloud(str(pcd_path), pcd)
                    file_size = pcd_path.stat().st_size / (1024 * 1024)
                    logger.info(f"    ✓ Saved point cloud PLY: {pcd_path.name} ({file_size:.1f} MB)")
            
            # Create connected surface meshes with different methods
            mesh_methods = [
                ("poisson", "Poisson Surface Reconstruction"),
                ("ball_pivoting", "Ball Pivoting Algorithm"), 
                ("convex_hull", "Convex Hull"),
                ("delaunay", "Delaunay Triangulation")
            ]
            
            for method_name, method_title in mesh_methods:
                try:
                    with suppress_stdout_stderr():
                        if method_name == "poisson":
                            # Poisson reconstruction - creates smooth connected surface
                            pcd.estimate_normals()
                            pcd.orient_normals_consistent_tangent_plane(100)
                            
                            mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                                pcd, depth=6, width=0, scale=1.1, linear_fit=False
                            )
                            
                        elif method_name == "ball_pivoting":
                            # Ball pivoting - good for creating connected surfaces
                            pcd.estimate_normals()
                            
                            distances = pcd.compute_nearest_neighbor_distance()
                            avg_dist = np.mean(distances)
                            radius = 1.5 * avg_dist
                            
                            mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
                                pcd, o3d.utility.DoubleVector([radius, radius * 2, radius * 4])
                            )
                            
                        elif method_name == "convex_hull":
                            # Convex hull - creates connected outer surface
                            mesh, _ = pcd.compute_convex_hull()
                            
                        elif method_name == "delaunay":
                            # Delaunay triangulation using alpha shapes with larger alpha
                            mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(
                                pcd, alpha=0.5
                            )
                        
                        # Clean up mesh and ensure connectivity
                        mesh.remove_degenerate_triangles()
                        mesh.remove_duplicated_triangles()
                        mesh.remove_duplicated_vertices()
                        mesh.remove_non_manifold_edges()
                        
                        # Add vertex colors based on proximity to original colored points
                        if len(mesh.vertices) > 0 and len(mesh.triangles) > 0:
                            # Smooth the mesh for better appearance
                            mesh = mesh.filter_smooth_simple(number_of_iterations=1)
                            mesh.compute_vertex_normals()
                            
                            # Get mesh stats
                            n_vertices = len(mesh.vertices)
                            n_triangles = len(mesh.triangles)
                            
                            logger.info(f"    ✓ {method_title}: {n_vertices} vertices, {n_triangles} triangles")
                            
                            # Create matplotlib visualization
                            vertices = np.asarray(mesh.vertices)
                            triangles = np.asarray(mesh.triangles)
                            
                            fig = plt.figure(figsize=(15, 10))
                            
                            # 3D mesh plot
                            ax1 = fig.add_subplot(221, projection='3d')
                            ax1.plot_trisurf(vertices[:, 0], vertices[:, 1], vertices[:, 2], 
                                            triangles=triangles, alpha=0.7, cmap='viridis')
                            ax1.set_xlabel(f'Dim {dim1}')
                            ax1.set_ylabel(f'Dim {dim2}') 
                            ax1.set_zlabel(f'Dim {dim3}')
                            ax1.set_title(f'{method_title}\n{n_vertices} vertices, {n_triangles} triangles')
                            
                            # Wireframe view
                            ax2 = fig.add_subplot(222, projection='3d')
                            ax2.plot_trisurf(vertices[:, 0], vertices[:, 1], vertices[:, 2],
                                            triangles=triangles, alpha=0.3, color='lightblue')
                            for triangle in triangles[:min(500, len(triangles))]:  # Show subset of edges
                                edge_points = vertices[triangle]
                                for i in range(3):
                                    start, end = edge_points[i], edge_points[(i+1)%3]
                                    ax2.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]], 
                                            'b-', alpha=0.4, linewidth=0.5)
                            ax2.set_title('Wireframe View')
                            
                            # 2D projections
                            ax3 = fig.add_subplot(223)
                            ax3.triplot(vertices[:, 0], vertices[:, 1], triangles, alpha=0.5)
                            ax3.set_xlabel(f'Dim {dim1}')
                            ax3.set_ylabel(f'Dim {dim2}')
                            ax3.set_title('2D Projection (dims 1-2)')
                            ax3.set_aspect('equal')
                            
                            ax4 = fig.add_subplot(224)
                            ax4.triplot(vertices[:, 0], vertices[:, 2], triangles, alpha=0.5)
                            ax4.set_xlabel(f'Dim {dim1}')
                            ax4.set_ylabel(f'Dim {dim3}')
                            ax4.set_title('2D Projection (dims 1-3)')
                            ax4.set_aspect('equal')
                            
                            plt.suptitle(f'{dataset_name} - {data_type.title()} - {method_title}\nDimensions ({dim1}, {dim2}, {dim3})', fontsize=14)
                            plt.tight_layout()
                            
                            # Save visualization with coordinate combination info
                            vis_path = mesh_dir / f"mesh_{method_name}_{data_type}_dims_{dim1}_{dim2}_{dim3}_visualization.png"
                            plt.savefig(vis_path, dpi=300, bbox_inches='tight')
                            plt.close()
                            
                            logger.info(f"      ✓ Saved visualization: {vis_path.name}")
                            
                            # Optionally save mesh PLY file
                            if save_ply:
                                mesh_ply_path = mesh_dir / f"mesh_{method_name}_{data_type}_dims_{dim1}_{dim2}_{dim3}.ply"
                                o3d.io.write_triangle_mesh(str(mesh_ply_path), mesh)
                                ply_size = mesh_ply_path.stat().st_size / (1024 * 1024)
                                logger.info(f"      ✓ Saved mesh PLY: {mesh_ply_path.name} ({ply_size:.1f} MB)")
                            
                        else:
                            logger.warning(f"    ✗ {method_title} failed: empty mesh")
                            
                except Exception as e:
                    logger.warning(f"    ✗ {method_title} failed: {str(e)}")
                    continue
        
        except Exception as e:
            logger.error(f"Mesh generation failed for combination dims ({dim1},{dim2},{dim3}): {str(e)}")
            continue


def save_visualization_metadata(
    args: argparse.Namespace,
    processed_datasets: List[Dict[str, Any]],
    output_dir: Path,
    logger: logging.Logger
) -> None:
    """Save comprehensive metadata about the visualization run."""
    
    from datetime import datetime
    import platform
    import sys
    
    logger.info("Saving comprehensive visualization metadata...")
    
    # Create metadata directory
    metadata_dir = output_dir / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    
    # Comprehensive metadata
    metadata = {
        "visualization_run_info": {
            "timestamp": datetime.now().isoformat(),
            "script_version": "step3_5_data_visualization.py",
            "task_name": "Step 3.5: Data Visualization Pipeline",
            "total_runtime_datasets": len(processed_datasets),
            "requested_max_datasets": args.max_datasets if args.max_datasets > 0 else "ALL",
            "data_directory": str(args.data_dir),
            "output_directory": str(args.output_dir)
        },
        
        "visualization_parameters": {
            "coordinate_combinations": args.n_combinations,
            "point_cloud_settings": {
                "max_vis_k": args.max_vis_k,
                "max_vis_n": args.max_vis_n,
                "generates_full_version": True,
                "generates_bounded_version": True,
                "full_version_note": "Shows all k batches with N samples each",
                "bounded_version_note": f"Shows max {args.max_vis_k} batches with max {args.max_vis_n} samples each"
            },
            "mesh_settings": {
                "max_mesh_k": args.max_mesh_k,
                "max_mesh_n": args.max_mesh_n,
                "mesh_methods": ["poisson", "ball_pivoting", "convex_hull", "delaunay"],
                "generates_ply_files": True,
                "generates_png_visualizations": True,
                "mesh_data_type": "clean"
            }
        },
        
        "system_info": {
            "python_version": sys.version,
            "platform": platform.platform(),
            "open3d_available": OPEN3D_AVAILABLE,
            "working_directory": str(Path.cwd())
        },
        
        "output_file_structure": {
            "point_clouds_per_dataset": args.n_combinations * 2 * 2 * 3,  # combinations × versions × plot_types × data_types
            "explanation": f"Each dataset generates {args.n_combinations} coordinate combinations × 2 versions (FULL/BOUNDED) × 2 plot types (3D/2D) × 3 data types (raw/clean/noisy)",
            "mesh_files_per_dataset": 8,  # 4 methods × 2 files each (PLY + PNG)
            "mesh_explanation": "Each dataset generates 4 mesh methods × 2 files each (PLY + PNG visualization) for clean data only"
        },
        
        "processed_datasets": processed_datasets,
        
        "file_naming_convention": {
            "point_clouds": {
                "3d_full": "3d_pointcloud_dims_{dim1}_{dim2}_{dim3}_FULL.png",
                "3d_bounded": "3d_pointcloud_dims_{dim1}_{dim2}_{dim3}_BOUNDED.png", 
                "2d_full": "2d_projection_dims_{dim1}_{dim2}_FULL.png",
                "2d_bounded": "2d_projection_dims_{dim1}_{dim2}_BOUNDED.png"
            },
            "meshes": {
                "ply_files": "mesh_{method}_{data_type}.ply",
                "png_visualizations": "mesh_{method}_{data_type}_visualization.png",
                "point_cloud": "pointcloud_{data_type}.ply"
            }
        },
        
        "dataset_parameter_summary": {
            "total_datasets_in_source": len(processed_datasets),
            "parameter_ranges": {},
            "base_types": [],
            "data_shapes": []
        }
    }
    
    # Calculate parameter ranges from processed datasets
    if processed_datasets:
        all_k = [d['metadata']['properties']['k'] for d in processed_datasets]
        all_N = [d['metadata']['properties']['N'] for d in processed_datasets]
        all_D = [d['metadata']['properties']['D'] for d in processed_datasets]
        all_d = [d['metadata']['properties']['d'] for d in processed_datasets]
        all_kernel_smoothness = [d['metadata']['properties']['kernel_smoothness'] for d in processed_datasets]
        all_base_types = [d['metadata']['properties']['base_type'] for d in processed_datasets]
        all_data_shapes = [d['data_shape'] for d in processed_datasets if 'data_shape' in d]
        
        metadata["dataset_parameter_summary"]["parameter_ranges"] = {
            "k": [min(all_k), max(all_k)],
            "N": [min(all_N), max(all_N)],
            "D": [min(all_D), max(all_D)],
            "d": [min(all_d), max(all_d)],
            "kernel_smoothness": [min(all_kernel_smoothness), max(all_kernel_smoothness)]
        }
        metadata["dataset_parameter_summary"]["base_types"] = list(set(all_base_types))
        metadata["dataset_parameter_summary"]["data_shapes"] = list(set(all_data_shapes))
    
    # Save metadata
    metadata_path = metadata_dir / "visualization_run_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"  ✓ Saved comprehensive metadata: {metadata_path}")
    
    # Save a human-readable summary
    summary_path = metadata_dir / "visualization_run_summary.txt"
    with open(summary_path, 'w') as f:
        f.write("STEP 3.5 DATA VISUALIZATION RUN SUMMARY\n")
        f.write("=" * 50 + "\n\n")
        
        f.write(f"Timestamp: {metadata['visualization_run_info']['timestamp']}\n")
        f.write(f"Output Directory: {metadata['visualization_run_info']['output_directory']}\n")
        f.write(f"Data Directory: {metadata['visualization_run_info']['data_directory']}\n")
        f.write(f"Total Datasets Processed: {metadata['visualization_run_info']['total_runtime_datasets']}\n\n")
        
        f.write("VISUALIZATION PARAMETERS:\n")
        f.write(f"  Coordinate Combinations: {metadata['visualization_parameters']['coordinate_combinations']}\n")
        f.write(f"  Point Cloud - Max Vis K: {metadata['visualization_parameters']['point_cloud_settings']['max_vis_k']}\n")
        f.write(f"  Point Cloud - Max Vis N: {metadata['visualization_parameters']['point_cloud_settings']['max_vis_n']}\n")
        f.write(f"  Mesh - Max K: {metadata['visualization_parameters']['mesh_settings']['max_mesh_k']}\n")
        f.write(f"  Mesh - Max N: {metadata['visualization_parameters']['mesh_settings']['max_mesh_n']}\n\n")
        
        f.write("OUTPUT FILES PER DATASET:\n")
        f.write(f"  Point Cloud Plots: {metadata['output_file_structure']['point_clouds_per_dataset']}\n")
        f.write(f"  Mesh Files: {metadata['output_file_structure']['mesh_files_per_dataset']}\n\n")
        
        if processed_datasets:
            f.write("DATASET PARAMETER RANGES:\n")
            for param, range_vals in metadata["dataset_parameter_summary"]["parameter_ranges"].items():
                f.write(f"  {param}: {range_vals[0]} to {range_vals[1]}\n")
            f.write(f"  Base Types: {', '.join(metadata['dataset_parameter_summary']['base_types'])}\n")
    
    logger.info(f"  ✓ Saved human-readable summary: {summary_path}")


def create_summary_visualization(
    datasets_info: List[Dict[str, Any]],
    output_dir: Path,
    logger: logging.Logger
) -> None:
    """Create summary visualization comparing different datasets."""
    
    logger.info("Creating summary comparison visualization...")
    
    # Create summary directory
    summary_dir = output_dir / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract information for comparison
    comparison_data = []
    for info in datasets_info:
        metadata = info['metadata']
        properties = metadata['properties']
        
        comparison_data.append({
            'dataset': info['dataset_name'],
            'k': properties['k'],
            'N': properties['N'],
            'D': properties['D'],
            'd': properties['d'],
            'kernel_smoothness': properties['kernel_smoothness'],
            'base_type': properties['base_type'],
            'data_shape': str(info.get('data_shape', 'N/A'))
        })
    
    # Create comparison plots
    if len(comparison_data) > 1:
        df = pd.DataFrame(comparison_data)
        
        # Parameter distribution plots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        # Plot distributions
        params = ['k', 'N', 'D', 'd', 'kernel_smoothness']
        for i, param in enumerate(params):
            if param in df.columns:
                if df[param].dtype in ['int64', 'float64']:
                    axes[i].hist(df[param], bins=20, alpha=0.7, edgecolor='black')
                    axes[i].set_xlabel(param)
                    axes[i].set_ylabel('Count')
                    axes[i].set_title(f'{param} Distribution')
                    axes[i].grid(True, alpha=0.3)
        
        # Base type distribution
        if 'base_type' in df.columns:
            base_type_counts = df['base_type'].value_counts()
            axes[5].pie(base_type_counts.values, labels=base_type_counts.index, autopct='%1.1f%%')
            axes[5].set_title('Base Type Distribution')
        
        plt.suptitle('Dataset Parameter Comparison', fontsize=16)
        plt.tight_layout()
        
        summary_path = summary_dir / 'parameter_comparison.png'
        plt.savefig(summary_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"  ✓ Saved parameter comparison: {summary_path}")
    
    # Save summary statistics
    summary_stats = {
        'total_datasets': len(datasets_info),
        'dataset_info': comparison_data,
        'parameter_ranges': {
            'k': [min(d['k'] for d in comparison_data), max(d['k'] for d in comparison_data)],
            'N': [min(d['N'] for d in comparison_data), max(d['N'] for d in comparison_data)],
            'D': [min(d['D'] for d in comparison_data), max(d['D'] for d in comparison_data)],
            'd': [min(d['d'] for d in comparison_data), max(d['d'] for d in comparison_data)],
            'kernel_smoothness': [min(d['kernel_smoothness'] for d in comparison_data), 
                                max(d['kernel_smoothness'] for d in comparison_data)],
        },
        'base_types': list(set(d['base_type'] for d in comparison_data))
    }
    
    stats_path = summary_dir / 'summary_statistics.json'
    with open(stats_path, 'w') as f:
        json.dump(summary_stats, f, indent=2)
    
    logger.info(f"  ✓ Saved summary statistics: {stats_path}")


def main():
    """Main visualization pipeline."""
    parser = argparse.ArgumentParser(description="Step 3.5: Data Visualization Pipeline")
    parser.add_argument("--data_dir", type=Path,
                       default=Path("data/data_250914_0100"),
                       help="Directory containing dataset folders")
    parser.add_argument("--output_dir", type=Path,
                       help="Output directory for visualizations (auto-generated if not specified)")
    parser.add_argument("--max_datasets", type=int, default=10,
                       help="Maximum number of datasets to process")
    parser.add_argument("--n_combinations", type=int, default=10,
                       help="Number of 3D coordinate combinations to generate")
    parser.add_argument("--max_vis_k", type=int, default=5,
                       help="Maximum number of batches to visualize (for clarity)")
    parser.add_argument("--max_vis_n", type=int, default=100,
                       help="Maximum number of samples per batch to visualize")
    parser.add_argument("--max_mesh_k", type=int, default=3,
                       help="Maximum number of batches for mesh generation")
    parser.add_argument("--max_mesh_n", type=int, default=200,
                       help="Maximum number of samples per batch for mesh")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Auto-generate output directory name if not specified
    if args.output_dir is None:
        from datetime import datetime
        timestamp = datetime.now().strftime("%y%m%d_%H%M")
        data_dir_name = args.data_dir.name
        args.output_dir = Path(f"plots/step3_5_data_viz_{data_dir_name}_{timestamp}")
    
    # Setup
    logger = setup_logging(args.verbose)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("Starting Step 3.5: Data Visualization Pipeline")
    logger.info(f"Data directory: {args.data_dir}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Max datasets: {args.max_datasets}")
    logger.info(f"Coordinate combinations: {args.n_combinations}")
    logger.info(f"Visualization limits: k≤{args.max_vis_k}, N≤{args.max_vis_n}")
    logger.info(f"Mesh limits: k≤{args.max_mesh_k}, N≤{args.max_mesh_n}")
    
    if OPEN3D_AVAILABLE:
        logger.info("Open3D available - mesh generation enabled")
    else:
        logger.info("Open3D not available - only point cloud visualization")
    
    # Find dataset directories
    dataset_dirs = [d for d in args.data_dir.iterdir() 
                   if d.is_dir() and d.name.startswith("dataset")]
    dataset_dirs = sorted(dataset_dirs, key=lambda x: int(x.name.replace("dataset", "")))
    
    if args.max_datasets > 0:
        dataset_dirs = dataset_dirs[:args.max_datasets]
    
    logger.info(f"Found {len(dataset_dirs)} datasets to process")
    
    # Process each dataset
    processed_datasets = []
    
    for i, dataset_dir in enumerate(dataset_dirs):
        dataset_name = dataset_dir.name
        logger.info(f"\n[{i+1}/{len(dataset_dirs)}] Processing {dataset_name}...")
        
        try:
            # Load dataset files
            files = load_dataset_files(dataset_dir)
            
            if 'metadata' not in files:
                logger.warning(f"  ✗ No metadata found for {dataset_name}")
                continue
            
            metadata = files['metadata']
            intrinsic = files.get('intrinsic', None)
            
            # Store dataset info
            dataset_info = {
                'dataset_name': dataset_name,
                'metadata': metadata,
                'intrinsic_available': intrinsic is not None
            }
            
            # Process each data type
            data_types = ['raw', 'clean', 'noisy']
            for data_type in data_types:
                data_key = f'{data_type}_data'
                if data_key in files:
                    data = files[data_key]
                    dataset_info['data_shape'] = data.shape
                    
                    logger.info(f"  Processing {data_type} data (shape: {data.shape})")
                    
                    # Create point cloud visualizations
                    create_point_cloud_plots(
                        data, intrinsic, metadata, args.output_dir, 
                        dataset_name, data_type, logger, args.n_combinations,
                        args.max_vis_k, args.max_vis_n
                    )
                    
                    # Create mesh visualization (only for one data type to save time)
                    if data_type == 'clean':  # Use clean data for mesh
                        create_mesh_visualization(
                            data, intrinsic, metadata, args.output_dir,
                            dataset_name, data_type, logger,
                            args.n_combinations, args.max_mesh_k, args.max_mesh_n,
                            save_ply=False  # Only save PNG plots, not PLY files for now
                        )
                else:
                    logger.warning(f"  ✗ No {data_type} data found for {dataset_name}")
            
            processed_datasets.append(dataset_info)
            logger.info(f"  ✓ Completed {dataset_name}")
            
        except Exception as e:
            logger.error(f"  ✗ Failed to process {dataset_name}: {str(e)}")
            continue
    
    # Create summary visualization
    if processed_datasets:
        create_summary_visualization(processed_datasets, args.output_dir, logger)
    
    # Save comprehensive metadata about this visualization run
    save_visualization_metadata(args, processed_datasets, args.output_dir, logger)
    
    logger.info(f"\n{'='*60}")
    logger.info("VISUALIZATION COMPLETE!")
    logger.info(f"{'='*60}")
    logger.info(f"Processed {len(processed_datasets)} datasets successfully")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info("Generated:")
    logger.info("  📊 Point cloud visualizations (FULL + BOUNDED versions)")
    logger.info("  🔗 Mesh reconstructions (if Open3D available)")
    logger.info("  📋 Summary comparison plots")
    logger.info("  📝 Comprehensive metadata logs")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()
