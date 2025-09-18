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
    """Create multiple 3D point cloud visualizations with different coordinate combinations."""
    
    properties = metadata['properties']
    k = properties['k']
    N = properties['N']
    D = properties['D']
    d = properties['d']
    
    logger.info(f"Creating point cloud plots for {dataset_name} ({data_type})")
    logger.info(f"  Data shape: {data.shape}, k={k}, N={N}, D={D}, d={d}")
    
    # Apply visualization limits for clearer plots
    vis_k = min(k, max_vis_k)
    vis_n = min(N, max_vis_n)
    total_vis_points = vis_k * vis_n
    
    logger.info(f"  Visualization limits: k={vis_k}/{k}, N={vis_n}/{N}, total={total_vis_points}")
    
    # Sample data for visualization
    vis_data = []
    for batch_idx in range(vis_k):
        start_idx = batch_idx * N
        end_idx = (batch_idx + 1) * N
        batch_data = data[start_idx:end_idx]
        
        # Sample vis_n points from this batch
        if vis_n < N:
            sample_indices = np.random.choice(N, vis_n, replace=False)
            batch_sample = batch_data[sample_indices]
        else:
            batch_sample = batch_data
        
        vis_data.append(batch_sample)
    
    vis_data = np.vstack(vis_data)
    logger.info(f"  Sampled data shape: {vis_data.shape}")
    
    # Get colors for visualization batches
    colors = get_colorblind_friendly_colors(vis_k)
    
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
    
    logger.info(f"  Creating {len(combinations)} coordinate combinations")
    
    # Create plots for each combination
    for idx, (dim1, dim2, dim3) in enumerate(combinations):
        
        # Create 3D point cloud plot
        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot each batch with different color
        for batch_idx in range(vis_k):
            start_idx = batch_idx * vis_n
            end_idx = (batch_idx + 1) * vis_n
            
            batch_data = vis_data[start_idx:end_idx]
            
            ax.scatter(
                batch_data[:, dim1],
                batch_data[:, dim2], 
                batch_data[:, dim3],
                c=colors[batch_idx],
                label=f'Batch {batch_idx + 1}',
                alpha=0.7,
                s=15
            )
        
        ax.set_xlabel(f'Dimension {dim1}')
        ax.set_ylabel(f'Dimension {dim2}')
        ax.set_zlabel(f'Dimension {dim3}')
        ax.set_title(f'{dataset_name} - {data_type.title()} Data\n'
                    f'3D Point Cloud: Dimensions ({dim1}, {dim2}, {dim3})\n'
                    f'Showing {vis_k}/{k} batches, {vis_n}/{N} samples/batch, σ={properties.get("kernel_smoothness", "N/A")}',
                    fontsize=12, pad=20)
        
        # Position legend outside the plot
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        
        # Save 3D plot
        plot_path = plot_dir / f'3d_pointcloud_dims_{dim1}_{dim2}_{dim3}.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create 2D projection plot
        fig, ax = plt.subplots(figsize=(10, 8))
        
        for batch_idx in range(vis_k):
            start_idx = batch_idx * vis_n
            end_idx = (batch_idx + 1) * vis_n
            
            batch_data = vis_data[start_idx:end_idx]
            
            ax.scatter(
                batch_data[:, dim1],
                batch_data[:, dim2],
                c=colors[batch_idx],
                label=f'Batch {batch_idx + 1}',
                alpha=0.7,
                s=15
            )
        
        ax.set_xlabel(f'Dimension {dim1}')
        ax.set_ylabel(f'Dimension {dim2}')
        ax.set_title(f'{dataset_name} - {data_type.title()} Data\n'
                    f'2D Projection: Dimensions ({dim1}, {dim2})\n'
                    f'Showing {vis_k}/{k} batches, {vis_n}/{N} samples/batch, σ={properties.get("kernel_smoothness", "N/A")}',
                    fontsize=12, pad=15)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save 2D plot
        plot_path_2d = plot_dir / f'2d_projection_dims_{dim1}_{dim2}.png'
        plt.savefig(plot_path_2d, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"    ✓ Saved combination {idx+1}/{len(combinations)}: 3D dims ({dim1},{dim2},{dim3}) + 2D projection ({dim1},{dim2})")


def create_mesh_visualization(
    data: np.ndarray,
    intrinsic: np.ndarray,
    metadata: Dict[str, Any],
    output_dir: Path,
    dataset_name: str,
    data_type: str,
    logger: logging.Logger,
    max_mesh_k: int = 3,
    max_mesh_n: int = 200
) -> None:
    """Create mesh visualization using Open3D with proper surface reconstruction."""
    
    if not OPEN3D_AVAILABLE:
        logger.warning("Open3D not available, skipping mesh generation")
        return
    
    properties = metadata['properties']
    k = properties['k']
    N = properties['N']
    
    # Apply limits for mesh generation (meshes need fewer points for good results)
    mesh_k = min(k, max_mesh_k)
    mesh_n = min(N, max_mesh_n)
    
    logger.info(f"Creating mesh visualization for {dataset_name} ({data_type})")
    logger.info(f"  Using {mesh_k}/{k} batches, {mesh_n}/{N} samples/batch for mesh generation")
    
    # Create directory for meshes
    mesh_dir = output_dir / "meshes" / dataset_name / data_type
    mesh_dir.mkdir(parents=True, exist_ok=True)
    
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
    
    try:
        # Sample data for mesh generation
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
        
        # Use first 3 dimensions for 3D coordinates
        points_3d = mesh_data[:, :3] if mesh_data.shape[1] >= 3 else np.column_stack([
            mesh_data[:, 0] if mesh_data.shape[1] > 0 else np.zeros(mesh_data.shape[0]),
            mesh_data[:, 1] if mesh_data.shape[1] > 1 else np.zeros(mesh_data.shape[0]),
            mesh_data[:, 2] if mesh_data.shape[1] > 2 else np.zeros(mesh_data.shape[0])
        ])
        
        # Create point cloud
        with suppress_stdout_stderr():
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points_3d)
            pcd.colors = o3d.utility.Vector3dVector(point_colors)
            
            # Save point cloud
            pcd_path = mesh_dir / f"pointcloud_{data_type}.ply"
            o3d.io.write_point_cloud(str(pcd_path), pcd)
            logger.info(f"    ✓ Saved point cloud: {pcd_path}")
        
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
                        
                        # Save mesh
                        mesh_path = mesh_dir / f"mesh_{method_name}_{data_type}.ply"
                        o3d.io.write_triangle_mesh(str(mesh_path), mesh)
                        
                        # Get mesh stats
                        n_vertices = len(mesh.vertices)
                        n_triangles = len(mesh.triangles)
                        file_size = mesh_path.stat().st_size / (1024 * 1024)
                        
                        logger.info(f"    ✓ {method_title}: {n_vertices} vertices, {n_triangles} triangles ({file_size:.1f} MB)")
                        
                        # Create matplotlib visualization
                        vertices = np.asarray(mesh.vertices)
                        triangles = np.asarray(mesh.triangles)
                        
                        fig = plt.figure(figsize=(15, 10))
                        
                        # 3D mesh plot
                        ax1 = fig.add_subplot(221, projection='3d')
                        ax1.plot_trisurf(vertices[:, 0], vertices[:, 1], vertices[:, 2], 
                                        triangles=triangles, alpha=0.7, cmap='viridis')
                        ax1.set_xlabel('X')
                        ax1.set_ylabel('Y') 
                        ax1.set_zlabel('Z')
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
                        ax3.set_xlabel('X')
                        ax3.set_ylabel('Y')
                        ax3.set_title('XY Projection')
                        ax3.set_aspect('equal')
                        
                        ax4 = fig.add_subplot(224)
                        ax4.triplot(vertices[:, 0], vertices[:, 2], triangles, alpha=0.5)
                        ax4.set_xlabel('X')
                        ax4.set_ylabel('Z')
                        ax4.set_title('XZ Projection')
                        ax4.set_aspect('equal')
                        
                        plt.suptitle(f'{dataset_name} - {data_type.title()} - {method_title}', fontsize=14)
                        plt.tight_layout()
                        
                        # Save visualization
                        vis_path = mesh_dir / f"mesh_{method_name}_{data_type}_visualization.png"
                        plt.savefig(vis_path, dpi=300, bbox_inches='tight')
                        plt.close()
                        
                        logger.info(f"      ✓ Saved visualization: {vis_path.name}")
                        
                    else:
                        logger.warning(f"    ✗ {method_title} failed: empty mesh")
                        
            except Exception as e:
                logger.warning(f"    ✗ {method_title} failed: {str(e)}")
                continue
        
    except Exception as e:
        logger.error(f"Mesh visualization failed: {str(e)}")


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
                            args.max_mesh_k, args.max_mesh_n
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
    
    logger.info(f"\n{'='*60}")
    logger.info("VISUALIZATION COMPLETE!")
    logger.info(f"{'='*60}")
    logger.info(f"Processed {len(processed_datasets)} datasets successfully")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info("Generated:")
    logger.info("  📊 Point cloud visualizations (multiple 3D combinations)")
    logger.info("  🔗 Mesh reconstructions (if Open3D available)")
    logger.info("  📋 Summary comparison plots")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()
