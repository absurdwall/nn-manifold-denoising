#!/usr/bin/env python3
"""
Visualization Summary and Cleanup

This script helps you understand what visualizations have been generated and optionally clean up old files.
"""

import argparse
from pathlib import Path
import shutil

def summarize_visualizations(base_dir: Path):
    """Summarize all visualizations that have been generated."""
    
    print("📊 VISUALIZATION SUMMARY")
    print("=" * 60)
    
    # Point cloud visualizations
    point_cloud_dir = base_dir / "point_clouds"
    if point_cloud_dir.exists():
        print("\n🎯 POINT CLOUD VISUALIZATIONS:")
        
        datasets = [d for d in point_cloud_dir.iterdir() if d.is_dir()]
        
        for dataset in sorted(datasets):
            print(f"\n  📂 {dataset.name}:")
            
            data_types = [d for d in dataset.iterdir() if d.is_dir()]
            for data_type in sorted(data_types):
                png_files = list(data_type.glob("*.png"))
                
                # Separate new and old format files
                new_3d = [f for f in png_files if f.name.startswith("3d_pointcloud_")]
                new_2d = [f for f in png_files if f.name.startswith("2d_projection_")]
                old_files = [f for f in png_files if f.name.startswith("pointcloud_dims_")]
                
                print(f"    📁 {data_type.name}:")
                if new_3d:
                    print(f"      ✅ {len(new_3d)} new 3D plots")
                if new_2d:
                    print(f"      ✅ {len(new_2d)} new 2D projections")
                if old_files:
                    print(f"      🗂️  {len(old_files)} old combined plots")
    
    # Mesh visualizations
    mesh_dir = base_dir / "meshes"
    if mesh_dir.exists():
        print("\n🔗 MESH VISUALIZATIONS:")
        
        datasets = [d for d in mesh_dir.iterdir() if d.is_dir()]
        
        for dataset in sorted(datasets):
            print(f"\n  📂 {dataset.name}:")
            
            data_types = [d for d in dataset.iterdir() if d.is_dir()]
            for data_type in sorted(data_types):
                ply_files = list(data_type.glob("*.ply"))
                
                point_clouds = [f for f in ply_files if "pointcloud" in f.name]
                meshes = [f for f in ply_files if "mesh_" in f.name]
                
                print(f"    📁 {data_type.name}:")
                if point_clouds:
                    print(f"      ☁️  {len(point_clouds)} point cloud PLY files")
                if meshes:
                    print(f"      🔷 {len(meshes)} mesh PLY files")
                    for mesh in meshes:
                        size_mb = mesh.stat().st_size / (1024 * 1024)
                        mesh_type = mesh.name.replace("mesh_", "").replace(f"_{data_type.name}.ply", "")
                        print(f"         - {mesh_type}: {size_mb:.1f} MB")
    
    # PLY views
    ply_views_dir = base_dir.parent / "ply_views"
    if ply_views_dir.exists():
        print("\n🖼️  PLY VISUALIZATION IMAGES:")
        png_files = list(ply_views_dir.glob("*.png"))
        print(f"  ✅ {len(png_files)} PLY visualization images generated")
        for png_file in sorted(png_files):
            size_mb = png_file.stat().st_size / (1024 * 1024)
            print(f"     - {png_file.name} ({size_mb:.1f} MB)")
    
    # Summary
    summary_dir = base_dir / "summary"
    if summary_dir.exists():
        print("\n📋 SUMMARY FILES:")
        summary_files = list(summary_dir.glob("*"))
        for summary_file in summary_files:
            size_kb = summary_file.stat().st_size / 1024
            print(f"  ✅ {summary_file.name} ({size_kb:.1f} KB)")


def cleanup_old_files(base_dir: Path, dry_run: bool = True):
    """Clean up old visualization files."""
    
    print("\n🧹 CLEANUP ANALYSIS")
    print("=" * 60)
    
    point_cloud_dir = base_dir / "point_clouds"
    old_files = []
    
    if point_cloud_dir.exists():
        # Find old format files
        for png_file in point_cloud_dir.rglob("*.png"):
            if png_file.name.startswith("pointcloud_dims_"):
                old_files.append(png_file)
    
    if old_files:
        print(f"\n📁 Found {len(old_files)} old format files:")
        total_size = 0
        for old_file in old_files:
            size_mb = old_file.stat().st_size / (1024 * 1024)
            total_size += size_mb
            print(f"  - {old_file.relative_to(base_dir)} ({size_mb:.1f} MB)")
        
        print(f"\n💾 Total size to be freed: {total_size:.1f} MB")
        
        if dry_run:
            print("\n⚠️  This is a DRY RUN. Use --confirm to actually delete files.")
        else:
            print("\n🗑️  Deleting old files...")
            for old_file in old_files:
                old_file.unlink()
                print(f"    ✅ Deleted {old_file.name}")
            print(f"✨ Cleanup complete! Freed {total_size:.1f} MB")
    else:
        print("✅ No old format files found. Nothing to clean up!")


def show_usage_examples():
    """Show examples of how to use the visualizations."""
    
    print("\n📖 USAGE EXAMPLES")
    print("=" * 60)
    
    print("""
🎯 View Point Cloud Plots:
   Open PNG files in plots/step3_5_data_visualization/point_clouds/
   
   Example files:
   - 3d_pointcloud_dims_0_1_2.png      (3D scatter plot)
   - 2d_projection_dims_0_1.png        (2D projection)

🔗 View Mesh Files:
   Use the PLY viewer script:
   
   # View all PLY files
   python scripts/view_ply_files.py
   
   # View specific PLY file
   python scripts/view_ply_files.py --ply_file plots/step3_5_data_visualization/meshes/dataset0/clean/pointcloud_clean.ply
   
   # Generate visualization images only
   python scripts/view_ply_files.py --method matplotlib

🖼️  View Generated PLY Images:
   Check plots/ply_views/ for PNG images of PLY files
   
🔄 Run More Visualizations:
   # Process more datasets
   python scripts/step3_5_data_visualization.py --max_datasets 5
   
   # More coordinate combinations
   python scripts/step3_5_data_visualization.py --n_combinations 10

📊 Alternative PLY Viewers:
   - MeshLab: Free 3D mesh viewer (meshlab.net)
   - CloudCompare: Point cloud viewer (cloudcompare.org)
   - Online viewers: threejs.org editor, viewstl.com
""")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Visualization Summary and Cleanup")
    parser.add_argument("--base_dir", type=Path,
                       default=Path("plots/step3_5_data_visualization"),
                       help="Base visualization directory")
    parser.add_argument("--cleanup", action="store_true",
                       help="Show cleanup analysis")
    parser.add_argument("--confirm", action="store_true",
                       help="Actually perform cleanup (use with --cleanup)")
    parser.add_argument("--examples", action="store_true",
                       help="Show usage examples")
    
    args = parser.parse_args()
    
    # Always show summary
    if args.base_dir.exists():
        summarize_visualizations(args.base_dir)
    else:
        print(f"❌ Directory {args.base_dir} does not exist")
        return
    
    # Show cleanup if requested
    if args.cleanup:
        cleanup_old_files(args.base_dir, dry_run=not args.confirm)
    
    # Show examples if requested
    if args.examples:
        show_usage_examples()
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
