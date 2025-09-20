# Curvature Neighborhood Analysis

This script analyzes how curvature estimates vary with different neighborhood sizes.

## Usage

```bash
# Basic usage with default parameters
python scripts/curvature_neighborhood_analysis.py

# Specify custom data directory and output
python scripts/curvature_neighborhood_analysis.py \
  --data_dir data/your_data_folder \
  --output_dir results/curvature_analysis

# Custom neighborhood size range
python scripts/curvature_neighborhood_analysis.py \
  --k_min 10 --k_max 80 --k_step 10 \
  --max_points 500

# Verbose output
python scripts/curvature_neighborhood_analysis.py -v
```

## Parameters

- `--data_dir`: Directory containing dataset folders (default: `data/data_250913_1749`)
- `--output_dir`: Output directory for results (default: `results/curvature_neighborhood_analysis`)
- `--k_min`: Minimum neighborhood size (default: 5)
- `--k_max`: Maximum neighborhood size (default: 100)
- `--k_step`: Step size for neighborhood range (default: 5)
- `--max_points`: Maximum points to analyze per dataset (default: 1000)
- `--verbose`: Enable verbose logging

## Output

The script generates:

1. **Individual dataset plots**: `curvature_vs_k_{dataset}.png`
   - 4 subplots showing:
     - Mean curvature vs k_neighbors
     - Success rate vs k_neighbors
     - PCA curvature vs k_neighbors
     - Method comparison

2. **Summary plot**: `curvature_vs_k_summary.png`
   - Comparison across all datasets
   - Stability analysis (coefficient of variation)

3. **Data files**:
   - `curvature_neighborhood_analysis.json`: Detailed results
   - `curvature_neighborhood_summary.csv`: Summary statistics

## Notes

- The analysis focuses only on clean (non-noisy) dataset coordinates
- Subsampling is used for efficiency when datasets are large
- Both mean curvature and PCA-based curvature methods are tested
- The script handles missing data and computation failures gracefully
