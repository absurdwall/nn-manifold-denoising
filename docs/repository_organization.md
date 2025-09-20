# Repository Organization Summary

This document describes the clean organization of the nn_manifold_denoising repository.

## Folder Structure

### Root Level Files
- `.gitignore` - Git ignore patterns
- `README.md` - Main project documentation
- `requirements.txt` - Python dependencies

### Main Directories

#### `/data/`
Contains all generated datasets with proper naming convention:
- `data_YYMMDD_HHMM/` - Timestamped dataset folders
- Example: `data_250914_0100/` (current main dataset)

#### `/results/`
Stores analysis results from experiments with descriptive names:
- `step2_data_250914_0100_250917_1813/` - Step 2 geometric analysis results
- Format: `step{N}_{data_source}_{timestamp}/`

#### `/plots/`
Contains all visualization outputs organized by analysis type:
- `step3_data_250914_0100_250917_1813/` - Step 3 plotting results
- `radius_analysis/` - Radius-based curvature analysis plots
- Format: `step{N}_{data_source}_{timestamp}/`

#### `/src/`
Core source code organized by functionality:
- `data_generation/` - Dataset generation and manifold creation
- `geometric_analysis/` - Curvature estimation and geometric analysis
- `network_training/` - Neural network definitions and training
- `plotting/` - Visualization utilities
- `utils/` - General utilities
- `analysis/` - Analysis framework components

#### `/scripts/`
Main workflow scripts in order of execution:
- `step1_generate_datasets.py` - Data generation
- `step2_geometric_analysis.py` - Geometric analysis
- `step2_comprehensive_analysis.py` - Comprehensive geometric analysis
- `step3_plotting_pipeline.py` - Visualization pipeline
- `step3_5_data_visualization.py` - Enhanced visualizations
- `analyze_curvature_results.py` - Result analysis utilities
- `visualization_summary.py` - Summary generation
- `create_radius_plots.py` - Radius analysis plots

#### `/testing/`
Temporary, experimental, and testing code:
- Debug scripts and experimental implementations
- Test files and validation scripts
- Code that can be safely deleted without affecting main workflow

#### `/examples/`
Illustration and demonstration code:
- Example usage scripts
- Tutorial implementations
- Sample code for documentation

#### `/docs/`
All documentation and markdown files:
- Technical documentation
- Analysis summaries
- Implementation guides
- README files from subdirectories
- Configuration templates for reference

#### `/logs/`
Log files from analysis runs and system outputs

## Naming Conventions

### Datasets
Format: `data_YYMMDD_HHMM`
- YY: Year (last 2 digits)
- MM: Month
- DD: Day  
- HH: Hour (24-hour format)
- MM: Minute

### Results
Format: `step{N}_{data_source}_{timestamp}`
- step{N}: Analysis step number
- data_source: Source dataset identifier
- timestamp: When analysis was performed

### Plots
Format: `step{N}_{data_source}_{timestamp}`
- Same convention as results for consistency

## File Organization Principles

1. **Core functionality** goes in `/src/` as reusable modules
2. **Main workflow scripts** go in `/scripts/` (including active config files)
3. **Temporary/experimental code** goes in `/testing/`
4. **Documentation** goes in `/docs/` (including config templates)
5. **Generated data/results/plots** use timestamped naming
6. **Root directory** only contains `.gitignore` and `README.md`

This organization ensures:
- Easy navigation and maintenance
- Clear separation of permanent vs temporary code
- Reproducible analysis with proper versioning
- Clean repository structure for collaboration
