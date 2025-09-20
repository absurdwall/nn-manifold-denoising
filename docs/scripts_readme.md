# Scripts Directory

This directory contains executable scripts for running the pipeline steps.
These scripts provide a user-friendly interface to the core functionality
without requiring users to read the source code.

## Pipeline Scripts:
- `step1_generate_datasets.py` - Generate manifold datasets
- `step2_geometric_analysis.py` - Analyze geometric properties  
- `step3_create_visualizations.py` - Generate plots and figures
- `step4_train_networks.py` - Train denoising neural networks
- `step5_analyze_results.py` - Analyze and summarize results

## Usage:
Each script can be run independently with command line arguments:
```bash
python step1_generate_datasets.py --ds 2,4,8 --Ds 100,300 --sigma 0.1,1,10
```

These scripts are user-facing and designed for easy replication of experiments.
