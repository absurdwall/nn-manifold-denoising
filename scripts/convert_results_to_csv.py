#!/usr/bin/env python3
"""
Convert JSON results to CSV format for analysis.
"""

import os
import json
import pandas as pd
import glob

def convert_exp1_to_csv(results_dir):
    """Convert experiment 1 JSON results to CSV."""
    json_files = glob.glob(os.path.join(results_dir, "dataset*_single_net.json"))
    
    csv_data = []
    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                result = json.load(f)
            
            row = {
                'experiment_id': result['experiment_id'],
                'dataset_name': result['dataset_name'],
                'test_loss': result['test_loss'],
                'best_val_loss': result['best_val_loss'],
                'epochs_completed': result['epochs_completed'],
                'training_time': result['training_time'],
                # Dataset properties
                'D': result['dataset_metadata']['D'],
                'd': result['dataset_metadata']['d'],
                'noise_level': result['dataset_metadata']['noise_level'],
                'N': result['dataset_metadata']['N'],
                # Network properties
                'network_type': result['network_config']['network_type'],
                'width': result['network_config']['width'],
                'depth': result['network_config']['depth'],
                'activation': result['network_config']['activation'],
                'norm_type': result['network_config']['norm_type'],
                'use_residual': result['network_config']['use_residual'],
                'init_scheme': result['network_config']['init_scheme'],
                # Training properties
                'learning_rate': result['training_config']['learning_rate'],
                'optimizer_name': result['training_config']['optimizer_name'],
                'scheduler_mode': result['training_config']['scheduler_mode'],
                'batch_size': result['training_config']['batch_size'],
            }
            csv_data.append(row)
        except Exception as e:
            print(f"Error processing {json_file}: {e}")
    
    if csv_data:
        df = pd.DataFrame(csv_data)
        csv_path = os.path.join(results_dir, 'experiment1_results.csv')
        df.to_csv(csv_path, index=False)
        print(f"Converted {len(csv_data)} results to CSV: {csv_path}")
        return csv_path
    return None

if __name__ == "__main__":
    # Convert experiment 1 results
    exp1_dir = "/home/tim/python_projects/nn_manifold_denoising/results/nn_train_exp1_250914_0100/tb20250921_2027"
    convert_exp1_to_csv(exp1_dir)
