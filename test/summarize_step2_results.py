#!/usr/bin/env python3
"""
Quick summary of Step 2 geometric analysis results.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path

def main():
    results_dir = Path("results/step2_analysis_250913_1749")
    
    # Load detailed results
    with open(results_dir / "detailed_results.json", 'r') as f:
        all_results = json.load(f)
    
    print("=" * 80)
    print("STEP 2 GEOMETRIC ANALYSIS SUMMARY")
    print("=" * 80)
    
    print(f"\nAnalyzed {len(all_results)} datasets from data_250913_1749")
    
    # Extract key information
    summary_data = []
    for result in all_results:
        if "error" in result:
            continue
        
        info = result["dataset_info"]
        basic = result["basic_info"]
        
        # Get true parameters from metadata
        true_d = info.get("properties", {}).get("d", "unknown")
        true_D = info.get("properties", {}).get("D", "unknown")
        true_k = info.get("properties", {}).get("k", "unknown")
        noise_sigma = info.get("properties", {}).get("noise_sigma", "unknown")
        
        # Get dimension estimates
        dim_ests = result.get("dimension_estimation", {})
        pca_est = dim_ests.get("PCA", np.nan)
        knn_est = dim_ests.get("k-NN", np.nan)
        twon_est = dim_ests.get("TwoNN", np.nan)
        mle_est = dim_ests.get("MLE", np.nan)
        
        # Get curvature info
        curv_stats = result.get("mean_curvature", {})
        mean_curv = curv_stats.get("mean", np.nan)
        curv_success = curv_stats.get("success_rate", 0)
        
        # Get geometric stats
        geom_stats = result.get("geometric_statistics", {})
        diameter = geom_stats.get("extrinsic_diameter", np.nan)
        
        summary_data.append({
            "dataset": info["name"],
            "true_d": true_d,
            "true_D": true_D,
            "true_k": true_k,
            "noise_sigma": noise_sigma,
            "PCA_est": pca_est,
            "kNN_est": knn_est,
            "TwoNN_est": twon_est,
            "MLE_est": mle_est,
            "mean_curvature": mean_curv,
            "curv_success_rate": curv_success,
            "diameter": diameter
        })
    
    df = pd.DataFrame(summary_data)
    
    # Group by true parameters
    print("\nDIMENSION ESTIMATION ACCURACY:")
    print("-" * 50)
    
    for true_d in sorted(df["true_d"].unique()):
        subset = df[df["true_d"] == true_d]
        if len(subset) == 0:
            continue
        
        print(f"\nTrue dimension d = {true_d} ({len(subset)} datasets):")
        
        for method in ["PCA_est", "kNN_est", "TwoNN_est", "MLE_est"]:
            values = subset[method].dropna()
            if len(values) > 0:
                mean_est = np.mean(values)
                std_est = np.std(values)
                mean_error = np.mean(np.abs(values - true_d))
                print(f"  {method:8}: {mean_est:6.2f} ± {std_est:5.2f} (avg error: {mean_error:5.2f})")
    
    print("\nCURVATURE ESTIMATION SUMMARY:")
    print("-" * 50)
    
    # Group by true parameters for curvature
    for true_d in sorted(df["true_d"].unique()):
        for true_k in sorted(df[df["true_d"] == true_d]["true_k"].unique()):
            subset = df[(df["true_d"] == true_d) & (df["true_k"] == true_k)]
            if len(subset) == 0:
                continue
            
            curv_values = subset["mean_curvature"].dropna()
            success_rates = subset["curv_success_rate"].dropna()
            
            if len(curv_values) > 0:
                print(f"d={true_d}, k={true_k}: curvature = {np.mean(curv_values):.3f} ± {np.std(curv_values):.3f}, success = {np.mean(success_rates):.2f}")
    
    print("\nGEOMETRIC PROPERTIES:")
    print("-" * 50)
    
    diameter_values = df["diameter"].dropna()
    print(f"Extrinsic diameter: {np.mean(diameter_values):.3f} ± {np.std(diameter_values):.3f}")
    print(f"Range: [{np.min(diameter_values):.3f}, {np.max(diameter_values):.3f}]")
    
    print("\nKEY OBSERVATIONS:")
    print("-" * 50)
    
    # Check dimension estimation accuracy
    d2_data = df[df["true_d"] == 2]
    d4_data = df[df["true_d"] == 4]
    
    if len(d2_data) > 0:
        pca_d2_error = np.mean(np.abs(d2_data["PCA_est"].dropna() - 2))
        knn_d2_error = np.mean(np.abs(d2_data["kNN_est"].dropna() - 2))
        print(f"• For d=2 manifolds: PCA error = {pca_d2_error:.2f}, k-NN error = {knn_d2_error:.2f}")
    
    if len(d4_data) > 0:
        pca_d4_error = np.mean(np.abs(d4_data["PCA_est"].dropna() - 4))
        knn_d4_error = np.mean(np.abs(d4_data["kNN_est"].dropna() - 4))
        print(f"• For d=4 manifolds: PCA error = {pca_d4_error:.2f}, k-NN error = {knn_d4_error:.2f}")
    
    # Check curvature success rate
    overall_success = np.mean(df["curv_success_rate"].dropna())
    print(f"• Overall curvature estimation success rate: {overall_success:.2f}")
    
    # Check embedding dimension effect
    d100_data = df[df["true_D"] == 100]
    d300_data = df[df["true_D"] == 300]
    
    if len(d100_data) > 0 and len(d300_data) > 0:
        d100_curv = np.mean(d100_data["mean_curvature"].dropna())
        d300_curv = np.mean(d300_data["mean_curvature"].dropna())
        print(f"• Mean curvature: D=100 → {d100_curv:.2f}, D=300 → {d300_curv:.2f}")
    
    print(f"\nDetailed results saved in: {results_dir}")
    print("Plots available in: plots/ subdirectory")
    print("CSV summary: analysis_summary.csv")

if __name__ == "__main__":
    main()
