import os
import sys
from pathlib import Path
import argparse
import numpy as np
import scanpy as sc
from sklearn.metrics import silhouette_score
from vcell.utils import load_yaml, set_seed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/data.yaml")
    args = parser.parse_args()

    # Load configuration parameters
    cfg = load_yaml(args.config)
    set_seed(int(cfg.get("seed", 42)))

    raw_h5ad = cfg["raw_h5ad"]
    target_col = cfg.get("target_col", "target_gene")
    batch_col = cfg.get("batch_col", "batch")  # Defaults to 'batch' if not explicit

    print(f"Loading H5AD: {raw_h5ad}")
    adata = sc.read_h5ad(raw_h5ad)
    print(f"Loaded AnnData shape: {adata.shape}")

    # Standardize data processing before running PCA
    print("Normalizing total counts to 1e4 and applying log1p...")
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    print("Extracting Highly Variable Genes (HVGs)...")
    sc.pp.highly_variable_genes(adata, n_top_genes=2000, flavor="seurat")
    
    print("Running PCA...")
    sc.tl.pca(adata, n_comps=30, use_highly_variable=True)
    pca_coords = adata.obsm["X_pca"]

    print("\n--- PCA Variance Investigation ---")

    # Evaluate Batch Variance
    if batch_col in adata.obs.columns:
        # Use a sample size of 10,000 cells to accelerate the calculation
        batch_score = silhouette_score(pca_coords, adata.obs[batch_col], sample_size=10000, random_state=42)
        print(f"Silhouette Score (Batch: '{batch_col}'): {batch_score:.4f}")
    else:
        print(f"Error: Batch column '{batch_col}' not found in metadata.")
        batch_score = None

    # Evaluate Perturbation Variance (Replacing Cell Type)
    if target_col in adata.obs.columns:
        pert_score = silhouette_score(pca_coords, adata.obs[target_col], sample_size=10000, random_state=42)
        print(f"Silhouette Score (Perturbation: '{target_col}'): {pert_score:.4f}")
    else:
        print(f"Warning: Perturbation column '{target_col}' not found in metadata.")
        pert_score = None

    print("\n--- CONCLUSION ---")
    if batch_score is not None and pert_score is not None:
        if batch_score > pert_score:
            print("Conclusion: Technical batch effects are driving more variance than the biological perturbations.")
            print("Recommendation: You may need to apply an integration method (e.g., Harmony) before modeling.")
        else:
            print("Conclusion: Biological perturbations are driving more variance than technical batch effects.")
            print("Recommendation: The dataset signal looks healthy! Proceed directly to pseudobulk preprocessing.")
    else:
        print("Analysis incomplete due to missing metadata columns.")


if __name__ == "__main__":
    main()