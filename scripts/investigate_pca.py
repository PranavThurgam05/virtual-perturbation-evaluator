import os
import sys
from pathlib import Path

root_dir = Path(__file__).resolve().parents[1]
if str(root_dir / "src") not in sys.path:
    sys.path.append(str(root_dir / "src"))

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

    # Set up and redirect Scanpy's figure saving directory
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)
    sc.settings.figdir = output_dir
    sc.settings.set_figure_params(dpi=150, format="png", color_map="viridis")

    raw_h5ad = cfg["raw_h5ad"]
    target_col = cfg.get("target_col", "target_gene")
    batch_col = cfg.get("batch_col", "batch")

    print(f"Loading H5AD: {raw_h5ad}")
    adata = sc.read_h5ad(raw_h5ad)
    print(f"Loaded AnnData shape: {adata.shape}")
    
    # Re-evaluate color targets now that adata is loaded
    color_targets = [col for col in [batch_col, target_col] if col in adata.obs.columns]

    # Standardize data processing before running PCA
    print("Normalizing total counts to 1e4 and applying log1p...")
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    print("Extracting Highly Variable Genes (HVGs)...")
    sc.pp.highly_variable_genes(adata, n_top_genes=2000, flavor="seurat")
    
    print("Running PCA (computing 30 components)...")
    sc.tl.pca(adata, n_comps=30, use_highly_variable=True)
    pca_coords = adata.obsm["X_pca"]

    print("\n--- PCA Variance Investigation ---")

    # Evaluate Batch Variance
    if batch_col in adata.obs.columns:
        batch_score = silhouette_score(pca_coords, adata.obs[batch_col], sample_size=10000, random_state=42)
        print(f"Silhouette Score (Batch: '{batch_col}'): {batch_score:.4f}")
    else:
        print(f"Error: Batch column '{batch_col}' not found in metadata.")
        batch_score = None

    # Evaluate Perturbation Variance
    if target_col in adata.obs.columns:
        pert_score = silhouette_score(pca_coords, adata.obs[target_col], sample_size=10000, random_state=42)
        print(f"Silhouette Score (Perturbation: '{target_col}'): {pert_score:.4f}")
    else:
        print(f"Warning: Perturbation column '{target_col}' not found in metadata.")
        pert_score = None

    # --- GENERATING VERIFICATION PLOTS ---
    print("\n--- Generating Verification Plots ---")
    
    # 1. Save Classic PCA Scatter Plots (PC1 vs PC2)
    print(f"Saving classic PC1 vs PC2 scatter plots to {output_dir}/pca_scatter.png...")
    if color_targets:
        # scanpy saves this automatically as 'pca_scatter.png' when using save="_scatter.png"
        sc.pl.pca(adata, color=color_targets, show=False, save="_scatter.png")

    # 2. Save PCA Variance Ratio Elbow Plot
    print(f"Saving PCA variance ratio (elbow plot) to {output_dir}/pca_variance_ratio.png...")
    sc.pl.pca_variance_ratio(adata, n_pcs=30, log=False, show=False, save=".png")

    # 3. Compute Embeddings and Save UMAP Plots
    print("Computing neighborhood graph and UMAP coordinates...")
    sc.pp.neighbors(adata, n_neighbors=15, n_pcs=30)
    sc.tl.umap(adata)
    
    print(f"Saving UMAP plot to {output_dir}/umap_investigation.png...")
    if color_targets:
        sc.pl.umap(adata, color=color_targets, show=False, save="_investigation.png")

    # 4. Save Violin Plots of top Principal Components
    print(f"Saving PC distribution violin plots to {output_dir}/...")
    if batch_col in adata.obs.columns:
        sc.pl.violin(adata, keys="PC1", groupby=batch_col, show=False, save=f"_pc1_by_batch.png")
    if target_col in adata.obs.columns:
        sc.pl.violin(adata, keys="PC1", groupby=target_col, show=False, save=f"_pc1_by_perturbation.png")

    print(f"\nAll verification graphics successfully saved inside the '{output_dir}/' directory!")

    print("\n--- CONCLUSION ---")
    if batch_score is not None and pert_score is not None:
        if batch_score > pert_score:
            print("Conclusion: Technical batch effects are driving more variance than the biological perturbations.")
        else:
            print("Conclusion: Biological perturbations are driving more variance than technical batch effects.")
            print("Recommendation: The dataset signal looks healthy! Proceed directly to pseudobulk preprocessing.")
    else:
        print("Analysis incomplete due to missing metadata columns.")


if __name__ == "__main__":
    main()