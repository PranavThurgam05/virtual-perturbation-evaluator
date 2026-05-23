import argparse
from pathlib import Path
import scanpy as sc
import numpy as np
from sklearn.metrics import silhouette_score
from vcell.utils import load_yaml, set_seed

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/data.yaml")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    set_seed(int(cfg.get("seed", 42)))

    raw_h5ad = cfg["raw_h5ad"]
    batch_col = cfg.get("batch_col", "batch")
    cell_type_col = cfg.get("cell_type_col", "cell_type")

    print(f"Loading H5AD: {raw_h5ad}")
    adata = sc.read_h5ad(raw_h5ad)

    # Basic QC and Normalization to prepare for PCA
    adata.var['mt'] = adata.var_names.str.startswith('MT-')
    sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)
    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)
    adata = adata[adata.obs.pct_counts_mt < 5, :]

    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    # Calculate highly variable genes and run PCA
    print("Running PCA...")
    sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
    sc.tl.pca(adata, svd_solver='arpack')

    pca_embeddings = adata.obsm['X_pca']

    # Investigate Variance via Silhouette Scores
    print("\n--- PCA Variance Investigation ---")
    
    if batch_col in adata.obs.columns:
        batch_score = silhouette_score(pca_embeddings, adata.obs[batch_col])
        print(f"Silhouette Score (Batch): {batch_score:.4f}")
    else:
        print(f"Batch column '{batch_col}' not found.")
        batch_score = -1

    if cell_type_col in adata.obs.columns:
        cell_type_score = silhouette_score(pca_embeddings, adata.obs[cell_type_col])
        print(f"Silhouette Score (Cell Type): {cell_type_score:.4f}")
    else:
        print(f"Cell type column '{cell_type_col}' not found.")
        cell_type_score = -1

    # Conclusion
    if batch_score > cell_type_score:
        print("\nCONCLUSION: Batch effect is driving more variance than cell type.")
        print("Recommendation: You may need to run Harmony or scVI integration before downstream evaluation.")
    elif cell_type_score > batch_score:
        print("\nCONCLUSION: Cell type biology is driving the primary variance.")
        print("Recommendation: Proceed with standard pseudobulking; batch effects are minimal.")

if __name__ == "__main__":
    main()