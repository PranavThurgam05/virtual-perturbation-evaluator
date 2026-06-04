"""
Preprocessing pipeline: QC -> normalization -> batch correction -> pseudobulking
"""

import argparse
from pathlib import Path

import numpy as np
import scanpy as sc
from scipy import sparse, issparse
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import pandas as pd
import scvi

from vcell.utils import load_yaml, save_json, set_seed


def to_dense_1d(x):
    if sparse.issparse(x):
        return np.asarray(x.toarray()).ravel()
    return np.asarray(x).ravel()


def get_group_mean(X, mask):
    sub = X[mask]
    mean = sub.mean(axis=0)
    return to_dense_1d(mean).astype(np.float32)


def compute_gene_features(control_X, gene_names, n_components=128, max_control_cells=8000, seed=42):
    """
    Creates target-gene features for every gene.

    Features are based on:
    - control mean expression
    - control variance
    - dropout fraction
    - PCA-reduced gene-expression patterns across control cells

    Returns:
        gene_features: (n_genes, n_components) float32
    """
    rng = np.random.default_rng(seed)

    n_controls = control_X.shape[0]
    if n_controls > max_control_cells:
        idx = rng.choice(n_controls, size=max_control_cells, replace=False)
        X = control_X[idx]
    else:
        X = control_X

    if sparse.issparse(X):
        X_dense = X.toarray().astype(np.float32)
    else:
        X_dense = np.asarray(X, dtype=np.float32)

    # cells x genes -> genes x cells
    gene_by_cell = X_dense.T

    mean    = gene_by_cell.mean(axis=1, keepdims=True)
    var     = gene_by_cell.var(axis=1, keepdims=True)
    dropout = (gene_by_cell <= 1e-8).mean(axis=1, keepdims=True)

    # Center each gene profile before PCA
    centered = gene_by_cell - gene_by_cell.mean(axis=1, keepdims=True)

    # PCA over genes, using expression profile across sampled control cells
    pca_components = max(1, n_components - 3)
    pca = PCA(n_components=pca_components, random_state=seed)
    pca_features = pca.fit_transform(centered)

    features = np.concatenate([mean, var, dropout, pca_features], axis=1)

    # Standardize features
    features = features.astype(np.float32)
    features = (features - features.mean(axis=0, keepdims=True)) / (
        features.std(axis=0, keepdims=True) + 1e-6
    )

    assert features.shape[0] == len(gene_names)
    return features.astype(np.float32)


def compute_de_gene_sets(
    adata,
    target_col,
    control_label,
    perturbation_genes,
    gene_names,
    alpha=0.05,
    min_abs_lfc=0.0,
):
    """
    Significance-based DE gene set per perturbation (the ground truth the real
    DES is scored against).

    Runs a Wilcoxon rank-sum test of each perturbation's cells vs the control
    (non-targeting) cells on the log-normalized expression in ``adata.X``, then
    keeps genes with FDR-adjusted p < alpha (and |log2FC| >= min_abs_lfc). This
    needs the per-cell data: significance depends on within-group variance and
    cell counts, which pseudobulk means discard.

    Returns:
        de_gene_sets: object ndarray (n_perts,), each an int64 array of gene
                      indices (into gene_names), aligned to ``perturbation_genes``.
    """
    print(f"Computing Wilcoxon DE gene sets (alpha={alpha}, min_abs_lfc={min_abs_lfc})...")

    adata.obs[target_col] = adata.obs[target_col].astype("category")
    categories = set(adata.obs[target_col].cat.categories)
    if control_label not in categories:
        raise ValueError(f"Control label {control_label!r} not present in {target_col}")

    groups = [g for g in perturbation_genes if g in categories]
    sc.tl.rank_genes_groups(
        adata,
        groupby=target_col,
        groups=groups,
        reference=control_label,
        method="wilcoxon",
        use_raw=False,
    )

    res = adata.uns["rank_genes_groups"]
    names, padj, lfc = res["names"], res["pvals_adj"], res["logfoldchanges"]

    gene_to_idx = {g: i for i, g in enumerate(gene_names)}
    de_sets = []
    n_sig = []
    for g in perturbation_genes:
        if g not in categories:
            de_sets.append(np.array([], dtype=np.int64))
            n_sig.append(0)
            continue
        gnames = np.asarray(names[g])
        sig = (np.asarray(padj[g]) < alpha) & (np.abs(np.asarray(lfc[g])) >= min_abs_lfc)
        idx = np.array(
            [gene_to_idx[gn] for gn in gnames[sig] if gn in gene_to_idx],
            dtype=np.int64,
        )
        de_sets.append(idx)
        n_sig.append(int(idx.size))

    n_sig = np.asarray(n_sig)
    print(
        f"DE sets: {int((n_sig > 0).sum())}/{len(perturbation_genes)} perturbations "
        f"have >=1 significant gene; median set size = {int(np.median(n_sig))}, "
        f"max = {int(n_sig.max()) if n_sig.size else 0}"
    )

    de_gene_sets = np.empty(len(perturbation_genes), dtype=object)
    de_gene_sets[:] = de_sets
    return de_gene_sets


def apply_qc(adata, min_genes=200, max_genes=8000, max_pct_mt=5):
    """
    Filters low-quality cells and uninformative genes.
    Thresholds are passed in from config rather than hardcoded.
    """
    print("Running Quality Control...")
    adata.var["mt"] = adata.var_names.str.startswith("MT-")
    sc.pp.calculate_qc_metrics(
        adata, qc_vars=["mt"], percent_top=None, log1p=False, inplace=True
    )

    n_before = adata.n_obs
    sc.pp.filter_cells(adata, min_genes=min_genes)
    adata = adata[adata.obs["n_genes_by_counts"] < max_genes].copy()
    adata = adata[adata.obs["pct_counts_mt"] < max_pct_mt].copy()
    print(f"Cells after QC: {adata.n_obs} (removed {n_before - adata.n_obs})")
    return adata


def pseudobulk(expr, target_labels, agg="mean"):
    """
    Aggregates per-cell expression into one vector per perturbation group.

    Args:
        expr:          (n_cells, n_genes) float32 ndarray  [scvi_norm layer]
        target_labels: (n_cells,) str array of perturbation labels
        agg:           aggregation method — "mean" | "median" | "sum"

    Returns:
        pb: pd.DataFrame of shape (n_groups, n_genes), index = perturbation label
    """
    df = pd.DataFrame(expr, index=target_labels)
    if agg == "mean":
        return df.groupby(level=0).mean()
    elif agg == "median":
        return df.groupby(level=0).median()
    elif agg == "sum":
        return df.groupby(level=0).sum()
    else:
        raise ValueError(f"Invalid aggregation method: {agg!r}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/data.yaml")
    parser.add_argument("--n-components", type=int, default=128)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    set_seed(int(cfg.get("seed", 42)))

    raw_h5ad     = cfg["raw_h5ad"]
    processed_npz = Path(cfg["processed_npz"])
    split_json    = Path(cfg["split_json"])
    control_label = cfg.get("control_label", "non-targeting")
    target_col    = cfg.get("target_col",    "target_gene")
    batch_col     = cfg.get("batch_col",     "batch")

    processed_npz.parent.mkdir(parents=True, exist_ok=True)
    split_json.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading H5AD: {raw_h5ad}")
    adata = sc.read_h5ad(raw_h5ad)
    print(f"Loaded AnnData shape: {adata.shape}")
    print(f"obs columns: {list(adata.obs.columns)}")

    if target_col not in adata.obs.columns:
        raise ValueError(f"Missing target column '{target_col}' in adata.obs")

    # 1. Quality Control
    adata = apply_qc(
        adata,
        min_genes   = int(cfg.get("min_genes",   200)),
        max_genes   = int(cfg.get("max_genes",   8000)),
        max_pct_mt  = float(cfg.get("max_pct_mt", 5.0)),
    )

    # Save raw counts before any normalisation (needed by scVI)
    adata.layers["counts"] = adata.X.copy()

    # 2. Normalize + Log-transform
    print("Normalizing...")
    target_sum = float(cfg.get("normalize_total", 1e4))
    sc.pp.normalize_total(adata, target_sum=target_sum)
    sc.pp.log1p(adata)

    # 3. Highly Variable Genes
    n_hvg = cfg.get("n_hvg", None)
    if n_hvg is not None:
        print(f"Selecting {n_hvg} highly variable genes...")
        sc.pp.highly_variable_genes(
            adata,
            n_top_genes=n_hvg,
            batch_key=batch_col,
            flavor="seurat_v3",
            layer="counts",
        )
        adata = adata[:, adata.var["highly_variable"]].copy()
        print(f"Genes after HVG selection: {adata.n_vars}")
    cfg["n_genes"] = adata.n_vars

    # 4. Batch Correction (scVI)
    print("Running scVI batch correction...")
    try:
        scvi.settings.seed = int(cfg.get("seed", 42))
        scvi.model.SCVI.setup_anndata(adata, layer="counts", batch_key=batch_col)
        vae = scvi.model.SCVI(adata, n_latent=50, n_layers=2)
        vae.train(max_epochs=30, early_stopping=True)
        adata.obsm["X_scVI"]      = vae.get_latent_representation()
        adata.layers["scvi_norm"] = vae.get_normalized_expression(library_size=1e4)
        print("scVI complete.")
    except Exception as e:
        print(f"Error running scVI batch correction: {e}")
        adata.layers["scvi_norm"] = adata.X.toarray() if issparse(adata.X) else adata.X
        raise

    # 5. Pseudobulk + Delta Calculation
    print("Computing pseudobulk means and deltas...")

    expr          = adata.layers["scvi_norm"]
    if issparse(expr):
        expr = expr.toarray()

    gene_names        = np.asarray(adata.var_names.astype(str))
    target_labels     = np.asarray(adata.obs[target_col].astype(str))
    control_mask      = target_labels == control_label
    perturbation_genes = sorted(
        g for g in np.unique(target_labels) if g != control_label
    )

    print(f"Control cells:      {control_mask.sum():,}")
    print(f"Perturbation genes: {len(perturbation_genes):,}")
    print(f"Genes measured:     {len(gene_names):,}")

    # Pseudobulk — one row per perturbation label (including control)
    pb = pseudobulk(expr, target_labels, agg="mean")

    # Control mean from the pseudobulk row
    control_mean = pb.loc[control_label].values.astype(np.float32)  # (n_genes,)
    adata.uns["ctrl_mean"] = control_mean

    # Per-perturbation mean and delta, in sorted order
    pert_means       = []
    pert_deltas      = []
    pert_cell_counts = []

    for gene in tqdm(perturbation_genes, desc="Pseudobulking"):
        mean  = pb.loc[gene].values.astype(np.float32)
        delta = mean - control_mean
        pert_means.append(mean)
        pert_deltas.append(delta)
        pert_cell_counts.append(int((target_labels == gene).sum()))

    pert_means        = np.stack(pert_means).astype(np.float32)   # (n_perts, n_genes)
    pert_deltas       = np.stack(pert_deltas).astype(np.float32)  # (n_perts, n_genes)
    pert_cell_counts  = np.asarray(pert_cell_counts, dtype=np.int64)

    # Gene index lookup — built once, used once
    gene_to_idx = {g: i for i, g in enumerate(gene_names)}
    target_gene_indices = np.asarray(
        [gene_to_idx.get(g, -1) for g in perturbation_genes],
        dtype=np.int64,
    )
    missing = [g for g, idx in zip(perturbation_genes, target_gene_indices) if idx < 0]
    if missing:
        print(f"WARNING: {len(missing)} perturbation target genes not found in var_names.")
        print(missing[:10])

    # Gene features from control cells
    print("Computing gene features from control cells...")
    gene_features = compute_gene_features(
        expr[control_mask],
        gene_names=gene_names,
        n_components=args.n_components,
        seed=int(cfg.get("seed", 42)),
    )

    # Significance-based DE gene sets (ground truth for the real DES).
    # Uses log-normalized adata.X and the per-cell data, so it must happen here
    # while the cells are still in memory.
    de_alpha       = float(cfg.get("de_alpha", 0.05))
    de_min_abs_lfc = float(cfg.get("de_min_abs_lfc", 0.0))
    de_gene_sets = compute_de_gene_sets(
        adata,
        target_col=target_col,
        control_label=control_label,
        perturbation_genes=perturbation_genes,
        gene_names=gene_names,
        alpha=de_alpha,
        min_abs_lfc=de_min_abs_lfc,
    )

    # Train / val split on perturbation genes
    print("Creating held-out perturbation split...")
    train_genes, val_genes = train_test_split(
        perturbation_genes,
        test_size=float(cfg.get("val_fraction", 0.2)),
        random_state=int(cfg.get("seed", 42)),
        shuffle=True,
    )
    split = {
        "seed":        int(cfg.get("seed", 42)),
        "train_genes": sorted(train_genes),
        "val_genes":   sorted(val_genes),
    }
    save_json(split, split_json)

    # Save
    print(f"Saving processed dataset to {processed_npz}")
    np.savez_compressed(
        processed_npz,
        gene_names          = gene_names,
        perturbation_genes  = np.asarray(perturbation_genes),
        target_gene_indices = target_gene_indices,
        pert_means          = pert_means,
        pert_deltas         = pert_deltas,
        control_mean        = control_mean,
        gene_features       = gene_features,
        pert_cell_counts    = pert_cell_counts,
        de_gene_sets        = de_gene_sets,
        de_alpha            = np.float32(de_alpha),
        de_min_abs_lfc      = np.float32(de_min_abs_lfc),
    )

    print("Done.")
    print(f"Train genes: {len(train_genes)}")
    print(f"Val genes:   {len(val_genes)}")


if __name__ == "__main__":
    main()