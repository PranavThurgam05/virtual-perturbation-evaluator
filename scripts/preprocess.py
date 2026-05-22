import argparse
from pathlib import Path

import numpy as np
import scanpy as sc
from scipy import sparse
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from tqdm import tqdm

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

    mean = gene_by_cell.mean(axis=1, keepdims=True)
    var = gene_by_cell.var(axis=1, keepdims=True)
    dropout = (gene_by_cell <= 1e-8).mean(axis=1, keepdims=True)

    # Center each gene profile before PCA.
    centered = gene_by_cell - gene_by_cell.mean(axis=1, keepdims=True)

    # PCA over genes, using expression profile across sampled control cells.
    pca_components = max(1, n_components - 3)
    pca = PCA(n_components=pca_components, random_state=seed)
    pca_features = pca.fit_transform(centered)

    features = np.concatenate([mean, var, dropout, pca_features], axis=1)

    # Standardize features.
    features = features.astype(np.float32)
    features = (features - features.mean(axis=0, keepdims=True)) / (
        features.std(axis=0, keepdims=True) + 1e-6
    )

    assert features.shape[0] == len(gene_names)
    return features.astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/data.yaml")
    parser.add_argument("--n-components", type=int, default=128)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    set_seed(int(cfg.get("seed", 42)))

    raw_h5ad = cfg["raw_h5ad"]
    processed_npz = Path(cfg["processed_npz"])
    split_json = Path(cfg["split_json"])
    control_label = cfg.get("control_label", "non-targeting")
    target_col = cfg.get("target_col", "target_gene")

    processed_npz.parent.mkdir(parents=True, exist_ok=True)
    split_json.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading H5AD: {raw_h5ad}")
    adata = sc.read_h5ad(raw_h5ad)

    print(f"Loaded AnnData shape: {adata.shape}")
    print(f"obs columns: {list(adata.obs.columns)}")

    if target_col not in adata.obs.columns:
        raise ValueError(f"Missing target column '{target_col}' in adata.obs")

    gene_names = np.asarray(adata.var_names.astype(str))
    target_labels = np.asarray(adata.obs[target_col].astype(str))

    print("Normalizing total counts to 1e4 and applying log1p...")
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    X = adata.X
    control_mask = target_labels == control_label
    perturbation_genes = sorted([g for g in np.unique(target_labels) if g != control_label])

    print(f"Control cells: {control_mask.sum():,}")
    print(f"Perturbation genes: {len(perturbation_genes):,}")
    print(f"Genes measured: {len(gene_names):,}")

    print("Computing control mean...")
    control_mean = get_group_mean(X, control_mask)

    print("Computing perturbation means and deltas...")
    pert_means = []
    pert_deltas = []
    pert_cell_counts = []

    for gene in tqdm(perturbation_genes):
        mask = target_labels == gene
        mean = get_group_mean(X, mask)
        delta = mean - control_mean
        pert_means.append(mean)
        pert_deltas.append(delta)
        pert_cell_counts.append(int(mask.sum()))

    pert_means = np.stack(pert_means).astype(np.float32)
    pert_deltas = np.stack(pert_deltas).astype(np.float32)
    pert_cell_counts = np.asarray(pert_cell_counts, dtype=np.int64)

    print("Computing gene features from control cells...")
    gene_features = compute_gene_features(
        X[control_mask],
        gene_names=gene_names,
        n_components=args.n_components,
        seed=int(cfg.get("seed", 42)),
    )

    gene_to_idx = {g: i for i, g in enumerate(gene_names)}
    target_gene_indices = np.asarray(
        [gene_to_idx[g] if g in gene_to_idx else -1 for g in perturbation_genes],
        dtype=np.int64,
    )

    missing = [g for g, idx in zip(perturbation_genes, target_gene_indices) if idx < 0]
    if missing:
        print(f"WARNING: {len(missing)} perturbation target genes not found in var_names.")
        print(missing[:10])

    print("Creating held-out perturbation split...")
    train_genes, val_genes = train_test_split(
        perturbation_genes,
        test_size=float(cfg.get("val_fraction", 0.2)),
        random_state=int(cfg.get("seed", 42)),
        shuffle=True,
    )

    split = {
        "seed": int(cfg.get("seed", 42)),
        "train_genes": sorted(train_genes),
        "val_genes": sorted(val_genes),
    }
    save_json(split, split_json)

    print(f"Saving processed dataset to {processed_npz}")
    np.savez_compressed(
        processed_npz,
        gene_names=gene_names,
        perturbation_genes=np.asarray(perturbation_genes),
        target_gene_indices=target_gene_indices,
        pert_means=pert_means,
        pert_deltas=pert_deltas,
        control_mean=control_mean.astype(np.float32),
        gene_features=gene_features.astype(np.float32),
        pert_cell_counts=pert_cell_counts,
    )

    print("Done.")
    print(f"Train genes: {len(train_genes)}")
    print(f"Val genes: {len(val_genes)}")


if __name__ == "__main__":
    main()