"""
Generate a small synthetic dataset matching the ProcessedPerturbationData
schema, so the training/eval pipeline can be exercised end-to-end without the
real Arc h5ad.

The signal is deliberately *learnable and discriminative*: each perturbation's
delta is a fixed linear function of its target gene's feature vector plus noise.
Because the model receives exactly that target-gene feature vector, a working
training loop should be able to drive pds_norm well above 0 — which is the whole
point of using this to validate the selection logic and metrics.
"""

import argparse
from pathlib import Path

import numpy as np

from vcell.utils import save_json


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-genes", type=int, default=400)
    p.add_argument("--n-perts", type=int, default=120)
    p.add_argument("--feature-dim", type=int, default=32)
    p.add_argument("--noise", type=float, default=0.3, help="delta noise std relative to signal")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-npz", type=str, default="data/processed/synthetic_dataset.npz")
    p.add_argument("--out-split", type=str, default="data/splits/synthetic_split.json")
    p.add_argument("--val-fraction", type=float, default=0.2)
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    n_genes, n_perts, fdim = args.n_genes, args.n_perts, args.feature_dim

    gene_names = np.asarray([f"G{i:05d}" for i in range(n_genes)])

    # Standardized gene features (same convention as the real pipeline).
    gene_features = rng.standard_normal((n_genes, fdim)).astype(np.float32)
    gene_features -= gene_features.mean(0, keepdims=True)
    gene_features /= gene_features.std(0, keepdims=True) + 1e-6

    # Perturbations are a random subset of genes (so target_gene_indices is valid).
    pert_idx = rng.choice(n_genes, size=n_perts, replace=False)
    pert_idx.sort()
    perturbation_genes = gene_names[pert_idx]
    target_gene_indices = pert_idx.astype(np.int64)

    # Ground-truth response operator: delta = features[target] @ W + noise.
    # This is exactly the mapping the model has access to, so it is learnable.
    W = rng.standard_normal((fdim, n_genes)).astype(np.float32) / np.sqrt(fdim)
    signal = gene_features[pert_idx] @ W  # (n_perts, n_genes)
    signal_std = signal.std()
    noise = args.noise * signal_std * rng.standard_normal(signal.shape).astype(np.float32)
    pert_deltas = (signal + noise).astype(np.float32)

    # A nonnegative control baseline; means follow from the deltas.
    control_mean = np.abs(rng.standard_normal(n_genes)).astype(np.float32) * 2.0
    pert_means = (control_mean[None, :] + pert_deltas).astype(np.float32)

    pert_cell_counts = rng.integers(50, 500, size=n_perts).astype(np.int64)

    out_npz = Path(args.out_npz)
    out_split = Path(args.out_split)
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    out_split.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        out_npz,
        gene_names=gene_names,
        perturbation_genes=perturbation_genes,
        target_gene_indices=target_gene_indices,
        pert_means=pert_means,
        pert_deltas=pert_deltas,
        control_mean=control_mean,
        gene_features=gene_features,
        pert_cell_counts=pert_cell_counts,
    )

    perm = rng.permutation(n_perts)
    n_val = max(1, int(round(args.val_fraction * n_perts)))
    val_genes = sorted(perturbation_genes[perm[:n_val]].tolist())
    train_genes = sorted(perturbation_genes[perm[n_val:]].tolist())
    save_json(
        {"seed": args.seed, "train_genes": train_genes, "val_genes": val_genes},
        out_split,
    )

    print(f"Wrote {out_npz}  (genes={n_genes}, perts={n_perts}, feat={fdim})")
    print(f"Wrote {out_split}  (train={len(train_genes)}, val={len(val_genes)})")


if __name__ == "__main__":
    main()
