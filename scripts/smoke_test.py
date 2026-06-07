import argparse
import tempfile
from pathlib import Path

import numpy as np
import torch

from vcell.data import PerturbationDeltaDataset, load_processed_npz
from vcell.metrics import evaluate_delta_predictions
from vcell.models.factory import build_model
from vcell.utils import save_json, set_seed


def make_synthetic_dataset(root):
    gene_names = np.asarray([f"GENE{i}" for i in range(16)])
    perturbation_genes = np.asarray(["GENE1", "GENE2", "GENE3", "GENE4"])
    target_gene_indices = np.asarray([1, 2, 3, 4], dtype=np.int64)
    control_mean = np.linspace(0.1, 1.6, len(gene_names)).astype(np.float32)

    rng = np.random.default_rng(42)
    pert_deltas = rng.normal(0.0, 0.05, size=(4, len(gene_names))).astype(np.float32)
    for row, idx in enumerate(target_gene_indices):
        pert_deltas[row, idx] -= 0.5
    pert_means = control_mean[None, :] + pert_deltas
    gene_features = rng.normal(0, 1, size=(len(gene_names), 8)).astype(np.float32)
    pert_cell_counts = np.asarray([10, 11, 12, 13], dtype=np.int64)

    npz_path = root / "synthetic.npz"
    split_path = root / "split.json"
    np.savez_compressed(
        npz_path,
        gene_names=gene_names,
        perturbation_genes=perturbation_genes,
        target_gene_indices=target_gene_indices,
        pert_means=pert_means,
        pert_deltas=pert_deltas,
        control_mean=control_mean,
        gene_features=gene_features,
        pert_cell_counts=pert_cell_counts,
    )
    save_json(
        {"seed": 42, "train_genes": ["GENE1", "GENE2"], "val_genes": ["GENE3", "GENE4"]},
        split_path,
    )
    return npz_path, split_path


def smoke_model(model_type, processed, batch):
    cfg = {
        "model_type": model_type,
        "model": {
            "hidden_dim": 16,
            "hidden_size": 8,
            "num_layers": 1,
            "num_heads": 2,
            "state_size": 4,
            "dropout": 0.0,
        },
    }
    model = build_model(cfg, processed, device=torch.device("cpu"))
    with torch.no_grad():
        pred = model(batch)
    assert pred.shape == batch["delta"].shape


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--include-sequence-models",
        action="store_true",
        help="Also instantiate tiny Transformer and Mamba models.",
    )
    args = parser.parse_args()

    set_seed(42)
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        npz_path, split_path = make_synthetic_dataset(root)
        processed = load_processed_npz(npz_path)

        train_ds = PerturbationDeltaDataset(processed, split_path, split="train")
        val_ds = PerturbationDeltaDataset(processed, split_path, split="val")
        assert len(train_ds) == 2
        assert len(val_ds) == 2

        batch = {
            key: torch.stack([train_ds[0][key], train_ds[1][key]])
            for key in ["target_gene_idx", "target_features", "delta", "true_mean"]
        }

        smoke_model("mlp", processed, batch)
        if args.include_sequence_models:
            smoke_model("transformer", processed, batch)
            smoke_model("mamba", processed, batch)

        metrics = evaluate_delta_predictions(
            pred_deltas=processed.pert_deltas,
            true_deltas=processed.pert_deltas,
            pred_means=processed.pert_means,
            true_means=processed.pert_means,
        )
        assert metrics["delta_mae"] == 0.0
        assert metrics["pds_top1"] == 1.0

    print("Smoke test passed.")


if __name__ == "__main__":
    main()
