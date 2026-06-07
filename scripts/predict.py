import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from vcell.data import load_processed_npz
from vcell.models.factory import build_model
from vcell.utils import ensure_parent, get_device, load_json, load_yaml


class TargetGeneDataset(Dataset):
    def __init__(self, processed, target_genes):
        self.processed = processed
        self.target_genes = [str(g) for g in target_genes]
        gene_to_idx = {str(g): i for i, g in enumerate(processed.gene_names)}
        self.target_indices = [gene_to_idx.get(g, -1) for g in self.target_genes]

        missing = [
            gene for gene, idx in zip(self.target_genes, self.target_indices) if idx < 0
        ]
        if missing:
            preview = ", ".join(missing[:10])
            raise ValueError(
                f"{len(missing)} target genes are missing from gene_names: {preview}"
            )

    def __len__(self):
        return len(self.target_genes)

    def __getitem__(self, idx):
        gene_idx = int(self.target_indices[idx])
        return {
            "target_gene_idx": torch.tensor(gene_idx, dtype=torch.long),
            "target_features": torch.tensor(
                self.processed.gene_features[gene_idx], dtype=torch.float32
            ),
        }


def move_batch(batch, device):
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}


def load_targets(args, cfg):
    target_col = args.target_col or cfg["data"].get("target_col", "target_gene")
    control_label = cfg["data"].get("control_label", "non-targeting")

    if args.targets:
        return list(
            dict.fromkeys(str(g) for g in args.targets if str(g) != control_label)
        )

    metadata_path = Path(args.metadata or cfg["data"].get("test_metadata", ""))
    if metadata_path.exists():
        metadata = pd.read_csv(metadata_path)
        if target_col not in metadata.columns:
            raise ValueError(
                f"Metadata file {metadata_path} does not contain '{target_col}'"
            )
        targets = metadata[target_col].dropna().astype(str).unique()
        return sorted(g for g in targets if g != control_label)

    split_json = Path(cfg["data"]["split_json"])
    split_data = load_json(split_json)
    split_name = args.split
    if split_name == "all":
        genes = split_data["train_genes"] + split_data["val_genes"]
    elif split_name == "train":
        genes = split_data["train_genes"]
    else:
        genes = split_data["val_genes"]

    print(
        f"No targets or metadata found; predicting the {split_name!r} split "
        f"from {split_json}."
    )
    return sorted(str(g) for g in genes if str(g) != control_label)


def load_checkpoint(model, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state = checkpoint.get("model_state", checkpoint)
    model.load_state_dict(state)
    return checkpoint


@torch.no_grad()
def predict(model, processed, target_genes, batch_size, device):
    dataset = TargetGeneDataset(processed, target_genes)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    model.eval()
    pred_deltas = []
    for batch in loader:
        batch = move_batch(batch, device)
        pred_deltas.append(model(batch).detach().cpu().numpy())

    pred_deltas = np.concatenate(pred_deltas, axis=0).astype(np.float32)
    pred_means = processed.control_mean[None, :] + pred_deltas
    return pred_deltas, pred_means.astype(np.float32)


def write_h5ad(
    path, processed, target_genes, pred_means, target_col, control_label, repeats
):
    try:
        import anndata as ad
        import pandas as pd
    except ImportError as exc:
        raise ImportError(
            "Writing H5AD predictions requires anndata and pandas. "
            "Run `uv sync --locked` first."
        ) from exc

    rows = []
    labels = []

    control = np.repeat(processed.control_mean[None, :], repeats, axis=0)
    rows.append(control)
    labels.extend([control_label] * repeats)

    for gene, mean in zip(target_genes, pred_means):
        rows.append(np.repeat(mean[None, :], repeats, axis=0))
        labels.extend([gene] * repeats)

    X = np.concatenate(rows, axis=0).astype(np.float32)
    obs = pd.DataFrame(
        {
            target_col: labels,
            "is_control": [label == control_label for label in labels],
        }
    )
    var = pd.DataFrame(index=pd.Index(processed.gene_names.astype(str), name="gene"))

    adata = ad.AnnData(X=X, obs=obs, var=var)
    path = Path(path)
    ensure_parent(path)
    adata.write_h5ad(path)


def main():
    parser = argparse.ArgumentParser(
        description="Generate perturbation predictions from a trained checkpoint."
    )
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--targets", nargs="+", default=None)
    parser.add_argument("--metadata", type=str, default=None)
    parser.add_argument("--target-col", type=str, default=None)
    parser.add_argument("--split", choices=["train", "val", "all"], default="val")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--output-npz", type=str, default=None)
    parser.add_argument("--output-h5ad", type=str, default=None)
    parser.add_argument("--h5ad-repeats", type=int, default=32)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    processed = load_processed_npz(cfg["data"]["processed_npz"])
    target_genes = load_targets(args, cfg)

    device = get_device()
    model = build_model(cfg, processed, device=device)

    checkpoint_path = Path(args.checkpoint or cfg["training"]["checkpoint_path"])
    load_checkpoint(model, checkpoint_path, device)

    batch_size = int(args.batch_size or cfg["training"].get("batch_size", 16))
    pred_deltas, pred_means = predict(
        model=model,
        processed=processed,
        target_genes=target_genes,
        batch_size=batch_size,
        device=device,
    )

    output_npz = Path(args.output_npz or cfg["data"]["prediction_output"])
    ensure_parent(output_npz)
    np.savez_compressed(
        output_npz,
        gene_names=processed.gene_names.astype(str),
        perturbation_genes=np.asarray(target_genes),
        pred_deltas=pred_deltas,
        pred_means=pred_means,
        control_mean=processed.control_mean,
        checkpoint=str(checkpoint_path),
    )
    print(f"Saved NPZ predictions -> {output_npz}")

    if args.output_h5ad:
        target_col = args.target_col or cfg["data"].get("target_col", "target_gene")
        control_label = cfg["data"].get("control_label", "non-targeting")
        write_h5ad(
            args.output_h5ad,
            processed=processed,
            target_genes=target_genes,
            pred_means=pred_means,
            target_col=target_col,
            control_label=control_label,
            repeats=int(args.h5ad_repeats),
        )
        print(f"Saved H5AD predictions -> {args.output_h5ad}")


if __name__ == "__main__":
    main()
