from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset

from vcell.utils import load_json


@dataclass
class ProcessedPerturbationData:
    gene_names: np.ndarray
    perturbation_genes: np.ndarray
    target_gene_indices: np.ndarray
    pert_means: np.ndarray
    pert_deltas: np.ndarray
    control_mean: np.ndarray
    gene_features: np.ndarray
    pert_cell_counts: np.ndarray


def load_processed_npz(path: str) -> ProcessedPerturbationData:
    data = np.load(path, allow_pickle=True)
    return ProcessedPerturbationData(
        gene_names=data["gene_names"],
        perturbation_genes=data["perturbation_genes"],
        target_gene_indices=data["target_gene_indices"],
        pert_means=data["pert_means"].astype(np.float32),
        pert_deltas=data["pert_deltas"].astype(np.float32),
        control_mean=data["control_mean"].astype(np.float32),
        gene_features=data["gene_features"].astype(np.float32),
        pert_cell_counts=data["pert_cell_counts"],
    )


class PerturbationDeltaDataset(Dataset):
    def __init__(self, processed: ProcessedPerturbationData, split_json: str, split: str):
        self.processed = processed
        split_data = load_json(split_json)

        if split == "train":
            genes = set(split_data["train_genes"])
        elif split in {"val", "eval"}:
            genes = set(split_data["val_genes"])
        else:
            raise ValueError(f"Unknown split: {split}")

        self.indices = [
            i for i, g in enumerate(processed.perturbation_genes)
            if str(g) in genes and processed.target_gene_indices[i] >= 0
        ]

        if len(self.indices) == 0:
            raise ValueError(f"No samples found for split={split}")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        target_gene_idx = int(self.processed.target_gene_indices[real_idx])

        target_features = self.processed.gene_features[target_gene_idx]
        delta = self.processed.pert_deltas[real_idx]
        true_mean = self.processed.pert_means[real_idx]

        return {
            "sample_index": torch.tensor(real_idx, dtype=torch.long),
            "target_gene_idx": torch.tensor(target_gene_idx, dtype=torch.long),
            "target_features": torch.tensor(target_features, dtype=torch.float32),
            "delta": torch.tensor(delta, dtype=torch.float32),
            "true_mean": torch.tensor(true_mean, dtype=torch.float32),
        }