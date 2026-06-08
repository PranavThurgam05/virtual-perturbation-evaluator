from vcell.models.mamba import MambaDeltaPredictor
from vcell.models.mlp import MLPDeltaPredictor
from vcell.models.transformer import TransformerDeltaPredictor


def build_model(cfg, processed, device=None):
    """Build the configured perturbation-delta predictor."""
    model_type = cfg["model_type"]
    n_genes = len(processed.gene_names)
    feature_dim = processed.gene_features.shape[1]
    mcfg = cfg["model"]

    if model_type == "mlp":
        model = MLPDeltaPredictor(
            feature_dim=feature_dim,
            n_genes=n_genes,
            hidden_dim=int(mcfg.get("hidden_dim", 512)),
            dropout=float(mcfg.get("dropout", 0.15)),
        )
    elif model_type == "mamba":
        model = MambaDeltaPredictor(
            n_genes=n_genes,
            feature_dim=feature_dim,
            control_mean=processed.control_mean,
            hidden_size=int(mcfg.get("hidden_size", 64)),
            num_layers=int(mcfg.get("num_layers", 2)),
            state_size=int(mcfg.get("state_size", 16)),
            dropout=float(mcfg.get("dropout", 0.1)),
        )
    elif model_type == "transformer":
        model = TransformerDeltaPredictor(
            n_genes=n_genes,
            feature_dim=feature_dim,
            control_mean=processed.control_mean,
            hidden_size=int(mcfg.get("hidden_size", 128)),
            num_layers=int(mcfg.get("num_layers", 4)),
            num_heads=int(mcfg.get("num_heads", 4)),
            dropout=float(mcfg.get("dropout", 0.1)),
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type!r}")

    return model.to(device) if device is not None else model
