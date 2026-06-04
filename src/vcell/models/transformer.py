import torch
import torch.nn as nn


class TransformerDeltaPredictor(nn.Module):
    """
    Transformer encoder over genes.

    Warning: full 18,080-gene self-attention can be memory-heavy.
    Use small hidden size/layers first.

    Ablation
    --------
    The four input streams can be toggled independently via the ``use_*`` flags.
    This is the knob the ablation harness (``scripts/ablation.py``) turns to ask
    "which inputs actually drive the prediction?" In particular, if disabling
    ``use_target`` and ``use_indicator`` barely changes the metrics, the model is
    ignoring the perturbation identity and collapsing toward the mean delta.
    ``use_target`` and ``use_indicator`` are the only perturbation-specific
    inputs, so at least one of them must be enabled.
    """

    def __init__(
        self,
        n_genes: int,
        feature_dim: int,
        control_mean,
        hidden_size: int = 128,
        num_layers: int = 4,
        num_heads: int = 4,
        dropout: float = 0.1,
        use_gene: bool = True,
        use_expr: bool = True,
        use_target: bool = True,
        use_indicator: bool = True,
    ):
        super().__init__()
        self.n_genes = n_genes
        self.use_gene = use_gene
        self.use_expr = use_expr
        self.use_target = use_target
        self.use_indicator = use_indicator

        if not (use_target or use_indicator):
            raise ValueError(
                "At least one perturbation-specific input (use_target or "
                "use_indicator) must be enabled, otherwise every perturbation "
                "shares identical inputs."
            )

        self.register_buffer("gene_ids", torch.arange(n_genes, dtype=torch.long))
        self.register_buffer(
            "control_mean",
            torch.tensor(control_mean, dtype=torch.float32).view(1, n_genes),
        )

        self.gene_embedding = nn.Embedding(n_genes, hidden_size)
        self.expr_proj = nn.Linear(1, hidden_size)
        self.target_proj = nn.Linear(feature_dim, hidden_size)
        self.indicator_proj = nn.Linear(1, hidden_size)

        layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, batch):
        target_features = batch["target_features"]
        target_gene_idx = batch["target_gene_idx"]

        batch_size = target_features.shape[0]
        device = target_features.device

        x = torch.zeros(batch_size, self.n_genes, self.head.in_features, device=device)

        if self.use_gene:
            gene_ids = self.gene_ids.to(device).unsqueeze(0).expand(batch_size, -1)
            x = x + self.gene_embedding(gene_ids)

        if self.use_expr:
            control_mean = self.control_mean.to(device).expand(batch_size, -1)
            x = x + self.expr_proj(control_mean.unsqueeze(-1))

        if self.use_target:
            x = x + self.target_proj(target_features).unsqueeze(1)

        if self.use_indicator:
            indicator = torch.zeros(batch_size, self.n_genes, device=device)
            indicator.scatter_(1, target_gene_idx.view(-1, 1), 1.0)
            x = x + self.indicator_proj(indicator.unsqueeze(-1))

        h = self.encoder(x)
        delta = self.head(h).squeeze(-1)
        return delta
