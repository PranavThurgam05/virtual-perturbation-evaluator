import torch
import torch.nn as nn
from transformers import MambaConfig, MambaModel


class MambaDeltaPredictor(nn.Module):
    """
    Sequence model over genes.

    For each gene position j, input embedding is:
      gene_id_embedding[j]
      + control_mean_embedding[j]
      + target_gene_feature_embedding
      + target_indicator_embedding[j]

    Output is one predicted perturbation delta per gene.
    """

    def __init__(
        self,
        n_genes: int,
        feature_dim: int,
        control_mean,
        hidden_size: int = 128,
        num_layers: int = 4,
        state_size: int = 16,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_genes = n_genes

        self.register_buffer("gene_ids", torch.arange(n_genes, dtype=torch.long))
        self.register_buffer(
            "control_mean",
            torch.tensor(control_mean, dtype=torch.float32).view(1, n_genes),
        )

        self.gene_embedding = nn.Embedding(n_genes, hidden_size)
        self.expr_proj = nn.Linear(1, hidden_size)
        self.target_proj = nn.Linear(feature_dim, hidden_size)
        self.indicator_proj = nn.Linear(1, hidden_size)
        self.dropout = nn.Dropout(dropout)

        config = MambaConfig(
            vocab_size=n_genes,
            hidden_size=hidden_size,
            num_hidden_layers=num_layers,
            state_size=state_size,
            pad_token_id=0,
        )
        self.backbone = MambaModel(config)
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, batch):
        target_features = batch["target_features"]
        target_gene_idx = batch["target_gene_idx"]

        batch_size = target_features.shape[0]
        device = target_features.device

        gene_ids = self.gene_ids.to(device).unsqueeze(0).expand(batch_size, -1)
        control_mean = self.control_mean.to(device).expand(batch_size, -1)

        gene_emb = self.gene_embedding(gene_ids)
        expr_emb = self.expr_proj(control_mean.unsqueeze(-1))

        target_emb = self.target_proj(target_features).unsqueeze(1)

        indicator = torch.zeros(batch_size, self.n_genes, device=device)
        indicator.scatter_(1, target_gene_idx.view(-1, 1), 1.0)
        indicator_emb = self.indicator_proj(indicator.unsqueeze(-1))

        x = gene_emb + expr_emb + target_emb + indicator_emb
        x = self.dropout(x)

        outputs = self.backbone(inputs_embeds=x)
        delta = self.head(outputs.last_hidden_state).squeeze(-1)
        return delta