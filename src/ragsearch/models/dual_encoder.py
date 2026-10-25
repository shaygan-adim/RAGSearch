"""Text dual encoder: separate query and product towers on top of a (domain-adapted) BERT.

Parameter names (``query_encoder.bert.*``, ``*.contrastive_head.*``, ``*.classification_head.*``)
match the checkpoints produced by the original notebooks, so old ``.pth`` files still load.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel


def _mlp(in_dim: int, out_dim: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, in_dim),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(in_dim, out_dim),
    )


class EncoderWithHeads(nn.Module):
    """BERT + CLS pooling + two heads.

    * contrastive head -> L2-normalised ``embedding_dim`` vector (used for retrieval),
    * classification head -> product-category logits (auxiliary training signal only).
    """

    def __init__(self, base_model: str | Path, num_classes: int, embedding_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.bert = AutoModel.from_pretrained(str(base_model))
        hidden = self.bert.config.hidden_size
        self.contrastive_head = _mlp(hidden, embedding_dim, dropout)
        self.classification_head = _mlp(hidden, num_classes, dropout)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        cls = self.bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0, :]
        embedding = F.normalize(self.contrastive_head(cls), p=2, dim=1)
        return embedding, self.classification_head(cls)


class RetrievalSystem(nn.Module):
    """Two independent towers initialised from the same base weights."""

    def __init__(self, base_model: str | Path, num_classes: int, embedding_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.query_encoder = EncoderWithHeads(base_model, num_classes, embedding_dim, dropout)
        self.product_encoder = EncoderWithHeads(base_model, num_classes, embedding_dim, dropout)

    def forward(self, batch: dict[str, torch.Tensor]):
        """Training forward: query, positive product and negative product."""
        q_emb, q_logits = self.query_encoder(batch["q_input_ids"], batch["q_attention_mask"])
        pos_emb, pos_logits = self.product_encoder(batch["pos_input_ids"], batch["pos_attention_mask"])
        neg_emb, neg_logits = self.product_encoder(batch["neg_input_ids"], batch["neg_attention_mask"])
        return q_emb, q_logits, pos_emb, pos_logits, neg_emb, neg_logits

    def encode_query(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return self.query_encoder(input_ids, attention_mask)[0]

    def encode_product(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return self.product_encoder(input_ids, attention_mask)[0]


CLASSIFIER_WEIGHT_KEY = "product_encoder.classification_head.3.weight"


def load_retrieval_system(
    checkpoint_path: str | Path,
    base_model: str | Path,
    embedding_dim: int = 128,
    dropout: float = 0.1,
    device: str = "cpu",
) -> RetrievalSystem:
    """Rebuild the architecture and load a ``state_dict`` saved by ``ragsearch contrastive``.

    The number of category classes is read from the checkpoint itself, so it need not match
    the catalogue you are currently serving.
    """
    state = torch.load(checkpoint_path, map_location="cpu")
    if CLASSIFIER_WEIGHT_KEY not in state:
        raise KeyError(f"{checkpoint_path} is not a RetrievalSystem checkpoint (missing '{CLASSIFIER_WEIGHT_KEY}')")
    num_classes = state[CLASSIFIER_WEIGHT_KEY].shape[0]

    model = RetrievalSystem(base_model, num_classes, embedding_dim, dropout)
    model.load_state_dict(state)
    return model.to(device).eval()
