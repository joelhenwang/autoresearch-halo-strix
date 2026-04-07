import torch
import torch.nn as nn

from src.components import register


@register("embedding", "standard")
class TokenEmbedding(nn.Module):
    """Standard token embedding lookup."""

    def __init__(self, vocab_size: int, n_embd: int, **kwargs):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, n_embd)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        return self.embed(idx)


@register("embedding", "value_residual")
class ValueEmbedding(nn.Module):
    """Per-token value embedding for ResFormer-style value residual.

    Used on alternating layers. Produces embeddings that get mixed into
    the attention value path via a learned gate.
    """

    def __init__(self, vocab_size: int, kv_dim: int, **kwargs):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, kv_dim)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        return self.embed(idx)
