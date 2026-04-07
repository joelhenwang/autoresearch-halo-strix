"""
TransformerBlock — assembled from registry components.
"""

import torch
import torch.nn as nn

from src.components import build
from src.model.config import BlockConfig


class TransformerBlock(nn.Module):
    """Single transformer layer with pre-norm residual connections.

    Components (attention, mlp, norm) are resolved from the registry.
    """

    def __init__(self, n_embd: int, n_head: int, n_kv_head: int, head_dim: int,
                 layer_idx: int, n_layer: int, block_config: BlockConfig,
                 max_seq_len: int = 2048):
        super().__init__()

        # Build norm modules
        self.norm1 = build("norm", block_config.norm, n_embd=n_embd)
        self.norm2 = build("norm", block_config.norm, n_embd=n_embd)

        # Build attention
        attn_kwargs = dict(
            n_embd=n_embd, n_head=n_head, n_kv_head=n_kv_head, head_dim=head_dim,
            layer_idx=layer_idx, n_layer=n_layer,
            has_value_embed=block_config.has_value_embed,
        )
        if block_config.attention == "sliding_window":
            attn_kwargs["window_size"] = block_config.window_size
            attn_kwargs["max_seq_len"] = max_seq_len
        self.attn = build("attention", block_config.attention, **attn_kwargs)

        # Build MLP
        self.mlp = build("mlp", block_config.mlp, n_embd=n_embd)

    def forward(self, x: torch.Tensor, ve: torch.Tensor | None,
                cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), ve, cos, sin)
        x = x + self.mlp(self.norm2(x))
        return x
