import torch
import torch.nn as nn
import torch.nn.functional as F

from src.components import register


@register("norm", "rmsnorm")
class RMSNorm(nn.Module):
    """RMS normalization (no learnable parameters, applied inline)."""

    def __init__(self, n_embd: int, **kwargs):
        super().__init__()
        self.n_embd = n_embd

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.rms_norm(x, (self.n_embd,))


@register("norm", "layernorm")
class LayerNorm(nn.Module):
    """Standard LayerNorm with learnable affine parameters."""

    def __init__(self, n_embd: int, **kwargs):
        super().__init__()
        self.ln = nn.LayerNorm(n_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ln(x)
