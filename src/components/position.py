import math
import torch
import torch.nn as nn

from src.components import register


@register("position", "rope")
class RotaryEmbedding(nn.Module):
    """Precomputed rotary position embeddings (RoPE)."""

    def __init__(self, head_dim: int, max_seq_len: int = 2048, base: float = 10000.0, **kwargs):
        super().__init__()
        self.head_dim = head_dim
        rope_len = max_seq_len * 10  # precompute extra for safety
        cos, sin = self._precompute(rope_len, head_dim, base)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    @staticmethod
    def _precompute(seq_len: int, head_dim: int, base: float):
        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))
        t = torch.arange(seq_len, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        cos, sin = freqs.cos().bfloat16(), freqs.sin().bfloat16()
        return cos[None, :, None, :], sin[None, :, None, :]

    def get_cos_sin(self, seq_len: int):
        return self.cos[:, :seq_len], self.sin[:, :seq_len]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Not used directly — attention modules call get_cos_sin
        return x


def apply_rotary_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply rotary embeddings to a 4D tensor [B, T, H, D]."""
    assert x.ndim == 4
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3)


@register("position", "alibi")
class ALiBi(nn.Module):
    """Attention with Linear Biases — no learned parameters.

    Generates per-head linear bias slopes added to attention scores.
    """

    def __init__(self, n_head: int, max_seq_len: int = 2048, **kwargs):
        super().__init__()
        self.n_head = n_head
        slopes = self._get_slopes(n_head)
        # Precompute bias matrix [1, n_head, max_seq_len, max_seq_len]
        positions = torch.arange(max_seq_len)
        rel_pos = positions[None, :] - positions[:, None]  # [T, T]
        rel_pos = rel_pos.float().clamp(max=0)  # causal: only negative or zero
        bias = slopes[:, None, None] * rel_pos[None, :, :]  # [H, T, T]
        self.register_buffer("bias", bias.unsqueeze(0), persistent=False)  # [1, H, T, T]

    @staticmethod
    def _get_slopes(n_head: int):
        """Get per-head slopes following the ALiBi paper."""
        def _get_slopes_power_of_2(n):
            start = 2 ** (-(2 ** -(math.log2(n) - 3)))
            ratio = start
            return [start * ratio ** i for i in range(n)]

        if math.log2(n_head).is_integer():
            return torch.tensor(_get_slopes_power_of_2(n_head))
        else:
            closest_power_of_2 = 2 ** math.floor(math.log2(n_head))
            slopes = _get_slopes_power_of_2(closest_power_of_2)
            extra = _get_slopes_power_of_2(2 * closest_power_of_2)
            slopes.extend(extra[0::2][:n_head - closest_power_of_2])
            return torch.tensor(slopes)

    def get_bias(self, seq_len: int):
        return self.bias[:, :, :seq_len, :seq_len]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x
