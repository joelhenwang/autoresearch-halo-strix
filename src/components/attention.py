import torch
import torch.nn as nn
import torch.nn.functional as F

from src.components import register
from src.components.position import apply_rotary_emb


def _norm(x: torch.Tensor) -> torch.Tensor:
    """Inline RMS norm for Q/K normalization."""
    return F.rms_norm(x, (x.size(-1),))


def _repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Repeat KV heads to match Q head count for SDPA (GQA support)."""
    if n_rep == 1:
        return x
    B, T, H, D = x.shape
    return x[:, :, :, None, :].expand(B, T, H, n_rep, D).reshape(B, T, H * n_rep, D)


@register("attention", "causal")
class CausalAttention(nn.Module):
    """Standard causal self-attention with RoPE, QK-norm, and optional value embeddings.

    Uses F.scaled_dot_product_attention (ROCm compatible).
    """

    def __init__(self, n_embd: int, n_head: int, n_kv_head: int, head_dim: int,
                 layer_idx: int, n_layer: int, has_value_embed: bool = False, **kwargs):
        super().__init__()
        self.n_head = n_head
        self.n_kv_head = n_kv_head
        self.head_dim = head_dim
        self.n_rep = n_head // n_kv_head

        self.c_q = nn.Linear(n_embd, n_head * head_dim, bias=False)
        self.c_k = nn.Linear(n_embd, n_kv_head * head_dim, bias=False)
        self.c_v = nn.Linear(n_embd, n_kv_head * head_dim, bias=False)
        self.c_proj = nn.Linear(n_head * head_dim, n_embd, bias=False)

        # Value embedding gate (ResFormer-style)
        self.ve_gate_channels = 32
        self.ve_gate = nn.Linear(self.ve_gate_channels, n_kv_head, bias=False) if has_value_embed else None

    def forward(self, x: torch.Tensor, ve: torch.Tensor | None,
                cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)

        # Value residual: mix in value embedding with input-dependent gate per head
        if ve is not None and self.ve_gate is not None:
            ve = ve.view(B, T, self.n_kv_head, self.head_dim)
            gate = 2 * torch.sigmoid(self.ve_gate(x[..., :self.ve_gate_channels]))
            v = v + gate.unsqueeze(-1) * ve

        # RoPE + QK-norm
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q, k = _norm(q), _norm(k)

        # Expand KV for GQA
        k = _repeat_kv(k, self.n_rep)
        v = _repeat_kv(v, self.n_rep)

        # SDPA expects [B, H, T, D]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, -1)
        return self.c_proj(y)


@register("attention", "sliding_window")
class SlidingWindowAttention(nn.Module):
    """Causal attention with sliding window via precomputed mask.

    Uses F.scaled_dot_product_attention with a band mask (ROCm compatible).
    """

    def __init__(self, n_embd: int, n_head: int, n_kv_head: int, head_dim: int,
                 layer_idx: int, n_layer: int, window_size: int = 1024,
                 max_seq_len: int = 2048, has_value_embed: bool = False, **kwargs):
        super().__init__()
        self.n_head = n_head
        self.n_kv_head = n_kv_head
        self.head_dim = head_dim
        self.n_rep = n_head // n_kv_head
        self.window_size = window_size

        self.c_q = nn.Linear(n_embd, n_head * head_dim, bias=False)
        self.c_k = nn.Linear(n_embd, n_kv_head * head_dim, bias=False)
        self.c_v = nn.Linear(n_embd, n_kv_head * head_dim, bias=False)
        self.c_proj = nn.Linear(n_head * head_dim, n_embd, bias=False)

        self.ve_gate_channels = 32
        self.ve_gate = nn.Linear(self.ve_gate_channels, n_kv_head, bias=False) if has_value_embed else None

        # Precompute sliding window mask [1, 1, T, T]
        mask = torch.ones(max_seq_len, max_seq_len, dtype=torch.bool)
        mask = torch.tril(mask)  # causal
        mask = mask & torch.triu(torch.ones_like(mask), diagonal=-(window_size - 1))
        # Convert to float mask: 0 for attend, -inf for blocked
        float_mask = torch.zeros(max_seq_len, max_seq_len)
        float_mask.masked_fill_(~mask, float("-inf"))
        self.register_buffer("attn_mask", float_mask[None, None, :, :], persistent=False)

    def forward(self, x: torch.Tensor, ve: torch.Tensor | None,
                cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)

        if ve is not None and self.ve_gate is not None:
            ve = ve.view(B, T, self.n_kv_head, self.head_dim)
            gate = 2 * torch.sigmoid(self.ve_gate(x[..., :self.ve_gate_channels]))
            v = v + gate.unsqueeze(-1) * ve

        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q, k = _norm(q), _norm(k)

        k = _repeat_kv(k, self.n_rep)
        v = _repeat_kv(v, self.n_rep)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        mask = self.attn_mask[:, :, :T, :T]
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        y = y.transpose(1, 2).contiguous().view(B, T, -1)
        return self.c_proj(y)
