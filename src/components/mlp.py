import torch
import torch.nn as nn
import torch.nn.functional as F

from src.components import register


@register("mlp", "relu_sq")
class ReLUSquaredMLP(nn.Module):
    """MLP with ReLU-squared activation (x → ReLU(x)²)."""

    def __init__(self, n_embd: int, expansion: int = 4, **kwargs):
        super().__init__()
        self.c_fc = nn.Linear(n_embd, expansion * n_embd, bias=False)
        self.c_proj = nn.Linear(expansion * n_embd, n_embd, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = F.relu(x).square()
        return self.c_proj(x)


@register("mlp", "swiglu")
class SwiGLUMLP(nn.Module):
    """MLP with SwiGLU activation (gated linear unit with SiLU)."""

    def __init__(self, n_embd: int, expansion: int = 4, **kwargs):
        super().__init__()
        # SwiGLU uses 2/3 of the standard hidden dim to keep param count comparable
        hidden = int(2 * expansion * n_embd / 3)
        hidden = ((hidden + 63) // 64) * 64  # round to multiple of 64
        self.w_gate = nn.Linear(n_embd, hidden, bias=False)
        self.w_up = nn.Linear(n_embd, hidden, bias=False)
        self.c_proj = nn.Linear(hidden, n_embd, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.c_proj(F.silu(self.w_gate(x)) * self.w_up(x))


@register("mlp", "gelu")
class GELUMLP(nn.Module):
    """Standard MLP with GELU activation."""

    def __init__(self, n_embd: int, expansion: int = 4, **kwargs):
        super().__init__()
        self.c_fc = nn.Linear(n_embd, expansion * n_embd, bias=False)
        self.c_proj = nn.Linear(expansion * n_embd, n_embd, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.c_proj(F.gelu(self.c_fc(x)))


@register("mlp", "geglu")
class GEGLUMLP(nn.Module):
    """MLP with GEGLU activation (gated linear unit with GELU)."""

    def __init__(self, n_embd: int, expansion: int = 4, **kwargs):
        super().__init__()
        hidden = int(2 * expansion * n_embd / 3)
        hidden = ((hidden + 63) // 64) * 64
        self.w_gate = nn.Linear(n_embd, hidden, bias=False)
        self.w_up = nn.Linear(n_embd, hidden, bias=False)
        self.c_proj = nn.Linear(hidden, n_embd, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.c_proj(F.gelu(self.w_gate(x)) * self.w_up(x))
