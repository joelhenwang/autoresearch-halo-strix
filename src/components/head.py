import torch
import torch.nn as nn
import torch.nn.functional as F

from src.components import register


@register("head", "standard")
class StandardHead(nn.Module):
    """Standard linear output head (logits projection)."""

    def __init__(self, n_embd: int, vocab_size: int, **kwargs):
        super().__init__()
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lm_head(x)


@register("head", "softcap")
class SoftcapHead(nn.Module):
    """Output head with tanh softcap to prevent logit explosion."""

    def __init__(self, n_embd: int, vocab_size: int, softcap: float = 15.0, **kwargs):
        super().__init__()
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        self.softcap = softcap

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.lm_head(x).float()
        return self.softcap * torch.tanh(logits / self.softcap)
