"""
Evaluation metrics: BPB (bits per byte) and MFU.
"""

import math
import torch
import torch.nn.functional as F

from src.data.tokenizer import Tokenizer

# RDNA 3.5 iGPU peak (40 CUs × 256 FP16 ops/CU/cycle × 2.9 GHz)
# This is approximate — benchmark on actual hardware to refine.
RDNA35_BF16_PEAK_FLOPS = 29.7e12


@torch.no_grad()
def evaluate_bpb(model, tokenizer: Tokenizer, eval_batches: list,
                 device: torch.device) -> float:
    """Compute validation bits-per-byte (BPB).

    BPB = total_nats / (ln(2) * total_bytes)

    Lower is better. Vocab-size independent.
    """
    model.eval()
    token_bytes = tokenizer.get_token_bytes(device=device)
    total_nats = 0.0
    total_bytes = 0

    for inputs, targets in eval_batches:
        logits = model(inputs)
        # Per-token cross entropy (nats)
        loss_per_token = F.cross_entropy(
            logits.view(-1, logits.size(-1)), targets.reshape(-1),
            ignore_index=-1, reduction='none',
        )
        # Byte counts per token
        byte_counts = token_bytes[targets.reshape(-1)]
        # Mask out special tokens (byte_length=0)
        mask = byte_counts > 0
        total_nats += (loss_per_token * mask).sum().item()
        total_bytes += byte_counts.sum().item()

    model.train()

    if total_bytes == 0:
        return float("inf")
    return total_nats / (math.log(2) * total_bytes)


def compute_mfu(flops_per_token: int, batch_size: int, dt: float,
                peak_flops: float = RDNA35_BF16_PEAK_FLOPS) -> float:
    """Compute Model FLOPs Utilization (MFU) as a percentage."""
    if dt <= 0:
        return 0.0
    return 100 * flops_per_token * batch_size / dt / peak_flops
