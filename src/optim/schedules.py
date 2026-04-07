"""
Learning rate, momentum, and weight decay schedules.
All based on progress = training_time / time_budget (0 to 1).
"""


def get_lr_multiplier(progress: float, warmup_ratio: float = 0.0,
                      warmdown_ratio: float = 0.5, final_lr_frac: float = 0.0) -> float:
    """LR multiplier: warmup → constant → warmdown."""
    if warmup_ratio > 0 and progress < warmup_ratio:
        return progress / warmup_ratio
    elif progress < 1.0 - warmdown_ratio:
        return 1.0
    else:
        if warmdown_ratio <= 0:
            return 1.0
        cooldown = (1.0 - progress) / warmdown_ratio
        return cooldown * 1.0 + (1 - cooldown) * final_lr_frac


def get_muon_momentum(step: int, ramp_steps: int = 300,
                      start: float = 0.85, end: float = 0.95) -> float:
    """Muon momentum ramp: start → end over ramp_steps."""
    frac = min(step / ramp_steps, 1.0)
    return (1 - frac) * start + frac * end


def get_weight_decay(progress: float, base_wd: float = 0.2) -> float:
    """Weight decay linear decay: base → 0 over training."""
    return base_wd * (1 - progress)
