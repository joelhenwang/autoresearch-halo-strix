"""
Component registry for modular GPT building blocks.

Usage:
    from src.components import register, build

    @register("attention", "causal")
    class CausalAttention(nn.Module): ...

    attn = build("attention", "causal", n_embd=768, n_head=6, ...)
"""

from typing import Dict, Type
import torch.nn as nn

REGISTRIES: Dict[str, Dict[str, Type[nn.Module]]] = {
    "attention": {},
    "mlp": {},
    "norm": {},
    "position": {},
    "embedding": {},
    "head": {},
}


def register(category: str, name: str):
    """Decorator to register a component class."""
    def wrapper(cls):
        if category not in REGISTRIES:
            raise KeyError(f"Unknown category '{category}'. Valid: {list(REGISTRIES.keys())}")
        REGISTRIES[category][name] = cls
        return cls
    return wrapper


def build(category: str, name: str, **kwargs) -> nn.Module:
    """Instantiate a registered component by category and name."""
    registry = REGISTRIES.get(category)
    if registry is None:
        raise KeyError(f"Unknown category '{category}'. Valid: {list(REGISTRIES.keys())}")
    cls = registry.get(name)
    if cls is None:
        raise KeyError(f"Unknown {category} '{name}'. Valid: {list(registry.keys())}")
    return cls(**kwargs)


def list_components(category: str = None) -> dict:
    """List registered components, optionally filtered by category."""
    if category:
        return {category: list(REGISTRIES.get(category, {}).keys())}
    return {cat: list(reg.keys()) for cat, reg in REGISTRIES.items()}
