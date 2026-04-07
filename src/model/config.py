"""
Experiment configuration — fully describes a training run.
Serializable to/from TOML.
"""

from dataclasses import dataclass, field, asdict
from typing import Optional
import tomllib
import tomli_w


@dataclass
class BlockConfig:
    attention: str = "causal"
    mlp: str = "relu_sq"
    norm: str = "rmsnorm"
    position: str = "rope"
    window_size: int = -1          # -1 = full context
    has_value_embed: bool = False


@dataclass
class ModelConfig:
    vocab_size: int = 50257        # GPT-2 tokenizer
    sequence_len: int = 2048
    n_layer: int = 8
    n_embd: int = 768
    n_head: int = 6
    n_kv_head: int = 6
    head: str = "softcap"
    softcap_value: float = 15.0
    use_residual_lambdas: bool = True
    use_x0_connection: bool = True
    block_pattern: str = "SSSL"    # S=sliding_window(T//2), L=full. Applied cyclically.
    param_limit: int = 200_000_000 # max parameters (enforced before training)
    # If set, overrides block_pattern with explicit per-layer configs
    block_configs: Optional[list] = None


@dataclass
class OptimizerConfig:
    kind: str = "muon_adamw"
    embedding_lr: float = 0.6
    unembedding_lr: float = 0.004
    matrix_lr: float = 0.04
    scalar_lr: float = 0.5
    weight_decay: float = 0.2
    adam_betas: tuple = (0.8, 0.95)
    warmup_ratio: float = 0.0
    warmdown_ratio: float = 0.5
    final_lr_frac: float = 0.0


@dataclass
class TrainingConfig:
    time_budget: int = 900         # 15 minutes
    total_batch_size: int = 2**19  # ~524K tokens per step
    device_batch_size: int = 128
    seed: int = 42
    dtype: str = "bfloat16"
    compile: bool = True
    smoke_test: bool = False       # 30s quick validation mode


@dataclass
class DataConfig:
    parquet_dir: str = ""
    text_column: str = "text"
    val_ratio: float = 0.1
    eval_tokens: int = 40 * 524288
    preload: bool = True           # unified memory: keep all tokens in RAM


@dataclass
class ExperimentConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    description: str = ""

    def to_dict(self) -> dict:
        d = {}
        d["description"] = self.description
        d["model"] = asdict(self.model)
        d["optimizer"] = asdict(self.optimizer)
        # Convert tuple to list for TOML
        d["optimizer"]["adam_betas"] = list(d["optimizer"]["adam_betas"])
        d["training"] = asdict(self.training)
        d["data"] = asdict(self.data)
        # Strip None block_configs for cleaner TOML
        if d["model"]["block_configs"] is None:
            del d["model"]["block_configs"]
        return d

    def to_toml(self) -> str:
        return tomli_w.dumps(self.to_dict())

    def save(self, path: str):
        with open(path, "wb") as f:
            tomli_w.dump(self.to_dict(), f)

    @classmethod
    def from_toml(cls, path: str) -> "ExperimentConfig":
        with open(path, "rb") as f:
            d = tomllib.load(f)
        model = ModelConfig(**{k: v for k, v in d.get("model", {}).items()
                               if k in ModelConfig.__dataclass_fields__})
        opt_d = d.get("optimizer", {})
        if "adam_betas" in opt_d:
            opt_d["adam_betas"] = tuple(opt_d["adam_betas"])
        optimizer = OptimizerConfig(**{k: v for k, v in opt_d.items()
                                       if k in OptimizerConfig.__dataclass_fields__})
        training = TrainingConfig(**{k: v for k, v in d.get("training", {}).items()
                                     if k in TrainingConfig.__dataclass_fields__})
        data = DataConfig(**{k: v for k, v in d.get("data", {}).items()
                             if k in DataConfig.__dataclass_fields__})
        return cls(
            model=model, optimizer=optimizer, training=training, data=data,
            description=d.get("description", ""),
        )


def has_ve(layer_idx: int, n_layer: int) -> bool:
    """Returns True if layer should have Value Embedding (alternating, last always included)."""
    return layer_idx % 2 == (n_layer - 1) % 2


def expand_block_pattern(config: ModelConfig) -> list[BlockConfig]:
    """Expand block_pattern string into per-layer BlockConfig list."""
    if config.block_configs is not None:
        return [BlockConfig(**bc) if isinstance(bc, dict) else bc
                for bc in config.block_configs]

    pattern = config.block_pattern.upper()
    assert all(c in "SL" for c in pattern), f"Invalid block_pattern: {pattern}"

    short_window = config.sequence_len // 2
    blocks = []
    for i in range(config.n_layer):
        char = pattern[i % len(pattern)]
        is_last = (i == config.n_layer - 1)
        window = -1 if (char == "L" or is_last) else short_window
        attn_type = "causal" if window == -1 else "sliding_window"
        blocks.append(BlockConfig(
            attention=attn_type,
            mlp="relu_sq",
            norm="rmsnorm",
            position="rope",
            window_size=window,
            has_value_embed=has_ve(i, config.n_layer),
        ))
    return blocks


def build_model_config_from_depth(depth: int, aspect_ratio: int = 64,
                                   head_dim: int = 128, **overrides) -> ModelConfig:
    """Convenience: compute n_embd and n_head from depth and aspect ratio."""
    base_dim = depth * aspect_ratio
    model_dim = ((base_dim + head_dim - 1) // head_dim) * head_dim
    num_heads = model_dim // head_dim
    defaults = dict(
        n_layer=depth, n_embd=model_dim,
        n_head=num_heads, n_kv_head=num_heads,
    )
    defaults.update(overrides)
    return ModelConfig(**defaults)
