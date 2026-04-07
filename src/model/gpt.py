"""
GPT model — assembled from modular components via config.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.components import build
from src.model.config import ModelConfig, BlockConfig, expand_block_pattern, has_ve
from src.model.block import TransformerBlock


def _norm(x: torch.Tensor) -> torch.Tensor:
    return F.rms_norm(x, (x.size(-1),))


class GPT(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.block_configs = expand_block_pattern(config)
        head_dim = config.n_embd // config.n_head

        # Token embedding
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                n_embd=config.n_embd, n_head=config.n_head,
                n_kv_head=config.n_kv_head, head_dim=head_dim,
                layer_idx=i, n_layer=config.n_layer,
                block_config=self.block_configs[i],
                max_seq_len=config.sequence_len,
            )
            for i in range(config.n_layer)
        ])

        # Output head
        head_kwargs = dict(n_embd=config.n_embd, vocab_size=config.vocab_size)
        if config.head == "softcap":
            head_kwargs["softcap"] = config.softcap_value
        self.head = build("head", config.head, **head_kwargs)

        # Per-layer residual scaling
        if config.use_residual_lambdas:
            self.resid_lambdas = nn.Parameter(torch.ones(config.n_layer))
        else:
            self.resid_lambdas = None

        if config.use_x0_connection:
            self.x0_lambdas = nn.Parameter(torch.zeros(config.n_layer))
        else:
            self.x0_lambdas = None

        # Value embeddings (on alternating layers)
        kv_dim = config.n_kv_head * head_dim
        self.value_embeds = nn.ModuleDict({
            str(i): nn.Embedding(config.vocab_size, kv_dim)
            for i in range(config.n_layer) if has_ve(i, config.n_layer)
        })

        # Rotary embeddings
        pos_type = self.block_configs[0].position
        if pos_type == "rope":
            rope = build("position", "rope", head_dim=head_dim, max_seq_len=config.sequence_len)
            self.register_buffer("rope_cos", rope.cos, persistent=False)
            self.register_buffer("rope_sin", rope.sin, persistent=False)
        elif pos_type == "alibi":
            alibi = build("position", "alibi", n_head=config.n_head, max_seq_len=config.sequence_len)
            self.register_buffer("alibi_bias", alibi.bias, persistent=False)
        self.pos_type = pos_type

    @torch.no_grad()
    def init_weights(self):
        n_embd = self.config.n_embd
        s = 3**0.5 * n_embd**-0.5

        # Embeddings
        nn.init.normal_(self.wte.weight, mean=0.0, std=1.0)

        # Output head
        if hasattr(self.head, "lm_head"):
            nn.init.normal_(self.head.lm_head.weight, mean=0.0, std=0.001)

        # Transformer blocks
        for block in self.blocks:
            attn = block.attn
            nn.init.uniform_(attn.c_q.weight, -s, s)
            nn.init.uniform_(attn.c_k.weight, -s, s)
            nn.init.uniform_(attn.c_v.weight, -s, s)
            nn.init.zeros_(attn.c_proj.weight)
            if attn.ve_gate is not None:
                nn.init.zeros_(attn.ve_gate.weight)

            mlp = block.mlp
            # Init first linear (gate/up) with Xavier-style, projection with zeros
            for name, param in mlp.named_parameters():
                if "c_proj" in name or (name == "c_proj.weight"):
                    nn.init.zeros_(param)
                elif "weight" in name:
                    nn.init.uniform_(param, -s, s)

        # Per-layer scalars
        if self.resid_lambdas is not None:
            self.resid_lambdas.fill_(1.0)
        if self.x0_lambdas is not None:
            self.x0_lambdas.fill_(0.1)

        # Value embeddings
        for ve in self.value_embeds.values():
            nn.init.uniform_(ve.weight, -s, s)

        # Cast embeddings to bf16
        self.wte.to(dtype=torch.bfloat16)
        for ve in self.value_embeds.values():
            ve.to(dtype=torch.bfloat16)

    def estimate_flops(self) -> int:
        """Estimated FLOPs per token (forward + backward = 6x param count + attention)."""
        nparams = sum(p.numel() for p in self.parameters())
        # Exclude non-matrix params from FLOPs count
        exclude = self.wte.weight.numel()
        exclude += sum(ve.weight.numel() for ve in self.value_embeds.values())
        if self.resid_lambdas is not None:
            exclude += self.resid_lambdas.numel()
        if self.x0_lambdas is not None:
            exclude += self.x0_lambdas.numel()

        head_dim = self.config.n_embd // self.config.n_head
        t = self.config.sequence_len
        attn_flops = 0
        for bc in self.block_configs:
            window = t if bc.window_size < 0 else min(bc.window_size, t)
            attn_flops += 12 * self.config.n_head * head_dim * window

        return 6 * (nparams - exclude) + attn_flops

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def setup_optimizer(self, opt_config):
        """Build parameter groups for MuonAdamW from OptimizerConfig."""
        from src.optim.muon_adamw import MuonAdamW

        model_dim = self.config.n_embd
        dmodel_lr_scale = (model_dim / 768) ** -0.5

        matrix_params = list(self.blocks.parameters())
        ve_params = list(self.value_embeds.parameters())
        embed_params = list(self.wte.parameters())
        head_params = [p for p in self.head.parameters()]
        resid_params = [self.resid_lambdas] if self.resid_lambdas is not None else []
        x0_params = [self.x0_lambdas] if self.x0_lambdas is not None else []

        betas = tuple(opt_config.adam_betas)

        param_groups = [
            dict(kind='adamw', params=head_params,
                 lr=opt_config.unembedding_lr * dmodel_lr_scale,
                 betas=betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=embed_params,
                 lr=opt_config.embedding_lr * dmodel_lr_scale,
                 betas=betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=ve_params,
                 lr=opt_config.embedding_lr * dmodel_lr_scale,
                 betas=betas, eps=1e-10, weight_decay=0.0),
        ]

        if resid_params:
            param_groups.append(dict(
                kind='adamw', params=resid_params,
                lr=opt_config.scalar_lr * 0.01, betas=betas, eps=1e-10, weight_decay=0.0,
            ))
        if x0_params:
            param_groups.append(dict(
                kind='adamw', params=x0_params,
                lr=opt_config.scalar_lr, betas=(0.96, 0.95), eps=1e-10, weight_decay=0.0,
            ))

        # Group matrix params by shape for Muon
        for shape in sorted({p.shape for p in matrix_params}):
            group = [p for p in matrix_params if p.shape == shape]
            param_groups.append(dict(
                kind='muon', params=group, lr=opt_config.matrix_lr,
                momentum=0.95, ns_steps=5, beta2=0.95,
                weight_decay=opt_config.weight_decay,
            ))

        optimizer = MuonAdamW(param_groups)
        for group in optimizer.param_groups:
            group["initial_lr"] = group["lr"]
        return optimizer

    def forward(self, idx: torch.Tensor, targets: torch.Tensor = None,
                reduction: str = "mean"):
        B, T = idx.size()

        # Position embeddings
        if self.pos_type == "rope":
            cos = self.rope_cos[:, :T]
            sin = self.rope_sin[:, :T]
        else:
            cos = sin = None  # ALiBi doesn't use cos/sin

        x = self.wte(idx)
        x = _norm(x)
        x0 = x

        for i, block in enumerate(self.blocks):
            # Residual scaling
            if self.resid_lambdas is not None:
                x = self.resid_lambdas[i] * x
            if self.x0_lambdas is not None:
                x = x + self.x0_lambdas[i] * x0

            # Value embedding for this layer
            ve = self.value_embeds[str(i)](idx) if str(i) in self.value_embeds else None
            x = block(x, ve, cos, sin)

        x = _norm(x)
        logits = self.head(x)

        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1),
                ignore_index=-1, reduction=reduction,
            )
            return loss
        return logits
