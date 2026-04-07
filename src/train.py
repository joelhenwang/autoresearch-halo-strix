"""
Config-driven training loop for modular GPT.
Time-budgeted, ROCm compatible.

Usage:
    uv run -m src.train --config configs/baseline.toml
    uv run -m src.train --config configs/baseline.toml > run.log 2>&1
"""

import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import gc
import math
import time
import argparse

import torch

# Import components to trigger registration
import src.components.attention  # noqa: F401
import src.components.mlp        # noqa: F401
import src.components.norm       # noqa: F401
import src.components.position   # noqa: F401
import src.components.embedding  # noqa: F401
import src.components.head       # noqa: F401

from src.model.config import ExperimentConfig, build_model_config_from_depth
from src.model.gpt import GPT
from src.data.prepare import prepare_data
from src.data.dataloader import UnifiedMemoryDataloader, make_eval_batches
from src.eval.metrics import evaluate_bpb, compute_mfu, RDNA35_BF16_PEAK_FLOPS
from src.optim.schedules import get_lr_multiplier, get_muon_momentum, get_weight_decay


def train(config: ExperimentConfig) -> dict:
    """Run a single training experiment. Returns results dict."""
    t_start = time.time()

    # Setup
    torch.manual_seed(config.training.seed)
    torch.cuda.manual_seed(config.training.seed)
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda")

    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16}
    dtype = dtype_map.get(config.training.dtype, torch.bfloat16)
    autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=dtype)

    # Data
    data = prepare_data(
        config.data.parquet_dir,
        text_column=config.data.text_column,
        val_ratio=config.data.val_ratio,
        seed=config.training.seed,
    )
    tokenizer = data["tokenizer"]
    print(f"Vocab size: {tokenizer.get_vocab_size():,}")

    # Model
    model_config = config.model
    print(f"Model config: n_layer={model_config.n_layer}, n_embd={model_config.n_embd}, "
          f"n_head={model_config.n_head}, n_kv_head={model_config.n_kv_head}")

    with torch.device("meta"):
        model = GPT(model_config)
    model.to_empty(device=device)
    model.init_weights()

    num_params = model.num_params()
    flops_per_token = model.estimate_flops()
    print(f"Parameters: {num_params / 1e6:.1f}M")
    print(f"FLOPs per token: {flops_per_token:e}")

    # Param limit check
    if num_params > model_config.param_limit:
        msg = f"Model has {num_params / 1e6:.1f}M params, exceeds limit of {model_config.param_limit / 1e6:.0f}M"
        print(f"FAIL: {msg}")
        return {"val_bpb": 0.0, "status": "crash", "error": msg,
                "num_steps": 0, "num_params_M": num_params / 1e6}

    # Smoke test overrides
    is_smoke = config.training.smoke_test
    if is_smoke:
        time_budget_override = 30  # 30 seconds
        print(f"SMOKE TEST MODE: {time_budget_override}s budget, validation skipped")

    # Optimizer
    optimizer = model.setup_optimizer(config.optimizer)

    # Compile
    if config.training.compile:
        model = torch.compile(model, dynamic=False)

    # Dataloader
    seq_len = model_config.sequence_len
    train_loader = UnifiedMemoryDataloader(
        data["train_docs"], config.training.device_batch_size, seq_len, device,
        seed=config.training.seed,
    )
    eval_batches = make_eval_batches(
        data["val_docs"], config.training.device_batch_size, seq_len,
        config.data.eval_tokens, device,
    )

    # Gradient accumulation
    tokens_per_fwdbwd = config.training.device_batch_size * seq_len
    assert config.training.total_batch_size % tokens_per_fwdbwd == 0, \
        f"total_batch_size ({config.training.total_batch_size}) must be divisible by " \
        f"device_batch_size * seq_len ({tokens_per_fwdbwd})"
    grad_accum_steps = config.training.total_batch_size // tokens_per_fwdbwd

    time_budget = time_budget_override if is_smoke else config.training.time_budget
    print(f"Time budget: {time_budget}s")
    print(f"Gradient accumulation steps: {grad_accum_steps}")

    # Prefetch first batch
    train_iter = iter(train_loader)
    x, y, epoch = next(train_iter)

    # Training loop
    t_start_training = time.time()
    smooth_train_loss = 0
    total_training_time = 0
    step = 0

    while True:
        torch.cuda.synchronize()
        t0 = time.time()

        for micro_step in range(grad_accum_steps):
            with autocast_ctx:
                loss = model(x, y)
            train_loss = loss.detach()
            loss = loss / grad_accum_steps
            loss.backward()
            x, y, epoch = next(train_iter)

        # Progress and schedules
        progress = min(total_training_time / time_budget, 1.0)
        lrm = get_lr_multiplier(progress, config.optimizer.warmup_ratio,
                                config.optimizer.warmdown_ratio, config.optimizer.final_lr_frac)
        muon_momentum = get_muon_momentum(step)
        muon_wd = get_weight_decay(progress, config.optimizer.weight_decay)

        for group in optimizer.param_groups:
            group["lr"] = group["initial_lr"] * lrm
            if group['kind'] == 'muon':
                group["momentum"] = muon_momentum
                group["weight_decay"] = muon_wd
        optimizer.step()
        model.zero_grad(set_to_none=True)

        train_loss_f = train_loss.item()

        # Fast fail
        if math.isnan(train_loss_f) or train_loss_f > 100:
            print("\nFAIL: loss exploded or NaN")
            return {"val_bpb": 0.0, "status": "crash", "error": "loss_explosion",
                    "num_steps": step, "num_params_M": num_params / 1e6}

        torch.cuda.synchronize()
        t1 = time.time()
        dt = t1 - t0

        if step > 10:
            total_training_time += dt

        # Logging
        ema_beta = 0.9
        smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss_f
        debiased_loss = smooth_train_loss / (1 - ema_beta ** (step + 1))
        pct = 100 * progress
        tok_sec = int(config.training.total_batch_size / dt)
        mfu = compute_mfu(flops_per_token, config.training.total_batch_size, dt)
        remaining = max(0, time_budget - total_training_time)

        print(f"\rstep {step:05d} ({pct:.1f}%) | loss: {debiased_loss:.6f} | lrm: {lrm:.2f} | "
              f"dt: {dt*1000:.0f}ms | tok/sec: {tok_sec:,} | mfu: {mfu:.1f}% | "
              f"epoch: {epoch} | remaining: {remaining:.0f}s    ", end="", flush=True)

        # GC management
        if step == 0:
            gc.collect()
            gc.freeze()
            gc.disable()
        elif (step + 1) % 5000 == 0:
            gc.collect()

        step += 1

        if step > 10 and total_training_time >= time_budget:
            break

    print()  # newline after training log

    # Smoke test: just check loss didn't explode, skip full eval
    if is_smoke:
        t_end = time.time()
        peak_vram_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
        passed = step > 0 and not math.isnan(train_loss_f) and train_loss_f < 100
        status = "PASS" if passed else "FAIL"
        print("---")
        print(f"smoke_test: {status}")
        print(f"final_loss: {train_loss_f:.6f}")
        print(f"num_steps: {step}")
        print(f"peak_vram_mb: {peak_vram_mb:.1f}")
        print(f"num_params_M: {num_params / 1e6:.1f}")
        return {
            "smoke_test": status, "final_loss": train_loss_f,
            "num_steps": step, "peak_vram_mb": peak_vram_mb,
            "num_params_M": num_params / 1e6, "status": "ok" if passed else "crash",
        }

    # Full evaluation
    model.eval()
    with autocast_ctx:
        val_bpb = evaluate_bpb(model, tokenizer, eval_batches, device)

    # Final metrics
    t_end = time.time()
    peak_vram_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
    total_tokens = step * config.training.total_batch_size
    steady_mfu = (compute_mfu(flops_per_token, config.training.total_batch_size * (step - 10),
                               total_training_time)
                  if total_training_time > 0 else 0)

    results = {
        "val_bpb": val_bpb,
        "training_seconds": total_training_time,
        "total_seconds": t_end - t_start,
        "peak_vram_mb": peak_vram_mb,
        "mfu_percent": steady_mfu,
        "total_tokens_M": total_tokens / 1e6,
        "num_steps": step,
        "num_params_M": num_params / 1e6,
        "depth": config.model.n_layer,
        "status": "ok",
    }

    print("---")
    for k, v in results.items():
        if isinstance(v, float):
            print(f"{k}: {v:.6f}" if "bpb" in k else f"{k}: {v:.1f}")
        else:
            print(f"{k}: {v}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Modular GPT training")
    parser.add_argument("--config", type=str, required=True, help="Path to TOML config file")
    parser.add_argument("--smoke", action="store_true", help="Smoke test mode (30s, no eval)")
    args = parser.parse_args()

    config = ExperimentConfig.from_toml(args.config)
    if args.smoke:
        config.training.smoke_test = True
    print(f"Experiment: {config.description}")
    train(config)


if __name__ == "__main__":
    main()
