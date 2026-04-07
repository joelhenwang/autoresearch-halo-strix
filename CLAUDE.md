# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Modular autonomous GPT research system. AI agents experiment with model architecture, optimizer, and hyperparameters via swappable nn.Module components and TOML configs. Each experiment trains for a fixed 15-minute time budget.

**Hardware**: AMD Ryzen AI Max 395+ APU, RDNA 3.5 iGPU, 106GB unified RAM, ROCm 7.12.
**Metric**: `val_bpb` (validation bits per byte) — lower is better, vocab-size-independent.
**Tokenizer**: GPT-2 tiktoken (50257 vocab).

## Commands

```bash
# Setup
uv sync                            # Install dependencies (ROCm PyTorch)

# Run a training experiment (15 min)
uv run -m src.train --config configs/baseline.toml
uv run -m src.train --config configs/baseline.toml > run.log 2>&1

# Run with experiment tracking (SQLite)
uv run -m src.experiment --config configs/baseline.toml --description "baseline"

# Extract metrics
grep "^val_bpb:" run.log
grep "^peak_vram_mb:" run.log

# Monitor experiments (Rust CLI)
autostrix status                   # Running + recent experiments
autostrix best                     # Top experiments by val_bpb
autostrix stats                    # Summary statistics
autostrix history                  # Full experiment history
autostrix show <id>                # Full details for one experiment
```

No unit tests or CI. Validation is purely empirical via `val_bpb`.

## Architecture

### Modular components (`src/components/`)

Registry-based swappable nn.Modules. Each component is registered with `@register(category, name)` and built with `build(category, name, **kwargs)`.

| Category | Available | Registry location |
|----------|-----------|-------------------|
| attention | `causal`, `sliding_window` | `src/components/attention.py` |
| mlp | `relu_sq`, `swiglu`, `gelu`, `geglu` | `src/components/mlp.py` |
| norm | `rmsnorm`, `layernorm` | `src/components/norm.py` |
| position | `rope`, `alibi` | `src/components/position.py` |
| embedding | `standard`, `value_residual` | `src/components/embedding.py` |
| head | `standard`, `softcap` | `src/components/head.py` |

### Model assembly (`src/model/`)

- `config.py` — `ExperimentConfig` (TOML-serializable dataclass): ModelConfig, OptimizerConfig, TrainingConfig, DataConfig
- `block.py` — `TransformerBlock`: assembled from registry by BlockConfig
- `gpt.py` — `GPT`: blocks + residual lambdas + x0 connection + value embeddings + output head

Block pattern: `"SSSL"` expands to per-layer configs (S=sliding_window at T//2, L=full context, last layer always full). Value embeddings on alternating layers.

### Optimizer (`src/optim/`)

- `muon.py` — Muon optimizer: polar express orthogonalization + NorMuon variance reduction (torch.compiled)
- `muon_adamw.py` — `MuonAdamW`: Muon for 2D matrix params, AdamW for embeddings/scalars
- `schedules.py` — LR multiplier (warmup→constant→warmdown), momentum ramp, weight decay decay

### Data pipeline (`src/data/`)

- `tokenizer.py` — GPT-2 tiktoken wrapper + `token_bytes` lookup for BPB metric
- `prepare.py` — Load parquet directory → tokenize → cache as `.pt` at `~/.cache/autoresearch/custom/`
- `dataloader.py` — `UnifiedMemoryDataloader`: best-fit packing, all tokens in RAM (no pin_memory, unified memory optimized)

### Evaluation (`src/eval/metrics.py`)

- `evaluate_bpb()` — bits per byte, excludes special tokens
- `compute_mfu()` — model FLOPs utilization vs RDNA 3.5 peak (~29.7 TFLOPS)

### Training loop (`src/train.py`)

Config-driven: loads TOML → builds model from registry → 15-min time-budgeted training. Fast-fail on NaN/loss explosion. GC management for stable training.

### Experiment tracking (`src/experiment.py`)

SQLite `experiments.db`. Autonomous loop support. `autostrix` Rust CLI reads the DB for monitoring.

## Multi-Agent Pipeline

The system runs as a sequential pipeline of 6 specialized agents, orchestrated by `autostrix`:

| Agent | Role | Produces |
|-------|------|----------|
| Researcher | Brainstorm hypotheses, study past results | `HYPOTHESIS.md` |
| Planner | Turn hypothesis into implementation plan + TOML config | `PLAN.md`, `config.toml` |
| Engineer | Build new components (skipped if config-only) | `src/components/` code, `README.md` |
| Trainer | Smoke test (30s) → full training (15 min) | `SMOKE_RESULTS.md`, `run.log` |
| Reporter | Analyze results, compare to history | `RESULTS.md` |
| Reviewer | Evaluate agents, update instructions if needed | `REVIEW_reviewer.md` |

Each agent is a separate `claude -p --model opus` process (NO subagents). Agent instructions are in `agents/<role>/CLAUDE.md`. Handoff schemas are in `agents/schemas/`.

### Running the pipeline

```bash
# Single cycle
autostrix run-cycle --experiment my-experiment-name

# Continuous autonomous loop
autostrix run-loop --max-cycles 10

# Monitor
autostrix status
autostrix best
autostrix stats
```

### Experiment folder structure

Each cycle creates `experiments/<code-name>/` containing all handoff documents, configs, logs, and reviews.

## Smoke Test Mode

```bash
uv run -m src.train --config configs/baseline.toml --smoke
```

30-second validation: checks model compiles, loss decreases, no OOM/NaN. Prints `smoke_test: PASS` or `smoke_test: FAIL`.

## Key Constraints

- ROCm only — no CUDA-specific code, uses `F.scaled_dot_product_attention` (not FA3)
- Unified memory — no `pin_memory`, tensors allocated directly on device
- 15-minute time budget per experiment (~4 experiments/hour)
- 200M parameter limit enforced at model construction
- GPT-2 tokenizer (50257 vocab), fixed
- Python 3.10+ managed by `uv`, PyTorch ROCm wheels
- Agents must NEVER spawn subagents — single-session processes only
