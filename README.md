# autoresearch-halo-strix

A modular, autonomous AI research lab for GPT model training. A sequential pipeline of 6 specialized AI agents experiments with architecture, optimizer, and hyperparameters — overnight, while you sleep.

Forked from [@karpathy's autoresearch](https://github.com/karpathy/autoresearch), rebuilt from the ground up for modularity, AMD ROCm, and multi-agent orchestration.

## What changed from the original

| | Original | This fork |
|---|---|---|
| **Hardware** | NVIDIA H100 (CUDA) | AMD Ryzen AI Max 395+ APU (ROCm 7.12) |
| **Memory** | Discrete GPU VRAM | 106GB unified CPU/GPU RAM |
| **Time budget** | 5 minutes | 15 minutes |
| **Model code** | Single monolithic `train.py` | Modular `src/` package with swappable nn.Modules |
| **Experiments** | Edit `train.py` directly | TOML config files in `configs/` |
| **Agents** | 1 agent edits code | 6 specialized agents (Researcher → Planner → Engineer → Trainer → Reporter → Reviewer) |
| **Tracking** | Manual `results.tsv` | SQLite `experiments.db` + `autostrix` Rust CLI |
| **Tokenizer** | Custom BPE (8192 vocab) | GPT-2 tiktoken (50257 vocab) |
| **Attention** | Flash Attention 3 (NVIDIA) | `F.scaled_dot_product_attention` (ROCm compatible) |

## How it works

### The AI Lab Pipeline

```
Researcher → Planner → Engineer → Trainer → Reporter → Reviewer → loop
```

Each agent runs as a **separate `claude` CLI process** (no subagents):

1. **Researcher** — Studies experiment history, checks human-provided ideas and suggestions, brainstorms hypotheses, writes `HYPOTHESIS.md`
2. **Planner** — Turns the hypothesis into a concrete implementation plan and TOML config
3. **Engineer** — Builds new nn.Module components if needed (skipped for config-only experiments)
4. **Trainer** — Runs a 30-second smoke test, then full 15-minute training
5. **Reporter** — Analyzes results, compares to history, writes recommendations
6. **Reviewer** — Evaluates all agents' performance, updates their instructions if needed

The `autostrix` Rust CLI orchestrates the pipeline and provides experiment monitoring.

### Modular Components

Architecture pieces are swappable like Legos via a registry system:

| Category | Options |
|----------|---------|
| Attention | `causal`, `sliding_window` |
| MLP | `relu_sq`, `swiglu`, `gelu`, `geglu` |
| Normalization | `rmsnorm`, `layernorm` |
| Position | `rope`, `alibi` |
| Embedding | `standard`, `value_residual` (ResFormer) |
| Output head | `standard`, `softcap` |

Each experiment is a TOML config — no code changes needed for most experiments:

```toml
[model]
n_layer = 8
n_embd = 768
block_pattern = "SSSL"
head = "softcap"

[optimizer]
kind = "muon_adamw"
matrix_lr = 0.04
```

## Quick start

**Requirements:** AMD APU or GPU with ROCm 7.12, Python 3.10+, [uv](https://docs.astral.sh/uv/), Rust toolchain.

```bash
# 1. Install dependencies
uv sync

# 2. Build the CLI
cd autostrix && cargo build --release
cd ..

# 3. Edit the baseline config — set your parquet dataset path
#    Edit configs/baseline.toml → parquet_dir = "path/to/your/parquets"

# 4. Run a smoke test (30 seconds)
uv run -m src.train --config configs/baseline.toml --smoke

# 5. Run a single training experiment (15 min)
uv run -m src.experiment --config configs/baseline.toml --description "baseline"

# 6. Check results
./autostrix/target/release/autostrix status
```

### Running the autonomous lab

```bash
# Single cycle (all 6 agents)
autostrix run-cycle --experiment my-experiment-name

# Continuous autonomous loop
autostrix run-loop --max-cycles 10

# Monitor
autostrix status          # Current + recent experiments
autostrix best            # Top experiments by val_bpb
autostrix stats           # Summary statistics
autostrix history         # Full experiment table

# Steer the research while it runs
autostrix idea "Differential Transformer - uses difference of two attention patterns" -t paper
autostrix suggest "Try 12 layers with SwiGLU and lower matrix_lr=0.02" -p 20
autostrix ideas           # See saved ideas
autostrix suggestions     # See queued suggestions
```

### Steering the research

The lab runs autonomously, but you can guide it at any time without stopping the loop:

- **`autostrix idea <text> [-t tag]`** — Save a paper, URL, concept, or observation to the ideas bank. The Researcher reads all ideas before brainstorming. Use `-t` to tag by category (e.g., `paper`, `architecture`, `optimizer`).
- **`autostrix suggest <text> [-p priority]`** — Queue a specific experiment direction. The Researcher prioritizes pending suggestions over its own ideas. Higher priority (default 10) gets tried sooner.

Both are stored in SQLite (`experiments.db`) and persist across cycles. You can seed ideas before bed and wake up to experiments informed by your input.

## Project structure

```
src/
  components/           # Swappable nn.Modules (attention, MLP, norm, position, etc.)
  model/                # Config system, TransformerBlock, GPT assembly
  optim/                # MuonAdamW optimizer + schedules
  data/                 # GPT-2 tokenizer, parquet loader, unified-memory dataloader
  eval/                 # evaluate_bpb metric, MFU calculation
  train.py              # Config-driven training loop
  experiment.py         # Experiment runner with SQLite tracking

agents/                 # Per-agent CLAUDE.md instructions
  researcher/           # Hypothesis generation
  planner/              # Implementation planning
  engineer/             # Component building
  trainer/              # Training execution
  reporter/             # Results analysis
  reviewer/             # Agent evaluation + self-improvement
  schemas/              # Handoff document templates

autostrix/              # Rust CLI — orchestrator + monitoring
configs/                # TOML experiment configs
experiments/            # Per-cycle artifacts (HYPOTHESIS.md, PLAN.md, run.log, etc.)
```

## Design choices

- **Modular components.** Architecture pieces are registered nn.Modules. Swap attention, MLP, normalization, or position encoding by changing a string in the config.
- **Config-driven experiments.** Each experiment is a TOML file, not a code change. This makes experiments reproducible, diffable, and git-trackable.
- **Unified memory optimization.** The AMD APU shares 106GB between CPU and GPU. The entire dataset (~1.2GB tokenized) lives in RAM with zero transfer overhead.
- **Multi-agent specialization.** Rather than one agent doing everything, six agents each focus on their strength: research, planning, engineering, training, reporting, reviewing.
- **Self-improving system.** The Reviewer agent can update other agents' instructions based on performance patterns, creating a feedback loop that improves the lab over time.
- **200M parameter limit.** Enforced at model construction. Keeps experiments within the compute budget of the 15-minute time window.

## Metric

**val_bpb** (validation bits per byte) — lower is better. Computed as cross-entropy in nats normalized by byte counts. Vocab-size-independent, so architectural changes that affect tokenization are fairly compared.

## License

MIT
