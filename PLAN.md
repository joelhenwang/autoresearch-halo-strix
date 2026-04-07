# Development Plan — Next Session (AMD Machine)

This document is a handoff for the next Claude Code session on the target AMD Ryzen AI Max 395+ machine with ROCm 7.12 + PyTorch installed.

## Current State

All code is written and passes non-GPU verification on the dev machine. What's done:

- `src/` — Full modular GPT training system (components, model, optimizer, data, eval, training loop, experiment runner)
- `agents/` — 6 agent CLAUDE.md files with instructions, schemas, no-subagent rules
- `autostrix/` — Rust CLI with orchestrator (`run-cycle`, `run-loop`) + monitoring commands
- `configs/baseline.toml` — Default experiment config (needs `parquet_dir` set)
- Smoke test mode (`--smoke` flag, 30s validation)
- 200M parameter limit enforcement
- SQLite experiment + cycle tracking

## What Needs to Be Done

### Step 1: Environment Setup

```bash
# Verify ROCm + PyTorch work
python -c "import torch; print(torch.cuda.is_available()); print(torch.version.hip)"

# Install project dependencies
uv sync

# If torch version conflicts with ROCm, you may need to adjust pyproject.toml:
# The current config points to https://download.pytorch.org/whl/rocm6.3
# If ROCm 7.12 needs a different index, update [tool.uv.sources] and [[tool.uv.index]]
```

### Step 2: Build autostrix CLI

```bash
cd autostrix && cargo build --release
# Add to PATH or create alias:
alias autostrix="./autostrix/target/release/autostrix"
```

### Step 3: Prepare Dataset

Set the `parquet_dir` in `configs/baseline.toml` to your dataset location:

```bash
# Check your dataset exists
ls datasets/gpt-training-small/*.parquet
ls datasets/babylm-strict-small/*.parquet

# Edit configs/baseline.toml:
# [data]
# parquet_dir = "datasets/gpt-training-small"
```

### Step 4: Smoke Test

```bash
uv run -m src.train --config configs/baseline.toml --smoke
# Expected: "smoke_test: PASS" within ~60 seconds
```

If smoke test fails, likely issues:
- **ROCm/PyTorch mismatch**: Check `torch.cuda.is_available()`. May need to update `pyproject.toml` torch index URL.
- **bfloat16 not supported**: Try changing `dtype = "float16"` in configs/baseline.toml
- **torch.compile fails on ROCm Triton**: Set `compile = false` in configs/baseline.toml
- **OOM**: Reduce `device_batch_size` (try 64 or 32)
- **Import errors**: Run `uv sync` again, check Python path

### Step 5: Baseline Experiment

```bash
uv run -m src.experiment --config configs/baseline.toml --description "baseline" > run.log 2>&1
grep "^val_bpb:" run.log
autostrix status
```

This establishes the val_bpb baseline for all future comparisons.

### Step 6: Install Skills

```bash
npx skills add https://github.com/tavily-ai/skills --skill search
npx skills add https://github.com/obra/superpowers --skill executing-plans
npx skills add https://github.com/obra/superpowers --skill writing-plans
npx skills add https://github.com/obra/superpowers --skill brainstorming
npx skills add https://github.com/affaan-m/everything-claude-code --skill pytorch-patterns
npx skills add https://github.com/charon-fan/agent-playbook --skill self-improving-agent
npx skills add https://github.com/huggingface/skills --skill hf-cli
npx skills add https://github.com/wshobson/agents --skill llm-evaluation
npx skills add https://github.com/supercent-io/skills-template --skill agent-evaluation
```

After installation:
1. Read each installed skill to understand what it provides
2. Check if any skill spawns subagents — if so, patch or remove that behavior
3. Update the relevant agent CLAUDE.md files to reference their skills

### Step 7: Test One Full Cycle

```bash
autostrix run-cycle --experiment test-cycle-001
```

Monitor the output. Verify:
- Each agent runs and produces its expected files
- Smoke test passes
- Full training completes
- Results are recorded in `experiments.db`
- `autostrix status` shows the experiment

### Step 8: Fix Issues

Common things that may need adjustment:

- **`claude` CLI not found**: Ensure Claude Code is installed and on PATH
- **Permission issues**: The orchestrator uses `--permission-mode acceptEdits` — verify this works
- **Agent tool permissions**: If an agent can't access a needed tool, update `allowed_tools_for_role()` in `autostrix/src/main.rs`
- **Agent max turns too low**: If agents run out of turns, increase in `max_turns_for_role()`
- **System prompt file path**: Ensure `agents/<role>/CLAUDE.md` paths resolve correctly from the repo root

### Step 9: Autonomous Operation

Once a full cycle works:

```bash
# Run continuously (stop with Ctrl+C)
autostrix run-loop

# Or with a limit
autostrix run-loop --max-cycles 20
```

## Known Limitations / TODO

### Must fix before autonomous operation:
- [ ] Verify ROCm PyTorch wheel URL in `pyproject.toml` matches ROCm 7.12
- [ ] Test `torch.compile` on ROCm — may need `compile = false` as fallback
- [ ] Calibrate `RDNA35_BF16_PEAK_FLOPS` in `src/eval/metrics.py` with actual benchmark
- [ ] Set correct `parquet_dir` in baseline config

### Nice to have:
- [ ] Add `autostrix tui` — full ratatui dashboard (currently minimal terminal output)
- [ ] Add checkpoint saving for interrupted training runs
- [ ] Add `--timeout` flag to autostrix for per-agent time limits
- [ ] Train/val split could be smarter (by document quality, not just random)
- [ ] Researcher could use web search skill for paper references
- [ ] Add a `babylm-strict-small` smoke test config separate from the main training config

### Future scaling (after progress stalls):
- [ ] Scale to ~1B params with bigger dataset
- [ ] Consider multi-GPU if available
- [ ] Add more component variants (mixture of experts, grouped query attention variants, etc.)

## File Reference

Key files to understand the system:

| File | What it does |
|------|-------------|
| `CLAUDE.md` | Instructions for Claude Code in this repo |
| `program.md` | Autonomous experiment loop instructions |
| `configs/baseline.toml` | Default experiment config |
| `src/components/__init__.py` | Component registry (register/build) |
| `src/model/config.py` | All config dataclasses + TOML serialization |
| `src/model/gpt.py` | GPT model (assembled from components) |
| `src/train.py` | Training loop + smoke test mode |
| `src/experiment.py` | Experiment runner + SQLite tracking |
| `autostrix/src/main.rs` | Rust CLI — orchestrator + monitoring |
| `agents/<role>/CLAUDE.md` | Per-agent instructions (6 agents) |
| `agents/schemas/*.md` | Handoff document templates |
