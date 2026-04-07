# Plan: <code-name>

## Implementation Type
<!-- One of: config-only | new-component -->

## Config
<!-- Full TOML config for this experiment. If config-only, this is the complete experiment definition. -->

```toml
# Paste or write the full config here
```

## New Component (if applicable)
<!-- Only if Implementation Type is new-component -->
- **File**: src/components/<file>.py
- **Class name**: <ClassName>
- **Registry**: @register("<category>", "<name>")
- **Interface**: forward(x, ...) → Tensor

## Training Notes
<!-- Any special considerations: batch size, LR adjustments, potential OOM -->

## Validation Criteria
<!-- What does success look like? Expected val_bpb range, what to watch for -->
