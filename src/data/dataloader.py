"""
Unified-memory optimized dataloader with best-fit packing.

Designed for systems with shared CPU/GPU memory (AMD APU with unified RAM).
Pre-loads all tokenized documents into memory for zero-transfer-cost iteration.
"""

import random
import torch


class UnifiedMemoryDataloader:
    """Best-fit packing dataloader optimized for unified memory.

    All tokens stay in RAM. Batches are allocated directly on the target device.
    No pin_memory, no async transfers — unified memory makes this unnecessary.
    """

    def __init__(self, docs: list[list[int]], batch_size: int, seq_len: int,
                 device: torch.device, seed: int = 42):
        self.docs = docs
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.device = device
        self.T = seq_len + 1  # need T+1 tokens to create (input, target) pair
        self.rng = random.Random(seed)
        self.epoch = 0

    def __iter__(self):
        return self._generate()

    def _generate(self):
        """Infinite generator yielding (inputs, targets, epoch) batches.

        Uses best-fit packing: fills rows of T+1 tokens from a document buffer.
        When no document fits the remaining space, crops the shortest one.
        Achieves ~100% token utilization with no padding.
        """
        while True:
            self.epoch += 1
            # Shuffle document order each epoch
            doc_indices = list(range(len(self.docs)))
            self.rng.shuffle(doc_indices)
            doc_iter = iter(doc_indices)

            # Buffer of available documents (sorted by length for best-fit)
            buffer = []
            buffer_target = 1000
            exhausted = False

            def refill_buffer():
                nonlocal exhausted
                while len(buffer) < buffer_target and not exhausted:
                    try:
                        idx = next(doc_iter)
                        doc = self.docs[idx]
                        if len(doc) > 0:
                            buffer.append(doc)
                    except StopIteration:
                        exhausted = True
                        break

            # Fill rows for one batch at a time
            while True:
                refill_buffer()
                if not buffer and exhausted:
                    break  # epoch done

                # Build batch_size rows of T+1 tokens each
                rows = []
                for _ in range(self.batch_size):
                    row = []
                    remaining = self.T

                    while remaining > 0:
                        refill_buffer()
                        if not buffer:
                            break

                        # Find best fit: largest doc that fits
                        best_idx = None
                        for i, doc in enumerate(buffer):
                            if len(doc) <= remaining:
                                if best_idx is None or len(doc) > len(buffer[best_idx]):
                                    best_idx = i

                        if best_idx is not None:
                            row.extend(buffer.pop(best_idx))
                            remaining = self.T - len(row)
                        else:
                            # No doc fits — crop the shortest one
                            shortest_idx = min(range(len(buffer)), key=lambda i: len(buffer[i]))
                            doc = buffer.pop(shortest_idx)
                            take = remaining
                            row.extend(doc[:take])
                            leftover = doc[take:]
                            if leftover:
                                buffer.append(leftover)
                            remaining = 0

                    if len(row) < self.T:
                        # Pad with zeros if we ran out of data (end of epoch)
                        row.extend([0] * (self.T - len(row)))

                    rows.append(row[:self.T])

                if len(rows) < self.batch_size:
                    break  # not enough data for a full batch

                # Convert to tensor on device
                batch = torch.tensor(rows, dtype=torch.long, device=self.device)
                inputs = batch[:, :-1]
                targets = batch[:, 1:]
                yield inputs, targets, self.epoch


def make_eval_batches(docs: list[list[int]], batch_size: int, seq_len: int,
                      eval_tokens: int, device: torch.device) -> list[tuple]:
    """Pre-build fixed evaluation batches (deterministic, no shuffling)."""
    T = seq_len + 1
    batches = []
    total_tokens = 0

    # Concatenate all val docs
    all_tokens = []
    for doc in docs:
        all_tokens.extend(doc)

    # Chunk into rows
    rows = []
    for start in range(0, len(all_tokens) - T + 1, T):
        rows.append(all_tokens[start:start + T])
        total_tokens += seq_len
        if total_tokens >= eval_tokens:
            break

    # Batch rows
    for i in range(0, len(rows) - batch_size + 1, batch_size):
        batch = torch.tensor(rows[i:i + batch_size], dtype=torch.long, device=device)
        batches.append((batch[:, :-1], batch[:, 1:]))

    return batches
