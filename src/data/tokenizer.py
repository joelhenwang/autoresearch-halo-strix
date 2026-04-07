"""
GPT-2 tiktoken tokenizer wrapper with byte-length lookup for BPB metric.
"""

import torch
import tiktoken


class Tokenizer:
    """Wraps tiktoken GPT-2 encoder with utilities for training."""

    def __init__(self):
        self.enc = tiktoken.get_encoding("gpt2")
        self.vocab_size = self.enc.n_vocab  # 50257
        self.eot_token = self.enc.eot_token  # <|endoftext|> = 50256
        self._token_bytes = None

    def encode(self, text: str) -> list[int]:
        return self.enc.encode(text, allowed_special={"<|endoftext|>"})

    def decode(self, tokens: list[int]) -> str:
        return self.enc.decode(tokens)

    def get_vocab_size(self) -> int:
        return self.vocab_size

    @property
    def bos_token(self) -> int:
        """Use <|endoftext|> as document separator/BOS."""
        return self.eot_token

    def get_token_bytes(self, device=None) -> torch.Tensor:
        """Return tensor of UTF-8 byte lengths per token (for BPB metric).

        Special tokens get byte_length=0 (excluded from BPB calculation).
        """
        if self._token_bytes is not None:
            if device is not None:
                return self._token_bytes.to(device)
            return self._token_bytes

        byte_lengths = []
        for i in range(self.vocab_size):
            try:
                token_bytes = self.enc.decode_single_token_bytes(i)
                byte_lengths.append(len(token_bytes))
            except Exception:
                byte_lengths.append(0)  # special tokens

        self._token_bytes = torch.tensor(byte_lengths, dtype=torch.int32)
        if device is not None:
            self._token_bytes = self._token_bytes.to(device)
        return self._token_bytes
