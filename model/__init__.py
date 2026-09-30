"""Decoder-only Transformer language model, implemented from PyTorch primitives."""

from .config import GPTConfig
from .gpt import GPT, IGNORE_INDEX, count_params

__all__ = ["GPTConfig", "GPT", "IGNORE_INDEX", "count_params"]
