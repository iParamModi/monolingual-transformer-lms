"""Batching for pretraining: fixed-length windows over a flat token array.

Phase 1 wrote each split as one ``.bin`` file -- a flat ``uint16`` array of
token ids with ``</s>`` (id 3) after every document, no header. This module
turns that array into training batches.

The window order is a seeded permutation regenerated per epoch, so the pair
``(seed, micro_step)`` determines a batch exactly. Resuming a run therefore
replays the identical stream instead of an approximation of it, which is what
lets ``tests/test_resume.py`` assert that an interrupted run and an
uninterrupted one end in the same place.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import numpy as np
import torch


class WindowedBinDataset:
    """Non-overlapping windows over a ``.bin`` file of uint16 token ids.

    Window ``i`` covers ``data[i*T : i*T+T]`` as input, with targets being the
    same span shifted one position right -- the causal language modelling
    objective, where the logit at position t predicts the token at t+1.

    Windows tile the corpus without overlap, so one epoch sees each token
    exactly once. They do cross document boundaries: a window may contain the
    tail of one article, an ``</s>``, and the head of the next. That is standard
    GPT pretraining practice and teaches the model that ``</s>`` resets context.

    Args:
        bin_path: Path to the ``.bin`` file written by Phase 1.
        block_size: Window length T, which must not exceed the model's
            ``max_seq_len``.
        seed: Base seed for the per-epoch permutation.

    Attributes:
        n_windows: Number of complete windows available.
    """

    def __init__(self, bin_path: str | Path, block_size: int, seed: int = 1337):
        self.path = Path(bin_path)
        if not self.path.exists():
            raise FileNotFoundError(f"token file not found: {self.path}")
        # mode="r" keeps the array on disk; the OS pages in only the windows
        # actually touched, so a 960 MB corpus costs almost no RAM.
        self.data = np.memmap(self.path, dtype=np.uint16, mode="r")
        self.block_size = block_size
        self.seed = seed

        # Cached so the counts stay readable after close().
        self._n_tokens = len(self.data)
        # The "- 1" reserves the one extra token that the last window's targets
        # need: window i reads up to index i*T + T, inclusive.
        self.n_windows = (self._n_tokens - 1) // block_size
        if self.n_windows < 1:
            raise ValueError(
                f"{self.path.name} holds {self._n_tokens} tokens, too few for a "
                f"single window of {block_size}"
            )

    @property
    def n_tokens(self) -> int:
        """Total tokens in the file."""
        return self._n_tokens

    def close(self) -> None:
        """Release the memory map.

        Windows refuses to delete or rename a file that is still mapped, so a
        long-lived process that opens several corpora -- or a test using a
        temporary directory -- must close them explicitly. Idempotent.
        """
        if self.data is None:
            return
        mapping = getattr(self.data, "_mmap", None)
        self.data = None  # drop the array before closing the mapping under it
        if mapping is not None:
            mapping.close()

    def __enter__(self) -> "WindowedBinDataset":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def __len__(self) -> int:
        return self.n_windows

    def batch(self, indices: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        """Materialise one batch from a list of window indices.

        Args:
            indices: Window indices to gather.

        Returns:
            ``(x, y)`` int64 CPU tensors of shape ``(len(indices), block_size)``.
        """
        if self.data is None:
            raise RuntimeError(f"{self.path.name} is closed; reopen the dataset to read from it")
        T = self.block_size
        # astype() copies out of the read-only memmap into owned, writable
        # memory -- torch.from_numpy warns on read-only arrays otherwise.
        x = np.stack([self.data[i * T: i * T + T] for i in indices]).astype(np.int64)
        y = np.stack([self.data[i * T + 1: i * T + T + 1] for i in indices]).astype(np.int64)
        return torch.from_numpy(x), torch.from_numpy(y)

    def epoch_order(self, epoch: int) -> np.ndarray:
        """Window order for one epoch: a permutation seeded by ``seed + epoch``.

        Derived rather than stored, so resuming needs only the epoch number.
        """
        return np.random.default_rng(self.seed + epoch).permutation(self.n_windows)

    def micro_batches_per_epoch(self, micro_batch: int) -> int:
        """How many full micro-batches one epoch yields (the tail is dropped)."""
        return self.n_windows // micro_batch

    def stream(
        self, micro_batch: int, start_micro_step: int = 0
    ) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        """Yield batches indefinitely, resuming exactly at ``start_micro_step``.

        Epochs run back to back; each reshuffles with a new seed. The final
        partial micro-batch of an epoch is dropped so every step sees the same
        batch size.

        Args:
            micro_batch: Windows per yielded batch.
            start_micro_step: How many micro-batches have already been consumed.
                Pass ``step * grad_accum`` when resuming.

        Yields:
            ``(x, y)`` int64 CPU tensors.
        """
        per_epoch = self.micro_batches_per_epoch(micro_batch)
        if per_epoch < 1:
            raise ValueError(
                f"micro_batch {micro_batch} exceeds the {self.n_windows} windows "
                f"available in {self.path.name}"
            )
        epoch, within = divmod(start_micro_step, per_epoch)
        while True:
            order = self.epoch_order(epoch)
            for m in range(within, per_epoch):
                yield self.batch(order[m * micro_batch: (m + 1) * micro_batch])
            epoch += 1
            within = 0

    def fixed_batches(
        self, n_batches: int, batch_size: int, seed: int | None = None
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Draw a fixed, reproducible set of batches -- used for validation.

        Every validation pass must score the *same* tokens, or the loss curve
        measures sampling noise as much as learning progress.

        Args:
            n_batches: Number of batches to draw.
            batch_size: Windows per batch.
            seed: Overrides the dataset seed if given.

        Returns:
            A list of ``(x, y)`` tensor pairs, held in memory.
        """
        needed = n_batches * batch_size
        if needed > self.n_windows:
            raise ValueError(
                f"asked for {needed} validation windows, only {self.n_windows} exist"
            )
        rng = np.random.default_rng(self.seed if seed is None else seed)
        chosen = rng.permutation(self.n_windows)[:needed]
        return [
            self.batch(chosen[i * batch_size: (i + 1) * batch_size])
            for i in range(n_batches)
        ]

    def describe(self) -> str:
        """One-line summary for the training log."""
        return (
            f"{self.path.name}: {self.n_tokens:,} tokens, "
            f"{self.n_windows:,} windows of {self.block_size}"
        )
