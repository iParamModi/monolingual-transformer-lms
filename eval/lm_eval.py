"""Intrinsic language-modelling evaluation: perplexity and bits-per-byte.

Two protocols, kept deliberately separate:

- **Packed perplexity** runs over the full held-out ``.bin`` file in
  non-overlapping windows -- the same shape of computation as training, so it
  is directly comparable to the validation numbers already in the training
  log, just on the test split instead.
- **Bits-per-byte** runs over the frozen text subset (``scripts/make_eval_subset.py``)
  with a sliding window, and normalises by UTF-8 byte count rather than token
  count. Vocabularies and fertility differ between Hindi and Nepali, so raw
  perplexity is not comparable across the two models; BPB is.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch

from model import GPT, IGNORE_INDEX


@torch.no_grad()
def packed_perplexity(
    model: GPT, bin_path: str | Path, block_size: int, device: str | torch.device,
    batch_size: int = 32,
) -> dict:
    """Cross-entropy and perplexity over a full token file, teacher-forced.

    Windows are non-overlapping, matching how the model was trained, so this
    is the direct held-out counterpart to the training-time validation loss --
    only run once, on the test split, and untouched until now.

    Args:
        model: A loaded, ``eval()``-mode GPT.
        bin_path: Path to a Phase 1 ``.bin`` file (uint16 token ids).
        block_size: Window length; should match training (512).
        device: Device to run on.
        batch_size: Windows per forward pass.

    Returns:
        Dict with ``cross_entropy``, ``perplexity``, ``tokens``.
    """
    data = np.memmap(bin_path, dtype=np.uint16, mode="r")
    n_windows = (len(data) - 1) // block_size

    total_nats, total_tokens = 0.0, 0
    for start in range(0, n_windows, batch_size):
        idx = range(start, min(start + batch_size, n_windows))
        x = np.stack([data[i * block_size: i * block_size + block_size] for i in idx]).astype(np.int64)
        y = np.stack([data[i * block_size + 1: i * block_size + block_size + 1] for i in idx]).astype(np.int64)
        x_t = torch.from_numpy(x).to(device)
        y_t = torch.from_numpy(y).to(device)
        logits, _, _ = model(x_t)
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)), y_t.reshape(-1), reduction="sum"
        )
        total_nats += loss.item()
        total_tokens += y_t.numel()

    ce = total_nats / total_tokens
    return {"cross_entropy": ce, "perplexity": math.exp(min(ce, 30.0)), "tokens": total_tokens}


@torch.no_grad()
def bits_per_byte(
    model: GPT, sp, subset_path: str | Path, device: str | torch.device,
    context: int = 512, stride: int = 256,
) -> dict:
    """Bits-per-byte on the frozen text subset -- comparable across tokenizers.

    Each document is scored with a sliding window: after the first window, only
    the *last* ``stride`` predicted tokens of every subsequent window are
    counted, so every scored token (beyond the first ``context - stride``) has
    at least ``context - stride`` tokens of real context rather than being
    scored cold at the start of an arbitrary chunk boundary.

    ``</s>`` is excluded from the nats sum: it has no corresponding UTF-8 bytes,
    so including it would inflate nats without inflating the denominator.

    Args:
        model: A loaded, ``eval()``-mode GPT.
        sp: The matching SentencePiece tokenizer.
        subset_path: Path to ``test_subset.jsonl`` (from make_eval_subset.py).
        device: Device to run on.
        context: Window length.
        stride: How far the window advances each step; only the newly-scored
            tail of each window (after the first) counts.

    Returns:
        Dict with ``bpb``, ``total_nats``, ``total_bytes``, ``total_tokens``, ``docs``.
    """
    eos_id = sp.eos_id() if sp.eos_id() >= 0 else 3
    total_nats, total_bytes, total_tokens, n_docs = 0.0, 0, 0, 0

    with open(subset_path, encoding="utf-8") as fh:
        for line in fh:
            doc = json.loads(line)
            text = doc["text"]
            ids = sp.encode(text, out_type=int) + [eos_id]
            if len(ids) < 2:
                continue
            n_docs += 1
            total_bytes += len(text.encode("utf-8"))

            pos = 0
            while pos < len(ids) - 1:
                window = ids[pos: pos + context + 1]
                if len(window) < 2:
                    break
                x = torch.tensor([window[:-1]], device=device)
                y = torch.tensor([window[1:]], device=device)
                logits, _, _ = model(x)
                nats = torch.nn.functional.cross_entropy(
                    logits.reshape(-1, logits.size(-1)), y.reshape(-1), reduction="none"
                )
                # First window: score everything. Later windows: score only the
                # newly-visible tail, so no token is double-counted across windows.
                score_from = 0 if pos == 0 else max(0, len(window) - 1 - stride)
                scored = nats[score_from:]
                # Drop the </s> position from the sum if it landed in this
                # window's scored region -- it has no byte cost.
                is_eos = [t == eos_id for t in window[1 + score_from:]]
                for nat, eos in zip(scored.tolist(), is_eos):
                    if not eos:
                        total_nats += nat
                        total_tokens += 1
                if len(window) <= context:
                    break
                pos += stride

    return {
        "bpb": (total_nats / math.log(2)) / total_bytes if total_bytes else float("nan"),
        "total_nats": total_nats,
        "total_bytes": total_bytes,
        "total_tokens": total_tokens,
        "bytes_per_token": total_bytes / total_tokens if total_tokens else float("nan"),
        "docs": n_docs,
    }


@torch.no_grad()
def score_sequence(model: GPT, ids: list[int], answer_start: int, device) -> float:
    """Sum log-likelihood (nats) the model assigns to ``ids[answer_start:]``.

    Factored out for Phase 3: multiple-choice reasoning accuracy is scored by
    comparing the likelihood the (frozen, pretrained) model assigns to each
    candidate answer, which is exactly this function applied per candidate.

    Args:
        model: A loaded, ``eval()``-mode GPT.
        ids: Full prompt+answer token ids (no BOS -- pretraining used none).
        answer_start: Index where the scored portion begins.
        device: Device to run on.

    Returns:
        Total log-likelihood (natural log) of the answer tokens, summed.
    """
    if len(ids) < 2:
        return 0.0
    x = torch.tensor([ids[:-1]], device=device)
    targets = torch.tensor([ids[1:]], device=device)
    mask = torch.full_like(targets, IGNORE_INDEX)
    # Position i in `targets` predicts ids[i+1]; answer tokens start at
    # answer_start, so their predicting positions start at answer_start - 1.
    keep_from = max(0, answer_start - 1)
    mask[:, keep_from:] = targets[:, keep_from:]
    logits, _, _ = model(x)
    loss = torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.size(-1)), mask.reshape(-1),
        ignore_index=IGNORE_INDEX, reduction="sum",
    )
    return -loss.item()


def unigram_baseline_from_counts(train_bin: str | Path, test_bin: str | Path, vocab_size: int) -> dict:
    """Cross-entropy of a unigram (frequency-only) model, as a sanity floor.

    Computed locally rather than from any pretrained resource: counts token
    frequencies directly from train.bin with a single bincount pass, then
    scores test.bin under that fixed distribution. Turns "perplexity 27" into
    "27 against a unigram floor of ~430", the difference between a number and
    evidence that the model learned something beyond word frequency.

    Args:
        train_bin: Training split, to estimate the unigram distribution from.
        test_bin: Test split, to score under that distribution.
        vocab_size: Tokenizer vocabulary size.

    Returns:
        Dict with ``cross_entropy`` and ``perplexity``.
    """
    train = np.memmap(train_bin, dtype=np.uint16, mode="r")
    counts = np.bincount(train, minlength=vocab_size).astype(np.float64)
    counts += 1.0  # add-one smoothing: an id absent from train must not get log(0)
    probs = counts / counts.sum()
    log_probs = np.log(probs)

    test = np.memmap(test_bin, dtype=np.uint16, mode="r")
    # Read in chunks; a 60M-token file is fine as one fancy-index, but chunking
    # keeps this safe on machines with less headroom.
    total_nats, total = 0.0, 0
    chunk = 5_000_000
    for start in range(0, len(test), chunk):
        piece = test[start:start + chunk]
        total_nats += -log_probs[piece].sum()
        total += len(piece)

    ce = total_nats / total
    return {"cross_entropy": ce, "perplexity": math.exp(min(ce, 30.0)), "tokens": int(total)}
