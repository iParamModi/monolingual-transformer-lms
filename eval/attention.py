"""Attention analysis: capture, entropy, mean distance, head specialisation.

Because the attention is hand-written (model/gpt.py), the weights are directly
inspectable via ``return_attn=True``. This module turns those weights into the
statistics and heatmap-ready arrays the brief asks for: per-head entropy, mean
attention distance, and enough structure to identify local/positional vs
content-based vs sink heads.

Functions take ``(model, ids)`` and return plain data -- no plotting here, and
no CLI-only logic, so Phase 3 can call these on a finetuned checkpoint and diff
against a pretrained run without any of this changing.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import torch


def first_sentence(subset_path: str | Path, min_words: int = 8, max_words: int = 22) -> str:
    """Pick one short sentence from the frozen subset, for the heatmap grid.

    Lives here rather than in run_eval.py so that figure-only scripts can
    import it without pulling in eval.metrics (and therefore sacrebleu).

    Documents in the subset are all >= 128 words by construction, so a short
    document does not exist -- split on Devanagari sentence terminators
    (danda, double danda, ?, !) and take the first sentence of a workable
    length. Falls back to truncating the first document if nothing fits.

    Args:
        subset_path: ``test_subset.jsonl``.
        min_words: Shortest acceptable sentence.
        max_words: Longest acceptable sentence (heatmap axes get unreadable
            much beyond this).

    Returns:
        A single sentence of text.
    """
    with open(subset_path, encoding="utf-8") as fh:
        for line in fh:
            text = json.loads(line)["text"]
            for sentence in re.split(r"[।॥?!]", text):
                sentence = sentence.strip()
                if min_words <= len(sentence.split()) <= max_words:
                    return sentence
        # Nothing matched: fall back to the head of the first document.
        fh.seek(0)
        first = json.loads(fh.readline())["text"]
        return " ".join(first.split()[:max_words])


@torch.no_grad()
def capture_attention(model, ids: torch.Tensor) -> list[np.ndarray]:
    """Run the model once and return every layer's attention weights.

    Args:
        model: A loaded, ``eval()``-mode GPT.
        ids: Token ids, ``(1, T)`` -- single sequence, since heatmaps and the
            per-head statistics below are computed one sequence at a time.

    Returns:
        List of length ``n_layers``, each ``(n_heads, T, T)`` numpy array.
    """
    if ids.dim() != 2 or ids.size(0) != 1:
        raise ValueError("capture_attention expects a single sequence, shape (1, T)")
    _, _, attentions = model(ids, return_attn=True)
    return [layer[0].cpu().numpy() for layer in attentions]  # drop the batch dim


def row_entropy(attn: np.ndarray) -> np.ndarray:
    """Per-query entropy (nats) of one head's attention distribution.

    Row t of a causal attention matrix has only t+1 non-zero entries, so raw
    entropy grows with position for a purely uniform head -- not a
    head-specialisation signal by itself. Kept alongside the normalised
    version below rather than replacing it, since the raw value is still
    useful for spotting rows that collapse to near-zero entropy regardless of
    position (a strongly peaked head).

    Args:
        attn: ``(T, T)`` attention matrix for one head.

    Returns:
        ``(T,)`` array, entropy per query position.
    """
    T = attn.shape[0]
    ent = np.zeros(T)
    for t in range(T):
        p = attn[t, :t + 1]
        p = p[p > 0]
        ent[t] = -(p * np.log(p)).sum()
    return ent


def normalized_entropy(attn: np.ndarray) -> np.ndarray:
    """Entropy divided by ln(t+1) -- the honest, position-independent version.

    A uniform head has normalised entropy 1.0 at every position, regardless of
    how many keys are available to spread over; a peaked head stays near 0.
    This is what should be compared across positions and across heads.
    """
    T = attn.shape[0]
    raw = row_entropy(attn)
    denom = np.log(np.arange(1, T + 1))
    denom[0] = 1.0  # position 0 has exactly one key; entropy is 0/0 -> define as 0
    return raw / denom


def mean_attention_distance(attn: np.ndarray) -> np.ndarray:
    """Per-query mean |query - key| distance, weighted by attention.

    Args:
        attn: ``(T, T)`` attention matrix for one head.

    Returns:
        ``(T,)`` array, mean distance per query position.
    """
    T = attn.shape[0]
    dist = np.zeros(T)
    for t in range(T):
        j = np.arange(t + 1)
        dist[t] = (attn[t, :t + 1] * (t - j)).sum()
    return dist


def normalized_distance(attn: np.ndarray) -> np.ndarray:
    """Mean distance divided by t -- position-independent local/long-range signal."""
    T = attn.shape[0]
    raw = mean_attention_distance(attn)
    denom = np.arange(T).astype(np.float64)
    denom[0] = 1.0  # position 0 can only attend to itself; distance is 0/0 -> 0
    return raw / denom


def previous_token_rate(attn: np.ndarray) -> float:
    """Mean weight each query (from t=1) puts on position t-1.

    High values mark induction/positional heads that mostly copy the previous
    token forward.
    """
    T = attn.shape[0]
    if T < 2:
        return 0.0
    return float(np.mean([attn[t, t - 1] for t in range(1, T)]))


def sink_rate(attn: np.ndarray) -> float:
    """Mean weight every query (from t=1) puts on position 0.

    High values mark an attention-sink head -- a well-documented pattern where
    heads default a chunk of their mass onto the first token when they have
    nothing more useful to attend to.
    """
    T = attn.shape[0]
    if T < 2:
        return 0.0
    return float(np.mean([attn[t, 0] for t in range(1, T)]))


def summarize_head(attn: np.ndarray) -> dict:
    """All scalar statistics for one head, averaged over query positions."""
    return {
        "entropy": float(row_entropy(attn).mean()),
        "normalized_entropy": float(normalized_entropy(attn).mean()),
        "mean_distance": float(mean_attention_distance(attn).mean()),
        "normalized_distance": float(normalized_distance(attn).mean()),
        "previous_token_rate": previous_token_rate(attn),
        "sink_rate": sink_rate(attn),
    }


@torch.no_grad()
def attention_stats_over_sequences(
    model, sequences: list[torch.Tensor],
) -> dict:
    """Average per-(layer, head) statistics over several held-out sequences.

    Aggregate rather than anecdotal: statistics from a single sentence are
    noisy, so this runs several sequences and averages, without ever holding
    every sequence's attention tensors in memory at once.

    Args:
        model: A loaded, ``eval()``-mode GPT.
        sequences: List of ``(1, T)`` token-id tensors on the model's device.

    Returns:
        Dict with ``n_layers``, ``n_heads``, and ``cells``: a flat list of
        per-(layer, head) dicts with averaged statistics, ready to write to CSV.
    """
    accum: dict[tuple[int, int], list[dict]] = {}
    n_layers = n_heads = None

    for ids in sequences:
        layers = capture_attention(model, ids)
        n_layers = len(layers)
        for layer_idx, layer_attn in enumerate(layers):
            n_heads = layer_attn.shape[0]
            for head_idx in range(n_heads):
                key = (layer_idx, head_idx)
                accum.setdefault(key, []).append(summarize_head(layer_attn[head_idx]))

    cells = []
    for (layer_idx, head_idx), stats_list in sorted(accum.items()):
        merged = {"layer": layer_idx, "head": head_idx}
        for stat_key in stats_list[0]:
            merged[stat_key] = float(np.mean([s[stat_key] for s in stats_list]))
        cells.append(merged)

    return {"n_layers": n_layers, "n_heads": n_heads, "n_sequences": len(sequences), "cells": cells}


def classify_head(cell: dict, distance_thresh: float = 0.15, entropy_thresh: float = 0.4) -> str:
    """Rough taxonomy label for a report table: local, content-based, or sink.

    Thresholds are heuristics for a short discussion, not a formal test --
    stated as such in the report.
    """
    if cell["sink_rate"] > 0.3:
        return "sink"
    if cell["normalized_distance"] < distance_thresh and cell["previous_token_rate"] > 0.3:
        return "local/positional"
    if cell["normalized_entropy"] > entropy_thresh:
        return "content-based"
    return "mixed"
