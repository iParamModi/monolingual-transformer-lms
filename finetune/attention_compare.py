#!/usr/bin/env python
"""Compare attention before and after reasoning finetuning.

PDF §3.2: *"Reuse the Phase 2 attention toolkit on the finetuned checkpoints.
Compare pretrained vs. finetuned heatmaps for at least one early and one late
layer per model. Comment on whether finetuning changed local vs. long-range
attention or head specialization, especially on comparative-reasoning
prompts."*

The Phase 2 toolkit is imported unchanged -- ``capture_attention`` for the raw
weights, ``attention_stats_over_sequences`` for the per-(layer, head) entropy
and distance statistics, ``eval.plots`` for the figures. Nothing about how
attention is measured differs from Phase 2, which is the point: if the numbers
move, that is the finetuning and not a change of instrument.

**Four things make this a valid comparison rather than two unrelated pictures.**

1. **The same inputs through both checkpoints.** Sequences are built once from
   ``test_templates.jsonl`` and reused. Those prompts are held out of training,
   so they are unseen by the finetuned model as well as by the pretrained one --
   any difference is a difference in the models, not in their familiarity with
   the text.
2. **Comparative-reasoning prompts**, as the brief asks, rather than corpus
   text. Phase 2 already measured attention on held-out corpus windows; the
   question here is what the heads do when the input is a comparison.
3. **A shared colour scale.** The heatmap grids are drawn with ``vmin=0,
   vmax=1`` (Phase 2's ``plot_attention_heatmap_grid`` already fixes this), so
   the two grids are directly comparable. Two independently autoscaled grids
   would show a difference that is an artifact of the colourbar.
4. **Delta figures.** ``finetuned - pretrained`` per (layer, head), on a
   diverging colormap centred at zero. The delta is the finding; two absolute
   grids side by side leave the reader doing the subtraction.

**On interpreting the result.** 1,875 steps at 1e-4 on a 24M model is a small
nudge. If the deltas are small, they should be reported as small, with the
numbers. Manufacturing a story out of a two-percent entropy shift measured on
one training run per language would not survive a viva question.

Usage::

    python -m finetune.attention_compare --lang hi \
        --pretrained hindi/checkpoints/ckpt_final.pt \
        --finetuned hindi/checkpoints_ft/ckpt_ft_final.pt \
        --data-dir hindi/reasoning --out-dir hindi/eval

    python -m finetune.attention_compare --lang ne \
        --pretrained nepali/checkpoints/ckpt_final.pt \
        --finetuned nepali/checkpoints_ft/ckpt_ft_final.pt \
        --data-dir nepali/reasoning --out-dir nepali/eval
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from eval.attention import attention_stats_over_sequences, capture_attention, classify_head
from eval.loader import load_for_eval
from eval.plots import plot_attention_heatmap_grid, plot_stat_heatmap, save, setup_devanagari_font
from finetune.data import encode_prompt, read_jsonl
from train.trainer import pick_device

#: Statistics compared before and after. Both are normalised by position in the
#: Phase 2 toolkit, so sequences of different length can be averaged together.
STATS = ["normalized_entropy", "normalized_distance", "previous_token_rate", "sink_rate"]


# --------------------------------------------------------------------------- #
# computation
# --------------------------------------------------------------------------- #


def build_sequences(records: list[dict], sp, device, n: int, max_len: int
                    ) -> tuple[list[torch.Tensor], list[dict]]:
    """Tokenise reasoning prompts into sequences for the attention pass.

    Only the prompt is used, not the answer: the question the brief asks is
    what the heads do while reading a comparison, and appending a gold answer
    the model did not produce would put text in the context that never appears
    at inference.

    Args:
        records: Rows from a reasoning split.
        sp: That language's tokenizer.
        device: Device the model is on.
        n: How many sequences to build.
        max_len: Skip anything longer, to keep the batch uniform in cost.

    Returns:
        ``(sequences, used)`` -- ``(1, T)`` tensors and the records behind them.
    """
    sequences, used = [], []
    for record in records:
        ids = encode_prompt(sp, record["prompt"])
        if len(ids) > max_len:
            continue
        sequences.append(torch.tensor([ids], device=device))
        used.append(record)
        if len(sequences) >= n:
            break
    if not sequences:
        raise ValueError("no prompt was short enough to use; raise --max-len")
    return sequences, used


def per_layer_means(cells: list[dict]) -> dict:
    """Average each statistic within a layer, for the report's layer-wise table."""
    layers: dict[int, list[dict]] = {}
    for cell in cells:
        layers.setdefault(cell["layer"], []).append(cell)
    return {
        str(layer): {
            stat: round(float(np.mean([c[stat] for c in group])), 4) for stat in STATS
        }
        for layer, group in sorted(layers.items())
    }


def overall_means(cells: list[dict]) -> dict:
    """Average each statistic over every head in the model."""
    return {stat: round(float(np.mean([c[stat] for c in cells])), 4) for stat in STATS}


def delta_cells(pre: list[dict], post: list[dict]) -> list[dict]:
    """Per-(layer, head) change, finetuned minus pretrained.

    Args:
        pre: Cells from the pretrained checkpoint.
        post: Cells from the finetuned checkpoint.

    Returns:
        Cells carrying the difference for each statistic.
    """
    index = {(c["layer"], c["head"]): c for c in pre}
    out = []
    for cell in post:
        before = index[(cell["layer"], cell["head"])]
        row = {"layer": cell["layer"], "head": cell["head"]}
        for stat in STATS:
            row[stat] = cell[stat] - before[stat]
        out.append(row)
    return out


def head_taxonomy(cells: list[dict]) -> dict:
    """Count how many heads fall into each behavioural class.

    Uses Phase 2's ``classify_head`` thresholds unchanged. They are heuristics
    for discussion, not a formal test -- but applying the *same* heuristics
    before and after makes the shift in counts meaningful even if the absolute
    labels are rough.
    """
    counts: dict[str, int] = {}
    for cell in cells:
        label = classify_head(cell)
        counts[label] = counts.get(label, 0) + 1
    return dict(sorted(counts.items()))


# --------------------------------------------------------------------------- #
# visualisation (kept separate from the computation above, per PDF guideline 4)
# --------------------------------------------------------------------------- #


def plot_delta_heatmap(cells: list[dict], n_layers: int, n_heads: int, stat_key: str,
                       title: str, out_path: str | Path) -> None:
    """Layer x head heatmap of a finetuned-minus-pretrained change.

    Diverging colormap centred at zero, with the limits set symmetrically from
    the largest absolute change, so a white cell means "no change" and the two
    directions are visually comparable. A sequential colormap here would make
    a tiny positive shift look like a large effect.
    """
    grid = np.full((n_layers, n_heads), np.nan)
    for cell in cells:
        grid[cell["layer"], cell["head"]] = cell[stat_key]

    limit = float(np.nanmax(np.abs(grid))) or 1e-9
    fig, ax = plt.subplots(figsize=(1.1 * n_heads + 2, 1.1 * n_layers + 1))
    im = ax.imshow(grid, cmap="coolwarm", aspect="auto", vmin=-limit, vmax=limit)
    for i in range(n_layers):
        for j in range(n_heads):
            ax.text(j, i, f"{grid[i, j]:+.3f}", ha="center", va="center", fontsize=7)

    ax.set_xticks(range(n_heads))
    ax.set_yticks(range(n_layers))
    ax.set_title(title, wrap=True)
    ax.set_xlabel("head")
    ax.set_ylabel("layer")
    fig.colorbar(im, ax=ax, label=f"change in {stat_key.replace('_', ' ')}")
    save(fig, out_path)


def plot_layer_profile(pre: dict, post: dict, stat_key: str, model_tag: str,
                       title: str, out_path: str | Path) -> None:
    """Layer-wise profile of one statistic, pretrained against finetuned.

    The single clearest picture of whether finetuning changed local versus
    long-range behaviour: two lines over depth, on one axis, same scale.
    """
    layers = sorted(pre, key=int)
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.plot([int(k) for k in layers], [pre[k][stat_key] for k in layers],
            marker="o", label="pretrained")
    ax.plot([int(k) for k in layers], [post[k][stat_key] for k in layers],
            marker="s", label="finetuned")
    ax.set_title(title, wrap=True)
    ax.set_xlabel("layer")
    ax.set_ylabel(stat_key.replace("_", " "))
    ax.set_xticks([int(k) for k in layers])
    ax.grid(alpha=0.3)
    ax.legend()
    save(fig, out_path)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    """Command-line interface."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--lang", required=True, choices=["hi", "ne"])
    parser.add_argument("--pretrained", required=True, help="Phase 2 checkpoint")
    parser.add_argument("--finetuned", required=True, help="Phase 3 checkpoint")
    parser.add_argument("--data-dir", required=True, help="directory with the reasoning JSONL")
    parser.add_argument("--split", default="test_templates",
                        help="which slice to draw prompts from (held out by default)")
    parser.add_argument("--out-dir", required=True, help="where the stats JSON goes")
    parser.add_argument("--figures-dir", default="report/figures")
    parser.add_argument("--font", default="report/fonts/NotoSansDevanagari-Regular.ttf")
    parser.add_argument("--n-sequences", type=int, default=64,
                        help="prompts to average the statistics over")
    parser.add_argument("--max-len", type=int, default=64,
                        help="skip prompts longer than this")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    return parser


def main(argv: list[str] | None = None) -> None:
    """Measure attention on both checkpoints and write the comparison."""
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):  # pragma: no cover - non-standard stream
        pass

    args = build_parser().parse_args(argv)
    device = pick_device(args.device)
    tag = "H" if args.lang == "hi" else "L"

    if not setup_devanagari_font(args.font):
        print(f"WARNING: {args.font} not found -- Devanagari labels will render as boxes")

    figures_dir = Path(args.figures_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    records = read_jsonl(Path(args.data_dir) / f"{args.split}.jsonl")

    # The pretrained model first, so the sequences it defines are the ones the
    # finetuned model is measured on too.
    pre_model, cfg, sp = load_for_eval(args.lang, args.pretrained, device)
    sequences, used = build_sequences(records, sp, device, args.n_sequences, args.max_len)
    print(f"Model {tag}: {len(sequences)} reasoning prompts from {args.split}")

    print("  pretrained ...")
    pre_stats = attention_stats_over_sequences(pre_model, sequences)

    post_model, post_cfg, _ = load_for_eval(args.lang, args.finetuned, device)
    if post_cfg.to_dict() != cfg.to_dict():
        raise SystemExit(
            "the two checkpoints have different architectures -- their attention "
            "statistics are not comparable"
        )
    print("  finetuned ...")
    post_stats = attention_stats_over_sequences(post_model, sequences)

    n_layers, n_heads = pre_stats["n_layers"], pre_stats["n_heads"]
    deltas = delta_cells(pre_stats["cells"], post_stats["cells"])

    # ---- heatmap grids on one shared prompt, early + mid + late layers ----
    # The same prompt and the same fixed 0-1 colour scale for both models, so
    # the two grids can be read against each other.
    prompt = min((r["prompt"] for r in used), key=lambda p: len(encode_prompt(sp, p)))
    ids = encode_prompt(sp, prompt)[:32]
    tokens = [sp.id_to_piece(i) for i in ids]
    layer_indices = sorted({0, n_layers // 2, n_layers - 1})
    head_indices = list(range(min(4, n_heads)))
    print(f"  heatmap prompt: {prompt[:70]}")

    for label, model in (("pre", pre_model), ("ft", post_model)):
        layers_attn = capture_attention(model, torch.tensor([ids], device=device))
        plot_attention_heatmap_grid(
            layers_attn, tokens, layer_indices, head_indices,
            f"{tag} ({'pretrained' if label == 'pre' else 'finetuned'}, reasoning prompt)",
            figures_dir / f"attn_{label}_{tag}_reasoning_grid.png",
        )

    # ---- per-head absolute maps and the deltas ----
    for label, stats in (("pre", pre_stats), ("ft", post_stats)):
        state = "pretrained" if label == "pre" else "finetuned"
        for stat in ("normalized_entropy", "normalized_distance"):
            plot_stat_heatmap(
                stats["cells"], n_layers, n_heads, stat,
                f"Model {tag} ({state}) — {stat.replace('_', ' ')} on reasoning prompts",
                figures_dir / f"attn_{label}_{tag}_{stat}.png",
            )

    for stat in ("normalized_entropy", "normalized_distance"):
        plot_delta_heatmap(
            deltas, n_layers, n_heads, stat,
            f"Model {tag} — change in {stat.replace('_', ' ')} after finetuning "
            f"(finetuned − pretrained)",
            figures_dir / f"attn_delta_{tag}_{stat}.png",
        )

    pre_layers, post_layers = per_layer_means(pre_stats["cells"]), per_layer_means(post_stats["cells"])
    for stat in ("normalized_entropy", "normalized_distance"):
        plot_layer_profile(
            pre_layers, post_layers, stat, tag,
            f"Model {tag} — {stat.replace('_', ' ')} by layer, before and after finetuning",
            figures_dir / f"attn_profile_{tag}_{stat}.png",
        )

    # ---- the numbers ----
    report = {
        "lang": args.lang,
        "model_tag": tag,
        "split": args.split,
        "n_sequences": len(sequences),
        "pretrained_checkpoint": str(args.pretrained),
        "finetuned_checkpoint": str(args.finetuned),
        "pretrained": {
            "overall": overall_means(pre_stats["cells"]),
            "per_layer": pre_layers,
            "head_taxonomy": head_taxonomy(pre_stats["cells"]),
            "cells": pre_stats["cells"],
        },
        "finetuned": {
            "overall": overall_means(post_stats["cells"]),
            "per_layer": post_layers,
            "head_taxonomy": head_taxonomy(post_stats["cells"]),
            "cells": post_stats["cells"],
        },
        "delta": {
            "overall": {
                stat: round(overall_means(post_stats["cells"])[stat]
                            - overall_means(pre_stats["cells"])[stat], 4)
                for stat in STATS
            },
            "largest_abs_change": {
                stat: max(
                    ({"layer": c["layer"], "head": c["head"], "change": round(c[stat], 4)}
                     for c in deltas),
                    key=lambda d: abs(d["change"]),
                )
                for stat in STATS
            },
            "cells": [
                {**{"layer": c["layer"], "head": c["head"]},
                 **{s: round(c[s], 5) for s in STATS}}
                for c in deltas
            ],
        },
    }
    out_path = out_dir / "attention_ft_stats.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8")

    print(f"\n  {'layer':>5} {'entropy pre':>12} {'entropy ft':>11} "
          f"{'distance pre':>13} {'distance ft':>12}")
    for layer in sorted(pre_layers, key=int):
        print(f"  {layer:>5} "
              f"{pre_layers[layer]['normalized_entropy']:>12.4f} "
              f"{post_layers[layer]['normalized_entropy']:>11.4f} "
              f"{pre_layers[layer]['normalized_distance']:>13.4f} "
              f"{post_layers[layer]['normalized_distance']:>12.4f}")
    print(f"\n  overall change: " + ", ".join(
        f"{stat.replace('_', ' ')} {report['delta']['overall'][stat]:+.4f}" for stat in STATS
    ))
    print(f"  wrote {out_path} and 10 figures to {figures_dir}")


if __name__ == "__main__":
    main()
