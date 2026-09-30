"""Generate every Phase 2 report figure from saved results.

Runs locally on CPU in about a minute. Most figures are pure re-plots of the
JSON that `eval/run_eval.py` already computed on the GPU, so they are identical
to what a re-run would produce. The two attention heatmap grids need one
32-token forward pass each, which is deterministic in eval() mode (no dropout,
no sampling) -- CPU and GPU differ only at float-precision level, invisible on
a 0-1 colour scale.

Deliberately does not import eval.generate or eval.metrics, so it runs without
sacrebleu installed: the generation metrics are read from generation.json
rather than recomputed.

Usage:
    python scripts/make_figures.py
    python scripts/make_figures.py --ckpt-dir ../outputs --skip-heatmap-grids
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

# Running `python scripts/make_figures.py` puts scripts/ on sys.path, not the
# repo root, so `import eval` would fail. Add the repo root explicitly, so the
# script works both as `python scripts/make_figures.py` and as
# `python -m scripts.make_figures`.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402  -- imported after the sys.path fix above

from eval.attention import capture_attention, first_sentence  # noqa: E402
from eval.loader import load_for_eval  # noqa: E402
from eval.plots import (  # noqa: E402
    plot_attention_heatmap_grid,
    plot_entropy_distance_comparison,
    plot_entropy_distance_scatter,
    plot_loss_comparison,
    plot_loss_curve,
    plot_metric_bars,
    plot_stat_heatmap,
    setup_devanagari_font,
)

LANGS = [
    {"lang": "hi", "dir": "hindi", "tag": "H", "name": "Hindi", "ckpt": "hi_model_final.pt"},
    {"lang": "ne", "dir": "nepali", "tag": "L", "name": "Nepali", "ckpt": "ne_model_final.pt"},
]


def read_train_log(path: Path) -> dict:
    """Load the training log CSV into columns, splitting out validation rows."""
    steps, train_loss, val_steps, val_loss = [], [], [], []
    with open(path, encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            steps.append(int(row["step"]))
            train_loss.append(float(row["train_loss"]))
            if row.get("val_loss"):
                val_steps.append(int(row["step"]))
                val_loss.append(float(row["val_loss"]))
    return {"steps": steps, "train_loss": train_loss,
            "val_steps": val_steps, "val_loss": val_loss}


def make_loss_figures(figures_dir: Path) -> dict:
    """Per-model loss curves plus the overlaid H-vs-L comparison."""
    logs = {}
    for spec in LANGS:
        log_path = Path(spec["dir"]) / "logs" / "train_log.csv"
        if not log_path.exists():
            print(f"  SKIP loss curve for {spec['name']}: {log_path} not found")
            continue
        log = read_train_log(log_path)
        logs[spec["tag"]] = log
        plot_loss_curve(
            log["steps"], log["train_loss"], log["val_steps"], log["val_loss"],
            f"Model {spec['tag']} ({spec['name']}) — pretraining loss",
            figures_dir / f"loss_{spec['tag']}.png",
        )
        print(f"  loss_{spec['tag']}.png")

    if len(logs) == 2:
        plot_loss_comparison(
            logs["H"]["val_steps"], logs["H"]["val_loss"],
            logs["L"]["val_steps"], logs["L"]["val_loss"],
            figures_dir / "loss_comparison.png",
        )
        print("  loss_comparison.png")
    return logs


def make_attention_stat_figures(figures_dir: Path) -> dict:
    """Layer x head heatmaps and the entropy-vs-distance scatter, per model."""
    all_stats = {}
    for spec in LANGS:
        stats_path = Path(spec["dir"]) / "eval" / "attention_stats.json"
        if not stats_path.exists():
            print(f"  SKIP attention figures for {spec['name']}: {stats_path} not found")
            continue
        stats = json.loads(stats_path.read_text(encoding="utf-8"))
        all_stats[spec["tag"]] = stats
        tag = spec["tag"]

        plot_stat_heatmap(
            stats["cells"], stats["n_layers"], stats["n_heads"], "normalized_entropy",
            f"Model {tag} — normalized attention entropy by layer and head",
            figures_dir / f"attn_{tag}_entropy_heatmap.png",
        )
        plot_stat_heatmap(
            stats["cells"], stats["n_layers"], stats["n_heads"], "normalized_distance",
            f"Model {tag} — normalized mean attention distance by layer and head",
            figures_dir / f"attn_{tag}_distance_heatmap.png",
        )
        plot_entropy_distance_scatter(
            stats["cells"], f"Model {tag} — attention head specialisation",
            figures_dir / f"attn_{tag}_entropy_vs_distance.png",
        )
        print(f"  attn_{tag}_entropy_heatmap.png, attn_{tag}_distance_heatmap.png, "
              f"attn_{tag}_entropy_vs_distance.png")

    if len(all_stats) == 2:
        plot_entropy_distance_comparison(
            all_stats["H"]["cells"], all_stats["L"]["cells"],
            figures_dir / "attn_HL_comparison.png",
        )
        print("  attn_HL_comparison.png")
    return all_stats


def make_generation_figures(figures_dir: Path) -> None:
    """Bar charts of generation metrics across decoding settings, per model."""
    for spec in LANGS:
        gen_path = Path(spec["dir"]) / "eval" / "generation.json"
        if not gen_path.exists():
            print(f"  SKIP generation figures for {spec['name']}: {gen_path} not found")
            continue
        rows = [r for r in json.loads(gen_path.read_text(encoding="utf-8")) if "setting" in r]
        tag = spec["tag"]
        plot_metric_bars(rows, "chrf++", "chrF++",
                         f"Model {tag} — chrF++ by decoding setting",
                         figures_dir / f"gen_{tag}_chrf.png")
        plot_metric_bars(rows, "distinct_2", "distinct-2",
                         f"Model {tag} — Distinct-2 by decoding setting",
                         figures_dir / f"gen_{tag}_distinct2.png")
        print(f"  gen_{tag}_chrf.png, gen_{tag}_distinct2.png")


def make_heatmap_grids(figures_dir: Path, ckpt_dir: Path, device: torch.device) -> None:
    """Attention heatmap grids -- the only figures needing a forward pass."""
    for spec in LANGS:
        ckpt = ckpt_dir / spec["dir"] / spec["ckpt"]
        subset = Path(spec["dir"]) / "data" / "eval" / "test_subset.jsonl"
        if not ckpt.exists():
            print(f"  SKIP heatmap grid for {spec['name']}: {ckpt} not found")
            continue
        if not subset.exists():
            print(f"  SKIP heatmap grid for {spec['name']}: {subset} not found")
            continue

        model, cfg, sp = load_for_eval(spec["lang"], ckpt, device)
        sentence = first_sentence(subset)
        ids = sp.encode(sentence, out_type=int)[:32]
        tokens = [sp.id_to_piece(i) for i in ids]
        layers_attn = capture_attention(model, torch.tensor([ids], device=device))

        n_layers = len(layers_attn)
        layer_idx = sorted({0, n_layers // 2, n_layers - 1})
        head_idx = list(range(min(4, layers_attn[0].shape[0])))
        plot_attention_heatmap_grid(
            layers_attn, tokens, layer_idx, head_idx, spec["tag"],
            figures_dir / f"attn_{spec['tag']}_heatmap_grid.png",
        )
        print(f"  attn_{spec['tag']}_heatmap_grid.png  "
              f"(layers {layer_idx}, heads {head_idx}, {len(ids)} tokens)")
        print(f"    sentence: {sentence[:60]}...")
        del model  # free before loading the next language


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--figures-dir", default="report/figures")
    ap.add_argument("--ckpt-dir", default="../outputs",
                    help="directory holding {hindi,nepali}/*_model_final.pt")
    ap.add_argument("--font", default="report/fonts/NotoSansDevanagari-Regular.ttf")
    ap.add_argument("--skip-heatmap-grids", action="store_true",
                    help="skip the two figures that need the model loaded")
    args = ap.parse_args()

    figures_dir = Path(args.figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    if setup_devanagari_font(args.font):
        print(f"font: {args.font}")
    else:
        print(f"WARNING: {args.font} not found -- Devanagari labels will render as boxes")

    print("\nloss curves:")
    make_loss_figures(figures_dir)

    print("\nattention statistics:")
    make_attention_stat_figures(figures_dir)

    print("\ngeneration metrics:")
    make_generation_figures(figures_dir)

    if not args.skip_heatmap_grids:
        print("\nattention heatmap grids (loads the models):")
        make_heatmap_grids(figures_dir, Path(args.ckpt_dir), torch.device("cpu"))

    figures = sorted(figures_dir.glob("*.png"))
    print(f"\n{len(figures)} figures in {figures_dir}:")
    for fig in figures:
        print(f"  {fig.name:38} {fig.stat().st_size / 1024:6.1f} KB")


if __name__ == "__main__":
    main()
