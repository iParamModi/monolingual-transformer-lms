#!/usr/bin/env python
"""Generate the Phase 3 report figures from the committed result files.

Reads only what is already on the branch -- the finetuning logs and the
reasoning result JSON -- so every figure is reproducible from the repository
without a GPU, and re-running this cannot change a number.

The attention figures are produced by ``finetune/attention_compare.py``, which
needs the checkpoints loaded; this script covers the two that do not:

* ``ft_loss_{H,L}.png`` -- finetuning train/val loss over steps.
* ``reasoning_accuracy.png`` -- pretrained vs. finetuned, per test slice, with
  the chance and positional floors drawn on.
* ``reasoning_by_template.png`` -- per-template-family accuracy, H against L.

PDF implementation guideline 3: every plot carries a title, axis labels and a
legend where one applies. Guideline 4: computation and visualisation stay in
separate functions -- the readers below return data, the plotters below draw it.

Usage (from the repository root)::

    python -m scripts.make_phase3_figures
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Runnable as a script from the repo root as well as with -m.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.plots import save, setup_devanagari_font  # noqa: E402

LANGS = [
    {"lang": "hi", "dir": "hindi", "tag": "H", "name": "Hindi"},
    {"lang": "ne", "dir": "nepali", "tag": "L", "name": "Nepali"},
]

SLICES = ["test_iid", "test_names", "test_templates"]

#: Readable names for the slices, since the axis labels carry the argument.
SLICE_LABELS = {
    "test_iid": "seen templates\nseen names",
    "test_names": "seen templates\nheld-out names",
    "test_templates": "held-out templates\nheld-out names",
}

#: Template family names, from finetune/gen_reasoning.py.
TEMPLATE_NAMES = {
    "0": "T0 pair", "1": "T1 chain→min", "2": "T2 chain→max",
    "3": "T3 A-vs-C*", "4": "T4 numeric", "5": "T5 equal", "6": "T6 middle*",
}


# --------------------------------------------------------------------------- #
# readers
# --------------------------------------------------------------------------- #


def read_finetune_log(path: Path) -> dict:
    """Load a finetuning log CSV into columns, splitting out validation rows."""
    steps, train_loss, val_steps, val_loss = [], [], [], []
    with open(path, encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            steps.append(int(row["step"]))
            train_loss.append(float(row["train_loss"]))
            if row.get("val_loss"):
                val_steps.append(int(row["step"]))
                val_loss.append(float(row["val_loss"]))
    return {"steps": steps, "train_loss": train_loss,
            "val_steps": val_steps, "val_loss": val_loss}


def read_reasoning(directory: Path, tag: str) -> dict | None:
    """Load one ``reasoning_{tag}.json``, or None if it has not been produced."""
    path = directory / "eval" / f"reasoning_{tag}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- #
# plotters
# --------------------------------------------------------------------------- #


def plot_finetune_loss(log: dict, title: str, out_path: Path) -> None:
    """Finetuning train and validation loss on one axis."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(log["steps"], log["train_loss"], label="train loss", alpha=0.8)
    if log["val_steps"]:
        ax.plot(log["val_steps"], log["val_loss"], marker="o", label="validation loss")
    ax.set_title(title, wrap=True)
    ax.set_xlabel("optimizer step")
    ax.set_ylabel("cross-entropy on answer tokens (nats)")
    ax.grid(alpha=0.3)
    ax.legend()
    save(fig, out_path)


def plot_reasoning_accuracy(results: dict, out_path: Path) -> None:
    """Pretrained vs. finetuned accuracy per test slice, one panel per model.

    Three bars per slice: the pretrained model's forced-choice accuracy, the
    finetuned model's forced-choice accuracy, and the finetuned model's exact
    match. The pretrained model's exact match is omitted from the bars because
    it is ~0 for a reason that is about output format rather than reasoning --
    it is in the JSON and discussed in the report instead.

    The chance and best-positional floors are drawn as horizontal lines: an
    accuracy without its floor is not evidence, and on the held-out-template
    slice the positional floor is meaningfully above chance.
    """
    tags = [spec["tag"] for spec in LANGS if spec["tag"] in results]
    fig, axes = plt.subplots(1, len(tags), figsize=(6.5 * len(tags), 4.8), sharey=True)
    axes = np.atleast_1d(axes)

    width = 0.26
    positions = np.arange(len(SLICES))

    for ax, tag in zip(axes, tags):
        pre, post = results[tag]["pretrained"], results[tag]["finetuned"]
        pre_ranked = [pre["slices"][s]["likelihood_acc"] for s in SLICES]
        post_ranked = [post["slices"][s]["likelihood_acc"] for s in SLICES]
        post_exact = [post["slices"][s]["exact_match"] for s in SLICES]

        ax.bar(positions - width, pre_ranked, width,
               label="pretrained (forced choice)", color="#B0B0B0")
        ax.bar(positions, post_ranked, width,
               label="finetuned (forced choice)", color="#4C72B0")
        ax.bar(positions + width, post_exact, width,
               label="finetuned (exact match)", color="#55A868")

        chance = [pre["slices"][s]["baselines"]["random"] for s in SLICES]
        positional = [pre["slices"][s]["baselines"]["positional_best"] for s in SLICES]
        ax.plot(positions, chance, linestyle="--", color="black",
                marker="_", markersize=18, linewidth=1, label="chance")
        ax.plot(positions, positional, linestyle=":", color="#C44E52",
                marker="_", markersize=18, linewidth=1, label="best positional rule")

        ax.set_xticks(positions)
        ax.set_xticklabels([SLICE_LABELS[s] for s in SLICES], fontsize=8)
        ax.set_ylim(0, 1.0)
        ax.set_title(f"Model {tag} ({results[tag]['name']})", wrap=True)
        ax.set_xlabel("test slice")
        ax.set_ylabel("accuracy")
        ax.grid(axis="y", alpha=0.3)

    axes[0].legend(fontsize=8, loc="upper right")
    fig.suptitle("Reasoning accuracy before and after finetuning, by generalisation slice",
                 y=1.02)
    save(fig, out_path)


def plot_reasoning_by_template(results: dict, out_path: Path) -> None:
    """Finetuned accuracy per template family, Model H against Model L.

    The aggregate can hide the finding. Near-ceiling on the transitive chains
    with a collapse on T6 would say the model learned to pick extremes rather
    than to order the entities, and only this breakdown shows it. Families
    marked ``*`` were held out of training entirely.
    """
    families: dict[str, dict[str, float]] = {}
    for spec in LANGS:
        tag = spec["tag"]
        if tag not in results:
            continue
        for slice_name in SLICES:
            by_template = results[tag]["finetuned"]["slices"][slice_name]["by_template"]
            for template, row in by_template.items():
                families.setdefault(template, {})[tag] = row["likelihood_acc"]

    order = sorted(families, key=int)
    tags = [spec["tag"] for spec in LANGS if spec["tag"] in results]
    positions = np.arange(len(order))
    width = 0.38

    fig, ax = plt.subplots(figsize=(1.5 * len(order) + 3, 4.5))
    for offset, (tag, colour) in enumerate(zip(tags, ("#4C72B0", "#DD8452"))):
        ax.bar(positions + (offset - 0.5) * width,
               [families[t].get(tag, 0.0) for t in order],
               width, label=f"Model {tag}", color=colour)

    ax.axhline(1 / 3, linestyle="--", color="black", linewidth=1, label="chance (3 candidates)")
    ax.set_xticks(positions)
    ax.set_xticklabels([TEMPLATE_NAMES.get(t, t) for t in order], fontsize=8, rotation=20)
    ax.set_ylim(0, 1.0)
    ax.set_title("Finetuned forced-choice accuracy by template family "
                 "(* = held out of training)", wrap=True)
    ax.set_xlabel("template family")
    ax.set_ylabel("accuracy")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=8)
    save(fig, out_path)


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #


def main(argv: list[str] | None = None) -> None:
    """Build every Phase 3 figure that the committed results support."""
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):  # pragma: no cover
        pass

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--figures-dir", default="report/figures")
    parser.add_argument("--font", default="report/fonts/NotoSansDevanagari-Regular.ttf")
    args = parser.parse_args(argv)

    figures_dir = Path(args.figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    if not setup_devanagari_font(args.font):
        print(f"WARNING: {args.font} not found -- Devanagari labels would render as boxes")

    print("finetuning loss curves:")
    for spec in LANGS:
        log_path = Path(spec["dir"]) / "logs" / "finetune_log.csv"
        if not log_path.exists():
            print(f"  SKIP {spec['name']}: {log_path} not found")
            continue
        plot_finetune_loss(
            read_finetune_log(log_path),
            f"Model {spec['tag']} ({spec['name']}) — reasoning finetuning loss",
            figures_dir / f"ft_loss_{spec['tag']}.png",
        )
        print(f"  ft_loss_{spec['tag']}.png")

    results: dict[str, dict] = {}
    for spec in LANGS:
        pre = read_reasoning(Path(spec["dir"]), "pretrained")
        post = read_reasoning(Path(spec["dir"]), "finetuned")
        if pre and post:
            results[spec["tag"]] = {"pretrained": pre, "finetuned": post,
                                    "name": spec["name"]}
        else:
            have = ", ".join(n for n, r in (("pretrained", pre), ("finetuned", post)) if r)
            print(f"  SKIP reasoning figures for {spec['name']}: "
                  f"have [{have or 'nothing'}], need both")

    print("\nreasoning accuracy:")
    if results:
        plot_reasoning_accuracy(results, figures_dir / "reasoning_accuracy.png")
        print("  reasoning_accuracy.png")
        plot_reasoning_by_template(results, figures_dir / "reasoning_by_template.png")
        print("  reasoning_by_template.png")
    else:
        print("  nothing to plot yet")

    figures = sorted(figures_dir.glob("*.png"))
    print(f"\n{len(figures)} figures in {figures_dir}")


if __name__ == "__main__":
    main()
