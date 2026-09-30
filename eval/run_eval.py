"""Run the full Phase 2 evaluation suite for one language.

Chains everything in eval/: intrinsic metrics (PPL, BPB), generation quality
(BLEU/chrF++/ROUGE-L, diversity, samples), and attention analysis (stats +
heatmaps), writing every artifact the brief's deliverables 5-7 require. Meant
to run once per language on the machine holding the trained checkpoint (Kaggle).

Usage:
    python -m eval.run_eval --lang hi \\
        --ckpt hindi/checkpoints/hi_model_final.pt \\
        --data-dir hindi/data/bin \\
        --subset hindi/data/eval/test_subset.jsonl \\
        --out-dir hindi/eval
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from eval.attention import attention_stats_over_sequences, capture_attention, first_sentence
from eval.generate import run_generation_eval, summary_table
from eval.lm_eval import bits_per_byte, packed_perplexity, unigram_baseline_from_counts
from eval.loader import load_for_eval
from eval.plots import (
    plot_attention_heatmap_grid,
    plot_entropy_distance_scatter,
    plot_metric_bars,
    plot_stat_heatmap,
    setup_devanagari_font,
)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lang", required=True, choices=["hi", "ne"])
    ap.add_argument("--ckpt", required=True, help="model_final.pt or ckpt_final.pt")
    ap.add_argument("--data-dir", required=True, help="directory with train.bin, val.bin, test.bin")
    ap.add_argument("--subset", required=True, help="test_subset.jsonl from make_eval_subset.py")
    ap.add_argument("--out-dir", required=True, help="where JSON/JSONL results go")
    ap.add_argument("--figures-dir", default="report/figures")
    ap.add_argument("--font", default="report/fonts/NotoSansDevanagari-Regular.ttf")
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--n-generation-examples", type=int, default=300)
    ap.add_argument("--n-attention-sequences", type=int, default=64)
    ap.add_argument("--attention-seq-len", type=int, default=256)
    ap.add_argument("--heatmap-sentence", default=None,
                    help="a short held-out sentence for the heatmap grid; "
                         "defaults to the first eligible line of the subset")
    return ap


def pick_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)




def run_intrinsic(model, sp, cfg, data_dir: Path, subset_path: Path, device, out_dir: Path) -> dict:
    """Deliverable 5: perplexity (val + test) and bits-per-byte."""
    print("intrinsic evaluation: val perplexity ...")
    val = packed_perplexity(model, data_dir / "val.bin", cfg.max_seq_len, device)
    print(f"  val  CE {val['cross_entropy']:.4f}  PPL {val['perplexity']:.2f}")

    print("intrinsic evaluation: test perplexity ...")
    test = packed_perplexity(model, data_dir / "test.bin", cfg.max_seq_len, device)
    print(f"  test CE {test['cross_entropy']:.4f}  PPL {test['perplexity']:.2f}")

    print("intrinsic evaluation: bits-per-byte on the frozen subset ...")
    bpb = bits_per_byte(model, sp, subset_path, device)
    print(f"  BPB {bpb['bpb']:.4f}  ({bpb['docs']} docs, {bpb['bytes_per_token']:.2f} bytes/token)")

    print("intrinsic evaluation: unigram baseline (train.bin -> test.bin) ...")
    baseline = unigram_baseline_from_counts(data_dir / "train.bin", data_dir / "test.bin", cfg.vocab_size)
    print(f"  unigram baseline PPL {baseline['perplexity']:.1f}")

    result = {"val": val, "test": test, "bpb": bpb, "unigram_baseline": baseline}
    (out_dir / "lm_eval.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def run_generation(model, sp, subset_path: Path, device, out_dir: Path, n_examples: int) -> dict:
    """Deliverable 6: generation quality, diversity, and saved samples."""
    print(f"generation evaluation: {n_examples} examples x 4 decoding settings ...")
    results = run_generation_eval(model, sp, subset_path, device, n_examples=n_examples)

    for name, r in results.items():
        if name == "examples_used":
            continue
        print(f"  {name:8} BLEU4 {r['bleu4']:.2f}  chrF++ {r['chrf++']:.2f}  "
              f"ROUGE-L {r['rouge_l']:.3f}  distinct2 {r['distinct_2']:.3f}  rep4 {r['rep_4']:.3f}")
        samples_path = out_dir / f"samples_{name}.jsonl"
        with open(samples_path, "w", encoding="utf-8") as fh:
            for rec in r["records"]:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    (out_dir / "generation.json").write_text(
        json.dumps(summary_table(results) + [{"examples_used": results["examples_used"]}],
                   indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return results


def run_attention(
    model, sp, data_dir: Path, subset_path: Path, device, out_dir: Path, figures_dir: Path,
    model_tag: str, n_sequences: int, seq_len: int, heatmap_sentence: str | None,
) -> dict:
    """Deliverable 7: aggregate attention statistics plus heatmap figures."""
    print(f"attention evaluation: {n_sequences} sequences of {seq_len} tokens ...")

    data = np.memmap(data_dir / "test.bin", dtype=np.uint16, mode="r")
    rng = np.random.default_rng(1337)
    high = len(data) - seq_len
    starts = rng.integers(0, max(1, high), size=n_sequences)
    sequences = [
        torch.tensor([data[s: s + seq_len].astype(np.int64)], device=device) for s in starts
    ]
    stats = attention_stats_over_sequences(model, sequences)
    (out_dir / "attention_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(f"  {len(stats['cells'])} (layer, head) cells averaged over {n_sequences} sequences")

    # Heatmap grid on one short, readable sentence. Every document in the
    # frozen subset is >= 128 words (make_eval_subset.py filters for length),
    # so a short *document* never exists -- take the first sentence of the
    # first document instead, splitting on Devanagari sentence terminators.
    if heatmap_sentence is None:
        heatmap_sentence = first_sentence(subset_path)
    ids = sp.encode(heatmap_sentence, out_type=int)[:32]
    print(f"  heatmap sentence: {heatmap_sentence[:70]}...")
    tokens = [sp.id_to_piece(i) for i in ids]
    layers_attn = capture_attention(model, torch.tensor([ids], device=device))
    n_layers = len(layers_attn)
    layer_idx = sorted({0, n_layers // 2, n_layers - 1})
    head_idx = list(range(min(4, layers_attn[0].shape[0])))

    figures_dir.mkdir(parents=True, exist_ok=True)
    plot_attention_heatmap_grid(
        layers_attn, tokens, layer_idx, head_idx, model_tag,
        figures_dir / f"attn_{model_tag}_heatmap_grid.png",
    )
    plot_stat_heatmap(
        stats["cells"], stats["n_layers"], stats["n_heads"], "normalized_entropy",
        f"Model {model_tag} — normalized attention entropy by layer and head",
        figures_dir / f"attn_{model_tag}_entropy_heatmap.png",
    )
    plot_stat_heatmap(
        stats["cells"], stats["n_layers"], stats["n_heads"], "normalized_distance",
        f"Model {model_tag} — normalized mean attention distance by layer and head",
        figures_dir / f"attn_{model_tag}_distance_heatmap.png",
    )
    plot_entropy_distance_scatter(
        stats["cells"], f"Model {model_tag} — attention head specialisation",
        figures_dir / f"attn_{model_tag}_entropy_vs_distance.png",
    )
    return stats


def main() -> None:
    args = build_parser().parse_args()
    device = pick_device(args.device)
    print(f"device: {device}")

    if not setup_devanagari_font(args.font):
        print(f"WARNING: {args.font} not found -- Devanagari heatmap labels will render as boxes")

    model, cfg, sp = load_for_eval(args.lang, args.ckpt, device)
    print(f"loaded {cfg.name} ({cfg.lang}), {cfg.vocab_size} vocab")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = Path(args.figures_dir)
    data_dir = Path(args.data_dir)
    subset_path = Path(args.subset)
    model_tag = "H" if args.lang == "hi" else "L"

    t0 = time.time()
    run_intrinsic(model, sp, cfg, data_dir, subset_path, device, out_dir)
    generation = run_generation(model, sp, subset_path, device, out_dir, args.n_generation_examples)
    run_attention(
        model, sp, data_dir, subset_path, device, out_dir, figures_dir, model_tag,
        args.n_attention_sequences, args.attention_seq_len, args.heatmap_sentence,
    )

    rows = summary_table(generation)
    plot_metric_bars(rows, "chrf++", "chrF++", f"Model {model_tag} — chrF++ by decoding setting",
                     figures_dir / f"gen_{model_tag}_chrf.png")
    plot_metric_bars(rows, "distinct_2", "distinct-2", f"Model {model_tag} — Distinct-2 by decoding setting",
                     figures_dir / f"gen_{model_tag}_distinct2.png")

    print(f"\ndone in {(time.time() - t0) / 60:.1f} min. results in {out_dir}, figures in {figures_dir}")


if __name__ == "__main__":
    main()
