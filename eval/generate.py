"""Generation quality: greedy + temperature sampling, scored against references.

For each decoding setting, generates continuations from held-out prefixes and
scores them with BLEU-4 / chrF++ / ROUGE-L against the true continuation, plus
diversity and repetition diagnostics that do not need a reference at all.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from eval.metrics import (
    corpus_bleu4,
    corpus_chrf_pp,
    corpus_rouge_l,
    distinct_n,
    max_repeated_run,
    repetition_rate,
)

DECODING_SETTINGS = [
    {"name": "greedy", "temperature": 0.0},
    {"name": "t0.5", "temperature": 0.5},
    {"name": "t1.0", "temperature": 1.0},
    {"name": "t1.5", "temperature": 1.5},
]


def load_eval_examples(subset_path: str | Path, sp, prefix_toks: int, gen_toks: int) -> list[dict]:
    """Build (prefix, reference) token pairs from the frozen held-out subset.

    Args:
        subset_path: ``test_subset.jsonl`` from ``scripts/make_eval_subset.py``.
        sp: SentencePiece tokenizer, matching the model's language.
        prefix_toks: Prompt length in tokens.
        gen_toks: Reference/generation length in tokens.

    Returns:
        List of dicts with ``prefix_ids``, ``ref_ids``, ``doc_id``/``source``.
    """
    examples = []
    with open(subset_path, encoding="utf-8") as fh:
        for line in fh:
            doc = json.loads(line)
            ids = sp.encode(doc["text"], out_type=int)
            if len(ids) < prefix_toks + gen_toks:
                continue
            examples.append({
                "prefix_ids": ids[:prefix_toks],
                "ref_ids": ids[prefix_toks: prefix_toks + gen_toks],
                "source": doc.get("source", ""),
                "manual": doc.get("manual", False),
            })
    return examples


@torch.no_grad()
def generate_for_setting(
    model, sp, examples: list[dict], setting: dict, gen_toks: int, device, seed: int = 1337,
    batch_size: int = 16,
) -> dict:
    """Generate continuations for one decoding setting over all examples.

    Args:
        model: A loaded, ``eval()``-mode GPT.
        sp: Matching tokenizer.
        examples: From :func:`load_eval_examples`.
        setting: One entry of ``DECODING_SETTINGS``.
        gen_toks: Number of new tokens to generate.
        device: Device to run on.
        seed: RNG seed, offset per setting so runs are reproducible.
        batch_size: Prompts generated per batch.

    Returns:
        Dict with per-metric scores plus the raw hyp/ref pairs and pooled
        generated token ids (for distinct-n / repetition, computed by the caller).
    """
    torch.manual_seed(seed + int(setting["temperature"] * 10))
    hyps, refs, records = [], [], []
    all_gen_ids: list[int] = []

    for start in range(0, len(examples), batch_size):
        batch = examples[start:start + batch_size]
        prefix = torch.tensor([e["prefix_ids"] for e in batch], device=device)
        out = model.generate(
            prefix, max_new_tokens=gen_toks,
            temperature=setting["temperature"], top_k=None, eos_id=None,
        )
        gen_ids_batch = out[:, prefix.size(1):].tolist()
        for ex, gen_ids in zip(batch, gen_ids_batch):
            hyp_text = sp.decode(gen_ids)
            ref_text = sp.decode(ex["ref_ids"])
            hyps.append(hyp_text)
            refs.append(ref_text)
            all_gen_ids.extend(gen_ids)
            records.append({
                "source": ex["source"], "manual": ex["manual"],
                "prefix": sp.decode(ex["prefix_ids"]),
                "hypothesis": hyp_text, "reference": ref_text,
            })

    return {
        "setting": setting["name"],
        "temperature": setting["temperature"],
        "n_examples": len(hyps),
        "bleu4": corpus_bleu4(hyps, refs),
        "chrf++": corpus_chrf_pp(hyps, refs),
        "rouge_l": corpus_rouge_l(hyps, refs),
        "distinct_1": distinct_n(all_gen_ids, 1),
        "distinct_2": distinct_n(all_gen_ids, 2),
        "rep_4": repetition_rate(all_gen_ids, 4),
        "max_repeated_run": max_repeated_run(all_gen_ids),
        "records": records,
    }


def run_generation_eval(
    model, sp, subset_path: str | Path, device,
    n_examples: int = 300, prefix_toks: int = 64, gen_toks: int = 64,
) -> dict:
    """Run every decoding setting and collect results.

    Args:
        model: A loaded, ``eval()``-mode GPT.
        sp: Matching tokenizer.
        subset_path: ``test_subset.jsonl``.
        device: Device to run on.
        n_examples: How many held-out documents to draw from.
        prefix_toks: Prompt length.
        gen_toks: Reference/generation length.

    Returns:
        Dict of ``{setting_name: result}`` plus ``"examples_used"``.
    """
    examples = load_eval_examples(subset_path, sp, prefix_toks, gen_toks)[:n_examples]
    if not examples:
        raise ValueError(
            f"no examples with >= {prefix_toks + gen_toks} tokens found in {subset_path}"
        )
    results = {}
    for setting in DECODING_SETTINGS:
        results[setting["name"]] = generate_for_setting(
            model, sp, examples, setting, gen_toks, device
        )
    results["examples_used"] = len(examples)
    return results


def summary_table(results: dict) -> list[dict]:
    """Strip the per-example records out, leaving just the metric table."""
    return [
        {k: v for k, v in r.items() if k != "records"}
        for name, r in results.items() if name != "examples_used"
    ]
