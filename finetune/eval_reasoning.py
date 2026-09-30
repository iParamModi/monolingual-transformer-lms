#!/usr/bin/env python
"""Score a checkpoint on the synthetic reasoning test slices.

PDF §3.1: *"Report pretrained vs. finetuned accuracy (or exact match) on the
synthetic test set, qualitative successes/failures, and a comparison between
Model H and Model L on these reasoning tasks."*

Run it four times -- pretrained and finetuned, for each language -- and the four
JSON files it writes are the reasoning half of deliverable 3 (metrics) and
deliverable 5 (comparison tables). The per-example JSONL files are where the
qualitative successes and failures come from, so the report quotes committed
data rather than whatever was on screen at the time.

**Two scoring protocols, and why both are needed.**

*Exact match* is what the brief names. Prompt the model, greedily decode up to
``--max-new`` tokens, stop at ``</s>``, compare against the gold answer.

*Likelihood ranking* scores each candidate answer instead: append candidate to
prompt, take the log-likelihood of the candidate's tokens, pick the argmax. This
is not decoration. A **pretrained** 24M model asked "सबसे छोटा कौन है?" does not
emit a bare name -- it continues the text as prose, so exact match scores it near
zero. Reporting "0% → 80%" would then credit a gain in output *formatting* as a
gain in reasoning. Forced choice over the three candidates measures whether the
model can order the entities at all, independent of whether it has learned to
answer in the expected shape. Both numbers go in the table, labelled.

**Normalisation matters.** SentencePiece normalises its input, and several
Devanagari nukta letters -- ``ड़`` in घड़ी, ``ज़`` in तेज़ -- are on Unicode's
composition-exclusion list, so a decode round-trip returns them as base +
U+093C. Canonically identical, different bytes. Comparing raw strings would mark
correct answers wrong on exactly the entities whose names carry a nukta, so both
sides are NFC-normalised before comparison.

**The prompt is encoded by the same function that built the training prefix**
(``finetune.data.encode_prompt``), so the model is asked to continue the
sequence it was trained to continue rather than a re-tokenised approximation
of it.

Usage::

    python -m finetune.eval_reasoning --lang hi \
        --ckpt hindi/checkpoints/ckpt_final.pt --tag pretrained \
        --data-dir hindi/reasoning --out-dir hindi/eval

    python -m finetune.eval_reasoning --lang hi \
        --ckpt hindi/checkpoints_ft/ckpt_ft_final.pt --tag finetuned \
        --data-dir hindi/reasoning --out-dir hindi/eval

    # ... and the same two with --lang ne
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import unicodedata
from collections import Counter
from pathlib import Path

import torch

from eval.lm_eval import score_sequence
from eval.loader import load_for_eval
from finetune.data import encode_answer, encode_prompt, read_jsonl
from train.trainer import pick_device

#: Test slices written by ``finetune.gen_reasoning``, in reporting order.
SLICES = ["test_iid", "test_names", "test_templates"]


def normalize(text: str) -> str:
    """Canonical form for string comparison: NFC plus collapsed whitespace.

    NFC because the tokenizer's normaliser decomposes composition-excluded
    Devanagari nukta letters, so the decoded output of a correct answer need not
    be byte-identical to the ``answer`` field it is compared against.
    """
    return " ".join(unicodedata.normalize("NFC", text).split())


@torch.no_grad()
def greedy_answer(model, sp, prompt: str, max_new: int, device) -> str:
    """Decode the model's answer to one prompt, greedily.

    Args:
        model: Loaded, eval-mode GPT.
        sp: That language's tokenizer.
        prompt: The question text.
        max_new: Token budget for the answer.
        device: Device to run on.

    Returns:
        The decoded continuation, truncated at the first ``</s>``.
    """
    prompt_ids = encode_prompt(sp, prompt)
    ids = torch.tensor([prompt_ids], device=device)
    out = model.generate(ids, max_new, temperature=0.0, eos_id=sp.eos_id())
    new_ids = out[0, len(prompt_ids):].tolist()
    if sp.eos_id() in new_ids:
        new_ids = new_ids[:new_ids.index(sp.eos_id())]
    return sp.decode(new_ids)


@torch.no_grad()
def rank_candidates(model, sp, prompt: str, candidates: list[str], device) -> dict:
    """Score every candidate answer and return the rankings.

    Each candidate is scored as a continuation of the same prompt, so the
    comparison is forced choice rather than free generation.

    Length normalisation is reported alongside the raw sum because candidates
    differ in token count and a summed log-likelihood mechanically favours the
    shorter one -- which in this dataset means favouring whichever entity name
    the tokenizer happened to split into fewer pieces.

    Args:
        model: Loaded, eval-mode GPT.
        sp: That language's tokenizer.
        prompt: The question text.
        candidates: Answer options.
        device: Device to run on.

    Returns:
        Dict with the argmax under each scoring rule and the per-candidate
        log-likelihoods in nats.
    """
    prompt_ids = encode_prompt(sp, prompt)
    totals, per_token = [], []

    for candidate in candidates:
        answer_ids = encode_answer(sp, candidate)
        total = score_sequence(model, prompt_ids + answer_ids, len(prompt_ids), device)
        totals.append(total)
        per_token.append(total / max(1, len(answer_ids)))

    return {
        "ranked": candidates[max(range(len(totals)), key=lambda i: totals[i])],
        "ranked_lennorm": candidates[
            max(range(len(per_token)), key=lambda i: per_token[i])
        ],
        "logprobs": [round(v, 4) for v in totals],
    }


def baselines(records: list[dict]) -> dict:
    """Floors the measured accuracy has to be read against.

    Without these an accuracy is a bare number. With them it is evidence.

    * ``random`` -- expected score of guessing uniformly among the candidates,
      averaged over examples (slices mix two- and three-candidate questions).
    * ``positional_best`` -- the best "always answer the entity mentioned k-th"
      rule. The generator balances answer position specifically so this stays
      near chance; on ``test_templates`` it sits higher because the middle
      entity of a three-entity chain is named in both premises and so is never
      mentioned last.
    * ``majority_answer`` -- always saying the single most frequent answer.

    Args:
        records: The slice's raw dataset records.

    Returns:
        The three baseline accuracies.
    """
    random_acc = sum(1.0 / len(r["candidates"]) for r in records) / len(records)

    scored = [r for r in records if r["answer_position"] >= 0]
    positional = max(
        (sum(1 for r in scored if r["answer_position"] == k) / len(records)
         for k in range(3)),
        default=0.0,
    )

    counts = Counter(r["answer"] for r in records)
    majority = counts.most_common(1)[0][1] / len(records)

    return {
        "random": round(random_acc, 4),
        "positional_best": round(positional, 4),
        "majority_answer": round(majority, 4),
    }


def score_slice(model, sp, records: list[dict], device, max_new: int) -> list[dict]:
    """Run both protocols over one slice.

    Args:
        model: Loaded, eval-mode GPT.
        sp: That language's tokenizer.
        records: Dataset records for this slice.
        device: Device to run on.
        max_new: Token budget for greedy decoding.

    Returns:
        One result dict per example, ready to write as JSONL.
    """
    results = []
    for i, record in enumerate(records, 1):
        gold = normalize(record["answer"])
        generated = greedy_answer(model, sp, record["prompt"], max_new, device)
        ranking = rank_candidates(model, sp, record["prompt"], record["candidates"], device)

        results.append({
            "id": record["id"],
            "template": record["template"],
            "prompt": record["prompt"],
            "gold": record["answer"],
            "generated": generated,
            "exact": normalize(generated) == gold,
            "prefix": normalize(generated).startswith(gold) and bool(gold),
            "ranked": ranking["ranked"],
            "ranked_correct": normalize(ranking["ranked"]) == gold,
            "ranked_lennorm": ranking["ranked_lennorm"],
            "ranked_lennorm_correct": normalize(ranking["ranked_lennorm"]) == gold,
            "logprobs": ranking["logprobs"],
            "candidates": record["candidates"],
        })

        if i % 200 == 0:
            print(f"    {i}/{len(records)}")
    return results


def summarize(results: list[dict], records: list[dict]) -> dict:
    """Aggregate one slice's per-example results.

    Reported per template family as well as overall: an aggregate can hide the
    finding. Near-ceiling on the transitive chains with a collapse on the
    middle-element family would say the model learned to pick extremes rather
    than to order, and only the breakdown shows that.
    """
    def rate(key: str, rows: list[dict]) -> float:
        return round(sum(1 for r in rows if r[key]) / len(rows), 4) if rows else 0.0

    by_template: dict[str, dict] = {}
    for template in sorted({r["template"] for r in results}):
        rows = [r for r in results if r["template"] == template]
        by_template[str(template)] = {
            "n": len(rows),
            "exact_match": rate("exact", rows),
            "likelihood_acc": rate("ranked_correct", rows),
        }

    return {
        "n": len(results),
        "exact_match": rate("exact", results),
        "prefix_match": rate("prefix", results),
        "likelihood_acc": rate("ranked_correct", results),
        "likelihood_acc_lennorm": rate("ranked_lennorm_correct", results),
        "by_template": by_template,
        "baselines": baselines(records),
    }


def build_parser() -> argparse.ArgumentParser:
    """Command-line interface."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--lang", required=True, choices=["hi", "ne"])
    parser.add_argument("--ckpt", required=True, help="checkpoint to score")
    parser.add_argument("--tag", required=True, choices=["pretrained", "finetuned"],
                        help="names the output files and the report row")
    parser.add_argument("--data-dir", required=True, help="directory with the reasoning JSONL")
    parser.add_argument("--out-dir", required=True, help="where the JSON and JSONL go")
    parser.add_argument("--max-new", type=int, default=8,
                        help="token budget for greedy decoding (answers are 1-3 tokens)")
    parser.add_argument("--limit", type=int, default=0,
                        help="score only the first N examples per slice (0 = all)")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    return parser


def main(argv: list[str] | None = None) -> None:
    """Score one checkpoint on all three test slices."""
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):  # pragma: no cover - non-standard stream
        pass

    args = build_parser().parse_args(argv)
    device = pick_device(args.device)

    # load_for_eval refuses to pair a checkpoint with another language's
    # tokenizer, so independence stays enforced rather than assumed.
    model, cfg, sp = load_for_eval(args.lang, args.ckpt, device)
    print(f"{args.tag}: {cfg.name} ({cfg.lang}) on {device}")

    data_dir, out_dir = Path(args.data_dir), Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    report = {
        "lang": args.lang,
        "tag": args.tag,
        "checkpoint": str(args.ckpt),
        "model": cfg.name,
        "max_new_tokens": args.max_new,
        "slices": {},
    }

    started = time.time()
    for name in SLICES:
        records = read_jsonl(data_dir / f"{name}.jsonl")
        if args.limit:
            records = records[:args.limit]
        print(f"  {name}: {len(records)} examples ...")

        results = score_slice(model, sp, records, device, args.max_new)
        report["slices"][name] = summarize(results, records)

        samples_path = out_dir / f"reasoning_samples_{args.tag}_{name}.jsonl"
        with open(samples_path, "w", encoding="utf-8", newline="\n") as handle:
            for row in results:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

        summary = report["slices"][name]
        print(f"    exact {summary['exact_match']:.3f}  "
              f"prefix {summary['prefix_match']:.3f}  "
              f"ranked {summary['likelihood_acc']:.3f}  "
              f"(random {summary['baselines']['random']:.3f}, "
              f"positional {summary['baselines']['positional_best']:.3f})")

    report["elapsed_s"] = round(time.time() - started, 1)
    out_path = out_dir / f"reasoning_{args.tag}.json"
    out_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"\nwrote {out_path} in {report['elapsed_s'] / 60:.1f} min")


if __name__ == "__main__":
    main()
