"""Freeze a small, reproducible sample of the test split for evaluation.

`data/splits/test.jsonl` (550-670 MB) is never committed and only exists on
this machine. Everything downstream that touches held-out text -- BPB,
generation, attention analysis -- needs to run from a file that IS committed,
so results are reproducible without the big split lying around.

This script:
  1. Reads test.jsonl (never test.bin -- we need raw text, not token ids).
  2. Keeps documents whose word count clears a floor, so a 64-token prompt and
     a 64-token continuation both fit with headroom (Hindi/Nepali fertility is
     ~1.5, so 128 words easily clears 192 tokens).
  3. Samples --n-docs of them with a fixed seed.
  4. Writes a small JSONL (~6-10 MB) plus a sidecar recording exactly how the
     sample was built, so it can be regenerated and checked byte-for-byte.

Usage:
    python scripts/make_eval_subset.py --lang hi
    python scripts/make_eval_subset.py --lang ne
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import random
from pathlib import Path

MIN_WORDS = 128  # ~192+ tokens at fertility ~1.5, enough for a 64+64 split


def file_sha256_head(path: Path, n_bytes: int = 1_000_000) -> str:
    """Hash of the file's first n_bytes -- enough to detect "wrong file", cheap
    enough not to hash 550 MB just to fingerprint it."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        h.update(fh.read(n_bytes))
    return h.hexdigest()


def eligible_documents(src: Path):
    """Stream documents from test.jsonl that clear the length floor.

    Args:
        src: Path to the language's test.jsonl.

    Yields:
        (line_index, doc) for every document with >= MIN_WORDS words.
    """
    with io.open(src, encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            line = line.strip()
            if not line:
                continue
            doc = json.loads(line)
            if len(doc["text"].split()) >= MIN_WORDS:
                yield i, doc


def build_subset(src: Path, n_docs: int, seed: int) -> tuple[list[dict], dict]:
    """Sample n_docs eligible documents deterministically.

    Reservoir sampling: the source is 550-670 MB and this reads it once,
    streaming, without loading the whole file or needing a document count
    up front.

    Args:
        src: Path to test.jsonl.
        n_docs: Target sample size.
        seed: RNG seed -- the same seed always yields the same sample.

    Returns:
        (sampled_docs, info) where info records how the sample was built.
    """
    rng = random.Random(seed)
    reservoir: list[dict] = []
    seen_eligible = 0
    total_lines = 0

    for total_lines, (_, doc) in enumerate(eligible_documents(src), start=1):
        seen_eligible += 1
        if len(reservoir) < n_docs:
            reservoir.append(doc)
        else:
            j = rng.randint(0, seen_eligible - 1)
            if j < n_docs:
                reservoir[j] = doc

    # Sort by (source, first 40 chars of text) rather than leaving reservoir
    # order, which depends on iteration order and is not a meaningful sort --
    # this way the output file is byte-identical across reruns and machines.
    reservoir.sort(key=lambda d: (d.get("source", ""), d["text"][:40]))

    info = {
        "seed": seed,
        "min_words": MIN_WORDS,
        "requested_docs": n_docs,
        "eligible_docs_seen": seen_eligible,
        "source_file": str(src),
        "source_sha256_head": file_sha256_head(src),
    }
    return reservoir, info


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lang", required=True, choices=["hi", "ne"])
    ap.add_argument("--n-docs", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    lang_dir = {"hi": "hindi", "ne": "nepali"}[args.lang]
    src = Path(lang_dir) / "data" / "splits" / "test.jsonl"
    out_dir = Path(lang_dir) / "data" / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "test_subset.jsonl"
    info_path = out_dir / "test_subset_info.json"

    if not src.exists():
        raise SystemExit(
            f"{src} not found. This script only runs on the machine that still "
            f"has the Phase 1 splits -- they are not committed and not on HF."
        )

    print(f"reading {src} ...")
    docs, info = build_subset(src, args.n_docs, args.seed)
    if len(docs) < args.n_docs:
        print(f"warning: only {len(docs)} eligible documents found, "
              f"fewer than the requested {args.n_docs}")

    with io.open(out_path, "w", encoding="utf-8") as fh:
        for doc in docs:
            fh.write(json.dumps(doc, ensure_ascii=False) + "\n")

    info["docs_written"] = len(docs)
    info["total_words"] = sum(len(d["text"].split()) for d in docs)
    info["manual_docs"] = sum(1 for d in docs if d.get("manual"))
    with io.open(info_path, "w", encoding="utf-8") as fh:
        json.dump(info, fh, indent=2, ensure_ascii=False)

    size_mb = out_path.stat().st_size / 1e6
    print(f"\n{out_path}  ({len(docs)} docs, {size_mb:.1f} MB)")
    print(f"{info_path}")
    print(f"eligible pool: {info['eligible_docs_seen']:,} / manual share: "
          f"{info['manual_docs'] / max(1, len(docs)):.1%}")


if __name__ == "__main__":
    main()
