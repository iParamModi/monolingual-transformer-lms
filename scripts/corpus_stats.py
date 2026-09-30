#!/usr/bin/env python3
"""Exact statistics for the cleaned Hindi and Nepali corpora (Phase 1, Step 0).

Counts characters, words and pre-tokens in every cleaned corpus file:

    <lang>/data/processed/<code>_manual_clean.jsonl      -- one combined file,
                                                            reported as a unit
    <lang>/data/processed/<code>_downloaded_clean.jsonl  -- one file containing
                                                            all sources, so it is
                                                            broken down PER SOURCE
                                                            using each document's
                                                            "source" field

(The downloaded corpus arrives as a single file rather than one file per source:
scripts/clean_downloaded.py streams every raw/*.jsonl through the pipeline as one
logical corpus and writes one output. Provenance survives on each document, so
the per-source split below is exact, not inferred from filenames.)

WHY THIS EXISTS
---------------
The project requires >=20% of each language's final *training tokens* to come
from manual collection (LMA_Individual_Project_v1.pdf, 1.2). Cleaning left far
more downloaded text than manual text, so the downloaded side has to be
downsampled to satisfy that ratio. Sizing that downsample correctly needs exact
counts, which is what this produces -- plus the arithmetic for the sample ratio
itself (see the CORPUS MIX section of the output).

ON THE THREE COUNTS
-------------------
characters  len(text) after cleaning.

words       len(text.split()) -- whitespace-delimited. The most stable of the
            three and the basis for the BPE projection.

pre-tokens  Orthographic words + individual punctuation marks: what a BPE
            tokenizer sees BEFORE any merges are applied, i.e. an upper bound
            on the vocabulary's job.

            The regex matters for Devanagari. A plain r"\\w+|[^\\w\\s]" is
            WRONG here: Python's \\w matches characters where str.isalnum() is
            true, which covers Devanagari consonants and independent vowels
            (category Lo) but NOT the dependent vowel signs (Mc) or the virama
            (Mn). Every matra and every conjunct therefore split a word into
            fragments, inflating the count to ~3.4x the word count. Explicitly
            including the whole Devanagari block U+0900-U+097F in the word class
            keeps letter+mark sequences together, which is what an orthographic
            word actually is.

            Real BPE token counts do not exist until the tokenizer is trained
            (Phase 1, 1.3). This is a proxy; the projection below brackets the
            real number using a typical Devanagari BPE fertility range.

Usage:
    python scripts/corpus_stats.py                  # exact: full read (~26 GB)
    python scripts/corpus_stats.py --lang hi
    python scripts/corpus_stats.py --sample 200000  # fast approximation
    python scripts/corpus_stats.py --json stats.json

Reading every cleaned file end to end is roughly 26 GB of I/O and takes a while;
progress and ETA are printed per file. --sample trades exactness for speed by
probing evenly spaced offsets instead (useful for a quick re-check, NOT for the
numbers that go in the report).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

# Word-ish run (Devanagari letters AND their combining marks, plus any other word
# character) OR a single non-space, non-word character.
#
# The Devanagari block is spelled out because \w misses the combining marks (see
# module docstring) -- but the block is split around U+0964/U+0965, the danda and
# double danda. Those live inside the block yet are sentence punctuation (category
# Po), so folding them into the word class would glue "खाया।" together as one
# token. Excluding them lets them fall through to the second branch and be counted
# individually, which is how a tokenizer actually treats them.
PRE_TOKEN_RE = re.compile(r"[\wऀ-ॣ०-ॿ]+|[^\w\s]", re.UNICODE)

LANGUAGES = [("hindi", "hi"), ("nepali", "ne")]
REPO_ROOT = Path(__file__).resolve().parent.parent

# Typical tokens-per-word for a 16k-32k Devanagari BPE vocabulary. Used only to
# bracket the eventual real token count.
FERTILITY_LOW, FERTILITY_HIGH = 1.4, 2.2


class Counts:
    """docs / chars / words / pre-tokens for one bucket."""

    __slots__ = ("docs", "chars", "words", "pretokens")

    def __init__(self) -> None:
        self.docs = self.chars = self.words = self.pretokens = 0

    def add(self, text: str) -> None:
        self.docs += 1
        self.chars += len(text)
        self.words += len(text.split())
        self.pretokens += len(PRE_TOKEN_RE.findall(text))

    def scale(self, factor: float) -> "Counts":
        c = Counts()
        c.docs = round(self.docs * factor)
        c.chars = round(self.chars * factor)
        c.words = round(self.words * factor)
        c.pretokens = round(self.pretokens * factor)
        return c

    def merge(self, other: "Counts") -> None:
        self.docs += other.docs
        self.chars += other.chars
        self.words += other.words
        self.pretokens += other.pretokens

    def as_dict(self) -> dict:
        return {
            "docs": self.docs,
            "chars": self.chars,
            "words": self.words,
            "pretokens": self.pretokens,
            "avg_words_per_doc": round(self.words / self.docs, 1) if self.docs else 0.0,
            "avg_chars_per_word": round(self.chars / self.words, 2) if self.words else 0.0,
        }


def fmt_time(seconds: float) -> str:
    if seconds != seconds or seconds < 0:
        return "--"
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


def row(label: str, c: Counts, indent: int = 2) -> str:
    pad = " " * indent
    return (
        f"{pad}{label:<24} docs={c.docs:>10,}  chars={c.chars:>15,}  "
        f"words={c.words:>14,}  pre-tokens={c.pretokens:>14,}"
    )


def scan_file(path: Path, sample: int | None) -> tuple[Counts, dict[str, Counts]]:
    """Return (total, per_source). Progress is tracked by bytes consumed, so no
    line-counting pre-pass is needed."""
    size = path.stat().st_size
    total = Counts()
    per_source: dict[str, Counts] = {}
    t0 = time.perf_counter()

    def take(doc: dict) -> None:
        text = doc.get("text", "")
        total.add(text)
        src = doc.get("source", "unknown")
        bucket = per_source.get(src)
        if bucket is None:
            bucket = per_source[src] = Counts()
        bucket.add(text)

    if sample:
        # Evenly spaced probes. The cleaned files are written source by source,
        # so a head-only sample would see only the first source -- the probes
        # must span the whole file to represent the mix.
        sampled_bytes = 0
        with open(path, "rb") as fh:
            for i in range(sample):
                fh.seek(size * i // sample)
                fh.readline()               # discard the partial line landed in
                line = fh.readline()
                if not line:
                    continue
                try:
                    doc = json.loads(line)
                except (json.JSONDecodeError, UnicodeDecodeError):
                    continue
                sampled_bytes += len(line)
                take(doc)
        # Scale back up to whole-file totals. Without this the counts are "per N
        # probes", and comparing two files of very different sizes at the same
        # probe count silently reports a meaningless ratio -- which is the one
        # number this script exists to produce.
        factor = (size / sampled_bytes) if sampled_bytes else 1.0
        scaled_total = total.scale(factor)
        scaled_sources = {s: c.scale(factor) for s, c in per_source.items()}
        sys.stdout.write(
            f"\r    {path.name}: {total.docs:,} probes over {size / 1e9:.1f} GB "
            f"-> scaled x{factor:,.0f}\n"
        )
        return scaled_total, scaled_sources

    read = 0
    next_tick = 0
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            read += len(line.encode("utf-8", "ignore"))
            line = line.strip()
            if line:
                try:
                    take(json.loads(line))
                except json.JSONDecodeError:
                    continue
            if total.docs >= next_tick:
                next_tick = total.docs + 20_000
                frac = read / size if size else 1.0
                el = time.perf_counter() - t0
                eta = (el / frac - el) if frac > 0 else float("nan")
                sys.stdout.write(
                    f"\r    {path.name}: {100 * frac:5.1f}%  {total.docs:,} docs  "
                    f"{read / 1e9:.1f}/{size / 1e9:.1f} GB  elapsed {fmt_time(el)}  "
                    f"ETA {fmt_time(eta)}   "
                )
                sys.stdout.flush()
    sys.stdout.write(
        f"\r    {path.name}: done -- {total.docs:,} docs in {fmt_time(time.perf_counter() - t0)}"
        + " " * 30 + "\n"
    )
    return total, per_source


def mix_plan(manual: Counts, downloaded: Counts, target_frac: float) -> dict:
    """How much of the downloaded corpus to keep so that manual reaches
    `target_frac` of the merged corpus.

        manual / (manual + keep) = f   =>   keep = manual * (1 - f) / f

    Computed on words: words are the stable quantity, and manual vs downloaded
    fertility is close enough that the word ratio tracks the token ratio. The
    real check is on token counts after encoding (Phase 1, Step 5).
    """
    m, d = manual.words, downloaded.words
    keep = m * (1 - target_frac) / target_frac if target_frac > 0 else 0.0
    capped = min(keep, d)                      # cannot keep more than exists
    ratio = (capped / d) if d else 0.0
    total_words = m + capped
    return {
        "target_manual_frac": target_frac,
        "downloaded_words_to_keep": round(capped),
        "downloaded_sample_ratio": round(ratio, 4),
        "total_words": round(total_words),
        "manual_frac_achieved": round(m / total_words, 4) if total_words else 0.0,
        "manual_limited": keep > d,            # true = not enough downloaded data
        "projected_tokens_low": round(total_words * FERTILITY_LOW),
        "projected_tokens_high": round(total_words * FERTILITY_HIGH),
    }


def analyse_language(root: Path, folder: str, code: str, sample: int | None) -> dict:
    processed = root / folder / "data" / "processed"
    manual_path = processed / f"{code}_manual_clean.jsonl"
    downloaded_path = processed / f"{code}_downloaded_clean.jsonl"

    print(f"\n{'=' * 78}")
    print(f"{folder.upper()} ({code})")
    print("=" * 78)

    missing = [p for p in (manual_path, downloaded_path) if not p.exists()]
    if missing:
        for p in missing:
            print(f"  [missing] {p}")
        return {}

    print("\n  scanning:")
    manual, _ = scan_file(manual_path, sample)
    downloaded, per_source = scan_file(downloaded_path, sample)

    print(f"\n  MANUAL  ({manual_path.name})")
    print(row("manual (combined)", manual))

    print(f"\n  DOWNLOADED  ({downloaded_path.name}) -- per source")
    for src, c in sorted(per_source.items(), key=lambda kv: -kv[1].words):
        print(row(src, c, indent=4))
    print(row("downloaded (all sources)", downloaded))

    combined = Counts()
    combined.merge(manual)
    combined.merge(downloaded)
    print("\n  IF EVERYTHING WERE USED AS-IS")
    print(row("manual + downloaded", combined))
    frac_now = manual.words / combined.words if combined.words else 0.0
    verdict = "OK" if frac_now >= 0.20 else "BELOW THE 20% REQUIREMENT"
    print(f"    manual share of words   : {100 * frac_now:5.2f}%   <-- {verdict}")

    print("\n  CORPUS MIX -- keep all manual, downsample downloaded")
    print(f"    {'target':>8}  {'keep downloaded':>17}  {'sample ratio':>12}  "
          f"{'total words':>14}  {'projected BPE tokens':>24}")
    plans = {}
    for target in (0.20, 0.25, 0.30):
        p = mix_plan(manual, downloaded, target)
        plans[f"{int(target * 100)}pct"] = p
        flag = "  (manual-limited)" if p["manual_limited"] else ""
        print(
            f"    {100 * target:6.0f}%  {p['downloaded_words_to_keep']:>17,}  "
            f"{p['downloaded_sample_ratio']:>11.2%}  {p['total_words']:>14,}  "
            f"{p['projected_tokens_low']:>10,} - {p['projected_tokens_high']:>10,}{flag}"
        )
    print(f"\n    projection uses fertility {FERTILITY_LOW}-{FERTILITY_HIGH} tokens/word;")
    print("    the real count only exists after the tokenizer is trained.")

    return {
        "manual": manual.as_dict(),
        "downloaded_total": downloaded.as_dict(),
        "downloaded_by_source": {s: c.as_dict() for s, c in per_source.items()},
        "combined_if_all_used": combined.as_dict(),
        "manual_word_frac_if_all_used": round(frac_now, 4),
        "mix_plans": plans,
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--root", default=str(REPO_ROOT),
                    help="repo root containing hindi/ and nepali/")
    ap.add_argument("--lang", choices=["hi", "ne"], default=None,
                    help="only this language (default: both)")
    ap.add_argument("--sample", type=int, default=None,
                    help="approximate from N evenly spaced probes per file instead of "
                         "reading in full. Fast re-check only -- report numbers should "
                         "come from an exact run.")
    ap.add_argument("--json", default=None, help="also write results to this JSON file")
    args = ap.parse_args()

    root = Path(args.root)
    langs = LANGUAGES if not args.lang else [(f, c) for f, c in LANGUAGES if c == args.lang]

    if args.sample:
        print(f"[approximate mode] {args.sample:,} probes per file -- not for the report")
    else:
        print("[exact mode] reading every cleaned file in full")

    t0 = time.perf_counter()
    results = {}
    for folder, code in langs:
        r = analyse_language(root, folder, code, args.sample)
        if r:
            results[folder] = r

    print(f"\n{'=' * 78}")
    print(f"total wall-clock: {fmt_time(time.perf_counter() - t0)}")
    if args.sample:
        print("NOTE: approximate (--sample). Re-run without it for report figures.")

    if args.json and results:
        out = Path(args.json)
        out.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
