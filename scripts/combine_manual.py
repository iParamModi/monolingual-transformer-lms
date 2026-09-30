#!/usr/bin/env python3
"""Merge the crawled manual-collection shards into one JSONL per language.

Second stage of the manual pipeline. scripts/scrape_manual.py writes many small
compressed shards; this collapses them into the single file per language that
scripts/clean_manual.py opens as its stage-0 input:

    <lang>/data/raw/manual/shards/shard-*.jsonl.zst
        |  combine_manual.py    (this script)
        v
    <lang>/data/raw/manual/<code>_manual_combined.jsonl
        |  clean_manual.py
        v
    <lang>/data/processed/<code>_manual_clean.jsonl

That combined JSONL is the only file this script writes. The per-source
register breakdown is printed to stdout for the report to quote.

Why shards at all: a multi-day crawl has to survive interruption, so a finished
shard is never reopened and a killed run costs at most the shard in progress.
That layout is right for crawling and wrong for everything downstream, which
wants one stream per language.

Three guards run during the merge. None is cosmetic -- each protects a number
that ends up in the report:

  * doc_id dedup      A resumed crawl refetches whatever was in flight when it
                      stopped, so one article can legitimately appear in two
                      shards. Counting it twice inflates the manual token share,
                      which is a graded quantity. Measured on the collected
                      corpus this removes 122,534 Hindi records (35% of the raw
                      stream) and 21,149 Nepali ones.
  * language guard    Keep only records whose `lang` belongs to this language.
                      www.bbc.com serves both Hindi and Nepali from one domain
                      and the two share Devanagari, so a misrouted record is a
                      realistic failure that a script-ratio check cannot catch.
  * source_type guard Keep only `manual` records, so nothing downloaded can leak
                      into the file the >=20% manual requirement is measured
                      from.

Token counts here are a deliberately naive regex proxy, not BPE tokens. The real
number only exists once the tokenizer is trained; build_tokenizer.py reports it.

Usage:
    python scripts/combine_manual.py                     # both languages
    python scripts/combine_manual.py --lang hi
    python scripts/combine_manual.py --shards-root ..    # crawl output elsewhere
    python scripts/combine_manual.py --limit 5000        # smoke test
"""

from __future__ import annotations

import argparse
import gzip
import io
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Naive stand-in for the eventual SentencePiece token count: a run of word
# characters is one token, every other non-space character is its own.
# Devanagari is entirely \w, so this approximates whitespace tokenisation with
# punctuation split off -- enough to size the corpus, never a real token count.
TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)

# Every spelling the `lang` field takes across the collected shards. It was
# recorded inconsistently over the collection window: Hindi shards carry both
# "hi" and "hindi", and Nepali shards carry only "nepali", never "ne". Matching
# on the ISO code alone would silently discard 100% of Nepali and ~72% of Hindi.
LANG_ALIASES = {
    "hi": {"hi", "hindi"},
    "ne": {"ne", "nepali"},
}

# repo directory, ISO code, and the crawl directory used before the collection
# scripts lived in this repo.
LANGUAGES = [
    ("hindi", "hi", "Hindi"),
    ("nepali", "ne", "Nepali"),
]


def load_registry(root: Path, folder: str):
    """Read <lang>/data/sources.yaml -> (category_of_name, name_of_domain).

    Two maps, because the shards label `source` inconsistently: the crawler was
    changed part-way through collection, so the same site appears both under its
    registry name ("amarujala") and under its domain ("www.amarujala.com").
    Left alone that splits every site across two rows and halves its apparent
    size, so the domain map folds the spellings back together.

    The registry only labels statistics and never gates the merge, so a missing
    file or missing PyYAML degrades to uncategorised rows.
    """
    path = root / folder / "data" / "sources.yaml"
    if not path.exists():
        print(f"  [warn] no source registry at {path}; rows will be uncategorised")
        return {}, {}
    try:
        import yaml
    except ImportError:
        print("  [warn] PyYAML not installed; rows will be uncategorised")
        return {}, {}

    with open(path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}
    categories, by_domain = {}, {}
    for site in cfg.get("manual_sites", []):
        if not isinstance(site, dict) or "name" not in site:
            continue
        categories[site["name"]] = site.get("category", "uncategorised")
        if site.get("domain"):
            by_domain[site["domain"]] = site["name"]
    return categories, by_domain


def open_shard(path: Path):
    """Open one shard as UTF-8 text, handling .zst / .gz / plain JSONL.

    zstandard is imported lazily because the plain and gzip paths do not need
    it. A .zst shard with the library missing is a hard error rather than a
    silent skip: skipping would quietly shrink the corpus.
    """
    suffix = path.suffix.lower()
    if suffix == ".zst":
        try:
            import zstandard as zstd
        except ImportError:
            sys.exit(
                f"error: {path.name} is zstd-compressed but 'zstandard' is not\n"
                "       installed.  pip install zstandard"
            )
        return io.TextIOWrapper(
            zstd.ZstdDecompressor().stream_reader(open(path, "rb")), encoding="utf-8"
        )
    if suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return open(path, "r", encoding="utf-8")


def find_shards(root: Path, folder: str, crawl_folder: str, shards_root: Path | None):
    """Locate a language's shard directory, or exit listing every path tried.

    Two layouts are supported: scrape_manual.py writes inside the repo, while
    the original collection ran from the working directory one level above it.
    """
    candidates = []
    if shards_root is not None:
        candidates += [shards_root / crawl_folder / "manual", shards_root]
    candidates += [
        root / folder / "data" / "raw" / "manual" / "shards",
        root.parent / crawl_folder / "manual",
    ]

    tried = []
    for cand in candidates:
        tried.append(cand)
        if not cand.is_dir():
            continue
        shards = sorted(
            p for p in cand.iterdir()
            if p.name.startswith("shard-") and ".jsonl" in p.name
        )
        if shards:
            return cand, shards
    listing = "\n".join(f"         {p}" for p in tried)
    sys.exit(
        f"error: no shard-*.jsonl* files found for {folder}. Looked in:\n{listing}\n"
        f"       Run scripts/scrape_manual.py first, or pass --shards-root."
    )


def combine_language(root: Path, folder: str, code: str, crawl_folder: str,
                     shards_root: Path | None, limit: int | None) -> dict:
    """Merge every shard for one language into a single deduplicated JSONL."""
    shard_dir, shards = find_shards(root, folder, crawl_folder, shards_root)
    categories, name_of_domain = load_registry(root, folder)
    aliases = LANG_ALIASES[code]

    # A --limit run is a smoke test, so it must NOT land on the real filename:
    # it would rename a few thousand documents over the full corpus and destroy
    # it. Sample runs get their own path.
    stem = f"{code}_manual_combined" + (".sample" if limit else "")
    out_path = root / folder / "data" / "raw" / "manual" / f"{stem}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Write to .partial and rename only on success, so an interrupted run cannot
    # leave a truncated file that looks complete to clean_manual.py.
    partial = out_path.with_suffix(out_path.suffix + ".partial")

    seen = set()
    kept, words_by_source, dropped = Counter(), Counter(), Counter()
    n_docs = n_chars = n_words = n_tokens = n_bytes = 0
    started = time.time()

    print(f"  shard dir  : {shard_dir}")
    print(f"  shards     : {len(shards)}")

    with open(partial, "w", encoding="utf-8", newline="\n") as out:
        for i, shard in enumerate(shards, 1):
            with open_shard(shard) as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        dropped["malformed_json"] += 1
                        continue

                    text = rec.get("text") or ""
                    if not text.strip():
                        dropped["empty_text"] += 1
                        continue
                    if rec.get("lang") not in aliases:
                        dropped["wrong_language"] += 1
                        continue
                    if rec.get("source_type", "manual") != "manual":
                        dropped["not_manual"] += 1
                        continue

                    # doc_id is the crawler's content hash; fall back to the URL
                    # for any older shard written before it was added.
                    key = rec.get("doc_id") or rec.get("url")
                    if key is not None:
                        if key in seen:
                            dropped["duplicate_doc_id"] += 1
                            continue
                        seen.add(key)

                    out.write(json.dumps(rec, ensure_ascii=False) + "\n")

                    raw_source = rec.get("source", "unknown")
                    source = name_of_domain.get(raw_source, raw_source)
                    words = len(text.split())
                    n_docs += 1
                    n_chars += len(text)
                    n_words += words
                    n_tokens += len(TOKEN_RE.findall(text))
                    n_bytes += len(text.encode("utf-8"))
                    kept[source] += 1
                    words_by_source[source] += words

                    if limit and n_docs >= limit:
                        break

            elapsed = max(time.time() - started, 1e-9)
            print(
                f"  [{i:>4}/{len(shards)}] kept={n_docs:,} "
                f"dropped={sum(dropped.values()):,} ({n_docs / elapsed:,.0f} docs/s)   ",
                end="\r", flush=True,
            )
            if limit and n_docs >= limit:
                print(f"\n  stopping early at --limit {limit:,}")
                break

    print()
    partial.replace(out_path)

    return {
        "language": folder,
        "lang_code": code,
        "shard_dir": str(shard_dir),
        "shards": len(shards),
        "docs": n_docs,
        "chars": n_chars,
        "words": n_words,
        "tokens_naive": n_tokens,
        "bytes_utf8": n_bytes,
        "dropped": dict(dropped),
        "sources_with_documents": len(kept),
        "sources_in_registry_with_no_documents": sorted(set(categories) - set(kept)),
        "per_source": [
            {
                "source": src,
                "category": categories.get(src, "uncategorised"),
                "docs": docs,
                "words": words_by_source[src],
                "doc_share": round(docs / n_docs, 4) if n_docs else 0.0,
            }
            for src, docs in kept.most_common()
        ],
        "out_path": str(out_path),
        "seconds": round(time.time() - started, 1),
    }


# --------------------------------------------------------------------- output
# Reporting is kept separate from the arithmetic above so the numbers can be
# reused (they are written to JSON) without going through the printer.

def print_language_report(stats: dict) -> None:
    """Human-readable summary for one language."""
    print(f"  documents  : {stats['docs']:,}")
    print(f"  characters : {stats['chars']:,}")
    print(f"  words      : {stats['words']:,}")
    print(f"  tokens*    : {stats['tokens_naive']:,}   (*naive proxy, not BPE)")
    if stats["dropped"]:
        reasons = ", ".join(f"{k}={v:,}" for k, v in sorted(stats["dropped"].items()))
        print(f"  dropped    : {sum(stats['dropped'].values()):,}   ({reasons})")
    print(f"  written    : {stats['out_path']}")

    print(f"\n  per source ({stats['sources_with_documents']} with documents):")
    header = f"    {'source':<22}{'category':<18}{'docs':>10}{'words':>16}{'share':>8}"
    print(header)
    print("    " + "-" * (len(header) - 4))
    for row in stats["per_source"]:
        print(
            f"    {row['source']:<22}{row['category']:<18}{row['docs']:>10,}"
            f"{row['words']:>16,}{row['doc_share'] * 100:>7.1f}%"
        )
    silent = stats["sources_in_registry_with_no_documents"]
    if silent:
        print(
            f"\n  [warn] {len(silent)} registry source(s) produced no documents: "
            f"{', '.join(silent)}"
        )


def print_summary(results: list) -> None:
    """Side-by-side totals for both languages."""
    print("\n=== Summary ===")
    header = (
        f"{'Language':<10}{'Shards':>8}{'Docs':>12}"
        f"{'Characters':>16}{'Words':>14}{'Tokens*':>14}"
    )
    print(header)
    print("-" * len(header))
    for s in results:
        print(
            f"{s['language']:<10}{s['shards']:>8}{s['docs']:>12,}"
            f"{s['chars']:>16,}{s['words']:>14,}{s['tokens_naive']:>14,}"
        )
    print(
        "\n*naive regex proxy (word runs + individual punctuation), not BPE tokens.\n"
        " The real per-language token count comes from scripts/build_tokenizer.py\n"
        " once the SentencePiece model is trained."
    )
    print("\nNext stage:  python scripts/clean_manual.py")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--root", default=str(REPO_ROOT),
                    help="repo root containing hindi/ and nepali/")
    ap.add_argument("--shards-root", default=None,
                    help="directory holding <Hindi|Nepali>/manual shard folders "
                         "(default: this repo, then the directory above it)")
    ap.add_argument("--lang", choices=["hi", "ne"], default=None,
                    help="combine one language only (default: both)")
    ap.add_argument("--limit", type=int, default=None,
                    help="stop after N documents per language. Writes to "
                         "<code>_manual_combined.sample.jsonl so a smoke test "
                         "cannot overwrite the real corpus")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    shards_root = Path(args.shards_root).resolve() if args.shards_root else None
    langs = [l for l in LANGUAGES if args.lang is None or l[1] == args.lang]

    results = []
    for folder, code, crawl_folder in langs:
        print(f"\n=== {folder} ({code}) ===")
        stats = combine_language(root, folder, code, crawl_folder, shards_root, args.limit)
        # The per-source breakdown is printed for the report to quote, not
        # written to disk: the combined JSONL is this script's only output.
        print_language_report(stats)
        results.append(stats)

    print_summary(results)


if __name__ == "__main__":
    main()
