"""
Phase 1 - Data Collection (Hindi) - ALL Sources, One Script
--------------------------------------------------------------
This single script downloads every public (non-manual) Hindi text
source we use for Phase 1. It replaces the old download_data.py +
download_hf_sources.py (now combined here).

Sources, in the order they run:
  1. Wikipedia      - official Hindi Wikipedia dump (direct download)
  2. CC-100          - web-crawled Hindi text
  3. IndicCorp v2    - AI4Bharat's Hindi corpus
  4. Sangraha (unverified) - AI4Bharat's raw web corpus
  5. Sangraha (verified)   - AI4Bharat's curated/OCR corpus (much bigger)
  6. mC4             - Hindi slice of Google's "Colossal Clean Crawled Corpus"
  7. MADLAD-400       - Google/AllenAI's cleaned multilingual web corpus
                        (only the "clean" files, not the "noisy" ones)

Sources 2-7 are big, so we "stream" them - reading documents one at a
time instead of downloading the whole dataset to disk first - and stop
after MAX_DOCS_PER_SOURCE documents each.

Every document is tagged with where it came from. All of this counts
as "manual": false, because we did not collect it ourselves - it's all
public data we downloaded. Separate scripts (not this one) handle the
required manual 20% (news scraping, OCR).

Before running, install the extra requirement:
    pip install -r requirements.txt

How to run:
    python download_all.py

Output (in the "raw" folder next to this script):
    raw/hiwiki-latest-pages-articles.xml.bz2   (Wikipedia dump)
    raw/hi_cc100.jsonl
    raw/hi_indiccorp.jsonl
    raw/hi_sangraha.jsonl                       (unverified)
    raw/hi_sangraha_verified.jsonl
    raw/hi_mc4.jsonl
    raw/hi_madlad.jsonl

Each JSONL line looks like:
    {"text": "...", "source": "cc100", "manual": false, "lang": "hi"}

Safe to re-run: every source is skipped once it has a finished output
file, so interrupting this script (Ctrl+C, closed terminal, closed
Colab session) and running it again just continues where it left off,
source by source.

Note on dataset names: Hugging Face dataset paths occasionally change.
If a source below fails to load, this script prints a warning and just
skips to the next source (one broken source should not stop the whole
run). If that happens, check the dataset's page on huggingface.co for
its current name/config and update the SOURCES list below.
"""

import json
import os
import requests
from tqdm import tqdm
from datasets import load_dataset

LANG_CODE = "hi"
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "raw")

# Wikimedia blocks the default requests user-agent, so we send a
# descriptive one instead (this is their standard policy, not a hack).
HEADERS = {"User-Agent": "LMA-college-project/1.0 (student data collection script)"}
WIKI_DUMP_URL = "https://dumps.wikimedia.org/hiwiki/latest/hiwiki-latest-pages-articles.xml.bz2"

# How many documents to pull from EACH streamed source below.
# Raise this if you still need more tokens after checking dataset_stats.py;
# lower it if a source is slow or you just want a quick test run.
MAX_DOCS_PER_SOURCE = 500_000


# ---------------------------------------------------------------------
# Part 1: Wikipedia dump (plain file download, not a Hugging Face source)
# ---------------------------------------------------------------------

def download_wikipedia_dump():
    """Download the official Hindi Wikipedia dump (one big XML file)."""

    save_path = os.path.join(OUT_DIR, WIKI_DUMP_URL.split("/")[-1])

    # Ask the server how big the file is, without downloading it yet
    head = requests.head(WIKI_DUMP_URL, headers=HEADERS, allow_redirects=True)
    head.raise_for_status()
    total_size = int(head.headers.get("content-length", 0))

    # Skip only if we already have a file of the correct, full size.
    # (Dump downloads can take a long time and get interrupted, e.g. by a
    # closed Colab session - this check stops us from mistaking a
    # half-downloaded file for a finished one.)
    if os.path.exists(save_path) and os.path.getsize(save_path) == total_size:
        print(f"Already downloaded, skipping: {save_path}")
        return

    print("Downloading Hindi Wikipedia dump...")
    response = requests.get(WIKI_DUMP_URL, stream=True, headers=HEADERS)
    response.raise_for_status()

    with open(save_path, "wb") as f, tqdm(
        total=total_size, unit="B", unit_scale=True, desc=os.path.basename(save_path)
    ) as progress_bar:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            f.write(chunk)
            progress_bar.update(len(chunk))

    print(f"Done. Saved to: {save_path}")


# ---------------------------------------------------------------------
# Part 2: Hugging Face streamed sources
# ---------------------------------------------------------------------

def load_cc100():
    """CC-100: web-crawled text.

    The normal way to load this ("cc100", lang="hi") runs a custom
    Python loading script, but Hugging Face's datasets library no longer
    allows that (security policy change). Instead, we load Hugging
    Face's own auto-converted Parquet copy of the same data directly -
    same text, just stored as Parquet files instead of a script.
    """
    return load_dataset(
        "statmt/cc100", data_files="hi/train/*.parquet",
        revision="refs/convert/parquet", split="train", streaming=True,
    )


def load_indiccorp():
    """IndicCorp v2 (AI4Bharat): Hindi is stored as three plain .txt files
    (data/hi-1.txt, hi-2.txt, hi-3.txt) directly in the repo - we point at
    them by name instead of using a data_dir/config shortcut, since that
    shortcut's folder name does not actually match the real files."""
    return load_dataset(
        "ai4bharat/IndicCorpV2",
        data_files=["data/hi-1.txt", "data/hi-2.txt", "data/hi-3.txt"],
        split="train", streaming=True,
    )


def load_sangraha_unverified():
    """Sangraha (AI4Bharat): 'unverified' split is plain web text."""
    return load_dataset(
        "ai4bharat/sangraha", data_dir="unverified/hin",
        split="train", streaming=True,
    )


def load_sangraha_verified():
    """Sangraha (AI4Bharat): 'verified' split - curated text (includes
    OCR/scraped sources AI4Bharat collected, not us, so it still counts
    as downloaded). Much larger than 'unverified'."""
    return load_dataset(
        "ai4bharat/sangraha", data_dir="verified/hin",
        split="train", streaming=True,
    )


def load_mc4():
    """mC4, via the allenai/c4 repo: same fix as CC-100 - load the
    auto-converted Parquet copy directly instead of running a script."""
    return load_dataset(
        "allenai/c4", data_files="hi/partial-train/*.parquet",
        revision="refs/convert/parquet", split="train", streaming=True,
    )


def load_madlad():
    """MADLAD-400 (Google/AllenAI): the dataset ships pre-split into
    'clean' and 'noisy' files by the dataset creators - we only use the
    'clean' ones."""
    return load_dataset(
        "allenai/MADLAD-400", data_files="data/hi/hi_clean_*.jsonl.gz",
        split="train", streaming=True,
    )


# List of (short name, loader function). Add/remove sources here.
SOURCES = [
    ("cc100", load_cc100),
    ("indiccorp", load_indiccorp),
    ("sangraha", load_sangraha_unverified),
    ("sangraha_verified", load_sangraha_verified),
    ("mc4", load_mc4),
    ("madlad", load_madlad),
]


def download_source(name, loader_fn, out_path, max_docs):
    """Stream one Hugging Face dataset and save its text lines as JSONL."""

    # If a finished file already exists, don't redo the work
    if os.path.exists(out_path):
        print(f"Already downloaded, skipping: {out_path}")
        return

    print(f"Starting source: {name}")
    try:
        dataset = loader_fn()
    except Exception as e:
        print(f"[warn] could not load source '{name}': {e}")
        return

    # Write to a ".partial" file first. Only rename it to the real name
    # once we're done - this way, if the run gets interrupted halfway,
    # the next run sees no finished file and safely starts over instead
    # of thinking a half-written file is complete.
    partial_path = out_path + ".partial"
    n = 0
    with open(partial_path, "w", encoding="utf-8") as f:
        try:
            for example in dataset:
                text = (example.get("text") or example.get("content") or "").strip()
                if not text:
                    continue
                row = {"text": text, "source": name, "manual": False, "lang": LANG_CODE}
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                n += 1
                if n % 20000 == 0:
                    print(f"  {name}: {n} docs written")
                if n >= max_docs:
                    break
        except Exception as e:
            print(f"[warn] source '{name}' stopped early after {n} docs: {e}")

    os.rename(partial_path, out_path)
    print(f"Done: {name} -> {n} docs -> {out_path}")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    download_wikipedia_dump()

    for name, loader_fn in SOURCES:
        out_path = os.path.join(OUT_DIR, f"{LANG_CODE}_{name}.jsonl")
        download_source(name, loader_fn, out_path, MAX_DOCS_PER_SOURCE)

    print("\nAll sources attempted. Run dataset_stats.py later to check total tokens.")


if __name__ == "__main__":
    main()
