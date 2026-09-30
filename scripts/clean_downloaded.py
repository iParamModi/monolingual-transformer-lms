#!/usr/bin/env python3
"""Clean + normalize the auto-downloaded (non-manual) corpus for Hindi and
Nepali, before SentencePiece BPE training (Phase 1, LMA_Individual_Project_v1.pdf).

Paths (all relative to the repo root, which --root defaults to):
    input   <lang>/data/raw/*.jsonl        (flat files written by
                                            <lang>/data/download_all.py: CC-100,
                                            IndicCorp v2, Sangraha unverified+
                                            verified, mC4, MADLAD-400)
    interim <lang>/data/interim/downloaded/0*.jsonl   (per-stage checkpoints)
    output  <lang>/data/processed/<code>_downloaded_clean.jsonl
where <lang> is hindi|nepali and <code> is hi|ne. Each input line is
{"text": str, "source": str, "manual": false, "lang": "hi"|"ne"}.

Note the raw/ glob is non-recursive, so the manual corpus staged alongside it
in <lang>/data/raw/manual/ (scripts/clean_manual.py's input) is never picked
up here -- the two pipelines stay cleanly separated.

Wikipedia dumps (*wiki*.xml.bz2) are DELIBERATELY EXCLUDED -- they're a
different format (compressed XML, needs WikiExtractor) and out of scope per
instruction. This script only ever looks at *.jsonl files in raw/, so the
dump files are never touched; empty (0-byte, failed-source) and *.partial
(still-downloading) files are skipped automatically and reported.

This is the DOWNLOADED (manual=false) side of the corpus, the complement of
scripts/clean_manual.py. The two share almost all cleaning logic (same
functions, same design), but thresholds differ deliberately:

  - We have far more raw volume here (tens of GB vs hundreds of MB for the
    manual slice) and no fragile ~100M-token floor to protect -- the ask was
    "I have a lot of tokens", so every filter below is picked on the STRICT
    side, not the lenient side. See stage_quality_filter for exact deltas
    vs. clean_manual.py.
  - Multiple named sources per language, and CC-100 / mC4 / MADLAD-400 /
    Sangraha-unverified are ALL substantially Common-Crawl-derived, so heavy
    cross-source duplication is expected and dedup matters far more here.
  - Scale changes the near-dedup cost/benefit. The first version of this stage
    used datasketch MinHash objects held in an in-memory LSH index, which at
    this document count would exhaust RAM long before it finished. It has been
    rewritten (see stage_near_dedup) to stream signatures into an on-disk
    numpy memmap and do band grouping with vectorised sorts, so memory stays
    flat and the work is resumable. It is still the slowest stage by a wide
    margin -- budget hours, not minutes -- but it runs to completion.

Pipeline (each stage reads the previous stage's output JSONL and writes a new
one, so any stage can be re-run/resumed independently):

    0 discover              -- find raw/*.jsonl (excluding wiki/empty/partial)
                                and report what's included/excluded. Sources are
                                streamed as one logical corpus, so there is no
                                physical concatenation pass.
    1 unicode_normalize     -- NFC + Indic normalize, strip control/zero-width
                                chars, strip URLs/HTML tags+entities/emails/
                                @handles, zero-English pass (blank every bare
                                Latin-letter run, keep digits), repair danda
                                punctuation, collapse whitespace. ALSO collects
                                the raw "before" statistics on the way past,
                                which used to be its own full streaming pass.
    2 lang_id_filter        -- drop documents fastText's LID model does not
                                classify as this language, above threshold.
                                Stricter default threshold than the manual
                                script (0.6 vs 0.5) since "Hindi should
                                contain only Hindi, Nepali only Nepali" and
                                we have surplus volume to afford being picky.
                                Optional dependency; falls back to pass-through
                                with a warning if fasttext/model is missing.
    3 boilerplate_removal   -- OFF BY DEFAULT (--boilerplate to enable). These
                                corpora store each document as a single line,
                                measured at 1.0 non-empty line/doc, so a stage
                                that removes frequently-recurring *lines* has
                                nothing to grip: it costs two full passes over
                                13+ GB and removes essentially nothing. What it
                                could legitimately catch, exact_dedup catches.
    4 quality_filter        -- Gopher-style filters, all tunable from the CLI:
                                min/max word count, min Devanagari-script
                                ratio, min sentence-final punctuation ratio
                                (default 0.0 -- see --min-terminal-frac), max
                                within-doc duplicate line fraction, and a
                                binary-payload signature check
    5 exact_dedup           -- drop exact (whitespace-normalized) duplicate
                                documents via hash -- also the main defense
                                against CC-100/mC4/MADLAD/Sangraha overlap
    6 perplexity_filter     -- score every document with an n-gram LM trained
                                FROM SCRATCH on a clean reference corpus, then
                                drop the high-perplexity tail (machine-
                                translated / ungrammatical / gibberish web
                                junk that all the rule-based filters above
                                let through). See stage_perplexity_filter.
    7 near_dedup            -- drop near-duplicate documents via banded
                                MinHash LSH. Reimplemented to actually scale
                                to this corpus: numpy-vectorised signatures
                                cached to an on-disk memmap, sort-based band
                                grouping, no per-document Python objects.
                                Still the slowest stage, but it now finishes
                                instead of exhausting RAM.
    8 report                -- final counts, retention %, per-source
                                breakdown, and a check against the downloaded-
                                corpus token floor (default 400M) using an
                                estimated BPE fertility range

Deliberately NOT done here (same as clean_manual.py, left for later steps):
    - cross-corpus (Hindi vs. Nepali) overlap removal
    - train/val/test splitting -- split before BPE training, since
      SentencePiece must only ever see the train split
    - merging with the manual-collection output (scripts/clean_manual.py) --
      that merge + the final combined split/dedup pass happens after both
      sides are independently cleaned

After every stage: documents / characters / words / naive-tokens remaining,
how many were dropped and why, time taken, and a live "which step is running
+ how much longer" progress readout, same mechanism as clean_manual.py.

Usage (run from anywhere; --root defaults to the repo root):
    python scripts/clean_downloaded.py
    python scripts/clean_downloaded.py --lang hi --skip-near-dedup
    python scripts/clean_downloaded.py --delete-intermediate   # reclaim disk as it goes

Dependencies:
    numpy                              # REQUIRED (already in requirements.txt)
                                        # -- stages 6 and 7 are built on it
    pip install indic-nlp-library      # optional, stage 1: nukta/vowel-sign
                                        # normalization (falls back to NFC)
    pip install fasttext-wheel         # optional, stage 2: LID

    Stage 2 also needs the pretrained fastText LID model file (one-time
    download, this is a LID classifier, not a pretrained LM/tokenizer, so
    it's on the project's allowed-tools list):
        https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin

    datasketch is NO LONGER needed -- stage 7 implements banded MinHash LSH
    directly on numpy so it can scale to this corpus.

No pretrained language model is used anywhere. The perplexity filter (stage 6)
trains its own n-gram LM from scratch on our own data, which the project rules
allow; downloading a pretrained LM to score text would not be.

Disk note: at tens of GB of raw input, keeping every stage's intermediate
file (clean_manual.py's default behavior) can require several times the raw
input size in scratch disk. Pass --delete-intermediate to remove each stage's
input file once the next stage has successfully written its own output.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
import time
import unicodedata
import zlib
from collections import Counter
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np

# --------------------------------------------------------------------------
# Text-level constants (identical to scripts/clean_manual.py)
# --------------------------------------------------------------------------

DEVA_RE = re.compile(r"[\u0900-\u097F]")
DEVA_TERMINAL_RE = re.compile(r"[।॥?!]\s*$")
CTRL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")
ZERO_WIDTH_RE = re.compile(r"[\u200b\u200c\u200d\ufeff]")
MULTI_SPACE_RE = re.compile(r"[ \t]+")
MULTI_BLANK_LINE_RE = re.compile(r"\n{3,}")
DOUBLE_DANDA_RE = re.compile(r"\|\|")
SINGLE_PIPE_RE = re.compile(r"(?<!\|)\|(?!\|)")

# Web/markup artifacts, merged into ONE alternation. These were six separate
# re.sub() calls; since they all collapse to the same replacement (a space),
# a single pass over the string does the same job. unicode_normalize is the
# heaviest stage in the pipeline precisely because it touches the full
# uncompressed corpus, so cutting six passes to one is the single best
# speedup available here. Order matters: URL and EMAIL come before the bare
# @handle rule so a full address is consumed as one unit.
WEB_ARTIFACT_RE = re.compile(
    r"<[^>]+>"                      # HTML tags
    r"|&#\d+;"                      # numeric entities
    r"|&[a-zA-Z]+;"                 # named entities
    r"|(?:https?://|www\.)\S+"      # URLs
    r"|\S+@\S+\.\S+"                # emails
    r"|(?<!\w)@\w+",                # @handles
    re.IGNORECASE,
)
HASHTAG_MARK_RE = re.compile(r"(?<!\w)#(?=\w)")
LATIN_ALPHA_RE = re.compile(r"[A-Za-z]+")  # zero-English rule; digits kept

BINARY_SIGNATURES = ("JFIF", "Exif", "%PDF", "Photoshop")

try:
    from indicnlp.normalize.indic_normalize import IndicNormalizerFactory

    _NORM_FACTORY = IndicNormalizerFactory()

    def _make_indic_normalizer(lang_code: str):
        try:
            return _NORM_FACTORY.get_normalizer(lang_code)
        except Exception:
            return _NORM_FACTORY.get_normalizer("hi")

except Exception:
    def _make_indic_normalizer(lang_code: str):
        return None


# --------------------------------------------------------------------------
# Stats helpers
# --------------------------------------------------------------------------

class Accumulator:
    """Running docs/chars/words totals, optionally split by source.

    Note what is NOT here: the old "naive token" count. It ran
    TOKEN_RE.findall() on every document at every stage -- allocating a
    ~1000-element list per document -- and the number it produced was
    misleading anyway (~3.4x the word count, because the virama splits
    Devanagari conjuncts into separate matches). Nothing depends on it: the
    BPE projection in stage_report is words x fertility, and the real token
    count only exists once SentencePiece has actually been trained. Dropping
    it removes a full regex scan per document per stage.

    Tracking per-source totals here (rather than re-reading the finished file
    at the end) removes one more full pass over the output.
    """

    __slots__ = ("docs", "chars", "words", "by_source")

    def __init__(self) -> None:
        self.docs = self.chars = self.words = 0
        self.by_source: dict[str, list[int]] = {}

    def add(self, text: str, source: str | None = None) -> None:
        c = len(text)
        w = len(text.split())
        self.docs += 1
        self.chars += c
        self.words += w
        if source is not None:
            e = self.by_source.get(source)
            if e is None:
                self.by_source[source] = [1, c, w]
            else:
                e[0] += 1
                e[1] += c
                e[2] += w

    def as_dict(self) -> dict:
        return {"docs": self.docs, "chars": self.chars, "words": self.words}


def fmt_time(seconds: float) -> str:
    if seconds != seconds or seconds < 0:
        return "--"
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m{s:02d}s"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


def fmt_stats(label: str, acc: "Accumulator | dict") -> str:
    d = acc.as_dict() if isinstance(acc, Accumulator) else acc
    return (
        f"{label:<24} docs={d['docs']:>10,}  chars={d['chars']:>14,}  "
        f"words={d['words']:>13,}"
    )


def print_by_source(acc: Accumulator) -> None:
    for src, (n, c, w) in sorted(acc.by_source.items(), key=lambda kv: -kv[1][2]):
        print(f"    {src:<20} docs={n:>10,}  chars={c:>14,}  words={w:>13,}")


# --------------------------------------------------------------------------
# Progress / ETA tracking (identical mechanism to clean_manual.py)
# --------------------------------------------------------------------------

class PipelineProgress:
    def __init__(self, lang_label: str, step_weights: "dict[str, float]") -> None:
        self.lang_label = lang_label
        self.weights = step_weights
        self.order = list(step_weights)
        self.total_weight = sum(step_weights.values())
        self.completed_weight = 0.0
        self.t_start = time.perf_counter()
        self._current: str | None = None
        self._step_t0: float = 0.0

    def start_step(self, name: str, note: str = "") -> None:
        self._current = name
        self._step_t0 = time.perf_counter()
        idx = self.order.index(name) + 1
        suffix = f" -- {note}" if note else ""
        print(f"\n>>> [{self.lang_label}] step {idx}/{len(self.order)} '{name}' STARTING{suffix}")

    def tick(self, done: int, total: int, unit: str = "docs") -> None:
        name = self._current
        assert name is not None
        frac_step = (done / total) if total else 1.0
        frac_overall = (self.completed_weight + self.weights[name] * frac_step) / self.total_weight
        elapsed_total = time.perf_counter() - self.t_start
        eta_total = (elapsed_total / frac_overall - elapsed_total) if frac_overall > 0 else float("nan")
        elapsed_step = time.perf_counter() - self._step_t0
        rate = done / elapsed_step if elapsed_step > 0 else 0.0
        sys.stdout.write(
            f"\r    [{self.lang_label}:{name}] {done:,}/{total:,} {unit} "
            f"({100 * frac_step:5.1f}%) | {rate:,.0f} {unit}/s | "
            f"pipeline elapsed {fmt_time(elapsed_total)} | "
            f"pipeline ETA remaining {fmt_time(eta_total)}   "
        )
        sys.stdout.flush()

    def end_step(self, name: str) -> None:
        self.completed_weight += self.weights[name]
        dt = time.perf_counter() - self._step_t0
        elapsed_total = time.perf_counter() - self.t_start
        print(f"\n<<< [{self.lang_label}] step '{name}' DONE in {fmt_time(dt)} (pipeline elapsed {fmt_time(elapsed_total)})")


def iter_progress(items: Iterable, total: int, progress: PipelineProgress, update_every: int = 20_000) -> Iterator:
    done = 0
    for item in items:
        yield item
        done += 1
        if done % update_every == 0 or done == total:
            progress.tick(done, total)


def count_lines(path: Path) -> int:
    n = 0
    with open(path, "r", encoding="utf-8") as fh:
        for _ in fh:
            n += 1
    return n


def read_jsonl(path: Path) -> Iterator[dict]:
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue  # tolerate the odd truncated/corrupt line in huge downloads


# --------------------------------------------------------------------------
# Stage 0: source discovery (multi-file, no physical combine pass)
# --------------------------------------------------------------------------

def discover_source_files(raw_dir: Path) -> tuple[list[Path], list[tuple[Path, str]]]:
    """Find raw/*.jsonl files to process, explicitly excluding Wikipedia dumps
    (different format, out of scope) and skipping empty/failed sources."""
    kept: list[Path] = []
    skipped: list[tuple[Path, str]] = []
    for p in sorted(raw_dir.glob("*.jsonl")):
        if "wiki" in p.name.lower():
            skipped.append((p, "Wikipedia dump -- excluded per instruction"))
            continue
        size = p.stat().st_size
        if size == 0:
            skipped.append((p, "empty (0 bytes) -- failed or not-yet-run source"))
            continue
        kept.append(p)
    return kept, skipped


def read_jsonl_multi(paths: Iterable[Path]) -> Iterator[dict]:
    for p in paths:
        yield from read_jsonl(p)


def count_lines_multi(paths: Iterable[Path]) -> int:
    return sum(count_lines(p) for p in paths)


# --------------------------------------------------------------------------
# Stage 1: Unicode / Indic normalization + artifact stripping
# --------------------------------------------------------------------------

def strip_web_artifacts(text: str) -> str:
    text = WEB_ARTIFACT_RE.sub(" ", text)  # tags/entities/URLs/emails/handles
    text = HASHTAG_MARK_RE.sub("", text)   # keep the word, drop the '#'
    text = LATIN_ALPHA_RE.sub(" ", text)   # zero-English rule
    return text


def clean_text_unicode(text: str, indic_normalizer) -> str:
    text = unicodedata.normalize("NFC", text)
    if indic_normalizer is not None:
        text = indic_normalizer.normalize(text)
    text = CTRL_RE.sub(" ", text)
    text = ZERO_WIDTH_RE.sub("", text)
    text = strip_web_artifacts(text)
    text = DOUBLE_DANDA_RE.sub("॥", text)
    text = SINGLE_PIPE_RE.sub("।", text)
    text = MULTI_SPACE_RE.sub(" ", text)
    text = MULTI_BLANK_LINE_RE.sub("\n\n", text)
    return text.strip()


def stage_unicode_normalize(
    in_paths: list[Path], out_path: Path, lang_code: str, progress: PipelineProgress
) -> tuple[Accumulator, Accumulator]:
    """Normalize the raw corpus AND collect the raw "before" statistics.

    The baseline numbers used to come from a dedicated stage that streamed the
    entire raw corpus purely to count things -- ~42 minutes of pure measurement
    on the Hindi side. This stage already reads exactly those bytes, so the raw
    stats are accumulated here (before cleaning each document) for free. Returns
    (baseline_stats, cleaned_stats).
    """
    progress.start_step("unicode_normalize", note=f"{len(in_paths)} source file(s), also collects baseline stats")
    total = count_lines_multi(in_paths)
    indic_normalizer = _make_indic_normalizer(lang_code)
    baseline = Accumulator()
    acc = Accumulator()
    dropped_empty = 0
    with open(out_path, "w", encoding="utf-8") as out:
        for doc in iter_progress(read_jsonl_multi(in_paths), total, progress):
            source = doc.get("source", "unknown")
            baseline.add(doc.get("text", ""), source)
            doc["text"] = clean_text_unicode(doc.get("text", ""), indic_normalizer)
            if not doc["text"]:
                dropped_empty += 1
                continue
            acc.add(doc["text"], source)
            out.write(json.dumps(doc, ensure_ascii=False) + "\n")
    progress.end_step("unicode_normalize")
    print(f"    dropped (empty after normalize): {dropped_empty:,}")
    print(fmt_stats("baseline (raw)", baseline))
    print_by_source(baseline)
    return baseline, acc


# --------------------------------------------------------------------------
# Stage 2: language-ID filter
# --------------------------------------------------------------------------

_FASTTEXT_MODEL_CACHE: dict = {}


def _load_fasttext_model(model_path: Path):
    key = str(model_path)
    if key in _FASTTEXT_MODEL_CACHE:
        return _FASTTEXT_MODEL_CACHE[key]
    model = None
    if not model_path.exists():
        print(f"    [skip] fastText LID model not found at {model_path}")
    else:
        try:
            import fasttext

            model = fasttext.load_model(str(model_path))
        except Exception as e:
            print(f"    [skip] could not load fastText LID model ({model_path}): {e}")
    _FASTTEXT_MODEL_CACHE[key] = model
    return model


def _predict_top_label(model, text: str) -> tuple[str, float]:
    """Top (label, probability) for one document.

    Deliberately bypasses fasttext's own .predict(): as of fasttext-wheel 0.9.2
    it ends with `np.array(probs, copy=False)`, which NumPy >= 2 rejects
    outright (ValueError: Unable to avoid copy). The raw C++ binding underneath
    returns plain (prob, label) tuples and is version-independent. The
    .predict() path is kept as a fallback for builds without `.f`.
    """
    try:
        res = model.f.predict(text, 1, 0.0, "strict")
    except AttributeError:
        labels, probs = model.predict(text, k=1)
        return (labels[0], float(probs[0])) if len(labels) else ("", 0.0)
    if not res:
        return "", 0.0
    prob, label = res[0]
    return label, float(prob)


def stage_lang_id_filter(
    in_path: Path,
    out_path: Path,
    progress: PipelineProgress,
    lang_code: str,
    model_path: Path,
    threshold: float,
    total: int,
) -> Accumulator:
    # `total` is passed in from the previous stage's Accumulator rather than
    # recomputed with count_lines(). Every stage used to re-read its entire
    # input just to get a denominator for the progress bar -- six redundant
    # full passes over multi-GB files across the pipeline.
    progress.start_step("lang_id_filter")
    model = _load_fasttext_model(model_path)
    acc = Accumulator()

    if model is None:
        print(
            "    [skip] fastText LID unavailable; copying through unchanged "
            "(pip install fasttext-wheel + download lid.176.bin to enable). "
            "WARNING: without this, wrong-language contamination from "
            "CC-100/mC4/MADLAD/Sangraha will NOT be filtered."
        )
        with open(out_path, "w", encoding="utf-8") as out:
            for doc in iter_progress(read_jsonl(in_path), total, progress):
                acc.add(doc.get("text", ""), doc.get("source", "unknown"))
                out.write(json.dumps(doc, ensure_ascii=False) + "\n")
        progress.end_step("lang_id_filter")
        return acc

    want = "__label__" + lang_code
    dropped = 0
    other_labels: Counter = Counter()
    with open(out_path, "w", encoding="utf-8") as out:
        for doc in iter_progress(read_jsonl(in_path), total, progress):
            text = doc.get("text", "")
            sample = text.replace("\n", " ")[:1000]
            label, prob = _predict_top_label(model, sample)
            if label == want and prob >= threshold:
                acc.add(text, doc.get("source", "unknown"))
                out.write(json.dumps(doc, ensure_ascii=False) + "\n")
            else:
                dropped += 1
                other_labels[label or "?"] += 1
    progress.end_step("lang_id_filter")
    print(f"    dropped (not-{lang_code} per fastText LID, threshold={threshold}): {dropped:,}")
    if other_labels:
        top = ", ".join(f"{lbl}={n:,}" for lbl, n in other_labels.most_common(5))
        print(f"    top predicted labels among dropped: {top}")
    return acc


# --------------------------------------------------------------------------
# Stage 3: boilerplate removal (two passes, per named source)
# --------------------------------------------------------------------------

def _hash_line(line: str) -> str:
    return hashlib.blake2b(line.strip().encode("utf-8"), digest_size=8).hexdigest()


def stage_boilerplate_removal(
    in_path: Path,
    out_path: Path,
    progress: PipelineProgress,
    total: int,
    line_freq_threshold: float = 0.02,
) -> Accumulator:
    # OFF BY DEFAULT -- enable with --boilerplate. Measurement on the real
    # corpus showed 1.0 non-empty line per document: these public web corpora
    # store each document as a single unbroken blob. This stage flags a *line*
    # that recurs in >2% of a source's documents, so with one line per document
    # it would have to see the same 500-word article repeated across 2% of the
    # source before removing anything -- which never happens. It was costing two
    # full passes over 13+ GB to delete essentially nothing, and anything it
    # could legitimately catch is caught properly by exact_dedup downstream.
    progress.start_step(
        "boilerplate_removal",
        note="opt-in; near no-op on single-line documents",
    )

    line_counts: dict[tuple[str, str], int] = Counter()
    docs_per_source: dict[str, int] = Counter()
    for doc in iter_progress(read_jsonl(in_path), total, progress):
        source = doc.get("source", "unknown")
        docs_per_source[source] += 1
        seen_in_doc = set()
        for line in doc.get("text", "").split("\n"):
            line = line.strip()
            if not line:
                continue
            h = _hash_line(line)
            if h in seen_in_doc:
                continue
            seen_in_doc.add(h)
            line_counts[(source, h)] += 1

    boilerplate_hashes: dict[str, set] = {}
    for (source, h), n in line_counts.items():
        if n / max(docs_per_source[source], 1) > line_freq_threshold:
            boilerplate_hashes.setdefault(source, set()).add(h)

    acc = Accumulator()
    dropped_empty = 0
    with open(out_path, "w", encoding="utf-8") as out:
        for doc in iter_progress(read_jsonl(in_path), total, progress):
            source = doc.get("source", "unknown")
            bad = boilerplate_hashes.get(source, set())
            kept_lines = [
                line for line in doc.get("text", "").split("\n")
                if _hash_line(line) not in bad or not line.strip()
            ]
            doc["text"] = "\n".join(kept_lines).strip()
            doc["text"] = MULTI_BLANK_LINE_RE.sub("\n\n", doc["text"])
            if not doc["text"]:
                dropped_empty += 1
                continue
            acc.add(doc["text"], source)
            out.write(json.dumps(doc, ensure_ascii=False) + "\n")

    progress.end_step("boilerplate_removal")
    n_boiler_lines = sum(len(v) for v in boilerplate_hashes.values())
    print(f"    boilerplate line patterns removed: {n_boiler_lines:,} (across {len(boilerplate_hashes):,} sources)")
    print(f"    dropped (empty after strip)      : {dropped_empty:,}")
    return acc


# --------------------------------------------------------------------------
# Stage 4: Gopher-style quality filter -- STRICT defaults
# --------------------------------------------------------------------------

def deva_ratio(text: str) -> float:
    chars = [c for c in text if not c.isspace()]
    if not chars:
        return 0.0
    return sum(1 for c in chars if DEVA_RE.match(c)) / len(chars)


def terminal_frac(text: str) -> float:
    lines = [l for l in text.split("\n") if l.strip()]
    if not lines:
        return 0.0
    return sum(1 for l in lines if DEVA_TERMINAL_RE.search(l)) / len(lines)


def dup_line_frac(text: str) -> float:
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    if not lines:
        return 0.0
    counts = Counter(lines)
    dup = sum(c for c in counts.values() if c > 1)
    return dup / len(lines)


def looks_binary(text: str) -> bool:
    return any(sig in text for sig in BINARY_SIGNATURES)


def quality_reject_reason(
    text: str,
    min_words: int,
    max_words: int,
    min_script_ratio: float,
    min_terminal_frac: float,
    max_dup_line_frac: float,
) -> str | None:
    if looks_binary(text):
        return "binary_payload"
    n_words = len(text.split())
    if n_words < min_words:
        return "too_short"
    if n_words > max_words:
        return "too_long"
    if deva_ratio(text) < min_script_ratio:
        return "low_script_ratio"
    if terminal_frac(text) < min_terminal_frac:
        return "low_terminal_punct"
    if dup_line_frac(text) > max_dup_line_frac:
        return "repetitive_lines"
    return None


def stage_quality_filter(
    in_path: Path,
    out_path: Path,
    progress: PipelineProgress,
    total: int,
    min_words: int,
    max_words: int,
    min_script_ratio: float,
    min_terminal_frac: float,
    max_dup_line_frac: float,
) -> Accumulator:
    progress.start_step("quality_filter")
    acc = Accumulator()
    reasons: Counter = Counter()
    with open(out_path, "w", encoding="utf-8") as out:
        for doc in iter_progress(read_jsonl(in_path), total, progress):
            text = doc.get("text", "")
            reason = quality_reject_reason(
                text, min_words, max_words, min_script_ratio, min_terminal_frac, max_dup_line_frac
            )
            if reason is not None:
                reasons[reason] += 1
                continue
            acc.add(text, doc.get("source", "unknown"))
            out.write(json.dumps(doc, ensure_ascii=False) + "\n")
    progress.end_step("quality_filter")
    kept = acc.docs
    for reason, n in reasons.most_common():
        print(f"    dropped ({reason}): {n:,}")
    print(f"    kept: {kept:,} / {kept + sum(reasons.values()):,}")
    return acc


# --------------------------------------------------------------------------
# Stage 5: exact dedup (also the main defense against cross-source overlap)
# --------------------------------------------------------------------------

def norm_key(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def stage_exact_dedup(
    in_path: Path, out_path: Path, progress: PipelineProgress, total: int
) -> Accumulator:
    progress.start_step(
        "exact_dedup",
        note="also catches most CC-100/mC4/MADLAD/Sangraha Common-Crawl overlap",
    )
    seen: set = set()
    acc = Accumulator()
    dropped = 0
    with open(out_path, "w", encoding="utf-8") as out:
        for doc in iter_progress(read_jsonl(in_path), total, progress):
            text = doc.get("text", "")
            # digest_size=8 (was 16): a 64-bit digest over a few million
            # documents has a collision probability far below the error rate of
            # every other filter here, and halves the set's memory footprint.
            h = hashlib.blake2b(norm_key(text).encode("utf-8"), digest_size=8).digest()
            if h in seen:
                dropped += 1
                continue
            seen.add(h)
            acc.add(text, doc.get("source", "unknown"))
            out.write(json.dumps(doc, ensure_ascii=False) + "\n")
    progress.end_step("exact_dedup")
    print(f"    dropped (exact duplicate): {dropped:,}")
    return acc


# --------------------------------------------------------------------------
# Stage 6: perplexity filter (n-gram LM trained from scratch)
# --------------------------------------------------------------------------
#
# The rule-based filters above are all *surface* checks -- length, script
# ratio, punctuation, duplicate lines. None of them can tell fluent Hindi from
# text that is grammatically broken, machine-translated, or keyword-spam that
# happens to be 100% Devanagari and correctly punctuated. Web corpora
# (CC-100/mC4/MADLAD/Sangraha-unverified) contain a lot of exactly that, and it
# is the single worst thing to feed a tokenizer: BPE will happily learn merges
# for junk collocations.
#
# The standard fix (CCNet) is to score every document with an n-gram LM built
# on known-good text and drop the high-perplexity tail. Two project-specific
# constraints shape the implementation:
#
#   1. NO PRETRAINED LMs. The LM here is a word n-gram model trained from
#      scratch, in-process, on our own data -- nothing is downloaded.
#   2. Reference corpus choice. By default we train on that language's own
#      CLEANED MANUAL corpus (processed/<code>_manual_clean.jsonl): it is
#      human-curated, already through the full manual pipeline, and in the
#      right language. HONEST CAVEAT: our manual corpus is news/caption-heavy,
#      so it under-represents encyclopedic and long-form registers. Documents
#      in those registers will score somewhat worse than they deserve. That is
#      why the default cutoff is a *percentile* (drop the worst tail) rather
#      than an absolute perplexity number -- a percentile adapts to whatever
#      the score distribution actually looks like, instead of hard-coding a
#      constant that was tuned on somebody else's corpus.
#      Use --ppl-reference self to train on a sample of the input instead.

_WORD_SPLIT_RE = re.compile(r"\S+")


class NgramLM:
    """Word n-gram LM with stupid backoff, trained from scratch.

    Deliberately small and simple: this is a *filter*, not a model we ship.
    Order 2 (bigram) by default -- unigram misses word-order breakage, and
    trigram costs more memory/time than the extra signal is worth here.
    """

    def __init__(self, order: int = 2, alpha: float = 0.4) -> None:
        self.order = max(1, order)
        self.alpha = alpha  # stupid-backoff discount
        self.counts: list[Counter] = [Counter() for _ in range(self.order + 1)]
        self.total_tokens = 0
        self.vocab_size = 1

    def train(self, texts: Iterable[str], max_tokens: int) -> None:
        seen = 0
        for text in texts:
            toks = _WORD_SPLIT_RE.findall(text)
            if not toks:
                continue
            self.counts[1].update(toks)
            for n in range(2, self.order + 1):
                if len(toks) >= n:
                    self.counts[n].update(
                        " ".join(toks[i:i + n]) for i in range(len(toks) - n + 1)
                    )
            seen += len(toks)
            if seen >= max_tokens:
                break
        self.total_tokens = max(sum(self.counts[1].values()), 1)
        # Prune hapax higher-order n-grams: they are mostly noise and dominate
        # memory (a bigram table over tens of millions of tokens is huge).
        for n in range(2, self.order + 1):
            self.counts[n] = Counter({k: v for k, v in self.counts[n].items() if v > 1})
        self.vocab_size = max(len(self.counts[1]), 1)

    def is_trained(self) -> bool:
        return self.total_tokens > 1 and self.vocab_size > 1

    def _unigram_logprob(self, tok: str) -> float:
        # add-1 smoothing so OOV words get a finite, low probability
        c = self.counts[1].get(tok, 0)
        return math.log((c + 1.0) / (self.total_tokens + self.vocab_size))

    def perplexity(self, text: str, max_words: int) -> float:
        toks = _WORD_SPLIT_RE.findall(text)
        if not toks:
            return float("inf")
        if len(toks) > max_words:
            toks = toks[:max_words]
        total_lp = 0.0
        for i, tok in enumerate(toks):
            lp = None
            if self.order >= 2 and i > 0:
                ctx = toks[i - 1]
                bg = self.counts[2].get(ctx + " " + tok, 0)
                if bg:
                    ctx_c = self.counts[1].get(ctx, 0)
                    if ctx_c:
                        lp = math.log(bg / ctx_c)
            if lp is None:
                lp = math.log(self.alpha) + self._unigram_logprob(tok)
            total_lp += lp
        return math.exp(-total_lp / len(toks))


def _reference_texts(reference_path: Path, max_docs: int) -> Iterator[str]:
    for i, doc in enumerate(read_jsonl(reference_path)):
        if i >= max_docs:
            break
        yield doc.get("text", "")


def stage_perplexity_filter(
    in_path: Path,
    out_path: Path,
    progress: PipelineProgress,
    reference_path: Path | None,
    order: int,
    percentile: float,
    abs_max: float | None,
    train_tokens: int,
    max_words: int,
    total: int,
    cutoff_sample: int = 200_000,
) -> Accumulator:
    progress.start_step(
        "perplexity_filter",
        note=f"n-gram LM trained from scratch (order={order})",
    )

    lm = NgramLM(order=order)
    if reference_path is not None and reference_path.exists():
        print(f"    training LM on reference corpus: {reference_path.name}")
        lm.train(_reference_texts(reference_path, max_docs=400_000), train_tokens)
    else:
        if reference_path is not None:
            print(f"    [warn] reference corpus not found at {reference_path}")
        print("    training LM on a sample of the input itself (self-reference)")
        lm.train(_reference_texts(in_path, max_docs=400_000), train_tokens)

    if not lm.is_trained():
        print("    [skip] LM training produced no usable counts; copying through unchanged")
        acc = Accumulator()
        with open(out_path, "w", encoding="utf-8") as out:
            for doc in iter_progress(read_jsonl(in_path), total, progress):
                acc.add(doc.get("text", ""), doc.get("source", "unknown"))
                out.write(json.dumps(doc, ensure_ascii=False) + "\n")
        progress.end_step("perplexity_filter")
        return acc

    print(
        f"    LM ready: {lm.vocab_size:,} vocab, {lm.total_tokens:,} training tokens, "
        f"{len(lm.counts[2]):,} kept bigrams"
    )

    # Establish the cutoff from an evenly-spaced SAMPLE rather than by scoring
    # the whole corpus first. The previous version made three passes: train,
    # score-everything, then write. A percentile estimated from ~200k documents
    # is statistically indistinguishable from one over the full few million --
    # the sampling error on a 95th percentile at n=200k is a fraction of a
    # percent -- so this buys back an entire pass for no measurable change in
    # which documents get dropped. An absolute --ppl-max skips sampling too.
    if abs_max is not None:
        cutoff = float(abs_max)
        basis = f"absolute --ppl-max={abs_max}"
    else:
        step = max(1, total // max(cutoff_sample, 1))
        sample: list[float] = []
        for i, doc in enumerate(read_jsonl(in_path)):
            if i % step:
                continue
            v = lm.perplexity(doc.get("text", ""), max_words)
            if math.isfinite(v):
                sample.append(v)
        if sample:
            arr = np.asarray(sample, dtype=np.float32)
            cutoff = float(np.percentile(arr, percentile))
            basis = f"{percentile:g}th percentile of a {arr.size:,}-doc sample (every {step}th)"
            q = np.percentile(arr, [5, 25, 50, 75, 95])
            print(
                f"    perplexity distribution: p5={q[0]:.0f} p25={q[1]:.0f} "
                f"median={q[2]:.0f} p75={q[3]:.0f} p95={q[4]:.0f}"
            )
        else:
            cutoff = float("inf")
            basis = "no finite scores -- filter disabled"
    print(f"    cutoff = {cutoff:.0f} ({basis}); documents above this are dropped")

    # Single scoring+writing pass.
    acc = Accumulator()
    dropped = 0
    with open(out_path, "w", encoding="utf-8") as out:
        for doc in iter_progress(read_jsonl(in_path), total, progress):
            text = doc.get("text", "")
            if not (lm.perplexity(text, max_words) <= cutoff):
                dropped += 1
                continue
            acc.add(text, doc.get("source", "unknown"))
            out.write(json.dumps(doc, ensure_ascii=False) + "\n")
    progress.end_step("perplexity_filter")
    print(f"    dropped (high perplexity): {dropped:,}")
    return acc


# --------------------------------------------------------------------------
# Stage 7: near dedup -- banded MinHash LSH, built to scale
# --------------------------------------------------------------------------
#
# Why this is not the textbook `datasketch` loop: that keeps one MinHash object
# per document in an in-memory index. At this corpus size that is tens of GB of
# Python objects and dies long before it finishes. This version instead:
#
#   * computes each signature with numpy (one vectorised min-reduction per doc
#     rather than num_perm Python-level updates per shingle),
#   * streams signatures straight into an on-disk uint32 memmap, so RAM stays
#     flat regardless of document count -- and the memmap doubles as a resume
#     cache, since signature computation is the expensive half,
#   * finds colliding bands with a vectorised sort instead of a dict with one
#     entry per (band, document).
#
# Shingles are word n-grams (default 5). Token hashing uses zlib.crc32 rather
# than Python's hash() because hash() is randomised per process, which would
# silently invalidate the cached signatures on the next run.

_SHINGLE_MIX = np.array(
    [0x9E3779B97F4A7C15, 0xBF58476D1CE4E5B9, 0x94D049BB133111EB,
     0x2545F4914F6CDD1D, 0xD6E8FEB86659FD93, 0xA0761D6478BD642F,
     0xE7037ED1A0B428DB, 0x8EBC6AF09C88C6E3],
    dtype=np.uint64,
)
_MINHASH_SEED = 1337


def _choose_bands_rows(num_perm: int, threshold: float) -> tuple[int, int]:
    """Pick (bands, rows) whose LSH S-curve inflection ~= the wanted Jaccard
    threshold. The approximation is (1/bands)**(1/rows)."""
    best = (num_perm, 1)
    best_err = float("inf")
    for bands in range(1, num_perm + 1):
        rows = num_perm // bands
        if rows < 1:
            continue
        err = abs((1.0 / bands) ** (1.0 / rows) - threshold)
        if err < best_err:
            best_err, best = err, (bands, rows)
    return best


def _shingle_hashes(text: str, shingle_size: int, max_shingles: int) -> np.ndarray:
    toks = _WORD_SPLIT_RE.findall(text.lower())
    if not toks:
        return np.empty(0, dtype=np.uint64)
    th = np.fromiter(
        (zlib.crc32(t.encode("utf-8")) for t in toks),
        dtype=np.uint64,
        count=len(toks),
    )
    k = min(shingle_size, len(th))
    n = len(th) - k + 1
    sh = np.zeros(n, dtype=np.uint64)
    for j in range(k):
        sh ^= th[j:j + n] * _SHINGLE_MIX[j % len(_SHINGLE_MIX)]
    if n > max_shingles:
        # Evenly subsample rather than truncate: a prefix would make two docs
        # that share an intro look identical even if the bodies differ.
        sh = sh[np.linspace(0, n - 1, max_shingles).astype(np.int64)]
    return sh


def _build_signatures(
    in_path: Path,
    sig_path: Path,
    total: int,
    num_perm: int,
    shingle_size: int,
    max_shingles: int,
    progress: PipelineProgress,
) -> np.ndarray:
    if sig_path.exists():
        try:
            cached = np.load(sig_path, mmap_mode="r")
            if cached.shape == (total, num_perm):
                print(f"    reusing cached signatures: {sig_path.name}")
                return cached
            print("    cached signatures have a stale shape; recomputing")
        except Exception:
            print("    cached signatures unreadable; recomputing")

    rng = np.random.default_rng(_MINHASH_SEED)
    a = rng.integers(1, 2**63, size=num_perm, dtype=np.uint64) | np.uint64(1)
    b = rng.integers(0, 2**63, size=num_perm, dtype=np.uint64)

    sig = np.lib.format.open_memmap(
        sig_path, mode="w+", dtype=np.uint32, shape=(total, num_perm)
    )
    empty_row = np.full(num_perm, np.iinfo(np.uint32).max, dtype=np.uint32)

    with np.errstate(over="ignore"):  # uint64 wraparound is the hash, not a bug
        for i, doc in enumerate(iter_progress(read_jsonl(in_path), total, progress)):
            if i >= total:
                break
            sh = _shingle_hashes(doc.get("text", ""), shingle_size, max_shingles)
            if sh.size == 0:
                sig[i] = empty_row
                continue
            perm = (a[:, None] * sh[None, :]) + b[:, None]
            sig[i] = (perm.min(axis=1) >> np.uint64(32)).astype(np.uint32)
    sig.flush()
    return sig


def stage_near_dedup(
    in_path: Path,
    out_path: Path,
    progress: PipelineProgress,
    interim_dir: Path,
    total: int,
    threshold: float = 0.85,
    num_perm: int = 64,
    shingle_size: int = 5,
    max_shingles: int = 2048,
) -> Accumulator:
    bands, rows = _choose_bands_rows(num_perm, threshold)
    achieved = (1.0 / bands) ** (1.0 / rows)
    progress.start_step(
        "near_dedup",
        note=(
            f"{bands} bands x {rows} rows over {num_perm} perms "
            f"(effective Jaccard threshold ~{achieved:.2f}); slowest stage, budget hours"
        ),
    )

    sig_path = interim_dir / f"minhash_sig_p{num_perm}_s{shingle_size}.npy"
    sig = _build_signatures(
        in_path, sig_path, total, num_perm, shingle_size, max_shingles, progress
    )

    # Band grouping: within a band, documents whose rows are identical collide.
    # A stable sort puts the lowest original index first in each group, so the
    # first member of every group is the canonical document we keep.
    print(f"\n    grouping {total:,} signatures across {bands} bands...")
    is_dup = np.zeros(total, dtype=bool)
    for band in range(bands):
        cols = np.asarray(sig[:, band * rows:(band + 1) * rows], dtype=np.uint64)
        key = np.full(total, 0xCBF29CE484222325, dtype=np.uint64)
        with np.errstate(over="ignore"):
            for c in range(cols.shape[1]):
                key = (key ^ cols[:, c]) * np.uint64(0x100000001B3)
        order = np.argsort(key, kind="stable")
        skey = key[order]
        first_in_group = np.empty(total, dtype=bool)
        first_in_group[0] = True
        np.not_equal(skey[1:], skey[:-1], out=first_in_group[1:])
        is_dup[order[~first_in_group]] = True
        sys.stdout.write(
            f"\r    band {band + 1}/{bands} -- cumulative near-duplicates: {int(is_dup.sum()):,}   "
        )
        sys.stdout.flush()
    print()

    acc = Accumulator()
    dropped = 0
    with open(out_path, "w", encoding="utf-8") as out:
        for i, doc in enumerate(iter_progress(read_jsonl(in_path), total, progress)):
            if i < total and is_dup[i]:
                dropped += 1
                continue
            text = doc.get("text", "")
            acc.add(text, doc.get("source", "unknown"))
            out.write(json.dumps(doc, ensure_ascii=False) + "\n")
    progress.end_step("near_dedup")
    print(f"    dropped (near duplicate, Jaccard ~{achieved:.2f}): {dropped:,}")
    return acc


# --------------------------------------------------------------------------
# Stage 7: final report + downloaded-token-floor sanity check
# --------------------------------------------------------------------------

def stage_report(
    lang_label: str,
    baseline: Accumulator,
    final: Accumulator,
    target_tokens: int,
    fertility_low: float,
    fertility_high: float,
) -> None:
    print(f"\n=== {lang_label}: cleaning summary ===")
    print(fmt_stats("baseline (raw downloaded)", baseline))
    print(fmt_stats("final (post-cleaning)", final))
    if baseline.docs:
        print(f"doc retention   : {100 * final.docs / baseline.docs:5.1f}%")
    if baseline.words:
        print(f"word retention  : {100 * final.words / baseline.words:5.1f}%")

    if final.by_source:
        # Accumulated during the final stage, not by re-reading the output file.
        print("\nper-source breakdown (final):")
        print_by_source(final)

    projected_low = int(final.words * fertility_low)
    projected_high = int(final.words * fertility_high)
    print(
        f"\nprojected BPE tokens (fertility {fertility_low}-{fertility_high}x words): "
        f"{projected_low:,} - {projected_high:,}"
    )
    if projected_low >= target_tokens:
        verdict = "PASS"
    elif projected_high >= target_tokens:
        verdict = "BORDERLINE"
    else:
        verdict = "FAIL"
    print(f"downloaded-corpus token floor ({target_tokens:,}): {verdict}")
    print(
        "NOTE: this is only a proxy. Re-check with the real count from "
        "tokenizer/eval_tokenizer.py / data/prepare_bin.py once the "
        f"{lang_label} SentencePiece tokenizer is trained."
    )


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------

# Relative wall-clock cost per stage, used only to drive the ETA readout.
#
# MEASURED, not guessed: throughput was benchmarked on 15k real Hindi documents
# and cross-checked against a production run. An earlier hand-guessed table had
# near_dedup at 8.0 -- 45% of the total -- which made the ETA read ~24h during
# the first stage. Reality is the inverse: the EARLY stages are expensive
# because they see the full uncompressed corpus, while near_dedup runs on the
# ~45%-of-original that survives filtering. Cost tracks each stage's INPUT SIZE
# far more than its algorithmic complexity.
#
# Weights below reflect the trimmed pipeline (baseline folded into
# unicode_normalize; per-stage count_lines and the naive token count removed).
STAGE_WEIGHTS = {
    "unicode_normalize": 3.00,    # heaviest by far: full corpus, NFC + Indic
                                   # normalize + regex stripping, and it now
                                   # carries the baseline stats too
    "lang_id_filter": 1.60,       # full corpus + per-doc fastText inference
    "boilerplate_removal": 0.90,  # opt-in only
    "quality_filter": 1.30,
    "exact_dedup": 0.50,
    "perplexity_filter": 0.80,    # sample for cutoff, then one scoring pass
    "near_dedup": 0.65,           # smallest input of any stage by this point
}

# (language folder, ISO code) -- folder names match the repo layout.
LANGUAGES = [("hindi", "hi"), ("nepali", "ne")]

# scripts/ lives directly under the repo root.
REPO_ROOT = Path(__file__).resolve().parent.parent


def clean_language(
    root: Path,
    folder: str,
    lang_code: str,
    skip_near_dedup: bool,
    target_tokens: int,
    lid_model_path: Path,
    lid_threshold: float,
    delete_intermediate: bool,
    skip_ppl_filter: bool,
    ppl_reference: str,
    ppl_order: int,
    ppl_percentile: float,
    ppl_max: float | None,
    ppl_train_tokens: int,
    ppl_max_words: int,
    near_dedup_threshold: float,
    near_dedup_num_perm: int,
    near_dedup_shingle: int,
    do_boilerplate: bool,
    min_words: int,
    max_words: int,
    min_script_ratio: float,
    min_terminal_frac: float,
    max_dup_line_frac: float,
) -> None:
    data_dir = root / folder / "data"
    raw_dir = data_dir / "raw"
    interim_dir = data_dir / "interim" / "downloaded"
    processed_dir = data_dir / "processed"
    interim_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    lang_label = f"{folder} ({lang_code})"

    kept_files, skipped_files = discover_source_files(raw_dir)
    print(f"\n=== {lang_label}: source discovery ({raw_dir}) ===")
    if not kept_files:
        raise SystemExit(f"no usable *.jsonl source files found in {raw_dir}")
    for p in kept_files:
        print(f"    [use]  {p.name:<35} {p.stat().st_size / 1e9:6.2f} GB")
    for p, reason in skipped_files:
        print(f"    [skip] {p.name:<35} {reason}")

    weights = dict(STAGE_WEIGHTS)
    if not do_boilerplate:
        del weights["boilerplate_removal"]
    if skip_ppl_filter:
        del weights["perplexity_filter"]
    if skip_near_dedup:
        del weights["near_dedup"]
    progress = PipelineProgress(lang_label, weights)

    def maybe_delete(*paths: Path) -> None:
        if delete_intermediate:
            for p in paths:
                try:
                    p.unlink()
                except FileNotFoundError:
                    pass

    p1 = interim_dir / "01_unicode.jsonl"
    p1b = interim_dir / "01b_langid.jsonl"
    p2 = interim_dir / "02_deboilerplate.jsonl"
    p3 = interim_dir / "03_quality.jsonl"
    p4 = interim_dir / "04_dedup_exact.jsonl"
    p5 = interim_dir / "05_perplexity.jsonl"
    p6 = interim_dir / "06_dedup_near.jsonl"
    final_path = processed_dir / f"{lang_code}_downloaded_clean.jsonl"

    # Stage 1 also produces the baseline "before" stats, so no separate
    # measurement pass over the raw corpus is needed.
    baseline, acc1 = stage_unicode_normalize(kept_files, p1, lang_code, progress)
    print(fmt_stats("after unicode_normalize", acc1))

    acc1b = stage_lang_id_filter(
        p1, p1b, progress, lang_code, lid_model_path, lid_threshold, acc1.docs
    )
    print(fmt_stats("after lang_id_filter", acc1b))
    maybe_delete(p1)

    if do_boilerplate:
        acc2 = stage_boilerplate_removal(p1b, p2, progress, acc1b.docs)
        print(fmt_stats("after boilerplate_removal", acc2))
        maybe_delete(p1b)
    else:
        p1b.replace(p2)
        acc2 = acc1b
        print("    [skip] boilerplate_removal off by default (no-op on single-line docs); --boilerplate to enable")

    acc3 = stage_quality_filter(
        p2, p3, progress, acc2.docs,
        min_words, max_words, min_script_ratio, min_terminal_frac, max_dup_line_frac,
    )
    print(fmt_stats("after quality_filter", acc3))
    maybe_delete(p2)

    acc4 = stage_exact_dedup(p3, p4, progress, acc3.docs)
    print(fmt_stats("after exact_dedup", acc4))
    maybe_delete(p3)

    # Perplexity before near-dedup on purpose: it is the cheaper of the two and
    # shrinks the input that the expensive signature build has to chew through.
    if skip_ppl_filter:
        p4.replace(p5)
        acc5 = acc4
        print("    [skip] perplexity_filter disabled via --skip-ppl-filter")
    else:
        reference_path = (
            None if ppl_reference == "self"
            else processed_dir / f"{lang_code}_manual_clean.jsonl"
        )
        acc5 = stage_perplexity_filter(
            p4, p5, progress, reference_path, ppl_order, ppl_percentile,
            ppl_max, ppl_train_tokens, ppl_max_words, acc4.docs,
        )
        print(fmt_stats("after perplexity_filter", acc5))
        maybe_delete(p4)

    if skip_near_dedup:
        p5.replace(p6)
        acc6 = acc5
        print("    [skip] near_dedup disabled via --skip-near-dedup")
    else:
        acc6 = stage_near_dedup(
            p5, p6, progress, interim_dir, acc5.docs,
            threshold=near_dedup_threshold,
            num_perm=near_dedup_num_perm,
            shingle_size=near_dedup_shingle,
        )
        print(fmt_stats("after near_dedup", acc6))
        maybe_delete(p5)

    p6.replace(final_path)

    # per-source totals came along with acc6; no re-read of the finished file.
    stage_report(
        lang_label,
        baseline,
        acc6,
        target_tokens=target_tokens,
        fertility_low=1.4,
        fertility_high=2.2,
    )
    print(f"final cleaned file: {final_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=str(REPO_ROOT), help="repo root containing hindi/ and nepali/ (default: the repo this script lives in)")
    ap.add_argument("--lang", choices=["hi", "ne"], default=None, help="clean only this language (default: both)")
    ap.add_argument("--skip-near-dedup", action="store_true", help="skip the MinHash near-dedup stage (recommended for a first pass at this scale)")
    ap.add_argument("--target-tokens", type=int, default=400_000_000, help="downloaded-corpus BPE token floor to check against (default 400M)")
    ap.add_argument("--lid-model", default="lid.176.bin", help="path to the fastText LID model file")
    ap.add_argument("--lid-threshold", type=float, default=0.6, help="min fastText confidence to keep a document as this language (default 0.6, stricter than clean_manual.py's 0.5)")
    ap.add_argument("--delete-intermediate", action="store_true", help="delete each stage's input file once the next stage has written its output, to cap disk usage")

    g = ap.add_argument_group("perplexity filter (stage 6)")
    g.add_argument("--skip-ppl-filter", action="store_true", help="skip the n-gram perplexity filter")
    g.add_argument("--ppl-reference", choices=["manual", "self"], default="manual", help="corpus the from-scratch LM is trained on: 'manual' = this language's cleaned manual corpus (default), 'self' = a sample of the input")
    g.add_argument("--ppl-order", type=int, default=2, help="n-gram order of the from-scratch LM (default 2 = bigram)")
    g.add_argument("--ppl-percentile", type=float, default=95.0, help="drop documents above this perplexity percentile (default 95 = worst 5%%)")
    g.add_argument("--ppl-max", type=float, default=None, help="absolute perplexity cutoff; overrides --ppl-percentile")
    g.add_argument("--ppl-train-tokens", type=int, default=20_000_000, help="max tokens used to train the LM (default 20M)")
    g.add_argument("--ppl-max-words", type=int, default=300, help="score only the first N words of each document (default 300)")

    g3 = ap.add_argument_group("quality filter thresholds")
    g3.add_argument("--boilerplate", action="store_true", help="re-enable the boilerplate-removal stage (off by default: measured as a no-op on these single-line documents)")
    g3.add_argument("--min-words", type=int, default=30, help="drop documents shorter than this (default 30; note CC-100 averages ~17 words/doc and is largely removed by this)")
    g3.add_argument("--max-words", type=int, default=100_000, help="drop pathologically long documents (default 100000)")
    g3.add_argument("--min-script-ratio", type=float, default=0.75, help="min fraction of non-space characters that must be Devanagari (default 0.75)")
    g3.add_argument("--min-terminal-frac", type=float, default=0.0,
                    help="min fraction of lines ending in Devanagari sentence punctuation. DEFAULT 0.0 (disabled): these corpora store each document as a single line, so this collapses to 'must end in danda' and rejects otherwise-good prose for its final character rather than its quality. Raise it only for multi-line corpora.")
    g3.add_argument("--max-dup-line-frac", type=float, default=0.25, help="drop documents whose duplicate-line fraction exceeds this (default 0.25; inert on single-line documents)")

    g2 = ap.add_argument_group("near dedup (stage 7)")
    g2.add_argument("--near-dedup-threshold", type=float, default=0.85, help="target Jaccard similarity for near-duplicates (default 0.85)")
    g2.add_argument("--near-dedup-num-perm", type=int, default=64, help="MinHash permutations (default 64; lower = faster + smaller signature cache)")
    g2.add_argument("--near-dedup-shingle", type=int, default=5, help="shingle size in words (default 5)")

    args = ap.parse_args()

    root = Path(args.root)
    languages = LANGUAGES
    if args.lang:
        languages = [(f, c) for f, c in languages if c == args.lang]

    pipeline_t0 = time.perf_counter()
    for folder, code in languages:
        clean_language(
            root, folder, code, args.skip_near_dedup, args.target_tokens,
            Path(args.lid_model), args.lid_threshold, args.delete_intermediate,
            args.skip_ppl_filter, args.ppl_reference, args.ppl_order,
            args.ppl_percentile, args.ppl_max, args.ppl_train_tokens,
            args.ppl_max_words, args.near_dedup_threshold,
            args.near_dedup_num_perm, args.near_dedup_shingle,
            args.boilerplate, args.min_words, args.max_words,
            args.min_script_ratio, args.min_terminal_frac, args.max_dup_line_frac,
        )
    print(f"\nTotal wall-clock time: {fmt_time(time.perf_counter() - pipeline_t0)}")


if __name__ == "__main__":
    main()
