#!/usr/bin/env python3
"""Clean + normalize the combined manual-collection JSONL for Hindi and Nepali,
before SentencePiece BPE training (Phase 1, LMA_Individual_Project_v1.pdf).

This is the ~20% *manual* slice of each language's corpus (scraped/OCR'd text
we collected ourselves, already unzipped+combined from the per-shard .zst
archives). The project requires >=20% of the final ~500M training tokens to be
manual per language, i.e. a floor of ~100M tokens for this slice alone after
BPE. Every filter threshold below is chosen on the *lenient* side for that
reason: it is cheap to run a second, stricter cleaning pass later if quality
demands it, but tokens dropped here are gone for good and count against that
floor.

Paths (all relative to the repo root, which --root defaults to):
    input   <lang>/data/raw/manual/<code>_manual_combined.jsonl
    interim <lang>/data/interim/manual/0*.jsonl      (per-stage checkpoints)
    output  <lang>/data/processed/<code>_manual_clean.jsonl
where <lang> is hindi|nepali and <code> is hi|ne.

Pipeline (each stage reads the previous stage's output JSONL and writes a new
one, so any stage can be re-run/resumed independently):

    0 baseline              -- just measure the combined input, no changes
    1 unicode_normalize     -- NFC + Indic normalize, strip control/zero-width
                                chars, strip URLs/HTML tags+entities/emails/
                                @handles (keep hashtag word, drop the '#'),
                                then a zero-English pass that blanks every
                                remaining bare Latin-letter run (adopted from
                                the friend's pipeline -- digits are kept),
                                repair danda punctuation, collapse whitespace
    2 lang_id_filter        -- drop documents fastText's LID model does not
                                classify as this language (hi/ne share
                                Devanagari with Marathi/Bhojpuri/Maithili/
                                Sanskrit, so script alone can't tell them
                                apart -- see stage_lang_id_filter). Optional
                                dependency (fasttext + lid.176.bin); falls
                                back to pass-through with a warning if either
                                is missing.
    3 boilerplate_removal   -- drop lines that recur across many documents of
                                the same source (nav menus, share bars, cookie
                                notices, copyright footers) -- two passes:
                                count line frequency per source, then strip
    4 quality_filter        -- Gopher-style filters, thresholds picked per-metric
                                from ours vs. the friend's pipeline (see
                                stage_quality_filter for the reasoning behind
                                each): min/max word count, min Devanagari-script
                                ratio, min sentence-final punctuation ratio
                                (disabled -- this corpus is caption/headline-
                                heavy), max within-doc duplicate line fraction,
                                and a binary/image-payload signature check
    5 exact_dedup           -- drop exact (whitespace-normalized) duplicate
                                documents via hash
    6 near_dedup            -- drop near-duplicate documents via MinHash LSH
                                (optional, --skip-near-dedup to turn off; this
                                is the most compute-heavy stage)
    7 report                -- final counts, retention %, and a check against
                                the manual-token floor using an estimated BPE
                                fertility range (real number comes later from
                                tokenizer/eval_tokenizer.py once the
                                tokenizer is actually trained)

Deliberately NOT done here (by request, left for later steps):
    - cross-corpus (Hindi vs. Nepali) overlap removal
    - train/val/test splitting -- this script's output is still a single
      cleaned JSONL per language; split it before BPE training, since
      SentencePiece must only ever see the train split.

After every stage: documents / characters / words / naive-tokens remaining,
how many were dropped and why, time taken, and a live "which step is running
+ how much longer" progress readout (see PipelineProgress / iter_progress
below) so a long Colab run is legible while it's happening.

Usage (run from anywhere; --root defaults to the repo root):
    python scripts/clean_manual.py
    python scripts/clean_manual.py --lang hi --skip-near-dedup

Optional dependencies:
    pip install indic-nlp-library      # stage 1: nukta/vowel-sign normalization
    pip install fasttext-wheel         # stage 2: LID (plain `fasttext` often
                                        # fails to build on Windows; the
                                        # `-wheel` fork ships prebuilt wheels)
    pip install datasketch             # stage 6: near-dedup

    Stage 2 also needs the pretrained fastText LID model file (this is a
    language-identification classifier, not an LM/tokenizer, so it's on the
    project's allowed-tools list). One-time download, then point
    --lid-model at it:
        https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Callable, Iterable, Iterator

# --------------------------------------------------------------------------
# Text-level constants
# --------------------------------------------------------------------------

# Devanagari Unicode block (covers Hindi and Nepali; U+0900-U+097F)
DEVA_RE = re.compile(r"[\u0900-\u097F]")
DEVA_TERMINAL_RE = re.compile(r"[।॥?!]\s*$")
CTRL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")
ZERO_WIDTH_RE = re.compile(r"[\u200b\u200c\u200d\ufeff]")
MULTI_SPACE_RE = re.compile(r"[ \t]+")
MULTI_BLANK_LINE_RE = re.compile(r"\n{3,}")
DOUBLE_DANDA_RE = re.compile(r"\|\|")
SINGLE_PIPE_RE = re.compile(r"(?<!\|)\|(?!\|)")

# Web/markup artifacts left over from scraping -- these are non-linguistic
# noise that would otherwise burn BPE vocab slots on garbage tokens.
HTML_TAG_RE = re.compile(r"<[^>]+>")
HTML_ENTITY_NUM_RE = re.compile(r"&#\d+;")
HTML_ENTITY_NAME_RE = re.compile(r"&[a-zA-Z]+;")
URL_RE = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)
EMAIL_RE = re.compile(r"\S+@\S+\.\S+")
HANDLE_RE = re.compile(r"(?<!\w)@\w+")
HASHTAG_MARK_RE = re.compile(r"(?<!\w)#(?=\w)")  # drop only the '#', keep the word
LATIN_ALPHA_RE = re.compile(r"[A-Za-z]+")  # zero-English rule (adopted): strip every
                                            # remaining Latin-letter run, not just
                                            # URLs/emails/handles. Digits are kept.

# Binary payload leftovers (image/PDF headers that leaked through as "text" from a
# broken extraction) -- adopted check, cheap and has zero legitimate false positives.
BINARY_SIGNATURES = ("JFIF", "Exif", "%PDF", "Photoshop")

# Naive pre-tokenizer-training token estimate, same convention as
# combine_manual.py: word-runs + each punctuation/symbol char as its own
# token. Real BPE token counts only exist once tokenizer/train_spm.py has run.
TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)

# Optional dependency: Indic NFC-adjacent normalization (nukta/vowel-sign
# unification). Falls back to plain NFC if indic_nlp_library isn't installed.
try:
    from indicnlp.normalize.indic_normalize import IndicNormalizerFactory

    _NORM_FACTORY = IndicNormalizerFactory()

    def _make_indic_normalizer(lang_code: str):
        try:
            return _NORM_FACTORY.get_normalizer(lang_code)
        except Exception:
            return _NORM_FACTORY.get_normalizer("hi")  # Devanagari fallback

except Exception:  # indic_nlp_library not installed
    def _make_indic_normalizer(lang_code: str):
        return None


# --------------------------------------------------------------------------
# Stats helpers
# --------------------------------------------------------------------------

def text_stats(text: str) -> tuple[int, int, int]:
    """(chars, words, naive_tokens) for one document's text."""
    return len(text), len(text.split()), len(TOKEN_RE.findall(text))


class Accumulator:
    """Running docs/chars/words/tokens totals for a stage."""

    __slots__ = ("docs", "chars", "words", "tokens")

    def __init__(self) -> None:
        self.docs = self.chars = self.words = self.tokens = 0

    def add(self, text: str) -> None:
        c, w, t = text_stats(text)
        self.docs += 1
        self.chars += c
        self.words += w
        self.tokens += t

    def as_dict(self) -> dict:
        return {"docs": self.docs, "chars": self.chars, "words": self.words, "tokens": self.tokens}


def fmt_time(seconds: float) -> str:
    if seconds != seconds or seconds < 0:  # NaN guard
        return "--"
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m{s:02d}s"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


def fmt_stats(label: str, acc: Accumulator | dict) -> str:
    d = acc.as_dict() if isinstance(acc, Accumulator) else acc
    return (
        f"{label:<22} docs={d['docs']:>10,}  chars={d['chars']:>14,}  "
        f"words={d['words']:>13,}  tokens={d['tokens']:>13,}"
    )


# --------------------------------------------------------------------------
# Progress / ETA tracking
# --------------------------------------------------------------------------

class PipelineProgress:
    """Tracks which step is currently running and estimates total time left.

    Steps are weighted by their expected relative cost (near-dedup and the
    two-pass boilerplate step are heavier than a single streaming pass), so
    the ETA reflects real wall-clock proportions rather than treating every
    step as equal-length.
    """

    def __init__(self, lang_label: str, step_weights: "dict[str, float]") -> None:
        self.lang_label = lang_label
        self.weights = step_weights
        self.order = list(step_weights)
        self.total_weight = sum(step_weights.values())
        self.completed_weight = 0.0
        self.t_start = time.perf_counter()
        self._current: str | None = None
        self._step_t0: float = 0.0

    def start_step(self, name: str) -> None:
        self._current = name
        self._step_t0 = time.perf_counter()
        idx = self.order.index(name) + 1
        print(
            f"\n>>> [{self.lang_label}] step {idx}/{len(self.order)} "
            f"'{name}' STARTING"
        )

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
        print(
            f"\n<<< [{self.lang_label}] step '{name}' DONE in {fmt_time(dt)} "
            f"(pipeline elapsed {fmt_time(elapsed_total)})"
        )


def iter_progress(
    items: Iterable,
    total: int,
    progress: PipelineProgress,
    update_every: int = 10_000,
) -> Iterator:
    """Yield items unchanged, calling progress.tick() every `update_every` items."""
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
                yield json.loads(line)


# --------------------------------------------------------------------------
# Stage 1: Unicode / Indic normalization
# --------------------------------------------------------------------------

def strip_web_artifacts(text: str) -> str:
    """Remove non-linguistic scraping leftovers: HTML tags/entities, URLs,
    emails, @handles. Hashtags keep their word (only the '#' is dropped) since
    the word itself is often real content, e.g. "#कोरोना" -> "कोरोना". Structural
    artifacts (URLs/emails/handles) are stripped as whole units *before* the
    blanket Latin-letter strip below, so we don't leave mangled partial tokens
    like a bare "@" or "http" behind."""
    text = HTML_TAG_RE.sub(" ", text)
    text = HTML_ENTITY_NUM_RE.sub(" ", text)
    text = HTML_ENTITY_NAME_RE.sub(" ", text)
    text = URL_RE.sub(" ", text)
    text = EMAIL_RE.sub(" ", text)
    text = HANDLE_RE.sub(" ", text)
    text = HASHTAG_MARK_RE.sub("", text)
    # Zero-English rule (adopted from the friend's pipeline): any remaining
    # bare Latin letters -- code-switched English words/phrases, not just
    # markup -- are stripped too. Runs before lang_id_filter (stage 2) and
    # quality_filter (stage 4), so both see already-purified text: LID isn't
    # confused by code-switching, and the script-ratio check no longer has to
    # do double duty catching English contamination.
    text = LATIN_ALPHA_RE.sub(" ", text)
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
    in_path: Path, out_path: Path, lang_code: str, progress: PipelineProgress
) -> Accumulator:
    progress.start_step("unicode_normalize")
    total = count_lines(in_path)
    indic_normalizer = _make_indic_normalizer(lang_code)
    acc = Accumulator()
    dropped_empty = 0
    with open(out_path, "w", encoding="utf-8") as out:
        for doc in iter_progress(read_jsonl(in_path), total, progress):
            doc["text"] = clean_text_unicode(doc.get("text", ""), indic_normalizer)
            if not doc["text"]:
                dropped_empty += 1
                continue
            acc.add(doc["text"])
            out.write(json.dumps(doc, ensure_ascii=False) + "\n")
    progress.end_step("unicode_normalize")
    print(f"    dropped (empty after normalize): {dropped_empty:,}")
    return acc


# --------------------------------------------------------------------------
# Stage 2: language-ID filter (drop docs that aren't actually this language)
# --------------------------------------------------------------------------
#
# Hindi and Nepali share the Devanagari script with Marathi, Bhojpuri,
# Maithili, and Sanskrit, so a script-ratio check (stage 4) cannot tell them
# apart -- a mislabeled/mixed source can leak wrong-language text straight
# into the tokenizer's BPE merges. fastText's public lid.176 model is a
# language-ID *classifier*, not a pretrained LM/tokenizer, so it's on the
# project's allowed-tools list (see module docstring for install/download).
#
# Both dependencies are optional: if `fasttext` isn't installed, or the model
# file isn't found at --lid-model, this stage copies the data through
# unchanged with a printed warning instead of failing the run.

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
    threshold: float = 0.5,
) -> Accumulator:
    progress.start_step("lang_id_filter")
    total = count_lines(in_path)
    model = _load_fasttext_model(model_path)
    acc = Accumulator()

    if model is None:
        print(
            "    [skip] fastText LID unavailable; copying through unchanged "
            "(pip install fasttext-wheel + download lid.176.bin to enable)"
        )
        with open(out_path, "w", encoding="utf-8") as out:
            for doc in iter_progress(read_jsonl(in_path), total, progress):
                acc.add(doc.get("text", ""))
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
                acc.add(text)
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
# Stage 3: dynamic boilerplate removal (two passes)
# --------------------------------------------------------------------------

def _hash_line(line: str) -> str:
    return hashlib.blake2b(line.strip().encode("utf-8"), digest_size=8).hexdigest()


def stage_boilerplate_removal(
    in_path: Path,
    out_path: Path,
    progress: PipelineProgress,
    line_freq_threshold: float = 0.02,  # a line must appear in >2% of a
                                         # source's docs to be considered
                                         # boilerplate (loose on purpose --
                                         # protects the manual-token floor)
) -> Accumulator:
    progress.start_step("boilerplate_removal")
    total = count_lines(in_path)

    # Pass A: per-source line-hash frequency + per-source doc count.
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
            if h in seen_in_doc:  # count each line at most once per doc
                continue
            seen_in_doc.add(h)
            line_counts[(source, h)] += 1

    boilerplate_hashes: dict[str, set] = {}
    for (source, h), n in line_counts.items():
        if n / max(docs_per_source[source], 1) > line_freq_threshold:
            boilerplate_hashes.setdefault(source, set()).add(h)

    # Pass B: strip boilerplate lines, drop docs that become too short.
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
            acc.add(doc["text"])
            out.write(json.dumps(doc, ensure_ascii=False) + "\n")

    progress.end_step("boilerplate_removal")
    n_boiler_lines = sum(len(v) for v in boilerplate_hashes.values())
    print(f"    boilerplate line patterns removed: {n_boiler_lines:,} (across {len(boilerplate_hashes):,} sources)")
    print(f"    dropped (empty after strip)      : {dropped_empty:,}")
    return acc


# --------------------------------------------------------------------------
# Stage 4: Gopher-style quality filter
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
    """Detect image/PDF/etc. header bytes that leaked through as 'text' from a
    broken extraction (adopted check -- essentially zero false-positive risk)."""
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
    # Every default below is the result of comparing our original (token-floor-
    # protective) values against the friend's pipeline's values and picking per
    # metric, not wholesale adopting either side -- see chat discussion.
    min_words: int = 10,             # adopted from friend's pipeline (was chars>=150,
                                      # stricter than his words>=10 in practice).
                                      # Preserves short captions/headlines, which this
                                      # corpus is full of -- directly helps the token floor.
    max_words: int = 100_000,        # adopted from friend's pipeline (we had no cap).
                                      # Only trims pathological giant/garbage docs;
                                      # negligible cost to the token floor.
    min_script_ratio: float = 0.70,  # compromise: ours was 0.60 (token-floor-safe but
                                      # loose), his was 0.75 (standard but riskier for
                                      # Nepali). Raised because stage 1 now strips ALL
                                      # Latin letters before this check runs, so it no
                                      # longer has to absorb English contamination --
                                      # it's now mainly catching other-script/junk residue.
    min_terminal_frac: float = 0.0,  # adopted from friend's pipeline (was 0.15). This
                                      # corpus is caption/headline-heavy and those
                                      # legitimately don't end in danda punctuation --
                                      # the old threshold was rejecting valid content.
    max_dup_line_frac: float = 0.30,  # adopted from friend's pipeline (was 0.40).
                                       # Duplicate-heavy docs contribute little unique
                                       # content anyway, so tightening this costs little.
) -> Accumulator:
    progress.start_step("quality_filter")
    total = count_lines(in_path)
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
            acc.add(text)
            out.write(json.dumps(doc, ensure_ascii=False) + "\n")
    progress.end_step("quality_filter")
    for reason, n in reasons.most_common():
        print(f"    dropped ({reason}): {n:,}")
    return acc


# --------------------------------------------------------------------------
# Stage 5: exact dedup
# --------------------------------------------------------------------------

def norm_key(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def stage_exact_dedup(
    in_path: Path, out_path: Path, progress: PipelineProgress
) -> Accumulator:
    progress.start_step("exact_dedup")
    total = count_lines(in_path)
    seen: set = set()
    acc = Accumulator()
    dropped = 0
    with open(out_path, "w", encoding="utf-8") as out:
        for doc in iter_progress(read_jsonl(in_path), total, progress):
            text = doc.get("text", "")
            h = hashlib.blake2b(norm_key(text).encode("utf-8"), digest_size=16).hexdigest()
            if h in seen:
                dropped += 1
                continue
            seen.add(h)
            acc.add(text)
            out.write(json.dumps(doc, ensure_ascii=False) + "\n")
    progress.end_step("exact_dedup")
    print(f"    dropped (exact duplicate): {dropped:,}")
    return acc


# --------------------------------------------------------------------------
# Stage 6: near dedup (MinHash LSH) -- optional, heaviest stage
# --------------------------------------------------------------------------

def stage_near_dedup(
    in_path: Path,
    out_path: Path,
    progress: PipelineProgress,
    threshold: float = 0.85,  # stricter than the master plan's 0.80 default:
                               # only near-*identical* docs are considered
                               # duplicates, so merely similar articles on
                               # the same topic are kept (token floor)
    num_perm: int = 64,
) -> Accumulator:
    progress.start_step("near_dedup")
    total = count_lines(in_path)
    try:
        from datasketch import MinHash, MinHashLSH
    except ImportError:
        print("    [skip] datasketch not installed; copying through unchanged")
        acc = Accumulator()
        with open(out_path, "w", encoding="utf-8") as out:
            for doc in iter_progress(read_jsonl(in_path), total, progress):
                acc.add(doc.get("text", ""))
                out.write(json.dumps(doc, ensure_ascii=False) + "\n")
        progress.end_step("near_dedup")
        return acc

    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    acc = Accumulator()
    dropped = 0
    with open(out_path, "w", encoding="utf-8") as out:
        for i, doc in enumerate(iter_progress(read_jsonl(in_path), total, progress)):
            text = doc.get("text", "")
            mh = MinHash(num_perm=num_perm)
            for tok in norm_key(text).split():
                mh.update(tok.encode("utf-8"))
            if lsh.query(mh):
                dropped += 1
                continue
            lsh.insert(f"d{i}", mh)
            acc.add(text)
            out.write(json.dumps(doc, ensure_ascii=False) + "\n")
    progress.end_step("near_dedup")
    print(f"    dropped (near duplicate, threshold={threshold}): {dropped:,}")
    return acc


# --------------------------------------------------------------------------
# Stage 7: final report + manual-token-floor sanity check
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
    print(fmt_stats("baseline (raw manual)", baseline))
    print(fmt_stats("final (post-cleaning)", final))
    if baseline.docs:
        print(f"doc retention   : {100 * final.docs / baseline.docs:5.1f}%")
    if baseline.tokens:
        print(f"token retention : {100 * final.tokens / baseline.tokens:5.1f}% (naive-token basis)")

    # Real BPE token count only exists after tokenizer/train_spm.py +
    # eval_tokenizer.py run on this cleaned text. Fertility (tokens per
    # whitespace word) for a 16k-32k Devanagari BPE model is typically in the
    # ~1.4-2.2 range; use that to sanity-check the manual-token floor now.
    projected_low = int(final.words * fertility_low)
    projected_high = int(final.words * fertility_high)
    print(
        f"projected BPE tokens (fertility {fertility_low}-{fertility_high}x words): "
        f"{projected_low:,} - {projected_high:,}"
    )
    if projected_low >= target_tokens:
        verdict = "PASS"
    elif projected_high >= target_tokens:
        verdict = "BORDERLINE"
    else:
        verdict = "FAIL"
    print(f"manual-token floor ({target_tokens:,}): {verdict}")
    print(
        "NOTE: this is only a proxy. Re-check with the real count from "
        "tokenizer/eval_tokenizer.py / data/prepare_bin.py once the "
        f"{lang_label} SentencePiece tokenizer is trained."
    )


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------

STAGE_WEIGHTS = {
    "baseline": 0.5,              # 1 read-only pass, no writing/filtering
    "unicode_normalize": 1.0,     # 1 pass
    "lang_id_filter": 1.2,        # 1 pass + per-doc model inference
    "boilerplate_removal": 2.0,   # 2 passes
    "quality_filter": 1.0,        # 1 pass
    "exact_dedup": 1.0,           # 1 pass, in-memory hash set
    "near_dedup": 3.0,            # 1 pass but MinHash construction is slow
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
) -> None:
    data_dir = root / folder / "data"
    combined_path = data_dir / "raw" / "manual" / f"{lang_code}_manual_combined.jsonl"
    interim_dir = data_dir / "interim" / "manual"
    processed_dir = data_dir / "processed"
    interim_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    if not combined_path.exists():
        raise SystemExit(
            f"{combined_path} not found -- combine the per-shard .zst archives into "
            "this file first"
        )

    lang_label = f"{folder} ({lang_code})"
    weights = dict(STAGE_WEIGHTS)
    if skip_near_dedup:
        del weights["near_dedup"]
    progress = PipelineProgress(lang_label, weights)

    # Stage 0: baseline stats only, no transform.
    progress.start_step("baseline")
    total0 = count_lines(combined_path)
    baseline = Accumulator()
    for doc in iter_progress(read_jsonl(combined_path), total0, progress):
        baseline.add(doc.get("text", ""))
    progress.end_step("baseline")
    print(fmt_stats("baseline (raw manual)", baseline))

    p1 = interim_dir / "01_unicode.jsonl"
    p1b = interim_dir / "01b_langid.jsonl"
    p2 = interim_dir / "02_deboilerplate.jsonl"
    p3 = interim_dir / "03_quality.jsonl"
    p4 = interim_dir / "04_dedup_exact.jsonl"
    p5 = interim_dir / "05_dedup_near.jsonl"
    final_path = processed_dir / f"{lang_code}_manual_clean.jsonl"

    acc1 = stage_unicode_normalize(combined_path, p1, lang_code, progress)
    print(fmt_stats("after unicode_normalize", acc1))

    acc1b = stage_lang_id_filter(p1, p1b, progress, lang_code, lid_model_path, lid_threshold)
    print(fmt_stats("after lang_id_filter", acc1b))

    acc2 = stage_boilerplate_removal(p1b, p2, progress)
    print(fmt_stats("after boilerplate_removal", acc2))

    acc3 = stage_quality_filter(p2, p3, progress)
    print(fmt_stats("after quality_filter", acc3))

    acc4 = stage_exact_dedup(p3, p4, progress)
    print(fmt_stats("after exact_dedup", acc4))

    if skip_near_dedup:
        p4.replace(p5)
        acc5 = acc4
        print("    [skip] near_dedup disabled via --skip-near-dedup")
    else:
        acc5 = stage_near_dedup(p4, p5, progress)
        print(fmt_stats("after near_dedup", acc5))

    p5.replace(final_path)

    stage_report(
        lang_label,
        baseline,
        acc5,
        target_tokens=target_tokens,
        fertility_low=1.4,
        fertility_high=2.2,
    )
    print(f"final cleaned file: {final_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=str(REPO_ROOT), help="repo root containing hindi/ and nepali/ (default: the repo this script lives in)")
    ap.add_argument("--lang", choices=["hi", "ne"], default=None, help="clean only this language (default: both)")
    ap.add_argument("--skip-near-dedup", action="store_true", help="skip the MinHash near-dedup stage (fastest, less thorough)")
    ap.add_argument("--target-tokens", type=int, default=100_000_000, help="manual-portion BPE token floor to check against (default 100M = 20%% of the 500M/language target)")
    ap.add_argument("--lid-model", default="lid.176.bin", help="path to the fastText LID model file (stage lang_id_filter; skipped with a warning if missing)")
    ap.add_argument("--lid-threshold", type=float, default=0.5, help="min fastText confidence to keep a document as this language (default 0.5)")
    args = ap.parse_args()

    root = Path(args.root)
    languages = LANGUAGES
    if args.lang:
        languages = [(f, c) for f, c in languages if c == args.lang]

    pipeline_t0 = time.perf_counter()
    for folder, code in languages:
        clean_language(
            root, folder, code, args.skip_near_dedup, args.target_tokens,
            Path(args.lid_model), args.lid_threshold,
        )
    print(f"\nTotal wall-clock time: {fmt_time(time.perf_counter() - pipeline_t0)}")


if __name__ == "__main__":
    main()
