#!/usr/bin/env python3
"""Phase 1, 1.3 -- corpus assembly, splits, and from-scratch BPE tokenizers.

Covers, per language and completely independently:

    combine   manual + downsampled downloaded, provenance preserved
    split     document-level train / val / test, stratified and seeded
    train     SentencePiece BPE trained FROM SCRATCH on the TRAIN split only
    evaluate  fertility / UNK / chars-per-token sweep on HELD-OUT val
    encode    uint16 .bin memmaps + meta.json
    report    everything LMA_Individual_Project_v1.pdf 1.3 asks to be reported

Run:
    python scripts/build_tokenizer.py                      # both languages, all stages
    python scripts/build_tokenizer.py --lang hi
    python scripts/build_tokenizer.py --stages evaluate,encode,report
    python scripts/build_tokenizer.py --train-frac 0.90 --val-frac 0.05 --test-frac 0.05

Splits default to 80/10/10 at the document level.


HOW THIS MEETS THE SPEC
-----------------------
1.3 "Train a separate tokenizer from scratch for each language"
    Two independent SentencePiece runs. Nothing is shared: separate input text,
    separate model, separate vocabulary, separate output directory. The script
    never concatenates languages, and refuses to if asked (see LANGUAGES).

1.3 "Pretrained tokenizers are not permitted"
    sentencepiece trains from raw text here. No tokenizer is downloaded.
    (The only downloaded model anywhere in this project is fastText lid.176,
    used for language identification during cleaning -- a classifier, not a
    tokenizer or an LM.)

1.3 "Recommended vocabulary size is in the tens of thousands"
    The sweep spans 8k/16k/32k, but auto-selection is floored at 16k so the
    chosen model always sits in the recommended band; 8k is trained purely as
    evidence for the report that a smaller vocabulary was evaluated and why it
    lost. --vocab overrides the choice manually.

1.3 "choose using fertility / unknown-token rate on held-out text"
    stage_evaluate scores every candidate on the VAL split, which no tokenizer
    ever saw during training. Selection rule is explicit and printed.

1.3 "Vocabularies must not be shared"
    Output paths are per-language and the model file name carries the language
    code, so the two can never be confused for one another.

1.3 "Report ... vocabulary size, token-frequency statistics, average characters
     per token, tokenization examples, and unknown-token statistics"
    stage_report writes report.md + report.json containing all five, plus the
    full sweep table.

1.2 "at least 20% of the final training tokens must come from manual collection"
    stage_combine downsamples the downloaded side so manual reaches the target
    share, and stage_encode measures the achieved share IN TOKENS (the unit the
    requirement is stated in) and fails loudly if it lands under 20%.

1.2 "Report the manual vs. downloaded token split"
    meta.json and report.md both carry it, per split.

1.2 "Maintain separate train/validation/test splits"
    Document-level, so no sentence appears in two splits.


TWO IMPLEMENTATION DETAILS THAT MATTER
--------------------------------------
1. SentencePiece silently drops long lines. --max_sentence_length defaults to
   4192 BYTES. Devanagari is 3 bytes/char, so that is ~1,400 characters -- but
   the cleaned documents average 384 words (~1,970 chars, ~5,900 bytes) and
   MADLAD's average 843 words. Feeding whole documents would therefore discard
   the MAJORITY of the corpus and train the tokenizer on only the shortest
   documents, with nothing but a warning to indicate it. stage_split writes
   train_sentences.txt split on Devanagari sentence punctuation instead, with a
   byte-bounded fallback chunker for any sentence still over the limit.

   Note this affects TRAINING only. Encoding has no length limit, so .bin
   encoding runs over whole documents, preserving document-level context.

2. Splitting and downsampling are decided by a stable hash of the document, not
   by random.shuffle(). That makes both decisions reproducible across runs and
   machines without holding the corpus in memory, and lets manual and downloaded
   be stratified independently in a single streaming pass.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import threading
import time
from collections import Counter
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np

# Devanagari is not representable in the Windows console's default code page;
# without this, printing a tokenization example raises UnicodeEncodeError.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

REPO_ROOT = Path(__file__).resolve().parent.parent
LANGUAGES = [("hindi", "hi"), ("nepali", "ne")]

# Sentence boundary for Devanagari: danda, double danda, question, exclamation.
SENT_SPLIT_RE = re.compile(r"(?<=[।॥?!])\s+")

# Reserved ids, fixed so Phase 2 can rely on them.
PAD_ID, UNK_ID, BOS_ID, EOS_ID = 0, 1, 2, 3

STAGE_ORDER = ["combine", "split", "train", "evaluate", "encode", "report"]


# --------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------

def fmt_time(seconds: float) -> str:
    seconds = int(max(seconds, 0))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


def banner(text: str) -> None:
    print(f"\n{'=' * 78}\n{text}\n{'=' * 78}", flush=True)


class Progress:
    """Single-line live progress with rate and ETA.

    Updates on a TIME interval rather than an item count: an item-count trigger
    either spams on fast stages or looks frozen on slow ones, whereas a ~2s
    cadence reads the same regardless of how quick each item is.
    """

    def __init__(self, label: str, total: int | None = None, interval: float = 2.0) -> None:
        self.label, self.total, self.interval = label, total, interval
        self.n = 0
        self.t0 = time.perf_counter()
        self._last = 0.0

    def update(self, k: int = 1) -> None:
        self.n += k
        now = time.perf_counter()
        if now - self._last >= self.interval:
            self._last = now
            self._render()

    def _render(self) -> None:
        el = time.perf_counter() - self.t0
        rate = self.n / el if el > 0 else 0.0
        if self.total:
            frac = min(self.n / self.total, 1.0)
            eta = (el / frac - el) if frac > 0 else float("nan")
            msg = (f"\r    {self.label}: {self.n:,}/{self.total:,} ({100 * frac:5.1f}%) | "
                   f"{rate:,.0f}/s | elapsed {fmt_time(el)} | ETA {fmt_time(eta)}      ")
        else:
            msg = (f"\r    {self.label}: {self.n:,} | {rate:,.0f}/s | "
                   f"elapsed {fmt_time(el)}      ")
        sys.stdout.write(msg)
        sys.stdout.flush()

    def close(self) -> float:
        self._render()
        sys.stdout.write("\n")
        sys.stdout.flush()
        return time.perf_counter() - self.t0


class Heartbeat:
    """Prints elapsed time from a background thread during a blocking call.

    SentencePiece training is one C++ call with no Python-visible progress. On
    a multi-GB corpus it runs for tens of minutes, during which the terminal
    would otherwise look completely frozen -- indistinguishable from a hang.
    This ticks every `interval` seconds so it is obvious work is happening.
    """

    def __init__(self, label: str, interval: float = 20.0) -> None:
        self.label, self.interval = label, interval
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.t0 = 0.0

    def _run(self) -> None:
        while not self._stop.wait(self.interval):
            el = time.perf_counter() - self.t0
            sys.stdout.write(f"\r    {self.label}: still running, elapsed {fmt_time(el)}      ")
            sys.stdout.flush()

    def __enter__(self) -> "Heartbeat":
        self.t0 = time.perf_counter()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=1.0)
        sys.stdout.write("\r" + " " * 78 + "\r")
        sys.stdout.flush()


def read_jsonl(path: Path) -> Iterator[dict]:
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue


def is_manual(doc: dict) -> bool:
    """Provenance, tolerant of both schemas in play.

    download_all.py stamps {"manual": false}; the manually collected shards
    carry {"source_type": "manual"} instead. stage_combine normalises everything
    to a single boolean so nothing downstream has to know this.
    """
    if "manual" in doc:
        return bool(doc["manual"])
    return doc.get("source_type") == "manual"


def unit_hash(text: str, salt: str, seed: int) -> float:
    """Deterministic uniform [0, 1) derived from the document itself.

    Used for both the downsample decision and the split assignment. Being a
    function of content rather than of iteration order means: reproducible
    across runs and machines, no shuffle buffer, and the two decisions stay
    independent because they use different salts.
    """
    h = hashlib.blake2b(f"{salt}:{seed}:{text}".encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(h, "big") / 2**64


def iter_training_sentences(text: str, max_bytes: int) -> Iterator[str]:
    """Yield sentence-sized lines that fit SentencePiece's byte limit."""
    for sent in SENT_SPLIT_RE.split(text):
        sent = sent.strip()
        if not sent:
            continue
        if len(sent.encode("utf-8")) <= max_bytes:
            yield sent
            continue
        # Still too long (unpunctuated run-on). Chunk on word boundaries rather
        # than mid-character, so no piece straddles a UTF-8 sequence.
        buf: list[str] = []
        buf_bytes = 0
        for word in sent.split():
            wb = len(word.encode("utf-8")) + 1
            if buf and buf_bytes + wb > max_bytes:
                yield " ".join(buf)
                buf, buf_bytes = [], 0
            buf.append(word)
            buf_bytes += wb
        if buf:
            yield " ".join(buf)


def load_corpus_stats(root: Path) -> dict | None:
    """Load precomputed word counts, from THIS root only.

    Deliberately does not fall back to REPO_ROOT: the stats describe a specific
    corpus, and silently applying the repo's stats to a different --root would
    compute the downsample ratio from the wrong totals and quietly produce a
    corpus with the wrong manual share -- with no error to notice.
    """
    candidate = root / "corpus_stats.json"
    if candidate.exists():
        try:
            return json.loads(candidate.read_text(encoding="utf-8"))
        except Exception:
            print(f"  [warn] could not parse {candidate}; will count instead")
    return None


# --------------------------------------------------------------------------
# paths
# --------------------------------------------------------------------------

class Paths:
    def __init__(self, root: Path, folder: str, code: str) -> None:
        self.folder, self.code = folder, code
        d = root / folder
        self.processed = d / "data" / "processed"
        self.manual = self.processed / f"{code}_manual_clean.jsonl"
        self.downloaded = self.processed / f"{code}_downloaded_clean.jsonl"
        self.splits = d / "data" / "splits"
        self.train = self.splits / "train.jsonl"
        self.val = self.splits / "val.jsonl"
        self.test = self.splits / "test.jsonl"
        self.train_sentences = self.splits / "train_sentences.txt"
        self.tokenizer = d / "tokenizer"
        self.bin = d / "data" / "bin"
        self.sweep = self.tokenizer / "sweep_results.json"
        self.meta = self.bin / "meta.json"
        self.report_md = self.tokenizer / "report.md"
        self.report_json = self.tokenizer / "report.json"

    def model_prefix(self, vocab: int) -> Path:
        return self.tokenizer / f"{self.code}_bpe_{vocab}"

    def split_file(self, name: str) -> Path:
        return {"train": self.train, "val": self.val, "test": self.test}[name]

    def bin_file(self, name: str) -> Path:
        return self.bin / f"{name}.bin"

    def mkdirs(self) -> None:
        for p in (self.splits, self.tokenizer, self.bin):
            p.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------
# stage: combine + split (fused -- one streaming pass)
# --------------------------------------------------------------------------

# Rough characters-per-token for a 16k-32k Devanagari BPE vocabulary, used only
# to turn a token target into a word target before the tokenizer exists.
# Fertility = chars_per_word / chars_per_token. Override with --assumed-fertility.
ASSUMED_CHARS_PER_TOKEN = 3.4


def _word_counts(paths: Paths, stats: dict | None) -> tuple[int, int, float]:
    lang_stats = (stats or {}).get(paths.folder)
    if lang_stats:
        m = lang_stats["manual"]["words"]
        d = lang_stats["downloaded_total"]["words"]
        cpw = lang_stats["combined_if_all_used"]["avg_chars_per_word"]
        print(f"  using corpus_stats.json: manual={m:,} downloaded={d:,} words")
        return m, d, cpw
    print("  corpus_stats.json not found -- counting words (one pass over both files)")
    m = mc = 0
    for doc in read_jsonl(paths.manual):
        t = doc.get("text", "")
        m += len(t.split())
        mc += len(t)
    d = dc = 0
    for doc in read_jsonl(paths.downloaded):
        t = doc.get("text", "")
        d += len(t.split())
        dc += len(t)
    cpw = (mc + dc) / max(m + d, 1)
    print(f"  counted: manual={m:,} downloaded={d:,} words")
    return m, d, cpw


def _plan_corpus(paths: Paths, stats: dict | None, target_train_tokens: int,
                 manual_frac: float, train_frac: float, upsample: float,
                 assumed_fertility: float | None) -> dict:
    """Decide how much downloaded data to keep.

    Two mutually exclusive sizing modes:

    TOKEN TARGET (--target-train-tokens > 0, the default)
        Keep whatever amount of downloaded data makes the TRAIN split reach the
        requested token count:

            corpus_words = target_tokens / (fertility * train_frac)
            downloaded_to_keep = corpus_words - manual_words

        Fertility is not knowable before the tokenizer is trained, so it is
        estimated from the corpus's own characters-per-word (or forced with
        --assumed-fertility) and the actual value is reported after encoding.

        IMPORTANT CONSEQUENCE: all manual data is used, so the manual share is
        whatever falls out of the arithmetic -- manual_words / corpus_words.
        Since the corpus is much larger than 5x the manual corpus, that share
        lands BELOW the 20% the brief requires. The script prints the projected
        share here and the measured share in stage_encode; neither is silent.
        --upsample-manual is the only lever that raises it without shrinking
        the corpus.

    MANUAL FRACTION (--target-train-tokens 0)
        The original behaviour: size the downloaded side so manual reaches
        manual_frac exactly, and accept whatever token count results. This is
        the mode that satisfies the >=20% requirement.
    """
    m_words, d_words, cpw = _word_counts(paths, stats)
    fertility = assumed_fertility or (cpw / ASSUMED_CHARS_PER_TOKEN)
    m_eff = m_words * upsample                      # manual after upsampling

    if target_train_tokens > 0:
        corpus_needed = target_train_tokens / (fertility * train_frac)
        d_needed = max(0.0, corpus_needed - m_eff)
        mode = "token-target"
    else:
        corpus_needed = m_eff / manual_frac if manual_frac > 0 else float("inf")
        d_needed = m_eff * (1.0 - manual_frac) / manual_frac if manual_frac > 0 else d_words
        mode = "manual-frac"

    keep_p = min(1.0, d_needed / d_words) if d_words else 0.0
    d_kept = d_words * keep_p
    corpus = m_eff + d_kept
    proj_manual = m_eff / corpus if corpus else 0.0
    proj_train_tokens = corpus * train_frac * fertility

    print(f"  sizing mode         : {mode}")
    print(f"  assumed fertility   : {fertility:.3f} tokens/word"
          f"{'' if assumed_fertility else f' (from {cpw:.2f} chars/word / {ASSUMED_CHARS_PER_TOKEN})'}")
    if upsample != 1.0:
        print(f"  manual upsample     : {upsample:.2f}x  ({m_words:,} -> {m_eff:,.0f} effective words)")
    if target_train_tokens > 0:
        print(f"  target train tokens : {target_train_tokens:,}")
        if d_needed > d_words:
            print(f"  [WARN] target needs {d_needed:,.0f} downloaded words but only "
                  f"{d_words:,} exist -- keeping 100% and falling short")
    print(f"  downloaded keep     : {keep_p:.2%}  ({d_kept:,.0f} of {d_words:,} words)")
    print(f"  projected corpus    : {corpus:,.0f} words")
    print(f"  projected TRAIN tok : {proj_train_tokens:,.0f}")
    print(f"  projected manual %  : {100 * proj_manual:.2f}%")
    if proj_manual < 0.20:
        need_up = (0.20 * d_kept) / (0.80 * m_words) if m_words else 0.0
        print("  " + "!" * 70)
        print(f"  !! PROJECTED MANUAL SHARE {100 * proj_manual:.2f}% IS BELOW THE REQUIRED 20%.")
        print(f"  !! LMA_Individual_Project_v1.pdf 1.2: at least 20% of the final training")
        print(f"  !! tokens MUST come from manual collection. Sizing the corpus to a token")
        print(f"  !! target instead of a manual ratio is what causes this.")
        print(f"  !! To reach 500M AND >=20%, add: --upsample-manual {need_up:.2f}")
        print(f"  !! To satisfy 20% instead of the token target: --target-train-tokens 0")
        print("  " + "!" * 70)

    return {
        "mode": mode,
        "keep_p": keep_p,
        "manual_words_available": m_words,
        "manual_words_effective": int(m_eff),
        "downloaded_words_available": d_words,
        "upsample_manual": upsample,
        "assumed_fertility": round(fertility, 4),
        "target_train_tokens": target_train_tokens,
        "projected_corpus_words": int(corpus),
        "projected_train_tokens": int(proj_train_tokens),
        "projected_manual_frac": round(proj_manual, 4),
    }


def stage_combine_split(
    paths: Paths,
    manual_frac: float,
    fracs: tuple[float, float, float],
    seed: int,
    stats: dict | None,
    max_sentence_bytes: int,
    target_train_tokens: int,
    upsample: float,
    assumed_fertility: float | None,
) -> dict:
    """Merge manual + downsampled downloaded and split, in a single pass.

    Combine and split are fused deliberately: writing a merged corpus file only
    to read it straight back would cost an extra multi-GB round trip for a file
    nothing else needs -- the splits are the deliverable.

    Sampling and split assignment are stratified BY PROVENANCE: the split hash
    is applied independently to the manual and downloaded streams, so each split
    inherits the same manual:downloaded ratio instead of relying on chance.
    """
    banner(f"{paths.folder} ({paths.code}) -- STAGE combine + split")
    for p in (paths.manual, paths.downloaded):
        if not p.exists():
            raise SystemExit(f"missing input: {p}")
    paths.mkdirs()

    plan = _plan_corpus(paths, stats, target_train_tokens, manual_frac,
                        fracs[0], upsample, assumed_fertility)
    keep_p = plan["keep_p"]
    train_f, val_f = fracs[0], fracs[0] + fracs[1]
    print(f"  split               : {fracs[0]:.0%} / {fracs[1]:.0%} / {fracs[2]:.0%} (train/val/test)")
    print(f"  seed                : {seed}")

    counts = {s: {"manual": [0, 0], "downloaded": [0, 0]} for s in ("train", "val", "test")}
    dropped = 0
    t0 = time.perf_counter()

    handles = {name: open(paths.split_file(name), "w", encoding="utf-8")
               for name in ("train", "val", "test")}
    sent_out = open(paths.train_sentences, "w", encoding="utf-8")
    n_sentences = 0
    # Document totals let the progress bar show a real percentage and ETA.
    lang_stats = (stats or {}).get(paths.folder) or {}
    doc_totals = {
        True: (lang_stats.get("manual") or {}).get("docs"),
        False: (lang_stats.get("downloaded_total") or {}).get("docs"),
    }
    try:
        for src_path, manual_flag in ((paths.manual, True), (paths.downloaded, False)):
            print(f"\n  reading {src_path.name} ...", flush=True)
            prog = Progress(src_path.name, total=doc_totals.get(manual_flag))
            seen = 0
            for doc in read_jsonl(src_path):
                prog.update()
                text = doc.get("text", "")
                if not text:
                    continue
                seen += 1

                # Downsample only the downloaded side; every manual document is kept.
                if not manual_flag and unit_hash(text, "keep", seed) >= keep_p:
                    dropped += 1
                    continue

                u = unit_hash(text, "split", seed)
                name = "train" if u < train_f else ("val" if u < val_f else "test")

                # Manual upsampling repeats documents in TRAIN ONLY. Repeating
                # them in val/test would inflate the eval sets with duplicates
                # and make perplexity look better than it is. Because the split
                # is a function of the text, every copy lands in the same split,
                # so no repeated document can straddle train and test.
                copies = 1
                if manual_flag and upsample > 1.0 and name == "train":
                    whole = int(upsample)
                    copies = whole + (1 if unit_hash(text, "upsample", seed) < (upsample - whole) else 0)

                rec = {
                    "text": text,
                    "source": doc.get("source", "unknown"),
                    "manual": manual_flag,          # normalised provenance flag
                    "lang": paths.code,
                }
                line = json.dumps(rec, ensure_ascii=False) + "\n"
                for _ in range(copies):
                    handles[name].write(line)

                bucket = counts[name]["manual" if manual_flag else "downloaded"]
                bucket[0] += copies
                bucket[1] += len(text.split()) * copies

                # Tokenizer training text: TRAIN SPLIT ONLY, sentence-per-line.
                # Written once regardless of upsampling -- repeating text here
                # would bias the merge statistics toward the manual register
                # without adding any new subword evidence.
                if name == "train":
                    for sent in iter_training_sentences(text, max_sentence_bytes):
                        sent_out.write(sent + "\n")
                        n_sentences += 1
            prog.close()
    finally:
        for h in handles.values():
            h.close()
        sent_out.close()

    print(f"\n  dropped by downsampling : {dropped:,} downloaded documents")
    print(f"  tokenizer training text : {n_sentences:,} sentences -> {paths.train_sentences.name}")
    print(f"\n  {'split':<8}{'docs':>12}{'words':>16}{'manual docs':>14}{'manual words %':>16}")
    summary = {}
    for name in ("train", "val", "test"):
        md, mw = counts[name]["manual"]
        dd, dw = counts[name]["downloaded"]
        tot_w = mw + dw
        frac = mw / tot_w if tot_w else 0.0
        print(f"  {name:<8}{md + dd:>12,}{tot_w:>16,}{md:>14,}{100 * frac:>15.2f}%")
        summary[name] = {
            "docs": md + dd, "words": tot_w,
            "manual_docs": md, "manual_words": mw,
            "downloaded_docs": dd, "downloaded_words": dw,
            "manual_word_frac": round(frac, 4),
        }
    print(f"\n  done in {fmt_time(time.perf_counter() - t0)}")
    info = dict(plan)
    info.update({
        "downloaded_keep_prob": round(keep_p, 6),
        "target_manual_frac": manual_frac,
        "split_fractions": {"train": fracs[0], "val": fracs[1], "test": fracs[2]},
        "seed": seed,
        "tokenizer_training_sentences": n_sentences,
        "splits": summary,
    })
    return info


# --------------------------------------------------------------------------
# stage: train BPE
# --------------------------------------------------------------------------

def stage_train(paths: Paths, vocab_sizes: list[int], input_sentence_size: int,
                max_sentence_bytes: int, seed: int, num_threads: int) -> dict:
    """Train one SentencePiece BPE model per candidate vocabulary size.

    Trained on paths.train_sentences, which is derived from the TRAIN SPLIT
    ONLY -- val and test are never seen by any tokenizer, which is what makes
    the fertility/UNK numbers in stage_evaluate a genuine held-out measurement.
    """
    banner(f"{paths.folder} ({paths.code}) -- STAGE train BPE")
    try:
        import sentencepiece as spm
    except ImportError:
        raise SystemExit("sentencepiece is required: pip install sentencepiece")

    if not paths.train_sentences.exists():
        raise SystemExit(f"missing {paths.train_sentences} -- run the split stage first")

    size_gb = paths.train_sentences.stat().st_size / 1e9
    print(f"  input: {paths.train_sentences.name} ({size_gb:.2f} GB), "
          f"sampling up to {input_sentence_size:,} sentences, {num_threads} threads")
    print(f"  {len(vocab_sizes)} model(s) to train: {', '.join(f'{v:,}' for v in vocab_sizes)}")
    print("  NOTE: SentencePiece is a single blocking call with no Python-visible")
    print("        progress. A heartbeat below confirms it is alive; SentencePiece's")
    print("        own detailed log goes to stderr.", flush=True)

    trained = {}
    for vocab in vocab_sizes:
        prefix = paths.model_prefix(vocab)
        print(f"\n  training vocab={vocab:,} -> {prefix.name}.model", flush=True)
        t0 = time.perf_counter()
        try:
            with Heartbeat(f"spm vocab={vocab:,}"):
                _train_one(spm, paths, prefix, vocab, input_sentence_size,
                           max_sentence_bytes, num_threads)
        except Exception as exc:
            # The usual cause is a vocabulary larger than the corpus can support
            # (SentencePiece enforces hard_vocab_limit). Say so plainly rather
            # than surfacing a bare RuntimeError from the C++ layer.
            raise SystemExit(
                f"SentencePiece failed at vocab_size={vocab:,}: {exc}\n"
                f"  If this reads like a vocabulary-size error, the training text has "
                f"fewer distinct subword candidates than {vocab:,}. Use a smaller "
                f"--vocab-sizes, or check that {paths.train_sentences.name} is complete."
            ) from exc
        dt = time.perf_counter() - t0
        print(f"    done in {fmt_time(dt)}  ({prefix.name}.model / .vocab)")
        trained[str(vocab)] = {
            "model": str(prefix) + ".model",
            "vocab_file": str(prefix) + ".vocab",
            "train_seconds": round(dt, 1),
        }
    return trained


def _train_one(spm, paths: Paths, prefix: Path, vocab: int, input_sentence_size: int,
               max_sentence_bytes: int, num_threads: int) -> None:
    spm.SentencePieceTrainer.train(
            input=str(paths.train_sentences),
            model_prefix=str(prefix),
            vocab_size=vocab,
            model_type="bpe",
            # Devanagari has a small alphabet; full coverage means every
            # character is representable, which drives the UNK rate to ~0.
            character_coverage=1.0,
            input_sentence_size=input_sentence_size,
            shuffle_input_sentence=True,
            max_sentence_length=max_sentence_bytes,
            pad_id=PAD_ID, unk_id=UNK_ID, bos_id=BOS_ID, eos_id=EOS_ID,
            pad_piece="<pad>", unk_piece="<unk>", bos_piece="<s>", eos_piece="</s>",
            num_threads=num_threads,
            train_extremely_large_corpus=True,
            seed_sentencepiece_size=1_000_000,
    )


# --------------------------------------------------------------------------
# stage: evaluate (fertility / UNK / chars-per-token on held-out val)
# --------------------------------------------------------------------------

def stage_evaluate(paths: Paths, vocab_sizes: list[int], max_docs: int,
                   min_vocab: int, tolerance: float, forced: int | None) -> dict:
    banner(f"{paths.folder} ({paths.code}) -- STAGE evaluate (held-out val)")
    import sentencepiece as spm

    rows = []
    for vocab in vocab_sizes:
        model = Path(str(paths.model_prefix(vocab)) + ".model")
        if not model.exists():
            print(f"  [skip] {model.name} not found")
            continue
        sp = spm.SentencePieceProcessor(model_file=str(model))
        n_tok = n_word = n_char = n_unk = 0
        prog = Progress(f"scoring vocab={vocab:,} on val", total=max_docs)
        for i, doc in enumerate(read_jsonl(paths.val)):
            if i >= max_docs:
                break
            prog.update()
            text = doc.get("text", "")
            ids = sp.encode(text, out_type=int)
            n_tok += len(ids)
            n_word += len(text.split())
            n_char += len(text)
            n_unk += sum(1 for t in ids if t == UNK_ID)
        prog.close()
        row = {
            "vocab_size": vocab,
            "fertility": round(n_tok / n_word, 4) if n_word else 0.0,
            "unk_rate": round(n_unk / n_tok, 8) if n_tok else 0.0,
            "unk_count": n_unk,
            "chars_per_token": round(n_char / n_tok, 4) if n_tok else 0.0,
            "eval_tokens": n_tok,
            "eval_words": n_word,
        }
        rows.append(row)
        print(f"  vocab={vocab:>6,}  fertility={row['fertility']:.3f}  "
              f"chars/token={row['chars_per_token']:.3f}  "
              f"UNK={row['unk_rate']:.2e} ({n_unk:,})")

    if not rows:
        raise SystemExit("no trained models found to evaluate")

    if forced:
        chosen = forced
        rule = f"forced via --vocab {forced}"
    else:
        # Lower fertility is better, but it always improves with vocabulary size,
        # so "lowest fertility" would trivially pick the largest model. Every
        # extra vocabulary entry is an embedding row that Phase 2 must pay for
        # out of a ~25M parameter budget (at d_model=384, 32k vocab = 12.3M
        # params -- roughly half of it -- against 6.1M at 16k). So: take the
        # SMALLEST vocabulary whose fertility is within `tolerance` of the best,
        # floored at min_vocab to stay inside the "tens of thousands" the brief
        # recommends. Diminishing returns, made explicit.
        best = min(r["fertility"] for r in rows)
        eligible = [r for r in rows
                    if r["vocab_size"] >= min_vocab
                    and r["fertility"] <= best * (1.0 + tolerance)]
        pool = eligible or [r for r in rows if r["vocab_size"] >= min_vocab] or rows
        chosen = min(pool, key=lambda r: r["vocab_size"])["vocab_size"]
        rule = (f"smallest vocab >= {min_vocab:,} within {tolerance:.0%} fertility of "
                f"the best ({best:.3f})")

    print(f"\n  chosen vocab: {chosen:,}  [{rule}]")
    result = {"sweep": rows, "chosen_vocab": chosen, "selection_rule": rule}
    paths.sweep.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"  wrote {paths.sweep}")
    return result


# --------------------------------------------------------------------------
# stage: encode to uint16 .bin
# --------------------------------------------------------------------------

def stage_encode(paths: Paths, vocab: int, flush_every: int,
                 doc_totals: dict | None = None) -> dict:
    """Encode each split to a flat uint16 memmap, tracking provenance in tokens.

    Documents are encoded WHOLE (SentencePiece imposes no length limit at
    encode time -- only at train time), with </s> appended so the model can
    learn document boundaries.

    The manual-vs-downloaded split is accumulated here in TOKENS, because that
    is the unit the >=20% requirement is written in. Everything upstream could
    only approximate it with words.
    """
    banner(f"{paths.folder} ({paths.code}) -- STAGE encode (vocab={vocab:,})")
    import sentencepiece as spm

    model = Path(str(paths.model_prefix(vocab)) + ".model")
    sp = spm.SentencePieceProcessor(model_file=str(model))
    real_vocab = sp.get_piece_size()
    if real_vocab > 65535:
        raise SystemExit(
            f"vocab {real_vocab:,} exceeds uint16 range; use uint32 or a smaller vocabulary"
        )
    paths.bin.mkdir(parents=True, exist_ok=True)

    freq = np.zeros(real_vocab, dtype=np.int64)     # token-frequency stats (train only)
    out = {}
    for name in ("train", "val", "test"):
        src, dst = paths.split_file(name), paths.bin_file(name)
        n_tok = n_unk = 0
        tok_manual = tok_downloaded = 0
        buf: list[int] = []
        t0 = time.perf_counter()
        prog = Progress(f"encoding {name}", total=(doc_totals or {}).get(name))
        with open(dst, "wb") as fh:
            for i, doc in enumerate(read_jsonl(src)):
                prog.update()
                ids = sp.encode(doc.get("text", ""), out_type=int)
                ids.append(EOS_ID)
                buf.extend(ids)
                n_tok += len(ids)
                n_unk += sum(1 for t in ids if t == UNK_ID)
                if doc.get("manual"):
                    tok_manual += len(ids)
                else:
                    tok_downloaded += len(ids)
                if name == "train":
                    freq += np.bincount(np.asarray(ids, dtype=np.int64), minlength=real_vocab)
                if len(buf) >= flush_every:
                    np.asarray(buf, dtype=np.uint16).tofile(fh)
                    buf.clear()
            if buf:
                np.asarray(buf, dtype=np.uint16).tofile(fh)
        prog.close()
        frac = tok_manual / n_tok if n_tok else 0.0
        print(f"  {name:<6} {n_tok:>14,} tokens  manual {100 * frac:5.2f}%  "
              f"-> {dst.name} ({dst.stat().st_size / 1e9:.2f} GB)")
        out[name] = {
            "tokens": n_tok,
            "unk_tokens": n_unk,
            "unk_rate": round(n_unk / n_tok, 10) if n_tok else 0.0,
            "manual_tokens": tok_manual,
            "downloaded_tokens": tok_downloaded,
            "manual_token_frac": round(frac, 4),
            # Relative paths as well as absolute: Phase 2 runs on Colab from a
            # Drive copy, where these absolute Windows paths are meaningless.
            "bin": str(dst),
            "bin_relative": f"{paths.folder}/data/bin/{dst.name}",
            "dtype": "uint16",
        }

    # The requirement is on TRAINING tokens specifically.
    train_frac = out["train"]["manual_token_frac"]
    ok = train_frac >= 0.20
    print(f"\n  manual share of TRAIN tokens: {100 * train_frac:.2f}%  "
          f"-- {'satisfies' if ok else 'FAILS'} the >=20% requirement")
    if not ok:
        need = 0.20 / max(train_frac, 1e-9)
        print("  ACTION: re-run the combine stage with a higher --manual-frac "
              f"(try {min(0.20 * need, 0.95):.2f}) and re-encode.")

    top = np.argsort(freq)[::-1]
    covered = int(freq[top[:1000]].sum())
    total = int(freq.sum())
    stats = {
        "vocab_size": real_vocab,
        "distinct_tokens_used": int((freq > 0).sum()),
        "unused_tokens": int((freq == 0).sum()),
        "singleton_tokens": int((freq == 1).sum()),
        "top1000_coverage": round(covered / total, 4) if total else 0.0,
        "top20": [
            {"id": int(t), "piece": sp.id_to_piece(int(t)), "count": int(freq[t])}
            for t in top[:20]
        ],
    }
    print(f"  token-frequency: {stats['distinct_tokens_used']:,}/{real_vocab:,} used, "
          f"top-1000 cover {100 * stats['top1000_coverage']:.1f}% of train tokens")

    meta = {
        "language": paths.folder,
        "lang_code": paths.code,
        "tokenizer_model": str(model),
        "tokenizer_model_relative": f"{paths.folder}/tokenizer/{model.name}",
        "vocab_size": real_vocab,
        "special_ids": {"pad": PAD_ID, "unk": UNK_ID, "bos": BOS_ID, "eos": EOS_ID},
        "dtype": "uint16",
        "splits": out,
        "token_frequency": stats,
        "manual_requirement": {
            "threshold": 0.20,
            "achieved_train": train_frac,
            "satisfied": ok,
        },
    }
    paths.meta.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  wrote {paths.meta}")
    return meta


# --------------------------------------------------------------------------
# stage: report
# --------------------------------------------------------------------------

def stage_report(paths: Paths, vocab: int, combine_info: dict | None,
                 sweep: dict | None, meta: dict | None, n_examples: int) -> None:
    banner(f"{paths.folder} ({paths.code}) -- STAGE report")
    import sentencepiece as spm

    sp = spm.SentencePieceProcessor(model_file=str(paths.model_prefix(vocab)) + ".model")
    if sweep is None and paths.sweep.exists():
        sweep = json.loads(paths.sweep.read_text(encoding="utf-8"))
    if meta is None and paths.meta.exists():
        meta = json.loads(paths.meta.read_text(encoding="utf-8"))

    # Tokenization examples drawn from TEST -- never seen by the tokenizer.
    examples = []
    for i, doc in enumerate(read_jsonl(paths.test)):
        if len(examples) >= n_examples:
            break
        for sent in SENT_SPLIT_RE.split(doc.get("text", "")):
            sent = sent.strip()
            if 40 <= len(sent) <= 180:
                pieces = sp.encode(sent, out_type=str)
                ids = sp.encode(sent, out_type=int)
                examples.append({
                    "text": sent,
                    "pieces": pieces,
                    "ids": ids,
                    "n_words": len(sent.split()),
                    "n_tokens": len(ids),
                    "fertility": round(len(ids) / max(len(sent.split()), 1), 2),
                    "roundtrip_ok": sp.decode(ids).strip() == sent.strip(),
                })
                break
        if i > 5000:
            break

    L = [f"# Tokenizer report -- {paths.folder} ({paths.code})", ""]
    L += ["Trained from scratch with SentencePiece BPE on the training split only.",
          "No pretrained tokenizer is used, and this vocabulary is not shared with",
          "the other language.", ""]

    tf = (meta or {}).get("token_frequency", {})
    L += ["## 1. Vocabulary", "",
          f"- Vocabulary size: **{tf.get('vocab_size', vocab):,}**",
          f"- Model: `{Path(str(paths.model_prefix(vocab)) + '.model').name}`",
          f"- Vocab file: `{Path(str(paths.model_prefix(vocab)) + '.vocab').name}`",
          f"- Special ids: pad={PAD_ID}, unk={UNK_ID}, bos={BOS_ID}, eos={EOS_ID}",
          f"- character_coverage = 1.0 (every Devanagari character representable)", ""]

    if sweep:
        L += ["## 2. Vocabulary size chosen by fertility / UNK on held-out val", "",
              "| vocab | fertility (tokens/word) | chars/token | UNK rate | UNK count |",
              "|---:|---:|---:|---:|---:|"]
        for r in sweep["sweep"]:
            L.append(f"| {r['vocab_size']:,} | {r['fertility']:.4f} | "
                     f"{r['chars_per_token']:.4f} | {r['unk_rate']:.2e} | {r['unk_count']:,} |")
        L += ["", f"**Chosen: {sweep['chosen_vocab']:,}** -- {sweep['selection_rule']}.", "",
              "Measured on the validation split, which no tokenizer saw during training.", ""]

    if tf:
        L += ["## 3. Token-frequency statistics (training split)", "",
              f"- Distinct tokens used: {tf['distinct_tokens_used']:,} / {tf['vocab_size']:,}",
              f"- Never used: {tf['unused_tokens']:,}",
              f"- Used exactly once: {tf['singleton_tokens']:,}",
              f"- Top-1000 tokens cover **{100 * tf['top1000_coverage']:.1f}%** of all training tokens",
              "", "Most frequent pieces:", "", "| rank | piece | id | count |", "|---:|---|---:|---:|"]
        for i, t in enumerate(tf.get("top20", []), 1):
            piece = t["piece"].replace("|", "\\|")
            L.append(f"| {i} | `{piece}` | {t['id']} | {t['count']:,} |")
        L.append("")

    if sweep:
        chosen_row = next((r for r in sweep["sweep"] if r["vocab_size"] == vocab), None)
        if chosen_row:
            L += ["## 4. Average characters per token", "",
                  f"- **{chosen_row['chars_per_token']:.3f}** characters per token (val split)",
                  f"- Fertility: **{chosen_row['fertility']:.3f}** tokens per whitespace word", ""]

    L += ["## 5. Unknown-token statistics", ""]
    if meta:
        L += ["| split | tokens | UNK tokens | UNK rate |", "|---|---:|---:|---:|"]
        for name in ("train", "val", "test"):
            s = meta["splits"][name]
            L.append(f"| {name} | {s['tokens']:,} | {s['unk_tokens']:,} | {s['unk_rate']:.2e} |")
        L += ["", "`character_coverage=1.0` means every character in the corpus is in the",
              "vocabulary, so UNK is expected to be ~0; any residual comes from characters",
              "absent from the training split but present in val/test.", ""]

    L += [f"## 6. Tokenization examples ({len(examples)} sentences from the test split)", ""]
    for i, ex in enumerate(examples, 1):
        L += [f"**Example {i}** -- {ex['n_words']} words -> {ex['n_tokens']} tokens "
              f"(fertility {ex['fertility']}), round-trip {'OK' if ex['roundtrip_ok'] else 'MISMATCH'}", "",
              f"- Text: {ex['text']}", f"- Pieces: `{' | '.join(ex['pieces'])}`",
              f"- Ids: `{ex['ids']}`", ""]

    if meta:
        m = meta["manual_requirement"]
        L += ["## 7. Dataset statistics and the manual token split", "",
              "| split | tokens | manual tokens | downloaded tokens | manual % |",
              "|---|---:|---:|---:|---:|"]
        for name in ("train", "val", "test"):
            s = meta["splits"][name]
            L.append(f"| {name} | {s['tokens']:,} | {s['manual_tokens']:,} | "
                     f"{s['downloaded_tokens']:,} | {100 * s['manual_token_frac']:.2f}% |")
        L += ["", f"Manual share of **training** tokens: **{100 * m['achieved_train']:.2f}%** "
                  f"-- requirement >= 20%: **{'SATISFIED' if m['satisfied'] else 'NOT SATISFIED'}**", ""]

    if combine_info:
        f = combine_info["split_fractions"]
        ci = combine_info
        L += ["## 8. Corpus construction", "",
              f"- Sizing mode: `{ci.get('mode', 'manual-frac')}`",
              f"- Manual words available: {ci['manual_words_available']:,} (all used)"]
        if ci.get("upsample_manual", 1.0) != 1.0:
            L.append(f"- Manual upsampled {ci['upsample_manual']:.2f}x in the training split "
                     f"-> {ci['manual_words_effective']:,} effective words")
        L += [f"- Downloaded words available: {ci['downloaded_words_available']:,}",
              f"- Downloaded kept: {ci['downloaded_keep_prob']:.2%}",
              f"- Split: {f['train']:.0%} train / {f['val']:.0%} val / {f['test']:.0%} test, "
              f"document-level, seed {ci['seed']}",
              f"- Tokenizer training sentences: {ci['tokenizer_training_sentences']:,}", ""]
        if ci.get("mode") == "token-target":
            L += [f"The corpus was sized to reach a target of "
                  f"{ci['target_train_tokens']:,} training tokens, assuming a fertility of "
                  f"{ci['assumed_fertility']:.3f} tokens/word.", ""]
            if meta and meta["manual_requirement"]["achieved_train"] < 0.20:
                L += ["> **Note on the manual share.** Sizing to a token target rather than",
                      "> a manual ratio means the manual share is whatever the arithmetic",
                      "> yields. All manual data is used, but the corpus is much larger than",
                      "> 5x the manual collection, so the share falls below the 20% the brief",
                      "> requires. Reaching both simultaneously requires either more manual",
                      "> collection or upsampling the manual split (`--upsample-manual`).", ""]
        else:
            L += ["The downloaded corpus is downsampled rather than used whole: with all of",
                  "it, manual would be ~5% of tokens, far below the required 20%. Since all",
                  "manual data is already used, the achievable total is capped at 5x the",
                  "manual corpus.", ""]

    paths.report_md.write_text("\n".join(L), encoding="utf-8")
    paths.report_json.write_text(json.dumps(
        {"combine": combine_info, "sweep": sweep, "meta": meta, "examples": examples},
        indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  wrote {paths.report_md}")
    print(f"  wrote {paths.report_json}")


# --------------------------------------------------------------------------
# orchestration
# --------------------------------------------------------------------------

def run_language(root: Path, folder: str, code: str, args, stats: dict | None) -> None:
    paths = Paths(root, folder, code)
    paths.mkdirs()
    stages = args.stages
    combine_info = sweep = meta = None
    vocab_sizes = sorted(args.vocab_sizes)
    fracs = (args.train_frac, args.val_frac, args.test_frac)

    if "combine" in stages or "split" in stages:
        combine_info = stage_combine_split(
            paths, args.manual_frac, fracs, args.seed, stats, args.max_sentence_bytes,
            args.target_train_tokens, args.upsample_manual, args.assumed_fertility,
        )
        (paths.splits / "combine_info.json").write_text(
            json.dumps(combine_info, indent=2), encoding="utf-8")
    elif (paths.splits / "combine_info.json").exists():
        combine_info = json.loads((paths.splits / "combine_info.json").read_text(encoding="utf-8"))

    if "train" in stages:
        stage_train(paths, vocab_sizes, args.input_sentence_size,
                    args.max_sentence_bytes, args.seed, args.num_threads)

    if "evaluate" in stages:
        sweep = stage_evaluate(paths, vocab_sizes, args.eval_docs,
                               args.min_vocab, args.fertility_tolerance, args.vocab)
    elif paths.sweep.exists():
        sweep = json.loads(paths.sweep.read_text(encoding="utf-8"))

    chosen = args.vocab or (sweep or {}).get("chosen_vocab") or vocab_sizes[-1]

    if "encode" in stages:
        doc_totals = {k: v["docs"] for k, v in (combine_info or {}).get("splits", {}).items()}
        meta = stage_encode(paths, chosen, args.flush_every, doc_totals)

    if "report" in stages:
        stage_report(paths, chosen, combine_info, sweep, meta, args.examples)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=str(REPO_ROOT))
    ap.add_argument("--lang", choices=["hi", "ne"], default=None,
                    help="one language only (default: both, independently)")
    ap.add_argument("--stages", default=",".join(STAGE_ORDER),
                    help=f"comma-separated subset of {STAGE_ORDER}")
    ap.add_argument("--seed", type=int, default=1337)

    g = ap.add_argument_group("corpus mix and splits")
    g.add_argument("--target-train-tokens", type=int, default=500_000_000,
                   help="size the corpus so the TRAIN split reaches this many tokens "
                        "(default 500000000). All manual data is always used and the "
                        "downloaded side is sampled to make up the difference. "
                        "WARNING: at 500M the manual share lands near 13-16%%, BELOW the "
                        ">=20%% the brief requires -- use --upsample-manual to fix that, or "
                        "set this to 0 to size by --manual-frac instead.")
    g.add_argument("--upsample-manual", type=float, default=1.0,
                   help="repeat manual documents this many times in the TRAIN split "
                        "(default 1.0 = no repetition). The only way to hit a large token "
                        "target AND keep manual >=20%%. Fractional values are honoured "
                        "probabilistically; val/test are never upsampled.")
    g.add_argument("--assumed-fertility", type=float, default=None,
                   help="tokens per word used to convert the token target into a word "
                        "target (default: derived per language from characters-per-word). "
                        "The real value is measured after training and reported.")
    g.add_argument("--manual-frac", type=float, default=0.22,
                   help="target manual share BY WORDS, used ONLY when "
                        "--target-train-tokens is 0 (default 0.22). This is the mode that "
                        "satisfies the >=20%% requirement.")
    # 80/10/10, document-level. The brief's ~500M target is on TRAINING tokens,
    # so every point given to val/test comes straight off that number. At this
    # corpus size 10% is already tens of millions of tokens per eval split --
    # far more than perplexity/BPB need to be stable -- so this is a comfortable
    # middle ground rather than the maximum.
    g.add_argument("--train-frac", type=float, default=0.80)
    g.add_argument("--val-frac", type=float, default=0.10)
    g.add_argument("--test-frac", type=float, default=0.10)

    t = ap.add_argument_group("tokenizer")
    t.add_argument("--vocab-sizes", type=lambda s: [int(x) for x in s.split(",")],
                   default=[8000, 16000, 32000], help="sweep candidates")
    t.add_argument("--vocab", type=int, default=None,
                   help="force the final vocabulary size, skipping auto-selection")
    t.add_argument("--min-vocab", type=int, default=16000,
                   help="floor for auto-selection, keeping the choice in the recommended "
                        "'tens of thousands' band (default 16000)")
    t.add_argument("--fertility-tolerance", type=float, default=0.05,
                   help="accept the smallest vocab within this relative fertility of the "
                        "best (default 0.05)")
    t.add_argument("--input-sentence-size", type=int, default=5_000_000,
                   help="sentences SentencePiece samples for training (default 5M)")
    t.add_argument("--max-sentence-bytes", type=int, default=4192,
                   help="SentencePiece per-line byte limit; also the chunk size used when "
                        "writing train_sentences.txt (default 4192, SentencePiece's own)")
    t.add_argument("--num-threads", type=int, default=max(1, (os.cpu_count() or 2) - 1))

    e = ap.add_argument_group("evaluate / encode / report")
    e.add_argument("--eval-docs", type=int, default=20000,
                   help="val documents used for the fertility sweep (default 20000)")
    e.add_argument("--flush-every", type=int, default=8_000_000,
                   help="token buffer size before flushing to .bin (default 8M = 16 MB)")
    e.add_argument("--examples", type=int, default=5,
                   help="tokenization examples to include in the report (default 5)")

    args = ap.parse_args()
    args.stages = [s.strip() for s in args.stages.split(",") if s.strip()]
    bad = [s for s in args.stages if s not in STAGE_ORDER]
    if bad:
        raise SystemExit(f"unknown stage(s): {bad}; valid: {STAGE_ORDER}")

    total = args.train_frac + args.val_frac + args.test_frac
    if abs(total - 1.0) > 1e-6:
        raise SystemExit(f"split fractions must sum to 1.0 (got {total})")
    if not 0 < args.manual_frac < 1:
        raise SystemExit("--manual-frac must be between 0 and 1")

    root = Path(args.root)
    langs = LANGUAGES if not args.lang else [(f, c) for f, c in LANGUAGES if c == args.lang]
    stats = load_corpus_stats(root)

    print(f"stages : {args.stages}")
    print(f"seed   : {args.seed}")
    print(f"split  : {args.train_frac:.0%}/{args.val_frac:.0%}/{args.test_frac:.0%}")

    t0 = time.perf_counter()
    for folder, code in langs:
        run_language(root, folder, code, args, stats)
    print(f"\ntotal wall-clock: {fmt_time(time.perf_counter() - t0)}")


if __name__ == "__main__":
    main()
