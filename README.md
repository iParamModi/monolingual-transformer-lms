[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/Q6gOCxoh)

# Building and Analyzing Monolingual Transformer LMs in Two Indian Languages

Two fully independent decoder-only language models, built from scratch.

| | Language | Script | Role |
|---|---|---|---|
| **Model H** | Hindi (`hi`) | Devanagari | Higher-resource |
| **Model L** | Nepali (`ne`) | Devanagari | Lower-resource |

No data, tokenizer, vocabulary, or weights are shared between the two models.

---

## Status

| Phase | Branch | State |
|---|---|---|
| **1 — Data collection & tokenizer** | `phase-1` | **Complete** |
| 2 — Model, pretraining, evaluation | `phase-2` | Not started |
| 3 — Reasoning finetuning & final report | `phase-3` | Not started |

**Phase 1 report: [`report/phase1.md`](report/phase1.md)**

---

## Phase 1 results

| | Hindi | Nepali |
|---|---:|---:|
| Training tokens | **479,808,984** | **487,001,673** |
| Validation tokens | 60,226,406 | 60,899,013 |
| Test tokens | 59,778,539 | 60,373,211 |
| Vocabulary size | 10,000 | 10,000 |
| Fertility (tokens/word) | 1.4675 | 1.5881 |
| Characters per token | 3.4715 | 3.9520 |
| UNK rate | 5.3 × 10⁻⁶ | 5.1 × 10⁻⁶ |
| Manual share of training tokens | 13.46% | 7.98% |

Corpora were assembled from five public sources (CC-100, Sangraha verified and
unverified, mC4, MADLAD-400) plus our own scraped collection, then cleaned
through an eight-stage pipeline and tokenized with SentencePiece BPE trained
from scratch on the training split only.


---

## Repository layout

```
hindi/  ·  nepali/          one self-contained directory per language
  data/sources.yaml         manual-crawl source registry — sites, categories,
                              crawl priority, per-domain rate limit
  data/download_all.py      downloads the public corpora
  data/raw/                 raw downloads              (HuggingFace — not in git)
    manual/shards/            crawler output, shard-*.jsonl.zst
    manual/frontier.sqlite3   resumable crawl queue
    manual/*_manual_combined.jsonl    merged manual corpus
  data/processed/           cleaned corpora            (HuggingFace — not in git)
  data/splits/              train/val/test             (not in git)
    combine_info.json         split definitions
  data/bin/                 uint16 token arrays        (not in git)
    meta.json                 corpus metadata
  tokenizer/                BPE models, vocabularies, sweep, per-language report

scripts/                    the pipeline, in execution order
  scrape_manual.py          0 — crawl the sites in sources.yaml → shards
  combine_manual.py         1 — merge shards → one JSONL per language
  clean_manual.py           2 — cleaning pipeline, manually collected corpus
  clean_downloaded.py       2 — cleaning pipeline, public corpora
  corpus_stats.py           3 — exact corpus statistics
  build_tokenizer.py        4 — corpus assembly, splits, BPE training, encoding

report/phase1.md            Phase 1 report
corpus_stats.json           per-language and per-source statistics
requirements.txt
```

Corpus data is excluded from git per the brief (see `.gitignore`); the
tokenizer models and vocabularies are ~1 MB each and are committed.

### Phase 1 deliverables

| # | Deliverable | Location |
|---|---|---|
| 1 | Dataset collection scripts | Manual: `scripts/scrape_manual.py` driven by `*/data/sources.yaml`. Downloaded: `hindi/data/download_all.py`, `nepali/data/download_all.py` |
| 2 | Preprocessing pipelines | `scripts/combine_manual.py` (merge + dedup), `scripts/clean_manual.py`, `scripts/clean_downloaded.py` |
| 3 | Dataset statistics | `corpus_stats.json`, [`report/phase1.md`](report/phase1.md) §4 |
| 4 | Train/val/test splits | Definitions in `*/data/splits/combine_info.json`; regenerable from the [cleaned corpora](https://huggingface.co/datasets/param-modi/hi-ne-monolingual-corpora) via `build_tokenizer.py` (seed 1337) |
| 5 | Tokenizer training code | `scripts/build_tokenizer.py` |
| 6 | Vocabulary files | `hindi/tokenizer/hi_bpe_10000.vocab`, `nepali/tokenizer/ne_bpe_10000.vocab` |
| 7 | Tokenizer model files | `hindi/tokenizer/hi_bpe_10000.model`, `nepali/tokenizer/ne_bpe_10000.model` |

The 5,000 and 8,000 candidates from the vocabulary sweep are also committed;
sweep results are in `*/tokenizer/sweep_results.json`.

---

## Large artifacts

Corpora are published as a public HuggingFace dataset — no account, token or
access request is needed to browse or download:

**https://huggingface.co/datasets/param-modi/hi-ne-monolingual-corpora**

| Artifact | Size | Link |
|---|---|---|
| Raw downloaded data — `*/data/raw/*.jsonl` | 27.98 GB | [`hindi/raw`](https://huggingface.co/datasets/param-modi/hi-ne-monolingual-corpora/tree/main/hindi/raw) · [`nepali/raw`](https://huggingface.co/datasets/param-modi/hi-ne-monolingual-corpora/tree/main/nepali/raw) |
| Manual corpus, merged — `*/data/raw/manual/*_manual_combined.jsonl` | 2.11 GB | [`hindi/manual`](https://huggingface.co/datasets/param-modi/hi-ne-monolingual-corpora/tree/main/hindi/manual) · [`nepali/manual`](https://huggingface.co/datasets/param-modi/hi-ne-monolingual-corpora/tree/main/nepali/manual) |
| Cleaned corpora — `*/data/processed/*.jsonl` | 24.07 GB | [`hindi/processed`](https://huggingface.co/datasets/param-modi/hi-ne-monolingual-corpora/tree/main/hindi/processed) · [`nepali/processed`](https://huggingface.co/datasets/param-modi/hi-ne-monolingual-corpora/tree/main/nepali/processed) |

**Phase 2 needs only the tokenized corpora**, which are not yet uploaded; the
splits can be regenerated from the cleaned corpora with `build_tokenizer.py`
(the split assignment is a deterministic hash of each document, seed 1337, so it
reproduces exactly without storing an index).

Download everything, or one subdirectory:

```bash
pip install huggingface_hub
hf download param-modi/hi-ne-monolingual-corpora --repo-type dataset --local-dir ./corpus
hf download param-modi/hi-ne-monolingual-corpora --repo-type dataset \
    --include "hindi/processed/*" --local-dir ./corpus
```

The two IndicCorp v2 files are absent by design: that source returned 0 bytes
for both languages and was excluded (see [`report/phase1.md`](report/phase1.md) §2).

> **Redistribution note.** `raw/` and `processed/` derive from CC-100, mC4,
> MADLAD-400 and Sangraha, each under its own upstream terms. `manual/` contains
> full-text news articles whose copyright remains with the original publishers;
> it is shared for research reproducibility only. Per-document `source`, `url`
> and `fetched_at` fields are preserved so any document can be traced.

---

## Setup

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1        # Windows PowerShell
pip install -r requirements.txt
```

The cleaning pipeline uses fastText's `lid.176` language-**identification**
classifier (not a language model or tokenizer). It is 131 MB, so it is not
committed:

```bash
curl -L -o lid.176.bin https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin
```

---

## Reproducing Phase 1

```bash
# 0. Crawl the manual corpus. --discover expands each site's sitemap (or
#    MediaWiki / WordPress API) into a SQLite frontier; the second call drains
#    it, fetching and extracting. Both are resumable — rerun after any
#    interruption and the crawl continues where it stopped.
python scripts/scrape_manual.py --lang hi --discover
python scripts/scrape_manual.py --lang hi
python scripts/scrape_manual.py --lang ne --discover
python scripts/scrape_manual.py --lang ne

# 1. Merge the crawl shards into one JSONL per language
#    (dedup by doc_id, language guard, per-source register breakdown)
python scripts/combine_manual.py

# 2. Download the public corpora
python hindi/data/download_all.py
python nepali/data/download_all.py

# 3. Clean both sides
python scripts/clean_manual.py --delete-intermediate
python scripts/clean_downloaded.py --delete-intermediate

# 4. Measure the cleaned corpora
python scripts/corpus_stats.py --json corpus_stats.json

# 5. Train tokenizers, build splits, encode
python scripts/build_tokenizer.py --vocab-sizes 5000,8000,10000 --min-vocab 5000 \
    --stages train,evaluate

python scripts/build_tokenizer.py --lang hi --assumed-fertility 1.4671 \
    --vocab-sizes 5000,8000,10000 --min-vocab 5000 \
    --stages combine,split,evaluate,encode,report

python scripts/build_tokenizer.py --lang ne --assumed-fertility 1.5842 \
    --vocab-sizes 5000,8000,10000 --min-vocab 5000 \
    --stages combine,split,evaluate,encode,report
```

Every stage prints live progress with an ETA. All randomness is seeded
(`--seed 1337`), and splits are derived from a stable hash of each document's
text, so they are reproducible on any machine without storing an index.


---

## Loading the tokenized data (Phase 2)

```python
import json, numpy as np, sentencepiece as spm

meta = json.load(open("hindi/data/bin/meta.json", encoding="utf-8"))
train = np.memmap("hindi/data/bin/train.bin", dtype=np.uint16, mode="r")
sp = spm.SentencePieceProcessor(model_file="hindi/tokenizer/hi_bpe_10000.model")

meta["vocab_size"]     # 10000
meta["special_ids"]    # {"pad": 0, "unk": 1, "bos": 2, "eos": 3}
sp.decode(train[:50].tolist())
```

Documents are separated by `</s>` (id 3). `meta.json` also records the absolute
paths the arrays were written from, plus repo-relative equivalents.

---

## AI tool usage

Claude (Anthropic) was used as a coding assistant: it wrote the pipeline scripts
in `scripts/`, and did the execution labour of running each stage and collating
the reported numbers.

The design decisions are the author's and are defended in
[`report/phase1.md`](report/phase1.md) — language pair (§1), source selection
(§2), cleaning stages and filter thresholds (§3), the 80/10/10 hash-based split
(§5), the vocabulary-selection rule and its trade-off against the ~25M parameter
budget (§6), and prioritising the ~500M-token target over the ≥20% manual share
(§9).

Every figure reported here was produced by running the committed code on the
collected data.

---
