[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/Q6gOCxoh)

# Building and Analyzing Monolingual Transformer LMs in Two Indian Languages

Two fully independent decoder-only language models, built from scratch — corpus
collection, tokenizer training, architecture, pretraining, and evaluation.

| | Language | Script | Role |
|---|---|---|---|
| **Model H** | Hindi (`hi`) | Devanagari | Higher-resource |
| **Model L** | Nepali (`ne`) | Devanagari | Lower-resource |

No data, tokenizer, vocabulary, or weights are shared between the two models.
The implementation in `model/`, `train/` and `eval/` is shared; every run takes
an explicit `--lang`, and `eval/loader.py` refuses to pair a checkpoint with a
mismatched tokenizer.

---

## Status

| Phase | Branch | State |
|---|---|---|
| **1 — Data collection & tokenizer** | `phase-1` | **Complete** |
| **2 — Model, pretraining, evaluation** | `phase-2` | **Complete** |
| 3 — Reasoning finetuning & final report | `phase-3` | Not started |

Reports: **[`report/phase1.md`](report/phase1.md)** · **[`report/phase2.md`](report/phase2.md)**

---

## Results

### Phase 2 — trained models

| | Model H (Hindi) | Model L (Nepali) |
|---|---:|---:|
| Trainable parameters | 24,285,184 | 24,285,184 |
| Training tokens seen | 475,136,000 | 475,136,000 |
| Test perplexity | **26.36** | **36.79** |
| **Bits per byte** | **0.5448** | **0.5018** |
| Unigram baseline perplexity | 1,355.2 | 2,419.0 |
| chrF++ (best setting) | 19.90 (t=0.5) | 18.35 (t=1.0) |
| Wall clock (Tesla T4) | 193.8 min | 203.1 min |

Both models were trained with identical architecture, hyperparameters, and token
budget, so every difference is attributable to the corpus. The headline finding
is that **Model L is 40% worse on perplexity but better on bits-per-byte** — the
two metrics disagree in direction, which [`report/phase2.md`](report/phase2.md)
§3.3 and §6 examine in detail.

### Phase 1 — corpora and tokenizers

| | Hindi | Nepali |
|---|---:|---:|
| Training tokens | 479,808,984 | 487,001,673 |
| Validation / test tokens | 60,226,406 / 59,778,539 | 60,899,013 / 60,373,211 |
| Vocabulary size | 10,000 | 10,000 |
| Fertility (tokens/word) | 1.4675 | 1.5881 |
| Characters per token | 3.4715 | 3.9520 |
| Manual share of training tokens | 13.46% | 7.98% |

Corpora were assembled from five public sources (CC-100, Sangraha verified and
unverified, mC4, MADLAD-400) plus our own scraped collection, cleaned through an
eight-stage pipeline, and tokenized with SentencePiece BPE trained from scratch
on the training split only.

---

## Large artifacts

### Model checkpoints (Phase 2)

**Google Drive:** https://drive.google.com/drive/folders/1gkc_ZYkfKCbN8fNtMQdkip5CZENGzH-z?usp=sharing

Shared as *Anyone with the link → Viewer*; no access request needed.

```
LMA-Phase2-Checkpoints/
  hindi/
    ckpt_final.pt          278 MB
    hi_model_final.pt       93 MB
  nepali/
    ckpt_final.pt          278 MB
    ne_model_final.pt       93 MB
```

| File | Size | Contents | Load with |
|---|---:|---|---|
| `{lang}/ckpt_final.pt` | 278 MB | model weights + optimizer + scheduler + training step + config + RNG and data-stream state | `train.checkpoint.load_checkpoint` — resumes training exactly |
| `{lang}/{code}_model_final.pt` | 93 MB | model weights + config only | `eval.loader.load_model` — what evaluation and Phase 3 use |

Both models are at step 14,500 (475,136,000 tokens). `ckpt_final.pt` is the
resume-capable checkpoint the brief requires; the weights-only export is
provided because it is a third of the size and is all that inference needs.

`ckpt_best.pt` and `ckpt_last.pt` are not uploaded: validation loss was lowest at
the final step, so their model weights are bit-identical to `ckpt_final.pt` and
they differ only in captured RNG and timing metadata.

### Corpora

Published as a public HuggingFace dataset — no account, token, or access request
needed:

**https://huggingface.co/datasets/param-modi/hi-ne-monolingual-corpora**

| Artifact | Size | Link |
|---|---|---|
| **Tokenized corpora** — `*/bin/*.bin` | 2.42 GB | [`hindi/bin`](https://huggingface.co/datasets/param-modi/hi-ne-monolingual-corpora/tree/main/hindi/bin) · [`nepali/bin`](https://huggingface.co/datasets/param-modi/hi-ne-monolingual-corpora/tree/main/nepali/bin) |
| Cleaned corpora — `*/processed/*.jsonl` | 24.07 GB | [`hindi/processed`](https://huggingface.co/datasets/param-modi/hi-ne-monolingual-corpora/tree/main/hindi/processed) · [`nepali/processed`](https://huggingface.co/datasets/param-modi/hi-ne-monolingual-corpora/tree/main/nepali/processed) |
| Manual corpus, merged | 2.11 GB | [`hindi/manual`](https://huggingface.co/datasets/param-modi/hi-ne-monolingual-corpora/tree/main/hindi/manual) · [`nepali/manual`](https://huggingface.co/datasets/param-modi/hi-ne-monolingual-corpora/tree/main/nepali/manual) |
| Raw downloads | 27.98 GB | [`hindi/raw`](https://huggingface.co/datasets/param-modi/hi-ne-monolingual-corpora/tree/main/hindi/raw) · [`nepali/raw`](https://huggingface.co/datasets/param-modi/hi-ne-monolingual-corpora/tree/main/nepali/raw) |

**Phase 2 needs only the tokenized corpora** (2.42 GB).

The two IndicCorp v2 files are absent by design: that source returned 0 bytes for
both languages and was excluded (see [`report/phase1.md`](report/phase1.md) §2).

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
# source .venv/bin/activate       # Linux / macOS
pip install -r requirements.txt
```

Two extra downloads, neither committed:

```bash
# fastText language-identification classifier (131 MB) -- Phase 1 cleaning only.
# This is a classifier, not a language model or tokenizer.
curl -L -o lid.176.bin https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin

# Devanagari font, so attention heatmap labels render as text rather than boxes
curl -L -o report/fonts/NotoSansDevanagari-Regular.ttf \
  https://github.com/googlefonts/noto-fonts/raw/main/hinted/ttf/NotoSansDevanagari/NotoSansDevanagari-Regular.ttf
```

Fetch the tokenized corpora:

```bash
hf download param-modi/hi-ne-monolingual-corpora --repo-type dataset \
    --include "hindi/bin/*" --include "nepali/bin/*" --local-dir ./data
```

---

## Running Phase 2

Run everything from the repository root.

### 1. Verify the implementation

```bash
python -m pytest tests/ -q          # 43 tests
python -m tests.test_causal         # causality evidence (report §1.5)
python -m tests.test_resume         # resume equivalence (report §2.3)
```

### 2. Parameter counts

```bash
python model/config.py --config hindi/configs/hi_25m.yaml
python model/config.py --config nepali/configs/ne_25m.yaml
```

Both print `24,285,184`.

### 3. Freeze the evaluation subsets

Needs the Phase 1 test splits locally. Already committed under
`*/data/eval/`, so this is only necessary if regenerating.

```bash
python scripts/make_eval_subset.py --lang hi
python scripts/make_eval_subset.py --lang ne
```

### 4. Check throughput before committing GPU hours

```bash
python -m train.benchmark --config hindi/configs/hi_25m.yaml \
    --data-dir ./data/hindi/bin --out-dir /tmp/bench --steps 50
```

Prints seconds/step, tokens/s, peak VRAM, and the projected total. If GPU memory
use is well under 70%, raise `--micro-batch` and lower `--grad-accum` to keep the
effective batch at 64.

### 5. Pretrain

About 3.3 hours per model on a Tesla T4.

```bash
python -m train.train --config hindi/configs/hi_25m.yaml \
    --data-dir ./data/hindi/bin --out-dir hindi/checkpoints

python -m train.train --config nepali/configs/ne_25m.yaml \
    --data-dir ./data/nepali/bin --out-dir nepali/checkpoints
```

After any interruption, add `--resume auto` — it picks up `ckpt_last.pt` and
replays the identical data stream.

Sanity check before a long run: `--overfit 200 --warmup-steps 20` trains on a
single batch repeatedly. The loss must fall from ~9.2 toward 0; if it plateaus,
gradients are not flowing.

### 6. Export weights-only checkpoints

```bash
python -m train.export --ckpt hindi/checkpoints/ckpt_final.pt \
    --out hindi/checkpoints/hi_model_final.pt
python -m train.export --ckpt nepali/checkpoints/ckpt_final.pt \
    --out nepali/checkpoints/ne_model_final.pt
```

278 MB → 93 MB, by dropping optimizer state.

### 7. Evaluate

About 20 minutes per model on a T4.

```bash
python -m eval.run_eval --lang hi --ckpt hindi/checkpoints/hi_model_final.pt \
    --data-dir ./data/hindi/bin --subset hindi/data/eval/test_subset.jsonl \
    --out-dir hindi/eval

python -m eval.run_eval --lang ne --ckpt nepali/checkpoints/ne_model_final.pt \
    --data-dir ./data/nepali/bin --subset nepali/data/eval/test_subset.jsonl \
    --out-dir nepali/eval
```

Writes perplexity, bits-per-byte, generation metrics, samples, and attention
statistics into `{lang}/eval/`.

### 8. Generate all figures

```bash
python scripts/make_figures.py
```

CPU, about a minute. Writes 16 PNGs to `report/figures/`.

All randomness is seeded (1337).

---

## Running Phase 1

```bash
# 0. Crawl the manual corpus. --discover expands each site's sitemap (or
#    MediaWiki / WordPress API) into a SQLite frontier; the second call drains
#    it. Both are resumable.
python scripts/scrape_manual.py --lang hi --discover
python scripts/scrape_manual.py --lang hi
python scripts/scrape_manual.py --lang ne --discover
python scripts/scrape_manual.py --lang ne

# 1. Merge the crawl shards into one JSONL per language
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

Splits are a deterministic hash of each document's text (seed 1337), so they
reproduce exactly on any machine without storing an index.

---

## Repository layout

```
model/                      the Transformer, from PyTorch primitives
  config.py                   GPTConfig, YAML loading, analytic parameter budget
  gpt.py                      CausalSelfAttention, FeedForward, Block, GPT

train/                      pretraining
  data.py                     windowed sampling over a .bin, seeded permutation
  scheduler.py                warmup + cosine, with state_dict
  checkpoint.py               atomic, resume-capable save/load
  trainer.py                  the training loop
  train.py                    CLI entry point
  benchmark.py                throughput and memory probe
  export.py                   full checkpoint -> weights-only

eval/                       evaluation
  loader.py                   load a checkpoint + its matching tokenizer
  lm_eval.py                  perplexity, bits-per-byte, unigram baseline
  generate.py                 continuation generation and scoring
  metrics.py                  BLEU-4, chrF++, ROUGE-L, distinct-n, repetition
  attention.py                entropy, distance, previous-token and sink rates
  plots.py                    every figure; enforces title/labels/legend
  run_eval.py                 runs the whole suite for one language

tests/                      43 tests
  test_causal.py              empirical proof the model cannot see the future
  test_model.py               shapes, attention, loss masking, parameters
  test_resume.py              interrupted run == uninterrupted run
  test_metrics.py             ROUGE-L on Devanagari, distinct-n, repetition

scripts/                    Phase 1 pipeline + evaluation utilities
  scrape_manual.py            0 -- crawl the sites in sources.yaml
  combine_manual.py           1 -- merge shards
  clean_manual.py             2 -- cleaning, manual corpus
  clean_downloaded.py         2 -- cleaning, public corpora
  corpus_stats.py             3 -- corpus statistics
  build_tokenizer.py          4 -- splits, BPE training, encoding
  make_eval_subset.py         freeze the held-out evaluation subset
  make_figures.py             regenerate every report figure

hindi/  ·  nepali/          one self-contained directory per language
  configs/*_25m.yaml          model configuration
  tokenizer/                  BPE models and vocabularies
  data/bin/                   uint16 token arrays        (not in git)
  data/splits/                train/val/test JSONL       (not in git)
  data/eval/test_subset.jsonl frozen 2,000-doc evaluation subset
  logs/                       train_log.csv, config snapshot, loss curve
  eval/                       lm_eval.json, generation.json, attention_stats.json,
                              samples_*.jsonl
  checkpoints/                (not in git -- Drive links above)

report/
  phase1.md                   Phase 1 report
  phase2.md                   Phase 2 report
  figures/                    16 figures
  fonts/                      Devanagari font, so figures regenerate identically
```

Corpora and checkpoints are excluded from git per the brief (see `.gitignore`);
tokenizer models and vocabularies are ~1 MB each and are committed.

---

## Deliverables

### Phase 2

| # | Deliverable | Location |
|---|---|---|
| 1 | Transformer implementation (MHA, positional embeddings, causal mask) | `model/gpt.py` |
| 2 | Model configuration files | `hindi/configs/hi_25m.yaml`, `nepali/configs/ne_25m.yaml` |
| 3 | Parameter counts, training scripts, checkpoint links | [`report/phase2.md`](report/phase2.md) §1.3, `train/`, Drive link above |
| 4 | Training logs and loss curves | `*/logs/train_log.csv`, `report/figures/loss_*.png` |
| 5 | Perplexity / BPB tables; BLEU, chrF, ROUGE-L | [`report/phase2.md`](report/phase2.md) §3–4, `*/eval/lm_eval.json`, `*/eval/generation.json` |
| 6 | Generated samples, diversity and repetition stats | [`report/phase2.md`](report/phase2.md) §4.5, `*/eval/samples_*.jsonl` |
| 7 | Attention heatmaps, entropy and distance summaries | `report/figures/attn_*.png`, `*/eval/attention_stats.json` |
| 8 | Resource-level comparison | [`report/phase2.md`](report/phase2.md) §6 |

### Phase 1

| # | Deliverable | Location |
|---|---|---|
| 1 | Dataset collection scripts | `scripts/scrape_manual.py` + `*/data/sources.yaml`; `*/data/download_all.py` |
| 2 | Preprocessing pipelines | `scripts/combine_manual.py`, `clean_manual.py`, `clean_downloaded.py` |
| 3 | Dataset statistics | `corpus_stats.json`, [`report/phase1.md`](report/phase1.md) §4 |
| 4 | Train/val/test splits | `*/data/splits/combine_info.json`; regenerable via `build_tokenizer.py` (seed 1337) |
| 5 | Tokenizer training code | `scripts/build_tokenizer.py` |
| 6 | Vocabulary files | `hindi/tokenizer/hi_bpe_10000.vocab`, `nepali/tokenizer/ne_bpe_10000.vocab` |
| 7 | Tokenizer model files | `hindi/tokenizer/hi_bpe_10000.model`, `nepali/tokenizer/ne_bpe_10000.model` |

The 5,000 and 8,000 vocabulary candidates from the sweep are also committed;
results are in `*/tokenizer/sweep_results.json`.

---

## Notes

**Hardware.** All training and evaluation ran on a Kaggle Tesla T4. There is no
local GPU, which is why checkpoint-resume was treated as a functional
requirement rather than a formality.

**Precision.** `torch.cuda.is_bf16_supported()` returns `True` on a T4, which has
no bfloat16 tensor cores. `train/trainer.py::pick_amp_dtype` gates on compute
capability ≥ 8 instead and falls back to fp16 with a gradient scaler.

**ROUGE-L.** The `rouge_score` package tokenizes with `[^a-z0-9]+` and silently
returns 0.0 for Devanagari. `eval/metrics.py` implements it directly;
`tests/test_metrics.py` guards against regression. `sacrebleu` is required for
BLEU and chrF++ but not for figure generation.

---

## AI tool usage

Claude (Anthropic) was used as a coding assistant across both phases: it wrote
the pipeline scripts in `scripts/`, the model and training code in `model/` and
`train/`, the evaluation suite in `eval/`, the test suite, and drafts of both
reports from the committed result files.

The design decisions are the author's and are defended in the reports —
language pair and source selection ([`phase1.md`](report/phase1.md) §1–2),
cleaning stages and thresholds (§3), the hash-based split (§5), vocabulary
selection against the parameter budget (§6); and for Phase 2, the architecture
and depth/width trade-off ([`phase2.md`](report/phase2.md) §1.2–1.3), the
equal-token-budget protocol (§2.1), the separation of perplexity from
bits-per-byte and the interpretation of their disagreement (§3), the choice of
chrF++ as the primary generation metric (§4.3), and the limits placed on what
the comparison can conclude (§6.4).

Every figure reported was produced by running the committed code on the
collected data.
