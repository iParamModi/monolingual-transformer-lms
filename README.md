# Building and Analyzing Monolingual Transformer LMs in Two Indian Languages

Two fully independent decoder-only language models, built from scratch — corpus
collection, tokenizer training, architecture, pretraining, reasoning finetuning,
and evaluation.

| | Language | Script | Role |
|---|---|---|---|
| **Model H** | Hindi (`hi`) | Devanagari | Higher-resource |
| **Model L** | Nepali (`ne`) | Devanagari | Lower-resource |

No data, tokenizer, vocabulary, or weights are shared between the two models.
The implementation in `model/`, `train/`, `eval/` and `finetune/` is shared code
taking an explicit `--lang`; `eval/loader.py` raises if a checkpoint is paired
with another language's tokenizer, and `finetune/finetune.py` refuses to
finetune a model on the wrong language's data.

Multi-head attention, positional embeddings and the causal mask are written from
PyTorch primitives in `model/gpt.py`. No pretrained model, no pretrained
tokenizer, no HuggingFace Transformer class, and deliberately no
`F.scaled_dot_product_attention`.

---

## Reports

**All results and analysis live in the reports.** This file covers where the
large artifacts are and how to reproduce everything.

| Report | Covers |
|---|---|
| [`report/phase1.md`](report/phase1.md) | Corpus collection, cleaning pipeline, splits, tokenizer construction |
| [`report/phase2.md`](report/phase2.md) | Architecture, pretraining, perplexity/BPB, generation quality, attention analysis |
| [**`report/phase3.md`**](report/phase3.md) | **Reasoning finetuning, attention before/after, and the consolidated final comparison** |

`phase-3` is the complete project snapshot.

| Phase | Branch | State |
|---|---|---|
| 1 — Data collection & tokenizer | `phase-1` | Complete |
| 2 — Model, pretraining, evaluation | `phase-2` | Complete |
| 3 — Reasoning finetuning & final report | `phase-3` | Complete |

---

## At a glance

| | Model H (Hindi) | Model L (Nepali) |
|---|---:|---:|
| Training tokens | 479,808,984 | 487,001,673 |
| Vocabulary | 10,000 | 10,000 |
| Trainable parameters | 24,285,184 | 24,285,184 |
| Test perplexity | 26.36 | 36.79 |
| Bits per byte | 0.5448 | 0.5018 |
| chrF++ (best decoding setting) | 19.90 | 18.35 |
| Reasoning — seen patterns | 83.4% | 83.7% |
| Reasoning — unseen entity names | 75.0% | 62.0% |

Interpretation of every number above is in [`report/phase3.md`](report/phase3.md).

---

## Downloads

Large artifacts are hosted externally per the brief. All links are open — no
account, token, or access request needed.

### Checkpoints — Google Drive

| Phase | Link |
|---|---|
| **Pretrained (H and L)** | https://drive.google.com/drive/folders/1gkc_ZYkfKCbN8fNtMQdkip5CZENGzH-z?usp=sharing |
| **Finetuned (H and L)** | https://drive.google.com/drive/folders/1PQPmxPVySYu94SfD0UjgafV_Gg2LykUK?usp=sharing |

```
LMA-Phase2-Checkpoints/          LMA-Phase3-Finetuned/
  hindi/                           Hindi/
    ckpt_final.pt      278 MB        ckpt_ft_final.pt       278 MB
    hi_model_final.pt   93 MB        train_args.json
  nepali/                            config_snapshot.yaml
    ckpt_final.pt      278 MB      Nepali/
    ne_model_final.pt   93 MB        ckpt_ft_final.pt       278 MB
                                     train_args.json
                                     config_snapshot.yaml
```

| File | Contents | Load with |
|---|---|---|
| `ckpt_final.pt` · `ckpt_ft_final.pt` | weights + optimizer + scheduler + step + config + RNG and data-stream state | `train.checkpoint.load_checkpoint` — resumes training exactly |
| `{code}_model_final.pt` | weights + config only | `eval.loader.load_model` — all inference needs |

Pretrained checkpoints are at step 14,500 (475,136,000 tokens). Finetuned
checkpoints are at step 1,875 (3 epochs over 20,000 reasoning examples).
`train_args.json` records every resolved hyperparameter plus the SHA-256 of the
parent Phase 2 checkpoint, so the lineage of each finetuned model is verifiable;
`config_snapshot.yaml` is the architecture it was trained with.

**Only `ckpt_ft_final.pt` is uploaded, and it is the checkpoint every number in
the reports was measured on.** `ckpt_last.pt` is a duplicate of it. `ckpt_best.pt`
is omitted deliberately: for Nepali it is the same step, but for Hindi the lowest
validation loss came at step 1,800 (0.11930) rather than the final step 1,875
(0.12083). Uploading it would place two different Hindi models on Drive with
nothing indicating which produced the reported results.

For Phase 2 the same reasoning applies in the simpler form — validation loss was
lowest at the final step for both models, so `ckpt_best.pt` and `ckpt_last.pt`
are bit-identical to `ckpt_final.pt` in their weights.

### Corpora — HuggingFace

**https://huggingface.co/datasets/param-modi/hi-ne-monolingual-corpora**

| Artifact | Size | Link |
|---|---:|---|
| **Tokenized corpora** — `*/bin/*.bin` | 2.42 GB | [`hindi/bin`](https://huggingface.co/datasets/param-modi/hi-ne-monolingual-corpora/tree/main/hindi/bin) · [`nepali/bin`](https://huggingface.co/datasets/param-modi/hi-ne-monolingual-corpora/tree/main/nepali/bin) |
| Cleaned corpora — `*/processed/*.jsonl` | 24.07 GB | [`hindi/processed`](https://huggingface.co/datasets/param-modi/hi-ne-monolingual-corpora/tree/main/hindi/processed) · [`nepali/processed`](https://huggingface.co/datasets/param-modi/hi-ne-monolingual-corpora/tree/main/nepali/processed) |
| Manual corpus, merged | 2.11 GB | [`hindi/manual`](https://huggingface.co/datasets/param-modi/hi-ne-monolingual-corpora/tree/main/hindi/manual) · [`nepali/manual`](https://huggingface.co/datasets/param-modi/hi-ne-monolingual-corpora/tree/main/nepali/manual) |
| Raw downloads | 27.98 GB | [`hindi/raw`](https://huggingface.co/datasets/param-modi/hi-ne-monolingual-corpora/tree/main/hindi/raw) · [`nepali/raw`](https://huggingface.co/datasets/param-modi/hi-ne-monolingual-corpora/tree/main/nepali/raw) |

Reproducing Phases 2 and 3 needs only the **tokenized corpora** (2.42 GB).

The two IndicCorp v2 files are absent by design: that source returned 0 bytes for
both languages and was excluded ([`phase1.md`](report/phase1.md) §2).

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

# Devanagari font, so figure labels render as text rather than boxes
curl -L -o report/fonts/NotoSansDevanagari-Regular.ttf \
  https://github.com/googlefonts/noto-fonts/raw/main/hinted/ttf/NotoSansDevanagari/NotoSansDevanagari-Regular.ttf
```

Fetch the tokenized corpora (skip if starting from Phase 1):

```bash
hf download param-modi/hi-ne-monolingual-corpora --repo-type dataset \
    --include "hindi/bin/*" --include "nepali/bin/*" --local-dir ./data
```

Run every command below **from the repository root**. All randomness is seeded
(1337).

---

## Reproduction — Phase 1: data and tokenizers

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

# 3. Clean both sides (eight stages each)
python scripts/clean_manual.py --delete-intermediate
python scripts/clean_downloaded.py --delete-intermediate

# 4. Measure the cleaned corpora
python scripts/corpus_stats.py --json corpus_stats.json

# 5. Train tokenizers, build splits, encode to .bin
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

## Reproduction — Phase 2: pretraining and evaluation

**1. Verify the implementation**

```bash
python -m pytest tests/ -q          # 43 tests
python -m tests.test_causal         # causality evidence (phase2.md §1.5)
python -m tests.test_resume         # resume equivalence (phase2.md §2.3)
```

**2. Parameter counts** — both print `24,285,184`

```bash
python model/config.py --config hindi/configs/hi_25m.yaml
python model/config.py --config nepali/configs/ne_25m.yaml
```

**3. Freeze the evaluation subsets** — already committed under `*/data/eval/`;
only needed if regenerating

```bash
python scripts/make_eval_subset.py --lang hi
python scripts/make_eval_subset.py --lang ne
```

**4. Check throughput before committing GPU hours**

```bash
python -m train.benchmark --config hindi/configs/hi_25m.yaml \
    --data-dir ./data/hindi/bin --out-dir /tmp/bench --steps 50
```

Prints seconds/step, tokens/s, peak VRAM, and the projected total. If GPU memory
use is well under 70%, raise `--micro-batch` and lower `--grad-accum` to keep the
effective batch at 64.

**5. Pretrain** — about 3.3 hours per model on a Tesla T4

```bash
python -m train.train --config hindi/configs/hi_25m.yaml \
    --data-dir ./data/hindi/bin --out-dir hindi/checkpoints

python -m train.train --config nepali/configs/ne_25m.yaml \
    --data-dir ./data/nepali/bin --out-dir nepali/checkpoints
```

After any interruption add `--resume auto` — it picks up `ckpt_last.pt` and
replays the identical data stream. Before a long run,
`--overfit 200 --warmup-steps 20` trains on a single batch repeatedly; loss must
fall from ~9.2 toward 0, and a plateau means gradients are not flowing.

**6. Export weights-only checkpoints** — 278 MB → 93 MB

```bash
python -m train.export --ckpt hindi/checkpoints/ckpt_final.pt \
    --out hindi/checkpoints/hi_model_final.pt
python -m train.export --ckpt nepali/checkpoints/ckpt_final.pt \
    --out nepali/checkpoints/ne_model_final.pt
```

**7. Evaluate** — about 20 minutes per model on a T4

```bash
python -m eval.run_eval --lang hi --ckpt hindi/checkpoints/hi_model_final.pt \
    --data-dir ./data/hindi/bin --subset hindi/data/eval/test_subset.jsonl \
    --out-dir hindi/eval

python -m eval.run_eval --lang ne --ckpt nepali/checkpoints/ne_model_final.pt \
    --data-dir ./data/nepali/bin --subset nepali/data/eval/test_subset.jsonl \
    --out-dir nepali/eval
```

Writes perplexity, bits-per-byte, generation metrics, samples and attention
statistics into `{lang}/eval/`.

**8. Generate the Phase 2 figures** — CPU, about a minute

```bash
python scripts/make_figures.py
```

---

## Reproduction — Phase 3: reasoning finetuning

Needs the Phase 2 checkpoints at `{lang}/checkpoints/ckpt_final.pt`.

**1. Generate the reasoning corpora**

```bash
# Which digit script does each corpus actually use? Reports and exits.
python -m finetune.gen_reasoning --lang hi --digit-audit
python -m finetune.gen_reasoning --lang ne --digit-audit

python -m finetune.gen_reasoning --lang hi --out-dir hindi/reasoning  --n-train 20000 --seed 1337
python -m finetune.gen_reasoning --lang ne --out-dir nepali/reasoning --n-train 20000 --seed 1337
```

Byte-identical output for a given seed. The committed corpora used seed 1337.

**2. Check the loss mask before spending GPU time**

```bash
python -m finetune.finetune --config hindi/configs/hi_ft.yaml --overfit 200
```

Drives one batch to near-zero loss and exits without training. A final loss above
0.05 means the prompt mask is misaligned.

**3. Finetune** — 1,875 steps each

```bash
python -m finetune.finetune --config hindi/configs/hi_ft.yaml
python -m finetune.finetune --config nepali/configs/ne_ft.yaml
```

Add `--resume auto` to continue from `{lang}/checkpoints_ft/ckpt_last.pt`.

**4. Score the reasoning test sets** — pretrained and finetuned, both languages

```bash
python -m finetune.eval_reasoning --lang hi --ckpt hindi/checkpoints/ckpt_final.pt \
    --tag pretrained --data-dir hindi/reasoning --out-dir hindi/eval
python -m finetune.eval_reasoning --lang hi --ckpt hindi/checkpoints_ft/ckpt_ft_final.pt \
    --tag finetuned --data-dir hindi/reasoning --out-dir hindi/eval
python -m finetune.eval_reasoning --lang ne --ckpt nepali/checkpoints/ckpt_final.pt \
    --tag pretrained --data-dir nepali/reasoning --out-dir nepali/eval
python -m finetune.eval_reasoning --lang ne --ckpt nepali/checkpoints_ft/ckpt_ft_final.pt \
    --tag finetuned --data-dir nepali/reasoning --out-dir nepali/eval
```

**5. Compare attention before and after**

```bash
python -m finetune.attention_compare --lang hi \
    --pretrained hindi/checkpoints/ckpt_final.pt \
    --finetuned hindi/checkpoints_ft/ckpt_ft_final.pt \
    --data-dir hindi/reasoning --out-dir hindi/eval
python -m finetune.attention_compare --lang ne \
    --pretrained nepali/checkpoints/ckpt_final.pt \
    --finetuned nepali/checkpoints_ft/ckpt_ft_final.pt \
    --data-dir nepali/reasoning --out-dir nepali/eval
```

**6. Generate the Phase 3 figures**

```bash
python -m scripts.make_phase3_figures
```

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

finetune/                   reasoning finetuning
  gen_reasoning.py            generates the synthetic reasoning corpus
  data.py                     encoding, prompt masking, batching
  finetune.py                 the finetuning loop, CLI entry point
  eval_reasoning.py           exact match + forced-choice scoring, baselines
  attention_compare.py        attention before vs after finetuning

tests/                      43 tests
  test_causal.py              empirical proof the model cannot see the future
  test_model.py               shapes, attention, loss masking, parameters
  test_resume.py              interrupted run == uninterrupted run
  test_metrics.py             ROUGE-L on Devanagari, distinct-n, repetition

scripts/                    Phase 1 pipeline + figure generation
  scrape_manual.py            0 -- crawl the sites in sources.yaml
  combine_manual.py           1 -- merge shards
  clean_manual.py             2 -- cleaning, manual corpus
  clean_downloaded.py         2 -- cleaning, public corpora
  corpus_stats.py             3 -- corpus statistics
  build_tokenizer.py          4 -- splits, BPE training, encoding
  make_eval_subset.py         freeze the held-out evaluation subset
  make_figures.py             regenerate every Phase 2 figure
  make_phase3_figures.py      regenerate every Phase 3 figure

hindi/  ·  nepali/          one self-contained directory per language
  configs/*_25m.yaml          model architecture
  configs/*_ft.yaml           finetuning hyperparameters
  tokenizer/                  BPE models and vocabularies
  data/bin/                   uint16 token arrays        (not in git)
  data/splits/                train/val/test JSONL       (not in git)
  data/eval/test_subset.jsonl frozen 2,000-doc evaluation subset
  reasoning/                  synthetic reasoning corpus: train, val,
                              three test slices, stats.json
  logs/                       train_log.csv, finetune_log.csv, config snapshots
  eval/                       lm_eval.json, generation.json, attention_stats.json,
                              samples_*.jsonl, reasoning_*.json,
                              reasoning_samples_*.jsonl, attention_ft_stats.json
  checkpoints/                pretrained      (not in git -- Drive links above)
  checkpoints_ft/             finetuned       (not in git -- Drive links above)

report/
  phase1.md  phase2.md  phase3.md
  figures/                    40 figures
  fonts/                      Devanagari font, so figures regenerate identically
```

Corpora and checkpoints are excluded from git per the brief (see `.gitignore`);
tokenizer models and vocabularies are ~1 MB each and are committed.

---

## Deliverables

### Phase 1 — Data collection and tokenizer construction

| # | Deliverable | Location |
|---|---|---|
| 1 | Dataset collection scripts | `scripts/scrape_manual.py` + `*/data/sources.yaml`; `*/data/download_all.py` |
| 2 | Preprocessing pipelines | `scripts/combine_manual.py`, `clean_manual.py`, `clean_downloaded.py` |
| 3 | Dataset statistics | `corpus_stats.json`, [`phase1.md`](report/phase1.md) §4 |
| 4 | Train/val/test splits | `*/data/splits/combine_info.json`; regenerable via `build_tokenizer.py` (seed 1337) |
| 5 | Tokenizer training code | `scripts/build_tokenizer.py` |
| 6 | Vocabulary files | `hindi/tokenizer/hi_bpe_10000.vocab`, `nepali/tokenizer/ne_bpe_10000.vocab` |
| 7 | Tokenizer model files | `hindi/tokenizer/hi_bpe_10000.model`, `nepali/tokenizer/ne_bpe_10000.model` |

The 5,000 and 8,000 vocabulary candidates from the sweep are also committed;
results in `*/tokenizer/sweep_results.json`.

### Phase 2 — Model implementation, pretraining, evaluation

| # | Deliverable | Location |
|---|---|---|
| 1 | Transformer implementation (MHA, positional embeddings, causal mask) | `model/gpt.py` |
| 2 | Model configuration files | `hindi/configs/hi_25m.yaml`, `nepali/configs/ne_25m.yaml` |
| 3 | Parameter counts, training scripts, checkpoint links | [`phase2.md`](report/phase2.md) §1.3, `train/`, Drive link above |
| 4 | Training logs and loss curves | `*/logs/train_log.csv`, `report/figures/loss_*.png` |
| 5 | Perplexity / BPB tables; BLEU, chrF, ROUGE-L | [`phase2.md`](report/phase2.md) §3–4, `*/eval/lm_eval.json`, `*/eval/generation.json` |
| 6 | Generated samples, diversity and repetition stats | [`phase2.md`](report/phase2.md) §4.5, `*/eval/samples_*.jsonl` |
| 7 | Attention heatmaps, entropy and distance summaries | `report/figures/attn_*.png`, `*/eval/attention_stats.json` |
| 8 | Resource-level comparison | [`phase2.md`](report/phase2.md) §6 |

### Phase 3 — Reasoning finetuning, attention analysis, final report

| # | Deliverable | Location |
|---|---|---|
| 1 | Reasoning data generation scripts, finetune scripts, configs | `finetune/gen_reasoning.py`, `finetune/{data,finetune}.py`, `*/configs/*_ft.yaml` |
| 2 | Finetuned checkpoints (H and L) | Drive link above |
| 3 | Finetuning logs, metrics, qualitative examples | `*/logs/finetune_log.csv`, `*/eval/reasoning_*.json`, `*/eval/reasoning_samples_*.jsonl`, [`phase3.md`](report/phase3.md) §7 |
| 4 | Pretrain vs. finetune attention comparison | `*/eval/attention_ft_stats.json`, `report/figures/attn_{pre,ft,delta,profile}_*.png`, [`phase3.md`](report/phase3.md) §8 |
| 5 | Final comparison tables and error analysis | [`phase3.md`](report/phase3.md) §6–7, §9 |
| 6 | Complete final report and reproducibility instructions | [`phase3.md`](report/phase3.md); commands above |

Reasoning datasets: `*/reasoning/{train,val,test_iid,test_names,test_templates}.jsonl`,
with sizes, template variety, leakage controls and balance statistics in
`*/reasoning/stats.json`.

---

## Notes

**Hardware.** Phase 2 pretraining and evaluation ran on a Kaggle Tesla T4. Phase 3
finetuning and evaluation ran locally on CPU (`torch 2.13.0+cpu`), which is why
its wall-clock times are ~75 minutes per language rather than the ~10 a T4 would
take; the computation and resulting weights are identical either way. There is no
local GPU, which is why checkpoint-resume was treated as a functional requirement
rather than a formality.

**Precision.** `torch.cuda.is_bf16_supported()` returns `True` on a T4, which has
no bfloat16 tensor cores. `train/trainer.py::pick_amp_dtype` gates on compute
capability ≥ 8 instead and falls back to fp16 with a gradient scaler.

**ROUGE-L.** The `rouge_score` package tokenizes with `[^a-z0-9]+` and silently
returns 0.0 for Devanagari. `eval/metrics.py` implements it directly;
`tests/test_metrics.py` guards against regression. `sacrebleu` is required for
BLEU and chrF++ but not for figure generation.

**Figure fonts.** `eval/plots.py::setup_devanagari_font` installs Noto Sans
Devanagari *and* DejaVu Sans as a fallback list. Noto covers Devanagari and
little else, so setting it alone renders every English label — titles, axis
labels, legends — as empty boxes.

**Manual data.** Collected by scraping, not OCR; the brief permits either. The
manual share of training tokens reached 13.46% (H) and 7.98% (L) against the
brief's 20% target — all collected manual data was used, and the shortfall is
discussed in [`phase3.md`](report/phase3.md) §2 and §10.

---

## AI tool usage

Claude (Anthropic) was used as a coding assistant across all three phases: it
wrote the pipeline scripts in `scripts/`, the model and training code in `model/`
and `train/`, the evaluation suite in `eval/`, the reasoning finetuning code in
`finetune/`, the test suite, and drafts of the three reports from the committed
result files.

The design decisions are the author's and are defended in the reports — language
pair and source selection ([`phase1.md`](report/phase1.md) §1–2), cleaning stages
and thresholds (§3), the hash-based split (§5), vocabulary selection against the
parameter budget (§6); the architecture and depth/width trade-off
([`phase2.md`](report/phase2.md) §1.2–1.3), the equal-token-budget protocol
(§2.1), the separation of perplexity from bits-per-byte and the interpretation of
their disagreement (§3), the choice of chrF++ as the primary generation metric
(§4.3), the limits placed on what the comparison can conclude (§6.4); and the
reasoning task design and three-slice leakage split
([`phase3.md`](report/phase3.md) §4), the decision to report forced-choice
scoring alongside exact match (§6.1), and the reading of what the results do and
do not establish (§6.4, §9.2, §10).

Every figure reported was produced by running the committed code on the collected
data.
