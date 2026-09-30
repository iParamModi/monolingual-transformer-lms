# Phase 1 — Data Collection and Tokenizer Construction

**Model H = Hindi (`hi`)  ·  Model L = Nepali (`ne`)**

Two independent pipelines. No document, tokenizer, or vocabulary is shared
between the languages.

| | Hindi | Nepali |
|---|---:|---:|
| **Training tokens** | **479,808,984** | **487,001,673** |
| Validation / test tokens | 60,226,406 / 59,778,539 | 60,899,013 / 60,373,211 |
| Vocabulary size | 10,000 | 10,000 |
| Fertility (tokens/word) | 1.4675 | 1.5881 |
| Characters per token | 3.4715 | 3.9520 |
| UNK rate (train) | 5.27 × 10⁻⁶ | 5.07 × 10⁻⁶ |
| Manual share of training tokens | 13.46% | 7.98% |


---

## 1. Language selection

**Model H — Hindi** (`hi`, Devanagari). The higher-resource choice: five public
corpora yielded **997.8M cleaned words**, comfortably above a 500M-token budget.

**Model L — Nepali** (`ne`, Devanagari). From the permitted lower-resource list.
Identical collection and cleaning yielded **679.5M cleaned words** — 68% of
Hindi's volume — and a manual corpus 55% the size of Hindi's.

**Why this pair.** Both use Devanagari, so holding the script constant means the
Model H / Model L comparison in Phases 2–3 isolates **data scale and quality**
rather than confounding it with script or encoding differences. They remain
distinct enough that language identification is a genuine filtering step.

---

## 2. Data sources

**Downloaded** (`manual: false`) — HuggingFace streaming, capped at 500,000
documents per source:

| Source | Notes |
|---|---|
| CC-100 | Common-Crawl derived |
| Sangraha (unverified) | AI4Bharat raw web corpus |
| Sangraha (verified) | AI4Bharat curated / OCR corpus |
| mC4 | Multilingual C4 language slice |
| MADLAD-400 | Google/AllenAI cleaned web corpus (`clean` files only) |
| IndicCorp v2 | Returned 0 bytes for both languages — excluded |

**Manual** (`manual: true`) — own sitemap-driven news scraping with robots.txt
compliance and rate limiting. Per-document provenance (`source`, `url`,
`fetched_at`, `doc_id`) preserved.

---

## 3. Cleaning steps

Both corpora pass through the same eight stages, with per-stage checkpoints.

| Stage | Operation |
|---|---|
| Unicode normalisation | NFC + IndicNLP nukta/vowel-sign unification; strip control and zero-width characters; repair danda punctuation; collapse whitespace |
| Web-artifact removal | HTML tags and entities, URLs, e-mail addresses, `@handles` |
| Zero-English rule | Every Latin-letter run `[A-Za-z]+` removed (digits retained) |
| Language identification | fastText `lid.176`; documents below the confidence threshold for the target language are dropped |
| Quality filtering | Gopher-style: min/max word count, Devanagari script ratio, sentence-terminal punctuation ratio, duplicate-line fraction, binary-payload signature check |
| Exact deduplication | BLAKE2b hash over whitespace-normalised text |
| Perplexity filtering | Word-bigram LM **trained from scratch** on our own cleaned manual corpus; highest-perplexity tail dropped |
| Near-deduplication | Banded MinHash LSH, word 5-gram shingles, 64 permutations |

**Language identification is necessary, not cosmetic.** Hindi, Nepali, Marathi,
Bhojpuri, Maithili and Sanskrit all share Devanagari, so a script-ratio check
cannot separate them. Validated on held-out samples, the stage retained 94.5% of
Hindi and 90.5% of Nepali documents, with Hindi↔Nepali confusion below 1%. The
material rejections were genuine foreign-language content — roughly 26% of the
Hindi mC4 slice and 40% of the Nepali mC4 slice were not the target language.

**Deduplication matters because the sources overlap.** CC-100, mC4, MADLAD-400
and Sangraha-unverified are all substantially Common-Crawl derived, so the same
page recurs across sources; exact and near deduplication run over the merged
stream rather than per source.

`lid.176` is a language-**identification classifier**, not a language model or
tokenizer. No pretrained language model or tokenizer is used anywhere.

---

## 4. Corpus statistics — after cleaning

### Hindi

| | Documents | Words | Characters | Words/doc | Chars/word |
|---|---:|---:|---:|---:|---:|
| **Manual** | 139,672 | 53,678,691 | 275,931,416 | 384.3 | 5.14 |
| **Downloaded** | 1,777,998 | 997,797,163 | 5,015,779,965 | 561.2 | 5.03 |
| — MADLAD-400 | 480,710 | 405,373,021 | 2,026,815,318 | 843.3 | 5.00 |
| — Sangraha | 483,077 | 241,465,380 | 1,222,187,494 | 499.8 | 5.06 |
| — Sangraha (verified) | 453,734 | 185,752,748 | 936,353,099 | 409.4 | 5.04 |
| — mC4 | 288,838 | 161,730,260 | 813,337,419 | 559.9 | 5.03 |
| — CC-100 | 71,639 | 3,475,754 | 17,086,635 | 48.5 | 4.92 |
| **Total available** | 1,917,670 | 1,051,475,854 | 5,291,711,381 | 548.3 | 5.03 |

### Nepali

| | Documents | Words | Characters | Words/doc | Chars/word |
|---|---:|---:|---:|---:|---:|
| **Manual** | 85,917 | 29,298,940 | 184,674,162 | 341.0 | 6.30 |
| **Downloaded** | 1,709,231 | 679,546,227 | 4,339,227,608 | 397.6 | 6.39 |
| — MADLAD-400 | 405,005 | 311,460,937 | 1,950,527,409 | 769.0 | 6.26 |
| — Sangraha | 472,651 | 133,777,923 | 880,269,287 | 283.0 | 6.58 |
| — Sangraha (verified) | 446,571 | 122,724,687 | 790,964,805 | 274.8 | 6.45 |
| — mC4 | 259,744 | 105,494,478 | 679,011,022 | 406.1 | 6.44 |
| — CC-100 | 125,260 | 6,088,202 | 38,455,085 | 48.6 | 6.32 |
| **Total available** | 1,795,148 | 708,845,167 | 4,523,901,770 | 394.9 | 6.38 |

Nepali words are ~27% longer than Hindi words (6.38 vs 5.03 characters), which
directly explains its higher fertility in §6. CC-100 contributes almost nothing
after cleaning (0.3% of Hindi words) — its documents average ~48 words and are
mostly removed by the minimum-length filter.

---

## 5. Splits

All manual data is used; the downloaded side is downsampled so the training
split reaches the ~500M-token target.

| | Hindi | Nepali |
|---|---:|---:|
| Manual words used | 53,678,691 (100%) | 29,298,940 (100%) |
| Downloaded kept | 37.32% | 53.75% |
| Resulting corpus | 426,010,496 words | 394,520,893 words |

**80 / 10 / 10 at the document level**, so no sentence appears in two splits.
Assignment is a deterministic hash of the document text rather than a shuffle:
the splits are exactly reproducible from the seed (1337) without storing an
index, and manual and downloaded documents are stratified independently — hence
the near-identical manual share across all three splits below.

### Hindi

| Split | Documents | Words | Manual docs | Manual words | Manual % |
|---|---:|---:|---:|---:|---:|
| train | 641,920 | 341,053,026 | 111,631 | 43,038,371 | 12.62% |
| val | 80,701 | 42,798,849 | 14,129 | 5,425,466 | 12.68% |
| test | 80,118 | 42,516,577 | 13,912 | 5,214,854 | 12.27% |

### Nepali

| Split | Documents | Words | Manual docs | Manual words | Manual % |
|---|---:|---:|---:|---:|---:|
| train | 804,526 | 316,005,269 | 68,758 | 23,360,397 | 7.39% |
| val | 99,726 | 39,498,387 | 8,558 | 2,973,818 | 7.53% |
| test | 99,919 | 39,193,647 | 8,601 | 2,964,725 | 7.56% |

---

## 6. Tokenizer

**SentencePiece BPE, trained from scratch on the training split only.** Neither
tokenizer has seen validation or test text, which is what makes the held-out
measurements below genuine.

| Setting | Value |
|---|---|
| `model_type` | `bpe` |
| `character_coverage` | 1.0 |
| `input_sentence_size` | 5,000,000 (shuffled sample) |
| `max_sentence_length` | 4,192 bytes |
| Special IDs | pad=0, unk=1, bos=2, eos=3 |
| Training sentences | Hindi 12,924,512 · Nepali 18,231,219 |

SentencePiece silently skips input lines longer than 4,192 **bytes**. Devanagari
is 3 bytes per character, so that is ~1,400 characters — while the cleaned
documents average ~1,970 characters and MADLAD-400's average 843 words. Feeding
whole documents would therefore have discarded most of the corpus, training the
tokenizer only on the shortest documents. The tokenizer input is instead split
on Devanagari sentence punctuation (`।`, `॥`, `?`, `!`) with a byte-bounded
chunker for unpunctuated run-ons. This affects tokenizer *training* only;
encoding runs over whole documents.

### Vocabulary size selection

Three sizes trained per language and scored on the **held-out validation split**.
The 5,000–10,000 range follows course guidance.

**Hindi**

| Vocab | Fertility | Chars/token | UNK rate | UNK count |
|---:|---:|---:|---:|---:|
| 5,000 | 1.9145 | 2.6610 | 8.18 × 10⁻⁶ | 129 |
| 8,000 | 1.5569 | 3.2722 | 1.01 × 10⁻⁵ | 129 |
| **10,000** | **1.4675** | **3.4715** | 1.07 × 10⁻⁵ | 129 |

**Nepali**

| Vocab | Fertility | Chars/token | UNK rate | UNK count |
|---:|---:|---:|---:|---:|
| 5,000 | 1.9533 | 3.2130 | 2.04 × 10⁻⁵ | 281 |
| 8,000 | 1.6769 | 3.7425 | 2.38 × 10⁻⁵ | 281 |
| **10,000** | **1.5881** | **3.9520** | 2.51 × 10⁻⁵ | 281 |

**Rule: the smallest vocabulary whose fertility is within 5% of the best.**
Fertility improves monotonically with vocabulary size, so "lowest fertility"
would trivially select the largest candidate; every extra entry is an embedding
row Phase 2 must pay for from a ~25M parameter budget. Both languages selected
**10,000** — neither smaller candidate came within 5% (Hindi 8k is 6.1% worse,
Nepali 8k 5.6% worse).

At `d_model = 512` a 10,000 vocabulary costs 5.12M embedding parameters, ~20% of
the budget, leaving ~20M for transformer blocks (≈6 layers). A 32,000 vocabulary
would have taken 16.4M — 66% of the budget, leaving room for ~3 layers.

UNK counts are identical across vocabulary sizes within each language because
`character_coverage = 1.0` means the residual unknowns are characters absent
from the training split entirely — a property of the data, not the merge count.

---

## 7. Tokenized corpora and token statistics

| | Hindi | Nepali |
|---|---:|---:|
| **Train tokens** | **479,808,984** | **487,001,673** |
| Validation tokens | 60,226,406 | 60,899,013 |
| Test tokens | 59,778,539 | 60,373,211 |
| Total | 599,813,929 | 608,273,897 |

Encoded to flat `uint16` arrays with `</s>` after each document.

### Unknown-token statistics

| Split | Hindi UNK | Hindi rate | Nepali UNK | Nepali rate |
|---|---:|---:|---:|---:|
| train | 2,529 | 5.27 × 10⁻⁶ | 2,469 | 5.07 × 10⁻⁶ |
| val | 440 | 7.31 × 10⁻⁶ | 480 | 7.88 × 10⁻⁶ |
| test | 711 | 1.19 × 10⁻⁵ | 539 | 8.93 × 10⁻⁶ |

Effectively zero, as expected from full character coverage. The residue is
characters present in validation/test but absent from training.

### Token-frequency statistics (training split)

| | Hindi | Nepali |
|---|---:|---:|
| Distinct tokens used | 9,998 / 10,000 | 9,997 / 10,000 |
| Never used | 2 | 3 |
| Used exactly once | 471 | 303 |
| Top-1,000 coverage | 72.52% | 62.67% |

Nepali's flatter distribution reflects richer surface morphology — the same
lexical content spread across more token types.

| Rank | Hindi piece | Count | Nepali piece | Count |
|---:|---|---:|---|---:|
| 1 | `▁के` | 13,509,005 | `▁।` | 13,401,324 |
| 2 | `।` | 10,886,023 | `,` | 8,639,186 |
| 3 | `▁है` | 10,529,626 | `▁र` | 5,830,319 |
| 4 | `▁में` | 9,916,233 | `▁छ` | 5,448,444 |
| 5 | `,` | 9,426,933 | `को` | 5,148,708 |
| 6 | `▁की` | 7,811,098 | `मा` | 4,350,381 |
| 7 | `▁से` | 6,272,882 | `।` | 3,873,310 |
| 8 | `▁को` | 6,096,684 | `का` | 3,654,996 |
| 9 | `▁और` | 5,784,593 | `ले` | 3,443,693 |
| 10 | `▁का` | 4,951,935 | `▁पनि` | 3,228,450 |

The top pieces are function words and case markers in both languages —
free-standing postpositions in Hindi (`के`, `में`, `को`), and case suffixes
attached as word-internal pieces in Nepali (`को`, `मा`, `ले`), consistent with
its more agglutinative morphology.

---

## 8. Tokenization examples

From the **test split**, never seen by either tokenizer.

### Hindi

**12 words → 19 tokens** (fertility 1.58)

> ट्रक ड्राइवर से वसूली करते एआरटीओ के सिपाही का वीडियो वायरल

`▁ट्रक | ▁ड्राइवर | ▁से | ▁व | सू | ली | ▁करते | ▁ए | आर | टी | ओ | ▁के | ▁सि | पा | ही | ▁का | ▁वीडियो | ▁वायरल`

Frequent words (`ट्रक`, `ड्राइवर`, `वीडियो`, `वायरल`) stay whole; the acronym
`एआरटीओ` and the rarer `वसूली`, `सिपाही` fragment.

**13 words → 19 tokens** (fertility 1.46)

> सोना 50 रुपये ट्रटा, चांदी में 450 रुपये का नुकसान

`▁सोना | ▁50 | ▁रुपये | ▁ट्र | टा | , | ▁चा | ंदी | ▁में | ▁4 | 50 | ▁रुपये | ▁का | ▁नुकसान`

### Nepali

**17 words → 21 tokens** (fertility 1.24)

> पर्यटन व्यवसायीहरूले नेपालको पर्यटन क्षेत्रलाई व्यवस्थित र लगानीमैत्री बनाउन व्यवसायीमैत्री पर्यटन ऐन आवश्यक भएको बताएका छन्।

`▁पर्यटन | ▁व्यवसायी | हरूले | ▁नेपालको | ▁पर्यटन | ▁क्षेत्रलाई | ▁व्यवस्थित | ▁र | ▁लगानी | मैत्री | ▁बनाउन | ▁व्यवसायी | मैत्री | ▁पर्यटन | ▁ऐन | ▁आवश्यक | ▁भएको | ▁बताएका | ▁छन् | ।`

The plural-ergative suffix `हरूले` and the compounding element `मैत्री` are
learned as reusable subword units — what BPE should discover for a
morphologically rich language.

**15 words → 30 tokens** (fertility 2.00)

> विकेभिएमले सेन्ट जोसेफ स्कुललाई ४४-२१ अंकले हराउँदै विराटनगर बास्केटबल कप प्रतियोगितामा विजयी सुरुवात गर्यो।

`▁वि | के | भि | एम | ले | ▁सेन्ट | ▁जो | से | फ | ▁स्कुल | लाई | ▁४४ | - | २ | १ | ▁अंकले | ▁हरा | उँदै | ▁विराटनगर | ▁बा | स्क | ेट | बल | ▁कप | ▁प्रतियोगितामा | ▁विजयी | ▁सुरुवात | ▁गर्यो | ।`

A hard case: an acronym, transliterated proper nouns, and Devanagari digits all
fragment, giving fertility well above the corpus average of 1.59.

Round-trip decoding reproduces the input exactly except for internal newlines,
which SentencePiece's `nmt_nfkc` normaliser collapses to spaces.

---

## 9. Manual vs. downloaded token split

The brief requires ≥20% of training tokens from manual collection. **This was
not reached.**

| Split | Hindi total | Hindi manual | % | Nepali total | Nepali manual | % |
|---|---:|---:|---:|---:|---:|---:|
| train | 479,808,984 | 64,596,209 | **13.46%** | 487,001,673 | 38,849,727 | **7.98%** |
| val | 60,226,406 | 8,182,852 | 13.59% | 60,899,013 | 4,912,835 | 8.07% |
| test | 59,778,539 | 7,811,956 | 13.07% | 60,373,211 | 4,896,014 | 8.11% |

Measured in **tokens**, the unit the requirement is stated in, rather than
approximated from word counts.
