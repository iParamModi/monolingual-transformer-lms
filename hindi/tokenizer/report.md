# Tokenizer report -- hindi (hi)

Trained from scratch with SentencePiece BPE on the training split only.
No pretrained tokenizer is used, and this vocabulary is not shared with
the other language.

## 1. Vocabulary

- Vocabulary size: **10,000**
- Model: `hi_bpe_10000.model`
- Vocab file: `hi_bpe_10000.vocab`
- Special ids: pad=0, unk=1, bos=2, eos=3
- character_coverage = 1.0 (every Devanagari character representable)

## 2. Vocabulary size chosen by fertility / UNK on held-out val

| vocab | fertility (tokens/word) | chars/token | UNK rate | UNK count |
|---:|---:|---:|---:|---:|
| 5,000 | 1.9145 | 2.6610 | 8.18e-06 | 129 |
| 8,000 | 1.5569 | 3.2722 | 1.01e-05 | 129 |
| 10,000 | 1.4675 | 3.4715 | 1.07e-05 | 129 |

**Chosen: 10,000** -- smallest vocab >= 5,000 within 5% fertility of the best (1.468).

Measured on the validation split, which no tokenizer saw during training.

## 3. Token-frequency statistics (training split)

- Distinct tokens used: 9,998 / 10,000
- Never used: 2
- Used exactly once: 471
- Top-1000 tokens cover **72.5%** of all training tokens

Most frequent pieces:

| rank | piece | id | count |
|---:|---|---:|---:|
| 1 | `▁के` | 11 | 13,509,005 |
| 2 | `।` | 6530 | 10,886,023 |
| 3 | `▁है` | 9 | 10,529,626 |
| 4 | `▁में` | 22 | 9,916,233 |
| 5 | `,` | 6535 | 9,426,933 |
| 6 | `▁की` | 26 | 7,811,098 |
| 7 | `▁से` | 31 | 6,272,882 |
| 8 | `▁को` | 29 | 6,096,684 |
| 9 | `▁और` | 42 | 5,784,593 |
| 10 | `▁का` | 39 | 4,951,935 |
| 11 | `▁हैं` | 52 | 4,240,015 |
| 12 | `▁पर` | 46 | 3,731,135 |
| 13 | `.` | 6545 | 3,654,518 |
| 14 | `▁ने` | 62 | 3,278,182 |
| 15 | `▁कि` | 36 | 3,091,393 |
| 16 | `▁एक` | 68 | 2,919,640 |
| 17 | `▁भी` | 70 | 2,780,891 |
| 18 | `▁लिए` | 81 | 2,635,630 |
| 19 | `▁नहीं` | 90 | 2,301,869 |
| 20 | `-` | 6552 | 2,256,288 |

## 4. Average characters per token

- **3.471** characters per token (val split)
- Fertility: **1.468** tokens per whitespace word

## 5. Unknown-token statistics

| split | tokens | UNK tokens | UNK rate |
|---|---:|---:|---:|
| train | 479,808,984 | 2,529 | 5.27e-06 |
| val | 60,226,406 | 440 | 7.31e-06 |
| test | 59,778,539 | 711 | 1.19e-05 |

`character_coverage=1.0` means every character in the corpus is in the
vocabulary, so UNK is expected to be ~0; any residual comes from characters
absent from the training split but present in val/test.

## 6. Tokenization examples (5 sentences from the test split)

**Example 1** -- 12 words -> 19 tokens (fertility 1.58), round-trip OK

- Text: : ट्रक ड्राइवर से वसूली करते एआरटीओ के सिपाही का वीडियो वायरल
- Pieces: `▁: | ▁ट्रक | ▁ड्राइवर | ▁से | ▁व | सू | ली | ▁करते | ▁ए | आर | टी | ओ | ▁के | ▁सि | पा | ही | ▁का | ▁वीडियो | ▁वायरल`
- Ids: `[356, 4884, 6187, 31, 21, 1984, 129, 268, 47, 1050, 210, 6562, 11, 198, 286, 56, 39, 1022, 2540]`

**Example 2** -- 8 words -> 17 tokens (fertility 2.12), round-trip MISMATCH

- Text: सात दिवसीय आवासीय कैंप का शुभारंभ
हरिद्वार, संवाददाता।
- Pieces: `▁सात | ▁दिव | सीय | ▁आ | वासी | य | ▁कैंप | ▁का | ▁शु | भार | ंभ | ▁हरि | द्व | ार | , | ▁संवाददाता | ।`
- Ids: `[2188, 1471, 4617, 27, 1441, 6521, 6133, 39, 701, 1543, 1685, 4548, 2388, 12, 6535, 4183, 6530]`

**Example 3** -- 13 words -> 19 tokens (fertility 1.46), round-trip MISMATCH

- Text: सोना 50 रुपये ट्रटा, चांदी में 450 रुपये का नुकसान
नई दिल्ली, एजेंसी।
- Pieces: `▁सोना | ▁50 | ▁रुपये | ▁ट्र | टा | , | ▁चा | ंदी | ▁में | ▁4 | 50 | ▁रुपये | ▁का | ▁नुकसान | ▁नई | ▁दिल्ली | , | ▁एजेंसी | ।`
- Ids: `[4346, 1760, 804, 694, 419, 6535, 1305, 916, 22, 451, 2403, 804, 39, 1821, 910, 624, 6535, 3705, 6530]`

**Example 4** -- 8 words -> 13 tokens (fertility 1.62), round-trip MISMATCH

- Text: दहेज नहीं मिलने पर विवाहिता को पीटा
रुड़की।
- Pieces: `▁दह | ेज | ▁नहीं | ▁मिलने | ▁पर | ▁विवाह | िता | ▁को | ▁पी | टा | ▁रु | ड़की | ।`
- Ids: `[5092, 665, 90, 1606, 46, 2978, 811, 29, 347, 419, 424, 6456, 6530]`

**Example 5** -- 11 words -> 16 tokens (fertility 1.45), round-trip OK

- Text: इस जीवनी लेख में सत्यापन हेतु अतिरिक्त सन्दर्भों की आवश्यकता है।
- Pieces: `▁इस | ▁जीव | नी | ▁लेख | ▁में | ▁सत्या | पन | ▁हेतु | ▁अतिरिक्त | ▁सन् | दर्भ | ों | ▁की | ▁आवश्यकता | ▁है | ।`
- Ids: `[58, 2479, 96, 1107, 22, 5457, 955, 3188, 2075, 2760, 3117, 24, 26, 1183, 9, 6530]`

## 7. Dataset statistics and the manual token split

| split | tokens | manual tokens | downloaded tokens | manual % |
|---|---:|---:|---:|---:|
| train | 479,808,984 | 64,596,209 | 415,212,775 | 13.46% |
| val | 60,226,406 | 8,182,852 | 52,043,554 | 13.59% |
| test | 59,778,539 | 7,811,956 | 51,966,583 | 13.07% |

Manual share of **training** tokens: **13.46%** -- requirement >= 20%: **NOT SATISFIED**

## 8. Corpus construction

- Sizing mode: `token-target`
- Manual words available: 53,678,691 (all used)
- Downloaded words available: 997,797,163
- Downloaded kept: 37.32%
- Split: 80% train / 10% val / 10% test, document-level, seed 1337
- Tokenizer training sentences: 12,924,512

The corpus was sized to reach a target of 500,000,000 training tokens, assuming a fertility of 1.467 tokens/word.

> **Note on the manual share.** Sizing to a token target rather than
> a manual ratio means the manual share is whatever the arithmetic
> yields. All manual data is used, but the corpus is much larger than
> 5x the manual collection, so the share falls below the 20% the brief
> requires. Reaching both simultaneously requires either more manual
> collection or upsampling the manual split (`--upsample-manual`).
