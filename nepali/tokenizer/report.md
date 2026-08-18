# Tokenizer report -- nepali (ne)

Trained from scratch with SentencePiece BPE on the training split only.
No pretrained tokenizer is used, and this vocabulary is not shared with
the other language.

## 1. Vocabulary

- Vocabulary size: **10,000**
- Model: `ne_bpe_10000.model`
- Vocab file: `ne_bpe_10000.vocab`
- Special ids: pad=0, unk=1, bos=2, eos=3
- character_coverage = 1.0 (every Devanagari character representable)

## 2. Vocabulary size chosen by fertility / UNK on held-out val

| vocab | fertility (tokens/word) | chars/token | UNK rate | UNK count |
|---:|---:|---:|---:|---:|
| 5,000 | 1.9533 | 3.2130 | 2.04e-05 | 281 |
| 8,000 | 1.6769 | 3.7425 | 2.38e-05 | 281 |
| 10,000 | 1.5881 | 3.9520 | 2.51e-05 | 281 |

**Chosen: 10,000** -- smallest vocab >= 5,000 within 5% fertility of the best (1.588).

Measured on the validation split, which no tokenizer saw during training.

## 3. Token-frequency statistics (training split)

- Distinct tokens used: 9,997 / 10,000
- Never used: 3
- Used exactly once: 303
- Top-1000 tokens cover **62.7%** of all training tokens

Most frequent pieces:

| rank | piece | id | count |
|---:|---|---:|---:|
| 1 | `▁।` | 14 | 13,401,324 |
| 2 | `,` | 7865 | 8,639,186 |
| 3 | `▁र` | 17 | 5,830,319 |
| 4 | `▁छ` | 27 | 5,448,444 |
| 5 | `को` | 5 | 5,148,708 |
| 6 | `मा` | 6 | 4,350,381 |
| 7 | `।` | 7852 | 3,873,310 |
| 8 | `का` | 9 | 3,654,996 |
| 9 | `ले` | 15 | 3,443,693 |
| 10 | `▁पनि` | 75 | 3,228,450 |
| 11 | `लाई` | 50 | 2,875,301 |
| 12 | `▁हो` | 91 | 1,910,394 |
| 13 | `▁छन्` | 122 | 1,854,982 |
| 14 | `▁भएको` | 123 | 1,782,478 |
| 15 | `बाट` | 118 | 1,622,596 |
| 16 | `▁लागि` | 140 | 1,596,005 |
| 17 | `▁'` | 141 | 1,592,876 |
| 18 | `▁भने` | 108 | 1,528,721 |
| 19 | `▁गर्न` | 144 | 1,514,444 |
| 20 | `▁।\` | 160 | 1,444,498 |

## 4. Average characters per token

- **3.952** characters per token (val split)
- Fertility: **1.588** tokens per whitespace word

## 5. Unknown-token statistics

| split | tokens | UNK tokens | UNK rate |
|---|---:|---:|---:|
| train | 487,001,673 | 2,469 | 5.07e-06 |
| val | 60,899,013 | 480 | 7.88e-06 |
| test | 60,373,211 | 539 | 8.93e-06 |

`character_coverage=1.0` means every character in the corpus is in the
vocabulary, so UNK is expected to be ~0; any residual comes from characters
absent from the training split but present in val/test.

## 6. Tokenization examples (5 sentences from the test split)

**Example 1** -- 17 words -> 21 tokens (fertility 1.24), round-trip OK

- Text: - पर्यटन व्यवसायीहरूले नेपालको पर्यटन क्षेत्रलाई व्यवस्थित र लगानीमैत्री बनाउन व्यवसायीमैत्री पर्यटन ऐन आवश्यक भएको बताएका छन्।
- Pieces: `▁- | ▁पर्यटन | ▁व्यवसायी | हरूले | ▁नेपालको | ▁पर्यटन | ▁क्षेत्रलाई | ▁व्यवस्थित | ▁र | ▁लगानी | मैत्री | ▁बनाउन | ▁व्यवसायी | मैत्री | ▁पर्यटन | ▁ऐन | ▁आवश्यक | ▁भएको | ▁बताएका | ▁छन् | ।`
- Ids: `[334, 1766, 1881, 604, 940, 1766, 4195, 3800, 17, 877, 5745, 1030, 1881, 5745, 1766, 2057, 698, 123, 1306, 122, 7852]`

**Example 2** -- 14 words -> 19 tokens (fertility 1.36), round-trip OK

- Text: काठमाडौँ - नेपाली कांग्रेसको इतर समूहले शेरबहादुर देउवालाई सभापतिको हैसियतमा कार्यालय प्रवेश गराएको छ।
- Pieces: `▁काठमाडौँ | ▁- | ▁नेपाली | ▁कांग्रेसको | ▁इ | तर | ▁समूहले | ▁शेरबहादुर | ▁देउवा | लाई | ▁सभापति | को | ▁हैसिय | तमा | ▁कार्यालय | ▁प्रवेश | ▁गराएको | ▁छ | ।`
- Ids: `[1285, 334, 350, 3885, 268, 1193, 4378, 3525, 1717, 50, 1899, 5, 4430, 979, 634, 1704, 3442, 27, 7852]`

**Example 3** -- 15 words -> 30 tokens (fertility 2.0), round-trip OK

- Text: - विकेभिएमले सेन्ट जोसेफ स्कुललाई ४४-२१ अंकले हराउँदै विराटनगर बास्केटबल कप प्रतियोगितामा विजयी सुरुवात गर्यो।
- Pieces: `▁- | ▁वि | के | भि | एम | ले | ▁सेन्ट | ▁जो | से | फ | ▁स्कुल | लाई | ▁४४ | - | २ | १ | ▁अंकले | ▁हरा | उँदै | ▁विराटनगर | ▁बा | स्क | ेट | बल | ▁कप | ▁प्रतियोगितामा | ▁विजयी | ▁सुरुवात | ▁गर्यो | ।`
- Ids: `[334, 47, 631, 178, 1955, 15, 6349, 374, 895, 7873, 3640, 50, 4990, 7885, 7881, 7880, 7699, 2344, 984, 3074, 154, 1297, 227, 1255, 2652, 3882, 3548, 4908, 2183, 7852]`

**Example 4** -- 10 words -> 15 tokens (fertility 1.5), round-trip OK

- Text: काठमाडौँ - पत्रकार किशोर श्रेष्ठलाई सरकारकी वकिलको कार्यालयमा लगिएको छ।
- Pieces: `▁काठमाडौँ | ▁- | ▁पत्रकार | ▁किशोर | ▁श्रेष्ठ | लाई | ▁सरकार | की | ▁वकि | लको | ▁कार्यालयमा | ▁लग | िएको | ▁छ | ।`
- Ids: `[1285, 334, 1359, 6255, 1167, 50, 220, 213, 7122, 1008, 3162, 425, 834, 27, 7852]`

**Example 5** -- 9 words -> 12 tokens (fertility 1.33), round-trip OK

- Text: काठमाडौँ - नेप्से परिसूचक आज सामान्य अङ्कले बढेको छ।
- Pieces: `▁काठमाडौँ | ▁- | ▁नेप्से | ▁परिसूचक | ▁आज | ▁सामान्य | ▁अ | ङ्क | ले | ▁बढेको | ▁छ | ।`
- Ids: `[1285, 334, 5183, 7768, 549, 1415, 23, 1656, 15, 1553, 27, 7852]`

## 7. Dataset statistics and the manual token split

| split | tokens | manual tokens | downloaded tokens | manual % |
|---|---:|---:|---:|---:|
| train | 487,001,673 | 38,849,727 | 448,151,946 | 7.98% |
| val | 60,899,013 | 4,912,835 | 55,986,178 | 8.07% |
| test | 60,373,211 | 4,896,014 | 55,477,197 | 8.11% |

Manual share of **training** tokens: **7.98%** -- requirement >= 20%: **NOT SATISFIED**

## 8. Corpus construction

- Sizing mode: `token-target`
- Manual words available: 29,298,940 (all used)
- Downloaded words available: 679,546,227
- Downloaded kept: 53.74%
- Split: 80% train / 10% val / 10% test, document-level, seed 1337
- Tokenizer training sentences: 18,231,219

The corpus was sized to reach a target of 500,000,000 training tokens, assuming a fertility of 1.584 tokens/word.

> **Note on the manual share.** Sizing to a token target rather than
> a manual ratio means the manual share is whatever the arithmetic
> yields. All manual data is used, but the corpus is much larger than
> 5x the manual collection, so the share falls below the 20% the brief
> requires. Reaching both simultaneously requires either more manual
> collection or upsampling the manual split (`--upsample-manual`).
