"""Generation-quality metrics: BLEU-4, chrF++, ROUGE-L, diversity, repetition.

ROUGE-L is implemented from scratch rather than using the ``rouge_score``
package: that package tokenizes with a ``[^a-z0-9]+`` regex, which strips every
Devanagari character and silently returns 0.0 for every Hindi or Nepali pair.
``tests/test_metrics.py`` checks this implementation against that failure mode
directly (a string scored against itself must be 1.0).
"""

from __future__ import annotations

from collections import Counter


def _sacrebleu():
    """Import sacrebleu on first use, with an actionable error if it is absent.

    Imported lazily rather than at module scope so that ROUGE-L, distinct-n and
    repetition -- none of which need it -- stay usable without the dependency.
    Otherwise a missing sacrebleu aborts collection of the whole test suite and
    blocks figure generation, neither of which involves BLEU or chrF++.
    """
    try:
        import sacrebleu
    except ImportError as exc:  # pragma: no cover -- environment-dependent
        raise ImportError(
            "BLEU and chrF++ need sacrebleu: pip install sacrebleu"
        ) from exc
    return sacrebleu


def corpus_bleu4(hyps: list[str], refs: list[str]) -> float:
    """Corpus-level BLEU-4.

    Uses sacrebleu's ``intl`` tokenizer rather than the default ``13a``: ``13a``
    is tuned for European languages and handles Devanagari punctuation
    (notably the danda `।`) poorly, while ``intl`` splits on Unicode
    punctuation classes generically. Neither loads a pretrained model --
    that would violate the "no pretrained tokenizer" constraint -- unlike
    sacrebleu's ``spm``/``flores200`` tokenizers, which must not be used here.

    Args:
        hyps: Generated continuations.
        refs: One reference continuation per hypothesis.

    Returns:
        BLEU-4 score, 0-100.
    """
    return _sacrebleu().corpus_bleu(hyps, [refs], tokenize="intl").score


def corpus_chrf_pp(hyps: list[str], refs: list[str]) -> float:
    """Corpus-level chrF++ (character n-grams + word order 2).

    Character-level, so it is script-agnostic and not sensitive to how the
    subword tokenizer happened to split a word -- the metric to actually
    compare on for Devanagari, where BLEU's word-level exact match is overly
    harsh on morphologically rich text.

    Args:
        hyps: Generated continuations.
        refs: One reference continuation per hypothesis.

    Returns:
        chrF++ score, 0-100.
    """
    return _sacrebleu().corpus_chrf(hyps, [refs], word_order=2).score


def _lcs_length(a: list[str], b: list[str]) -> int:
    """Length of the longest common subsequence of two token lists."""
    prev = [0] * (len(b) + 1)
    for token_a in a:
        curr = [0] * (len(b) + 1)
        for j, token_b in enumerate(b, start=1):
            curr[j] = prev[j - 1] + 1 if token_a == token_b else max(prev[j], curr[j - 1])
        prev = curr
    return prev[-1]


def rouge_l(hyp: str, ref: str) -> dict:
    """Sentence-level ROUGE-L: longest-common-subsequence precision/recall/F1.

    Tokenizes on whitespace only, so it works for any script -- unlike
    ``rouge_score``, which is English-only and returns 0.0 here.

    Args:
        hyp: Generated text.
        ref: Reference text.

    Returns:
        Dict with ``precision``, ``recall``, ``fmeasure``.
    """
    hyp_toks, ref_toks = hyp.split(), ref.split()
    if not hyp_toks or not ref_toks:
        return {"precision": 0.0, "recall": 0.0, "fmeasure": 0.0}
    lcs = _lcs_length(hyp_toks, ref_toks)
    precision = lcs / len(hyp_toks)
    recall = lcs / len(ref_toks)
    fmeasure = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return {"precision": precision, "recall": recall, "fmeasure": fmeasure}


def corpus_rouge_l(hyps: list[str], refs: list[str]) -> float:
    """Mean ROUGE-L F1 across a corpus of hypothesis/reference pairs."""
    scores = [rouge_l(h, r)["fmeasure"] for h, r in zip(hyps, refs)]
    return sum(scores) / len(scores) if scores else 0.0


def distinct_n(token_ids: list[int], n: int) -> float:
    """Distinct-n: fraction of n-grams that are unique.

    Low values indicate the model is repeating itself; measured over token ids
    (not text) so it is exact regardless of how pieces render.

    Args:
        token_ids: A stream of generated token ids, pooled across samples.
        n: N-gram order.

    Returns:
        unique n-grams / total n-grams, or 0.0 if there are fewer than n tokens.
    """
    if len(token_ids) < n:
        return 0.0
    grams = list(zip(*[token_ids[i:] for i in range(n)]))
    return len(set(grams)) / len(grams)


def repetition_rate(token_ids: list[int], n: int = 4) -> float:
    """Repetition rate: 1 - (unique n-grams / total n-grams).

    Reported at n=4 in addition to distinct-2 because distinct-3 tends to
    saturate near 1.0 on short generations and stops being informative;
    4-grams still show degeneration (a model stuck in a loop) clearly.

    Args:
        token_ids: A stream of generated token ids.
        n: N-gram order.

    Returns:
        Repetition rate in [0, 1]; 0 means every n-gram is unique.
    """
    if len(token_ids) < n:
        return 0.0
    grams = list(zip(*[token_ids[i:] for i in range(n)]))
    return 1 - len(set(grams)) / len(grams)


def max_repeated_run(token_ids: list[int]) -> int:
    """Longest run of the same token id repeated consecutively.

    Catches degenerate loops (e.g. one token repeated 40 times) that n-gram
    based measures can under-report if the rest of the sequence is varied.
    """
    if not token_ids:
        return 0
    best = run = 1
    for prev, cur in zip(token_ids, token_ids[1:]):
        run = run + 1 if cur == prev else 1
        best = max(best, run)
    return best


def top_ngram_counts(token_ids: list[int], n: int, top_k: int = 5) -> list[tuple[tuple[int, ...], int]]:
    """Most frequent n-grams, for spot-checking what a model is repeating."""
    if len(token_ids) < n:
        return []
    grams = Counter(zip(*[token_ids[i:] for i in range(n)]))
    return grams.most_common(top_k)
