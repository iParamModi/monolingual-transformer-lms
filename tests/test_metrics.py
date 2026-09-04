"""Tests for eval/metrics.py -- especially the ROUGE-L trap.

The ``rouge_score`` package tokenizes with a ``[^a-z0-9]+`` regex and silently
returns 0.0 for any Devanagari text. The test below is the guard against ever
reintroducing that dependency: an identical string scored against itself must
be a perfect 1.0, in Hindi exactly as in English.
"""

from __future__ import annotations

import pytest

from eval.metrics import (
    corpus_rouge_l,
    distinct_n,
    max_repeated_run,
    repetition_rate,
    rouge_l,
)

HINDI_SENTENCE = "भारतीय क्रिकेटर मोहम्मद शमी पत्नी से रिश्ते को लेकर विवादों में हैं"
NEPALI_SENTENCE = "पर्यटन व्यवसायीहरूले नेपालको पर्यटन क्षेत्रलाई व्यवस्थित बनाउन चाहन्छन्"


@pytest.mark.parametrize("text", [HINDI_SENTENCE, NEPALI_SENTENCE, "hello world this is english"])
def test_rouge_l_self_score_is_one(text):
    """A string against itself must score a perfect match on any script.

    This is exactly the case where ``rouge_score`` (the pip package) returns
    0.0 for Devanagari -- guards against that regression.
    """
    result = rouge_l(text, text)
    assert result["fmeasure"] == pytest.approx(1.0)
    assert result["precision"] == pytest.approx(1.0)
    assert result["recall"] == pytest.approx(1.0)


def test_rouge_l_no_overlap_scores_zero():
    result = rouge_l("बिल्कुल अलग शब्द यहाँ", "पूर्णतः भिन्न पाठ वहाँ")
    assert result["fmeasure"] == 0.0


def test_rouge_l_partial_overlap():
    """LCS-based F1 for a hand-checkable case."""
    # hyp = "a b c d", ref = "a b x d" -> LCS = "a b d" (length 3)
    result = rouge_l("a b c d", "a b x d")
    assert result["precision"] == pytest.approx(3 / 4)
    assert result["recall"] == pytest.approx(3 / 4)


def test_rouge_l_empty_string_scores_zero():
    result = rouge_l("", "कुछ पाठ")
    assert result["fmeasure"] == 0.0


def test_corpus_rouge_l_averages_sentence_scores():
    hyps = ["a b c", "x y z"]
    refs = ["a b c", "p q r"]
    # first pair: perfect (1.0), second: no overlap (0.0) -> mean 0.5
    assert corpus_rouge_l(hyps, refs) == pytest.approx(0.5)


def test_distinct_n_all_unique():
    assert distinct_n([1, 2, 3, 4, 5], 2) == pytest.approx(1.0)


def test_distinct_n_all_repeated():
    # [1,1] x5 -> every bigram is (1,1) -> 1 unique / 4 total
    assert distinct_n([1, 1, 1, 1, 1], 2) == pytest.approx(1 / 4)


def test_distinct_n_too_short_returns_zero():
    assert distinct_n([1], 2) == 0.0


def test_repetition_rate_is_inverse_of_distinct_n():
    ids = [1, 2, 1, 2, 1, 2, 1, 2]
    assert repetition_rate(ids, 2) == pytest.approx(1 - distinct_n(ids, 2))


def test_repetition_rate_no_repeats():
    assert repetition_rate([1, 2, 3, 4, 5, 6], 3) == 0.0


def test_max_repeated_run():
    assert max_repeated_run([1, 2, 2, 2, 3, 4, 4]) == 3


def test_max_repeated_run_no_repeats():
    assert max_repeated_run([1, 2, 3]) == 1


def test_max_repeated_run_empty():
    assert max_repeated_run([]) == 0
