"""Tests for finetune/data.py -- the prompt mask, and proof that padding is inert.

Two failure modes here are silent, which is why this file exists.

**An off-by-one in the mask boundary changes what the model learns and nothing
else.** Mask one position too few and the model is also trained to predict the
last prompt token; one too many and it is never trained to produce the first
answer token at all. In both cases the loss falls, the curve looks healthy, and
the only symptom is an accuracy number that is worse than it should be with no
explanation. So the index is asserted directly, on hand-built token lists,
without a tokenizer or a model in the way.

**Padding that is not inert corrupts the loss in proportion to how ragged the
batch is.** The claim -- right-padding with ``pad_id`` in the inputs and
``IGNORE_INDEX`` in the targets contributes nothing, because the causal mask
already blocks rightward attention -- is checked numerically against a real
model rather than argued from the architecture.

The tokenizer-dependent tests skip when the Phase 1 artifacts are absent, so
the suite stays green on a fresh clone.
"""

from __future__ import annotations

import itertools
import unicodedata
from pathlib import Path

import pytest
import torch

from finetune.data import (
    ReasoningDataset,
    build_examples,
    collate,
    encode_answer,
    encode_prompt,
    mask_targets,
    read_jsonl,
)
from model import GPT, GPTConfig, IGNORE_INDEX

REPO_ROOT = Path(__file__).resolve().parents[1]
LANGS = {"hi": "hindi", "ne": "nepali"}
EOS = 3
PAD = 0


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture
def cfg() -> GPTConfig:
    """Small config with the real vocabulary size, dropout off for determinism."""
    return GPTConfig(
        name="tiny", lang="hi", vocab_size=10000, d_model=32, n_layers=2,
        n_heads=4, d_ff=64, max_seq_len=128, dropout=0.0, attn_dropout=0.0,
    )


def synthetic(prompt_ids: list[int], answer_ids: list[int]) -> dict:
    """An example dict built straight from token ids, with no tokenizer."""
    input_ids, target_ids = mask_targets(prompt_ids, answer_ids, EOS)
    return {
        "id": "synthetic",
        "prompt": "", "answer": "", "candidates": [], "template": 0,
        "input_ids": input_ids,
        "target_ids": target_ids,
        "prompt_len": len(prompt_ids),
        "n_supervised": len(answer_ids) + 1,
    }


def tokenizer_or_skip(lang: str):
    """Load the Phase 1 tokenizer, skipping if it or sentencepiece is missing."""
    spm = pytest.importorskip("sentencepiece")
    model = REPO_ROOT / LANGS[lang] / "tokenizer" / f"{lang}_bpe_10000.model"
    if not model.exists():
        pytest.skip(f"tokenizer not found: {model}")
    return spm.SentencePieceProcessor(model_file=str(model))


def nfc(text: str) -> str:
    """Canonical form, for comparing text that has been through the tokenizer.

    SentencePiece normalises its input (``nmt_nfkc`` by default), and several
    Devanagari nukta letters -- ``ड़`` in घड़ी, ``ज़`` in तेज़ -- are on Unicode's
    composition-exclusion list, so they come back as base + U+093C rather than
    as the precomposed character. That is canonically the same string and a
    different sequence of bytes. Comparing without normalising would make these
    tests fail on a difference that does not exist.

    Task 4's exact-match scorer has to do the same thing for the same reason:
    the model emits whatever the tokenizer's normaliser produced, which need not
    be byte-identical to the ``answer`` field in the JSONL.
    """
    return unicodedata.normalize("NFC", text)


def split_or_skip(lang: str, split: str = "train") -> Path:
    """Path to a generated split, skipping if Task 1 has not been run."""
    path = REPO_ROOT / LANGS[lang] / "reasoning" / f"{split}.jsonl"
    if not path.exists():
        pytest.skip(f"{path} not generated yet -- run finetune.gen_reasoning")
    return path


# --------------------------------------------------------------------------- #
# The mask boundary
# --------------------------------------------------------------------------- #


def test_mask_boundary_is_exact():
    """Targets are IGNORE_INDEX for exactly the first prompt_len - 1 positions.

    Worked by hand so the expected tensor is written out rather than computed
    by the same arithmetic being tested. With prompt [10, 11, 12] and answer
    [20, 21]:

        ids     = [10, 11, 12, 20, 21,  3]
        inputs  = [10, 11, 12, 20, 21]
        targets = [11, 12, 20, 21,  3]

    Position 2 (input 12, the last prompt token) is where the model must first
    produce the answer, so target index 2 = P - 1 is the first supervised one.
    """
    input_ids, target_ids = mask_targets([10, 11, 12], [20, 21], EOS)
    assert input_ids == [10, 11, 12, 20, 21]
    assert target_ids == [IGNORE_INDEX, IGNORE_INDEX, 20, 21, 3]


def test_single_token_prompt_supervises_everything():
    """With a one-token prompt, keep_from is 0 and nothing is masked."""
    input_ids, target_ids = mask_targets([10], [20, 21], EOS)
    assert input_ids == [10, 20, 21]
    assert target_ids == [20, 21, 3]
    assert IGNORE_INDEX not in target_ids


@pytest.mark.parametrize("n_prompt", [1, 2, 5, 17])
@pytest.mark.parametrize("n_answer", [1, 3, 8])
def test_supervised_count_is_answer_plus_eos(n_prompt, n_answer):
    """Exactly len(answer) + 1 positions are supervised -- the +1 is </s>.

    The model has to learn to stop, so the EOS prediction is trained too.
    """
    prompt_ids = list(range(100, 100 + n_prompt))
    answer_ids = list(range(200, 200 + n_answer))
    input_ids, target_ids = mask_targets(prompt_ids, answer_ids, EOS)

    supervised = [t for t in target_ids if t != IGNORE_INDEX]
    assert len(supervised) == n_answer + 1
    assert supervised == answer_ids + [EOS]
    assert len(input_ids) == len(target_ids) == n_prompt + n_answer


def test_masked_prefix_is_contiguous_and_leading():
    """No supervised position ever precedes a masked one."""
    _, target_ids = mask_targets([1, 2, 3, 4], [5, 6], EOS)
    masked = [t == IGNORE_INDEX for t in target_ids]
    assert masked == sorted(masked, reverse=True)


@pytest.mark.parametrize("prompt_ids,answer_ids", [([], [1]), ([1], [])])
def test_empty_side_raises(prompt_ids, answer_ids):
    """An empty prompt or answer is a data bug, not something to pad around."""
    with pytest.raises(ValueError):
        mask_targets(prompt_ids, answer_ids, EOS)


# --------------------------------------------------------------------------- #
# Collation
# --------------------------------------------------------------------------- #


def test_collate_pads_both_tensors_correctly():
    """Ragged examples become a rectangle: pad_id in inputs, -100 in targets."""
    batch = [synthetic([10, 11, 12], [20]), synthetic([30], [40, 41])]
    x, y = collate(batch, pad_id=PAD)

    assert x.shape == y.shape == (2, 4)
    assert x.dtype == y.dtype == torch.long

    # Row 0 is the longest (3 + 1 = 4), so it is unpadded.
    assert x[0].tolist() == [10, 11, 12, 20]
    assert y[0].tolist() == [IGNORE_INDEX, IGNORE_INDEX, 20, EOS]

    # Row 1 is length 3 and gets one column of padding.
    assert x[1].tolist() == [30, 40, 41, PAD]
    assert y[1].tolist() == [40, 41, EOS, IGNORE_INDEX]


def test_collate_width_matches_longest_example():
    batch = [synthetic([1], [2]), synthetic(list(range(10, 30)), [99])]
    x, _ = collate(batch, pad_id=PAD)
    assert x.shape[1] == max(len(e["input_ids"]) for e in batch)


# --------------------------------------------------------------------------- #
# Padding is inert -- checked against a real model
# --------------------------------------------------------------------------- #


def test_padded_batch_loss_equals_per_example_loss(cfg):
    """A ragged padded batch scores identically to the examples scored alone.

    This is the whole padding argument, as a number. ``F.cross_entropy`` with
    ``ignore_index`` reduces by the mean over *non-ignored* positions, so the
    batch loss is the supervised-count-weighted mean of the per-example losses.
    If padding leaked into either the attention or the loss, these would differ.

    Deliberately ragged: lengths 4, 3 and 8, so two rows carry padding and one
    does not.
    """
    torch.manual_seed(0)
    model = GPT(cfg).eval()
    batch = [
        synthetic([10, 11, 12], [20]),
        synthetic([30], [40, 41]),
        synthetic([50, 51, 52, 53, 54], [60, 61]),
    ]

    with torch.no_grad():
        x, y = collate(batch, pad_id=PAD)
        _, batch_loss, _ = model(x, y)

        total_nats, total_supervised = 0.0, 0
        for example in batch:
            xi = torch.tensor([example["input_ids"]], dtype=torch.long)
            yi = torch.tensor([example["target_ids"]], dtype=torch.long)
            _, loss_i, _ = model(xi, yi)
            n = sum(1 for t in example["target_ids"] if t != IGNORE_INDEX)
            total_nats += loss_i.item() * n
            total_supervised += n

    assert total_supervised == sum(e["n_supervised"] for e in batch)
    # 1e-4 rather than something tighter: a batched forward and a single-row
    # forward reduce their matmuls in a different order, which moves a loss of
    # ~9 by about 1e-5 in fp32. A masking error would move it by 0.1 or more.
    assert batch_loss.item() == pytest.approx(total_nats / total_supervised, abs=1e-4)


def test_padding_contributes_no_gradient(cfg):
    """Widening the padding must not change the gradient at all.

    The control for the test above: identical batches differing only in how
    much padding they carry must produce bit-comparable gradients. If padded
    positions reached the loss, extra padding would move them.
    """
    torch.manual_seed(0)
    model = GPT(cfg).eval()
    batch = [synthetic([10, 11, 12], [20]), synthetic([30], [40, 41])]

    def grad_norm(extra_pad: int) -> float:
        model.zero_grad(set_to_none=True)
        x, y = collate(batch, pad_id=PAD)
        if extra_pad:
            pad_x = torch.full((x.size(0), extra_pad), PAD, dtype=torch.long)
            pad_y = torch.full((y.size(0), extra_pad), IGNORE_INDEX, dtype=torch.long)
            x = torch.cat([x, pad_x], dim=1)
            y = torch.cat([y, pad_y], dim=1)
        _, loss, _ = model(x, y)
        loss.backward()
        return torch.nn.utils.clip_grad_norm_(model.parameters(), 1e9).item()

    assert grad_norm(0) == pytest.approx(grad_norm(6), rel=1e-4)


# --------------------------------------------------------------------------- #
# ReasoningDataset
# --------------------------------------------------------------------------- #


@pytest.fixture
def dataset() -> ReasoningDataset:
    """64 synthetic examples of varying length."""
    examples = [
        synthetic(list(range(10, 10 + 1 + (i % 5))), [100 + i, 200 + i][: 1 + (i % 2)])
        for i in range(64)
    ]
    return ReasoningDataset(examples, seed=1337, pad_id=PAD, source="synthetic.jsonl")


def test_epoch_order_is_a_permutation(dataset):
    order = dataset.epoch_order(0)
    assert sorted(order.tolist()) == list(range(len(dataset)))


def test_epochs_differ_but_are_reproducible(dataset):
    """Each epoch reshuffles; the same epoch always gives the same order."""
    assert dataset.epoch_order(0).tolist() != dataset.epoch_order(1).tolist()
    assert dataset.epoch_order(3).tolist() == dataset.epoch_order(3).tolist()


def test_stream_resumes_exactly(dataset):
    """Resuming at step k replays the same batches as skipping k from step 0.

    This is what makes the finetuning checkpoint exactly resumable rather than
    approximately: the order is derived from (seed, epoch), so a resumed run
    needs only the step count to land on the identical stream. Deliberately
    crosses an epoch boundary -- 2 batches of 32 per epoch, resuming at 3.
    """
    fresh = list(itertools.islice(dataset.stream(32, start_step=0), 6))
    resumed = list(itertools.islice(dataset.stream(32, start_step=3), 3))

    for offset, (x_r, y_r) in enumerate(resumed):
        x_f, y_f = fresh[3 + offset]
        assert torch.equal(x_f, x_r), f"inputs differ at resumed batch {offset}"
        assert torch.equal(y_f, y_r), f"targets differ at resumed batch {offset}"


def test_stream_drops_the_partial_batch(dataset):
    """Every optimizer step must see the same batch size."""
    assert dataset.batches_per_epoch(30) == 2  # 64 // 30, the last 4 dropped
    for x, _ in itertools.islice(dataset.stream(30), 5):
        assert x.shape[0] == 30


def test_fixed_batches_cover_every_example_once(dataset):
    """Validation scores the whole split, including the trailing partial batch."""
    batches = dataset.fixed_batches(10)
    assert sum(x.shape[0] for x, _ in batches) == len(dataset)
    assert batches[-1][0].shape[0] == 4
    # Same tokens on every pass, or the val curve measures resampling noise.
    again = dataset.fixed_batches(10)
    assert all(torch.equal(a[0], b[0]) for a, b in zip(batches, again))


def test_batch_larger_than_dataset_raises(dataset):
    with pytest.raises(ValueError):
        next(dataset.stream(len(dataset) + 1))


def test_state_records_the_source(dataset):
    state = dataset.state()
    assert state["n_examples"] == 64
    assert state["seed"] == 1337
    assert state["source"] == "synthetic.jsonl"


# --------------------------------------------------------------------------- #
# Against the real corpus and tokenizer
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("lang", ["hi", "ne"])
def test_build_examples_on_the_real_corpus(lang):
    """Encoding the generated corpus produces well-formed, BOS-free examples."""
    sp = tokenizer_or_skip(lang)
    examples = build_examples(split_or_skip(lang, "test_iid"), sp, max_len=128)

    assert len(examples) == 1000
    for example in examples:
        assert len(example["input_ids"]) == len(example["target_ids"])
        assert sp.bos_id() not in example["input_ids"]
        assert example["prompt_len"] >= 1
        supervised = [t for t in example["target_ids"] if t != IGNORE_INDEX]
        assert len(supervised) == example["n_supervised"]
        assert supervised[-1] == sp.eos_id()


@pytest.mark.parametrize("lang", ["hi", "ne"])
def test_training_prefix_matches_the_generation_prompt(lang):
    """The ids the model trains on are the ids generation will hand it.

    The reason prompt and answer are encoded separately. If ``build_examples``
    ever switched to encoding ``prompt + " " + answer`` as one string,
    SentencePiece could merge across the boundary and the training prefix would
    stop being reachable from the prompt alone -- which is exactly the sequence
    Task 4 decodes from.
    """
    sp = tokenizer_or_skip(lang)
    examples = build_examples(split_or_skip(lang, "test_iid"), sp, max_len=128)

    for example in examples[:200]:
        prompt_ids = encode_prompt(sp, example["prompt"])
        assert example["prompt_len"] == len(prompt_ids)
        assert example["input_ids"][:len(prompt_ids)] == prompt_ids


@pytest.mark.parametrize("lang", ["hi", "ne"])
def test_supervised_tokens_decode_to_the_answer(lang):
    """The supervised span is the answer and nothing else.

    Guards the mask boundary end to end: decoding only the positions that carry
    a label must reproduce the gold answer, in the model's own script.
    """
    sp = tokenizer_or_skip(lang)
    examples = build_examples(split_or_skip(lang, "test_iid"), sp, max_len=128)

    for example in examples[:200]:
        supervised = [t for t in example["target_ids"] if t != IGNORE_INDEX]
        assert nfc(sp.decode(supervised[:-1]).strip()) == nfc(example["answer"])


@pytest.mark.parametrize("lang", ["hi", "ne"])
def test_full_sequence_round_trips_to_prompt_and_answer(lang):
    """Decoding the whole input reproduces 'prompt answer'.

    Confirms the separator: SentencePiece's word-boundary marker supplies the
    space between the two halves, so the documented ``prompt + " " + answer``
    format is what actually reaches the model even though no space token is
    ever inserted by hand.
    """
    sp = tokenizer_or_skip(lang)
    examples = build_examples(split_or_skip(lang, "test_iid"), sp, max_len=128)

    for example in examples[:200]:
        text = sp.decode(example["input_ids"])
        expected = f"{example['prompt']} {example['answer']}"
        assert nfc(text.strip()) == nfc(expected.strip())


@pytest.mark.parametrize("lang", ["hi", "ne"])
def test_over_long_sequences_raise_rather_than_truncate(lang):
    """A too-tight cap must fail loudly.

    Truncating from the left would corrupt the premises and from the right
    would delete the answer, so neither is a recoverable outcome and silently
    doing either would produce a corpus that trains without complaint.
    """
    sp = tokenizer_or_skip(lang)
    with pytest.raises(ValueError, match="above the"):
        build_examples(split_or_skip(lang, "test_iid"), sp, max_len=8)


@pytest.mark.parametrize("lang", ["hi", "ne"])
def test_metadata_survives_encoding(lang):
    """Task 4 scores against candidates, so they must come through intact."""
    sp = tokenizer_or_skip(lang)
    path = split_or_skip(lang, "test_iid")
    records = read_jsonl(path)
    examples = build_examples(path, sp, max_len=128)

    for record, example in zip(records, examples):
        assert example["id"] == record["id"]
        assert example["answer"] == record["answer"]
        assert example["candidates"] == record["candidates"]
        assert example["answer"] in example["candidates"]


@pytest.mark.parametrize("lang", ["hi", "ne"])
def test_answer_encoding_is_context_independent(lang):
    """The same answer string always encodes to the same ids.

    Task 4 ranks candidates by appending each to the prompt. That comparison is
    only fair if a candidate's token ids do not depend on which prompt precedes
    them -- which holds because the two sides are encoded separately.
    """
    sp = tokenizer_or_skip(lang)
    examples = build_examples(split_or_skip(lang, "test_iid"), sp, max_len=128)

    seen: dict[str, list[int]] = {}
    for example in examples:
        ids = encode_answer(sp, example["answer"])
        assert seen.setdefault(example["answer"], ids) == ids
