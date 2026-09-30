"""Encoding, prompt masking, and batching for reasoning finetuning.

Pretraining streamed fixed-width windows out of a `.bin` memmap. Finetuning
iterates variable-length prompt/answer pairs and supervises only the answer, so
the data path is different even though the training contract around it is the
same. Three things live here, and each one has a way of going wrong quietly.

**Prompt and answer are encoded separately, then concatenated.**
``sp.encode(prompt) + sp.encode(answer)`` -- never
``sp.encode(prompt + " " + answer)``. The two are not guaranteed to agree at the
boundary: SentencePiece could merge the last prompt token with the first answer
token into a piece that generation, which is only ever handed the prompt, can
never produce. Training on a sequence the inference path cannot reach is a bug
that shows up only as unexplained accuracy loss. Encoding the two sides
separately makes the training prefix byte-identical to the generation prompt by
construction, and :func:`encode_prompt` is the single function both paths call.

The space between them is not lost. SentencePiece marks word-initial pieces
with ``U+2581`` and adds that marker to the start of any string it encodes, so
``decode(encode("क") + encode("ख"))`` is ``"क ख"``. The documented
``prompt + " " + answer`` format is exactly what this produces.

**The mask boundary is an index, and it is off by one from the obvious guess.**
For ``ids = [p_0 ... p_{P-1}, a_0 ... a_{A-1}, eos]``, inputs are ``ids[:-1]``
and targets are ``ids[1:]``, so target position ``i`` is the prediction made
*from* input position ``i``. The first answer token ``a_0`` sits at index ``P``
in ``ids``, and it is predicted from index ``P-1``. So the supervised region of
the targets starts at ``P-1``, not at ``P``: positions ``[0 : P-1]`` are masked.
Masking ``[0 : P]`` instead would drop the one prediction that matters most --
the model's first guess at the answer -- while masking ``[0 : P-2]`` would
quietly train it to predict the final prompt token as well. Neither shows up in
the loss curve. :func:`mask_targets` is a pure function of two token lists
precisely so a test can pin the index without a tokenizer or a model.

**Padding is inert, and it is worth knowing why.** Batches are right-padded
with ``pad_id`` (0, reserved in Phase 1) in the inputs and ``IGNORE_INDEX``
(-100) in the targets. No attention mask is needed: the causal mask already
stops any real token from attending to a position to its right, and padding
only ever sits to the right. The targets being -100 keeps padding out of the
loss and out of the gradient. ``tests/test_finetune_data.py`` asserts this
numerically rather than taking it on trust.

**No BOS.** Pretraining packed documents as ``... doc </s> doc ...`` with no
BOS, so token id 2 was never trained and its embedding is still at
initialisation. Putting it in front of every finetuning example would ask the
model to condition on a vector it has never seen. :func:`build_examples`
rejects any sequence containing it.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import torch

from model import IGNORE_INDEX


def encode_prompt(sp, prompt: str) -> list[int]:
    """Token ids for a prompt, exactly as generation will be handed them.

    Task 4's evaluator calls this same function to build the context it decodes
    from. Keeping it in one place is what guarantees the finetuned model is
    asked to continue the sequence it was trained to continue.

    Args:
        sp: The language's SentencePieceProcessor.
        prompt: Prompt text.

    Returns:
        Token ids, no BOS, no trailing separator.
    """
    return sp.encode(prompt, out_type=int)


def encode_answer(sp, answer: str) -> list[int]:
    """Token ids for an answer, as a word-initial span.

    SentencePiece prefixes the encoding with its word-boundary marker, so
    concatenating this after :func:`encode_prompt` reproduces
    ``prompt + " " + answer`` on decode without any explicit space token.

    Args:
        sp: The language's SentencePieceProcessor.
        answer: Answer text.

    Returns:
        Token ids.
    """
    return sp.encode(answer, out_type=int)


def mask_targets(prompt_ids: list[int], answer_ids: list[int],
                 eos_id: int) -> tuple[list[int], list[int]]:
    """Build one training pair with the prompt masked out of the loss.

    Pure function of token ids: no tokenizer, no file, no model, so the mask
    boundary can be asserted directly.

    Args:
        prompt_ids: Prompt token ids (at least one).
        answer_ids: Answer token ids (at least one).
        eos_id: End-of-sequence id to append; it *is* supervised, because the
            model has to learn to stop after the answer rather than carry on
            generating.

    Returns:
        ``(input_ids, target_ids)``, both of length
        ``len(prompt_ids) + len(answer_ids)``. Exactly
        ``len(answer_ids) + 1`` target positions carry a real label.

    Raises:
        ValueError: If either side is empty.
    """
    if not prompt_ids:
        raise ValueError("prompt encoded to zero tokens")
    if not answer_ids:
        raise ValueError("answer encoded to zero tokens")

    ids = list(prompt_ids) + list(answer_ids) + [eos_id]
    input_ids = ids[:-1]
    target_ids = ids[1:]

    # Target position i is the prediction made from input position i, so the
    # first answer token (ids[P]) is predicted at index P-1. See module docstring.
    keep_from = len(prompt_ids) - 1
    target_ids = [IGNORE_INDEX] * keep_from + target_ids[keep_from:]
    return input_ids, target_ids


def read_jsonl(path: str | Path) -> list[dict]:
    """Read a reasoning split into memory.

    The splits are at most 20,000 short records, so streaming buys nothing and
    a list makes the seeded shuffling in :class:`ReasoningDataset` simple.
    """
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def build_examples(path: str | Path, sp, max_len: int = 128) -> list[dict]:
    """Encode a reasoning split into supervised training pairs.

    Args:
        path: A ``{train,val,test_*}.jsonl`` from ``finetune.gen_reasoning``.
        sp: The language's SentencePieceProcessor. Passing the wrong language's
            tokenizer is not detectable here -- callers get the language from
            ``eval.loader.load_for_eval``, which does check.
        max_len: Hard ceiling on encoded length. Exceeding it raises rather
            than truncating: cutting the left would corrupt the premises and
            cutting the right would remove the answer, so neither is a
            recoverable outcome.

    Returns:
        One dict per record with ``input_ids``, ``target_ids``, ``prompt_len``,
        ``n_supervised``, and the metadata Task 4 scores against (``id``,
        ``prompt``, ``answer``, ``candidates``, ``template``).

    Raises:
        ValueError: On an over-long sequence, an empty side, or a stray BOS.
    """
    eos_id, bos_id = sp.eos_id(), sp.bos_id()
    examples: list[dict] = []

    for record in read_jsonl(path):
        prompt_ids = encode_prompt(sp, record["prompt"])
        answer_ids = encode_answer(sp, record["answer"])
        input_ids, target_ids = mask_targets(prompt_ids, answer_ids, eos_id)

        total = len(input_ids) + 1  # the sequence before the shift dropped one
        if total > max_len:
            raise ValueError(
                f"{record.get('id', '?')} encodes to {total} tokens, above the "
                f"{max_len} cap -- regenerate with a lower --max-len or shorten "
                f"the template"
            )
        if bos_id in prompt_ids or bos_id in answer_ids:
            raise ValueError(
                f"{record.get('id', '?')} contains BOS (id {bos_id}); pretraining "
                f"never used it and its embedding is untrained"
            )

        examples.append({
            "id": record.get("id"),
            "prompt": record["prompt"],
            "answer": record["answer"],
            "candidates": record.get("candidates", []),
            "template": record.get("template"),
            "input_ids": input_ids,
            "target_ids": target_ids,
            "prompt_len": len(prompt_ids),
            "n_supervised": len(answer_ids) + 1,
        })

    if not examples:
        raise ValueError(f"{path} produced no examples")
    return examples


def collate(batch: list[dict], pad_id: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """Right-pad a list of examples into rectangular input and target tensors.

    Inputs are padded with ``pad_id`` and targets with ``IGNORE_INDEX``, so the
    padded positions contribute nothing to the loss. Padding sits only to the
    right, and the causal mask already prevents any real position from
    attending rightwards, so no attention mask is required.

    Args:
        batch: Examples from :func:`build_examples`.
        pad_id: Padding token, 0 by Phase 1 convention.

    Returns:
        ``(x, y)`` int64 CPU tensors of shape ``(len(batch), max_len_in_batch)``.
    """
    width = max(len(example["input_ids"]) for example in batch)
    x = torch.full((len(batch), width), pad_id, dtype=torch.long)
    y = torch.full((len(batch), width), IGNORE_INDEX, dtype=torch.long)
    for row, example in enumerate(batch):
        length = len(example["input_ids"])
        x[row, :length] = torch.tensor(example["input_ids"], dtype=torch.long)
        y[row, :length] = torch.tensor(example["target_ids"], dtype=torch.long)
    return x, y


class ReasoningDataset:
    """Seeded, resumable batching over encoded reasoning examples.

    Mirrors ``train.data.WindowedBinDataset``: the order for an epoch is
    *derived* from ``seed + epoch`` rather than stored, so resuming a run needs
    only the step count to replay the identical stream. That is what lets the
    finetuning checkpoint be exactly resumable in the same sense pretraining's
    was, rather than merely approximately.
    """

    def __init__(self, examples: list[dict], seed: int = 1337, pad_id: int = 0,
                 source: str | Path | None = None):
        """
        Args:
            examples: Output of :func:`build_examples`.
            seed: Base seed for the per-epoch permutation.
            pad_id: Padding token for :func:`collate`.
            source: Path the examples came from, recorded in :meth:`state` so a
                checkpoint says which file it was trained on.
        """
        if not examples:
            raise ValueError("ReasoningDataset needs at least one example")
        self.examples = examples
        self.seed = seed
        self.pad_id = pad_id
        self.source = str(source) if source is not None else None

    def __len__(self) -> int:
        return len(self.examples)

    def epoch_order(self, epoch: int) -> np.ndarray:
        """Example order for one epoch: a permutation seeded by ``seed + epoch``."""
        return np.random.default_rng(self.seed + epoch).permutation(len(self.examples))

    def batches_per_epoch(self, batch_size: int) -> int:
        """Full batches in one epoch; the trailing partial batch is dropped."""
        return len(self.examples) // batch_size

    def stream(self, batch_size: int, start_step: int = 0
               ) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        """Yield training batches indefinitely, resuming exactly at ``start_step``.

        Epochs run back to back, each with a fresh permutation. The trailing
        partial batch of an epoch is dropped so every optimizer step sees the
        same batch size.

        Args:
            batch_size: Examples per batch.
            start_step: Batches already consumed, for resume.

        Yields:
            ``(x, y)`` int64 CPU tensors.

        Raises:
            ValueError: If a batch would be larger than the whole dataset.
        """
        per_epoch = self.batches_per_epoch(batch_size)
        if per_epoch < 1:
            raise ValueError(
                f"batch_size {batch_size} exceeds the {len(self.examples)} "
                f"examples available"
            )
        epoch, within = divmod(start_step, per_epoch)
        while True:
            order = self.epoch_order(epoch)
            for index in range(within, per_epoch):
                chosen = order[index * batch_size: (index + 1) * batch_size]
                yield collate([self.examples[i] for i in chosen], self.pad_id)
            epoch += 1
            within = 0

    def fixed_batches(self, batch_size: int) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Every example once, in a fixed order -- for validation.

        Covers the whole split including a final partial batch, so validation
        loss is measured on all of it rather than on a resampled subset. Every
        validation pass must score the same tokens, or the curve reports
        sampling noise alongside learning.

        Args:
            batch_size: Examples per batch.

        Returns:
            A list of ``(x, y)`` pairs held in memory.
        """
        order = self.epoch_order(0)
        return [
            collate([self.examples[i] for i in order[start: start + batch_size]], self.pad_id)
            for start in range(0, len(order), batch_size)
        ]

    def state(self) -> dict:
        """Loader identity for the checkpoint's ``data_state`` field."""
        return {
            "kind": "reasoning",
            "source": self.source,
            "n_examples": len(self.examples),
            "seed": self.seed,
            "pad_id": self.pad_id,
        }

    def describe(self) -> str:
        """One-line summary for the finetuning log."""
        lengths = [len(example["input_ids"]) for example in self.examples]
        supervised = [example["n_supervised"] for example in self.examples]
        name = Path(self.source).name if self.source else "examples"
        return (
            f"{name}: {len(self.examples):,} examples, "
            f"mean {sum(lengths) / len(lengths):.1f} tokens "
            f"(max {max(lengths)}), "
            f"mean {sum(supervised) / len(supervised):.1f} supervised"
        )
