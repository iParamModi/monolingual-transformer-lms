#!/usr/bin/env python
"""Finetune a pretrained checkpoint on that language's reasoning corpus.

The brief's protocol (PDF §3.1) is four requirements, and each maps to
something concrete here:

* *"Start from that language's own pretrained checkpoint"* -- ``--init``, and
  the architecture is read **out of** that checkpoint rather than re-specified.
  ``GPTConfig`` never appears on the command line, so a finetuning run cannot
  silently disagree with the model it is loading.
* *"Keep that language's tokenizer and vocabulary fixed"* -- no token is added
  and the embedding is never resized. The prompt/answer boundary is an index
  into the target tensor (``finetune/data.py``), not a special token, precisely
  so the vocabulary stays byte-identical to Phase 1's.
* *"Document all hyperparameters"* -- ``{lang}/configs/{code}_ft.yaml`` holds
  every one, and ``write_run_config`` drops the architecture snapshot and the
  resolved arguments next to the checkpoints.
* *"Save finetuned checkpoints in the same resume-capable format as
  pretraining"* -- ``train/checkpoint.py`` is imported verbatim, format
  unchanged, so Phase 2's loader reads a finetuned checkpoint without knowing
  anything about Phase 3.

**Why this is a separate class and not a subclass of ``train.Trainer``.**
Phase 2's trainer docstring anticipated subclassing. In practice its
``__init__`` opens two ``WindowedBinDataset`` memmaps inline, and reasoning data
is a list of ragged prompt/answer pairs with no ``.bin`` behind it -- so a
subclass would have to override the entire constructor, which is copying with
extra steps. Refactoring ``Trainer`` instead would mean editing Phase 2 code
that 43 passing tests depend on, on the branch that is the final submission.
Everything genuinely shared *is* imported rather than duplicated:
``pick_device``, ``pick_amp_dtype``, ``CosineWarmupLR``, and the whole of
``train/checkpoint.py``.

**What differs from pretraining, and why.**

======================  ==================  ==========================================
Setting                 Pretraining         Finetuning
======================  ==================  ==========================================
Peak LR                 6e-4                1e-4 -- adapt the model, do not overwrite it
Warmup                  500 steps           100 steps -- the run is 8x shorter
Optimizer state         fresh               fresh: AdamW's second moments from a
                                            6e-4 cosine run are wrong for 1e-4
Batch                   16 x 4 accum = 64   32, no accumulation (~900 tokens/batch)
Loss                    every token         answer tokens and ``</s>`` only
======================  ==================  ==========================================

Everything is held identical between Model H and Model L. If the reasoning
results differ, that is the data, not the budget -- the same control discipline
Phase 2 used.

Usage::

    python -m finetune.finetune --config hindi/configs/hi_ft.yaml
    python -m finetune.finetune --config nepali/configs/ne_ft.yaml

    # sanity gate before spending GPU time (see --overfit)
    python -m finetune.finetune --config hindi/configs/hi_ft.yaml --overfit 200

    # resume after an interrupted session
    python -m finetune.finetune --config hindi/configs/hi_ft.yaml --resume auto
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml

from eval.loader import load_tokenizer
from finetune.data import ReasoningDataset, build_examples
from model import GPT, GPTConfig, IGNORE_INDEX, count_params
from train.checkpoint import (
    load_checkpoint,
    resolve_resume,
    restore_rng,
    save_checkpoint,
    write_run_config,
)
from train.scheduler import CosineWarmupLR
from train.trainer import pick_amp_dtype, pick_device

LOG_COLUMNS = [
    "step", "epoch", "lr", "train_loss", "val_loss", "val_ppl",
    "answer_token_acc", "grad_norm", "tokens_seen", "supervised_seen",
    "peak_mem_gb", "elapsed_s",
]

#: Defaults for every hyperparameter. A YAML config overrides these, and an
#: explicit command-line flag overrides the YAML -- so the config file is the
#: record of what a run used, and nothing is hidden in argparse.
DEFAULTS: dict = {
    "lang": None,
    "init": None,
    "data_dir": None,
    "out_dir": None,
    "log_dir": None,
    "epochs": 3,
    "batch_size": 32,
    "max_steps": None,          # derived from epochs when not given
    "lr": 1e-4,
    "min_lr": 1e-5,
    "warmup_steps": 100,
    "weight_decay": 0.1,
    "beta1": 0.9,
    "beta2": 0.95,
    "grad_clip": 1.0,
    "max_len": 128,
    "eval_every": 100,
    "log_every": 20,
    "save_every": 250,
    "eval_batch_size": 64,
    "seed": 1337,
    "device": "auto",
    "resume": None,
    "overfit": 0,
}


def file_sha256(path: str | Path, chunk: int = 1 << 20) -> str:
    """SHA-256 of a file, read in chunks.

    Recorded in ``train_args`` so a finetuned checkpoint names not just the path
    of the model it started from but the exact bytes. Paths get reused; hashes
    do not.
    """
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


class FinetuneTrainer:
    """Finetunes one pretrained model on one language's reasoning corpus.

    Args:
        config: Architecture, taken from the pretrained checkpoint -- never
            re-specified by the caller.
        args: Resolved hyperparameters (see :data:`DEFAULTS`).
        train_data: Training examples.
        val_data: Validation examples, from the same language.
        device: Device to train on.
    """

    def __init__(self, config: GPTConfig, args, train_data: ReasoningDataset,
                 val_data: ReasoningDataset, device: torch.device | None = None):
        self.cfg = config
        self.args = args
        self.device = device or pick_device(getattr(args, "device", "auto"))
        self.out_dir = Path(args.out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir = Path(getattr(args, "log_dir", None) or args.out_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        torch.manual_seed(args.seed)
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(args.seed)

        # Same guard as Phase 2: validation runs inside the logging branch, so
        # an eval interval that is not a multiple of the log interval would
        # never fire and the run would finish with an empty val column.
        if args.eval_every % args.log_every != 0:
            raise ValueError(
                f"--eval-every ({args.eval_every}) must be a multiple of "
                f"--log-every ({args.log_every}), or validation never runs"
            )

        self.model = GPT(config).to(self.device)
        self.n_params = count_params(self.model)

        self.optimizer = torch.optim.AdamW(
            self.model.optimizer_param_groups(args.weight_decay),
            lr=args.lr, betas=(args.beta1, args.beta2), eps=1e-8,
        )
        self.scheduler = CosineWarmupLR(
            self.optimizer, peak_lr=args.lr, min_lr=args.min_lr,
            warmup_steps=args.warmup_steps, max_steps=args.max_steps,
        )

        self.amp_dtype, needs_scaler = pick_amp_dtype(self.device)
        self.scaler = torch.amp.GradScaler(self.device.type, enabled=needs_scaler)

        self.train_data = train_data
        self.val_data = val_data
        self.val_batches = val_data.fixed_batches(args.eval_batch_size)

        self.step = 0
        self.tokens_seen = 0
        self.supervised_seen = 0
        self.best_val = float("inf")
        self.elapsed = 0.0
        self._stream = None
        self.log_path = self.log_dir / "finetune_log.csv"
        self.jsonl_path = self.log_dir / "finetune_log.jsonl"

    # ------------------------------------------------------------------ #
    # initialisation from the pretrained checkpoint
    # ------------------------------------------------------------------ #

    def init_from(self, path: str | Path) -> None:
        """Load **only** the weights from the pretrained checkpoint.

        Optimizer state, scheduler state and step counter are deliberately left
        fresh. Finetuning is a new objective on a new distribution at a sixth of
        the learning rate; inheriting AdamW's second-moment estimates from a
        6e-4 cosine run would size the first updates for a schedule that is no
        longer running.

        Args:
            path: ``ckpt_final.pt`` or the weights-only ``{lang}_model_final.pt``.

        Raises:
            ValueError: If the checkpoint's architecture differs from ``cfg``.
        """
        ckpt = load_checkpoint(path, map_location=self.device)
        if ckpt["config"] != self.cfg.to_dict():
            raise ValueError(
                f"{Path(path).name} has a different architecture than the config "
                f"this trainer was built with -- refusing to load mismatched weights"
            )
        self.model.load_state_dict(ckpt["model"])

    # ------------------------------------------------------------------ #
    # data
    # ------------------------------------------------------------------ #

    def _ensure_stream(self) -> None:
        """Open the batch stream at the current step, if it is not already open."""
        if self._stream is None:
            self._stream = self.train_data.stream(
                self.args.batch_size, start_step=self.step
            )

    def _next_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Next training batch, moved to the device."""
        self._ensure_stream()
        x, y = next(self._stream)
        return x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)

    def _autocast(self):
        """Mixed-precision context, or a no-op on CPU."""
        if self.amp_dtype is None:
            return contextlib.nullcontext()
        return torch.autocast(device_type=self.device.type, dtype=self.amp_dtype)

    @property
    def batches_per_epoch(self) -> int:
        """Optimizer steps in one pass over the training split."""
        return self.train_data.batches_per_epoch(self.args.batch_size)

    @property
    def epoch(self) -> int:
        """Epochs completed so far."""
        return self.step // max(1, self.batches_per_epoch)

    # ------------------------------------------------------------------ #
    # training
    # ------------------------------------------------------------------ #

    def train_step(self, fixed_batch=None) -> tuple[float, float, float]:
        """Run one optimizer step.

        No gradient accumulation: a batch of 32 reasoning examples is under a
        thousand tokens, so the effective batch is reached in one forward pass.

        Args:
            fixed_batch: If given, reuse this batch instead of drawing one.
                Used by :meth:`overfit_batch`.

        Returns:
            ``(loss, grad_norm, lr)``.
        """
        self.model.train()
        lr = self.scheduler.step_to(self.step)
        self.optimizer.zero_grad(set_to_none=True)

        x, y = fixed_batch if fixed_batch is not None else self._next_batch()
        with self._autocast():
            _, loss, _ = self.model(x, targets=y)
        self.scaler.scale(loss).backward()

        # Unscale before clipping, or the threshold is meaningless: the
        # gradients still carry the scaler's multiplier.
        self.scaler.unscale_(self.optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.args.grad_clip
        )
        # A no-op if the scaler saw inf/nan this iteration.
        self.scaler.step(self.optimizer)
        self.scaler.update()

        self.step += 1
        # Padding is excluded from both counts: pad id 0 is reserved and never
        # produced by the tokenizer, so this is exact rather than approximate.
        self.tokens_seen += int((x != self.train_data.pad_id).sum())
        self.supervised_seen += int((y != IGNORE_INDEX).sum())
        return float(loss.item()), float(grad_norm), lr

    @torch.no_grad()
    def evaluate(self) -> tuple[float, float, float]:
        """Loss, perplexity and teacher-forced token accuracy on the val split.

        Losses are summed and divided by the total number of supervised
        positions rather than averaged per batch. The batches are ragged -- the
        last one is partial and examples differ in answer length -- so a mean of
        means would weight a short batch the same as a full one.

        ``answer_token_acc`` is teacher-forced accuracy over supervised
        positions: a cheap per-step signal, **not** the reported metric. Task 4
        scores whole answers by greedy decoding and by candidate likelihood.

        Returns:
            ``(mean_loss, perplexity, answer_token_accuracy)``.
        """
        self.model.eval()
        total_nats, total_tokens, correct = 0.0, 0, 0

        for x, y in self.val_batches:
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)
            with self._autocast():
                logits, _, _ = self.model(x)
            total_nats += float(F.cross_entropy(
                logits.reshape(-1, logits.size(-1)).float(), y.reshape(-1),
                ignore_index=IGNORE_INDEX, reduction="sum",
            ).item())
            mask = y != IGNORE_INDEX
            total_tokens += int(mask.sum())
            correct += int((logits.argmax(dim=-1)[mask] == y[mask]).sum())

        self.model.train()
        mean = total_nats / max(1, total_tokens)
        return mean, math.exp(min(mean, 20.0)), correct / max(1, total_tokens)

    def overfit_batch(self, steps: int = 200, disable_dropout: bool = True) -> float:
        """Drive one batch to near-zero loss -- the masking sanity gate.

        If the prompt mask is misaligned the model is being asked to predict
        tokens from positions that cannot see them, and a single batch will not
        reach near-zero however long it trains. Cheap, and it fails before any
        GPU time is committed.

        Dropout is switched off for the duration. The gate measures whether the
        supervised positions are learnable at all; leaving dropout on would put
        a noise floor under the loss and turn a clean pass/fail into a
        judgement call about where that floor sits. It is restored afterwards.

        Args:
            steps: Optimizer steps to take on the one batch.
            disable_dropout: Set False to measure the loss with dropout active.

        Returns:
            The final loss.
        """
        saved: list[tuple[torch.nn.Dropout, float]] = []
        if disable_dropout:
            for module in self.model.modules():
                if isinstance(module, torch.nn.Dropout):
                    saved.append((module, module.p))
                    module.p = 0.0
        try:
            batch = self._next_batch()
            loss = float("nan")
            for _ in range(steps):
                loss, _, _ = self.train_step(fixed_batch=batch)
            return loss
        finally:
            for module, p in saved:
                module.p = p

    # ------------------------------------------------------------------ #
    # checkpointing
    # ------------------------------------------------------------------ #

    def data_state(self) -> dict:
        """Loader position, so a resumed run replays the identical stream."""
        state = self.train_data.state()
        state.update({"batch_size": self.args.batch_size, "step": self.step})
        return state

    def save(self, name: str, metrics: dict | None = None) -> Path:
        """Write a checkpoint in Phase 2's format, unchanged.

        All five contents the brief names are present -- model weights,
        optimizer state, scheduler state, training step, configuration -- plus
        the RNG and data-stream state that make a resumed run identical rather
        than merely similar.

        ``supervised_seen`` and ``elapsed_s`` are written on *every* save, not
        only when a caller happens to pass a metrics row. They are resumable
        counters: ``tokens_seen`` is a first-class checkpoint field and
        restores on its own, but these two ride in ``metrics``, so a save that
        omitted them would make the resumed run's log restart them from zero
        mid-run. A caller's explicit values win.
        """
        payload = {
            "supervised_seen": self.supervised_seen,
            "elapsed_s": self.elapsed,
        }
        if metrics:
            payload.update(metrics)
        return save_checkpoint(
            self.out_dir / name,
            model=self.model, optimizer=self.optimizer,
            scheduler=self.scheduler, scaler=self.scaler,
            config=self.cfg, step=self.step, tokens_seen=self.tokens_seen,
            best_val=self.best_val, train_args=vars(self.args),
            data_state=self.data_state(), metrics=payload,
        )

    def resume_from(self, path: str | Path) -> None:
        """Restore a finetuning run so it continues exactly where it stopped.

        Order matters: the model has already been constructed, which consumed
        RNG, so the RNG state is restored *last* and overwrites what
        construction left behind. Without that, dropout masks diverge after the
        resume and the run is only approximately the same.
        """
        ckpt = load_checkpoint(path, map_location=self.device)
        if ckpt["config"] != self.cfg.to_dict():
            raise ValueError(
                f"{Path(path).name} was trained with a different architecture; "
                "refusing to load mismatched weights"
            )
        self.model.load_state_dict(ckpt["model"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.scheduler.load_state_dict(ckpt["scheduler"])
        if ckpt.get("scaler") is not None:
            self.scaler.load_state_dict(ckpt["scaler"])
        self.step = ckpt["step"]
        self.tokens_seen = ckpt["tokens_seen"]
        self.best_val = ckpt["best_val"]
        metrics = ckpt.get("metrics", {})
        self.supervised_seen = metrics.get("supervised_seen", 0)
        self.elapsed = metrics.get("elapsed_s", 0.0)
        self._stream = None  # reopened at the restored step
        restore_rng(ckpt)

    # ------------------------------------------------------------------ #
    # logging
    # ------------------------------------------------------------------ #

    def _prepare_log(self, resumed: bool) -> None:
        """Create the log, or trim the rows a resumed run is about to rewrite."""
        if not resumed or not self.log_path.exists():
            with open(self.log_path, "w", newline="", encoding="utf-8") as fh:
                csv.writer(fh).writerow(LOG_COLUMNS)
            return
        with open(self.log_path, newline="", encoding="utf-8") as fh:
            rows = list(csv.reader(fh))
        header, body = rows[0], rows[1:]
        kept = [r for r in body if r and int(r[0]) < self.step]
        with open(self.log_path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(header)
            writer.writerows(kept)

    def _log(self, row: dict) -> None:
        """Append one row to both the CSV and the JSONL log."""
        with open(self.log_path, "a", newline="", encoding="utf-8") as fh:
            csv.writer(fh).writerow([row.get(c, "") for c in LOG_COLUMNS])
        with open(self.jsonl_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(row) + "\n")

    def _peak_mem_gb(self) -> float:
        """Peak GPU memory in GB, or 0 on CPU."""
        if self.device.type != "cuda":
            return 0.0
        return torch.cuda.max_memory_allocated(self.device) / 1e9

    def describe(self) -> str:
        """Multi-line run summary, printed once at startup."""
        return "\n".join([
            f"  model        : {self.cfg.name} ({self.cfg.lang}), "
            f"{self.n_params:,} parameters",
            f"  device       : {self.device} "
            f"(amp: {self.amp_dtype or 'off'}, scaler: {self.scaler.is_enabled()})",
            f"  train data   : {self.train_data.describe()}",
            f"  val data     : {self.val_data.describe()}",
            f"  schedule     : {self.args.epochs} epochs x {self.batches_per_epoch} "
            f"batches = {self.args.max_steps} steps, "
            f"lr {self.args.lr:g} -> {self.args.min_lr:g} "
            f"after {self.args.warmup_steps} warmup",
            f"  batch        : {self.args.batch_size} examples, no accumulation",
        ])

    # ------------------------------------------------------------------ #
    # the loop
    # ------------------------------------------------------------------ #

    def fit(self, resumed: bool = False) -> None:
        """Train to ``max_steps``, evaluating, logging and checkpointing.

        Args:
            resumed: True when continuing an interrupted run, so the log is
                trimmed rather than truncated.
        """
        self._prepare_log(resumed)
        start = time.time() - self.elapsed
        target = self.args.max_steps

        while self.step < target:
            loss, grad_norm, lr = self.train_step()

            if self.step % self.args.log_every == 0 or self.step == target:
                row = {
                    "step": self.step, "epoch": self.epoch, "lr": round(lr, 8),
                    "train_loss": round(loss, 5), "grad_norm": round(grad_norm, 4),
                    "tokens_seen": self.tokens_seen,
                    "supervised_seen": self.supervised_seen,
                    "peak_mem_gb": round(self._peak_mem_gb(), 3),
                    "elapsed_s": round(time.time() - start, 1),
                }
                if self.step % self.args.eval_every == 0 or self.step == target:
                    val_loss, val_ppl, acc = self.evaluate()
                    row.update({
                        "val_loss": round(val_loss, 5),
                        "val_ppl": round(val_ppl, 3),
                        "answer_token_acc": round(acc, 5),
                    })
                    if val_loss < self.best_val:
                        self.best_val = val_loss
                        self.save("ckpt_best.pt", metrics=row)
                    print(f"  step {self.step:>5}/{target}  loss {loss:.4f}  "
                          f"val {val_loss:.4f}  acc {acc:.3f}  lr {lr:.2e}")
                else:
                    print(f"  step {self.step:>5}/{target}  loss {loss:.4f}  "
                          f"lr {lr:.2e}")
                self.elapsed = row["elapsed_s"]
                self._log(row)

            if self.step % self.args.save_every == 0:
                self.save("ckpt_last.pt")  # save() records the resumable counters

        val_loss, val_ppl, acc = self.evaluate()
        final_metrics = {
            "elapsed_s": round(time.time() - start, 1),
            "supervised_seen": self.supervised_seen,
            "val_loss": val_loss, "val_ppl": val_ppl, "answer_token_acc": acc,
        }
        self.save("ckpt_last.pt", metrics=final_metrics)
        self.save("ckpt_ft_final.pt", metrics=final_metrics)
        print(f"\nfinished {self.step} steps in {final_metrics['elapsed_s'] / 60:.1f} min")
        print(f"  val loss {val_loss:.4f}  ppl {val_ppl:.3f}  "
              f"answer-token accuracy {acc:.3f}")
        print(f"  best val loss {self.best_val:.4f}  ->  {self.out_dir}")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    """Command-line interface.

    Every hyperparameter defaults to ``None`` so :func:`resolve_args` can tell
    "not given" from "given the same value the config file has", and the
    precedence command line > YAML > :data:`DEFAULTS` is unambiguous.
    """
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", default=None,
                        help="finetuning hyperparameters YAML, e.g. hindi/configs/hi_ft.yaml")

    group = parser.add_argument_group("what to finetune")
    group.add_argument("--lang", choices=["hi", "ne"], default=None)
    group.add_argument("--init", default=None, help="pretrained checkpoint to start from")
    group.add_argument("--data-dir", default=None, help="directory with the reasoning JSONL")
    group.add_argument("--out-dir", default=None, help="where checkpoints go")
    group.add_argument("--log-dir", default=None, help="where logs go (default: --out-dir)")

    group = parser.add_argument_group("schedule")
    group.add_argument("--epochs", type=int, default=None)
    group.add_argument("--batch-size", type=int, default=None)
    group.add_argument("--max-steps", type=int, default=None,
                       help="overrides epochs x batches-per-epoch")
    group.add_argument("--max-len", type=int, default=None)

    group = parser.add_argument_group("optimizer")
    group.add_argument("--lr", type=float, default=None)
    group.add_argument("--min-lr", type=float, default=None)
    group.add_argument("--warmup-steps", type=int, default=None)
    group.add_argument("--weight-decay", type=float, default=None)
    group.add_argument("--beta1", type=float, default=None)
    group.add_argument("--beta2", type=float, default=None)
    group.add_argument("--grad-clip", type=float, default=None)

    group = parser.add_argument_group("evaluation and saving")
    group.add_argument("--eval-every", type=int, default=None)
    group.add_argument("--log-every", type=int, default=None)
    group.add_argument("--save-every", type=int, default=None)
    group.add_argument("--eval-batch-size", type=int, default=None)

    group = parser.add_argument_group("run control")
    group.add_argument("--seed", type=int, default=None)
    group.add_argument("--device", default=None, choices=["auto", "cuda", "cpu"])
    group.add_argument("--resume", default=None,
                       help="'auto' to continue from ckpt_last.pt, or a path")
    group.add_argument("--overfit", type=int, default=None, metavar="STEPS",
                       help="masking sanity gate: drive one batch to near-zero loss")
    return parser


def resolve_args(cli: argparse.Namespace) -> argparse.Namespace:
    """Merge command line over YAML over :data:`DEFAULTS`.

    Args:
        cli: Parsed arguments, with ``None`` wherever a flag was not given.

    Returns:
        A namespace with every key of :data:`DEFAULTS` filled in.

    Raises:
        SystemExit: If a required setting is missing after merging.
    """
    from_yaml: dict = {}
    if cli.config:
        from_yaml = yaml.safe_load(Path(cli.config).read_text(encoding="utf-8")) or {}

    resolved = {}
    for key, fallback in DEFAULTS.items():
        given = getattr(cli, key, None)
        if given is not None:
            resolved[key] = given
        elif key in from_yaml:
            resolved[key] = from_yaml[key]
        else:
            resolved[key] = fallback

    resolved["config"] = cli.config
    # Carried through only so it lands in train_args.json; the config file is
    # the record of the run and should say what it was called.
    resolved["name"] = from_yaml.get("name")

    missing = [k for k in ("lang", "init", "data_dir", "out_dir") if not resolved[k]]
    if missing:
        raise SystemExit(
            f"missing required setting(s): {', '.join(missing)} -- "
            f"pass --config or the matching flags"
        )
    return argparse.Namespace(**resolved)


def main(argv: list[str] | None = None) -> None:
    """Finetune one model end to end."""
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):  # pragma: no cover - non-standard stream
        pass

    # Run from the repository root: the tokenizer and data paths in the config
    # files are repo-relative, matching every other entry point in this project.
    args = resolve_args(build_parser().parse_args(argv))
    device = pick_device(args.device)

    # The architecture travels inside the checkpoint. Reading it from there
    # rather than from a config file means a finetuning run cannot be started
    # against weights it does not match.
    init_path = Path(args.init)
    ckpt = load_checkpoint(init_path, map_location="cpu")
    cfg = GPTConfig.from_dict(ckpt["config"])
    if cfg.lang != args.lang:
        raise SystemExit(
            f"{init_path.name} was pretrained as lang={cfg.lang!r} but --lang "
            f"{args.lang!r} was requested -- refusing to finetune a model on "
            f"another language's data"
        )
    del ckpt  # the weights are loaded again, onto the device, by init_from

    # Phase 1's tokenizer for this language, unchanged: no token added, no
    # embedding resized. That is the brief's "keep the tokenizer and vocabulary
    # fixed", enforced rather than asserted.
    sp = load_tokenizer(args.lang, cfg.vocab_size)

    data_dir = Path(args.data_dir)
    train_data = ReasoningDataset(
        build_examples(data_dir / "train.jsonl", sp, args.max_len),
        seed=args.seed, source=data_dir / "train.jsonl",
    )
    val_data = ReasoningDataset(
        build_examples(data_dir / "val.jsonl", sp, args.max_len),
        seed=args.seed, source=data_dir / "val.jsonl",
    )

    if args.max_steps is None:
        args.max_steps = args.epochs * train_data.batches_per_epoch(args.batch_size)
    args.init_sha256 = file_sha256(init_path)

    trainer = FinetuneTrainer(cfg, args, train_data, val_data, device=device)

    resume_path = resolve_resume(args.out_dir, args.resume)
    if resume_path is not None:
        trainer.resume_from(resume_path)
        print(f"resumed from {resume_path} at step {trainer.step}")
    else:
        trainer.init_from(init_path)
        print(f"initialised from {init_path}")
        print(f"  sha256 {args.init_sha256[:16]}...")

    print(trainer.describe())

    # Hyperparameters and the architecture snapshot land next to the
    # checkpoints (PDF implementation guideline 6).
    write_run_config(trainer.out_dir, cfg, vars(args))

    if args.overfit:
        print(f"\noverfit gate: {args.overfit} steps on one batch ...")
        final = trainer.overfit_batch(args.overfit)
        verdict = "PASS" if final < 0.05 else "FAIL"
        print(f"  final loss {final:.5f}  -> {verdict} (want < 0.05)")
        if verdict == "FAIL":
            print("  a single batch that will not overfit means the prompt mask "
                  "is wrong -- check finetune/data.py before spending GPU time")
        return

    print()
    trainer.fit(resumed=resume_path is not None)


if __name__ == "__main__":
    main()
