"""The pretraining loop, as a reusable class.

``Trainer`` owns the model, optimizer, scheduler, data streams and logging.
``train/train.py`` is a thin CLI around it, ``train/benchmark.py`` borrows its
``train_step`` to time the real work rather than an approximation of it, and
Phase 3's finetuning subclasses it rather than copying it.

Recipe, all of it standard for a ~25M decoder:
  AdamW (0.9, 0.95), weight decay on matrices only, linear warmup into cosine
  decay, gradient clipping at 1.0, mixed precision, gradient accumulation to
  reach the target batch size on a small GPU.
"""

from __future__ import annotations

import csv
import json
import math
import time
from pathlib import Path

import torch

from model import GPT, GPTConfig, count_params
from train.checkpoint import (
    load_checkpoint,
    restore_rng,
    save_checkpoint,
    write_run_config,
)
from train.data import WindowedBinDataset
from train.scheduler import CosineWarmupLR

LOG_COLUMNS = [
    "step", "tokens_seen", "lr", "train_loss", "val_loss", "val_ppl",
    "grad_norm", "tok_per_s", "peak_mem_gb", "elapsed_s",
]


def pick_device(requested: str = "auto") -> torch.device:
    """Choose the compute device.

    Args:
        requested: ``auto``, ``cuda``, or ``cpu``.

    Returns:
        The selected device.
    """
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def pick_amp_dtype(device: torch.device) -> tuple[torch.dtype | None, bool]:
    """Choose the mixed-precision dtype for this GPU.

    bfloat16 is preferred where supported (Ampere and newer) because it has the
    same exponent range as float32 and so cannot overflow. Kaggle's T4 is
    Turing and has no bfloat16, so it falls back to float16, which *can*
    overflow and therefore needs a gradient scaler.

    Args:
        device: The device training will run on.

    Returns:
        ``(dtype, needs_scaler)``. ``dtype`` is None when running without AMP.
    """
    if device.type != "cuda":
        return None, False  # CPU runs in plain float32; this path is for tests
    # Compute capability, not torch.cuda.is_bf16_supported(): that returns True
    # on Turing (T4, sm_75) and even on Pascal, where bfloat16 is emulated in
    # software rather than run on tensor cores. Taking it at its word makes a T4
    # run several times slower than it should. Native bf16 starts at Ampere.
    major, _ = torch.cuda.get_device_capability(device)
    if major >= 8:
        return torch.bfloat16, False
    return torch.float16, True


class Trainer:
    """Pretrains one model on one language's corpus.

    Args:
        config: Architecture config, loaded from the language's YAML file.
        args: Parsed CLI arguments (see ``train/train.py``).
        device: Device to train on.
    """

    def __init__(self, config: GPTConfig, args, device: torch.device | None = None):
        self.cfg = config
        self.args = args
        self.device = device or pick_device(getattr(args, "device", "auto"))
        self.out_dir = Path(args.out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

        torch.manual_seed(args.seed)
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(args.seed)

        if config.max_seq_len < args.block_size:
            raise ValueError(
                f"block_size {args.block_size} exceeds the model's max_seq_len "
                f"{config.max_seq_len}"
            )
        # Validation runs inside the logging branch, so an eval interval that is
        # not a multiple of the logging interval would silently never fire and
        # the run would finish with an empty val column.
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

        # Data. Train and val come from the same language, never mixed.
        data_dir = Path(args.data_dir)
        self.train_data = WindowedBinDataset(
            data_dir / args.train_bin, args.block_size, seed=args.seed
        )
        self.val_data = WindowedBinDataset(
            data_dir / args.val_bin, args.block_size, seed=args.seed
        )
        self.val_batches = self.val_data.fixed_batches(
            args.eval_batches, args.eval_batch_size, seed=args.seed
        )

        self.step = 0
        self.tokens_seen = 0
        self.best_val = float("inf")
        self.elapsed = 0.0
        self._stream = None
        self.log_path = self.out_dir / "train_log.csv"
        self.jsonl_path = self.out_dir / "train_log.jsonl"

    # ------------------------------------------------------------------ #
    # properties
    # ------------------------------------------------------------------ #

    @property
    def tokens_per_step(self) -> int:
        """Training tokens consumed by one optimizer step."""
        return self.args.micro_batch * self.args.grad_accum * self.args.block_size

    # ------------------------------------------------------------------ #
    # data
    # ------------------------------------------------------------------ #

    def _ensure_stream(self) -> None:
        """Open the training stream at the position implied by ``self.step``."""
        if self._stream is None:
            self._stream = self.train_data.stream(
                self.args.micro_batch,
                start_micro_step=self.step * self.args.grad_accum,
            )

    def _next_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Fetch one micro-batch and move it to the device."""
        self._ensure_stream()
        x, y = next(self._stream)
        return (x.to(self.device, non_blocking=True),
                y.to(self.device, non_blocking=True))

    def _autocast(self):
        """Mixed-precision context, or a no-op on CPU."""
        return torch.autocast(
            device_type=self.device.type,
            dtype=self.amp_dtype,
            enabled=self.amp_dtype is not None,
        )

    # ------------------------------------------------------------------ #
    # training
    # ------------------------------------------------------------------ #

    def train_step(self, fixed_batch=None) -> tuple[float, float, float]:
        """Run one full optimizer step, including gradient accumulation.

        Args:
            fixed_batch: If given, reuse this one batch for every micro-step.
                Used by the overfit sanity check and the benchmark.

        Returns:
            ``(loss, grad_norm, lr)`` where loss is averaged over micro-steps.
        """
        self.model.train()
        lr = self.scheduler.step_to(self.step)
        self.optimizer.zero_grad(set_to_none=True)

        total_loss = 0.0
        for _ in range(self.args.grad_accum):
            x, y = fixed_batch if fixed_batch is not None else self._next_batch()
            with self._autocast():
                _, loss, _ = self.model(x, targets=y)
                # Scale so the accumulated gradient equals the gradient of the
                # mean loss over the full effective batch.
                loss = loss / self.args.grad_accum
            self.scaler.scale(loss).backward()
            total_loss += loss.item()

        # Unscale before clipping, or the threshold is meaningless: the
        # gradients still carry the scaler's multiplier.
        self.scaler.unscale_(self.optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.args.grad_clip
        )
        # step() becomes a no-op if the scaler found inf/nan this iteration.
        self.scaler.step(self.optimizer)
        self.scaler.update()

        self.step += 1
        self.tokens_seen += self.tokens_per_step
        return total_loss, float(grad_norm), lr

    @torch.no_grad()
    def evaluate(self) -> tuple[float, float]:
        """Loss and perplexity on the fixed validation batches.

        The same batches every time, so the curve tracks learning rather than
        which tokens happened to be sampled.

        Returns:
            ``(mean_loss, perplexity)``.
        """
        self.model.eval()
        total = 0.0
        for x, y in self.val_batches:
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)
            with self._autocast():
                _, loss, _ = self.model(x, targets=y)
            total += loss.item()
        self.model.train()
        mean = total / len(self.val_batches)
        return mean, math.exp(min(mean, 20.0))  # cap guards against inf early on

    # ------------------------------------------------------------------ #
    # checkpointing
    # ------------------------------------------------------------------ #

    def data_state(self) -> dict:
        """Where the loader is, so a resumed run replays the same stream."""
        return {
            "seed": self.args.seed,
            "micro_step": self.step * self.args.grad_accum,
            "train_bin": str(self.train_data.path.name),
            "train_tokens": self.train_data.n_tokens,
            "n_windows": self.train_data.n_windows,
        }

    def save(self, name: str, metrics: dict | None = None) -> Path:
        """Write a checkpoint into the run directory."""
        return save_checkpoint(
            self.out_dir / name,
            model=self.model, optimizer=self.optimizer,
            scheduler=self.scheduler, scaler=self.scaler,
            config=self.cfg, step=self.step, tokens_seen=self.tokens_seen,
            best_val=self.best_val, train_args=vars(self.args),
            data_state=self.data_state(), metrics=metrics,
        )

    def close(self) -> None:
        """Release the corpus memory maps.

        Windows will not delete a file that is still mapped, so anything that
        creates trainers in a loop -- or in a temporary directory -- must close
        them. Idempotent, and safe to call on a trainer that never ran.
        """
        self._stream = None  # the generator holds a reference to the dataset
        self.train_data.close()
        self.val_data.close()

    def __enter__(self) -> "Trainer":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def resume_from(self, path: Path) -> None:
        """Restore a run from a checkpoint so it continues exactly.

        Order matters: the model is already built (which consumed RNG during
        initialisation), so RNG state is restored *last*, overwriting whatever
        construction left behind.
        """
        ckpt = load_checkpoint(path, map_location=self.device)
        saved_cfg = ckpt["config"]
        if saved_cfg != self.cfg.to_dict():
            raise ValueError(
                f"{path.name} was trained with a different architecture; "
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
        self.elapsed = ckpt.get("metrics", {}).get("elapsed_s", 0.0)
        self._stream = None  # reopened at the restored step
        restore_rng(ckpt)

    # ------------------------------------------------------------------ #
    # logging
    # ------------------------------------------------------------------ #

    def _prepare_log(self, resumed: bool) -> None:
        """Create the log, or trim rows a resumed run is about to rewrite."""
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
        return torch.cuda.max_memory_allocated() / 1e9

    def describe(self) -> str:
        """Run header printed at startup and worth pasting into the report."""
        gpu = torch.cuda.get_device_name(0) if self.device.type == "cuda" else "cpu"
        precision = str(self.amp_dtype).replace("torch.", "") if self.amp_dtype else "float32"
        return "\n".join([
            f"model        {self.cfg.name}  ({self.cfg.lang})",
            f"parameters   {self.n_params:,}",
            f"device       {gpu}  [{precision}]",
            f"train data   {self.train_data.describe()}",
            f"val data     {self.val_data.describe()}",
            f"batch        {self.args.micro_batch} x {self.args.grad_accum} accum "
            f"x {self.args.block_size} tok = {self.tokens_per_step:,} tokens/step",
            f"schedule     {self.args.max_steps:,} steps -> "
            f"{self.args.max_steps * self.tokens_per_step:,} tokens "
            f"({self.args.max_steps * self.tokens_per_step / self.train_data.n_tokens:.2f} epochs)",
            f"lr           {self.args.lr:g} peak, {self.args.min_lr:g} floor, "
            f"{self.args.warmup_steps} warmup",
        ])

    # ------------------------------------------------------------------ #
    # main loop
    # ------------------------------------------------------------------ #

    def fit(self, resumed: bool = False) -> None:
        """Train to ``max_steps``, evaluating and checkpointing as it goes."""
        write_run_config(self.out_dir, self.cfg, vars(self.args))
        self._prepare_log(resumed)
        print(self.describe(), flush=True)
        if resumed:
            print(f"resumed at step {self.step:,} ({self.tokens_seen:,} tokens seen)", flush=True)

        start_wall = time.time()
        start_step = self.step
        window_start = time.time()
        window_tokens = 0

        try:
            while self.step < self.args.max_steps:
                loss, grad_norm, lr = self.train_step()
                window_tokens += self.tokens_per_step

                if self.step % self.args.log_every == 0 or self.step == self.args.max_steps:
                    now = time.time()
                    tok_per_s = window_tokens / max(now - window_start, 1e-9)
                    window_start, window_tokens = now, 0
                    self.elapsed = now - start_wall

                    row = {
                        "step": self.step,
                        "tokens_seen": self.tokens_seen,
                        "lr": round(lr, 8),
                        "train_loss": round(loss, 6),
                        "grad_norm": round(grad_norm, 4),
                        "tok_per_s": round(tok_per_s, 1),
                        "peak_mem_gb": round(self._peak_mem_gb(), 3),
                        "elapsed_s": round(self.elapsed, 1),
                    }

                    due = (self.step % self.args.eval_every == 0
                           or self.step == self.args.max_steps)
                    if due:
                        val_loss, val_ppl = self.evaluate()
                        row["val_loss"] = round(val_loss, 6)
                        row["val_ppl"] = round(val_ppl, 3)
                        if val_loss < self.best_val:
                            self.best_val = val_loss
                            self.save("ckpt_best.pt", metrics=row)
                        # Evaluation costs wall time but no tokens; restart the
                        # throughput window so it does not skew tok/s.
                        window_start = time.time()

                    self._log(row)
                    msg = (f"step {self.step:>6,}/{self.args.max_steps:,} | "
                           f"loss {loss:.4f} | lr {lr:.2e} | "
                           f"{tok_per_s / 1000:.1f}k tok/s")
                    if "val_loss" in row:
                        msg += f" | val {row['val_loss']:.4f} (ppl {row['val_ppl']:.1f})"
                    print(msg, flush=True)

                if self.step % self.args.save_every == 0:
                    self.save("ckpt_last.pt")

        except KeyboardInterrupt:
            print("\ninterrupted -- saving ckpt_last.pt so --resume auto can continue", flush=True)
            self.save("ckpt_last.pt")
            raise

        self.save("ckpt_last.pt")
        self.save("ckpt_final.pt")
        done = self.step - start_step
        print(
            f"\nfinished {done:,} steps in {self.elapsed / 60:.1f} min | "
            f"{self.tokens_seen:,} tokens seen | best val {self.best_val:.4f}",
            flush=True,
        )

    def overfit_batch(self, steps: int = 200) -> float:
        """Train repeatedly on a single batch -- the fastest test that it learns.

        Every structural test passes on a model whose gradients never reach the
        early layers. This one does not: memorising one batch should drive the
        loss towards zero within a couple hundred steps. If it plateaus near
        ``ln(vocab_size)``, something is broken, and finding that out here costs
        minutes instead of GPU-hours.

        Args:
            steps: How many times to train on the same batch.

        Returns:
            The final loss.
        """
        batch = self._next_batch()
        first = last = None
        for i in range(steps):
            loss, _, _ = self.train_step(fixed_batch=batch)
            if i == 0:
                first = loss
            last = loss
            if i % max(1, steps // 10) == 0 or i == steps - 1:
                print(f"  overfit step {i:>4} | loss {loss:.4f}", flush=True)
        print(f"\nstart {first:.4f} -> end {last:.4f} "
              f"(random guessing would be {math.log(self.cfg.vocab_size):.4f})", flush=True)
        return last
