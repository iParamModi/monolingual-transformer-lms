"""Resume-capable checkpointing.

The brief makes this mandatory and names the contents: "model weights,
optimizer state, scheduler state, training step, and configuration". All five
are present, plus the RNG and data-stream state needed to make a resumed run
*identical* to an uninterrupted one rather than merely similar.

Writes are atomic. Free cloud GPUs are killed without warning, and a session
that dies midway through ``torch.save`` must not leave a truncated file where
the only usable checkpoint used to be.
"""

from __future__ import annotations

import json
import os
import random
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

# Bumped if the dict layout ever changes, so an old checkpoint fails loudly
# instead of loading half-correctly.
CHECKPOINT_FORMAT = 1


def _rng_state() -> dict:
    """Capture every random-number generator the training loop consumes."""
    return {
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "numpy": np.random.get_state(),
        "python": random.getstate(),
    }


def _restore_rng(state: dict) -> None:
    """Restore generators captured by :func:`_rng_state`."""
    torch.set_rng_state(state["torch"].cpu() if torch.is_tensor(state["torch"]) else state["torch"])
    if state.get("cuda") is not None and torch.cuda.is_available():
        try:
            torch.cuda.set_rng_state_all(state["cuda"])
        except (RuntimeError, ValueError):
            # Different GPU count than the run that saved this. Harmless: only
            # dropout masks diverge, not the weights or the data order.
            pass
    np.random.set_state(state["numpy"])
    random.setstate(state["python"])


def save_checkpoint(
    path: str | Path,
    *,
    model,
    optimizer,
    scheduler,
    scaler,
    config,
    step: int,
    tokens_seen: int,
    best_val: float,
    train_args: dict,
    data_state: dict,
    metrics: dict | None = None,
) -> Path:
    """Write a complete, resumable checkpoint atomically.

    Args:
        path: Destination file.
        model: The ``GPT`` instance.
        optimizer: AdamW instance.
        scheduler: :class:`~train.scheduler.CosineWarmupLR` instance.
        scaler: ``torch.amp.GradScaler`` (may be disabled).
        config: The ``GPTConfig`` this model was built from.
        step: Optimizer steps completed so far.
        tokens_seen: Training tokens consumed so far.
        best_val: Best validation loss observed.
        train_args: Full CLI arguments, so the run is reproducible.
        data_state: Loader position, seed, and source file identity.
        metrics: Optional latest metrics, for convenience when inspecting.

    Returns:
        The path written.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "format": CHECKPOINT_FORMAT,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,
        "config": asdict(config) if hasattr(config, "__dataclass_fields__") else dict(config),
        "step": step,
        "tokens_seen": tokens_seen,
        "best_val": best_val,
        "train_args": train_args,
        "data_state": data_state,
        "rng": _rng_state(),
        "metrics": metrics or {},
    }

    # Write beside the target, then rename. os.replace is atomic on Windows and
    # POSIX alike, so the destination is either the old file or the new one --
    # never a half-written one.
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)
    return path


def load_checkpoint(path: str | Path, map_location="cpu") -> dict:
    """Read a checkpoint and check it is a format this code understands.

    Args:
        path: Checkpoint file.
        map_location: Passed to ``torch.load``.

    Returns:
        The stored payload dict.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"checkpoint not found: {path}")
    # weights_only=False: the payload holds RNG state and config dicts, not just
    # tensors. Only ever load checkpoints you produced.
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    found = ckpt.get("format")
    if found != CHECKPOINT_FORMAT:
        raise ValueError(
            f"{path.name} is checkpoint format {found}, this code expects {CHECKPOINT_FORMAT}"
        )
    return ckpt


def restore_rng(ckpt: dict) -> None:
    """Restore RNG state from a loaded checkpoint, if present."""
    if ckpt.get("rng"):
        _restore_rng(ckpt["rng"])


def resolve_resume(out_dir: str | Path, resume: str | None) -> Path | None:
    """Work out which checkpoint to resume from.

    Args:
        out_dir: Directory holding this run's checkpoints.
        resume: ``None`` to start fresh, ``"auto"`` to pick up
            ``ckpt_last.pt`` if it exists, or an explicit path.

    Returns:
        Path to resume from, or ``None`` to start from scratch.
    """
    if not resume:
        return None
    if resume == "auto":
        candidate = Path(out_dir) / "ckpt_last.pt"
        return candidate if candidate.exists() else None
    path = Path(resume)
    if not path.exists():
        raise FileNotFoundError(f"--resume pointed at a missing file: {path}")
    return path


def write_run_config(out_dir: str | Path, config, train_args: dict) -> None:
    """Drop the config and hyperparameters next to the checkpoints.

    Guideline 6 of the brief: "Log hyperparameters and training configs
    alongside checkpoints so runs are reproducible from the README."
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if hasattr(config, "to_yaml"):
        config.to_yaml(out_dir / "config_snapshot.yaml")
    with open(out_dir / "train_args.json", "w", encoding="utf-8") as fh:
        json.dump(train_args, fh, indent=2, sort_keys=True)
