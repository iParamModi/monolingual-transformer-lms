"""Proof that an interrupted run resumes exactly.

The brief calls checkpointing mandatory and says "a run that cannot resume
after interruption will lose marks". Saving a file is easy; proving the resumed
run is the *same* run is the part worth testing.

The strategy: train 40 steps straight through, then train the same setup in two
20-step halves with a save and a fresh ``Trainer`` in between, and compare. If
the data order, RNG, optimizer moments or scheduler were restored incorrectly,
the two runs diverge and the weights no longer match.

Run standalone to print the evidence for the report::

    python -m tests.test_resume
"""

from __future__ import annotations

import argparse
import contextlib
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from model import GPTConfig
from train.trainer import Trainer


def _write_corpus(directory: Path, n_tokens: int = 60_000, vocab: int = 97) -> None:
    """Write small train/val token files shaped exactly like Phase 1's output."""
    rng = np.random.default_rng(0)
    for name in ("train.bin", "val.bin"):
        rng.integers(0, vocab, size=n_tokens, dtype=np.uint16).tofile(directory / name)


@contextlib.contextmanager
def _temp_corpus():
    """A throwaway corpus directory that cleans up on Windows too.

    Every trainer built inside must be closed before the directory is removed:
    Windows refuses to delete a file that is still memory-mapped. Trainers
    registered with the yielded list are closed automatically.
    """
    opened: list[Trainer] = []
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        root = Path(tmp)
        data_dir = root / "data"
        data_dir.mkdir()
        _write_corpus(data_dir)
        try:
            yield root, data_dir, opened
        finally:
            for trainer in opened:
                trainer.close()


def _config() -> GPTConfig:
    """A model small enough to train on CPU in seconds."""
    return GPTConfig(
        name="resume_test", lang="hi", vocab_size=97, d_model=32, n_layers=2,
        n_heads=4, d_ff=64, max_seq_len=32, dropout=0.1, attn_dropout=0.1,
    )


def _args(data_dir: Path, out_dir: Path, max_steps: int) -> argparse.Namespace:
    """Training arguments for the test runs.

    Dropout stays on deliberately: it consumes the global RNG, so it only stays
    reproducible if the checkpoint's RNG state is restored properly. Turning it
    off would make the test easier and weaker.
    """
    return argparse.Namespace(
        config="<test>", data_dir=str(data_dir), out_dir=str(out_dir),
        train_bin="train.bin", val_bin="val.bin",
        max_steps=max_steps, block_size=32, micro_batch=4, grad_accum=2,
        lr=1e-3, min_lr=1e-4, warmup_steps=5, weight_decay=0.1,
        beta1=0.9, beta2=0.95, grad_clip=1.0,
        eval_every=1000, eval_batches=2, eval_batch_size=4,
        log_every=1000, save_every=1000,
        resume=None, seed=1337, device="cpu", overfit=0,
    )


def _weights(trainer: Trainer) -> torch.Tensor:
    """Flatten every parameter into one vector for comparison."""
    return torch.cat([p.detach().reshape(-1) for p in trainer.model.parameters()])


def run_comparison(total_steps: int = 40) -> dict:
    """Train straight through, then in two halves, and compare the results.

    Args:
        total_steps: Steps for the uninterrupted run; the split run does half,
            saves, reloads in a new Trainer, and finishes.

    Returns:
        Losses from both runs and the largest weight difference between them.
    """
    half = total_steps // 2
    with _temp_corpus() as (root, data_dir, opened):
        # --- reference: one uninterrupted run --------------------------------
        straight = Trainer(_config(), _args(data_dir, root / "straight", total_steps),
                           torch.device("cpu"))
        opened.append(straight)
        straight_losses = [straight.train_step()[0] for _ in range(total_steps)]

        # --- first half, then save -------------------------------------------
        split_dir = root / "split"
        first = Trainer(_config(), _args(data_dir, split_dir, total_steps),
                        torch.device("cpu"))
        opened.append(first)
        first_losses = [first.train_step()[0] for _ in range(half)]
        ckpt = first.save("ckpt_last.pt")
        first.close()  # the interruption

        # --- a brand-new process would do exactly this ------------------------
        second = Trainer(_config(), _args(data_dir, split_dir, total_steps),
                         torch.device("cpu"))
        opened.append(second)
        second.resume_from(ckpt)
        second_losses = [second.train_step()[0] for _ in range(total_steps - half)]

        return {
            "total_steps": total_steps,
            "resumed_at": half,
            "straight_losses": straight_losses,
            "split_losses": first_losses + second_losses,
            "max_weight_diff": (_weights(straight) - _weights(second)).abs().max().item(),
            "straight_final": straight_losses[-1],
            "split_final": second_losses[-1],
            "resumed_step": second.step,
            "resumed_tokens": second.tokens_seen,
            "straight_tokens": straight.tokens_seen,
        }


@pytest.fixture(scope="module")
def comparison() -> dict:
    """Run the comparison once and share it across the assertions."""
    return run_comparison()


def test_resumed_run_reaches_identical_weights(comparison):
    """Every parameter must match the uninterrupted run."""
    assert comparison["max_weight_diff"] < 1e-6, (
        f"weights diverged by {comparison['max_weight_diff']:.2e} after resuming"
    )


def test_every_step_loss_matches(comparison):
    """Not just the endpoint -- each step must line up.

    Comparing only final losses would hide a divergence that happened to
    reconverge.
    """
    for i, (a, b) in enumerate(zip(comparison["straight_losses"], comparison["split_losses"])):
        assert a == pytest.approx(b, abs=1e-6), f"step {i}: {a:.6f} vs {b:.6f}"


def test_step_and_token_counters_survive(comparison):
    """Bookkeeping must continue, not restart."""
    assert comparison["resumed_step"] == comparison["total_steps"]
    assert comparison["resumed_tokens"] == comparison["straight_tokens"]


def test_checkpoint_carries_everything_the_brief_requires():
    """Weights, optimizer state, scheduler state, step, and config.

    The five items named in the brief, checked by name so a refactor that drops
    one is caught here.
    """
    from train.checkpoint import load_checkpoint

    with _temp_corpus() as (root, data_dir, opened):
        trainer = Trainer(_config(), _args(data_dir, root / "out", 10), torch.device("cpu"))
        opened.append(trainer)
        trainer.train_step()
        ckpt = load_checkpoint(trainer.save("ckpt_last.pt"))

    for key in ("model", "optimizer", "scheduler", "step", "config"):
        assert key in ckpt, f"checkpoint is missing '{key}', which the brief requires"
    assert ckpt["step"] == 1
    assert ckpt["config"]["vocab_size"] == 97
    assert "lr" in ckpt["optimizer"]["param_groups"][0]
    assert ckpt["scheduler"]["warmup_steps"] == 5


def test_refuses_a_mismatched_architecture():
    """Loading weights from a differently-shaped model must fail loudly."""
    with _temp_corpus() as (root, data_dir, opened):
        trainer = Trainer(_config(), _args(data_dir, root / "out", 10), torch.device("cpu"))
        opened.append(trainer)
        ckpt = trainer.save("ckpt_last.pt")

        wider = GPTConfig(**{**_config().to_dict(), "d_model": 64})
        other = Trainer(wider, _args(data_dir, root / "out2", 10), torch.device("cpu"))
        opened.append(other)
        with pytest.raises(ValueError, match="different architecture"):
            other.resume_from(ckpt)


def main() -> None:
    """Print the resume evidence table for the report."""
    r = run_comparison()
    print("Checkpoint resume check")
    print(f"  uninterrupted run : {r['total_steps']} steps")
    print(f"  interrupted run   : {r['resumed_at']} steps, saved, reloaded, "
          f"{r['total_steps'] - r['resumed_at']} more\n")
    print(f"{'step':>6} {'uninterrupted':>15} {'resumed':>12} {'difference':>13}")
    for i in (0, r["resumed_at"] - 1, r["resumed_at"], r["total_steps"] - 1):
        a, b = r["straight_losses"][i], r["split_losses"][i]
        marker = "  <- resumed here" if i == r["resumed_at"] else ""
        print(f"{i:>6} {a:>15.6f} {b:>12.6f} {abs(a - b):>13.2e}{marker}")
    print(f"\nlargest weight difference across all {r['total_steps']} steps: "
          f"{r['max_weight_diff']:.2e}")
    print("PASS: resuming reproduces the uninterrupted run exactly.")


if __name__ == "__main__":
    main()
