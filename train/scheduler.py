"""Learning-rate schedule: linear warmup followed by cosine decay.

Written as a stateful object with ``state_dict``/``load_state_dict`` rather than
as a bare function of the step count. The brief requires every checkpoint to
carry "scheduler state", so the checkpoint gets a literal ``scheduler`` entry
and there is nothing to argue about. The schedule is still a pure function of
the step, so restoring it can never drift from the optimizer.
"""

from __future__ import annotations

import math


class CosineWarmupLR:
    """Linear warmup to ``peak_lr``, then cosine decay to ``min_lr``.

    Warmup exists because AdamW's second-moment estimate is unreliable in the
    first few hundred steps; starting at full learning rate there is the usual
    cause of an early loss spike. Cosine decay then anneals smoothly, which
    reliably reaches a lower final loss than holding the rate constant.

    Args:
        optimizer: Optimizer whose param groups get the rate written into them.
        peak_lr: Highest learning rate, reached at the end of warmup.
        min_lr: Floor the cosine decays to at ``max_steps``.
        warmup_steps: Steps spent ramping linearly from ~0 to ``peak_lr``.
        max_steps: Step at which the cosine reaches ``min_lr``.
    """

    def __init__(
        self,
        optimizer,
        peak_lr: float,
        min_lr: float,
        warmup_steps: int,
        max_steps: int,
    ):
        if warmup_steps >= max_steps:
            raise ValueError(
                f"warmup_steps ({warmup_steps}) must be less than max_steps ({max_steps})"
            )
        self.optimizer = optimizer
        self.peak_lr = peak_lr
        self.min_lr = min_lr
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.last_lr = 0.0

    def lr_at(self, step: int) -> float:
        """Learning rate for a given step, without touching the optimizer.

        Args:
            step: Zero-based optimizer step about to be taken.

        Returns:
            The learning rate for that step.
        """
        if step < self.warmup_steps:
            # step + 1 so the very first step trains at a non-zero rate.
            return self.peak_lr * (step + 1) / self.warmup_steps
        if step >= self.max_steps:
            return self.min_lr
        progress = (step - self.warmup_steps) / (self.max_steps - self.warmup_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))  # 1 -> 0
        return self.min_lr + (self.peak_lr - self.min_lr) * cosine

    def step_to(self, step: int) -> float:
        """Write the rate for ``step`` into every optimizer param group.

        Args:
            step: Zero-based optimizer step about to be taken.

        Returns:
            The learning rate applied.
        """
        lr = self.lr_at(step)
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        self.last_lr = lr
        return lr

    def state_dict(self) -> dict:
        """Schedule parameters, as stored in the checkpoint."""
        return {
            "peak_lr": self.peak_lr,
            "min_lr": self.min_lr,
            "warmup_steps": self.warmup_steps,
            "max_steps": self.max_steps,
            "last_lr": self.last_lr,
        }

    def load_state_dict(self, state: dict) -> None:
        """Restore schedule parameters from a checkpoint."""
        self.peak_lr = state["peak_lr"]
        self.min_lr = state["min_lr"]
        self.warmup_steps = state["warmup_steps"]
        self.max_steps = state["max_steps"]
        self.last_lr = state.get("last_lr", 0.0)
