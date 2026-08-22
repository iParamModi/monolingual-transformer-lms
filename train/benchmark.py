"""Measure throughput before committing to a multi-hour run.

Runs a handful of real optimizer steps -- the same ``Trainer.train_step`` the
real run uses, not an approximation -- and reports speed, memory, and the
implied wall clock. Use it to size ``--micro-batch`` to the GPU before starting
Hindi, and again if a session lands on different hardware.

Example::

    python -m train.benchmark --config hindi/configs/hi_25m.yaml \
        --data-dir hindi/data/bin --out-dir /tmp/bench --steps 50
"""

from __future__ import annotations

import time

import torch

from model import GPTConfig
from train.train import build_parser
from train.trainer import Trainer, pick_device


def run_benchmark(trainer: Trainer, steps: int, warmup: int = 5) -> dict:
    """Time ``steps`` optimizer steps, discarding the first ``warmup``.

    The early steps include CUDA context setup, memory-pool growth and kernel
    autotuning, so timing them would understate steady-state throughput.

    Args:
        trainer: A configured trainer.
        steps: Total steps to run.
        warmup: Leading steps excluded from the timing.

    Returns:
        Measurements: seconds per step, tokens per second, peak memory.
    """
    if steps <= warmup:
        raise ValueError(f"--steps ({steps}) must exceed the {warmup} warmup steps")

    for _ in range(warmup):
        trainer.train_step()

    if trainer.device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    timed = steps - warmup
    start = time.time()
    losses = []
    for _ in range(timed):
        loss, _, _ = trainer.train_step()
        losses.append(loss)
    if trainer.device.type == "cuda":
        torch.cuda.synchronize()  # kernels are async; sync before reading the clock
    duration = time.time() - start

    per_step = duration / timed
    return {
        "steps_timed": timed,
        "seconds_per_step": per_step,
        "tokens_per_second": trainer.tokens_per_step / per_step,
        "peak_mem_gb": (torch.cuda.max_memory_allocated() / 1e9
                        if trainer.device.type == "cuda" else 0.0),
        "mean_loss": sum(losses) / len(losses),
    }


def report(trainer: Trainer, result: dict, max_steps: int) -> str:
    """Format the benchmark as a short, decision-ready summary."""
    total_h = result["seconds_per_step"] * max_steps / 3600
    total_gb = (torch.cuda.get_device_properties(0).total_memory / 1e9
                if trainer.device.type == "cuda" else 0.0)
    lines = [
        "",
        f"steps timed        {result['steps_timed']}",
        f"seconds per step   {result['seconds_per_step']:.3f}",
        f"tokens per second  {result['tokens_per_second'] / 1000:.1f}k",
        f"mean loss          {result['mean_loss']:.4f}",
    ]
    if total_gb:
        used = result["peak_mem_gb"]
        lines.append(f"peak GPU memory    {used:.2f} GB of {total_gb:.1f} GB "
                     f"({100 * used / total_gb:.0f}% used)")
        if used / total_gb < 0.7:
            lines.append("                   -> headroom left: raise --micro-batch and "
                         "lower --grad-accum to keep the effective batch fixed")
    lines += [
        "",
        f"projected total    {total_h:.1f} h for {max_steps:,} steps",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    """Run the benchmark and print the summary."""
    parser = build_parser()
    parser.add_argument("--steps", type=int, default=50, help="steps to run, warmup included")
    args = parser.parse_args(argv)

    cfg = GPTConfig.from_yaml(args.config)
    trainer = Trainer(cfg, args, pick_device(args.device))
    print(trainer.describe(), flush=True)
    print(f"\nbenchmarking {args.steps} steps...", flush=True)

    result = run_benchmark(trainer, args.steps)
    print(report(trainer, result, args.max_steps), flush=True)


if __name__ == "__main__":
    main()
