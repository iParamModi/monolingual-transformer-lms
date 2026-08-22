"""Pretrain one monolingual model. Resume-capable.

Model H and Model L are trained by separate invocations of this script, from
separate configs, over separate corpora. Neither is ever initialised from the
other -- the brief forbids it, and nothing here reads a checkpoint except the
one belonging to the run being resumed.

Examples:
    Hindi, from scratch::

        python -m train.train --config hindi/configs/hi_25m.yaml \\
            --data-dir hindi/data/bin --out-dir hindi/checkpoints

    Continue after a session was killed::

        python -m train.train --config hindi/configs/hi_25m.yaml \\
            --data-dir hindi/data/bin --out-dir hindi/checkpoints --resume auto

    Check the model can learn before spending GPU hours::

        python -m train.train --config hindi/configs/hi_25m.yaml \\
            --data-dir hindi/data/bin --out-dir /tmp/overfit --overfit 200
"""

from __future__ import annotations

import argparse
from pathlib import Path

from model import GPTConfig
from train.checkpoint import resolve_resume
from train.trainer import Trainer, pick_device


def build_parser() -> argparse.ArgumentParser:
    """Command-line interface. Defaults are the values planned for both runs."""
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)

    g = p.add_argument_group("what to train")
    g.add_argument("--config", required=True, help="model config YAML")
    g.add_argument("--data-dir", required=True, help="directory holding the .bin files")
    g.add_argument("--out-dir", required=True, help="where checkpoints and logs go")
    g.add_argument("--train-bin", default="train.bin")
    g.add_argument("--val-bin", default="val.bin")

    g = p.add_argument_group("schedule")
    g.add_argument("--max-steps", type=int, default=14500,
                   help="14500 x 64 x 512 = 475,136,000 tokens, ~1 epoch for both corpora")
    g.add_argument("--block-size", type=int, default=512, help="tokens per window")
    g.add_argument("--micro-batch", type=int, default=16, help="windows per forward pass")
    g.add_argument("--grad-accum", type=int, default=4,
                   help="micro-batches per optimizer step; effective batch = micro_batch x this")

    g = p.add_argument_group("optimizer")
    g.add_argument("--lr", type=float, default=6e-4, help="peak learning rate")
    g.add_argument("--min-lr", type=float, default=6e-5, help="cosine floor")
    g.add_argument("--warmup-steps", type=int, default=500)
    g.add_argument("--weight-decay", type=float, default=0.1)
    g.add_argument("--beta1", type=float, default=0.9)
    g.add_argument("--beta2", type=float, default=0.95, help="0.95 suits noisy small-batch grads")
    g.add_argument("--grad-clip", type=float, default=1.0)

    g = p.add_argument_group("evaluation and saving")
    g.add_argument("--eval-every", type=int, default=250)
    g.add_argument("--eval-batches", type=int, default=40)
    g.add_argument("--eval-batch-size", type=int, default=16)
    g.add_argument("--log-every", type=int, default=50)
    g.add_argument("--save-every", type=int, default=500)

    g = p.add_argument_group("run control")
    g.add_argument("--resume", default=None,
                   help="'auto' to continue from ckpt_last.pt, or a checkpoint path")
    g.add_argument("--seed", type=int, default=1337)
    g.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    g.add_argument("--overfit", type=int, default=0, metavar="STEPS",
                   help="sanity check: train on a single batch for STEPS steps and exit")
    return p


def main(argv: list[str] | None = None) -> None:
    """Parse arguments, build the trainer, and run."""
    args = build_parser().parse_args(argv)
    cfg = GPTConfig.from_yaml(args.config)
    device = pick_device(args.device)

    trainer = Trainer(cfg, args, device)

    if args.overfit:
        print(trainer.describe(), flush=True)
        print(f"\noverfitting one batch for {args.overfit} steps "
              f"-- the loss must fall towards 0\n", flush=True)
        trainer.overfit_batch(args.overfit)
        return

    resume_path = resolve_resume(args.out_dir, args.resume)
    if resume_path is not None:
        trainer.resume_from(resume_path)
        trainer.fit(resumed=True)
    else:
        if args.resume == "auto":
            print("no ckpt_last.pt found -- starting from scratch", flush=True)
        trainer.fit(resumed=False)


if __name__ == "__main__":
    main()
