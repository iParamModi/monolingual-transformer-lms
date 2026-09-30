"""Strip a training checkpoint down to a weights-only model file.

A full checkpoint is ~292 MB because AdamW keeps two momentum buffers per
parameter. Evaluation and Phase 3 need none of that -- only the weights and the
config. Exporting gives a ~97 MB file that is quicker to upload, quicker to
download, and unambiguous about what it is.

Keep both: ``ckpt_final.pt`` is what the brief means by a resumable checkpoint,
``model_final.pt`` is what everything downstream actually loads.

Example::

    python -m train.export --ckpt hindi/checkpoints/ckpt_final.pt \
        --out hindi/checkpoints/hi_model_final.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from train.checkpoint import CHECKPOINT_FORMAT, load_checkpoint


def export_weights(ckpt_path: str | Path, out_path: str | Path) -> Path:
    """Write a weights-and-config-only copy of a checkpoint.

    Args:
        ckpt_path: Full training checkpoint.
        out_path: Destination for the slim file.

    Returns:
        The path written.
    """
    ckpt = load_checkpoint(ckpt_path, map_location="cpu")
    payload = {
        "format": CHECKPOINT_FORMAT,
        "model": ckpt["model"],
        "config": ckpt["config"],
        "step": ckpt["step"],
        "tokens_seen": ckpt["tokens_seen"],
        "best_val": ckpt["best_val"],
        "metrics": ckpt.get("metrics", {}),
    }
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out_path)
    return out_path


def main(argv: list[str] | None = None) -> None:
    """Export a checkpoint and report the size saved."""
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True, help="full training checkpoint")
    ap.add_argument("--out", required=True, help="destination for the weights-only file")
    args = ap.parse_args(argv)

    out = export_weights(args.ckpt, args.out)
    before = Path(args.ckpt).stat().st_size / 1e6
    after = out.stat().st_size / 1e6
    print(f"{args.ckpt}  {before:.0f} MB\n{out}  {after:.0f} MB  "
          f"({100 * (1 - after / before):.0f}% smaller)")


if __name__ == "__main__":
    main()
