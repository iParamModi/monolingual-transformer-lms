"""Load a trained model and its tokenizer from a checkpoint.

Every evaluation script starts the same way -- load the weights, load the
matching tokenizer, put the model in eval mode -- so that sequence lives here
once rather than four times.
"""

from __future__ import annotations

from pathlib import Path

import sentencepiece as spm
import torch

from model import GPT, GPTConfig
from train.checkpoint import load_checkpoint

# Tokenizer path convention fixed by Phase 1: {lang_dir}/tokenizer/{code}_bpe_{vocab}.model
LANG_DIRS = {"hi": "hindi", "ne": "nepali"}


def load_model(ckpt_path: str | Path, device: str | torch.device = "cpu") -> tuple[GPT, GPTConfig]:
    """Load a model from a checkpoint (full or weights-only export).

    Args:
        ckpt_path: Path to ``ckpt_final.pt`` or ``{lang}_model_final.pt``.
        device: Device to place the model on.

    Returns:
        ``(model, config)``, model in ``eval()`` mode.
    """
    ckpt = load_checkpoint(ckpt_path, map_location=device)
    cfg = GPTConfig.from_dict(ckpt["config"])
    model = GPT(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, cfg


def load_tokenizer(lang: str, vocab_size: int = 10000) -> spm.SentencePieceProcessor:
    """Load the Phase 1 SentencePiece tokenizer for a language.

    Args:
        lang: ``hi`` or ``ne``.
        vocab_size: Vocabulary size used in the model filename (10000 here).

    Returns:
        A ready SentencePieceProcessor.
    """
    if lang not in LANG_DIRS:
        raise ValueError(f"lang must be one of {list(LANG_DIRS)}, got {lang!r}")
    path = Path(LANG_DIRS[lang]) / "tokenizer" / f"{lang}_bpe_{vocab_size}.model"
    if not path.exists():
        raise FileNotFoundError(f"tokenizer not found: {path}")
    return spm.SentencePieceProcessor(model_file=str(path))


def load_for_eval(
    lang: str, ckpt_path: str | Path, device: str | torch.device = "cpu"
) -> tuple[GPT, GPTConfig, spm.SentencePieceProcessor]:
    """Convenience wrapper: model, config, and matching tokenizer together.

    Args:
        lang: ``hi`` or ``ne``. Determines which tokenizer is loaded --
            deliberately explicit rather than inferred from the checkpoint, so
            a Hindi checkpoint can never be silently paired with the Nepali
            tokenizer.
        ckpt_path: Checkpoint to load.
        device: Device to place the model on.

    Returns:
        ``(model, config, tokenizer)``.
    """
    model, cfg = load_model(ckpt_path, device)
    if cfg.lang != lang:
        raise ValueError(
            f"checkpoint was trained as lang={cfg.lang!r}, but --lang {lang!r} "
            f"was requested -- refusing to pair it with the wrong tokenizer"
        )
    sp = load_tokenizer(lang, cfg.vocab_size)
    return model, cfg, sp
