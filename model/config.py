"""Model configuration for the decoder-only Transformer.

One :class:`GPTConfig` fully determines the architecture. Configs are stored as
YAML (``hindi/configs/hi_25m.yaml``, ``nepali/configs/ne_25m.yaml``) and are
embedded inside every checkpoint, so a checkpoint alone is enough to rebuild the
model -- Phase 3 finetuning never re-specifies the architecture.

Run as a script to print the parameter budget for a config file::

    python -m model.config --config hindi/configs/hi_25m.yaml
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import yaml


@dataclass
class GPTConfig:
    """Architecture of one monolingual decoder-only Transformer.

    Attributes:
        name: Identifier used in log and checkpoint filenames.
        lang: Language code, ``hi`` or ``ne``. Kept so a checkpoint can never be
            paired with the wrong tokenizer by accident.
        vocab_size: Size of this language's own vocabulary (Phase 1). The two
            models must never share a vocabulary.
        d_model: Residual-stream width.
        n_layers: Number of stacked Transformer blocks.
        n_heads: Attention heads per block; ``d_model`` must divide evenly.
        d_ff: Inner width of the position-wise feed-forward network.
        max_seq_len: Context length. With learned positional embeddings this is
            a hard cap: there is no embedding row for position >= max_seq_len.
        dropout: Dropout on embeddings and both sublayer outputs.
        attn_dropout: Dropout applied to the attention probabilities.
        tie_weights: Share the output projection with the token embedding.
        pos_encoding: ``learned`` for learned absolute positions, ``none`` for
            the no-positional-information ablation (bonus).
        bias: Whether attention projections carry a bias term. The feed-forward
            network always uses biases.
        init_std: Standard deviation of the normal weight initialisation.
    """

    name: str
    lang: str
    vocab_size: int
    d_model: int
    n_layers: int
    n_heads: int
    d_ff: int
    max_seq_len: int
    dropout: float = 0.1
    attn_dropout: float = 0.1
    tie_weights: bool = True
    pos_encoding: str = "learned"
    bias: bool = False
    init_std: float = 0.02

    def __post_init__(self) -> None:
        if self.d_model % self.n_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})"
            )
        if self.pos_encoding not in {"learned", "none"}:
            raise ValueError(
                f"pos_encoding must be 'learned' or 'none', got {self.pos_encoding!r}"
            )
        if self.lang not in {"hi", "ne"}:
            raise ValueError(f"lang must be 'hi' or 'ne', got {self.lang!r}")

    @property
    def d_head(self) -> int:
        """Per-head subspace width, ``d_model // n_heads``."""
        return self.d_model // self.n_heads

    @classmethod
    def from_yaml(cls, path: str | Path) -> "GPTConfig":
        """Load a config from a YAML file, ignoring unknown keys."""
        with open(path, encoding="utf-8") as fh:
            raw = yaml.safe_load(fh)
        known = {f.name for f in fields(cls)}
        unknown = set(raw) - known
        if unknown:
            raise ValueError(f"unknown config keys in {path}: {sorted(unknown)}")
        return cls(**raw)

    @classmethod
    def from_dict(cls, raw: dict) -> "GPTConfig":
        """Rebuild a config from the dict stored inside a checkpoint."""
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in raw.items() if k in known})

    def to_dict(self) -> dict:
        """Plain-dict form, as embedded in checkpoints."""
        return asdict(self)

    def to_yaml(self, path: str | Path) -> None:
        """Write the config to YAML (used to snapshot a run's exact config)."""
        with open(path, "w", encoding="utf-8") as fh:
            yaml.safe_dump(self.to_dict(), fh, sort_keys=False, allow_unicode=True)

    def param_budget(self) -> dict[str, int]:
        """Analytic parameter count, component by component.

        Computed from the config alone -- no model is built. ``tests/test_params.py``
        asserts this matches the real ``count_params(GPT(cfg))``, which is what
        catches a silent architecture change.

        Returns:
            Mapping from component name to parameter count, including ``total``.
        """
        d, ff, v = self.d_model, self.d_ff, self.vocab_size
        attn = 4 * d * d + (4 * d if self.bias else 0)
        mlp = (d * ff + ff) + (ff * d + d)
        norms = 2 * (2 * d)
        per_block = attn + mlp + norms

        budget = {
            "token_embedding": v * d,
            "positional_embedding": self.max_seq_len * d if self.pos_encoding == "learned" else 0,
            "attention_per_block": attn,
            "feedforward_per_block": mlp,
            "layernorm_per_block": norms,
            "per_block_total": per_block,
            "all_blocks": self.n_layers * per_block,
            "final_layernorm": 2 * d,
            "output_head": 0 if self.tie_weights else v * d,
        }
        budget["total"] = (
            budget["token_embedding"]
            + budget["positional_embedding"]
            + budget["all_blocks"]
            + budget["final_layernorm"]
            + budget["output_head"]
        )
        return budget

    def summary(self) -> str:
        """Human-readable architecture and parameter summary."""
        b = self.param_budget()
        lines = [
            f"config           {self.name}  (lang={self.lang})",
            f"vocab_size       {self.vocab_size:,}",
            f"d_model          {self.d_model}   ({self.n_heads} heads x {self.d_head} dims)",
            f"n_layers         {self.n_layers}",
            f"d_ff             {self.d_ff}",
            f"max_seq_len      {self.max_seq_len}   (hard cap: {self.pos_encoding} positions)",
            f"dropout          {self.dropout} (attn {self.attn_dropout})",
            f"tie_weights      {self.tie_weights}",
            "",
            "parameter budget",
            f"  token embedding      {b['token_embedding']:>12,}",
            f"  positional embedding {b['positional_embedding']:>12,}",
            f"  per block            {b['per_block_total']:>12,}"
            f"   (attn {b['attention_per_block']:,} + ffn {b['feedforward_per_block']:,}"
            f" + ln {b['layernorm_per_block']:,})",
            f"  x {self.n_layers} blocks           {b['all_blocks']:>12,}",
            f"  final layernorm      {b['final_layernorm']:>12,}",
            f"  output head          {b['output_head']:>12,}"
            f"   {'(tied to token embedding)' if self.tie_weights else ''}",
            f"  TOTAL                {b['total']:>12,}   ({b['total'] / 1e6:.2f}M)",
        ]
        return "\n".join(lines)


def main() -> None:
    """Print the parameter budget for a config file."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True, help="path to a YAML config file")
    args = ap.parse_args()
    print(GPTConfig.from_yaml(args.config).summary())


if __name__ == "__main__":
    main()
