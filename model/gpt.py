"""Decoder-only (GPT-style) Transformer language model, built from primitives.

Only ``nn.Linear``, ``nn.Embedding``, ``nn.LayerNorm`` and ``nn.Dropout`` are
used. There is no ``nn.Transformer*``, no HuggingFace class, and deliberately no
``F.scaled_dot_product_attention`` -- the attention maths is written out so that
every tensor operation in the forward pass is visible, explainable, and
inspectable by the attention analysis in Phase 2.

Forward pass, with shapes for a batch of B sequences of length T:

    ids            (B, T)              token ids
    token + pos    (B, T, d_model)     embeddings, added, then dropout
    per block:
      Q, K, V      (B, h, T, d_head)   three linear projections, split into heads
      scores       (B, h, T, T)        Q K^T / sqrt(d_head)
      masked       (B, h, T, T)        future positions set to -inf
      attn         (B, h, T, T)        softmax over keys; each row sums to 1
      context      (B, h, T, d_head)   attn @ V
      out          (B, T, d_model)     heads concatenated, output projection
      + FFN sublayer, residual around both
    logits         (B, T, vocab_size)  final LayerNorm then the output head
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import GPTConfig

# Label value that cross-entropy ignores. Pretraining never masks, but Phase 3
# finetuning masks prompt tokens so the loss lands only on answer tokens.
IGNORE_INDEX = -100


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention, implemented from first principles.

    Each head works in a ``d_head = d_model // n_heads`` dimensional subspace.
    Heads are folded into the batch dimension so all of them are computed by a
    single batched matmul, then concatenated back and mixed by ``W_o``.
    """

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.d_head = cfg.d_head
        # 1/sqrt(d_head): dot products are sums of d_head terms, so their
        # variance grows with d_head. Unscaled, the softmax saturates towards
        # one-hot and its gradient vanishes. Scaling keeps score variance ~1.
        self.scale = 1.0 / math.sqrt(self.d_head)

        self.W_q = nn.Linear(cfg.d_model, cfg.d_model, bias=cfg.bias)
        self.W_k = nn.Linear(cfg.d_model, cfg.d_model, bias=cfg.bias)
        self.W_v = nn.Linear(cfg.d_model, cfg.d_model, bias=cfg.bias)
        self.W_o = nn.Linear(cfg.d_model, cfg.d_model, bias=cfg.bias)

        self.attn_dropout = nn.Dropout(cfg.attn_dropout)
        self.resid_dropout = nn.Dropout(cfg.dropout)

        # Lower-triangular boolean mask: True = "query t may attend to key j".
        # Registered as a non-persistent buffer, so it follows .to(device) but
        # is not written into the checkpoint (it is derivable from the config).
        causal = torch.tril(
            torch.ones(cfg.max_seq_len, cfg.max_seq_len, dtype=torch.bool)
        )
        self.register_buffer("causal", causal.view(1, 1, cfg.max_seq_len, cfg.max_seq_len),
                             persistent=False)

    def forward(
        self, x: torch.Tensor, return_attn: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Apply causal self-attention.

        Args:
            x: Input of shape ``(B, T, d_model)``.
            return_attn: If True, also return the attention probabilities of
                shape ``(B, n_heads, T, T)``, detached and taken *before*
                attention dropout so they are the true distribution.

        Returns:
            ``(output, attn)`` where output is ``(B, T, d_model)`` and attn is
            ``None`` unless ``return_attn`` is set.
        """
        B, T, C = x.shape

        # (B, T, d) -> (B, T, h, d_head) -> (B, h, T, d_head).
        # The transpose puts heads next to the batch dim so the matmuls below
        # treat (B, h) as independent problems.
        q = self.W_q(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = self.W_k(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = self.W_v(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        # (B, h, T, d_head) @ (B, h, d_head, T) -> (B, h, T, T)
        scores = (q @ k.transpose(-2, -1)) * self.scale

        # Causal mask: position t must not see t+1.. . Applied *before* softmax
        # so the surviving weights renormalise to sum to 1. Row t always keeps
        # at least its own key (the diagonal), so no row is fully -inf and the
        # softmax cannot produce NaN.
        scores = scores.masked_fill(~self.causal[:, :, :T, :T], float("-inf"))

        attn = torch.softmax(scores, dim=-1)  # (B, h, T, T), rows sum to 1
        weights = attn.detach() if return_attn else None

        # Dropout on the probabilities (after capture, and inactive in eval()).
        context = self.attn_dropout(attn) @ v  # (B, h, T, d_head)

        # Concatenate heads: (B, h, T, d_head) -> (B, T, h, d_head) -> (B, T, d)
        context = context.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.W_o(context)), weights


class FeedForward(nn.Module):
    """Position-wise feed-forward network: Linear -> GELU -> Linear.

    Applied independently at every position. The inner width ``d_ff`` is wider
    than ``d_model`` (4x here), giving the block capacity to compute non-linear
    per-token features that attention alone cannot.
    """

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.fc_in = nn.Linear(cfg.d_model, cfg.d_ff)
        self.fc_out = nn.Linear(cfg.d_ff, cfg.d_model)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map ``(B, T, d_model)`` to ``(B, T, d_model)``."""
        return self.dropout(self.fc_out(F.gelu(self.fc_in(x))))


class Block(nn.Module):
    """One pre-norm Transformer block.

    Pre-norm means ``x + Sublayer(LayerNorm(x))`` rather than
    ``LayerNorm(x + Sublayer(x))``. The residual path then carries an unmodified
    signal from the embeddings to the final norm, so gradients reach layer 0
    without passing through a normalisation at every hop -- markedly more stable
    for deep stacks, and it removes the need for a long warmup to avoid early
    divergence.
    """

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.ln_attn = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.ln_ffn = nn.LayerNorm(cfg.d_model)
        self.ffn = FeedForward(cfg)

    def forward(
        self, x: torch.Tensor, return_attn: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Run both sublayers with residual connections around each."""
        attn_out, weights = self.attn(self.ln_attn(x), return_attn=return_attn)
        x = x + attn_out
        x = x + self.ffn(self.ln_ffn(x))
        return x, weights


class GPT(nn.Module):
    """Decoder-only Transformer language model.

    Args:
        cfg: Architecture configuration. The model is fully determined by it.
    """

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg

        self.token_embedding = nn.Embedding(cfg.vocab_size, cfg.d_model)
        # Self-attention is permutation invariant -- without positional
        # information the model sees the context as a bag of tokens. Learned
        # absolute embeddings inject it, at the cost of a hard context cap:
        # there is simply no row for position >= max_seq_len.
        if cfg.pos_encoding == "learned":
            self.position_embedding = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        else:  # "none" -- the bonus ablation
            self.position_embedding = None

        self.embed_dropout = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.ln_final = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        if cfg.tie_weights:
            # One weight matrix serving as both input embedding and output
            # projection: saves vocab_size * d_model parameters and ties the
            # representation a token has as input to the one it is scored by.
            self.head.weight = self.token_embedding.weight

        self.apply(self._init_weights)
        # GPT-2 style residual scaling: every block adds two contributions to
        # the residual stream, so without this the stream's variance grows with
        # depth. Applied to the projections that write into the residual.
        residual_std = cfg.init_std / math.sqrt(2 * cfg.n_layers)
        for block in self.blocks:
            nn.init.normal_(block.attn.W_o.weight, mean=0.0, std=residual_std)
            nn.init.normal_(block.ffn.fc_out.weight, mean=0.0, std=residual_std)

    def _init_weights(self, module: nn.Module) -> None:
        """Initialise Linear and Embedding weights from N(0, init_std)."""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=self.cfg.init_std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.cfg.init_std)

    def forward(
        self,
        ids: torch.Tensor,
        targets: torch.Tensor | None = None,
        return_attn: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None, list[torch.Tensor] | None]:
        """Run the language model.

        Args:
            ids: Token ids, ``(B, T)``.
            targets: Next-token labels, ``(B, T)``. Positions labelled
                ``IGNORE_INDEX`` contribute nothing to the loss.
            return_attn: Collect per-layer attention probabilities.

        Returns:
            ``(logits, loss, attentions)``. ``logits`` is ``(B, T, vocab_size)``,
            ``loss`` is a scalar or None, ``attentions`` is a list of
            ``(B, n_heads, T, T)`` tensors (one per layer) or None.
        """
        B, T = ids.shape
        if T > self.cfg.max_seq_len:
            raise ValueError(
                f"sequence length {T} exceeds max_seq_len {self.cfg.max_seq_len}; "
                "learned positional embeddings have no row beyond that position"
            )

        x = self.token_embedding(ids)  # (B, T, d_model)
        if self.position_embedding is not None:
            positions = torch.arange(T, device=ids.device)
            x = x + self.position_embedding(positions)  # broadcast over batch
        x = self.embed_dropout(x)

        attentions: list[torch.Tensor] | None = [] if return_attn else None
        for block in self.blocks:
            x, weights = block(x, return_attn=return_attn)
            if return_attn:
                attentions.append(weights)

        logits = self.head(self.ln_final(x))  # (B, T, vocab_size)

        loss = None
        if targets is not None:
            # Causal LM objective: logits at position t predict the token at
            # t+1. The shift already lives in the targets built by the data
            # loader, so positions line up one-to-one here.
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=IGNORE_INDEX,
            )
        return logits, loss, attentions

    @torch.no_grad()
    def generate(
        self,
        ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
        eos_id: int | None = None,
    ) -> torch.Tensor:
        """Autoregressively extend ``ids`` by sampling from the model.

        Args:
            ids: Prompt token ids, ``(B, T)``.
            max_new_tokens: Number of tokens to append.
            temperature: ``0`` selects the argmax (greedy); higher values
                flatten the distribution and increase diversity.
            top_k: If set, sample only from the k most likely tokens.
            eos_id: If set, stop early once every sequence has emitted it.

        Returns:
            ``(B, T + n)`` token ids, prompt included.
        """
        was_training = self.training
        self.eval()
        finished = torch.zeros(ids.size(0), dtype=torch.bool, device=ids.device)

        for _ in range(max_new_tokens):
            # Crop to the context window: positions beyond max_seq_len have no
            # positional embedding, so the model can only condition on the last
            # max_seq_len tokens.
            context = ids[:, -self.cfg.max_seq_len:]
            logits, _, _ = self.forward(context)
            logits = logits[:, -1, :]  # (B, vocab_size), last position only

            if temperature == 0:
                next_ids = logits.argmax(dim=-1, keepdim=True)
            else:
                logits = logits / temperature
                if top_k is not None:
                    kth = torch.topk(logits, top_k, dim=-1).values[:, -1:]
                    logits = logits.masked_fill(logits < kth, float("-inf"))
                next_ids = torch.multinomial(torch.softmax(logits, dim=-1), num_samples=1)

            if eos_id is not None:
                # Once a sequence is done, keep emitting eos so shapes stay
                # rectangular; callers truncate at the first eos.
                next_ids = torch.where(finished.unsqueeze(1), torch.full_like(next_ids, eos_id), next_ids)
                finished |= next_ids.squeeze(1) == eos_id

            ids = torch.cat([ids, next_ids], dim=1)
            if eos_id is not None and bool(finished.all()):
                break

        if was_training:
            self.train()
        return ids

    def optimizer_param_groups(self, weight_decay: float) -> list[dict]:
        """Split parameters into decayed and non-decayed groups.

        Matmul weights and embeddings are decayed; LayerNorm gains and biases
        are not -- decaying a 1-D scale or offset towards zero just fights the
        normalisation. Tied weights are a single tensor and appear exactly once.
        """
        decay, no_decay, seen = [], [], set()
        for param in self.parameters():
            if not param.requires_grad or id(param) in seen:
                continue
            seen.add(id(param))
            (decay if param.dim() >= 2 else no_decay).append(param)
        return [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]


def count_params(model: nn.Module, trainable_only: bool = True) -> int:
    """Count parameters, counting a tied weight matrix only once.

    Args:
        model: The module to measure.
        trainable_only: Skip parameters with ``requires_grad=False``.

    Returns:
        Total number of distinct parameter values.
    """
    seen, total = set(), 0
    for param in model.parameters():
        if trainable_only and not param.requires_grad:
            continue
        if param.data_ptr() in seen:  # tied weights share storage
            continue
        seen.add(param.data_ptr())
        total += param.numel()
    return total
