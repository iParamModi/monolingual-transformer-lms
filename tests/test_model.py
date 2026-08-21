"""Structural tests: tensor shapes, attention properties, and parameter counts."""

from __future__ import annotations

import math

import pytest
import torch

from model import GPT, GPTConfig, IGNORE_INDEX, count_params


@pytest.fixture
def cfg() -> GPTConfig:
    """Small config that exercises every code path cheaply."""
    return GPTConfig(
        name="tiny", lang="hi", vocab_size=50, d_model=32, n_layers=2,
        n_heads=4, d_ff=64, max_seq_len=16, dropout=0.0, attn_dropout=0.0,
    )


# --------------------------------------------------------------------------- #
# shapes
# --------------------------------------------------------------------------- #

def test_logit_shape(cfg):
    """(B, T) ids must produce (B, T, vocab_size) logits."""
    model = GPT(cfg).eval()
    logits, loss, attn = model(torch.randint(0, cfg.vocab_size, (3, 10)))
    assert logits.shape == (3, 10, cfg.vocab_size)
    assert loss is None and attn is None


def test_attention_shapes(cfg):
    """Returned attention is one (B, h, T, T) tensor per layer."""
    model = GPT(cfg).eval()
    _, _, attn = model(torch.randint(0, cfg.vocab_size, (2, 7)), return_attn=True)
    assert len(attn) == cfg.n_layers
    for layer in attn:
        assert layer.shape == (2, cfg.n_heads, 7, 7)


def test_sequence_longer_than_context_is_rejected(cfg):
    """The learned positional table caps context; going past it must raise."""
    model = GPT(cfg).eval()
    too_long = torch.randint(0, cfg.vocab_size, (1, cfg.max_seq_len + 1))
    with pytest.raises(ValueError, match="max_seq_len"):
        model(too_long)


# --------------------------------------------------------------------------- #
# attention properties
# --------------------------------------------------------------------------- #

def test_attention_rows_sum_to_one(cfg):
    """Softmax over keys: every query's weights form a distribution."""
    model = GPT(cfg).eval()
    _, _, attn = model(torch.randint(0, cfg.vocab_size, (2, 12)), return_attn=True)
    for layer in attn:
        torch.testing.assert_close(
            layer.sum(dim=-1), torch.ones_like(layer.sum(dim=-1)), atol=1e-5, rtol=0
        )


def test_attention_is_strictly_lower_triangular(cfg):
    """Every weight above the diagonal must be exactly zero, not merely small."""
    model = GPT(cfg).eval()
    T = 12
    _, _, attn = model(torch.randint(0, cfg.vocab_size, (2, T)), return_attn=True)
    future = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)
    for i, layer in enumerate(attn):
        assert layer[:, :, future].abs().max().item() == 0.0, f"layer {i} attends to the future"


# --------------------------------------------------------------------------- #
# loss
# --------------------------------------------------------------------------- #

def test_untrained_loss_is_near_uniform(cfg):
    """At init the model should be about as good as guessing: loss ~ ln(V)."""
    torch.manual_seed(0)
    model = GPT(cfg).eval()
    ids = torch.randint(0, cfg.vocab_size, (8, 16))
    _, loss, _ = model(ids, targets=ids)
    assert abs(loss.item() - math.log(cfg.vocab_size)) < 0.5


def test_ignore_index_excludes_masked_positions(cfg):
    """Positions labelled IGNORE_INDEX must not contribute to the loss.

    Phase 3 relies on this to train only on answer tokens.
    """
    torch.manual_seed(0)
    model = GPT(cfg).eval()
    ids = torch.randint(0, cfg.vocab_size, (4, 8))
    targets = ids.clone()

    _, full_loss, _ = model(ids, targets=targets)
    masked = targets.clone()
    masked[:, :4] = IGNORE_INDEX
    _, tail_loss, _ = model(ids, targets=masked)

    # Recompute the tail-only loss by hand and check it matches.
    logits, _, _ = model(ids)
    manual = torch.nn.functional.cross_entropy(
        logits[:, 4:].reshape(-1, cfg.vocab_size), targets[:, 4:].reshape(-1)
    )
    torch.testing.assert_close(tail_loss, manual, atol=1e-6, rtol=0)
    assert not torch.isclose(full_loss, tail_loss)


# --------------------------------------------------------------------------- #
# parameters
# --------------------------------------------------------------------------- #

def test_analytic_budget_matches_real_model(cfg):
    """The config's parameter arithmetic must match the built model exactly."""
    assert count_params(GPT(cfg)) == cfg.param_budget()["total"]


@pytest.mark.parametrize("path", ["hindi/configs/hi_25m.yaml", "nepali/configs/ne_25m.yaml"])
def test_project_configs_hit_the_budget(path):
    """Both shipped configs must be the planned 24,285,184 parameters."""
    cfg = GPTConfig.from_yaml(path)
    assert cfg.param_budget()["total"] == 24_285_184
    assert count_params(GPT(cfg)) == 24_285_184


def test_tied_weights_share_storage(cfg):
    """Weight tying must be real sharing, not a copy."""
    model = GPT(cfg)
    assert model.head.weight.data_ptr() == model.token_embedding.weight.data_ptr()


def test_untied_model_is_larger_by_one_embedding(cfg):
    """Untying must cost exactly vocab_size * d_model extra parameters."""
    untied = GPTConfig(**{**cfg.to_dict(), "tie_weights": False})
    assert count_params(GPT(untied)) - count_params(GPT(cfg)) == cfg.vocab_size * cfg.d_model


def test_optimizer_groups_cover_every_parameter_once(cfg):
    """Decay/no-decay split must partition the parameters, with no duplicates."""
    model = GPT(cfg)
    groups = model.optimizer_param_groups(weight_decay=0.1)
    ids = [id(p) for g in groups for p in g["params"]]
    assert len(ids) == len(set(ids)), "a parameter appears in two groups"
    assert sum(p.numel() for g in groups for p in g["params"]) == count_params(model)
    assert all(p.dim() >= 2 for p in groups[0]["params"])
    assert all(p.dim() < 2 for p in groups[1]["params"])


# --------------------------------------------------------------------------- #
# ablation variant (bonus)
# --------------------------------------------------------------------------- #

def test_no_positional_variant_drops_the_table(cfg):
    """pos_encoding='none' removes the positional embedding entirely."""
    ablated = GPTConfig(**{**cfg.to_dict(), "pos_encoding": "none"})
    model = GPT(ablated)
    assert model.position_embedding is None
    assert count_params(model) == count_params(GPT(cfg)) - cfg.max_seq_len * cfg.d_model


def _last_logits(model, ids):
    """Logits at the final position, which attends over the whole sequence."""
    with torch.no_grad():
        logits, _, _ = model(ids)
    return logits[0, -1]


def test_single_layer_without_positions_is_permutation_invariant(cfg):
    """Self-attention alone cannot tell order apart -- the reason positions exist.

    Checked with one layer, where every key and value is a function of its own
    token only. The final position attends over the same multiset of keys
    whichever order they arrive in, so its logits are unchanged.
    """
    ablated = GPTConfig(**{**cfg.to_dict(), "pos_encoding": "none", "n_layers": 1})
    torch.manual_seed(0)
    model = GPT(ablated).eval()
    ids = torch.randint(0, cfg.vocab_size, (1, 6))
    shuffled = ids.clone()
    shuffled[0, :5] = ids[0, :5].flip(0)  # reorder everything before the last token
    torch.testing.assert_close(_last_logits(model, ids), _last_logits(model, shuffled),
                               atol=1e-5, rtol=0)


def test_positional_embeddings_break_that_symmetry(cfg):
    """The same single-layer model *with* positions must notice the reordering.

    This is the control for the test above: it shows the invariance comes from
    the missing positional information, not from the model ignoring its input.
    """
    with_pos = GPTConfig(**{**cfg.to_dict(), "n_layers": 1})
    torch.manual_seed(0)
    model = GPT(with_pos).eval()
    ids = torch.randint(0, cfg.vocab_size, (1, 6))
    shuffled = ids.clone()
    shuffled[0, :5] = ids[0, :5].flip(0)
    delta = (_last_logits(model, ids) - _last_logits(model, shuffled)).abs().max()
    assert delta > 1e-4


def test_causal_mask_leaks_weak_position_information_at_depth(cfg):
    """With two or more layers, removing positions is not full invariance.

    Layer 2 reads representations that layer 1 built from each token's own
    causal prefix, and prefix *length* varies by position. So a stack deeper
    than one layer can still distinguish some orderings even with no positional
    embeddings -- the ablated model degrades sharply but does not become a pure
    bag of tokens. Worth stating explicitly in the bonus ablation write-up.
    """
    ablated = GPTConfig(**{**cfg.to_dict(), "pos_encoding": "none", "n_layers": 2})
    torch.manual_seed(0)
    model = GPT(ablated).eval()
    ids = torch.randint(0, cfg.vocab_size, (1, 6))
    shuffled = ids.clone()
    shuffled[0, :5] = ids[0, :5].flip(0)
    delta = (_last_logits(model, ids) - _last_logits(model, shuffled)).abs().max()
    assert delta > 1e-6


# --------------------------------------------------------------------------- #
# generation
# --------------------------------------------------------------------------- #

def test_generate_appends_exactly_n_tokens(cfg):
    """Generation must extend the prompt by the requested length."""
    model = GPT(cfg).eval()
    prompt = torch.randint(0, cfg.vocab_size, (2, 5))
    out = model.generate(prompt, max_new_tokens=7, temperature=1.0)
    assert out.shape == (2, 12)
    assert torch.equal(out[:, :5], prompt), "the prompt must be preserved"


def test_greedy_generation_is_deterministic(cfg):
    """temperature=0 must give identical output across calls."""
    model = GPT(cfg).eval()
    prompt = torch.randint(0, cfg.vocab_size, (1, 4))
    a = model.generate(prompt, max_new_tokens=6, temperature=0.0)
    b = model.generate(prompt, max_new_tokens=6, temperature=0.0)
    assert torch.equal(a, b)


def test_generation_respects_the_context_cap(cfg):
    """Generating past max_seq_len must work by cropping, not by crashing."""
    model = GPT(cfg).eval()
    prompt = torch.randint(0, cfg.vocab_size, (1, cfg.max_seq_len))
    out = model.generate(prompt, max_new_tokens=4, temperature=0.0)
    assert out.shape == (1, cfg.max_seq_len + 4)
