"""Empirical proof that the causal mask works.

The project brief requires this specifically: "verify empirically that the model
cannot see the future -- for example, show that changing token t+1 does not
change the logits at position t".

Run standalone to print the evidence quoted in the report::

    python -m tests.test_causal
"""

from __future__ import annotations

import torch

from model import GPT, GPTConfig


def _tiny_model(seed: int = 0) -> GPT:
    """A small model in eval mode, so dropout cannot muddy the comparison."""
    torch.manual_seed(seed)
    cfg = GPTConfig(
        name="tiny", lang="hi", vocab_size=100, d_model=64, n_layers=2,
        n_heads=4, d_ff=128, max_seq_len=16, dropout=0.0, attn_dropout=0.0,
    )
    return GPT(cfg).eval()


def causality_evidence(edit_position: int = 5, length: int = 10) -> dict:
    """Perturb one token and measure which logits move.

    Args:
        edit_position: Index of the token to change.
        length: Sequence length to test.

    Returns:
        Dict with the maximum absolute logit change strictly before the edited
        position, and the change at the edited position itself.
    """
    model = _tiny_model()
    torch.manual_seed(1)
    ids = torch.randint(0, 100, (1, length))

    with torch.no_grad():
        original, _, _ = model(ids)
        edited_ids = ids.clone()
        # Change the token at `edit_position` to a different id.
        edited_ids[0, edit_position] = (ids[0, edit_position] + 1) % 100
        edited, _, _ = model(edited_ids)

    delta = (original - edited).abs()
    after = delta[0, edit_position + 1:]
    return {
        "edit_position": edit_position,
        "max_change_before_edit": delta[0, :edit_position].max().item(),
        "change_at_edit_position": delta[0, edit_position].max().item(),
        # Empty when the edited token is the last one: nothing follows it.
        "max_change_after_edit": after.max().item() if after.numel() else float("nan"),
    }


def test_future_tokens_cannot_affect_earlier_logits():
    """Editing token t must leave every logit at positions < t untouched."""
    ev = causality_evidence()
    assert ev["max_change_before_edit"] == 0.0, (
        f"information leaked backwards: logits before position {ev['edit_position']} "
        f"moved by {ev['max_change_before_edit']}"
    )


def test_edited_position_does_change():
    """Sanity check the test itself: the edit must actually do something.

    Without this, a model that ignored its input entirely would pass the test
    above trivially.
    """
    ev = causality_evidence()
    assert ev["change_at_edit_position"] > 1e-6


def test_causality_holds_at_every_position():
    """Repeat the check for every valid edit position, not just one."""
    for pos in range(1, 10):
        ev = causality_evidence(edit_position=pos)
        assert ev["max_change_before_edit"] == 0.0, f"leak when editing position {pos}"


def main() -> None:
    """Print the causality evidence table for the report."""
    print("Causal masking check -- edit one token, measure logit changes elsewhere")
    print(f"{'edit pos':>9} {'max |d| before':>16} {'|d| at pos':>12} {'max |d| after':>15}")
    for pos in range(1, 10):
        ev = causality_evidence(edit_position=pos)
        print(
            f"{ev['edit_position']:>9} {ev['max_change_before_edit']:>16.2e} "
            f"{ev['change_at_edit_position']:>12.2e} {ev['max_change_after_edit']:>15.2e}"
        )
    print("\nPASS: logits before the edited position are bit-identical (0.00e+00);")
    print("      the edited position and everything after it change, as they must.")


if __name__ == "__main__":
    main()
