"""Phase 3 -- reasoning finetuning.

Shared code, explicit ``--lang`` on every entry point. Exactly like ``model/``,
``train/`` and ``eval/`` in Phase 2: the *code* is shared, the *data*, the
*tokenizer* and the *weights* never are. That is what the brief's independence
requirement covers, and ``eval.loader.load_for_eval`` still refuses to pair a
checkpoint with the wrong language's tokenizer.

Modules
-------
``gen_reasoning``
    Generates the synthetic comparative-reasoning corpus for one language.
``data``
    Encodes those examples, masks the prompt out of the loss, and batches them.
``finetune``
    Finetunes one pretrained checkpoint on its own language's reasoning corpus.
``eval_reasoning``
    Scores a checkpoint on the test slices: greedy exact match and candidate
    likelihood ranking, against the chance and positional baselines.
``attention_compare``
    Measures attention on the same reasoning prompts through the pretrained and
    the finetuned checkpoint, and reports the difference.
"""
