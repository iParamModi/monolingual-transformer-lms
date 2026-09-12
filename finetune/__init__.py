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
"""
