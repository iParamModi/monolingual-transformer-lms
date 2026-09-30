"""Evaluation: language-modelling metrics, generation, attention analysis, plots.

Every function here takes a model and returns data (dicts, arrays, JSON-ready
structures) -- no function prints a final report or hardcodes a file path.
Plotting is likewise separated into eval/plots.py so computation and drawing
never mix, per the brief's design guideline. Phase 3 imports these functions
directly for its pretrained-vs-finetuned attention comparison.
"""
