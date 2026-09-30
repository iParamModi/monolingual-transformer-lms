#!/usr/bin/env python
"""Generate the synthetic comparative-reasoning finetuning corpus for one language.

The brief (Phase 3, §3.1) requires a *self-generated* dataset -- programmatic,
so the ground-truth labels are known by construction -- in the target language's
own script and natural phrasing, with train/val/test splits and a documented
story about how train-test leakage is prevented. This script is that dataset.

Design decisions, and why
-------------------------

**Seven template families, two held out.** T0-T2, T4 and T5 are used for
training; T3 (transitivity asked as a relation between the two *non-adjacent*
entities) and T6 (name the *middle* element, which needs the full ordering
rather than an extremum) appear only in ``test_templates.jsonl``. A model that
scores above chance there has generalised the relation, not the surface pattern.

**Three test slices, not one.** ``test_iid`` (train templates, train names),
``test_names`` (train templates, held-out names) and ``test_templates``
(held-out templates *and* held-out names). A single test number cannot separate
"memorised these names in these patterns" from "learned the comparison"; the
gaps between these three can.

**Premise order and premise polarity are randomised.** A chain stated only ever
as "A > B. B > C." makes "the largest is the first name mentioned" a correct
rule for T2 without any comparison at all. Each chain is therefore emitted in
one of four arrangements -- premises in either order, phrased with either the
"big" or the "small" adjective -- so mention position carries no answer.

**Answer position is balanced by construction.** On top of the arrangement
shuffle, each example is drawn against a round-robin target position and
resampled (bounded) until the answer lands there. The always-answer-the-
first-mentioned-entity heuristic therefore scores near chance, which is what
makes the reported accuracy mean anything.

**All entities in one example share grammatical gender.** Hindi and Nepali
adjectives agree with their subject (लंबा/लंबी, अग्लो/अग्ली). If a three-entity
question mixed genders, the adjective in the *question* would leak which entity
is the answer. Same-gender triples keep the question form uninformative and the
text grammatical.

**Entity pools are dimension-appropriate.** People have heights and ages,
objects have weights and prices, vehicles have speeds. Attaching महंगा to a
person's name is not text anyone would write, and the brief asks for natural
phrasing rather than translated templates.

**Digit script is measured, not assumed.** ``--digits auto`` (the default)
counts ASCII against Devanagari digits in that language's own ``train.bin`` and
uses whichever the corpus actually uses. Writing ``२४`` into a corpus that only
ever saw ``24`` would make the numeric templates fail for tokenisation reasons
that have nothing to do with reasoning.

**No BOS.** Pretraining packed documents as ``... doc </s> doc ...`` with no
BOS, so token id 2 was never trained and its embedding is still at
initialisation. Finetuning examples are ``prompt + " " + answer + </s>``. This
file only writes the text; the id 2 decision is enforced in ``finetune/data.py``,
but it is the reason the prompts here carry no lead-in marker.

Usage
-----
::

    # 1. Which digit script does each corpus actually use?
    python -m finetune.gen_reasoning --lang hi --digit-audit
    python -m finetune.gen_reasoning --lang ne --digit-audit

    # 2. Generate (digit script decided by the same audit, automatically)
    python -m finetune.gen_reasoning --lang hi --out-dir hindi/reasoning  --n-train 20000 --seed 1337
    python -m finetune.gen_reasoning --lang ne --out-dir nepali/reasoning --n-train 20000 --seed 1337

Run from the repository root. Output is five JSONL files plus ``stats.json``.
Regenerating with the same seed reproduces byte-identical files -- there are no
timestamps and no set iteration anywhere in this script.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

DEVANAGARI_DIGITS = "०१२३४५६७८९"
ASCII_DIGITS = "0123456789"

#: Directory name for each language code, fixed by Phase 1.
LANG_DIRS = {"hi": "hindi", "ne": "nepali"}

#: Template metadata. ``role`` is which splits a family may appear in;
#: ``answer_is_entity`` is False only for the equality family, whose answer is
#: बराबर rather than one of the named entities.
TEMPLATES: dict[int, dict] = {
    0: {"name": "pair_direct", "n_entities": 2, "numeric": False,
        "role": "train", "answer_is_entity": True},
    1: {"name": "chain_smallest", "n_entities": 3, "numeric": False,
        "role": "train", "answer_is_entity": True},
    2: {"name": "chain_largest", "n_entities": 3, "numeric": False,
        "role": "train", "answer_is_entity": True},
    3: {"name": "chain_pair_relation", "n_entities": 3, "numeric": False,
        "role": "test", "answer_is_entity": True},
    4: {"name": "numeric_extreme", "n_entities": 3, "numeric": True,
        "role": "train", "answer_is_entity": True},
    5: {"name": "numeric_equal", "n_entities": 2, "numeric": True,
        "role": "train", "answer_is_entity": False},
    6: {"name": "chain_middle", "n_entities": 3, "numeric": False,
        "role": "test", "answer_is_entity": True},
}

#: Families used for training (and therefore for val and the two in-pattern test
#: slices). Weights are relative. T5 is kept near 12% so "the answer is always
#: one of the named entities" is not something the model can assume. T0 is the
#: lightest of the rest because it is the family with the smallest space of
#: distinct prompts (one premise over a pair), and over-weighting it would just
#: repeat the same few thousand strings.
TRAIN_TEMPLATE_WEIGHTS: dict[int, int] = {0: 15, 1: 25, 2: 25, 4: 23, 5: 12}

#: Families held out of training entirely.
HELDOUT_TEMPLATE_WEIGHTS: dict[int, int] = {3: 50, 6: 50}

#: Bound on resampling attempts per example (position target + deduplication).
MAX_SAMPLE_TRIES = 200


# --------------------------------------------------------------------------- #
# Language resources
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Dim:
    """One comparison dimension.

    Attributes:
        big: Adjective for the greater pole, keyed by grammatical gender.
        small: Adjective for the lesser pole, keyed by grammatical gender.
        noun: The dimension noun used by the numeric templates (उम्र, तौल, ...).
        noun_gender: Gender of ``noun``; Hindi picks का/की from it.
        unit: Unit written after the value (साल, किलो, रुपये, ...).
        lo: Inclusive lower bound for generated values.
        hi: Inclusive upper bound for generated values.
        pool: Which entity pool this dimension applies to.
    """

    big: dict
    small: dict
    noun: str
    noun_gender: str
    unit: str
    lo: int
    hi: int
    pool: str


# Pools are (split -> pool -> gender -> names). "Gender" is the grammatical
# gender the adjectives agree with: m/f for Hindi throughout, m/f for Nepali
# people and n for Nepali inanimates (which do not inflect).
HI_POOLS: dict[str, dict[str, dict[str, tuple[str, ...]]]] = {
    "train": {
        "people": {
            "m": ("राम", "श्याम", "मोहन", "अनिल", "सुनील", "विकास",
                  "रवि", "संजय", "दीपक", "राजेश", "अमित", "विनोद"),
            "f": ("गीता", "सीता", "रीता", "कविता", "मीरा", "अनीता",
                  "प्रीति", "रेखा", "ममता", "सुनीता", "शारदा", "उर्मिला"),
        },
        "objects": {
            "f": ("किताब", "कुर्सी", "मेज़", "घड़ी", "टोपी", "कमीज़", "थाली", "दरी"),
            "m": ("बैग", "झोला", "जूता", "पर्दा", "तकिया", "गमला"),
        },
        "vehicles": {
            "f": ("गाड़ी", "साइकिल", "बस", "ट्रेन", "नाव", "जीप", "लॉरी", "टैक्सी"),
            "m": ("स्कूटर", "ट्रक", "ट्रैक्टर", "रिक्शा", "तांगा", "टेम्पो"),
        },
    },
    "test": {
        "people": {
            "m": ("अर्जुन", "करण", "रोहित", "मनीष", "नितिन", "पंकज", "आशीष", "गौरव"),
            # रचना, not स्नेहा: स्नेहा literally contains नेहा, and mention order is
            # recovered by substring search. See validate_pools().
            "f": ("नेहा", "पूजा", "दिव्या", "रचना", "वंदना", "किरण", "अंजलि", "ज्योति"),
        },
        "objects": {
            "f": ("छतरी", "चादर", "पेटी", "बाल्टी", "कंघी"),
            "m": ("थैला", "बर्तन", "डिब्बा", "ताला", "कटोरा"),
        },
        "vehicles": {
            "f": ("रेलगाड़ी", "नौका", "मेट्रो", "कार", "वैन"),
            "m": ("ऑटो", "जहाज़", "हेलिकॉप्टर", "ठेला", "टैंकर"),
        },
    },
}

NE_POOLS: dict[str, dict[str, dict[str, tuple[str, ...]]]] = {
    "train": {
        "people": {
            "m": ("राम", "हरि", "कृष्ण", "बिनोद", "प्रकाश", "रमेश",
                  "दीपक", "नारायण", "सुरेश", "गोपाल", "मोहन", "शंकर"),
            "f": ("सीता", "गीता", "सरिता", "अनिता", "कमला", "सुनिता",
                  "शान्ति", "राधा", "पार्वती", "लक्ष्मी", "मीना", "बिन्दु"),
        },
        "objects": {
            "n": ("किताब", "कुर्सी", "टेबल", "घडी", "झोला", "टोपी", "कमिज",
                  "ब्याग", "थाल", "ओछ्यान", "गमला", "ताल्चा", "बाल्टिन", "सिरानी"),
        },
        "vehicles": {
            "n": ("गाडी", "साइकल", "बस", "रेल", "डुङ्गा", "स्कुटर", "ट्रक",
                  "जीप", "ट्र्याक्टर", "रिक्सा", "ट्याक्सी", "भ्यान", "टेम्पो", "एम्बुलेन्स"),
        },
    },
    "test": {
        "people": {
            "m": ("निरज", "सञ्जय", "दिपेश", "सुजन", "बिकास", "अनुप", "प्रदीप", "राजु"),
            "f": ("पूजा", "रञ्जना", "अस्मिता", "सविता", "निर्मला", "सम्झना", "सुष्मा", "मन्जु"),
        },
        "objects": {
            "n": ("छाता", "पर्दा", "थैलो", "दराज", "कचौरा", "चम्चा", "ऐना", "कोट", "जुत्ता", "डब्बा"),
        },
        "vehicles": {
            "n": ("हवाइजहाज", "हेलिकप्टर", "रेलगाडी", "मालगाडी", "बाइक",
                  "माइक्रो", "लरी", "कार", "पानीजहाज", "स्कुटी"),
        },
    },
}

HI_DIMS: tuple[Dim, ...] = (
    Dim(big={"m": "लंबा", "f": "लंबी"}, small={"m": "छोटा", "f": "छोटी"},
        noun="लंबाई", noun_gender="f", unit="सेंटीमीटर", lo=140, hi=190, pool="people"),
    Dim(big={"m": "बड़ा", "f": "बड़ी"}, small={"m": "छोटा", "f": "छोटी"},
        noun="उम्र", noun_gender="f", unit="साल", lo=8, hi=70, pool="people"),
    Dim(big={"m": "भारी", "f": "भारी"}, small={"m": "हल्का", "f": "हल्की"},
        noun="वजन", noun_gender="m", unit="किलो", lo=2, hi=40, pool="objects"),
    Dim(big={"m": "महंगा", "f": "महंगी"}, small={"m": "सस्ता", "f": "सस्ती"},
        noun="कीमत", noun_gender="f", unit="रुपये", lo=50, hi=950, pool="objects"),
    Dim(big={"m": "तेज़", "f": "तेज़"}, small={"m": "धीमा", "f": "धीमी"},
        noun="गति", noun_gender="f", unit="किलोमीटर प्रति घंटा", lo=10, hi=90, pool="vehicles"),
)

NE_DIMS: tuple[Dim, ...] = (
    Dim(big={"m": "अग्लो", "f": "अग्ली", "n": "अग्लो"},
        small={"m": "होचो", "f": "होची", "n": "होचो"},
        noun="उचाइ", noun_gender="n", unit="सेन्टिमिटर", lo=140, hi=190, pool="people"),
    Dim(big={"m": "जेठो", "f": "जेठी", "n": "जेठो"},
        small={"m": "कान्छो", "f": "कान्छी", "n": "कान्छो"},
        noun="उमेर", noun_gender="n", unit="वर्ष", lo=8, hi=70, pool="people"),
    Dim(big={"m": "गह्रौँ", "f": "गह्रौँ", "n": "गह्रौँ"},
        small={"m": "हलुका", "f": "हलुका", "n": "हलुका"},
        noun="तौल", noun_gender="n", unit="किलो", lo=2, hi=40, pool="objects"),
    Dim(big={"m": "महँगो", "f": "महँगी", "n": "महँगो"},
        small={"m": "सस्तो", "f": "सस्ती", "n": "सस्तो"},
        noun="मूल्य", noun_gender="n", unit="रुपैयाँ", lo=50, hi=950, pool="objects"),
    Dim(big={"m": "छिटो", "f": "छिटो", "n": "छिटो"},
        small={"m": "ढिलो", "f": "ढिलो", "n": "ढिलो"},
        noun="गति", noun_gender="n", unit="किलोमिटर प्रति घण्टा", lo=10, hi=90, pool="vehicles"),
)


# --------------------------------------------------------------------------- #
# Grammar
# --------------------------------------------------------------------------- #


class Grammar:
    """Sentence construction for one language.

    Clauses are returned *without* terminal punctuation so the assembler can
    join two of them with और / र for template 3 and terminate the rest with a
    danda. Every method takes the grammatical gender of the entities in the
    example, which is shared across all entities by construction.
    """

    equal_word: str = ""

    def premise(self, subject: str, other: str, adjective: str, gender: str) -> str:
        """Clause asserting ``subject`` exceeds ``other`` on ``adjective``."""
        raise NotImplementedError

    def value_clause(self, entity: str, dim: Dim, value: str) -> str:
        """Clause stating that ``entity`` has ``value`` on ``dim``."""
        raise NotImplementedError

    def q_superlative(self, adjective: str, gender: str, pool: str) -> str:
        """Question: which of them is the most ``adjective``?"""
        raise NotImplementedError

    def q_of_two(self, adjective: str, gender: str, pool: str) -> str:
        """Question: of the two, which is the more ``adjective``?"""
        raise NotImplementedError

    def q_named_pair(self, first: str, second: str, adjective: str, gender: str, pool: str) -> str:
        """Question: between these two named entities, which is more ``adjective``?"""
        raise NotImplementedError

    def q_middle(self, gender: str, pool: str) -> str:
        """Question: which one is in the middle?"""
        raise NotImplementedError

    def join_two(self, first: str, second: str) -> str:
        """Join two premise clauses into one sentence with 'and'."""
        raise NotImplementedError

    def sentence(self, clause: str) -> str:
        """Terminate a clause as a sentence."""
        return clause + "।"


class HindiGrammar(Grammar):
    """Hindi. Adjectives agree with the subject; का/की agrees with the dimension noun."""

    equal_word = "बराबर"

    def _who(self, gender: str, pool: str) -> str:
        """Interrogative pronoun: कौन for people, कौन सा / कौन सी for things."""
        if pool == "people":
            return "कौन"
        return "कौन सी" if gender == "f" else "कौन सा"

    def premise(self, subject: str, other: str, adjective: str, gender: str) -> str:
        return f"{subject}, {other} से {adjective} है"

    def value_clause(self, entity: str, dim: Dim, value: str) -> str:
        possessive = "की" if dim.noun_gender == "f" else "का"
        return f"{entity} {possessive} {dim.noun} {value} {dim.unit} है"

    def q_superlative(self, adjective: str, gender: str, pool: str) -> str:
        return f"सबसे {adjective} {self._who(gender, pool)} है?"

    def q_of_two(self, adjective: str, gender: str, pool: str) -> str:
        return f"दोनों में {adjective} {self._who(gender, pool)} है?"

    def q_named_pair(self, first: str, second: str, adjective: str, gender: str, pool: str) -> str:
        return f"तो {first} और {second} में {adjective} {self._who(gender, pool)} है?"

    def q_middle(self, gender: str, pool: str) -> str:
        return f"बीच में {self._who(gender, pool)} है?"

    def join_two(self, first: str, second: str) -> str:
        return f"{first} और {second}"


class NepaliGrammar(Grammar):
    """Nepali. Adjectives and copula agree with people; inanimates take the -ो form."""

    equal_word = "बराबर"

    def _who(self, pool: str) -> str:
        """Interrogative pronoun: को for people, कुन for things."""
        return "को" if pool == "people" else "कुन"

    def _copula(self, gender: str) -> str:
        """Declarative copula: छिन् for feminine people, छ otherwise."""
        return "छिन्" if gender == "f" else "छ"

    def _q_copula(self, gender: str) -> str:
        """Interrogative copula: हुन् for feminine people, हो otherwise."""
        return "हुन्" if gender == "f" else "हो"

    def premise(self, subject: str, other: str, adjective: str, gender: str) -> str:
        return f"{subject} {other} भन्दा {adjective} {self._copula(gender)}"

    def value_clause(self, entity: str, dim: Dim, value: str) -> str:
        return f"{entity}को {dim.noun} {value} {dim.unit} छ"

    def q_superlative(self, adjective: str, gender: str, pool: str) -> str:
        return f"सबैभन्दा {adjective} {self._who(pool)} {self._q_copula(gender)}?"

    def q_of_two(self, adjective: str, gender: str, pool: str) -> str:
        return f"दुईमध्ये {adjective} {self._who(pool)} {self._q_copula(gender)}?"

    def q_named_pair(self, first: str, second: str, adjective: str, gender: str, pool: str) -> str:
        return (f"त्यसैले {first} र {second} मध्ये {adjective} "
                f"{self._who(pool)} {self._q_copula(gender)}?")

    def q_middle(self, gender: str, pool: str) -> str:
        return f"बीचमा {self._who(pool)} {self._q_copula(gender)}?"

    def join_two(self, first: str, second: str) -> str:
        return f"{first} र {second}"


@dataclass(frozen=True)
class LangSpec:
    """Everything language-specific, bundled."""

    code: str
    dir_name: str
    dims: tuple[Dim, ...]
    pools: dict
    grammar: Grammar


def language_spec(lang: str) -> LangSpec:
    """Return the resource bundle for a language code.

    Args:
        lang: ``hi`` or ``ne``.

    Returns:
        The :class:`LangSpec` for that language.

    Raises:
        ValueError: If the code is not one of the two project languages.
    """
    if lang == "hi":
        return LangSpec("hi", "hindi", HI_DIMS, HI_POOLS, HindiGrammar())
    if lang == "ne":
        return LangSpec("ne", "nepali", NE_DIMS, NE_POOLS, NepaliGrammar())
    raise ValueError(f"lang must be 'hi' or 'ne', got {lang!r}")


# --------------------------------------------------------------------------- #
# Validation of the hand-written resources
# --------------------------------------------------------------------------- #


def validate_pools(spec: LangSpec, min_group: int = 3) -> None:
    """Check the entity pools are usable before a single example is generated.

    Three things must hold, and all three are easy to break by editing the
    tables above:

    1. Train and test names are disjoint -- the whole point of ``test_names``.
    2. No name in a group is a substring of another in the same group. Mention
       order is recovered with ``prompt.index(name)``, which would find the
       wrong position if one name were contained in another.
    3. Every gender group has at least ``min_group`` names, since the
       three-entity templates draw three same-gender entities.

    Args:
        spec: The language bundle to check.
        min_group: Smallest usable gender group.

    Raises:
        ValueError: Naming exactly what is wrong.
    """
    problems: list[str] = []

    for pool_name in sorted(spec.pools["train"]):
        train_names = {n for group in spec.pools["train"][pool_name].values() for n in group}
        test_names = {n for group in spec.pools["test"][pool_name].values() for n in group}
        shared = sorted(train_names & test_names)
        if shared:
            problems.append(f"{pool_name}: names in both train and test pools: {shared}")

    for split in ("train", "test"):
        for pool_name in sorted(spec.pools[split]):
            for gender in sorted(spec.pools[split][pool_name]):
                names = spec.pools[split][pool_name][gender]
                if len(names) < min_group:
                    problems.append(
                        f"{split}/{pool_name}/{gender}: only {len(names)} names, "
                        f"need at least {min_group}"
                    )
                for a in names:
                    for b in names:
                        if a != b and a in b:
                            problems.append(
                                f"{split}/{pool_name}/{gender}: {a!r} is a substring of {b!r}"
                            )

    for dim in spec.dims:
        if dim.pool not in spec.pools["train"]:
            problems.append(f"dimension {dim.noun!r} wants pool {dim.pool!r}, which does not exist")

    if problems:
        raise ValueError("entity pools are not usable:\n  - " + "\n  - ".join(problems))


def validate_tokenization(spec: LangSpec, sp) -> None:
    """Check every hand-written string survives the tokenizer without ``<unk>``.

    A name that tokenises to ``<unk>`` cannot be learned and cannot be scored,
    so it is worth catching here rather than discovering it in the accuracy
    table. Adjectives, dimension nouns and units are checked too.

    Args:
        spec: The language bundle to check.
        sp: A loaded SentencePieceProcessor for the same language.

    Raises:
        ValueError: Listing every offending string.
    """
    unk_id = sp.unk_id()
    offenders: list[str] = []

    strings: list[str] = []
    for split in ("train", "test"):
        for pool_name in sorted(spec.pools[split]):
            for gender in sorted(spec.pools[split][pool_name]):
                strings.extend(spec.pools[split][pool_name][gender])
    for dim in spec.dims:
        strings.extend(sorted(dim.big.values()))
        strings.extend(sorted(dim.small.values()))
        strings.extend([dim.noun, dim.unit])
    strings.append(spec.grammar.equal_word)

    for text in strings:
        if unk_id in sp.encode(text, out_type=int):
            offenders.append(text)

    if offenders:
        raise ValueError(
            "these strings tokenise to <unk> under "
            f"the {spec.code} tokenizer and must be replaced: {sorted(set(offenders))}"
        )


# --------------------------------------------------------------------------- #
# Tokenizer and digit audit
# --------------------------------------------------------------------------- #


def load_meta(spec: LangSpec, repo_root: Path) -> dict:
    """Read the Phase 1 ``meta.json`` for a language.

    That file is the authoritative record of where the tokenizer lives and what
    the special ids are, so nothing here hardcodes a path.

    Args:
        spec: The language bundle.
        repo_root: Repository root.

    Returns:
        The parsed metadata dict.
    """
    path = repo_root / spec.dir_name / "data" / "bin" / "meta.json"
    if not path.exists():
        raise FileNotFoundError(f"Phase 1 metadata not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_tokenizer(spec: LangSpec, repo_root: Path):
    """Load the language's SentencePiece model, as recorded by Phase 1.

    Args:
        spec: The language bundle.
        repo_root: Repository root.

    Returns:
        A ready ``SentencePieceProcessor``.
    """
    import sentencepiece as spm  # imported lazily so --help works without it

    meta = load_meta(spec, repo_root)
    model_path = repo_root / meta["tokenizer_model_relative"]
    if not model_path.exists():
        raise FileNotFoundError(f"tokenizer model not found: {model_path}")
    return spm.SentencePieceProcessor(model_file=str(model_path))


def digit_audit(spec: LangSpec, repo_root: Path, sp) -> dict:
    """Count ASCII against Devanagari digits in the language's training corpus.

    Counting is done over token *pieces* rather than decoded text: each vocab
    id is scored once for how many digits of each kind its piece contains, then
    a single ``bincount`` over ``train.bin`` turns that into corpus totals. That
    is exact and runs in seconds over 480M tokens.

    Args:
        spec: The language bundle.
        repo_root: Repository root.
        sp: The loaded tokenizer for this language.

    Returns:
        Dict with both counts, the resulting share, the recommended script, and
        how each digit glyph tokenises.
    """
    import numpy as np

    meta = load_meta(spec, repo_root)
    bin_path = repo_root / meta["splits"]["train"]["bin_relative"]
    if not bin_path.exists():
        raise FileNotFoundError(
            f"{bin_path} not found -- the digit audit reads the training corpus. "
            f"Pass --digits ascii or --digits deva to skip it."
        )

    vocab = sp.get_piece_size()
    ascii_per_id = np.zeros(vocab, dtype=np.int64)
    deva_per_id = np.zeros(vocab, dtype=np.int64)
    for i in range(vocab):
        piece = sp.id_to_piece(i)
        ascii_per_id[i] = sum(ch in ASCII_DIGITS for ch in piece)
        deva_per_id[i] = sum(ch in DEVANAGARI_DIGITS for ch in piece)

    data = np.memmap(bin_path, dtype=np.uint16, mode="r")
    counts = np.zeros(vocab, dtype=np.int64)
    chunk = 20_000_000
    for start in range(0, len(data), chunk):
        counts += np.bincount(data[start:start + chunk], minlength=vocab).astype(np.int64)

    n_ascii = int((counts * ascii_per_id).sum())
    n_deva = int((counts * deva_per_id).sum())
    total = n_ascii + n_deva

    glyphs = {}
    for label, digits in (("ascii", ASCII_DIGITS), ("deva", DEVANAGARI_DIGITS)):
        for ch in digits:
            ids = sp.encode(ch, out_type=int)
            glyphs[f"{label}:{ch}"] = {"ids": ids, "is_unk": sp.unk_id() in ids}

    return {
        "corpus_tokens": int(len(data)),
        "ascii_digit_chars": n_ascii,
        "devanagari_digit_chars": n_deva,
        "devanagari_share": (n_deva / total) if total else None,
        "recommended": "deva" if n_deva > n_ascii else "ascii",
        "glyph_tokenization": glyphs,
    }


def render_number(value: int, script: str) -> str:
    """Write an integer in the chosen digit script.

    Args:
        value: The number.
        script: ``ascii`` or ``deva``.

    Returns:
        The rendered string.
    """
    text = str(value)
    if script == "ascii":
        return text
    return "".join(DEVANAGARI_DIGITS[int(ch)] for ch in text)


# --------------------------------------------------------------------------- #
# Example construction
# --------------------------------------------------------------------------- #


def draw_values(rng: random.Random, lo: int, hi: int, n: int, min_gap: int = 3) -> list[int]:
    """Draw ``n`` values that are pairwise at least ``min_gap`` apart.

    The gap keeps the task a comparison rather than a test of numeric
    precision: 45 against 47 asks the model to resolve a difference it has no
    reason to represent.

    Args:
        rng: Seeded generator.
        lo: Inclusive lower bound.
        hi: Inclusive upper bound.
        n: How many values.
        min_gap: Minimum pairwise separation.

    Returns:
        A list of ``n`` values in draw order.
    """
    for _ in range(500):
        values = [rng.randint(lo, hi) for _ in range(n)]
        ordered = sorted(values)
        if all(b - a >= min_gap for a, b in zip(ordered, ordered[1:])):
            return values
    raise RuntimeError(f"could not draw {n} values in [{lo}, {hi}] with gap {min_gap}")


def _mention_order(prompt: str, names: list[str]) -> list[str]:
    """Order entity names by where they first appear in the prompt.

    ``str.index`` raises if a name is absent, which is the assertion we want:
    an entity that does not appear in its own prompt is a rendering bug.
    """
    return sorted(names, key=prompt.index)


def sample_example(rng: random.Random, spec: LangSpec, template: int, split_pool: str,
                   script: str) -> dict:
    """Build one reasoning example.

    Args:
        rng: Seeded generator.
        spec: Language bundle.
        template: Template family id (a key of :data:`TEMPLATES`).
        split_pool: ``train`` or ``test`` -- which entity pool to draw from.
        script: Digit script for numeric templates.

    Returns:
        A record with the prompt, the answer, the ground truth it was derived
        from, and the bookkeeping the tests and the evaluator need.
    """
    meta = TEMPLATES[template]
    n_entities = meta["n_entities"]
    grammar = spec.grammar

    dim = rng.choice(list(spec.dims))
    pool = spec.pools[split_pool][dim.pool]
    genders = [g for g in sorted(pool) if len(pool[g]) >= n_entities]
    gender = rng.choice(genders)
    # Drawn in logical order: entities[0] is the greatest on this dimension.
    entities = rng.sample(list(pool[gender]), n_entities)

    big = dim.big[gender]
    small = dim.small[gender]

    values: dict[str, int] | None = None
    order: list[str] | None = None
    pair: list[str] | None = None
    ask: str

    if meta["numeric"]:
        if template == 5:
            # Equality: both entities carry the same value, so neither wins.
            value = rng.randint(dim.lo, dim.hi)
            values = {name: value for name in entities}
            order = None
        else:
            drawn = draw_values(rng, dim.lo, dim.hi, n_entities)
            values = {name: v for name, v in zip(entities, drawn)}
            order = sorted(entities, key=lambda n: -values[n])
        clauses = [grammar.value_clause(name, dim, render_number(values[name], script))
                   for name in entities]
        rng.shuffle(clauses)
        sentences = [grammar.sentence(c) for c in clauses]
        ask = rng.choice(["big", "small"])
        adjective = big if ask == "big" else small

        if template == 5:
            question = grammar.q_of_two(adjective, gender, dim.pool)
            answer = grammar.equal_word
        else:
            question = grammar.q_superlative(adjective, gender, dim.pool)
            answer = order[0] if ask == "big" else order[-1]
        prompt = " ".join(sentences) + " " + question

    else:
        order = list(entities)
        # Premises are stated with either the "big" or the "small" adjective and
        # in either order. Without this, mention position alone answers T1/T2.
        premise_polarity = rng.choice(["big", "small"])
        if premise_polarity == "big":
            links = [(order[i], order[i + 1], big) for i in range(n_entities - 1)]
        else:
            links = [(order[i + 1], order[i], small) for i in range(n_entities - 1)]
        clauses = [grammar.premise(a, b, adj, gender) for a, b, adj in links]
        rng.shuffle(clauses)

        if template == 0:
            ask = rng.choice(["big", "small"])
            adjective = big if ask == "big" else small
            sentences = [grammar.sentence(clauses[0])]
            question = grammar.q_of_two(adjective, gender, dim.pool)
            answer = order[0] if ask == "big" else order[-1]
        elif template in (1, 2):
            ask = "small" if template == 1 else "big"
            adjective = big if ask == "big" else small
            sentences = [grammar.sentence(c) for c in clauses]
            question = grammar.q_superlative(adjective, gender, dim.pool)
            answer = order[0] if ask == "big" else order[-1]
        elif template == 3:
            ask = rng.choice(["big", "small"])
            adjective = big if ask == "big" else small
            sentences = [grammar.sentence(grammar.join_two(clauses[0], clauses[1]))]
            # The asked pair is the non-adjacent one; no premise states it
            # directly, so answering needs transitivity.
            pair = [order[0], order[-1]]
            rng.shuffle(pair)
            question = grammar.q_named_pair(pair[0], pair[1], adjective, gender, dim.pool)
            answer = order[0] if ask == "big" else order[-1]
        elif template == 6:
            ask = "middle"
            sentences = [grammar.sentence(c) for c in clauses]
            question = grammar.q_middle(gender, dim.pool)
            answer = order[1]
        else:
            raise ValueError(f"unhandled template {template}")

        prompt = " ".join(sentences) + " " + question

    mention = _mention_order(prompt, list(entities))
    candidates = list(mention)
    if template == 5:
        candidates = candidates + [grammar.equal_word]

    return {
        "prompt": prompt,
        "answer": answer,
        "template": template,
        "template_name": meta["name"],
        "dim": dim.noun,
        "dim_pool": dim.pool,
        "gender": gender,
        "ask": ask,
        "n_entities": n_entities,
        "entities": mention,
        "order": order,
        "values": values,
        "pair": pair,
        "candidates": candidates,
        "answer_index": candidates.index(answer),
        "answer_position": mention.index(answer) if answer in mention else -1,
    }


def weighted_template(rng: random.Random, weights: dict[int, int]) -> int:
    """Pick a template family according to fixed integer weights."""
    keys = sorted(weights)
    return rng.choices(keys, weights=[weights[k] for k in keys], k=1)[0]


def generate_split(rng: random.Random, spec: LangSpec, *, n: int, weights: dict[int, int],
                   split_pool: str, script: str, blocked: set[str], split_name: str,
                   require_unique: bool) -> list[dict]:
    """Generate one split.

    Three constraints are applied while sampling:

    * **Hard, always** -- a prompt used by an *earlier* split is never emitted.
      This is what makes the splits disjoint by construction rather than by
      filtering afterwards, and it is asserted by the test suite.
    * **Hard for the evaluation splits** -- ``require_unique`` also forbids
      repeats *within* the split, so no test example is scored twice.
    * **Soft** -- each example is drawn against a round-robin target answer
      position and resampled until it matches, falling back to the first
      otherwise-valid draw once the attempt budget runs out. Some families
      cannot place the answer at every position (T6's middle entity is named in
      both premises, so it is never mentioned last); the fallback lets those
      spread evenly over the positions they *can* reach instead of failing.

    ``require_unique`` is deliberately False for ``train``. The T0 family has
    only a few thousand distinct prompts, so a 20,000-example training set must
    repeat some of them; that is ordinary for synthetic data and the exact
    number of distinct prompts is reported in ``stats.json`` rather than hidden.

    Args:
        rng: Seeded generator.
        spec: Language bundle.
        n: Number of examples.
        weights: Template family weights.
        split_pool: ``train`` or ``test`` entity pool.
        script: Digit script.
        blocked: Prompts used by earlier splits. Read here, updated by the
            caller once this split is finished.
        split_name: Used for the record ids.
        require_unique: Forbid repeats within this split.

    Returns:
        The generated records.
    """
    records: list[dict] = []
    seen_here: set[str] = set()
    cycles: dict[int, int] = {}

    for i in range(n):
        template = weighted_template(rng, weights)
        meta = TEMPLATES[template]
        # The equality family answers बराबर, which occupies no mention position,
        # so there is nothing to balance and the only valid target is -1.
        positions = list(range(meta["n_entities"])) if meta["answer_is_entity"] else [-1]
        target = positions[cycles.get(template, 0) % len(positions)]
        cycles[template] = cycles.get(template, 0) + 1

        chosen: dict | None = None
        fallback: dict | None = None
        fallback_fresh = False

        for _ in range(MAX_SAMPLE_TRIES):
            example = sample_example(rng, spec, template, split_pool, script)
            if example["prompt"] in blocked:
                continue
            fresh = example["prompt"] not in seen_here
            if require_unique and not fresh:
                continue
            if fallback is None or (fresh and not fallback_fresh):
                fallback, fallback_fresh = example, fresh
            if example["answer_position"] == target and fresh:
                chosen = example
                break

        chosen = chosen or fallback
        if chosen is None:
            raise RuntimeError(
                f"{split_name}: could not draw a usable prompt for template {template} "
                f"after {MAX_SAMPLE_TRIES} tries -- the entity pool is too small for "
                f"the requested split sizes. Reduce --n-test, or enlarge the pools."
            )

        chosen["id"] = f"{spec.code}-{split_name}-{i:06d}"
        records.append(chosen)
        seen_here.add(chosen["prompt"])

    return records


# --------------------------------------------------------------------------- #
# Statistics and output
# --------------------------------------------------------------------------- #


def split_stats(records: list[dict], sp, max_len: int) -> dict:
    """Summarise one split, and check the hard limits while doing it.

    Args:
        records: The split's records.
        sp: Tokenizer, or ``None`` to skip length and UNK statistics.
        max_len: Length ceiling to assert against.

    Returns:
        A dict of counters and length statistics.
    """
    templates = Counter(r["template"] for r in records)
    dims = Counter(r["dim"] for r in records)
    asks = Counter(r["ask"] for r in records)
    genders = Counter(r["gender"] for r in records)
    answers = Counter(r["answer"] for r in records)

    # Answer position is only comparable within a fixed number of entities:
    # a two-entity example has no third position to land in.
    positions: dict[str, Counter] = {}
    for record in records:
        key = f"{record['n_entities']}_entities"
        positions.setdefault(key, Counter())[str(record["answer_position"])] += 1

    # The number the accuracy table needs a floor from: how well would
    # "always answer the entity mentioned k-th" do on this split?
    scored = [r for r in records if r["answer_position"] >= 0]
    positional_best = max(
        (sum(1 for r in scored if r["answer_position"] == k) / len(scored) for k in range(3)),
        default=0.0,
    )

    stats = {
        "n": len(records),
        "distinct_prompts": len({r["prompt"] for r in records}),
        "positional_heuristic_best": round(positional_best, 4),
        "templates": {str(k): templates[k] for k in sorted(templates)},
        "dims": {k: dims[k] for k in sorted(dims)},
        "ask": {k: asks[k] for k in sorted(asks)},
        "gender": {k: genders[k] for k in sorted(genders)},
        "answer_position": {k: {p: positions[k][p] for p in sorted(positions[k])}
                            for k in sorted(positions)},
        "most_common_answers": [[name, count] for name, count in answers.most_common(5)],
        "distinct_answers": len(answers),
    }

    if sp is not None:
        unk_id = sp.unk_id()
        lengths, unks = [], 0
        for record in records:
            ids = sp.encode(record["prompt"] + " " + record["answer"], out_type=int)
            lengths.append(len(ids) + 1)  # +1 for the </s> that finetuning appends
            unks += sum(1 for i in ids if i == unk_id)
        lengths.sort()
        longest = lengths[-1]
        if longest > max_len:
            raise ValueError(f"an example encodes to {longest} tokens, above the {max_len} cap")
        stats["tokens"] = {
            "mean": round(sum(lengths) / len(lengths), 2),
            "p95": lengths[int(0.95 * (len(lengths) - 1))],
            "max": longest,
        }
        stats["unk_tokens"] = unks

    return stats


def write_jsonl(path: Path, records: list[dict]) -> None:
    """Write records as UTF-8 JSONL with stable key order.

    ``sort_keys`` and ``ensure_ascii=False`` keep the output byte-identical
    across runs and readable in Devanagari.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def pool_summary(spec: LangSpec) -> dict:
    """Record the exact entity pools used, for the report and for the tests."""
    return {
        split: {
            pool: {gender: list(spec.pools[split][pool][gender])
                   for gender in sorted(spec.pools[split][pool])}
            for pool in sorted(spec.pools[split])
        }
        for split in ("train", "test")
    }


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    """Command-line interface."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--lang", required=True, choices=["hi", "ne"])
    parser.add_argument("--out-dir", default=None,
                        help="default: {hindi|nepali}/reasoning")
    parser.add_argument("--repo-root", default=".", help="repository root (default: cwd)")
    parser.add_argument("--n-train", type=int, default=20000)
    parser.add_argument("--n-val", type=int, default=2000)
    parser.add_argument("--n-test", type=int, default=1000,
                        help="per test slice; there are three")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--digits", default="auto", choices=["auto", "ascii", "deva"],
                        help="digit script for numeric templates; 'auto' measures the corpus")
    parser.add_argument("--max-len", type=int, default=128,
                        help="hard ceiling on encoded example length")
    parser.add_argument("--digit-audit", action="store_true",
                        help="report the corpus digit counts and exit")
    parser.add_argument("--no-tokenizer", action="store_true",
                        help="skip UNK and length checks (used by the determinism test)")
    return parser


def main(argv: list[str] | None = None) -> None:
    """Generate the corpus for one language."""
    # Every message this script prints may contain Devanagari, and the default
    # Windows console codepage cannot encode it. Without this the run dies on a
    # print, not on anything that matters.
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):  # pragma: no cover - non-standard stream
        pass

    args = build_parser().parse_args(argv)
    repo_root = Path(args.repo_root).resolve()
    spec = language_spec(args.lang)

    validate_pools(spec)

    sp = None if args.no_tokenizer else load_tokenizer(spec, repo_root)
    if sp is not None:
        validate_tokenization(spec, sp)

    audit = None
    if args.digit_audit or args.digits == "auto":
        if sp is None:
            raise SystemExit("--digits auto needs the tokenizer; pass --digits ascii|deva")
        audit = digit_audit(spec, repo_root, sp)

    if args.digit_audit:
        print(f"digit audit -- {spec.dir_name} train.bin ({audit['corpus_tokens']:,} tokens)")
        print(f"  ASCII digit characters      : {audit['ascii_digit_chars']:,}")
        print(f"  Devanagari digit characters : {audit['devanagari_digit_chars']:,}")
        share = audit["devanagari_share"]
        print(f"  Devanagari share            : {share:.4f}" if share is not None
              else "  Devanagari share            : n/a (no digits found)")
        print(f"  -> recommended digit script : {audit['recommended']}")
        unk_glyphs = sorted(k for k, v in audit["glyph_tokenization"].items() if v["is_unk"])
        print(f"  digit glyphs that tokenise to <unk>: {unk_glyphs or 'none'}")
        return

    script = audit["recommended"] if args.digits == "auto" else args.digits
    out_dir = Path(args.out_dir) if args.out_dir else repo_root / spec.dir_name / "reasoning"

    rng = random.Random(args.seed)
    blocked: set[str] = set()

    # Test slices are generated first so that train and val are forced to avoid
    # their prompts, rather than the other way round: the test sets are the
    # scarce resource and must not be diluted by whatever train happened to use.
    plan = [
        ("test_templates", args.n_test, HELDOUT_TEMPLATE_WEIGHTS, "test", True),
        ("test_names", args.n_test, TRAIN_TEMPLATE_WEIGHTS, "test", True),
        ("test_iid", args.n_test, TRAIN_TEMPLATE_WEIGHTS, "train", True),
        ("val", args.n_val, TRAIN_TEMPLATE_WEIGHTS, "train", True),
        ("train", args.n_train, TRAIN_TEMPLATE_WEIGHTS, "train", False),
    ]

    splits: dict[str, list[dict]] = {}
    for name, count, weights, pool, unique in plan:
        print(f"generating {name}: {count} examples ...")
        splits[name] = generate_split(
            rng, spec, n=count, weights=weights, split_pool=pool,
            script=script, blocked=blocked, split_name=name, require_unique=unique,
        )
        # Only now do this split's prompts become off-limits, so a split is
        # never asked to avoid itself.
        blocked.update(record["prompt"] for record in splits[name])

    for name in ("train", "val", "test_iid", "test_names", "test_templates"):
        write_jsonl(out_dir / f"{name}.jsonl", splits[name])

    stats = {
        "lang": spec.code,
        "language": spec.dir_name,
        "seed": args.seed,
        "digit_script": script,
        "digit_audit": audit,
        "max_len": args.max_len,
        "splits": {name: split_stats(splits[name], sp, args.max_len)
                   for name in ("train", "val", "test_iid", "test_names", "test_templates")},
        "templates": {
            str(k): {
                "name": v["name"],
                "n_entities": v["n_entities"],
                "numeric": v["numeric"],
                "held_out": v["role"] == "test",
            }
            for k, v in sorted(TEMPLATES.items())
        },
        "template_weights": {
            "train_val_iid_names": {str(k): v for k, v in sorted(TRAIN_TEMPLATE_WEIGHTS.items())},
            "test_templates": {str(k): v for k, v in sorted(HELDOUT_TEMPLATE_WEIGHTS.items())},
        },
        "dimensions": [
            {"noun": d.noun, "pool": d.pool, "unit": d.unit, "range": [d.lo, d.hi]}
            for d in spec.dims
        ],
        "pools": pool_summary(spec),
        "leakage_control": {
            "held_out_entity_names": "test_names and test_templates draw only from the test pools",
            "held_out_templates": "T3 and T6 never appear in train, val or test_iid",
            "prompt_disjointness": "every split's prompts are blocked from all later splits",
            "answer_position": "round-robin target position per template, resampled to match",
            "gender": "all entities in an example share grammatical gender, so the "
                      "adjective in the question does not identify the answer",
        },
        "known_confounds": [
            "In a three-entity chain the middle entity is necessarily named in both "
            "premises, so 'the entity mentioned twice' identifies T6's answer without "
            "any ordering. T6 is held out of training, so the model cannot have "
            "learned that shortcut here, but T6 accuracy should not be read as proof "
            "of full ordering on its own.",
            "For the same reason T6's answer is never the last entity mentioned: it is "
            "at mention position 0 or 1, roughly half the time each. A fixed-position "
            "heuristic therefore scores about 50% on T6 alone and in the low forties on "
            "test_templates as a whole, against about 33% elsewhere. The exact figure "
            "per split is recorded as positional_heuristic_best and is the floor the "
            "reasoning accuracy table must be read against.",
            "Numeric values are drawn uniformly in a plausible band and are not "
            "world-accurate; a 30 kg book is possible. The task is comparison, not "
            "world knowledge.",
        ],
    }
    (out_dir / "stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    print(f"\nwrote {out_dir}")
    for name in ("train", "val", "test_iid", "test_names", "test_templates"):
        s = stats["splits"][name]
        tokens = s.get("tokens", {})
        print(f"  {name:<16} {s['n']:>6} examples, {s['distinct_prompts']:>6} distinct"
              + (f", mean {tokens['mean']:.1f} / max {tokens['max']} tokens" if tokens else ""))
    print(f"  digit script: {script}")
    print("\nsample prompts:")
    for name in ("train", "test_templates"):
        for record in splits[name][:2]:
            print(f"  [{name}/T{record['template']}] {record['prompt']}")
            print(f"      -> {record['answer']}")


if __name__ == "__main__":
    sys.exit(main())
