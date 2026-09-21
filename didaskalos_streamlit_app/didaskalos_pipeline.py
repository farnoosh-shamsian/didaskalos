from __future__ import annotations

import math
import os
import re
import unicodedata
import xml.etree.ElementTree as ET
from collections import Counter
from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import quote

import numpy as np
import pandas as pd
from markdown import markdown as markdown_to_html

import constructions

try:
    from i18n import DEFAULT_LANG, is_rtl, t
except ImportError:  # imported as a package rather than a flat module
    from .i18n import DEFAULT_LANG, is_rtl, t

try:
    from treebank_parsers import parse_agdt_xml, parse_treebank_file
except ImportError:
    from .treebank_parsers import parse_agdt_xml, parse_treebank_file

try:
    from work_catalog import format_citation, tlg_work_key
except ImportError:
    from .work_catalog import format_citation, tlg_work_key

# Back-compat alias for the AGDT parser's historical name.
parse_treebank_xml = parse_agdt_xml


def _force_utf8_stdio() -> None:
    os.environ.setdefault("PYTHONUTF8", "1")
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")

    import sys

    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass


_force_utf8_stdio()


CASE_MAP = {"n": "nominative", "g": "genitive", "d": "dative", "a": "accusative", "v": "vocative"}
TENSE_MAP = {
    "p": "present",
    "i": "imperfect",
    "f": "future",
    "a": "aorist",
    "r": "perfect",
    "l": "pluperfect",
    "t": "future perfect",
}
MOOD_MAP = {"i": "indicative", "s": "subjunctive", "o": "optative", "m": "imperative", "n": "infinitive", "p": "participle"}
VOICE_MAP = {"a": "active", "m": "middle", "p": "passive", "e": "middle/passive"}
GENDER_MAP = {"m": "masculine", "f": "feminine", "n": "neuter"}
# The degree slot only ever carries a comparative or a superlative; a positive
# adjective leaves it empty, so there is no code for it to decode.
DEGREE_MAP = {"c": "comparative", "s": "superlative"}
SIMPLE_POS_LABELS = {
    "d": "adverb",
    "r": "preposition",
    "g": "particle",
    "c": "conjunction",
    "i": "interjection",
}
POS_CATEGORY_MAP = {
    "v": "verb",
    "l": "article",
    "p": "pronoun",
    **SIMPLE_POS_LABELS,
}

# The inflected classes, the only ones a parsing exercise can ask about. Named
# separately from POS_CATEGORY_MAP because that one folds nouns and adjectives
# into a single category, and an answer key has to tell them apart.
POSTAG_POS_NAMES = {"n": "noun", "a": "adjective", "v": "verb", "l": "article", "p": "pronoun"}

# Postag first letters of the content words (noun, adjective, verb, adverb,
# pronoun); everything else counts as a function word.
CONTENT_POS_PREFIXES = ("n", "a", "v", "d", "p")

# Difficulty blends mean content-word rarity, the rarest word, and length.
DIFFICULTY_WEIGHT_MEAN_RARITY = 0.35
DIFFICULTY_WEIGHT_RAREST_WORD = 0.35
DIFFICULTY_WEIGHT_LENGTH = 0.30

# Prefer exercise sentences whose lemmas were mostly introduced already.
KNOWN_LEMMA_COVERAGE_THRESHOLD = 0.70
KNOWN_FUNCTION_LEMMA_SEED_COUNT = 50

# A reading passage runs whole citation units up to a word budget, so a cut never
# lands inside a chapter, section or verse. The texts already mark their own
# divisions in subdoc, and those are where the subject changes: a Herodotus
# chapter, one Aesop fable, seven Homer verse-sentences. word_count counts tokens
# including punctuation, so the budget is roughly ninety words of running text.
PASSAGE_WORD_BUDGET = 100
PASSAGE_MIN_WORDS = 40
PASSAGE_MAX_WORDS = 160
PASSAGE_COUNT = 25

# One long work would otherwise take every slot, since it supplies most of the
# candidates and epic repeats its vocabulary: unguarded, the Iliad took 1,187 of
# 1,322 candidates in a four-work build. Ignored when only one work was picked.
PASSAGE_MAX_WORK_SHARE = 0.4


GREEK_MARK_RE = re.compile(r"[\u0370-\u03FF\u1F00-\u1FFF]")


def clean_text(element):
    return " ".join(element.itertext()).split() if element is not None else []


def list_treebanks(folder: str | Path) -> pd.DataFrame:
    folder = Path(folder)
    rows = []

    if not folder.exists():
        return pd.DataFrame(columns=["file", "title", "author"])

    for xml_file in sorted(folder.glob("*.xml")):
        root = ET.parse(xml_file).getroot()
        title = " ".join(clean_text(root.find(".//title"))) or None
        author = " ".join(clean_text(root.find(".//author"))) or None

        rows.append(
            {
                "file": xml_file.name,
                "title": title,
                "author": author,
            }
        )

    return pd.DataFrame(rows)


def _decode(code_map, ch: str) -> str:
    return "unknown" if ch == "-" else code_map.get(ch, ch)


def parse_postag(postag: str) -> str:
    if not isinstance(postag, str) or not postag:
        return "NA"

    pos = postag[0]

    if pos in {"n", "a"} and len(postag) > 7:
        return _decode(CASE_MAP, postag[7])

    if pos == "v" and len(postag) > 5:
        return ", ".join(
            [
                _decode(TENSE_MAP, postag[3]),
                _decode(MOOD_MAP, postag[4]),
                _decode(VOICE_MAP, postag[5]),
            ]
        )

    if pos == "l":
        return "article"

    if pos == "p":
        return "pronoun"

    if pos in SIMPLE_POS_LABELS:
        return SIMPLE_POS_LABELS[pos]

    return "NA"


def parse_pos_category(postag: str) -> str:
    if not isinstance(postag, str) or not postag:
        return "other"
    return POS_CATEGORY_MAP.get(postag[0], "noun/adjective" if postag[0] in {"n", "a"} else "other")


def normalize_frequency_row_name(label: str) -> str:
    if not isinstance(label, str):
        return label

    normalized = label.strip().lower()
    normalized = normalized.replace(", ", "_").replace(",", "_")
    normalized = normalized.replace("/", "_").replace(" ", "_")
    normalized = normalized.replace("(", "").replace(")", "")
    normalized = re.sub(r"_+", "_", normalized).strip("_")
    return normalized


# Called millions of times per build over a few thousand distinct lemmas, so
# memoizing is the biggest CPU win in textbook generation.
@lru_cache(maxsize=None)
def _normalize_greek_lemma_cached(lemma: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", lemma.lower().strip()) if unicodedata.category(c) != "Mn")


def normalize_greek_lemma(lemma: str) -> str:
    if not isinstance(lemma, str):
        return ""
    return _normalize_greek_lemma_cached(lemma)


# Perseus homonym numbers (λέγω3, θύω1) hide the ending that identifies a verb's
# conjugation class, so they are stripped before any ending is inspected.
_HOMONYM_DIGITS_RE = re.compile(r"\d+$")


@lru_cache(maxsize=None)
def lemma_conjugation_key(lemma: str) -> str:
    return _HOMONYM_DIGITS_RE.sub("", normalize_greek_lemma(lemma))


def lemma_headword(lemma: str) -> str:
    # The lemma as a dictionary prints it: the homonym digit is an internal key,
    # and Logeion answers a request for "λέγω1" with "Could not find λέγω1", so it
    # comes off before a lemma is shown or linked. Accents stay, unlike
    # lemma_conjugation_key, because a headword without them is not lookup-able.
    if not isinstance(lemma, str):
        return ""
    return _HOMONYM_DIGITS_RE.sub("", lemma.strip())


IRREGULAR_VERB_BUCKET = "irregular"


def parse_verb_subcategory(lemma: str, postag: str | None = None) -> str:
    if postag and not str(postag).startswith("v"):
        return ""

    lemma_n = lemma_conjugation_key(lemma)
    if not lemma_n:
        return ""
    # Deponents go by conjugation class: thematic -ομαι follows the -ω paradigms,
    # athematic -μαι the -μι ones. Deponency itself is carried by
    # is_deponent_lemma, not by a bucket.
    if lemma_n.endswith("ομαι"):
        return "w"
    if lemma_n.endswith("μαι") or lemma_n.endswith("μι"):
        return "mi"
    if lemma_n.endswith("ω"):
        return "w"
    return IRREGULAR_VERB_BUCKET


def is_deponent_lemma(lemma: str) -> bool:
    # Middle-only ("deponent") verb: the dictionary form ends in -μαι.
    return lemma_conjugation_key(lemma).endswith("μαι")


# First and second person pronouns, accents off. They inflect for case and
# number only.
GENDERLESS_PRONOUN_LEMMAS = frozenset({"εγω", "συ", "ημεις", "υμεις", "νω", "σφω"})


def is_genderless_pronoun_lemma(lemma: str) -> bool:
    if not isinstance(lemma, str) or not lemma:
        return False
    return normalize_greek_lemma(lemma_headword(lemma)) in GENDERLESS_PRONOUN_LEMMAS


# The particles, accents off. A closed class the treebanks disagree about:
# Perseus tags most of them "g", Gorman files them with the adverbs, so γάρ and
# μέν arrive looking like content words. Particle-hood is a property of the word
# rather than of the slot it fills, so the book carries the list itself. Held to
# lemmas with no content-word homograph once the accents come off -- μήν the
# particle and μήν "month" normalize alike, so neither is here.
PARTICLE_LEMMAS = frozenset(
    {"δε", "γαρ", "μεν", "τε", "ουν", "αν", "δη", "γε", "αρα", "μεντοι", "καιτοι", "τοινυν"}
)


def is_particle_lemma(lemma: str) -> bool:
    if not isinstance(lemma, str) or not lemma:
        return False
    return normalize_greek_lemma(lemma_headword(lemma)) in PARTICLE_LEMMAS


@lru_cache(maxsize=None)
def _is_greek_lemma_cached(lemma: str) -> bool:
    return bool(GREEK_MARK_RE.search(lemma))


def is_greek_lemma(lemma: str) -> bool:
    return isinstance(lemma, str) and _is_greek_lemma_cached(lemma)


# AGDT 9-position postag indices.
POSTAG_PERSON_INDEX = 1
POSTAG_NUMBER_INDEX = 2
POSTAG_TENSE_INDEX = 3
POSTAG_MOOD_INDEX = 4
POSTAG_VOICE_INDEX = 5
POSTAG_GENDER_INDEX = 6
POSTAG_CASE_INDEX = 7
POSTAG_DEGREE_INDEX = 8

# The label drives the lesson filename via normalize_frequency_row_name, so it
# must match the module file names in lessons/en/. The code is only an internal
# key, never exported: part of speech, declension, then gender for the first two
# declensions and stem type for the third.
NOUN_DECLENSION_LABELS = {
    "noun-1-fem": "first declension feminine nouns",
    "noun-1-masc": "first declension masculine nouns",
    "noun-2-masc": "second declension masculine nouns",
    "noun-2-neut": "second declension neuter nouns",
    # Labial and velar stems share a lesson: both keep the stop visible in the
    # nominative (-ψ, -ξ) and behave alike. Dentals drop it, so they get their own.
    "noun-3-labial-velar": "third declension labial and velar stem nouns",
    "noun-3-dental": "third declension dental stem nouns",
    "noun-3-iota-ups": "third declension iota upsilon stem nouns",
    "noun-3-nasal-liq": "third declension nasal liquid stem nouns",
    # noun-3-other and adj-3-two-end are residual buckets, but named for what is
    # in them: frequency order can put one first, and "Other Adjectives" is no
    # title for the first adjective lesson a learner meets. The code says "other"
    # where the label says "sigma stem" because the bucket also holds the -ευς,
    # -αυς and -ω stems. See also LESSON_PREREQUISITE_KINDS.
    "noun-3-other": "sigma stem and irregular nouns",
}

ADJECTIVE_DECLENSION_LABELS = {
    "adj-1-2": "first second declension adjectives",
    "adj-3-three-end": "third declension adjectives",
    "adj-3-two-end": "two ending and irregular adjectives",
}

DECLENSION_LABELS = {**NOUN_DECLENSION_LABELS, **ADJECTIVE_DECLENSION_LABELS}

# Keyed on _classification_key output. Only lemmas whose nominative-singular
# ending points to the wrong class need listing.
IRREGULAR_NOUN_LEXICON = {
    "γυνη": "noun-3-labial-velar",  # γυναικός: velar stem despite ending in -η
    "παισ": "noun-3-dental",  # παιδός: dental stem despite ending in -ις
    "ελπισ": "noun-3-dental",  # ἐλπίδος
    "χαρισ": "noun-3-dental",  # χάριτος
    "ορνισ": "noun-3-dental",  # ὄρνιθος
    "ερισ": "noun-3-dental",  # ἔριδος
    "κλεισ": "noun-3-dental",  # κλειδός
    "νυξ": "noun-3-dental",  # νυκτός: the -ξ writes κτ + ς, the stem is dental
    "εισ": "noun-3-nasal-liq",  # ἑνός: the numeral is tagged a noun, stem ἑν-
    "νουσ": "noun-2-masc",  # second declension contract
    "πλουσ": "noun-2-masc",
    "ζευσ": "noun-3-other",
    "γραυσ": "noun-3-other",
    "γηρασ": "noun-3-other",
    "κερασ": "noun-3-other",
    "τερασ": "noun-3-other",
    "κρεασ": "noun-3-other",
    "υδωρ": "noun-3-nasal-liq",
}

IRREGULAR_ADJECTIVE_LEXICON = {
    "πολυσ": "adj-3-two-end",  # mixed 2nd/3rd declension paradigm
    "μεγασ": "adj-3-two-end",  # mixed 2nd/3rd declension paradigm
}


def _classification_key(text: str) -> str:
    return lemma_conjugation_key(text).replace("ς", "σ")  # final sigma -> sigma


def _genitive_singular_signal(forms: list[str]) -> str | None:
    # Vote on the declension from the attested genitive singulars: "d12" (-ου),
    # "d1" (-ης/-ας), "d3i" (-εως), "d3s" (-ους), "d3" (-ος), or None.
    counts: Counter[str] = Counter()
    for form in forms:
        key = _classification_key(form)
        if key.endswith("εωσ"):
            counts["d3i"] += 1
        elif key.endswith("ουσ"):
            counts["d3s"] += 1
        elif key.endswith("οσ"):
            counts["d3"] += 1
        elif key.endswith("ου"):
            counts["d12"] += 1
        elif key.endswith(("ησ", "ασ")):
            counts["d1"] += 1
    if not counts:
        return None
    return counts.most_common(1)[0][0]


STEM_CONSONANT_CLASSES = {
    **{letter: "labial-velar" for letter in "πβφκγχ"},
    **{letter: "dental" for letter in "τδθ"},
    **{letter: "nasal-liquid" for letter in "νρλ"},
}

CONSONANT_STEM_CODES = {
    "labial-velar": "noun-3-labial-velar",
    "dental": "noun-3-dental",
    "nasal-liquid": "noun-3-nasal-liq",
}


def _consonant_stem_signal(forms: list[str]) -> str | None:
    # Vote on the stem class from the genitive singulars in -ος: the letter
    # before that ending is the stem's real final consonant, which the
    # nominative usually hides. νυκτός gives τ, so νύξ is dental despite the -ξ,
    # and χειρός gives ρ, so χείρ is liquid despite failing the -ηρ/-ωρ test.
    counts: Counter[str] = Counter()
    for form in forms:
        key = _classification_key(form)
        if not key.endswith("οσ") or len(key) < 3:
            continue
        stem_class = STEM_CONSONANT_CLASSES.get(key[-3])
        if stem_class:
            counts[stem_class] += 1
    if not counts:
        return None
    return counts.most_common(1)[0][0]


def _consonant_stem_class(stem_signal: str | None) -> str:
    # Dental is the fallback: it is much the larger class, mostly on the strength
    # of the -μα, -ματος neuters.
    return CONSONANT_STEM_CODES.get(str(stem_signal), "noun-3-dental")


def classify_noun_declension(
    lemma: str,
    gender: str = "-",
    genitive_signal: str | None = None,
    stem_signal: str | None = None,
) -> str:
    # gender is the AGDT postag gender character, ideally the lemma's majority
    # gender across the corpus; genitive_signal comes from
    # _genitive_singular_signal and stem_signal from _consonant_stem_signal.
    key = _classification_key(lemma)
    if not key:
        return "noun-3-other"
    if key in IRREGULAR_NOUN_LEXICON:
        return IRREGULAR_NOUN_LEXICON[key]

    third_declension_evidence = genitive_signal in {"d3", "d3s", "d3i"}

    # Unambiguously third-declension nominative endings.
    if key.endswith(("ευσ", "αυσ", "ουσ", "ω")):
        return "noun-3-other"  # βασιλεύς, ναῦς, βοῦς, πειθώ
    if key.endswith(("ην", "ων", "ηρ", "ωρ")):
        return "noun-3-nasal-liq"  # ποιμήν, δαίμων, πατήρ, ῥήτωρ
    if key.endswith("ισ"):
        # πόλις (-εως, iota stem) vs. ἐλπίς (-ίδος, dental stem).
        if genitive_signal == "d3":
            return "noun-3-dental"  # -ιδος, -ιτος, -ιθος are all dental
        return "noun-3-iota-ups"
    if key.endswith(("υσ", "υ")):
        return "noun-3-iota-ups"  # ἰχθύς, ἄστυ

    # -μα, -ματος neuters are dental stems: σῶμα, πρᾶγμα.
    if gender == "n" and key.endswith("μα"):
        return "noun-3-dental"

    if gender == "f" and key.endswith(("α", "η")):
        return _consonant_stem_class(stem_signal) if third_declension_evidence else "noun-1-fem"
    if gender == "m" and key.endswith(("ασ", "ησ")):
        # πολίτης (-ου, 1st decl) vs. Σωκράτης (-ους, sigma stem) vs. γίγας (-αντος).
        if genitive_signal == "d3s":
            return "noun-3-other"
        if genitive_signal == "d3":
            return _consonant_stem_class(stem_signal)
        return "noun-1-masc"

    if key.endswith("οσ"):
        if gender == "n":
            return "noun-3-other"  # γένος, τεῖχος: sigma-stem neuters
        if third_declension_evidence:
            return _consonant_stem_class(stem_signal)
        return "noun-2-masc"  # masc λόγος (rare feminines like ὁδός also land here)
    if gender == "n" and key.endswith("ον"):
        return "noun-2-neut"

    # Remaining lemmas ending in a consonant: φύλαξ, νύξ, Ἑλλάς, χείρ, ...
    if key.endswith(("ρ", "ν")):
        return "noun-3-nasal-liq"  # χείρ, πῦρ: the -ηρ/-ωρ test above misses these
    if key.endswith(("ξ", "ψ", "σ")):
        if stem_signal:
            return _consonant_stem_class(stem_signal)
        # No genitive attested, so fall back to the nominative: the stop that
        # survives in -ξ/-ψ is the stem's own, while -ς has swallowed a dental.
        return "noun-3-labial-velar" if key.endswith(("ξ", "ψ")) else "noun-3-dental"

    return "noun-3-other"


def classify_adjective_declension(lemma: str) -> str:
    key = _classification_key(lemma)
    if not key:
        return "adj-3-two-end"
    if key in IRREGULAR_ADJECTIVE_LEXICON:
        return IRREGULAR_ADJECTIVE_LEXICON[key]
    if key.endswith(("οσ", "ουσ")):
        return "adj-1-2"  # ἀγαθός, δίκαιος, contract χρυσοῦς
    if key.endswith(("υσ", "εισ", "ασ")):
        return "adj-3-three-end"  # three-ending 3rd decl: ταχύς, χαρίεις, πᾶς, μέλας
    return "adj-3-two-end"  # two-ending 3rd decl (-ης, -ων), comparatives, irregulars


def _noun_rows_with_keys(combined_df: pd.DataFrame, key_func=None) -> pd.DataFrame:
    postag = combined_df["postag"].astype(str)
    noun_mask = postag.str.startswith("n") & combined_df["lemma"].apply(is_greek_lemma)
    rows = combined_df.loc[noun_mask, ["lemma", "form", "postag"]].copy()
    if rows.empty:
        return rows.assign(key="", gender="")
    rows["key"] = rows["lemma"].apply(key_func or _classification_key)
    rows["gender"] = rows["postag"].astype(str).str.slice(POSTAG_GENDER_INDEX, POSTAG_GENDER_INDEX + 1)
    return rows


def _noun_lemma_signals(noun_rows: pd.DataFrame) -> tuple[dict[str, str], dict[str, list[str]]]:
    # Majority gender and the attested genitive singulars, per lemma key. The
    # declension classifier votes on the ending; the dictionary citation line
    # prints the form itself. Both want the same two groupbys, so they share them.
    gendered = noun_rows[noun_rows["gender"].isin(["m", "f", "n"])]
    majority_gender = (
        gendered.groupby("key")["gender"].agg(lambda genders: genders.value_counts().idxmax()).to_dict()
        if not gendered.empty
        else {}
    )

    postag = noun_rows["postag"].astype(str)
    genitive_singular_mask = (
        postag.str.slice(POSTAG_CASE_INDEX, POSTAG_CASE_INDEX + 1).eq("g")
        & postag.str.slice(POSTAG_NUMBER_INDEX, POSTAG_NUMBER_INDEX + 1).eq("s")
    )
    genitive_forms = {
        key: group["form"].astype(str).tolist()
        for key, group in noun_rows[genitive_singular_mask].groupby("key")
    }
    return majority_gender, genitive_forms


GENDER_ARTICLES = {"m": "ὁ", "f": "ἡ", "n": "τό"}

# How one-gendered a lemma's noun-tagged occurrences must be before the citation
# line names an article for it. Below this it is a word inflecting for all three
# genders that the tagger happened to call a noun -- μηδείς, πᾶς, τοιοῦτος --
# and a majority vote would print "μηδείς, τό".
CITATION_GENDER_MAJORITY = 0.9


def _mainly_noun_keys(combined_df: pd.DataFrame, noun_rows: pd.DataFrame) -> set[str]:
    # Lemmas the corpus mostly tags as nouns. μηδείς and πᾶς are tagged noun in
    # the odd substantive passage, and a citation line built from those few rows
    # gives them a gender and a genitive they do not have as headwords.
    if noun_rows.empty:
        return set()
    all_keys = combined_df["lemma"].map(lemma_headword)
    totals = all_keys.value_counts()
    noun_counts = noun_rows["key"].value_counts()
    shares = noun_counts / totals.reindex(noun_counts.index)
    return set(shares[shares >= 0.5].index)


def _single_gender_noun_keys(noun_rows: pd.DataFrame) -> set[str]:
    gendered = noun_rows[noun_rows["gender"].isin(["m", "f", "n"])]
    if gendered.empty:
        return set()
    shares = gendered.groupby("key")["gender"].agg(lambda genders: genders.value_counts(normalize=True).max())
    return set(shares[shares >= CITATION_GENDER_MAJORITY].index)

_GRAVE, _ACUTE = "̀", "́"


def to_citation_accent(form: str) -> str:
    # A final acute turns grave in running text, so the corpus attests ἀνδρὸς
    # where a dictionary prints ἀνδρός. Put the acute back for the citation line.
    if not form or _GRAVE not in unicodedata.normalize("NFD", form):
        return form
    decomposed = unicodedata.normalize("NFD", form).replace(_GRAVE, _ACUTE)
    return unicodedata.normalize("NFC", decomposed)


def build_lemma_citation_index(combined_df: pd.DataFrame) -> dict[str, dict[str, str]]:
    # Dictionary-style citation data per noun lemma: the article its majority
    # gender implies, and the genitive singular the corpus actually attests. A
    # lemma with no attested genitive gets none, and the entry prints short --
    # better a bare headword than an ending we guessed.
    #
    # Keyed on the accented headword, not the classifier's key: that one strips
    # accents and breathings, which would let the preposition εἰς collect the
    # article of the numeral εἷς.
    if combined_df is None or combined_df.empty or "postag" not in combined_df.columns:
        return {}

    noun_rows = _noun_rows_with_keys(combined_df, key_func=lemma_headword)
    if noun_rows.empty:
        return {}

    majority_gender, genitive_forms = _noun_lemma_signals(noun_rows)
    noun_keys = _mainly_noun_keys(combined_df, noun_rows)
    one_gender_keys = _single_gender_noun_keys(noun_rows) & noun_keys

    citation_index: dict[str, dict[str, str]] = {}
    for key in set(majority_gender) | set(genitive_forms):
        forms = genitive_forms.get(key, [])
        article = GENDER_ARTICLES.get(majority_gender.get(key, ""), "") if key in one_gender_keys else ""
        citation_index[key] = {
            "genitive": to_citation_accent(Counter(forms).most_common(1)[0][0]) if forms and key in noun_keys else "",
            "article": article,
        }
    return citation_index


def add_declension_features(combined_df: pd.DataFrame) -> pd.DataFrame:
    # Add declension_code / declension_label for noun and adjective rows.
    out = combined_df.copy()
    out["declension_code"] = ""
    out["declension_label"] = ""

    if out.empty or "postag" not in out.columns or "lemma" not in out.columns:
        return out

    postag = out["postag"].astype(str)
    greek_lemma_mask = out["lemma"].apply(is_greek_lemma)
    noun_mask = postag.str.startswith("n") & greek_lemma_mask
    adjective_mask = postag.str.startswith("a") & greek_lemma_mask

    if noun_mask.any():
        noun_rows = out.loc[noun_mask, ["lemma", "form", "postag"]].copy()
        noun_rows["key"] = noun_rows["lemma"].apply(_classification_key)
        noun_rows["gender"] = noun_rows["postag"].astype(str).str.slice(
            POSTAG_GENDER_INDEX, POSTAG_GENDER_INDEX + 1
        )

        majority_gender, genitive_forms = _noun_lemma_signals(noun_rows)
        genitive_signals = {key: _genitive_singular_signal(forms) for key, forms in genitive_forms.items()}
        stem_signals = {key: _consonant_stem_signal(forms) for key, forms in genitive_forms.items()}

        code_by_key = {
            row["key"]: classify_noun_declension(
                row["lemma"],
                majority_gender.get(row["key"], "-"),
                genitive_signals.get(row["key"]),
                stem_signals.get(row["key"]),
            )
            for _, row in noun_rows.drop_duplicates("key").iterrows()
        }
        out.loc[noun_mask, "declension_code"] = noun_rows["key"].map(code_by_key)

    if adjective_mask.any():
        adjective_code_cache: dict[str, str] = {}

        def adjective_code(lemma: str) -> str:
            key = _classification_key(lemma)
            if key not in adjective_code_cache:
                adjective_code_cache[key] = classify_adjective_declension(lemma)
            return adjective_code_cache[key]

        out.loc[adjective_mask, "declension_code"] = out.loc[adjective_mask, "lemma"].map(adjective_code)

    out["declension_label"] = out["declension_code"].map(DECLENSION_LABELS).fillna("")
    return out


def apply_declension_syllabus(combined_df: pd.DataFrame) -> pd.DataFrame:
    # Swap the case-based syllabus of noun/adjective rows for declension labels.
    out = combined_df if "declension_label" in combined_df.columns else add_declension_features(combined_df)
    out = out.copy()

    noun_adjective_mask = out["postag"].astype(str).str.startswith(("n", "a"))
    has_label = out["declension_label"].astype(str).ne("")

    out.loc[noun_adjective_mask, "syllabus"] = "NA"
    out.loc[noun_adjective_mask & has_label, "syllabus"] = out.loc[
        noun_adjective_mask & has_label, "declension_label"
    ]
    return out


def build_combined_df(
    folder: str | Path,
    selected_files: list[str],
    syllabus_mode: str = "case",
    formats: Mapping[str, str | None] | None = None,
) -> pd.DataFrame:
    # formats maps a filename to its registry format; a missing entry lets the
    # dispatcher auto-detect, which keeps uploads and ad-hoc URLs working.
    formats = formats or {}
    all_dfs = []

    for filename in selected_files:
        file_path = Path(folder) / filename
        df = parse_treebank_file(file_path, formats.get(filename))
        df["file"] = os.path.basename(file_path)
        all_dfs.append(df)

    if not all_dfs:
        return pd.DataFrame()

    combined_df = pd.concat(all_dfs, ignore_index=True)

    # Editors differ on whether a breathing is its own codepoint, so the same
    # word arrives as both ὁ and ο +  ̔ . Left alone they count as two words and
    # print as two vocabulary entries.
    for column in ("lemma", "form"):
        if column in combined_df.columns:
            combined_df[column] = combined_df[column].map(
                lambda value: unicodedata.normalize("NFC", value) if isinstance(value, str) else value
            )

    combined_df["syllabus"] = combined_df["postag"].apply(parse_postag)
    combined_df["pos_category"] = combined_df["postag"].apply(parse_pos_category)
    combined_df["verb_subcategory"] = combined_df.apply(
        lambda row: parse_verb_subcategory(row["lemma"], row["postag"]) if row["pos_category"] == "verb" else "",
        axis=1,
    )
    combined_df["is_deponent"] = (combined_df["pos_category"] == "verb") & combined_df["lemma"].map(is_deponent_lemma)

    if syllabus_mode == "declension":
        combined_df = add_declension_features(combined_df)
        combined_df = apply_declension_syllabus(combined_df)
        # Both are spent once the label is in "syllabus": the code is an internal
        # key and the label now duplicates the syllabus text. Dropping them keeps
        # the exported columns the same in either syllabus mode.
        combined_df = combined_df.drop(columns=["declension_code", "declension_label"])

    # Object columns dominate memory (~220 MB for 258k tokens) and got the
    # container OOM-killed; these few repeat a small vocabulary, so categoricals
    # roughly halve the frame. The rest stay objects: syllabus/pos_category in
    # particular are reassigned and grouped on in build_frequency_syllabus.
    for column in ("document_id", "subdoc", "postag", "relation", "head",
                   "file", "verb_subcategory"):
        if column in combined_df.columns:
            combined_df[column] = combined_df[column].astype("category")

    return combined_df


IRREGULAR_LESSON_LABEL = "irregular verbs"
IRREGULAR_LESSON_FILENAME = "irregular_verbs.md"

TENSE_NAMES = frozenset(TENSE_MAP.values())
MOOD_NAMES = frozenset(MOOD_MAP.values())
VOICE_NAMES = frozenset(VOICE_MAP.values())
CASE_NAMES = frozenset(CASE_MAP.values())
DECLENSION_LABEL_NAMES = frozenset(DECLENSION_LABELS.values())


def is_decodable_verb_label(label: str) -> bool:
    # True when a verb row names a real tense, mood and voice. parse_postag lets
    # unmarked and undefined slots through, so rows like "present, unknown,
    # active" reach the syllabus looking like paradigms. Concept lessons, which
    # are not tense/mood/voice rows at all, are exempt.
    text = str(label)
    if text in {IRREGULAR_LESSON_LABEL, DEPONENT_LESSON_LABEL}:
        return True
    base_label, _ = split_syllabus_label_and_bucket(text)
    parts = [part.strip() for part in base_label.split(",")]
    if len(parts) != 3:
        return False
    tense, mood, voice = parts
    return tense in TENSE_NAMES and mood in MOOD_NAMES and voice in VOICE_NAMES


def is_decodable_nominal_label(label: str) -> bool:
    # True when a noun/adjective row names a real case or declension class. An
    # unmarked or absent case slot produces rows no lesson can serve. Both
    # syllabus modes are accepted.
    return str(label) in CASE_NAMES or str(label) in DECLENSION_LABEL_NAMES

# Tenses where the middle and the passive are the same form, so one lesson
# teaches both. The aorist and future build the passive on a separate -θη- stem
# and need their own lessons.
VOICE_SYNCRETIC_TENSES = ("present", "imperfect", "perfect", "pluperfect", "future perfect")

# ...except these two, which have a single-voice module of their own and must
# not be folded into the middle/passive one.
UNMERGED_SINGLE_VOICE_LABELS = frozenset(
    {"present, indicative, middle (w)", "imperfect, indicative, middle (w)"}
)


def _syncretic_voice_merges() -> dict[str, str]:
    merges: dict[str, str] = {}
    for tense in VOICE_SYNCRETIC_TENSES:
        for mood in MOOD_MAP.values():
            for bucket in ("", " (w)", " (mi)"):
                target = f"{tense}, {mood}, middle/passive{bucket}"
                for voice in ("middle", "passive"):
                    source = f"{tense}, {mood}, {voice}{bucket}"
                    if source not in UNMERGED_SINGLE_VOICE_LABELS:
                        merges[source] = target
    return merges


# Syllabus rows that share another row's lesson. Merged at row level only: the
# token-level "syllabus" value is untouched, so answer keys and the exported rows
# still name the real case and voice. The vocative repeats the nominative bar a
# few singular endings, so nominative.md covers both and there is no vocative.md.
MERGED_SYLLABUS_LABELS = {"vocative": "nominative", **_syncretic_voice_merges()}


def build_frequency_syllabus(combined_df: pd.DataFrame) -> pd.DataFrame:
    if combined_df is None or combined_df.empty:
        return pd.DataFrame(columns=["syllabus", "pos_category", "frequency", "syllabus_normalized"])

    verb_mask = (
        combined_df["pos_category"].eq("verb")
        & combined_df["verb_subcategory"].notna()
        & combined_df["verb_subcategory"].astype(str).ne("")
    )

    syllabus_with_verb_bucket = combined_df["syllabus"].where(
        ~verb_mask,
        combined_df["syllabus"].astype(str) + " (" + combined_df["verb_subcategory"].astype(str) + ")",
    )

    # The irregular verbs are not a conjugation class but ~150 lemmas each
    # defective in its own way. Split by tense/mood/voice they made 57 thin
    # lessons, so they collapse into one concept lesson, as deponency does.
    irregular_mask = verb_mask & combined_df["verb_subcategory"].astype(str).eq(IRREGULAR_VERB_BUCKET)
    syllabus_with_verb_bucket = syllabus_with_verb_bucket.where(~irregular_mask, IRREGULAR_LESSON_LABEL)

    frequency_syllabus = (
        pd.DataFrame(
            {
                "syllabus": syllabus_with_verb_bucket.replace(MERGED_SYLLABUS_LABELS),
                "pos_category": combined_df["pos_category"],
            }
        )
        .groupby(["syllabus", "pos_category"], dropna=False)
        .size()
        .reset_index(name="frequency")
        .sort_values("frequency", ascending=False, ignore_index=True)
    )
    frequency_syllabus["syllabus_normalized"] = frequency_syllabus["syllabus"].apply(normalize_frequency_row_name)

    # Skip placeholder rows like NA/unknown in the "other" POS bucket.
    skip_labels = {"na", "unknown", ""}
    skip_mask = (
        frequency_syllabus["pos_category"].astype(str).eq("other")
        & frequency_syllabus["syllabus_normalized"].astype(str).isin(skip_labels)
    )

    # Same for rows the treebank left unmarked or mis-tagged: a slot that decodes
    # to nothing is not a paradigm, so no lesson can be written for it.
    pos_series = frequency_syllabus["pos_category"].astype(str)
    undecodable_mask = (
        pos_series.eq("verb") & ~frequency_syllabus["syllabus"].apply(is_decodable_verb_label)
    ) | (
        pos_series.eq("noun/adjective")
        & ~frequency_syllabus["syllabus"].apply(is_decodable_nominal_label)
    )

    frequency_syllabus = frequency_syllabus.loc[~(skip_mask | undecodable_mask)].reset_index(drop=True)

    return frequency_syllabus


def syllabus_to_filename(syllabus_label: str) -> str | None:
    if pd.isna(syllabus_label):
        return None

    normalized = normalize_frequency_row_name(str(syllabus_label))
    if normalized in {"na", "unknown", ""}:
        return None
    return normalized + ".md"


SIMPLE_POS_LESSONS = {
    "adverb": "d",
    "preposition": "r",
    "particle": "g",
    "conjunction": "c",
    "interjection": "i",
}


POS_LABEL_FOR_PROMPT = {
    "verb": "verb",
    "noun/adjective": "noun or adjective",
    "article": "article",
    "pronoun": "pronoun",
    "adverb": "adverb",
    "preposition": "preposition",
    "particle": "particle",
    "conjunction": "conjunction",
    "interjection": "interjection",
}


PERSON_MAP = {"1": "1st person", "2": "2nd person", "3": "3rd person", "-": "not marked"}
NUMBER_MAP = {"s": "singular", "p": "plural", "d": "dual", "-": "not marked"}


# Bidi handling for RTL languages. Greek is strongly left-to-right, but inside an
# RTL paragraph the bidi algorithm misplaces the neutral characters at the edges
# of a Greek run, so Greek runs are isolated explicitly.
_GREEK_LETTER = "Ͱ-Ͽἀ-῿"
_GREEK_MARKS = "̀-ͯ᾽᾿’'"
_GREEK_TOKEN = f"[{_GREEK_LETTER}][{_GREEK_LETTER}{_GREEK_MARKS}]*"
# The Arabic comma and semicolon separate too: a Persian lesson writes its Greek
# lists with them, and ending the run there reversed the whole paradigm.
_GREEK_SEP = "[  ,،;؛.··‐‑-]+"
# A parenthesised tail stays inside the run: a movable ν left outside is isolated
# on its own and jumps past the form it belongs to.
_GREEK_WORD = rf"{_GREEK_TOKEN}(?:\([{_GREEK_LETTER}{_GREEK_MARKS}]+\))?"
# A tag-free phrase: Greek words joined by spaces/neutral punctuation.
_GREEK_PHRASE = f"{_GREEK_WORD}(?:{_GREEK_SEP}{_GREEK_WORD})*"
# A phrase inside one balanced inline element, so emphasis within a Greek
# sentence does not split the run and reverse the word order.
_GREEK_ELEM = "(?:" + "|".join(
    f"<{tag}>{_GREEK_PHRASE}</{tag}>" for tag in ("u", "em", "strong", "b", "i")
) + ")"
_GREEK_ATOM = f"(?:{_GREEK_PHRASE}|{_GREEK_ELEM})"
# A run: atoms joined by separators, with optional attached hyphens.
_GREEK_RUN_RE = re.compile(f"-?{_GREEK_ATOM}(?:{_GREEK_SEP}{_GREEK_ATOM})*-?")


# The starter modules mark transliteration ⟨like this⟩. The brackets are mirrored
# characters, so they point outward only while the preceding strong character is
# left-to-right. Isolating the Greek beside them takes that context away and they
# render reversed, so the transliteration needs isolating too.
_TRANSLIT_RUN_RE = re.compile("⟨[^⟨⟩]*⟩")


_HTML_TAG_SPLIT_RE = re.compile(r"(<[^>]*>)")


# A cell holding only Greek, transliteration or a Latin citation is an LTR block,
# not an LTR run: isolating the text fixes its word order but leaves the cell
# aligned against the right edge with its final period on the wrong side.
_TABLE_CELL_RE = re.compile(r"<(td|th)([^>]*)>(.*?)</\1>", re.DOTALL)
_ANY_TAG_RE = re.compile(r"<[^>]*>")
_ARABIC_RE = re.compile("[؀-ۿݐ-ݿﭐ-﷿ﹰ-﻿]")
# Latin here has to reach past ASCII: the transliteration writes macrons and
# acutes (ē, ō, ḗ), and the pronunciation column is IPA, so a cell holding only
# those would otherwise be left in the RTL flow.
_LTR_LETTER_RE = re.compile(f"[{_GREEK_LETTER}A-Za-zÀ-ʯḀ-ỿ]")


def _ltr_table_cells(html: str) -> str:
    def flip(match: re.Match) -> str:
        tag, attrs, inner = match.group(1), match.group(2), match.group(3)
        text = _ANY_TAG_RE.sub("", inner)
        if _ARABIC_RE.search(text) or not _LTR_LETTER_RE.search(text):
            return match.group(0)
        return f'<{tag}{attrs} dir="ltr">{inner}</{tag}>'

    return _TABLE_CELL_RE.sub(flip, html)


def wrap_ltr_runs_in_html(html: str) -> str:
    # Cells first, while their contents are still plain text.
    html = _ltr_table_cells(html)
    # Text nodes only. Heading ids and the links to them are slugged from the
    # heading text, so Greek does reach markup, and wrapping it there would break
    # the attribute.
    parts = _HTML_TAG_SPLIT_RE.split(html)
    for index in range(0, len(parts), 2):
        text = _GREEK_RUN_RE.sub(
            lambda match: f'<bdi lang="grc" dir="ltr">{match.group(0)}</bdi>',
            parts[index],
        )
        parts[index] = _TRANSLIT_RUN_RE.sub(
            lambda match: f'<bdi dir="ltr">{match.group(0)}</bdi>',
            text,
        )
    return "".join(parts)


def _ltr_isolate(text: str, rtl: bool) -> str:
    # LTR span around a Greek fragment so word order survives an RTL paragraph.
    if not rtl:
        return text
    return f'<span lang="grc" dir="ltr">{text}</span>'


def _ltr_block(lines: list[str], rtl: bool) -> list[str]:
    # Greek-only lines are their own LTR island. Isolating the text fixes word
    # order but leaves the block against the right margin with its list markers
    # there too. Blank lines inside keep GitHub and md_in_html parsing the
    # markdown within, as on the title page.
    if not rtl:
        return lines
    return ['<div class="ltr-block" dir="ltr" markdown="1">', "", *lines, "", "</div>", ""]


def _citation_suffix(row: Mapping[str, Any], rtl: bool) -> str:
    # Source citation for an exercise line, e.g. "  (*Hom. Il. 1.1-1.7*)"; empty
    # when the provenance cannot be resolved. LTR-isolated for RTL layouts.
    citation = format_citation(row.get("file"), row.get("document_id"), row.get("subdoc"))
    if not citation:
        return ""
    return f"  (*{_ltr_isolate(citation, rtl)}*)"


def _pos_label(lesson_pos_category: str, lang: str) -> str:
    key = "pos_label_" + re.sub(r"[^a-z0-9]+", "_", str(lesson_pos_category).lower()).strip("_")
    value = t(key, lang)
    if value == key:
        return POS_LABEL_FOR_PROMPT.get(lesson_pos_category, "target form")
    return value


def _feature_label(feature_value: str, lang: str) -> str:
    key = "feat_" + re.sub(r"[^a-z0-9]+", "_", str(feature_value).lower()).strip("_")
    value = t(key, lang)
    return feature_value if value == key else value


def _feature_name_label(feature_name: str, lang: str) -> str:
    key = "featname_" + re.sub(r"[^a-z0-9]+", "_", str(feature_name).lower()).strip("_")
    value = t(key, lang)
    return feature_name if value == key else value


def _postag_pos_label(postag: str, lang: str) -> str:
    pos_name = POSTAG_POS_NAMES.get(postag[0] if postag else "", "other")
    key = "pos_label_" + pos_name
    value = t(key, lang)
    return pos_name if value == key else value


def format_parsed_features(postag: str, lang: str, lemma: str = "") -> str:
    pairs = parse_form_features(postag, lemma)
    if not pairs:
        return ""
    separator = t("tb_feature_separator", lang)
    return separator.join(
        t("tb_feature_pair", lang, name=_feature_name_label(name, lang), value=_feature_label(value, lang))
        for name, value in pairs
    )


def _parse_answer_line(row: Mapping[str, Any], lang: str, rtl: bool) -> str | None:
    # One answer-key line for one token, or None when the form does not inflect and
    # so has nothing to parse. Every exercise that prints a key uses this, so a
    # verb, a participle and a noun all answer in the same shape.
    postag = str(row.get("postag") or "")
    lemma = str(row.get("lemma", ""))
    features = format_parsed_features(postag, lang, lemma)
    if not features:
        return None

    answer = t(
        "tb_parse_answer",
        lang,
        form=_ltr_isolate(str(row.get("form", "")), rtl),
        lemma=_ltr_isolate(lemma, rtl),
        pos=_postag_pos_label(postag, lang),
        features=features,
    )
    if postag.startswith("v") and is_deponent_lemma(lemma):
        answer += f" ({t('feat_deponent_note', lang)})"
    return answer


def _parse_answers_for_rows(target_rows: pd.DataFrame | None, lang: str, rtl: bool) -> tuple[list[str], bool]:
    # Answers for the target words of one sentence, deduplicated by form, plus
    # whether any of them had a parse at all. A word with nothing to parse still
    # answers with itself, so the sentence exercise on an adverb or preposition
    # lesson keeps naming the forms it asked for; the flag is what stops that
    # lesson's prompt from telling the student to parse them.
    if target_rows is None or target_rows.empty:
        return [], False

    answers = []
    parsed_any = False
    seen = set()
    for _, row in target_rows.iterrows():
        form = str(row.get("form", ""))
        dedupe_key = _normalize_answer_word(form)
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        parsed = _parse_answer_line(row, lang, rtl)
        parsed_any = parsed_any or parsed is not None
        answers.append(parsed or _ltr_isolate(form, rtl))
    return answers, parsed_any


def split_syllabus_label_and_bucket(syllabus_label: str) -> tuple[str, str | None]:
    if not isinstance(syllabus_label, str):
        return syllabus_label, None
    match = re.match(r"^(.*)\s\(([^()]*)\)$", syllabus_label.strip())
    if not match:
        return syllabus_label, None
    return match.group(1), match.group(2)


def _postag_feature(postag: str, index: int, code_map: Mapping[str, str]) -> str | None:
    # An empty slot is left out of a parse rather than answered, so "-" is None
    # here even though PERSON_MAP and NUMBER_MAP give it a label of its own.
    code = postag[index] if len(postag) > index else "-"
    if code == "-":
        return None
    return code_map.get(code)


def parse_form_features(postag: str, lemma: str = "") -> list[tuple[str, str]]:
    # The full parse of one form, as ordered (feature, value) pairs in the order a
    # grammar book asks for them. Only what the form actually carries is reported:
    # an infinitive has no person, a participle has a case. Slots left empty by the
    # tagger are dropped rather than answered "not marked", and a postag outside
    # the inflected classes yields nothing at all, which is what keeps the parsing
    # exercise off the preposition and conjunction lessons.
    if not isinstance(postag, str) or not postag:
        return []

    pos = postag[0]
    if pos not in POSTAG_POS_NAMES:
        return []

    pairs: list[tuple[str, str | None]] = []

    if pos == "v":
        mood = _postag_feature(postag, POSTAG_MOOD_INDEX, MOOD_MAP)
        if mood in {"infinitive", "participle"}:
            pairs = [
                ("tense", _postag_feature(postag, POSTAG_TENSE_INDEX, TENSE_MAP)),
                ("voice", _postag_feature(postag, POSTAG_VOICE_INDEX, VOICE_MAP)),
                ("mood", mood),
            ]
            if mood == "participle":
                pairs += [
                    ("case", _postag_feature(postag, POSTAG_CASE_INDEX, CASE_MAP)),
                    ("number", _postag_feature(postag, POSTAG_NUMBER_INDEX, NUMBER_MAP)),
                    ("gender", _postag_feature(postag, POSTAG_GENDER_INDEX, GENDER_MAP)),
                ]
        else:
            pairs = [
                ("person", _postag_feature(postag, POSTAG_PERSON_INDEX, PERSON_MAP)),
                ("number", _postag_feature(postag, POSTAG_NUMBER_INDEX, NUMBER_MAP)),
                ("tense", _postag_feature(postag, POSTAG_TENSE_INDEX, TENSE_MAP)),
                ("mood", mood),
                ("voice", _postag_feature(postag, POSTAG_VOICE_INDEX, VOICE_MAP)),
            ]
    else:
        if pos == "p":
            # Only the personal pronouns are marked for person; the rest leave the
            # slot empty and drop out below.
            pairs.append(("person", _postag_feature(postag, POSTAG_PERSON_INDEX, PERSON_MAP)))
        gender = _postag_feature(postag, POSTAG_GENDER_INDEX, GENDER_MAP)
        if is_genderless_pronoun_lemma(lemma):
            # ἐγώ and σύ have no gender. Treebanks tag them masculine by default,
            # which would print as grammar the student then has to unlearn.
            gender = None
        pairs += [
            ("case", _postag_feature(postag, POSTAG_CASE_INDEX, CASE_MAP)),
            ("number", _postag_feature(postag, POSTAG_NUMBER_INDEX, NUMBER_MAP)),
            ("gender", gender),
        ]
        if pos == "a":
            pairs.append(("degree", _postag_feature(postag, POSTAG_DEGREE_INDEX, DEGREE_MAP)))

    return [(name, value) for name, value in pairs if value]


DEPONENT_LESSON_LABEL = "deponent verbs"
DEPONENT_LESSON_FILENAME = "deponent_verbs.md"

CONDITIONAL_LESSON_LABEL = "conditional sentences"
CONDITIONAL_LESSON_FILENAME = "conditionals.md"

DIDAKTA_LESSON_LABEL = "didakta"
DIDAKTA_LESSON_FILENAME = "didakta.md"

# The starter module that teaches dictionary lookup.
DICTIONARY_LESSON_MODULE = "using_a_dictionary"

# The last starter module, which carries the generated core function-word table:
# the hand-over point between the preset modules and the corpus-driven lessons.
# Named rather than matched on position.
DIALECTS_LESSON_MODULE = "greek_dialects"
# The reference appendix, not a lesson: it never enters the syllabus.
SYNTAX_REFERENCE_FILENAME = "syntax_reference.md"


def get_topic_rows_for_label(syllabus_label: str, combined_df: pd.DataFrame) -> pd.DataFrame:
    # The "deponent verbs" concept lesson draws on every deponent verb token,
    # whatever its tense, mood or voice.
    if normalize_frequency_row_name(str(syllabus_label)) == normalize_frequency_row_name(DEPONENT_LESSON_LABEL):
        if "is_deponent" in combined_df.columns:
            return combined_df[(combined_df["pos_category"] == "verb") & combined_df["is_deponent"]].copy()
        return combined_df.iloc[0:0].copy()

    # Likewise the "irregular verbs" lesson.
    if normalize_frequency_row_name(str(syllabus_label)) == normalize_frequency_row_name(IRREGULAR_LESSON_LABEL):
        return combined_df[
            (combined_df["pos_category"] == "verb")
            & combined_df["verb_subcategory"].astype(str).eq(IRREGULAR_VERB_BUCKET)
        ].copy()

    # A lesson that absorbed other rows draws on the tokens of all of them, so
    # its exercises can show any of those forms.
    candidate_labels = [str(syllabus_label)] + [
        source for source, target in MERGED_SYLLABUS_LABELS.items() if target == syllabus_label
    ]

    matched = []
    for candidate in candidate_labels:
        base_label, verb_bucket = split_syllabus_label_and_bucket(candidate)
        rows = combined_df[combined_df["syllabus"] == base_label]
        if verb_bucket is not None:
            rows = rows[
                (rows["pos_category"] == "verb") & (rows["verb_subcategory"] == verb_bucket)
            ]
        if not rows.empty:
            matched.append(rows)

    if matched:
        return pd.concat(matched).copy() if len(matched) > 1 else matched[0].copy()

    normalized_target = normalize_frequency_row_name(syllabus_label)
    normalized_series = combined_df["syllabus"].apply(normalize_frequency_row_name)

    verb_suffix_map = {
        "_w": "w",
        "_mi": "mi",
    }

    for suffix, raw_bucket in verb_suffix_map.items():
        if normalized_target.endswith(suffix):
            base_norm = normalized_target[: -len(suffix)]
            return combined_df[
                (normalized_series == base_norm)
                & (combined_df["pos_category"] == "verb")
                & (combined_df["verb_subcategory"].isin([raw_bucket, suffix.lstrip("_")]))
            ].copy()

    return combined_df[normalized_series == normalized_target].copy()


def filter_topic_rows_by_lesson_rules(
    syllabus_label: str,
    lesson_pos_category: str,
    topic_rows: pd.DataFrame,
) -> pd.DataFrame:
    case_lessons = {"accusative", "dative", "genitive", "nominative", "vocative"}
    if syllabus_label == "article":
        return topic_rows[topic_rows["postag"].str.startswith("l", na=False)]
    if syllabus_label in case_lessons:
        return topic_rows[topic_rows["postag"].str.startswith(("n", "a"), na=False)]
    if lesson_pos_category == "verb":
        return topic_rows[topic_rows["postag"].str.startswith("v", na=False)]
    if lesson_pos_category == "pronoun":
        return topic_rows[topic_rows["postag"].str.startswith("p", na=False)]
    if lesson_pos_category in SIMPLE_POS_LESSONS:
        prefix = SIMPLE_POS_LESSONS[lesson_pos_category]
        return topic_rows[topic_rows["postag"].str.startswith(prefix, na=False)]
    return topic_rows


def mark_topic_words_in_sentence(sentence_text: str, target_forms: set[str]) -> str:
    if not target_forms:
        return sentence_text

    marked_text = sentence_text
    for form in sorted(target_forms, key=len, reverse=True):
        if not form:
            continue
        marked_text = re.sub(rf"(?<!\w)({re.escape(form)})(?!\w)", r"<u>\1</u>", marked_text)
    return marked_text


def get_topic_words(
    syllabus_label: str,
    lesson_pos_category: str,
    combined_df: pd.DataFrame,
    num_words: int = 15,
) -> pd.DataFrame:
    topic_rows = get_topic_rows_for_label(syllabus_label, combined_df)
    if topic_rows.empty:
        return pd.DataFrame()

    topic_rows = topic_rows.dropna(subset=["form", "lemma", "postag"]).copy()
    topic_rows["form"] = topic_rows["form"].astype(str).str.strip()
    topic_rows["lemma"] = topic_rows["lemma"].astype(str).str.strip()
    topic_rows["postag"] = topic_rows["postag"].astype(str).str.strip()
    topic_rows = topic_rows[(topic_rows["form"] != "") & (topic_rows["lemma"] != "") & (topic_rows["postag"] != "")]
    topic_rows = topic_rows[topic_rows["lemma"].apply(is_greek_lemma)]
    topic_rows = drop_word_fragments(topic_rows)

    if topic_rows.empty:
        return pd.DataFrame()

    topic_rows = filter_topic_rows_by_lesson_rules(syllabus_label, lesson_pos_category, topic_rows)
    if topic_rows.empty:
        return pd.DataFrame()

    if "lemma_frequency" not in topic_rows.columns:
        local_counts = topic_rows["lemma"].value_counts()
        topic_rows["lemma_frequency"] = topic_rows["lemma"].map(local_counts)

    topic_rows["lemma_frequency"] = pd.to_numeric(topic_rows["lemma_frequency"], errors="coerce").fillna(0)
    topic_rows = topic_rows.sort_values("lemma_frequency", ascending=False)
    # One entry per lemma, then one per printed form: the interrogative τίς and
    # the indefinite τις are two lemmas that can surface as the same string with
    # the same parse, which asks the student the same question twice.
    topic_words = topic_rows.drop_duplicates(subset=["lemma"], keep="first")
    topic_words = topic_words.drop_duplicates(subset=["form"], keep="first").head(num_words)
    return topic_words[["form", "lemma", "postag", "token_index", "sentence_index"]]


# How many words each kind of lesson offers. Nouns and adjectives get one shot per
# declension class, and the closed classes are worth learning nearly whole, so both
# run to 20; verb lessons are many and every entry in them is new, so 10 each still
# adds up. These are targets, not quotas: a lesson prints fewer rather than padding.
VOCAB_LIST_SIZES = {
    "verb": 10,
    "noun/adjective": 20,
}
VOCAB_LIST_SIZE_DEFAULT = 20
# How many function words the last starter module hands over up front.
VOCAB_CORE_WORD_COUNT = 20
# A function word attested once is a scribal accident, not vocabulary.
VOCAB_MIN_FUNCTION_WORD_COUNT = 2
# Share of a lemma's tokens that must carry a function-word tag before the core
# table will take it. Membership is per lemma, not per token.
FUNCTION_WORD_LEMMA_SHARE = 0.5

_VOCAB_COLUMNS = ["headword", "ledger_key", "frequency", "is_name", "is_deponent", "pos_category"]


# One notion of "the same word" for the whole book: accents and homonym digits
# off, so λέγω1 and λέγω3 are one entry in the ledger and one word to a learner.
# Cached because the sentence-coverage scan runs it over every token.
@lru_cache(maxsize=None)
def vocabulary_ledger_key(lemma: str) -> str:
    return normalize_greek_lemma(lemma_headword(lemma))


def _is_proper_name(headword: str) -> bool:
    # Treebank lemmas are lemmatized, so an initial capital marks a name and not
    # merely a word that opened a sentence.
    return bool(headword) and headword[:1].isupper()


def _vocabulary_list_size(lesson_pos_category: str) -> int:
    return VOCAB_LIST_SIZES.get(lesson_pos_category, VOCAB_LIST_SIZE_DEFAULT)


def _clean_lemma_rows(rows: pd.DataFrame) -> pd.DataFrame:
    rows = rows.dropna(subset=["lemma", "postag"]).copy()
    rows["lemma"] = rows["lemma"].astype(str).str.strip()
    rows["postag"] = rows["postag"].astype(str).str.strip()
    rows = rows[(rows["lemma"] != "") & (rows["postag"] != "")]
    return rows[rows["lemma"].apply(is_greek_lemma)]


def _dominant_value(rows: pd.DataFrame, column: str) -> pd.Series:
    # The commonest value of a column per ledger key, ties going to whichever the
    # groupby met first.
    return (
        rows.groupby(["ledger_key", column], sort=False)
        .size()
        .sort_values(ascending=False)
        .reset_index()
        .drop_duplicates("ledger_key")
        .set_index("ledger_key")[column]
    )


def _aggregate_vocabulary_rows(rows: pd.DataFrame) -> pd.DataFrame:
    # One row per headword, ranked by how often the corpus uses it.
    if rows.empty:
        return pd.DataFrame(columns=_VOCAB_COLUMNS)

    rows = rows.copy()
    rows["headword"] = rows["lemma"].map(lemma_headword)
    rows["ledger_key"] = rows["lemma"].map(vocabulary_ledger_key)
    rows = rows[rows["headword"] != ""]
    if rows.empty:
        return pd.DataFrame(columns=_VOCAB_COLUMNS)

    if "lemma_frequency" in rows.columns:
        rows["_frequency"] = pd.to_numeric(rows["lemma_frequency"], errors="coerce").fillna(0)
    else:
        rows["_frequency"] = rows["lemma"].map(rows["lemma"].value_counts()).fillna(0)

    rows["_deponent"] = (
        rows["is_deponent"].fillna(False).astype(bool) if "is_deponent" in rows.columns else False
    )
    rows["_pos"] = rows["pos_category"].astype(str) if "pos_category" in rows.columns else ""

    aggregated = (
        rows.groupby("ledger_key", sort=False)
        .agg(
            frequency=("_frequency", "max"),
            is_deponent=("_deponent", "any"),
        )
        .reset_index()
    )

    # Spelling and tag both come from the commonest row rather than whichever one
    # sorted first. The ledger key drops breathings, so εἰς and εἷς share an entry
    # and "first" could print the numeral over the preposition's count; the
    # treebanks also tag inconsistently, and one stray row in 326 made ὅς an
    # article.
    aggregated["headword"] = aggregated["ledger_key"].map(_dominant_value(rows, "headword"))
    aggregated["pos_category"] = aggregated["ledger_key"].map(_dominant_value(rows, "_pos")).fillna("")

    aggregated["is_name"] = aggregated["headword"].map(_is_proper_name)
    return aggregated.sort_values("frequency", ascending=False, ignore_index=True)


def _reserved_verb_lemmas(topic_rows: pd.DataFrame) -> set[str]:
    # Deponents and irregulars are taught as lexical classes in their own concept
    # lessons, so the paradigm lessons they happen to appear in leave them alone.
    reserved = pd.Series(False, index=topic_rows.index)
    if "is_deponent" in topic_rows.columns:
        reserved |= topic_rows["is_deponent"].fillna(False).astype(bool)
    if "verb_subcategory" in topic_rows.columns:
        reserved |= topic_rows["verb_subcategory"].astype(str).eq(IRREGULAR_VERB_BUCKET)
    return {vocabulary_ledger_key(str(lemma)) for lemma in topic_rows.loc[reserved, "lemma"]}


def _attach_citations(
    words: pd.DataFrame,
    citation_index: Mapping[str, Mapping[str, str]] | None,
) -> pd.DataFrame:
    words = words.copy()
    if words.empty:
        words["genitive"] = ""
        words["article"] = ""
        return words

    index = citation_index or {}
    keys = words["headword"].map(lemma_headword)
    words["genitive"] = keys.map(lambda key: index.get(key, {}).get("genitive", ""))
    words["article"] = keys.map(lambda key: index.get(key, {}).get("article", ""))
    return words


def get_lesson_vocabulary(
    syllabus_label: str,
    lesson_pos_category: str,
    combined_df: pd.DataFrame,
    introduced_lemmas: set[str] | None = None,
    citation_index: Mapping[str, Mapping[str, str]] | None = None,
) -> pd.DataFrame:
    # The most frequent words of this lesson's own part of speech, counted over
    # the whole corpus. The lesson's exercises play no part in the choice.
    introduced = introduced_lemmas or set()
    empty = pd.DataFrame(columns=_VOCAB_COLUMNS)

    topic_rows = get_topic_rows_for_label(syllabus_label, combined_df)
    if topic_rows.empty:
        return empty

    topic_rows = _clean_lemma_rows(topic_rows)
    topic_rows = filter_topic_rows_by_lesson_rules(syllabus_label, lesson_pos_category, topic_rows)
    if topic_rows.empty:
        return empty

    normalized_label = normalize_frequency_row_name(str(syllabus_label))
    teaches_reserved_class = normalized_label in {
        normalize_frequency_row_name(DEPONENT_LESSON_LABEL),
        normalize_frequency_row_name(IRREGULAR_LESSON_LABEL),
    }

    candidates = _aggregate_vocabulary_rows(topic_rows)
    if candidates.empty:
        return empty

    if lesson_pos_category == "verb" and not teaches_reserved_class:
        candidates = candidates[~candidates["ledger_key"].isin(_reserved_verb_lemmas(topic_rows))]
    elif lesson_pos_category not in VOCAB_LIST_SIZES:
        # A closed class: print the inventory rather than a sample of it.
        candidates = candidates[candidates["frequency"] >= VOCAB_MIN_FUNCTION_WORD_COUNT]

    # Proper names are frequent but they are not vocabulary: a lexicon will not
    # help with Κῦρος, and in a corpus like Herodotus they would crowd out the
    # nouns worth learning.
    candidates = candidates[~candidates["is_name"]]
    candidates = candidates[~candidates["ledger_key"].isin(introduced)]
    words = candidates.head(_vocabulary_list_size(lesson_pos_category))
    return _attach_citations(words, citation_index).reset_index(drop=True)


def assemble_sentences(df: pd.DataFrame) -> pd.DataFrame:
    attach_to_prev = {",", ".", ";", ":", "!", "?", ")", "']"}

    def join_forms(forms: list[str]) -> str:
        words = []
        # A trailing-hyphen token (the first half of a crasis) waits here to be
        # glued to the following word.
        pending_prefix = ""
        for form in forms:
            token = str(form).strip()
            if not token:
                continue

            # A bare hyphen is a stray join marker with nothing to attach; drop it.
            if set(token) == {"-"}:
                continue

            # Enclitic marked with a leading hyphen: glue it to the word before,
            # dropping the seam marker.
            if token.startswith("-"):
                glued = token.lstrip("-")
                if words:
                    words[-1] += glued
                else:
                    words.append(glued)
                continue

            # Crasis first half: hold it and prepend it to the next word.
            if token.endswith("-"):
                pending_prefix += token.rstrip("-")
                continue

            if pending_prefix:
                token = pending_prefix + token
                pending_prefix = ""

            if token in attach_to_prev and words:
                words[-1] += token
            else:
                words.append(token)

        # A dangling crasis prefix is kept rather than lost.
        if pending_prefix:
            words.append(pending_prefix)

        text = " ".join(words)
        text = re.sub(r"\s+([,.:;!?\)])", r"\1", text)
        text = re.sub(r"([\(\[])\s+", r"\1", text)

        # Bracketed index markers from the source data, like [0], [12].
        text = re.sub(r"\[\s*\d+\s*\]", "", text)

        # Hidden Unicode formatting chars, which show up as odd symbols.
        text = re.sub(r"[\u200b-\u200f\u2060\ufeff]", "", text)

        text = re.sub(r"\s+", " ", text).strip()
        return text

    # Columns as plain lists, addressed by group position: a per-group
    # sort_values over ~19k sentences dominated build time, and every parser
    # already appends a sentence's tokens contiguously and in order.
    forms = df["form"].tolist()
    doc_ids = df["document_id"].tolist() if "document_id" in df.columns else None
    subdocs = df["subdoc"].tolist() if "subdoc" in df.columns else None
    file_ids = df["file"].tolist() if "file" in df.columns else None

    rows = []
    for sent_id, positions in df.groupby("sentence_id", sort=False).indices.items():
        first = positions[0]
        rows.append(
            {
                "sentence_id": sent_id,
                "document_id": doc_ids[first] if doc_ids is not None else None,
                "subdoc": subdocs[first] if subdocs is not None else None,
                "file": file_ids[first] if file_ids is not None else None,
                "sentence_text": join_forms([forms[position] for position in positions]),
                "word_count": len(positions),
            }
        )

    sentences = pd.DataFrame(rows)
    return _blank_whole_work_subdocs(sentences)


def _blank_whole_work_subdocs(sentences: pd.DataFrame) -> pd.DataFrame:
    # Some Perseus files tag every sentence with one whole-work range (Lysias 1
    # is "1-50" throughout), which is noise, not a citation. Blank a subdoc that
    # is constant across a file and looks like a range, so the citation degrades
    # to the work label. Varying refs (Homer's line ranges) are untouched.
    if sentences.empty or "subdoc" not in sentences.columns or "file" not in sentences.columns:
        return sentences

    for _, idx in sentences.groupby("file", sort=False).groups.items():
        values = sentences.loc[idx, "subdoc"].dropna().unique()
        if len(values) == 1 and "-" in str(values[0]):
            sentences.loc[idx, "subdoc"] = None

    return sentences


def add_sentence_scores(sentences_df: pd.DataFrame, combined_df: pd.DataFrame) -> pd.DataFrame:
    out = sentences_df.copy()

    greek = combined_df[combined_df["lemma"].apply(is_greek_lemma)].copy()
    if "lemma_frequency" in greek.columns:
        greek["lemma_frequency"] = pd.to_numeric(greek["lemma_frequency"], errors="coerce").fillna(0.0)
    else:
        counts = greek["lemma"].value_counts()
        greek["lemma_frequency"] = greek["lemma"].map(counts).astype(float)

    # Content words only: function words are frequent enough to drown out the
    # ones that gate comprehension. Log frequencies tame the Zipf skew.
    content = greek[greek["postag"].astype(str).str.startswith(CONTENT_POS_PREFIXES)].copy()

    stat_columns = ["avg_log_lemma_freq", "min_log_lemma_freq"]
    out = out.drop(columns=stat_columns, errors="ignore")
    if content.empty:
        out["avg_log_lemma_freq"] = 0.0
        out["min_log_lemma_freq"] = 0.0
    else:
        content["log_lemma_freq"] = np.log1p(content["lemma_frequency"])
        sent_stats = content.groupby("sentence_id", as_index=False).agg(
            avg_log_lemma_freq=("log_lemma_freq", "mean"),
            min_log_lemma_freq=("log_lemma_freq", "min"),
        )
        out = out.merge(sent_stats, on="sentence_id", how="left")
        # No content words means no lexical load, so score them easy.
        for column in stat_columns:
            max_value = out[column].max()
            out[column] = out[column].fillna(max_value if pd.notna(max_value) else 0.0)

    def to_0_100(series: pd.Series) -> pd.Series:
        max_value = series.max()
        if pd.notna(max_value) and max_value > 0:
            return series / max_value * 100
        return pd.Series(0.0, index=series.index)

    out["mean_rarity_score"] = 100 - to_0_100(pd.to_numeric(out["avg_log_lemma_freq"], errors="coerce").fillna(0.0))
    out["rarest_word_score"] = 100 - to_0_100(pd.to_numeric(out["min_log_lemma_freq"], errors="coerce").fillna(0.0))
    out["sentence_length_score"] = pd.to_numeric(to_0_100(out["word_count"]), errors="coerce").fillna(0.0)
    out["difficulty_score"] = (
        DIFFICULTY_WEIGHT_MEAN_RARITY * out["mean_rarity_score"]
        + DIFFICULTY_WEIGHT_RAREST_WORD * out["rarest_word_score"]
        + DIFFICULTY_WEIGHT_LENGTH * out["sentence_length_score"]
    )
    return out


def build_known_lemma_seed(
    combined_df: pd.DataFrame,
    top_n: int = KNOWN_FUNCTION_LEMMA_SEED_COUNT,
) -> set[str]:
    # The top function-word lemmas, which every reader meets from the first page,
    # seed the known-vocabulary set used for stage-aware sentence selection.
    if combined_df is None or combined_df.empty:
        return set()
    greek = combined_df[combined_df["lemma"].apply(is_greek_lemma)]
    function_rows = greek[~greek["postag"].astype(str).str.startswith(CONTENT_POS_PREFIXES)]
    top_lemmas = function_rows["lemma"].value_counts().head(top_n).index
    return {vocabulary_ledger_key(str(lemma)) for lemma in top_lemmas}


def _known_lemma_coverage_by_sentence(combined_df: pd.DataFrame, known_lemmas: set[str]) -> pd.Series:
    # Fraction of content-word lemmas per sentence_index that are known.
    greek = combined_df[combined_df["lemma"].apply(is_greek_lemma)]
    content = greek[greek["postag"].astype(str).str.startswith(CONTENT_POS_PREFIXES)]
    if content.empty:
        return pd.Series(dtype=float)
    known = content["lemma"].astype(str).map(vocabulary_ledger_key).isin(known_lemmas)
    return known.groupby(content["sentence_index"]).mean()


def _sentence_known_lemma_counts(combined_df: pd.DataFrame, known_lemmas: set[str]) -> pd.DataFrame:
    # Content-word totals and known counts per sentence_index. Only two columns
    # are materialized: the boolean-index copies _known_lemma_coverage_by_sentence
    # takes carry all fifteen, which one lesson's rows can afford and a whole
    # select-all frame cannot.
    empty = pd.DataFrame(columns=["content_words", "known_words"])
    if combined_df is None or combined_df.empty or "sentence_index" not in combined_df.columns:
        return empty

    lemmas = combined_df["lemma"].astype(str)
    content = combined_df["postag"].astype(str).str.startswith(CONTENT_POS_PREFIXES) & lemmas.map(is_greek_lemma)
    if not content.any():
        return empty

    known = lemmas[content].map(vocabulary_ledger_key).isin(known_lemmas)
    grouped = known.groupby(combined_df["sentence_index"][content], sort=False)
    return pd.DataFrame({"content_words": grouped.size(), "known_words": grouped.sum()})


def _citation_units(sentences: pd.DataFrame) -> list[dict]:
    # Runs of consecutive sentences sharing one subdoc, per file, in document
    # order. A run-length walk rather than a groupby: Homer's "1.9" is followed by
    # "1.9-1.12", and a reference that recurs later must stay two units instead of
    # fusing into one range that never existed. Sentences whose subdoc was blanked
    # group under "", which _split_oversized_unit then cuts on sentence bounds.
    if sentences is None or sentences.empty:
        return []

    ordered = sentences.sort_values(["file", "sentence_index"], kind="stable")
    units: list[dict] = []
    current: dict | None = None

    for row in ordered.itertuples(index=False):
        subdoc = row.subdoc.strip() if isinstance(row.subdoc, str) else ""
        if current is None or current["file"] != row.file or current["subdoc"] != subdoc:
            current = {
                "file": row.file,
                "document_id": row.document_id,
                "subdoc": subdoc,
                "sentences": [],
                "word_count": 0,
            }
            units.append(current)
        current["sentences"].append(row)
        current["word_count"] += int(row.word_count)

    return units


def _split_oversized_unit(unit: dict) -> list[dict]:
    # A unit too big to be a passage is cut on sentence boundaries instead. Plato's
    # Euthyphro carries one subdoc across all 6,349 of its words and Lysias 1 has
    # its whole-work range blanked, so without this those two textbooks would end
    # on an empty appendix. Cuts still never land inside a sentence.
    if unit["word_count"] <= PASSAGE_MAX_WORDS:
        return [unit]

    pieces: list[dict] = []
    current: dict | None = None
    for sentence in unit["sentences"]:
        words = int(sentence.word_count)
        if current is not None and (
            current["word_count"] >= PASSAGE_WORD_BUDGET
            or current["word_count"] + words > PASSAGE_MAX_WORDS
        ):
            current = None
        if current is None:
            current = {**unit, "sentences": [], "word_count": 0}
            pieces.append(current)
        current["sentences"].append(sentence)
        current["word_count"] += words

    return pieces


def _pack_passages(units: list[dict]) -> list[dict]:
    # Whole units accumulated to the budget, cut only on a unit boundary. The
    # lookahead earns its keep: closing as soon as the budget is reached lets a
    # 97-word Aesop fable swallow the next one, and the 195-word result then fails
    # the maximum and is dropped, so Aesop would contribute nothing at all.
    passages: list[dict] = []
    current: dict | None = None

    def close() -> None:
        if current is not None and PASSAGE_MIN_WORDS <= current["word_count"] <= PASSAGE_MAX_WORDS:
            passages.append(current)

    for unit in units:
        for piece in _split_oversized_unit(unit):
            if current is not None and (
                current["file"] != piece["file"]
                or current["word_count"] >= PASSAGE_WORD_BUDGET
                or current["word_count"] + piece["word_count"] > PASSAGE_MAX_WORDS
            ):
                close()
                current = None
            if current is None:
                current = {
                    "file": piece["file"],
                    "document_id": piece["document_id"],
                    "first_subdoc": "",
                    "last_subdoc": "",
                    "sentences": [],
                    "word_count": 0,
                }
            current["sentences"].extend(piece["sentences"])
            current["word_count"] += piece["word_count"]
            # Only referenced units move the span, so a passage running from a
            # referenced unit into an unreferenced one still cites what it can.
            if piece["subdoc"]:
                current["first_subdoc"] = current["first_subdoc"] or piece["subdoc"]
                current["last_subdoc"] = piece["subdoc"]

    close()
    return passages


def _subdoc_span(first_subdoc: str, last_subdoc: str) -> str:
    # One reference covering both ends. A unit that is itself a range contributes
    # its outer edge, so Homer's "1.1-1.7" through "1.29-1.31" joins to "1.1-1.31".
    # Either end may be missing where a work references only part of itself, and a
    # half-open "1.1.1-" is worse than the narrower reference that is certain.
    start = (first_subdoc or "").split("-")[0].strip()
    end = (last_subdoc or "").split("-")[-1].strip()
    if not start or not end:
        return start or end
    return start if start == end else f"{start}-{end}"


def build_reading_passages(
    sentences_df: pd.DataFrame,
    combined_df: pd.DataFrame,
    known_lemmas: set[str],
    count: int = PASSAGE_COUNT,
) -> list[dict]:
    # Passages ranked by how much of their vocabulary the finished book taught, so
    # the appendix opens with what a reader who worked through it can already read.
    if sentences_df is None or sentences_df.empty:
        return []

    packed = _pack_passages(_citation_units(sentences_df))
    if not packed:
        return []

    counts = _sentence_known_lemma_counts(combined_df, known_lemmas)
    content_by_sentence = counts["content_words"].to_dict() if not counts.empty else {}
    known_by_sentence = counts["known_words"].to_dict() if not counts.empty else {}

    scored: list[dict] = []
    for passage in packed:
        indices = [sentence.sentence_index for sentence in passage["sentences"]]
        content_words = sum(int(content_by_sentence.get(index, 0)) for index in indices)
        if content_words <= 0:
            # A run of nothing but punctuation and function words; ranking it by
            # coverage would put it first on a division by nothing.
            continue
        known_words = sum(int(known_by_sentence.get(index, 0)) for index in indices)
        difficulties = [float(getattr(sentence, "difficulty_score", 0.0) or 0.0) for sentence in passage["sentences"]]
        scored.append(
            {
                "file": passage["file"],
                "document_id": passage["document_id"],
                "work_key": tlg_work_key(passage["file"], passage["document_id"]) or passage["file"],
                "citation": format_citation(
                    passage["file"],
                    passage["document_id"],
                    _subdoc_span(passage["first_subdoc"], passage["last_subdoc"]),
                ),
                "word_count": passage["word_count"],
                "coverage": known_words / content_words,
                "difficulty": sum(difficulties) / len(difficulties) if difficulties else 0.0,
                "order": min(indices),
                "text": " ".join(str(sentence.sentence_text) for sentence in passage["sentences"]),
            }
        )

    # Difficulty breaks ties on coverage; file and position only make the build
    # repeatable.
    def ranking(passage: dict) -> tuple:
        return -passage["coverage"], passage["difficulty"], str(passage["file"]), passage["order"]

    scored.sort(key=ranking)

    per_work_cap = count
    if len({passage["work_key"] for passage in scored}) > 1:
        per_work_cap = max(1, int(count * PASSAGE_MAX_WORK_SHARE))

    selected: list[dict] = []
    taken: dict[str, int] = {}
    for passage in scored:
        work_key = passage["work_key"]
        if taken.get(work_key, 0) >= per_work_cap:
            continue
        selected.append(passage)
        taken[work_key] = taken.get(work_key, 0) + 1
        if len(selected) >= count:
            break

    # A cap that starves the appendix is worse than an unbalanced one, so fill any
    # shortfall from what it held back.
    if len(selected) < count:
        chosen = {id(passage) for passage in selected}
        for passage in scored:
            if len(selected) >= count:
                break
            if id(passage) not in chosen:
                selected.append(passage)
        selected.sort(key=ranking)

    return selected


def get_topic_sentences(
    syllabus_label: str,
    combined_df: pd.DataFrame,
    sentences_df: pd.DataFrame,
    num_sentences: int = 20,
    known_lemmas: set[str] | None = None,
) -> pd.DataFrame:
    matching_rows = get_topic_rows_for_label(syllabus_label, combined_df)
    if matching_rows.empty:
        return pd.DataFrame()

    matching_sentence_indices = set(matching_rows["sentence_index"].unique())
    if not matching_sentence_indices:
        return pd.DataFrame()

    topic_sentences = sentences_df[sentences_df["sentence_index"].isin(matching_sentence_indices)].copy()
    if topic_sentences.empty:
        return pd.DataFrame()

    if not known_lemmas:
        return topic_sentences.sort_values("difficulty_score").head(num_sentences)

    # The lesson's own target lemmas are being taught now, so they count as known.
    effective_known = known_lemmas | {
        normalize_greek_lemma(str(lemma)) for lemma in matching_rows["lemma"].dropna()
    }
    candidate_rows = combined_df[combined_df["sentence_index"].isin(matching_sentence_indices)]
    coverage = _known_lemma_coverage_by_sentence(candidate_rows, effective_known)
    topic_sentences["known_lemma_coverage"] = topic_sentences["sentence_index"].map(coverage).fillna(1.0)

    # Stage-appropriate sentences first; the rest, ranked by difficulty alone,
    # fill the quota when the corpus cannot.
    qualified_mask = topic_sentences["known_lemma_coverage"] >= KNOWN_LEMMA_COVERAGE_THRESHOLD
    qualified = topic_sentences[qualified_mask].sort_values("difficulty_score")
    remainder = topic_sentences[~qualified_mask].sort_values("difficulty_score")
    return pd.concat([qualified, remainder]).head(num_sentences)


def format_parsing_exercise(topic_words: pd.DataFrame, lang: str = DEFAULT_LANG) -> str:
    # Words that do not inflect are dropped rather than asked about, so the whole
    # exercise disappears from the adverb, preposition, conjunction, particle and
    # interjection lessons; those keep the sentence exercise alone. The prompt names
    # no part of speech because a case lesson mixes nouns and adjectives.
    if topic_words is None or topic_words.empty:
        return ""

    rtl = is_rtl(lang)
    items = []
    answers = []
    for _, row in topic_words.iterrows():
        answer = _parse_answer_line(row, lang, rtl)
        if not answer:
            continue
        items.append(
            t(
                "tb_ex_parsing_item",
                lang,
                form=_ltr_isolate(str(row["form"]), rtl),
                lemma=_ltr_isolate(str(row["lemma"]), rtl),
            )
        )
        answers.append(answer)

    if not items:
        return ""

    lines = [
        f"### {t('tb_ex_parsing_header', lang)}",
        "",
        t("tb_ex_parsing_prompt", lang),
        "",
    ]
    lines += [f"{idx}. {item}" for idx, item in enumerate(items, 1)]
    lines += ["", f"#### {t('tb_answer_key_header', lang)}", ""]
    lines += [f"{idx}. {answer}" for idx, answer in enumerate(answers, 1)]
    lines.append("")
    return "\n".join(lines)


LOGEION_URL = "https://logeion.uchicago.edu/{headword}"
PERSEUS_URL = "https://www.perseus.tufts.edu/hopper/morph?l={headword}&la=greek"


def logeion_url(headword: str) -> str:
    # Logeion takes the accented headword straight in the path, but answers a
    # request carrying a Perseus homonym digit with "Could not find λέγω1", so the
    # digit has to be gone before the link is built.
    return LOGEION_URL.format(headword=quote(lemma_headword(headword), safe=""))


def perseus_url(headword: str) -> str:
    # The Perseus word study tool, which parses the form and then offers LSJ,
    # Middle Liddell, Slater and Autenrieth for it.
    return PERSEUS_URL.format(headword=quote(lemma_headword(headword), safe=""))


def _vocabulary_frequency(row: pd.Series, lang: str) -> str:
    return t("tb_vocab_freq", lang, count=int(row.get("frequency", 0) or 0))


def _vocabulary_tagged_entry(row: pd.Series, lang: str, rtl: bool) -> str:
    # Headword plus its part of speech. A mis-tagged token leaves pos_category as
    # "other", and _pos_label answers that with the exercise wording "target
    # form", which says nothing in a word list -- so the tag is simply dropped.
    headword = _ltr_isolate(str(row["headword"]), rtl)
    pos_category = str(row.get("pos_category", "") or "")
    if not pos_category or pos_category == "other":
        return headword
    return t("tb_vocab_word_with_pos", lang, headword=headword, pos_label=_pos_label(pos_category, lang))


def _vocabulary_word_cell(row: pd.Series, lang: str, rtl: bool) -> str:
    # A dictionary entry's opening line, as far as the corpus can attest it:
    # headword, genitive singular, article. Missing parts are left out rather
    # than guessed, so a noun with no attested genitive prints as a bare headword.
    entry = _ltr_isolate(str(row["headword"]), rtl)
    genitive = str(row.get("genitive", "") or "")
    article = str(row.get("article", "") or "")

    if genitive and article:
        entry = t(
            "tb_vocab_entry_full",
            lang,
            headword=entry,
            genitive=_ltr_isolate(genitive, rtl),
            article=_ltr_isolate(article, rtl),
        )
    elif article:
        entry = t("tb_vocab_entry_article", lang, headword=entry, article=_ltr_isolate(article, rtl))

    if bool(row.get("is_deponent", False)):
        entry = f"{entry} *({t('tb_vocab_deponent_tag', lang)})*"
    return entry


def _vocabulary_lookup_cell(headword: str, lang: str, rtl: bool) -> str:
    # The lexicon names carry the links; the locale string holds the markdown so
    # a translator controls the link text along with the sentence around it.
    return t(
        "tb_vocab_lookup_cell",
        lang,
        headword=_ltr_isolate(headword, rtl),
        perseus_url=perseus_url(headword),
        logeion_url=logeion_url(headword),
    )


def _vocabulary_table(rows: pd.DataFrame, lang: str, rtl: bool, word_cell) -> list[str]:
    header = [t("tb_vocab_col_word", lang), t("tb_vocab_col_frequency", lang), t("tb_vocab_col_lookup", lang)]
    lines = ["| " + " | ".join(header) + " |", "|---|---|---|"]
    for _, row in rows.iterrows():
        lines.append(
            "| "
            + " | ".join(
                [
                    word_cell(row, lang, rtl),
                    _vocabulary_frequency(row, lang),
                    _vocabulary_lookup_cell(str(row["headword"]), lang, rtl),
                ]
            )
            + " |"
        )
    return lines


def format_vocabulary_section(
    vocabulary: pd.DataFrame | None,
    lang: str = DEFAULT_LANG,
) -> str:
    if vocabulary is None or vocabulary.empty:
        return ""

    rtl = is_rtl(lang)
    lines = [
        f"## {t('tb_vocab_header', lang)}",
        "",
        t("tb_vocab_prompt", lang),
        "",
        t("tb_vocab_lookup_hint", lang),
        "",
        *_vocabulary_table(vocabulary, lang, rtl, _vocabulary_word_cell),
    ]
    return "\n".join(lines) + "\n"


COVERAGE_RING_RADIUS = 26
COVERAGE_RING_CIRCUMFERENCE = 2 * math.pi * COVERAGE_RING_RADIUS


def _count_new_lemma_tokens(
    ledger_keys,
    counted: set[str],
    token_counts: pd.Series,
) -> int:
    # Only lemmas not already counted, so a word handed over twice cannot inflate
    # the running total.
    fresh = {str(key) for key in ledger_keys} - counted
    if not fresh:
        return 0
    counted |= fresh
    return int(token_counts.reindex(sorted(fresh)).fillna(0).sum())


def _coverage_fraction(covered: int, total: int) -> float:
    if total <= 0:
        return 0.0
    return min(max(covered / total, 0.0), 1.0)


def _coverage_ring_svg(fraction: float, label_key: str, aria_key: str, lang: str) -> str:
    # A donut rather than a bar: the covered arc is one circle stroked over a full
    # circle, started at twelve o'clock by rotating it instead of computing a path.
    percent = int(round(fraction * 100))
    dash = COVERAGE_RING_CIRCUMFERENCE * fraction
    gap = COVERAGE_RING_CIRCUMFERENCE - dash
    label = t(label_key, lang)
    aria = t(aria_key, lang, percent=percent).replace('"', "&quot;")
    return (
        '<figure class="coverage-ring">'
        f'<svg viewBox="0 0 64 64" width="76" height="76" role="img" aria-label="{aria}">'
        f'<circle class="coverage-track" cx="32" cy="32" r="{COVERAGE_RING_RADIUS}"/>'
        f'<circle class="coverage-fill" cx="32" cy="32" r="{COVERAGE_RING_RADIUS}"'
        f' stroke-dasharray="{dash:.2f} {gap:.2f}" transform="rotate(-90 32 32)"/>'
        '<text class="coverage-value" x="32" y="32" dy="0.35em" text-anchor="middle">'
        f"{percent}%</text>"
        "</svg>"
        f"<figcaption>{label}</figcaption>"
        "</figure>"
    )


def render_coverage_gauges(
    vocabulary_fraction: float,
    morphology_fraction: float,
    lang: str = DEFAULT_LANG,
) -> str:
    # The book's argument for its own ordering, made once per lesson: the learner
    # sees how much of the corpus they can already read, not just how often this
    # lesson's forms happen to occur.
    #
    # One line of raw HTML on purpose. The document is handed to md_in_html inside
    # the RTL wrapper, and a block with no internal newline passes through whole.
    lead = t("tb_coverage_lead", lang).strip()
    lead_html = f'<p class="coverage-lead">{lead}</p>' if lead else ""
    rings = _coverage_ring_svg(
        vocabulary_fraction, "tb_coverage_vocab", "tb_coverage_vocab_aria", lang
    ) + _coverage_ring_svg(
        morphology_fraction, "tb_coverage_morph", "tb_coverage_morph_aria", lang
    )
    return f'<div class="coverage">{lead_html}<div class="coverage-rings">{rings}</div></div>'


def get_core_function_words(
    combined_df: pd.DataFrame,
    top_n: int = VOCAB_CORE_WORD_COUNT,
) -> pd.DataFrame:
    # The function words a reader meets from the first corpus sentence onward.
    # Their own lessons are ordered by frequency and may land late or fall outside
    # the lesson count altogether, so the book hands them over up front instead.
    if combined_df is None or combined_df.empty or "postag" not in combined_df.columns:
        return pd.DataFrame(columns=_VOCAB_COLUMNS)

    rows = _clean_lemma_rows(combined_df)
    if rows.empty:
        return pd.DataFrame(columns=_VOCAB_COLUMNS)

    # Judged over the lemma rather than the token. Filtering row by row let one
    # mis-tagged token carry a content word in: ὅς is tagged article once in 326
    # and οὐ conjunction once in 315, and both then ranked here on their full
    # lemma count, ahead of genuine function words. The particles join on the
    # list, since the tag alone files most of them under the adverbs.
    is_particle = rows["lemma"].map(is_particle_lemma)
    is_function = ~rows["postag"].astype(str).str.startswith(CONTENT_POS_PREFIXES) | is_particle
    function_share = is_function.groupby(rows["lemma"].map(vocabulary_ledger_key)).transform("mean")
    function_rows = rows[function_share > FUNCTION_WORD_LEMMA_SHARE]
    if function_rows.empty:
        return pd.DataFrame(columns=_VOCAB_COLUMNS)

    core = _aggregate_vocabulary_rows(function_rows).head(top_n)
    # The commonest tag would call δέ an adverb, which is what the list is for.
    core.loc[core["headword"].map(is_particle_lemma), "pos_category"] = "particle"
    return core


def format_core_function_words(core_words: pd.DataFrame | None, lang: str = DEFAULT_LANG) -> str:
    if core_words is None or core_words.empty:
        return ""

    rtl = is_rtl(lang)
    lines = [
        f"## {t('tb_core_words_header', lang)}",
        "",
        t("tb_core_words_note", lang),
        "",
        *_vocabulary_table(core_words, lang, rtl, _vocabulary_tagged_entry),
    ]
    return "\n".join(lines) + "\n"


def _build_sentence_target_rows(
    syllabus_label: str,
    lesson_pos_category: str,
    combined_df: pd.DataFrame,
) -> pd.DataFrame:
    topic_rows = get_topic_rows_for_label(syllabus_label, combined_df)
    if topic_rows.empty:
        return pd.DataFrame()

    topic_rows = topic_rows.dropna(subset=["form", "postag", "sentence_index"]).copy()
    topic_rows["form"] = topic_rows["form"].astype(str).str.strip()
    topic_rows["lemma"] = topic_rows["lemma"].astype(str).str.strip()
    topic_rows["postag"] = topic_rows["postag"].astype(str).str.strip()
    topic_rows = topic_rows[(topic_rows["form"] != "") & (topic_rows["postag"] != "")]
    topic_rows = drop_word_fragments(topic_rows)
    topic_rows = filter_topic_rows_by_lesson_rules(syllabus_label, lesson_pos_category, topic_rows)

    if "token_index" not in topic_rows.columns:
        if "word_id" in topic_rows.columns:
            topic_rows["token_index"] = pd.to_numeric(topic_rows["word_id"], errors="coerce")
        else:
            topic_rows["token_index"] = pd.Series(range(len(topic_rows)), index=topic_rows.index, dtype="int64")

    return topic_rows


# Fragments and one-word "sentences" make poor full-sentence exercises. Three
# words still admitted things like "οὐ γὰρ οὖν.", which show the student nothing.
MIN_EXERCISE_SENTENCE_WORDS = 5

# The picker throws candidates away — too short, or answering nothing the lesson
# has not already answered — so it is handed several times what it will keep.
EXERCISE_SENTENCE_POOL_FACTOR = 5

# A token counts as a word only if it has a letter, so standalone punctuation
# does not count toward the minimum length.
_WORD_TOKEN_RE = re.compile(r"\w", re.UNICODE)


def _count_words(text: str) -> int:
    return sum(1 for token in text.split() if _WORD_TOKEN_RE.search(token))


# Elision and crasis leave pieces of words in the treebanks: "τ-" from τἆλλα,
# "-τε" from οὔτε. They are not forms a student can parse or point to.
_FRAGMENT_FORM_RE = re.compile(r"^[-‐-―]|[-‐-―]$")


def is_word_fragment(form: str) -> bool:
    text = str(form).strip()
    return not text or bool(_FRAGMENT_FORM_RE.search(text))


def drop_word_fragments(rows: pd.DataFrame) -> pd.DataFrame:
    if rows is None or rows.empty or "form" not in rows.columns:
        return rows
    return rows[~rows["form"].map(is_word_fragment)]


# Treebanks mark elision with whichever apostrophe their editor used, so the
# same clause arrives twice as μὰ Δί̓ and μὰ Δί’ unless they are folded together.
_APOSTROPHE_CHARS = "'’ʼʽ̓᾽´"
_APOSTROPHE_RE = re.compile(f"[{_APOSTROPHE_CHARS}]")


def _fold_apostrophes(text: str) -> str:
    # NFC first: a combining smooth breathing belongs to its vowel and composes
    # away, leaving loose only the ones marking an elided word — the apostrophes
    # this is meant to drop.
    return _APOSTROPHE_RE.sub("", unicodedata.normalize("NFC", str(text)))


def _normalize_answer_word(word: str) -> str:
    # A final acute turns grave in running text, so δέ and δὲ are one word and
    # should not be answered twice in the same key.
    return _fold_apostrophes(to_citation_accent(str(word).strip()).lower())


def _normalize_sentence_key(text: str) -> str:
    return _fold_apostrophes(re.sub(r"\s+", " ", str(text).strip()))


def _pick_unique_exercise_sentences(
    topic_sentences: pd.DataFrame,
    topic_rows: pd.DataFrame,
    max_sentences: int = 20,
) -> tuple[pd.DataFrame, dict[object, pd.DataFrame]]:
    if topic_sentences is None or topic_sentences.empty or topic_rows is None or topic_rows.empty:
        return pd.DataFrame(), {}

    selected_sentence_ids = []
    selected_targets_by_sentence: dict[object, pd.DataFrame] = {}
    used_sentence_texts: set[str] = set()
    used_answer_words: set[str] = set()

    # Sort the target rows once and index them by position per sentence. Sorting
    # and copying every sentence group up front cost ~135k tiny sort_values per
    # build, though only a handful of sentences are ever selected.
    topic_rows = topic_rows.sort_values("token_index")
    group_positions = topic_rows.groupby("sentence_index", sort=False).indices
    target_forms = topic_rows["form"].to_numpy(dtype=object)

    sentence_index_values = topic_sentences["sentence_index"].to_numpy()
    sentence_text_values = topic_sentences.get("sentence_text")
    sentence_text_values = (
        sentence_text_values.astype(str).to_numpy()
        if sentence_text_values is not None
        else np.array([""] * len(topic_sentences), dtype=object)
    )

    for sentence_index, sentence_text in zip(sentence_index_values, sentence_text_values):
        if len(selected_sentence_ids) >= max_sentences:
            break

        sentence_text_key = _normalize_sentence_key(sentence_text)

        if not sentence_text_key or sentence_text_key in used_sentence_texts:
            continue

        if _count_words(sentence_text_key) < MIN_EXERCISE_SENTENCE_WORDS:
            continue

        positions = group_positions.get(sentence_index)
        if positions is None or len(positions) == 0:
            continue

        # A sentence earns its place by introducing a form no earlier sentence
        # answered — but once chosen it keeps every target it contains, or the
        # key would tell the student that a word they correctly found is wrong.
        answer_positions = [
            position
            for position in positions
            if _normalize_answer_word(target_forms[position])
        ]
        if not answer_positions:
            continue

        introduces_new_form = any(
            _normalize_answer_word(target_forms[position]) not in used_answer_words
            for position in answer_positions
        )
        if not introduces_new_form:
            continue

        chosen_targets = topic_rows.iloc[answer_positions]
        selected_sentence_ids.append(sentence_index)
        selected_targets_by_sentence[sentence_index] = chosen_targets
        used_sentence_texts.add(sentence_text_key)
        used_answer_words.update(_normalize_answer_word(form) for form in chosen_targets["form"].tolist())

    if not selected_sentence_ids:
        return pd.DataFrame(), {}

    selected_sentences = topic_sentences[topic_sentences["sentence_index"].isin(selected_sentence_ids)].copy()
    selected_sentences = selected_sentences.drop_duplicates(subset=["sentence_text"], keep="first")
    return selected_sentences, selected_targets_by_sentence


def _format_exercise_nonverb(
    lesson_pos_category: str,
    exercise_sentences: pd.DataFrame,
    sentence_target_rows: Mapping[Any, pd.DataFrame],
    lang: str = DEFAULT_LANG,
) -> str:
    if exercise_sentences is None or exercise_sentences.empty:
        return ""

    rtl = is_rtl(lang)
    pos_label = _pos_label(lesson_pos_category, lang)

    # Answers come first: whether anything could be parsed decides which prompt the
    # exercise opens with, and a preposition lesson must not ask for a parse.
    answer_lines = []
    parsed_any = False
    for _, row in exercise_sentences.iterrows():
        answers, parsed = _parse_answers_for_rows(sentence_target_rows.get(row["sentence_index"]), lang, rtl)
        parsed_any = parsed_any or parsed
        answer_lines.append(" | ".join(answers) if answers else t("tb_no_target_form", lang))

    prompt_key = "tb_ex_sentences_prompt" if parsed_any else "tb_ex_sentences_identify_prompt"
    lines = [
        f"### {t('tb_ex_sentences_header', lang)}",
        "",
        t(prompt_key, lang, pos_label=pos_label),
        "",
    ]
    items = [
        f"{idx}. {_ltr_isolate(str(row['sentence_text']), rtl)}{_citation_suffix(row, rtl)}"
        for idx, (_, row) in enumerate(exercise_sentences.iterrows(), 1)
    ]
    lines += _ltr_block(items, rtl)
    lines.append("")
    lines.append(f"#### {t('tb_answer_key_header', lang)}")
    lines.append("")
    lines += [f"{idx}. {answer}" for idx, answer in enumerate(answer_lines, 1)]

    lines.append("")
    return "\n".join(lines)


def _format_exercise_verb(
    exercise_sentences: pd.DataFrame,
    sentence_verb_rows: Mapping[Any, pd.DataFrame],
    lang: str = DEFAULT_LANG,
) -> str:
    if exercise_sentences is None or exercise_sentences.empty:
        return ""

    rtl = is_rtl(lang)
    lines = [
        f"### {t('tb_ex_sentences_header', lang)}",
        "",
        t("tb_ex_sentences_verb_prompt", lang),
        "",
    ]

    items = []
    for idx, (_, row) in enumerate(exercise_sentences.iterrows(), 1):
        sentence_rows = sentence_verb_rows.get(row["sentence_index"])
        forms = set()
        if sentence_rows is not None and not sentence_rows.empty:
            forms = set(sentence_rows["form"].tolist())
        marked = mark_topic_words_in_sentence(row["sentence_text"], forms)
        items.append(f"{idx}. {_ltr_isolate(marked, rtl)}{_citation_suffix(row, rtl)}")
    lines += _ltr_block(items, rtl)

    lines.append("")
    lines.append(f"#### {t('tb_answer_key_header', lang)}")
    lines.append("")

    for idx, (_, row) in enumerate(exercise_sentences.iterrows(), 1):
        answers, _ = _parse_answers_for_rows(sentence_verb_rows.get(row["sentence_index"]), lang, rtl)
        lines.append(f"{idx}. " + (" | ".join(answers) if answers else t("tb_no_marked_verbs", lang)))

    lines.append("")
    return "\n".join(lines)


def generate_exercises_for_topic(
    syllabus_label: str,
    lesson_pos_category: str,
    combined_df: pd.DataFrame,
    sentences_df: pd.DataFrame,
    num_sentences: int = 20,
    lang: str = DEFAULT_LANG,
    topic_words: pd.DataFrame | None = None,
    known_lemmas: set[str] | None = None,
) -> str:
    exercise_blocks = []

    if topic_words is None:
        topic_words = get_topic_words(syllabus_label, lesson_pos_category, combined_df, num_words=15)
    words_exercise = format_parsing_exercise(topic_words, lang=lang)
    if words_exercise:
        exercise_blocks.append(words_exercise)

    topic_sentences = get_topic_sentences(
        syllabus_label=syllabus_label,
        combined_df=combined_df,
        sentences_df=sentences_df,
        num_sentences=num_sentences * EXERCISE_SENTENCE_POOL_FACTOR,
        known_lemmas=known_lemmas,
    )

    if not topic_sentences.empty:
        topic_rows = _build_sentence_target_rows(syllabus_label, lesson_pos_category, combined_df)

        if not topic_rows.empty:
            selected_sentences, selected_targets_by_sentence = _pick_unique_exercise_sentences(
                topic_sentences,
                topic_rows,
                max_sentences=num_sentences,
            )

            if selected_sentences.empty:
                return "\n".join(exercise_blocks)

            if lesson_pos_category == "verb":
                exercise_blocks.append(_format_exercise_verb(selected_sentences, selected_targets_by_sentence, lang=lang))
            else:
                exercise_blocks.append(
                    _format_exercise_nonverb(lesson_pos_category, selected_sentences, selected_targets_by_sentence, lang=lang)
                )

    return "\n".join(exercise_blocks)


# Lesson files carry their own display title, so titles are normalized on the way
# into the contents: the textbook already numbers and labels each entry, so a
# "Lesson:" prefix is redundant and emphasis markup makes one entry shout.
LESSON_TITLE_PREFIX_RE = re.compile(
    r"^\s*(?:lesson|module|unit|chapter|درس|بخش|مبحث|فصل)\s*[:：]\s*",
    re.IGNORECASE,
)


def normalize_lesson_title(title: str) -> str:
    cleaned = re.sub(r"^\s*#{1,6}\s+", "", str(title))
    cleaned = LESSON_TITLE_PREFIX_RE.sub("", cleaned)
    cleaned = re.sub(r"\*{1,3}(.+?)\*{1,3}", r"\1", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" #*_")
    return cleaned


# Lessons that read as a contrast with a class the learner already knows, which
# frequency order alone can put first. A lesson of the kind on the left waits
# for one lesson of the kind on the right; with none in the syllabus it keeps
# its frequency position.
LESSON_PREREQUISITE_KINDS: dict[str, str] = {
    # An irregular form only reads as irregular against a paradigm.
    "irregular verb": "regular verb",
    "irregular noun": "regular noun",
    "irregular adjective": "regular adjective",
    # An adjective agrees with a noun, so a noun class comes first.
    "adjective": "noun",
}

REGULAR_NOUN_LESSONS = frozenset(
    normalize_frequency_row_name(label) for code, label in NOUN_DECLENSION_LABELS.items() if code != "noun-3-other"
)
IRREGULAR_NOUN_LESSON = normalize_frequency_row_name(NOUN_DECLENSION_LABELS["noun-3-other"])
REGULAR_ADJECTIVE_LESSONS = frozenset(
    normalize_frequency_row_name(label)
    for code, label in ADJECTIVE_DECLENSION_LABELS.items()
    if code != "adj-3-two-end"
)
IRREGULAR_ADJECTIVE_LESSON = normalize_frequency_row_name(ADJECTIVE_DECLENSION_LABELS["adj-3-two-end"])


def lesson_kinds(lesson: Mapping[str, Any]) -> frozenset[str]:
    # The prerequisite kinds a lesson belongs to, as a dependent and as a
    # prerequisite for others. Case-mode noun/adjective lessons name a case
    # rather than an inflection class, so they belong to none and stay put.
    if lesson.get("is_starter"):
        return frozenset()
    normalized = normalize_frequency_row_name(str(lesson.get("label", "")))
    if normalized in REGULAR_NOUN_LESSONS:
        return frozenset({"noun", "regular noun"})
    if normalized == IRREGULAR_NOUN_LESSON:
        return frozenset({"noun", "irregular noun"})
    if normalized in REGULAR_ADJECTIVE_LESSONS:
        return frozenset({"adjective", "regular adjective"})
    if normalized == IRREGULAR_ADJECTIVE_LESSON:
        return frozenset({"adjective", "irregular adjective"})
    if str(lesson.get("pos_category", "")) != "verb":
        return frozenset()
    if normalized == normalize_frequency_row_name(IRREGULAR_LESSON_LABEL):
        return frozenset({"verb", "irregular verb"})
    if normalized == normalize_frequency_row_name(DEPONENT_LESSON_LABEL):
        return frozenset({"verb"})  # a concept lesson, not a paradigm to contrast against
    return frozenset({"verb", "regular verb"})


def apply_lesson_prerequisite_order(lesson_data: list[dict[str, Any]]) -> list[dict[str, Any]]:
    # Move each dependent lesson just past the first lesson satisfying each
    # prerequisite it lacks; everything else keeps its frequency position.
    kinds = [lesson_kinds(lesson) for lesson in lesson_data]

    for _ in range(len(lesson_data)):
        for position, lesson_kind_set in enumerate(kinds):
            target = position
            for kind in lesson_kind_set:
                required = LESSON_PREREQUISITE_KINDS.get(kind)
                if required is None:
                    continue
                first = next((index for index, other in enumerate(kinds) if required in other), None)
                if first is None or first < position:
                    continue
                target = max(target, first)
            if target == position:
                continue
            lesson_data.insert(target, lesson_data.pop(position))
            kinds.insert(target, kinds.pop(position))
            break
        else:
            break

    for index, lesson in enumerate(lesson_data, 1):
        lesson["rank"] = index
    return lesson_data


# A concept lesson teaches what no single paradigm carries, so it has no
# frequency row and no position of its own: it is placed against what the reader
# has already met, and left out of a book that never gets there rather than
# taught over forms nobody has seen. Inserted after the frequency cut is taken,
# so it costs no lesson slot.
#
# "requires" is matched as a substring of the normalized label, which gates on
# tense and mood but not on voice or conjugation: the optative is met at the
# first optative lesson of any kind. "after" says which satisfying lesson to
# follow — the last one for a lesson that needs all of them in hand, the first
# for one that only contrasts with a class the reader now knows.
#
# "body_only" marks a lesson that is entirely its own file: no generated
# vocabulary, no generated exercises, no coverage gauge. Nothing above the word
# is a paradigm, so there are no forms to hand over and nothing for the
# morphology gauge to move.
CONCEPT_LESSONS: dict[str, dict[str, Any]] = {
    DEPONENT_LESSON_LABEL: {
        "filename": DEPONENT_LESSON_FILENAME,
        "pos_category": "verb",
        "requires": ("middle",),
        "after": "first",
        "frequency": "deponent_count",
    },
    # The seven types between them need three indicative tenses and both
    # non-indicative moods. The future is left out although the future more
    # vivid uses it: it is rarer than the optative in both test corpora and
    # would delay the whole lesson for one row of the table. εἰ, ἐάν and ἄν are
    # function words the lesson introduces itself, not lessons to wait for.
    CONDITIONAL_LESSON_LABEL: {
        "filename": CONDITIONAL_LESSON_FILENAME,
        "pos_category": "verb",
        "requires": ("present_indicative", "imperfect_indicative", "aorist", "subjunctive", "optative"),
        "after": "last",
        "subtitle_key": "tb_module_type_syntax",
        "body_only": True,
    },
    # Four lessons of the nominal system, whatever the syllabus mode calls them.
    # Declension mode has one per inflection class; case mode has exactly four,
    # since vocative merges into nominative in MERGED_SYLLABUS_LABELS, so the
    # same number reads as "once the cases are done" there.
    DIDAKTA_LESSON_LABEL: {
        "filename": DIDAKTA_LESSON_FILENAME,
        "pos_category": "reference",
        "requires_nominal_count": 4,
        "subtitle_key": "tb_module_type_reference",
        "body_only": True,
    },
}


def _concept_lesson_position(lesson_data: list[dict[str, Any]], entry: Mapping[str, Any]) -> int | None:
    # Where the lesson goes, or None when the book never satisfies it.
    nominal_count = entry.get("requires_nominal_count")
    if nominal_count:
        nominals = [index for index, lesson in enumerate(lesson_data) if lesson["pos_category"] == "noun/adjective"]
        if len(nominals) < nominal_count:
            return None
        return nominals[nominal_count - 1] + 1

    positions = []
    for requirement in entry.get("requires", ()):
        found = next(
            (
                index
                for index, lesson in enumerate(lesson_data)
                if not lesson["is_starter"] and requirement in normalize_frequency_row_name(str(lesson["label"]))
            ),
            None,
        )
        if found is None:
            return None
        positions.append(found)

    if not positions:
        return None
    return (max(positions) if entry.get("after") == "last" else min(positions)) + 1


def insert_concept_lessons(
    lesson_data: list[dict[str, Any]], combined_df: pd.DataFrame | None = None
) -> list[dict[str, Any]]:
    for label, entry in CONCEPT_LESSONS.items():
        position = _concept_lesson_position(lesson_data, entry)
        if position is None:
            continue

        frequency: int | str = "—"
        if entry.get("frequency") == "deponent_count" and combined_df is not None and "is_deponent" in combined_df.columns:
            frequency = int(combined_df["is_deponent"].sum())

        lesson_data.insert(
            position,
            {
                "rank": 0,
                "label": label,
                "pos_category": entry["pos_category"],
                "frequency": frequency,
                "filename": entry["filename"],
                "is_starter": False,
                "body_only": bool(entry.get("body_only")),
                "subtitle_key": entry.get("subtitle_key"),
            },
        )

    for index, lesson in enumerate(lesson_data, 1):
        lesson["rank"] = index
    return lesson_data


def _split_lesson_title(lesson_text: str) -> tuple[str | None, str]:
    # (leading heading, remaining markdown). Leading YAML frontmatter is dropped
    # so its metadata does not leak into the rendered textbook.
    lines = lesson_text.splitlines()
    start = 0
    if lines and lines[0].strip() == "---":
        for index in range(1, len(lines)):
            if lines[index].strip() in ("---", "..."):
                start = index + 1
                break
    for index in range(start, len(lines)):
        stripped = lines[index].strip()
        if not stripped:
            continue
        match = re.match(r"#{1,6}\s+(.+)", stripped)
        if match:
            title = match.group(1).strip()
            body = "\n".join(lines[index + 1:]).lstrip("\n")
            return (title or None), body
        break
    return None, "\n".join(lines[start:]).lstrip("\n")


def _render_source_summary(
    source_summary: Mapping[str, Any],
    lesson_count: int,
    syllabus_mode: str,
    lang: str = DEFAULT_LANG,
) -> list[str]:
    # Markdown for the "About This Textbook" front matter: the build setting, the
    # works and corpora behind it, and the textbook's own licence. Empty when
    # there is nothing to show, so the caller can skip the section.
    works = list(source_summary.get("works") or [])
    corpora = list(source_summary.get("corpora") or [])
    has_custom = bool(source_summary.get("has_custom_sources"))
    if not works and not corpora and not has_custom:
        return []

    mode_key = (
        "tb_syllabus_mode_declension" if syllabus_mode == "declension" else "tb_syllabus_mode_case"
    )

    lines: list[str] = [f"# {t('tb_about_header', lang)}", ""]

    setting_body = t(
        "tb_setting_body",
        lang,
        lesson_count=int(lesson_count),
        work_count=int(source_summary.get("work_count", len(works))),
        token_count=int(source_summary.get("token_count", 0)),
        mode=t(mode_key, lang),
    )
    lines.append(f"**{t('tb_setting_label', lang)}** {setting_body}")
    lines.append("")

    # Texts included, grouped by author.
    if works:
        lines.append(f"**{t('tb_texts_label', lang)}**")
        lines.append("")
        grouped: dict[str, list[str]] = {}
        for author, work in works:
            key = author or t("tb_unknown_author", lang)
            grouped.setdefault(key, []).append(work)
        for author in sorted(grouped):
            lines.append(f"- **{author}** — {', '.join(grouped[author])}")
        lines.append("")

    # Corpora used: name — license (source link). Description.
    if corpora or has_custom:
        lines.append(f"**{t('tb_corpora_label', lang)}**")
        lines.append("")
        for corpus in corpora:
            name = (corpus.get("name") or "").strip()
            if not name:
                continue
            license_name = (corpus.get("license") or "").strip()
            head = [
                t("tb_corpus_line", lang, name=name, license=license_name)
                if license_name
                else f"**{name}**"
            ]
            source_url = (corpus.get("source_url") or "").strip()
            if source_url:
                head.append(f"([{t('tb_corpus_source_label', lang)}]({source_url}))")
            line = f"- {' '.join(head)}"
            desc_key = f"tb_corpus_desc_{corpus.get('id')}"
            desc = t(desc_key, lang)
            if desc and desc != desc_key:
                line += f". {desc}"
            lines.append(line)
        if has_custom:
            lines.append(f"- {t('tb_corpus_custom', lang)}")
        lines.append("")

    # Where the answer keys come from, and what the coverage figures do and do not
    # measure. Same reasoning as the licence below: the file outlives the session,
    # so a reader who never saw the app has no other way to learn any of it.
    lines.append(f"**{t('tb_caveats_label', lang)}** {t('tb_caveats_body', lang)}")
    lines.append("")

    # Terms for the textbook itself. An exported file leaves the app entirely, so
    # this is the only place a later reader can learn them, and the ShareAlike
    # corpora above oblige us to state them.
    lines.append(f"**{t('tb_license_label', lang)}** {t('tb_copyright', lang, year=date.today().year)}")
    lines.append("")
    lines.append(t("tb_license_body", lang))
    lines.append("")

    return lines


# Cover logo. The markdown export links the committed PNG so the .md file stays
# small and readable; the HTML export swaps this src for an inlined data URI so a
# downloaded file still shows the logo offline (see generate_textbook_html).
TEXTBOOK_LOGO_URL = (
    "https://raw.githubusercontent.com/farnoosh-shamsian/didaskalos/main/docs/assets/logo-el-ink.png"
)

_HEADING_SLUG_STRIP_RE = re.compile(r"[^\w\s-]", re.UNICODE)
_HEADING_SLUG_SPACE_RE = re.compile(r"\s+")


def heading_slug(text: str, separator: str = "-") -> str:
    # GitHub-compatible anchor, used for both the contents links and the ids
    # rendered into the HTML. Python-Markdown's own slugify is ASCII-only, which
    # would collapse every Greek and Persian heading to a bare number.
    slug = _HEADING_SLUG_STRIP_RE.sub("", str(text)).strip().lower()
    return _HEADING_SLUG_SPACE_RE.sub(separator, slug)


def _title_page_works_line(source_summary: Mapping[str, Any] | None, lang: str) -> str | None:
    # "Author — Work1, Work2; Author2 — Work3", grouped and separated the same way
    # as the "Texts included" list further down (_render_source_summary), so the
    # two mentions of the same works read consistently.
    if not source_summary:
        return None
    works = list(source_summary.get("works") or [])
    if not works:
        return None
    grouped: dict[str, list[str]] = {}
    for author, work in works:
        key = author or t("tb_unknown_author", lang)
        grouped.setdefault(key, []).append(work)
    return "; ".join(f"{author} — {', '.join(titles)}" for author, titles in grouped.items())


def _render_title_page(lang: str, source_summary: Mapping[str, Any] | None = None) -> list[str]:
    # Cover: logo, title, included works, build date. The blank lines inside the
    # div are what keep GitHub and md_in_html parsing the markdown within it, and
    # align="center" rather than a style attribute because GitHub strips styles.
    lines = [
        '<div class="title-page" align="center" markdown="1">',
        "",
        f'<img class="textbook-logo" src="{TEXTBOOK_LOGO_URL}" alt="Didaskalos" width="360">',
        "",
        f"# {t('tb_doc_title', lang)}",
        "",
    ]
    works_line = _title_page_works_line(source_summary, lang)
    if works_line:
        lines.append(works_line)
        lines.append("")
    lines.append(t("tb_built_on", lang, date=date.today().isoformat()))
    lines.append("")
    lines.append("</div>")
    lines.append("")
    return lines


# Sections of the syntax reference name themselves in the lesson file, so the
# authored prose decides what is covered and in what order; the corpus only
# fills in what it can attest.
SYNTAX_PLACEHOLDER_RE = re.compile(r"<!--\s*syntax:([a-z_]+)\s*-->")


def _syntax_category_label(category: str, lang: str) -> str:
    # Clause categories are a tense/mood pair and reuse the parsing vocabulary;
    # named constructions carry a syn_ key of their own.
    if "|" in category:
        tense, mood = category.split("|", 1)
        return f"{_feature_label(tense, lang)} {_feature_label(mood, lang)}"
    key = f"syn_{category}"
    value = t(key, lang)
    return category.replace("_", " ") if value == key else value


def _syntax_table(rows: pd.DataFrame, entry_name: str, sentence_text, lang: str) -> list[str]:
    rtl = is_rtl(lang)
    order = constructions.SYNTAX_ENTRIES[entry_name].get("order")
    counts = rows["category"].value_counts()
    if order:
        categories = [c for c in order if c in counts] + [c for c in counts.index if c not in order]
    else:
        categories = list(counts.index)

    lines = [
        f"| {t('tb_syntax_col_category', lang)} | {t('tb_syntax_col_count', lang)} | {t('tb_syntax_col_example', lang)} |",
        "| --- | --- | --- |",
    ]
    for category in categories:
        subset = rows[rows["category"] == category]
        if len(subset) < constructions.MIN_ATTESTED:
            continue
        # Shortest first: the briefest attested sentence is the one a reader can
        # take in whole, and length is the gate that mattered most in testing.
        subset = subset.sort_values("words")
        example = ""
        for row in subset.itertuples(index=False):
            text = sentence_text.get(row.sentence_id)
            if text:
                example = _ltr_isolate(text, rtl) + _citation_suffix(row._asdict(), rtl)
                break
        lines.append(
            f"| {_syntax_category_label(category, lang)} | {len(subset)} | {example} |"
        )

    if len(lines) == 2:
        return [f"*{t('tb_syntax_unattested', lang)}*"]
    return lines


def detect_syntax(
    combined_df: pd.DataFrame | None,
    sentences_df: pd.DataFrame | None,
    fmt: str = "agdt-xml",
) -> tuple[dict, dict] | None:
    # Every construction the corpus attests, plus the sentences to quote them
    # from. Built once and handed to each body that carries a placeholder: the
    # ClauseIndex behind it costs more than the classification does.
    if combined_df is None or combined_df.empty:
        return None

    sentence_text = {}
    if sentences_df is not None and not sentences_df.empty:
        sentence_text = dict(zip(sentences_df["sentence_id"], sentences_df["sentence_text"]))

    return constructions.all_reference_rows(combined_df, fmt), sentence_text


def fill_syntax_placeholders(body: str, detection: tuple[dict, dict] | None, lang: str = DEFAULT_LANG) -> str:
    # Each <!--syntax:name--> replaced by what the selected texts actually
    # attest. A placeholder naming an entry that is not attested says so rather
    # than disappearing, so the book never implies a construction is absent from
    # Greek when it is only absent from this corpus.
    if detection is None:
        return SYNTAX_PLACEHOLDER_RE.sub("", body)

    detected, sentence_text = detection

    def replace(match: re.Match) -> str:
        name = match.group(1)
        rows = detected.get(name)
        if rows is None or rows.empty:
            return f"*{t('tb_syntax_unattested', lang)}*"
        return "\n".join(_syntax_table(rows, name, sentence_text, lang))

    return SYNTAX_PLACEHOLDER_RE.sub(replace, body)


def format_passage_appendix(passages: list[dict], lang: str = DEFAULT_LANG) -> str:
    # The passages themselves. Greek is LTR-isolated here rather than left to the
    # HTML export, because wrap_ltr_runs_in_html only runs on that path and the
    # markdown download would otherwise reorder the words on an RTL page.
    rtl = is_rtl(lang)
    lines = [t("tb_passages_intro", lang), ""]

    for number, passage in enumerate(passages, 1):
        citation = passage.get("citation") or ""
        if citation:
            heading = t("tb_passage_heading", lang, number=number, citation=_ltr_isolate(citation, rtl))
        else:
            heading = str(number)
        lines.append(f"## {heading}")
        lines.append("")
        lines += _ltr_block([_ltr_isolate(passage["text"], rtl)], rtl)
        lines.append("")

    return "\n".join(lines).rstrip()


def generate_textbook_markdown(
    frequency_syllabus: pd.DataFrame,
    grammar_folder: str | Path,
    lesson_count: int = 40,
    combined_df: pd.DataFrame | None = None,
    syllabus_mode: str = "case",
    lang: str = DEFAULT_LANG,
    source_summary: Mapping[str, Any] | None = None,
) -> str:
    # Keep in sync with STARTER_LESSON_FILES in app.py, which drives the download.
    starter_modules = [
        "about",
        "alphabet",
        "introduction_nouns",
        "introduction_adjectives",
        "introduction_verbs",
        DICTIONARY_LESSON_MODULE,
        DIALECTS_LESSON_MODULE,
    ]
    rtl = is_rtl(lang)
    if syllabus_mode == "declension":
        intro_text = t("tb_intro_declension", lang)
        mode_key = "tb_syllabus_mode_declension"
    else:
        intro_text = t("tb_intro_case", lang)
        mode_key = "tb_syllabus_mode_case"

    lesson_rows = frequency_syllabus[
        frequency_syllabus["syllabus"].notna() & (frequency_syllabus["syllabus"] != "NA")
    ].head(int(lesson_count))

    # Every analyzable token in the corpus, whether or not its lesson made the cut.
    # build_frequency_syllabus has already dropped the rows no lesson could teach,
    # so a complete book approaches 100% instead of stalling at an unexplained
    # ceiling.
    total_forms = int(
        pd.to_numeric(frequency_syllabus["frequency"], errors="coerce").fillna(0).sum()
    )

    lesson_data = []
    rank = 0

    for module_name in starter_modules:
        rank += 1
        lesson_data.append(
            {
                "rank": rank,
                "label": module_name,
                "pos_category": "module",
                "frequency": "core",
                "filename": f"{module_name}.md",
                "is_starter": True,
            }
        )

    for _, row in lesson_rows.iterrows():
        rank += 1
        label = row["syllabus"]
        pos_category = row.get("pos_category", "other")
        freq = row["frequency"]
        filename = syllabus_to_filename(label)

        if filename is None:
            continue

        lesson_data.append(
            {
                "rank": rank,
                "label": label,
                "pos_category": pos_category,
                "frequency": freq,
                "filename": filename,
                "is_starter": False,
            }
        )

    apply_lesson_prerequisite_order(lesson_data)
    insert_concept_lessons(lesson_data, combined_df)

    grammar_folder = Path(grammar_folder)

    # Bodies are loaded up front so the contents can use each file's own H1 title
    # (localized in a translated folder) rather than the raw syllabus label.
    for lesson in lesson_data:
        lesson["display_label"] = lesson["label"]
        lesson_path = grammar_folder / lesson["filename"]
        if not lesson_path.exists():
            lesson["body"] = f"*{t('tb_module_not_found', lang, filename=lesson['filename'])}*"
            continue
        try:
            title, body = _split_lesson_title(lesson_path.read_text(encoding="utf-8"))
        except Exception as exc:
            lesson["body"] = f"*{t('tb_error_reading', lang, error=exc)}*"
            continue
        if title:
            lesson["display_label"] = normalize_lesson_title(title) or lesson["label"]
        lesson["body"] = body

    markdown_content = []
    markdown_content.extend(_render_title_page(lang, source_summary))
    markdown_content.append(intro_text)
    markdown_content.append("")

    if source_summary:
        markdown_content.extend(
            _render_source_summary(source_summary, lesson_count, syllabus_mode, lang)
        )

    markdown_content.append(f"# {t('tb_toc_header', lang)}")
    markdown_content.append("")

    # Linked to the heading each lesson will emit below, slugged the same way.
    for lesson in lesson_data:
        anchor = heading_slug(f"{lesson['rank']}. {lesson['display_label']}")
        markdown_content.append(f"{lesson['rank']}. [{lesson['display_label']}](#{anchor})")

    # Held open for the back matter, which cannot be built until known_lemmas is
    # final and that only happens after the lesson loop. A slot no section claims
    # is dropped once they have all run, so a blank line never splits the contents
    # into two lists and restarts the numbering.
    syntax_toc_slot = len(markdown_content)
    markdown_content.append("")
    passages_toc_slot = len(markdown_content)
    markdown_content.append("")
    next_toc_slot = len(markdown_content)
    markdown_content.append("")

    markdown_content.append("")

    working_combined_df = None
    working_sentences_df = None
    known_lemmas: set[str] = set()
    introduced_lemmas: set[str] = set()
    citation_index: dict[str, dict[str, str]] = {}
    core_function_words = pd.DataFrame()

    # Running coverage of the corpus, carried across the lesson loop.
    ledger_token_counts = pd.Series(dtype="int64")
    total_greek_tokens = 0
    counted_lemma_keys: set[str] = set()
    covered_tokens = 0
    covered_forms = 0

    if combined_df is not None and not combined_df.empty:
        # The passed frame is worked on directly: a full .copy() duplicated the
        # whole token table and drove the OOM kills, whereas the two derived
        # columns cost a few MB. They also show up in the combined-rows export.
        working_combined_df = combined_df

        # Counted over the distinct lemmas rather than every token: the Greek test
        # then runs a few thousand times instead of a few hundred thousand, and the
        # counts are needed here even when lemma_frequency survives from an earlier
        # build, because the frame is worked on in place.
        all_lemma_counts = working_combined_df["lemma"].value_counts()
        greek_mask = np.array(all_lemma_counts.index.map(is_greek_lemma), dtype=bool)
        lemma_counts = all_lemma_counts[greek_mask]

        if "lemma_frequency" not in working_combined_df.columns:
            working_combined_df["lemma_frequency"] = working_combined_df["lemma"].map(lemma_counts).fillna(0)

        # Coverage is counted in running text, over the same merged identity the
        # vocabulary ledger uses, so λέγω1 and λέγω3 are one word here too.
        ledger_token_counts = lemma_counts.groupby(
            lemma_counts.index.map(vocabulary_ledger_key)
        ).sum()
        total_greek_tokens = int(lemma_counts.sum())

        working_sentences_df = assemble_sentences(working_combined_df)
        if not working_sentences_df.empty:
            working_sentences_df["sentence_index"] = range(len(working_sentences_df))
            working_combined_df["sentence_index"] = working_combined_df.groupby("sentence_id", sort=False).ngroup()
            working_sentences_df = add_sentence_scores(working_sentences_df, working_combined_df)

        known_lemmas = build_known_lemma_seed(working_combined_df)
        citation_index = build_lemma_citation_index(working_combined_df)
        # The last starter module hands the closed classes over before the corpus
        # lessons begin, because their own lessons are frequency-ordered and may
        # land late or not at all. The ledger starts from the words that table
        # printed.
        core_function_words = get_core_function_words(working_combined_df)
        if not core_function_words.empty:
            introduced_lemmas.update(core_function_words["ledger_key"])
            covered_tokens += _count_new_lemma_tokens(
                core_function_words["ledger_key"], counted_lemma_keys, ledger_token_counts
            )

    # Shared by the concept lessons that carry a placeholder and by the appendix.
    syntax_detection = detect_syntax(working_combined_df, working_sentences_df)

    for lesson in lesson_data:
        # A rule then an H1: the lesson title outranks every heading its own body
        # uses, which is what tells a reader one lesson has ended and another
        # begun. The HTML export turns the same H1 into a banded, page-breaking
        # heading.
        markdown_content.append("---")
        markdown_content.append("")
        markdown_content.append(f"# {lesson['rank']}. {lesson['display_label']}")
        if lesson.get("is_starter"):
            markdown_content.append(t("tb_module_type_core", lang))
        elif lesson.get("subtitle_key"):
            # Built above the word, so there is no part of speech to name and no
            # form count to print.
            markdown_content.append(t(lesson["subtitle_key"], lang))
        else:
            markdown_content.append(t("tb_pos_family", lang, pos=_pos_label(lesson["pos_category"], lang)))
            markdown_content.append(t("tb_frequency", lang, frequency=lesson["frequency"]))
        markdown_content.append("")

        # Filled in at the end of this iteration. The figure worth showing is what
        # the reader knows once the lesson is done, and the ledger only reaches
        # that state further down the loop. Sitting after the subtitle keeps the
        # h1 + p rule matching, so the grey subtitle keeps its styling.
        coverage_slot = len(markdown_content)
        markdown_content.append("")

        markdown_content.append("")
        markdown_content.append(fill_syntax_placeholders(lesson["body"], syntax_detection, lang))

        if lesson.get("is_starter") or lesson.get("body_only"):
            if lesson["label"] == DIALECTS_LESSON_MODULE:
                core_words_table = format_core_function_words(core_function_words, lang=lang)
                if core_words_table:
                    markdown_content.append("")
                    markdown_content.append(core_words_table)
        else:
            corpus_available = (
                working_combined_df is not None
                and working_sentences_df is not None
                and not working_sentences_df.empty
            )
            vocabulary_markdown = ""
            exercises = ""

            if corpus_available:
                vocabulary = get_lesson_vocabulary(
                    lesson["label"],
                    lesson["pos_category"],
                    working_combined_df,
                    introduced_lemmas=introduced_lemmas,
                    citation_index=citation_index,
                )
                vocabulary_markdown = format_vocabulary_section(vocabulary, lang=lang)

                # Words this lesson handed over count from here on: the ledger keeps
                # later lessons off them, and known_lemmas steers sentence selection
                # toward what the student has already met.
                if not vocabulary.empty:
                    introduced_lemmas.update(vocabulary["ledger_key"])
                    known_lemmas.update(vocabulary["ledger_key"])
                    covered_tokens += _count_new_lemma_tokens(
                        vocabulary["ledger_key"], counted_lemma_keys, ledger_token_counts
                    )

                topic_words = get_topic_words(
                    lesson["label"], lesson["pos_category"], working_combined_df, num_words=15
                )
                exercises = generate_exercises_for_topic(
                    lesson["label"],
                    lesson["pos_category"],
                    working_combined_df,
                    working_sentences_df,
                    lang=lang,
                    topic_words=topic_words,
                    known_lemmas=known_lemmas,
                )
                if not topic_words.empty:
                    known_lemmas.update(
                        vocabulary_ledger_key(str(lemma)) for lemma in topic_words["lemma"]
                    )

            if vocabulary_markdown:
                markdown_content.append("")
                markdown_content.append(vocabulary_markdown)

            markdown_content.append("")
            markdown_content.append(f"## {t('tb_exercises_header', lang)}")
            markdown_content.append("")

            if not corpus_available:
                markdown_content.append(f"*{t('tb_exercises_unavailable', lang)}*")
            elif exercises:
                markdown_content.append(exercises)
            else:
                markdown_content.append(f"*{t('tb_no_exercises', lang, label=lesson['display_label'])}*")

            # Deponency is a lexical class, not a paradigm: its tokens were already
            # counted under the voice lessons they appear in, so the concept lesson
            # adds no forms of its own. Starter modules never reach here, and a
            # gauge reading 0% morphology on the alphabet page would argue against
            # the very thing it exists to argue for.
            frequency = lesson["frequency"]
            if lesson["label"] != DEPONENT_LESSON_LABEL and isinstance(
                frequency, (int, np.integer)
            ):
                covered_forms += int(frequency)

            if total_greek_tokens and total_forms:
                markdown_content[coverage_slot] = render_coverage_gauges(
                    _coverage_fraction(covered_tokens, total_greek_tokens),
                    _coverage_fraction(covered_forms, total_forms),
                    lang,
                )

        markdown_content.append("")

    # Syntax closes the book rather than entering the syllabus: these
    # constructions are built above the word, so the morphology grid the lessons
    # are generated from has no cell for them. Placed before the passages, which
    # are where a reader meets them running.
    appendix_rank = len(lesson_data)
    syntax_path = grammar_folder / SYNTAX_REFERENCE_FILENAME
    if syntax_path.exists():
        _, syntax_body = _split_lesson_title(syntax_path.read_text(encoding="utf-8"))
        syntax_body = fill_syntax_placeholders(syntax_body, syntax_detection, lang)
        appendix_rank += 1
        syntax_title = t("tb_syntax_header", lang)
        anchor = heading_slug(f"{appendix_rank}. {syntax_title}")
        markdown_content[syntax_toc_slot] = f"{appendix_rank}. [{syntax_title}](#{anchor})"
        markdown_content.append("---")
        markdown_content.append("")
        markdown_content.append(f"# {appendix_rank}. {syntax_title}")
        markdown_content.append("")
        markdown_content.append(syntax_body)
        markdown_content.append("")

    # Continuous reading to close on, drawn from the same texts the lessons were
    # built from. known_lemmas now holds everything the book taught, which is what
    # ranks the passages.
    passages: list[dict] = []
    if working_combined_df is not None and working_sentences_df is not None and not working_sentences_df.empty:
        passages = build_reading_passages(working_sentences_df, working_combined_df, known_lemmas)

    passages_rank = appendix_rank
    if passages:
        passages_rank = appendix_rank + 1
        passages_title = t("tb_passages_header", lang)
        anchor = heading_slug(f"{passages_rank}. {passages_title}")
        markdown_content[passages_toc_slot] = f"{passages_rank}. [{passages_title}](#{anchor})"
        markdown_content.append("---")
        markdown_content.append("")
        markdown_content.append(f"# {passages_rank}. {passages_title}")
        markdown_content.append("")
        markdown_content.append(format_passage_appendix(passages, lang=lang))
        markdown_content.append("")

    # A book ends where its settings end, not where the language does, so the last
    # page names those settings back and says what to change to carry on.
    next_rank = passages_rank + 1
    next_title = t("tb_next_header", lang)
    anchor = heading_slug(f"{next_rank}. {next_title}")
    markdown_content[next_toc_slot] = f"{next_rank}. [{next_title}](#{anchor})"
    markdown_content.append("---")
    markdown_content.append("")
    markdown_content.append(f"# {next_rank}. {next_title}")
    markdown_content.append("")
    markdown_content.append(
        t("tb_next_body", lang, lesson_count=int(lesson_count), mode=t(mode_key, lang))
    )
    markdown_content.append("")

    # Descending, so the indices of the slots still to check stay valid.
    for slot in sorted([syntax_toc_slot, passages_toc_slot, next_toc_slot], reverse=True):
        if not markdown_content[slot]:
            del markdown_content[slot]

    document = "\n".join(markdown_content)

    if rtl:
        # Base paragraph direction for the document. The blank lines keep GitHub
        # parsing the inner markdown; markdown="1" does the same for md_in_html.
        document = f'<div dir="rtl" markdown="1">\n\n{document}\n\n</div>\n'

    return document


def generate_textbook_html(
    frequency_syllabus: pd.DataFrame,
    grammar_folder: str | Path,
    lesson_count: int = 40,
    doc_title: str | None = None,
    combined_df: pd.DataFrame | None = None,
    syllabus_mode: str = "case",
    lang: str = DEFAULT_LANG,
    markdown_content: str | None = None,
    source_summary: Mapping[str, Any] | None = None,
    logo_data_uri: str | None = None,
) -> str:
    rtl = is_rtl(lang)
    if doc_title is None:
        doc_title = t("tb_doc_title", lang)

    if markdown_content is None:
        markdown_content = generate_textbook_markdown(
            frequency_syllabus=frequency_syllabus,
            grammar_folder=grammar_folder,
            lesson_count=lesson_count,
            combined_df=combined_df,
            syllabus_mode=syllabus_mode,
            lang=lang,
            source_summary=source_summary,
        )

    # A downloaded HTML file is read offline as often as not, so the cover logo is
    # inlined here rather than left pointing at GitHub.
    if logo_data_uri:
        markdown_content = markdown_content.replace(TEXTBOOK_LOGO_URL, logo_data_uri)

    # "extra" bundles md_in_html, which parses the markdown inside the RTL wrapper.
    body_html = markdown_to_html(
        markdown_content,
        extensions=["extra", "toc", "tables"],
        extension_configs={"toc": {"slugify": heading_slug}},
    )

    if rtl:
        body_html = wrap_ltr_runs_in_html(body_html)

    dir_attr = "rtl" if rtl else "ltr"
    rtl_font_link = ""
    rtl_style = ""
    if rtl:
        rtl_font_link = (
            '\n    <link rel="stylesheet" '
            'href="https://fonts.googleapis.com/css2?family=Noto+Naskh+Arabic:wght@400;500;600;700&display=swap">'
        )
        rtl_style = """
        body {
            font-family: 'Noto Naskh Arabic', 'B Lotus', 'Segoe UI', Tahoma, sans-serif;
        }
        /* Code blocks stay left-to-right even in an RTL document. */
        pre, code {
            direction: ltr;
            text-align: left;
        }
        /* Greek-only blocks and cells are their own LTR island. */
        .ltr-block, td[dir="ltr"], th[dir="ltr"] {
            text-align: left;
        }
        .ltr-block ol, .ltr-block ul {
            padding-inline-start: 1.5rem;
        }
"""

    return f"""<!doctype html>
<html lang=\"{lang}\" dir=\"{dir_attr}\">
<head>
    <meta charset=\"utf-8\">
    <meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">
    <title>{doc_title}</title>{rtl_font_link}
    <style>
        @page {{
            margin: 18mm;
        }}
        body {{
            margin: 0;
            padding: 2rem;
            font-family: Arial, sans-serif;
            line-height: 1.7;
            color: #222;
            background: #fff;
        }}
        h1, h2, h3, h4 {{
            line-height: 1.3;
        }}
        /* Each lesson opens on an H1 banded in the logo's ink, so the start of a
           lesson is unmistakable on screen and lands on a fresh printed page. */
        h1 {{
            margin: 4rem 0 1.5rem;
            padding: 0.7rem 1rem;
            font-size: 2rem;
            color: #3A1712;
            background: #f6f1ea;
            border-inline-start: 6px solid #3A1712;
            break-before: page;
            page-break-before: always;
        }}
        h1 + p {{
            margin-top: -0.8rem;
            color: #6b5b4d;
            font-size: 0.92rem;
        }}
        h2 {{
            margin: 2.4rem 0 0.8rem;
            font-size: 1.45rem;
            color: #3A1712;
        }}
        h3 {{
            margin: 1.8rem 0 0.6rem;
            font-size: 1.15rem;
        }}
        h4 {{
            margin: 1.4rem 0 0.5rem;
            font-size: 1rem;
            color: #6b5b4d;
        }}
        /* links */
        a, a:visited {{
            color: #3e6182;
            text-decoration-color: #a3b6c8;
        }}
        /* contents entries: the numbered list already reads as navigation */
        li > a {{
            text-decoration: none;
        }}
        a:hover {{
            color: #2c4762;
            text-decoration: underline;
            text-decoration-color: #2c4762;
        }}
        hr {{
            border: 0;
            border-top: 1px solid #e0d6ca;
            margin: 3rem 0 0;
            break-after: avoid;
        }}
        .title-page {{
            padding: 4rem 0 5rem;
            break-after: page;
            page-break-after: always;
        }}
        .title-page .textbook-logo {{
            width: min(360px, 60%);
            height: auto;
        }}
        .title-page h1 {{
            margin: 2rem 0 1rem;
            padding: 0;
            font-size: 2.4rem;
            background: none;
            border: 0;
            break-before: auto;
            page-break-before: auto;
        }}
        .title-page h1 + p {{
            margin-top: 0;
            color: #444;
            font-size: 1.05rem;
            max-width: 34rem;
            margin-inline: auto;
        }}
        pre {{
            padding: 1rem;
            background: #f6f8fa;
            overflow-x: auto;
        }}
        code {{
            font-family: Consolas, Monaco, monospace;
        }}
        table {{
            border-collapse: collapse;
            width: 100%;
            margin: 1rem 0;
        }}
        th, td {{
            border: 1px solid #ccc;
            padding: 0.5rem;
            text-align: start;
        }}
        th {{
            background: #f0f0f0;
        }}
        /* The two coverage donuts under a lesson heading. Kept on the heading's
           page: orphaned from it they say nothing. */
        .coverage {{
            margin: 0 0 1.8rem;
            break-inside: avoid;
            page-break-inside: avoid;
            break-after: avoid;
        }}
        .coverage-lead {{
            margin: 0 0 0.5rem;
            color: #6b5b4d;
            font-size: 0.92rem;
        }}
        .coverage-rings {{
            display: flex;
            gap: 2.2rem;
        }}
        .coverage-ring {{
            margin: 0;
            text-align: center;
        }}
        .coverage-ring figcaption {{
            margin-top: 0.1rem;
            color: #6b5b4d;
            font-size: 0.85rem;
            line-height: 1.3;
        }}
        .coverage-track {{
            fill: none;
            stroke: #e0d6ca;
            stroke-width: 8;
        }}
        .coverage-fill {{
            fill: none;
            stroke: #3A1712;
            stroke-width: 8;
        }}
        /* The numeral stays left-to-right in an RTL book, or the per-cent sign
           lands on the wrong side of the digits. */
        .coverage-value {{
            fill: #3A1712;
            font-family: Arial, sans-serif;
            font-size: 15px;
            font-weight: bold;
            direction: ltr;
            unicode-bidi: isolate;
        }}
{rtl_style}    </style>
</head>
<body>
{body_html}
</body>
</html>"""
