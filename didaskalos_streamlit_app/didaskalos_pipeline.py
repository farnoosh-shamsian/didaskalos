from __future__ import annotations

import os
import re
import unicodedata
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
from markdown import markdown as markdown_to_html

try:
    from i18n import DEFAULT_LANG, is_rtl, t
except ImportError:  # imported as part of a package rather than as a flat module
    from .i18n import DEFAULT_LANG, is_rtl, t

try:
    from treebank_parsers import parse_agdt_xml, parse_treebank_file
except ImportError:  # imported as part of a package rather than as a flat module
    from .treebank_parsers import parse_agdt_xml, parse_treebank_file

# Back-compat alias: parse_treebank_xml was the historical name for the AGDT
# parser before formats were pluggable. Kept so external callers keep working.
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

# First postag character of the word classes whose lemmas carry lexical content
# (noun, adjective, verb, adverb, pronoun). Everything else (article, particle,
# conjunction, preposition, interjection, punctuation) is treated as a function
# word for difficulty scoring and known-vocabulary coverage.
CONTENT_POS_PREFIXES = ("n", "a", "v", "d", "p")

# Sentence difficulty blends how rare the sentence's content words are on
# average, how rare its single rarest word is (the word that gates
# comprehension), and how long the sentence is.
DIFFICULTY_WEIGHT_MEAN_RARITY = 0.35
DIFFICULTY_WEIGHT_RAREST_WORD = 0.35
DIFFICULTY_WEIGHT_LENGTH = 0.30

# Stage-aware exercise selection: prefer sentences whose content lemmas are
# mostly vocabulary already introduced in earlier lessons.
KNOWN_LEMMA_COVERAGE_THRESHOLD = 0.70
KNOWN_FUNCTION_LEMMA_SEED_COUNT = 50


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


def normalize_greek_lemma(lemma: str) -> str:
    if not isinstance(lemma, str):
        return ""
    return "".join(c for c in unicodedata.normalize("NFD", lemma.lower().strip()) if unicodedata.category(c) != "Mn")


def parse_verb_subcategory(lemma: str, postag: str | None = None) -> str:
    if postag and not str(postag).startswith("v"):
        return ""

    lemma_n = normalize_greek_lemma(lemma)
    if not lemma_n:
        return ""
    if lemma_n.endswith("μαι"):
        return "deponent"
    if lemma_n.endswith("μι"):
        return "mi"
    if lemma_n.endswith("ω"):
        return "w"
    return "irregular"


def is_greek_lemma(lemma: str) -> bool:
    return isinstance(lemma, str) and bool(GREEK_MARK_RE.search(lemma))


# ---------------------------------------------------------------------------
# Declension classification (declension-based textbook mode)
# ---------------------------------------------------------------------------

# AGDT 9-position postag indices.
POSTAG_NUMBER_INDEX = 2
POSTAG_GENDER_INDEX = 6
POSTAG_CASE_INDEX = 7

# Label text drives the lesson filename via normalize_frequency_row_name
# (e.g. "first declension feminine nouns" -> first_declension_feminine_nouns.md),
# so it must stay in sync with the lessons-decl/ file names. The short code
# (N1..N8, ADJ1..ADJ3) is kept only as the dict key / declension_code column.
NOUN_DECLENSION_LABELS = {
    "N1": "first declension feminine nouns",
    "N2": "first declension masculine nouns",
    "N3": "second declension masculine nouns",
    "N4": "second declension neuter nouns",
    "N5": "third declension consonant stem nouns",
    "N6": "third declension iota upsilon stem nouns",
    "N7": "third declension nasal liquid stem nouns",
    "N8": "other third declension and irregular nouns",
}

ADJECTIVE_DECLENSION_LABELS = {
    "ADJ1": "first second declension adjectives",
    "ADJ2": "third declension adjectives",
    "ADJ3": "other adjectives",
}

DECLENSION_LABELS = {**NOUN_DECLENSION_LABELS, **ADJECTIVE_DECLENSION_LABELS}

# Keys are diacritic-stripped, lowercased, with final sigma normalized to σ
# (the output of _classification_key). Only lemmas whose nominative-singular
# ending points to the wrong class need to be listed here.
IRREGULAR_NOUN_LEXICON = {
    "γυνη": "N5",  # γυναικός: consonant stem despite ending in -η
    "παισ": "N5",  # παιδός: dental stem despite ending in -ις
    "ελπισ": "N5",  # ἐλπίδος
    "χαρισ": "N5",  # χάριτος
    "ορνισ": "N5",  # ὄρνιθος
    "ερισ": "N5",  # ἔριδος
    "κλεισ": "N5",  # κλειδός
    "νουσ": "N3",  # second declension contract
    "πλουσ": "N3",
    "ζευσ": "N8",
    "γραυσ": "N8",
    "γηρασ": "N8",
    "κερασ": "N8",
    "τερασ": "N8",
    "κρεασ": "N8",
    "υδωρ": "N7",
}

IRREGULAR_ADJECTIVE_LEXICON = {
    "πολυσ": "ADJ3",  # mixed 2nd/3rd declension paradigm
    "μεγασ": "ADJ3",  # mixed 2nd/3rd declension paradigm
}


def _classification_key(text: str) -> str:
    normalized = normalize_greek_lemma(text)
    normalized = re.sub(r"\d+$", "", normalized)
    return normalized.replace("ς", "σ")  # final sigma -> sigma


def _genitive_singular_signal(forms: list[str]) -> str | None:
    """Vote on the declension using the genitive-singular forms attested in the corpus.

    Returns one of: "d12" (-ου: 1st masc / 2nd decl), "d1" (-ης/-ας: 1st decl),
    "d3i" (-εως: 3rd decl iota stem), "d3s" (-ους: 3rd decl sigma stem),
    "d3" (-ος: other 3rd decl), or None when no genitive singular is attested.
    """
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


def classify_noun_declension(lemma: str, gender: str = "-", genitive_signal: str | None = None) -> str:
    """Classify a noun lemma into N1..N8.

    ``gender`` is the AGDT postag gender character ("m", "f", "n" or "-"),
    ideally the majority gender of the lemma across the corpus.
    ``genitive_signal`` is the output of _genitive_singular_signal for the lemma.
    """
    key = _classification_key(lemma)
    if not key:
        return "N8"
    if key in IRREGULAR_NOUN_LEXICON:
        return IRREGULAR_NOUN_LEXICON[key]

    third_declension_evidence = genitive_signal in {"d3", "d3s", "d3i"}

    # Unambiguously third-declension nominative endings.
    if key.endswith(("ευσ", "αυσ", "ουσ", "ω")):
        return "N8"  # βασιλεύς, ναῦς, βοῦς, πειθώ
    if key.endswith(("ην", "ων", "ηρ", "ωρ")):
        return "N7"  # ποιμήν, δαίμων, πατήρ, ῥήτωρ
    if key.endswith("ισ"):
        # πόλις (-εως, iota stem) vs. ἐλπίς (-ίδος, dental stem).
        if genitive_signal == "d3":
            return "N5"
        return "N6"
    if key.endswith(("υσ", "υ")):
        return "N6"  # ἰχθύς, ἄστυ

    # -μα, -ματος neuters are consonant (dental) stems: σῶμα, πρᾶγμα.
    if gender == "n" and key.endswith("μα"):
        return "N5"

    if gender == "f" and key.endswith(("α", "η")):
        return "N5" if third_declension_evidence else "N1"
    if gender == "m" and key.endswith(("ασ", "ησ")):
        # πολίτης (-ου, 1st decl) vs. Σωκράτης (-ους, sigma stem) vs. γίγας (-αντος).
        if genitive_signal == "d3s":
            return "N8"
        if genitive_signal == "d3":
            return "N5"
        return "N2"

    if key.endswith("οσ"):
        if gender == "n":
            return "N8"  # γένος, τεῖχος: sigma-stem neuters
        if third_declension_evidence:
            return "N5"
        return "N3"  # masc λόγος (rare feminines like ὁδός also land here)
    if gender == "n" and key.endswith("ον"):
        return "N4"

    # Remaining lemmas ending in a consonant: φύλαξ, νύξ, Ἑλλάς, χείρ, ...
    if key.endswith(("ξ", "ψ", "ρ", "ν", "σ")):
        return "N5"

    return "N8"


def classify_adjective_declension(lemma: str) -> str:
    key = _classification_key(lemma)
    if not key:
        return "ADJ3"
    if key in IRREGULAR_ADJECTIVE_LEXICON:
        return IRREGULAR_ADJECTIVE_LEXICON[key]
    if key.endswith(("οσ", "ουσ")):
        return "ADJ1"  # ἀγαθός, δίκαιος, contract χρυσοῦς
    if key.endswith(("υσ", "εισ", "ασ")):
        return "ADJ2"  # three-ending 3rd decl: ταχύς, χαρίεις, πᾶς, μέλας
    return "ADJ3"  # two-ending 3rd decl (-ης, -ων), comparatives, irregulars


def add_declension_features(combined_df: pd.DataFrame) -> pd.DataFrame:
    """Add declension_code / declension_label columns for noun and adjective rows."""
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

        gendered = noun_rows[noun_rows["gender"].isin(["m", "f", "n"])]
        majority_gender = (
            gendered.groupby("key")["gender"].agg(lambda genders: genders.value_counts().idxmax()).to_dict()
            if not gendered.empty
            else {}
        )

        genitive_singular_mask = (
            noun_rows["postag"].astype(str).str.slice(POSTAG_CASE_INDEX, POSTAG_CASE_INDEX + 1).eq("g")
            & noun_rows["postag"].astype(str).str.slice(POSTAG_NUMBER_INDEX, POSTAG_NUMBER_INDEX + 1).eq("s")
        )
        genitive_signals = {
            key: _genitive_singular_signal(group["form"].astype(str).tolist())
            for key, group in noun_rows[genitive_singular_mask].groupby("key")
        }

        code_by_key = {
            row["key"]: classify_noun_declension(
                row["lemma"],
                majority_gender.get(row["key"], "-"),
                genitive_signals.get(row["key"]),
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
    """Replace the case-based syllabus of noun/adjective rows with declension labels."""
    out = combined_df if "declension_label" in combined_df.columns else add_declension_features(combined_df)
    out = out.copy()

    noun_adjective_mask = out["postag"].astype(str).str.startswith(("n", "a"))
    has_label = out["declension_label"].astype(str).ne("")

    out.loc[noun_adjective_mask, "syllabus"] = "NA"
    out.loc[noun_adjective_mask & has_label, "syllabus"] = out.loc[
        noun_adjective_mask & has_label, "declension_label"
    ]
    return out


def build_declension_summary(combined_df: pd.DataFrame) -> pd.DataFrame:
    """Per-category token counts, lemma counts and example lemmas, sorted by frequency."""
    columns = ["declension_code", "declension_label", "tokens", "distinct_lemmas", "example_lemmas"]
    if combined_df is None or combined_df.empty:
        return pd.DataFrame(columns=columns)

    df = combined_df if "declension_code" in combined_df.columns else add_declension_features(combined_df)
    classified = df[df["declension_code"].astype(str).ne("")]
    if classified.empty:
        return pd.DataFrame(columns=columns)

    summary_rows = []
    for code, label in DECLENSION_LABELS.items():
        subset = classified[classified["declension_code"] == code]
        if subset.empty:
            continue
        lemma_counts = subset["lemma"].value_counts()
        summary_rows.append(
            {
                "declension_code": code,
                "declension_label": label,
                "tokens": int(len(subset)),
                "distinct_lemmas": int(lemma_counts.size),
                "example_lemmas": ", ".join(lemma_counts.head(5).index.astype(str).tolist()),
            }
        )

    return pd.DataFrame(summary_rows, columns=columns).sort_values("tokens", ascending=False, ignore_index=True)


def build_combined_df(
    folder: str | Path,
    selected_files: list[str],
    syllabus_mode: str = "case",
    formats: Mapping[str, str | None] | None = None,
) -> pd.DataFrame:
    # ``formats`` maps a selected filename to its declared corpus format (from the
    # registry). Missing/None entries let the dispatcher auto-detect by extension
    # or content, which keeps uploads and ad-hoc URLs working.
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
    combined_df["syllabus"] = combined_df["postag"].apply(parse_postag)
    combined_df["pos_category"] = combined_df["postag"].apply(parse_pos_category)
    combined_df["verb_subcategory"] = combined_df.apply(
        lambda row: parse_verb_subcategory(row["lemma"], row["postag"]) if row["pos_category"] == "verb" else "",
        axis=1,
    )

    if syllabus_mode == "declension":
        combined_df = add_declension_features(combined_df)
        combined_df = apply_declension_syllabus(combined_df)

    return combined_df


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

    frequency_syllabus = (
        pd.DataFrame(
            {
                "syllabus": syllabus_with_verb_bucket,
                "pos_category": combined_df["pos_category"],
            }
        )
        .groupby(["syllabus", "pos_category"], dropna=False)
        .size()
        .reset_index(name="frequency")
        .sort_values("frequency", ascending=False, ignore_index=True)
    )
    frequency_syllabus["syllabus_normalized"] = frequency_syllabus["syllabus"].apply(normalize_frequency_row_name)

    # Always skip placeholder rows like NA/unknown in the "other" POS bucket.
    skip_labels = {"na", "unknown", ""}
    skip_mask = (
        frequency_syllabus["pos_category"].astype(str).eq("other")
        & frequency_syllabus["syllabus_normalized"].astype(str).isin(skip_labels)
    )
    frequency_syllabus = frequency_syllabus.loc[~skip_mask].reset_index(drop=True)

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


# Bidi handling for RTL output languages (e.g. Persian). Greek is strongly
# left-to-right; inside an RTL paragraph the Unicode bidi algorithm misplaces
# neutral characters (hyphens, punctuation) that sit at the edges of a Greek
# run, so Greek runs get isolated explicitly.
_GREEK_LETTER = "Ͱ-Ͽἀ-῿"
_GREEK_MARKS = "̀-ͯ᾽᾿’'"
_GREEK_TOKEN = f"[{_GREEK_LETTER}][{_GREEK_LETTER}{_GREEK_MARKS}]*"
_GREEK_SEP = "[  ,;.··‐‑-]+"
# A tag-free phrase: Greek tokens joined by spaces/neutral punctuation.
_GREEK_PHRASE = f"{_GREEK_TOKEN}(?:{_GREEK_SEP}{_GREEK_TOKEN})*"
# A phrase wrapped in one balanced inline element (e.g. <strong>δέ</strong>,
# rendered from **δέ**), so emphasis inside a Greek sentence does not split
# the run into separate isolates and reverse the word order.
_GREEK_ELEM = "(?:" + "|".join(
    f"<{tag}>{_GREEK_PHRASE}</{tag}>" for tag in ("u", "em", "strong", "b", "i")
) + ")"
_GREEK_ATOM = f"(?:{_GREEK_PHRASE}|{_GREEK_ELEM})"
# A run: atoms joined by separators, with optional attached hyphens
# (endings like "-η", stems like "λυ-").
_GREEK_RUN_RE = re.compile(f"-?{_GREEK_ATOM}(?:{_GREEK_SEP}{_GREEK_ATOM})*-?")


def wrap_greek_runs_in_html(html: str) -> str:
    """Wrap runs of Greek text in ``<bdi dir="ltr">`` isolates.

    Safe to apply to rendered HTML because Greek characters only ever occur in
    text content, never inside tag markup.
    """
    return _GREEK_RUN_RE.sub(
        lambda match: f'<bdi lang="grc" dir="ltr">{match.group(0)}</bdi>',
        html,
    )


def _ltr_isolate(text: str, rtl: bool) -> str:
    """Wrap a fully-Greek fragment (possibly containing inline tags such as
    ``<u>``) in an LTR span so word order survives an RTL paragraph."""
    if not rtl:
        return text
    return f'<span lang="grc" dir="ltr">{text}</span>'


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


def split_syllabus_label_and_bucket(syllabus_label: str) -> tuple[str, str | None]:
    if not isinstance(syllabus_label, str):
        return syllabus_label, None
    match = re.match(r"^(.*)\s\(([^()]*)\)$", syllabus_label.strip())
    if not match:
        return syllabus_label, None
    return match.group(1), match.group(2)


def decode_marked_verb_features(postag: str) -> dict[str, str]:
    if not isinstance(postag, str) or len(postag) < 6:
        return {
            "person": "unknown",
            "number": "unknown",
            "tense": "unknown",
            "voice": "unknown",
            "mood": "unknown",
        }

    person_code = postag[1] if len(postag) > 1 else "-"
    number_code = postag[2] if len(postag) > 2 else "-"
    tense_code = postag[3] if len(postag) > 3 else "-"
    mood_code = postag[4] if len(postag) > 4 else "-"
    voice_code = postag[5] if len(postag) > 5 else "-"

    return {
        "person": PERSON_MAP.get(person_code, "unknown"),
        "number": NUMBER_MAP.get(number_code, "unknown"),
        "tense": TENSE_MAP.get(tense_code, "unknown") if tense_code != "-" else "not marked",
        "voice": VOICE_MAP.get(voice_code, "unknown") if voice_code != "-" else "not marked",
        "mood": MOOD_MAP.get(mood_code, "unknown") if mood_code != "-" else "not marked",
    }


def get_topic_rows_for_label(syllabus_label: str, combined_df: pd.DataFrame) -> pd.DataFrame:
    base_label, verb_bucket = split_syllabus_label_and_bucket(syllabus_label)
    if verb_bucket is None:
        direct = combined_df[combined_df["syllabus"] == syllabus_label].copy()
        if not direct.empty:
            return direct
    else:
        direct = combined_df[combined_df["syllabus"] == base_label].copy()
        if not direct.empty:
            return direct[(direct["pos_category"] == "verb") & (direct["verb_subcategory"] == verb_bucket)]

    normalized_target = normalize_frequency_row_name(syllabus_label)
    normalized_series = combined_df["syllabus"].apply(normalize_frequency_row_name)

    verb_suffix_map = {
        "_w": "w",
        "_mi": "mi",
        "_deponent": "deponent",
        "_irregular": "irregular",
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
    topic_words = topic_rows.drop_duplicates(subset=["lemma"], keep="first").head(num_words)
    return topic_words[["form", "lemma", "postag", "token_index", "sentence_index"]]


def assemble_sentences(df: pd.DataFrame) -> pd.DataFrame:
    attach_to_prev = {",", ".", ";", ":", "!", "?", ")", "']"}

    def join_forms(forms: list[str]) -> str:
        words = []
        for form in forms:
            token = str(form)
            if token in attach_to_prev and words:
                words[-1] += token
            else:
                words.append(token)

        text = " ".join(words)
        text = re.sub(r"\s+([,.:;!?\)])", r"\1", text)
        text = re.sub(r"([\(\[])\s+", r"\1", text)

        # Drop bracketed index markers from source data like [0], [12].
        text = re.sub(r"\[\s*\d+\s*\]", "", text)

        # Remove hidden Unicode formatting chars that can appear as odd symbols.
        text = re.sub(r"[\u200b-\u200f\u2060\ufeff]", "", text)

        text = re.sub(r"\s+", " ", text).strip()
        return text

    rows = []
    for sent_id, group in df.groupby("sentence_id", sort=False):
        group = group.sort_values("token_index" if "token_index" in group.columns else "word_id")
        first = group.iloc[0]
        rows.append(
            {
                "sentence_id": sent_id,
                "document_id": first.get("document_id"),
                "subdoc": first.get("subdoc"),
                "file": first.get("file"),
                "sentence_text": join_forms(group["form"].tolist()),
                "word_count": len(group),
            }
        )

    return pd.DataFrame(rows)


def add_sentence_scores(sentences_df: pd.DataFrame, combined_df: pd.DataFrame) -> pd.DataFrame:
    out = sentences_df.copy()

    greek = combined_df[combined_df["lemma"].apply(is_greek_lemma)].copy()
    if "lemma_frequency" in greek.columns:
        greek["lemma_frequency"] = pd.to_numeric(greek["lemma_frequency"], errors="coerce").fillna(0.0)
    else:
        counts = greek["lemma"].value_counts()
        greek["lemma_frequency"] = greek["lemma"].map(counts).astype(float)

    # Function words are frequent enough to drown out the words that actually
    # gate comprehension, so lexical difficulty looks at content words only.
    # Log frequencies tame the Zipf skew: a couple of articles no longer make a
    # sentence full of rare words look easy.
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
        # Sentences with no content words carry no lexical load; score them easy.
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
    """Top function-word lemmas (articles, particles, conjunctions, ...) that
    every reader meets from the first page; they seed the known-vocabulary set
    used for stage-aware sentence selection."""
    if combined_df is None or combined_df.empty:
        return set()
    greek = combined_df[combined_df["lemma"].apply(is_greek_lemma)]
    function_rows = greek[~greek["postag"].astype(str).str.startswith(CONTENT_POS_PREFIXES)]
    top_lemmas = function_rows["lemma"].value_counts().head(top_n).index
    return {normalize_greek_lemma(str(lemma)) for lemma in top_lemmas}


def _known_lemma_coverage_by_sentence(combined_df: pd.DataFrame, known_lemmas: set[str]) -> pd.Series:
    """Fraction of content-word lemmas per sentence_index that are known."""
    greek = combined_df[combined_df["lemma"].apply(is_greek_lemma)]
    content = greek[greek["postag"].astype(str).str.startswith(CONTENT_POS_PREFIXES)]
    if content.empty:
        return pd.Series(dtype=float)
    known = content["lemma"].astype(str).map(normalize_greek_lemma).isin(known_lemmas)
    return known.groupby(content["sentence_index"]).mean()


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

    # The lesson's own target lemmas are being taught right now, so they count
    # as known when judging whether a sentence fits the learner's stage.
    effective_known = known_lemmas | {
        normalize_greek_lemma(str(lemma)) for lemma in matching_rows["lemma"].dropna()
    }
    candidate_rows = combined_df[combined_df["sentence_index"].isin(matching_sentence_indices)]
    coverage = _known_lemma_coverage_by_sentence(candidate_rows, effective_known)
    topic_sentences["known_lemma_coverage"] = topic_sentences["sentence_index"].map(coverage).fillna(1.0)

    # Stage-appropriate sentences first; if the corpus cannot fill the quota,
    # fall back to the remaining sentences ranked by difficulty alone.
    qualified_mask = topic_sentences["known_lemma_coverage"] >= KNOWN_LEMMA_COVERAGE_THRESHOLD
    qualified = topic_sentences[qualified_mask].sort_values("difficulty_score")
    remainder = topic_sentences[~qualified_mask].sort_values("difficulty_score")
    return pd.concat([qualified, remainder]).head(num_sentences)


def format_exercise_set1(topic_words: pd.DataFrame, lesson_pos_category: str, lang: str = DEFAULT_LANG) -> str:
    if topic_words is None or topic_words.empty:
        return ""

    rtl = is_rtl(lang)
    pos_label = _pos_label(lesson_pos_category, lang)
    lines = [
        f"### {t('tb_ex1_header', lang)}",
        "",
        t("tb_ex1_prompt", lang, pos_label=pos_label),
        "",
    ]
    for idx, (_, row) in enumerate(topic_words.iterrows(), 1):
        item = t(
            "tb_ex1_item",
            lang,
            form=_ltr_isolate(str(row["form"]), rtl),
            lemma=_ltr_isolate(str(row["lemma"]), rtl),
        )
        lines.append(f"{idx}. {item}")
    lines.append("")
    return "\n".join(lines)


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
    topic_rows = filter_topic_rows_by_lesson_rules(syllabus_label, lesson_pos_category, topic_rows)

    if "token_index" not in topic_rows.columns:
        if "word_id" in topic_rows.columns:
            topic_rows["token_index"] = pd.to_numeric(topic_rows["word_id"], errors="coerce")
        else:
            topic_rows["token_index"] = pd.Series(range(len(topic_rows)), index=topic_rows.index, dtype="int64")

    return topic_rows


def _normalize_answer_word(word: str) -> str:
    return str(word).strip().lower()


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

    grouped_targets = {
        sent_idx: grp.sort_values("token_index")
        for sent_idx, grp in topic_rows.groupby("sentence_index", sort=False)
    }

    for _, sentence_row in topic_sentences.iterrows():
        if len(selected_sentence_ids) >= max_sentences:
            break

        sentence_index = sentence_row["sentence_index"]
        sentence_text = str(sentence_row.get("sentence_text", "")).strip()
        sentence_text_key = re.sub(r"\s+", " ", sentence_text)

        if not sentence_text_key or sentence_text_key in used_sentence_texts:
            continue

        sentence_targets = grouped_targets.get(sentence_index)
        if sentence_targets is None or sentence_targets.empty:
            continue

        candidate_rows = []
        for _, target_row in sentence_targets.iterrows():
            answer_form = _normalize_answer_word(target_row.get("form", ""))
            if not answer_form or answer_form in used_answer_words:
                continue
            candidate_rows.append(target_row)

        if not candidate_rows:
            continue

        chosen_targets = pd.DataFrame(candidate_rows)
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
    sentence_form_lookup: dict[object, list[str]],
    lang: str = DEFAULT_LANG,
) -> str:
    if exercise_sentences is None or exercise_sentences.empty:
        return ""

    rtl = is_rtl(lang)
    pos_label = _pos_label(lesson_pos_category, lang)
    lines = [
        f"### {t('tb_ex2_header', lang)}",
        "",
        t("tb_ex2_prompt", lang, pos_label=pos_label),
        "",
    ]
    for idx, (_, row) in enumerate(exercise_sentences.iterrows(), 1):
        lines.append(f"{idx}. {_ltr_isolate(str(row['sentence_text']), rtl)}")
    lines.append("")
    lines.append(f"#### {t('tb_answer_key_header', lang)}")
    lines.append("")

    for idx, (_, row) in enumerate(exercise_sentences.iterrows(), 1):
        targets = sentence_form_lookup.get(row["sentence_index"], [])
        if targets:
            answer = _ltr_isolate(", ".join(targets), rtl)
        else:
            answer = t("tb_no_target_form", lang)
        lines.append(f"{idx}. {answer}")

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
        f"### {t('tb_ex2_verb_header', lang)}",
        "",
        t("tb_ex2_verb_prompt", lang),
        "",
    ]

    for idx, (_, row) in enumerate(exercise_sentences.iterrows(), 1):
        sentence_rows = sentence_verb_rows.get(row["sentence_index"])
        forms = set()
        if sentence_rows is not None and not sentence_rows.empty:
            forms = set(sentence_rows["form"].tolist())
        marked = mark_topic_words_in_sentence(row["sentence_text"], forms)
        lines.append(f"{idx}. {_ltr_isolate(marked, rtl)}")

    lines.append("")
    lines.append(f"#### {t('tb_answer_key_header', lang)}")
    lines.append("")

    for idx, (_, row) in enumerate(exercise_sentences.iterrows(), 1):
        sentence_rows = sentence_verb_rows.get(row["sentence_index"])
        if sentence_rows is None or sentence_rows.empty:
            lines.append(f"{idx}. " + t("tb_no_marked_verbs", lang))
            continue

        sentence_answers = []
        for _, verb_row in sentence_rows.iterrows():
            features = decode_marked_verb_features(verb_row.get("postag", ""))
            sentence_answers.append(
                t(
                    "tb_verb_answer",
                    lang,
                    form=_ltr_isolate(str(verb_row.get("form", "")), rtl),
                    lemma=_ltr_isolate(str(verb_row.get("lemma", "")), rtl),
                    person=_feature_label(features["person"], lang),
                    number=_feature_label(features["number"], lang),
                    tense=_feature_label(features["tense"], lang),
                    voice=_feature_label(features["voice"], lang),
                    mood=_feature_label(features["mood"], lang),
                )
            )

        lines.append(f"{idx}. " + " | ".join(sentence_answers))

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
    words_exercise = format_exercise_set1(topic_words, lesson_pos_category, lang=lang)
    if words_exercise:
        exercise_blocks.append(words_exercise)

    topic_sentences = get_topic_sentences(
        syllabus_label=syllabus_label,
        combined_df=combined_df,
        sentences_df=sentences_df,
        num_sentences=num_sentences,
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
                sentence_form_lookup: dict[object, list[str]] = {}
                for sent_idx, grp in selected_targets_by_sentence.items():
                    ordered_forms = list(dict.fromkeys(grp["form"].tolist()))
                    sentence_form_lookup[sent_idx] = ordered_forms
                exercise_blocks.append(_format_exercise_nonverb(lesson_pos_category, selected_sentences, sentence_form_lookup, lang=lang))

    return "\n".join(exercise_blocks)


def _split_lesson_title(lesson_text: str) -> tuple[str | None, str]:
    """Return ``(title, body)`` where ``title`` is the lesson file's leading
    heading (if any) and ``body`` is the remaining markdown.

    Leading YAML frontmatter is dropped so its metadata does not leak into the
    rendered textbook.
    """
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


def generate_textbook_markdown(
    frequency_syllabus: pd.DataFrame,
    grammar_folder: str | Path,
    lesson_count: int = 35,
    combined_df: pd.DataFrame | None = None,
    syllabus_mode: str = "case",
    lang: str = DEFAULT_LANG,
) -> str:
    starter_modules = ["about", "alphabet", "introduction_nouns", "introduction_adjectives", "introduction_verbs"]
    lesson_separator = "════════════════════ ⟡ ════════════════════"
    lesson_separator_markup = f"<div align=\"center\" style=\"font-size: 200%; line-height: 1.2;\">{lesson_separator}</div>"

    rtl = is_rtl(lang)
    if syllabus_mode == "declension":
        intro_text = t("tb_intro_declension", lang)
    else:
        intro_text = t("tb_intro_case", lang)

    lesson_rows = frequency_syllabus[
        frequency_syllabus["syllabus"].notna() & (frequency_syllabus["syllabus"] != "NA")
    ].head(int(lesson_count))

    lesson_data = []
    rank = 0

    # Always prepend core starter modules in this fixed order.
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

    grammar_folder = Path(grammar_folder)

    # Load lesson bodies up front so the table of contents can use each file's
    # own H1 title (localized in translated lesson folders) instead of the raw
    # syllabus label.
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
            lesson["display_label"] = title
        lesson["body"] = body

    markdown_content = []
    markdown_content.append(f"# {t('tb_doc_title', lang)}")
    markdown_content.append("")
    markdown_content.append(intro_text)
    markdown_content.append("")
    markdown_content.append(f"## {t('tb_toc_header', lang)}")
    markdown_content.append("")

    for lesson in lesson_data:
        markdown_content.append(f"{lesson['rank']}. {lesson['display_label']}")

    markdown_content.append("")
    markdown_content.append(lesson_separator_markup)
    markdown_content.append("")

    working_combined_df = None
    working_sentences_df = None
    known_lemmas: set[str] = set()

    if combined_df is not None and not combined_df.empty:
        working_combined_df = combined_df.copy()

        if "lemma_frequency" not in working_combined_df.columns:
            greek_rows = working_combined_df[working_combined_df["lemma"].apply(is_greek_lemma)]
            lemma_counts = greek_rows["lemma"].value_counts()
            working_combined_df["lemma_frequency"] = working_combined_df["lemma"].map(lemma_counts).fillna(0)

        working_sentences_df = assemble_sentences(working_combined_df)
        if not working_sentences_df.empty:
            working_sentences_df["sentence_index"] = range(len(working_sentences_df))
            working_combined_df["sentence_index"] = working_combined_df.groupby("sentence_id", sort=False).ngroup()
            working_sentences_df = add_sentence_scores(working_sentences_df, working_combined_df)

        known_lemmas = build_known_lemma_seed(working_combined_df)

    for lesson in lesson_data:
        markdown_content.append(f"## {lesson['rank']}. {lesson['display_label']}")
        if lesson.get("is_starter"):
            markdown_content.append(t("tb_module_type_core", lang))
        else:
            markdown_content.append(t("tb_pos_family", lang, pos=_pos_label(lesson["pos_category"], lang)))
            markdown_content.append(t("tb_frequency", lang, frequency=lesson["frequency"]))
        markdown_content.append("")
        markdown_content.append(lesson["body"])

        if not lesson.get("is_starter"):
            markdown_content.append("")
            markdown_content.append(f"### {t('tb_exercises_header', lang)}")
            markdown_content.append("")

            if working_combined_df is not None and working_sentences_df is not None and not working_sentences_df.empty:
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
                # Vocabulary introduced here counts as known for later lessons.
                if not topic_words.empty:
                    known_lemmas.update(
                        normalize_greek_lemma(str(lemma)) for lemma in topic_words["lemma"]
                    )
                if exercises:
                    markdown_content.append(exercises)
                else:
                    markdown_content.append(f"*{t('tb_no_exercises', lang, label=lesson['display_label'])}*")
            else:
                markdown_content.append(f"*{t('tb_exercises_unavailable', lang)}*")

        markdown_content.append("")
        markdown_content.append(lesson_separator_markup)
        markdown_content.append("")

    document = "\n".join(markdown_content)

    if rtl:
        # Set the base paragraph direction for the whole document. The blank
        # lines make renderers such as GitHub keep parsing the inner markdown;
        # markdown="1" does the same for python-markdown's md_in_html.
        document = f'<div dir="rtl" markdown="1">\n\n{document}\n\n</div>\n'

    return document


def generate_textbook_html(
    frequency_syllabus: pd.DataFrame,
    grammar_folder: str | Path,
    lesson_count: int = 35,
    doc_title: str | None = None,
    combined_df: pd.DataFrame | None = None,
    syllabus_mode: str = "case",
    lang: str = DEFAULT_LANG,
    markdown_content: str | None = None,
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
        )
    # "extra" bundles md_in_html, which keeps parsing the markdown inside the
    # <div dir="rtl" markdown="1"> document wrapper.
    body_html = markdown_to_html(markdown_content, extensions=["extra", "toc", "tables"])

    if rtl:
        body_html = wrap_greek_runs_in_html(body_html)

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
"""

    return f"""<!doctype html>
<html lang=\"{lang}\" dir=\"{dir_attr}\">
<head>
    <meta charset=\"utf-8\">
    <meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">
    <title>{doc_title}</title>{rtl_font_link}
    <style>
        body {{
            margin: 0;
            padding: 2rem;
            font-family: Arial, sans-serif;
            line-height: 1.7;
            color: #222;
            background: #fff;
        }}
        h1, h2, h3 {{
            line-height: 1.3;
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
{rtl_style}    </style>
</head>
<body>
{body_html}
</body>
</html>"""
