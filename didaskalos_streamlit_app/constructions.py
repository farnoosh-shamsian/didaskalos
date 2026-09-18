# Clause-level constructions read off the dependency annotation the treebanks
# already carry but nothing downstream ever used. Each entry in SYNTAX_ENTRIES
# becomes one section of the syntax reference at the end of the book: a table of
# the categories attested in the chosen corpus, with a few cited examples.
#
# An instance the classifier cannot place is dropped, never guessed at. The
# reference needs a handful of examples per row, not every instance, so a rule
# can discard most of its candidates and still fill the table.
from __future__ import annotations

import re

import pandas as pd

from work_catalog import WORK_CATALOG, tlg_work_key

# Treebanks number homograph lemmas inconsistently: Gorman writes the modal
# particle an1 (390 tokens) beside a plain an (51), so an unnormalized count
# reads a third of the truth.
_LEMMA_DIGITS = re.compile(r"\d+$")

AN = "ἄν"
EI = "εἰ"
EAN = "ἐάν"
ARTICLE = "ὁ"
TE = "τε"
TEOS_SUFFIX = "τέος"

# AGDT hangs the clause verb below its subordinator, which is an AuxC; UD hangs
# the subordinator below the verb as a mark. Measured: ei 975 child / 215 parent
# in AGDT, hina 281 parent / 1 child in PROIEL.
HOP_DIRECTION = {"agdt-xml": "child", "conllu": "parent"}

# Authors whose text is verse. Random draws from the verse treebanks produced
# elided hoste, vocative interjections and tragic word order, none of which
# reads as an example of anything to a beginner.
VERSE_AUTHORS = frozenset(
    {
        "Homer",
        "Hesiod",
        "Sophocles",
        "Aeschylus",
        "Euripides",
        "Aristophanes",
        "Pindar",
        "Homeric Hymns",
    }
)

# ei introduces indirect questions ("whether") as well as conditions, and after
# these verbs it nearly always does. Without the guard, "ho d' ephasken, ei
# polla eie" was offered as a past general condition.
ASKING_VERBS = frozenset(
    {
        "φάσκω",
        "φημί",
        "λέγω",
        "εἶπον",
        "ἐρωτάω",
        "πυνθάνομαι",
        "οἶδα",
        "σκοπέω",
        "ἀκούω",
        "θαυμάζω",
        "ὁράω",
    }
)

EXAMPLE_MIN_WORDS = 5
EXAMPLE_MAX_WORDS = 25
# How many attested examples one reference row prints, and the floor below which
# a row is reported as unattested rather than illustrated.
EXAMPLES_PER_ROW = 3
MIN_ATTESTED = 3

# Deliberately the same vocabulary parse_form_features decodes into, so a
# category label renders through the existing feat_* locale keys rather than a
# second set. Spelled out here rather than imported: didaskalos_pipeline imports
# this module.
MOOD = {"i": "indicative", "s": "subjunctive", "o": "optative", "m": "imperative", "n": "infinitive", "p": "participle"}
TENSE = {"p": "present", "i": "imperfect", "f": "future", "a": "aorist", "r": "perfect", "l": "pluperfect", "t": "future perfect"}


def normalize_lemma(lemma) -> str:
    if lemma is None:
        return ""
    return _LEMMA_DIGITS.sub("", str(lemma)).strip()


class ClauseIndex:
    # The tree lookups every classifier needs, built once per frame. The joins
    # are vectorized and the per-node maps cover only verbs and modal particles;
    # a dict over every token costs more than parsing the treebank does.
    def __init__(self, df: pd.DataFrame, fmt: str = "agdt-xml"):
        self.direction = HOP_DIRECTION.get(fmt, "child")

        keep = [c for c in ("sentence_id","word_id","head","postag","lemma","form","relation","file","document_id","subdoc") if c in df.columns]
        d = df[keep].copy()
        d["word_id"] = d["word_id"].astype(str)
        d["head"] = d["head"].astype(str)
        # postag and relation are categorical in combined_df to keep the frame
        # small, and filling a categorical with a category it does not have is an
        # error, so both go back to plain objects first.
        d["postag"] = d["postag"].astype(object).fillna("").astype(str)
        d["lemma_n"] = d["lemma"].map(normalize_lemma)
        d["rel"] = d["relation"].astype(object).fillna("").astype(str).str.split("_").str[0]
        self.d = d

        verbs = d[d["postag"].str[:1] == "v"]
        # Only verbs are ever looked up by node: every caller tests for one, and
        # a miss falls through to the same answer a non-verb would have given.
        self.verb_postag = dict(zip(zip(verbs["sentence_id"], verbs["word_id"]), verbs["postag"]))
        self.verb_lemma = dict(zip(zip(verbs["sentence_id"], verbs["word_id"]), verbs["lemma_n"]))

        self.verb_child = {}
        for sid, head, wid, tag in zip(
            verbs["sentence_id"], verbs["head"], verbs["word_id"], verbs["postag"]
        ):
            self.verb_child.setdefault((sid, head), (wid, tag))

        coords = d[d["rel"] == "COORD"]
        self.coord_children = {}
        for sid, head, wid in zip(coords["sentence_id"], coords["head"], coords["word_id"]):
            self.coord_children.setdefault((sid, head), []).append(wid)

        modal = d[d["lemma_n"] == AN]
        self.has_an = set(zip(modal["sentence_id"], modal["head"]))

        # The treebanks split eite into ei + enclitic -te, so the disjunction is
        # only visible as a dependent of the ei.
        self.parent = dict(zip(zip(d["sentence_id"], d["word_id"]), d["head"]))

        enclitic = d[d["lemma_n"] == TE]
        self.has_te = set(zip(enclitic["sentence_id"], enclitic["head"]))

        self.sentence_words = d.groupby("sentence_id", observed=True).size().to_dict()

    def clause_verb(self, sid: str, node_id: str):
        # The verb of the clause a subordinator introduces. AGDT hangs it below
        # the subordinator, UD above; trying below first and then above answers
        # for either, which also covers a mixed selection of corpora in one book.
        if self.direction != "parent":
            found = self.verb_child.get((sid, node_id))
            if found:
                return found
            # Coordinated clauses hang their verbs under a COORD node.
            for coord_id in self.coord_children.get((sid, node_id), []):
                found = self.verb_child.get((sid, coord_id))
                if found:
                    return found
        up = self.parent.get((sid, node_id))
        tag = self.verb_postag.get((sid, up)) if up else None
        return (up, tag) if tag else None

    def apodosis_verb(self, sid: str, subordinator_id: str, protasis_id: str):
        # The verb the protasis hangs off, found by walking up from it and
        # stepping over the subordinator, which sits above the protasis verb in
        # AGDT and below it in UD.
        node = protasis_id
        for _ in range(6):
            node = self.parent.get((sid, node))
            if not node or node == "0":
                return None
            if node == subordinator_id:
                continue
            tag = self.verb_postag.get((sid, node))
            if tag:
                return (node, tag)
        return None

    def head_verb(self, sid: str, node_id: str):
        # The verb the whole clause depends on: a conditional's apodosis, or the
        # matrix verb of a subordinate clause.
        tag = self.verb_postag.get((sid, node_id))
        if tag:
            return (node_id, tag)
        return self.clause_verb(sid, node_id)


def _mood(tag: str):
    return MOOD.get(tag[4:5]) if tag and len(tag) > 4 else None


def _tense(tag: str):
    return TENSE.get(tag[3:4]) if tag and len(tag) > 3 else None


# The six types a textbook teaches, or None. ei carrying an an dependent is ean
# written apart, which Perseus does and which otherwise loses roughly sixty per
# cent of the future more vivid conditions.
def _conditional_type(protasis_tag: str, apodosis_tag: str, is_ean: bool, apodosis_an: bool):
    pm, pt = _mood(protasis_tag), _tense(protasis_tag)
    am, at = _mood(apodosis_tag), _tense(apodosis_tag)

    # A prescription with an imperative apodosis is not a general condition, and
    # letting it through labelled one instance of it a present general.
    if am not in ("indicative", "optative"):
        return None

    if is_ean and pm == "subjunctive":
        if at == "future":
            return "future_more_vivid"
        if at == "present" and am == "indicative":
            return "present_general"
        return None
    if pm == "optative":
        if am == "optative" and apodosis_an:
            return "future_less_vivid"
        if at == "imperfect" and am == "indicative":
            return "past_general"
        return None
    if pm == "indicative" and pt == "imperfect" and apodosis_an:
        return "present_contrafactual"
    if pm == "indicative" and pt in ("aorist", "pluperfect") and apodosis_an:
        return "past_contrafactual"
    if pm == "indicative" and not apodosis_an:
        return "simple"
    return None


def _mood_tense_label(tag: str):
    mood, tense = _mood(tag), _tense(tag)
    if not mood:
        return None
    return f"{tense}|{mood}" if tense else mood


# One entry per reference section. "trigger" selects the tokens worth looking at,
# "classify" places each in a category or returns None to drop it.
SYNTAX_ENTRIES: dict[str, dict] = {
    "an": {
        "lemmas": {AN},
        "kind": "modal",
        "order": ["potential", "indefinite", "contrafactual"],
    },
    "conditionals": {
        "lemmas": {EI, EAN},
        "kind": "conditional",
        "order": [
            "simple",
            "present_general",
            "future_more_vivid",
            "future_less_vivid",
            "present_contrafactual",
            "past_contrafactual",
            "past_general",
        ],
    },
    "purpose": {"lemmas": {"ἵνα"}, "kind": "clause"},
    # Kept apart from hina: after verbs of striving hopos takes a future
    # indicative and the clause is one of effort, not purpose, so merging the
    # two would hide the very contrast the section describes.
    "effort": {"lemmas": {"ὅπως"}, "kind": "clause"},
    "result": {"lemmas": {"ὥστε"}, "kind": "clause"},
    "temporal": {"lemmas": {"ἐπεί", "ἐπειδή", "ὅτε", "ἕως"}, "kind": "clause"},
    "prin": {"lemmas": {"πρίν"}, "kind": "clause"},
    "hoti": {"lemmas": {"ὅτι"}, "kind": "clause"},
}


def catalog_author(file_name, document_id=None):
    # Only the curated catalog answers here. A work it does not know is treated
    # as prose rather than dropped, since the verse treebanks are all catalogued.
    entry = WORK_CATALOG.get(tlg_work_key(file_name or "", document_id) or "")
    return entry[0] if entry else None


def _is_prose(file_name, document_id, author_lookup) -> bool:
    lookup = author_lookup or catalog_author
    return lookup(file_name, document_id) not in VERSE_AUTHORS


def reference_rows(
    df: pd.DataFrame,
    entry_name: str,
    fmt: str = "agdt-xml",
    author_lookup=None,
    index: ClauseIndex | None = None,
) -> pd.DataFrame:
    # Every classified instance of one construction, with the provenance an
    # example line needs. Length and prose gates are applied here so a caller
    # cannot accidentally print a forty-word line of tragedy.
    entry = SYNTAX_ENTRIES[entry_name]
    index = index or ClauseIndex(df, fmt)
    d = index.d

    triggers = d[d["lemma_n"].isin(entry["lemmas"])]
    kind = entry["kind"]
    out = []

    for row in triggers.itertuples(index=False):
        sid, wid = row.sentence_id, row.word_id
        words = index.sentence_words.get(sid, 0)
        if not (EXAMPLE_MIN_WORDS <= words <= EXAMPLE_MAX_WORDS):
            continue
        if not _is_prose(getattr(row, "file", None), getattr(row, "document_id", None), author_lookup):
            continue

        category = None
        if kind == "modal":
            verb = index.head_verb(sid, row.head)
            if not verb:
                continue
            mood, tense = _mood(verb[1]), _tense(verb[1])
            if mood == "optative":
                category = "potential"
            elif mood == "subjunctive":
                category = "indefinite"
            elif mood == "indicative" and tense in ("imperfect", "aorist", "pluperfect"):
                category = "contrafactual"
        elif kind == "conditional":
            # eite ... eite is "whether ... or", a disjunction filed under ei; it
            # is not a conditional and reads as a poor example of one.
            if (sid, wid) in index.has_te:
                continue
            protasis = index.clause_verb(sid, wid)
            if not protasis:
                continue
            apodosis = index.apodosis_verb(sid, wid, protasis[0])
            if not apodosis:
                continue
            if row.lemma_n == EI and index.verb_lemma.get((sid, apodosis[0])) in ASKING_VERBS:
                continue
            is_ean = row.lemma_n == EAN or (sid, wid) in index.has_an
            category = _conditional_type(
                protasis[1], apodosis[1], is_ean, (sid, apodosis[0]) in index.has_an
            )
        else:
            verb = index.clause_verb(sid, wid)
            if not verb:
                continue
            category = _mood_tense_label(verb[1])

        if not category:
            continue
        out.append(
            {
                "entry": entry_name,
                "category": category,
                "trigger": row.lemma_n,
                "sentence_id": sid,
                "file": getattr(row, "file", None),
                "document_id": getattr(row, "document_id", None),
                "subdoc": getattr(row, "subdoc", None),
                "words": words,
            }
        )

    return pd.DataFrame(out)


def all_reference_rows(df: pd.DataFrame, fmt: str = "agdt-xml", author_lookup=None):
    # One index serves every entry; rebuilding it per section cost more than the
    # whole detection does.
    index = ClauseIndex(df, fmt)
    return {
        name: reference_rows(df, name, fmt, author_lookup, index=index)
        for name in SYNTAX_ENTRIES
    }
