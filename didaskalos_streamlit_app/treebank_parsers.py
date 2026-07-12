"""Pluggable treebank parsers.

Every corpus format Didaskalos understands is parsed by an adapter registered in
``PARSERS``. An adapter takes a file path and returns a token ``DataFrame`` with
the schema the rest of the pipeline consumes::

    sentence_id, document_id, subdoc, word_id, token_index,
    form, lemma, postag, relation, head

The crucial contract is the ``postag`` column: it is always the Perseus / Ancient
Greek Dependency Treebank (AGDT) 9-character string, which is the single morphology
vocabulary the downstream pipeline decodes (``parse_postag`` / ``parse_pos_category``
in didaskalos_pipeline.py). Adding a new format therefore means writing one adapter
that normalizes its native morphology into that 9-character layout; nothing else in
the pipeline has to change.

AGDT postag layout (position -> feature)::

    0 part-of-speech  1 person  2 number  3 tense  4 mood
    5 voice           6 gender  7 case    8 degree
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd


# Strong sentence-ending punctuation (including the Greek ano teleia). Native
# sentence boundaries are primary; this only sub-splits a sentence into segments,
# and is shared by every adapter so segmentation is format-independent.
_END_PUNCT = {".", "?", ";", "!", ":", "·"}


# ---------------------------------------------------------------------------
# AGDT / Perseus XML  (format id: "agdt-xml")
# ---------------------------------------------------------------------------
def parse_agdt_xml(file_path: str | Path) -> pd.DataFrame:
    file_path = Path(file_path)
    tree = ET.parse(file_path)
    root = tree.getroot()

    data = []
    token_index = 0

    for fallback_counter, sentence in enumerate(root.findall(".//sentence"), 1):
        document_id = sentence.get("document_id")
        subdoc = sentence.get("subdoc")
        native_id = sentence.get("id") or f"s{fallback_counter}"
        segment = 1

        for word in sentence.findall("word"):
            token_index += 1
            form = word.get("form") or ""

            data.append(
                {
                    # The file stem keeps ids unique when several treebanks are
                    # combined into one DataFrame; without it, sentences from
                    # different works would merge in assemble_sentences.
                    "sentence_id": f"{file_path.stem}|{native_id}|{segment}",
                    "document_id": document_id,
                    "subdoc": subdoc,
                    "word_id": word.get("id"),
                    "token_index": token_index,
                    "form": form,
                    "lemma": word.get("lemma"),
                    "postag": word.get("postag"),
                    "relation": word.get("relation"),
                    "head": word.get("head"),
                }
            )

            if form in _END_PUNCT:
                segment += 1

    return pd.DataFrame(data)


# ---------------------------------------------------------------------------
# CoNLL-U  (Universal Dependencies / PROIEL; format id: "conllu")
# ---------------------------------------------------------------------------
# UD universal POS tag -> AGDT postag position 0 (part-of-speech letter).
_UPOS_TO_AGDT = {
    "NOUN": "n",
    "PROPN": "n",
    "ADJ": "a",
    "VERB": "v",
    "AUX": "v",
    "DET": "l",
    "PRON": "p",
    "ADV": "d",
    "ADP": "r",
    "PART": "g",
    "CCONJ": "c",
    "SCONJ": "c",
    "INTJ": "i",
    "NUM": "m",
    "PUNCT": "u",
    "SYM": "u",
    "X": "-",
}
# UD FEATS value -> AGDT single-letter code, one map per postag position. These
# mirror the CASE_MAP / TENSE_MAP / MOOD_MAP / VOICE_MAP decode tables in
# didaskalos_pipeline.py, reversed.
_FEAT_CASE = {"Nom": "n", "Gen": "g", "Dat": "d", "Acc": "a", "Voc": "v"}
_FEAT_TENSE = {"Pres": "p", "Imp": "i", "Fut": "f", "Aor": "a", "Perf": "r", "Pqp": "l", "FutPerf": "t"}
_FEAT_MOOD = {"Ind": "i", "Sub": "s", "Opt": "o", "Imp": "m", "Inf": "n", "Part": "p"}
_FEAT_VOICE = {"Act": "a", "Mid": "m", "Pass": "p", "MidPass": "e"}


def _parse_feats(feats: str) -> dict[str, str]:
    result: dict[str, str] = {}
    if feats and feats != "_":
        for pair in feats.split("|"):
            key, sep, value = pair.partition("=")
            if sep:
                result[key.strip()] = value.strip()
    return result


def _agdt_postag_from_ud(upos: str, xpos: str, feats: dict[str, str]) -> str:
    # Prefer the original AGDT tag when the UD source preserved it in XPOS (as
    # Perseus-UD does): a full 9-character alpha-initial tag is exactly what the
    # pipeline decodes, so pass it through unchanged. PROIEL's short XPOS ("Nb",
    # "V-") fails this length check and falls through to FEATS synthesis.
    if xpos and xpos != "_" and len(xpos) >= 8 and xpos[0].isalpha():
        return xpos

    slots = ["-"] * 9
    slots[0] = _UPOS_TO_AGDT.get(upos, "-")

    tense = _FEAT_TENSE.get(feats.get("Tense", ""))
    if not tense and feats.get("Aspect") == "Perf":
        tense = "r"  # Greek perfect is often encoded as Aspect=Perf in UD.
    if tense:
        slots[3] = tense
    mood = _FEAT_MOOD.get(feats.get("Mood", ""))
    if mood:
        slots[4] = mood
    voice = _FEAT_VOICE.get(feats.get("Voice", ""))
    if voice:
        slots[5] = voice
    case = _FEAT_CASE.get(feats.get("Case", ""))
    if case:
        slots[7] = case

    return "".join(slots)


def parse_conllu(file_path: str | Path) -> pd.DataFrame:
    file_path = Path(file_path)
    data = []
    token_index = 0
    sentence_counter = 0
    native_id: str | None = None
    document_id: str | None = None
    segment = 1

    with file_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.rstrip("\n")

            if not line.strip():
                # Blank line ends a sentence; reset per-sentence state.
                native_id = None
                segment = 1
                continue

            if line.startswith("#"):
                key, _, value = line[1:].partition("=")
                key = key.strip()
                value = value.strip()
                if key == "sent_id":
                    native_id = value or None
                elif key in ("newdoc id", "newdoc"):
                    document_id = value or document_id
                continue

            fields = line.split("\t")
            if len(fields) < 8:
                continue

            word_id = fields[0]
            # Skip multiword-token ranges ("7-8") and empty nodes ("7.1"); their
            # component words carry the morphology.
            if "-" in word_id or "." in word_id:
                continue

            form = fields[1] or ""
            lemma = fields[2]
            upos = fields[3]
            xpos = fields[4]
            feats = _parse_feats(fields[5])
            head = fields[6]
            deprel = fields[7]

            if native_id is None:
                sentence_counter += 1
                native_id = f"s{sentence_counter}"

            token_index += 1
            data.append(
                {
                    "sentence_id": f"{file_path.stem}|{native_id}|{segment}",
                    "document_id": document_id,
                    "subdoc": None,
                    "word_id": word_id,
                    "token_index": token_index,
                    "form": form,
                    "lemma": None if lemma in ("_", "") else lemma,
                    "postag": _agdt_postag_from_ud(upos, xpos, feats),
                    "relation": None if deprel in ("_", "") else deprel,
                    "head": None if head in ("_", "") else head,
                }
            )

            if form in _END_PUNCT:
                segment += 1

    return pd.DataFrame(data)


# ---------------------------------------------------------------------------
# Format registry + dispatcher
# ---------------------------------------------------------------------------
PARSERS = {
    "agdt-xml": parse_agdt_xml,
    "conllu": parse_conllu,
}


def detect_format(file_path: str | Path) -> str:
    """Best-effort format detection for uploads / ad-hoc URLs with no manifest.

    Uses the file extension first, then sniffs the leading bytes (XML documents
    start with '<'; CoNLL-U is tab-separated plain text).
    """
    file_path = Path(file_path)
    suffix = file_path.suffix.lower()
    if suffix in (".conllu", ".conll"):
        return "conllu"
    if suffix == ".xml":
        return "agdt-xml"

    try:
        with file_path.open("r", encoding="utf-8", errors="ignore") as handle:
            head = handle.read(4096)
    except OSError:
        return "agdt-xml"
    return "agdt-xml" if head.lstrip().startswith("<") else "conllu"


def parse_treebank_file(file_path: str | Path, fmt: str | None = None) -> pd.DataFrame:
    """Parse a treebank file with the adapter for ``fmt`` (or auto-detected)."""
    parser = PARSERS.get(fmt) if fmt else None
    if parser is None:
        parser = PARSERS[detect_format(file_path)]
    return parser(file_path)
