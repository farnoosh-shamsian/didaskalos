# Pluggable treebank parsers, one adapter per corpus format registered in PARSERS: each takes a file path and returns a token DataFrame of sentence_id, document_id, subdoc, word_id, token_index, form, lemma, postag, relation, head, where postag is always the AGDT 9-character string (0 part-of-speech, 1 person, 2 number, 3 tense, 4 mood, 5 voice, 6 gender, 7 case, 8 degree), the one morphology vocabulary didaskalos_pipeline.py decodes, so a new format only needs an adapter that normalizes its own morphology into that layout.
from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd


# Strong sentence-ending punctuation, ";" being the Greek question mark, with the ano teleia and colon excluded as mid-sentence pauses in Greek; native sentence boundaries stay primary, this only sub-splits them into segments.
_END_PUNCT = {".", "?", ";", "!"}


# AGDT / Perseus XML (format id "agdt-xml").
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
            # Nodes the annotators inserted for gapping/ellipsis are not part of the written text, so they would be phantom words in the counts.
            if word.get("artificial"):
                continue

            token_index += 1
            form = word.get("form") or ""

            data.append(
                {
                    # The file stem keeps ids unique across combined treebanks, so assemble_sentences cannot merge two works' sentences.
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


# CoNLL-U (Universal Dependencies / PROIEL; format id "conllu"): UD universal POS tag -> AGDT postag position 0.
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
# UD FEATS value -> AGDT letter, one map per postag position; these are the decode tables in didaskalos_pipeline.py, reversed.
_FEAT_CASE = {"Nom": "n", "Gen": "g", "Dat": "d", "Acc": "a", "Voc": "v"}
_FEAT_TENSE = {"Pres": "p", "Imp": "i", "Fut": "f", "Aor": "a", "Perf": "r", "Pqp": "l", "FutPerf": "t"}
_FEAT_MOOD = {"Ind": "i", "Sub": "s", "Opt": "o", "Imp": "m", "Inf": "n", "Part": "p"}
_FEAT_VOICE = {"Act": "a", "Mid": "m", "Pass": "p", "MidPass": "e"}
_FEAT_PERSON = {"1": "1", "2": "2", "3": "3"}
_FEAT_NUMBER = {"Sing": "s", "Plur": "p", "Dual": "d"}
_FEAT_GENDER = {"Masc": "m", "Fem": "f", "Neut": "n"}


def _parse_feats(feats: str) -> dict[str, str]:
    result: dict[str, str] = {}
    if feats and feats != "_":
        for pair in feats.split("|"):
            key, sep, value = pair.partition("=")
            if sep:
                result[key.strip()] = value.strip()
    return result


def _agdt_postag_from_ud(upos: str, xpos: str, feats: dict[str, str]) -> str:
    # Prefer the original AGDT tag where the UD source kept it in XPOS, as Perseus-UD does; PROIEL's short XPOS ("Nb", "V-") fails the length check and falls through to FEATS synthesis.
    if xpos and xpos != "_" and len(xpos) >= 8 and xpos[0].isalpha():
        return xpos

    slots = ["-"] * 9
    slots[0] = _UPOS_TO_AGDT.get(upos, "-")

    person = _FEAT_PERSON.get(feats.get("Person", ""))
    if person:
        slots[1] = person
    number = _FEAT_NUMBER.get(feats.get("Number", ""))
    if number:
        slots[2] = number
    # Sources naming the Greek tense outright map directly; PROIEL splits it across Tense + Aspect, where Past+Perf is the aorist and Past+Imp the imperfect, and a bare Aspect=Perf is the last-resort perfect.
    tense = _FEAT_TENSE.get(feats.get("Tense", ""))
    if not tense:
        if feats.get("Tense") == "Past":
            tense = "i" if feats.get("Aspect") == "Imp" else "a"
        elif feats.get("Aspect") == "Perf":
            tense = "r"
    if tense:
        slots[3] = tense
    mood = _FEAT_MOOD.get(feats.get("Mood", ""))
    if mood:
        slots[4] = mood
    voice = _FEAT_VOICE.get(feats.get("Voice", ""))
    if voice:
        slots[5] = voice
    gender = _FEAT_GENDER.get(feats.get("Gender", ""))
    if gender:
        slots[6] = gender
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
    subdoc: str | None = None
    segment = 1

    with file_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.rstrip("\n")

            if not line.strip():
                # Blank line ends a sentence; reset per-sentence state.
                native_id = None
                subdoc = None
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
                elif key == "source":
                    # PROIEL tags each sentence with its locus ("Histories, Book 1, chapter 1"), and the work is already in the citation siglum, so only the trailing number is kept as the passage ref.
                    match = re.search(r"(\d+)\s*$", value)
                    subdoc = match.group(1) if match else None
                continue

            fields = line.split("\t")
            if len(fields) < 8:
                continue

            word_id = fields[0]
            # Multiword ranges ("7-8") and empty nodes ("7.1") carry no morphology of their own; their component words do.
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
                    "subdoc": subdoc,
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


PARSERS = {
    "agdt-xml": parse_agdt_xml,
    "conllu": parse_conllu,
}


def detect_format(file_path: str | Path) -> str:
    # For uploads and ad-hoc URLs with no manifest: extension first, then the leading bytes (XML starts with '<'; CoNLL-U is tab-separated text).
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
    # Parse with the adapter for fmt, or for the detected format.
    parser = PARSERS.get(fmt) if fmt else None
    if parser is None:
        parser = PARSERS[detect_format(file_path)]
    return parser(file_path)
