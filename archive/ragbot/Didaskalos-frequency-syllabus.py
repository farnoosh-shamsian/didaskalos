# Frequency syllabus: generates frequency_syllabus.csv, the most frequent morph in the treebanks, for the Smyth RAG chatbot; noun and adjective declension is not counted, the datasets carry no info on it yet.

import re
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path().resolve()
FOLDER = BASE_DIR.parent / "treebanks" / "perseus"
OUTPUT_CSV = BASE_DIR / "frequency_syllabus.csv"

if not FOLDER.exists():
    raise FileNotFoundError(f"Perseus treebanks folder not found: {FOLDER}")

case_map = {
    "n": "nominative",
    "g": "genitive",
    "d": "dative",
    "a": "accusative",
    "v": "vocative",
}

tense_map = {
    "p": "present",
    "i": "imperfect",
    "f": "future",
    "a": "aorist",
    "r": "perfect",
    "l": "pluperfect",
    "t": "future perfect",
}

mood_map = {
    "i": "indicative",
    "s": "subjunctive",
    "o": "optative",
    "m": "imperative",
    "n": "infinitive",
    "p": "participle",
}

voice_map = {
    "a": "active",
    "m": "middle",
    "p": "passive",
    "e": "middle/passive",
}


def normalize_greek_lemma(lemma):
    if not isinstance(lemma, str):
        return ""
    decomposed = __import__("unicodedata").normalize("NFD", lemma.lower().strip())
    return "".join(char for char in decomposed if __import__("unicodedata").category(char) != "Mn")


def parse_postag(postag):
    if not isinstance(postag, str) or len(postag) < 9:
        return "NA"

    pos = postag[0]

    if pos in {"n", "a"}:
        return case_map.get(postag[7], postag[7] if postag[7] != "-" else "unknown")
    if pos == "v":
        ten = tense_map.get(postag[3], postag[3] if postag[3] != "-" else "unknown")
        moo = mood_map.get(postag[4], postag[4] if postag[4] != "-" else "unknown")
        voi = voice_map.get(postag[5], postag[5] if postag[5] != "-" else "unknown")
        return f"{ten}, {moo}, {voi}"
    if pos == "l":
        return "article"
    if pos == "p":
        return "pronouns"
    return "NA"


def parse_pos_category(postag):
    if not isinstance(postag, str) or not postag:
        return "other"
    pos = postag[0]
    if pos == "v":
        return "verb"
    if pos in {"n", "a"}:
        return "noun/adjective"
    if pos == "l":
        return "article"
    if pos == "p":
        return "pronoun"
    return "other"


def parse_verb_subcategory(lemma, postag=None):
    if postag and not str(postag).startswith("v"):
        return ""
    normalized_lemma = normalize_greek_lemma(lemma)
    if not normalized_lemma:
        return ""
    if normalized_lemma.endswith("ω"):
        return "ω"
    if normalized_lemma.endswith("μαι"):
        return "deponent"
    if normalized_lemma.endswith("μι"):
        return "μι"
    return "irregulars"


def parse_treebank_xml(file_path):
    tree = ET.parse(file_path)
    root = tree.getroot()

    data = []
    sentence_counter = 1
    token_index = 0
    current_sentence_id = f"tb_{sentence_counter}"

    for sentence in root.findall(".//sentence"):
        document_id = sentence.get("document_id")
        subdoc = sentence.get("subdoc")

        for word in sentence.findall("word"):
            token_index += 1
            form = word.get("form") or ""

            data.append(
                {
                    "sentence_id": current_sentence_id,
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

            if form in {".", "?", ";", "!", ":"}:
                sentence_counter += 1
                current_sentence_id = f"tb_{sentence_counter}"

    return pd.DataFrame(data)


all_dfs = []
for xml_file in sorted(FOLDER.glob("*.xml")):
    df = parse_treebank_xml(xml_file)
    if not df.empty:
        df["file"] = xml_file.name
        all_dfs.append(df)

if not all_dfs:
    raise ValueError(f"No XML treebanks were found in {FOLDER}")

combined_df = pd.concat(all_dfs, ignore_index=True)
combined_df["syllabus"] = combined_df["postag"].apply(parse_postag)
combined_df["pos_category"] = combined_df["postag"].apply(parse_pos_category)
combined_df["verb_subcategory"] = combined_df.apply(
    lambda row: parse_verb_subcategory(row["lemma"], row["postag"]) if row["pos_category"] == "verb" else "",
    axis=1,
)

frequency_syllabus = combined_df.copy()
frequency_syllabus["syllabus"] = frequency_syllabus.apply(
    lambda row: f"{row['syllabus']} ({row['verb_subcategory']})"
    if row["pos_category"] == "verb" and row.get("verb_subcategory")
    else row["syllabus"],
    axis=1,
)
frequency_syllabus = (
    frequency_syllabus.groupby(["syllabus", "pos_category"], dropna=False)
    .size()
    .reset_index(name="frequency")
    .sort_values("frequency", ascending=False)
    .reset_index(drop=True)
)

frequency_syllabus.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
print(f"Saved {len(frequency_syllabus)} rows to {OUTPUT_CSV.resolve()}")
print(frequency_syllabus.head(20).to_string(index=False))
