"""Validate a treebank file against Didaskalos's parser adapters.

Run this before committing a new treebank so you know it parses into the schema
the pipeline expects, and that its morphology decodes into recognizable syllabus
categories.

Usage::

    py -3 validate_treebank.py <file> [--format agdt-xml|conllu]

With no --format the format is auto-detected (by extension, then content).
Exit code is non-zero when the file yields no tokens.
"""

from __future__ import annotations

import argparse
from pathlib import Path

try:
    from treebank_parsers import PARSERS, detect_format, parse_treebank_file
    from didaskalos_pipeline import parse_pos_category, parse_postag
except ImportError:  # imported as part of a package rather than as a flat module
    from .treebank_parsers import PARSERS, detect_format, parse_treebank_file
    from .didaskalos_pipeline import parse_pos_category, parse_postag


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("file", help="Path to the treebank file to validate.")
    parser.add_argument(
        "--format",
        choices=sorted(PARSERS),
        default=None,
        help="Force a parser instead of auto-detecting.",
    )
    args = parser.parse_args()

    path = Path(args.file)
    if not path.exists():
        print(f"File not found: {path}")
        return 1

    fmt = args.format or detect_format(path)
    print(f"File:   {path}")
    print(f"Format: {fmt}")

    df = parse_treebank_file(path, args.format)
    if df.empty:
        print("FAIL: no tokens parsed. The file did not yield any word/token rows.")
        return 1

    postags = df["postag"].fillna("")
    missing = int((postags == "").sum())
    syllabus = postags.apply(parse_postag)
    pos_category = postags.apply(parse_pos_category)
    # "Undecodable" = a real word (not punctuation/other) whose postag the pipeline
    # cannot classify into a syllabus label. A high count signals a bad postag map.
    undecodable = int(((syllabus == "NA") & (pos_category != "other")).sum())

    print(f"Sentences:            {df['sentence_id'].nunique()}")
    print(f"Tokens:               {len(df)}")
    print(f"Missing postag:       {missing}")
    print(f"Undecodable postag:   {undecodable}")
    print("POS categories:")
    for category, count in pos_category.value_counts().items():
        print(f"  {str(category):16} {count}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
