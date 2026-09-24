# Run before committing a new treebank, to check it parses into the schema the pipeline expects and that its morphology decodes into syllabus categories: py -3 validate_treebank.py <file> [--format agdt-xml|conllu], the format auto-detected without --format, and the exit code non-zero when the file yields no tokens.
from __future__ import annotations

import argparse
from pathlib import Path

try:
    from treebank_parsers import PARSERS, detect_format, parse_treebank_file
    from didaskalos_pipeline import parse_pos_category, parse_postag
except ImportError:  # imported as a package rather than a flat module
    from .treebank_parsers import PARSERS, detect_format, parse_treebank_file
    from .didaskalos_pipeline import parse_pos_category, parse_postag


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate a treebank file against Didaskalos's parser adapters."
    )
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
    # A real word whose postag yields no syllabus label; a high count means the postag map is wrong.
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
