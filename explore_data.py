"""Exploratory analysis of the resume dataset.

Prints the dataset's shape, columns, missing values, duplicates, category
distribution, text-length statistics and a before/after look at cleaning.

The TF-IDF / GloVe / Sentence-BERT experiments that used to live at the end of
this script are now in matcher.py (implementation) and evaluate.py (comparison).

Usage:
    python explore_data.py               # uses data/Resume.csv (or RESUME_DATA_PATH)
    python explore_data.py path/to/Resume.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from resume_matcher.config import get_settings
from resume_matcher.data_loader import DatasetError, detect_columns, load_resumes


def explore(path: Path) -> None:
    corpus = load_resumes(path)  # validates the file first, with a friendly error
    raw = pd.read_csv(path, on_bad_lines="skip")
    print(f"File: {path}")
    print(f"Raw shape: {raw.shape}")
    print(f"Columns: {list(raw.columns)}")

    text_col, category_col, id_col = detect_columns(raw)
    print(f"Detected columns -> text: {text_col}, category: {category_col}, id: {id_col}")
    print("\nMissing values per column:")
    print(raw.isna().sum().to_string())
    print(f"\nExact duplicate resume texts (raw): {raw[text_col].duplicated().sum()}")

    df = corpus.df
    print(f"\nAfter cleaning: {len(df)} resumes. Stats: {corpus.stats}")

    if corpus.has_categories:
        counts = df["category"].value_counts()
        print(f"\n{len(counts)} categories:")
        print(counts.to_string())

    words = df["clean_text"].str.split().str.len()
    print("\nWords per cleaned resume:")
    print(words.describe().round(1).to_string())

    print("\nCleaning example (first 200 characters):")
    for _, row in df.head(3).iterrows():
        print(f"- [{row['resume_id']}] original: {row['text'][:200]!r}")
        print(f"  {'':>{len(str(row['resume_id'])) + 2}} cleaned:  {row['clean_text'][:200]!r}")


def main() -> int:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else get_settings().dataset_path
    pd.set_option("display.width", 160)
    try:
        explore(path)
    except (DatasetError, FileNotFoundError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
