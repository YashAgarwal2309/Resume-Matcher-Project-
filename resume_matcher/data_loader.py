"""Load, validate and clean resumes: from a dataset CSV or from uploaded files.

The CSV loader does not assume fixed column names. It looks for well-known
names first (``Resume_str``, ``Category``, ``ID`` in the Kaggle dataset)
and falls back to sensible guesses, so other CSV files also work.
Uploaded resumes (PDF, DOCX, TXT or MD) go through the same cleaning as the dataset.
"""

from __future__ import annotations

import html
import io
import json
import logging
import re
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from resume_matcher.preprocessing import (
    clean_text,
    decode_text_file,
    fix_mojibake,
    normalize_whitespace,
)

logger = logging.getLogger(__name__)

TEXT_COLUMN_CANDIDATES = ("resume_str", "resume_text", "resume", "text", "content", "cv")
CATEGORY_COLUMN_CANDIDATES = ("category", "job_category", "label", "role", "position")
ID_COLUMN_CANDIDATES = ("id", "resume_id", "candidate_id")
RESUME_FILE_TYPES = ("pdf", "docx", "txt", "md")

# A .docx file is a zip archive; its text is in word/document.xml. The XML is read with plain
# regular expressions (no XML parser, so no entity expansion), and its unpacked size is capped
# so a small "zip bomb" upload cannot use up memory.
MAX_DOCX_XML_BYTES = 30 * 1024 * 1024
_DOCX_PARAGRAPH_END_RE = re.compile(r"</w:p>|<w:br\b[^>]*/>|<w:cr\b[^>]*/>")
_DOCX_TAB_RE = re.compile(r"<w:tab\b[^>]*/>")
_DOCX_TEXT_RE = re.compile(r"<w:t(?:\s[^>]*)?>([^<]*)</w:t>|(\n|\t)")


class DatasetError(Exception):
    """Raised when the resume dataset cannot be used."""


@dataclass
class ResumeCorpus:
    """Cleaned resumes ready for matching.

    ``df`` always has these columns:
        resume_id  - stable identifier (from the dataset, or "row-<n>")
        category   - category label, or None if the dataset has none
        text       - original resume text (kept for display/traceability)
        clean_text - normalized text used by TF-IDF and GloVe
    """

    df: pd.DataFrame
    source: str
    text_column: str
    category_column: str | None = None
    id_column: str | None = None
    stats: dict[str, int] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.df)

    @property
    def has_categories(self) -> bool:
        return self.df["category"].notna().any()


def _find_column(columns: list[str], candidates: tuple[str, ...]) -> str | None:
    """Return the first column whose lowercase name is in ``candidates``."""
    lookup = {col.strip().lower(): col for col in columns}
    for name in candidates:
        if name in lookup:
            return lookup[name]
    return None


def detect_columns(df: pd.DataFrame) -> tuple[str, str | None, str | None]:
    """Work out which columns hold resume text, category and ID.

    Returns ``(text_column, category_column, id_column)``; the last two may be None.
    Raises DatasetError if no usable text column exists.
    """
    columns = [str(c) for c in df.columns]
    text_col = _find_column(columns, TEXT_COLUMN_CANDIDATES)

    if text_col is None:
        # Fallback: the string column with the longest average length,
        # ignoring HTML copies of the text (e.g. "Resume_html").
        best_len = 0.0
        for col in columns:
            is_text = pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_string_dtype(df[col])
            if "html" in col.lower() or not is_text:
                continue
            avg_len = df[col].dropna().astype(str).str.len().mean()
            if pd.notna(avg_len) and avg_len > best_len:
                text_col, best_len = col, avg_len
        if text_col is None or best_len < 20:
            raise DatasetError(
                f"Could not find a resume text column. Columns found: {columns}. "
                "Rename the text column to 'Resume_str' or 'text'."
            )
        logger.warning("No standard text column name found; using '%s'.", text_col)

    category_col = _find_column(columns, CATEGORY_COLUMN_CANDIDATES)
    id_col = _find_column(columns, ID_COLUMN_CANDIDATES)
    return text_col, category_col, id_col


def prepare_corpus(raw: pd.DataFrame, source: str = "<dataframe>") -> ResumeCorpus:
    """Validate a raw DataFrame and turn it into a ResumeCorpus."""
    if raw.empty:
        raise DatasetError(f"The dataset at {source} is empty.")

    text_col, category_col, id_col = detect_columns(raw)
    stats = {"rows_read": len(raw)}

    df = pd.DataFrame(index=raw.index)
    # Stable IDs: use the dataset's ID column, otherwise the original row number.
    if id_col is not None:
        ids = raw[id_col].astype(str).str.strip()
        fallback = pd.Series([f"row-{i}" for i in range(len(raw))], index=raw.index)
        df["resume_id"] = ids.where(raw[id_col].notna() & (ids != ""), fallback)
    else:
        df["resume_id"] = [f"row-{i}" for i in range(len(raw))]

    if category_col is not None:
        categories = raw[category_col].map(normalize_whitespace)
        df["category"] = categories.where(categories != "", None)
    else:
        df["category"] = None

    df["text"] = raw[text_col].map(fix_mojibake).map(normalize_whitespace)
    df["clean_text"] = df["text"].map(clean_text)

    # Drop rows whose text is missing or becomes empty after cleaning.
    empty = df["clean_text"] == ""
    stats["dropped_empty"] = int(empty.sum())
    df = df[~empty]

    # Drop exact duplicate resumes (same cleaned text), keeping the first one.
    duplicated = df.duplicated(subset="clean_text", keep="first")
    stats["dropped_duplicates"] = int(duplicated.sum())
    df = df[~duplicated]

    # Make IDs unique in case the dataset's ID column has repeats.
    repeated = df["resume_id"].duplicated(keep=False)
    if repeated.any():
        df.loc[repeated, "resume_id"] = (
            df.loc[repeated, "resume_id"] + "-" + df.loc[repeated].groupby("resume_id").cumcount().astype(str)
        )

    if df.empty:
        raise DatasetError(f"No usable resume text found in {source} after cleaning.")

    df = df.reset_index(drop=True)
    stats["rows_used"] = len(df)
    for key in ("dropped_empty", "dropped_duplicates"):
        if stats[key]:
            logger.info("%s: %d rows (%s)", key, stats[key], source)

    return ResumeCorpus(
        df=df,
        source=source,
        text_column=text_col,
        category_column=category_col,
        id_column=id_col,
        stats=stats,
    )


def load_resumes(path: str | Path) -> ResumeCorpus:
    """Read a resume CSV file and return a cleaned ResumeCorpus.

    Raises DatasetError with a readable message for missing, empty or
    unusable files.
    """
    path = Path(path)
    if not path.is_file():
        raise DatasetError(
            f"Resume dataset not found at '{path}'. Download Resume.csv "
            "(see data/README.md) and place it in the data/ folder, or set RESUME_DATA_PATH."
        )

    # All columns are read (not just the ones we need) so that pandas can spot and
    # skip malformed rows with too many fields; with `usecols` it accepts them silently.
    try:
        raw = pd.read_csv(path, on_bad_lines="skip", encoding_errors="replace")
    except pd.errors.EmptyDataError as exc:
        raise DatasetError(f"The dataset at {path} is empty.") from exc
    except (pd.errors.ParserError, ValueError) as exc:
        raise DatasetError(f"Could not read {path} as CSV: {exc}") from exc

    corpus = prepare_corpus(raw, source=str(path))
    logger.info(
        "Columns found %s; using text=%s, category=%s, id=%s",
        list(raw.columns), corpus.text_column, corpus.category_column, corpus.id_column,
    )
    return corpus


def load_sample_jobs(path: str | Path) -> list[dict[str, str]]:
    """Load the hand-written sample job descriptions (not part of the resume dataset).

    Returns a list of dicts with ``title``, ``target_category`` and ``text``.
    Returns an empty list if the file is missing or invalid.
    """
    path = Path(path)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        jobs = [j for j in data.get("jobs", []) if isinstance(j, dict) and str(j.get("text", "")).strip()]
    except (OSError, ValueError, AttributeError) as exc:
        logger.warning("Could not load sample job descriptions from %s: %s", path, exc)
        return []
    return jobs


def read_docx(name: str, data: bytes) -> str:
    """Text of a Word .docx file: paragraphs, line breaks and tabs, without formatting."""
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            info = archive.getinfo("word/document.xml")
            if info.file_size > MAX_DOCX_XML_BYTES:
                raise DatasetError(f"'{name}' is too large to read as a DOCX file.")
            xml = archive.read(info).decode("utf-8", errors="replace")
    except (zipfile.BadZipFile, KeyError, OSError, ValueError) as exc:
        raise DatasetError(f"Could not read '{name}' as a DOCX file: {exc}") from exc
    xml = _DOCX_PARAGRAPH_END_RE.sub("\n", _DOCX_TAB_RE.sub("\t", xml))
    parts = (text if text else sep for text, sep in _DOCX_TEXT_RE.findall(xml))
    return html.unescape("".join(parts))


def read_resume_file(name: str, data: bytes) -> str:
    """Return the text of one uploaded resume file (.pdf, .docx, .txt or .md).

    Raises DatasetError for unsupported or unreadable files. A scanned PDF (only
    images, no text layer) returns an empty string, since we do not run OCR.
    Old binary Word files (.doc) are not supported.
    """
    suffix = Path(name).suffix.lower().lstrip(".")
    if suffix not in RESUME_FILE_TYPES:
        raise DatasetError(f"'{name}' is not a supported file type ({', '.join(RESUME_FILE_TYPES)}).")
    if suffix == "docx":
        return read_docx(name, data)
    if suffix != "pdf":
        return decode_text_file(data)

    from pypdf import PdfReader
    from pypdf.errors import PdfReadError

    try:
        reader = PdfReader(io.BytesIO(data))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    except (PdfReadError, ValueError, OSError) as exc:
        raise DatasetError(f"Could not read '{name}' as a PDF: {exc}") from exc


def corpus_from_files(files: list[tuple[str, str]]) -> ResumeCorpus:
    """Build a ResumeCorpus from ``(file name, text)`` pairs; the file name becomes the resume ID."""
    if not files:
        raise DatasetError("No resumes uploaded.")
    raw = pd.DataFrame({"ID": [name for name, _ in files], "Resume": [text for _, text in files]})
    return prepare_corpus(raw, source="uploaded files")
