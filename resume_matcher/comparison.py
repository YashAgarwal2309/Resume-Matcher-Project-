"""Candidate-level analysis of one matching run: evidence, KPIs, comparison and exports.

Everything here is derived from the ranking returned by ``BaseMatcher.match`` and the
resume text; the similarity scores are passed through unchanged.

    candidates = build_candidates(results, corpus, requirements, job_description, method)
    kpis = summarize_run(candidates, resumes_processed=len(corpus), threshold=60)
    selected = select_candidates(candidates, ["a.pdf", "b.pdf"])   # 2 to 5 resume IDs
    comparison_summary(selected), skill_matrix(selected), comparison_csv(selected, shortlist)

No UI code and no file or network access: the app decides what to show and keeps any
state (selection, shortlist) in the user's session only.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import pandas as pd

from resume_matcher.config import METHOD_GLOVE, METHOD_HYBRID, METHOD_LABELS, METHOD_SBERT, METHOD_SHORT, METHOD_TFIDF
from resume_matcher.data_loader import ResumeCorpus
from resume_matcher.preprocessing import STOP_WORDS, _as_text, mask_contact_info, normalize_whitespace, tokenize
from resume_matcher.skills import Requirement, compare_requirements, normalize_for_matching

MIN_COMPARE = 2
MAX_COMPARE = 5
NOT_AVAILABLE = "Not available"

# Default "strong match" threshold per method, on the 0-100 similarity scale. Methods use different
# scales (averaged GloVe vectors score most resumes 80-95), so one number cannot fit all of them.
# These defaults separate the three Data Science resumes in demo_samples/ from the rest for every
# method; they are a starting point that the user can change in the app, not a calibrated cut-off.
DEFAULT_STRONG_THRESHOLDS = {
    METHOD_TFIDF: 25.0,
    METHOD_GLOVE: 90.0,
    METHOD_SBERT: 70.0,
    METHOD_HYBRID: 60.0,
}

# Columns of the table returned by build_candidates (one row per ranked resume).
CANDIDATE_COLUMNS = [
    "rank", "resume_id", "score", "cosine_similarity", "method",
    "requirements", "matched", "unconfirmed", "matched_count", "requirement_count",
    "experience", "education", "keywords", "preview", "text",
]

# ---- Evidence taken from the resume text (shown as quotes, never inferred)

_NUMBER_WORDS = (
    "one and a half|two and a half|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|"
    "fifteen|twenty"
)
_EXPERIENCE_RE = re.compile(
    r"(?:\b(?:about|around|over|more than|nearly|almost|approximately|a little over|close to)\s+)?"
    rf"\b(?:\d{{1,2}}(?:\.\d)?\s*\+?|(?:{_NUMBER_WORDS})\b\+?)\s*(?:years?|yrs?)\b"
    r"(?:\s+[\w-]+){0,3}?\s+experience\b"
    rf"|\bexperience\s+of\s+(?:about\s+|over\s+)?(?:\d{{1,2}}\s*\+?|(?:{_NUMBER_WORDS})\b)\s*(?:years?|yrs?)\b",
    re.IGNORECASE,
)
_MAX_EVIDENCE = 3

# Degree names. Abbreviations are case-sensitive and need their dots ("B.E.", "M.S."), so ordinary
# words such as "be", "ms" (MS Excel) or "ba" are not mistaken for degrees.
_DEGREE_RE = re.compile(
    r"(?<![A-Za-z])(?:"
    r"Ph\.?\s?D\b\.?|Doctorate\b|"
    r"[Mm]aster'?s?(?:\s+of|\s+in|\s+degree)\b|M\.\s?Tech\b|MTech\b|M\.\s?Sc\b|MSc\b|M\.S\.|M\.E\.|MBA\b|MCA\b|"
    r"M\.\s?Com\b|M\.A\.|"
    r"[Bb]achelor'?s?(?:\s+of|\s+in|\s+degree)\b|B\.\s?Tech\b|BTech\b|B\.\s?Sc\b|BSc\b|B\.S\.|B\.E\b\.?|BCA\b|"
    r"BBA\b|B\.\s?Com\b|B\.A\.|"
    r"[Dd]iploma\s+in\b|[Aa]ssociate'?s?\s+degree\b"
    r")"
)
# A degree mention ends at a field separator, a full stop followed by a space, or a year.
_DEGREE_END_RE = re.compile(r"\s*(?:[|;•▪●\n]|\.\s|\b(?:19|20)\d\d\b)")
_MAX_DEGREE_CHARS = 80

# Words that appear in most job descriptions and resumes and say nothing about fit.
_GENERIC_KEYWORDS = frozenset(
    (
        "experience", "experienced", "work", "working", "worked", "team", "teams", "role", "years",
        "year", "skills", "skill", "strong", "good", "ability", "knowledge", "using", "used", "use",
        "including", "plus", "required", "requirements", "preferred", "responsibilities", "job",
        "candidate", "company", "looking", "build", "building", "built", "new", "high", "well",
        "based", "etc", "will", "able", "help", "make", "related", "within", "across", "also",
    )
)
_MAX_KEYWORDS = 10


def extract_experience(text: object) -> tuple[str, ...]:
    """Phrases that state years of experience ("4 years of experience"), as written in the resume.

    Only explicit statements are returned; nothing is computed from employment dates.
    """
    found: list[str] = []
    for match in _EXPERIENCE_RE.finditer(normalize_whitespace(_as_text(text))):
        phrase = match.group().strip()
        if phrase.lower() not in (f.lower() for f in found):
            found.append(phrase)
        if len(found) == _MAX_EVIDENCE:
            break
    return tuple(found)


def extract_education(text: object) -> tuple[str, ...]:
    """Degree mentions such as "B.Tech, Computer Engineering", cut at the next separator or year."""
    text = normalize_whitespace(_as_text(text))
    found: list[str] = []
    for match in _DEGREE_RE.finditer(text):
        tail = text[match.start() : match.start() + _MAX_DEGREE_CHARS + 1]
        end = _DEGREE_END_RE.search(tail, len(match.group()))
        mention = tail[: end.start()] if end else tail
        if len(mention) > _MAX_DEGREE_CHARS:
            mention = mention[:_MAX_DEGREE_CHARS].rsplit(" ", 1)[0]
        mention = re.sub(r"\s+([,.;:)])", r"\1", mention).strip(" ,.-(")
        if mention and not any(mention in f or f in mention for f in found):
            found.append(mention)
        if len(found) == _MAX_EVIDENCE:
            break
    return tuple(found)


def shared_keywords(job_description: object, resume_text: object, exclude: tuple[str, ...] = ()) -> tuple[str, ...]:
    """Words used in both the job description and the resume, most frequent in the job first.

    ``exclude`` takes requirement names that are already listed elsewhere, so the keywords add
    evidence instead of repeating it.
    """
    job_tokens = [t.strip(".") for t in tokenize(job_description, remove_stop_words=True)]
    resume_tokens = {t.strip(".") for t in tokenize(resume_text, remove_stop_words=True)}
    skip = {w for name in exclude for w in normalize_for_matching(name).split()}
    counts: dict[str, int] = {}
    for token in job_tokens:
        if (
            len(token) < 3
            or token in STOP_WORDS
            or token in _GENERIC_KEYWORDS
            or token in skip
            or not any(ch.isalpha() for ch in token)
            or token not in resume_tokens
        ):
            continue
        counts[token] = counts.get(token, 0) + 1
    # dicts keep insertion order, so ties stay in job-description order.
    ranked = sorted(counts, key=lambda word: -counts[word])
    return tuple(ranked[:_MAX_KEYWORDS])


# ---- One row per ranked candidate


def build_candidates(
    results: pd.DataFrame,
    corpus: ResumeCorpus,
    requirements: list[Requirement] | tuple[Requirement, ...],
    job_description: str,
    method: str,
) -> pd.DataFrame:
    """Add requirement checks and resume evidence to the ranking returned by ``BaseMatcher.match``.

    ``rank``, ``resume_id``, ``score`` and ``cosine_similarity`` are copied as they are; the
    other columns only explain the ranking.
    """
    if method not in METHOD_LABELS:
        raise ValueError(f"Unknown method '{method}'.")
    texts = corpus.df.set_index("resume_id")["text"]
    rows = []
    for row in results.itertuples(index=False):
        text = _as_text(texts.get(row.resume_id, ""))
        comparison = compare_requirements(requirements, text)
        rows.append(
            {
                "rank": int(row.rank),
                "resume_id": row.resume_id,
                "score": float(row.score),
                "cosine_similarity": float(row.cosine_similarity),
                "method": method,
                "requirements": tuple(r.name for r in requirements),
                "matched": comparison.included,
                "unconfirmed": comparison.missing,
                "matched_count": len(comparison.included),
                "requirement_count": comparison.total,
                "experience": extract_experience(text),
                "education": extract_education(text),
                "keywords": shared_keywords(job_description, text, exclude=comparison.included),
                "preview": row.preview,
                "text": text,
            }
        )
    return pd.DataFrame(rows, columns=CANDIDATE_COLUMNS)


@dataclass(frozen=True)
class RunSummary:
    """Headline numbers for one matching run."""

    resumes_processed: int
    candidates_ranked: int
    strong_matches: int
    threshold: float
    average_score: float | None  # None when nothing was ranked


def summarize_run(candidates: pd.DataFrame, resumes_processed: int, threshold: float) -> RunSummary:
    """KPIs from the ranked candidates. A strong match has a similarity score >= ``threshold``."""
    scores = candidates["score"].astype(float)
    return RunSummary(
        resumes_processed=int(resumes_processed),
        candidates_ranked=len(candidates),
        strong_matches=int((scores >= threshold).sum()),
        threshold=float(threshold),
        average_score=float(scores.mean()) if len(scores) else None,
    )


# ---- Searching, filtering and sorting the ranking

SORT_OPTIONS = {
    "rank": "Rank (best first)",
    "score_asc": "Similarity score (lowest first)",
    "requirements": "Requirements found (most first)",
    "name": "Candidate (A–Z)",
}


def filter_candidates(
    candidates: pd.DataFrame,
    query: str = "",
    shortlist: set[str] | frozenset[str] = frozenset(),
    shortlisted_only: bool = False,
    strong_only: bool = False,
    threshold: float = 0.0,
    must_include: tuple[str, ...] | list[str] = (),
) -> pd.DataFrame:
    """Rows matching every active filter. ``query`` searches file names and requirement names."""
    keep = pd.Series(True, index=candidates.index)
    query = query.strip().lower()
    if query:
        keep &= candidates.apply(
            lambda row: query in str(row["resume_id"]).lower()
            or any(query in name.lower() for name in (*row["matched"], *row["unconfirmed"])),
            axis=1,
        )
    if shortlisted_only:
        keep &= candidates["resume_id"].isin(shortlist)
    if strong_only:
        keep &= candidates["score"] >= threshold
    for name in must_include:
        keep &= candidates["matched"].map(lambda matched, name=name: name in matched)
    return candidates[keep]


def sort_candidates(candidates: pd.DataFrame, key: str) -> pd.DataFrame:
    """Sort by one of SORT_OPTIONS; ties always fall back to rank."""
    if key == "rank":
        return candidates.sort_values("rank", kind="stable")
    if key == "score_asc":
        return candidates.sort_values(["score", "rank"], ascending=[True, False], kind="stable")
    if key == "requirements":
        return candidates.sort_values(["matched_count", "rank"], ascending=[False, True], kind="stable")
    if key == "name":
        return candidates.sort_values(
            ["resume_id", "rank"], key=lambda col: col.str.lower() if col.name == "resume_id" else col, kind="stable"
        )
    raise ValueError(f"Unknown sort key '{key}'. Choose from {list(SORT_OPTIONS)}.")


# ---- Comparison of 2 to 5 candidates


@dataclass(frozen=True)
class SelectionStatus:
    ok: bool
    message: str


def validate_selection(resume_ids: list[str] | tuple[str, ...], available: int | None = None) -> SelectionStatus:
    """Whether ``resume_ids`` can be compared (between MIN_COMPARE and MAX_COMPARE distinct IDs)."""
    count = len(set(resume_ids))
    if available is not None and available < MIN_COMPARE:
        return SelectionStatus(False, f"Comparison needs at least {MIN_COMPARE} ranked candidates.")
    if count != len(resume_ids):
        return SelectionStatus(False, "Each candidate can only be selected once.")
    if count < MIN_COMPARE:
        needed = MIN_COMPARE - count
        return SelectionStatus(False, f"Select {needed} more candidate{'s' if needed > 1 else ''} to compare.")
    if count > MAX_COMPARE:
        return SelectionStatus(False, f"Select at most {MAX_COMPARE} candidates (currently {count}).")
    return SelectionStatus(True, f"{count} candidates ready to compare.")


def select_candidates(candidates: pd.DataFrame, resume_ids: list[str] | tuple[str, ...]) -> pd.DataFrame:
    """The selected rows from one run, in rank order. Raises ValueError for an invalid selection,
    including IDs that are not part of this run (for example left over from an older run)."""
    status = validate_selection(list(resume_ids), available=len(candidates))
    if not status.ok:
        raise ValueError(status.message)
    unknown = [rid for rid in resume_ids if rid not in set(candidates["resume_id"])]
    if unknown:
        raise ValueError(f"Not part of the current ranking: {', '.join(unknown)}.")
    selected = candidates[candidates["resume_id"].isin(resume_ids)]
    if selected["method"].nunique() > 1:
        raise ValueError("Candidates ranked with different methods cannot be compared.")
    return selected.sort_values("rank", kind="stable")


def requirement_names(selected: pd.DataFrame) -> list[str]:
    """The job's requirements in the order they appear in the job description."""
    return [] if selected.empty else list(selected["requirements"].iloc[0])


def skill_matrix(selected: pd.DataFrame) -> pd.DataFrame:
    """Candidate-by-requirement table of booleans: True when the requirement was found in the resume.

    Rows are requirements, columns are resume IDs in rank order. Empty when the job
    description had no detectable requirements.
    """
    names = requirement_names(selected)
    data = {row.resume_id: [name in row.matched for name in names] for row in selected.itertuples(index=False)}
    return pd.DataFrame(data, index=pd.Index(names, name="requirement"))


@dataclass(frozen=True)
class SharedSkills:
    found_in_all: tuple[str, ...]
    found_in_some: tuple[str, ...]
    found_in_none: tuple[str, ...]


def shared_skills(selected: pd.DataFrame) -> SharedSkills:
    matrix = skill_matrix(selected)
    if matrix.empty:
        return SharedSkills((), (), ())
    hits = matrix.sum(axis=1)
    columns = matrix.shape[1]
    return SharedSkills(
        tuple(matrix.index[hits == columns]),
        tuple(matrix.index[(hits > 0) & (hits < columns)]),
        tuple(matrix.index[hits == 0]),
    )


def _names(ids: list[str]) -> str:
    if len(ids) == 1:
        return ids[0]
    return ", ".join(ids[:-1]) + " and " + ids[-1]


def comparison_summary(selected: pd.DataFrame) -> list[str]:
    """Plain-language findings for the selected candidates, using only the ranking data.

    Each sentence states the criterion it is based on. It never rates people or recommends a hire.
    """
    if len(selected) < MIN_COMPARE:
        return []
    method = METHOD_SHORT[selected["method"].iloc[0]]
    lines = []

    top_score = selected["score"].max()
    low_score = selected["score"].min()
    best = selected.loc[selected["score"] == top_score, "resume_id"].tolist()
    lowest = selected.loc[selected["score"] == low_score, "resume_id"].tolist()
    if top_score == low_score:
        lines.append(f"All selected candidates have the same {method} similarity score ({top_score:.2f}).")
    else:
        lines.append(
            f"{_names(best)} {'has' if len(best) == 1 else 'have'} the highest {method} similarity score "
            f"({top_score:.2f}) and {_names(lowest)} the lowest ({low_score:.2f}), a gap of "
            f"{top_score - low_score:.2f} points on the 0–100 scale."
        )

    total = int(selected["requirement_count"].iloc[0])
    if total == 0:
        lines.append(
            "No specific requirements were detected in the job description, so only the similarity "
            "scores can be compared."
        )
        return lines

    most = selected["matched_count"].max()
    least = selected["matched_count"].min()
    leaders = selected.loc[selected["matched_count"] == most, "resume_id"].tolist()
    if most == least:
        lines.append(f"Every selected resume mentions {most} of the {total} detected job requirements.")
    else:
        lines.append(
            f"{_names(leaders)} {'mentions' if len(leaders) == 1 else 'each mention'} the most detected "
            f"job requirements ({most} of {total}); the fewest is {least} of {total}."
        )
        if set(leaders).isdisjoint(best):
            lines.append(
                f"The highest similarity score and the most requirements found belong to different "
                f"candidates: {_names(best)} is closest to the job description as a whole, while "
                f"{_names(leaders)} mentions more of the listed requirements."
            )

    shared = shared_skills(selected)
    if shared.found_in_all:
        lines.append(f"Found in every selected resume: {', '.join(shared.found_in_all)}.")
    for row in selected.itertuples(index=False):
        only = [name for name in row.matched if name in shared.found_in_some
                and sum(name in other for other in selected["matched"]) == 1]
        if only:
            lines.append(f"Only {row.resume_id} mentions {', '.join(only)}.")
    if shared.found_in_none:
        lines.append(
            f"Not found in any selected resume: {', '.join(shared.found_in_none)} "
            "(the resumes may still describe these in other words)."
        )
    return lines


# ---- CSV exports

_FORMULA_PREFIXES = ("=", "+", "-", "@", "\t", "\r")


def _safe_cell(value: object) -> object:
    """Stop spreadsheet apps from running text that looks like a formula (CSV injection)."""
    if isinstance(value, str) and value.startswith(_FORMULA_PREFIXES):
        return "'" + value
    return value


def _join(values: tuple[str, ...] | list[str]) -> str:
    return "; ".join(values) if values else ""


def comparison_table(selected: pd.DataFrame, shortlist: set[str] | frozenset[str] = frozenset()) -> pd.DataFrame:
    """One row per selected candidate with the comparison fields (no resume text or contact details)."""
    rows = []
    for row in selected.itertuples(index=False):
        rows.append(
            {
                "candidate": row.resume_id,
                "rank": row.rank,
                "similarity_score": round(row.score, 2),
                "cosine_similarity": row.cosine_similarity,
                "matching_method": METHOD_LABELS[row.method],
                "requirements_found": f"{row.matched_count} of {row.requirement_count}" if row.requirement_count else "",
                "matched_skills": _join(row.matched),
                "unconfirmed_skills": _join(row.unconfirmed),
                "experience_mentions": _join(row.experience) or NOT_AVAILABLE,
                "education_mentions": _join(row.education) or NOT_AVAILABLE,
                "shortlisted": "yes" if row.resume_id in shortlist else "no",
            }
        )
    return pd.DataFrame(rows)


def to_csv_bytes(table: pd.DataFrame, index: bool = False) -> bytes:
    """CSV with formula-safe text cells, encoded as UTF-8 with a BOM so Excel shows accents correctly.

    pandas quotes cells that contain commas, quotes or line breaks.
    """
    safe = table.map(_safe_cell)
    if index:
        safe.index = safe.index.map(_safe_cell)
    return safe.to_csv(index=index).encode("utf-8-sig")


def comparison_csv(selected: pd.DataFrame, shortlist: set[str] | frozenset[str] = frozenset()) -> bytes:
    return to_csv_bytes(comparison_table(selected, shortlist))


def skill_matrix_csv(selected: pd.DataFrame) -> bytes:
    """The skill matrix as CSV: one row per requirement, "found" / "not found" per candidate."""
    matrix = skill_matrix(selected).replace({True: "found", False: "not found"})
    return to_csv_bytes(matrix, index=True)


def masked_text(text: object) -> str:
    """Parsed resume text with e-mail addresses, links and phone numbers hidden, for display."""
    return mask_contact_info(text)


def score_bins(scores: pd.Series | np.ndarray, width: int = 10) -> pd.DataFrame:
    """Histogram counts of 0-100 scores in ``width``-point bins (every bin listed, also empty ones)."""
    edges = np.arange(0, 100 + width, width)
    counts, _ = np.histogram(np.clip(np.asarray(scores, dtype=float), 0, 100), bins=edges)
    labels = [f"{lo}–{hi}" for lo, hi in zip(edges[:-1], edges[1:])]
    return pd.DataFrame({"range": labels, "start": edges[:-1], "candidates": counts})
