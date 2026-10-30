"""Candidate-level analysis of one matching run: evidence, KPIs, comparison and exports.

Everything here is derived from the ranking returned by ``BaseMatcher.match`` and the
resume text; the similarity scores are passed through unchanged.

    candidates = build_candidates(results, corpus, requirements, job_description, method)
    kpis = summarize_run(candidates, resumes_processed=len(corpus), threshold=60)
    selected = select_candidates(candidates, ["a.pdf", "b.pdf"])   # 2 to 5 resume IDs
    comparison_summary(selected), skill_matrix(selected), comparison_csv(selected, shortlist)

Two separate measures are kept side by side and never mixed: ``score`` is the matcher's document
similarity (cosine x 100); ``coverage`` is the percent of the job's requirements the resume shows
evidence for (coverage.py). Neither is a hiring probability.

No UI code and no file or network access: the app decides what to show and keeps any
state (selection, shortlist) in the user's session only.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from resume_matcher.config import (
    METHOD_GLOVE,
    METHOD_HYBRID,
    METHOD_LABELS,
    METHOD_SBERT,
    METHOD_SHORT,
    METHOD_TFIDF,
)
from resume_matcher.coverage import (
    STATUS_MATCHED,
    STATUS_RELATED,
    STATUS_SEMANTIC,
    RequirementCheck,
    assess,
    extract_education,
    extract_experience,
    job_requirements,
)
from resume_matcher.data_loader import ResumeCorpus
from resume_matcher.preprocessing import (
    STOP_WORDS,
    _as_text,
    mask_contact_info,
    tokenize,
)
from resume_matcher.skills import Requirement, normalize_for_matching

MIN_COMPARE = 2
MAX_COMPARE = 5
NOT_AVAILABLE = "Not available"

# Default "strong match" threshold per method, on the 0-100 similarity scale. Methods use different
# scales (averaged GloVe vectors score most resumes 80-95), so one number cannot fit all of them.
# These defaults separate the three Data Science resumes in demo_samples/ from the rest for every
# method; they are a starting point that the user can change in the app, not a calibrated cut-off.
DEFAULT_STRONG_THRESHOLDS = {
    METHOD_TFIDF: 20.0,
    METHOD_GLOVE: 82.0,
    METHOD_SBERT: 70.0,
    METHOD_HYBRID: 60.0,
}

# Columns of the table returned by build_candidates (one row per ranked resume).
CANDIDATE_COLUMNS = [
    "rank", "resume_id", "score", "cosine_similarity", "method",
    "requirements", "matched", "unconfirmed", "matched_count", "requirement_count",
    "coverage", "related", "checks",
    "experience", "education", "keywords", "preview", "text",
]

# ---- Evidence taken from the resume text (shown as quotes, never inferred)
# extract_experience and extract_education live in coverage.py and are re-exported here.

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
    """Add requirement coverage and resume evidence to the ranking returned by ``BaseMatcher.match``.

    ``rank``, ``resume_id``, ``score`` and ``cosine_similarity`` are copied as they are; the
    other columns only explain the ranking. ``requirements`` are the job's skill requirements
    (skills.extract_requirements); the degree level and years of experience it asks for are
    read from ``job_description`` and checked as well. ``unconfirmed`` lists every requirement
    that is not matched (related or missing); ``related`` the ones with a related skill only.
    """
    if method not in METHOD_LABELS:
        raise ValueError(f"Unknown method '{method}'.")
    job = job_requirements(job_description, skills=requirements)
    texts = dict(zip(corpus.df["resume_id"], corpus.df["text"]))
    rows = []
    for row in results.itertuples(index=False):
        text = _as_text(texts.get(row.resume_id, ""))
        report = assess(job, text)
        matched = report.matched
        rows.append(
            {
                "rank": int(row.rank),
                "resume_id": row.resume_id,
                "score": float(row.score),
                "cosine_similarity": float(row.cosine_similarity),
                "method": method,
                "requirements": job.names,
                "matched": matched,
                "unconfirmed": tuple(c.name for c in report.checks if c.status != STATUS_MATCHED),
                "matched_count": len(matched),
                "requirement_count": report.total,
                "coverage": report.score,
                "related": report.related,
                "checks": report.checks,
                "experience": extract_experience(text),
                "education": extract_education(text),
                "keywords": shared_keywords(job_description, text, exclude=matched),
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
    average_coverage: float | None = None  # None when the job lists no checkable requirements


def summarize_run(candidates: pd.DataFrame, resumes_processed: int, threshold: float) -> RunSummary:
    """KPIs from the ranked candidates. A strong match has a similarity score >= ``threshold``."""
    scores = candidates["score"].astype(float)
    coverage = candidates["coverage"].dropna().astype(float) if "coverage" in candidates else pd.Series(dtype=float)
    return RunSummary(
        resumes_processed=int(resumes_processed),
        candidates_ranked=len(candidates),
        strong_matches=int((scores >= threshold).sum()),
        threshold=float(threshold),
        average_score=float(scores.mean()) if len(scores) else None,
        average_coverage=float(coverage.mean()) if len(coverage) else None,
    )


# ---- Searching, filtering and sorting the ranking

SORT_OPTIONS = {
    "rank": "Rank (highest similarity first)",
    "score_asc": "Similarity score (lowest first)",
    "coverage": "Requirement coverage (highest first)",
    "requirements": "Requirements found (most first)",
    "name": "Candidate (A–Z)",
}

# "Show Top K Resumes" choices in the dashboard; None shows every ranked resume (over several pages).
# They only limit what is displayed: every processed resume is always ranked.
RESULT_LIMITS = {"Top 5": 5, "Top 10": 10, "Top 20": 20, "Top 30": 30, "Top 50": 50, "All Resumes": None}
DEFAULT_RESULT_LIMIT = "Top 30"


def filter_candidates(
    candidates: pd.DataFrame,
    query: str = "",
    shortlist: set[str] | frozenset[str] = frozenset(),
    shortlisted_only: bool = False,
    strong_only: bool = False,
    threshold: float = 0.0,
    must_include: tuple[str, ...] | list[str] = (),
    min_score: float = 0.0,
    min_coverage: float = 0.0,
) -> pd.DataFrame:
    """Rows matching every active filter. ``query`` searches file names and requirement names.

    ``min_score`` and ``min_coverage`` are optional floors (0 keeps everything); resumes below
    them stay ranked, they are only hidden from this view.
    """
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
    if min_score > 0:
        keep &= candidates["score"] >= min_score
    if min_coverage > 0:
        keep &= candidates["coverage"].fillna(0) >= min_coverage
    return candidates[keep]


def limit_results(candidates: pd.DataFrame, limit: int | None) -> pd.DataFrame:
    """The first ``limit`` rows (all of them when ``limit`` is None). Only limits what is shown."""
    return candidates if limit is None else candidates.head(limit)


def paginate(candidates: pd.DataFrame, page: int, page_size: int) -> tuple[pd.DataFrame, int]:
    """Rows of ``page`` (1-based, clamped to the valid range) and the number of pages (at least 1)."""
    pages = max(1, -(-len(candidates) // page_size))
    page = min(max(1, int(page)), pages)
    return candidates.iloc[(page - 1) * page_size : page * page_size], pages


def sort_candidates(candidates: pd.DataFrame, key: str) -> pd.DataFrame:
    """Sort by one of SORT_OPTIONS; ties always fall back to rank."""
    if key == "rank":
        return candidates.sort_values("rank", kind="stable")
    if key == "score_asc":
        return candidates.sort_values(["score", "rank"], ascending=[True, False], kind="stable")
    if key == "coverage":
        return candidates.sort_values(
            ["coverage", "rank"], ascending=[False, True], kind="stable", na_position="last"
        )
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


MATRIX_FOUND = "Found"
MATRIX_RELATED = "Related (unverified)"
MATRIX_SEMANTIC = "Semantically related (unverified)"
MATRIX_NOT_FOUND = "Not found"
_MATRIX_STATES = {STATUS_MATCHED: MATRIX_FOUND, STATUS_RELATED: MATRIX_RELATED, STATUS_SEMANTIC: MATRIX_SEMANTIC}


def status_matrix(selected: pd.DataFrame) -> pd.DataFrame:
    """Like skill_matrix, with the states Found, Related (unverified), Semantically related
    (unverified, only when semantic evidence was added to the checks) and Not found."""
    names = requirement_names(selected)
    data = {}
    for row in selected.itertuples(index=False):
        status = {check.name: check.status for check in row.checks}
        data[row.resume_id] = [_MATRIX_STATES.get(status.get(name), MATRIX_NOT_FOUND) for name in names]
    return pd.DataFrame(data, index=pd.Index(names, name="requirement"))


def evidence_lines(checks: tuple[RequirementCheck, ...], limit: int | None = None) -> list[str]:
    """"Requirement: quote" for each matched or related requirement that has evidence."""
    shown = (STATUS_MATCHED, STATUS_RELATED, STATUS_SEMANTIC)
    lines = [f"{c.name}: {c.evidence}" for c in checks if c.evidence and c.status in shown]
    return lines if limit is None else lines[:limit]


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
                "coverage_percent": "" if row.coverage is None else row.coverage,
                "matched_skills": _join(row.matched),
                "unconfirmed_skills": _join(row.unconfirmed),
                "related_unverified": _join(row.related),
                "evidence": " | ".join(evidence_lines(row.checks)),
                "experience_mentions": _join(row.experience) or NOT_AVAILABLE,
                "education_mentions": _join(row.education) or NOT_AVAILABLE,
                "shortlisted": "yes" if row.resume_id in shortlist else "no",
                **_method_columns(row),
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
    """The skill matrix as CSV: one row per requirement, "found" / "related (unverified)" /
    "not found" per candidate."""
    matrix = status_matrix(selected).map(str.lower)
    return to_csv_bytes(matrix, index=True)


def results_table(candidates: pd.DataFrame, shortlist: set[str] | frozenset[str] = frozenset()) -> pd.DataFrame:
    """The ranking export: one row per candidate with both scores, the requirement check and evidence.

    The first columns are those of the earlier results CSV (rank, resume_id, score,
    cosine_similarity, preview, method); the coverage columns follow.
    """
    rows = []
    for row in candidates.itertuples(index=False):
        rows.append(
            {
                "rank": row.rank,
                "resume_id": row.resume_id,
                "score": round(row.score, 2),
                "cosine_similarity": row.cosine_similarity,
                "preview": row.preview,
                "method": row.method,
                "coverage_percent": "" if row.coverage is None else row.coverage,
                "requirements_matched": f"{row.matched_count} of {row.requirement_count}" if row.requirement_count else "",
                "matched_requirements": _join(row.matched),
                "related_unverified": _join(row.related),
                "missing_requirements": _join([n for n in row.unconfirmed if n not in row.related]),
                "evidence": " | ".join(evidence_lines(row.checks)),
                "shortlisted": "yes" if row.resume_id in shortlist else "no",
                **_method_columns(row),
            }
        )
    return pd.DataFrame(rows)


# Optional per-method columns a run can carry (see app.add_method_details), and their export names.
METHOD_DETAIL_COLUMNS = {
    "tfidf_score": "tfidf_similarity", "glove_score": "glove_similarity", "sbert_score": "sbert_similarity",
    "sbert_document": "sbert_whole_resume_similarity", "best_section": "sbert_best_section",
}


def _method_columns(row) -> dict[str, object]:
    out = {}
    for column, export in METHOD_DETAIL_COLUMNS.items():
        value = getattr(row, column, None)
        if value is not None:
            out[export] = "" if isinstance(value, float) and value != value else value
    semantic = [c.name for c in getattr(row, "checks", ()) if c.status == STATUS_SEMANTIC]
    if semantic:
        out["semantically_related"] = _join(semantic)
    return out


def results_csv(candidates: pd.DataFrame, shortlist: set[str] | frozenset[str] = frozenset()) -> bytes:
    return to_csv_bytes(results_table(candidates, shortlist))


def masked_text(text: object) -> str:
    """Parsed resume text with e-mail addresses, links and phone numbers hidden, for display."""
    return mask_contact_info(text)


def score_bins(scores: pd.Series | np.ndarray, width: int = 10) -> pd.DataFrame:
    """Histogram counts of 0-100 scores in ``width``-point bins (every bin listed, also empty ones)."""
    edges = np.arange(0, 100 + width, width)
    counts, _ = np.histogram(np.clip(np.asarray(scores, dtype=float), 0, 100), bins=edges)
    labels = [f"{lo}–{hi}" for lo, hi in zip(edges[:-1], edges[1:])]
    return pd.DataFrame({"range": labels, "start": edges[:-1], "candidates": counts})


# ---- The same resumes ranked by several methods


def combine_method_rankings(results: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """One row per resume with its rank and score under each method (``BaseMatcher.match`` output).

    Resumes are joined by ID, never by position, so a resume keeps its own numbers under every
    method. A method that ranked fewer resumes leaves the others' cells empty instead of guessing.
    """
    table = None
    for method, ranking in results.items():
        part = ranking[["resume_id", "rank", "score"]].rename(
            columns={"rank": f"rank_{method}", "score": f"score_{method}"}
        )
        table = part if table is None else table.merge(part, on="resume_id", how="outer")
    if table is None:
        return pd.DataFrame(columns=["resume_id"])
    ranks = table[[f"rank_{m}" for m in results]]
    table["rank_spread"] = ranks.max(axis=1) - ranks.min(axis=1)
    return table


def ranking_agreement(table: pd.DataFrame, methods: list[str], top: int = 10) -> pd.DataFrame:
    """Pairwise agreement between methods: Kendall's tau over all resumes and overlap of the top ``top``."""
    from scipy.stats import kendalltau

    rows = []
    for i, a in enumerate(methods):
        for b in methods[i + 1 :]:
            both = table[[f"rank_{a}", f"rank_{b}"]].dropna()
            tau = kendalltau(both.iloc[:, 0], both.iloc[:, 1]).statistic if len(both) > 1 else np.nan
            k = min(top, len(both))
            top_a = set(both.nsmallest(k, f"rank_{a}").index)
            top_b = set(both.nsmallest(k, f"rank_{b}").index)
            rows.append({"methods": f"{METHOD_SHORT[a]} vs {METHOD_SHORT[b]}", "kendall_tau": round(float(tau), 3),
                         f"top_{top}_overlap": f"{len(top_a & top_b)} of {k}"})
    return pd.DataFrame(rows)
