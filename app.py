"""Streamlit web interface for the Resume Matcher: a recruitment analytics dashboard.

Run with:  streamlit run app.py
The user types one job description and uploads resumes (PDF, DOCX, TXT or MD); nothing is preloaded.
Every readable resume is ranked; the result selector and pages only change how many are shown.
Matching lives in resume_matcher/matcher.py and candidate analysis (requirements, KPIs,
comparison, exports) in resume_matcher/comparison.py; this file only handles the UI and the
per-session state (results, comparison selection, shortlist).

Privacy: resume text, matchers and the shortlist are kept in this browser session only
(st.session_state). They are not shared between users or written to disk.
"""

from __future__ import annotations

import contextlib
import dataclasses
import hashlib
import logging
import uuid
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from resume_matcher.comparison import (
    DEFAULT_RESULT_LIMIT,
    DEFAULT_STRONG_THRESHOLDS,
    MATRIX_FOUND,
    MATRIX_RELATED,
    MATRIX_SEMANTIC,
    MAX_COMPARE,
    NOT_AVAILABLE,
    RESULT_LIMITS,
    SORT_OPTIONS,
    build_candidates,
    combine_method_rankings,
    comparison_csv,
    comparison_summary,
    filter_candidates,
    limit_results,
    masked_text,
    paginate,
    ranking_agreement,
    results_csv,
    score_bins,
    select_candidates,
    shared_skills,
    skill_matrix_csv,
    sort_candidates,
    status_matrix,
    summarize_run,
    validate_selection,
)
from resume_matcher.config import (
    MAX_RESUME_BYTES,
    METHOD_GLOVE,
    METHOD_HYBRID,
    METHOD_LABELS,
    METHOD_SBERT,
    METHOD_SHORT,
    METHOD_TFIDF,
    SEMANTIC_EVIDENCE_THRESHOLDS,
    Settings,
    get_settings,
)
from resume_matcher.coverage import (
    STATUS_MATCHED,
    STATUS_RELATED,
    STATUS_SEMANTIC,
    CoverageReport,
    semantic_evidence_batch,
)
from resume_matcher.data_loader import (
    RESUME_FILE_TYPES,
    DatasetError,
    ResumeCorpus,
    corpus_from_files,
    read_resume_file,
)
from resume_matcher.matcher import (
    BaseMatcher,
    EmptyQueryError,
    HybridMatcher,
    MatcherError,
    MethodUnavailableError,
    SbertMatcher,
    create_matcher,
    load_glove_vectors,
    load_sbert_model,
    method_status,
    validate_hybrid_weights,
)
from resume_matcher.preprocessing import (
    fix_mojibake,
    make_preview,
    normalize_whitespace,
)
from resume_matcher.skills import (
    Requirement,
    RequirementComparison,
    extract_requirements,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
for noisy in ("httpx", "httpx2", "urllib3", "huggingface_hub"):
    logging.getLogger(noisy).setLevel(logging.WARNING)

CSS_PATH = Path(__file__).resolve().parent / "assets" / "app.css"

METHOD_HELP = {
    METHOD_TFIDF: (
        "Counts how often each word appears in a resume, down-weighting words that are common across "
        "all uploaded resumes (TF-IDF). The job description is turned into the same kind of vector and "
        "compared with cosine similarity. Fast and transparent, but only rewards exact word overlap "
        "(\"developer\" and \"programmer\" are unrelated to it)."
    ),
    METHOD_GLOVE: (
        "Represents each document as the average of pretrained GloVe word vectors, so related words "
        "(\"physician\" / \"doctor\") land close together. The direction all averaged vectors share is "
        "removed, so scores spread out instead of all being 80–95. Averaging loses word order, and newer "
        "terms missing from the 2014 GloVe vocabulary (e.g. Kubernetes, TensorFlow) are ignored."
    ),
    METHOD_SBERT: (
        "Encodes text with a Sentence-BERT transformer model (default: BAAI/bge-small-en-v1.5, trained for "
        "search) into embeddings that capture meaning, so related wording still matches. The model reads at "
        "most 512 word-pieces, so each resume is also split into section-aware chunks; the score averages the "
        "whole-resume similarity and the best-matching chunk, so a relevant part late in a long resume counts."
    ),
    METHOD_HYBRID: (
        "Recommended. Puts the TF-IDF, GloVe and Sentence-BERT similarities on one scale (0–1 within this "
        "run for each method) and combines them with configurable weights (equal by default), so exact skill "
        "words, related words and overall meaning all count. The score is relative to the uploaded resumes; "
        "each method's own score is shown next to it. See README §13 for the evaluation."
    ),
}

# One-line descriptions shown under the method picker and in the header.
METHOD_TAGLINE = {
    METHOD_TFIDF: "Matches the exact words of the job description. Fastest.",
    METHOD_GLOVE: "Also counts related words (e.g. doctor / physician).",
    METHOD_SBERT: "Compares the overall meaning of the texts.",
    METHOD_HYBRID: "Recommended: combines TF-IDF, GloVe and Sentence-BERT on one scale.",
}

# Result tabs. The tab bar is stateful (key "view"), so "Compare Candidates" can switch to it.
TAB_RANKING = ":material/leaderboard: Ranking"
TAB_INSIGHTS = ":material/insights: Insights"
TAB_COMPARE = ":material/compare_arrows: Compare"
TABS = [TAB_RANKING, TAB_INSIGHTS, TAB_COMPARE]

# Chart colours, from the validated categorical palette (blue, orange); the lighter blue is a
# step of the same hue for scores below the strong-match threshold.
COLOR_FOUND = "#2a78d6"
COLOR_NOT_FOUND = "#eb6834"
COLOR_STRONG = "#2a78d6"
COLOR_BELOW = "#9ec5f4"

# Matchers kept per session (each holds the vectors of one set of uploaded resumes).
MAX_SESSION_MATCHERS = 4

# A set of uploaded resumes, as ((file name, text), ...).
UploadedFiles = tuple[tuple[str, str], ...]


@dataclass(frozen=True, eq=False)
class MatchRun:
    """Everything one click on "Match Resumes" produced. Later views only read from it, so the
    ranking, comparison and exports always belong to the same job description and method."""

    run_id: str  # new for every run; keys the comparison selection
    inputs_key: str  # method + resumes + job description, to notice later edits
    files_key: str
    shortlist_scope: str  # resumes + job description: the shortlist survives a method change
    method: str
    job_description: str
    requirements: tuple[Requirement, ...]  # the job's skill requirements
    results: pd.DataFrame  # as returned by BaseMatcher.match, every resume ranked
    candidates: pd.DataFrame  # results plus requirement coverage and evidence (comparison.py)
    resumes_processed: int  # readable, de-duplicated resumes (all of them are ranked)
    resumes_uploaded: int = 0
    files_failed: int = 0  # unreadable, empty, too large or unsupported
    duplicates_removed: int = 0
    methods_used: tuple[str, ...] = ()  # for Hybrid: the parts that contributed
    methods_skipped: tuple[tuple[str, str], ...] = ()  # (method, reason) for parts that could not read the job
    hybrid_weights: tuple[float, float, float] | None = None

    @property
    def requirement_names(self) -> tuple[str, ...]:
        """Every checked requirement (skills, degree, years of experience), in display order."""
        return tuple(self.candidates["requirements"].iloc[0]) if len(self.candidates) else ()


# --------------------------------------------------------------------------- keys and caches


def _digest(*parts: str) -> str:
    digest = hashlib.sha256()
    for part in parts:
        digest.update(part.encode("utf-8", errors="replace") + b"\x00")
    return digest.hexdigest()[:16]


def files_key(files: UploadedFiles) -> str:
    return _digest(*(f"{name}\x01{text}" for name, text in files))


def inputs_key(method: str, files: UploadedFiles, job_description: str, weights=None) -> str:
    return _digest(method, files_key(files), job_description, repr(weights) if method == METHOD_HYBRID else "")


def _session_lru(name: str) -> OrderedDict:
    """A per-session cache: nothing in it is visible to other users or kept after the session."""
    return st.session_state.setdefault(name, OrderedDict())


@st.cache_resource(show_spinner=False)
def get_glove_vectors(path: str):
    """Load the public GloVe table once; it is shared by every session (it holds no resume data)."""
    return load_glove_vectors(path)


@st.cache_resource(show_spinner=False)
def get_sbert_model(model_name: str, device: str | None):
    """Load the public Sentence-BERT model once (it holds no resume data)."""
    return load_sbert_model(model_name, device)


def matcher_settings(settings: Settings) -> Settings:
    """Settings for matching uploaded resumes: no on-disk embedding cache, so nothing derived from
    a candidate's resume is written to disk (evaluate.py still caches the public datasets)."""
    return dataclasses.replace(settings, cache_dir=None)


def get_corpus(files: UploadedFiles, key: str) -> ResumeCorpus:
    cache = _session_lru("_corpora")
    if key not in cache:
        cache[key] = corpus_from_files(list(files))
        while len(cache) > 2:
            cache.popitem(last=False)
    return cache[key]


def get_matcher(method: str, files: UploadedFiles, key: str, settings: Settings) -> BaseMatcher:
    """Build each matcher once per set of uploaded resumes in this session; resume vectors are
    reused across queries, so changing only the job description is fast."""
    cache = _session_lru("_matchers")
    cache_key = (method, key)
    if cache_key in cache:
        cache.move_to_end(cache_key)
        return cache[cache_key]
    word_vectors = sbert_model = None
    if method in (METHOD_GLOVE, METHOD_HYBRID):
        word_vectors = get_glove_vectors(str(settings.glove_path))
    if method in (METHOD_SBERT, METHOD_HYBRID):
        sbert_model = get_sbert_model(settings.sbert_model_name, settings.sbert_device)
    matcher = create_matcher(
        method, get_corpus(files, key), matcher_settings(settings), word_vectors=word_vectors, sbert_model=sbert_model
    )
    cache[cache_key] = matcher
    while len(cache) > MAX_SESSION_MATCHERS:
        cache.popitem(last=False)
    return matcher


def read_uploaded_resumes(uploaded) -> tuple[UploadedFiles, tuple[bytes, ...], list[str]]:
    """Return the readable resumes as (name, text) pairs, their original bytes (same order), and a
    message for each file that was skipped. Text is extracted once per file and session."""
    previous = st.session_state.get("_file_texts", {})
    texts: dict[tuple[str, str], str | DatasetError] = {}
    files, sources, problems = [], [], []
    new = [item for item in uploaded or [] if (item.name, item.file_id) not in previous]
    with st.spinner(f"Reading {len(new)} file(s)...") if new else contextlib.nullcontext():
        for item in uploaded or []:
            if item.size > MAX_RESUME_BYTES:
                problems.append(f"'{item.name}' is too large (limit {MAX_RESUME_BYTES // (1024 * 1024)} MB).")
                continue
            key = (item.name, item.file_id)
            if key in previous:
                texts[key] = previous[key]
            else:
                try:
                    texts[key] = read_resume_file(item.name, item.getvalue())
                except DatasetError as exc:
                    texts[key] = exc
            text = texts[key]
            if isinstance(text, DatasetError):
                problems.append(str(text))
                continue
            if not text.strip():
                problems.append(f"'{item.name}' has no readable text (a scanned PDF needs OCR first).")
                continue
            files.append((item.name, text))
            sources.append(item.getvalue())
    st.session_state["_file_texts"] = texts  # only the current uploads are kept
    return tuple(files), tuple(sources), problems


def original_files(files: UploadedFiles, sources: tuple[bytes, ...], corpus: ResumeCorpus) -> dict[str, tuple[str, bytes]]:
    """Map each resume ID to its uploaded (file name, bytes). IDs can differ from file names when
    two files share a name, so files are matched on their cleaned text, as the corpus was built."""
    by_text: dict[str, int] = {}
    for index, (_, text) in enumerate(files):
        by_text.setdefault(normalize_whitespace(fix_mojibake(text)), index)
    mapping = {}
    for row in corpus.df.itertuples(index=False):
        index = by_text.get(row.text)
        if index is not None:
            mapping[row.resume_id] = (files[index][0], sources[index])
    return mapping


# --------------------------------------------------------------------------- session state


def clear_results() -> None:
    """Callback: forget old results when the uploaded resumes change (they belong to the old set)."""
    st.session_state.pop("run", None)


def _forget_run_widgets() -> None:
    """Remove widget state that belonged to an older run (comparison selection, skill filter)."""
    for key in [k for k in st.session_state if str(k).startswith(("compare_sel_", "rank_must_"))]:
        del st.session_state[key]


def shortlist_for(run: MatchRun) -> set[str]:
    return st.session_state.setdefault("shortlists", {}).setdefault(run.shortlist_scope, set())


def _on_shortlist(scope: str, resume_id: str, key: str) -> None:
    shortlist = st.session_state.setdefault("shortlists", {}).setdefault(scope, set())
    if st.session_state[key]:
        shortlist.add(resume_id)
    else:
        shortlist.discard(resume_id)


def shortlist_toggle(run: MatchRun, resume_id: str, where: str) -> None:
    """A toggle bound to the session shortlist. Its state is reset from the shortlist on every run,
    so the toggles in the ranking and in the comparison always agree."""
    key = f"sl_{where}_{run.shortlist_scope}_{resume_id}"
    st.session_state[key] = resume_id in shortlist_for(run)
    st.toggle(
        "Shortlisted", key=key, on_change=_on_shortlist, args=(run.shortlist_scope, resume_id, key),
        help="Kept for this browser session only.",
    )


def selection_key(run: MatchRun) -> str:
    return f"compare_sel_{run.run_id}"


def _add_to_comparison(key: str, resume_id: str) -> None:
    selected = list(st.session_state.get(key, []))
    if resume_id in selected:
        selected.remove(resume_id)
    elif len(selected) < MAX_COMPARE:
        selected.append(resume_id)
    st.session_state[key] = selected


def _open_compare() -> None:
    st.session_state["view"] = TAB_COMPARE


# --------------------------------------------------------------------------- small UI helpers


def step_header(number: int, title: str, subtitle: str) -> None:
    """Numbered heading at the top of an input card."""
    st.markdown(
        f'<div class="step"><span class="num">{number}</span>'
        f'<div><div class="title">{title}</div><div class="sub">{subtitle}</div></div></div>',
        unsafe_allow_html=True,
    )


def section_title(title: str, subtitle: str = "") -> None:
    sub = f'<p class="section-sub">{subtitle}</p>' if subtitle else ""
    st.markdown(f'<p class="section-title">{title}</p>{sub}', unsafe_allow_html=True)


def field_label(text: str) -> None:
    st.markdown(f'<p class="field-label">{text}</p>', unsafe_allow_html=True)


def _badge_text(name: str) -> str:
    return name.replace("[", "(").replace("]", ")").replace(":", "\\:")


def badges(names: tuple[str, ...] | list[str], color: str, empty: str) -> None:
    if names:
        st.markdown(" ".join(f":{color}-badge[{_badge_text(n)}]" for n in names))
    else:
        st.markdown(f'<span class="na">{empty}</span>', unsafe_allow_html=True)


def evidence(values: tuple[str, ...]) -> None:
    """Quotes taken from the resume, or "Not available" when nothing reliable was found."""
    if values:
        st.markdown("\n".join(f"- {make_preview(v, 120)}" for v in values))
    else:
        st.markdown(f'<span class="na">{NOT_AVAILABLE}</span>', unsafe_allow_html=True)


def render_requirement_comparison(comparison: RequirementComparison) -> None:
    """Two-column table: the job's requirements found in this resume, and those not found."""
    if comparison.total == 0:
        st.info(
            "No specific skills or requirements were detected in the job description, so there is "
            "nothing to compare. Listing them (for example after \"Required:\") helps.",
            icon=":material/info:",
        )
        return
    st.table(
        pd.DataFrame(
            {
                "Included in Resume": [", ".join(comparison.included) or "None detected"],
                "Not Found in Resume": [", ".join(comparison.missing) or "None, all requirements found"],
            }
        ),
        hide_index=True,
    )


def threshold_for(method: str) -> float:
    return float(st.session_state.get(f"threshold_{method}", DEFAULT_STRONG_THRESHOLDS[method]))


# --------------------------------------------------------------------------- charts


# Height per bar row, so every candidate label has room however many rows there are.
ROW_STEP = alt.Step(30)


def score_chart(candidates: pd.DataFrame, threshold: float, title: str = "Similarity score") -> alt.Chart:
    """Horizontal bars on a fixed 0-100 scale, in rank order, with the strong-match threshold."""
    data = candidates[["rank", "resume_id", "score"]].assign(
        band=lambda d: d["score"].ge(threshold).map({True: "Strong match", False: "Below threshold"}),
        label=lambda d: "#" + d["rank"].astype(str) + " " + d["resume_id"],
    )
    order = data.sort_values("rank")["label"].tolist()
    bars = alt.Chart(data).mark_bar(cornerRadiusEnd=4, height={"band": 0.7}).encode(
        x=alt.X("score:Q", title=title + " (0–100)", scale=alt.Scale(domain=[0, 100])),
        y=alt.Y("label:N", sort=order, title=None, axis=alt.Axis(labelLimit=220)),
        color=alt.Color(
            "band:N", title=None,
            scale=alt.Scale(domain=["Strong match", "Below threshold"], range=[COLOR_STRONG, COLOR_BELOW]),
            legend=alt.Legend(orient="top"),
        ),
        tooltip=[
            alt.Tooltip("resume_id:N", title="Candidate"),
            alt.Tooltip("rank:Q", title="Rank"),
            alt.Tooltip("score:Q", title="Similarity", format=".2f"),
        ],
    )
    rule = alt.Chart(pd.DataFrame({"threshold": [threshold]})).mark_rule(strokeDash=[4, 4], color="#334155").encode(
        x="threshold:Q", tooltip=[alt.Tooltip("threshold:Q", title="Strong-match threshold")]
    )
    return (bars + rule).properties(height=ROW_STEP)


def distribution_chart(candidates: pd.DataFrame) -> alt.Chart:
    bins = score_bins(candidates["score"])
    return alt.Chart(bins).mark_bar(cornerRadiusEnd=4, color=COLOR_FOUND).encode(
        x=alt.X("start:O", title="Similarity score (10-point ranges, by lower bound)", axis=alt.Axis(labelAngle=0)),
        y=alt.Y("candidates:Q", title="Candidates", axis=alt.Axis(tickMinStep=1)),
        tooltip=[alt.Tooltip("range:N", title="Range"), alt.Tooltip("candidates:Q", title="Candidates")],
    ).properties(height=240)


def requirements_chart(candidates: pd.DataFrame) -> alt.Chart:
    """Stacked bars: requirements found vs not found in each resume."""
    rows = []
    for row in candidates.itertuples(index=False):
        label = f"#{row.rank} {row.resume_id}"
        rows.append({"label": label, "rank": row.rank, "status": "Found in resume", "count": row.matched_count})
        rows.append({"label": label, "rank": row.rank, "status": "Not found",
                     "count": row.requirement_count - row.matched_count})
    data = pd.DataFrame(rows)
    order = candidates.sort_values("rank").apply(lambda r: f"#{r['rank']} {r['resume_id']}", axis=1).tolist()
    return alt.Chart(data).mark_bar(stroke="#FFFFFF", strokeWidth=2, height={"band": 0.7}).encode(
        x=alt.X("count:Q", title="Job requirements", axis=alt.Axis(tickMinStep=1)),
        y=alt.Y("label:N", sort=order, title=None, axis=alt.Axis(labelLimit=220)),
        color=alt.Color(
            "status:N", title=None,
            scale=alt.Scale(domain=["Found in resume", "Not found"], range=[COLOR_FOUND, COLOR_NOT_FOUND]),
            legend=alt.Legend(orient="top"),
        ),
        order=alt.Order("status:N", sort="ascending"),
        tooltip=[alt.Tooltip("label:N", title="Candidate"), alt.Tooltip("status:N", title="Status"),
                 alt.Tooltip("count:Q", title="Requirements")],
    ).properties(height=ROW_STEP)


def coverage_chart(candidates: pd.DataFrame, requirements: tuple[str, ...]) -> alt.Chart:
    """How many of the ranked resumes mention each job requirement."""
    data = pd.DataFrame(
        {
            "requirement": list(requirements),
            "resumes": [int(candidates["matched"].map(lambda m, n=name: n in m).sum()) for name in requirements],
        }
    )
    total = len(candidates)
    return alt.Chart(data).mark_bar(cornerRadiusEnd=4, color=COLOR_FOUND, height={"band": 0.7}).encode(
        x=alt.X("resumes:Q", title=f"Ranked resumes that mention it (of {total})",
                scale=alt.Scale(domain=[0, max(total, 1)]), axis=alt.Axis(tickMinStep=1)),
        y=alt.Y("requirement:N", sort=list(requirements), title=None, axis=alt.Axis(labelLimit=200)),
        tooltip=[alt.Tooltip("requirement:N", title="Requirement"), alt.Tooltip("resumes:Q", title="Resumes")],
    ).properties(height=ROW_STEP)


def similarity_coverage_chart(candidates: pd.DataFrame, method: str) -> alt.Chart:
    """One dot per ranked resume: document similarity (x) against requirement coverage (y)."""
    data = candidates[["rank", "resume_id", "score", "coverage", "matched_count", "requirement_count"]].dropna(
        subset=["coverage"]
    )
    return alt.Chart(data).mark_circle(size=90, color=COLOR_FOUND, opacity=0.75, stroke="#FFFFFF", strokeWidth=1).encode(
        x=alt.X("score:Q", title=f"{METHOD_SHORT[method]} similarity (0–100)", scale=alt.Scale(domain=[0, 100])),
        y=alt.Y("coverage:Q", title="Requirement coverage (%)", scale=alt.Scale(domain=[0, 100])),
        tooltip=[
            alt.Tooltip("resume_id:N", title="Candidate"),
            alt.Tooltip("rank:Q", title="Rank"),
            alt.Tooltip("score:Q", title="Similarity", format=".2f"),
            alt.Tooltip("coverage:Q", title="Coverage %", format=".0f"),
            alt.Tooltip("matched_count:Q", title="Requirements matched"),
        ],
    ).properties(height=300)


# --------------------------------------------------------------------------- page sections


def render_header(method: str) -> None:
    with st.container(key="hero"):
        left, right = st.columns([3, 1.25], vertical_alignment="center", gap="large")
        with left:
            st.title("Resume Matcher", anchor=False)
            st.markdown(
                '<p class="subtitle">AI-Powered Recruitment Analytics</p>'
                '<p class="lede">Rank uploaded resumes against a job description with TF-IDF, GloVe, '
                "Sentence-BERT or a Hybrid of all three, see which job requirements each resume mentions, "
                "and compare up to five candidates side by side.</p>",
                unsafe_allow_html=True,
            )
        with right:
            st.markdown(
                f'<div class="algo"><span class="label">Selected algorithm</span>'
                f'<span class="name">{METHOD_SHORT[method]}</span>'
                f'<span class="sub">{METHOD_TAGLINE[method]}</span></div>',
                unsafe_allow_html=True,
            )


def render_sidebar(settings: Settings) -> None:
    with st.sidebar:
        st.markdown('<p class="side-title">Method status</p>', unsafe_allow_html=True)
        for method, label in METHOD_LABELS.items():
            ok, message = method_status(method, settings)
            st.markdown(
                f'<div class="method-row"><span class="dot {"ok" if ok else "off"}"></span>'
                f"<div><b>{label}</b><br><small>{message}</small></div></div>",
                unsafe_allow_html=True,
            )

        st.markdown('<p class="side-title">How it works</p>', unsafe_allow_html=True)
        for method, label in METHOD_LABELS.items():
            with st.expander(label):
                st.write(METHOD_HELP[method])

        st.markdown('<p class="side-title">Limitations</p>', unsafe_allow_html=True)
        st.caption(
            "The score (0–100) is cosine similarity × 100, with negative values shown as 0. It measures "
            "how similar the texts are, **not** the probability that a candidate is suitable or will be "
            "hired. Scores are not comparable across methods. Rankings can reflect wording and biases in "
            "the resumes, so use them to assist human review, never to make decisions automatically. "
            "Scanned PDFs (images only) cannot be read."
        )
        st.markdown('<p class="side-title">Privacy</p>', unsafe_allow_html=True)
        st.caption(
            "Uploaded resumes, results and the shortlist stay in this browser session's memory. "
            "Nothing is written to disk; closing the tab discards them. Exports you download may "
            "contain candidate information, so store and share them carefully."
        )


@dataclass
class Inputs:
    job_description: str
    files: UploadedFiles
    sources: tuple[bytes, ...]
    corpus: ResumeCorpus | None
    files_key: str
    method: str
    clicked: bool
    uploaded: int = 0
    failed: int = 0
    hybrid_weights: tuple[float, float, float] | None = None


def render_inputs(settings: Settings) -> Inputs:
    left, right = st.columns([1.15, 1], gap="medium")
    with left, st.container(border=True, key="card_job", height="stretch"):
        step_header(1, "Job description", "Paste the role, responsibilities and required skills")
        job_description = st.text_area(
            "Job description", key="jd_text", height=230, placeholder="Paste or type the job description here.",
            label_visibility="collapsed",
        )
        if job_description.strip():
            names = [r.name for r in extract_requirements(job_description)]
            if names:
                shown = ", ".join(names[:12]) + (f" and {len(names) - 12} more" if len(names) > 12 else "")
                st.caption(f":material/checklist: **{len(names)}** requirement(s) detected: {shown}")
            else:
                st.caption(
                    ":material/info: No specific requirements detected. Listing skills (e.g. after "
                    "\"Required:\") enables the requirement checks."
                )

    with right, st.container(border=True, key="card_resumes", height="stretch"):
        types = ", ".join(t.upper() for t in RESUME_FILE_TYPES)
        step_header(2, "Candidate resumes", f"{types} · up to {MAX_RESUME_BYTES // (1024 * 1024)} MB each · select several files")
        uploaded = st.file_uploader(
            f"Resumes ({types}, select several files)",
            type=list(RESUME_FILE_TYPES),
            accept_multiple_files=True,
            key="resume_files",
            on_change=clear_results,
            label_visibility="collapsed",
        )
        files, sources, problems = read_uploaded_resumes(uploaded)
        for message in problems[:10]:
            st.warning(message)
        if len(problems) > 10:
            st.warning(f"{len(problems) - 10} more file(s) could not be read.")
        corpus: ResumeCorpus | None = None
        key = files_key(files)
        if files:
            try:
                corpus = get_corpus(files, key)
            except DatasetError as exc:
                st.error(str(exc))
        if corpus is not None:
            st.caption(f":material/check_circle: **{len(corpus)}** resume(s) ready to match.")
            skipped = [f"{len(uploaded or [])} uploaded"]
            if problems:
                skipped.append(f"{len(problems)} unreadable")
            if corpus.stats["dropped_duplicates"]:
                skipped.append(f"{corpus.stats['dropped_duplicates']} duplicate file(s) ignored")
            if corpus.stats["dropped_empty"]:
                skipped.append(f"{corpus.stats['dropped_empty']} without usable text")
            st.caption(" · ".join(skipped))

    with st.container(border=True, key="card_settings"):
        step_header(3, "Matching settings", "Choose the algorithm. Every readable resume is ranked.")
        c1, c3 = st.columns([2.4, 1.1], vertical_alignment="bottom")
        method = c1.selectbox(
            "Matching algorithm", list(METHOD_LABELS), format_func=METHOD_LABELS.get, key="method",
            index=list(METHOD_LABELS).index(METHOD_HYBRID),
        )
        clicked = c3.button("Match Resumes", type="primary", width="stretch", icon=":material/manage_search:")
        hybrid_weights = None
        if method == METHOD_HYBRID:
            with st.expander("Hybrid weights", icon=":material/tune:"):
                w1, w2, w3 = st.columns(3)
                defaults = settings.hybrid_weights
                raw = (
                    w1.number_input("TF-IDF", 0.0, 1.0, round(defaults[0], 2), 0.05, key="w_tfidf"),
                    w2.number_input("GloVe", 0.0, 1.0, round(defaults[1], 2), 0.05, key="w_glove"),
                    w3.number_input("Sentence-BERT", 0.0, 1.0, round(defaults[2], 2), 0.05, key="w_sbert"),
                )
                try:
                    hybrid_weights = validate_hybrid_weights(raw)
                    st.caption(
                        "Rescaled to sum to 1: "
                        + ", ".join(f"{name} {w:.2f}" for name, w in zip(("TF-IDF", "GloVe", "Sentence-BERT"), hybrid_weights))
                        + ". Each method's similarities are put on a 0–1 scale within the run before weighting. "
                        "Equal weights were the most consistent choice in our evaluation; no tuned set was "
                        "better on every dataset (README §13)."
                    )
                except ValueError as exc:
                    st.caption(f":material/warning: {exc} Using the defaults.")
                    hybrid_weights = validate_hybrid_weights(defaults)
        ok, status = method_status(method, settings)
        details = [f"**{METHOD_SHORT[method]}:** {METHOD_TAGLINE[method]}"]
        if method in (METHOD_SBERT, METHOD_HYBRID):
            details.append(f"Sentence-BERT model: `{settings.sbert_model_name}` (set `SBERT_MODEL` to change).")
        details.append(("" if ok else ":material/warning: ") + status)
        st.caption("  \n".join(details))

    return Inputs(
        job_description, files, sources, corpus, key, method, clicked, len(uploaded or []), len(problems), hybrid_weights
    )


def run_matching(inputs: Inputs, settings: Settings) -> None:
    """Validate the inputs, rank the resumes and store the run in the session."""
    st.session_state.pop("run", None)
    if inputs.corpus is None:
        st.warning("Please upload at least one resume first.")
        return
    if not inputs.job_description.strip():
        st.warning("Please enter a job description first.")
        return
    label = METHOD_LABELS[inputs.method]
    try:
        with st.status(f"Matching {len(inputs.corpus)} resumes with {METHOD_SHORT[inputs.method]}...") as status:
            st.write(f"Building the {label} index (the first run of a model can take a while)...")
            matcher = get_matcher(inputs.method, inputs.files, inputs.files_key, settings)
            if isinstance(matcher, HybridMatcher) and inputs.hybrid_weights:
                matcher.weights = inputs.hybrid_weights  # weights only combine scores; no refit needed
            st.write(f"Ranking all {len(inputs.corpus)} resumes...")
            results = matcher.match(inputs.job_description, top_k=len(inputs.corpus))
            st.write("Checking the job requirements in each resume...")
            requirements = tuple(extract_requirements(inputs.job_description))
            candidates = build_candidates(
                results, inputs.corpus, requirements, inputs.job_description, inputs.method
            )
            candidates = add_method_details(candidates, matcher, inputs.corpus, inputs.job_description)
            status.update(
                label=f"Ranked {len(results)} of {len(inputs.corpus)} resumes with {METHOD_SHORT[inputs.method]}.",
                state="complete", expanded=False,
            )
    except MethodUnavailableError as exc:
        st.error(f"{label} is not available: {exc}")
        return
    except EmptyQueryError as exc:
        st.warning(str(exc))
        return
    except MatcherError as exc:
        st.error(f"Matching failed: {exc}")
        return

    used = tuple(matcher.last_used) if isinstance(matcher, HybridMatcher) else (inputs.method,)
    skipped = tuple(matcher.last_skipped.items()) if isinstance(matcher, HybridMatcher) else ()
    _forget_run_widgets()
    st.session_state["run"] = MatchRun(
        run_id=uuid.uuid4().hex[:12],
        inputs_key=inputs_key(inputs.method, inputs.files, inputs.job_description, inputs.hybrid_weights),
        files_key=inputs.files_key,
        shortlist_scope=_digest(inputs.files_key, inputs.job_description),
        method=inputs.method,
        job_description=inputs.job_description,
        requirements=requirements,
        results=results,
        candidates=candidates,
        resumes_processed=len(inputs.corpus),
        resumes_uploaded=inputs.uploaded,
        files_failed=inputs.failed,
        duplicates_removed=inputs.corpus.stats.get("dropped_duplicates", 0),
        methods_used=used,
        methods_skipped=skipped,
        hybrid_weights=matcher.weights if isinstance(matcher, HybridMatcher) else None,
    )
    st.session_state["view"] = TAB_RANKING


def _score(values: np.ndarray) -> np.ndarray:
    return np.round(np.clip(values, 0.0, 1.0) * 100, 2)


def add_method_details(candidates: pd.DataFrame, matcher: BaseMatcher, corpus: ResumeCorpus, job: str) -> pd.DataFrame:
    """Per-method scores behind a ranking, aligned to each candidate by resume ID.

    Hybrid: each part's own similarity (0-100, before the per-run rescaling). Sentence-BERT (alone
    or inside Hybrid): the whole-resume similarity and the section of the best-matching chunk.
    Missing values stay empty; nothing is filled in for a method that did not run.
    """
    order = {rid: i for i, rid in enumerate(corpus.df["resume_id"])}
    rows = candidates["resume_id"].map(order).to_numpy()
    extra: dict[str, object] = {}
    sbert = None
    if isinstance(matcher, HybridMatcher):
        for name in HybridMatcher.PART_NAMES:
            values = matcher.last_components.get(name)
            extra[f"{name}_score"] = _score(values[rows]) if values is not None else np.full(len(rows), np.nan)
        sbert = matcher.parts[2] if METHOD_SBERT in matcher.last_components else None
    elif isinstance(matcher, SbertMatcher):
        sbert = matcher
    if sbert is not None:
        extra["sbert_document"] = _score(sbert.document_similarities(job)[rows])
        best = sbert.best_chunks(job)
        extra["best_section"] = [best[r][0] if best[r] else "" for r in rows]
        extra["best_chunk"] = [make_preview(best[r][1], 400) if best[r] else "" for r in rows]
    return candidates.assign(**extra)


def render_empty_state() -> None:
    st.markdown(
        '<div class="empty"><b>Your candidate ranking will appear here.</b>'
        "<ol><li>Paste a job description.</li><li>Upload the candidates' resumes.</li>"
        "<li>Choose an algorithm and click <b>Match Resumes</b>.</li></ol></div>",
        unsafe_allow_html=True,
    )


def render_kpis(run: MatchRun, threshold: float) -> None:
    summary = summarize_run(run.candidates, run.resumes_processed, threshold)
    k1, k2, k3, k4 = st.columns(4)
    k1.metric(
        "Resumes uploaded", run.resumes_uploaded, border=True, icon=":material/upload_file:",
        help="Files selected in the uploader for this run.",
    )
    k2.metric(
        "Resumes processed", summary.resumes_processed, border=True, icon=":material/description:",
        help=f"Readable resumes after removing {run.duplicates_removed} exact duplicate(s).",
    )
    k3.metric(
        "Failed files", run.files_failed, border=True, icon=":material/error:",
        help="Unreadable, empty, too large or unsupported files (listed above the results).",
    )
    k4.metric(
        "Candidates ranked", summary.candidates_ranked, border=True, icon=":material/format_list_numbered:",
        help="Every processed resume is ranked; the result selector only changes how many are shown.",
    )
    k5, k6, k7, k8 = st.columns(4)
    k5.metric(
        "Strong matches", summary.strong_matches, border=True, icon=":material/verified:",
        help=f"Ranked resumes with a {METHOD_SHORT[run.method]} similarity score of at least {threshold:g}. "
        "Change the threshold with the button above.",
    )
    k6.metric(
        "Average similarity", "—" if summary.average_score is None else f"{summary.average_score:.1f}",
        border=True, icon=":material/functions:",
        help="Mean document similarity of the ranked candidates in this run (0–100). Only meaningful "
        "within one run and method; it is not a probability.",
    )
    k7.metric(
        "Average coverage", "—" if summary.average_coverage is None else f"{summary.average_coverage:.0f}%",
        border=True, icon=":material/checklist:",
        help="Mean share of the job's requirements that each resume shows evidence for. A separate "
        "measure from similarity; not a probability.",
    )
    k8.metric(
        "Requirements checked", len(run.requirement_names), border=True, icon=":material/rule:",
        help="Skills, degree level and years of experience detected in the job description.",
    )


def render_threshold_control(run: MatchRun) -> float:
    key = f"threshold_{run.method}"
    current = threshold_for(run.method)
    with st.popover(f"Strong match ≥ {current:g}", icon=":material/tune:", width="stretch"):
        st.number_input(
            f"Strong-match threshold ({METHOD_SHORT[run.method]})", min_value=0.0, max_value=100.0,
            value=DEFAULT_STRONG_THRESHOLDS[run.method], step=1.0, key=key,
        )
        st.caption(
            "A ranked resume counts as a strong match when its similarity score is at least this value. "
            f"Default for {METHOD_SHORT[run.method]}: {DEFAULT_STRONG_THRESHOLDS[run.method]:g}. Each method "
            "has its own default because the methods score on different scales (GloVe gives most resumes "
            "80–95). The threshold only changes the labels and counts, never the ranking."
        )
    return threshold_for(run.method)


def render_selection_bar(run: MatchRun) -> list[str]:
    """Pick 2 to 5 candidates from the current run; the button opens the Compare tab."""
    key = selection_key(run)
    ids = run.candidates["resume_id"].tolist()
    if key in st.session_state:  # keep only IDs that belong to this run, at most MAX_COMPARE
        valid = [rid for rid in st.session_state[key] if rid in ids][:MAX_COMPARE]
        if valid != list(st.session_state[key]):
            st.session_state[key] = valid
    labels = {row.resume_id: f"#{row.rank} · {row.resume_id} · {row.score:.1f}" for row in run.candidates.itertuples()}
    with st.container(border=True, key="card_select"):
        c1, c2 = st.columns([2.4, 1.2], vertical_alignment="bottom")
        selected = c1.multiselect(
            f"Candidates to compare (choose 2 to {MAX_COMPARE})", ids, format_func=labels.get,
            max_selections=MAX_COMPARE, key=key, placeholder="Choose candidates from this ranking",
            disabled=len(ids) < 2,
        )
        status = validate_selection(selected, available=len(ids))
        c2.button(
            "Compare Candidates", type="primary", width="stretch", icon=":material/compare_arrows:",
            disabled=not status.ok, on_click=_open_compare, key="compare_button",
        )
        st.caption(f"**{len(selected)} of {MAX_COMPARE}** selected · {status.message}")
    return selected


PAGE_SIZE = 25

_STATUS_ICONS = {
    STATUS_MATCHED: ":green[:material/check_circle:]",
    STATUS_RELATED: ":violet[:material/help:]",
    STATUS_SEMANTIC: ":blue[:material/lightbulb:]",
}
PART_LABELS = {METHOD_TFIDF: "TF-IDF", METHOD_GLOVE: "GloVe", METHOD_SBERT: "Sentence-BERT"}
SECTION_NAMES = {"summary": "Summary", "skills": "Skills", "experience": "Experience", "projects": "Projects",
                 "education": "Education", "certifications": "Certifications", "other": "Other"}


def similarity_label(method: str) -> str:
    if method == METHOD_HYBRID:
        return "Hybrid score (relative, 0–100)"
    return f"{METHOD_SHORT[method]} similarity (0–100)"


def sbert_for_run(run: MatchRun) -> SbertMatcher | None:
    """The fitted Sentence-BERT matcher behind this run (alone or inside Hybrid), if any."""
    matcher = _session_lru("_matchers").get((run.method, run.files_key))
    if isinstance(matcher, HybridMatcher):
        return matcher.parts[2]
    return matcher if isinstance(matcher, SbertMatcher) else None


def semantic_checks(run: MatchRun, rows: pd.DataFrame) -> dict[str, tuple]:
    """Requirement checks with semantic evidence added (Sentence-BERT runs only), per resume ID.

    Computed only for the rows shown, in one batch, and cached for the run, so paging back is
    instant and large uploads do not pay for resumes nobody looks at.
    """
    plain = dict(zip(rows["resume_id"], rows["checks"]))
    sbert = sbert_for_run(run)
    threshold = SEMANTIC_EVIDENCE_THRESHOLDS.get(get_settings().sbert_model_name)
    if sbert is None or threshold is None or run.method not in (METHOD_SBERT, METHOD_HYBRID):
        return plain
    cache = st.session_state.setdefault("_semantic", {})
    if cache.get("run_id") != run.run_id:
        cache.clear()
        cache["run_id"] = run.run_id
    todo = rows[~rows["resume_id"].isin(cache)]
    if len(todo):
        reports = semantic_evidence_batch(
            [CoverageReport(c) for c in todo["checks"]], todo["text"].tolist(),
            sbert.encode_queries, sbert.encode_passages, threshold,
        )
        for rid, report in zip(todo["resume_id"], reports, strict=True):
            cache[rid] = report.checks
    return {rid: cache.get(rid, plain[rid]) for rid in rows["resume_id"]}


def render_checks(row, checks: tuple | None = None) -> None:
    """One line per job requirement: matched (exact or alias), related, semantically related or not
    found, with the quote from the resume."""
    if not row.requirement_count:
        return
    lines = []
    for check in checks if checks is not None else row.checks:
        icon = _STATUS_ICONS.get(check.status, ":orange[:material/cancel:]")
        if check.status == STATUS_MATCHED:
            how = {"exact": "exact match", "alias": "recognised alias", "stated": "as stated"}.get(check.match_type, "")
            detail = f" ({how})" + (f": “{check.evidence}”" if check.evidence else "")
        elif check.status == STATUS_RELATED:
            detail = f": related only, not verified. {check.note} “{check.evidence}”"
        elif check.status == STATUS_SEMANTIC:
            detail = f": not found by name; semantically related wording, not verified. “{check.evidence}” ({check.note})"
        else:
            detail = ": not found" + (f". {check.note}" if check.note else "")
        lines.append(f"- {icon} **{check.name}**{detail}")
    st.markdown("\n".join(lines))


def render_method_details(row) -> None:
    """Hybrid parts' own scores and Sentence-BERT's best-matching section, when the run has them."""
    parts = [
        f"{PART_LABELS[name]} {getattr(row, f'{name}_score'):.1f}"
        for name in HybridMatcher.PART_NAMES
        if hasattr(row, f"{name}_score") and pd.notna(getattr(row, f"{name}_score"))
    ]
    if parts:
        st.caption("Each method's own similarity (0–100): " + " · ".join(parts))
    if getattr(row, "best_chunk", ""):
        st.caption(
            f"Sentence-BERT: whole-resume similarity {row.sbert_document:.1f}; best-matching part "
            f"({SECTION_NAMES.get(row.best_section, row.best_section)} section): “{row.best_chunk}”"
        )


def render_ranking_tab(run: MatchRun, threshold: float, shortlist: set[str], selected: list[str]) -> pd.DataFrame:
    """The ranked table with search, filters, result limit and pages. Returns the rows in view
    (every page), which the Insights charts reuse."""
    candidates = run.candidates
    c1, c2 = st.columns([1.4, 1], vertical_alignment="bottom")
    query = c1.text_input("Search", key="rank_search", placeholder="File name or skill",
                          icon=":material/search:")
    sort_key = c2.selectbox("Sort by", list(SORT_OPTIONS), format_func=SORT_OPTIONS.get, key="rank_sort")
    filters = st.pills("Show only", ["Shortlisted", "Strong matches"], selection_mode="multi", key="rank_filters")
    d1, d2, d3 = st.columns([1.6, 0.8, 0.8], vertical_alignment="bottom")
    must_include: list[str] = []
    if run.requirement_names:
        must_include = d1.multiselect(
            "Must mention", list(run.requirement_names), key=f"rank_must_{run.run_id}",
            placeholder="Requirements matched in the resume",
        )
    min_score = d2.number_input(
        "Min. similarity", min_value=0.0, max_value=100.0, value=0.0, step=5.0, key="rank_min_score",
        help="Optional. Hides resumes below this similarity score; they stay ranked. 0 shows all.",
    )
    min_coverage = d3.number_input(
        "Min. coverage %", min_value=0.0, max_value=100.0, value=0.0, step=10.0, key="rank_min_coverage",
        help="Optional. Hides resumes that match fewer of the job's requirements; they stay ranked.",
    )
    filtered = sort_candidates(
        filter_candidates(
            candidates, query=query, shortlist=shortlist, shortlisted_only="Shortlisted" in filters,
            strong_only="Strong matches" in filters, threshold=threshold, must_include=must_include,
            min_score=min_score, min_coverage=min_coverage,
        ),
        sort_key,
    )

    with st.container(border=True, key="card_topk"):
        heading = st.empty()  # filled once the Top K choice below is known
        top_k, _, download = st.columns([1.2, 0.6, 1.2], vertical_alignment="bottom")
        limit_label = top_k.selectbox(
            "Show Top K Resumes", list(RESULT_LIMITS), index=list(RESULT_LIMITS).index(DEFAULT_RESULT_LIMIT),
            key="top_k_results",
            help="How many of the best-ranked resumes to show. Every resume is always ranked and scored; this "
            "only changes the view (table, candidate details, charts and the results CSV).",
        )
        in_view = limit_results(filtered, RESULT_LIMITS[limit_label])
        heading.markdown(
            f"**{len(candidates)} resumes ranked · {METHOD_SHORT[run.method]}** "
            f"<span style='color:#64748B'>· {len(filtered)} match the filters · showing {len(in_view)}</span>",
            unsafe_allow_html=True,
        )
        download.download_button(
            "Download results as CSV",
            results_csv(in_view, shortlist),
            file_name=f"resume_matches_{run.method}.csv",
            mime="text/csv",
            icon=":material/download:",
            width="stretch",
            help=f"All {len(in_view)} result(s) in view (every page), with both scores and the evidence. "
            "Contains candidate data; store it carefully.",
        )
    if in_view.empty:
        st.info("No candidates match the current search and filters.", icon=":material/filter_alt_off:")
        return in_view

    pages = max(1, -(-len(in_view) // PAGE_SIZE))
    page = 1
    if pages > 1:
        p1, p2 = st.columns([1, 3], vertical_alignment="center")
        page = p1.number_input(
            f"Page (of {pages})", min_value=1, max_value=pages, value=1, step=1, key=f"rank_page_{pages}"
        )
    shown, pages = paginate(in_view, page, PAGE_SIZE)
    first = (min(max(1, int(page)), pages) - 1) * PAGE_SIZE + 1
    if pages > 1:
        p2.caption(f"Showing results {first}–{first + len(shown) - 1} of {len(in_view)} ({PAGE_SIZE} per page).")

    table = pd.DataFrame(
        {
            "rank": shown["rank"],
            "shortlisted": shown["resume_id"].isin(shortlist),
            "resume_id": shown["resume_id"],
            "score": shown["score"],
            "coverage": shown["coverage"],
            "requirements": [
                f"{m} of {t}" if t else "—" for m, t in zip(shown["matched_count"], shown["requirement_count"])
            ],
            "matched": shown["matched"].map(list),
            "related": shown["related"].map(list),
            "unconfirmed": [[n for n in u if n not in r] for u, r in zip(shown["unconfirmed"], shown["related"])],
            "method": METHOD_SHORT[run.method],
        }
    )
    method_columns = {}
    for name in HybridMatcher.PART_NAMES:
        if f"{name}_score" in shown:
            table[f"{name}_score"] = shown[f"{name}_score"]
            method_columns[f"{name}_score"] = st.column_config.NumberColumn(
                f"{PART_LABELS[name]} (0–100)", format="%.1f", width="small",
                help=f"{PART_LABELS[name]}'s own similarity before the Hybrid puts the methods on one scale.",
            )
    if "sbert_document" in shown:
        table["sbert_document"] = shown["sbert_document"]
        table["best_section"] = shown["best_section"].map(lambda s: SECTION_NAMES.get(s, s))
        method_columns["sbert_document"] = st.column_config.NumberColumn(
            "SBERT whole resume", format="%.1f", width="small",
            help="Sentence-BERT similarity of the whole resume alone (the model reads its first 512 word-pieces).",
        )
        method_columns["best_section"] = st.column_config.TextColumn(
            "Best-matching section", width="small",
            help="The resume section whose chunk is closest to the job description.",
        )
    st.dataframe(
        table,
        hide_index=True,
        width="stretch",
        column_config=method_columns | {
            "rank": st.column_config.NumberColumn("Rank", width="small"),
            "shortlisted": st.column_config.CheckboxColumn("Shortlist", width="small"),
            "resume_id": st.column_config.TextColumn("Candidate", width="medium"),
            "score": st.column_config.ProgressColumn(
                similarity_label(run.method), min_value=0, max_value=100, format="%.2f",
                help="Document similarity: cosine similarity × 100 between the whole resume and the job "
                "description. Not a hiring probability.",
            ),
            "coverage": st.column_config.ProgressColumn(
                "Requirement coverage %", min_value=0, max_value=100, format="%.0f%%",
                help="Share of the job's requirements matched with evidence in the resume. A separate "
                "measure; empty when the job lists no checkable requirements.",
            ),
            "requirements": st.column_config.TextColumn("Matched", width="small"),
            "matched": st.column_config.ListColumn("Matched requirements", width="medium"),
            "related": st.column_config.ListColumn(
                "Related (unverified)", width="small",
                help="Not found, but the resume names a related technology (e.g. Flask for Django). "
                "Not counted as matched.",
            ),
            "unconfirmed": st.column_config.ListColumn(
                "Not found in resume", width="medium",
                help="Requirements not detected in the resume text. The candidate may still have them.",
            ),
            "method": st.column_config.TextColumn("Method", width="small"),
        },
    )
    st.caption(
        "Similarity compares whole documents; coverage checks the job's requirements one by one. "
        "They are separate measures, and neither is a hiring probability."
        + (" The Hybrid score is relative to the uploaded resumes (each method rescaled to 0–1 within this "
           "run); the methods' own similarities are shown next to it." if run.method == METHOD_HYBRID else "")
    )

    section_title(
        "Candidate details",
        f"{len(run.requirement_names)} requirement(s) checked. A requirement is matched when the resume names "
        "it or a recognised alias (e.g. \"k8s\" for Kubernetes); a related technology is shown but never "
        "counted. This explains the ranking; it does not change the scores.",
    )
    sel_key = selection_key(run)
    checks_by_resume = semantic_checks(run, shown)
    for row in shown.itertuples(index=False):
        label = f"#{row.rank} · {row.resume_id} · {row.score:.2f}"
        if row.requirement_count:
            label += f" · {row.matched_count} of {row.requirement_count} requirements"
        with st.expander(label, expanded=row.rank == 1, icon=":material/star:" if row.resume_id in shortlist else None):
            a, b, _ = st.columns([1.3, 1.7, 0.6], vertical_alignment="center")
            with a:
                shortlist_toggle(run, row.resume_id, "rank")
            in_comparison = row.resume_id in selected
            b.button(
                "Remove from comparison" if in_comparison else "Add to comparison",
                key=f"cmp_{run.run_id}_{row.resume_id}", on_click=_add_to_comparison, args=(sel_key, row.resume_id),
                disabled=not in_comparison and (len(selected) >= MAX_COMPARE or len(candidates) < 2),
                icon=":material/remove:" if in_comparison else ":material/add:",
            )
            render_requirement_comparison(RequirementComparison(row.matched, row.unconfirmed))
            render_method_details(row)
            render_checks(row, checks_by_resume[row.resume_id])
            e1, e2, e3 = st.columns(3)
            with e1:
                field_label("Experience (as stated)")
                evidence(row.experience)
            with e2:
                field_label("Education")
                evidence(row.education)
            with e3:
                field_label("Shared keywords")
                st.write(", ".join(row.keywords) if row.keywords else NOT_AVAILABLE)
            st.caption("Resume preview (contact details hidden)")
            st.write(make_preview(row.text, 1200))
    return in_view


MAX_CHART_ROWS = 50


def render_insights_tab(run: MatchRun, threshold: float, in_view: pd.DataFrame) -> None:
    """Charts for the current run. Per-candidate charts follow the Ranking tab's filters and result
    limit (at most MAX_CHART_ROWS bars); the distribution and coverage charts use every ranked resume."""
    candidates = run.candidates
    per_candidate = in_view.head(MAX_CHART_ROWS) if len(in_view) else candidates.head(MAX_CHART_ROWS)
    note = f"{len(per_candidate)} candidate(s) in view" + (
        f" (first {MAX_CHART_ROWS} shown)" if len(in_view) > MAX_CHART_ROWS else ""
    )
    c1, c2 = st.columns([1.4, 1], gap="medium")
    with c1, st.container(border=True, key="card_chart_scores"):
        section_title("Similarity score by candidate", f"{note} · dashed line: strong-match threshold ({threshold:g})")
        st.altair_chart(score_chart(per_candidate, threshold), width="stretch")
    with c2, st.container(border=True, key="card_chart_dist"):
        section_title("Score distribution", f"All {len(candidates)} ranked candidates per 10-point range")
        st.altair_chart(distribution_chart(candidates), width="stretch")

    render_method_comparison(run)
    if not run.requirement_names:
        st.info(
            "No specific requirements were detected in the job description, so there are no "
            "requirement charts. Listing skills (for example after \"Required:\") enables them.",
            icon=":material/info:",
        )
        return
    with st.container(border=True, key="card_chart_scatter"):
        section_title(
            "Similarity vs requirement coverage",
            f"All {len(candidates)} ranked candidates. The two measures are separate: a resume can be close "
            "to the job description overall while matching few listed requirements, or the reverse.",
        )
        st.altair_chart(similarity_coverage_chart(candidates, run.method), width="stretch")
    c3, c4 = st.columns(2, gap="medium")
    with c3, st.container(border=True, key="card_chart_reqs"):
        section_title("Requirements found per resume", f"{note} · found vs not found in the text")
        st.altair_chart(requirements_chart(per_candidate), width="stretch")
    with c4, st.container(border=True, key="card_chart_coverage"):
        section_title("Requirement coverage", f"How many of the {len(candidates)} ranked resumes match each requirement")
        st.altair_chart(coverage_chart(candidates, run.requirement_names), width="stretch")


def _rank_all_methods(run: MatchRun) -> None:
    """Callback: rank this run's resumes and job description with every available method."""
    settings = get_settings()
    files = st.session_state.get("_run_files", {}).get(run.files_key)
    results, problems = {}, {}
    for method in (METHOD_TFIDF, METHOD_GLOVE, METHOD_SBERT, METHOD_HYBRID):
        ok, status = method_status(method, settings)
        if not ok:
            problems[method] = status
            continue
        try:
            matcher = get_matcher(method, files, run.files_key, settings)
            if isinstance(matcher, HybridMatcher) and run.hybrid_weights:
                matcher.weights = run.hybrid_weights
            results[method] = matcher.match(run.job_description, top_k=run.resumes_processed)
        except (MethodUnavailableError, EmptyQueryError, MatcherError, DatasetError) as exc:
            problems[method] = str(exc)
    st.session_state[f"_method_ranks_{run.run_id}"] = (results, problems)


def render_method_comparison(run: MatchRun) -> None:
    """How the same resumes rank under TF-IDF, GloVe, Sentence-BERT and Hybrid."""
    with st.container(border=True, key="card_method_compare"):
        section_title(
            "Ranking by method",
            "How the same resumes rank under each method for this job description. Scores of different "
            "methods are on different scales, so compare ranks, not scores.",
        )
        stored = st.session_state.get(f"_method_ranks_{run.run_id}")
        if stored is None:
            st.button(
                "Rank with every available method", icon=":material/compare:", key=f"rank_all_{run.run_id}",
                on_click=_rank_all_methods, args=(run,),
                help="Loads the GloVe and Sentence-BERT models if needed; the first time can take a while.",
            )
            return
        results, problems = stored
        for method, reason in problems.items():
            st.caption(f":material/block: {METHOD_SHORT[method]} not included: {reason}")
        if not results:
            return
        methods = list(results)
        table = combine_method_rankings(results)
        order_by = f"rank_{run.method}" if run.method in results else f"rank_{methods[0]}"
        table = table.sort_values(order_by, kind="stable")
        config = {"resume_id": st.column_config.TextColumn("Candidate", width="medium")}
        for m in methods:
            config[f"rank_{m}"] = st.column_config.NumberColumn(
                METHOD_SHORT[m], width="small", help=f"Rank under {METHOD_LABELS[m]} (1 = closest)."
            )
        config["rank_spread"] = st.column_config.NumberColumn(
            "Rank spread", width="small", help="Largest minus smallest rank across the methods."
        )
        shown = table[["resume_id", *[f"rank_{m}" for m in methods], "rank_spread"]]
        st.dataframe(shown, hide_index=True, width="stretch", column_config=config, height=min(420, 36 * (len(shown) + 1)))
        if len(methods) > 1:
            st.caption("Agreement between methods (Kendall's tau: 1 = same order, 0 = unrelated).")
            agreement = ranking_agreement(table.reset_index(drop=True), methods)
            st.dataframe(agreement, hide_index=True, width="stretch", column_config={
                "methods": "Methods", "kendall_tau": st.column_config.NumberColumn("Kendall's tau", format="%.3f"),
                "top_10_overlap": "Top-10 overlap",
            })


def _matrix_style(value: str) -> str:
    if value == MATRIX_FOUND:
        return "background-color: #E3EEFB; color: #0F172A; font-weight: 600"
    if value in (MATRIX_RELATED, MATRIX_SEMANTIC):
        return "background-color: #EEEAFB; color: #3B2A7A"
    return "background-color: #FDEDE6; color: #7C2D12"


def render_compare_tab(
    run: MatchRun, selected_ids: list[str], shortlist: set[str], originals: dict[str, tuple[str, bytes]], threshold: float
) -> None:
    status = validate_selection(selected_ids, available=len(run.candidates))
    if not status.ok:
        st.info(
            f"{status.message} Choose {2}–{MAX_COMPARE} candidates in the selector above, then click "
            "**Compare Candidates**.",
            icon=":material/compare_arrows:",
        )
        return
    try:
        selected = select_candidates(run.candidates, selected_ids)
    except ValueError as exc:  # e.g. IDs from an older run
        st.warning(str(exc))
        return
    enriched = semantic_checks(run, selected)  # adds semantic evidence for Sentence-BERT runs
    selected = selected.assign(checks=selected["resume_id"].map(enriched))

    section_title(
        f"Comparing {len(selected)} candidates",
        f"Same run, same method ({METHOD_LABELS[run.method]}) and same job description, so the scores "
        "are on one scale.",
    )

    with st.container(border=True, key="card_summary"):
        field_label("Summary")
        st.markdown("\n".join(f"- {line}" for line in comparison_summary(selected)))
        st.caption(
            "Based only on similarity scores and on which detected requirements appear in each resume's "
            "text. It does not judge candidates or recommend a hire; review the resumes yourself."
        )

    columns = st.columns(len(selected), gap="small")
    for column, row in zip(columns, selected.itertuples(index=False)):
        with column, st.container(border=True, key=f"card_cmp_{row.rank}"):
            st.markdown(
                f'<div class="cand-head"><span class="cand-rank">#{row.rank}</span>'
                f'<span class="cand-name">{row.resume_id}</span></div>',
                unsafe_allow_html=True,
            )
            st.progress(min(max(row.score / 100, 0.0), 1.0), text=f"Similarity **{row.score:.2f}** / 100")
            if row.coverage is not None:
                st.progress(row.coverage / 100, text=f"Requirement coverage **{row.coverage:.0f}%**")
            strong = row.score >= threshold
            st.caption(
                f"{METHOD_SHORT[row.method]} · {'Strong match' if strong else 'Below threshold'} (≥ {threshold:g})"
            )
            shortlist_toggle(run, row.resume_id, "compare")
            field_label(
                f"Skills matched ({row.matched_count} of {row.requirement_count})" if row.requirement_count
                else "Skills matched"
            )
            if row.requirement_count:
                badges(row.matched, "blue", "None detected")
                if row.related:
                    field_label("Related, unverified")
                    badges(row.related, "violet", "")
                semantic = [c.name for c in row.checks if c.status == STATUS_SEMANTIC]
                if semantic:
                    field_label("Semantically related, unverified")
                    badges(semantic, "blue", "")
                field_label("Not found in resume")
                badges([n for n in row.unconfirmed if n not in row.related and n not in semantic], "orange",
                       "None, all requirements found")
                with st.expander("Evidence", icon=":material/format_quote:"):
                    render_method_details(row)
                    render_checks(row)
            else:
                st.markdown(f'<span class="na">{NOT_AVAILABLE} (no requirements detected)</span>', unsafe_allow_html=True)
            field_label("Experience (as stated)")
            evidence(row.experience)
            field_label("Education")
            evidence(row.education)
            field_label("Relevant keywords")
            if row.keywords:
                badges(row.keywords, "gray", "")
            else:
                st.markdown(f'<span class="na">{NOT_AVAILABLE}</span>', unsafe_allow_html=True)
            with st.expander("Resume text", icon=":material/article:"):
                st.caption("Parsed text, contact details hidden")
                with st.container(height=260, border=False):
                    st.text(masked_text(row.text))
            source = originals.get(row.resume_id)
            if source is not None:
                st.download_button(
                    "Original file", source[1], file_name=source[0], key=f"orig_{run.run_id}_{row.resume_id}",
                    icon=":material/attach_file:", width="stretch",
                )
            else:
                st.caption(f"Original file: {NOT_AVAILABLE}")

    c1, c2 = st.columns([1.3, 1], gap="medium")
    with c1, st.container(border=True, key="card_cmp_scores", height="stretch"):
        section_title("Similarity scores", "Same 0–100 scale for every selected candidate")
        st.altair_chart(score_chart(selected, threshold), width="stretch")
    with c2, st.container(border=True, key="card_cmp_shared", height="stretch"):
        section_title("Shared skills", "Detected job requirements across the selected resumes")
        if run.requirement_names:
            shared = shared_skills(selected)
            for title, names, color in (
                ("Found in every selected resume", shared.found_in_all, "blue"),
                ("Found in some", shared.found_in_some, "violet"),
                ("Not found in any", shared.found_in_none, "orange"),
            ):
                field_label(title)
                badges(names, color, "None")
        else:
            st.markdown(f'<span class="na">{NOT_AVAILABLE}: no requirements were detected.</span>',
                        unsafe_allow_html=True)

    with st.container(border=True, key="card_cmp_matrix"):
        section_title(
            "Candidate × requirement matrix",
            "Found: the resume names it (or an alias). Related (unverified): only a related technology is "
            "named. Semantically related (unverified): similar wording found by Sentence-BERT. Not found: not "
            "detected in the text.",
        )
        matrix = status_matrix(selected)
        if matrix.empty:
            st.markdown(f'<span class="na">{NOT_AVAILABLE}: no requirements were detected in the job description.</span>',
                        unsafe_allow_html=True)
        else:
            shown = matrix.copy()
            shown.columns = [f"#{r} {c}" for r, c in zip(selected["rank"], shown.columns)]
            st.dataframe(shown.style.map(_matrix_style), width="stretch", height=35 * (len(shown) + 1) + 2)

    with st.container(border=True, key="card_export"):
        section_title("Export comparison")
        e1, e2, _ = st.columns([1, 1, 1.2])
        e1.download_button(
            "Download Comparison CSV", comparison_csv(selected, shortlist), file_name=f"candidate_comparison_{run.method}.csv",
            mime="text/csv", icon=":material/download:", width="stretch", key="dl_comparison",
        )
        if run.requirement_names:
            e2.download_button(
                "Download Skill Matrix CSV", skill_matrix_csv(selected), file_name=f"skill_matrix_{run.method}.csv",
                mime="text/csv", icon=":material/grid_on:", width="stretch", key="dl_matrix",
            )
        st.caption(
            ":material/shield: Exports contain candidate file names, skills, short evidence quotes and the "
            "experience and education phrases from the resumes (no full resume text or contact details). Treat "
            "them as confidential candidate data."
        )


def render_dashboard(run: MatchRun, inputs: Inputs) -> None:
    if run.inputs_key != inputs_key(inputs.method, inputs.files, inputs.job_description, inputs.hybrid_weights):
        st.info(
            "The job description, algorithm or Hybrid weights changed after this ranking. Everything "
            "below still belongs to the previous run; click **Match Resumes** to update it.",
            icon=":material/history:",
        )

    if run.method == METHOD_HYBRID:
        used = ", ".join(PART_LABELS[m] for m in run.methods_used)
        weights = ", ".join(f"{PART_LABELS[m]} {w:.2f}" for m, w in zip(HybridMatcher.PART_NAMES, run.hybrid_weights or ()))
        if run.methods_skipped:
            reasons = "; ".join(f"{PART_LABELS[m]}: {reason}" for m, reason in run.methods_skipped)
            st.warning(
                f"This Hybrid ranking uses only {used}. Left out because they could not read the job "
                f"description: {reasons} The remaining weights were rescaled.",
                icon=":material/warning:",
            )
        else:
            st.caption(f":material/merge: Hybrid of {used} · weights {weights} · each rescaled to 0–1 within this run.")

    head, control = st.columns([3, 1], vertical_alignment="center")
    with head:
        section_title(
            "Results",
            f"{METHOD_LABELS[run.method]} · {len(run.candidates)} of {run.resumes_processed} resumes ranked",
        )
    with control:
        threshold = render_threshold_control(run)
    render_kpis(run, threshold)

    shortlist = shortlist_for(run)
    selected = render_selection_bar(run)
    originals = original_files(inputs.files, inputs.sources, inputs.corpus) if inputs.corpus is not None else {}

    ranking, insights, compare = st.tabs(TABS, key="view", on_change="rerun")
    with ranking:
        in_view = render_ranking_tab(run, threshold, shortlist, selected)
    with insights:
        render_insights_tab(run, threshold, in_view)
    with compare:
        render_compare_tab(run, selected, shortlist, originals, threshold)


def main() -> None:
    st.set_page_config(page_title="Resume Matcher", page_icon=":material/manage_search:", layout="wide")
    st.markdown(f"<style>{CSS_PATH.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)
    settings = get_settings()

    render_header(st.session_state.get("method", METHOD_HYBRID))
    render_sidebar(settings)
    st.write("")
    inputs = render_inputs(settings)

    st.session_state["_run_files"] = {inputs.files_key: inputs.files}  # for "Rank with every method"
    if inputs.clicked:
        run_matching(inputs, settings)

    run: MatchRun | None = st.session_state.get("run")
    if run is not None and (inputs.corpus is None or run.files_key != inputs.files_key):
        st.session_state.pop("run", None)  # the resumes changed, so the old results no longer apply
        run = None
    if run is None:
        render_empty_state()
        return
    render_dashboard(run, inputs)


if __name__ == "__main__":
    main()
