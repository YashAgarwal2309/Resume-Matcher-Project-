"""Streamlit web interface for the Resume Matcher: a recruitment analytics dashboard.

Run with:  streamlit run app.py
The user types one job description and uploads resumes (PDF, TXT or MD); nothing is preloaded.
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
import pandas as pd
import streamlit as st

from resume_matcher.comparison import (
    DEFAULT_STRONG_THRESHOLDS,
    MAX_COMPARE,
    NOT_AVAILABLE,
    SORT_OPTIONS,
    build_candidates,
    comparison_csv,
    comparison_summary,
    filter_candidates,
    masked_text,
    score_bins,
    select_candidates,
    shared_skills,
    skill_matrix,
    skill_matrix_csv,
    sort_candidates,
    summarize_run,
    validate_selection,
)
from resume_matcher.config import (
    DEFAULT_TOP_K,
    MAX_RESUME_BYTES,
    MAX_TOP_K,
    METHOD_GLOVE,
    METHOD_HYBRID,
    METHOD_LABELS,
    METHOD_SBERT,
    METHOD_SHORT,
    METHOD_TFIDF,
    Settings,
    get_settings,
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
    MatcherError,
    MethodUnavailableError,
    create_matcher,
    load_glove_vectors,
    load_sbert_model,
    method_status,
)
from resume_matcher.preprocessing import fix_mojibake, make_preview, normalize_whitespace
from resume_matcher.skills import (
    Requirement,
    RequirementComparison,
    compare_requirements,
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
        "(\"physician\" / \"doctor\") land close together. Averaging loses word order and can blur long "
        "resumes; words missing from the GloVe vocabulary are ignored."
    ),
    METHOD_SBERT: (
        "Encodes the whole text with a Sentence-BERT transformer model (default: BAAI/bge-small-en-v1.5, "
        "trained for search) into one embedding that captures meaning, so related wording still matches. "
        "Slower than TF-IDF, and it reads only roughly the first 512 word-pieces of each resume."
    ),
    METHOD_HYBRID: (
        "Recommended. Averages the TF-IDF, GloVe and Sentence-BERT similarities (one third each), so exact "
        "skill words, related words and overall meaning all count. It is one of the best-scoring methods "
        "in our evaluation (see README)."
    ),
}

# One-line descriptions shown under the method picker and in the header.
METHOD_TAGLINE = {
    METHOD_TFIDF: "Matches the exact words of the job description. Fastest.",
    METHOD_GLOVE: "Also counts related words (e.g. doctor / physician).",
    METHOD_SBERT: "Compares the overall meaning of the texts.",
    METHOD_HYBRID: "Recommended: averages TF-IDF, GloVe and Sentence-BERT.",
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
    inputs_key: str  # method + Top K + resumes + job description, to notice later edits
    files_key: str
    shortlist_scope: str  # resumes + job description: the shortlist survives a method change
    method: str
    top_k: int
    job_description: str
    requirements: tuple[Requirement, ...]
    results: pd.DataFrame  # as returned by BaseMatcher.match (the results CSV uses it unchanged)
    candidates: pd.DataFrame  # results plus requirement checks and evidence (comparison.py)
    resumes_processed: int


# --------------------------------------------------------------------------- keys and caches


def _digest(*parts: str) -> str:
    digest = hashlib.sha256()
    for part in parts:
        digest.update(part.encode("utf-8", errors="replace") + b"\x00")
    return digest.hexdigest()[:16]


def files_key(files: UploadedFiles) -> str:
    return _digest(*(f"{name}\x01{text}" for name, text in files))


def inputs_key(method: str, top_k: int, files: UploadedFiles, job_description: str) -> str:
    return _digest(method, str(int(top_k)), files_key(files), job_description)


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
    top_k: int
    clicked: bool


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
        for message in problems:
            st.warning(message)
        corpus: ResumeCorpus | None = None
        key = files_key(files)
        if files:
            try:
                corpus = get_corpus(files, key)
            except DatasetError as exc:
                st.error(str(exc))
        if corpus is not None:
            st.caption(f":material/check_circle: **{len(corpus)}** resume(s) ready to match.")
            if corpus.stats["dropped_duplicates"]:
                st.caption(f"{corpus.stats['dropped_duplicates']} duplicate file(s) ignored.")

    with st.container(border=True, key="card_settings"):
        step_header(3, "Matching settings", "Choose the algorithm and how many candidates to rank")
        c1, c2, c3 = st.columns([2, 1, 1], vertical_alignment="bottom")
        method = c1.selectbox(
            "Matching algorithm", list(METHOD_LABELS), format_func=METHOD_LABELS.get, key="method",
            index=list(METHOD_LABELS).index(METHOD_HYBRID),
        )
        count = len(corpus) if corpus is not None else DEFAULT_TOP_K
        top_k = c2.number_input(
            "Candidates to display", min_value=1, max_value=min(MAX_TOP_K, count), value=min(MAX_TOP_K, count),
            step=1, help=f"Top K: how many of the best-scoring resumes to rank (at most {MAX_TOP_K}).",
        )
        clicked = c3.button("Match Resumes", type="primary", width="stretch", icon=":material/manage_search:")
        ok, status = method_status(method, settings)
        details = [f"**{METHOD_SHORT[method]}:** {METHOD_TAGLINE[method]}"]
        if method in (METHOD_SBERT, METHOD_HYBRID):
            details.append(f"Sentence-BERT model: `{settings.sbert_model_name}` (set `SBERT_MODEL` to change).")
        details.append(("" if ok else ":material/warning: ") + status)
        st.caption("  \n".join(details))

    return Inputs(job_description, files, sources, corpus, key, method, int(top_k), clicked)


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
            st.write("Ranking resumes...")
            results = matcher.match(inputs.job_description, top_k=inputs.top_k)
            st.write("Checking the job requirements in each resume...")
            requirements = tuple(extract_requirements(inputs.job_description))
            candidates = build_candidates(
                results, inputs.corpus, requirements, inputs.job_description, inputs.method
            )
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

    _forget_run_widgets()
    st.session_state["run"] = MatchRun(
        run_id=uuid.uuid4().hex[:12],
        inputs_key=inputs_key(inputs.method, inputs.top_k, inputs.files, inputs.job_description),
        files_key=inputs.files_key,
        shortlist_scope=_digest(inputs.files_key, inputs.job_description),
        method=inputs.method,
        top_k=inputs.top_k,
        job_description=inputs.job_description,
        requirements=requirements,
        results=results,
        candidates=candidates,
        resumes_processed=len(inputs.corpus),
    )
    st.session_state["view"] = TAB_RANKING


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
        "Resumes processed", summary.resumes_processed, border=True, icon=":material/description:",
        help="Uploaded resumes with readable text (unreadable files and exact duplicates are skipped).",
    )
    k2.metric(
        "Candidates ranked", summary.candidates_ranked, border=True, icon=":material/format_list_numbered:",
        help=f"The top {run.top_k} resumes by similarity score.",
    )
    k3.metric(
        "Strong matches", summary.strong_matches, border=True, icon=":material/verified:",
        help=f"Ranked resumes with a {METHOD_SHORT[run.method]} similarity score of at least {threshold:g}. "
        "Change the threshold with the button above.",
    )
    k4.metric(
        "Average similarity", "—" if summary.average_score is None else f"{summary.average_score:.1f}",
        border=True, icon=":material/functions:",
        help="Mean similarity score of the ranked candidates in this run (0–100). Only meaningful "
        "within one run and method; it is not a probability.",
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
        c1, c2 = st.columns([3, 1.1], vertical_alignment="bottom")
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


def render_ranking_tab(run: MatchRun, threshold: float, shortlist: set[str], selected: list[str]) -> None:
    candidates = run.candidates
    c1, c2, c3 = st.columns([1.6, 1.2, 1.4], vertical_alignment="bottom")
    query = c1.text_input("Search", key="rank_search", placeholder="File name or skill",
                          icon=":material/search:")
    sort_key = c2.selectbox("Sort by", list(SORT_OPTIONS), format_func=SORT_OPTIONS.get, key="rank_sort")
    filters = c3.pills("Show only", ["Shortlisted", "Strong matches"], selection_mode="multi", key="rank_filters")
    must_include: list[str] = []
    if run.requirements:
        must_include = st.multiselect(
            "Must mention", [r.name for r in run.requirements], key=f"rank_must_{run.run_id}",
            placeholder="Filter by job requirements found in the resume",
        )

    shown = sort_candidates(
        filter_candidates(
            candidates, query=query, shortlist=shortlist, shortlisted_only="Shortlisted" in filters,
            strong_only="Strong matches" in filters, threshold=threshold, must_include=must_include,
        ),
        sort_key,
    )

    head, download = st.columns([3, 1], vertical_alignment="center")
    head.markdown(
        f"**Top {len(candidates)} resumes · {METHOD_SHORT[run.method]}** "
        f"<span style='color:#64748B'>· showing {len(shown)} of {len(candidates)}</span>",
        unsafe_allow_html=True,
    )
    download.download_button(
        "Download results as CSV",
        run.results.drop(columns="category").assign(method=run.method).to_csv(index=False).encode("utf-8"),
        file_name=f"resume_matches_{run.method}.csv",
        mime="text/csv",
        icon=":material/download:",
        width="stretch",
    )
    if shown.empty:
        st.info("No candidates match the current search and filters.", icon=":material/filter_alt_off:")
        return

    table = pd.DataFrame(
        {
            "rank": shown["rank"],
            "shortlisted": shown["resume_id"].isin(shortlist),
            "resume_id": shown["resume_id"],
            "score": shown["score"],
            "method": METHOD_SHORT[run.method],
            "requirements": [
                f"{m} of {t}" if t else "—" for m, t in zip(shown["matched_count"], shown["requirement_count"])
            ],
            "matched": shown["matched"].map(list),
            "unconfirmed": shown["unconfirmed"].map(list),
        }
    )
    st.dataframe(
        table,
        hide_index=True,
        width="stretch",
        column_config={
            "rank": st.column_config.NumberColumn("Rank", width="small"),
            "shortlisted": st.column_config.CheckboxColumn("Shortlist", width="small"),
            "resume_id": st.column_config.TextColumn("Candidate", width="medium"),
            "score": st.column_config.ProgressColumn(
                "Similarity (0–100)", min_value=0, max_value=100, format="%.2f",
                help="Cosine similarity × 100 for the selected method. Not a hiring probability.",
            ),
            "method": st.column_config.TextColumn("Method", width="small"),
            "requirements": st.column_config.TextColumn(
                "Found", width="small", help="Detected job requirements found in the resume text."
            ),
            "matched": st.column_config.ListColumn("Matched skills", width="medium"),
            "unconfirmed": st.column_config.ListColumn(
                "Not found in resume", width="medium",
                help="Required skills not detected in the resume text. The candidate may still have them.",
            ),
        },
    )
    st.caption("Similarity score = cosine similarity × 100. It is not a hiring probability.")

    section_title(
        "Candidate details",
        f"{len(run.requirements)} requirement(s) detected in the job description. A requirement counts as "
        "included when the resume mentions it or a common alias (e.g. \"k8s\" for Kubernetes). This "
        "explains the ranking; it does not change the scores.",
    )
    texts = dict(zip(candidates["resume_id"], candidates["text"]))
    sel_key = selection_key(run)
    for row in shown.itertuples(index=False):
        text = texts.get(row.resume_id, "")
        comparison = compare_requirements(run.requirements, text)
        label = f"#{row.rank} · {row.resume_id} · {row.score:.2f}"
        if comparison.total:
            label += f" · {len(comparison.included)} of {comparison.total} requirements"
        with st.expander(label, expanded=row.rank == 1, icon=":material/star:" if row.resume_id in shortlist else None):
            a, b, _ = st.columns([1, 1.3, 2], vertical_alignment="center")
            with a:
                shortlist_toggle(run, row.resume_id, "rank")
            in_comparison = row.resume_id in selected
            b.button(
                "Remove from comparison" if in_comparison else "Add to comparison",
                key=f"cmp_{run.run_id}_{row.resume_id}", on_click=_add_to_comparison, args=(sel_key, row.resume_id),
                disabled=not in_comparison and (len(selected) >= MAX_COMPARE or len(candidates) < 2),
                icon=":material/remove:" if in_comparison else ":material/add:",
            )
            render_requirement_comparison(comparison)
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
            st.write(make_preview(text, 1200))


def render_insights_tab(run: MatchRun, threshold: float) -> None:
    candidates = run.candidates
    c1, c2 = st.columns([1.4, 1], gap="medium")
    with c1, st.container(border=True, key="card_chart_scores"):
        section_title("Similarity score by candidate", f"Dashed line: strong-match threshold ({threshold:g})")
        st.altair_chart(score_chart(candidates, threshold), width="stretch")
    with c2, st.container(border=True, key="card_chart_dist"):
        section_title("Score distribution", "Ranked candidates per 10-point range")
        st.altair_chart(distribution_chart(candidates), width="stretch")

    if not run.requirements:
        st.info(
            "No specific requirements were detected in the job description, so there are no "
            "requirement charts. Listing skills (for example after \"Required:\") enables them.",
            icon=":material/info:",
        )
        return
    c3, c4 = st.columns(2, gap="medium")
    with c3, st.container(border=True, key="card_chart_reqs"):
        section_title("Requirements found per resume", "Detected job requirements found vs not found in the text")
        st.altair_chart(requirements_chart(candidates), width="stretch")
    with c4, st.container(border=True, key="card_chart_coverage"):
        section_title("Requirement coverage", "How many ranked resumes mention each requirement")
        st.altair_chart(coverage_chart(candidates, tuple(r.name for r in run.requirements)), width="stretch")


def _matrix_style(value: str) -> str:
    if value == "Found":
        return "background-color: #E3EEFB; color: #0F172A; font-weight: 600"
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
                field_label("Not found in resume")
                badges(row.unconfirmed, "orange", "None, all requirements found")
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
        if run.requirements:
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
        section_title("Candidate × requirement matrix", "Found: the resume text mentions it. Not found: not detected.")
        matrix = skill_matrix(selected)
        if matrix.empty:
            st.markdown(f'<span class="na">{NOT_AVAILABLE}: no requirements were detected in the job description.</span>',
                        unsafe_allow_html=True)
        else:
            shown = matrix.replace({True: "Found", False: "Not found"})
            shown.columns = [f"#{r} {c}" for r, c in zip(selected["rank"], shown.columns)]
            st.dataframe(shown.style.map(_matrix_style), width="stretch", height=35 * (len(shown) + 1) + 2)

    with st.container(border=True, key="card_export"):
        section_title("Export comparison")
        e1, e2, _ = st.columns([1, 1, 1.2])
        e1.download_button(
            "Download Comparison CSV", comparison_csv(selected, shortlist), file_name=f"candidate_comparison_{run.method}.csv",
            mime="text/csv", icon=":material/download:", width="stretch", key="dl_comparison",
        )
        if run.requirements:
            e2.download_button(
                "Download Skill Matrix CSV", skill_matrix_csv(selected), file_name=f"skill_matrix_{run.method}.csv",
                mime="text/csv", icon=":material/grid_on:", width="stretch", key="dl_matrix",
            )
        st.caption(
            ":material/shield: Exports contain candidate file names, skills and the experience and education "
            "phrases quoted from the resumes (no full resume text or contact details). Treat them as "
            "confidential candidate data."
        )


def render_dashboard(run: MatchRun, inputs: Inputs) -> None:
    if run.inputs_key != inputs_key(inputs.method, inputs.top_k, inputs.files, inputs.job_description):
        st.info(
            "The job description, algorithm or number of candidates changed after this ranking. Everything "
            "below still belongs to the previous run; click **Match Resumes** to update it.",
            icon=":material/history:",
        )

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
        render_ranking_tab(run, threshold, shortlist, selected)
    with insights:
        render_insights_tab(run, threshold)
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
