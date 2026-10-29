"""Streamlit web interface for the Resume Matcher.

Run with:  streamlit run app.py
The user uploads resumes (PDF, TXT or MD) and types one job description; nothing is preloaded.
All matching logic lives in matcher.py; this file only handles the UI.
"""

from __future__ import annotations

import logging

import pandas as pd
import streamlit as st

from resume_matcher.config import (
    DEFAULT_TOP_K,
    MAX_RESUME_BYTES,
    MAX_TOP_K,
    METHOD_GLOVE,
    METHOD_HYBRID,
    METHOD_LABELS,
    METHOD_SBERT,
    METHOD_TFIDF,
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
from resume_matcher.preprocessing import make_preview
from resume_matcher.skills import (
    RequirementComparison,
    compare_requirements,
    extract_requirements,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
for noisy in ("httpx", "httpx2", "urllib3", "huggingface_hub"):
    logging.getLogger(noisy).setLevel(logging.WARNING)

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

# Short names and one-line descriptions shown under the method picker and in the result summary.
METHOD_SHORT = {METHOD_TFIDF: "TF-IDF", METHOD_GLOVE: "GloVe", METHOD_SBERT: "Sentence-BERT", METHOD_HYBRID: "Hybrid"}
METHOD_TAGLINE = {
    METHOD_TFIDF: "Matches the exact words of the job description. Fastest.",
    METHOD_GLOVE: "Also counts related words (e.g. doctor / physician).",
    METHOD_SBERT: "Compares the overall meaning of the texts.",
    METHOD_HYBRID: "Recommended: averages TF-IDF, GloVe and Sentence-BERT.",
}

# Extra styling on top of the theme in .streamlit/config.toml. Containers created with
# st.container(key="...") get the CSS class "st-key-<key>", which is what these rules target.
APP_CSS = """
<style>
.block-container {padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1240px;}
header[data-testid="stHeader"] {background: transparent;}

/* Header banner */
.st-key-hero {
    background: linear-gradient(120deg, #0A1226 0%, #13306B 60%, #1D4ED8 100%);
    border-radius: 18px; padding: 1.9rem 2.2rem 1.6rem; box-shadow: 0 12px 32px rgba(15, 23, 42, 0.18);
}
.st-key-hero h1 {color: #FFFFFF; font-weight: 800; letter-spacing: -0.02em; padding: 0 0 0.2rem 0;}
.st-key-hero p {color: #C9DBF5; font-size: 1.02rem; margin-bottom: 0.4rem;}
.eyebrow {color: #22D3EE !important; font-size: 0.78rem !important; font-weight: 700;
          letter-spacing: 0.14em; text-transform: uppercase; margin: 0 !important;}
.chip {display: inline-block; margin: 0.35rem 0.4rem 0 0; padding: 0.22rem 0.75rem; border-radius: 999px;
       font-size: 0.8rem; font-weight: 600; color: #E2E8F0; background: rgba(255, 255, 255, 0.08);
       border: 1px solid rgba(255, 255, 255, 0.22);}

/* White cards (all containers whose key starts with "card") */
[class*="st-key-card"] {background: #FFFFFF; box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04), 0 8px 24px rgba(15, 23, 42, 0.05);}
.step {display: flex; align-items: center; gap: 0.75rem; margin-bottom: 0.3rem;}
.step .num {flex: none; width: 2rem; height: 2rem; border-radius: 50%; background: #2563EB; color: #FFFFFF;
            display: flex; align-items: center; justify-content: center; font-weight: 700; font-size: 0.95rem;}
.step .title {font-weight: 700; font-size: 1.05rem; color: #0F172A; line-height: 1.2;}
.step .sub {font-size: 0.82rem; color: #64748B;}

/* Result summary tiles */
[data-testid="stMetric"] {background: #FFFFFF; box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);}
[data-testid="stMetricValue"] {font-size: 1.45rem; font-weight: 700;}

/* Empty state before the first match */
.empty {text-align: center; padding: 2.2rem 1rem; color: #64748B; border: 1.5px dashed #CBD5E1;
        border-radius: 14px; background: #FFFFFF;}
.empty b {color: #0F172A;}

/* Sidebar */
.side-title {font-size: 0.75rem; font-weight: 700; letter-spacing: 0.12em; text-transform: uppercase;
             color: #22D3EE; margin: 0.6rem 0 0.4rem;}
.method-row {display: flex; gap: 0.6rem; align-items: flex-start; margin-bottom: 0.7rem;}
.method-row .dot {flex: none; width: 0.6rem; height: 0.6rem; border-radius: 50%; margin-top: 0.42rem;}
.method-row .ok {background: #34D399; box-shadow: 0 0 0 3px rgba(52, 211, 153, 0.18);}
.method-row .off {background: #64748B;}
.method-row b {color: #FFFFFF; font-size: 0.9rem;}
.method-row small {color: #8FA3C2; font-size: 0.78rem;}
</style>
"""


def step_header(number: int, title: str, subtitle: str) -> None:
    """Numbered heading at the top of an input card."""
    st.markdown(
        f'<div class="step"><span class="num">{number}</span>'
        f'<div><div class="title">{title}</div><div class="sub">{subtitle}</div></div></div>',
        unsafe_allow_html=True,
    )

# A set of uploaded resumes, as ((file name, text), ...). Tuples so Streamlit can use them as cache keys.
UploadedFiles = tuple[tuple[str, str], ...]


@st.cache_data(show_spinner=False, max_entries=500)
def read_file(name: str, data: bytes) -> str:
    """Extract the text of one uploaded file once (PDF parsing is the slow part)."""
    return read_resume_file(name, data)


@st.cache_resource(show_spinner=False)
def get_glove_vectors(path: str):
    """Load the GloVe table once; it is shared by every set of uploaded resumes."""
    return load_glove_vectors(path)


@st.cache_resource(show_spinner=False)
def get_sbert_model(model_name: str, device: str | None):
    """Load the Sentence-BERT model once."""
    return load_sbert_model(model_name, device)


@st.cache_resource(show_spinner=False, max_entries=8)
def get_corpus(files: UploadedFiles) -> ResumeCorpus:
    return corpus_from_files(list(files))


@st.cache_resource(show_spinner=False, max_entries=8)
def get_matcher(method: str, files: UploadedFiles) -> BaseMatcher:
    """Build each matcher once per set of uploaded resumes; resume vectors are reused across queries."""
    settings = get_settings()
    word_vectors = sbert_model = None
    if method in (METHOD_GLOVE, METHOD_HYBRID):
        word_vectors = get_glove_vectors(str(settings.glove_path))
    if method in (METHOD_SBERT, METHOD_HYBRID):
        sbert_model = get_sbert_model(settings.sbert_model_name, settings.sbert_device)
    return create_matcher(method, get_corpus(files), settings, word_vectors=word_vectors, sbert_model=sbert_model)


def read_uploaded_resumes(uploaded) -> tuple[UploadedFiles, list[str]]:
    """Return the readable resumes as (name, text) pairs, plus a message for each file that was skipped."""
    files, problems = [], []
    for item in uploaded or []:
        if item.size > MAX_RESUME_BYTES:
            problems.append(f"'{item.name}' is too large (limit {MAX_RESUME_BYTES // (1024 * 1024)} MB).")
            continue
        try:
            text = read_file(item.name, item.getvalue())
        except DatasetError as exc:
            problems.append(str(exc))
            continue
        if not text.strip():
            problems.append(f"'{item.name}' has no readable text (a scanned PDF needs OCR first).")
            continue
        files.append((item.name, text))
    return tuple(files), problems


def clear_results() -> None:
    """Callback: forget old results when the uploaded resumes change (they belong to the old set)."""
    st.session_state.pop("results", None)


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


def render_sidebar(settings) -> None:
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


def main() -> None:
    st.set_page_config(page_title="Resume Matcher", page_icon="📄", layout="wide")
    st.markdown(APP_CSS, unsafe_allow_html=True)
    settings = get_settings()

    # ---- Header banner
    with st.container(key="hero"):
        st.markdown('<p class="eyebrow">AI-assisted resume screening</p>', unsafe_allow_html=True)
        st.title("Resume Matcher", anchor=False)
        st.markdown(
            "<p>Upload resumes, describe the job, and get the resumes ranked from best to worst match.</p>"
            + "".join(f'<span class="chip">{name}</span>' for name in METHOD_SHORT.values()),
            unsafe_allow_html=True,
        )
    render_sidebar(settings)
    st.write("")

    # ---- Inputs: resumes on the left, job description on the right
    left, right = st.columns(2, gap="medium")
    with left, st.container(border=True, key="card_resumes", height="stretch"):
        step_header(1, "Upload resumes", f"{', '.join(t.upper() for t in RESUME_FILE_TYPES)} · select several files")
        uploaded = st.file_uploader(
            f"Resumes ({', '.join(t.upper() for t in RESUME_FILE_TYPES)}, select several files)",
            type=list(RESUME_FILE_TYPES),
            accept_multiple_files=True,
            key="resume_files",
            on_change=clear_results,
            label_visibility="collapsed",
        )
        files, problems = read_uploaded_resumes(uploaded)
        for message in problems:
            st.warning(message)
        corpus: ResumeCorpus | None = None
        if files:
            try:
                corpus = get_corpus(files)
            except DatasetError as exc:
                st.error(str(exc))
        if corpus is not None:
            st.caption(f":material/check_circle: **{len(corpus)}** resume(s) ready to match.")
            if corpus.stats["dropped_duplicates"]:
                st.caption(f"{corpus.stats['dropped_duplicates']} duplicate file(s) ignored.")
    with right, st.container(border=True, key="card_job", height="stretch"):
        step_header(2, "Describe the job", "Paste or type the job description")
        job_description = st.text_area(
            "Job description", key="jd_text", height=230, placeholder="Paste or type the job description here.",
            label_visibility="collapsed",
        )

    # ---- Method, Top K and the Match button
    with st.container(border=True, key="card_settings"):
        step_header(3, "Choose a method and rank", "The score shows how similar each resume is to the job (0–100)")
        c1, c2, c3 = st.columns([2, 1, 1], vertical_alignment="bottom")
        method = c1.selectbox(
            "Matching algorithm", list(METHOD_LABELS), format_func=METHOD_LABELS.get, key="method",
            index=list(METHOD_LABELS).index(METHOD_HYBRID),
        )
        count = len(corpus) if corpus is not None else DEFAULT_TOP_K
        top_k = c2.number_input("Top K resumes", min_value=1, max_value=min(MAX_TOP_K, count),
                                value=min(MAX_TOP_K, count), step=1)
        clicked = c3.button("Match Resumes", type="primary", width="stretch", icon=":material/manage_search:")
        st.caption(f"**{METHOD_SHORT[method]}:** {METHOD_TAGLINE[method]}")

    if clicked:
        st.session_state.pop("results", None)
        if corpus is None:
            st.warning("Please upload at least one resume first.")
        elif not job_description.strip():
            st.warning("Please enter a job description first.")
        else:
            try:
                with st.spinner(f"Building {METHOD_LABELS[method]} index (first run can take a while)..."):
                    matcher = get_matcher(method, files)
                with st.spinner("Ranking resumes..."):
                    results = matcher.match(job_description, top_k=int(top_k))
                # Stored with the results, so editing the job description later cannot mismatch them.
                st.session_state["results"] = (method, results, extract_requirements(job_description))
            except MethodUnavailableError as exc:
                st.error(f"{METHOD_LABELS[method]} is not available: {exc}")
            except EmptyQueryError as exc:
                st.warning(str(exc))
            except MatcherError as exc:
                st.error(f"Matching failed: {exc}")

    # ---- Results (kept in session state so the download button does not clear them)
    if "results" not in st.session_state or corpus is None:
        st.markdown(
            '<div class="empty"><b>Your ranking will appear here.</b><br>'
            "Upload resumes, add a job description and click <b>Match Resumes</b>.</div>",
            unsafe_allow_html=True,
        )
        return

    method_used, results, requirements = st.session_state["results"]
    best = results.iloc[0]
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Resumes ranked", f"{len(results)} of {len(corpus)}", border=True)
    m2.metric("Best match", best["resume_id"], border=True)
    m3.metric("Best score", f"{best['score']:.1f} / 100", border=True)
    m4.metric("Method", METHOD_SHORT[method_used], border=True)

    with st.container(border=True, key="card_results"):
        head, download = st.columns([3, 1], vertical_alignment="center")
        head.subheader(f"Top {len(results)} resumes · {METHOD_SHORT[method_used]}", anchor=False)
        download.download_button(
            "Download results as CSV",
            results.drop(columns="category").assign(method=method_used).to_csv(index=False).encode("utf-8"),
            file_name=f"resume_matches_{method_used}.csv",
            mime="text/csv",
            icon=":material/download:",
            width="stretch",
        )
        st.dataframe(
            results.drop(columns=["cosine_similarity", "category"]),
            hide_index=True,
            width="stretch",
            column_config={
                "rank": st.column_config.NumberColumn("Rank", width="small"),
                "resume_id": st.column_config.TextColumn("Resume file", width="medium"),
                "score": st.column_config.ProgressColumn(
                    "Similarity (0–100)", min_value=0, max_value=100, format="%.2f"
                ),
                "preview": st.column_config.TextColumn("Resume preview", width="large"),
            },
        )
        st.caption("Similarity score = cosine similarity × 100. It is not a hiring probability.")

        st.markdown("**Job requirements in each resume**")
        st.caption(
            f"{len(requirements)} requirement(s) detected in the job description. A requirement counts as "
            "included when the resume mentions it or a common alias (e.g. \"k8s\" for Kubernetes). This "
            "explains the ranking; it does not change the scores."
        )
        texts = corpus.df.set_index("resume_id")["text"]
        for row in results.itertuples():
            text = texts.get(row.resume_id, "")
            comparison = compare_requirements(requirements, text)
            label = f"#{row.rank} · {row.resume_id} · {row.score:.2f}"
            if comparison.total:
                label += f" · {len(comparison.included)} of {comparison.total} requirements"
            with st.expander(label, expanded=row.rank == 1):
                render_requirement_comparison(comparison)
                st.caption("Resume preview")
                st.write(make_preview(text, 1200))


if __name__ == "__main__":
    main()
