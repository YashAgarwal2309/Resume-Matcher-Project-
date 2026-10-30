"""Central configuration for the Resume Matcher project.

Every setting has a sensible default and can be overridden with an
environment variable, so nothing here is specific to one computer.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# The project root is the folder above this package.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
CACHE_DIR = PROJECT_ROOT / ".cache"
SAMPLE_JOBS_PATH = DATA_DIR / "sample_job_descriptions.json"

# Resume datasets the app offers (label -> file), in order of preference. See data/README.md.
DATASET_OPTIONS = {
    "Tech roles (UpdatedResumeDataSet.csv)": DATA_DIR / "UpdatedResumeDataSet.csv",
    "All industries (Resume.csv)": DATA_DIR / "Resume.csv",
}
# The original explore_data.py read "Resume.csv" from the project root, so that
# location is still supported as a fallback.
LEGACY_DATASET_PATH = PROJECT_ROOT / "Resume.csv"
DEFAULT_DATASET_CANDIDATES = (*DATASET_OPTIONS.values(), LEGACY_DATASET_PATH)

# Places we look for pretrained GloVe vectors when GLOVE_PATH is not set.
DEFAULT_GLOVE_CANDIDATES = (
    MODELS_DIR / "glove-wiki-gigaword-100.gz",
    MODELS_DIR / "glove.6B.100d.txt",
)

# Matching method identifiers used across the project.
METHOD_TFIDF = "tfidf"
METHOD_GLOVE = "glove"
METHOD_SBERT = "sbert"
METHOD_HYBRID = "hybrid"
METHOD_LABELS = {
    METHOD_TFIDF: "TF-IDF + Cosine Similarity",
    METHOD_GLOVE: "GloVe Average Word Embeddings",
    METHOD_SBERT: "Sentence-BERT Embeddings",
    METHOD_HYBRID: "Hybrid: TF-IDF + GloVe + Sentence-BERT",
}
# Short names for compact places (badges, chart labels, summary sentences).
METHOD_SHORT = {METHOD_TFIDF: "TF-IDF", METHOD_GLOVE: "GloVe", METHOD_SBERT: "Sentence-BERT", METHOD_HYBRID: "Hybrid"}

# Default Sentence-BERT model. bge-small-en-v1.5 is trained for search (short query vs. long document),
# reads up to 512 word-pieces and is about the same size as all-MiniLM-L6-v2, which it replaced (README §13).
DEFAULT_SBERT_MODEL = "BAAI/bge-small-en-v1.5"
# BGE models expect this instruction in front of the query (here: the job description), not the resumes.
SBERT_QUERY_PREFIXES = {
    "BAAI/bge-small-en-v1.5": "Represent this sentence for searching relevant passages: ",
    "BAAI/bge-base-en-v1.5": "Represent this sentence for searching relevant passages: ",
}

# GloVe document vectors: "mean" (plain average, the default) or "idf" (damped counts x IDF). IDF
# weighting scored lower on both evaluation datasets (README §13), so it is opt-in: GLOVE_WEIGHTING=idf.
GLOVE_WEIGHTING = "mean"

# Sentence-BERT long-resume handling (README §13): chunks of at most this many word-pieces, and the
# weight of the whole-resume embedding in the final score (the rest is the best-matching chunk).
# Chunking makes the one-off encoding of an upload a few times slower; SBERT_CHUNK_TOKENS=0 turns it
# off (whole-resume embedding only, the earlier behaviour).
SBERT_CHUNK_TOKENS = 256
SBERT_DOC_WEIGHT = 0.5

# Hybrid: weights of TF-IDF, GloVe and Sentence-BERT (rescaled to sum to 1) and how their similarities
# are put on one scale before combining. Equal weights after per-query min-max normalization were the
# most consistent choice across the evaluation sets; weights tuned on one dataset did not carry over
# to the others (README §13). Override with HYBRID_WEIGHTS="0.4,0.2,0.4" and HYBRID_NORMALIZATION=none.
HYBRID_WEIGHTS = (1 / 3, 1 / 3, 1 / 3)
HYBRID_NORMALIZATION = "minmax"

# Requirement-level semantic evidence: a resume sentence counts as semantically related to a job
# capability requirement when its Sentence-BERT cosine reaches this value. Calibrated by hand on real
# demo-resume sentences for this model only (tests/test_real_models.py): at 0.72 none of the 55
# unrelated pairs reviewed passed (the closest scored 0.703) and 5 of 11 related ones did (lowest
# 0.725), so it shows little but rarely misleads; the margin is thin. For other models the check is
# switched off rather than guessed.
SEMANTIC_EVIDENCE_THRESHOLDS = {"BAAI/bge-small-en-v1.5": 0.72}

DEFAULT_TOP_K = 5
MAX_TOP_K = 50
PREVIEW_CHARS = 300
MAX_RESUME_BYTES = 5 * 1024 * 1024  # per uploaded resume file; real resumes are well under this


def _int_from_env(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


def _weights_from_env(name: str) -> tuple[float, float, float]:
    """Three comma-separated weights from an environment variable; the default when unset or invalid."""
    value = os.environ.get(name)
    if not value:
        return HYBRID_WEIGHTS
    try:
        weights = tuple(float(part) for part in value.split(","))
    except ValueError:
        return HYBRID_WEIGHTS
    if len(weights) != 3 or any(w < 0 for w in weights) or sum(weights) <= 0:
        return HYBRID_WEIGHTS
    return weights


def _first_existing(paths: tuple[Path, ...]) -> Path:
    """Return the first path that exists, or the first candidate if none do."""
    for path in paths:
        if path.exists():
            return path
    return paths[0]


def _path_from_env(name: str, candidates: tuple[Path, ...]) -> Path:
    value = os.environ.get(name)
    if value:
        return Path(value).expanduser()
    return _first_existing(candidates)


@dataclass(frozen=True)
class Settings:
    """Read-only bundle of all runtime settings."""

    dataset_path: Path = field(
        default_factory=lambda: _path_from_env("RESUME_DATA_PATH", DEFAULT_DATASET_CANDIDATES)
    )
    glove_path: Path = field(
        default_factory=lambda: _path_from_env("GLOVE_PATH", DEFAULT_GLOVE_CANDIDATES)
    )
    sbert_model_name: str = field(
        default_factory=lambda: os.environ.get("SBERT_MODEL", DEFAULT_SBERT_MODEL)
    )
    # None lets sentence-transformers pick (CUDA / Apple MPS / CPU). Set to "cpu" to force CPU.
    sbert_device: str | None = field(default_factory=lambda: os.environ.get("SBERT_DEVICE") or None)
    sbert_batch_size: int = 32
    cache_dir: Path = CACHE_DIR
    sbert_chunk_tokens: int = field(default_factory=lambda: _int_from_env("SBERT_CHUNK_TOKENS", SBERT_CHUNK_TOKENS))
    sbert_doc_weight: float = SBERT_DOC_WEIGHT
    glove_weighting: str = field(default_factory=lambda: os.environ.get("GLOVE_WEIGHTING", GLOVE_WEIGHTING).strip().lower())
    hybrid_weights: tuple[float, float, float] = field(default_factory=lambda: _weights_from_env("HYBRID_WEIGHTS"))
    hybrid_normalization: str = field(
        default_factory=lambda: os.environ.get("HYBRID_NORMALIZATION", HYBRID_NORMALIZATION).strip().lower()
    )


def get_settings() -> Settings:
    """Build a fresh Settings object (re-reads environment variables)."""
    return Settings()


def available_datasets() -> dict[str, Path]:
    """Datasets the user can choose from, as {label: path}.

    If RESUME_DATA_PATH is set, only that file is offered. Otherwise every
    known dataset file that exists is offered. May return an empty dict.
    """
    custom = os.environ.get("RESUME_DATA_PATH")
    if custom:
        path = Path(custom).expanduser()
        return {f"Custom ({path.name})": path}
    found = {label: path for label, path in DATASET_OPTIONS.items() if path.is_file()}
    if not found and LEGACY_DATASET_PATH.is_file():
        found["All industries (Resume.csv)"] = LEGACY_DATASET_PATH
    return found
