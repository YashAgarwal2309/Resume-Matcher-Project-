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

DEFAULT_TOP_K = 5
MAX_TOP_K = 50
PREVIEW_CHARS = 300
MAX_RESUME_BYTES = 5 * 1024 * 1024  # per uploaded resume file; real resumes are well under this


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
