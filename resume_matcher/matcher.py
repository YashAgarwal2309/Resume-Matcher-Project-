"""Resume-to-job-description matching with TF-IDF, GloVe, Sentence-BERT and a hybrid of all three.

All matchers share one interface:

    matcher = create_matcher("tfidf", corpus, settings)
    results = matcher.match(job_description, top_k=5)

``match`` always returns a DataFrame with RESULT_COLUMNS. If a method
cannot run (missing model, package or file) it raises MethodUnavailableError
instead of falling back to a different algorithm.
"""

from __future__ import annotations

import gzip
import hashlib
import importlib.util
import logging
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from resume_matcher.config import (
    METHOD_GLOVE,
    METHOD_HYBRID,
    METHOD_LABELS,
    METHOD_SBERT,
    METHOD_TFIDF,
    PREVIEW_CHARS,
    SBERT_QUERY_PREFIXES,
    Settings,
)
from resume_matcher.data_loader import ResumeCorpus
from resume_matcher.preprocessing import STOP_WORDS, clean_text, make_preview, normalize_whitespace, tokenize

logger = logging.getLogger(__name__)

RESULT_COLUMNS = ["rank", "resume_id", "category", "score", "cosine_similarity", "preview"]


class MatcherError(Exception):
    """Base error for matching problems."""


class MethodUnavailableError(MatcherError):
    """The requested method cannot run in this environment (e.g. model missing)."""


class EmptyQueryError(MatcherError, ValueError):
    """The job description is blank or has no usable words."""


def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    """Scale each row to unit length; all-zero rows stay zero (no division by zero)."""
    matrix = np.asarray(matrix, dtype=np.float32)
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def similarity_to_score(similarity: np.ndarray) -> np.ndarray:
    """Convert cosine similarity to a 0-100 display score.

    Negative similarities (possible with embeddings) are shown as 0. This is a
    similarity score, not a probability of being hired.
    """
    return np.round(np.clip(similarity, 0.0, 1.0) * 100, 2)


class BaseMatcher:
    """Shared ranking logic. Subclasses implement ``fit`` and ``similarities``."""

    method = ""

    def __init__(self, corpus: ResumeCorpus):
        if len(corpus) == 0:
            raise MatcherError("Cannot build a matcher from an empty resume corpus.")
        self.corpus = corpus
        self.is_fitted = False

    @property
    def label(self) -> str:
        return METHOD_LABELS[self.method]

    def fit(self) -> "BaseMatcher":
        """Prepare resume vectors. Must set ``self.is_fitted = True``."""
        raise NotImplementedError

    def similarities(self, job_description: str) -> np.ndarray:
        """Return one cosine similarity per resume (same order as the corpus)."""
        raise NotImplementedError

    def match(self, job_description: str, top_k: int = 5) -> pd.DataFrame:
        """Rank resumes against ``job_description`` and return the top ``top_k``."""
        if not isinstance(job_description, str) or not job_description.strip():
            raise EmptyQueryError("Please enter a job description.")
        if not isinstance(top_k, (int, np.integer)) or top_k < 1:
            raise ValueError("top_k must be a positive integer.")
        if not self.is_fitted:
            self.fit()

        sims = np.asarray(self.similarities(job_description), dtype=np.float64).ravel()
        if sims.shape[0] != len(self.corpus):
            raise MatcherError("Internal error: similarity count does not match resume count.")

        top_k = min(int(top_k), len(sims))
        # Stable sort: ties keep dataset order, so results are reproducible.
        order = np.argsort(-sims, kind="stable")[:top_k]
        rows = self.corpus.df.iloc[order]

        return pd.DataFrame(
            {
                "rank": np.arange(1, top_k + 1),
                "resume_id": rows["resume_id"].to_numpy(),
                "category": rows["category"].to_numpy(),
                "score": similarity_to_score(sims[order]),
                "cosine_similarity": np.round(sims[order], 4),
                "preview": [make_preview(t, PREVIEW_CHARS) for t in rows["text"]],
            },
            columns=RESULT_COLUMNS,
        )


# --------------------------------------------------------------------------- TF-IDF


class TfidfMatcher(BaseMatcher):
    """Bag-of-words matching: TF-IDF vectors compared with cosine similarity."""

    method = METHOD_TFIDF

    def __init__(self, corpus: ResumeCorpus):
        super().__init__(corpus)
        # Text is already cleaned, so split on whitespace. This keeps terms like
        # "c++", "c#" and single letters such as "c" or "r" that the default
        # sklearn tokenizer would drop.
        self.vectorizer = TfidfVectorizer(
            tokenizer=str.split,
            token_pattern=None,
            lowercase=False,
            stop_words=sorted(STOP_WORDS),
            sublinear_tf=True,
        )
        self.resume_matrix = None

    def fit(self) -> "TfidfMatcher":
        try:
            self.resume_matrix = self.vectorizer.fit_transform(self.corpus.df["clean_text"])
        except ValueError as exc:  # "empty vocabulary" when every word is a stop word
            raise MatcherError(f"TF-IDF could not build a vocabulary from the resumes: {exc}") from exc
        self.is_fitted = True
        return self

    def similarities(self, job_description: str) -> np.ndarray:
        query = self.vectorizer.transform([clean_text(job_description)])
        if query.nnz == 0:
            raise EmptyQueryError(
                "None of the job description's words appear in the resume vocabulary "
                "(after removing stop words). Add more specific skills or duties."
            )
        # TfidfVectorizer rows are already L2-normalized, so the dot product is cosine similarity.
        return (self.resume_matrix @ query.T).toarray().ravel()


# --------------------------------------------------------------------------- GloVe


def load_glove_vectors(path: str | Path) -> dict[str, np.ndarray]:
    """Load GloVe vectors from a text file (``.txt`` or gzipped ``.gz``).

    Supports the original Stanford format (``word v1 v2 ...``) and the gensim-data
    word2vec format, which has an extra ``<count> <dim>`` header line.
    """
    path = Path(path)
    if not path.is_file():
        raise MethodUnavailableError(
            f"GloVe vectors not found at '{path}'. See README > 'Optional: GloVe' "
            "for the download command, or set GLOVE_PATH."
        )
    opener = gzip.open if path.suffix == ".gz" else open
    vectors: dict[str, np.ndarray] = {}
    dim = None
    try:
        with opener(path, "rt", encoding="utf-8", errors="replace") as handle:
            for line_no, line in enumerate(handle):
                parts = line.rstrip().split(" ")
                if line_no == 0 and len(parts) == 2:
                    continue  # word2vec header "<vocab_size> <dimensions>"
                if dim is None:
                    dim = len(parts) - 1
                if len(parts) != dim + 1:
                    continue  # skip malformed lines
                vectors[parts[0]] = np.asarray(parts[1:], dtype=np.float32)
    except (OSError, ValueError, EOFError) as exc:
        raise MethodUnavailableError(f"Could not read GloVe file '{path}': {exc}") from exc
    if not vectors:
        raise MethodUnavailableError(f"No word vectors found in '{path}'.")
    logger.info("Loaded %d GloVe vectors of dimension %d from %s", len(vectors), dim, path)
    return vectors


class GloveMatcher(BaseMatcher):
    """Average of pretrained GloVe word vectors per document, compared with cosine similarity."""

    method = METHOD_GLOVE

    def __init__(self, corpus: ResumeCorpus, word_vectors: Mapping[str, np.ndarray]):
        super().__init__(corpus)
        if not word_vectors:
            raise MethodUnavailableError("GloVe word vectors are empty.")
        self.word_vectors = word_vectors
        self.dim = len(next(iter(word_vectors.values())))
        self.resume_matrix: np.ndarray | None = None
        self.zero_vector_resumes = 0

    def document_vector(self, text: str) -> tuple[np.ndarray, int]:
        """Average the vectors of known, non-stop-word tokens.

        Returns ``(vector, known_word_count)``; the vector is all zeros when no
        word is in the GloVe vocabulary.
        """
        known = [self.word_vectors[t] for t in tokenize(text, remove_stop_words=True) if t in self.word_vectors]
        if not known:
            return np.zeros(self.dim, dtype=np.float32), 0
        return np.mean(known, axis=0), len(known)

    def fit(self) -> "GloveMatcher":
        vectors, counts = zip(*(self.document_vector(t) for t in self.corpus.df["clean_text"]), strict=True)
        self.zero_vector_resumes = sum(1 for c in counts if c == 0)
        if self.zero_vector_resumes:
            logger.warning("%d resumes have no GloVe-known words; their similarity is 0.", self.zero_vector_resumes)
        self.resume_matrix = l2_normalize(np.vstack(vectors))
        self.is_fitted = True
        return self

    def similarities(self, job_description: str) -> np.ndarray:
        vector, known = self.document_vector(job_description)
        if known == 0:
            raise EmptyQueryError("None of the job description's words are in the GloVe vocabulary.")
        return self.resume_matrix @ l2_normalize(vector).ravel()


# --------------------------------------------------------------------------- Sentence-BERT


def sbert_installed() -> bool:
    """True if the sentence-transformers package can be imported."""
    return importlib.util.find_spec("sentence_transformers") is not None


def load_sbert_model(model_name: str, device: str | None = None):
    """Load a sentence-transformers model, raising MethodUnavailableError on failure."""
    if not sbert_installed():
        raise MethodUnavailableError(
            "The 'sentence-transformers' package is not installed. Run: pip install sentence-transformers"
        )
    try:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(model_name, device=device)
    except Exception as exc:  # network errors, bad model names, corrupt cache, ...
        raise MethodUnavailableError(
            f"Could not load Sentence-BERT model '{model_name}': {exc}. "
            "The first run needs internet access to download the model."
        ) from exc
    logger.info("Loaded Sentence-BERT model '%s' on %s", model_name, model.device)
    return model


class SbertMatcher(BaseMatcher):
    """Sentence-BERT document embeddings compared with cosine similarity.

    Resume embeddings are computed once and cached on disk (keyed by model
    name and corpus content), so later runs and queries are fast.
    """

    method = METHOD_SBERT

    def __init__(
        self,
        corpus: ResumeCorpus,
        model,
        model_name: str = "custom",
        batch_size: int = 32,
        cache_dir: Path | None = None,
        query_prefix: str = "",
    ):
        super().__init__(corpus)
        self.model = model
        self.model_name = model_name
        # Some models (e.g. BGE) are trained with an instruction in front of the search query.
        self.query_prefix = query_prefix
        self.batch_size = batch_size
        self.cache_dir = cache_dir
        self.resume_matrix: np.ndarray | None = None
        self.loaded_from_cache = False

    @staticmethod
    def prepare_text(text: str) -> str:
        """Sentence-BERT reads natural text, so we only normalize whitespace.

        Both resumes and job descriptions go through this same function.
        """
        return normalize_whitespace(text)

    def _encode(self, texts: list[str]) -> np.ndarray:
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        # Normalize again ourselves in case a custom model ignores the flag.
        return l2_normalize(np.asarray(embeddings))

    def _cache_path(self, texts: list[str]) -> Path | None:
        if self.cache_dir is None:
            return None
        digest = hashlib.sha256()
        digest.update(self.model_name.encode())
        for text in texts:
            digest.update(b"\x00" + text.encode("utf-8", errors="replace"))
        safe_name = "".join(c if c.isalnum() else "_" for c in self.model_name)
        return Path(self.cache_dir) / f"sbert_{safe_name}_{digest.hexdigest()[:16]}.npy"

    def fit(self) -> "SbertMatcher":
        texts = [self.prepare_text(t) for t in self.corpus.df["text"]]
        cache_path = self._cache_path(texts)
        if cache_path is not None and cache_path.is_file():
            try:
                cached = np.load(cache_path)
                if cached.shape[0] == len(texts):
                    self.resume_matrix = cached
                    self.loaded_from_cache = True
                    self.is_fitted = True
                    logger.info("Loaded cached Sentence-BERT embeddings from %s", cache_path)
                    return self
            except (OSError, ValueError) as exc:
                logger.warning("Ignoring unreadable embedding cache %s: %s", cache_path, exc)

        try:
            self.resume_matrix = self._encode(texts)
        except Exception as exc:
            raise MatcherError(f"Sentence-BERT failed to encode the resumes: {exc}") from exc

        if cache_path is not None:
            try:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                np.save(cache_path, self.resume_matrix)
            except OSError as exc:
                logger.warning("Could not save embedding cache: %s", exc)
        self.is_fitted = True
        return self

    def similarities(self, job_description: str) -> np.ndarray:
        query = self._encode([self.query_prefix + self.prepare_text(job_description)])
        return self.resume_matrix @ query.ravel()


# --------------------------------------------------------------------------- Hybrid


class HybridMatcher(BaseMatcher):
    """Equal average of the TF-IDF, GloVe and Sentence-BERT cosine similarities.

    TF-IDF rewards exact skill words ("pandas", "Spring Boot"), GloVe rewards related
    words, and Sentence-BERT rewards related meaning. Averaging the three cosines keeps
    the score on the same 0-100 scale.
    """

    method = METHOD_HYBRID

    def __init__(self, corpus: ResumeCorpus, tfidf: TfidfMatcher, glove: GloveMatcher, sbert: SbertMatcher):
        super().__init__(corpus)
        self.parts = (tfidf, glove, sbert)

    def fit(self) -> "HybridMatcher":
        for part in self.parts:
            part.fit()
        self.is_fitted = True
        return self

    def similarities(self, job_description: str) -> np.ndarray:
        scores = []
        for part in self.parts:
            try:
                scores.append(part.similarities(job_description))
            except EmptyQueryError:
                continue  # this method knows none of the job description's words; average the others
        if not scores:
            raise EmptyQueryError("None of the methods could read the job description. Add more specific words.")
        return np.mean(scores, axis=0)


# --------------------------------------------------------------------------- factory


def method_status(method: str, settings: Settings) -> tuple[bool, str]:
    """Cheap check (no model loading) of whether a method can probably run."""
    if method == METHOD_TFIDF:
        return True, "Ready (no external model needed)."
    if method == METHOD_GLOVE:
        if Path(settings.glove_path).is_file():
            return True, f"GloVe file found: {Path(settings.glove_path).name}"
        return False, f"GloVe file not found at {settings.glove_path}."
    if method == METHOD_SBERT:
        if sbert_installed():
            return True, f"Package installed; model '{settings.sbert_model_name}' loads on first use."
        return False, "Package 'sentence-transformers' is not installed."
    if method == METHOD_HYBRID:
        missing = [name for name in (METHOD_GLOVE, METHOD_SBERT) if not method_status(name, settings)[0]]
        if missing:
            return False, "Needs " + " and ".join(METHOD_LABELS[m] for m in missing) + "."
        return True, "Ready (uses TF-IDF, GloVe and Sentence-BERT)."
    raise ValueError(f"Unknown method '{method}'. Choose from {list(METHOD_LABELS)}.")


def create_matcher(
    method: str,
    corpus: ResumeCorpus,
    settings: Settings,
    word_vectors: Mapping[str, np.ndarray] | None = None,
    sbert_model=None,
) -> BaseMatcher:
    """Build and fit the matcher for ``method``.

    ``word_vectors`` / ``sbert_model`` can pass in an already-loaded GloVe table or
    Sentence-BERT model (the web app keeps them in memory); otherwise they are loaded here.
    Raises MethodUnavailableError if the method's model or package is missing.
    """
    if method not in METHOD_LABELS:
        raise ValueError(f"Unknown method '{method}'. Choose from {list(METHOD_LABELS)}.")

    def glove() -> GloveMatcher:
        return GloveMatcher(corpus, word_vectors or load_glove_vectors(settings.glove_path))

    def sbert() -> SbertMatcher:
        return SbertMatcher(
            corpus,
            sbert_model or load_sbert_model(settings.sbert_model_name, settings.sbert_device),
            model_name=settings.sbert_model_name,
            batch_size=settings.sbert_batch_size,
            cache_dir=settings.cache_dir,
            query_prefix=SBERT_QUERY_PREFIXES.get(settings.sbert_model_name, ""),
        )

    if method == METHOD_TFIDF:
        matcher: BaseMatcher = TfidfMatcher(corpus)
    elif method == METHOD_GLOVE:
        matcher = glove()
    elif method == METHOD_SBERT:
        matcher = sbert()
    else:
        matcher = HybridMatcher(corpus, TfidfMatcher(corpus), glove(), sbert())
    return matcher.fit()
