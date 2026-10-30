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
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer

from resume_matcher.chunking import SECTION_OTHER, chunk_text
from resume_matcher.config import (
    GLOVE_WEIGHTING,
    HYBRID_NORMALIZATION,
    HYBRID_WEIGHTS,
    METHOD_GLOVE,
    METHOD_HYBRID,
    METHOD_LABELS,
    METHOD_SBERT,
    METHOD_TFIDF,
    PREVIEW_CHARS,
    SBERT_CHUNK_TOKENS,
    SBERT_DOC_WEIGHT,
    SBERT_QUERY_PREFIXES,
    Settings,
)
from resume_matcher.data_loader import ResumeCorpus
from resume_matcher.preprocessing import clean_text, make_preview, normalize_whitespace
from resume_matcher.text_features import (
    TFIDF_STOP_WORDS,
    canonicalize_skills,
    prepare_tfidf_text,
    tfidf_tokens,
)

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
        # Stable sort: ties keep dataset order, so results are reproducible. Similarities are rounded
        # to 12 decimals for sorting only, so float noise (1e-17) cannot reorder genuinely tied resumes.
        order = np.argsort(-np.round(sims, 12), kind="stable")[:top_k]
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
    """Bag-of-words matching: TF-IDF vectors compared with cosine similarity.

    Text goes through text_features.prepare_tfidf_text (skill aliases spelled one way, technical
    terms kept), then single-word tokens without stop words or number-only tokens.

    The IDF weights are fitted on all resumes *plus* the job description, as if a TfidfVectorizer
    (sublinear TF, smoothed IDF, L2 norm) were fitted on ``resumes + [job description]``. So words
    of the job description that no resume uses still count in its vector and lower every score
    equally, instead of being silently dropped (which inflated scores). To avoid re-tokenizing
    every resume for each query, the resume term counts are computed once in ``fit`` and only the
    IDF weighting is redone per query, on sparse matrices.

    Settings were chosen on the two evaluation datasets (README §13): adding word pairs
    (ngram_range=(1, 2)) gave no consistent gain while making the vocabulary 10-17x larger, so
    known multi-word skills become single tokens instead ("machine_learning"). min_df=1 and
    max_df=1.0 because an upload can be just a few resumes; sublinear TF so that repeating a
    keyword many times does not keep raising the score.
    """

    method = METHOD_TFIDF

    def __init__(self, corpus: ResumeCorpus):
        super().__init__(corpus)
        self.vectorizer = CountVectorizer(tokenizer=tfidf_tokens, token_pattern=None, lowercase=False)
        self.resume_tf: sparse.csr_matrix | None = None  # sublinear term frequencies, one row per resume
        self.resume_df: np.ndarray | None = None  # number of resumes containing each term

    def fit(self) -> "TfidfMatcher":
        texts = [prepare_tfidf_text(t) for t in self.corpus.df["text"]]
        try:
            counts = self.vectorizer.fit_transform(texts).astype(np.float64).tocsr()
        except ValueError as exc:  # "empty vocabulary" when every word is a stop word
            raise MatcherError(f"TF-IDF could not build a vocabulary from the resumes: {exc}") from exc
        self.resume_df = np.bincount(counts.indices, minlength=counts.shape[1]).astype(np.float64)
        counts.data = 1.0 + np.log(counts.data)  # sublinear TF
        self.resume_tf = counts
        self.is_fitted = True
        return self

    def query_tokens(self, job_description: str) -> list[str]:
        return tfidf_tokens(prepare_tfidf_text(job_description))

    def similarities(self, job_description: str) -> np.ndarray:
        tokens = self.query_tokens(job_description)
        vocabulary = self.vectorizer.vocabulary_
        if not any(token in vocabulary for token in tokens):
            raise EmptyQueryError(
                "None of the job description's words appear in the resume vocabulary "
                "(after removing stop words). Add more specific skills or duties."
            )
        known: dict[int, int] = {}
        unknown: dict[str, int] = {}  # job-description words that no resume uses
        for token in tokens:
            if token in vocabulary:
                known[vocabulary[token]] = known.get(vocabulary[token], 0) + 1
            else:
                unknown[token] = unknown.get(token, 0) + 1

        # Smoothed IDF over resumes + job description: idf = ln((1 + n) / (1 + df)) + 1.
        n_docs = self.resume_tf.shape[0] + 1
        df = self.resume_df.copy()
        columns = np.fromiter(known, dtype=np.int64)
        df[columns] += 1
        idf = np.log((1.0 + n_docs) / (1.0 + df)) + 1.0
        unknown_idf = np.log((1.0 + n_docs) / 2.0) + 1.0  # df = 1: only the job description

        weighted = self.resume_tf.multiply(idf).tocsr()  # sparse: only terms each resume uses
        resume_norms = np.sqrt(np.asarray(weighted.multiply(weighted).sum(axis=1)).ravel())
        query = (1.0 + np.log(np.fromiter(known.values(), dtype=np.float64))) * idf[columns]
        unknown_weights = (1.0 + np.log(np.fromiter(unknown.values(), dtype=np.float64))) * unknown_idf
        query_norm = np.sqrt(np.sum(query**2) + np.sum(unknown_weights**2))

        dots = np.asarray(weighted[:, columns] @ query).ravel()
        with np.errstate(divide="ignore", invalid="ignore"):
            sims = np.where(resume_norms > 0, dots / (resume_norms * query_norm), 0.0)
        return sims


# --------------------------------------------------------------------------- GloVe


class GloveVectors(Mapping):
    """A GloVe table as one float32 matrix plus a word -> row index.

    Behaves like a read-only ``{word: vector}`` mapping, so code (and tests) that pass a plain
    dict still work, while document vectors can be computed for many texts in one sparse
    matrix product. One matrix also uses less memory than 400,000 separate arrays.
    """

    def __init__(self, words: list[str], matrix: np.ndarray):
        self.index = {word: i for i, word in enumerate(words)}
        self.matrix = np.ascontiguousarray(matrix, dtype=np.float32)
        self.dim = self.matrix.shape[1]
        self._common_direction: np.ndarray | None = None

    @classmethod
    def from_mapping(cls, vectors: Mapping[str, np.ndarray]) -> "GloveVectors":
        if isinstance(vectors, GloveVectors):
            return vectors
        words = list(vectors)
        return cls(words, np.vstack([np.asarray(vectors[w], dtype=np.float32) for w in words]))

    def __getitem__(self, word: str) -> np.ndarray:
        return self.matrix[self.index[word]]

    def __contains__(self, word: object) -> bool:
        return word in self.index

    def __iter__(self):
        return iter(self.index)

    def __len__(self) -> int:
        return len(self.index)

    def common_direction(self) -> np.ndarray | None:
        """The direction shared by averaged word vectors, from the vocabulary itself.

        Averages of GloVe vectors all lean the same way, which squeezes every document's cosine
        into roughly 0.8-0.95. Removing this direction (as in "SIF" sentence embeddings, Arora et
        al. 2017) spreads the scores out. It is the first principal direction of the most frequent
        words, weighted by an estimate of their frequency (GloVe files are sorted by frequency),
        so it does not depend on which or how many resumes are uploaded. ``None`` for small toy
        vocabularies, where it would not be meaningful.
        """
        if self._common_direction is None and len(self) >= GLOVE_MIN_VOCAB_FOR_DIRECTION:
            top = min(len(self), 50_000)
            weights = 1.0 / (np.arange(top, dtype=np.float64) + 10.0)  # Zipf estimate from frequency rank
            weighted = np.sqrt(weights / weights.sum())[:, None] * self.matrix[:top]
            self._common_direction = np.linalg.svd(weighted, full_matrices=False)[2][0].astype(np.float32)
        return self._common_direction


GLOVE_MIN_VOCAB_FOR_DIRECTION = 1000
_GLOVE_CACHE: dict[tuple[str, int, int], GloveVectors] = {}
_GLOVE_BLOCK_LINES = 20_000


def _parse_glove_block(words: list[str], rows: list[str], dim: int) -> tuple[list[str], np.ndarray]:
    """Parse a block of "v1 v2 ..." strings; lines with the wrong number of values are skipped."""
    keep_words, values = [], []
    for word, row in zip(words, rows, strict=True):
        parts = row.split(" ")
        if len(parts) == dim:
            keep_words.append(word)
            values.append(parts)
    if not values:
        return [], np.zeros((0, dim), dtype=np.float32)
    return keep_words, np.asarray(values, dtype=np.float32)


def load_glove_vectors(path: str | Path) -> GloveVectors:
    """Load GloVe vectors from a text file (``.txt`` or gzipped ``.gz``), once per process.

    Supports the original Stanford format (``word v1 v2 ...``) and the gensim-data
    word2vec format, which has an extra ``<count> <dim>`` header line. Malformed lines are
    skipped and counted; a file with fewer vectors than its header announces is reported as
    possibly incomplete. The parsed table is kept in memory and reused for the same file
    (same path, size and modification time).
    """
    path = Path(path)
    if not path.is_file():
        raise MethodUnavailableError(
            f"GloVe vectors not found at '{path}'. See README > 'Optional: GloVe' "
            "for the download command, or set GLOVE_PATH."
        )
    stat = path.stat()
    key = (str(path.resolve()), stat.st_size, int(stat.st_mtime))
    if key in _GLOVE_CACHE:
        return _GLOVE_CACHE[key]

    opener = gzip.open if path.suffix == ".gz" else open
    words: list[str] = []
    blocks: list[np.ndarray] = []
    matrix: np.ndarray | None = None  # filled in place when the header gives the size
    expected = dim = None
    skipped = 0

    def flush(block_words: list[str], block_rows: list[str]) -> None:
        nonlocal skipped, matrix
        kept, values = _parse_glove_block(block_words, block_rows, dim)
        skipped += len(block_words) - len(kept)
        if matrix is not None and len(words) + len(kept) <= len(matrix):
            matrix[len(words) : len(words) + len(kept)] = values
        else:
            if matrix is not None:  # more rows than the header said: fall back to stacking
                blocks.append(matrix[: len(words)].copy())
                matrix = None
            blocks.append(values)
        words.extend(kept)

    try:
        with opener(path, "rt", encoding="utf-8", errors="replace") as handle:
            block_words, block_rows = [], []
            for line_no, line in enumerate(handle):
                line = line.rstrip()
                if not line:
                    continue
                word, _, rest = line.partition(" ")
                if line_no == 0 and len(line.split(" ")) == 2:
                    expected = int(word)  # word2vec header "<vocab_size> <dimensions>"
                    continue
                if dim is None:
                    dim = len(rest.split(" "))
                    if expected:
                        matrix = np.empty((expected, dim), dtype=np.float32)
                block_words.append(word)
                block_rows.append(rest)
                if len(block_words) == _GLOVE_BLOCK_LINES:
                    flush(block_words, block_rows)
                    block_words, block_rows = [], []
            if block_words:
                flush(block_words, block_rows)
    except (OSError, ValueError, EOFError) as exc:
        raise MethodUnavailableError(f"Could not read GloVe file '{path}': {exc}") from exc
    if not words:
        raise MethodUnavailableError(f"No word vectors found in '{path}'.")
    if skipped:
        logger.warning("Skipped %d malformed lines in %s", skipped, path)
    if expected is not None and len(words) < expected:
        logger.warning("GloVe file %s may be incomplete: %d of %d vectors read.", path, len(words), expected)
    table = matrix[: len(words)] if matrix is not None else np.vstack(blocks)
    vectors = GloveVectors(words, table)
    logger.info("Loaded %d GloVe vectors of dimension %d from %s", len(vectors), vectors.dim, path)
    _GLOVE_CACHE.clear()  # keep one table in memory
    _GLOVE_CACHE[key] = vectors
    return vectors


def glove_tokens(text: object, vocabulary: Mapping[str, object]) -> list[str]:
    """Words of ``text`` that have a GloVe vector, after the same skill normalization as TF-IDF.

    Known skills and aliases are spelled one way first ("k8s" -> "kubernetes", "Node JS" ->
    "node.js"); multi-word skills are then split back into words ("machine learning"), because
    GloVe has vectors for words, not for our joined tokens. Stop words are dropped except words
    that matter in job ads ("front", "back", "full", "system"). A token GloVe does not know is
    tried in pieces ("asp.net" -> "asp", "net") before being skipped.
    """
    words = []
    for token in clean_text(canonicalize_skills(text).replace("_", " ")).split():
        if token in TFIDF_STOP_WORDS:
            continue
        if token in vocabulary:
            words.append(token)
            continue
        for piece in token.replace(".", " ").replace("+", " ").replace("#", " ").split():
            if piece in vocabulary and piece not in TFIDF_STOP_WORDS:
                words.append(piece)
    return words


class GloveMatcher(BaseMatcher):
    """Average of pretrained GloVe word vectors per document, compared with cosine similarity.

    Each document is the plain average of the vectors of its known words (a word used twice
    counts twice). The direction that all averaged vectors share is then removed (see
    GloveVectors.common_direction), so scores are no longer squeezed into 80-95; finally vectors
    are L2-normalized and compared by cosine. Alternatives tried on the evaluation data and not
    adopted (README §13): IDF weighting, counting each word once, damped counts, SIF word weights
    and removing a direction learned from the uploaded resumes (unstable for small uploads).
    """

    method = METHOD_GLOVE

    def __init__(self, corpus: ResumeCorpus, word_vectors: Mapping[str, np.ndarray], weighting: str = GLOVE_WEIGHTING):
        super().__init__(corpus)
        if not word_vectors:
            raise MethodUnavailableError("GloVe word vectors are empty.")
        if weighting not in ("mean", "idf"):
            raise ValueError("weighting must be 'mean' or 'idf'.")
        # "mean": every known word counts once per use. "idf" (optional, evaluated and not the default):
        # each distinct word weighted by (1 + log count) x IDF over the uploaded resumes.
        self.weighting = weighting
        self.idf: np.ndarray | None = None
        self.word_vectors = GloveVectors.from_mapping(word_vectors)
        self.dim = self.word_vectors.dim
        self.direction = self.word_vectors.common_direction()
        self.resume_matrix: np.ndarray | None = None
        self.zero_vector_resumes = 0
        self.known_word_share: float | None = None  # share of resume words with a GloVe vector

    def _averages(self, texts: list[str]) -> tuple[np.ndarray, np.ndarray]:
        """Average vectors of many texts at once (sparse word counts x vector matrix)."""
        index = self.word_vectors.index
        rows, cols, counts = [], [], []
        known = np.zeros(len(texts), dtype=np.int64)
        for row, text in enumerate(texts):
            per_word: dict[int, int] = {}
            for word in glove_tokens(text, index):
                per_word[index[word]] = per_word.get(index[word], 0) + 1
            for col, n in per_word.items():
                rows.append(row)
                cols.append(col)
                counts.append(n)
            known[row] = sum(per_word.values())
        values = np.asarray(counts, dtype=np.float32)
        if self.weighting == "idf" and self.idf is not None and len(values):
            values = (1.0 + np.log(values)) * self.idf[np.asarray(cols)]
        weights = sparse.csr_matrix((values, (rows, cols)), shape=(len(texts), len(index)))
        totals = np.asarray(weights.sum(axis=1)).ravel()
        sums = np.asarray(weights @ self.word_vectors.matrix)
        return sums / np.where(totals > 0, totals, 1.0)[:, None], known

    def document_vector(self, text: str) -> tuple[np.ndarray, int]:
        """Average of the known words' vectors. Returns ``(vector, known_word_count)``; the
        vector is all zeros when no word is in the GloVe vocabulary."""
        vectors, known = self._averages([text])
        return vectors[0].astype(np.float32), int(known[0])

    def _finish(self, vectors: np.ndarray) -> np.ndarray:
        if self.direction is not None:
            vectors = vectors - np.outer(vectors @ self.direction, self.direction)
        return l2_normalize(vectors)

    def fit(self) -> "GloveMatcher":
        if self.weighting == "idf":
            index = self.word_vectors.index
            df = np.zeros(len(index), dtype=np.float32)
            for text in self.corpus.df["text"]:
                for col in {index[w] for w in glove_tokens(text, index)}:
                    df[col] += 1
            n = len(self.corpus)
            self.idf = (np.log((1.0 + n) / (1.0 + df)) + 1.0).astype(np.float32)
        vectors, known = self._averages(self.corpus.df["text"].tolist())
        self.zero_vector_resumes = int((known == 0).sum())
        if self.zero_vector_resumes:
            logger.warning("%d resumes have no GloVe-known words; their similarity is 0.", self.zero_vector_resumes)
        total_words = sum(len(clean_text(t).split()) for t in self.corpus.df["text"])
        self.known_word_share = float(known.sum() / total_words) if total_words else None
        self.resume_matrix = self._finish(vectors)
        self.resume_matrix[known == 0] = 0.0  # no known words: no direction, similarity 0
        self.is_fitted = True
        return self

    def similarities(self, job_description: str) -> np.ndarray:
        vector, known = self.document_vector(job_description)
        if known == 0:
            raise EmptyQueryError("None of the job description's words are in the GloVe vocabulary.")
        return self.resume_matrix @ self._finish(vector).ravel()


# --------------------------------------------------------------------------- Sentence-BERT


def sbert_installed() -> bool:
    """True if the sentence-transformers package can be imported."""
    return importlib.util.find_spec("sentence_transformers") is not None


def load_sbert_model(model_name: str, device: str | None = None):
    """Load a sentence-transformers model, raising MethodUnavailableError on failure.

    If the normal load fails (for example without internet access), a copy already in the local
    Hugging Face cache is tried before giving up.
    """
    if not sbert_installed():
        raise MethodUnavailableError(
            "The 'sentence-transformers' package is not installed. Run: pip install sentence-transformers"
        )
    from sentence_transformers import SentenceTransformer

    try:
        model = SentenceTransformer(model_name, device=device)
    except Exception as first_error:  # noqa: BLE001 - network errors, bad model names, corrupt cache, ...
        try:
            model = SentenceTransformer(model_name, device=device, local_files_only=True)
            logger.warning("Loaded '%s' from the local cache after: %s", model_name, first_error)
        except Exception:  # noqa: BLE001 - any failure here means the model is unavailable
            raise MethodUnavailableError(
                f"Could not load Sentence-BERT model '{model_name}': {first_error}. "
                "The first run needs internet access to download the model."
            ) from first_error
    logger.info("Loaded Sentence-BERT model '%s' on %s", model_name, model.device)
    return model


class SbertMatcher(BaseMatcher):
    """Sentence-BERT embeddings of whole resumes and of resume chunks, compared with cosine similarity.

    The model reads at most ``max_seq_length`` word-pieces (512 for BGE) and ignores the rest, and
    many resumes are longer. So each resume is also split into section-aware chunks of at most
    ``chunk_tokens`` word-pieces (chunking.py) and the score is

        similarity = doc_weight * cosine(job, whole resume) + (1 - doc_weight) * max over chunks cosine(job, chunk)

    The whole-resume term keeps the overall picture; the best chunk makes sure a relevant part
    deep in a long resume still counts. ``document_similarities`` and ``best_chunks`` keep the two
    parts available separately. All embeddings are L2-normalized (BGE uses CLS pooling followed by
    normalization), so dot products are cosines. Resume and chunk embeddings are computed once, in
    one batch, and can be cached on disk (keyed by model, chunk settings and text).
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
        chunk_tokens: int = SBERT_CHUNK_TOKENS,
        doc_weight: float = SBERT_DOC_WEIGHT,
    ):
        super().__init__(corpus)
        if not 0.0 <= doc_weight <= 1.0:
            raise ValueError("doc_weight must be between 0 and 1.")
        self.model = model
        self.model_name = model_name
        # Some models (e.g. BGE) are trained with an instruction in front of the search query.
        self.query_prefix = query_prefix
        self.batch_size = batch_size
        self.cache_dir = cache_dir
        self.chunk_tokens = chunk_tokens
        self.doc_weight = doc_weight
        self.resume_matrix: np.ndarray | None = None  # one embedding per resume (first max_seq_length pieces)
        self.chunk_matrix: np.ndarray | None = None  # one embedding per chunk
        self.chunk_owner: np.ndarray | None = None  # resume row of each chunk
        self.chunks: list[tuple[str, str]] = []  # (section, text) of each chunk
        self.loaded_from_cache = False
        self._last: tuple[str, np.ndarray, np.ndarray] | None = None  # (query, document sims, best chunk index)

    @staticmethod
    def prepare_text(text: str) -> str:
        """Sentence-BERT reads natural text, so we only normalize whitespace.

        Both resumes and job descriptions go through this same function.
        """
        return normalize_whitespace(text)

    def _count_tokens(self, text: str) -> int:
        tokenizer = getattr(self.model, "tokenizer", None)
        if tokenizer is None:
            return int(len(text.split()) * 1.3) + 1
        return len(tokenizer(text, add_special_tokens=False)["input_ids"])

    def _max_tokens(self) -> int:
        return int(getattr(self.model, "max_seq_length", 512) or 512) - 2  # room for [CLS] and [SEP]

    def _encode(self, texts: list[str]) -> np.ndarray:
        kwargs = {
            "batch_size": self.batch_size, "show_progress_bar": False, "convert_to_numpy": True,
            "normalize_embeddings": True,
        }
        try:
            embeddings = self.model.encode(texts, **kwargs)
        except RuntimeError as exc:  # e.g. an Apple MPS or CUDA failure: retry once on the CPU
            device = str(getattr(self.model, "device", "cpu"))
            if device == "cpu" or not hasattr(self.model, "to"):
                raise
            logger.warning("Sentence-BERT failed on %s (%s); retrying on the CPU.", device, exc)
            self.model.to("cpu")
            embeddings = self.model.encode(texts, **kwargs)
        # Normalize again ourselves in case a custom model ignores the flag.
        return l2_normalize(np.asarray(embeddings))

    def _cache_path(self, texts: list[str]) -> Path | None:
        if self.cache_dir is None:
            return None
        digest = hashlib.sha256()
        digest.update(f"{self.model_name}|chunks={self.chunk_tokens}|v2".encode())
        for text in texts:
            digest.update(b"\x00" + text.encode("utf-8", errors="replace"))
        safe_name = "".join(c if c.isalnum() else "_" for c in self.model_name)
        return Path(self.cache_dir) / f"sbert_{safe_name}_{digest.hexdigest()[:16]}.npy"

    def fit(self) -> "SbertMatcher":
        texts = [self.prepare_text(t) for t in self.corpus.df["text"]]
        limit = min(self.chunk_tokens, self._max_tokens())
        owner, chunks = [], []
        for row, text in enumerate(texts):
            if limit <= 0:  # chunking switched off (SBERT_CHUNK_TOKENS=0): whole-resume embedding only
                break
            pieces = chunk_text(text, limit, self._count_tokens) or [(SECTION_OTHER, text)]
            for section, chunk in pieces:
                owner.append(row)
                chunks.append((section, chunk))
        self.chunks, self.chunk_owner = chunks, np.asarray(owner, dtype=np.int64)

        everything = texts + [chunk for _, chunk in chunks]
        cache_path = self._cache_path(texts)
        embeddings = None
        if cache_path is not None and cache_path.is_file():
            try:
                cached = np.load(cache_path)
                if cached.shape[0] == len(everything):
                    embeddings, self.loaded_from_cache = cached, True
                    logger.info("Loaded cached Sentence-BERT embeddings from %s", cache_path)
            except (OSError, ValueError) as exc:
                logger.warning("Ignoring unreadable embedding cache %s: %s", cache_path, exc)
        if embeddings is None:
            try:
                embeddings = self._encode(everything)  # whole resumes and chunks in one batch
            except Exception as exc:
                raise MatcherError(f"Sentence-BERT failed to encode the resumes: {exc}") from exc
            if cache_path is not None:
                try:
                    cache_path.parent.mkdir(parents=True, exist_ok=True)
                    np.save(cache_path, embeddings)
                except OSError as exc:
                    logger.warning("Could not save embedding cache: %s", exc)
        if embeddings.ndim != 2 or embeddings.shape[0] != len(everything):
            raise MatcherError("Sentence-BERT returned an unexpected number of embeddings.")
        self.resume_matrix = embeddings[: len(texts)]
        self.chunk_matrix = embeddings[len(texts) :]
        self.is_fitted = True
        return self

    def query_vector(self, job_description: str) -> np.ndarray:
        """Embedding of the job description; a description longer than the model's limit is
        chunked and its chunk embeddings averaged, so no part of it is ignored."""
        text = self.prepare_text(job_description)
        limit = self._max_tokens() - self._count_tokens(self.query_prefix)
        if self._count_tokens(text) <= limit:
            return self._encode([self.query_prefix + text]).ravel()
        parts = [chunk for _, chunk in chunk_text(text, limit, self._count_tokens)]
        return l2_normalize(self._encode([self.query_prefix + part for part in parts]).mean(axis=0)).ravel()

    def _parts(self, job_description: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(whole-resume similarities, best chunk similarities, index of each resume's best chunk)."""
        query = self.query_vector(job_description)
        document = self.resume_matrix @ query
        if not self.chunks:
            self._last = (job_description, document, np.full(len(document), -1, dtype=np.int64))
            return document, document, self._last[2]
        chunk_sims = self.chunk_matrix @ query
        order = np.lexsort((-chunk_sims, self.chunk_owner))  # by resume, best chunk first
        first = np.ones(len(order), dtype=bool)
        first[1:] = self.chunk_owner[order][1:] != self.chunk_owner[order][:-1]
        best_index = np.full(len(document), -1, dtype=np.int64)
        best_index[self.chunk_owner[order][first]] = order[first]
        best = np.where(best_index >= 0, chunk_sims[np.maximum(best_index, 0)], document)
        self._last = (job_description, document, best_index)
        return document, best, best_index

    def similarities(self, job_description: str) -> np.ndarray:
        document, best, _ = self._parts(job_description)
        if not self.chunks:
            return document
        return self.doc_weight * document + (1.0 - self.doc_weight) * best

    def document_similarities(self, job_description: str) -> np.ndarray:
        """Whole-resume cosine only (the earlier single-embedding score, kept as a separate metric)."""
        if self._last is not None and self._last[0] == job_description:
            return self._last[1]
        return self._parts(job_description)[0]

    def encode_queries(self, texts: list[str]) -> np.ndarray:
        """Embeddings of short search texts (with the model's query instruction, if any)."""
        return self._encode([self.query_prefix + self.prepare_text(t) for t in texts])

    def encode_passages(self, texts: list[str]) -> np.ndarray:
        """Embeddings of resume passages (no instruction)."""
        return self._encode([self.prepare_text(t) for t in texts])

    def best_chunks(self, job_description: str) -> list[tuple[str, str] | None]:
        """For each resume, the (section, text) of the chunk closest to the job description."""
        if self._last is None or self._last[0] != job_description:
            self._parts(job_description)
        return [self.chunks[i] if i >= 0 else None for i in self._last[2]]


# --------------------------------------------------------------------------- Hybrid


def minmax_normalize(values: np.ndarray) -> np.ndarray:
    """Scale one method's similarities for one query to 0-1 (best resume 1, worst 0).

    When all resumes score the same (for example a single resume), there is no spread to
    rescale, so the cosines are kept, clipped to 0-1.
    """
    values = np.asarray(values, dtype=np.float64)
    low, high = values.min(), values.max()
    if high - low < 1e-9:
        return np.clip(values, 0.0, 1.0)
    return (values - low) / (high - low)


class HybridMatcher(BaseMatcher):
    """Weighted combination of TF-IDF, GloVe and Sentence-BERT, after putting them on one scale.

    The three methods' cosines are not comparable: per job description their spread across
    resumes is about 3 points for TF-IDF, 5 for Sentence-BERT and 18 for GloVe (README §13), so a
    plain average of cosines is ranked mostly by GloVe, the weakest method. Each method's
    similarities are therefore min-max normalized per query (``normalization="minmax"``) and then
    combined with ``weights`` (TF-IDF, GloVe, Sentence-BERT; rescaled to sum to 1):

        hybrid = sum(w_m * minmax(similarity_m)) over the methods that could read the query

    The result is relative to the uploaded resumes: 1 for a resume that is best under every method,
    0 for one that is worst under every method. ``normalization="none"`` restores the plain
    weighted average of cosines. A method that knows none of the job description's words is left
    out and the remaining weights are rescaled; ``last_used`` / ``last_skipped`` say which.
    ``last_components`` keeps each method's raw cosines for display.
    """

    method = METHOD_HYBRID
    PART_NAMES = (METHOD_TFIDF, METHOD_GLOVE, METHOD_SBERT)

    def __init__(
        self,
        corpus: ResumeCorpus,
        tfidf: TfidfMatcher,
        glove: GloveMatcher,
        sbert: SbertMatcher,
        weights: tuple[float, float, float] = HYBRID_WEIGHTS,
        normalization: str = HYBRID_NORMALIZATION,
    ):
        super().__init__(corpus)
        self.parts = (tfidf, glove, sbert)
        self.weights = validate_hybrid_weights(weights)
        if normalization not in ("minmax", "none"):
            raise ValueError("normalization must be 'minmax' or 'none'.")
        self.normalization = normalization
        self.last_components: dict[str, np.ndarray] = {}
        self.last_used: tuple[str, ...] = ()
        self.last_skipped: dict[str, str] = {}

    def fit(self) -> "HybridMatcher":
        for part in self.parts:
            part.fit()
        self.is_fitted = True
        return self

    def similarities(self, job_description: str) -> np.ndarray:
        components, skipped = {}, {}
        for name, part in zip(self.PART_NAMES, self.parts, strict=True):
            try:
                components[name] = np.asarray(part.similarities(job_description), dtype=np.float64)
            except EmptyQueryError as exc:  # this method knows none of the job description's words
                skipped[name] = str(exc)
        self.last_components, self.last_skipped = components, skipped
        used = [name for name in self.PART_NAMES if name in components and self.weights[self.PART_NAMES.index(name)] > 0]
        self.last_used = tuple(used)
        if not used:
            raise EmptyQueryError("None of the methods could read the job description. Add more specific words.")
        weights = np.array([self.weights[self.PART_NAMES.index(name)] for name in used])
        weights = weights / weights.sum()
        scale = minmax_normalize if self.normalization == "minmax" else (lambda v: v)
        return sum(w * scale(components[name]) for w, name in zip(weights, used, strict=True))


def validate_hybrid_weights(weights) -> tuple[float, float, float]:
    """Three non-negative weights (TF-IDF, GloVe, Sentence-BERT), rescaled to sum to 1."""
    values = tuple(float(w) for w in weights)
    if len(values) != 3 or any(not np.isfinite(w) or w < 0 for w in values) or sum(values) <= 0:
        raise ValueError("Hybrid weights must be three non-negative numbers that are not all zero.")
    total = sum(values)
    return tuple(w / total for w in values)


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
        return GloveMatcher(corpus, word_vectors or load_glove_vectors(settings.glove_path), weighting=settings.glove_weighting)

    def sbert() -> SbertMatcher:
        return SbertMatcher(
            corpus,
            sbert_model or load_sbert_model(settings.sbert_model_name, settings.sbert_device),
            model_name=settings.sbert_model_name,
            batch_size=settings.sbert_batch_size,
            cache_dir=settings.cache_dir,
            query_prefix=SBERT_QUERY_PREFIXES.get(settings.sbert_model_name, ""),
            chunk_tokens=settings.sbert_chunk_tokens,
            doc_weight=settings.sbert_doc_weight,
        )

    if method == METHOD_TFIDF:
        matcher: BaseMatcher = TfidfMatcher(corpus)
    elif method == METHOD_GLOVE:
        matcher = glove()
    elif method == METHOD_SBERT:
        matcher = sbert()
    else:
        matcher = HybridMatcher(
            corpus, TfidfMatcher(corpus), glove(), sbert(),
            weights=settings.hybrid_weights, normalization=settings.hybrid_normalization,
        )
    return matcher.fit()
