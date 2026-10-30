"""GloVe, Sentence-BERT, chunking, Hybrid and semantic-evidence tests with offline fakes."""

import gzip
import logging

import numpy as np
import pandas as pd
import pytest

import resume_matcher.matcher as matcher_module
from resume_matcher.chunking import (
    SECTION_EDUCATION,
    SECTION_EXPERIENCE,
    SECTION_SKILLS,
    chunk_text,
    split_sections,
)
from resume_matcher.comparison import (
    combine_method_rankings,
    ranking_agreement,
    status_matrix,
)
from resume_matcher.coverage import (
    STATUS_MISSING,
    STATUS_SEMANTIC,
    assess,
    job_requirements,
    resume_sentences,
    semantic_evidence,
    semantic_evidence_batch,
)
from resume_matcher.data_loader import corpus_from_files, prepare_corpus
from resume_matcher.matcher import (
    EmptyQueryError,
    GloveMatcher,
    GloveVectors,
    HybridMatcher,
    MatcherError,
    MethodUnavailableError,
    SbertMatcher,
    TfidfMatcher,
    glove_tokens,
    l2_normalize,
    load_glove_vectors,
    load_sbert_model,
)
from tests.conftest import FAKE_GLOVE, FakeSentenceModel

# ---------------------------------------------------------------- GloVe loading


def write_glove(path, rows, header=None):
    lines = ([header] if header else []) + [f"{w} " + " ".join(str(v) for v in vec) for w, vec in rows]
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "wt", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def test_glove_table_is_one_matrix_and_behaves_like_a_dict(tmp_path):
    path = tmp_path / "g.txt"
    write_glove(path, [("python", [0.1, 0.2]), ("nurse", [0.3, 0.4]), ("broken", [0.5])])
    vectors = load_glove_vectors(path)
    assert isinstance(vectors, GloveVectors) and vectors.matrix.shape == (2, 2) and vectors.dim == 2
    assert vectors.matrix.dtype == np.float32
    assert "python" in vectors and "broken" not in vectors and len(vectors) == 2
    np.testing.assert_allclose(vectors["nurse"], [0.3, 0.4])
    assert set(vectors) == {"python", "nurse"}


def test_glove_file_is_loaded_once(tmp_path, monkeypatch):
    path = tmp_path / "g.gz"
    write_glove(path, [("python", [0.1, 0.2])], header="1 2")
    first = load_glove_vectors(path)
    monkeypatch.setattr(matcher_module.gzip, "open", lambda *a, **k: pytest.fail("file read again"))
    assert load_glove_vectors(path) is first  # same table, not re-read


def test_incomplete_or_malformed_glove_file_is_reported(tmp_path, caplog):
    path = tmp_path / "g.gz"
    write_glove(path, [("python", [0.1, 0.2]), ("bad", [0.1, 0.2, 0.3])], header="5 2")
    with caplog.at_level(logging.WARNING):
        vectors = load_glove_vectors(path)
    assert len(vectors) == 1
    assert "malformed" in caplog.text and "incomplete" in caplog.text


def test_missing_glove_file_has_a_clear_message(tmp_path):
    with pytest.raises(MethodUnavailableError, match="not found.*GLOVE_PATH"):
        load_glove_vectors(tmp_path / "nope.gz")


# ---------------------------------------------------------------- GloVe matching

VOCAB = {**FAKE_GLOVE, "front": np.array([0.5, 0.0, 0.5], np.float32), "end": np.array([0.4, 0.1, 0.5], np.float32),
         "node.js": np.array([0.9, 0.0, 0.1], np.float32), "machine": np.array([0.7, 0.2, 0.1], np.float32),
         "learning": np.array([0.6, 0.3, 0.1], np.float32)}


def test_glove_tokens_keep_tech_words_and_normalize_aliases():
    assert glove_tokens("Front-end developer, Node JS and Machine-Learning; the of", VOCAB) == [
        "front", "end", "developer", "node.js", "machine", "learning",
    ]
    assert glove_tokens("asp.net python", VOCAB) == ["python"]  # unknown pieces are skipped
    assert glove_tokens("", VOCAB) == [] and glove_tokens(None, VOCAB) == []


def test_glove_oov_and_empty_documents_score_zero_without_errors():
    corpus = prepare_corpus(pd.DataFrame({"text": ["python developer", "qwerty asdfgh", "2019 2020"]}))
    matcher = GloveMatcher(corpus, VOCAB).fit()
    assert matcher.zero_vector_resumes == 2
    sims = matcher.similarities("python developer")
    assert sims[0] > 0 and sims[1] == 0 and sims[2] == 0
    with pytest.raises(EmptyQueryError):
        matcher.similarities("zzzz qqqq")


def test_glove_repeated_words_count_once_per_use():
    matcher = GloveMatcher(prepare_corpus(pd.DataFrame({"text": ["a"]})), VOCAB)
    vector, known = matcher.document_vector("python python nurse")
    assert known == 3
    np.testing.assert_allclose(vector, (2 * VOCAB["python"] + VOCAB["nurse"]) / 3, rtol=1e-6)


def test_glove_idf_weighting_is_optional_and_down_weights_common_words():
    corpus = prepare_corpus(pd.DataFrame({"text": ["python developer", "nurse developer", "chef developer"]}))
    mean = GloveMatcher(corpus, VOCAB).fit()
    idf = GloveMatcher(corpus, VOCAB, weighting="idf").fit()
    assert idf.idf is not None and mean.idf is None
    index = idf.word_vectors.index
    assert idf.idf[index["developer"]] < idf.idf[index["python"]]  # in every resume, so less weight
    assert not np.allclose(mean.resume_matrix, idf.resume_matrix)
    with pytest.raises(ValueError):
        GloveMatcher(corpus, VOCAB, weighting="tfidf")


def big_vocabulary(n=2000, dim=20, seed=0):
    """A vocabulary large enough for the common direction, with a shared offset like real GloVe."""
    rng = np.random.default_rng(seed)
    shared = np.ones(dim, np.float32)
    words = [f"w{i}" for i in range(n)]
    return words, rng.normal(size=(n, dim)).astype(np.float32) * 0.5 + shared, shared


def test_common_direction_is_removed_for_real_sized_vocabularies():
    words, matrix, shared = big_vocabulary()
    vectors = GloveVectors(words, matrix)
    direction = vectors.common_direction()
    assert abs(direction @ (shared / np.linalg.norm(shared))) > 0.95  # finds the shared offset
    assert GloveVectors.from_mapping(FAKE_GLOVE).common_direction() is None  # toy vocabularies: skipped
    texts = [" ".join(words[i : i + 30]) for i in range(0, 600, 30)]
    corpus = corpus_from_files([(f"r{i}.txt", t) for i, t in enumerate(texts)])
    matcher = GloveMatcher(corpus, vectors).fit()
    plain = l2_normalize(matcher._averages(texts)[0]) @ l2_normalize(matcher._averages([texts[0]])[0]).ravel()
    spread = matcher.similarities(texts[0])
    assert plain.min() > 0.8  # averaged vectors all look alike...
    assert spread.std() > 3 * plain.std()  # ...until the shared direction is removed


# ---------------------------------------------------------------- chunking

LONG_RESUME = (
    "JANE DOE Data Engineer SUMMARY " + "Reliable engineer who enjoys teamwork. " * 120
    + "TECHNICAL SKILLS Python, Spark, Airflow, Kafka. WORK EXPERIENCE Built nurse scheduling for hospital patients. "
    + "EDUCATION B.Tech, Computer Science | 2018"
)


def test_sections_are_detected_in_flattened_text():
    sections = [name for name, _ in split_sections(LONG_RESUME)]
    assert sections[0] == "other" and {SECTION_SKILLS, SECTION_EXPERIENCE, SECTION_EDUCATION} <= set(sections)
    assert split_sections("") == []


def test_chunks_cover_the_text_and_respect_the_limit():
    chunks = chunk_text(LONG_RESUME, max_tokens=64)
    assert len(chunks) > 5
    assert all(int(len(c.split()) * 1.3) + 1 <= 64 + 10 for _, c in chunks)
    assert "".join(c for _, c in chunks).replace(" ", "") == LONG_RESUME.replace(" ", "")
    long_sentence = "word " * 500
    assert all(len(c.split()) <= 60 for _, c in chunk_text(long_sentence, max_tokens=64))


# ---------------------------------------------------------------- Sentence-BERT


class TruncatingModel(FakeSentenceModel):
    """Fake model that, like a real transformer, only reads the first ``max_seq_length`` words."""

    max_seq_length = 40

    def encode(self, texts, **kwargs):
        return super().encode([" ".join(t.split()[: self.max_seq_length]) for t in texts], **kwargs)


def test_long_resume_is_matched_on_its_later_sections():
    filler = "Reliable engineer who enjoys teamwork. " * 30
    corpus = corpus_from_files([
        ("late.txt", filler + "WORK EXPERIENCE Registered nurse caring for hospital patients."),
        ("none.txt", filler + "WORK EXPERIENCE Organised office events."),
    ])
    model = TruncatingModel()
    matcher = SbertMatcher(corpus, model, chunk_tokens=32).fit()
    document = matcher.document_similarities("nurse for hospital patients")
    assert np.isclose(document[0], document[1])  # the whole-resume embedding never sees the nursing part
    sims = matcher.similarities("nurse for hospital patients")
    assert sims[0] > sims[1]  # the best chunk does
    section, text = matcher.best_chunks("nurse for hospital patients")[0]
    assert section == SECTION_EXPERIENCE and "nurse" in text
    assert model.calls == 3  # fit: one batch for resumes + chunks; then one per query


def test_score_blends_whole_resume_and_best_chunk():
    corpus = corpus_from_files([("a.txt", "python developer. SKILLS python"), ("b.txt", "chef in a kitchen")])
    matcher = SbertMatcher(corpus, FakeSentenceModel(), doc_weight=0.25).fit()
    document, best, _ = matcher._parts("python")
    np.testing.assert_allclose(matcher.similarities("python"), 0.25 * document + 0.75 * best)
    with pytest.raises(ValueError):
        SbertMatcher(corpus, FakeSentenceModel(), doc_weight=1.5)


def test_all_sbert_embeddings_are_normalized():
    corpus = corpus_from_files([("a.txt", LONG_RESUME), ("b.txt", "Head chef")])
    matcher = SbertMatcher(corpus, FakeSentenceModel(), chunk_tokens=64).fit()
    np.testing.assert_allclose(np.linalg.norm(matcher.chunk_matrix, axis=1), 1.0, rtol=1e-5)
    np.testing.assert_allclose(np.linalg.norm(matcher.resume_matrix, axis=1), 1.0, rtol=1e-5)
    assert set(matcher.chunk_owner) == {0, 1}


def test_long_job_description_is_chunked_not_cut():
    corpus = corpus_from_files([("a.txt", "nurse"), ("b.txt", "chef")])
    model = TruncatingModel()
    matcher = SbertMatcher(corpus, model, chunk_tokens=32).fit()
    long_job = "Reliable team member. " * 40 + "Registered nurse for hospital patients."
    assert matcher.similarities(long_job)[0] > matcher.similarities(long_job)[1]


class FlakyDeviceModel(FakeSentenceModel):
    device = "mps"

    def __init__(self):
        super().__init__()
        self.moved_to = None

    def to(self, device):
        self.moved_to = device
        self.device = device

    def encode(self, texts, **kwargs):
        if self.device != "cpu":
            raise RuntimeError("MPS backend out of memory")
        return super().encode(texts, **kwargs)


def test_gpu_failure_falls_back_to_cpu():
    corpus = corpus_from_files([("a.txt", "nurse"), ("b.txt", "chef")])
    model = FlakyDeviceModel()
    matcher = SbertMatcher(corpus, model).fit()
    assert model.moved_to == "cpu" and matcher.is_fitted


def test_encoding_failure_is_reported_not_hidden():
    class Broken(FakeSentenceModel):
        def encode(self, texts, **kwargs):
            raise ValueError("bad input")

    with pytest.raises(MatcherError, match="failed to encode"):
        SbertMatcher(corpus_from_files([("a.txt", "nurse")]), Broken()).fit()


def test_model_loads_from_local_cache_when_offline(monkeypatch):
    calls = []

    class FakeST:
        def __init__(self, name, device=None, local_files_only=False):
            calls.append(local_files_only)
            if not local_files_only:
                raise OSError("no internet")
            self.device = "cpu"

    import sentence_transformers

    monkeypatch.setattr(matcher_module, "sbert_installed", lambda: True)
    monkeypatch.setattr(sentence_transformers, "SentenceTransformer", FakeST)
    assert isinstance(load_sbert_model("some/model"), FakeST) and calls == [False, True]

    class NeverST(FakeST):
        def __init__(self, *a, **k):
            raise OSError("no internet and no cache")

    monkeypatch.setattr(sentence_transformers, "SentenceTransformer", NeverST)
    with pytest.raises(MethodUnavailableError, match="needs internet access"):
        load_sbert_model("some/model")


# ---------------------------------------------------------------- Hybrid on one candidate set


def synthetic(n):
    topics = ["python developer databases", "nurse hospital patient", "chef kitchen", "teacher", "software developer"]
    return [(f"c{i:04d}.txt", f"{topics[i % 5]} candidate {i} " + "kitchen " * (i % 3)) for i in range(n)]


@pytest.mark.parametrize("n", [30, 50, 100, 250, 1000])
def test_every_method_ranks_every_resume_with_aligned_ids(n):
    corpus = corpus_from_files(synthetic(n))
    parts = TfidfMatcher(corpus), GloveMatcher(corpus, FAKE_GLOVE), SbertMatcher(corpus, FakeSentenceModel())
    hybrid = HybridMatcher(corpus, *parts).fit()
    results = hybrid.match("python developer", top_k=len(corpus))
    assert len(results) == n and results["resume_id"].is_unique and results["score"].is_monotonic_decreasing
    assert all(len(v) == n for v in hybrid.last_components.values())
    # The Hybrid score of each resume is computed from that same resume's part scores.
    position = {rid: i for i, rid in enumerate(corpus.df["resume_id"])}
    combined = hybrid.similarities("python developer")
    for rid, cosine in zip(results["resume_id"].head(20), results["cosine_similarity"].head(20)):
        assert np.isclose(round(combined[position[rid]], 4), cosine)


def test_duplicate_resumes_are_ranked_once_by_every_method():
    files = synthetic(10) + [("copy.txt", synthetic(10)[0][1])]
    corpus = corpus_from_files(files)
    hybrid = HybridMatcher(corpus, TfidfMatcher(corpus), GloveMatcher(corpus, FAKE_GLOVE), SbertMatcher(corpus, FakeSentenceModel())).fit()
    assert len(hybrid.match("python", top_k=50)) == 10


# ---------------------------------------------------------------- semantic evidence


def fake_encoder(words):
    """Bag-of-concepts encoder: texts sharing a concept get cosine 1; texts without one encode to zero."""
    concepts = {"rest": ["rest apis", "endpoints", "http"], "testing": ["unit testing", "pytest"]}

    def encode(texts):
        out = np.zeros((len(texts), len(concepts)), np.float32)
        for i, text in enumerate(texts):
            low = text.lower()
            hits = [j for j, words_ in enumerate(concepts.values()) if any(w in low for w in words_)]
            if hits:
                out[i, hits[0]] = 1.0
        return out

    return encode


def test_semantic_evidence_is_shown_but_never_counted():
    job = job_requirements("Required: REST APIs, Kubernetes and Docker.")
    resume = "Built backend endpoints with Flask for the app; contact dev@example.com. Managed payroll for the team."
    report = assess(job, resume)
    encode = fake_encoder(None)
    calls = []

    def counting(texts):
        calls.append(len(texts))
        return encode(texts)

    enriched = semantic_evidence(report, resume, counting, counting, threshold=0.9)
    status = {c.name: c for c in enriched.checks}
    assert status["REST APIs"].status == STATUS_SEMANTIC and "endpoints" in status["REST APIs"].evidence
    assert "dev@example.com" not in status["REST APIs"].evidence
    assert status["Kubernetes"].status == STATUS_MISSING
    assert enriched.score == report.score and "REST APIs" not in enriched.matched
    assert enriched.semantic == ("REST APIs",) and len(calls) == 2  # one query batch, one passage batch


def test_semantic_evidence_batch_keeps_each_resume_to_its_own_sentences():
    job = job_requirements("Required: REST APIs and Unit Testing.")
    resumes = ["Designed HTTP services for mobile apps.", "Wrote pytest suites for every module.", ""]
    reports = semantic_evidence_batch([assess(job, r) for r in resumes], resumes, fake_encoder(None), fake_encoder(None), 0.9)
    names = [{c.name: c.status for c in r.checks} for r in reports]
    assert names[0]["REST APIs"] == STATUS_SEMANTIC and names[0]["Unit Testing"] == STATUS_MISSING
    assert names[1]["Unit Testing"] == STATUS_SEMANTIC and names[1]["REST APIs"] == STATUS_MISSING
    assert all(s == STATUS_MISSING for s in names[2].values())


def test_products_contact_lines_and_fragments_are_never_semantic_evidence():
    job = job_requirements("Required: Python, Docker and REST APIs.")
    always_similar = lambda texts: np.ones((len(texts), 2), np.float32) / np.sqrt(2)  # noqa: E731
    resume = "Contact: jane@example.com. HTTP APIs. Head chef managing a kitchen team of eight."
    report = semantic_evidence(assess(job, resume), resume, always_similar, always_similar, threshold=0.5)
    status = {c.name: c for c in report.checks}
    assert status["Python"].status == STATUS_MISSING and status["Docker"].status == STATUS_MISSING  # products
    assert status["REST APIs"].status == STATUS_SEMANTIC  # a capability...
    assert status["REST APIs"].evidence == "Head chef managing a kitchen team of eight."  # ...but never the e-mail or a fragment


def test_resume_sentences_are_short_pieces():
    pieces = resume_sentences("One. Two | three • four " + "x " * 100)
    assert pieces[:3] == ["One.", "Two", "three"] and max(len(p.split()) for p in pieces) <= 40


def test_semantic_status_reaches_the_matrix():
    job = job_requirements("Required: REST APIs.")
    resume = "Built endpoints for the app."
    row = {"resume_id": "a", "rank": 1, "method": "sbert", "requirements": job.names,
           "checks": semantic_evidence(assess(job, resume), resume, fake_encoder(None), fake_encoder(None), 0.9).checks}
    matrix = status_matrix(pd.DataFrame([row]))
    assert matrix.loc["REST APIs", "a"] == "Semantically related (unverified)"


# ---------------------------------------------------------------- method comparison


def test_method_rankings_are_joined_by_resume_id():
    tfidf = pd.DataFrame({"resume_id": ["a", "b", "c"], "rank": [1, 2, 3], "score": [50.0, 20.0, 0.0]})
    sbert = pd.DataFrame({"resume_id": ["c", "a", "b"], "rank": [1, 2, 3], "score": [70.0, 60.0, 50.0]})
    table = combine_method_rankings({"tfidf": tfidf, "sbert": sbert}).set_index("resume_id")
    assert table.loc["c", "rank_tfidf"] == 3 and table.loc["c", "rank_sbert"] == 1 and table.loc["c", "rank_spread"] == 2
    agreement = ranking_agreement(table.reset_index(), ["tfidf", "sbert"], top=2)
    assert agreement.loc[0, "methods"] == "TF-IDF vs Sentence-BERT" and agreement.loc[0, "top_2_overlap"] == "1 of 2"


def test_chunking_can_be_switched_off():
    corpus = corpus_from_files([("a.txt", LONG_RESUME), ("b.txt", "Head chef")])
    model = FakeSentenceModel()
    matcher = SbertMatcher(corpus, model, chunk_tokens=0).fit()
    assert matcher.chunks == [] and matcher.chunk_matrix.shape[0] == 0
    np.testing.assert_allclose(matcher.similarities("nurse"), matcher.document_similarities("nurse"))
    assert matcher.best_chunks("nurse") == [None, None]
