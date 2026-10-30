import gzip
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import resume_matcher.matcher as matcher_module
from resume_matcher.config import (
    METHOD_GLOVE,
    METHOD_HYBRID,
    METHOD_SBERT,
    METHOD_TFIDF,
    Settings,
)
from resume_matcher.data_loader import prepare_corpus
from resume_matcher.matcher import (
    RESULT_COLUMNS,
    EmptyQueryError,
    GloveMatcher,
    HybridMatcher,
    MatcherError,
    MethodUnavailableError,
    SbertMatcher,
    TfidfMatcher,
    create_matcher,
    load_glove_vectors,
    method_status,
    minmax_normalize,
    similarity_to_score,
)
from tests.conftest import FAKE_GLOVE, FakeSentenceModel


@pytest.fixture
def all_matchers(corpus, tmp_path):
    """One fitted matcher per method, using offline fakes for GloVe and SBERT."""
    return {
        METHOD_TFIDF: TfidfMatcher(corpus).fit(),
        METHOD_GLOVE: GloveMatcher(corpus, FAKE_GLOVE).fit(),
        METHOD_SBERT: SbertMatcher(corpus, FakeSentenceModel(), cache_dir=tmp_path).fit(),
        METHOD_HYBRID: HybridMatcher(
            corpus, TfidfMatcher(corpus), GloveMatcher(corpus, FAKE_GLOVE), SbertMatcher(corpus, FakeSentenceModel())
        ).fit(),
    }


JOB = "Looking for a Python software developer with databases experience"


# ---------------------------------------------------------------- shared behaviour


@pytest.mark.parametrize("method", [METHOD_TFIDF, METHOD_GLOVE, METHOD_SBERT, METHOD_HYBRID])
def test_consistent_output_format(all_matchers, method):
    results = all_matchers[method].match(JOB, top_k=3)
    assert list(results.columns) == RESULT_COLUMNS
    assert results["rank"].tolist() == [1, 2, 3]
    assert results["resume_id"].is_unique


@pytest.mark.parametrize("method", [METHOD_TFIDF, METHOD_GLOVE, METHOD_SBERT, METHOD_HYBRID])
def test_scores_bounded_and_descending(all_matchers, method):
    results = all_matchers[method].match(JOB, top_k=5)
    assert results["score"].between(0, 100).all()
    assert results["cosine_similarity"].between(-1.0001, 1.0001).all()
    assert results["cosine_similarity"].is_monotonic_decreasing
    assert results["score"].is_monotonic_decreasing


@pytest.mark.parametrize("method", [METHOD_TFIDF, METHOD_GLOVE, METHOD_SBERT, METHOD_HYBRID])
def test_relevant_resume_ranked_first(all_matchers, method):
    results = all_matchers[method].match(JOB, top_k=2)
    assert results["category"].iloc[0] == "INFORMATION-TECHNOLOGY"
    nurse = all_matchers[method].match("nurse for patient care in hospital", top_k=1)
    assert nurse["resume_id"].iloc[0] == "102"


@pytest.mark.parametrize("method", [METHOD_TFIDF, METHOD_GLOVE, METHOD_SBERT, METHOD_HYBRID])
@pytest.mark.parametrize("bad", ["", "   ", None, 123])
def test_empty_job_description(all_matchers, method, bad):
    with pytest.raises(EmptyQueryError):
        all_matchers[method].match(bad, top_k=3)


@pytest.mark.parametrize("top_k, expected", [(1, 1), (3, 3), (5, 5), (100, 5)])
def test_top_k_limits(all_matchers, top_k, expected):
    assert len(all_matchers[METHOD_TFIDF].match(JOB, top_k=top_k)) == expected


@pytest.mark.parametrize("top_k", [0, -1, 2.5, "3"])
def test_invalid_top_k(all_matchers, top_k):
    with pytest.raises(ValueError):
        all_matchers[METHOD_TFIDF].match(JOB, top_k=top_k)


def test_previews_mask_contact_details(all_matchers):
    results = all_matchers[METHOD_TFIDF].match("python django developer", top_k=1)
    assert results["resume_id"].iloc[0] == "101"
    assert "dev@example.com" not in results["preview"].iloc[0]
    assert "[email]" in results["preview"].iloc[0]


def test_missing_metadata_does_not_crash():
    corpus = prepare_corpus(pd.DataFrame({"text": ["python developer", "chef in kitchen", "nurse"]}))
    results = TfidfMatcher(corpus).match("python", top_k=2)
    assert results["resume_id"].iloc[0] == "row-0"
    assert results["category"].isna().all()


def test_similarity_to_score_clips_negative_values():
    assert similarity_to_score(np.array([-0.5, 0.0, 0.1234, 1.0])).tolist() == [0.0, 0.0, 12.34, 100.0]


def test_ties_keep_dataset_order():
    corpus = prepare_corpus(pd.DataFrame({"text": ["java developer", "java engineer", "chef cook"]}))
    results = TfidfMatcher(corpus).match("java", top_k=3)
    assert results["cosine_similarity"].iloc[0] == results["cosine_similarity"].iloc[1]
    assert results["resume_id"].tolist()[:2] == ["row-0", "row-1"]


# ---------------------------------------------------------------- TF-IDF


def test_tfidf_unknown_words_raise(all_matchers):
    with pytest.raises(EmptyQueryError):
        all_matchers[METHOD_TFIDF].match("zzzqqq xxyyzz", top_k=3)
    with pytest.raises(EmptyQueryError):
        all_matchers[METHOD_TFIDF].match("the and of", top_k=3)  # only stop words


def test_tfidf_empty_vocabulary():
    corpus = prepare_corpus(pd.DataFrame({"text": ["the and of", "is it the"]}))
    with pytest.raises(MatcherError, match="vocabulary"):
        TfidfMatcher(corpus).fit()


def test_tfidf_keeps_programming_language_tokens(corpus):
    m = TfidfMatcher(corpus).fit()
    assert "c++" in m.vectorizer.vocabulary_
    assert m.match("C++ engineer", top_k=1)["resume_id"].iloc[0] == "104"


# ---------------------------------------------------------------- GloVe


def test_glove_oov_query_raises(all_matchers):
    with pytest.raises(EmptyQueryError, match="GloVe vocabulary"):
        all_matchers[METHOD_GLOVE].match("zzzqqq", top_k=2)


def test_glove_resume_without_known_words_scores_zero():
    corpus = prepare_corpus(pd.DataFrame({"text": ["python developer", "qwerty asdfgh"]}))
    m = GloveMatcher(corpus, FAKE_GLOVE).fit()
    assert m.zero_vector_resumes == 1
    results = m.match("python", top_k=2)
    assert results["resume_id"].tolist() == ["row-0", "row-1"]
    assert results["cosine_similarity"].iloc[1] == 0


def test_glove_document_vector_is_average(corpus):
    m = GloveMatcher(corpus, FAKE_GLOVE)
    vector, known = m.document_vector("Python and nurse unknownword")
    assert known == 2
    np.testing.assert_allclose(vector, [0.5, 0.5, 0.0])


def test_load_glove_text_and_gzip_formats(tmp_path):
    txt = tmp_path / "glove.txt"
    txt.write_text("python 0.1 0.2 0.3\nnurse 0.4 0.5 0.6\nbroken 0.1\n")
    vectors = load_glove_vectors(txt)
    assert set(vectors) == {"python", "nurse"}
    np.testing.assert_allclose(vectors["nurse"], [0.4, 0.5, 0.6])

    gz = tmp_path / "glove.gz"
    with gzip.open(gz, "wt") as handle:
        handle.write("2 3\npython 0.1 0.2 0.3\nnurse 0.4 0.5 0.6\n")  # word2vec header line
    assert set(load_glove_vectors(gz)) == {"python", "nurse"}


def test_glove_missing_file_is_unavailable(tmp_path, corpus):
    with pytest.raises(MethodUnavailableError, match="not found"):
        load_glove_vectors(tmp_path / "missing.txt")
    settings = replace(Settings(), glove_path=tmp_path / "missing.txt")
    assert method_status(METHOD_GLOVE, settings)[0] is False
    with pytest.raises(MethodUnavailableError):
        create_matcher(METHOD_GLOVE, corpus, settings)


def test_create_glove_matcher_from_file(tmp_path, corpus):
    path = tmp_path / "glove.txt"
    path.write_text("\n".join(f"{w} " + " ".join(map(str, v)) for w, v in FAKE_GLOVE.items()))
    settings = replace(Settings(), glove_path=path)
    assert method_status(METHOD_GLOVE, settings)[0] is True
    results = create_matcher(METHOD_GLOVE, corpus, settings).match("hospital nurse", top_k=1)
    assert results["resume_id"].iloc[0] == "102"


def test_create_glove_matcher_with_preloaded_vectors(corpus):
    settings = replace(Settings(), glove_path=Path("does-not-exist.txt"))  # the file is never read
    results = create_matcher(METHOD_GLOVE, corpus, settings, word_vectors=FAKE_GLOVE).match("hospital nurse", top_k=1)
    assert results["resume_id"].iloc[0] == "102"


def test_create_sbert_matcher_with_preloaded_model(corpus, tmp_path):
    settings = replace(Settings(), cache_dir=tmp_path)
    results = create_matcher(METHOD_SBERT, corpus, settings, sbert_model=FakeSentenceModel()).match("chef", top_k=1)
    assert results["resume_id"].iloc[0] == "103"


# ---------------------------------------------------------------- Sentence-BERT


def test_sbert_package_missing(monkeypatch, corpus):
    monkeypatch.setattr(matcher_module, "sbert_installed", lambda: False)
    assert method_status(METHOD_SBERT, Settings())[0] is False
    with pytest.raises(MethodUnavailableError, match="not installed"):
        create_matcher(METHOD_SBERT, corpus, Settings())


def test_sbert_model_download_failure(monkeypatch, corpus):
    class FailingModel:
        def __init__(self, *args, **kwargs):
            raise OSError("network unreachable")

    fake_module = type("FakeModule", (), {"SentenceTransformer": FailingModel})
    monkeypatch.setattr(matcher_module, "sbert_installed", lambda: True)
    monkeypatch.setitem(__import__("sys").modules, "sentence_transformers", fake_module)
    with pytest.raises(MethodUnavailableError, match="network unreachable"):
        create_matcher(METHOD_SBERT, corpus, Settings())


def test_sbert_embeddings_are_cached(corpus, tmp_path):
    first = SbertMatcher(corpus, FakeSentenceModel(), model_name="fake", cache_dir=tmp_path).fit()
    assert not first.loaded_from_cache
    assert len(list(tmp_path.glob("sbert_fake_*.npy"))) == 1

    model = FakeSentenceModel()
    second = SbertMatcher(corpus, model, model_name="fake", cache_dir=tmp_path).fit()
    assert second.loaded_from_cache
    assert model.calls == 0  # resumes were not re-encoded
    np.testing.assert_allclose(first.resume_matrix, second.resume_matrix)


def test_sbert_resume_embeddings_computed_once_per_matcher(corpus):
    model = FakeSentenceModel()
    m = SbertMatcher(corpus, model).fit()
    m.match(JOB, top_k=2)
    m.match("chef", top_k=2)
    assert model.calls == 3  # 1 for all resumes + 1 per query


def test_sbert_embeddings_normalized(all_matchers):
    norms = np.linalg.norm(all_matchers[METHOD_SBERT].resume_matrix, axis=1)
    np.testing.assert_allclose(norms, 1.0, rtol=1e-5)


class RecordingModel(FakeSentenceModel):
    """FakeSentenceModel that remembers every text it was asked to encode."""

    def __init__(self):
        super().__init__()
        self.seen = []

    def encode(self, texts, **kwargs):
        self.seen.extend(texts)
        return super().encode(texts, **kwargs)


def test_sbert_query_prefix_is_added_to_the_job_description_only(corpus):
    model = RecordingModel()
    matcher = SbertMatcher(corpus, model, query_prefix="search: ").fit()
    matcher.match("hospital nurse", top_k=1)
    assert model.seen[-1] == "search: hospital nurse"
    assert not any(text.startswith("search: ") for text in model.seen[:-1])  # resumes have no prefix


def test_default_model_is_bge_with_its_query_instruction(corpus, tmp_path):
    settings = replace(Settings(), cache_dir=tmp_path)
    assert settings.sbert_model_name == "BAAI/bge-small-en-v1.5"
    matcher = create_matcher(METHOD_SBERT, corpus, settings, sbert_model=FakeSentenceModel())
    assert matcher.query_prefix.startswith("Represent this sentence")


# ---------------------------------------------------------------- Hybrid


def make_hybrid(corpus):
    return HybridMatcher(
        corpus, TfidfMatcher(corpus), GloveMatcher(corpus, FAKE_GLOVE), SbertMatcher(corpus, FakeSentenceModel())
    ).fit()


def test_hybrid_is_equal_average_of_the_normalized_methods(corpus):
    hybrid = make_hybrid(corpus)
    tfidf, glove, sbert = hybrid.parts
    parts = [tfidf.similarities(JOB), glove.similarities(JOB), sbert.similarities(JOB)]
    expected = sum(minmax_normalize(p) for p in parts) / 3
    np.testing.assert_allclose(hybrid.similarities(JOB), expected, rtol=1e-6)
    assert hybrid.last_used == ("tfidf", "glove", "sbert") and not hybrid.last_skipped
    np.testing.assert_allclose(hybrid.last_components["glove"], parts[1])  # raw cosines kept for display


def test_hybrid_without_normalization_is_the_plain_weighted_average(corpus):
    tfidf, glove, sbert = make_hybrid(corpus).parts
    hybrid = HybridMatcher(corpus, tfidf, glove, sbert, weights=(2, 1, 1), normalization="none").fit()
    expected = 0.5 * tfidf.similarities(JOB) + 0.25 * glove.similarities(JOB) + 0.25 * sbert.similarities(JOB)
    np.testing.assert_allclose(hybrid.similarities(JOB), expected, rtol=1e-6)


def test_hybrid_weights_are_validated_and_rescaled(corpus):
    tfidf, glove, sbert = make_hybrid(corpus).parts
    assert HybridMatcher(corpus, tfidf, glove, sbert, weights=(1, 0, 1)).weights == (0.5, 0.0, 0.5)
    for bad in [(0, 0, 0), (1, -1, 1), (1, 2), (float("nan"), 1, 1)]:
        with pytest.raises(ValueError):
            HybridMatcher(corpus, tfidf, glove, sbert, weights=bad)
    with pytest.raises(ValueError):
        HybridMatcher(corpus, tfidf, glove, sbert, normalization="zscore")


def test_hybrid_with_zero_weight_ignores_that_method(corpus):
    tfidf, glove, sbert = make_hybrid(corpus).parts
    hybrid = HybridMatcher(corpus, tfidf, glove, sbert, weights=(0, 0, 1)).fit()
    np.testing.assert_allclose(hybrid.similarities(JOB), minmax_normalize(sbert.similarities(JOB)))
    assert hybrid.last_used == ("sbert",)


def test_hybrid_reports_methods_that_could_not_read_the_query(corpus):
    hybrid = make_hybrid(corpus)
    tfidf, glove, sbert = hybrid.parts
    query = "astronaut"  # in no resume (TF-IDF) and not in the fake GloVe table, so both raise EmptyQueryError
    for part in (tfidf, glove):
        with pytest.raises(EmptyQueryError):
            part.similarities(query)
    np.testing.assert_allclose(hybrid.similarities(query), minmax_normalize(sbert.similarities(query)))
    assert hybrid.last_used == ("sbert",) and set(hybrid.last_skipped) == {"tfidf", "glove"}
    assert len(hybrid.match(query, top_k=2)) == 2


def test_minmax_normalize_handles_constant_scores():
    np.testing.assert_allclose(minmax_normalize(np.array([0.2, 0.6, 0.4])), [0.0, 1.0, 0.5])
    np.testing.assert_allclose(minmax_normalize(np.array([0.7])), [0.7])  # one resume: nothing to rescale
    np.testing.assert_allclose(minmax_normalize(np.array([1.5, 1.5])), [1.0, 1.0])


def test_create_hybrid_matcher_and_status(monkeypatch, corpus, tmp_path):
    settings = replace(Settings(), cache_dir=tmp_path, glove_path=tmp_path / "missing.txt")
    matcher = create_matcher(METHOD_HYBRID, corpus, settings, word_vectors=FAKE_GLOVE, sbert_model=FakeSentenceModel())
    assert isinstance(matcher, HybridMatcher) and matcher.is_fitted
    assert [type(p) for p in matcher.parts] == [TfidfMatcher, GloveMatcher, SbertMatcher]

    ok, message = method_status(METHOD_HYBRID, settings)  # GloVe file is missing
    assert ok is False and "GloVe" in message
    monkeypatch.setattr(matcher_module, "sbert_installed", lambda: False)
    assert "Sentence-BERT" in method_status(METHOD_HYBRID, settings)[1]
    with pytest.raises(MethodUnavailableError):
        create_matcher(METHOD_HYBRID, corpus, settings, sbert_model=FakeSentenceModel())  # no GloVe vectors


def test_unknown_method(corpus):
    with pytest.raises(ValueError):
        create_matcher("word2vec", corpus, Settings())


def test_empty_corpus_rejected(corpus):
    corpus.df = corpus.df.iloc[0:0]
    with pytest.raises(MatcherError):
        TfidfMatcher(corpus)
