"""TF-IDF preprocessing, scoring and scale tests (all resumes synthetic)."""

import random
import time

import numpy as np
import pandas as pd
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer

from resume_matcher.comparison import build_candidates
from resume_matcher.config import METHOD_TFIDF
from resume_matcher.data_loader import corpus_from_files, prepare_corpus
from resume_matcher.matcher import EmptyQueryError, TfidfMatcher
from resume_matcher.skills import extract_requirements
from resume_matcher.text_features import (
    canonicalize_skills,
    prepare_tfidf_text,
    tfidf_tokens,
)

JOB = (
    "Backend Engineer. Required: Python, Django, PostgreSQL, Docker, Kubernetes and AWS. "
    "Experience with REST APIs and CI/CD. Bachelor's degree in Computer Science. 3+ years of experience."
)

SKILL_POOL = [
    "Python", "Django", "Flask", "PostgreSQL", "MySQL", "Docker", "Kubernetes", "AWS", "Azure", "Java",
    "Spring Boot", "React", "JavaScript", "SQL", "pandas", "Tableau", "Excel", "Linux", "Jenkins", "Git",
]
OTHER_WORK = [
    "Registered nurse providing patient care on a busy hospital ward.",
    "Head chef managing a kitchen team, menu planning and food safety.",
    "Primary school teacher planning lessons and managing a classroom.",
    "Sales executive handling key accounts and quarterly targets.",
]


def synthetic_resumes(n: int, seed: int = 7) -> list[tuple[str, str]]:
    """``n`` distinct resumes: tech profiles with random skill sets, plus some unrelated jobs."""
    rng = random.Random(seed)
    files = []
    for i in range(n):
        if i % 5 == 4:
            text = f"{OTHER_WORK[i % len(OTHER_WORK)]} Employee number {i}."
        else:
            skills = rng.sample(SKILL_POOL, k=rng.randint(2, 8))
            text = (
                f"Software engineer {i} with {rng.randint(1, 9)} years of experience. "
                f"Skills: {', '.join(skills)}. Built services and dashboards."
            )
        files.append((f"candidate_{i:04d}.txt", text))
    return files


def reference_similarities(texts: list[str], job: str) -> np.ndarray:
    """Plain scikit-learn: one TfidfVectorizer fitted on all resumes plus the job description."""
    vectorizer = TfidfVectorizer(tokenizer=tfidf_tokens, token_pattern=None, lowercase=False, sublinear_tf=True)
    matrix = vectorizer.fit_transform([prepare_tfidf_text(t) for t in texts] + [prepare_tfidf_text(job)])
    return (matrix[:-1] @ matrix[-1].T).toarray().ravel()


# ---- preprocessing


def test_aliases_and_spellings_become_one_token():
    tokens = tfidf_tokens(prepare_tfidf_text(
        "k8s and Kubernetes; sklearn and Scikit-Learn; Node JS and node.js; postgres; ML; golang; CI/CD"
    ))
    assert tokens.count("kubernetes") == 2 and tokens.count("scikit_learn") == 2 and tokens.count("node.js") == 2
    assert {"postgresql", "machine_learning", "golang", "ci_cd"} <= set(tokens)


def test_technical_terms_survive_cleaning():
    tokens = tfidf_tokens(prepare_tfidf_text(
        "C++, C#, .NET, front-end, back-end, full-stack, IT support, HTML5, S3 and EC2, Power BI, A/B testing"
    ))
    for term in ("c++", "c#", ".net", "front", "back", "full", "information_technology", "s3", "ec2",
                 "power_bi", "a_b_testing", "html"):
        assert term in tokens, term


def test_related_technologies_are_not_merged():
    tokens = tfidf_tokens(prepare_tfidf_text("Flask, Django, MySQL, PostgreSQL, React, Angular"))
    assert {"flask", "django", "mysql", "postgresql", "react", "angular"} <= set(tokens)


def test_noise_is_removed():
    tokens = tfidf_tokens(prepare_tfidf_text("Contact me at dev@example.com or https://x.io. Born 1990, +91 98765 43210."))
    assert not any(c.isdigit() for t in tokens for c in t if t.isdigit())
    assert "dev" not in tokens and "example.com" not in tokens and "x.io" not in tokens
    assert all(any(ch.isalpha() for ch in t) for t in tokens)


def test_short_words_are_not_mistaken_for_skills():
    assert "r_lang" not in canonicalize_skills("R&D team") and "golang" not in canonicalize_skills("good to go")
    assert "information_technology" not in prepare_tfidf_text("it was good")


def test_preprocessing_handles_empty_input():
    assert prepare_tfidf_text(None) == "" and prepare_tfidf_text("") == "" and tfidf_tokens("") == []


# ---- scoring


def test_scores_equal_a_vectorizer_fitted_on_resumes_plus_job():
    files = synthetic_resumes(60)
    corpus = corpus_from_files(files)
    matcher = TfidfMatcher(corpus).fit()
    expected = reference_similarities(corpus.df["text"].tolist(), JOB)
    assert np.allclose(matcher.similarities(JOB), expected, atol=1e-12)


def test_job_words_missing_from_all_resumes_lower_scores_but_not_the_order():
    corpus = corpus_from_files(synthetic_resumes(40))
    matcher = TfidfMatcher(corpus).fit()
    base = matcher.match(JOB, top_k=40)
    padded = matcher.match(JOB + " Quasar zephyrine blorptastic synergies.", top_k=40)
    assert padded["resume_id"].tolist() == base["resume_id"].tolist()
    before, after = matcher.similarities(JOB), matcher.similarities(JOB + " Quasar zephyrine blorptastic synergies.")
    ratio = after[before > 0] / before[before > 0]
    assert np.allclose(ratio, ratio[0]) and ratio[0] < 1  # every score shrinks by the same factor: no inflation


def test_alias_wording_still_matches():
    corpus = corpus_from_files([
        ("aliases.txt", "Deployed services on k8s with postgres; models in sklearn."),
        ("unrelated.txt", "Head chef running a kitchen."),
    ])
    results = TfidfMatcher(corpus).match("Kubernetes, PostgreSQL and scikit-learn", top_k=2)
    assert results["resume_id"].iloc[0] == "aliases.txt" and results["score"].iloc[0] > 0
    assert results["score"].iloc[1] == 0


def test_exact_keywords_rank_above_low_similarity_resumes():
    corpus = corpus_from_files([
        ("low.txt", "Sales executive handling key accounts."),
        ("exact.txt", "Python Django PostgreSQL Docker Kubernetes AWS engineer."),
        ("partial.txt", "Python developer using Flask and MySQL."),
    ])
    results = TfidfMatcher(corpus).match(JOB, top_k=3)
    assert results["resume_id"].tolist() == ["exact.txt", "partial.txt", "low.txt"]
    assert results["score"].is_monotonic_decreasing
    assert results["score"].iloc[-1] == 0  # very low similarity is reported as such, not dropped


def test_repeating_a_keyword_does_not_beat_broader_evidence():
    corpus = corpus_from_files([
        ("stuffed.txt", "Python " * 40 + "developer"),
        ("broad.txt", "Python, Django, PostgreSQL, Docker, Kubernetes and AWS developer"),
    ])
    assert TfidfMatcher(corpus).match(JOB, top_k=1)["resume_id"].iloc[0] == "broad.txt"


def test_job_without_known_words_raises():
    matcher = TfidfMatcher(corpus_from_files(synthetic_resumes(10))).fit()
    with pytest.raises(EmptyQueryError):
        matcher.match("zzqx blorp", top_k=3)
    with pytest.raises(EmptyQueryError):
        matcher.match("2019 2020 the of", top_k=3)  # numbers and stop words only


def test_resume_with_only_numbers_scores_zero_without_crashing():
    corpus = prepare_corpus(pd.DataFrame({"ID": ["nums", "py"], "Resume": ["2015 2016 2017", "Python developer"]}))
    results = TfidfMatcher(corpus).match("Python", top_k=2)
    assert results.set_index("resume_id").loc["nums", "score"] == 0


# ---- every resume is ranked, at any size


@pytest.mark.parametrize("n", [4, 30, 50, 100, 250, 1000])
def test_every_resume_is_processed_and_ranked(n):
    files = synthetic_resumes(n)
    corpus = corpus_from_files(files)
    assert len(corpus) == n
    start = time.perf_counter()
    results = TfidfMatcher(corpus).match(JOB, top_k=len(corpus))
    candidates = build_candidates(results, corpus, extract_requirements(JOB), JOB, METHOD_TFIDF)
    elapsed = time.perf_counter() - start
    assert len(candidates) == n and candidates["rank"].tolist() == list(range(1, n + 1))
    assert candidates["resume_id"].is_unique and set(candidates["resume_id"]) == {name for name, _ in files}
    assert candidates["score"].is_monotonic_decreasing
    # Scores stay attached to the right file: recompute them directly.
    expected = dict(zip(corpus.df["resume_id"], np.round(np.clip(reference_similarities(corpus.df["text"].tolist(), JOB), 0, 1) * 100, 2)))
    assert all(expected[r] == s for r, s in zip(candidates["resume_id"], candidates["score"]))
    assert elapsed < 60  # generous bound; see README for measured timings


def test_empty_and_duplicate_resumes_are_counted_not_ranked():
    files = synthetic_resumes(20)
    files += [("empty.txt", "   "), ("only-punctuation.txt", "--- ... ---"), ("copy.txt", files[0][1])]
    corpus = corpus_from_files(files)
    assert len(corpus) == 20
    assert corpus.stats["dropped_empty"] == 2 and corpus.stats["dropped_duplicates"] == 1
    results = TfidfMatcher(corpus).match(JOB, top_k=len(corpus))
    assert len(results) == 20 and "copy.txt" not in set(results["resume_id"])
