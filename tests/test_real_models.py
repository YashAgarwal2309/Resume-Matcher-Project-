"""Checks against the real GloVe file and Sentence-BERT model. Skipped when they are not on this machine
(the Sentence-BERT model is only loaded from the local cache, never downloaded by the tests)."""

import numpy as np
import pytest

from resume_matcher.config import (
    SBERT_QUERY_PREFIXES,
    SEMANTIC_EVIDENCE_THRESHOLDS,
    get_settings,
)
from resume_matcher.matcher import load_glove_vectors

# Real lines from the demo resumes, labelled by hand for job requirements they do not name: 1 = genuine
# evidence for the requirement, 0 = not. These fix the semantic-evidence threshold in config.py (the
# closest unrelated lines are kept; 55 unrelated and 11 related pairs were reviewed).
REAL_PAIRS = [
    ("Create process flow diagrams", "Created as-is and to-be process maps for the attendance regularisation flow.", 1),
    ("deploying models as APIs", "Deployed two models as Flask APIs in Docker with help from the backend team.", 1),
    ("deploying models as APIs", "Package models with FastAPI and Docker for the backend team.", 1),
    ("Word embeddings", "Built an FAQ search using sentence embeddings and cosine similarity.", 1),
    ("REST API development", "Developed backend endpoints using Flask.", 1),
    ("deploying models as APIs", "Develop and maintain the shipment tracking APIs (Django REST Framework) used by our mobile app and around 300 warehouse partners.", 0),
    ("deploying models as APIs", "Good with REST APIs, Hibernate, MySQL and writing proper unit tests.", 0),
    ("deploying models as APIs", "Worked with data engineering to move training data from ad-hoc SQL extracts to versioned tables in Snowflake.", 0),
    ("Create process flow diagrams", "Track experiments in MLflow and write a short evaluation note (confusion matrix, error examples) for every model.", 0),
    ("Support user acceptance testing", "Good with REST APIs, Hibernate, MySQL and writing proper unit tests.", 0),
    ("Machine Learning", "Solved 400+ problems on LeetCode and HackerRank in Python.", 0),
    ("Feature Engineering", "Built Power BI dashboards for product managers showing feature usage and ticket trends.", 0),
    ("REST APIs", "Wrote SQL queries to pull claims data from MySQL.", 0),
]

# Clean, hand-written pairs (an earlier calibration set): unrelated ones must stay below the threshold.
LABELLED_PAIRS = [
    ("REST API development", "Developed backend endpoints using Flask.", 1),
    ("REST APIs", "Designed HTTP services consumed by the mobile app.", 1),
    ("Machine Learning", "Built a churn prediction model with gradient boosting.", 1),
    ("Docker", "Containerised the services and deployed them on ECS.", 1),
    ("Kubernetes", "Managed container orchestration on EKS clusters.", 1),
    ("Data Visualization", "Created Tableau dashboards for the sales team.", 1),
    ("Unit Testing", "Wrote pytest test cases for every module.", 1),
    ("NLP", "Fine-tuned DistilBERT for intent classification of chat messages.", 1),
    ("Statistics", "Ran A/B tests and hypothesis tests on campaign results.", 1),
    ("CI/CD", "Set up Jenkins pipelines that build and deploy on every merge.", 1),
    ("REST API development", "Managed payroll for 300 employees.", 0),
    ("Machine Learning", "Handled recruitment for sales roles.", 0),
    ("Docker", "Head chef managing a kitchen team of eight.", 0),
    ("Kubernetes", "Wrote BRDs and user stories with stakeholders.", 0),
    ("Data Visualization", "Maintained Java Spring Boot microservices.", 0),
    ("Unit Testing", "Organised the annual company offsite.", 0),
    ("NLP", "Designed the PostgreSQL schema for the returns module.", 0),
    ("Statistics", "Built React components for the checkout page.", 0),
    ("CI/CD", "Negotiated vendor contracts for office supplies.", 0),
    ("Python", "Developed Java applications with Spring.", 0),
    ("Django", "Built REST APIs with Flask.", 0),
]


@pytest.fixture(scope="module")
def sbert():
    pytest.importorskip("sentence_transformers")
    from sentence_transformers import SentenceTransformer

    name = get_settings().sbert_model_name
    try:
        return name, SentenceTransformer(name, local_files_only=True)
    except Exception:  # noqa: BLE001 - not cached, offline, ...: skip
        pytest.skip(f"{name} is not in the local model cache")


def similarities(model, name, pairs):
    prefix = SBERT_QUERY_PREFIXES.get(name, "")
    queries = model.encode([prefix + r for r, _, _ in pairs], normalize_embeddings=True)
    passages = model.encode([s for _, s, _ in pairs], normalize_embeddings=True)
    return (queries * passages).sum(axis=1), np.array([label for *_, label in pairs])


def test_semantic_threshold_has_no_false_positives_on_real_resume_lines(sbert):
    name, model = sbert
    threshold = SEMANTIC_EVIDENCE_THRESHOLDS[name]
    sims, labels = similarities(model, name, REAL_PAIRS)
    assert (sims[labels == 0] < threshold).all()  # the closest unrelated lines are never shown as evidence
    assert (sims[labels == 1] >= threshold).sum() >= 4  # clear evidence is found (all 5 when calibrated)
    clean, clean_labels = similarities(model, name, LABELLED_PAIRS)
    assert (clean[clean_labels == 0] < threshold).all()


def test_model_uses_cls_pooling_normalization_and_512_pieces(sbert):
    _, model = sbert
    modules = [type(m).__name__ for m in model]
    assert modules == ["Transformer", "Pooling", "Normalize"] and model.max_seq_length == 512
    assert model[1].get_config_dict()["pooling_mode"] == "cls"


def test_real_glove_file_is_complete():
    path = get_settings().glove_path
    if not path.is_file():
        pytest.skip("GloVe file not downloaded")
    vectors = load_glove_vectors(path)
    assert vectors.dim == 100 and len(vectors) == 400_000 and vectors.matrix.dtype == np.float32
    assert "python" in vectors and vectors.common_direction() is not None
