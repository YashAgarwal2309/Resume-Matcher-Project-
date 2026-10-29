import json

import pandas as pd
import pytest

import evaluate
from resume_matcher import config


def test_precision_at_k():
    assert evaluate.precision_at_k(["A", "B", "A", "C"], {"A"}) == 0.5
    assert evaluate.precision_at_k(["A", "B"], {"A", "B"}) == 1.0
    assert evaluate.precision_at_k([], {"A"}) == 0.0


@pytest.fixture
def sample_jobs(tmp_path, monkeypatch):
    path = tmp_path / "jobs.json"
    path.write_text(json.dumps({"jobs": [
        {"title": "Nurse", "relevant_categories": ["HEALTHCARE"], "text": "nurse patient care hospital"},
        {"title": "Mobile", "relevant_categories": [], "text": "android developer java"},
    ]}))
    monkeypatch.setattr(evaluate, "SAMPLE_JOBS_PATH", path)


def test_evaluate_scores_only_labelled_jobs(resume_csv, sample_jobs):
    results = evaluate.evaluate(["tfidf"], top_k=2, dataset_path=resume_csv)
    nurse = results[results["job"] == "Nurse"].iloc[0]
    mobile = results[results["job"] == "Mobile"].iloc[0]
    assert nurse["top1_category"] == "HEALTHCARE" and nurse["top1_relevant"] is True
    assert nurse["precision_at_2"] == 0.5
    assert pd.isna(mobile["precision_at_2"]) and pd.isna(mobile["top1_relevant"])

    summary = evaluate.summarize(results, 2)
    row = summary.iloc[0]
    assert (row["jobs"], row["labelled_jobs"], row["top1_relevant"], row["mean_precision_at_k"]) == (2, 1, 1, 0.5)


def test_evaluate_reports_unavailable_method(resume_csv, sample_jobs, monkeypatch, tmp_path):
    monkeypatch.setenv("GLOVE_PATH", str(tmp_path / "missing.txt"))
    results = evaluate.evaluate(["glove"], top_k=2, dataset_path=resume_csv)
    assert results["status"].iloc[0].startswith("unavailable")


def test_available_datasets(monkeypatch, tmp_path):
    a, b = tmp_path / "a.csv", tmp_path / "b.csv"
    b.write_text("x")
    monkeypatch.delenv("RESUME_DATA_PATH", raising=False)
    monkeypatch.setattr(config, "DATASET_OPTIONS", {"A": a, "B": b})
    monkeypatch.setattr(config, "LEGACY_DATASET_PATH", tmp_path / "legacy.csv")
    assert config.available_datasets() == {"B": b}  # missing files are not offered

    b.unlink()
    assert config.available_datasets() == {}

    monkeypatch.setenv("RESUME_DATA_PATH", str(a))
    assert config.available_datasets() == {"Custom (a.csv)": a}
