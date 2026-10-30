"""Compare the matching methods on the same resumes and sample job descriptions.

Usage:
    python evaluate.py                      # every available dataset, all methods, top-5
    python evaluate.py --methods tfidf sbert --top-k 10
    python evaluate.py --data data/UpdatedResumeDataSet.csv

What it reports, per dataset, method and job description:
    * time to build the resume index and to answer one query
    * the category of the top-ranked resume
    * Precision@K, Recall@K and NDCG@K, using the resume's dataset category as a *proxy* relevance
      label (a resume counts as relevant if its category is in the job's relevant_categories)
    * graded NDCG@5 on the seven demo resumes, whose relevance was labelled by hand
      (demo_samples/relevance_labels.json; 3 = strong fit ... 0 = unrelated)

Precision@K here is NOT a human-judged accuracy: the datasets have no relevance
judgments between resumes and these job descriptions. Category agreement is
only a rough signal of whether the ranking is sensible. Jobs whose relevant
categories do not exist in a dataset are ranked but not scored.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from resume_matcher.config import (
    METHOD_GLOVE,
    METHOD_HYBRID,
    METHOD_LABELS,
    METHOD_SBERT,
    PROJECT_ROOT,
    SAMPLE_JOBS_PATH,
    available_datasets,
    get_settings,
)
from resume_matcher.data_loader import (
    DatasetError,
    corpus_from_files,
    load_resumes,
    load_sample_jobs,
    read_resume_file,
)
from resume_matcher.matcher import (
    MatcherError,
    MethodUnavailableError,
    create_matcher,
    load_glove_vectors,
    load_sbert_model,
)

RESULTS_DIR = PROJECT_ROOT / "results"
DEMO_DIR = PROJECT_ROOT / "demo_samples"
DEMO_LABELS_PATH = DEMO_DIR / "relevance_labels.json"


def precision_at_k(categories: list, relevant: set[str]) -> float:
    """Fraction of the retrieved resumes whose category is in ``relevant``."""
    if not categories:
        return 0.0
    return sum(1 for c in categories if c in relevant) / len(categories)


def recall_at_k(gains_in_rank_order, k: int) -> float:
    """Share of all relevant resumes (gain > 0) that appear in the top ``k``."""
    gains = np.asarray(gains_in_rank_order, dtype=float)
    total = (gains > 0).sum()
    return float((gains[:k] > 0).sum() / total) if total else 0.0


def ndcg_at_k(gains_in_rank_order, k: int) -> float:
    """Normalized discounted cumulative gain of the top ``k`` (binary or graded gains)."""
    gains = np.asarray(gains_in_rank_order, dtype=float)
    discounts = 1.0 / np.log2(np.arange(2, k + 2))
    dcg = ((2 ** gains[:k] - 1) * discounts[: len(gains[:k])]).sum()
    ideal = np.sort(gains)[::-1][:k]
    idcg = ((2**ideal - 1) * discounts[: len(ideal)]).sum()
    return float(dcg / idcg) if idcg > 0 else 0.0


_MODELS: dict[str, object] = {}


def shared_models(method: str, settings) -> dict[str, object]:
    """Load GloVe and Sentence-BERT at most once per evaluation run."""
    kwargs = {}
    if method in (METHOD_GLOVE, METHOD_HYBRID):
        kwargs["word_vectors"] = load_glove_vectors(settings.glove_path)
    if method in (METHOD_SBERT, METHOD_HYBRID):
        if "sbert" not in _MODELS:
            _MODELS["sbert"] = load_sbert_model(settings.sbert_model_name, settings.sbert_device)
        kwargs["sbert_model"] = _MODELS["sbert"]
    return kwargs


def evaluate(methods: list[str], top_k: int, dataset_path: Path) -> pd.DataFrame:
    """Run every method on every sample job for one dataset; one row per (method, job)."""
    settings = get_settings()
    corpus = load_resumes(dataset_path)
    jobs = load_sample_jobs(SAMPLE_JOBS_PATH)
    if not jobs:
        raise DatasetError(f"No sample job descriptions found in {SAMPLE_JOBS_PATH}.")

    dataset = Path(corpus.source).name
    known_categories = set(corpus.df["category"].dropna())
    print(f"\n=== Dataset: {dataset} ({len(corpus)} resumes after cleaning, stats={corpus.stats})")
    print(f"Sample job descriptions: {len(jobs)} | top-K = {top_k}")

    rows = []
    for method in methods:
        start = time.perf_counter()
        try:
            matcher = create_matcher(method, corpus, settings, **shared_models(method, settings))
        except (MethodUnavailableError, MatcherError) as exc:
            print(f"[{method}] UNAVAILABLE: {exc}")
            rows.append({"dataset": dataset, "method": method, "status": f"unavailable: {exc}"})
            continue
        build_seconds = time.perf_counter() - start
        note = " (embeddings loaded from cache)" if getattr(matcher, "loaded_from_cache", False) else ""
        print(f"[{method}] index built in {build_seconds:.2f}s{note}")

        for job in jobs:
            start = time.perf_counter()
            ranking = matcher.match(job["text"], top_k=len(corpus))  # full ranking, for recall
            query_seconds = time.perf_counter() - start
            results = ranking.head(top_k)

            relevant = set(job.get("relevant_categories", [])) & known_categories
            gains = ranking["category"].isin(relevant).astype(float).to_numpy()
            top1 = results["category"].iloc[0]
            rows.append(
                {
                    "dataset": dataset,
                    "method": method,
                    "status": "ok",
                    "job": job["title"],
                    "relevant_categories": "; ".join(sorted(relevant)) or None,
                    "top1_category": top1,
                    "top1_score": results["score"].iloc[0],
                    "top1_relevant": (top1 in relevant) if relevant else None,
                    f"precision_at_{top_k}": (
                        round(precision_at_k(results["category"].tolist(), relevant), 3) if relevant else None
                    ),
                    f"recall_at_{top_k}": round(recall_at_k(gains, top_k), 3) if relevant else None,
                    f"ndcg_at_{top_k}": round(ndcg_at_k(gains, top_k), 3) if relevant else None,
                    "build_seconds": round(build_seconds, 3),
                    "query_seconds": round(query_seconds, 4),
                }
            )
    return pd.DataFrame(rows)


def summarize(results: pd.DataFrame, top_k: int) -> pd.DataFrame:
    """Per dataset and method: mean Precision@K over labelled jobs, top-1 hits and timings."""
    ok = results[results["status"] == "ok"].copy()
    if ok.empty:
        return pd.DataFrame()
    ok["labelled"] = ok["top1_relevant"].notna()
    ok["top1_hit"] = ok["top1_relevant"].fillna(False).astype(bool)
    return (
        ok.groupby(["dataset", "method"], sort=False)
        .agg(
            jobs=("job", "count"),
            labelled_jobs=("labelled", "sum"),
            mean_precision_at_k=(f"precision_at_{top_k}", "mean"),
            mean_recall_at_k=(f"recall_at_{top_k}", "mean"),
            mean_ndcg_at_k=(f"ndcg_at_{top_k}", "mean"),
            top1_relevant=("top1_hit", "sum"),
            build_seconds=("build_seconds", "first"),
            mean_query_seconds=("query_seconds", "mean"),
        )
        .round(4)
        .reset_index()
    )


def evaluate_labelled(methods: list[str], k: int = 5) -> pd.DataFrame:
    """Graded NDCG@k on the demo resumes, using the hand-assigned labels in relevance_labels.json."""
    if not DEMO_LABELS_PATH.is_file():
        return pd.DataFrame()
    labels = json.loads(DEMO_LABELS_PATH.read_text(encoding="utf-8"))["jobs"]
    jobs = {j["title"]: j["text"] for j in load_sample_jobs(SAMPLE_JOBS_PATH)}
    files = sorted((DEMO_DIR / "pdf").glob("*.pdf"))
    corpus = corpus_from_files([(p.stem, read_resume_file(p.name, p.read_bytes())) for p in files])
    settings = get_settings()
    rows = []
    for method in methods:
        try:
            matcher = create_matcher(method, corpus, settings, **shared_models(method, settings))
        except (MethodUnavailableError, MatcherError) as exc:
            rows.append({"method": method, "status": f"unavailable: {exc}"})
            continue
        for title, grades in labels.items():
            if title not in jobs:
                continue
            ranking = matcher.match(jobs[title], top_k=len(corpus))
            gains = ranking["resume_id"].map(grades).fillna(0).to_numpy()
            rows.append({"method": method, "status": "ok", "job": title, f"ndcg_at_{k}": round(ndcg_at_k(gains, k), 3),
                         "top1": ranking["resume_id"].iloc[0], "top1_grade": int(gains[0])})
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--methods", nargs="+", default=list(METHOD_LABELS), choices=list(METHOD_LABELS))
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--data", type=Path, default=None, help="Path to one resume CSV (default: all datasets)")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

    paths = [args.data] if args.data else list(available_datasets().values()) or [get_settings().dataset_path]
    try:
        results = pd.concat([evaluate(args.methods, args.top_k, path) for path in paths], ignore_index=True)
    except DatasetError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 20)
    pd.set_option("display.max_colwidth", 40)
    ok = results[results["status"] == "ok"]
    if not ok.empty:
        print("\nPer-job results:")
        print(ok.drop(columns=["status", "relevant_categories"]).to_string(index=False))
        summary = summarize(results, args.top_k)
        print(f"\nSummary (Precision@{args.top_k} uses dataset category as a proxy relevance label;")
        print("only jobs with a matching category in that dataset are scored):")
        print(summary.to_string(index=False))

        RESULTS_DIR.mkdir(exist_ok=True)
        results.to_csv(RESULTS_DIR / "evaluation_per_job.csv", index=False)
        summary.to_csv(RESULTS_DIR / "evaluation_summary.csv", index=False)

        labelled = evaluate_labelled(args.methods)
        if not labelled.empty and (labelled["status"] == "ok").any():
            ok_labelled = labelled[labelled["status"] == "ok"]
            print("\nHand-labelled demo resumes (graded NDCG@5, 3 = strong fit ... 0 = unrelated):")
            print(ok_labelled.groupby("method", sort=False).agg(
                jobs=("job", "count"), mean_ndcg_at_5=("ndcg_at_5", "mean"),
                top1_strong=("top1_grade", lambda g: int((g == 3).sum())),
            ).round(3).to_string())
            labelled.to_csv(RESULTS_DIR / "evaluation_labelled.csv", index=False)
        print(f"\nSaved CSV files to {RESULTS_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
