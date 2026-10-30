"""Tests for candidate analysis, comparison and exports (resume_matcher/comparison.py).

All resumes are synthetic; scores come from the real TF-IDF matcher.
"""

import io

import pandas as pd
import pytest

from resume_matcher.comparison import (
    DEFAULT_STRONG_THRESHOLDS,
    MAX_COMPARE,
    NOT_AVAILABLE,
    SORT_OPTIONS,
    build_candidates,
    comparison_csv,
    comparison_summary,
    comparison_table,
    extract_education,
    extract_experience,
    filter_candidates,
    masked_text,
    score_bins,
    select_candidates,
    shared_keywords,
    shared_skills,
    skill_matrix,
    skill_matrix_csv,
    sort_candidates,
    summarize_run,
    validate_selection,
)
from resume_matcher.config import METHOD_LABELS, METHOD_TFIDF
from resume_matcher.data_loader import corpus_from_files
from resume_matcher.matcher import TfidfMatcher
from resume_matcher.skills import extract_requirements

JOB = "Backend developer. Required: Python, SQL, Django, Docker and AWS."

RESUMES = [
    ("alice.txt", "Python developer with 4 years of experience in Django, SQL and Docker. "
                  "B.Tech, Computer Science | Some University | 2015 - 2019"),
    ("bob.txt", "Java and C++ engineer. 6+ years experience building backend systems with SQL and AWS. "
                "Master of Science in Software Engineering, 2014."),
    ("carol.txt", "Registered nurse providing patient care. Comfortable with MS Excel."),
    ("dave.txt", "Python scripting, pandas and SQL reports for a logistics team."),
    ("erin.txt", "Head chef managing kitchen staff and menu planning."),
    ("frank.txt", "DevOps engineer: Docker, Kubernetes, AWS and Python automation."),
]


def run_tfidf(files, job=JOB, top_k=None):
    corpus = corpus_from_files(files)
    results = TfidfMatcher(corpus).fit().match(job, top_k=top_k or len(corpus))
    requirements = extract_requirements(job)
    return results, build_candidates(results, corpus, requirements, job, METHOD_TFIDF)


@pytest.fixture
def ranked():
    return run_tfidf(RESUMES)


# ---- building the candidate table


def test_scores_and_ranks_are_copied_from_the_matcher(ranked):
    results, candidates = ranked
    assert candidates["rank"].tolist() == results["rank"].tolist()
    assert candidates["resume_id"].tolist() == results["resume_id"].tolist()
    assert candidates["score"].tolist() == results["score"].tolist()
    assert candidates["cosine_similarity"].tolist() == results["cosine_similarity"].tolist()
    assert set(candidates["method"]) == {METHOD_TFIDF}


def test_requirements_are_split_into_matched_and_unconfirmed(ranked):
    _, candidates = ranked
    alice = candidates.set_index("resume_id").loc["alice.txt"]
    assert alice["matched"] == ("Python", "SQL", "Django", "Docker")
    assert alice["unconfirmed"] == ("AWS",)
    assert (alice["matched_count"], alice["requirement_count"]) == (4, 5)
    assert alice["requirements"] == ("Python", "SQL", "Django", "Docker", "AWS")


def test_unknown_method_is_rejected(ranked):
    results, _ = ranked
    with pytest.raises(ValueError):
        build_candidates(results, corpus_from_files(RESUMES), [], JOB, "magic")


# ---- evidence from the resume text


def test_experience_is_quoted_only_when_stated():
    assert extract_experience("Developer with 4 years of experience in Django.") == ("4 years of experience",)
    assert extract_experience("About one and a half years of industry experience in NLP.") == (
        "About one and a half years of industry experience",
    )
    assert extract_experience("6+ years experience building systems") == ("6+ years experience",)
    assert extract_experience("Experience of 3 years in sales") == ("Experience of 3 years",)
    # Durations that are not about the candidate's experience are ignored.
    assert extract_experience("Migrated two years of old data. Worked 2016 - 2020.") == ()
    assert extract_experience("") == ()
    assert extract_experience(None) == ()


def test_education_mentions_need_a_real_degree_name():
    assert extract_education("B.Tech, Computer Science | Some University | 2015 - 2019") == ("B.Tech, Computer Science",)
    assert extract_education("Master of Science in Software Engineering, 2014.") == (
        "Master of Science in Software Engineering",
    )
    assert extract_education("MBA (Human Resource Management) | Amity | 2017 - 2019 B.Com (Hons.) | DU") == (
        "MBA (Human Resource Management)", "B.Com (Hons.)",
    )
    # "MS Excel", "be", "ba" and "masters the craft" are not degrees.
    assert extract_education("Comfortable with MS Excel, able to be flexible; masters the craft.") == ()
    assert extract_education(None) == ()


def test_missing_evidence_is_reported_as_not_available(ranked):
    _, candidates = ranked
    erin = candidates.set_index("resume_id").loc["erin.txt"]
    assert erin["experience"] == () and erin["education"] == ()
    table = comparison_table(select_candidates(candidates, ["alice.txt", "erin.txt"]))
    assert table.set_index("candidate").loc["erin.txt", "experience_mentions"] == NOT_AVAILABLE
    assert table.set_index("candidate").loc["erin.txt", "education_mentions"] == NOT_AVAILABLE


def test_shared_keywords_skip_generic_words_and_listed_skills():
    job = "Senior data engineer with strong experience building Spark pipelines and Airflow scheduling."
    resume = "Data engineer: Spark pipelines, Airflow scheduling, strong experience."
    keywords = shared_keywords(job, resume, exclude=("Apache Spark",))
    assert "experience" not in keywords and "strong" not in keywords
    assert "spark" not in keywords  # already listed as a matched requirement
    assert keywords[:2] == ("data", "engineer")
    assert "pipelines" in keywords and "scheduling" in keywords


def test_masked_text_hides_contact_details():
    text = masked_text("Mail me at jane@example.com or call +91 98765 43210, see https://example.com/jane")
    assert "jane@example.com" not in text and "98765" not in text and "https://" not in text
    assert "[email]" in text and "[phone]" in text and "[link]" in text


# ---- KPIs, filtering, sorting


def test_run_summary_counts_strong_matches_with_the_threshold(ranked):
    _, candidates = ranked
    summary = summarize_run(candidates, resumes_processed=8, threshold=20)
    assert summary.resumes_processed == 8
    assert summary.candidates_ranked == len(candidates)
    assert summary.strong_matches == int((candidates["score"] >= 20).sum())
    assert summary.average_score == pytest.approx(candidates["score"].mean())
    assert summarize_run(candidates, 8, threshold=101).strong_matches == 0
    assert summarize_run(candidates.iloc[0:0], 0, threshold=50).average_score is None


def test_default_thresholds_exist_for_every_method():
    assert set(DEFAULT_STRONG_THRESHOLDS) == set(METHOD_LABELS)
    assert all(0 < value < 100 for value in DEFAULT_STRONG_THRESHOLDS.values())


def test_filters_combine(ranked):
    _, candidates = ranked
    assert filter_candidates(candidates, query="ALICE")["resume_id"].tolist() == ["alice.txt"]
    by_skill = filter_candidates(candidates, query="docker")  # matched or unconfirmed requirement
    assert len(by_skill) == len(candidates)  # every row lists every requirement
    shortlisted = filter_candidates(candidates, shortlist={"bob.txt", "erin.txt"}, shortlisted_only=True)
    assert set(shortlisted["resume_id"]) == {"bob.txt", "erin.txt"}
    strong = filter_candidates(candidates, strong_only=True, threshold=10)
    assert (strong["score"] >= 10).all() and len(strong) == int((candidates["score"] >= 10).sum())
    with_aws = filter_candidates(candidates, must_include=["AWS", "Docker"])
    assert set(with_aws["resume_id"]) == {"frank.txt"}
    assert filter_candidates(candidates, query="no such thing").empty


def test_sorting(ranked):
    _, candidates = ranked
    assert sort_candidates(candidates.sample(frac=1, random_state=1), "rank")["rank"].tolist() == sorted(candidates["rank"])
    assert sort_candidates(candidates, "score_asc")["score"].is_monotonic_increasing
    by_reqs = sort_candidates(candidates, "requirements")
    assert by_reqs["matched_count"].is_monotonic_decreasing
    assert sort_candidates(candidates, "name")["resume_id"].tolist() == sorted(candidates["resume_id"])
    by_coverage = sort_candidates(candidates, "coverage")
    assert by_coverage["coverage"].is_monotonic_decreasing
    assert set(SORT_OPTIONS) == {"rank", "score_asc", "coverage", "requirements", "name"}
    with pytest.raises(ValueError):
        sort_candidates(candidates, "age")


# ---- selection rules


@pytest.mark.parametrize("count", [2, 3, 4, 5])
def test_two_to_five_candidates_can_be_compared(ranked, count):
    _, candidates = ranked
    ids = candidates["resume_id"].tolist()[::-1][:count]  # deliberately not in rank order
    assert validate_selection(ids).ok
    selected = select_candidates(candidates, ids)
    assert len(selected) == count
    assert selected["rank"].is_monotonic_increasing  # always shown in rank order


@pytest.mark.parametrize("count", [0, 1, MAX_COMPARE + 1])
def test_fewer_than_two_or_more_than_five_are_refused(ranked, count):
    _, candidates = ranked
    ids = candidates["resume_id"].tolist()[:count]
    status = validate_selection(ids)
    assert not status.ok
    assert ("more candidate" in status.message) if count < 2 else ("at most 5" in status.message)
    with pytest.raises(ValueError):
        select_candidates(candidates, ids)


def test_duplicate_or_unknown_ids_are_refused(ranked):
    _, candidates = ranked
    assert not validate_selection(["alice.txt", "alice.txt"]).ok
    with pytest.raises(ValueError, match="Not part of the current ranking"):
        select_candidates(candidates, ["alice.txt", "someone-from-an-old-run.pdf"])


def test_a_run_with_one_candidate_cannot_be_compared():
    _, candidates = run_tfidf(RESUMES, top_k=1)
    status = validate_selection([], available=len(candidates))
    assert not status.ok and "at least 2 ranked candidates" in status.message


def test_candidates_from_different_methods_cannot_be_mixed(ranked):
    _, candidates = ranked
    mixed = candidates.copy()
    mixed.loc[mixed["resume_id"] == "bob.txt", "method"] = "sbert"
    with pytest.raises(ValueError, match="different methods"):
        select_candidates(mixed, ["alice.txt", "bob.txt"])


# ---- comparison content


def test_skill_matrix_and_shared_skills(ranked):
    _, candidates = ranked
    selected = select_candidates(candidates, ["alice.txt", "frank.txt", "erin.txt"])
    matrix = skill_matrix(selected)
    assert matrix.index.tolist() == ["Python", "SQL", "Django", "Docker", "AWS"]
    assert matrix.columns.tolist() == selected["resume_id"].tolist()
    assert matrix.loc["Django", "alice.txt"] and not matrix.loc["Django", "frank.txt"]
    assert not matrix["erin.txt"].any()
    shared = shared_skills(selected)
    assert shared.found_in_all == ()
    assert set(shared.found_in_some) == {"Python", "SQL", "Django", "Docker", "AWS"}
    assert shared.found_in_none == ()


def test_summary_states_its_criteria_and_makes_no_hiring_claims(ranked):
    _, candidates = ranked
    selected = select_candidates(candidates, ["alice.txt", "bob.txt", "frank.txt"])
    summary = " ".join(comparison_summary(selected))
    top = selected.loc[selected["score"].idxmax()]
    assert f"{top['resume_id']} has the highest TF-IDF similarity score ({top['score']:.2f})" in summary
    assert "detected job requirements" in summary
    assert "Only alice.txt mentions Django" in summary
    for word in ("hire", "best candidate", "should", "recommend", "age", "gender"):
        assert word not in summary.lower()


def test_summary_explains_when_score_and_requirements_disagree():
    files = [
        ("words.txt", "Backend developer. Backend developer for backend Python services."),
        ("skills.txt", "Gardening, painting, travel, cooking, music, hiking, photography, chess, reading, "
                       "volunteering, baking, cycling, yoga, knitting. Tools: Python, SQL, Django, Docker, AWS."),
        ("other.txt", "Nurse caring for patients in a hospital ward."),
    ]
    _, candidates = run_tfidf(files)
    summary = comparison_summary(select_candidates(candidates, ["words.txt", "skills.txt"]))
    best_score = candidates.iloc[0]["resume_id"]
    most_reqs = candidates.loc[candidates["matched_count"].idxmax(), "resume_id"]
    assert best_score != most_reqs
    assert any("belong to different candidates" in line for line in summary)


def test_job_without_requirements_compares_scores_only():
    _, candidates = run_tfidf(RESUMES, job="backend developer")
    selected = select_candidates(candidates, candidates["resume_id"].tolist()[:2])
    assert skill_matrix(selected).empty
    assert shared_skills(selected).found_in_all == ()
    summary = comparison_summary(selected)
    assert any("only the similarity scores can be compared" in line for line in summary)
    row = comparison_table(selected).iloc[0]
    assert row["requirements_found"] == "" and row["matched_skills"] == ""


def test_duplicate_file_names_stay_separate_candidates():
    files = [("cv.pdf", "Python developer with SQL"), ("cv.pdf", "Chef managing a kitchen"), ("x.txt", "Python")]
    _, candidates = run_tfidf(files, job="Required: Python and SQL")
    ids = candidates["resume_id"].tolist()
    assert len(set(ids)) == 3 and {"cv.pdf-0", "cv.pdf-1"} <= set(ids)
    selected = select_candidates(candidates, ["cv.pdf-0", "cv.pdf-1"])
    assert skill_matrix(selected).loc["SQL"].tolist() == [True, False]


def test_empty_resume_text_is_dropped_before_ranking():
    files = [("empty.txt", "   "), ("a.txt", "Python developer"), ("b.txt", "SQL analyst")]
    _, candidates = run_tfidf(files, job="Required: Python")
    assert "empty.txt" not in candidates["resume_id"].tolist()
    assert len(candidates) == 2


# ---- CSV exports


def read_csv(data: bytes) -> pd.DataFrame:
    assert data.startswith(b"\xef\xbb\xbf")  # UTF-8 BOM so spreadsheet apps detect Unicode
    return pd.read_csv(io.BytesIO(data), encoding="utf-8-sig", keep_default_na=False)


def test_comparison_csv_has_one_row_per_candidate(ranked):
    _, candidates = ranked
    selected = select_candidates(candidates, ["frank.txt", "alice.txt"])
    table = read_csv(comparison_csv(selected, shortlist={"alice.txt"}))
    assert table["candidate"].tolist() == selected["resume_id"].tolist()
    assert list(table.columns) == [
        "candidate", "rank", "similarity_score", "cosine_similarity", "matching_method", "requirements_found",
        "coverage_percent", "matched_skills", "unconfirmed_skills", "related_unverified", "evidence",
        "experience_mentions", "education_mentions", "shortlisted",
    ]
    alice = table.set_index("candidate").loc["alice.txt"]
    assert alice["similarity_score"] == pytest.approx(candidates.set_index("resume_id").loc["alice.txt", "score"])
    assert alice["matched_skills"] == "Python; SQL; Django; Docker"
    assert alice["unconfirmed_skills"] == "AWS"
    assert alice["requirements_found"] == "4 of 5"
    assert alice["experience_mentions"] == "4 years of experience"
    assert alice["shortlisted"] == "yes"
    assert table.set_index("candidate").loc["frank.txt", "shortlisted"] == "no"
    assert "text" not in table.columns and "preview" not in table.columns  # no raw resume text


def test_csv_escapes_commas_quotes_newlines_unicode_and_formulas():
    files = [
        ('=HYPERLINK("x"),"quoted"\nname.txt', "Python developer, Zürich — naïve résumé"),
        ("Ünïcødé, résumé.txt", "Python and SQL"),
    ]
    _, candidates = run_tfidf(files, job="Required: Python")
    selected = select_candidates(candidates, candidates["resume_id"].tolist())
    data = comparison_csv(selected)
    table = read_csv(data)
    assert len(table) == 2
    names = table["candidate"].tolist()
    assert "'" + '=HYPERLINK("x"),"quoted"\nname.txt' in names  # formula neutralised, content intact
    assert "Ünïcødé, résumé.txt" in names
    matrix = pd.read_csv(io.BytesIO(skill_matrix_csv(selected)), encoding="utf-8-sig", index_col=0)
    assert matrix.loc["Python"].tolist() == ["found", "found"]
    assert "Ünïcødé, résumé.txt" in matrix.columns


def test_score_bins_cover_the_full_scale():
    bins = score_bins(pd.Series([0, 5, 12, 99.5, 100]))
    assert len(bins) == 10 and bins["candidates"].sum() == 5
    assert bins.loc[0, "candidates"] == 2 and bins.loc[9, "candidates"] == 2
