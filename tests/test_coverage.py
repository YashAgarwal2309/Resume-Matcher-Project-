"""Job-description requirement coverage (resume_matcher/coverage.py). All texts are synthetic."""

import pytest

from resume_matcher.coverage import (
    MATCH_ALIAS,
    MATCH_EXACT,
    MATCH_STATED,
    RELATED_GROUPS,
    STATUS_MATCHED,
    STATUS_MISSING,
    STATUS_RELATED,
    assess,
    degree_level,
    job_requirements,
    snippet,
    years_in,
)
from resume_matcher.skills import vocabulary

JOB = (
    "Backend Engineer. Required: Python, Django, PostgreSQL, Docker, Kubernetes and AWS. "
    "Bachelor's degree in Computer Science or a related field. 3+ years of experience building APIs."
)


def checks(resume: str, job: str = JOB) -> dict:
    return {c.name: c for c in assess(job_requirements(job), resume).checks}


def test_job_requirements_cover_skills_degree_and_years():
    job = job_requirements(JOB)
    assert job.names == (
        "Python", "Django", "PostgreSQL", "Docker", "Kubernetes", "AWS",
        "Bachelor's degree or higher", "3+ years of experience",
    )
    assert job.degree_text.startswith("Bachelor's degree") and "3+ years" in job.years_text


@pytest.mark.parametrize("text, degree, years", [
    ("Master's or PhD in statistics required", "Master's degree or higher", ""),
    ("BS in Computer Science; MS preferred", "Bachelor's degree or higher", ""),
    ("5-7 years experience in sales", "", "5+ years of experience"),
    ("Minimum of four years' experience", "", "4+ years of experience"),
    ("360 degree feedback, MS Excel and great teamwork", "", ""),
    ("Join a team of 12 engineers working 5 days a week", "", ""),
])
def test_degree_and_years_parsing(text, degree, years):
    job = job_requirements(text)
    assert (job.degree_label, job.years_label) == (degree, years)


def test_exact_and_alias_matches_have_evidence():
    found = checks("Built Python services with Django on Postgres; deployed with k8s and Docker on AWS.")
    assert found["Python"].status == STATUS_MATCHED and found["Python"].match_type == MATCH_EXACT
    assert found["PostgreSQL"].match_type == MATCH_ALIAS and "Postgres" in found["PostgreSQL"].evidence
    assert found["Kubernetes"].match_type == MATCH_ALIAS and "k8s" in found["Kubernetes"].evidence
    for name in ("Python", "Django", "PostgreSQL", "Docker", "Kubernetes", "AWS"):
        assert found[name].evidence, name  # every match quotes the resume


def test_related_technology_is_not_a_match():
    found = checks("Python developer: Flask APIs, MySQL, Azure.")
    assert found["Django"].status == STATUS_RELATED and found["Django"].related_to == ("Flask",)
    assert "Flask" in found["Django"].evidence and "not the same" in found["Django"].note
    assert found["PostgreSQL"].status == STATUS_RELATED and found["AWS"].status == STATUS_RELATED
    report = assess(job_requirements(JOB), "Python developer: Flask APIs, MySQL, Azure.")
    assert "Django" not in report.matched and "Django" in report.related
    assert report.score == pytest.approx(100 * 1 / 8, abs=0.1)  # only Python counts


def test_missing_requirements_have_no_evidence():
    found = checks("Head chef managing a kitchen.")
    assert all(c.status == STATUS_MISSING for c in found.values())
    assert found["Python"].evidence == ""


def test_degree_level_is_compared_not_the_wording():
    assert checks("B.Tech, Computer Engineering | VIT | 2016 - 2020")["Bachelor's degree or higher"].status == STATUS_MATCHED
    master = checks("M.Tech, Computer Science | IIT | 2018")["Bachelor's degree or higher"]
    assert master.status == STATUS_MATCHED and master.match_type == MATCH_STATED and "M.Tech" in master.evidence
    diploma = checks("Diploma in Computer Engineering")["Bachelor's degree or higher"]
    assert diploma.status == STATUS_MISSING and "Diploma" in diploma.note
    assert "No degree found" in checks("Self-taught developer")["Bachelor's degree or higher"].note


def test_years_are_compared_only_when_stated():
    assert checks("Developer with 4 years of experience")["3+ years of experience"].status == STATUS_MATCHED
    short = checks("Developer with 2 years of experience")["3+ years of experience"]
    assert short.status == STATUS_MISSING and "States 2 years" in short.note
    unstated = checks("Developer at Acme, 2015 - 2023")["3+ years of experience"]
    assert unstated.status == STATUS_MISSING and "not added up" in unstated.note


def test_score_is_none_without_requirements():
    report = assess(job_requirements("We are hiring a friendly person."), "Python developer")
    assert report.total == 0 and report.score is None


def test_evidence_hides_contact_details():
    found = checks("Python developer, mail dev@example.com, phone +91 98765 43210, see https://dev.example.com")
    assert "dev@example.com" not in found["Python"].evidence and "98765" not in found["Python"].evidence


def test_snippet_is_bounded():
    text = "word " * 200 + "Python" + " word" * 200
    quote = snippet(text, text.index("Python"), text.index("Python") + 6)
    assert "Python" in quote and len(quote) < 200 and quote.startswith("…") and quote.endswith("…")


def test_helpers():
    assert years_in("about one and a half years of industry experience") == 1.5
    assert years_in("6+ years experience") == 6
    assert years_in("experience") is None
    assert degree_level("PhD in Physics") > degree_level("M.Sc Chemistry") > degree_level("B.Com (Hons.)") > 0


def test_related_groups_use_known_skills():
    names = {r.name for r in vocabulary()}
    assert all(name in names for group in RELATED_GROUPS for name in group)
