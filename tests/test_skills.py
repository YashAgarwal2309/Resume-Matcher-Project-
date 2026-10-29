import pytest

from resume_matcher.skills import (
    SKILLS,
    RequirementComparison,
    compare_requirements,
    extract_requirements,
    normalize_for_matching,
)


def names(job_description):
    return [r.name for r in extract_requirements(job_description)]


def compare(job_description, resume_text):
    return compare_requirements(extract_requirements(job_description), resume_text)


def test_vocabulary_names_are_unique():
    keys = [" ".join(normalize_for_matching(s.name).split()) for s in SKILLS]
    assert len(keys) == len(set(keys))


def test_normalize_keeps_length():
    text = "Machine-Learning / CI/CD_pipelines – İstanbul"
    assert len(normalize_for_matching(text)) == len(text)


def test_requirements_in_job_description_order():
    assert names("Required: Python, SQL, Git, Docker and AWS.") == ["Python", "SQL", "Git", "Docker", "AWS"]


def test_case_aliases_and_separators_are_one_requirement():
    assert names("machine-learning, Machine Learning, ML, k8s and Kubernetes") == ["Machine Learning", "Kubernetes"]


def test_longest_overlapping_skill_wins():
    assert names("Experience with SQL Server and React Native") == ["SQL Server", "React Native"]
    assert names("SQL Server and plain SQL") == ["SQL Server", "SQL"]


@pytest.mark.parametrize(
    "resume, expected",
    [
        ("Built apps in JavaScript", False),  # word boundary: Java is not in JavaScript
        ("Java developer", True),
        ("JAVA, Spring", True),  # case is ignored
    ],
)
def test_word_boundaries(resume, expected):
    assert bool(compare("Java", resume).included) is expected


def test_symbol_skills_and_exact_case_letters():
    comparison = compare("C, C++, C#, R and Go", "Expert in C++ and C#. Led R&D. Go-getter, go to market.")
    assert comparison.included == ("C++", "C#")
    assert comparison.missing == ("C", "R", "Go")
    assert compare("C, R and Go", "Languages: C, R (tidyverse), Go").included == ("C", "R", "Go")


def test_aliases_match_resume_wording():
    comparison = compare(
        "Must have: Kubernetes, AWS, Node.js, scikit-learn, NLP and CI/CD",
        "Deployed on k8s in Amazon Web Services; nodejs APIs; sklearn and natural language processing; "
        "continuous integration with Jenkins.",
    )
    assert comparison.missing == ()
    assert len(comparison.included) == 6


def test_plural_forms_match():
    assert compare("REST API design", "Designed REST APIs").included == ("REST APIs",)


def test_multi_word_skills_need_all_words():
    comparison = compare("Data analysis and machine learning", "Analysis of machine data; learning new tools")
    assert comparison.included == ()
    assert comparison.missing == ("Data Analysis", "Machine Learning")


def test_listed_items_outside_vocabulary_are_extracted():
    found = names("Requirements: wound care, IV therapy and patient care. 3+ years of experience; "
                  "Bachelor's degree in nursing.")
    assert found == ["Wound care", "IV therapy", "Patient Care"]
    assert compare("Requirements: wound care, IV therapy", "Provided wound-care daily").included == ("Wound care",)


def test_category_before_such_as_is_not_a_requirement():
    assert names("Experience with cloud platforms such as GCP or Azure") == ["GCP", "Azure"]


def test_text_already_covered_by_vocabulary_is_not_repeated():
    assert names("Experience with Docker containers and Python programming") == ["Docker", "Python"]


@pytest.mark.parametrize("job_description", ["", "   ", None, "We need someone friendly and reliable."])
def test_no_requirements(job_description):
    assert extract_requirements(job_description) == []
    assert compare_requirements([], "Python developer") == RequirementComparison((), ())


@pytest.mark.parametrize("resume_text", ["", None, float("nan")])
def test_empty_resume_misses_everything(resume_text):
    comparison = compare_requirements(extract_requirements("Python and SQL"), resume_text)
    assert comparison.included == ()
    assert comparison.missing == ("Python", "SQL")
    assert comparison.total == 2


def test_each_resume_is_compared_separately():
    requirements = extract_requirements("Required: Python, SQL, Docker")
    first = compare_requirements(requirements, "Python and SQL")
    second = compare_requirements(requirements, "Docker only")
    assert first == RequirementComparison(("Python", "SQL"), ("Docker",))
    assert second == RequirementComparison(("Docker",), ("Python", "SQL"))
