"""Smoke tests for the Streamlit UI using Streamlit's built-in AppTest runner.

Resumes are uploaded through the file uploader (synthetic text and a tiny generated PDF),
and GloVe is pointed at a missing file to check the "unavailable" message.
"""

from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

from tests.conftest import SYNTHETIC_RESUMES, make_pdf

APP_PATH = str(Path(__file__).resolve().parents[1] / "app.py")


def resume_files():
    """The five synthetic resumes as uploaded .txt files, named after their IDs."""
    return [
        (f"{rid}.txt", text.encode("utf-8"), "text/plain")
        for rid, text in zip(SYNTHETIC_RESUMES["ID"], SYNTHETIC_RESUMES["Resume_str"])
    ]


@pytest.fixture
def app(monkeypatch, tmp_path):
    monkeypatch.setenv("GLOVE_PATH", str(tmp_path / "missing-glove.txt"))
    at = AppTest.from_file(APP_PATH, default_timeout=60)
    at.run()
    assert not at.exception
    return at


def upload(at, files):
    at.file_uploader(key="resume_files").set_value(files)
    at.run()
    assert not at.exception


def click_match(at):
    next(b for b in at.button if b.label == "Match Resumes").click().run()


def test_app_starts_empty(app):
    assert app.title[0].value.endswith("Resume Matcher")
    assert not app.error
    assert app.text_area(key="jd_text").value == ""  # nothing is prefilled
    assert len(app.text_area) == 1 and len(app.file_uploader) == 1  # one JD box, one resume uploader
    assert app.file_uploader(key="resume_files").accept_multiple_files
    assert app.selectbox(key="method").value == "hybrid"  # recommended method is the default
    assert not app.dataframe


def test_match_without_resumes_shows_warning(app):
    app.text_area(key="jd_text").input("python developer")
    click_match(app)
    assert "upload at least one resume" in app.warning[0].value


def test_empty_job_description_shows_warning(app):
    upload(app, resume_files())
    click_match(app)
    assert "enter a job description" in app.warning[0].value


def test_tfidf_match_shows_ranked_results(app):
    upload(app, resume_files())
    assert any("5" in c.value and "ready to match" in c.value for c in app.caption)
    app.text_area(key="jd_text").input("Hospital nurse for patient care")
    app.selectbox(key="method").select("tfidf")
    click_match(app)
    assert not app.exception and not app.error
    table = app.dataframe[0].value
    assert table["resume_id"].iloc[0] == "102.txt"
    assert table["rank"].tolist() == list(range(1, 6))  # Top K defaults to all uploaded resumes
    assert "category" not in table.columns
    assert len(app.get("download_button")) == 1


def test_pdf_resumes_are_read(app):
    upload(app, [("nurse.pdf", make_pdf("Registered nurse for hospital patient care"), "application/pdf"),
                 ("chef.pdf", make_pdf("Head chef managing a busy kitchen"), "application/pdf")])
    app.text_area(key="jd_text").input("nurse")
    app.selectbox(key="method").select("tfidf")
    click_match(app)
    assert app.dataframe[0].value["resume_id"].iloc[0] == "nurse.pdf"


def test_unreadable_file_is_skipped_with_warning(app):
    upload(app, [("broken.pdf", b"not a pdf", "application/pdf"), *resume_files()[:2]])
    assert "Could not read 'broken.pdf'" in app.warning[0].value
    assert any("2" in c.value and "ready to match" in c.value for c in app.caption)


def test_changing_uploads_clears_results(app):
    upload(app, resume_files())
    app.text_area(key="jd_text").input("chef kitchen")
    app.selectbox(key="method").select("tfidf")
    click_match(app)
    assert app.dataframe
    upload(app, resume_files()[:3])
    assert not app.dataframe  # old results belonged to the other set of resumes


def test_unavailable_glove_is_reported(app):
    upload(app, resume_files())
    app.text_area(key="jd_text").input("python developer")
    app.selectbox(key="method").select("glove")
    click_match(app)
    assert "not available" in app.error[0].value
    assert not app.dataframe


def result_cards(at):
    """The per-resume result expanders (the sidebar's "How it works" expanders have no '#')."""
    return [e for e in at.expander if e.label.startswith("#")]


def test_every_ranked_resume_shows_requirement_columns(app):
    upload(app, resume_files())
    app.text_area(key="jd_text").input("Python developer. Required: Python, SQL, Django, Docker and AWS.")
    app.selectbox(key="method").select("tfidf")
    click_match(app)
    assert not app.exception and not app.error
    ranked = app.dataframe[0].value["resume_id"].tolist()
    cards = result_cards(app)
    assert [c.label.split(" · ")[1] for c in cards] == ranked  # one card per ranked resume, in rank order
    for card in cards:
        assert len(card.table) == 1
        table = card.table[0].value
        assert list(table.columns) == ["Included in Resume", "Not Found in Resume"]
    by_resume = {c.label.split(" · ")[1]: c.table[0].value.iloc[0].tolist() for c in cards}
    assert by_resume["101.txt"] == ["Python, SQL, Django", "Docker, AWS"]
    assert by_resume["103.txt"] == ["None detected", "Python, SQL, Django, Docker, AWS"]
    assert cards[ranked.index("101.txt")].label.endswith("3 of 5 requirements")
    assert len(app.get("download_button")) == 1  # CSV export is still offered


def test_requirement_columns_use_the_matched_job_description(app):
    upload(app, resume_files())
    app.text_area(key="jd_text").input("Required: Python and Java")
    app.selectbox(key="method").select("tfidf")
    click_match(app)
    app.text_area(key="jd_text").input("Required: cooking").run()  # edited but not re-matched
    tables = [c.table[0].value.iloc[0].tolist() for c in result_cards(app)]
    assert ["Python", "Java"] in tables and ["Java", "Python"] in tables


def test_job_description_without_requirements_says_so(app):
    upload(app, resume_files())
    app.text_area(key="jd_text").input("hospital nurse")
    app.selectbox(key="method").select("tfidf")
    click_match(app)
    cards = result_cards(app)
    assert len(cards) == 5
    for card in cards:
        assert not card.table
        assert "No specific skills or requirements were detected" in card.info[0].value


def test_hybrid_needs_glove_and_says_so(app):
    upload(app, resume_files())
    app.text_area(key="jd_text").input("python developer")
    click_match(app)  # hybrid is the default method; the fixture points GloVe at a missing file
    assert "not available" in app.error[0].value
    assert not app.dataframe
