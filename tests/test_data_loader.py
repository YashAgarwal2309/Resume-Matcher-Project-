import pandas as pd
import pytest

from resume_matcher.config import SAMPLE_JOBS_PATH
from resume_matcher.data_loader import (
    DatasetError,
    corpus_from_files,
    detect_columns,
    load_resumes,
    load_sample_jobs,
    prepare_corpus,
    read_resume_file,
)
from tests.conftest import make_docx, make_pdf


def test_load_valid_csv(resume_csv):
    corpus = load_resumes(resume_csv)
    assert len(corpus) == 5
    assert corpus.text_column == "Resume_str"
    assert corpus.category_column == "Category"
    assert corpus.id_column == "ID"
    assert list(corpus.df.columns) == ["resume_id", "category", "text", "clean_text"]
    assert corpus.df["resume_id"].tolist() == ["101", "102", "103", "104", "105"]


def test_original_text_is_preserved(corpus):
    assert "dev@example.com" in corpus.df.loc[0, "text"]
    assert "dev@example.com" not in corpus.df.loc[0, "clean_text"]


def test_missing_file(tmp_path):
    with pytest.raises(DatasetError, match="not found"):
        load_resumes(tmp_path / "nope.csv")


def test_completely_empty_file(tmp_path):
    path = tmp_path / "empty.csv"
    path.write_text("")
    with pytest.raises(DatasetError, match="empty"):
        load_resumes(path)


def test_header_only_file(tmp_path):
    path = tmp_path / "header.csv"
    path.write_text("ID,Resume_str,Category\n")
    with pytest.raises(DatasetError, match="empty"):
        load_resumes(path)


def test_no_text_column():
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    with pytest.raises(DatasetError, match="text column"):
        prepare_corpus(df)


def test_detects_alternative_column_names():
    df = pd.DataFrame({"resume_text": ["some resume text here"], "Label": ["X"]})
    assert detect_columns(df) == ("resume_text", "Label", None)


def test_fallback_detection_ignores_html_column():
    long = "experienced accountant handling ledgers and audits " * 3
    df = pd.DataFrame({"Resume_html": ["<div>" + long * 5 + "</div>"], "body": [long], "n": [1]})
    assert detect_columns(df)[0] == "body"


def test_null_empty_and_duplicate_rows_are_removed():
    df = pd.DataFrame(
        {
            "Resume_str": ["Python developer", None, "   ", "!!!", "python   DEVELOPER", 42],
            "Category": ["IT", "IT", "IT", "IT", "IT", None],
        }
    )
    corpus = prepare_corpus(df)
    assert corpus.stats == {"rows_read": 6, "dropped_empty": 3, "dropped_duplicates": 1, "rows_used": 2}
    assert corpus.df["text"].tolist() == ["Python developer", "42"]


def test_missing_optional_metadata_gets_defaults():
    corpus = prepare_corpus(pd.DataFrame({"text": ["first resume text", "second resume text"]}))
    assert corpus.df["resume_id"].tolist() == ["row-0", "row-1"]
    assert corpus.df["category"].isna().all()
    assert not corpus.has_categories


def test_duplicate_ids_are_made_unique():
    df = pd.DataFrame({"ID": [7, 7, None], "Resume_str": ["alpha resume", "beta resume", "gamma resume"]})
    ids = prepare_corpus(df).df["resume_id"].tolist()
    assert len(set(ids)) == 3
    assert ids[2] == "row-2"


def test_malformed_rows_are_skipped(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text("ID,Resume_str,Category\n1,good resume text,IT\n2,too,many,fields,here\n3,another resume,HR\n")
    corpus = load_resumes(path)
    assert corpus.df["resume_id"].tolist() == ["1", "3"]


def test_sample_jobs_file_is_valid():
    jobs = load_sample_jobs(SAMPLE_JOBS_PATH)
    assert len(jobs) >= 3
    assert all({"title", "relevant_categories", "text"} <= set(job) for job in jobs)
    assert all(isinstance(job["relevant_categories"], list) for job in jobs)


def test_sample_jobs_missing_file(tmp_path):
    assert load_sample_jobs(tmp_path / "missing.json") == []


def test_mojibake_is_repaired_on_load():
    df = pd.DataFrame({"Category": ["Data Science"], "Resume": ["Machine learning: NaÃ¯ve Bayes, SVM â€“ Python"]})
    corpus = prepare_corpus(df)
    assert corpus.df.loc[0, "text"] == "Machine learning: Naïve Bayes, SVM – Python"
    assert corpus.text_column == "Resume"


def test_tech_dataset_shape_without_id_column():
    df = pd.DataFrame({"Category": ["Java Developer"] * 3, "Resume": ["java spring", "java spring", "python django"]})
    corpus = prepare_corpus(df)
    assert corpus.stats["dropped_duplicates"] == 1
    assert corpus.df["resume_id"].tolist() == ["row-0", "row-2"]  # IDs keep the original row numbers


# ---------------------------------------------------------------- uploaded resume files


def test_read_text_resume_file():
    assert read_resume_file("cv.txt", "Python developer – Django".encode()) == "Python developer – Django"
    assert read_resume_file("cv.MD", "Java developer".encode("cp1252")) == "Java developer"


def test_read_pdf_resume_file():
    text = read_resume_file("cv.pdf", make_pdf("Registered nurse with hospital experience"))
    assert "Registered nurse with hospital experience" in text


def test_unsupported_or_broken_files_are_rejected():
    with pytest.raises(DatasetError, match="not a supported file type"):
        read_resume_file("cv.doc", b"anything")  # old binary Word files are not supported
    with pytest.raises(DatasetError, match="Could not read"):
        read_resume_file("cv.pdf", b"this is not a pdf")
    with pytest.raises(DatasetError, match="Could not read 'cv.docx' as a DOCX"):
        read_resume_file("cv.docx", b"this is not a zip archive")


def test_read_docx_resume_file():
    text = read_resume_file("cv.docx", make_docx("Jane Doe, Data Scientist", "Skills: Python, SQL & Tableau", "Zürich"))
    assert text.splitlines() == ["Jane Doe, Data Scientist", "Skills: Python, SQL & Tableau", "Zürich"]


def test_docx_without_document_or_too_large_is_rejected(monkeypatch):
    import io
    import zipfile

    empty = io.BytesIO()
    with zipfile.ZipFile(empty, "w") as archive:
        archive.writestr("other.xml", "<x/>")
    with pytest.raises(DatasetError, match="Could not read"):
        read_resume_file("cv.docx", empty.getvalue())
    monkeypatch.setattr("resume_matcher.data_loader.MAX_DOCX_XML_BYTES", 10)
    with pytest.raises(DatasetError, match="too large"):
        read_resume_file("cv.docx", make_docx("A long enough paragraph"))


def test_corpus_from_files_uses_file_names_as_ids():
    corpus = corpus_from_files([("a.pdf", "Python developer"), ("b.txt", "Head chef"), ("c.txt", "Python developer")])
    assert corpus.df["resume_id"].tolist() == ["a.pdf", "b.txt"]  # exact duplicate c.txt is dropped
    assert corpus.stats["dropped_duplicates"] == 1
    assert not corpus.has_categories
    with pytest.raises(DatasetError):
        corpus_from_files([])
