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


# ---- Dashboard, comparison and shortlist

REQUIRED_JOB = "Python developer. Required: Python, SQL, Django, Docker and AWS."


def match_tfidf(at, files=None, job=REQUIRED_JOB):
    upload(at, files or resume_files())
    at.text_area(key="jd_text").input(job)
    at.selectbox(key="method").select("tfidf")
    click_match(at)
    assert not at.exception and not at.error


def compare_select(at):
    return next(m for m in at.multiselect if m.key.startswith("compare_sel_"))


def compare_button(at):
    return next(b for b in at.button if b.label == "Compare Candidates")


def downloads(at):
    return [d.proto.label for d in at.get("download_button")]


def shortlist_toggle(at, where, resume_id):
    return next(t for t in at.toggle if t.key.startswith(f"sl_{where}_") and t.key.endswith(f"_{resume_id}"))


def metrics(at):
    return {m.label: m.value for m in at.metric}


def test_header_shows_product_and_selected_algorithm(app):
    text = " ".join(m.value for m in app.markdown)
    assert "AI-Powered Recruitment Analytics" in text and "Hybrid" in text
    app.selectbox(key="method").select("tfidf").run()
    assert '<span class="name">TF-IDF</span>' in " ".join(m.value for m in app.markdown)


def test_kpis_and_scores_come_from_the_backend(app):
    from resume_matcher.comparison import DEFAULT_STRONG_THRESHOLDS
    from resume_matcher.data_loader import corpus_from_files
    from resume_matcher.matcher import TfidfMatcher

    match_tfidf(app)
    expected = TfidfMatcher(corpus_from_files([(n, d.decode()) for n, d, _ in resume_files()])).fit().match(
        REQUIRED_JOB, top_k=5
    )
    table = app.dataframe[0].value
    assert table["resume_id"].tolist() == expected["resume_id"].tolist()
    assert table["score"].tolist() == expected["score"].tolist()  # same values as BaseMatcher.match
    kpis = metrics(app)
    assert kpis["Resumes processed"] == "5" and kpis["Candidates ranked"] == "5"
    strong = int((expected["score"] >= DEFAULT_STRONG_THRESHOLDS["tfidf"]).sum())
    assert kpis["Strong matches"] == str(strong)
    assert kpis["Average similarity"] == f"{expected['score'].mean():.1f}"


def test_strong_match_threshold_is_configurable(app):
    match_tfidf(app)
    app.number_input(key="threshold_tfidf").set_value(0.0).run()
    assert metrics(app)["Strong matches"] == "5"
    app.number_input(key="threshold_tfidf").set_value(100.0).run()
    assert metrics(app)["Strong matches"] == "0"
    assert app.dataframe[0].value["rank"].tolist() == [1, 2, 3, 4, 5]  # ranking unchanged


def test_insights_tab_draws_charts_from_the_run(app):
    match_tfidf(app)
    # scores, distribution, similarity vs coverage, requirements per resume, coverage per requirement
    assert len(app.get("vega_lite_chart")) == 5


@pytest.mark.parametrize("count", [2, 3, 4, 5])
def test_two_to_five_candidates_can_be_compared(app, count):
    match_tfidf(app)
    ranked = app.dataframe[0].value["resume_id"].tolist()
    compare_select(app).set_value(ranked[:count]).run()
    assert not compare_button(app).disabled
    compare_button(app).click().run()
    assert not app.exception
    assert app.session_state["view"].endswith("Compare")  # the button opens the Compare tab
    assert "Download Comparison CSV" in downloads(app) and "Download Skill Matrix CSV" in downloads(app)
    assert downloads(app).count("Original file") == count
    matrix = app.dataframe[1].value
    assert len(matrix.columns) == count and matrix.index.tolist() == ["Python", "SQL", "Django", "Docker", "AWS"]
    assert matrix.loc["Django", f"#1 {ranked[0]}"] == ("Found" if ranked[0] == "101.txt" else "Not found")


def test_comparison_needs_at_least_two_candidates(app):
    match_tfidf(app)
    assert compare_button(app).disabled  # nothing selected yet
    compare_select(app).set_value(["101.txt"]).run()
    assert compare_button(app).disabled
    assert any("Select 1 more candidate" in i.value for i in app.info)
    assert "Download Comparison CSV" not in downloads(app)


def test_more_than_five_candidates_are_capped(app):
    match_tfidf(app, files=[*resume_files(), ("106.txt", b"Python SQL analyst", "text/plain")])
    show_top(app, 6)  # all six in the top K, so all six can be picked
    compare_select(app).set_value([f"{i}.txt" for i in range(101, 107)]).run()  # the browser stops at 5
    assert len(compare_select(app).value) == 5
    assert not compare_button(app).disabled


def test_add_to_comparison_buttons_fill_the_selection(app):
    match_tfidf(app)
    for _ in range(2):
        next(b for b in app.button if b.label == "Add to comparison").click().run()
    assert len(compare_select(app).value) == 2
    assert not compare_button(app).disabled


def test_new_run_clears_stale_comparison(app):
    match_tfidf(app)
    old_key = compare_select(app).key
    compare_select(app).set_value(["101.txt", "104.txt"]).run()
    compare_button(app).click().run()
    click_match(app)  # same inputs, but a new run replaces the results
    assert compare_select(app).key != old_key and compare_select(app).value == []
    assert old_key not in app.session_state
    assert app.session_state["view"].endswith("Ranking")
    assert "Download Comparison CSV" not in downloads(app)


def test_shortlist_is_shown_filtered_and_shared_with_comparison(app):
    match_tfidf(app)
    shortlist_toggle(app, "rank", "103.txt").set_value(True).run()
    table = app.dataframe[0].value.set_index("resume_id")
    assert table.loc["103.txt", "shortlisted"] and not table.loc["101.txt", "shortlisted"]
    app.button_group[0].set_value(["Shortlisted"]).run()
    assert app.dataframe[0].value["resume_id"].tolist() == ["103.txt"]
    app.button_group[0].set_value([]).run()
    compare_select(app).set_value(["101.txt", "103.txt"]).run()
    assert shortlist_toggle(app, "compare", "103.txt").value  # same shortlist in the comparison
    shortlist_toggle(app, "compare", "103.txt").set_value(False).run()
    assert not shortlist_toggle(app, "rank", "103.txt").value
    assert not app.dataframe[0].value["shortlisted"].any()


def test_shortlist_survives_a_rerun_but_not_new_uploads(app):
    match_tfidf(app)
    shortlist_toggle(app, "rank", "102.txt").set_value(True).run()
    click_match(app)  # re-run with the same resumes and job description
    assert shortlist_toggle(app, "rank", "102.txt").value
    upload(app, resume_files()[:3])
    match_tfidf(app, files=resume_files()[:3])
    assert not shortlist_toggle(app, "rank", "102.txt").value


def test_search_and_requirement_filters(app):
    match_tfidf(app)
    app.text_input(key="rank_search").input("104").run()
    assert app.dataframe[0].value["resume_id"].tolist() == ["104.txt"]
    app.text_input(key="rank_search").input("").run()
    next(m for m in app.multiselect if m.key.startswith("rank_must_")).set_value(["Django"]).run()
    assert app.dataframe[0].value["resume_id"].tolist() == ["101.txt"]
    app.text_input(key="rank_search").input("nobody").run()
    assert any("No candidates match" in i.value for i in app.info)


def test_editing_the_job_marks_results_as_outdated(app):
    match_tfidf(app)
    assert not any("changed after this ranking" in i.value for i in app.info)
    app.text_area(key="jd_text").input("Required: cooking").run()
    assert any("changed after this ranking" in i.value for i in app.info)
    assert app.dataframe  # still shown, clearly labelled, until Match Resumes is clicked again


def test_duplicate_file_names_are_separate_candidates(app):
    files = [("cv.txt", b"Python developer with Django and SQL", "text/plain"),
             ("cv.txt", b"Head chef managing a kitchen", "text/plain")]
    match_tfidf(app, files=files)
    ids = app.dataframe[0].value["resume_id"].tolist()
    assert sorted(ids) == ["cv.txt-0", "cv.txt-1"]
    compare_select(app).set_value(ids).run()
    assert downloads(app).count("Original file") == 2


def test_uploaded_resumes_are_not_cached_on_disk():
    import app as app_module
    from resume_matcher.config import get_settings

    assert get_settings().cache_dir is not None  # evaluate.py may cache the public datasets
    assert app_module.matcher_settings(get_settings()).cache_dir is None  # the app never does


# ---- All resumes ranked, result limits, pages, coverage

def many_resumes(n):
    from tests.test_tfidf import synthetic_resumes

    return [(name, text.encode(), "text/plain") for name, text in synthetic_resumes(n)]


def ranking_head(at):
    return next(m.value for m in at.markdown if "ranked resumes" in m.value)


def show_top(at, value):
    at.number_input(key="top_k_results").set_value(value).run()


def test_every_uploaded_resume_is_ranked_and_limits_only_change_the_view(app):
    match_tfidf(app, files=many_resumes(60), job="Required: Python, Django, PostgreSQL, Docker and AWS.")
    kpis = metrics(app)
    assert kpis["Resumes uploaded"] == "60" and kpis["Resumes processed"] == "60"
    assert kpis["Failed files"] == "0" and kpis["Candidates ranked"] == "60"
    assert "Showing 5 of 60 ranked resumes" in ranking_head(app)  # top 5 by default
    assert app.dataframe[0].value["rank"].tolist() == [1, 2, 3, 4, 5]
    show_top(app, 30)
    assert app.dataframe[0].value["rank"].tolist() == list(range(1, 26))  # page 1 of 2, 25 per page
    app.number_input(key="rank_page_2").set_value(2).run()
    assert app.dataframe[0].value["rank"].tolist() == list(range(26, 31))
    show_top(app, 60)
    assert "Showing 60 of 60 ranked resumes" in ranking_head(app)
    app.number_input(key="rank_page_3").set_value(3).run()
    assert app.dataframe[0].value["rank"].tolist() == list(range(51, 61))
    assert len(compare_select(app).options) == 60  # every ranked resume can be compared


def test_minimum_score_filter_is_optional_and_never_unranks(app):
    match_tfidf(app, files=many_resumes(20), job="Required: Python, Django, PostgreSQL, Docker and AWS.")
    show_top(app, 20)
    everyone = app.dataframe[0].value
    assert len(everyone) == 20 and (everyone["score"] == 0).any()  # zero-similarity resumes are kept
    app.number_input(key="rank_min_score").set_value(10.0).run()
    shown = app.dataframe[0].value
    assert (shown["score"] >= 10).all() and len(shown) < 20
    assert metrics(app)["Candidates ranked"] == "20"


def test_failed_and_duplicate_files_are_counted(app):
    files = [*resume_files(), ("broken.pdf", b"not a pdf", "application/pdf"), ("copy.txt", resume_files()[0][1], "text/plain")]
    match_tfidf(app, files=files)
    kpis = metrics(app)
    assert kpis["Resumes uploaded"] == "7" and kpis["Failed files"] == "1"
    assert kpis["Resumes processed"] == "5" and kpis["Candidates ranked"] == "5"


def test_docx_resumes_are_read(app):
    from tests.conftest import make_docx

    docx = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    match_tfidf(app, files=[("nurse.docx", make_docx("Registered nurse", "Hospital patient care"), docx),
                            ("chef.docx", make_docx("Head chef managing a kitchen"), docx)],
                job="Hospital nurse for patient care")
    assert app.dataframe[0].value["resume_id"].iloc[0] == "nurse.docx"


def test_similarity_and_coverage_are_shown_separately(app):
    files = [("flask.txt", b"Python developer building Flask APIs with MySQL", "text/plain"),
             ("django.txt", b"Python developer building Django APIs with PostgreSQL. 5 years of experience", "text/plain"),
             ("chef.txt", b"Head chef managing a kitchen", "text/plain")]
    match_tfidf(app, files=files, job="Required: Python, Django and PostgreSQL. 3+ years of experience.")
    table = app.dataframe[0].value.set_index("resume_id")
    assert table.loc["django.txt", "coverage"] == 100.0
    assert table.loc["flask.txt", "coverage"] == 25.0  # Python only; Flask is related to Django, not a match
    assert list(table.loc["flask.txt", "related"]) == ["Django", "PostgreSQL"]
    assert table.loc["chef.txt", "coverage"] == 0.0 and table.loc["chef.txt", "score"] == 0.0
    assert "Average coverage" in metrics(app)


def test_fresh_run_is_not_marked_as_outdated(app):
    match_tfidf(app)
    assert not any("changed after this ranking" in i.value for i in app.info)
    app.selectbox(key="method").select("hybrid").run()  # Hybrid has weight inputs; switching marks the run outdated
    assert any("changed after this ranking" in i.value for i in app.info)


# ---- "Number of Top Resumes to Display"

def test_top_resumes_input_shows_the_best_k_of_the_ranked_resumes(app):
    match_tfidf(app, files=many_resumes(7), job="Required: Python, Django, PostgreSQL, Docker and AWS.")
    field = app.number_input(key="top_k_results")
    assert field.label == "Number of Top Resumes to Display" and field.value == 5 and field.min == 1
    show_top(app, 7)
    full = app.dataframe[0].value  # all 7, best first
    assert full["rank"].tolist() == list(range(1, 8))
    for entered, expected in [(5, 5), (3, 3), (7, 7), (10, 7)]:  # 10 > 7 available: capped at 7
        show_top(app, entered)
        shown = app.dataframe[0].value
        assert f"Showing {expected} of 7 ranked resumes" in ranking_head(app)
        assert shown["resume_id"].tolist() == full["resume_id"].head(expected).tolist()  # best first
        assert shown["score"].tolist() == full["score"].head(expected).tolist()  # scores unchanged
        assert app.number_input(key="top_k_results").value == expected  # the field shows the capped number
        assert metrics(app)["Candidates ranked"] == "7"  # every resume is still ranked
    assert any("Only 7 resumes are ranked, so all 7 are shown" in c.value for c in app.caption)  # after 10
    show_top(app, 5)
    assert not any("Only 7 resumes are ranked" in c.value for c in app.caption)


# Below the minimum of 1, Streamlit rejects the number (the default stays); an emptied field restores the
# default; decimals are rounded down.
@pytest.mark.parametrize("entered, expected", [(0, 5), (-4, 5), (2.7, 2), (None, 5)])
def test_top_resumes_input_handles_invalid_numbers(app, entered, expected):
    match_tfidf(app, files=many_resumes(7), job="Required: Python, Django, PostgreSQL, Docker and AWS.")
    show_top(app, entered)
    assert not app.exception
    assert len(app.dataframe[0].value) == expected and f"Showing {expected} of 7" in ranking_head(app)
    assert app.number_input(key="top_k_results").value == expected


def test_top_resumes_choice_is_kept_when_the_page_changes(app):
    match_tfidf(app, files=many_resumes(60), job="Required: Python, Django, PostgreSQL, Docker and AWS.")
    show_top(app, 50)
    app.number_input(key="rank_page_2").set_value(2).run()
    assert app.dataframe[0].value["rank"].tolist() == list(range(26, 51))
    assert app.number_input(key="top_k_results").value == 50


def test_top_resumes_limits_the_csv_and_follows_filters(app):
    match_tfidf(app, files=many_resumes(20), job="Required: Python, Django, PostgreSQL, Docker and AWS.")
    show_top(app, 3)
    assert "Download top 3 as CSV" in downloads(app)
    top3 = app.dataframe[0].value["resume_id"].tolist()
    app.text_input(key="rank_search").input(top3[0]).run()  # filters work inside the top K
    assert "Showing 3 of 20 ranked resumes" in ranking_head(app)
    assert any("1 of the top 3 match the search and filters" in c.value for c in app.caption)
    assert app.dataframe[0].value["resume_id"].tolist() == [top3[0]]


# ---- The top K drives the whole dashboard (KPIs, ranking, insights, compare, CSV)

TOP_K_JOB = "Required: Python, Django, PostgreSQL, Docker and AWS."


def chart_data(at):
    """The data behind each Insights chart, in drawing order: scores, distribution, similarity vs
    coverage, requirements per resume, coverage per requirement."""
    import pyarrow as pa

    return [
        [pa.ipc.open_stream(d.data.data).read_all().to_pandas() for d in chart.proto.datasets]
        for chart in at.get("vega_lite_chart")
    ]


@pytest.fixture
def csv_spy(monkeypatch):
    """Records the rows passed to the results CSV (the app imports results_csv on every run)."""
    from resume_matcher import comparison

    exported = []
    real = comparison.results_csv
    monkeypatch.setattr(comparison, "results_csv", lambda rows, *a, **k: exported.append(rows) or real(rows, *a, **k))
    return exported


def full_ranking(at, n):
    show_top(at, n)
    return at.dataframe[0].value.reset_index(drop=True)


@pytest.mark.parametrize("k", [5, 3, 7])
def test_every_dashboard_component_uses_only_the_top_k(app, csv_spy, k):
    from resume_matcher.comparison import DEFAULT_STRONG_THRESHOLDS

    match_tfidf(app, files=many_resumes(7), job=TOP_K_JOB)
    full = full_ranking(app, 7)  # every resume is ranked first
    top = full.head(k)
    ids = top["resume_id"].tolist()
    show_top(app, k)

    # Ranking: the K best, in rank order, with unchanged scores
    table = app.dataframe[0].value
    assert table["resume_id"].tolist() == ids and table["rank"].tolist() == list(range(1, k + 1))
    assert table["score"].tolist() == top["score"].tolist()
    assert f"Showing {k} of 7 ranked resumes" in ranking_head(app)

    # KPIs: counts cover the run, scores and coverage only the top K
    kpis = metrics(app)
    assert kpis["Candidates ranked"] == "7" and kpis["Resumes processed"] == "7"
    assert kpis["Average similarity"] == f"{top['score'].mean():.1f}"
    assert kpis["Average coverage"] == f"{top['coverage'].mean():.0f}%"
    assert kpis["Strong matches"] == str(int((top["score"] >= DEFAULT_STRONG_THRESHOLDS["tfidf"]).sum()))

    # Insights: every chart is drawn from the same K resumes
    scores, distribution, scatter, per_resume, coverage = chart_data(app)
    assert sorted(scores[0]["resume_id"]) == sorted(ids)
    assert distribution[0]["candidates"].sum() == k
    assert sorted(scatter[0]["resume_id"]) == sorted(ids)
    assert per_resume[0]["count"].sum() == k * 5  # found + not found = 5 requirements per resume
    expected = {name: sum(name in matched for matched in top["matched"]) for name in coverage[0]["requirement"]}
    assert dict(zip(coverage[0]["requirement"], coverage[0]["resumes"])) == expected
    assert coverage[0]["resumes"].max() <= k

    # Compare: only the top K can be picked
    assert len(compare_select(app).options) == k
    assert all(any(rid in option for rid in ids) for option in compare_select(app).options)

    # CSV: exactly the top K
    assert f"Download top {k} as CSV" in downloads(app)
    assert csv_spy[-1]["resume_id"].tolist() == ids


def test_compare_tab_compares_only_top_k_candidates(app):
    match_tfidf(app, files=many_resumes(7), job=TOP_K_JOB)
    ids = full_ranking(app, 7)["resume_id"].tolist()
    show_top(app, 5)
    compare_select(app).set_value(ids[3:5]).run()  # ranks 4 and 5
    compare_button(app).click().run()
    assert not app.exception and app.session_state["view"].endswith("Compare")
    assert [c.split(" ", 1)[1] for c in app.dataframe[-1].value.columns] == ids[3:5]
    show_top(app, 3)  # ranks 4 and 5 leave the top K: they leave the comparison too
    assert compare_select(app).value == []
    assert any("Select 2 more candidates" in i.value for i in app.info)
    assert "Download Comparison CSV" not in downloads(app)


def test_method_comparison_shows_only_top_k_rows(app):
    match_tfidf(app, files=many_resumes(7), job=TOP_K_JOB)
    full = full_ranking(app, 7)
    run_id = next(b.key for b in app.button if b.key and b.key.startswith("rank_all_")).removeprefix("rank_all_")
    # Stored rankings as _rank_all_methods saves them (here TF-IDF only, so no model is loaded)
    app.session_state[f"_method_ranks_{run_id}"] = ({"tfidf": full[["resume_id", "rank", "score"]]}, {})
    show_top(app, 4)
    by_method = next(df.value for df in app.dataframe if "rank_spread" in df.value.columns)
    assert by_method["resume_id"].tolist() == full["resume_id"].head(4).tolist()
    assert by_method["rank_tfidf"].tolist() == [1, 2, 3, 4]  # ranks among all 7, unchanged


def test_shortlist_outside_top_k_is_kept_but_left_out(app):
    match_tfidf(app, files=many_resumes(7), job=TOP_K_JOB)
    fifth = full_ranking(app, 7)["resume_id"].iloc[4]
    show_top(app, 5)
    shortlist_toggle(app, "rank", fifth).set_value(True).run()
    show_top(app, 3)
    app.pills(key="rank_filters").set_value(["Shortlisted"]).run()
    assert any("No candidates match" in i.value for i in app.info)  # rank 5 is outside the top 3
    show_top(app, 5)
    assert app.dataframe[0].value["resume_id"].tolist() == [fifth]  # still shortlisted, not discarded


def test_new_ranking_method_makes_a_new_top_k(app):
    match_tfidf(app, files=many_resumes(7), job=TOP_K_JOB)
    show_top(app, 3)
    first_run = compare_select(app).key
    click_match(app)  # a new run (here the same method) builds a new ranking and a new top-K selection
    assert compare_select(app).key != first_run
    assert app.number_input(key="top_k_results").value == 3 and len(app.dataframe[0].value) == 3
