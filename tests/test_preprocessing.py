import math

import pytest

from resume_matcher.preprocessing import (
    clean_text,
    decode_text_file,
    fix_mojibake,
    make_preview,
    mask_contact_info,
    normalize_whitespace,
    tokenize,
)


@pytest.mark.parametrize("value", [None, math.nan, "", "   \n\t "])
def test_clean_text_handles_null_and_blank(value):
    assert clean_text(value) == ""


def test_clean_text_handles_non_string():
    assert clean_text(12345) == "12345"


def test_clean_text_lowercases_and_collapses_whitespace():
    assert clean_text("  Hello,\n\nWORLD!!  ") == "hello world"


def test_clean_text_keeps_technical_terms():
    cleaned = clean_text("Skills: C++, C#, Node.js, .NET and Python 3.10. Also R.")
    tokens = cleaned.split()
    for term in ["c++", "c#", "node.js", ".net", "python", "3.10", "r"]:
        assert term in tokens, term


def test_clean_text_removes_urls_and_emails():
    cleaned = clean_text("Email me at john.doe@mail.com or visit https://linkedin.com/in/john")
    assert "john.doe" not in cleaned
    assert "linkedin" not in cleaned


def test_clean_text_drops_lone_symbols():
    assert clean_text("• Java + SQL # . ") == "java sql"


def test_tokenize_stop_words():
    assert tokenize("The engineer and the manager", remove_stop_words=True) == ["engineer", "manager"]
    assert "the" in tokenize("The engineer")


def test_normalize_whitespace():
    assert normalize_whitespace(" a \n b\t c ") == "a b c"
    assert normalize_whitespace(None) == ""


def test_mask_contact_info():
    text = "Call +1 (555) 123-4567, mail a@b.com, see www.site.com. Worked 2012 - 2016."
    masked = mask_contact_info(text)
    assert "[phone]" in masked and "[email]" in masked and "[link]" in masked
    assert "2012 - 2016" in masked  # date ranges are not phone numbers


def test_make_preview_truncates_on_word_boundary():
    preview = make_preview("word " * 100, max_chars=30)
    assert preview.endswith(" ...")
    assert len(preview) <= 34


@pytest.mark.parametrize(
    "broken, fixed",
    [("NaÃ¯ve Bayes", "Naïve Bayes"), ("Skills â€¢ Java", "Skills • Java"), ("plain text", "plain text"), (None, "")],
)
def test_fix_mojibake(broken, fixed):
    assert fix_mojibake(broken) == fixed


def test_fix_mojibake_leaves_undecodable_text_alone():
    assert fix_mojibake("Ã alone") == "Ã alone"


def test_fix_mojibake_mixed_with_correct_characters():
    assert fix_mojibake("Café • PowerPoint â€¢Excel, studentsâ€™ needs (1995â€”2003)") == (
        "Café • PowerPoint •Excel, students’ needs (1995—2003)"
    )


@pytest.mark.parametrize(
    "data, expected",
    [
        ("Développeur Java – café".encode("utf-8"), "Développeur Java – café"),
        ("\ufeffBackend Developer".encode("utf-8"), "Backend Developer"),  # UTF-8 with byte-order mark
        ("Développeur Java – café".encode("cp1252"), "Développeur Java – café"),  # saved on Windows
        (b"", ""),
    ],
)
def test_decode_text_file(data, expected):
    assert decode_text_file(data) == expected
