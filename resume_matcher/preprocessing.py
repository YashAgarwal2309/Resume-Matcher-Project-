"""Text preprocessing shared by every matching method.

The same functions are applied to resumes and job descriptions so both
sides of a comparison are always cleaned the same way.
"""

from __future__ import annotations

import re

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

URL_RE = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)
EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b")
# Candidate phone numbers: digits with optional separators. We only mask a match
# if it holds at least 10 digits, so date ranges such as "2012 - 2016" are kept.
PHONE_CANDIDATE_RE = re.compile(r"\+?\(?\d[\d\s().-]{7,}\d")

# Keep letters, digits and the symbols that matter in skill names (c++, c#, node.js).
UNWANTED_CHARS_RE = re.compile(r"[^a-z0-9+#.\s]")
# A dot followed by whitespace / end of text is sentence punctuation, not part of a term.
TRAILING_DOT_RE = re.compile(r"\.+(?=\s|$)")
# "+", "#" or "." standing alone (e.g. bullet points) carry no meaning.
LONE_SYMBOL_RE = re.compile(r"(?<!\S)[+#.]+(?!\S)")
WHITESPACE_RE = re.compile(r"\s+")

STOP_WORDS = frozenset(ENGLISH_STOP_WORDS)


def _as_text(text: object) -> str:
    """Convert any input (None, NaN, numbers) into a string safely."""
    if text is None:
        return ""
    if isinstance(text, float) and text != text:  # NaN check without importing numpy
        return ""
    return text if isinstance(text, str) else str(text)


# A UTF-8 lead byte (shown as Â..ô) followed by continuation bytes, as they look after
# being wrongly decoded as Latin-1 (\x80-\xbf) or Windows-1252 (€ ‚ „ … ‘ ’ “ ” • – — ™ ...).
MOJIBAKE_RUN_RE = re.compile(
    "[Â-ô][\u0080-¿€‚ƒ„…†‡ˆ‰Š‹Œ"
    "Ž‘’“”•–—˜™š›œžŸ]+"
)


def _repair_run(match: re.Match) -> str:
    run = match.group()
    for wrong_encoding in ("cp1252", "latin-1"):
        try:
            return run.encode(wrong_encoding).decode("utf-8")
        except (UnicodeEncodeError, UnicodeDecodeError):
            continue
    return run


def fix_mojibake(text: object) -> str:
    """Repair UTF-8 text that was wrongly decoded as Latin-1 or Windows-1252.

    Examples: "NaÃ¯ve" -> "Naïve", "â€¢" -> "•". Some public datasets contain
    this damage. Each broken sequence is fixed separately and only if it
    re-decodes cleanly, so correct characters elsewhere are never touched.
    """
    return MOJIBAKE_RUN_RE.sub(_repair_run, _as_text(text))


def decode_text_file(data: bytes) -> str:
    """Decode an uploaded text file: UTF-8 first, then Windows-1252 (common for files saved on Windows)."""
    try:
        return data.decode("utf-8-sig")  # "-sig" also strips a byte-order mark if present
    except UnicodeDecodeError:
        return data.decode("cp1252", errors="replace")


def normalize_whitespace(text: object) -> str:
    """Collapse all runs of whitespace into single spaces and trim the ends."""
    return WHITESPACE_RE.sub(" ", _as_text(text)).strip()


def clean_text(text: object) -> str:
    """Normalize text for TF-IDF and GloVe.

    Steps: lowercase -> drop URLs and e-mail addresses -> remove unwanted
    characters (keeping + # . inside terms like ``c++``, ``c#``, ``node.js``)
    -> drop sentence-ending dots and lone symbols -> collapse whitespace.
    """
    text = _as_text(text).lower()
    text = URL_RE.sub(" ", text)
    text = EMAIL_RE.sub(" ", text)
    text = UNWANTED_CHARS_RE.sub(" ", text)
    text = TRAILING_DOT_RE.sub(" ", text)
    text = LONE_SYMBOL_RE.sub(" ", text)
    return WHITESPACE_RE.sub(" ", text).strip()


def tokenize(text: object, remove_stop_words: bool = False) -> list[str]:
    """Clean ``text`` and split it into tokens.

    Single-character tokens are kept on purpose because "c" and "r" are
    programming languages.
    """
    tokens = clean_text(text).split()
    if remove_stop_words:
        tokens = [token for token in tokens if token not in STOP_WORDS]
    return tokens


def _mask_phone(match: re.Match) -> str:
    digits = sum(ch.isdigit() for ch in match.group())
    return "[phone]" if digits >= 10 else match.group()


def mask_contact_info(text: object) -> str:
    """Hide e-mail addresses, URLs and phone numbers before displaying text."""
    text = _as_text(text)
    text = EMAIL_RE.sub("[email]", text)
    text = URL_RE.sub("[link]", text)
    return PHONE_CANDIDATE_RE.sub(_mask_phone, text)


def make_preview(text: object, max_chars: int = 300) -> str:
    """Return a short, single-line, contact-masked preview of ``text``."""
    text = normalize_whitespace(mask_contact_info(text))
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars].rsplit(" ", 1)[0]
    return cut + " ..."
