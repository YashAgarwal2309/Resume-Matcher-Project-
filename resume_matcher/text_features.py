"""Text preparation for TF-IDF: keep technical terms intact and spell each skill one way.

    prepare_tfidf_text("Built k8s pipelines with Scikit-Learn and Node JS")
    -> "built kubernetes pipelines with scikit_learn and node.js"

Steps (the same for resumes and job descriptions, so both share one vocabulary):

1. Known skills and their aliases (the vocabulary in skills.py) are replaced by one canonical
   token: "k8s" -> "kubernetes", "sklearn" / "scikit-learn" -> "scikit_learn", "ML" ->
   "machine_learning". Multi-word skills become a single token, so "machine learning" is one
   feature instead of two common words. Where two skills overlap, the longer one wins
   ("SQL Server" rather than "SQL").
2. The usual cleaning: lowercase, drop URLs and e-mail addresses, keep the symbols that matter
   in skill names (c++, c#, node.js), drop punctuation, collapse whitespace.

Related technologies are never merged: Flask stays "flask" and Django stays "django".
"""

from __future__ import annotations

import re
from functools import cache, lru_cache

from resume_matcher.preprocessing import (
    EMAIL_RE,
    LONE_SYMBOL_RE,
    STOP_WORDS,
    TRAILING_DOT_RE,
    URL_RE,
    WHITESPACE_RE,
    _as_text,
)
from resume_matcher.skills import (
    _AFTER,
    _BEFORE,
    _EXACT_AFTER,
    _EXACT_BEFORE,
    SKILLS,
    normalize_for_matching,
)

# sklearn's English stop-word list removes words that carry meaning in job ads: "front-end",
# "back-end", "full-stack", "system design". They are kept for TF-IDF. ("it" and "go" stay stop
# words; the upper-case "IT" and the Go language become their own tokens in step 1 instead.)
KEEP_WORDS = frozenset({"system", "front", "back", "full"})
TFIDF_STOP_WORDS = frozenset(STOP_WORDS - KEEP_WORDS)

# Like preprocessing.UNWANTED_CHARS_RE, but "_" survives because canonical skill tokens use it.
_UNWANTED_CHARS_RE = re.compile(r"[^a-z0-9+#._\s]")
# Short canonical names that would collide with ordinary words or letters.
_TOKEN_OVERRIDES = {"Go": "golang", "R": "r_lang", "C": "c_lang"}
# Exact-case abbreviations that are not skills but would otherwise be lost as stop words.
_EXTRA_EXACT = {"IT": "information_technology"}


def skill_token(name: str) -> str:
    """The single TF-IDF token for a skill name: "Machine Learning" -> "machine_learning"."""
    if name in _TOKEN_OVERRIDES:
        return _TOKEN_OVERRIDES[name]
    return re.sub(r"[^a-z0-9+#.]+", "_", name.lower()).strip("_")


def _alias_key(text: str) -> str:
    return " ".join(normalize_for_matching(text).split())


@cache
def _canonical_patterns() -> tuple[re.Pattern, dict[str, str], re.Pattern | None, dict[str, str]]:
    """One case-insensitive pattern for every alias, plus one for the exact-case names."""
    alias_to_token: dict[str, str] = {}
    exact_to_token: dict[str, str] = {}
    for skill in SKILLS:
        token = skill_token(skill.name)
        for alias in skill.aliases:
            alias_to_token.setdefault(_alias_key(alias), token)
        for spelling in skill.exact:
            exact_to_token.setdefault(spelling, token)
    for spelling, token in _EXTRA_EXACT.items():
        exact_to_token.setdefault(spelling, token)
    # Longest alternatives first: at a given position the regex takes the first one that fits.
    bodies = []
    for key in sorted(alias_to_token, key=len, reverse=True):
        body = r"\s+".join(re.escape(word) for word in key.split())
        bodies.append(body + (r"(?:e?s)?" if key[-1].isalpha() else ""))
    alias_re = re.compile(_BEFORE + "(?:" + "|".join(bodies) + ")" + _AFTER)
    exact_re = None
    if exact_to_token:
        alternatives = "|".join(re.escape(e) for e in sorted(exact_to_token, key=len, reverse=True))
        exact_re = re.compile(_EXACT_BEFORE + "(?:" + alternatives + ")" + _EXACT_AFTER)
    return alias_re, alias_to_token, exact_re, exact_to_token


def _lookup(matched: str, table: dict[str, str]) -> str | None:
    key = _alias_key(matched)
    if key in table:
        return table[key]
    for suffix in ("es", "s"):  # plural forms ("microservices", "REST APIs")
        if key.endswith(suffix) and key[: -len(suffix)] in table:
            return table[key[: -len(suffix)]]
    return None


def canonicalize_skills(text: object) -> str:
    """Replace every recognised skill mention with its canonical token (see module docstring)."""
    text = _as_text(text)
    if not text:
        return ""
    return _canonicalize_cached(text)


@lru_cache(maxsize=4096)
def _canonicalize_cached(text: str) -> str:
    """Cached, because TF-IDF and GloVe (both used by Hybrid) prepare the same resumes."""
    alias_re, alias_to_token, exact_re, exact_to_token = _canonical_patterns()
    normalized = normalize_for_matching(text)  # same length as text, so spans line up
    spans = []
    for match in alias_re.finditer(normalized):
        token = _lookup(match.group(), alias_to_token)
        if token:
            spans.append((match.start(), match.end(), token))
    if exact_re is not None:
        for match in exact_re.finditer(text):
            token = exact_to_token.get(match.group())
            if token:
                spans.append((match.start(), match.end(), token))
    # Longest span wins where two overlap; then rebuild the text left to right.
    spans.sort(key=lambda s: (s[0] - s[1], s[0]))
    taken: list[tuple[int, int, str]] = []
    for start, end, token in spans:
        if all(end <= t_start or start >= t_end for t_start, t_end, _ in taken):
            taken.append((start, end, token))
    taken.sort()
    out, position = [], 0
    for start, end, token in taken:
        out.append(text[position:start])
        out.append(f" {token} ")
        position = end
    out.append(text[position:])
    return "".join(out)


def prepare_tfidf_text(text: object) -> str:
    """Resume or job description -> cleaned text for the TF-IDF vectorizer (see module docstring)."""
    text = _as_text(text)
    text = URL_RE.sub(" ", text)
    text = EMAIL_RE.sub(" ", text)
    text = canonicalize_skills(text).lower()
    text = _UNWANTED_CHARS_RE.sub(" ", text)
    text = TRAILING_DOT_RE.sub(" ", text)
    text = LONE_SYMBOL_RE.sub(" ", text)
    return WHITESPACE_RE.sub(" ", text).strip()


def tfidf_tokens(text: str) -> list[str]:
    """Tokenizer for prepared text: whitespace split, dropping stop words and number-only tokens
    (years, phone fragments), which only add noise. Terms like "c++", "s3" or "html5" stay."""
    return [t for t in text.split() if t not in TFIDF_STOP_WORDS and any(c.isalpha() for c in t)]
