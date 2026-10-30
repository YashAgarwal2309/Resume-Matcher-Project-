"""Split resumes into sections and model-sized chunks for Sentence-BERT.

Sentence-BERT models read a limited number of word-pieces (512 for BAAI/bge-small-en-v1.5) and
silently ignore the rest. Many resumes are longer than that (about 43% of the tech dataset and
90% of the all-industries dataset), so a single embedding would describe only their first part.
Instead each resume is split into chunks that fit the model:

    chunks = chunk_text(text, max_tokens=256, count_tokens=model_token_counter)
    [(section, chunk_text), ...]    # in reading order; every word of the resume is in one chunk

1. Section headings ("SKILLS", "Work History", "Education Details", ...) start a new section, so
   a chunk never mixes the end of one section with the start of the next.
2. Each section is cut into chunks of at most ``max_tokens`` at sentence, bullet or field
   boundaries; a single over-long sentence is cut at word boundaries.

Section detection is a heuristic on flattened text (line breaks are not kept), so a heading word
used in a sentence can occasionally start a section. That only moves a chunk boundary.
"""

from __future__ import annotations

import re
from collections.abc import Callable

from resume_matcher.preprocessing import _as_text, normalize_whitespace

SECTION_OTHER = "other"
SECTION_SUMMARY = "summary"
SECTION_SKILLS = "skills"
SECTION_EXPERIENCE = "experience"
SECTION_PROJECTS = "projects"
SECTION_EDUCATION = "education"
SECTION_CERTIFICATIONS = "certifications"

# Heading text -> section. Matched in upper case anywhere, or in Title Case when it is a typical
# heading phrase (single title-case words such as "Experience" are too often ordinary words).
_HEADINGS = {
    "PROFESSIONAL SUMMARY": SECTION_SUMMARY, "CAREER OBJECTIVE": SECTION_SUMMARY, "SUMMARY": SECTION_SUMMARY,
    "OBJECTIVE": SECTION_SUMMARY, "PROFILE SUMMARY": SECTION_SUMMARY, "ABOUT": SECTION_SUMMARY,
    "HIGHLIGHTS": SECTION_SUMMARY, "TECHNICAL SKILLS": SECTION_SKILLS, "SKILLS": SECTION_SKILLS,
    "SKILL DETAILS": SECTION_SKILLS, "CORE COMPETENCIES": SECTION_SKILLS, "WORK EXPERIENCE": SECTION_EXPERIENCE,
    "PROFESSIONAL EXPERIENCE": SECTION_EXPERIENCE, "EXPERIENCE": SECTION_EXPERIENCE, "WORK HISTORY": SECTION_EXPERIENCE,
    "EMPLOYMENT HISTORY": SECTION_EXPERIENCE, "COMPANY DETAILS": SECTION_EXPERIENCE, "PROJECTS": SECTION_PROJECTS,
    "ACADEMIC PROJECTS": SECTION_PROJECTS, "EDUCATION": SECTION_EDUCATION, "EDUCATION DETAILS": SECTION_EDUCATION,
    "CERTIFICATIONS": SECTION_CERTIFICATIONS, "CERTIFICATION": SECTION_CERTIFICATIONS,
    "ACCOMPLISHMENTS": SECTION_OTHER, "ACHIEVEMENTS": SECTION_OTHER, "INTERESTS": SECTION_OTHER,
    "HOBBIES": SECTION_OTHER, "LANGUAGES": SECTION_OTHER, "PERSONAL DETAILS": SECTION_OTHER,
}
_TITLE_CASE_HEADINGS = (
    "Professional Summary", "Career Objective", "Technical Skills", "Skill Details", "Work Experience",
    "Professional Experience", "Work History", "Employment History", "Company Details", "Education Details",
    "Academic Projects", "Certifications", "Accomplishments", "Highlights", "Summary", "Skills", "Education",
)
_HEADING_RE = re.compile(
    r"(?<![A-Za-z])(?:"
    + "|".join(re.escape(h) for h in sorted(_HEADINGS, key=len, reverse=True))
    + r")(?![a-z])|(?<![A-Za-z])(?:"
    + "|".join(re.escape(h) for h in sorted(_TITLE_CASE_HEADINGS, key=len, reverse=True))
    + r")\b(?=\s*[:•\-–A-Z0-9])"
)
# Places where a chunk may end: after a full stop, bullet, pipe, semicolon or line break.
_BREAK_RE = re.compile(r"(?<=[.!?;|•▪●])\s+|\s+(?=[•▪●])")


def _words_estimate(text: str) -> int:
    """Rough word-piece count when no tokenizer is given (English text: ~1.3 pieces per word)."""
    return int(len(text.split()) * 1.3) + 1


def split_sections(text: object) -> list[tuple[str, str]]:
    """``(section, text)`` pairs in reading order. Text before the first heading is "other"."""
    text = normalize_whitespace(_as_text(text))
    if not text:
        return []
    sections: list[tuple[str, str]] = []
    position, current = 0, SECTION_OTHER
    for match in _HEADING_RE.finditer(text):
        body = text[position : match.start()].strip()
        if body:
            sections.append((current, body))
        current = _HEADINGS.get(match.group().upper(), SECTION_OTHER)
        position = match.start()  # keep the heading in its section's text
    tail = text[position:].strip()
    if tail:
        sections.append((current, tail))
    return sections


def _pieces(text: str, max_tokens: int, count: Callable[[str], int]) -> list[tuple[str, int]]:
    """Sentence-like pieces with their token counts, each within max_tokens (long ones are cut at
    word boundaries)."""
    out = []
    for piece in (p.strip() for p in _BREAK_RE.split(text)):
        if not piece:
            continue
        size = count(piece)
        if size <= max_tokens:
            out.append((piece, size))
            continue
        current, current_size = [], 0
        for word in piece.split():
            word_size = count(word)
            if current and current_size + word_size > max_tokens:
                out.append((" ".join(current), current_size))
                current, current_size = [], 0
            current.append(word)
            current_size += word_size
        if current:
            out.append((" ".join(current), current_size))
    return out


def chunk_text(
    text: object, max_tokens: int = 256, count_tokens: Callable[[str], int] | None = None
) -> list[tuple[str, str]]:
    """``(section, chunk)`` pairs covering the whole text, each chunk within about ``max_tokens``.

    Pieces are counted once and their counts added up, which matches the model's count closely
    (joining pieces adds no word-pieces beyond the spaces between them).
    """
    count = count_tokens or _words_estimate
    chunks: list[tuple[str, str]] = []
    for section, body in split_sections(text):
        current: list[str] = []
        current_size = 0
        for piece, size in _pieces(body, max_tokens, count):
            if current and current_size + size > max_tokens:
                chunks.append((section, " ".join(current)))
                current, current_size = [], 0
            current.append(piece)
            current_size += size
        if current:
            chunks.append((section, " ".join(current)))
    return chunks
