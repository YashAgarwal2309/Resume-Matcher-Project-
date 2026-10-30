"""Job-description requirement coverage: which requirements a resume shows evidence for.

This is a second, separate measure next to the similarity score. Similarity compares whole
documents; coverage checks the job's individual requirements one by one:

    job = job_requirements(job_description)          # skills, degree level, years of experience
    report = assess(job, resume_text)
    report.score      # percent of requirements matched (None when the job lists none)
    report.checks     # one RequirementCheck per requirement, with the resume quote as evidence

Each requirement ends up as:

* matched - the resume names it, either exactly ("Python") or by a recognised alias ("k8s" for
  Kubernetes, "sklearn" for scikit-learn); a degree at or above the required level; or a stated
  number of years at or above the required number. The quote from the resume is kept as evidence.
* related - not found, but the resume names a related technology (Flask when Django is required).
  Related skills are shown for the reviewer to judge and never count as matched.
* missing - no evidence in the resume text. The candidate may still have it.

Nothing here is a hiring probability, and nothing is inferred beyond what the text states.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import cache

from resume_matcher.preprocessing import (
    _as_text,
    mask_contact_info,
    normalize_whitespace,
)
from resume_matcher.skills import (
    Requirement,
    extract_requirements,
    normalize_for_matching,
    vocabulary,
)

STATUS_MATCHED = "matched"
STATUS_RELATED = "related"
STATUS_MISSING = "missing"

MATCH_EXACT = "exact"
MATCH_ALIAS = "alias"
MATCH_STATED = "stated"  # degree or years of experience as written in the resume

KIND_SKILL = "skill"
KIND_EDUCATION = "education"
KIND_EXPERIENCE = "experience"

# Technologies that solve similar problems but are NOT interchangeable. A resume that names one
# of them when the job asks for another gets a "related (unverified)" note, never a match.
RELATED_GROUPS: tuple[tuple[str, ...], ...] = (
    ("Django", "Flask", "FastAPI"),
    ("React", "Angular", "Vue.js"),
    ("React Native", "Flutter"),
    ("Android", "iOS"),
    ("Node.js", "Express.js"),
    ("Spring Boot", "Spring Framework"),
    ("Java", "Kotlin", "Scala"),
    ("JavaScript", "TypeScript"),
    ("C", "C++"),
    ("SQL", "PL/SQL", "T-SQL"),
    ("MySQL", "PostgreSQL", "SQL Server", "Oracle Database", "SQLite"),
    ("MongoDB", "Cassandra", "DynamoDB", "Firebase"),
    ("AWS", "Azure", "GCP"),
    ("Docker", "Kubernetes"),
    ("Terraform", "Ansible"),
    ("TensorFlow", "PyTorch", "Keras"),
    ("Machine Learning", "Deep Learning"),
    ("NLP", "Large Language Models"),
    ("Apache Spark", "Hadoop", "Hive", "Databricks"),
    ("Tableau", "Power BI", "Looker", "Qlik"),
    ("Agile", "Scrum", "Kanban"),
    ("Linux", "Unix"),
    ("Jenkins", "CI/CD"),
)

# ---- Years of experience and degrees, as written

_NUMBER_WORDS = (
    "one and a half|two and a half|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|"
    "fifteen|twenty"
)
_WORD_VALUES = {
    "one and a half": 1.5, "two and a half": 2.5, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "fifteen": 15,
    "twenty": 20,
}
_EXPERIENCE_RE = re.compile(
    r"(?:\b(?:about|around|over|more than|nearly|almost|approximately|a little over|close to)\s+)?"
    rf"\b(?:\d{{1,2}}(?:\.\d)?\s*\+?|(?:{_NUMBER_WORDS})\b\+?)\s*(?:years?|yrs?)\b"
    r"(?:\s+[\w-]+){0,3}?\s+experience\b"
    rf"|\bexperience\s+of\s+(?:about\s+|over\s+)?(?:\d{{1,2}}\s*\+?|(?:{_NUMBER_WORDS})\b)\s*(?:years?|yrs?)\b",
    re.IGNORECASE,
)
_YEARS_VALUE_RE = re.compile(rf"(\d{{1,2}}(?:\.\d)?)|\b({_NUMBER_WORDS})\b", re.IGNORECASE)
_MAX_EVIDENCE = 3

# Degree names. Abbreviations are case-sensitive and need their dots ("B.E.", "M.S."), so ordinary
# words such as "be", "ms" (MS Excel) or "ba" are not mistaken for degrees.
_DOCTORATE = r"Ph\.?\s?D\b\.?|Doctorate\b"
_MASTERS = (
    r"[Mm]aster'?s?(?:\s+of|\s+in|\s+degree)\b|M\.\s?Tech\b|MTech\b|M\.\s?Sc\b|MSc\b|M\.S\.|M\.E\.|MBA\b|MCA\b|"
    r"M\.\s?Com\b|M\.A\."
)
_BACHELORS = (
    r"[Bb]achelor'?s?(?:\s+of|\s+in|\s+degree)\b|B\.\s?Tech\b|BTech\b|B\.\s?Sc\b|BSc\b|B\.S\.|B\.E\b\.?|BCA\b|"
    r"BBA\b|B\.\s?Com\b|B\.A\."
)
_DIPLOMA = r"[Dd]iploma\s+in\b|[Aa]ssociate'?s?\s+degree\b"
_DEGREE_RE = re.compile(rf"(?<![A-Za-z])(?:{_DOCTORATE}|{_MASTERS}|{_BACHELORS}|{_DIPLOMA})")
_DEGREE_LEVELS = (
    (re.compile(r"(?<![A-Za-z])(?:" + _DOCTORATE + ")"), 4, "Doctorate (PhD)"),
    (re.compile(r"(?<![A-Za-z])(?:" + _MASTERS + ")"), 3, "Master's degree"),
    (re.compile(r"(?<![A-Za-z])(?:" + _BACHELORS + ")"), 2, "Bachelor's degree"),
    (re.compile(r"(?<![A-Za-z])(?:" + _DIPLOMA + ")"), 1, "Diploma or associate degree"),
)
# A degree mention ends at a field separator, a full stop followed by a space, or a year.
_DEGREE_END_RE = re.compile(r"\s*(?:[|;•▪●\n]|\.\s|\b(?:19|20)\d\d\b)")
_MAX_DEGREE_CHARS = 80

# Short degree abbreviations count only when followed by "in", "degree", "or", "and", "/", "," or
# the end of the text, so "MS Excel" or "BA team" are not read as degrees.
_ABBREVIATION_CONTEXT = r"(?=\s+(?:in|degree|or|and)\b|\s*[/,)]|\.?\s*$)"
# In a job description: the degree level it asks for. Plain-word forms are matched in any case
# ("bachelor's degree", "PhD"); short abbreviations only in their usual spelling (BS, B.S., MS in).
_JD_DEGREE_PATTERNS = (
    (re.compile(r"\b(?:ph\.?\s?d|doctorate|doctoral)\b", re.IGNORECASE), 4),
    (re.compile(r"\bmaster'?s?\b|\bpost-?graduate\b|\bm\.?\s?tech\b|\bmba\b|\bmca\b", re.IGNORECASE), 3),
    (re.compile(r"\b(?:MS|M\.S\.|MSc|M\.Sc)" + _ABBREVIATION_CONTEXT), 3),
    (re.compile(r"\bbachelor'?s?\b|\bundergraduate degree\b|\bb\.?\s?tech\b|\bb\.?\s?e\.?\s+in\b|\bgraduate degree\b",
                re.IGNORECASE), 2),
    (re.compile(r"\b(?:BS|B\.S\.|BSc|B\.Sc|BA|B\.A\.)" + _ABBREVIATION_CONTEXT), 2),
    (re.compile(r"\b(?:university|college|academic)?\s*degree\s+in\b", re.IGNORECASE), 2),
    (re.compile(r"\bdiploma\b", re.IGNORECASE), 1),
)
_LEVEL_LABELS = {4: "Doctorate (PhD)", 3: "Master's degree", 2: "Bachelor's degree", 1: "Diploma or degree"}
_JD_YEARS_RE = re.compile(
    rf"(?:at\s+least|minimum\s+(?:of\s+)?|min\.?\s+)?\b(\d{{1,2}}(?:\.\d)?|{_NUMBER_WORDS})\s*\+?\s*"
    rf"(?:(?:-|–|to)\s*(?:\d{{1,2}}|{_NUMBER_WORDS})\s*)?\+?\s*(?:years?|yrs?)\b",
    re.IGNORECASE,
)
_SENTENCE_RE = re.compile(r"[^.\n;•]+")
_SNIPPET_RADIUS = 70


def extract_experience(text: object) -> tuple[str, ...]:
    """Phrases that state years of experience ("4 years of experience"), as written in the resume.

    Only explicit statements are returned; nothing is computed from employment dates.
    """
    found: list[str] = []
    for match in _EXPERIENCE_RE.finditer(normalize_whitespace(_as_text(text))):
        phrase = match.group().strip()
        if phrase.lower() not in (f.lower() for f in found):
            found.append(phrase)
        if len(found) == _MAX_EVIDENCE:
            break
    return tuple(found)


def extract_education(text: object) -> tuple[str, ...]:
    """Degree mentions such as "B.Tech, Computer Engineering", cut at the next separator or year."""
    text = normalize_whitespace(_as_text(text))
    found: list[str] = []
    for match in _DEGREE_RE.finditer(text):
        tail = text[match.start() : match.start() + _MAX_DEGREE_CHARS + 1]
        end = _DEGREE_END_RE.search(tail, len(match.group()))
        mention = tail[: end.start()] if end else tail
        if len(mention) > _MAX_DEGREE_CHARS:
            mention = mention[:_MAX_DEGREE_CHARS].rsplit(" ", 1)[0]
        mention = re.sub(r"\s+([,.;:)])", r"\1", mention).strip(" ,.-(")
        if mention and not any(mention in f or f in mention for f in found):
            found.append(mention)
        if len(found) == _MAX_EVIDENCE:
            break
    return tuple(found)


def years_in(phrase: str) -> float | None:
    """The number of years in an experience phrase ("about one and a half years" -> 1.5)."""
    match = _YEARS_VALUE_RE.search(phrase)
    if match is None:
        return None
    if match.group(1):
        return float(match.group(1))
    return float(_WORD_VALUES[match.group(2).lower()])


def degree_level(mention: str) -> int:
    """Level of a degree mention from the resume: 4 doctorate, 3 master's, 2 bachelor's, 1 diploma."""
    for pattern, level, _ in _DEGREE_LEVELS:
        if pattern.search(mention):
            return level
    return 0


# ---- What the job asks for


@dataclass(frozen=True)
class JobRequirements:
    """The checkable requirements of one job description."""

    skills: tuple[Requirement, ...] = ()
    degree_level: int | None = None  # minimum level asked for (see degree_level)
    min_years: float | None = None
    degree_text: str = ""  # the job description's own wording, for display
    years_text: str = ""

    @property
    def degree_label(self) -> str:
        return f"{_LEVEL_LABELS[self.degree_level]} or higher" if self.degree_level else ""

    @property
    def years_label(self) -> str:
        return f"{self.min_years:g}+ years of experience" if self.min_years is not None else ""

    @property
    def names(self) -> tuple[str, ...]:
        """All requirement names in display order: skills, then degree, then experience."""
        extra = tuple(label for label in (self.degree_label, self.years_label) if label)
        return tuple(r.name for r in self.skills) + extra

    def __len__(self) -> int:
        return len(self.names)


def _job_degree(text: str) -> tuple[int | None, str]:
    """Lowest degree level the job mentions in an education context, and that sentence."""
    best: int | None = None
    sentence_found = ""
    for sentence in _SENTENCE_RE.findall(text):
        levels = [level for pattern, level in _JD_DEGREE_PATTERNS if pattern.search(sentence)]
        if levels and (best is None or min(levels) < best):
            best, sentence_found = min(levels), sentence.strip()
    return best, sentence_found


def _job_years(text: str) -> tuple[float | None, str]:
    """Minimum years of experience the job asks for (first "N+ years ... experience" sentence)."""
    for sentence in _SENTENCE_RE.findall(text):
        if "experience" not in sentence.lower():
            continue
        match = _JD_YEARS_RE.search(sentence)
        if match:
            value = match.group(1)
            years = float(value) if value[0].isdigit() else float(_WORD_VALUES[value.lower()])
            return years, sentence.strip()
    return None, ""


def job_requirements(job_description: object, skills: list[Requirement] | tuple[Requirement, ...] | None = None) -> JobRequirements:
    """Skills (skills.extract_requirements), degree level and years of experience from a job description."""
    text = normalize_whitespace(_as_text(job_description))
    if skills is None:
        skills = extract_requirements(text)
    level, degree_text = _job_degree(text)
    years, years_text = _job_years(text)
    return JobRequirements(tuple(skills), level, years, degree_text, years_text)


# ---- Checking one resume


@dataclass(frozen=True)
class RequirementCheck:
    """The outcome for one requirement in one resume."""

    name: str
    kind: str
    status: str
    match_type: str = ""  # exact, alias or stated (for matched requirements)
    evidence: str = ""  # quote from the resume (contact details hidden)
    related_to: tuple[str, ...] = ()  # related skills the resume does name
    note: str = ""


@dataclass(frozen=True)
class CoverageReport:
    checks: tuple[RequirementCheck, ...] = field(default_factory=tuple)

    def _names(self, status: str) -> tuple[str, ...]:
        return tuple(c.name for c in self.checks if c.status == status)

    @property
    def matched(self) -> tuple[str, ...]:
        return self._names(STATUS_MATCHED)

    @property
    def related(self) -> tuple[str, ...]:
        return self._names(STATUS_RELATED)

    @property
    def missing(self) -> tuple[str, ...]:
        return self._names(STATUS_MISSING)

    @property
    def semantic(self) -> tuple[str, ...]:
        """Requirements with semantically similar wording only (see semantic_evidence)."""
        return self._names("semantic")

    @property
    def total(self) -> int:
        return len(self.checks)

    @property
    def score(self) -> float | None:
        """Percent of requirements matched, or None when there is nothing to check."""
        return round(100 * len(self.matched) / self.total, 1) if self.total else None


def snippet(text: str, start: int, end: int, radius: int = _SNIPPET_RADIUS) -> str:
    """A short single-line quote around text[start:end], with contact details hidden."""
    left = max(0, start - radius)
    right = min(len(text), end + radius)
    if left > 0:  # do not start or end in the middle of a word
        space = text.find(" ", left, start)
        left = space + 1 if space != -1 else left
    if right < len(text):
        space = text.rfind(" ", end, right)
        right = space if space != -1 else right
    quote = normalize_whitespace(mask_contact_info(text[left:right]))
    return ("…" if left > 0 else "") + quote + ("…" if right < len(text) else "")


@cache
def _vocabulary_by_name() -> dict[str, Requirement]:
    return {requirement.name: requirement for requirement in vocabulary()}


@cache
def _related_names() -> dict[str, tuple[str, ...]]:
    related: dict[str, list[str]] = {}
    for group in RELATED_GROUPS:
        for name in group:
            related.setdefault(name, []).extend(other for other in group if other != name)
    return {name: tuple(dict.fromkeys(others)) for name, others in related.items()}


def _alias_key(text: str) -> str:
    return " ".join(normalize_for_matching(text).split())


def _match_type(requirement: Requirement, matched_text: str) -> str:
    """Exact when the resume uses the requirement's own name (any case, plural allowed)."""
    if matched_text == requirement.name:
        return MATCH_EXACT
    key, name = _alias_key(matched_text), _alias_key(requirement.name)
    return MATCH_EXACT if key in (name, name + "s", name + "es") else MATCH_ALIAS


def _check_skill(requirement: Requirement, text: str, normalized: str) -> RequirementCheck:
    span = requirement.first_span(text, normalized)
    if span:
        start, end = span
        return RequirementCheck(
            requirement.name, KIND_SKILL, STATUS_MATCHED,
            match_type=_match_type(requirement, text[start:end]), evidence=snippet(text, start, end),
        )
    by_name = _vocabulary_by_name()
    found = []
    for other in _related_names().get(requirement.name, ()):
        other_requirement = by_name.get(other)
        other_span = other_requirement.first_span(text, normalized) if other_requirement else None
        if other_span:
            found.append((other, other_span))
    if found:
        other, (start, end) = found[0]
        return RequirementCheck(
            requirement.name, KIND_SKILL, STATUS_RELATED, evidence=snippet(text, start, end),
            related_to=tuple(name for name, _ in found),
            note=f"Mentions {', '.join(name for name, _ in found)}, which is related but not the same.",
        )
    return RequirementCheck(requirement.name, KIND_SKILL, STATUS_MISSING)


def _check_degree(job: JobRequirements, text: str) -> RequirementCheck:
    mentions = extract_education(text)
    best = max(mentions, key=degree_level, default="")
    if best and degree_level(best) >= job.degree_level:
        return RequirementCheck(job.degree_label, KIND_EDUCATION, STATUS_MATCHED, MATCH_STATED, evidence=best)
    note = f"Highest degree found: {best}." if best else "No degree found in the resume text."
    return RequirementCheck(job.degree_label, KIND_EDUCATION, STATUS_MISSING, evidence=best, note=note)


def _check_years(job: JobRequirements, text: str) -> RequirementCheck:
    stated = [(years_in(phrase), phrase) for phrase in extract_experience(text)]
    stated = [(years, phrase) for years, phrase in stated if years is not None]
    if stated:
        years, phrase = max(stated)
        if years >= job.min_years:
            return RequirementCheck(job.years_label, KIND_EXPERIENCE, STATUS_MATCHED, MATCH_STATED, evidence=phrase)
        return RequirementCheck(
            job.years_label, KIND_EXPERIENCE, STATUS_MISSING, evidence=phrase,
            note=f"States {years:g} years; the job asks for {job.min_years:g}+.",
        )
    return RequirementCheck(
        job.years_label, KIND_EXPERIENCE, STATUS_MISSING,
        note="No total years of experience stated (employment dates are not added up).",
    )


def assess(job: JobRequirements, resume_text: object) -> CoverageReport:
    """Check every requirement of ``job`` against one resume (see module docstring)."""
    text = normalize_whitespace(_as_text(resume_text))
    normalized = normalize_for_matching(text)
    checks = [_check_skill(requirement, text, normalized) for requirement in job.skills]
    if job.degree_level:
        checks.append(_check_degree(job, text))
    if job.min_years is not None:
        checks.append(_check_years(job, text))
    return CoverageReport(tuple(checks))


# ---- Semantic evidence (Sentence-BERT), shown separately and never counted as a match

STATUS_SEMANTIC = "semantic"

# Semantic evidence is only looked for when a requirement names a capability or practice ("REST APIs",
# "Unit Testing", "Data Visualization"), never for a specific product (Python, Django, AWS): for a
# product, similar wording is not evidence, and on real resumes the model rated unrelated lines (an
# e-mail address, "MS Excel") as close to "Python" or "PostgreSQL". Items listed in the job description
# that are not in the skill vocabulary qualify when they have at least two words ("performance tuning").
SEMANTIC_CAPABILITIES = frozenset({
    "REST APIs", "Microservices", "Machine Learning", "Deep Learning", "Artificial Intelligence", "NLP",
    "Computer Vision", "Large Language Models", "Generative AI", "Data Analysis", "Data Visualization",
    "Statistics", "Regression", "Classification", "Clustering", "Feature Engineering", "Time Series",
    "A/B Testing", "Big Data", "Data Mining", "Data Modeling", "ETL", "Data Warehousing", "CI/CD", "DevOps",
    "MLOps", "Serverless", "Cybersecurity", "Networking", "Agile", "Unit Testing", "Test Automation",
    "Object-Oriented Programming", "Data Structures", "Algorithms", "System Design", "Software Development",
    "Web Development", "UI/UX", "Graphic Design", "Project Management", "Product Management",
    "Stakeholder Management", "Communication", "Leadership", "Teamwork", "Problem Solving", "Time Management",
    "Customer Service", "Business Development", "Digital Marketing", "SEO", "SEM", "Social Media",
    "Content Writing", "Copywriting", "Email Marketing", "Market Research", "Negotiation", "Budgeting",
    "Forecasting", "Financial Analysis", "Financial Modeling", "Financial Reporting", "Accounting",
    "Bookkeeping", "Accounts Payable", "Accounts Receivable", "Payroll", "Auditing", "Taxation",
    "Risk Management", "Compliance", "Reconciliation", "Recruitment", "Onboarding", "Employee Relations",
    "Performance Management", "Training and Development", "Supply Chain Management", "Logistics",
    "Inventory Management", "Procurement", "Vendor Management", "Operations Management", "Quality Assurance",
    "Quality Control", "Data Entry", "Event Planning", "Public Relations", "Patient Care",
    "Medication Administration", "Clinical Research", "Lesson Planning", "Classroom Management",
    "Curriculum Development", "Food Safety", "Menu Planning",
})
# Lines that cannot be evidence: contact details, or fragments of fewer than four words.
_CONTACT_RE = re.compile(r"@|https?://|www\.|\+?\d[\d\s()-]{7,}")
_MIN_EVIDENCE_WORDS = 4


def semantic_eligible(name: str) -> bool:
    """Whether semantic evidence may be looked for (see SEMANTIC_CAPABILITIES)."""
    if name in SEMANTIC_CAPABILITIES:
        return True
    return name not in _vocabulary_by_name() and len(name.split()) >= 2


_SENTENCE_BREAK_RE = re.compile(r"(?<=[.!?;|•▪●])\s+|\s+(?=[•▪●])|\s+[|]\s+")
_MAX_SENTENCE_WORDS = 40


def resume_sentences(text: object, max_words: int = _MAX_SENTENCE_WORDS) -> list[str]:
    """Short sentence-like pieces of a resume (bullets, sentences, fields), each at most ``max_words``."""
    pieces = []
    for piece in _SENTENCE_BREAK_RE.split(normalize_whitespace(_as_text(text))):
        words = piece.strip(" |•▪●").split()
        for start in range(0, len(words), max_words):
            part = " ".join(words[start : start + max_words])
            if len(part) > 2 and any(ch.isalpha() for ch in part):
                pieces.append(part)
    return pieces


def semantic_evidence_batch(
    reports: list[CoverageReport],
    resume_texts: list[object],
    encode_queries,
    encode_passages,
    threshold: float,
) -> list[CoverageReport]:
    """Mark requirements that are missing but semantically close to a sentence of the same resume.

    ``encode_queries`` / ``encode_passages`` turn lists of texts into L2-normalized embeddings
    (for BGE models the queries get the model's search instruction); each is called at most once
    for the whole batch. A missing capability requirement (``semantic_eligible``) whose closest
    sentence in that resume (four or more words, no contact details) reaches ``threshold`` becomes
    STATUS_SEMANTIC with that sentence as evidence. This shows related wording for a reviewer ("developed backend endpoints using Flask"
    for "REST APIs"); it is not proof of the skill and does not change ``report.score``, which
    counts only matched requirements.
    """
    def candidate(check: RequirementCheck) -> bool:
        return check.status == STATUS_MISSING and check.kind == KIND_SKILL and semantic_eligible(check.name)

    names = sorted({c.name for r in reports for c in r.checks if candidate(c)})
    sentences = [
        [x for x in resume_sentences(text) if len(x.split()) >= _MIN_EVIDENCE_WORDS and not _CONTACT_RE.search(x)]
        for text in resume_texts
    ]
    flat = [sentence for per_resume in sentences for sentence in per_resume]
    if not names or not flat:
        return list(reports)
    query_index = {name: i for i, name in enumerate(names)}
    queries = encode_queries(names)
    passages = encode_passages(flat)
    out, start = [], 0
    for report, own in zip(reports, sentences, strict=True):
        block = passages[start : start + len(own)]
        start += len(own)
        if not own:
            out.append(report)
            continue
        checks = []
        for check in report.checks:
            if candidate(check):
                similarity = block @ queries[query_index[check.name]]
                best = int(similarity.argmax())
                if similarity[best] >= threshold:
                    check = RequirementCheck(
                        check.name, check.kind, STATUS_SEMANTIC, evidence=mask_contact_info(own[best]),
                        note=f"Semantically similar wording (cosine {similarity[best]:.2f}); not a verified skill.",
                    )
            checks.append(check)
        out.append(CoverageReport(tuple(checks)))
    return out


def semantic_evidence(report: CoverageReport, resume_text: object, encode_queries, encode_passages, threshold: float) -> CoverageReport:
    """semantic_evidence_batch for one resume."""
    return semantic_evidence_batch([report], [resume_text], encode_queries, encode_passages, threshold)[0]
