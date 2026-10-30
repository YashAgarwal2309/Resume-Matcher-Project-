"""Find the job description's requirements and check which ones each resume mentions.

This only explains a ranking; it never changes the similarity scores.

    requirements = extract_requirements(job_description)
    comparison = compare_requirements(requirements, resume_text)
    comparison.included, comparison.missing   # lists of requirement names

Requirements come from two places, both read from the job description itself:

1. A vocabulary of common skills (SKILLS) with aliases, so "ML", "machine-learning"
   and "Machine Learning" are one requirement, and a resume that says "k8s" has
   "Kubernetes".
2. Short items listed after cue phrases such as "Required:", "Skills:",
   "experience with" or "knowledge of", which catches skills the vocabulary lacks.

Matching ignores case (except for one- or two-letter names such as "R", "C" and "Go"),
respects word boundaries ("Java" is not found in "JavaScript", "C" not in "C++"),
treats hyphens, slashes and underscores like spaces, and accepts plural forms.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import cache

from resume_matcher.preprocessing import STOP_WORDS, _as_text


@dataclass(frozen=True)
class Skill:
    """A vocabulary entry: display ``name`` plus other spellings.

    ``aliases`` are matched case-insensitively; ``exact`` spellings only with this exact case
    (needed for names that are also ordinary words or letters, like "R" or "Go").
    """

    name: str
    aliases: tuple[str, ...] = ()
    exact: tuple[str, ...] = ()


def _skill(name: str, *aliases: str, exact: tuple[str, ...] = ()) -> Skill:
    # The display name is itself a case-insensitive alias unless it is listed as exact-case only.
    if name not in exact:
        aliases = (name, *aliases)
    return Skill(name, aliases, exact)


SKILLS: tuple[Skill, ...] = (
    # Programming languages
    _skill("Python"),
    _skill("Java"),
    _skill("JavaScript", "js", "ecmascript"),
    _skill("TypeScript"),
    _skill("C++", "cpp"),
    _skill("C#", "c sharp", "csharp"),
    _skill("C", exact=("C",)),
    _skill("R", "r programming", "r language", "rstudio", exact=("R",)),
    _skill("Go", "golang", "go lang", "go programming", exact=("Go",)),
    _skill("Rust"),
    _skill("Ruby"),
    _skill("Ruby on Rails", "rails"),
    _skill("PHP"),
    _skill("Swift"),
    _skill("Kotlin"),
    _skill("Scala"),
    _skill("Perl"),
    _skill("MATLAB"),
    _skill("SAS"),
    _skill("Objective-C"),
    _skill("Dart"),
    _skill("Shell Scripting", "bash", "shell script", "unix shell"),
    _skill("PowerShell"),
    _skill("VBA", "visual basic for applications", "excel vba"),
    _skill("HTML", "html5"),
    _skill("CSS", "css3"),
    _skill("SQL"),
    _skill("NoSQL"),
    _skill("PL/SQL", "plsql"),
    _skill("T-SQL", "tsql"),
    # Web and application frameworks
    _skill("React", "react.js", "reactjs"),
    _skill("React Native"),
    _skill("Angular", "angularjs", "angular.js"),
    _skill("Vue.js", "vue", "vuejs"),
    _skill("Node.js", "nodejs", "node js"),
    _skill("Express.js", "expressjs"),
    _skill("Django"),
    _skill("Flask"),
    _skill("FastAPI"),
    _skill("Spring Boot", "springboot"),
    _skill("Spring Framework", "spring mvc"),
    _skill(".NET", "dotnet", "asp.net", ".net core"),
    _skill("jQuery"),
    _skill("REST APIs", "rest api", "restful", "restful api", "restful services", "rest services"),
    _skill("GraphQL"),
    _skill("Microservices", "microservice", "micro services"),
    _skill("Flutter"),
    _skill("Android"),
    _skill("iOS"),
    # Data science and machine learning
    _skill("Machine Learning", exact=("ML",)),
    _skill("Deep Learning"),
    _skill("Artificial Intelligence", exact=("AI",)),
    _skill("NLP", "natural language processing"),
    _skill("Computer Vision"),
    _skill("Large Language Models", "llm", "llms", "large language model"),
    _skill("Generative AI", "genai", "gen ai"),
    _skill("Data Analysis", "data analytics"),
    _skill("Data Visualization", "data visualisation"),
    _skill("Statistics", "statistical analysis", "statistical modeling", "statistical modelling"),
    _skill("Regression", "linear regression", "logistic regression"),
    _skill("Classification"),
    _skill("Clustering"),
    _skill("Feature Engineering"),
    _skill("Time Series", "time series analysis", "forecasting models"),
    _skill("A/B Testing", "ab testing", "split testing"),
    _skill("Big Data"),
    _skill("Data Mining"),
    _skill("Data Modeling", "data modelling"),
    _skill("ETL", "extract transform load"),
    _skill("Data Warehousing", "data warehouse"),
    _skill("pandas"),
    _skill("NumPy"),
    _skill("SciPy"),
    _skill("scikit-learn", "sklearn"),
    _skill("TensorFlow"),
    _skill("Keras"),
    _skill("PyTorch"),
    _skill("XGBoost"),
    _skill("OpenCV"),
    _skill("NLTK"),
    _skill("spaCy"),
    _skill("Hugging Face", "huggingface"),
    _skill("Matplotlib"),
    _skill("Seaborn"),
    _skill("Jupyter", "jupyter notebook"),
    _skill("Apache Spark", "spark", "pyspark"),
    _skill("Hadoop"),
    _skill("Hive"),
    _skill("Kafka", "apache kafka"),
    _skill("Airflow", "apache airflow"),
    _skill("Databricks"),
    _skill("Snowflake"),
    _skill("Tableau"),
    _skill("Power BI", "powerbi"),
    _skill("Looker"),
    _skill("Qlik", "qlikview", "qlik sense"),
    _skill("SPSS"),
    _skill("Excel", "ms excel", "microsoft excel", "advanced excel"),
    _skill("Google Analytics"),
    # Databases
    _skill("MySQL"),
    _skill("PostgreSQL", "postgres"),
    _skill("Oracle Database", "oracle db", "oracle sql", "oracle 11g", "oracle 12c"),
    _skill("SQL Server", "mssql", "ms sql", "microsoft sql server"),
    _skill("MongoDB", "mongo"),
    _skill("Redis"),
    _skill("Cassandra"),
    _skill("Elasticsearch", "elastic search"),
    _skill("DynamoDB"),
    _skill("SQLite"),
    _skill("Firebase"),
    # Cloud, DevOps and engineering practice
    _skill("AWS", "amazon web services"),
    _skill("Azure", "microsoft azure"),
    _skill("GCP", "google cloud", "google cloud platform"),
    _skill("Docker"),
    _skill("Kubernetes", "k8s"),
    _skill("Terraform"),
    _skill("Ansible"),
    _skill("Jenkins"),
    _skill("CI/CD", "continuous integration", "continuous delivery", "continuous deployment"),
    _skill("Git"),
    _skill("GitHub"),
    _skill("GitLab"),
    _skill("Linux"),
    _skill("Unix"),
    _skill("DevOps"),
    _skill("MLOps"),
    _skill("Serverless"),
    _skill("Cybersecurity", "cyber security", "information security", "infosec"),
    _skill("Networking", "computer networks", "network administration"),
    _skill("TCP/IP"),
    _skill("Agile", "agile methodology", "agile methodologies"),
    _skill("Scrum"),
    _skill("Kanban"),
    _skill("Jira"),
    _skill("Confluence"),
    _skill("Selenium"),
    _skill("Unit Testing", "unit test"),
    _skill("Test Automation", "automation testing", "automated testing"),
    _skill("Postman"),
    _skill("Object-Oriented Programming", "object oriented", "oop", "ood"),
    _skill("Data Structures", "data structure"),
    _skill("Algorithms", "algorithm"),
    _skill("System Design"),
    _skill("Software Development", "software engineering"),
    _skill("Web Development"),
    # Design and engineering tools
    _skill("Figma"),
    _skill("Adobe Photoshop", "photoshop"),
    _skill("Adobe Illustrator", "illustrator"),
    _skill("Adobe InDesign", "indesign"),
    _skill("Adobe Premiere Pro", "premiere pro", "adobe premiere"),
    _skill("After Effects", "adobe after effects"),
    _skill("UI/UX", "ux design", "ui design", "user experience", "user interface design", "ux"),
    _skill("Graphic Design"),
    _skill("AutoCAD", "auto cad"),
    _skill("SolidWorks"),
    _skill("Revit"),
    # Business software
    _skill("SAP"),
    _skill("Salesforce"),
    _skill("ERP", "enterprise resource planning"),
    _skill("CRM", "customer relationship management"),
    _skill("QuickBooks"),
    _skill("Tally", "tally erp"),
    _skill("Microsoft Office", "ms office", "office 365", "microsoft 365"),
    _skill("Microsoft Word", "ms word"),
    _skill("PowerPoint", "power point", "ms powerpoint"),
    # Business, finance and people skills
    _skill("Project Management"),
    _skill("Product Management"),
    _skill("Stakeholder Management"),
    _skill("Communication", "communication skills", "communications"),
    _skill("Leadership", "team leadership"),
    _skill("Teamwork", "team player"),
    _skill("Problem Solving"),
    _skill("Time Management"),
    _skill("Customer Service", "customer support", "client service"),
    _skill("Business Development"),
    _skill("Digital Marketing"),
    _skill("SEO", "search engine optimization", "search engine optimisation"),
    _skill("SEM", "search engine marketing"),
    _skill("Social Media", "social media marketing"),
    _skill("Content Writing", "content creation"),
    _skill("Copywriting"),
    _skill("Email Marketing"),
    _skill("Market Research"),
    _skill("Negotiation", "negotiating"),
    _skill("Budgeting", "budget management", "budget planning"),
    _skill("Forecasting"),
    _skill("Financial Analysis"),
    _skill("Financial Modeling", "financial modelling"),
    _skill("Financial Reporting", "financial statements"),
    _skill("Accounting"),
    _skill("Bookkeeping"),
    _skill("Accounts Payable"),
    _skill("Accounts Receivable"),
    _skill("Payroll"),
    _skill("Auditing", "audit"),
    _skill("Taxation", "tax preparation"),
    _skill("GAAP"),
    _skill("IFRS"),
    _skill("Risk Management"),
    _skill("Compliance"),
    _skill("Reconciliation", "account reconciliation", "bank reconciliation"),
    _skill("Recruitment", "recruiting", "talent acquisition"),
    _skill("Onboarding"),
    _skill("Employee Relations"),
    _skill("Performance Management"),
    _skill("HRIS"),
    _skill("Training and Development", "learning and development"),
    _skill("Supply Chain Management", "supply chain"),
    _skill("Logistics"),
    _skill("Inventory Management", "inventory control"),
    _skill("Procurement", "purchasing"),
    _skill("Vendor Management"),
    _skill("Operations Management"),
    _skill("Quality Assurance"),
    _skill("Quality Control"),
    _skill("Six Sigma", "lean six sigma"),
    _skill("Data Entry"),
    _skill("Event Planning", "event management"),
    _skill("Public Relations"),
    # Healthcare, education and hospitality
    _skill("Patient Care"),
    _skill("Medication Administration"),
    _skill("Electronic Health Records", "ehr", "emr", "electronic medical records"),
    _skill("CPR"),
    _skill("BLS", "basic life support"),
    _skill("HIPAA"),
    _skill("Phlebotomy"),
    _skill("Clinical Research"),
    _skill("Lesson Planning", "lesson plans", "lesson plan"),
    _skill("Classroom Management"),
    _skill("Curriculum Development", "curriculum design"),
    _skill("Food Safety"),
    _skill("Menu Planning"),
)

# Hyphens, dashes, slashes and underscores separate words just like spaces do ("machine-learning",
# "CI/CD", "scikit_learn"). One character becomes one character, so match positions stay valid.
_SEPARATORS = str.maketrans({ch: " " for ch in "-‐‑‒–—_/\\"})

# Word boundaries that also respect the symbols used in skill names: "Java" must not match inside
# "JavaScript", "C" not inside "C++" or "C#", and "js" not inside "node.js".
_BEFORE = r"(?<![a-z0-9+#.])"
_AFTER = r"(?![a-z0-9+#])"
# Exact-case names are short and easy to confuse, so they also must not touch "-", "&" or "'"
# ("C-level", "R&D", "Go-getter"), and are not followed by ".x" (initials such as "R.J.").
_EXACT_BEFORE = r"(?<![A-Za-z0-9+#.&'\-])"
_EXACT_AFTER = r"(?![A-Za-z0-9+#&'\-]|\.[A-Za-z])"


def normalize_for_matching(text: object) -> str:
    """Lowercase ``text`` and turn word separators into spaces, keeping its length unchanged."""
    text = _as_text(text)
    lowered = text.lower()
    # Character-by-character only when needed: some characters lower-case to two ("İ" -> "i̇"),
    # which would shift match positions, and Greek capital sigma lower-cases by context.
    if len(lowered) != len(text) or "Σ" in text:
        lowered = "".join(ch.lower() if len(ch.lower()) == 1 else ch for ch in text)
    return lowered.translate(_SEPARATORS)


def _alias_pattern(alias: str) -> str:
    """Regex for one case-insensitive alias: any whitespace between words, optional plural ending."""
    words = normalize_for_matching(alias).split()
    body = r"\s+".join(re.escape(word) for word in words)
    if words and words[-1][-1].isalpha():
        body += r"(?:e?s)?"
    return _BEFORE + body + _AFTER


@dataclass(frozen=True)
class Requirement:
    """One requirement found in the job description, with the patterns that detect it."""

    name: str
    pattern: re.Pattern | None = None  # case-insensitive aliases, run on normalize_for_matching(text)
    exact_pattern: re.Pattern | None = None  # exact-case spellings, run on the original text
    # Literal text every match must contain: the first word of each alias (normalized) and each
    # exact spelling. A quick substring test skips the regex when none of them occurs.
    alias_needles: tuple[str, ...] = ()
    exact_needles: tuple[str, ...] = ()

    def _may_match(self, text: str, normalized: str) -> tuple[bool, bool]:
        alias = self.pattern is not None and (not self.alias_needles or any(n in normalized for n in self.alias_needles))
        exact = self.exact_pattern is not None and (not self.exact_needles or any(n in text for n in self.exact_needles))
        return alias, exact

    def spans(self, text: str, normalized: str) -> list[tuple[int, int]]:
        found = []
        if self.pattern is not None:
            found += [m.span() for m in self.pattern.finditer(normalized)]
        if self.exact_pattern is not None:
            found += [m.span() for m in self.exact_pattern.finditer(text)]
        return found

    def first_span(self, text: str, normalized: str) -> tuple[int, int] | None:
        """The earliest match (like min(spans(...))), without collecting every match."""
        alias, exact = self._may_match(text, normalized)
        found = []
        if alias and (m := self.pattern.search(normalized)):
            found.append(m.span())
        if exact and (m := self.exact_pattern.search(text)):
            found.append(m.span())
        return min(found) if found else None

    def found_in(self, text: str, normalized: str | None = None) -> bool:
        if normalized is None:
            normalized = normalize_for_matching(text)
        alias, exact = self._may_match(text, normalized)
        return bool((alias and self.pattern.search(normalized)) or (exact and self.exact_pattern.search(text)))


def _make_requirement(name: str, aliases: tuple[str, ...] = (), exact: tuple[str, ...] = ()) -> Requirement:
    pattern = exact_pattern = None
    if aliases:
        pattern = re.compile("|".join(f"(?:{_alias_pattern(a)})" for a in aliases))
    if exact:
        exact_pattern = re.compile(_EXACT_BEFORE + "(?:" + "|".join(re.escape(e) for e in exact) + ")" + _EXACT_AFTER)
    alias_needles = tuple(dict.fromkeys(w[0] for a in aliases if (w := normalize_for_matching(a).split())))
    return Requirement(name, pattern, exact_pattern, alias_needles, tuple(exact))


@cache
def vocabulary() -> tuple[Requirement, ...]:
    """The compiled SKILLS vocabulary (built once)."""
    return tuple(_make_requirement(s.name, s.aliases, s.exact) for s in SKILLS)


# ---- Items listed after cue phrases ("Required: X, Y and Z", "experience with X or Y")

_CUE_RE = re.compile(
    r"\b(?:requirements?|required|qualifications?|skills?|must[ -]haves?|must have|nice to have|"
    r"(?:proficien(?:t|cy)|experience|experienced|expertise|knowledge|familiar(?:ity)?|"
    r"hands[ -]on(?: experience)?|competen(?:t|cy|ce))\s+(?:in|with|of|using)|"
    r"tech(?:nical)? stack|technologies|tools)\b\s*:?",
    re.IGNORECASE,
)
# Sentence and bullet boundaries. A dot only ends a sentence when followed by space ("node.js" stays).
_SENTENCE_SPLIT_RE = re.compile(r"[\n;•▪●◦·]|\.(?=\s|$)")
# "<category> such as X, Y": the category ("cloud platforms") is not a requirement by itself.
_EXAMPLES_RE = re.compile(r"[^,;:()]*\b(?:such as|including|e\.g\.?|for example|like)\b", re.IGNORECASE)
_ITEM_SPLIT_RE = re.compile(r"[,;:()\[\]/&|+]|\b(?:and|or|as well as|etc|plus)\b", re.IGNORECASE)
_LEADING_FILLER = frozenset(
    (
        "a", "an", "the", "strong", "solid", "excellent", "good", "great", "basic", "advanced",
        "working", "proven", "demonstrated", "practical", "some", "any", "prior", "relevant",
        "thorough", "sound", "in", "in-depth", "with", "of", "using", "to", "is", "are", "be", "also",
        "experience", "experienced", "knowledge", "understanding", "familiarity", "proficiency",
        "proficient", "expertise", "ability", "skills", "skill", "hands-on",
    )
)
_TRAILING_FILLER = frozenset(
    (
        "skills", "skill", "experience", "knowledge", "expertise", "proficiency", "background", "is",
        "are", "a", "plus", "preferred", "required", "desired", "desirable", "mandatory", "essential",
        "beneficial", "advantage", "advantageous", "highly", "helpful",
    )
)
# Items containing one of these words describe the job or the candidate, not a skill to look for.
_GENERIC_WORDS = frozenset(
    (
        "year", "years", "yr", "yrs", "degree", "bachelor", "bachelors", "bachelor's", "master",
        "masters", "master's", "phd", "diploma", "field", "related", "equivalent", "candidate",
        "candidates", "role", "position", "company", "job", "opportunity", "salary", "benefits", "we",
        "you", "our", "your", "team", "work", "environment", "ability", "responsibilities", "duties",
    )
)
_MAX_ITEM_WORDS = 4


def _clean_item(item: str) -> str:
    words = item.strip(" \t.,:;'\"!?*-").split()
    while words and words[0].lower() in _LEADING_FILLER:
        words.pop(0)
    while words and words[-1].lower().strip(".") in _TRAILING_FILLER:
        words.pop()
    return " ".join(words).strip(" .")


def _is_requirement_item(item: str) -> bool:
    words = item.lower().split()
    if not words or len(words) > _MAX_ITEM_WORDS or len(item) < 2:
        return False
    if not any(ch.isalpha() for ch in item):
        return False
    if any(w.strip(".'") in _GENERIC_WORDS for w in words) or any(ch.isdigit() for ch in item):
        return False
    return not all(w in STOP_WORDS for w in words)


_MASK = "\x00"  # stands in for text already matched by the vocabulary


def _sentences(text: str) -> list[tuple[int, str]]:
    """Split ``text`` into ``(start offset, sentence)`` pairs at line breaks, bullets and full stops."""
    out, start = [], 0
    for boundary in _SENTENCE_SPLIT_RE.finditer(text):
        out.append((start, text[start : boundary.start()]))
        start = boundary.end()
    out.append((start, text[start:]))
    return out


def _listed_items(text: str) -> list[tuple[int, str]]:
    """``(position, item)`` for short items listed after a cue phrase, in the order they appear.

    Pieces that contain masked (vocabulary) text are skipped: "Docker containers" is already
    covered by "Docker" and should not also produce "containers".
    """
    items = []
    for start, sentence in _sentences(text):
        cue = _CUE_RE.search(sentence)
        if cue is None:
            continue
        tail_start = start + cue.end()
        # Blank out "<category> such as" while keeping offsets, so only the examples remain.
        tail = _EXAMPLES_RE.sub(lambda m: "," + " " * (len(m.group()) - 1), sentence[cue.end() :])
        position = 0
        for piece in _ITEM_SPLIT_RE.split(tail):
            offset = tail.find(piece, position)
            position = offset + len(piece)
            if _MASK in piece:
                continue
            item = _clean_item(piece)
            if _is_requirement_item(item):
                items.append((tail_start + offset, item))
    return items


def _listed_requirement(item: str) -> Requirement:
    # One- or two-letter items ("Go", "R") are only safe to match with their exact case.
    if len(item) <= 2:
        return _make_requirement(item, exact=(item,))
    # "model evaluation" -> "Model evaluation", so it reads like the vocabulary names; "iOS" stays.
    name = item[0].upper() + item[1:] if item.islower() else item
    return _make_requirement(name, aliases=(item,))


def _key(name: str) -> str:
    return " ".join(normalize_for_matching(name).split())


def extract_requirements(job_description: object) -> list[Requirement]:
    """Return the requirements named in ``job_description``, in the order they first appear.

    Vocabulary skills are found anywhere in the text; where two overlap, the longer one wins
    ("SQL Server" rather than "SQL", "React Native" rather than "React"). Other skills are
    taken from short lists after cue phrases such as "Required:" or "experience with".
    Returns an empty list for blank input or when nothing is recognized.
    """
    text = _as_text(job_description)
    if not text.strip():
        return []
    normalized = normalize_for_matching(text)

    candidates = [
        (start, end, requirement)
        for requirement in vocabulary()
        for start, end in requirement.spans(text, normalized)
    ]
    taken: list[tuple[int, int]] = []
    found: dict[str, tuple[int, Requirement]] = {}
    for start, end, requirement in sorted(candidates, key=lambda c: (c[0] - c[1], c[0])):
        if any(start < t_end and t_start < end for t_start, t_end in taken):
            continue
        taken.append((start, end))
        key = _key(requirement.name)
        if key not in found or start < found[key][0]:
            found[key] = (start, requirement)

    masked = list(text)
    for start, end in taken:
        masked[start:end] = _MASK * (end - start)
    for position, item in _listed_items("".join(masked)):
        key = _key(item)
        if key not in found:
            found[key] = (position, _listed_requirement(item))

    return [requirement for _, requirement in sorted(found.values(), key=lambda pair: pair[0])]


@dataclass(frozen=True)
class RequirementComparison:
    """Which of the job's requirements one resume mentions (``included``) and which it does not."""

    included: tuple[str, ...]
    missing: tuple[str, ...]

    @property
    def total(self) -> int:
        return len(self.included) + len(self.missing)


def compare_requirements(requirements: list[Requirement] | tuple[Requirement, ...], resume_text: object) -> RequirementComparison:
    """Check each requirement against the text of one resume (keeps the requirements' order)."""
    text = _as_text(resume_text)
    normalized = normalize_for_matching(text)
    included, missing = [], []
    for requirement in requirements:
        (included if requirement.found_in(text, normalized) else missing).append(requirement.name)
    return RequirementComparison(tuple(included), tuple(missing))
