"""Shared pytest fixtures.

All resumes here are tiny SYNTHETIC examples written for unit tests only.
They are not taken from the real project dataset (data/Resume.csv).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from resume_matcher.data_loader import prepare_corpus

SYNTHETIC_RESUMES = pd.DataFrame(
    {
        "ID": [101, 102, 103, 104, 105],
        "Resume_str": [
            "Python developer with Django, SQL databases and REST API experience. Contact: dev@example.com",
            "Registered nurse providing patient care, medication administration and hospital records.",
            "Head chef managing kitchen staff, menu planning, cooking and food safety.",
            "Java and C++ software engineer building backend systems and databases.",
            "Primary school teacher planning lessons and managing classroom learning.",
        ],
        "Category": ["INFORMATION-TECHNOLOGY", "HEALTHCARE", "CHEF", "INFORMATION-TECHNOLOGY", "TEACHER"],
    }
)

# Small fake "GloVe" table: 3-dimensional vectors grouped by topic.
FAKE_GLOVE = {
    "python": np.array([1.0, 0.0, 0.0], dtype=np.float32),
    "developer": np.array([0.9, 0.1, 0.0], dtype=np.float32),
    "software": np.array([0.95, 0.05, 0.0], dtype=np.float32),
    "databases": np.array([0.8, 0.0, 0.2], dtype=np.float32),
    "nurse": np.array([0.0, 1.0, 0.0], dtype=np.float32),
    "patient": np.array([0.0, 0.9, 0.1], dtype=np.float32),
    "hospital": np.array([0.0, 0.95, 0.05], dtype=np.float32),
    "chef": np.array([0.0, 0.0, 1.0], dtype=np.float32),
    "kitchen": np.array([0.1, 0.0, 0.9], dtype=np.float32),
    "teacher": np.array([0.3, 0.3, 0.3], dtype=np.float32),
}


class FakeSentenceModel:
    """Offline stand-in for a SentenceTransformer: embeds text using FAKE_GLOVE words."""

    def __init__(self):
        self.calls = 0

    def encode(self, texts, **kwargs):
        self.calls += 1
        out = []
        for text in texts:
            words = [w.strip(".,:").lower() for w in text.split()]
            vecs = [FAKE_GLOVE[w] for w in words if w in FAKE_GLOVE]
            out.append(np.mean(vecs, axis=0) if vecs else np.full(3, 0.01, dtype=np.float32))
        return np.asarray(out)


@pytest.fixture
def raw_resumes() -> pd.DataFrame:
    return SYNTHETIC_RESUMES.copy()


@pytest.fixture
def corpus(raw_resumes):
    return prepare_corpus(raw_resumes, source="synthetic")


@pytest.fixture
def resume_csv(tmp_path, raw_resumes) -> Path:
    path = tmp_path / "Resume.csv"
    raw_resumes.assign(Resume_html="<p>ignored</p>").to_csv(path, index=False)
    return path


def make_pdf(text: str) -> bytes:
    """A minimal one-page PDF containing ``text`` (Helvetica), so tests need no PDF files on disk."""
    stream = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode("latin-1")
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        (b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R "
         b"/Resources << /Font << /F1 5 0 R >> >> >>"),
        b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out, offsets = bytearray(b"%PDF-1.4\n"), []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += b"%d 0 obj\n" % number + body + b"\nendobj\n"
    xref = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objects) + 1)
    out += b"".join(b"%010d 00000 n \n" % offset for offset in offsets)
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (len(objects) + 1, xref)
    return bytes(out)
