"""Build the interview demo files from the plain-text resumes in demo_samples/resumes/.

Usage (from the project root):
    .venv/bin/python demo_samples/build_samples.py

Writes:
    demo_samples/demo_resumes.csv   - the seven resumes in the CSV format the app loads
    demo_samples/pdf/resume_N.pdf   - a one-page PDF of each resume (printed with headless Chrome)

The resumes are fictional sample data written for the demo (see README.md). The .txt files are the
single source: the app reads exactly the same text that is printed in the PDFs.
"""

from __future__ import annotations

import csv
import html
import re
import subprocess
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESUME_DIR = HERE / "resumes"
PDF_DIR = HERE / "pdf"
CSV_PATH = HERE / "demo_resumes.csv"
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"

# Category of each sample resume (used by the app's "category" column, like the real datasets).
CATEGORIES = {
    "resume_1": "Data Science",
    "resume_2": "Python Developer",
    "resume_3": "HR",
    "resume_4": "Data Science",
    "resume_5": "Java Developer",
    "resume_6": "Data Science",
    "resume_7": "Business Analyst",
}

# Three different layouts so the PDFs don't all look like they came from one template.
THEMES = {
    "classic": """
        body { font-family: Georgia, 'Times New Roman', serif; color: #111; }
        header { text-align: center; border-bottom: 1.5px solid #111; padding-bottom: 6px; }
        h1 { font-size: 2em; letter-spacing: 1px; text-transform: uppercase; margin: 0; }
        .headline { font-style: italic; margin: 2px 0; }
        h2 { font-size: 1em; letter-spacing: 1.5px; border-bottom: 0.75px solid #777;
             margin: 11px 0 4px; padding-bottom: 1px; }
    """,
    "modern": """
        body { font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif; color: #222; }
        header { border-left: 5px solid #1f4e79; padding-left: 12px; }
        h1 { font-size: 2.2em; color: #1f4e79; margin: 0; font-weight: 700; }
        .headline { font-size: 1.15em; color: #555; margin: 1px 0 3px; }
        h2 { font-size: 1em; color: #1f4e79; letter-spacing: 1px; margin: 12px 0 4px; }
    """,
    "compact": """
        body { font-family: Arial, Helvetica, sans-serif; color: #000; }
        header { background: #eef1f4; padding: 8px 10px; }
        h1 { font-size: 1.85em; margin: 0; }
        .headline { font-weight: bold; color: #444; margin: 1px 0; }
        h2 { font-size: 1em; background: #eef1f4; padding: 2px 6px; margin: 10px 0 4px; }
    """,
}
RESUME_THEMES = {
    "resume_1": "modern", "resume_2": "classic", "resume_3": "compact", "resume_4": "classic",
    "resume_5": "modern", "resume_6": "compact", "resume_7": "classic",
}

BASE_CSS = """
    @page { size: A4; margin: 14mm 15mm; }
    body { margin: 0; width: 180mm; line-height: 1.32; }
    .contact { margin: 2px 0 0; }
    p { margin: 2px 0; }
    ul { margin: 2px 0 4px; padding-left: 17px; }
    li { margin: 1px 0; }
    .role { display: flex; justify-content: space-between; margin-top: 6px; }
    .date { white-space: nowrap; }
"""

SECTION_RE = re.compile(r"^[A-Z][A-Z &/]+$")
DATE_RE = re.compile(r"^(\w+ )?\d{4}( - ((\w+ )?\d{4}|Present))?$")

# Shrinks the font until the resume fits one A4 page (180 x 269 mm printable area at 96 dpi),
# so every PDF fills exactly one page whatever the amount of text.
FIT_SCRIPT = """
<script>
  const maxHeight = 269 / 25.4 * 96;
  let size = 12;
  document.body.style.fontSize = size + "pt";
  while (document.body.scrollHeight > maxHeight && size > 8.5) {
    size -= 0.1;
    document.body.style.fontSize = size.toFixed(1) + "pt";
  }
</script>
"""


def to_html(name: str, text: str) -> str:
    """Turn the simple text format into HTML.

    Format: line 1 = name, line 2 = headline, line 3 = contact; then ALL-CAPS section headings,
    "- " bullets, and "Title | Place | Dates" lines for jobs and degrees.
    """
    lines = text.strip().splitlines()
    full_name, headline, contact = (html.escape(line) for line in lines[:3])
    body: list[str] = []
    in_list = False
    for raw in lines[3:]:
        line = raw.strip()
        if line.startswith("- "):
            if not in_list:
                body.append("<ul>")
                in_list = True
            body.append(f"<li>{html.escape(line[2:])}</li>")
            continue
        if in_list:
            body.append("</ul>")
            in_list = False
        if not line:
            continue
        if SECTION_RE.match(line):
            body.append(f"<h2>{html.escape(line)}</h2>")
        elif " | " in line:
            parts = [html.escape(p) for p in line.split(" | ")]
            dates = [p for p in parts[1:] if DATE_RE.match(p)]
            right = dates[0] if dates else ""
            if right:
                parts.remove(right)
            left = f"<b>{parts[0]}</b>" + "".join(f", {p}" for p in parts[1:])
            body.append(f'<div class="role"><span>{left}</span><span class="date">{right}</span></div>')
        else:
            body.append(f"<p>{html.escape(line)}</p>")
    if in_list:
        body.append("</ul>")
    css = BASE_CSS + THEMES[RESUME_THEMES[name]]
    return (
        f"<!doctype html><html><head><meta charset='utf-8'><title>{full_name}</title><style>{css}</style>"
        f"</head><body><header><h1>{full_name}</h1><div class='headline'>{headline}</div>"
        f"<div class='contact'>{contact}</div></header>{''.join(body)}{FIT_SCRIPT}</body></html>"
    )


def print_pdf(page: str, pdf_path: Path) -> None:
    with tempfile.NamedTemporaryFile("w", suffix=".html", delete=False, encoding="utf-8") as f:
        f.write(page)
    subprocess.run(
        [CHROME, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
         f"--print-to-pdf={pdf_path}", Path(f.name).as_uri()],
        check=True, capture_output=True,
    )
    Path(f.name).unlink()


def main() -> None:
    PDF_DIR.mkdir(exist_ok=True)
    rows = []
    for path in sorted(RESUME_DIR.glob("resume_*.txt")):
        name, text = path.stem, path.read_text(encoding="utf-8")
        print_pdf(to_html(name, text), PDF_DIR / f"{name}.pdf")
        rows.append({"ID": name, "Category": CATEGORIES[name], "Resume": " ".join(text.split())})
        print(f"Saved pdf/{name}.pdf")
    with CSV_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["ID", "Category", "Resume"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {CSV_PATH.name} ({len(rows)} resumes)")


if __name__ == "__main__":
    main()
