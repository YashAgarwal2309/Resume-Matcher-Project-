# Interview demo samples

A small, ready-to-show input for Resume Matcher: **one job description and seven resumes**.

| File | What it is |
|---|---|
| `job_description.txt` | "Data Scientist / Machine Learning Engineer" — the hand-written sample job from `data/sample_job_descriptions.json` |
| `resumes/resume_1.txt` … `resume_7.txt` | The seven resumes as plain text (the single source for the other files) |
| `pdf/resume_1.pdf` … `resume_7.pdf` | The same resumes as one-page PDFs, for the upload demo |
| `demo_resumes.csv` | The same seven resumes in the CSV format the app loads (`ID`, `Category`, `Resume`) |
| `build_samples.py` | Rebuilds `demo_resumes.csv` and `pdf/` from the `.txt` files |

## Where the resumes come from

All seven are **fictional sample resumes written for this demo**. The names, employers and projects are
made up, email addresses use the reserved `example.com` domain, and phone numbers are masked
(`+91 98XXX XX214`). No real person's data is included. They are deliberately a mix of strong, partial and
unrelated matches so the ranking is easy to judge.

| Resume | Role | Expected fit for the job |
|---|---|---|
| resume_1 | Data Science | Strong |
| resume_2 | Python Developer | Partial (Python, no ML) |
| resume_3 | HR | Unrelated |
| resume_4 | Data Science | Strong |
| resume_5 | Java Developer | Weak |
| resume_6 | Data Science | Strong |
| resume_7 | Business Analyst | Weak |

## How to run the demo

Start the app (`.venv/bin/streamlit run app.py`), upload the seven files from `pdf/`, paste
`job_description.txt` into the job description box, choose a method and click **Match Resumes**.

To rebuild the CSV and PDFs after editing a `.txt` resume:

```bash
.venv/bin/python demo_samples/build_samples.py
```

## Results (real run, 30 Oct 2026, the seven PDFs, all resumes ranked)

| Rank | TF-IDF | GloVe | Sentence-BERT | Hybrid (relative) |
|---|---|---|---|---|
| 1 | resume_6 · 30.13 | resume_6 · 90.83 | resume_6 · 79.75 | resume_6 · 100.00 |
| 2 | resume_4 · 25.29 | resume_4 · 87.74 | resume_1 · 77.25 | resume_1 · 85.92 |
| 3 | resume_1 · 23.63 | resume_1 · 86.45 | resume_4 · 74.71 | resume_4 · 85.63 |
| 4 | resume_2 · 4.34 | resume_2 · 78.73 | resume_2 · 63.79 | resume_2 · 40.78 |
| 5 | resume_7 · 3.08 | resume_5 · 74.21 | resume_7 · 59.51 | resume_7 · 27.74 |
| 6 | resume_5 · 1.16 | resume_7 · 71.18 | resume_5 · 56.09 | resume_5 · 23.73 |
| 7 | resume_3 · 0.56 | resume_3 · 50.13 | resume_3 · 53.47 | resume_3 · 0.00 |

Score = document similarity (cosine similarity × 100); the dashboard shows requirement coverage separately. Things worth pointing out:

- **Every method** puts the three Data Science resumes on top, the Python developer fourth and the HR resume last.
- **GloVe** scores now spread from 50 to 91 because the direction all averaged word vectors share is removed;
  before, they were bunched at 78–94.
- **Sentence-BERT** reads every part of these resumes (6 of 7 are longer than its 512-word-piece limit) through
  section-aware chunks; its best-matching section is shown in the app.
- **Hybrid** scores are relative to these seven resumes (each method rescaled to 0–1, equal weights), so the
  best one is 100 by construction; the methods' own scores are shown next to it in the app.
- With only seven resumes, TF-IDF's IDF weights are learned from these seven documents, so its scores differ
  from a run on the full dataset.
