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

## Results (real run, 30 Oct 2026, the seven PDFs, top-k = 7)

| Rank | TF-IDF | GloVe | Sentence-BERT | Hybrid |
|---|---|---|---|---|
| 1 | resume_6 · 39.71 | resume_6 · 93.83 | resume_6 · 76.87 | resume_6 · 70.14 |
| 2 | resume_4 · 32.12 | resume_4 · 91.63 | resume_1 · 74.32 | resume_4 · 65.84 |
| 3 | resume_1 · 31.64 | resume_1 · 91.35 | resume_4 · 73.78 | resume_1 · 65.77 |
| 4 | resume_2 · 7.02 | resume_2 · 88.63 | resume_2 · 61.61 | resume_2 · 52.42 |
| 5 | resume_7 · 4.00 | resume_5 · 86.77 | resume_7 · 60.19 | resume_7 · 49.96 |
| 6 | resume_5 · 1.76 | resume_7 · 85.70 | resume_5 · 54.05 | resume_5 · 47.53 |
| 7 | resume_3 · 0.58 | resume_3 · 78.45 | resume_3 · 52.53 | resume_3 · 43.86 |

Score = cosine similarity × 100. Things worth pointing out:

- **Every method** puts the three Data Science resumes on top, the Python developer fourth and the HR resume last.
- **GloVe** scores are bunched together (78–94) because averaging word vectors makes most documents look
  similar. This matches the evaluation results, where GloVe was the weakest method on the tech dataset.
- With only seven resumes, TF-IDF's IDF weights are learned from these seven documents, so its scores differ
  from a run on the full dataset.
