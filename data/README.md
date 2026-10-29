# Data folder

The app works with two real, public resume datasets. Neither is committed to Git (they contain
third-party personal data); download them yourself and put the CSV files in this folder.
The app shows a dataset selector in the sidebar listing whichever of these files exist.

## 1. Tech roles: `UpdatedResumeDataSet.csv` (default)

Kaggle "Resume Dataset" by Gaurav Dutta: <https://www.kaggle.com/datasets/gauravduttakiit/resume-dataset>
(a free Kaggle login is needed). An MIT-licensed copy is on Hugging Face, which needs no login:

```bash
curl -L -o data/UpdatedResumeDataSet.csv \
  https://huggingface.co/datasets/brackozi/Resume/resolve/main/UpdatedResumeDataSet.csv
```

Columns: `Category`, `Resume`. 962 rows, but only **166 unique** resumes (the loader removes the
796 exact duplicates). Categories are mostly technical: Java Developer, Python Developer, DotNet
Developer, SAP Developer, ETL Developer, Data Science, DevOps Engineer, Testing, Automation Testing,
Database, Hadoop, Blockchain, Business Analyst, Web Designing, Network Security Engineer, plus a few
non-tech ones (HR, Sales, Advocate, ...). 13 resumes contain broken characters such as `NaÃ¯ve`,
which the loader repairs.

## 2. All industries: `Resume.csv`

Kaggle "Resume Dataset" by Snehaan Bhawal: <https://www.kaggle.com/datasets/snehaanbhawal/resume-dataset>

1. Download and unzip it.
2. Copy `Resume/Resume.csv` into this folder, so the path is `data/Resume.csv`.

Columns: `ID`, `Resume_str`, `Resume_html`, `Category` (2,484 resumes in 24 industry categories).
Only `ID`, `Resume_str` and `Category` are used. `Resume.csv` in the project root (where the original
`explore_data.py` expected it) is also accepted.

## Using another CSV

The loader looks for a text column named `Resume_str`, `resume_text`, `resume`, `text`, `content`
or `cv` (case-insensitive). Category (`Category`, `label`, ...) and ID (`ID`, `resume_id`, ...)
columns are optional. To use only one specific file:

```bash
export RESUME_DATA_PATH=/path/to/your_resumes.csv
```

## sample_job_descriptions.json (committed)

Sixteen technical job descriptions **written by hand for demos and evaluation**: SDE, SWE,
Application, Backend, Frontend, Full-Stack, Mobile and Systems Software roles, Data Analyst,
Data Scientist / ML Engineer, AI Engineer (NLP), IT Business Analyst, DevOps, QA / Automation Test,
Database Developer and Data Engineer. They are not taken from any dataset. Each lists
`relevant_categories`, the dataset categories that `evaluate.py` treats as a rough relevance label.
The list is empty when no dataset has a fitting category.
