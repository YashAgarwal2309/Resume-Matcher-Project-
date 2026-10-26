import pandas as pd
df = pd.read_csv('Resume.csv')
print(df.shape)
print(df.columns)
print(df[['Resume_str', 'Category']].head())

import re

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)  # strip punctuation/special characters
    text = re.sub(r'\s+', ' ', text).strip()   # collapse multiple spaces/newlines into one
    return text

df['clean_resume'] = df['Resume_str'].apply(clean_text)
print(df[['Resume_str', 'clean_resume']].head())

print(df['Category'].unique())

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Write one short job description per category — 2-3 sentences, realistic posting language
job_descriptions = {
    "HR": "Looking for an HR specialist to manage recruitment, employee relations, and HR operations. Experience with onboarding and policy development required.",
    "INFORMATION-TECHNOLOGY": "Seeking a software engineer skilled in Python, databases, and system design. Experience building and maintaining applications required.",
    "SALES": "Sales representative needed to manage client relationships and meet revenue targets. Strong communication and negotiation skills required.",
    # add 2-3 more matching categories from your printout
}

# Vectorize: fit on cleaned resumes, transform both resumes and job descriptions into the same vector space
vectorizer = TfidfVectorizer(stop_words='english')
resume_vectors = vectorizer.fit_transform(df['clean_resume'])
job_vectors = vectorizer.transform(list(job_descriptions.values()))

# Score every job description against every resume
similarity_matrix = cosine_similarity(job_vectors, resume_vectors)

# For each job, show the top matching resume's actual category
for i, job_category in enumerate(job_descriptions.keys()):
    best_match_idx = similarity_matrix[i].argmax()
    predicted_category = df.iloc[best_match_idx]['Category']
    score = similarity_matrix[i][best_match_idx]
    print(f"Job: {job_category} -> Best match category: {predicted_category} (score: {score:.3f})")


import gensim.downloader as api
import numpy as np

# Downloads a pretrained GloVe model (~128MB, one-time download)
word_vectors = api.load('glove-wiki-gigaword-100')

def get_average_vector(text, model, vector_size=100):
    words = text.split()
    vectors = [model[word] for word in words if word in model]
    if len(vectors) == 0:
        return np.zeros(vector_size)
    return np.mean(vectors, axis=0)

resume_vectors_w2v = np.array([get_average_vector(t, word_vectors) for t in df['clean_resume']])
job_vectors_w2v = np.array([get_average_vector(t, word_vectors) for t in job_descriptions.values()])

similarity_matrix_w2v = cosine_similarity(job_vectors_w2v, resume_vectors_w2v)

for i, job_category in enumerate(job_descriptions.keys()):
    best_match_idx = similarity_matrix_w2v[i].argmax()
    predicted_category = df.iloc[best_match_idx]['Category']
    score = similarity_matrix_w2v[i][best_match_idx]
    print(f"Job: {job_category} -> Best match category: {predicted_category} (score: {score:.3f})")


from sentence_transformers import SentenceTransformer

# Load a pretrained Sentence-BERT model — small, fast on CPU, widely used in real semantic search systems
sbert_model = SentenceTransformer('all-MiniLM-L6-v2')

# Encode resumes and job descriptions into full-sentence embeddings (not word-by-word averaging)
resume_vectors_sbert = sbert_model.encode(df['clean_resume'].tolist(), show_progress_bar=True)
job_vectors_sbert = sbert_model.encode(list(job_descriptions.values()), show_progress_bar=True)

similarity_matrix_sbert = cosine_similarity(job_vectors_sbert, resume_vectors_sbert)

for i, job_category in enumerate(job_descriptions.keys()):
    best_match_idx = similarity_matrix_sbert[i].argmax()
    predicted_category = df.iloc[best_match_idx]['Category']
    score = similarity_matrix_sbert[i][best_match_idx]
    print(f"Job: {job_category} -> Best match category: {predicted_category} (score: {score:.3f})")