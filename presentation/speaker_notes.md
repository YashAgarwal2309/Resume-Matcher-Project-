# Speaker notes — Milestone 3

Simple script for each slide. Say it in your own words; you do not need to read it exactly.

## Slide 1

Good morning. We are Batch 3: Yash Agarwal and Raghav Soni. Our project is Resume Matcher. In one line: you give it a job description and some resumes, and it sorts the resumes from the best match to the worst match. It does this by comparing the text of the job with the text of each resume. We built and compared four ways of doing this comparison: TF-IDF, GloVe, Sentence-BERT, and a Hybrid that combines all three.

## Slide 2

First, the problem. For one job opening, a company can get hundreds of resumes. Reading every one by hand is slow and tiring. Searching for keywords is faster, but it misses people who write the same skill in different words, like 'ML' instead of 'Machine Learning'.

Our idea: the recruiter gives our app the job description and the resumes. The app gives each resume a score from 0 to 100 and sorts them. The recruiter can start reading from the top. The app only helps; a human still makes the final decision.

To test it we used two real, public resume datasets from Kaggle. The tech dataset is our main one. It has 962 rows, but 796 of them were exact copies, so after removing copies we have 166 unique resumes in 25 job types. The second dataset has about 2,500 resumes from all industries. We use it as an extra check. Our code counts and cleans these numbers itself (data_loader.py).

## Slide 3

This is how the app works, in five steps.

Step 1, Read: we take the text out of each resume file. PDF, TXT and MD files are supported.
Step 2, Clean: we make everything lowercase, remove links, e-mail addresses and extra symbols, and fix broken characters. We are careful to keep skill names like C++ and C#, which normal cleaning would destroy. The job description is cleaned in exactly the same way, so both sides are fair.
Step 3, Convert: a computer cannot compare sentences directly, so we turn each text into a list of numbers called a vector. The four methods on the next slide are four different ways of doing this step.
Step 4, Compare: we use cosine similarity. Imagine each vector as an arrow. If the job's arrow and the resume's arrow point the same way, the texts are similar and the score is high.
Step 5, Rank: we sort the resumes by score and show the top ones.

Important: the score is only text similarity, not a hiring probability. And there is no chatbot or LLM generating text; the app only measures and sorts.

## Slide 4

Step 3 can be done in different ways. We built four, and all four use the same cosine-similarity comparison and the same ranking code.

1. TF-IDF matches the same words. It counts the words in each resume, but rare, specific words like 'Selenium' count more than common words like 'work'. It is very fast and easy to explain, but it only sees exact words: 'developer' and 'programmer' look unrelated to it.

2. GloVe knows related words. It is a ready-made list of 400,000 English words, where each word is 100 numbers, and words with similar meanings have similar numbers. We represent a resume as the average of its words. The weakness is that averaging a whole resume blurs the details, so most resumes end up with close scores.

3. Sentence-BERT understands meaning. A small pre-trained AI model called bge-small reads the whole text and gives 384 numbers that capture its meaning. So 'built ML models in Python' still matches a Machine Learning job. It is slower and reads only about the first page (512 word pieces) of each resume. We first used a model called MiniLM; bge-small scored better in our tests, so we switched.

4. Hybrid simply averages the three scores, like three judges. A resume needs the right skills, related words and the right meaning to rank high. This gave the best result, so it is the default in our app.

## Slide 5

Our code is split into small modules, and each one matches a box in the design. preprocessing.py does the cleaning. data_loader.py reads resume files and the datasets and removes empty and duplicate resumes. matcher.py has the four methods and the ranking. app.py is the web page, made with Streamlit. evaluate.py compares the methods on the real datasets.

Live demo (run: streamlit run app.py, then open localhost:8501). Upload the seven sample resumes from demo_samples/pdf, paste the Data Scientist job from demo_samples/job_description.txt, keep the Hybrid method, and click Match Resumes. The three data-science resumes come first and the HR resume comes last, which is what we expect. Then show the download button for the CSV. If time allows, switch to TF-IDF and show that the order changes a little.

## Slide 6

How do we know which method works best? We wrote 16 tech job descriptions, like Backend Developer, Data Scientist and DevOps. For each job, we let every method pick its top 5 resumes from the tech dataset, and we checked how many of those 5 come from a fitting job type. For example, for a Backend job, a Java or Python Developer resume counts as a good fit. This is called Precision at 5. Two jobs, Mobile and Systems, have no fitting job type in the data, so 14 jobs are scored.

Results: Hybrid 0.83, Sentence-BERT 0.81, TF-IDF 0.79, GloVe 0.44. So with Hybrid, about 4 out of 5 top resumes were a good fit, and the number-1 resume was a good fit for 14 of the 14 jobs. TF-IDF does well because tech jobs use exact skill names like Spring Boot or Selenium. GloVe does worst because averaging blurs these specific words.

On the all-industries dataset all methods scored low, around 0.27 to 0.37, because it has very few software resumes.

We also have 119 automated tests (pytest) that check loading, cleaning, every method, ranking and the web page. All pass.

To be honest about limits: 14 jobs is a small test, and 'job type fits' is only a rough check of relevance, so these numbers are a good sign, not a guarantee. All numbers come from evaluate.py (results/evaluation_summary.csv).

## Slide 7

To sum up the two Milestone 3 points. First, design and code match: every box in our design is a real module, resumes and jobs are cleaned the same way, and all four methods share the same compare-and-rank code. Second, the modules work: four methods tested on two real datasets, a working web app, an evaluation script and 119 automated tests.

Our limitations: the test is small, 'job type fits' is only a rough check, and we cannot read Word files or scanned PDFs yet. Next, we want real people to mark good matches, show which skills match or are missing, and support more file types.

Thank you. The code is on GitHub at the link shown. We are happy to take questions.

## Possible questions (simple answers)

**What is cosine similarity?**

Think of each text as an arrow. Cosine similarity measures the angle between the job's arrow and the resume's arrow. Same direction gives 1 (we show 100), at right angles gives 0. It ignores length, so a long resume is not favoured just for being long.

**What does TF-IDF stand for?**

Term Frequency, Inverse Document Frequency. Term frequency: how often a word appears in this resume. Inverse document frequency: words that appear in almost every resume (like 'work') get a low weight, rare words (like 'Kubernetes') get a high weight.

**Is this an LLM or ChatGPT-style app?**

No. Nothing generates text. Sentence-BERT is a small pre-trained model that only turns text into numbers. The app measures similarity and sorts.

**Why is GloVe so weak here?**

We average all the word vectors in a resume. Most tech resumes share many general computer words, so their averages look alike and the specific skills get blurred.

**Why use the Hybrid?**

Each method catches something different: exact skills (TF-IDF), related words (GloVe) and overall meaning (Sentence-BERT). Averaging them gave the best result on the tech dataset and put a good fit at number 1 for all 14 scored jobs.

**Why did you switch the Sentence-BERT model?**

We first used all-MiniLM-L6-v2 (Precision@5 0.69). BAAI/bge-small-en-v1.5 is about the same size, is trained for search, and scored 0.81 on the same test, so it became the default.

**Is the score the chance of being hired?**

No. It is only how similar the texts are. The app should help a recruiter decide what to read first, never reject anyone automatically. Text similarity can also copy biases in how people write.

**Is the data real?**

Yes. Both datasets are public Kaggle resume datasets, and all results come from real runs. The 7 demo resumes in demo_samples/ are made-up sample resumes prepared only for the live demo, so we show no real person's data.
