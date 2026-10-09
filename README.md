# AI Resume & Job Matcher

A Python-based NLP project that compares a resume with a job description using TF-IDF and cosine similarity.

## Features
- Upload a resume in PDF format
- Enter a job description
- Calculate text similarity
- Identify matching and missing skills
- Interactive interface built with Streamlit

## Tech Stack
- Python
- Streamlit
- Scikit-learn
- NLP (TF-IDF and cosine similarity)
- PDF text extraction

## How It Works
1. Extract text from the uploaded resume.
2. Process the resume and job description.
3. Convert both texts into TF-IDF vectors.
4. Calculate cosine similarity.
5. Display the similarity score and skill comparison.

## Installation

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Limitations
- Similarity measures text overlap, not actual candidate suitability.
- Skill detection depends on a predefined list of skills.
- Scanned PDFs may require OCR.
- This is a learning project, not a validated recruitment decision system.

## Project Status
Initial prototype. Local testing and further improvements are planned.

## Author
Dev Rai
B.Tech in Artificial Intelligence and Machine Learning
