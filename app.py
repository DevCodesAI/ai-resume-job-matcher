import re
from io import BytesIO

import streamlit as st
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

SKILLS = [
    'python', 'sql', 'machine learning', 'deep learning', 'natural language processing',
    'nlp', 'computer vision', 'pytorch', 'tensorflow', 'scikit-learn', 'pandas',
    'numpy', 'streamlit', 'git', 'docker', 'aws', 'data analysis', 'opencv',
    'fastapi', 'flask', 'power bi', 'excel', 'java', 'javascript', 'linux'
]


def extract_pdf_text(uploaded_file):
    reader = PdfReader(BytesIO(uploaded_file.getvalue()))
    return '\n'.join(page.extract_text() or '' for page in reader.pages)


def detect_skills(text):
    normalized = re.sub(r'[^a-z0-9+#.-]+', ' ', text.lower())
    return {skill for skill in SKILLS if re.search(r'(?<!\w)' + re.escape(skill) + r'(?!\w)', normalized)}


def compare_texts(resume, job_description):
    vectors = TfidfVectorizer(stop_words='english', ngram_range=(1, 2)).fit_transform(
        [resume, job_description]
    )
    return float(cosine_similarity(vectors[0:1], vectors[1:2])[0, 0]) * 100


st.set_page_config(page_title='AI Resume & Job Matcher', page_icon='📄')
st.title('📄 AI Resume & Job Matcher')
st.caption('A transparent TF-IDF text similarity baseline, not a hiring decision tool.')
resume_file = st.file_uploader('Upload a text-based resume PDF', type=['pdf'])
job_description = st.text_area('Paste a job description', height=200)

if st.button('Analyze match', type='primary'):
    if not resume_file or not job_description.strip():
        st.warning('Please upload a PDF and paste a job description.')
    else:
        try:
            resume_text = extract_pdf_text(resume_file)
            if not resume_text.strip():
                st.error('No selectable text found. Scanned PDFs need OCR, which this demo does not support.')
            else:
                score = compare_texts(resume_text, job_description)
                resume_skills = detect_skills(resume_text)
                job_skills = detect_skills(job_description)
                st.metric('TF-IDF text similarity', f'{score:.1f}%')
                st.progress(min(score / 100, 1.0))
                st.subheader('Skills detected in job description')
                st.write(', '.join(sorted(job_skills)) if job_skills else 'None from the built-in skills list')
                st.subheader('Matching skills')
                st.write(', '.join(sorted(resume_skills & job_skills)) or 'None detected')
                st.subheader('Skills not detected in resume')
                st.write(', '.join(sorted(job_skills - resume_skills)) or 'None detected')
                st.info('The score is a text-similarity measure, not an employability probability. Skill matching uses a limited keyword dictionary.')
        except Exception as exc:
            st.error(f'Unable to read or analyze this PDF: {exc}')
