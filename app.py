# app.py
import streamlit as st
import os
import pdfplumber
from matcher import match_resumes_to_jd, extract_skills, clean_text

st.set_page_config(page_title="AI Resume Matcher", page_icon="📄", layout="wide")
st.title("AI-Powered Resume Matcher")
st.write("Paste a job description and upload PDF resumes to see semantic matches and skill gaps.")

jd_text_input = st.text_area("✍️ Paste Job Description Here", height=300)
uploaded_pdfs = st.file_uploader("📄 Upload Resumes (PDF only)", type=['pdf'], accept_multiple_files=True)

def pdf_to_text(file_obj):
    text = ""
    with pdfplumber.open(file_obj) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

if st.button("Run analysis"):
    if not jd_text_input.strip():
        st.error("Please paste a job description.")
    elif not uploaded_pdfs:
        st.error("Please upload at least one PDF resume.")
    else:
        jd_text = clean_text(jd_text_input)

        
        resume_dir = 'uploaded_resumes'
        os.makedirs(resume_dir, exist_ok=True)
        for old in os.listdir(resume_dir):
            os.remove(os.path.join(resume_dir, old))

        
        for pdf in uploaded_pdfs:
            resume_text = pdf_to_text(pdf)
            with open(os.path.join(resume_dir, pdf.name + ".txt"), 'w', encoding='utf-8') as f:
                f.write(resume_text)

        st.subheader("Matching Results")
        df = match_resumes_to_jd(resume_dir, jd_text)
        st.dataframe(df)

        # Skill gap analysis
        st.subheader("Skill Gap Analysis")
        jd_skills = set(extract_skills(jd_text))
        for resume_name in df['Resume']:
            with open(os.path.join(resume_dir, resume_name), 'r', encoding='utf-8') as f:
                resume_text = f.read()
            resume_skills = set(extract_skills(resume_text))
            missing = jd_skills - resume_skills
            st.markdown(
                f"**{resume_name}** missing: {', '.join(list(missing)[:20]) if missing else 'No major gaps detected!'}"
            )
