# matcher.py
import os
import re
import spacy
import pandas as pd
from sentence_transformers import SentenceTransformer, util

nlp = spacy.load("en_core_web_sm")
model = SentenceTransformer('all-MiniLM-L6-v2')

def clean_text(text: str) -> str:
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def extract_skills(text: str):
    doc = nlp(text)
    skills = set()
    for chunk in doc.noun_chunks:
        skills.add(chunk.text.lower())
    for token in doc:
        if token.pos_ == 'PROPN':
            skills.add(token.text.lower())
    return list(skills)

def encode_text(text: str):
    return model.encode([text], convert_to_tensor=True)

def compute_similarity(resume_text: str, jd_text: str) -> float:
    emb_resume = encode_text(resume_text)
    emb_jd = encode_text(jd_text)
    score = util.pytorch_cos_sim(emb_resume, emb_jd)
    return float(score.item())

def match_resumes_to_jd(resume_dir: str, jd_text: str):
    jd_text = clean_text(jd_text)
    results = []
    for fname in os.listdir(resume_dir):
        if fname.endswith('.txt'):
            with open(os.path.join(resume_dir, fname), 'r', encoding='utf-8') as f:
                resume_text = clean_text(f.read())
                sim = compute_similarity(resume_text, jd_text)
                results.append((fname, sim))
    df = pd.DataFrame(results, columns=['Resume', 'Similarity'])
    df.sort_values(by='Similarity', ascending=False, inplace=True)
    return df
