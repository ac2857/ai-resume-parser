from __future__ import annotations

import shutil
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Set

import pandas as pd
import streamlit as st
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import spacy
from spacy.lang.en.stop_words import STOP_WORDS as SPACY_STOP

###############################################################################
# Config & constants
###############################################################################

st.set_page_config(page_title="AI Resume‑JD Matcher", layout="wide")

UPLOADS_DIR = Path("uploads")
DATA_CSV = Path("data.csv")

CUSTOM_STOP: Set[str] = {
    "responsibilities", "responsibility", "requirement", "requirements", "role",
    "job", "duties", "skill", "skills", "experience", "candidate", "position",
    "description", "year", "years", "ability", "including", "related", "knowledge",
}
STOPWORDS: Set[str] = set().union(SPACY_STOP, CUSTOM_STOP)

###############################################################################
# spaCy loader – auto‑installs model if missing
###############################################################################

@st.cache_resource(show_spinner=False)
def get_nlp():
    model_name = "en_core_web_sm"
    try:
        return spacy.load(model_name, disable=["ner", "parser"])
    except OSError:
        with st.spinner("Downloading spaCy language model… (~10 MB one‑time)"):
            from spacy.cli import download as spacy_download
            spacy_download(model_name)
        return spacy.load(model_name, disable=["ner", "parser"])

###############################################################################
# Helper functions
###############################################################################

def ensure_dirs() -> None:
    UPLOADS_DIR.mkdir(exist_ok=True)
    if not DATA_CSV.exists():
        DATA_CSV.write_text("file_path,coverage,similarity,timestamp\n")


def preprocess(text: str) -> List[str]:
    nlp = get_nlp()
    doc = nlp(text.lower())
    return [
        tok.lemma_ for tok in doc
        if tok.is_alpha and len(tok) > 2
        and tok.pos_ in {"NOUN", "PROPN", "VERB", "ADJ"}
        and tok.lemma_ not in STOPWORDS
    ]


def extract_pdf_text(file: Path | st.runtime.uploaded_file_manager.UploadedFile) -> str:
    reader = PdfReader(file if isinstance(file, Path) else file)
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def evaluate_fit(resume_tokens: List[str], jd_tokens: List[str]):
    jd_set, resume_set = set(jd_tokens), set(resume_tokens)
    matched, missing = jd_set & resume_set, jd_set - resume_set
    coverage = 100 * len(matched) / max(1, len(jd_set))
    vect = TfidfVectorizer(tokenizer=lambda x: x.split(), lowercase=False)
    tfidf = vect.fit_transform([" ".join(resume_tokens), " ".join(jd_tokens)])
    similarity = float(cosine_similarity(tfidf[0], tfidf[1])[0, 0])
    return coverage, similarity, matched, missing


def save_files_and_log(files: Iterable[st.runtime.uploaded_file_manager.UploadedFile],
                       coverage: float, similarity: float) -> None:
    ts = datetime.now().isoformat(timespec="seconds")
    rows = []
    for f in files:
        dest = UPLOADS_DIR / f.name
        with open(dest, "wb") as out:
            shutil.copyfileobj(f, out)
        rows.append({"file_path": str(dest), "coverage": f"{coverage:.1f}",
                     "similarity": f"{similarity:.3f}", "timestamp": ts})
    df_old = pd.read_csv(DATA_CSV) if DATA_CSV.exists() else pd.DataFrame()
    pd.concat([df_old, pd.DataFrame(rows)], ignore_index=True).to_csv(DATA_CSV, index=False)

###############################################################################
# UI
###############################################################################

ensure_dirs()

st.title("🧮 AI Resume ↔️ JD Matcher (Smart Extract)")

with st.form("matcher_form", clear_on_submit=True):
    col1, col2 = st.columns(2)
    with col1:
        uploaded_files = st.file_uploader("Upload resume PDF", type=["pdf"], accept_multiple_files=True)
    with col2:
        jd_text = st.text_area("Job description", height=250,
                               placeholder="Paste the target job description here…")
    submitted = st.form_submit_button("⚖️ Evaluate fit", type="primary")

if submitted:
    if not uploaded_files:
        st.warning("Please upload at least one PDF resume.")
        st.stop()
    if not jd_text.strip():
        st.warning("Job description cannot be empty.")
        st.stop()

    jd_tokens = preprocess(jd_text)
    if not jd_tokens:
        st.error("Job description yielded no usable keywords after cleaning.")
        st.stop()

    resume_text = extract_pdf_text(uploaded_files[0])
    resume_tokens = preprocess(resume_text)

    coverage, sim, matched_kw, missing_kw = evaluate_fit(resume_tokens, jd_tokens)
    save_files_and_log(uploaded_files, coverage, sim)

    st.success("Evaluation complete!")
    st.metric("Keyword coverage (%)", f"{coverage:.1f}")
    st.metric("Cosine similarity", f"{sim:.3f}")

    with st.expander("🟢 Matched keywords"):
        st.write(", ".join(sorted(matched_kw)) or "None")
    with st.expander("🔴 Missing keywords"):
        st.write(", ".join(sorted(missing_kw)) or "None")

if DATA_CSV.exists():
    st.subheader("📜 Evaluation history")
    st.dataframe(pd.read_csv(DATA_CSV), use_container_width=True)

with st.sidebar.expander("ℹ️ About", expanded=False):
    st.markdown("Downloads spaCy model on first run if necessary; after that, starts instantly.")
