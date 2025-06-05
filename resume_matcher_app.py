
from __future__ import annotations

import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Iterable

import pandas as pd
import streamlit as st
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

###############################################################################
# Config & constants
###############################################################################

st.set_page_config(page_title="AI Resume‑JD Matcher", layout="wide")

UPLOADS_DIR = Path("uploads")
DATA_CSV = Path("data.csv")
STOPWORDS: set[str]
try:
    # scikit‑learn's built‑in English stopwords
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS as STOPWORDS  # type: ignore
except ImportError:
    STOPWORDS = set()

###############################################################################
# Helpers
###############################################################################

def ensure_dirs() -> None:
    UPLOADS_DIR.mkdir(exist_ok=True)
    if not DATA_CSV.exists():
        DATA_CSV.write_text("file_path,coverage,similarity,timestamp\n")


def preprocess(text: str) -> list[str]:
    """Lower‑case, keep alphabetic tokens ≥3 chars, drop stopwords."""
    tokens = re.findall(r"[A-Za-z]{3,}", text.lower())
    return [t for t in tokens if t not in STOPWORDS]


def extract_pdf_text(uploaded_file: st.runtime.uploaded_file_manager.UploadedFile | Path) -> str:
    """Return concatenated text from all pages of a PDF."""
    if isinstance(uploaded_file, Path):
        f_obj = open(uploaded_file, "rb")
        reader = PdfReader(f_obj)
    else:
        reader = PdfReader(uploaded_file)
    text = "\n".join(page.extract_text() or "" for page in reader.pages)
    return text


def evaluate_fit(resume_tokens: list[str], jd_tokens: list[str]) -> tuple[float, float, set[str], set[str]]:
    """Return (coverage %, cosine similarity, matched, missing)."""
    jd_unique = set(jd_tokens)
    resume_unique = set(resume_tokens)
    matched = jd_unique & resume_unique
    missing = jd_unique - resume_unique
    coverage = 100 * len(matched) / max(1, len(jd_unique))

    vect = TfidfVectorizer(stop_words="english")
    try:
        tfidf = vect.fit_transform([" ".join(resume_tokens), " ".join(jd_tokens)])
        similarity = float(cosine_similarity(tfidf[0], tfidf[1])[0, 0])
    except ValueError:
        # happens if either doc is empty after preprocessing
        similarity = 0.0

    return coverage, similarity, matched, missing


def save_files_and_log(files: Iterable[st.runtime.uploaded_file_manager.UploadedFile],
                       jd_text: str,
                       coverage: float,
                       similarity: float) -> None:
    ts = datetime.now().isoformat(timespec="seconds")

    log_rows = []
    for f in files:
        dest = UPLOADS_DIR / f.name
        with open(dest, "wb") as out:
            shutil.copyfileobj(f, out)
        log_rows.append({
            "file_path": str(dest),
            "coverage": f"{coverage:.1f}",
            "similarity": f"{similarity:.3f}",
            "timestamp": ts,
        })

    df_existing = pd.read_csv(DATA_CSV) if DATA_CSV.exists() else pd.DataFrame()
    df_new = pd.DataFrame(log_rows)
    pd.concat([df_existing, df_new], ignore_index=True).to_csv(DATA_CSV, index=False)

###############################################################################
# UI
###############################################################################

ensure_dirs()

st.title("🧮 AI Resume ↔️ JD Matcher")

with st.form("matcher_form", clear_on_submit=True):
    col1, col2 = st.columns(2)

    with col1:
        uploaded_files = st.file_uploader(
            "Upload resume PDF (1 per submission preferred)",
            type=["pdf"],
            accept_multiple_files=True,
            help="Drag‑and‑drop a PDF resume",
        )

    with col2:
        jd_text = st.text_area(
            "Job description", height=250,
            placeholder="Paste the target job description here…",
        )

    submitted = st.form_submit_button("⚖️ Evaluate fit", type="primary")

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

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

    # For now evaluate only first resume if multiple
    f0 = uploaded_files[0]
    resume_text = extract_pdf_text(f0)
    resume_tokens = preprocess(resume_text)

    coverage, sim, matched_kw, missing_kw = evaluate_fit(resume_tokens, jd_tokens)

    # Save
    save_files_and_log(uploaded_files, jd_text, coverage, sim)

    # Results UI
    st.success("Evaluation complete!")
    st.metric("Keyword coverage (%)", f"{coverage:.1f}")
    #st.metric("Cosine similarity", f"{sim:.3f}")

    with st.expander("🟢 Matched keywords"):
        st.write(", ".join(sorted(matched_kw)) or "None")
    with st.expander("🔴 Missing keywords"):
        st.write(", ".join(sorted(missing_kw)) or "None")

###############################################################################
# Log display
###############################################################################

if DATA_CSV.exists():
    st.subheader("📜 Evaluation history")
    st.dataframe(pd.read_csv(DATA_CSV), use_container_width=True)

###############################################################################
# Sidebar
###############################################################################

with st.sidebar.expander("ℹ️ About", expanded=False):
    st.markdown(
        """
        **AI Resume‑JD Matcher** quickly gauges how well a resume aligns
        with a job description by comparing keyword overlap and TF‑IDF
        similarity. Use it to spot skill gaps or tailor your CV before
        applying.
        """
    )
