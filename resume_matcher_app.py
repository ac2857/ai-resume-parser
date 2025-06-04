from __future__ import annotations

import shutil
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

###############################################################################
# Page config & helpers
###############################################################################

st.set_page_config(page_title="PDF + Notes Collector", layout="wide")

UPLOADS_DIR = Path("uploads")
DATA_CSV = Path("data.csv")


def ensure_dirs() -> None:
    UPLOADS_DIR.mkdir(exist_ok=True)
    if not DATA_CSV.exists():
        DATA_CSV.write_text("file_path,notes,timestamp\n")


def save_files(files: list[st.runtime.uploaded_file_manager.UploadedFile], notes: str) -> pd.DataFrame:
    """Persist *files* to disk and return log rows as a DataFrame."""
    ts = datetime.now().isoformat(timespec="seconds")
    rows = []
    for f in files:
        dest = UPLOADS_DIR / f.name
        with open(dest, "wb") as out:
            shutil.copyfileobj(f, out)
        rows.append({"file_path": str(dest), "notes": notes, "timestamp": ts})
    return pd.DataFrame(rows)


def load_log() -> pd.DataFrame:
    return pd.read_csv(DATA_CSV) if DATA_CSV.exists() else pd.DataFrame(columns=["file_path", "notes", "timestamp"])


def write_log(df: pd.DataFrame) -> None:
    df.to_csv(DATA_CSV, index=False)

###############################################################################
# Main UI
###############################################################################

ensure_dirs()
st.title("AI Resume Parser and Analyzer")

###############################################################################
# Upload + notes form (auto‑clears on submit)
###############################################################################

with st.form("upload_form", clear_on_submit=True):
    left, right = st.columns(2)

    with left:
        uploaded_files = st.file_uploader(
            "Upload PDF file(s)",
            type=["pdf"],
            accept_multiple_files=True,
            help="Drag‑and‑drop one or more PDF resumes, papers, etc.",
        )

    with right:
        notes = st.text_area(
            "Notes / description",
            height=250,
            placeholder="Enter notes that relate to the uploaded PDF(s)…",
        )

    submitted = st.form_submit_button("💾 Save entry", type="primary")

if submitted:
    if uploaded_files:
        new_rows_df = save_files(uploaded_files, notes)

        if "log_df" not in st.session_state:
            st.session_state.log_df = load_log()
        st.session_state.log_df = pd.concat([st.session_state.log_df, new_rows_df], ignore_index=True)
        write_log(st.session_state.log_df)

        st.success(f"Resume uploaded")
    else:
        st.warning("Please upload at least one PDF before saving.")

###############################################################################
# Display log
###############################################################################

# log_df = st.session_state.get("log_df", load_log())

# st.subheader("📑 Existing Entries")
# if log_df.empty:
#     st.info("No entries yet. Upload a PDF and add some notes to get started!")
# else:
#     st.dataframe(log_df, use_container_width=True)

###############################################################################
# Sidebar help
###############################################################################

with st.sidebar.expander("ℹ️ How to use this app", expanded=False):
    st.markdown(
        """
        1. **Upload** your resume using the left‑hand widget.
        2. **Type** job role description on the right.
        3. Press **Upload** – 
        4. Your job role matching is generated
        """
    )
