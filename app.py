"""Streamlit UI: upload a transcript -> structured action items.

Run:  streamlit run app.py
"""
import inspect
from datetime import date
from pathlib import Path

import pandas as pd
import streamlit as st

from meeting_actions.evaluate import BackendFallbackError, evaluate_dataset
from meeting_actions.pipeline import run_pipeline
from meeting_actions.schema import STATUSES
from meeting_actions.ui_helpers import UNASSIGNED, filter_rows, items_to_rows, owners_in, summary_line

ROOT = Path(__file__).parent
SAMPLES = ROOT / "data" / "sample_transcripts"


def _stretch(fn) -> dict:
    """Full-width kwarg that works on old and new Streamlit (use_container_width was replaced by width='stretch')."""
    try:
        if "width" in inspect.signature(fn).parameters:
            return {"width": "stretch"}
    except (TypeError, ValueError):
        pass
    return {"use_container_width": True}


st.set_page_config(page_title="Meeting Action-Item Extractor", page_icon="✅", layout="wide")
st.title("✅ Meeting Action-Item Extractor")
st.caption("Upload a transcript and get structured action items: task, owner, deadline, status, confidence.")

with st.sidebar:
    st.header("Settings")
    backend = st.selectbox("Extraction backend", ["rules", "llm", "auto", "transformer"],
                           help="rules = offline; llm = Anthropic API; auto = llm if a key is set, else rules")
    api_key = st.text_input("Anthropic API key (llm/auto)", type="password") or None
    meeting_date = st.date_input("Meeting date", value=date.today(),
                                 help="Relative deadlines ('Friday', 'next week') are resolved from this date.")
    attendees = st.text_input("Attendees who don't speak (comma-separated)",
                              help="So tasks can still be assigned to them, e.g. 'Priya, Marcus'")
    min_conf = st.slider("Hide items below confidence", 0.0, 1.0, 0.3, 0.05)

tab_run, tab_eval = st.tabs(["Extract", "Evaluate"])

with tab_run:
    c1, c2 = st.columns(2)
    uploaded = c1.file_uploader("Transcript (.txt, .vtt, .srt, .md)", type=["txt", "vtt", "srt", "md"])
    sample = c2.selectbox("...or load a sample", ["(none)"] + sorted(p.name for p in SAMPLES.iterdir()))
    text = ""
    if uploaded is not None:
        text = uploaded.getvalue().decode("utf-8", errors="replace")
    elif sample != "(none)":
        text = (SAMPLES / sample).read_text(encoding="utf-8")
    text = st.text_area("Transcript", text, height=240, placeholder="Speaker: sentence ...")

    if st.button("Extract action items", type="primary", disabled=not text.strip()):
        try:
            res = run_pipeline(text, meeting_date, backend=backend, min_confidence=min_conf, api_key=api_key,
                               extra_participants=[a for a in attendees.split(",") if a.strip()])
        except Exception as e:  # keep the UI alive on unexpected input
            st.error(f"Could not process this transcript: {e}")
            st.stop()
        for w in res.warnings:
            st.warning(w)
        st.write(f"**Backend:** {res.backend} · **Participants:** {', '.join(res.participants) or 'n/a'}")
        if res.items:
            all_rows = items_to_rows(res.items)
            pick = st.multiselect("Filter by owner", owners_in(all_rows) + [UNASSIGNED])
            only_review = st.checkbox("Show only items that need review")
            rows = filter_rows(all_rows, pick, only_review)
            st.caption("Edit any cell to correct the result (set status to 'done' when finished), then download.")
            edited = st.data_editor(
                pd.DataFrame(rows), hide_index=True, num_rows="dynamic",
                disabled=["flags", "source", "confidence"],
                column_config={"status": st.column_config.SelectboxColumn(options=list(STATUSES))},
                **_stretch(st.data_editor),
            )
            d1, d2 = st.columns(2)
            d1.download_button("Download JSON", res.to_json(), "action_items.json", "application/json")
            d2.download_button("Download CSV (with your edits)", edited.to_csv(index=False), "action_items.csv", "text/csv")
            st.info(summary_line(res.items))
        else:
            st.info("No action items found.")
        if res.issues:
            with st.expander(f"Validation log ({len(res.issues)})"):
                st.dataframe(pd.DataFrame(res.issues), hide_index=True, **_stretch(st.dataframe))

with tab_eval:
    st.write("Scores the extractor against the gold annotations in `data/*/annotations.json`.")
    e1, e2 = st.columns(2)
    eval_backend = e1.selectbox("Backend to evaluate", ["rules", "llm", "transformer"], key="eval_backend",
                                help="llm needs an API key (sidebar or ANTHROPIC_API_KEY) and costs API calls")
    eval_limit = e2.number_input("Meetings per dataset (0 = all; use a small number for llm)", 0, 100,
                                 0 if eval_backend == "rules" else 5, key="eval_limit")
    if st.button("Run evaluation"):
        data_root = ROOT / "data"
        for d in [data_root] + sorted(p.parent for p in data_root.glob("*/annotations.json")):
            try:
                rep = evaluate_dataset(d, backend=eval_backend, limit=int(eval_limit),
                                       **({"api_key": api_key} if eval_backend == "llm" else {}))
            except BackendFallbackError as e:
                st.error(f"The {eval_backend} backend did not run, so no scores were produced: {e}")
                break
            st.subheader(f"Dataset: {d.name} ({rep['n_meetings']} meetings, {rep['n_gold_items']} gold items)")
            st.dataframe(pd.DataFrame(rep["by_threshold"]), hide_index=True, **_stretch(st.dataframe))
