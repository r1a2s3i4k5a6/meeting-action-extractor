"""Streamlit UI: upload a transcript -> structured action items.

Run:  streamlit run app.py
"""
from datetime import date
from pathlib import Path

import pandas as pd
import streamlit as st

from meeting_actions.evaluate import evaluate_dataset
from meeting_actions.pipeline import run_pipeline

SAMPLES = Path(__file__).parent / "data" / "sample_transcripts"

st.set_page_config(page_title="Meeting Action-Item Extractor", page_icon="✅", layout="wide")
st.title("✅ Meeting Action-Item Extractor")
st.caption("Upload a transcript and get structured action items: task, owner, deadline, status, confidence.")

with st.sidebar:
    st.header("Settings")
    backend = st.selectbox("Extraction backend", ["rules", "llm", "auto"],
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
        res = run_pipeline(text, meeting_date, backend=backend, min_confidence=min_conf, api_key=api_key,
                           extra_participants=[a for a in attendees.split(",") if a.strip()])
        for w in res.warnings:
            st.warning(w)
        st.write(f"**Backend:** {res.backend} · **Participants:** {', '.join(res.participants) or 'n/a'}")
        if res.items:
            df = pd.DataFrame([i.to_dict() for i in res.items])
            df["flags"] = df["flags"].apply(", ".join)
            owners = sorted({o for o in df["owner"].dropna()})
            pick = st.multiselect("Filter by owner", owners + ["(unassigned)"])
            if pick:
                mask = df["owner"].isin(pick) | (df["owner"].isna() & ("(unassigned)" in pick))
                df = df[mask]
            only_review = st.checkbox("Show only items that need review")
            if only_review:
                df = df[df["status"] == "needs_review"]
            st.caption("Edit any cell to correct the result, then download.")
            edited = st.data_editor(
                df[["task", "owner", "deadline", "status", "confidence", "flags", "source"]],
                use_container_width=True, hide_index=True, num_rows="dynamic",
                disabled=["flags", "source", "confidence"],
                column_config={"status": st.column_config.SelectboxColumn(options=["open", "needs_review", "done"])},
            )
            d1, d2 = st.columns(2)
            d1.download_button("Download JSON", res.to_json(), "action_items.json", "application/json")
            d2.download_button("Download CSV (with your edits)", edited.to_csv(index=False), "action_items.csv", "text/csv")
            n_review = sum(i.status == "needs_review" for i in res.items)
            st.info(f"{len(res.items)} action items, {n_review} need review (missing owner, past deadline, low confidence...).")
        else:
            st.info("No action items found.")
        if res.issues:
            with st.expander(f"Validation log ({len(res.issues)})"):
                st.dataframe(pd.DataFrame(res.issues), use_container_width=True, hide_index=True)

with tab_eval:
    st.write("Scores the extractor against gold annotations in `data/annotations.json`.")
    if st.button("Run evaluation"):
        data_root = Path(__file__).parent / "data"
        for d in [data_root] + sorted(p.parent for p in data_root.glob("*/annotations.json")):
            rep = evaluate_dataset(d, backend="rules")
            st.subheader(f"Dataset: {d.name}")
            st.dataframe(pd.DataFrame(rep["by_threshold"]), use_container_width=True, hide_index=True)
