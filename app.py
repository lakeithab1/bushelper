"""Streamlit chat page for the Syracuse bus helper.

Run with:  streamlit run app.py
"""
import os
from datetime import datetime

import streamlit as st

import assistant
import gtfs_tools

# Get the API key from .streamlit/secrets.toml if it exists (never put it in code).
try:
    if "OPENAI_API_KEY" in st.secrets:
        os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
except Exception:
    pass

MAX_QUESTIONS = 25  # per browser session, to keep API costs in check

st.set_page_config(page_title="Syracuse bus helper", page_icon=None)
st.title("Syracuse bus helper (class prototype)")
st.caption("Answers come from Centro's scheduled data and verified notes. Not live arrivals.")

cov = gtfs_tools.feed_coverage()
today = datetime.now(gtfs_tools.TZ).strftime("%Y%m%d")
feed_is_current = cov["start"] <= today <= cov["end"]

with st.sidebar:
    st.subheader("Schedule file")
    st.write(f"Runs {cov['start']} to {cov['end']}")
    st.write(cov["version"])
    if not feed_is_current:
        st.warning(
            "This schedule file does not cover today. Use a test date below, "
            "or replace the files in the data folder with the current Centro feed."
        )
    use_test = st.checkbox("Use a test date and time", value=not feed_is_current)
    test_date = st.text_input("Test date (YYYY-MM-DD)", "2026-07-15") if use_test else None
    test_time = st.text_input("Test time (HH:MM)", "08:00") if use_test else None
    show_sources = st.checkbox("Show lookups the assistant made", value=True)

if "history" not in st.session_state:
    st.session_state.history = []

for m in st.session_state.history:
    with st.chat_message(m["role"]):
        st.write(m["content"])

if "count" not in st.session_state:
    st.session_state.count = 0

question = st.chat_input("Ask about Centro buses near campus")
if question and st.session_state.count >= MAX_QUESTIONS:
    st.info("This demo is limited to 25 questions per session. Refresh the page to start over.")
elif question:
    st.session_state.count += 1
    st.session_state.history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)
    with st.chat_message("assistant"):
        try:
            answer, tool_log = assistant.ask(
                st.session_state.history, test_date=test_date, test_time=test_time
            )
        except Exception as e:
            answer, tool_log = f"Something went wrong: {e}", []
        st.write(answer)
        if show_sources and tool_log:
            with st.expander("Lookups made"):
                for t in tool_log:
                    st.code(f"{t['tool']}({t['args']})")
    st.session_state.history.append({"role": "assistant", "content": answer})
