import asyncio
import sys
from pathlib import Path

import streamlit as st
import pandas as pd

# make `src` importable when launched via `streamlit run src/ui/ui.py`
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.agent.agent import answer
from src.models import Evidence, PDFEvidence, SQLEvidence

BLURB = (
    "Ask natural-language questions about Apple, Microsoft, and Alphabet's "
    "10-K filings (FY2023-2025). The system routes to a financial database "
    "for exact numbers and the filings for narrative context."
)


def render_sql_evidence(index: int, ev: SQLEvidence) -> None:
    st.markdown(f"**{index}. SQL query**")
    st.code(ev.query, language="sql")
    if ev.rows:
        # static table renders all columns/rows at full width, no scroll widget to miss
        st.table(pd.DataFrame(ev.rows))
    else:
        st.caption("No rows returned.")


def render_pdf_evidence(index: int, ev: PDFEvidence) -> None:
    score = ev.score if ev.score is not None else 0.0
    header = (
        f"**{index}. PDF — {ev.company_ticker} FY{ev.fiscal_year} "
        f"· p.{ev.page_number} · score {score:.2f}**"
    )
    st.markdown(header)
    # fixed-height scrollable box 
    with st.container(height=180):
        st.text(ev.text)


def render_evidence(evidence: list[Evidence]) -> None:
    with st.expander(f"Evidence ({len(evidence)})", expanded=False):
        if not evidence:
            st.write("No evidence was retrieved for this answer.")
            return
        for i, ev in enumerate(evidence, start=1):
            if ev.source == "sql":
                render_sql_evidence(i, ev)
            else:
                render_pdf_evidence(i, ev)
            st.divider()


st.set_page_config(page_title="Sumer's RAG Application", layout="wide")
st.markdown(BLURB)

user_input = st.text_area(
    label="Enter your natural language query here",
    placeholder="Type your query here",
    height=250,
)

ask = st.button("Ask", type="primary")

if ask:
    if not user_input.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Thinking..."):
            response = asyncio.run(answer(user_input))

        st.markdown("### Answer")
        st.markdown(response.answer)

        if response.sources_used:
            sources = ", ".join(response.sources_used)
        else:
            sources = "none"
        st.caption(f"Sources used: {sources}")

        render_evidence(response.evidence)
