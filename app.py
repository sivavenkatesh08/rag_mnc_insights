import streamlit as st
from pathlib import Path
from rag_pipeline_gemini import rag_query
from fpdf import FPDF
import pandas as pd
import unicodedata

import unicodedata

def safe_text(text):
    """Convert unicode text to latin-1 compatible format."""
    return unicodedata.normalize("NFKD", text).encode("latin-1", "ignore").decode("latin-1")


# Constants
INDEX_PATH = Path(r"C:\22ad053\Navigate Labs\rag_mnc_insights\data\outputs\mnc_faiss_index")

st.set_page_config(page_title="RAG MNC Insights", layout="wide")
st.title("🔍 MNC Insights Assistant - RAG Powered by Gemini")

st.markdown("""
Ask questions based on quarterly earnings call transcripts of top MNCs like Microsoft, Apple, Google, etc.
""")

# --- Filter Section ---
st.subheader("📁 Filter Options (Optional)")
col1, col2, col3 = st.columns(3)
company = col1.selectbox("Company", ["", "MSFT", "AAPL", "AMZN", "GOOGL", "NVDA"])
year = col2.selectbox("Year", ["", "2016", "2017", "2018", "2019", "2020", "2021", "2022", "2023", "2024"])
quarter = col3.selectbox("Quarter", ["", "Q1", "Q2", "Q3", "Q4"])

# --- Question Input ---
st.subheader("🧠 Ask your question")
user_question = st.text_input("Type your question here")

if st.button("Get Answer") and user_question:
    with st.spinner("Thinking with Gemini..."):
        result = rag_query(
            INDEX_PATH,
            user_question,
            streamlit_mode=True,
            company=company,
            year=year,
            quarter=quarter
        )

        st.success("Answer retrieved!")

        st.markdown("## 📌 Answer")
        st.write(result["answer"])

        st.markdown("## 📄 Sources")
        for src in result["sources"]:
            st.markdown(f"- {src}")

       # PDF Download
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.multi_cell(0, 10, safe_text(f"Question: {user_question}"))
        pdf.ln()

        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 10, safe_text("Answer:"), ln=True)
        pdf.set_font("Arial", size=12)
        pdf.multi_cell(0, 10, safe_text(result["answer"]))

        pdf.ln()
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 10, safe_text("Sources:"), ln=True)
        for src in result["sources"]:
            pdf.cell(0, 10, safe_text(f"- {src}"), ln=True)

        pdf_path = "rag_answer.pdf"
        pdf.output(pdf_path)
        with open(pdf_path, "rb") as f:
            st.download_button("⬇️ Download Answer as PDF", f, file_name="RAG_Answer.pdf")


# --- Evaluation Section ---
with st.sidebar:
    st.markdown("---")
    st.subheader("📊 Evaluation Dashboard")
    show_eval = st.checkbox("Show Evaluation Metrics")

if show_eval:
    st.header("📊 RAG System Evaluation")
    try:
        df = pd.read_csv("evaluation_results.csv")

        st.subheader("✅ Keyword Match (%)")
        st.bar_chart(df["keywords_matched"] * 100)

        st.subheader("🧠 Text Similarity (%)")
        st.bar_chart(df["text_similarity"] * 100)

        st.subheader("📋 Raw Results")
        st.dataframe(df[["question", "keywords_matched", "text_similarity"]])

    except FileNotFoundError:
        st.error("❌ evaluation_results.csv not found. Please run the evaluator first.")
