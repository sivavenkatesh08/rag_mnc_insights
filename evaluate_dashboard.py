import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Load results
df = pd.read_csv("evaluation_results.csv")

st.title("📊 RAG System Evaluation Dashboard")

# Show table
st.subheader("🔍 Evaluation Summary")
st.dataframe(df.style.background_gradient(cmap="YlGn"))

# Show average scores
st.subheader("📈 Average Metrics")
col1, col2 = st.columns(2)
col1.metric("Avg. Keyword Match", f"{df['keywords_matched'].mean():.2f}")
col2.metric("Avg. Text Similarity", f"{df['text_similarity'].mean():.2f}")

# Plot scores
st.subheader("📉 Evaluation Scores by Question")

fig, ax = plt.subplots(figsize=(10, 5))
df.plot.bar(x="question", y=["keywords_matched", "text_similarity"], ax=ax)
plt.xticks(rotation=45, ha="right")
plt.ylabel("Score")
plt.ylim(0, 1)
st.pyplot(fig)

# Filters
st.sidebar.header("🔍 Filter Questions")
selected_q = st.sidebar.selectbox("Choose a question", ["All"] + df["question"].tolist())
if selected_q != "All":
    st.write("### 🎯 Selected Question")
    st.write(df[df["question"] == selected_q])
