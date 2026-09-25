import streamlit as st
from src.pipeline import ChessRAG

st.set_page_config(page_title="Chess Mentor RAG", layout="wide")

st.title("♟️ Chess Mentor (RAG System)")
st.write("Ask beginner chess questions. The system answers using a knowledge base.")

st.sidebar.title("💡 Try these questions")

sample_qs = [
    "What is castling?",
    "How does a pawn move?",
    "What is a fork in chess?",
    "Why control the center?"
]

for q in sample_qs:
    if st.sidebar.button(q):
        st.session_state["question"] = q

# Init RAG
@st.cache_resource
def load_rag():
    return ChessRAG()

rag = load_rag()

# Input
question = st.text_input("Enter your question:", key="question")

if st.button("Ask"):
    if question.strip() == "":
        st.warning("Please enter a question.")
    else:
        result = rag.ask(question)

        st.subheader("🧠 Answer")

        if result["confidence"] < 5:
            st.warning("I’m not confident enough to answer this question based on my knowledge base.")
        else:
            st.success(result["answer"])

        st.subheader("📊 Confidence")

        score = result["confidence"]

        if score > 7:
            st.success(f"High confidence ({round(score,2)})")
        elif score > 4:
            st.warning(f"Medium confidence ({round(score,2)})")
        else:
            st.error(f"Low confidence ({round(score,2)})")

        st.subheader("📚 Retrieved Chunks")
        for c in result["chunks"]:
            with st.expander(f"{c['chunk_id']} (score: {round(c['score'],2)})"):
                st.write(c["text"])

        st.subheader("🔍 Why this answer?")
        st.write("The system retrieved relevant knowledge chunks and generated the answer based only on them.")


import pandas as pd

st.sidebar.title("📈 Evaluation")

if st.sidebar.button("Run Evaluation"):
    import subprocess
    subprocess.run(["python", "src/evaluate.py"])

    df = pd.read_csv("eval/results.csv")

    st.sidebar.write("Results Preview:")
    st.sidebar.dataframe(df.head())