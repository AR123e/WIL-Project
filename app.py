"""Chess Mentor - Streamlit front end.  Run: streamlit run app.py"""
import json
import pandas as pd
import streamlit as st

from src.config import EVAL_DIR, MIN_SCORE, MAX_QUESTIONS
from src.pipeline import ChessRAG

st.set_page_config(page_title="Chess Mentor", page_icon="♟️", layout="wide")


@st.cache_resource
def load_rag():
    return ChessRAG()


rag = load_rag()
tab_ask, tab_eval = st.tabs(["♟️ Ask the mentor", "📊 Evaluation dashboard"])

with tab_ask:
    st.title("Chess Mentor")
    st.caption("Beginner chess answers grounded in the FIDE Basic Rules and Wikibooks opening principles. "
               "Every answer cites its source; if the knowledge base can't answer, the mentor says so.")
    examples = ["How does the knight move?", "Why shouldn't I bring my queen out early?",
                "What is stalemate?", "How does the chess clock work?"]
    cols = st.columns(len(examples))
    for col, ex in zip(cols, examples):
        if col.button(ex, use_container_width=True):
            st.session_state["q"] = ex
    q = st.text_input("Your question", key="q", placeholder="e.g. Can I castle out of check?")
    if q:
        asked=st.session_state.setdefault("asked",[])
        if q not in asked:
            if len(asked)>=MAX_QUESTIONS:
                st.warning(f"ypu've reached the limit of {MAX_QUESTIONS} questions for this session. Refresh the page to start a new session.")
                st.stop()
            asked.append(q)
            st.caption(f"Questioned used :{len(asked)}/{MAX_QUESTIONS}")
            
            try:
                with st.spinner("Thinking..."):
                    res = rag.ask(q)
            except Exception as e:
                st.error(f"Could not reach Ollama. Is it running (`ollama serve`) with llama3.1 pulled? ({e})")
                st.stop()
            if res["declined"]:
                st.warning(res["answer"])
            else:
                st.success(res["answer"])
            c1, c2 = st.columns(2)
        c1.metric("Retrieval confidence (BM25)", f"{res['confidence']:.2f}", help=f"Declines below {MIN_SCORE}")
        c2.metric("Latency", f"{res['latency']:.1f}s")
        with st.expander("Sources used"):
            for c in res["chunks"]:
                st.markdown(f"**[{c['chunk_id']}]** · {c['doc_title']} · *{c['section']}* · score {c['score']:.2f}")
                st.write(c["text"])

with tab_eval:
    st.title("Evaluation dashboard")
    st.write("Run `python -m src.evaluate --stage full --baseline` to refresh these results.")
    for stage in ("full", "retrieval"):
        path = EVAL_DIR / f"results_{stage}.csv"
        if path.exists():
            df = pd.read_csv(path)
            ins = df[df["type"] != "out_of_scope"]
            oos = df[df["type"] == "out_of_scope"]
            m = st.columns(5)
            m[0].metric("hit@1", f"{100*ins['hit_at_1'].mean():.0f}%")
            m[1].metric("hit@3", f"{100*ins['hit_at_3'].mean():.0f}%")
            m[2].metric("MRR", f"{ins['rr'].mean():.2f}")
            m[3].metric("False refusals", f"{100*ins['unanswered'].mean():.0f}%")
            m[4].metric("Correct refusals", f"{100*oos['unanswered'].mean():.0f}%")
            if stage == "full" and "faithful" in df:
                a = ins[~ins["declined"].astype(bool)]
                g = st.columns(4)
                g[0].metric("Faithfulness", f"{100*a['faithful'].astype(float).mean():.0f}%")
                g[1].metric("Citation correct", f"{100*a['cite_correct'].astype(float).mean():.0f}%")
                g[2].metric("Keyword correctness", f"{100*a['keyword_score'].mean():.0f}%")
                g[3].metric("Flesch ease", f"{a['flesch'].mean():.0f}")
            st.subheader("Hit@3 by question type")
            st.bar_chart(ins.groupby("type")["hit_at_3"].mean())
            st.subheader("Per-question results")
            st.dataframe(df, use_container_width=True)
            break
    else:
        st.info("No results yet — run the evaluation first.")
