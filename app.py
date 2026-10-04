"""Chess Mentor - Streamlit front end.  Run: streamlit run app.py"""
import pandas as pd
import streamlit as st

from src import config, dashboard, evaluate
from src.generate import get_llm
from src.pipeline import ChessRAG
from src.usage_log import log_question, log_feedback, read_events, usage_summary

st.set_page_config(page_title="Chess Mentor", page_icon="♟️", layout="wide")

REASONS = {
    "low_retrieval_confidence": "Nothing in the knowledge base matched your question well enough.",
    "unknown_terms": "Your question uses words that never appear in the knowledge base.",
    "no_content_words": "I couldn't find a real topic in that question - try naming a piece, rule or term.",
    "low_coverage": "The best match only covers a small part of your question.",
    "llm_declined": "The retrieved passages did not contain the answer.",
    "citation_check_failed": "The draft answer did not cite its sources, so it was withheld.",
}


@st.cache_resource
def load_rag():
    return ChessRAG()


rag = load_rag()

with st.sidebar:
    st.header("Settings")
    k = st.slider("Passages given to the model (k)", 1, 5, config.TOP_K)
    debug = st.checkbox("Show retrieval signals", value=False)
    ollama_ok = get_llm().available(config.MODEL)
    if ollama_ok:
        st.success(f"Ollama model `{config.MODEL}` ready")
    else:
        st.error(f"Ollama / `{config.MODEL}` not reachable. Run `ollama serve` and `ollama pull {config.MODEL}`.")
    st.caption(f"Retriever: {config.RETRIEVER}")

tab_ask, tab_eval, tab_live = st.tabs(["♟️ Ask the mentor", "📊 Evaluation dashboard", "📡 Live usage"])


# ---------------------------------------------------------------------------------------------------
# Ask
# ---------------------------------------------------------------------------------------------------
def render_answer(res, qid):
    if res["declined"]:
        st.warning(res["answer"])
        st.caption(REASONS.get(res["reason"], res["reason"]))
    else:
        st.success(res["answer"])
        if res["invalid_citations"]:
            st.error("Warning: the answer cites passages that were not retrieved: " + ", ".join(res["invalid_citations"]))
        elif res["uncited"]:
            st.info("This answer did not cite a source - double-check it against the passages below.")
    c1, c2, c3 = st.columns(3)
    c1.metric("Best BM25 score", f"{res['confidence']:.2f}")
    c2.metric("Latency", f"{res['latency']:.1f}s")
    c3.metric("Route", res["routed"] or "keyword search")
    f1, f2, _ = st.columns([1, 1, 6])
    if f1.button("👍", key="fb_up", help="This answer helped"):
        log_feedback(qid, "up")
        st.toast("Thanks for the feedback")
    if f2.button("👎", key="fb_down", help="This answer was wrong or unhelpful"):
        log_feedback(qid, "down")
        st.toast("Thanks - logged as unhelpful")
    with st.expander("Sources used", expanded=not res["declined"]):
        for c in res["chunks"]:
            mark = "✅ " if c["chunk_id"] in res["cited"] else ""
            st.markdown(f"{mark}**[{c['chunk_id']}]** · {c['doc_title']} · *{c['section']}* · license: {c['license']}")
            if debug:
                st.caption(f"bm25 {c['bm25']:.2f} · char-sim {c['char_sim']:.2f} · coverage {c['coverage']:.2f} · "
                           f"unknown-term ratio {c['oov_ratio']:.2f}")
            st.write(c["text"])


with tab_ask:
    st.title("Chess Mentor")
    st.caption("Beginner chess answers grounded in the FIDE Basic Rules, Wikibooks opening principles and three "
               "project-authored beginner guides. Every answer cites its sources; if the knowledge base can't answer, "
               "the mentor says so.")
    examples = ["What is chess?", "How does the knight move?", "What is a fork?", "Why shouldn't I bring my queen out early?",
                "How does the chess clock work?"]
    cols = st.columns(len(examples))
    for col, ex in zip(cols, examples):
        if col.button(ex):
            st.session_state["q"] = ex
    q = st.text_input("Your question", key="q", placeholder="e.g. Can I castle out of check?")
    if q:
        # Streamlit reruns on every click (including the feedback buttons), so remember the last result
        # instead of asking and logging the same question twice.
        if st.session_state.get("last_key") != (q, k):
            try:
                with st.spinner("Thinking..."):
                    res = rag.ask(q, k=k)
            except Exception as e:
                st.error(f"Could not get an answer from the model. Is Ollama running with `{config.MODEL}` pulled? ({e})")
                st.stop()
            st.session_state.update(last_key=(q, k), last_res=res, last_id=log_question(q, res))
        render_answer(st.session_state["last_res"], st.session_state["last_id"])


# ---------------------------------------------------------------------------------------------------
# Evaluation dashboard
# ---------------------------------------------------------------------------------------------------
def run_panel():
    """Run a fresh evaluation from the page (collapsed by default; it sits above the scorecard so results update at once)."""
    with st.expander("🔄 Run a new check"):
        c1, c2, c3, c4 = st.columns(4)
        stage = c1.selectbox("Check", ["retrieval", "full"], format_func=lambda x: "Quick (finding sources)" if x == "retrieval" else "Full (also writes answers)")
        split = c2.selectbox("Questions", ["test", "dev", "all"], format_func=lambda x: {"test": "Test set", "dev": "Dev set", "all": "All"}[x])
        limit = c3.number_input("Max questions (0 = all)", 0, 200, 0)
        baseline = c4.checkbox("Compare with plain AI (no sources)", value=False, disabled=stage != "full")
        if stage == "full" and not ollama_ok:
            st.error("The full check needs Ollama running. Start it, then reload this page.")
        elif st.button("Run check", type="primary"):
            bar, status = st.progress(0.0), st.empty()

            def progress(i, n, question):
                bar.progress(i / n)
                status.caption(f"[{i}/{n}] {question}")
            with st.spinner("Checking..."):
                _, out = evaluate.run_and_save(stage, split, baseline, int(limit) or None, progress)
            bar.progress(1.0)
            status.success("Done - showing the new results below.")
            st.session_state["eval_pref"] = out.name
            st.session_state["eval_nonce"] = st.session_state.get("eval_nonce", 0) + 1


with tab_eval:
    st.title("How well does the mentor work?")
    st.caption("A scorecard from labelled test questions. Higher is better on every number.")
    run_panel()

    files = sorted(config.EVAL_DIR.glob("results_*.csv"), key=lambda f: f.stat().st_mtime)
    if not files:
        st.info("No results yet. Open **Run a new check** above, or run `py -m src.evaluate --stage retrieval --split test`.")
    else:
        names = [f.name for f in files]
        pref = st.session_state.get("eval_pref")
        current = pref if pref in names else dashboard.pick_default_file(files).name
        choice = st.selectbox("Results shown", names, index=names.index(current),
                              key=f"eval_select_{st.session_state.get('eval_nonce', 0)}",
                              on_change=lambda: st.session_state.update(
                                  eval_pref=st.session_state[f"eval_select_{st.session_state.get('eval_nonce', 0)}"]))
        df = pd.read_csv(config.EVAL_DIR / choice)
        sc = dashboard.scorecard(df)

        text, notes = dashboard.verdict(sc, choice)
        st.info(text)
        for n in notes:
            st.warning(n)

        shown = dashboard.tiles(sc)
        for row_start in range(0, len(shown), 3):
            cols = st.columns(3)
            for col, (label, value, help_text) in zip(cols, shown[row_start:row_start + 3]):
                col.metric(f"{dashboard.band(value)} {label}", "n/a" if value is None else f"{value:.0f}%", help=help_text)
                col.caption(help_text)
        if sc["rag_keyword"] is not None:
            st.success(f"With sources: {sc['rag_keyword']:.0f}% of key facts correct, versus {sc['base_keyword']:.0f}% "
                       "for the same AI model answering with no sources.")

        st.subheader("How it does by kind of question")
        kinds = dashboard.by_type(df)
        if len(kinds):
            st.bar_chart(kinds, y_label="Found the right source (%)")

        st.subheader("Where it struggles")
        weak = dashboard.struggles(df)
        if weak.empty:
            st.success("No mistakes in this set of questions.")
        else:
            st.dataframe(weak, hide_index=True, width="stretch")
            st.caption("Showing up to 10. These are the questions to improve next.")

        with st.expander("🔍 How to read this page"):
            st.markdown(
                "- **Finds the right source**: the system first looks up passages, then writes an answer from them. "
                "This checks that the lookup found the passage that really answers the question.\n"
                "- **Answers valid questions / Declines off-topic questions**: it should answer chess questions and say "
                "'I don't know' to anything else.\n"
                "- **Test set vs dev set**: the dev set was used to tune the system, so the test set gives the honest numbers.\n"
                "- Green is 90% or more, yellow 75-89%, red below 75%.")

        with st.expander("⚙️ More detail (technical)"):
            st.dataframe(df, width="stretch")
            st.download_button("Download this file (CSV)", df.to_csv(index=False), file_name=choice)
            hist = evaluate.read_history()
            if hist:
                h = pd.DataFrame(hist)
                h["when"] = pd.to_datetime(h["ts"], unit="s").dt.strftime("%Y-%m-%d %H:%M")
                cols = [c for c in ("when", "stage", "split", "n", "hit1", "hit3", "mrr", "false_refusal", "faithfulness") if c in h.columns]
                st.caption("Every run is recorded in eval/history.jsonl")
                st.dataframe(h[cols].tail(8).iloc[::-1].round(2), hide_index=True, width="stretch")


# ---------------------------------------------------------------------------------------------------
# Live usage (auto-refreshes every few seconds)
# ---------------------------------------------------------------------------------------------------
with tab_live:
    st.title("Live usage")
    st.caption("Questions asked in this app, logged locally to logs/usage.jsonl (git-ignored). Updates every 5 seconds.")

    @st.fragment(run_every="5s")
    def live_panel():
        events = read_events()
        s = usage_summary(events)
        if not s["n_questions"]:
            st.info("No questions yet - ask something in the first tab.")
            return
        m = st.columns(5)
        m[0].metric("Questions", s["n_questions"])
        m[1].metric("Declined", f"{s['decline_rate']:.0f}%")
        m[2].metric("Answers without citation", f"{s['uncited_rate']:.0f}%" if s["uncited_rate"] == s["uncited_rate"] else "-")
        m[3].metric("Latency mean / p95", f"{s['latency_mean']:.1f}s / {s['latency_p95']:.1f}s")
        m[4].metric("👍 / 👎", f"{s['thumbs_up']} / {s['thumbs_down']}")
        c1, c2 = st.columns(2)
        if s["reasons"]:
            c1.subheader("Why questions were declined")
            c1.bar_chart(pd.Series(s["reasons"]))
        c2.subheader("How questions were routed")
        c2.bar_chart(pd.Series(s["routes"]))
        asks = pd.DataFrame([e for e in events if e.get("type") == "ask"]).tail(15)
        asks["when"] = pd.to_datetime(asks["ts"], unit="s")
        st.subheader("Most recent questions")
        st.dataframe(asks[["when", "question", "declined", "reason", "routed", "latency"]].iloc[::-1], width="stretch")
        st.caption("Declined or 👎 questions are your next evaluation questions: copy them into eval/test_questions.json "
                   "with a label and re-run the evaluation.")

    live_panel()
