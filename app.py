"""ExamPrep Studio - Streamlit app.  Run:  streamlit run app.py"""
import streamlit as st

from examprep.exporters import flashcards_to_csv, study_pack_to_markdown
from examprep.generators import (generate_flashcards, generate_mcqs, generate_short_questions,
                                 generate_summary)
from examprep.grader import grade_answer, grade_mcq
from examprep.llm import get_llm
from examprep.ui_sidebar import render_sidebar

st.set_page_config(page_title="ExamPrep Studio", page_icon="🎓", layout="wide")
st.title("🎓 ExamPrep Studio")
st.caption("Upload your notes → pick a topic → get cited MCQs, short answers, flashcards and a summary. "
           "Every answer is graded against your own material.")

retriever, k = render_sidebar()

try:
    llm = get_llm()
except RuntimeError as e:
    st.error(str(e))
    st.stop()

if retriever is None:
    st.info("👈 Upload notes (or tick the sample notes) and press **Build index** to begin.")
    st.stop()

# ===================== Part 1: topic input + generation =====================
st.subheader("Pick a topic")
topic = st.text_input("Topic or chapter", placeholder="e.g. overfitting and regularization")
c1, c2, c3, c4 = st.columns(4)
n_mcq = c1.number_input("MCQs", 1, 10, 5)
n_short = c2.number_input("Short answers", 1, 6, 3)
n_cards = c3.number_input("Flashcards", 2, 20, 8)
difficulty = c4.selectbox("Difficulty", ["easy", "medium", "hard"], index=1)

if st.button("✨ Generate study pack", type="primary", disabled=not topic.strip()):
    with st.spinner("Retrieving passages and generating..."):
        try:
            hits = retriever.retrieve(topic, k=k)
            st.session_state["pack"] = {
                "topic": topic, "hits": hits,
                "summary": generate_summary(llm, hits, topic),
                "mcqs": generate_mcqs(llm, hits, int(n_mcq), difficulty),
                "shorts": generate_short_questions(llm, hits, int(n_short), difficulty),
                "cards": generate_flashcards(llm, hits, int(n_cards)),
            }
            st.session_state["grades"] = {}
        except Exception as e:  # network / API / JSON problems
            st.error(f"Generation failed: {e}")

pack = st.session_state.get("pack")
if not pack:
    st.stop()


def show_citations(citations):
    st.caption("📖 " + (" · ".join(citations) if citations else "no citation returned"))


tab_sum, tab_mcq, tab_short, tab_cards, tab_src, tab_export = st.tabs(
    ["📝 Summary", "✅ MCQs", "✍️ Short answers", "🃏 Flashcards", "🔎 Sources", "⬇️ Export"])

with tab_sum:
    s = pack["summary"]
    st.write(s.text)
    for p in s.key_points:
        st.markdown(f"- {p}")
    show_citations(s.citations)

with tab_cards:
    for i, c in enumerate(pack["cards"], 1):
        with st.expander(f"{i}. {c.front}"):
            st.write(c.back)
            show_citations(c.citations)

# ===================== Part 2: MCQ quiz =====================
with tab_mcq:
    for i, q in enumerate(pack["mcqs"]):
        st.markdown(f"**Q{i + 1}. {q.question}**")
        choice = st.radio("Your answer", range(len(q.options)), index=None, key=f"mcq_{i}",
                          format_func=lambda j, q=q: f"{chr(65 + j)}. {q.options[j]}", label_visibility="collapsed")
        if choice is not None:
            if grade_mcq(q, choice):
                st.success("Correct!")
            else:
                st.error(f"Not quite. Correct answer: {chr(65 + q.answer_index)}. {q.options[q.answer_index]}")
            st.write(q.explanation)
            show_citations(q.citations)
        st.divider()

# ===================== Part 3: short answers graded against the source =====================
with tab_short:
    for i, q in enumerate(pack["shorts"]):
        st.markdown(f"**Q{i + 1}. {q.question}**")
        answer = st.text_area("Your answer", key=f"short_{i}", height=110)
        if st.button("Grade my answer", key=f"grade_{i}"):
            with st.spinner("Grading against your notes..."):
                try:
                    st.session_state["grades"][i] = grade_answer(
                        llm, retriever, q.question, answer, reference_answer=q.model_answer)
                except Exception as e:
                    st.error(f"Grading failed: {e}")
        g = st.session_state.get("grades", {}).get(i)
        if g:
            icon = {"correct": "🟢", "partial": "🟡", "incorrect": "🔴"}[g.verdict]
            st.metric("Score", f"{g.score}/10", help=g.verdict)
            st.write(f"{icon} **{g.verdict.title()}** — {g.feedback}")
            if g.covered:
                st.markdown("**You covered:** " + "; ".join(g.covered))
            if g.missed:
                st.markdown("**You missed:** " + "; ".join(g.missed))
            show_citations(g.citations)
            with st.expander("Model answer"):
                st.write(q.model_answer)
        st.divider()

# ===================== Part 4: sources + export =====================
with tab_src:
    st.caption("These are the retrieved passages every output was generated from.")
    for i, h in enumerate(pack["hits"], 1):
        with st.expander(f"[S{i}] {h.chunk.citation}  ·  relevance {h.score:.2f}"):
            st.write(h.chunk.text)

with tab_export:
    st.download_button("Download flashcards (Anki CSV)", flashcards_to_csv(pack["cards"]),
                       file_name="flashcards.csv", mime="text/csv")
    st.download_button("Download full study pack (Markdown)",
                       study_pack_to_markdown(pack["topic"], pack["summary"], pack["mcqs"],
                                              pack["shorts"], pack["cards"]),
                       file_name="study_pack.md", mime="text/markdown")
