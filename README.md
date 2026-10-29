# 🎓 ExamPrep Studio
**A RAG-powered study-pack generator with cited questions and source-grounded answer grading**

Upload your syllabus or notes (PDF / TXT / MD), pick a topic, and ExamPrep Studio generates **MCQs, short-answer
questions, flashcards and a summary**, all grounded in retrieved passages with **page citations**. Write your own
answer and the app **grades it against your notes** and explains exactly what you missed.

## Why RAG?
A plain LLM can invent facts. Here the model may only use passages retrieved from *your* material, and citations are
resolved from real chunk IDs, so every page reference is genuine.

## Architecture
```
 PDF/TXT/MD ──► ingestion ──► chunking ──► embeddings ──► vector store
                (pages)      (sentence-     (SBERT or      (cosine + MMR)
                              aware)         TF-IDF)             │
                                                                 ▼
 topic ──────────────────────────────────────────────────► Retriever.retrieve(topic, k)
                                                                 │ top-k Hits (text + page)
                                                                 ▼
                              prompts ──► LLM (Claude) ──► JSON ──► generators ──► Summary / MCQ / Short / Flashcards
 student answer ──► Retriever (re-retrieve evidence) ──► LLM grader ──► score, covered, missed, feedback, citations
                                                                 │
                                                          Streamlit UI + CSV/Markdown export
```

## Setup
```bash
git clone <your-repo-url> && cd examprep-studio-rag
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -r requirements-embeddings.txt             # recommended: semantic retrieval
cp .env.example .env     # add a FREE Gemini key from aistudio.google.com/app/apikey (see .env.example)
streamlit run app.py
```
**AI options (all selectable in `.env`):** `gemini` (free tier), `ollama` (free, runs on your laptop), `openai_compat` (Groq, OpenRouter...), `anthropic` (paid), or `mock` (fake answers for a UI demo, no key). Free tiers have low rate limits, so the app retries automatically when it is told to slow down.

## Tests and retrieval evaluation
```bash
pytest -q
python scripts/eval_retrieval.py --backend tfidf --k 3     # compare with --backend sbert
```

## Team and ownership
| Area | Owner | Files |
|---|---|---|
| Data and retrieval (ingestion, chunking, embeddings, vector store, retriever, sidebar UI, CI, eval) | **Person A** | `config.py models.py ingestion.py chunking.py embeddings.py vectorstore.py retriever.py pipeline.py ui_sidebar.py scripts/ sample_data/ .github/` |
| LLM and app (prompts, generators, grader, exporters, main UI) | **Person B** | `llm.py prompts.py generators.py grader.py exporters.py app.py docs/` |

The only interface between them: `Retriever.retrieve(query, k) -> list[Hit]` (see `models.py`).

## Limitations / future work
Scanned PDFs need OCR; TF-IDF fallback is lexical only; add a re-ranker, persistent index and spaced-repetition scheduling.
