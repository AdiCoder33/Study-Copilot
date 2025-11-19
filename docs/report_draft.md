# EduWeave - AI Study Copilot for PDFs (Draft Report)

## 1. Title & Objective
- Working title and a short paragraph describing the student pain-point (PDF overload) and EduWeave's goal.

## 2. Tools / Frameworks
- Python, Streamlit, FAISS, `pypdf`.
- `sentence-transformers/all-MiniLM-L6-v2` for embeddings.
- Hugging Face Inference API (e.g., `HuggingFaceH4/zephyr-7b-beta`) for Q&A, summaries, and MCQs.
- Brief rationale for each choice (open weights, free-tier friendly, quick iteration).

## 3. Approach / Workflow Summary
- Bullet the flow: upload -> extraction -> cleaning -> chunking -> embeddings -> FAISS -> retrieval -> generation.
- Include note about RAG mode vs. baseline "whole corpus" mode.

## 4. Key Implementation Steps
- Extraction & cleaning details (pypdf quirks, newline cleanup).
- Chunking experiments (sizes, overlaps) and what worked best.
- Vector store construction (batching into FAISS, handling metadata).
- Prompting patterns and temperature tuning for Q&A, summaries, MCQs.

## 5. Results / Observations
- Summaries of experiment log entries: what parameters changed, how answer quality or hallucinations shifted.
- Examples or screenshots from the Streamlit app.

## 6. Learnings / Future Improvements
- What delivered the best value (e.g., longer chunks reduced fragmentation).
- Future ideas: reranking, better UI, persistent vector store, progress tracking.

> Keep this document to ~2-3 pages when polished. Use `docs/experiments.md` to inject real data into section 5.
