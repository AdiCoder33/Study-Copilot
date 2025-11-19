# EduWeave - AI Study Copilot for PDFs

EduWeave is a Streamlit demo that helps students make sense of dense PDF notes.  
Upload one or more PDFs, build an in-memory FAISS index with Sentence Transformer embeddings, and explore:

- Retrieval-grounded Q&A with the supporting chunks shown
- Structured summaries aimed at engineering undergrads
- Auto-generated MCQ practice sets with answers and explanations

The emphasis is on workflow clarity and experimentation (chunks, retrieval top-k, temperatures) rather than UI polish.

## Tech Stack

- Python 3.11+
- Streamlit for the UI
- `pypdf` for parsing PDFs
- `sentence-transformers/all-MiniLM-L6-v2` for embeddings
- Hugging Face Inference API (defaults to `HuggingFaceH4/zephyr-7b-beta`) for generation
- FAISS in-memory vector store

## Getting Started

```bash
python -m venv .venv
.venv\Scripts\activate  # PowerShell on Windows
pip install -r requirements.txt
```

Create a `.env` (or use the Streamlit sidebar) with:

```
HF_TOKEN=hf_...
```

Tokens are free from https://huggingface.co/settings/tokens and are stored only in memory during a session.

Then launch:

```bash
streamlit run app.py
```

## Workflow Overview

1. **Upload PDFs** - Text is extracted per page with `pypdf`, cleaned, chunked with adjustable size/overlap, and embedded via Sentence Transformers before building FAISS.
2. **Ask Questions** - The Q&A tab retrieves the top-k chunks, constrains the Hugging Face model with that context, and shows the retrieved snippets. A baseline "whole corpus" mode is available for comparison.
3. **Summaries & MCQs** - Summaries use a structured prompt, while MCQs return JSON payloads rendered into readable quiz cards (with a raw output fallback if parsing fails).
4. **Experiment Logging** - Optional controls append notes and parameter settings to `docs/experiments.md`, which later feeds the project report.

## Repository Layout

```
app.py
eduweave/
  |- config.py
  |- pdf_utils.py
  |- text_processing.py
  |- vector_store.py
  |- rag.py
  |- generation.py
  |- experiment_tracker.py
docs/
  |- experiments.md
  |- report_draft.md
  |- video_script.md
data/
  |- samples/
  |- vector_store/
```

Use the `docs` folder as you iterate - log experiment findings, expand the project report, and refine your demo script.
