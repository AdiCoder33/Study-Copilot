# Experiment Log

Use this living document to capture what you tried and what you learned.

Each entry added through the Streamlit UI (or manually) should follow:

```
### YYYY-MM-DD HH:MM - Experiment Title

**Parameters:** chunk_size=800, chunk_overlap=150, top_k=5

**Observation:** Brief qualitative summary about answer quality, hallucinations, MCQ diversity, etc.

---
```

Key things to vary:
- Chunk size / overlap (400-1200, 0-200)
- Retrieval `top_k` (3/5/8)
- Generation temperatures (0.2 vs 0.7)
- RAG vs baseline mode

This log feeds directly into the final report's "Results / Observations" section.
