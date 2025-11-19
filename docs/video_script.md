# EduWeave Demo - Draft Video Script (3-5 min)

1. **Intro (20-30s)**
   - Quick self-introduction and context of the internship challenge.
   - Frame the problem: students drown in PDF notes and need a smarter study workflow.

2. **Solution Overview (30s)**
   - Show the Streamlit interface layout.
   - Mention FAISS + Sentence Transformers + locally-downloaded Gemma 2B IT as the core stack.

3. **Feature Walkthrough (~2 min)**
   - **Upload tab:** highlight PDF ingestion, chunk controls, experiment logging toggle.
   - **Q&A tab:** compare baseline vs. RAG answers, point out the "Retrieved context" expander.
   - **Summary tab:** run a summary and show the structured output.
   - **MCQ tab:** generate a short set, show how parsed questions look (and mention raw fallback if parsing fails).

4. **Experiments & Learnings (~1 min)**
   - Reference the experiment log (chunk sizes, overlaps, top-k, temperatures).
   - Share notable takeaways (e.g., higher temp improved MCQ diversity but needed careful review).

5. **Closing (20s)**
   - Summarize learnings, mention future extensions (persistent stores, better UI, analytics).
   - Thank reviewers and invite them to read the report / explore the repo.
