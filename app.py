"""Streamlit interface for EduWeave - AI Study Copilot for PDFs."""

from __future__ import annotations

import io
from typing import List

import streamlit as st
from dotenv import load_dotenv

from eduweave.config import AppConfig, ChunkConfig, RetrievalConfig
from eduweave.experiment_tracker import log_experiment
from eduweave.generation import generate_mcqs, generate_summary
from eduweave.pdf_utils import combine_texts, extract_text_from_pdf
from eduweave.rag import answer_question
from eduweave.text_processing import chunk_text_with_metadata
from eduweave.vector_store import VectorStoreManager

load_dotenv()

APP_CONFIG = AppConfig()


def init_state() -> None:
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None
    if "corpus_text" not in st.session_state:
        st.session_state.corpus_text = ""
    if "document_names" not in st.session_state:
        st.session_state.document_names = []


def ensure_corpus_ready() -> bool:
    vector_store: VectorStoreManager | None = st.session_state.get("vector_store")
    if not vector_store or not vector_store.is_ready():
        st.info("Upload and process at least one PDF in the first tab to continue.")
        return False
    return True


def sidebar() -> None:
    st.sidebar.header("Setup")
    st.sidebar.write(
        "1. Upload PDFs.\n"
        "2. Tune chunk sizes / overlaps.\n"
        "3. Explore RAG Q&A, summaries, and MCQs."
    )
    st.sidebar.markdown("---")
    st.sidebar.caption("EduWeave prioritizes workflow clarity over flashy UI.")


def upload_tab() -> None:
    st.subheader("1. Upload & Process PDFs")
    uploaded_files = st.file_uploader(
        "Upload one or more study PDFs",
        type=["pdf"],
        accept_multiple_files=True,
    )
    chunk_size = st.slider("Chunk size (characters)", 400, 1600, APP_CONFIG.chunk.chunk_size, step=100)
    chunk_overlap = st.slider("Chunk overlap (characters)", 0, 400, APP_CONFIG.chunk.chunk_overlap, step=25)
    record_logs = st.checkbox("Record this run in docs/experiments.md")
    note = st.text_area("Experiment note (optional)", placeholder="e.g., Larger chunk size felt more coherent.")

    if st.button("Process PDFs"):
        if not uploaded_files:
            st.warning("Please upload at least one PDF.")
            return

        pdf_texts: List[str] = []
        doc_names: List[str] = []

        with st.spinner("Extracting text from PDFs..."):
            for file in uploaded_files:
                data = file.read()
                text = extract_text_from_pdf(io.BytesIO(data), filename=file.name)
                if text:
                    pdf_texts.append(text)
                    doc_names.append(file.name)

        if not pdf_texts:
            st.error("No extractable text found in the uploaded PDFs.")
            return

        chunk_config = ChunkConfig(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=APP_CONFIG.chunk.separators,
        )

        all_chunks: List[str] = []
        all_meta = []
        for name, text in zip(doc_names, pdf_texts):
            chunks, metadata = chunk_text_with_metadata(text, chunk_config, source=name)
            all_chunks.extend(chunks)
            all_meta.extend(metadata)

        st.write(f"Prepared {len(all_chunks)} chunks from {len(pdf_texts)} documents.")

        with st.spinner("Creating embeddings and FAISS index..."):
            vector_store = VectorStoreManager(APP_CONFIG.embedding_model)
            vector_store.build(all_chunks, all_meta)
            vector_store.full_text = combine_texts(pdf_texts)

        st.session_state.vector_store = vector_store
        st.session_state.corpus_text = vector_store.full_text
        st.session_state.document_names = doc_names

        if record_logs:
            log_experiment(
                "Vector store build",
                {
                    "chunk_size": chunk_size,
                    "chunk_overlap": chunk_overlap,
                    "documents": len(doc_names),
                },
                note or "Processed new documents.",
            )

        st.success("Vector store ready! Explore the other tabs.")


def qa_tab() -> None:
    st.subheader("2. RAG Q&A")
    if not ensure_corpus_ready():
        return

    question = st.text_input("Ask a question about your uploaded material")
    top_k = st.slider("Retrieved chunks (top-k)", 2, 8, APP_CONFIG.retrieval.top_k)
    temperature = st.slider("Answer creativity (temperature)", 0.0, 1.0, APP_CONFIG.generation.answer_temperature, step=0.05)
    use_rag = st.radio("Answer mode", ["RAG retrieval", "Baseline (entire text)"], horizontal=True) == "RAG retrieval"

    if st.button("Get answer", key="qa"):
        if not question.strip():
            st.warning("Enter a question first.")
            return
        try:
            retrieval_cfg = RetrievalConfig(
                top_k=top_k,
                max_k=APP_CONFIG.retrieval.max_k,
                distance_cutoff=APP_CONFIG.retrieval.distance_cutoff,
            )
            response = answer_question(
                question,
                st.session_state.vector_store,
                retrieval_cfg,
                APP_CONFIG.generation,
                use_rag=use_rag,
                temperature=temperature,
            )
        except Exception as exc:
            st.error(f"Failed to get answer: {exc}")
            return

        st.markdown("**Answer:**")
        st.write(response.answer)
        if response.sources:
            with st.expander("Retrieved context"):
                for source in response.sources:
                    meta = source.get("metadata", {})
                    label = meta.get("source", "chunk")
                    st.markdown(f"**{label}** - distance {source.get('distance', 0):.4f}")
                    st.caption(source["text"])

        st.success(f"Answered using {'RAG' if response.mode == 'rag' else 'baseline'} mode.")

        note = st.text_area("Optional experiment note", key="qa_note")
        if st.button("Log this Q&A experiment"):
            log_experiment(
                "Q&A",
                {"mode": response.mode, "top_k": top_k, "temp": temperature},
                note or "No additional notes.",
            )
            st.info("Logged to docs/experiments.md")


def summary_tab() -> None:
    st.subheader("3. Summary Generator")
    if not ensure_corpus_ready():
        return
    temperature = st.slider("Summary temperature", 0.0, 0.8, APP_CONFIG.generation.summary_temperature, step=0.05)
    if st.button("Generate summary"):
        try:
            summary = generate_summary(st.session_state.corpus_text, APP_CONFIG.generation, temperature=temperature)
        except Exception as exc:
            st.error(f"Failed to create summary: {exc}")
            return
        st.markdown("### Summary")
        st.write(summary)
        note = st.text_area("Notes about this summary run", key="summary_note")
        if st.button("Log summary run"):
            log_experiment(
                "Summary generation",
                {"temp": temperature},
                note or "Summary generated.",
            )
            st.info("Logged to docs/experiments.md")


def mcq_tab() -> None:
    st.subheader("4. MCQ Practice")
    if not ensure_corpus_ready():
        return
    count = st.slider("Number of MCQs", 3, 15, 5)
    temperature = st.slider("MCQ temperature", 0.0, 1.0, APP_CONFIG.generation.mcq_temperature, step=0.05)
    if st.button("Generate MCQs"):
        try:
            questions, raw = generate_mcqs(
                st.session_state.corpus_text,
                count,
                APP_CONFIG.generation,
                temperature=temperature,
            )
        except Exception as exc:
            st.error(f"Failed to generate MCQs: {exc}")
            return

        if questions:
            for idx, mcq in enumerate(questions, start=1):
                st.markdown(f"**Q{idx}. {mcq.get('question', '').strip()}**")
                for option in mcq.get("options", []):
                    st.write(option)
                st.caption(f"Answer: {mcq.get('answer', '?')} - {mcq.get('explanation', '').strip()}")
        else:
            st.warning("Could not parse JSON response, showing raw output.")
            st.code(raw)

        note = st.text_area("Notes about MCQ quality", key="mcq_note")
        if st.button("Log MCQ run"):
            log_experiment(
                "MCQ generation",
                {"temp": temperature, "questions": count},
                note or "MCQs generated.",
            )
            st.info("Logged to docs/experiments.md")


def main() -> None:
    st.set_page_config(page_title="EduWeave - Study Copilot", layout="wide")
    st.title(APP_CONFIG.project_title)
    st.caption("Upload PDFs -> explore RAG Q&A, summaries, and MCQ practice.")

    init_state()
    sidebar()

    tab_upload, tab_qa, tab_summary, tab_mcq = st.tabs(
        ["Upload & Process", "Q&A", "Summaries", "MCQs"]
    )

    with tab_upload:
        upload_tab()
    with tab_qa:
        qa_tab()
    with tab_summary:
        summary_tab()
    with tab_mcq:
        mcq_tab()


if __name__ == "__main__":
    main()
