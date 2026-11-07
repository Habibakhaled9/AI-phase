"""
Upload & Document Management Page.

Enforces:
1. Multi-format support: PDF, Word (docx), PowerPoint (pptx), Plain Text (txt/md).
2. Archive unpacking: ZIP, RAR, 7z, TAR.GZ.
3. Strict Deduplication:
   - 🚫 Duplicate file is NOT saved to disk (pdfs/).
   - 🚫 Backend pipeline is NOT applied.
   - ⚠️ Clear alert shows matching paper title and hash.
4. Genuinely new files are indexed, saved to pdfs/, and appended to metadata.csv.
5. Unified searchable registry table of all documents in the knowledge base.
"""

from pathlib import Path
import streamlit as st
import pandas as pd

from graph_rag.config import settings
from graph_rag.ingestion.archive_handler import archive_handler
from graph_rag.ingestion.file_handler import file_handler
from graph_rag.ingestion.metadata_manager import metadata_manager
from graph_rag.ingestion.pipeline import pipeline


def render_upload_page() -> None:
    """Render Document Knowledge Base & Upload Management interface."""
    user = st.session_state.get("user")
    user_id = user.id if user else None

    st.subheader("📁 Document Knowledge Base & Ingestion")
    st.markdown(
        "Upload new research papers to expand the knowledge base. "
        "The system parses, chunks, generates vector embeddings, and extracts graph entities."
    )

    # ── Upload Section ──
    with st.expander("📤 Upload New Documents (Single, Multiple, or Archive)", expanded=True):
        st.markdown(
            "**Supported Formats:** PDF (`.pdf`), Word (`.docx`), PowerPoint (`.pptx`), Text (`.txt`, `.md`), "
            "or Compressed Archives (`.zip`, `.rar`, `.7z`, `.tar.gz`)."
        )

        uploaded_files = st.file_uploader(
            "Choose files or archives to ingest",
            type=["pdf", "docx", "pptx", "txt", "md", "zip", "rar", "7z", "tar", "gz"],
            accept_multiple_files=True,
            help=f"Maximum allowed file size: {settings.MAX_UPLOAD_SIZE_MB}MB",
        )

        if uploaded_files:
            if st.button("🚀 Process & Ingest Files", type="primary"):
                progress_bar = st.progress(0)
                status_placeholder = st.empty()

                total_files = len(uploaded_files)
                for idx, uploaded_file in enumerate(uploaded_files, start=1):
                    filename = uploaded_file.name
                    file_bytes = uploaded_file.read()
                    size_mb = len(file_bytes) / (1024 * 1024)

                    if size_mb > settings.MAX_UPLOAD_SIZE_MB:
                        st.error(f"❌ File '{filename}' exceeds maximum limit of {settings.MAX_UPLOAD_SIZE_MB}MB.")
                        continue

                    status_placeholder.text(f"Processing ({idx}/{total_files}): {filename}...")

                    # Check if archive or single document
                    if archive_handler.is_archive(filename):
                        with st.spinner(f"Unpacking and scanning archive: {filename}..."):
                            archive_results = pipeline.ingest_uploaded_archive(
                                archive_bytes=file_bytes,
                                archive_name=filename,
                                user_id=user_id,
                            )
                        for res in archive_results:
                            _render_ingestion_result(res)
                    else:
                        with st.spinner(f"Verifying and ingesting: {filename}..."):
                            res = pipeline.ingest_uploaded_file(
                                file_bytes=file_bytes,
                                filename=filename,
                                user_id=user_id,
                            )
                        _render_ingestion_result(res)

                    progress_bar.progress(idx / total_files)

                status_placeholder.success("Processing complete!")
                st.rerun()

    st.divider()

    # ── Document Registry Table (from metadata.csv) ──
    st.subheader("📚 Registered Knowledge Base Documents")
    df = metadata_manager.load_metadata()

    if not df.empty:
        col_s, col_filter = st.columns([3, 1])
        with col_s:
            search_query = st.text_input("🔍 Search documents by title, author, or topic", placeholder="e.g. Diarization, BERT, Transformer...")
        with col_filter:
            source_filter = st.selectbox("Source Origin", options=["All", "Initial Papers (pdfs/)", "User Uploads"])

        # Filtering
        filtered_df = df.copy()
        if search_query:
            q = search_query.lower()
            mask = (
                filtered_df["title"].astype(str).str.lower().str.contains(q)
                | filtered_df["topic"].astype(str).str.lower().str.contains(q)
                | filtered_df["authors"].astype(str).str.lower().str.contains(q)
                | filtered_df["local_pdf"].astype(str).str.lower().str.contains(q)
            )
            filtered_df = filtered_df[mask]

        if source_filter == "Initial Papers (pdfs/)":
            filtered_df = filtered_df[filtered_df["source"] == "initial"]
        elif source_filter == "User Uploads":
            filtered_df = filtered_df[filtered_df["source"] == "upload"]

        st.caption(f"Showing **{len(filtered_df)}** of **{len(df)}** registered documents")

        # Format display columns
        display_cols = ["title", "topic", "authors", "source", "local_pdf", "ingested_at"]
        available_cols = [c for c in display_cols if c in filtered_df.columns]
        st.dataframe(
            filtered_df[available_cols],
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("No documents are currently registered in metadata.csv.")


def _render_ingestion_result(res) -> None:
    """Display clean result message for an ingestion attempt."""
    if res.is_duplicate:
        st.warning(
            f"🚫 **Duplicate Skipped**: `{res.filename}`  \n"
            f"Reason: {res.match_reason}  \n"
            f"*(File was not saved to disk and no backend compute was spent)*"
        )
    elif res.status == "completed":
        st.success(
            f"✅ **Successfully Ingested**: `{res.filename}`  \n"
            f"Title: **{res.title}** | Created **{res.num_chunks}** vector chunks in ChromaDB & extracted graph entities into Neo4j."
        )
    else:
        st.error(
            f"❌ **Failed to Ingest**: `{res.filename}`  \n"
            f"Error: {res.error_message}"
        )
