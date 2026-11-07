"""
UI Components and Theme Styling module for Streamlit.

Provides:
- Modern CSS tokens (dark mode accents, glassmorphic cards, score badges)
- Global Sidebar navigation & session switcher
- Metric badges and source citation accordions
"""

import streamlit as st
from graph_rag.chat.chat_service import chat_service
from graph_rag.embeddings.vector_store import vector_store
from graph_rag.graph.graph_builder import graph_builder
from graph_rag.ingestion.metadata_manager import metadata_manager

CUSTOM_CSS = """
<style>
/* Modern typography and theme styling */
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
}

/* Glassmorphism cards */
.glass-card {
    background: rgba(255, 255, 255, 0.05);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 12px;
    padding: 18px;
    margin-bottom: 14px;
    backdrop-filter: blur(10px);
}

/* Metric chip pills */
.badge-faith {
    background: #0d3b24;
    color: #4ade80;
    padding: 3px 9px;
    border-radius: 12px;
    font-size: 0.78rem;
    font-weight: 600;
    display: inline-block;
    border: 1px solid #166534;
}

.badge-relevance {
    background: #172554;
    color: #60a5fa;
    padding: 3px 9px;
    border-radius: 12px;
    font-size: 0.78rem;
    font-weight: 600;
    display: inline-block;
    border: 1px solid #1e40af;
}

.badge-latency {
    background: #31135e;
    color: #c084fc;
    padding: 3px 9px;
    border-radius: 12px;
    font-size: 0.78rem;
    font-weight: 600;
    display: inline-block;
    border: 1px solid #581c87;
}

.badge-source-initial {
    background: #1e293b;
    color: #94a3b8;
    padding: 2px 8px;
    border-radius: 8px;
    font-size: 0.75rem;
    font-weight: 500;
}

.badge-source-upload {
    background: #3730a3;
    color: #c7d2fe;
    padding: 2px 8px;
    border-radius: 8px;
    font-size: 0.75rem;
    font-weight: 500;
}

/* Source citation box */
.citation-box {
    background: rgba(0, 0, 0, 0.25);
    border-left: 3px solid #3b82f6;
    padding: 10px 14px;
    margin: 8px 0;
    border-radius: 4px;
    font-size: 0.88rem;
}
</style>
"""


def apply_theme() -> None:
    """Inject custom CSS design tokens."""
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def render_sidebar() -> None:
    """Render unified navigation and session sidebar."""
    with st.sidebar:
        user = st.session_state.get("user")
        if not user:
            return

        st.markdown(f"### 🧬 **Graph RAG Studio**")
        st.caption(f"👤 Logged in as **{user.username}**")

        st.divider()

        # Navigation menu
        pages = {
            "chat": "💬 Research Chat",
            "upload": "📁 Document Knowledge Base",
            "eval": "📊 Live Evaluation & Logs",
        }
        current_page = st.session_state.get("page", "chat")

        nav_choice = st.radio(
            "Navigation",
            options=list(pages.keys()),
            format_func=lambda k: pages[k],
            index=list(pages.keys()).index(current_page) if current_page in pages else 0,
            label_visibility="collapsed",
        )
        if nav_choice != current_page:
            st.session_state["page"] = nav_choice
            st.rerun()

        st.divider()

        # Chat Sessions management (visible on chat page)
        if nav_choice == "chat":
            col_new, col_trash = st.columns([4, 1])
            with col_new:
                if st.button("➕ New Chat", use_container_width=True, type="primary"):
                    new_sess = chat_service.create_session(user_id=user.id)
                    st.session_state["active_session_id"] = new_sess.id
                    st.rerun()

            sessions = chat_service.list_user_sessions(user_id=user.id)
            if sessions:
                st.markdown("**Your Conversations**")
                active_id = st.session_state.get("active_session_id")
                if not active_id and sessions:
                    active_id = sessions[0].id
                    st.session_state["active_session_id"] = active_id

                for sess in sessions[:15]:
                    is_active = sess.id == active_id
                    btn_label = f"{'▶ ' if is_active else ''}{sess.title}"
                    if st.button(
                        btn_label,
                        key=f"sess_btn_{sess.id}",
                        use_container_width=True,
                    ):
                        st.session_state["active_session_id"] = sess.id
                        st.rerun()

            st.divider()

        # System Knowledge Base Status
        doc_count = metadata_manager.get_document_count()
        chunk_count = vector_store.get_total_chunks()
        neo4j_stats = graph_builder.get_graph_stats()

        st.markdown("**Knowledge Base Health**")
        st.markdown(f"📄 **{doc_count}** Indexed Papers")
        st.markdown(f"🧩 **{chunk_count}** Vector Chunks")
        if neo4j_stats.get("status") == "online":
            st.markdown(f"🕸️ **{neo4j_stats.get('nodes', 0)}** Graph Entities")
        else:
            st.caption("🕸️ Neo4j Graph (Offline)")

        st.divider()

        if st.button("🚪 Log Out", use_container_width=True):
            st.session_state.clear()
            st.rerun()
