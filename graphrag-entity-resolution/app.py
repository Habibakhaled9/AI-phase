"""
Main Streamlit Application Entry Point for Graph RAG Studio.

Usage:
    uv run streamlit run app.py
"""

import sys
from pathlib import Path
import streamlit as st

# Ensure project src directory is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
PROJECT_SRC = PROJECT_ROOT / "src"
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

from graph_rag.logging_config import configure_logging
from graph_rag.database.db import init_db
from graph_rag.ui.chat_page import render_chat_page
from graph_rag.ui.components import apply_theme, render_sidebar
from graph_rag.ui.eval_page import render_eval_page
from graph_rag.ui.login_page import render_login_page
from graph_rag.ui.upload_page import render_upload_page

# Set page configuration
st.set_page_config(
    page_title="Graph RAG Studio",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)


def main() -> None:
    configure_logging()
    # Ensure database tables exist on startup
    init_db()

    # Apply global styles
    apply_theme()

    # Authentication Gate
    user = st.session_state.get("user")
    if not user:
        render_login_page()
        return

    # Render global sidebar navigation
    render_sidebar()

    # Page Routing
    current_page = st.session_state.get("page", "chat")
    if current_page == "chat":
        render_chat_page()
    elif current_page == "upload":
        render_upload_page()
    elif current_page == "eval":
        render_eval_page()
    else:
        render_chat_page()


if __name__ == "__main__":
    main()
