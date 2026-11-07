"""
Chat Interface UI Page.

Features:
- Multi-turn conversation display with user/assistant bubbles
- Expandable paper citations & page numbers
- Inline RAG Quality Badges (Faithfulness, Relevance, Latency)
- Live Session Evaluation drawer
"""

import streamlit as st
from graph_rag.chat.chat_service import chat_service
from graph_rag.retriever.query_engine import query_engine
from graph_rag.evaluation.eval_service import eval_service


def render_chat_page() -> None:
    """Render the main chat interface."""
    user = st.session_state.get("user")
    active_session_id = st.session_state.get("active_session_id")

    if not active_session_id:
        sessions = chat_service.list_user_sessions(user_id=user.id)
        if sessions:
            active_session_id = sessions[0].id
            st.session_state["active_session_id"] = active_session_id
        else:
            new_sess = chat_service.create_session(user_id=user.id)
            active_session_id = new_sess.id
            st.session_state["active_session_id"] = active_session_id

    current_session = chat_service.get_session(active_session_id)
    session_title = current_session.title if current_session else "Research Discussion"

    # Header
    col_t, col_actions = st.columns([4, 1])
    with col_t:
        st.subheader(f"💬 {session_title}")
    with col_actions:
        if st.button("🗑️ Clear Chat", help="Delete this session"):
            chat_service.delete_session(active_session_id)
            st.session_state["active_session_id"] = None
            st.rerun()

    # Load and display message history
    messages = chat_service.get_session_messages(active_session_id)

    # Load session evaluation logs to attach badges to assistant messages
    eval_logs = {
        log["message_id"]: log
        for log in eval_service.get_session_eval_logs(active_session_id)
        if log.get("message_id")
    }

    if not messages:
        st.info("💡 **Tip**: Ask questions about NLP, Speech Diarization, Computer Vision, or upload your own papers in the Document Knowledge Base tab!")

    for msg in messages:
        with st.chat_message(msg.role):
            st.markdown(msg.content)

            # If assistant message with sources, render sources
            if msg.role == "assistant" and msg.sources:
                with st.expander(f"📚 View {len(msg.sources)} Source Citations", expanded=False):
                    for s in msg.sources:
                        st.markdown(f"- {s}")

            # If evaluation log exists for this message, show badges
            ev = eval_logs.get(msg.id)
            if ev:
                c1, c2, c3, c4 = st.columns([1.2, 1.2, 1.2, 3])
                with c1:
                    faith = ev.get("faithfulness", 0.0)
                    st.markdown(f"<span class='badge-faith'>✅ Grounding: {int(faith * 100)}%</span>", unsafe_allow_html=True)
                with c2:
                    rel = ev.get("answer_relevance", 0.0)
                    st.markdown(f"<span class='badge-relevance'>🎯 Relevance: {int(rel * 100)}%</span>", unsafe_allow_html=True)
                with c3:
                    total_ms = ev.get("total_ms", 0)
                    st.markdown(f"<span class='badge-latency'>⏱️ {total_ms}ms</span>", unsafe_allow_html=True)

    # Chat Input Box
    user_input = st.chat_input("Ask a question about the research papers...")
    if user_input:
        # 1. Save and display user message
        user_msg = chat_service.add_message(
            session_id=active_session_id,
            user_id=user.id,
            role="user",
            content=user_input,
        )

        with st.chat_message("user"):
            st.markdown(user_input)

        # 2. Query Hybrid Engine with progress spinner
        with st.chat_message("assistant"):
            with st.spinner("Searching knowledge graph & generating scientific answer..."):
                resp = query_engine.query(
                    question=user_input,
                    session_id=active_session_id,
                    user_id=user.id,
                    run_evaluation=True,
                )

                st.markdown(resp.answer)

                if resp.sources:
                    with st.expander(f"📚 View {len(resp.sources)} Source Citations", expanded=False):
                        for s in resp.sources:
                            st.markdown(f"- {s}")

                # Inline evaluation scores
                if resp.evaluation:
                    c1, c2, c3, c4 = st.columns([1.2, 1.2, 1.2, 3])
                    with c1:
                        st.markdown(f"<span class='badge-faith'>✅ Grounding: {int(resp.evaluation.faithfulness * 100)}%</span>", unsafe_allow_html=True)
                    with c2:
                        st.markdown(f"<span class='badge-relevance'>🎯 Relevance: {int(resp.evaluation.answer_relevance * 100)}%</span>", unsafe_allow_html=True)
                    with c3:
                        st.markdown(f"<span class='badge-latency'>⏱️ {resp.latency.total_ms}ms</span>", unsafe_allow_html=True)

                # 3. Save assistant response into database
                chat_service.add_message(
                    session_id=active_session_id,
                    user_id=user.id,
                    role="assistant",
                    content=resp.answer,
                    sources=resp.sources,
                )

        st.rerun()

    # Expandable Live Session Logs Drawer
    with st.expander("📋 Live Session Logs & Telemetry", expanded=False):
        recent_evals = eval_service.get_session_eval_logs(active_session_id)
        if recent_evals:
            for log in reversed(recent_evals[-5:]):
                st.markdown(
                    f"**Q:** {log.get('question')}  \n"
                    f"⏱️ Retrieval: `{log.get('retrieval_ms', 0)}ms` | Generation: `{log.get('generation_ms', 0)}ms` | Eval: `{log.get('eval_ms', 0)}ms`  \n"
                    f"⭐ Faithfulness: `{log.get('faithfulness', 0.0)}` | Relevance: `{log.get('answer_relevance', 0.0)}`"
                )
                st.divider()
        else:
            st.caption("No logs recorded yet for this session.")
