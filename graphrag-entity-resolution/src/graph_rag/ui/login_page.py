"""
Login and Registration UI Page.
"""

import streamlit as st
from graph_rag.auth.auth_service import auth_service
from graph_rag.chat.chat_service import chat_service


def render_login_page() -> None:
    """Render the authentication portal."""
    st.markdown("<br><br>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown(
            """
            <div style='text-align: center; margin-bottom: 25px;'>
                <h1 style='margin-bottom: 5px;'>🔬 Graph RAG Studio</h1>
                <p style='color: #94a3b8;'>Scientific Research Assistant with Hybrid Graph + Vector RAG</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        tab_login, tab_register = st.tabs(["🔑 Sign In", "📝 Create Account"])

        # ── Sign In Tab ──
        with tab_login:
            with st.form("login_form"):
                username = st.text_input("Username", placeholder="e.g. researcher1")
                password = st.text_input("Password", type="password", placeholder="••••••••")
                submitted = st.form_submit_button("Sign In", use_container_width=True, type="primary")

                if submitted:
                    if not username or not password:
                        st.error("Please enter both username and password.")
                    else:
                        success, msg, token, user = auth_service.login_user(username, password)
                        if success and user:
                            st.session_state["user"] = user
                            st.session_state["token"] = token
                            # Start each login in a fresh conversation while preserving history.
                            new_sess = chat_service.create_session(user_id=user.id)
                            st.session_state["active_session_id"] = new_sess.id
                            st.success(msg)
                            st.rerun()
                        else:
                            st.error(msg)

        # ── Create Account Tab ──
        with tab_register:
            with st.form("register_form"):
                reg_username = st.text_input("Username", placeholder="Choose a username")
                reg_email = st.text_input("Email", placeholder="you@university.edu")
                reg_password = st.text_input("Password", type="password", placeholder="At least 6 characters")
                reg_submitted = st.form_submit_button("Register", use_container_width=True)

                if reg_submitted:
                    if not reg_username or not reg_email or not reg_password:
                        st.error("All fields are required.")
                    else:
                        success, msg, user = auth_service.register_user(
                            username=reg_username,
                            email=reg_email,
                            password=reg_password,
                        )
                        if success:
                            st.success("Account created successfully! Please switch to the Sign In tab to log in.")
                        else:
                            st.error(msg)
