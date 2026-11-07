"""
Live Evaluation & Telemetry Dashboard Page.

Visualizes:
1. Aggregate RAG quality KPIs (Faithfulness, Answer Relevance, Context Relevance).
2. Latency breakdown (Retrieval vs Generation vs Eval).
3. Detailed log table of all historical query evaluations with LLM judge reasoning.
"""

import json
import streamlit as st
import pandas as pd
from graph_rag.evaluation.eval_service import eval_service


def render_eval_page() -> None:
    """Render the RAG Evaluation & Telemetry Dashboard."""
    st.subheader("📊 Live Session Evaluation & RAG Telemetry")
    st.markdown(
        "Real-time quality scores computed by LLM-as-a-Judge alongside per-stage latency telemetry."
    )

    summary = eval_service.get_overall_summary()
    total_evals = summary.get("total_evaluations", 0)

    # ── Top KPIs ──
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1:
        st.metric(
            label="Queries Evaluated",
            value=f"{total_evals:,}",
            help="Total number of RAG queries evaluated in live sessions",
        )
    with kpi2:
        faith_val = summary.get("avg_faithfulness", 0.0)
        st.metric(
            label="Avg Grounding / Faithfulness",
            value=f"{int(faith_val * 100)}%",
            help="Measures whether claims in the answer are supported by retrieved context",
        )
    with kpi3:
        rel_val = summary.get("avg_answer_relevance", 0.0)
        st.metric(
            label="Avg Answer Relevance",
            value=f"{int(rel_val * 100)}%",
            help="Measures whether the answer addresses the user's specific prompt",
        )
    with kpi4:
        avg_total_ms = summary.get("avg_total_ms", 0)
        st.metric(
            label="Avg Total Latency",
            value=f"{avg_total_ms:,.0f} ms",
            help="Average end-to-end response time",
        )

    st.divider()

    # ── Latency Breakdown ──
    st.markdown("#### ⏱️ Latency Stage Breakdown (Average)")
    lat1, lat2, lat3 = st.columns(3)
    with lat1:
        st.metric("Vector & Graph Retrieval", f"{summary.get('avg_retrieval_ms', 0):,.0f} ms")
    with lat2:
        st.metric("LLM Answer Generation", f"{summary.get('avg_generation_ms', 0):,.0f} ms")
    with lat3:
        ctx_rel = summary.get("avg_context_relevance", 0.0)
        st.metric("Avg Context Relevance", f"{int(ctx_rel * 100)}%")

    st.divider()

    # ── Recent Evaluation Logs ──
    st.markdown("#### 📜 Live Query Logs & LLM Judge Audits")
    recent_logs = eval_service.get_recent_logs(limit=30)

    if recent_logs:
        for log in recent_logs:
            with st.expander(
                f"[{log.get('created_at', '')[:19]}] Q: {log.get('question', '')[:65]}... (Faith: {int(log.get('faithfulness', 0)*100)}% | Total: {log.get('total_ms', 0)}ms)",
                expanded=False,
            ):
                c_q, c_a = st.columns([1, 1])
                with c_q:
                    st.markdown("**User Question:**")
                    st.info(log.get("question"))
                    st.markdown("**Retrieved Context Excerpt:**")
                    st.text(log.get("context_chunks", "")[:400] + "...")

                with c_a:
                    st.markdown("**Generated Answer:**")
                    st.success(log.get("answer"))

                st.markdown("**Detailed Judge Reasoning:**")
                reasoning_str = log.get("eval_reasoning")
                if reasoning_str:
                    try:
                        reasoning_dict = json.loads(reasoning_str)
                        st.json(reasoning_dict)
                    except Exception:
                        st.text(reasoning_str)

                st.caption(
                    f"Session: {log.get('session_title', 'General')} | "
                    f"Retrieval: {log.get('retrieval_ms')}ms | "
                    f"Generation: {log.get('generation_ms')}ms | "
                    f"Evaluation: {log.get('eval_ms')}ms"
                )
    else:
        st.info("No queries have been evaluated yet. Ask questions in the Research Chat to generate live evaluation telemetry.")
