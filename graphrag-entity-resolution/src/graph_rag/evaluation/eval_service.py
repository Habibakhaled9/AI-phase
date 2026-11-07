"""
Evaluation Service module.

Executes live RAG evaluations, persists scores to SQLite evaluation_logs table,
and exposes aggregation functions for the Live Session Evaluation Dashboard.
"""

import json
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from graph_rag.config import settings
from graph_rag.database.db import get_db_connection
from graph_rag.evaluation.latency_tracker import LatencyMetrics, StageTimer
from graph_rag.evaluation.metrics import (
    evaluate_answer_relevance,
    evaluate_context_relevance,
    evaluate_faithfulness,
)

logger = logging.getLogger(__name__)


@dataclass
class EvaluationRecord:
    """Full evaluation assessment for a single query response."""
    faithfulness: float
    answer_relevance: float
    context_relevance: float
    faithfulness_reasoning: str
    answer_relevance_reasoning: str
    context_relevance_reasoning: str
    latency: LatencyMetrics


class EvaluationService:
    """Handles evaluation logging and dashboard metrics generation."""

    def evaluate_turn(
        self,
        question: str,
        answer: str,
        context: str,
        message_id: Optional[int] = None,
        session_id: Optional[int] = None,
        user_id: Optional[int] = None,
        latency: Optional[LatencyMetrics] = None,
    ) -> EvaluationRecord:
        """Run full evaluation suite on an answered turn and persist to database."""
        lat = latency or LatencyMetrics()

        with StageTimer() as eval_timer:
            faith_score, faith_reason = evaluate_faithfulness(question, answer, context)
            relevance_score, relevance_reason = evaluate_answer_relevance(question, answer)
            ctx_score, ctx_reason = evaluate_context_relevance(question, context)

        lat.eval_ms = eval_timer.elapsed_ms
        lat.total_ms = lat.retrieval_ms + lat.generation_ms + lat.eval_ms

        eval_record = EvaluationRecord(
            faithfulness=faith_score,
            answer_relevance=relevance_score,
            context_relevance=ctx_score,
            faithfulness_reasoning=faith_reason,
            answer_relevance_reasoning=relevance_reason,
            context_relevance_reasoning=ctx_reason,
            latency=lat,
        )

        # Log into SQLite
        try:
            with get_db_connection() as conn:
                conn.execute(
                    """
                    INSERT INTO evaluation_logs
                    (message_id, session_id, user_id, question, answer, context_chunks,
                     faithfulness, answer_relevance, context_relevance,
                     retrieval_ms, generation_ms, eval_ms, total_ms, eval_reasoning)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        message_id,
                        session_id,
                        user_id,
                        question,
                        answer,
                        context[:1500],
                        faith_score,
                        relevance_score,
                        ctx_score,
                        lat.retrieval_ms,
                        lat.generation_ms,
                        lat.eval_ms,
                        lat.total_ms,
                        json.dumps({
                            "faithfulness": faith_reason,
                            "answer_relevance": relevance_reason,
                            "context_relevance": ctx_reason,
                        }),
                    ),
                )
        except Exception as e:
            logger.warning(f"Could not persist evaluation log to database: {e}")

        logger.info(
            "Evaluation scores: faithfulness=%.3f answer_relevance=%.3f "
            "context_relevance=%.3f retrieval_ms=%d generation_ms=%d "
            "eval_ms=%d total_ms=%d",
            faith_score,
            relevance_score,
            ctx_score,
            lat.retrieval_ms,
            lat.generation_ms,
            lat.eval_ms,
            lat.total_ms,
        )
        return eval_record

    def get_session_eval_logs(self, session_id: int) -> List[Dict[str, Any]]:
        """Retrieve evaluation logs for a given chat session."""
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT id, message_id, question, answer, faithfulness,
                       answer_relevance, context_relevance, retrieval_ms,
                       generation_ms, eval_ms, total_ms, eval_reasoning, created_at
                FROM evaluation_logs
                WHERE session_id = ?
                ORDER BY created_at ASC
                """,
                (session_id,),
            )
            return [dict(row) for row in cursor.fetchall()]

    def get_recent_logs(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Retrieve most recent evaluation logs across all sessions for live view."""
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT e.*, u.username, s.title AS session_title
                FROM evaluation_logs e
                LEFT JOIN users u ON e.user_id = u.id
                LEFT JOIN chat_sessions s ON e.session_id = s.id
                ORDER BY e.created_at DESC
                LIMIT ?
                """,
                (limit,),
            )
            return [dict(row) for row in cursor.fetchall()]

    def get_overall_summary(self) -> Dict[str, Any]:
        """Aggregate summary metrics across all sessions."""
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT
                    COUNT(*) as total_evaluations,
                    ROUND(AVG(faithfulness), 2) as avg_faithfulness,
                    ROUND(AVG(answer_relevance), 2) as avg_answer_relevance,
                    ROUND(AVG(context_relevance), 2) as avg_context_relevance,
                    ROUND(AVG(retrieval_ms), 0) as avg_retrieval_ms,
                    ROUND(AVG(generation_ms), 0) as avg_generation_ms,
                    ROUND(AVG(total_ms), 0) as avg_total_ms
                FROM evaluation_logs
                """
            )
            row = cursor.fetchone()
            if row and row["total_evaluations"]:
                return dict(row)
            return {
                "total_evaluations": 0,
                "avg_faithfulness": 0.0,
                "avg_answer_relevance": 0.0,
                "avg_context_relevance": 0.0,
                "avg_retrieval_ms": 0,
                "avg_generation_ms": 0,
                "avg_total_ms": 0,
            }


eval_service = EvaluationService()
