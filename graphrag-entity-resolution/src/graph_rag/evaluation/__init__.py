"""RAG evaluation package."""

from graph_rag.evaluation.eval_service import EvaluationRecord, EvaluationService, eval_service
from graph_rag.evaluation.latency_tracker import LatencyMetrics, StageTimer
from graph_rag.evaluation.metrics import (
    evaluate_answer_relevance,
    evaluate_context_relevance,
    evaluate_faithfulness,
)

__all__ = [
    "EvaluationRecord",
    "EvaluationService",
    "eval_service",
    "LatencyMetrics",
    "StageTimer",
    "evaluate_faithfulness",
    "evaluate_answer_relevance",
    "evaluate_context_relevance",
]
