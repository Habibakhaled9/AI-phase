"""
Query Engine module.

Coordinates:
1. Hybrid Retrieval (Vector + Graph)
2. Prompt Context Assembly
3. Local LLM (qwen2.5) Generation
4. Stage Latency Instrumentation
5. Real-Time RAG Evaluation
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from graph_rag.config import settings
from graph_rag.evaluation.eval_service import EvaluationRecord, eval_service
from graph_rag.evaluation.latency_tracker import LatencyMetrics, StageTimer
from graph_rag.llm.ollama_client import ollama_client
from graph_rag.retriever.hybrid_retriever import hybrid_retriever

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are an expert scientific AI research assistant with access to a Graph RAG knowledge base.
Your goal is to answer the user's question accurately, faithfully, and insightfully based on the retrieved research context.

Guidelines:
1. Ground your answer in the provided context excerpts and knowledge graph relations.
2. If the context does not contain enough information to answer definitively, clearly state what is known from the papers and what is missing.
3. Cite the relevant paper titles and pages when making factual claims.
4. Keep explanations clear, well-structured, and scientific.
"""

USER_PROMPT_TEMPLATE = """Research Context:
{context}

User Question:
{question}

Answer:"""


@dataclass
class QueryEngineResponse:
    """Consolidated response from the Graph RAG query engine."""
    question: str
    answer: str
    sources: List[str] = field(default_factory=list)
    context_text: str = ""
    latency: LatencyMetrics = field(default_factory=LatencyMetrics)
    evaluation: Optional[EvaluationRecord] = None


class QueryEngine:
    """Main interface for querying the Graph RAG system."""

    def __init__(self) -> None:
        self.retriever = hybrid_retriever
        self.llm = ollama_client
        self.eval = eval_service

    def query(
        self,
        question: str,
        session_id: Optional[int] = None,
        user_id: Optional[int] = None,
        message_id: Optional[int] = None,
        run_evaluation: bool = True,
    ) -> QueryEngineResponse:
        """Process user question through full hybrid RAG pipeline."""
        latency = LatencyMetrics()
        logger.info("Query started: session_id=%s user_id=%s question=%r", session_id, user_id, question[:2000])

        # Stage 1: Hybrid Retrieval
        with StageTimer() as ret_timer:
            retrieval_res = self.retriever.retrieve(query=question, top_k=5)
        latency.retrieval_ms = ret_timer.elapsed_ms

        context_text = retrieval_res.assembled_text
        sources = retrieval_res.sources
        logger.info(
            "Retrieval completed: chunks=%d graph_facts=%d sources=%d latency_ms=%d sources=%r",
            len(retrieval_res.chunks),
            len(retrieval_res.graph_facts),
            len(sources),
            latency.retrieval_ms,
            sources,
        )

        # Stage 2: LLM Generation
        prompt = USER_PROMPT_TEMPLATE.format(context=context_text, question=question)

        with StageTimer() as gen_timer:
            try:
                answer = self.llm.generate(
                    prompt=prompt,
                    system_prompt=SYSTEM_PROMPT,
                    temperature=0.2,
                )
            except Exception as e:
                logger.error(f"LLM Generation failed: {e}")
                answer = f"I encountered an error generating an answer: {e}. Please ensure Ollama is running."
        latency.generation_ms = gen_timer.elapsed_ms
        logger.info(
            "LLM response completed: latency_ms=%d answer=%r",
            latency.generation_ms,
            answer[:4000],
        )

        # Stage 3: Live Evaluation (if enabled)
        eval_record = None
        if run_evaluation and settings.RAG_EVAL_ENABLED:
            try:
                eval_record = self.eval.evaluate_turn(
                    question=question,
                    answer=answer,
                    context=context_text,
                    message_id=message_id,
                    session_id=session_id,
                    user_id=user_id,
                    latency=latency,
                )
            except Exception as e:
                logger.warning(f"Evaluation failed: {e}")

        latency.total_ms = latency.retrieval_ms + latency.generation_ms + (eval_record.latency.eval_ms if eval_record else 0)
        if eval_record:
            logger.info(
                "Evaluation completed: faithfulness=%.3f answer_relevance=%.3f "
                "context_relevance=%.3f eval_ms=%d total_ms=%d",
                eval_record.faithfulness,
                eval_record.answer_relevance,
                eval_record.context_relevance,
                eval_record.latency.eval_ms,
                latency.total_ms,
            )
        else:
            logger.info("Evaluation skipped: enabled=%s requested=%s", settings.RAG_EVAL_ENABLED, run_evaluation)

        return QueryEngineResponse(
            question=question,
            answer=answer,
            sources=sources,
            context_text=context_text,
            latency=latency,
            evaluation=eval_record,
        )


query_engine = QueryEngine()
