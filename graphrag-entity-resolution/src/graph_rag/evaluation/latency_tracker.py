"""
Latency Tracker module.

Times per-stage pipeline execution:
- retrieval_ms: vector store & graph query time
- generation_ms: LLM inference time
- eval_ms: evaluation metrics computation time
- total_ms: total end-to-end turnaround time
"""

import time
from dataclasses import dataclass
from typing import Optional


@dataclass
class LatencyMetrics:
    """Breakdown of processing latency across stages."""
    retrieval_ms: int = 0
    generation_ms: int = 0
    eval_ms: int = 0
    total_ms: int = 0


class StageTimer:
    """Context manager for timing individual operations."""

    def __init__(self) -> None:
        self.start_time: Optional[float] = None
        self.elapsed_ms: int = 0

    def __enter__(self) -> "StageTimer":
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self.start_time is not None:
            self.elapsed_ms = int((time.perf_counter() - self.start_time) * 1000)
