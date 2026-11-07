"""
Ollama Client module.

Wraps the Ollama API for:
- Text Generation (qwen2.5)
- Text Embeddings (nomic-embed-text)
- Health check and model availability
"""

import logging
from typing import Any, Dict, List, Optional
import httpx
import ollama

from graph_rag.config import settings

logger = logging.getLogger(__name__)


class OllamaClient:
    """Wrapper for local Ollama LLM and Embedding interactions."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        llm_model: Optional[str] = None,
        embed_model: Optional[str] = None,
    ) -> None:
        self.base_url = base_url or settings.OLLAMA_BASE_URL
        self.llm_model = llm_model or settings.OLLAMA_LLM_MODEL
        self.embed_model = embed_model or settings.OLLAMA_EMBED_MODEL
        self.client = ollama.Client(host=self.base_url, timeout=180.0)

    def is_available(self) -> bool:
        """Check if Ollama server is running and accessible."""
        try:
            with httpx.Client(timeout=3.0) as client:
                res = client.get(f"{self.base_url}/api/tags")
                return res.status_code == 200
        except Exception:
            return False

    def list_models(self) -> List[str]:
        """List all models currently available in local Ollama."""
        try:
            models_info = self.client.list()
            # ollama-python returns ListResponse object with 'models'
            return [m.model for m in models_info.models]
        except Exception as e:
            logger.warning(f"Failed to query Ollama models: {e}")
            return []

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.2,
    ) -> str:
        """Generate a text response using Ollama LLM with automatic model fallback."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        # Try configured model, fallback to any available qwen or local model
        target_model = self.llm_model
        try:
            response = self.client.chat(
                model=target_model,
                messages=messages,
                options={"temperature": temperature},
            )
            return response["message"]["content"].strip()
        except Exception as e:
            # Check if 404 not found and attempt fallback
            err_str = str(e).lower()
            if "not found" in err_str:
                available = self.list_models()
                fallback = next((m for m in available if "qwen" in m.lower() or "deepseek" in m.lower()), None)
                if fallback and fallback != target_model:
                    logger.info(f"Model '{target_model}' not found, falling back to installed '{fallback}'")
                    self.llm_model = fallback
                    response = self.client.chat(
                        model=fallback,
                        messages=messages,
                        options={"temperature": temperature},
                    )
                    return response["message"]["content"].strip()
            raise

    def get_embedding(self, text: str) -> List[float]:
        """Generate embedding vector for a single text chunk."""
        # Use nomic-embed-text
        res = self.client.embeddings(
            model=self.embed_model,
            prompt=text,
        )
        return res["embedding"]

    def get_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embedding vectors for a batch of text chunks."""
        if not texts:
            return []

        # Ollama's embed endpoint accepts the complete batch in one request.
        # This avoids paying model-loading and HTTP overhead once per chunk.
        response = self.client.embed(model=self.embed_model, input=texts)
        return [list(embedding) for embedding in response.embeddings]


ollama_client = OllamaClient()
