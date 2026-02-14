"""
Ollama API Client — lightweight, dependency-free (just requests + numpy).

Wraps the Ollama REST API for embeddings and chat generation.
No LangChain, no LlamaIndex, just raw HTTP.
"""
import numpy as np
import requests
from typing import Dict, List, Optional

OLLAMA_BASE = "http://localhost:11434"

# Defaults
EMBED_MODEL = "nomic-embed-text"
CHAT_MODEL = "llama3"


def get_embedding(text: str, model: str = EMBED_MODEL) -> np.ndarray:
    """
    Get a normalized embedding vector from Ollama.

    Args:
        text: The text to embed.
        model: Ollama embedding model name.

    Returns:
        Normalized float32 numpy array (768-dim for nomic-embed-text).
    """
    resp = requests.post(
        f"{OLLAMA_BASE}/api/embeddings",
        json={"model": model, "prompt": text},
        timeout=30,
    )
    resp.raise_for_status()

    vec = np.array(resp.json()["embedding"], dtype=np.float32)

    # Normalize for cosine distance (HNSW uses 1 - dot)
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec /= norm

    return vec


def chat(
    messages: List[Dict[str, str]],
    model: str = CHAT_MODEL,
    temperature: float = 0.7,
) -> str:
    """
    Send a conversation to Ollama and return the assistant's response.

    Args:
        messages: List of {"role": "system"|"user"|"assistant", "content": "..."}.
        model: Ollama chat model name.
        temperature: Sampling temperature.

    Returns:
        The assistant's response text.
    """
    resp = requests.post(
        f"{OLLAMA_BASE}/api/chat",
        json={
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature},
        },
        timeout=120,
    )
    resp.raise_for_status()

    return resp.json()["message"]["content"]


def is_available() -> bool:
    """Check if the Ollama server is running."""
    try:
        resp = requests.get(f"{OLLAMA_BASE}/api/tags", timeout=5)
        return resp.status_code == 200
    except requests.ConnectionError:
        return False


def list_models() -> List[str]:
    """List all locally available Ollama models."""
    try:
        resp = requests.get(f"{OLLAMA_BASE}/api/tags", timeout=5)
        resp.raise_for_status()
        return [m["name"] for m in resp.json().get("models", [])]
    except (requests.ConnectionError, requests.HTTPError):
        return []
