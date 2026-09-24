"""
Embedding generation via Ollama's /api/embeddings — confirmed
available on 10.10.160.51 (config.OLLAMA_URL): nomic-embed-text:latest
IS pulled there (per `ollama list` on that host), so the earlier
detour to a local sentence-transformers model wasn't actually
necessary — the original model/approach works fine once pointed at
the host that's actually reachable, with no vector-space change and
no new dependency.
"""

import json
import requests
from config import OLLAMA_EMBED_URL, EMBED_MODEL


def generate_embedding(text):
    payload = {
        "model": EMBED_MODEL,
        "prompt": text
    }
    response = requests.post(
        OLLAMA_EMBED_URL,
        json=payload,
        proxies={"http": None, "https": None}
    )
    response.raise_for_status()
    return response.json()["embedding"]