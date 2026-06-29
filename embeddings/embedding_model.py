import requests
from config import OLLAMA_EMBED_URL


def generate_embedding(text):
    payload = {
        "model": "nomic-embed-text:latest",
        "prompt": text
    }
    response = requests.post(
        OLLAMA_EMBED_URL,
        json=payload,
        proxies={"http": None, "https": None}
    )
    response.raise_for_status()
    return response.json()["embedding"]