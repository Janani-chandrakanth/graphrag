import os
from dotenv import load_dotenv

load_dotenv()

# ─── Ollama ───────────────────────────────────────────────
OLLAMA_URL        = os.getenv("OLLAMA_URL", "http://52.206.209.141:8002")
OLLAMA_EMBED_URL  = f"{OLLAMA_URL}/api/embeddings"
OLLAMA_HOST       = OLLAMA_URL

# ─── Neo4j ────────────────────────────────────────────────
NEO4J_URI         = os.getenv("NEO4J_URI")
NEO4J_USERNAME    = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD    = os.getenv("NEO4J_PASSWORD")

# ─── ChromaDB ─────────────────────────────────────────────
CHROMA_PATH       = os.getenv("CHROMA_PATH", "./chroma_db")
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "requirements_chunks")