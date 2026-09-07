import os
from dotenv import load_dotenv

load_dotenv()

# ───────────────────────────────────────────────────────────────
# Ollama
# ───────────────────────────────────────────────────────────────

# Main Ollama server
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://52.206.209.141:8002")
OLLAMA_HOST = OLLAMA_URL

# Embedding model
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text:latest")
OLLAMA_EMBED_URL = f"{OLLAMA_URL}/api/embeddings"

# All models are now served from the same Ollama instance
OLLAMA_CODER_URL = os.getenv("OLLAMA_CODER_URL", OLLAMA_URL)

# ───────────────────────────────────────────────────────────────
# Entity Extraction Model (graph/entity_extractor.py)
# ───────────────────────────────────────────────────────────────
# Uses qwen2.5-coder:7b for consistent snake_case entity IDs.

EXTRACTION_MODEL = os.getenv(
    "EXTRACTION_MODEL",
    "qwen2.5-coder:7b"
)

EXTRACTION_MODEL_URL = os.getenv(
    "EXTRACTION_MODEL_URL",
    OLLAMA_URL
)

# ───────────────────────────────────────────────────────────────
# Cross-Requirement Linker (graph/cross_reference_linker.py)
# ───────────────────────────────────────────────────────────────

CROSS_REF_MODEL = os.getenv(
    "CROSS_REF_MODEL",
    "gemma4:12b"
)

CROSS_REF_MODEL_URL = os.getenv(
    "CROSS_REF_MODEL_URL",
    OLLAMA_URL
)

# ───────────────────────────────────────────────────────────────
# Workflow / Sequence Extractor (graph/sequence_extractor.py)
# ───────────────────────────────────────────────────────────────
# qwen3:latest supports tools + thinking mode, making it well-suited
# for the reasoning-heavy workflow sequence extraction task.

WORKFLOW_MODEL = os.getenv(
    "WORKFLOW_MODEL",
    "qwen3:latest"
)

WORKFLOW_MODEL_URL = os.getenv(
    "WORKFLOW_MODEL_URL",
    OLLAMA_URL
)

# ───────────────────────────────────────────────────────────────
# Neo4j
# ───────────────────────────────────────────────────────────────

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
# Optional. The raw neo4j driver (graph/neo4j_manager.py's d.session()
# with no args) asks the server for its default database via routing
# and just works. langchain-neo4j's Neo4jGraph does NOT do that -- it
# defaults to the LITERAL string "neo4j" client-side, which breaks
# with Neo.ClientError.Database.DatabaseNotFound on any instance whose
# actual database isn't named exactly that (check Aura console -> your
# instance -> Connection details for the real name). Leave unset if
# graph/langchain_qa.py's Ask the Graph works fine as-is.
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE")

# ───────────────────────────────────────────────────────────────
# ChromaDB
# ───────────────────────────────────────────────────────────────

CHROMA_PATH = os.getenv("CHROMA_PATH", "./chroma_db")
CHROMA_COLLECTION = os.getenv(
    "CHROMA_COLLECTION",
    "requirements_chunks"
)

# ───────────────────────────────────────────────────────────────
# Vision Model (parser/vision_extractor.py)
# ───────────────────────────────────────────────────────────────
# Qwen2.5-VL is used for extracting information from diagrams/images.

VISION_MODEL = os.getenv(
    "VISION_MODEL",
    "qwen2.5vl:latest"
)

VISION_MODEL_URL = os.getenv(
    "VISION_MODEL_URL",
    OLLAMA_URL
)