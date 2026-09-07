"""
FastAPI Entry Point
Registers all route groups and starts the API server.

Run with:
    uvicorn api.main:app --reload --port 8000

Docs available at:
    http://localhost:8000/docs        (Swagger UI)
    http://localhost:8000/redoc       (ReDoc)
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import document, graph, community, query

app = FastAPI(
    title="Requirement GraphRAG API",
    description=(
        "REST API for the GraphRAG system. "
        "Upload requirement documents, build knowledge graphs, "
        "detect communities, and query using GraphRAG."
    ),
    version="1.0.0"
)

# Allow all origins for development
# Restrict in production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# Register route groups
app.include_router(document.router,  prefix="/api/document",  tags=["Document"])
app.include_router(graph.router,     prefix="/api/graph",     tags=["Graph"])
app.include_router(community.router, prefix="/api/community", tags=["Community"])
app.include_router(query.router,     prefix="/api/query",     tags=["Query"])


@app.get("/")
def root():
    return {
        "message": "Requirement GraphRAG API is running",
        "docs":    "http://localhost:8000/docs",
        "version": "1.0.0"
    }


@app.get("/health")
def health():
    return {"status": "ok"}