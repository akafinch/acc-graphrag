"""
Default configuration for the GraphRAG system.
"""

import os
from pathlib import Path

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EMBEDDINGS_DIR = DATA_DIR / "embeddings"

# Ollama configuration
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
LLM_MODEL = os.getenv("LLM_MODEL", "llama3")
LLM_CONTEXT_WINDOW = 8192
SYSTEM_PROMPT = """You are an AI assistant for a cloud architecture team. 
Provide clear, accurate information about partner technologies, integration approaches,
and best practices based on the documentation provided. If you're unsure, be transparent
about limitations."""

# Neo4j configuration
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "graphrag")

# API configuration
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))
API_WORKERS = int(os.getenv("API_WORKERS", "4"))

# RAG configuration
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50
TOP_K_RETRIEVAL = 5

# Graph schema
NODE_TYPES = {
    "Partner": {
        "properties": ["name", "description", "website"]
    },
    "Document": {
        "properties": ["title", "source", "partner", "date_published"]
    },
    "Section": {
        "properties": ["title", "content", "embedding"]
    },
    "Concept": {
        "properties": ["name", "description", "embedding"]
    },
    "Product": {
        "properties": ["name", "version", "description", "embedding"]
    },
    "Service": {
        "properties": ["name", "category", "description", "embedding"]
    },
    "Technology": {
        "properties": ["name", "category", "description", "embedding"]
    },
    "UseCase": {
        "properties": ["name", "description", "embedding"]
    }
}

RELATIONSHIP_TYPES = [
    "HAS_DOCUMENT",    # Partner -> Document
    "HAS_ENTITY",      # Partner -> Any Entity Type
    "HAS_SECTION",     # Document -> Section
    "CONTAINS",        # Document -> Section
    "MENTIONS",        # Section -> Concept/Product/Service/Technology
    "RELATES_TO",      # Concept -> Concept
    "IMPLEMENTS",      # Product -> Technology
    "DEPENDS_ON",      # Product -> Product
    "PART_OF",         # Service -> Product
    "ENABLES",         # Technology -> UseCase
    "ALTERNATIVE_TO"   # Product -> Product
]

# UI configuration
UI_PORT = int(os.getenv("UI_PORT", "8501"))
PAGE_TITLE = "Cloud Partner Knowledge Explorer"
PAGE_ICON = "☁️" 