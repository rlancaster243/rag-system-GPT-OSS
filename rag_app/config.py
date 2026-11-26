"""
Configuration module for RAG system.
All environment-specific settings are centralized here.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# chromadb 0.4.x telemetry is incompatible with current posthog releases and only logs
# "Failed to send telemetry event" noise; opt out unless the user explicitly set it.
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CHROMA_PERSIST_DIR = PROJECT_ROOT / "chroma_store"
REPORTS_DIR = PROJECT_ROOT / "reports"

# Groq API configuration
def _load_groq_api_key():
    """Environment / .env first, then Streamlit secrets (Streamlit Cloud writes a secrets.toml).

    st.secrets is only touched when a secrets.toml actually exists: accessing it without one
    renders a red "No secrets files found" banner inside the app (Streamlit 1.30) and raises
    StreamlitSecretNotFoundError on newer versions, even when the key is set via .env.
    """
    key = os.getenv("GROQ_API_KEY")
    if key:
        return key
    secrets_files = [Path.home() / ".streamlit" / "secrets.toml", Path.cwd() / ".streamlit" / "secrets.toml"]
    if not any(p.is_file() for p in secrets_files):
        return None
    try:
        import streamlit as st
        return st.secrets.get("GROQ_API_KEY")
    except Exception:
        return None


GROQ_API_KEY = _load_groq_api_key()

# Note: We don't raise an error here anymore to allow imports to succeed
# Individual modules will check for the key when needed

# LLM configuration
DEFAULT_MODEL = "openai/gpt-oss-20b"
LLM_TEMPERATURE = 0.0

# Embedding configuration
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # Smaller model for Streamlit Cloud
EMBEDDING_DEVICE = "cpu"  # Change to "cuda" if GPU is available

# Chunking configuration
CHUNK_SIZE = 500  # Reduced for Streamlit Cloud memory limits
CHUNK_OVERLAP = 50  # Reduced for Streamlit Cloud memory limits

# Retrieval configuration
DEFAULT_K = 4  # Number of documents to retrieve

# Chroma configuration
CHROMA_COLLECTION_NAME = "rag_documents"

# Supported file extensions
SUPPORTED_EXTENSIONS = {
    ".pdf",
    ".txt",
    ".md",
    ".html",
    ".htm",
    ".py",
    ".ipynb",
}
