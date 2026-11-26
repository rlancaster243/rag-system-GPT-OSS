"""
Embeddings module using local sentence-transformers.
Provides sentence-transformer embeddings (all-MiniLM-L6-v2 by default) without external APIs.
"""
from functools import lru_cache
from typing import List
from sentence_transformers import SentenceTransformer
from langchain.embeddings.base import Embeddings
from rag_app.config import EMBEDDING_MODEL, EMBEDDING_DEVICE


class LocalEmbeddings(Embeddings):
    """
    Local embeddings using a sentence-transformers model.
    Implements LangChain's Embeddings interface for compatibility.
    """

    def __init__(self, model_name: str = EMBEDDING_MODEL, device: str = EMBEDDING_DEVICE):
        """
        Initialize the local embedding model.

        Args:
            model_name: Name of the sentence-transformers model
            device: Device to run the model on ('cpu' or 'cuda')
        """
        self.model = SentenceTransformer(model_name, device=device)
        self.model_name = model_name

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a list of documents.

        Args:
            texts: List of text strings to embed

        Returns:
            List of embedding vectors
        """
        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return embeddings.tolist()

    def embed_query(self, text: str) -> List[float]:
        """
        Embed a single query string.

        Args:
            text: Query text to embed

        Returns:
            Embedding vector
        """
        embedding = self.model.encode(
            text,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return embedding.tolist()


@lru_cache(maxsize=4)
def get_embeddings(model_name: str = EMBEDDING_MODEL, device: str = EMBEDDING_DEVICE) -> LocalEmbeddings:
    """
    Return a process-wide cached embeddings model.

    Loading a SentenceTransformer takes seconds and hundreds of MB of RAM; previously every
    query (and every evaluation run) loaded a fresh copy. Caching keeps one instance per
    (model, device) pair.
    """
    return LocalEmbeddings(model_name=model_name, device=device)
