"""
Shared fixtures. Tests run fully offline: a deterministic hashing embedder replaces the
sentence-transformers model and a fake chat model replaces Groq, so CI needs no model
download and no API key.
"""
import hashlib
import re

import numpy as np
import pytest
from langchain.schema import Document
from langchain_core.embeddings import Embeddings

DIM = 256


class HashingEmbeddings(Embeddings):
    """Bag-of-words hashing embedder: texts sharing words get high cosine similarity."""

    def _embed(self, text: str):
        vec = np.zeros(DIM)
        for token in re.findall(r"[a-z0-9]+", text.lower()):
            vec[int(hashlib.md5(token.encode()).hexdigest(), 16) % DIM] += 1.0
        norm = np.linalg.norm(vec)
        return (vec / norm if norm else vec).tolist()

    def embed_documents(self, texts):
        return [self._embed(t) for t in texts]

    def embed_query(self, text):
        return self._embed(text)


@pytest.fixture
def fake_embeddings():
    return HashingEmbeddings()


@pytest.fixture
def corpus():
    return [
        Document(page_content="Chroma is the vector database used for persistent storage.",
                 metadata={"source": "guide.md", "file_type": "text"}),
        Document(page_content="The embedding model is all-MiniLM-L6-v2 producing 384 dimensional vectors.",
                 metadata={"source": "guide.md", "file_type": "text"}),
        Document(page_content="Bananas are yellow fruit rich in potassium.",
                 metadata={"source": "fruit.txt", "file_type": "text"}),
        Document(page_content="Groq serves the gpt-oss-20b language model with low latency.",
                 metadata={"source": "guide.md", "file_type": "text"}),
    ]


@pytest.fixture
def store(tmp_path, fake_embeddings, corpus):
    from rag_app.vectorstore import VectorStore

    vs = VectorStore(persist_directory=tmp_path / "chroma", collection_name="test",
                     embeddings=fake_embeddings)
    vs.add_documents([Document(page_content=d.page_content, metadata=dict(d.metadata)) for d in corpus])
    return vs
