"""Vector store dedup, retrieval, RAG chain and API behaviour (offline)."""
import pytest
from fastapi.testclient import TestClient
from langchain.schema import Document
from langchain_community.chat_models.fake import FakeListChatModel

from rag_app import api
from rag_app.rag_chain import answer_question, format_documents
from rag_app.retrieval import format_sources, retrieve_documents
from rag_app.vectorstore import VectorStore


def test_store_counts_corpus(store, corpus):
    assert store.count() == len(corpus)


def test_reingest_adds_nothing(store, corpus):
    copies = [Document(page_content=d.page_content, metadata=dict(d.metadata)) for d in corpus]
    assert store.add_documents(copies) == 0
    assert store.count() == len(corpus)


def test_duplicates_within_one_batch_are_indexed_once(tmp_path, fake_embeddings):
    vs = VectorStore(persist_directory=tmp_path / "c", collection_name="dup", embeddings=fake_embeddings)
    doc = lambda: Document(page_content="same text", metadata={"source": "a.txt"})
    assert vs.add_documents([doc(), doc(), doc()]) == 1
    assert vs.count() == 1


def test_empty_add_returns_zero(tmp_path, fake_embeddings):
    vs = VectorStore(persist_directory=tmp_path / "e", collection_name="empty", embeddings=fake_embeddings)
    assert vs.add_documents([]) == 0
    assert vs.count() == 0


def test_store_persists_across_instances(tmp_path, fake_embeddings, store):
    reopened = VectorStore(persist_directory=store.persist_directory, collection_name="test",
                           embeddings=fake_embeddings)
    assert reopened.count() == store.count()


def test_retrieval_ranks_relevant_chunk_first(store):
    docs = retrieve_documents("which vector database is used", k=2, vectorstore=store)
    assert len(docs) == 2
    assert "Chroma" in docs[0].page_content


def test_format_sources_truncates_and_locates():
    long_doc = Document(page_content="x" * 500, metadata={"source": "a.pdf", "file_type": "pdf", "page": 3})
    src = format_sources([long_doc])[0]
    assert src["location"] == "Page 3"
    assert len(src["snippet"]) == 203 and src["snippet"].endswith("...")


def test_format_documents_includes_source_and_location():
    ctx = format_documents([Document(page_content="hello", metadata={"source": "nb.ipynb",
                                                                       "file_type": "notebook",
                                                                       "cell_index": 2})])
    assert "nb.ipynb" in ctx and "Cell 2" in ctx and "hello" in ctx


def test_answer_question_uses_injected_llm(store):
    llm = FakeListChatModel(responses=["Chroma [source: guide.md, text, N/A]"])
    out = answer_question("which vector database?", k=2, vectorstore=store, llm=llm)
    assert out["answer"].startswith("Chroma")
    assert out["retrieved_docs"] == 2
    assert {s["filename"] for s in out["sources"]} <= {"guide.md", "fruit.txt"}


def test_answer_question_with_empty_store_skips_llm(tmp_path, fake_embeddings):
    empty = VectorStore(persist_directory=tmp_path / "x", collection_name="none", embeddings=fake_embeddings)

    class ExplodingLLM:
        def __getattr__(self, name):
            raise AssertionError("LLM must not be called when nothing is retrieved")

    out = answer_question("anything", vectorstore=empty, llm=ExplodingLLM())
    assert out["retrieved_docs"] == 0 and out["sources"] == []


def test_get_llm_without_key_raises(monkeypatch):
    import rag_app.models as models
    monkeypatch.setattr(models, "GROQ_API_KEY", None)
    with pytest.raises(ValueError, match="GROQ_API_KEY"):
        models.get_llm()


@pytest.fixture
def client():
    return TestClient(api.app)


def test_api_health(client):
    assert client.get("/health").json()["status"] == "healthy"


def test_api_query_success(client, monkeypatch):
    monkeypatch.setattr(api, "answer_question", lambda q, k: {
        "answer": f"echo {q} {k}",
        "sources": [{"filename": "a.md", "file_type": "text", "location": "N/A", "snippet": "s"}],
        "retrieved_docs": 1,
    })
    r = client.post("/query", json={"question": "hi", "k": 3})
    assert r.status_code == 200
    assert r.json()["answer"] == "echo hi 3"


def test_api_query_validation(client):
    assert client.post("/query", json={"question": ""}).status_code == 422
    assert client.post("/query", json={"question": "q", "k": 50}).status_code == 422


def test_api_query_error_is_500(client, monkeypatch):
    def boom(q, k):
        raise RuntimeError("groq down")
    monkeypatch.setattr(api, "answer_question", boom)
    r = client.post("/query", json={"question": "hi"})
    assert r.status_code == 500 and "groq down" in r.json()["detail"]
