"""Evaluation metrics: loading, relevance, retrieval metrics and answer scoring (offline)."""
import json
from pathlib import Path

import pytest
from langchain.schema import Document
from langchain_community.chat_models.fake import FakeListChatModel

import rag_app.embeddings as embeddings_mod
from rag_app.evaluation import (
    cosine_similarity,
    evaluate_answers,
    evaluate_rag,
    evaluate_retrieval,
    is_relevant,
    load_eval_set,
)

REPO_EVAL_SET = Path(__file__).resolve().parent.parent / "eval" / "sample_eval.jsonl"


def test_cosine_similarity_edge_cases():
    assert cosine_similarity([1, 0], [1, 0]) == pytest.approx(1.0)
    assert cosine_similarity([1, 0], [0, 1]) == pytest.approx(0.0)
    assert cosine_similarity([0, 0], [1, 1]) == 0.0


def test_load_jsonl_and_csv(tmp_path):
    jl = tmp_path / "e.jsonl"
    jl.write_text('{"question": "q1", "relevant_keywords": ["a"]}\n\n{"question": ""}\n')
    items = load_eval_set(jl)
    assert len(items) == 1 and items[0]["answer_keywords"] == []

    cv = tmp_path / "e.csv"
    cv.write_text("question,expected_answer,relevant_keywords\nq,a,foo | bar\n")
    assert load_eval_set(cv)[0]["relevant_keywords"] == ["foo", "bar"]


def test_load_rejects_bad_inputs(tmp_path):
    with pytest.raises(ValueError):
        load_eval_set(tmp_path / "x.txt")
    empty = tmp_path / "empty.jsonl"
    empty.write_text("")
    with pytest.raises(ValueError):
        load_eval_set(empty)


def test_repo_eval_set_is_valid():
    items = load_eval_set(REPO_EVAL_SET)
    assert len(items) >= 5
    for item in items:
        assert item["expected_source"] and item["relevant_keywords"] and item["expected_answer"]


def test_repo_eval_keywords_exist_in_sample_document():
    """Guards against the eval set drifting from the document it is labelled against."""
    text = (REPO_EVAL_SET.parent.parent / "data" / "sample_document.md").read_text().lower()
    for item in load_eval_set(REPO_EVAL_SET):
        assert any(kw.lower() in text for kw in item["relevant_keywords"]), item["question"]


def test_is_relevant_rules():
    doc = Document(page_content="Chroma store", metadata={"source": "a.md"})
    assert is_relevant(doc, {"expected_source": "a.md", "relevant_keywords": ["chroma"]})
    assert not is_relevant(doc, {"expected_source": "b.md", "relevant_keywords": ["chroma"]})
    assert not is_relevant(doc, {"expected_source": "a.md", "relevant_keywords": ["banana"]})
    assert is_relevant(doc, {"expected_source": "a.md", "relevant_keywords": []})
    assert not is_relevant(doc, {"relevant_keywords": []})


def test_retrieval_metrics(store):
    items = [
        {"question": "which vector database", "expected_source": "guide.md", "relevant_keywords": ["Chroma"]},
        {"question": "yellow fruit potassium", "expected_source": "fruit.txt", "relevant_keywords": ["Bananas"]},
        {"question": "which vector database", "expected_source": "guide.md", "relevant_keywords": ["unicorn"]},
    ]
    s = evaluate_retrieval(items, k=2, vectorstore=store)["summary"]
    assert s["hit_rate"] == pytest.approx(2 / 3)
    assert s["mrr"] == pytest.approx(2 / 3)  # two rank-1 hits, one miss
    assert s["precision_at_k"] == pytest.approx((0.5 + 0.5 + 0) / 3)


def test_answer_metrics_and_error_handling(store, fake_embeddings, monkeypatch):
    monkeypatch.setattr(embeddings_mod, "get_embeddings", lambda: fake_embeddings)
    items = [
        {"question": "which vector database", "expected_answer": "Chroma vector database",
         "answer_keywords": ["Chroma", "persistent"]},
    ]
    llm = FakeListChatModel(responses=["Chroma vector database"])
    res = evaluate_answers(items, k=2, vectorstore=store, llm=llm)
    row = res["results"][0]
    assert row["similarity_score"] == pytest.approx(1.0)
    assert row["keyword_recall"] == pytest.approx(0.5)
    assert res["summary"]["source_coverage"] == 1.0 and res["summary"]["num_errors"] == 0

    class FailingLLM(FakeListChatModel):
        def _call(self, *a, **kw):
            raise RuntimeError("401 invalid key")

    res = evaluate_answers(items, k=2, vectorstore=store, llm=FailingLLM(responses=["x"]))
    assert res["summary"]["num_errors"] == 1 and res["summary"]["num_answered"] == 0
    assert res["summary"]["average_similarity"] is None  # errors are not averaged in as 0.0


def test_evaluate_rag_writes_report(tmp_path, store):
    eval_file = tmp_path / "e.jsonl"
    eval_file.write_text(json.dumps({"question": "which vector database",
                                     "expected_source": "guide.md", "relevant_keywords": ["Chroma"]}))
    out = tmp_path / "out" / "report.json"
    report = evaluate_rag(eval_file, k=2, output_file=out, retrieval_only=True, vectorstore=store)
    assert out.exists() and "answers" not in report
    assert json.loads(out.read_text())["retrieval"]["summary"]["hit_rate"] == 1.0
