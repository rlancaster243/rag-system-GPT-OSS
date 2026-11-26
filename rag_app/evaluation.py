"""
Evaluation module for the RAG system.

Two layers, so the system can be measured with or without an LLM key:

1. Retrieval metrics (offline, deterministic, no API calls)
   - hit_rate@k : share of questions where a relevant chunk appears in the top-k
   - MRR        : mean reciprocal rank of the first relevant chunk
   - precision@k: share of retrieved chunks that are relevant
   A chunk is "relevant" when it comes from ``expected_source`` (if given) AND contains
   at least one of ``relevant_keywords`` (case-insensitive, if given).

2. Answer metrics (requires GROQ_API_KEY)
   - semantic similarity between the generated and the expected answer (cosine of embeddings)
   - keyword recall: share of ``answer_keywords`` that appear in the generated answer
   - source coverage: share of answers returned with at least one source
   Questions that error out are reported separately and excluded from the averages
   instead of silently counting as a 0.0 score.

Eval file format (JSONL, one object per line, or CSV with the same column names;
list fields in CSV are ``|``-separated):
    {"question": "...", "expected_answer": "...", "expected_source": "sample_document.md",
     "relevant_keywords": ["..."], "answer_keywords": ["..."]}
"""
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from rag_app.config import DEFAULT_K, REPORTS_DIR

LIST_FIELDS = ("relevant_keywords", "answer_keywords")


def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """Cosine similarity between two vectors (0.0 if either is all zeros)."""
    v1 = np.asarray(vec1, dtype=float)
    v2 = np.asarray(vec2, dtype=float)
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(np.dot(v1, v2) / (norm1 * norm2))


def load_eval_set(eval_file: Path) -> List[Dict]:
    """Load and normalise an evaluation set from JSONL or CSV."""
    eval_file = Path(eval_file)
    if eval_file.suffix == ".jsonl":
        with open(eval_file, "r", encoding="utf-8") as f:
            items = [json.loads(line) for line in f if line.strip()]
    elif eval_file.suffix == ".csv":
        with open(eval_file, "r", encoding="utf-8") as f:
            items = list(csv.DictReader(f))
        for item in items:
            for field in LIST_FIELDS:
                if isinstance(item.get(field), str):
                    item[field] = [v.strip() for v in item[field].split("|") if v.strip()]
    else:
        raise ValueError(f"Unsupported file format: {eval_file.suffix}")

    items = [item for item in items if item.get("question")]
    if not items:
        raise ValueError(f"No evaluation questions found in {eval_file}")
    for item in items:
        for field in LIST_FIELDS:
            item.setdefault(field, [])
    return items


def is_relevant(doc, item: Dict) -> bool:
    """Decide whether a retrieved chunk is relevant to an eval item."""
    expected_source = item.get("expected_source")
    if expected_source and doc.metadata.get("source") != expected_source:
        return False
    keywords = item.get("relevant_keywords") or []
    if keywords:
        content = doc.page_content.lower()
        return any(kw.lower() in content for kw in keywords)
    # No keywords: matching the source is enough (if no source either, nothing is relevant).
    return bool(expected_source)


def evaluate_retrieval(items: List[Dict], k: int = DEFAULT_K, vectorstore=None) -> Dict:
    """Compute hit_rate@k, MRR and precision@k over an eval set. No LLM calls."""
    from rag_app.retrieval import retrieve_documents

    per_question = []
    for item in items:
        docs = retrieve_documents(item["question"], k=k, vectorstore=vectorstore)
        flags = [is_relevant(doc, item) for doc in docs]
        first_rank = next((i + 1 for i, rel in enumerate(flags) if rel), None)
        per_question.append({
            "question": item["question"],
            "first_relevant_rank": first_rank,
            "reciprocal_rank": 1.0 / first_rank if first_rank else 0.0,
            "precision": sum(flags) / k if k else 0.0,
            "retrieved_sources": [doc.metadata.get("source") for doc in docs],
        })

    n = len(per_question)
    summary = {
        "k": k,
        "num_questions": n,
        "hit_rate": sum(q["first_relevant_rank"] is not None for q in per_question) / n,
        "mrr": sum(q["reciprocal_rank"] for q in per_question) / n,
        "precision_at_k": sum(q["precision"] for q in per_question) / n,
    }
    return {"summary": summary, "results": per_question}


def evaluate_answers(items: List[Dict], k: int = DEFAULT_K, vectorstore=None, llm=None) -> Dict:
    """Generate answers with the full RAG chain and score them against references."""
    from rag_app.embeddings import get_embeddings
    from rag_app.rag_chain import answer_question

    embeddings = get_embeddings()
    results, errors = [], []

    for i, item in enumerate(items, 1):
        question = item["question"]
        print(f"[{i}/{len(items)}] {question[:70]}")
        try:
            out = answer_question(question, k=k, vectorstore=vectorstore, llm=llm)
        except Exception as e:  # network/auth/rate-limit errors are reported, not scored
            errors.append({"question": question, "error": str(e)})
            continue

        answer = out["answer"]
        row = {
            "question": question,
            "generated_answer": answer,
            "num_sources": len(out["sources"]),
            "has_sources": bool(out["sources"]),
        }
        if item.get("expected_answer"):
            row["expected_answer"] = item["expected_answer"]
            row["similarity_score"] = cosine_similarity(
                embeddings.embed_query(item["expected_answer"]), embeddings.embed_query(answer)
            )
        if item["answer_keywords"]:
            hits = [kw for kw in item["answer_keywords"] if kw.lower() in answer.lower()]
            row["keyword_recall"] = len(hits) / len(item["answer_keywords"])
        results.append(row)

    def _mean(key):
        vals = [r[key] for r in results if key in r]
        return float(np.mean(vals)) if vals else None

    summary = {
        "num_questions": len(items),
        "num_answered": len(results),
        "num_errors": len(errors),
        "average_similarity": _mean("similarity_score"),
        "average_keyword_recall": _mean("keyword_recall"),
        "source_coverage": (sum(r["has_sources"] for r in results) / len(results)) if results else None,
    }
    return {"summary": summary, "results": results, "errors": errors}


def evaluate_rag(
    eval_file: Path,
    k: int = DEFAULT_K,
    output_file: Optional[Path] = None,
    retrieval_only: bool = False,
    vectorstore=None,
    llm=None,
) -> Dict:
    """Run retrieval (and optionally answer) evaluation and save a JSON report."""
    items = load_eval_set(eval_file)
    print(f"📊 Evaluating {len(items)} questions (k={k})...")

    report = {"eval_file": str(eval_file), "retrieval": evaluate_retrieval(items, k, vectorstore)}
    if not retrieval_only:
        report["answers"] = evaluate_answers(items, k, vectorstore, llm)

    print_summary(report)

    if output_file is None:
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        output_file = REPORTS_DIR / "eval_summary.json"
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"💾 Report saved to: {output_file}")
    return report


def print_summary(report: Dict) -> None:
    """Pretty-print the headline metrics of a report."""
    r = report["retrieval"]["summary"]
    print("\n" + "=" * 70)
    print("📊 RETRIEVAL")
    print(f"  hit_rate@{r['k']}: {r['hit_rate']:.3f}   MRR: {r['mrr']:.3f}   "
          f"precision@{r['k']}: {r['precision_at_k']:.3f}   (n={r['num_questions']})")
    if "answers" in report:
        a = report["answers"]["summary"]
        fmt = lambda v: "n/a" if v is None else f"{v:.3f}"
        print("📝 ANSWERS")
        print(f"  answered: {a['num_answered']}/{a['num_questions']}   errors: {a['num_errors']}")
        print(f"  similarity: {fmt(a['average_similarity'])}   keyword recall: "
              f"{fmt(a['average_keyword_recall'])}   source coverage: {fmt(a['source_coverage'])}")
    print("=" * 70)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate the RAG system")
    parser.add_argument("eval_file", type=Path)
    parser.add_argument("-k", type=int, default=DEFAULT_K)
    parser.add_argument("--retrieval-only", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    evaluate_rag(args.eval_file, k=args.k, output_file=args.output, retrieval_only=args.retrieval_only)
