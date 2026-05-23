import asyncio
import json
import sys
from pathlib import Path
from uuid import UUID

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from app.core.config import settings
from app.services.retrieval import hybrid_retrieve_relevant_chunks


BENCHMARK_PATH = PROJECT_ROOT / "evals" / "benchmark_questions.json"


def load_benchmark_questions() -> list[dict]:
    with open(BENCHMARK_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


def calculate_hit_at_k(
    retrieved_chunk_indexes: list[int],
    expected_chunk_indexes: list[int],
) -> int:
    return int(bool(set(retrieved_chunk_indexes) & set(expected_chunk_indexes)))


def calculate_precision_at_k(
    retrieved_chunk_indexes: list[int],
    expected_chunk_indexes: list[int],
) -> float:
    if not retrieved_chunk_indexes:
        return 0.0

    retrieved_set = set(retrieved_chunk_indexes)
    expected_set = set(expected_chunk_indexes)

    correct = len(retrieved_set & expected_set)
    return correct / len(retrieved_chunk_indexes)


def calculate_mrr(
    retrieved_chunk_indexes: list[int],
    expected_chunk_indexes: list[int],
) -> float:
    expected_set = set(expected_chunk_indexes)

    for rank, chunk_index in enumerate(retrieved_chunk_indexes, start=1):
        if chunk_index in expected_set:
            return 1.0 / rank

    return 0.0


async def evaluate_retrieval(top_k: int = 3) -> dict:
    if not settings.EVAL_DOCUMENT_ID:
        raise ValueError("EVAL_DOCUMENT_ID is missing in settings/.env")

    document_id = UUID(settings.EVAL_DOCUMENT_ID)
    questions = load_benchmark_questions()

    results = []

    total_hit = 0
    total_precision = 0.0
    total_mrr = 0.0

    for item in questions:
        question = item["question"]
        expected_chunk_indexes = item["expected_chunk_indexes"]

        retrieved_chunks = await hybrid_retrieve_relevant_chunks(
            query=question,
            top_k=top_k,
            document_id=document_id,
        )

        retrieved_chunk_indexes = [
            chunk["chunk_index"]
            for chunk in retrieved_chunks
        ]

        hit_at_k = calculate_hit_at_k(
            retrieved_chunk_indexes=retrieved_chunk_indexes,
            expected_chunk_indexes=expected_chunk_indexes,
        )

        precision_at_k = calculate_precision_at_k(
            retrieved_chunk_indexes=retrieved_chunk_indexes,
            expected_chunk_indexes=expected_chunk_indexes,
        )

        mrr = calculate_mrr(
            retrieved_chunk_indexes=retrieved_chunk_indexes,
            expected_chunk_indexes=expected_chunk_indexes,
        )

        total_hit += hit_at_k
        total_precision += precision_at_k
        total_mrr += mrr

        results.append(
            {
                "id": item["id"],
                "question": question,
                "expected_chunk_indexes": expected_chunk_indexes,
                "retrieved_chunk_indexes": retrieved_chunk_indexes,
                "hit_at_k": hit_at_k,
                "precision_at_k": precision_at_k,
                "mrr": mrr,
                "top_results": [
                    {
                        "chunk_id": chunk["chunk_id"],
                        "chunk_index": chunk["chunk_index"],
                        "page_number": chunk["page_number"],
                        "dense_score": chunk["dense_score"],
                        "sparse_score": chunk["sparse_score"],
                        "hybrid_score": chunk["hybrid_score"],
                    }
                    for chunk in retrieved_chunks
                ],
            }
        )

    question_count = len(questions)

    summary = {
        "top_k": top_k,
        "question_count": question_count,
        "hit_at_k": total_hit / question_count if question_count else 0.0,
        "precision_at_k": total_precision / question_count if question_count else 0.0,
        "mrr": total_mrr / question_count if question_count else 0.0,
    }

    return {
        "summary": summary,
        "results": results,
    }


def print_report(report: dict) -> None:
    summary = report["summary"]

    print("\nRetrieval Evaluation Report")
    print("=" * 32)
    print(f"top_k:          {summary['top_k']}")
    print(f"questions:      {summary['question_count']}")
    print(f"hit@k:          {summary['hit_at_k']:.3f}")
    print(f"precision@k:    {summary['precision_at_k']:.3f}")
    print(f"MRR:            {summary['mrr']:.3f}")

    print("\nQuestion Results")
    print("=" * 32)

    for item in report["results"]:
        print(f"\n[{item['id']}] {item['question']}")
        print(f"expected:  {item['expected_chunk_indexes']}")
        print(f"retrieved: {item['retrieved_chunk_indexes']}")
        print(f"hit@k:     {item['hit_at_k']}")
        print(f"p@k:       {item['precision_at_k']:.3f}")
        print(f"mrr:       {item['mrr']:.3f}")


async def main() -> None:
    report = await evaluate_retrieval(top_k=3)
    print_report(report)

    output_path = PROJECT_ROOT / "evals" / "retrieval_eval_report.json"

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)

    print(f"\nSaved JSON report to: {output_path}")


if __name__ == "__main__":
    asyncio.run(main())