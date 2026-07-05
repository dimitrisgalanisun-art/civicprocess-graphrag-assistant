import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.graph_retrieval import retrieve_graphrag_context


DEFAULT_QUESTIONS_FILE = PROJECT_ROOT / "evaluation" / "evaluation_questions.example.jsonl"
RESULTS_DIR = PROJECT_ROOT / "evaluation" / "results"


def load_jsonl(path: Path) -> list[dict]:
    """Load evaluation questions from JSONL."""
    if not path.exists():
        raise FileNotFoundError(f"Evaluation file not found: {path}")

    records = []

    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                records.append(json.loads(line))

    return records


def normalize_text(text: str) -> str:
    """Normalize text for simple keyword matching."""
    return text.lower().replace("ς", "σ")


def collect_retrieved_text(retrieval: dict) -> str:
    """Collect text from vector and graph results for evaluation."""
    parts = []

    for record in retrieval.get("vector_results", []):
        parts.append(record.get("question", ""))
        parts.append(record.get("answer", ""))
        parts.append(record.get("section_title", ""))

    for record in retrieval.get("graph_entry_points", []):
        parts.append(record.get("question_text", ""))
        parts.append(record.get("answer_text", ""))
        parts.append(record.get("section_name", ""))

    for record in retrieval.get("graph_results", []):
        parts.append(record.get("question", ""))
        parts.append(record.get("answer", ""))
        parts.append(record.get("section_title", ""))
        parts.extend(record.get("shared_keywords", []))

    return "\n".join(part for part in parts if part)


def evaluate_question(
    item: dict,
    top_k: int,
    section_limit: int,
    keyword_limit: int,
) -> dict:
    """Run one retrieval evaluation item."""
    question = item["question"]
    expected_keywords = item.get("expected_keywords", [])

    retrieval = retrieve_graphrag_context(
        query=question,
        vector_top_k=top_k,
        section_limit=section_limit,
        keyword_limit=keyword_limit,
    )

    retrieved_text = normalize_text(collect_retrieved_text(retrieval))

    matched_keywords = []

    for keyword in expected_keywords:
        normalized_keyword = normalize_text(keyword)

        if normalized_keyword in retrieved_text:
            matched_keywords.append(keyword)

    total_keywords = len(expected_keywords)
    matched_count = len(matched_keywords)

    score = matched_count / total_keywords if total_keywords else 0.0

    top_vector_ids = [
        record.get("id")
        for record in retrieval.get("vector_results", [])
    ]

    graph_entry_ids = [
        record.get("id")
        for record in retrieval.get("graph_entry_points", [])
    ]

    graph_neighbor_ids = [
        record.get("id")
        for record in retrieval.get("graph_results", [])
    ]

    return {
        "id": item.get("id"),
        "question": question,
        "expected_keywords": expected_keywords,
        "matched_keywords": matched_keywords,
        "score": round(score, 3),
        "top_vector_ids": top_vector_ids,
        "graph_entry_ids": graph_entry_ids,
        "graph_neighbor_ids": graph_neighbor_ids,
    }


def print_result(result: dict) -> None:
    """Print one evaluation result."""
    print("\n" + "=" * 90)
    print(f"{result['id']} | {result['question']}")
    print("=" * 90)

    print(f"Score: {result['score']}")
    print(f"Matched keywords: {', '.join(result['matched_keywords']) or 'None'}")

    print("\nVector IDs:")
    for record_id in result["top_vector_ids"]:
        print(f"- {record_id}")

    print("\nGraph entry IDs:")
    for record_id in result["graph_entry_ids"]:
        print(f"- {record_id}")

    print("\nGraph neighbor IDs:")
    for record_id in result["graph_neighbor_ids"][:10]:
        print(f"- {record_id}")


def save_results(results: list[dict]) -> Path:
    """Save evaluation results to a private ignored results folder."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    output_path = RESULTS_DIR / "retrieval_evaluation_results.json"

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(results, file, ensure_ascii=False, indent=2)

    return output_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--questions-file",
        type=Path,
        default=DEFAULT_QUESTIONS_FILE,
        help="Path to evaluation questions JSONL file.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of vector matches.",
    )
    parser.add_argument(
        "--section-limit",
        type=int,
        default=0,
        help="Number of same-section graph neighbors.",
    )
    parser.add_argument(
        "--keyword-limit",
        type=int,
        default=1,
        help="Number of keyword graph neighbors.",
    )

    args = parser.parse_args()

    questions = load_jsonl(args.questions_file)

    print(f"Loaded {len(questions)} evaluation questions.")
    print(f"Vector top-k: {args.top_k}")
    print(f"Same-section graph neighbors: {args.section_limit}")
    print(f"Keyword graph neighbors: {args.keyword_limit}")

    results = []

    for item in questions:
        result = evaluate_question(
            item=item,
            top_k=args.top_k,
            section_limit=args.section_limit,
            keyword_limit=args.keyword_limit,
        )
        results.append(result)
        print_result(result)

    average_score = (
        sum(result["score"] for result in results) / len(results)
        if results
        else 0.0
    )

    print("\n" + "=" * 90)
    print("SUMMARY")
    print("=" * 90)
    print(f"Average retrieval keyword score: {average_score:.3f}")

    output_path = save_results(results)
    print(f"Saved results to: {output_path}")


if __name__ == "__main__":
    main()