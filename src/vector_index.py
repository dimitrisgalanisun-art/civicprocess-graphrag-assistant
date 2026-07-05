import argparse
import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from src.config import EMBEDDING_MODEL, QA_JSONL_FILE, VECTOR_DB_DIR


EMBEDDINGS_FILE = VECTOR_DB_DIR / "embeddings.npy"
RECORDS_FILE = VECTOR_DB_DIR / "records.jsonl"


def load_jsonl(path: Path) -> list[dict]:
    """Load Q/A records from a JSONL file."""
    if not path.exists():
        raise FileNotFoundError(
            f"JSONL dataset not found: {path}\n"
            "Run: python -m src.build_dataset"
        )

    records = []

    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                records.append(json.loads(line))

    return records


def save_jsonl(records: list[dict], path: Path) -> None:
    """Save records to JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def build_vector_index() -> None:
    """
    Build a local vector index from private JSONL records.

    The index is saved under vector_db/, which is ignored by Git.
    """
    records = load_jsonl(QA_JSONL_FILE)

    texts = [
        record.get("text_for_embedding")
        or f"{record.get('question', '')}\n{record.get('answer', '')}"
        for record in records
    ]

    print(f"Loading embedding model: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)

    print(f"Encoding {len(texts)} records...")
    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    VECTOR_DB_DIR.mkdir(parents=True, exist_ok=True)

    np.save(EMBEDDINGS_FILE, embeddings)
    save_jsonl(records, RECORDS_FILE)

    print(f"Saved embeddings to: {EMBEDDINGS_FILE}")
    print(f"Saved records to: {RECORDS_FILE}")
    print(f"Vector index size: {len(records)} records")


def load_vector_index() -> tuple[np.ndarray, list[dict]]:
    """Load the local vector index."""
    if not EMBEDDINGS_FILE.exists() or not RECORDS_FILE.exists():
        raise FileNotFoundError(
            "Vector index not found.\n"
            "Run: python -m src.vector_index --build"
        )

    embeddings = np.load(EMBEDDINGS_FILE)
    records = load_jsonl(RECORDS_FILE)

    return embeddings, records


def search_vector_index(query: str, top_k: int = 5) -> list[dict]:
    """
    Search the vector index using cosine similarity.

    Because embeddings are normalized, dot product equals cosine similarity.
    """
    embeddings, records = load_vector_index()

    model = SentenceTransformer(EMBEDDING_MODEL)

    query_embedding = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )[0]

    scores = np.dot(embeddings, query_embedding)

    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []

    for rank, index in enumerate(top_indices, start=1):
        record = records[int(index)].copy()
        record["rank"] = rank
        record["score"] = float(scores[int(index)])
        results.append(record)

    return results


def print_search_results(results: list[dict]) -> None:
    """Pretty-print search results in the terminal."""
    for result in results:
        print("=" * 80)
        print(f"Rank: {result['rank']}")
        print(f"Score: {result['score']:.4f}")
        print(f"ID: {result['id']}")
        print(f"Section: {result.get('section_title', '')}")
        print(f"Question: {result.get('question', '')}")
        print("-" * 80)
        print(result.get("answer", "")[:1000])
        print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--build",
        action="store_true",
        help="Build the local vector index.",
    )
    parser.add_argument(
        "--search",
        type=str,
        help="Search the local vector index.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of search results to return.",
    )

    args = parser.parse_args()

    if args.build:
        build_vector_index()

    if args.search:
        results = search_vector_index(args.search, top_k=args.top_k)
        print_search_results(results)

    if not args.build and not args.search:
        parser.print_help()


if __name__ == "__main__":
    main()