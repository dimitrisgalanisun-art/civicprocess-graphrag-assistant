import argparse

from neo4j import GraphDatabase

from src.config import (
    NEO4J_DATABASE,
    NEO4J_PASSWORD,
    NEO4J_URI,
    NEO4J_USERNAME,
    validate_env,
)
from src.vector_index import search_vector_index


def get_driver():
    """Create Neo4j driver."""
    validate_env()

    return GraphDatabase.driver(
        NEO4J_URI,
        auth=(NEO4J_USERNAME, NEO4J_PASSWORD),
    )


def run_read_query(driver, query: str, parameters: dict | None = None) -> list[dict]:
    """Run a read query and return dictionaries."""
    parameters = parameters or {}

    with driver.session(database=NEO4J_DATABASE) as session:
        result = session.run(query, parameters)
        return [dict(record) for record in result]

def get_graph_entry_points(record_ids: list[str]) -> list[dict]:
    """
    Retrieve the exact Neo4j graph nodes corresponding to vector search results.

    These are not graph-expanded neighbors.
    They are the vector records reloaded from Neo4j to prove the ID bridge:

        vector result ID -> Neo4j CivicRAGRecord node
    """
    if not record_ids:
        return []

    driver = get_driver()

    query = """
    MATCH (record:CivicRAGRecord)
    WHERE record.id IN $record_ids

    MATCH (record)-[:HAS_QUESTION]->(question:CivicRAGQuestion)
    MATCH (record)-[:HAS_ANSWER]->(answer:CivicRAGAnswer)

    OPTIONAL MATCH (record)-[:FROM_SOURCE]->(source:CivicRAGSource)
    OPTIONAL MATCH (record)-[:IN_SECTION]->(section:CivicRAGSection)
    OPTIONAL MATCH (record)-[:MENTIONS]->(keyword:CivicRAGKeyword)

    RETURN
        record {
            .id,
            .question,
            .answer,
            .source,
            .section_title,
            .heading_level,
            .chunk_index
        } AS record,
        question.text AS question_text,
        answer.text AS answer_text,
        source.name AS source_name,
        section.name AS section_name,
        collect(DISTINCT keyword.name)[0..15] AS keywords
    """

    try:
        driver.verify_connectivity()

        rows = run_read_query(
            driver,
            query,
            {"record_ids": record_ids},
        )

        records_by_id = {}

        for row in rows:
            record = row["record"]
            record["graph_entry"] = True
            record["question_text"] = row.get("question_text")
            record["answer_text"] = row.get("answer_text")
            record["source_name"] = row.get("source_name")
            record["section_name"] = row.get("section_name")
            record["keywords"] = row.get("keywords", [])
            records_by_id[record["id"]] = record

        # Preserve vector search order.
        return [
            records_by_id[record_id]
            for record_id in record_ids
            if record_id in records_by_id
        ]

    finally:
        driver.close()


def expand_by_section(driver, seed_id: str, limit: int = 3) -> list[dict]:
    """
    Retrieve nearby records from the same document section.

    This is useful because neighboring chunks often contain continuation,
    definitions, examples, and implementation details.
    """
    query = """
    MATCH (seed:CivicRAGRecord {id: $seed_id})
          -[:IN_SECTION]->(section:CivicRAGSection)
          <-[:IN_SECTION]-(record:CivicRAGRecord)
    WHERE record.id <> seed.id
    WITH seed, record,
         abs(coalesce(record.chunk_index, 0) - coalesce(seed.chunk_index, 0)) AS distance
    ORDER BY distance ASC, record.chunk_index ASC
    LIMIT $limit
    RETURN
        record {
            .id,
            .question,
            .answer,
            .source,
            .section_title,
            .heading_level,
            .chunk_index
        } AS record,
        seed.id AS seed_id,
        "same_section" AS graph_reason,
        distance AS graph_score
    """

    return run_read_query(
        driver,
        query,
        {
            "seed_id": seed_id,
            "limit": limit,
        },
    )


def expand_by_keywords(driver, seed_id: str, limit: int = 5) -> list[dict]:
    """
    Retrieve records connected through shared technical keywords.

    Keywords come from backticked identifiers extracted during graph upload.
    """
    query = """
    MATCH (seed:CivicRAGRecord {id: $seed_id})
          -[:MENTIONS]->(keyword:CivicRAGKeyword)
          <-[:MENTIONS]-(record:CivicRAGRecord)
    WHERE record.id <> seed.id
    WITH record,
         collect(DISTINCT keyword.name) AS shared_keywords,
         count(DISTINCT keyword) AS keyword_overlap
    ORDER BY keyword_overlap DESC, record.chunk_index ASC
    LIMIT $limit
    RETURN
        record {
            .id,
            .question,
            .answer,
            .source,
            .section_title,
            .heading_level,
            .chunk_index
        } AS record,
        $seed_id AS seed_id,
        "shared_keywords" AS graph_reason,
        keyword_overlap AS graph_score,
        shared_keywords[0..10] AS shared_keywords
    """

    return run_read_query(
        driver,
        query,
        {
            "seed_id": seed_id,
            "limit": limit,
        },
    )


def graph_expand_records(
    seed_ids: list[str],
    section_limit: int = 3,
    keyword_limit: int = 5,
) -> list[dict]:
    """
    Expand vector-retrieved record IDs through the Neo4j graph.

    Returns related records with graph metadata.
    """
    driver = get_driver()

    expanded = []
    seen_record_ids = set(seed_ids)

    try:
        driver.verify_connectivity()

        for seed_id in seed_ids:
            section_rows = expand_by_section(
                driver,
                seed_id=seed_id,
                limit=section_limit,
            )

            keyword_rows = expand_by_keywords(
                driver,
                seed_id=seed_id,
                limit=keyword_limit,
            )

            for row in section_rows + keyword_rows:
                record = row["record"]

                if record["id"] in seen_record_ids:
                    continue

                seen_record_ids.add(record["id"])

                record["seed_id"] = row["seed_id"]
                record["graph_reason"] = row["graph_reason"]
                record["graph_score"] = row["graph_score"]

                if "shared_keywords" in row:
                    record["shared_keywords"] = row["shared_keywords"]

                expanded.append(record)

    finally:
        driver.close()

    return expanded


def retrieve_graphrag_context(
    query: str,
    vector_top_k: int = 4,
    section_limit: int = 2,
    keyword_limit: int = 3,
) -> dict:
    """
    Full retrieval step:

    user query
    -> vector search
    -> seed record IDs
    -> exact Neo4j graph entry points
    -> graph expansion
    """
    vector_results = search_vector_index(query, top_k=vector_top_k)
    seed_ids = [record["id"] for record in vector_results]

    graph_entry_points = get_graph_entry_points(seed_ids)

    graph_results = graph_expand_records(
        seed_ids=seed_ids,
        section_limit=section_limit,
        keyword_limit=keyword_limit,
    )

    return {
        "query": query,
        "vector_results": vector_results,
        "graph_entry_points": graph_entry_points,
        "graph_results": graph_results,
    }


def print_records(title: str, records: list[dict], max_chars: int = 700) -> None:
    """Pretty-print retrieval records."""
    print("\n" + "=" * 90)
    print(title)
    print("=" * 90)

    if not records:
        print("No records found.")
        return

    for record in records:
        print("-" * 90)
        print(f"ID: {record.get('id')}")
        print(f"Section: {record.get('section_title')}")
        print(f"Chunk: {record.get('chunk_index')}")

        if "rank" in record:
            print(f"Vector rank: {record.get('rank')}")
            print(f"Vector score: {record.get('score'):.4f}")

        if "graph_reason" in record:
            print(f"Graph reason: {record.get('graph_reason')}")
            print(f"Seed ID: {record.get('seed_id')}")
            print(f"Graph score: {record.get('graph_score')}")

        if "shared_keywords" in record:
            print(f"Shared keywords: {record.get('shared_keywords')}")

        print(f"Question: {record.get('question')}")
        print()
        print(record.get("answer", "")[:max_chars])
        print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--query",
        type=str,
        required=True,
        help="User query to retrieve GraphRAG context for.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=4,
        help="Number of vector search seed records.",
    )
    parser.add_argument(
        "--section-limit",
        type=int,
        default=2,
        help="Number of same-section graph records per seed.",
    )
    parser.add_argument(
        "--keyword-limit",
        type=int,
        default=3,
        help="Number of shared-keyword graph records per seed.",
    )

    args = parser.parse_args()

    result = retrieve_graphrag_context(
        query=args.query,
        vector_top_k=args.top_k,
        section_limit=args.section_limit,
        keyword_limit=args.keyword_limit,
    )

    print(f"\nQuery: {result['query']}")

    print_records("VECTOR RESULTS", result["vector_results"])
    print_records("GRAPH ENTRY POINTS", result["graph_entry_points"])
    print_records("GRAPH-EXPANDED RESULTS", result["graph_results"])


if __name__ == "__main__":
    main()