import argparse
from functools import lru_cache

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.config import HF_TOKEN, MODEL_NAME
from src.graph_retrieval import retrieve_graphrag_context


MAX_CONTEXT_CHARS = 9000


def format_record(record: dict, source_type: str) -> str:
    """Format one retrieved record for the LLM context."""
    lines = [
        "[RETRIEVED RECORD]",
        f"Source type: {source_type}",
        f"Record ID: {record.get('id')}",
        f"Section: {record.get('section_title')}",
        f"Chunk: {record.get('chunk_index')}",
    ]

    if "rank" in record:
        lines.append(f"Vector rank: {record.get('rank')}")
        lines.append(f"Vector score: {record.get('score'):.4f}")

    if "graph_reason" in record:
        lines.append(f"Graph reason: {record.get('graph_reason')}")
        lines.append(f"Seed ID: {record.get('seed_id')}")
        lines.append(f"Graph score: {record.get('graph_score')}")

    if "shared_keywords" in record:
        lines.append(f"Shared keywords: {record.get('shared_keywords')}")

    lines.extend(
        [
            "",
            "Question stored in record:",
            record.get("question", ""),
            "",
            "Content:",
            record.get("answer", ""),
        ]
    )

    return "\n".join(lines)


def build_vector_context(vector_results: list[dict]) -> str:
    """Build context from direct vector matches."""
    blocks = []

    for i, record in enumerate(vector_results, start=1):
        blocks.append(
            f"""
Vector Match {i}
Record ID: {record.get("id")}
Section: {record.get("section_title")}
Chunk: {record.get("chunk_index")}
Similarity Score: {record.get("score"):.4f}
Question: {record.get("question", "")}
Answer: {record.get("answer", "")}
""".strip()
        )

    return "\n\n".join(blocks)


def build_graph_entry_context(graph_entry_points: list[dict]) -> str:
    """Build context from exact Neo4j graph entry points."""
    if not graph_entry_points:
        return "No graph entry points found."

    blocks = []

    for i, record in enumerate(graph_entry_points, start=1):
        keywords = ", ".join(record.get("keywords", [])) or "None"

        blocks.append(
            f"""
Graph Entry Point {i}
Record ID: {record.get("id")}
Source: {record.get("source_name") or record.get("source")}
Section: {record.get("section_name") or record.get("section_title")}
Chunk: {record.get("chunk_index")}
Keywords: {keywords}
Question: {record.get("question_text") or record.get("question", "")}
Answer: {record.get("answer_text") or record.get("answer", "")}
""".strip()
        )

    return "\n\n".join(blocks)


def build_graph_expansion_context(graph_results: list[dict]) -> str:
    """Build context from graph-expanded neighbor records."""
    if not graph_results:
        return "No graph-expanded neighbors found."

    blocks = []

    for i, record in enumerate(graph_results, start=1):
        shared_keywords = ", ".join(record.get("shared_keywords", [])) or "None"

        blocks.append(
            f"""
Graph Neighbor {i}
Related Record ID: {record.get("id")}
Starting Vector Record: {record.get("seed_id")}
Reason: {record.get("graph_reason")}
Graph Score: {record.get("graph_score")}
Section: {record.get("section_title")}
Chunk: {record.get("chunk_index")}
Shared Keywords: {shared_keywords}
Question: {record.get("question", "")}
Answer: {record.get("answer", "")}
""".strip()
        )

    return "\n\n".join(blocks)


def build_full_context(
    vector_results: list[dict],
    graph_entry_points: list[dict],
    graph_results: list[dict],
) -> str:
    """Build the three-part Exercise-5b-style context."""
    context = f"""
DIRECT VECTOR MATCHES:
{build_vector_context(vector_results)}

GRAPH ENTRY POINTS:
{build_graph_entry_context(graph_entry_points)}

GRAPH-EXPANDED NEIGHBORS:
{build_graph_expansion_context(graph_results)}
""".strip()

    if len(context) > MAX_CONTEXT_CHARS:
        context = context[:MAX_CONTEXT_CHARS] + "\n\n[CONTEXT TRUNCATED]"

    return context

def detect_answer_language(user_question: str) -> str:
    """Detect whether the answer should be Greek or English."""
    greek_chars = sum(1 for char in user_question if "\u0370" <= char <= "\u03FF")

    if greek_chars > 0:
        return "Greek"

    return "English"

def build_prompt(user_question: str, context: str) -> list[dict]:
    """Build chat-style prompt for the instruction-tuned model."""
    answer_language = detect_answer_language(user_question)

    system_message = f"""
You are a careful assistant answering questions about municipality ERP documentation.

You MUST answer in {answer_language}.

Use only the retrieved documentation context.

Important rules:
- Give the final answer only.
- If the context is insufficient, say so clearly.
""".strip()

    user_message = f"""
Question:
{user_question}

Retrieved documentation context:
{context}

Write only the final answer in {answer_language}.
Use 3 to 6 bullet points.
Do not include retrieval metadata.
""".strip()

    return [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message},
    ]

@lru_cache(maxsize=1)
def load_llm():
    """Load tokenizer and local Hugging Face language model."""
    print(f"Loading LLM: {MODEL_NAME}")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        token=HF_TOKEN,
        trust_remote_code=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        token=HF_TOKEN,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True,
    )

    return tokenizer, model


def generate_answer(
    user_question: str,
    vector_top_k: int = 4,
    section_limit: int = 2,
    keyword_limit: int = 3,
    max_new_tokens: int = 500,
) -> dict:
    """
    Full GraphRAG pipeline:

    question
    -> vector retrieval
    -> graph expansion
    -> context formatting
    -> LLM generation
    """
    retrieval = retrieve_graphrag_context(
        query=user_question,
        vector_top_k=vector_top_k,
        section_limit=section_limit,
        keyword_limit=keyword_limit,
    )

    context = build_full_context(
        vector_results=retrieval["vector_results"],
        graph_entry_points=retrieval["graph_entry_points"],
        graph_results=retrieval["graph_results"],
    )

    messages = build_prompt(user_question, context)

    tokenizer, model = load_llm()

    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated_ids = output_ids[0][inputs["input_ids"].shape[-1]:]
    answer = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    return {
        "question": user_question,
        "answer": answer,
        "context": context,
        "vector_results": retrieval["vector_results"],
        "graph_entry_points": retrieval["graph_entry_points"],
        "graph_results": retrieval["graph_results"],
    }


def print_answer(result: dict) -> None:
    """Print final answer and compact source trace."""
    print("\n" + "=" * 90)
    print("ANSWER")
    print("=" * 90)
    print(result["answer"])

    print("\n" + "=" * 90)
    print("VECTOR SOURCES")
    print("=" * 90)

    for record in result["vector_results"]:
        print(
            f"- {record.get('id')} | "
            f"rank={record.get('rank')} | "
            f"score={record.get('score'):.4f} | "
            f"section={record.get('section_title')} | "
            f"chunk={record.get('chunk_index')}"
        )

    print("\n" + "=" * 90)
    print("GRAPH ENTRY POINTS")
    print("=" * 90)

    for record in result["graph_entry_points"]:
        print(
            f"- {record.get('id')} | "
            f"section={record.get('section_name') or record.get('section_title')} | "
            f"chunk={record.get('chunk_index')}"
        )

    print("\n" + "=" * 90)
    print("GRAPH-EXPANDED SOURCES")
    print("=" * 90)

    for record in result["graph_results"][:10]:
        print(
            f"- {record.get('id')} | "
            f"reason={record.get('graph_reason')} | "
            f"seed={record.get('seed_id')} | "
            f"section={record.get('section_title')} | "
            f"chunk={record.get('chunk_index')}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--question",
        type=str,
        required=True,
        help="Question to answer using GraphRAG.",
    )
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--section-limit", type=int, default=2)
    parser.add_argument("--keyword-limit", type=int, default=3)
    parser.add_argument("--max-new-tokens", type=int, default=500)

    args = parser.parse_args()

    result = generate_answer(
        user_question=args.question,
        vector_top_k=args.top_k,
        section_limit=args.section_limit,
        keyword_limit=args.keyword_limit,
        max_new_tokens=args.max_new_tokens,
    )

    print_answer(result)


if __name__ == "__main__":
    main()