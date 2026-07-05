import sys
from pathlib import Path

import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.rag_pipeline import generate_answer
from src.graph_retrieval import retrieve_graphrag_context


st.set_page_config(
    page_title="CivicProcess GraphRAG Assistant",
    page_icon="🏛️",
    layout="wide",
)

st.title("🏛️ CivicProcess GraphRAG Assistant")

st.write(
    """
This prototype answers questions over private municipality ERP documentation.

The architecture follows a Graph-Enhanced Vector RAG pattern:

1. Vector search retrieves semantically relevant records.
2. Retrieved record IDs become Neo4j graph entry points.
3. Neo4j expands to related records through graph relationships.
4. The LLM receives vector matches, graph entry points, and graph-expanded neighbors.
"""
)

st.warning(
    """
Prototype note: retrieval and graph expansion are the main focus of this project.
The local generation model is small, so final answers may be imperfect.
Always inspect the retrieved evidence below the answer.
"""
)

question = st.text_area(
    "Ask a question:",
    height=100,
)

col1, col2, col3, col4 = st.columns(4)

with col1:
    top_k = st.slider(
        "Vector matches",
        min_value=1,
        max_value=10,
        value=3,
    )

with col2:
    section_limit = st.slider(
        "Same-section graph neighbors",
        min_value=0,
        max_value=5,
        value=0,
    )

with col3:
    keyword_limit = st.slider(
        "Keyword graph neighbors",
        min_value=0,
        max_value=10,
        value=1,
    )

with col4:
    max_new_tokens = st.slider(
        "Max new tokens",
        min_value=100,
        max_value=800,
        value=350,
        step=50,
    )


if st.button("Generate GraphRAG Answer"):
    if not question.strip():
        st.warning("Please enter a question.")

    else:
        with st.spinner("Running GraphRAG retrieval and answer generation..."):
            result = generate_answer(
                user_question=question,
                vector_top_k=top_k,
                section_limit=section_limit,
                keyword_limit=keyword_limit,
                max_new_tokens=max_new_tokens,
            )

        st.success("Done.")

        st.header("Generated Answer")
        st.write(result["answer"])

        st.header("Direct Vector Matches")

        for record in result["vector_results"]:
            with st.expander(
                f"{record.get('id')} | "
                f"score={record.get('score'):.4f} | "
                f"section={record.get('section_title')} | "
                f"chunk={record.get('chunk_index')}"
            ):
                st.write("**Question stored in record:**")
                st.write(record.get("question", ""))

                st.write("**Answer / content:**")
                st.write(record.get("answer", ""))

        st.header("Neo4j Graph Entry Points")

        for record in result["graph_entry_points"]:
            with st.expander(
                f"{record.get('id')} | "
                f"section={record.get('section_name') or record.get('section_title')} | "
                f"chunk={record.get('chunk_index')}"
            ):
                st.write("**Keywords:**")
                st.write(", ".join(record.get("keywords", [])) or "None")

                st.write("**Question:**")
                st.write(record.get("question_text") or record.get("question", ""))

                st.write("**Answer / content:**")
                st.write(record.get("answer_text") or record.get("answer", ""))

        st.header("Graph-Expanded Neighbor Records")

        if not result["graph_results"]:
            st.info("No graph-expanded neighbors found.")

        for record in result["graph_results"]:
            with st.expander(
                f"{record.get('id')} | "
                f"reason={record.get('graph_reason')} | "
                f"seed={record.get('seed_id')} | "
                f"section={record.get('section_title')} | "
                f"chunk={record.get('chunk_index')}"
            ):
                st.write("**Shared keywords:**")
                st.write(", ".join(record.get("shared_keywords", [])) or "None")

                st.write("**Question:**")
                st.write(record.get("question", ""))

                st.write("**Answer / content:**")
                st.write(record.get("answer", ""))


st.divider()

st.subheader("Retrieval-only quick test")

quick_query = st.text_input(
    "Search the vector index without running the LLM:",
)

if st.button("Run Retrieval Test"):
    retrieval = retrieve_graphrag_context(
        query=quick_query,
        vector_top_k=top_k,
        section_limit=section_limit,
        keyword_limit=keyword_limit,
    )

    st.write("### Vector Results")
    for record in retrieval["vector_results"]:
        st.write(
            f"- **{record.get('id')}** | "
            f"score={record.get('score'):.4f} | "
            f"section={record.get('section_title')} | "
            f"chunk={record.get('chunk_index')}"
        )

    st.write("### Graph Entry Points")
    for record in retrieval["graph_entry_points"]:
        st.write(
            f"- **{record.get('id')}** | "
            f"section={record.get('section_name') or record.get('section_title')} | "
            f"chunk={record.get('chunk_index')}"
        )

    st.write("### Graph-Expanded Neighbors")
    for record in retrieval["graph_results"]:
        st.write(
            f"- **{record.get('id')}** | "
            f"reason={record.get('graph_reason')} | "
            f"seed={record.get('seed_id')} | "
            f"section={record.get('section_title')} | "
            f"chunk={record.get('chunk_index')}"
        )
