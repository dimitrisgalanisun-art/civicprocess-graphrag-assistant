# CivicProcess GraphRAG Assistant

A Graph-Enhanced Vector RAG prototype for answering questions over private municipality ERP documentation.

The project demonstrates how semantic vector search can be combined with a Neo4j graph layer and a local language model. The main goal is to make the retrieval path visible and inspectable: the app shows direct vector matches, Neo4j graph entry points, and graph-expanded neighbor records.

---

## Project Goal

Municipality ERP documentation often contains long technical descriptions of workflows, database tables, statuses, stored procedures, UI flows, and business rules.

A simple chatbot over this kind of documentation can easily fail because:

- the documentation is long,
- related concepts are spread across different sections,
- table names and process names are technical,
- the final answer must be grounded in the retrieved evidence.

This project implements a GraphRAG-style assistant that retrieves relevant documentation chunks and then expands from those chunks through a Neo4j graph.

---

## Architecture

The high-level flow is:

```text
Private Markdown documentation
→ chunked JSONL records
→ multilingual vector embeddings
→ local vector search
→ Neo4j graph upload
→ vector result IDs become graph entry points
→ graph expansion through shared keywords
→ local LLM answer generation
→ Streamlit demo UI
```

The runtime question-answering flow is:

```text
User question
→ vector search over documentation chunks
→ retrieve top record IDs
→ find the same IDs in Neo4j
→ expand to related graph records
→ build context
→ generate answer with local LLM
→ show answer and evidence trace
```

The important idea is that the vector database and Neo4j graph share the same stable record IDs.

Example:

```text
Vector result:
municipality_record_0045

Neo4j graph node:
(:CivicRAGRecord {id: "municipality_record_0045"})
```

This makes the vector layer a semantic entry point and the Neo4j layer a relational expansion layer.

---

## Main Components

### `src/build_dataset.py`

Builds private JSONL records from the private Markdown documentation.

It performs:

- Markdown section parsing,
- text cleaning,
- chunking,
- content-based question generation,
- technical keyword extraction,
- embedding-text construction.

The current chunking settings are:

```python
CHUNK_SIZE = 600
CHUNK_OVERLAP = 100
```

Smaller chunks improved retrieval precision because evaluation, ranking, scoring, and document-checking content are less likely to be mixed into the same record.

---

### `src/vector_index.py`

Builds and searches the local vector index.

It uses:

```text
sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
```

This multilingual embedding model is used because the source documentation and many demo questions are in Greek.

The vector index retrieves the most semantically similar records for a user question.

---

### `src/graph_upload.py`

Uploads the private JSONL records into Neo4j.

The graph contains nodes such as:

```text
CivicRAGRecord
CivicRAGQuestion
CivicRAGAnswer
CivicRAGSource
CivicRAGSection
CivicRAGKeyword
```

Records are connected to their source, section, question, answer, and technical keywords.

Technical keywords are extracted from generated `technical_terms`, backticked identifiers, and snake_case database identifiers.

---

### `src/graph_retrieval.py`

Implements the GraphRAG retrieval flow.

It does three things:

1. Runs vector search.
2. Uses the vector result IDs as Neo4j graph entry points.
3. Expands from those graph records to related records.

There are two graph expansion modes:

```text
Same-section graph neighbors
```

Finds records from the same documentation section.

```text
Keyword graph neighbors
```

Finds records that share technical keyword nodes, such as table names, field names, or process identifiers.

For this dataset, keyword expansion is usually more precise than same-section expansion, because some sections are broad.

---

### `src/rag_pipeline.py`

Combines retrieval with local LLM answer generation.

The current local model is:

```text
Qwen/Qwen2.5-0.5B-Instruct
```

This is intentionally small and lightweight, but it has limitations. The retrieval layer is stronger than the generation layer.

---

### `src/app.py`

Streamlit demo app.

The app displays:

- generated answer,
- direct vector matches,
- Neo4j graph entry points,
- graph-expanded neighbor records.

This makes the retrieval path inspectable instead of hiding everything behind the final generated answer.

---

## Privacy

The repository is designed so private data is not committed.

The public repository contains code and safe examples only.

Private/local files are ignored, including:

```text
.env
data_source/private/
data_source/*.md
data_source/*.pdf
data_source/*.docx
data_source/*.jsonl
vector_db/
evaluation/results/
```

The private documentation, generated JSONL records, vector embeddings, Neo4j credentials, and evaluation result outputs remain local.

---

## Setup

Create and activate your Python environment, then install dependencies:

```powershell
python -m pip install -r requirements.txt
```

Create a local `.env` file from the example:

```powershell
copy .env.example .env
```

Fill in your local values:

```env
HF_TOKEN=your_huggingface_token_here
NEO4J_URI=your_neo4j_uri_here
NEO4J_USERNAME=your_neo4j_username_here
NEO4J_PASSWORD=your_neo4j_password_here
NEO4J_DATABASE=your_neo4j_database_here
MODEL_NAME=Qwen/Qwen2.5-0.5B-Instruct
EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
```

---

## Private Data Location

Place the private Markdown documentation here:

```text
data_source/private/municipality_applications.md
```

This file is ignored by Git.

---

## Build the Dataset

Generate private JSONL records:

```powershell
python -m src.build_dataset
```

This creates:

```text
data_source/private/municipality_qa.jsonl
```

The JSONL file is ignored by Git.

---

## Build the Vector Index

Build local embeddings:

```powershell
python -m src.vector_index --build
```

This creates local vector files under:

```text
vector_db/
```

The vector database is ignored by Git.

Test vector search:

```powershell
python -m src.vector_index --search "Τι είναι ο πίνακας κατάταξης;" --top-k 5
```

---

## Upload to Neo4j

Upload the private records to Neo4j:

```powershell
python -m src.graph_upload --reset --upload --summary
```

This creates the GraphRAG graph structure in Neo4j.

The `--reset` flag deletes the existing graph contents in the configured Neo4j database before uploading.

Use it only when the database is dedicated to this project.

---

## Test GraphRAG Retrieval

Run:

```powershell
python -m src.graph_retrieval --query "Πώς γίνεται η αξιολόγηση αίτησης;" --top-k 5
```

Another useful test:

```powershell
python -m src.graph_retrieval --query "Τι είναι ο πίνακας κατάταξης;" --top-k 5
```

Expected output sections:

```text
VECTOR RESULTS
GRAPH ENTRY POINTS
GRAPH-EXPANDED RESULTS
```

The vector results and graph entry points should share the same record IDs.

---

## Run the Streamlit App

Run:

```powershell
streamlit run src/app.py
```

Recommended demo settings:

```text
Question: Τι είναι ο πίνακας κατάταξης;
Vector matches: 3
Same-section graph neighbors: 0
Keyword graph neighbors: 1
Max new tokens: 220
```

Alternative demo question:

```text
Πώς γίνεται η αξιολόγηση αίτησης;
```

Recommended settings:

```text
Vector matches: 3
Same-section graph neighbors: 0
Keyword graph neighbors: 1
Max new tokens: 220
```

Same-section expansion is set to `0` for the demo because some document sections are broad and can introduce noisy context. Keyword graph expansion is more focused because it follows technical identifiers.

---

## Evaluation

The project includes a small retrieval evaluation harness.

Evaluation files:

```text
evaluation/evaluation_questions.example.jsonl
evaluation/evaluate_retrieval.py
```

Run:

```powershell
python evaluation/evaluate_retrieval.py --top-k 3 --section-limit 0 --keyword-limit 1
```

The evaluation checks whether expected keywords appear in the retrieved vector, graph-entry, and graph-expanded context.

The current evaluation set contains five safe public example questions:

```text
Πώς γίνεται η αξιολόγηση αίτησης;
Τι είναι ο πίνακας κατάταξης;
Πώς υπολογίζεται η βαθμολογία μιας αίτησης;
Πώς ελέγχονται τα δικαιολογητικά μιας αίτησης;
```

With the current demo settings, the retrieval keyword score is:

```text
Average retrieval keyword score: 0.600
```

The evaluation also confirms that:

```text
vector result IDs match Neo4j graph entry IDs
keyword graph expansion returns related neighbor records
```

Generated evaluation results are saved locally under:

```text
evaluation/results/
```

This folder is ignored by Git.

---

## Demo Explanation

This is not just a simple vector chatbot.

A simple vector chatbot does:

```text
question
→ vector search
→ LLM answer
```

This project does:

```text
question
→ vector search
→ stable record IDs
→ Neo4j graph entry points
→ graph expansion
→ LLM answer
→ evidence trace
```

The app shows the retrieval evidence, so the user can inspect whether the answer is grounded in relevant documentation.

---

## Known Limitations

The main limitation is generation quality.

The local model:

```text
Qwen/Qwen2.5-0.5B-Instruct
```

is small. It can generate answers, but it may sometimes:

- copy context too literally,
- miss important details,
- fail to synthesize broad processes,
- be distracted by noisy graph-expanded context.

The retrieval architecture is stronger than the final generation layer.

This is why the app exposes the retrieved evidence below the answer.

---

## Future Improvements

Possible improvements:

- use a stronger local model, such as a 2B+ instruct model,
- add a reranker before LLM generation,
- improve graph keyword extraction,
- improve graph expansion scoring,
- add a larger manual evaluation set,
- evaluate faithfulness of generated answers,
- add better UI filtering for vector results and graph neighbors,
- support more document types beyond Markdown.

---

## Project Status

Implemented:

```text
Private dataset building
Content-based chunk records
Multilingual vector search
Neo4j graph upload
Graph entry point retrieval
Keyword-based graph expansion
Streamlit demo app
Retrieval evaluation harness
Privacy-safe GitHub structure
```

Current status:

```text
GraphRAG retrieval pipeline works end to end.
Final answer quality is limited by the small local LLM.
```
## Example Evaluation Question Analysis

This section explains what happens internally when the evaluation script runs one example question.

Example evaluation question:

```text
How is an application connected to the ranking table?
```

The system does not send this question directly to the language model. First, it passes through the retrieval pipeline.

The question is converted into an embedding. This means that the text of the question becomes a numerical vector that represents its semantic meaning. The same process has already been applied to the documentation records created from the private Markdown file. Because of this, the system can compare the question embedding with the embeddings of the stored documentation chunks.

Next, vector search is performed. The system looks for the records whose embeddings are closest to the question embedding. Since the question asks about the relationship between an application and a ranking table, the most relevant vector results are expected to be records that mention concepts such as:

```text
ranking table
application
ranking position
ranking result
application result
```

These direct vector matches are the primary retrieval results. They are important because they come from semantic similarity between the user question and the documentation chunks.

After vector search returns the top records, their stable record IDs are used as entry points into Neo4j. This is the key GraphRAG step. The graph is not searched independently from the beginning. Instead, the system takes the record IDs found by vector search and looks for the corresponding `CivicRAGRecord` nodes in Neo4j.

For example:

```text
Vector result:
municipality_record_0286

Neo4j graph node:
(:CivicRAGRecord {id: "municipality_record_0286"})
```

This connects the semantic retrieval layer with the graph retrieval layer. The vector index finds the most relevant documentation records, and Neo4j confirms that the same records exist as graph nodes.

The evaluation script then reports these as Graph Entry Points. These should match the vector result IDs. If the vector result IDs and graph entry IDs match, this confirms that the vector database and the Neo4j graph are connected correctly through shared stable IDs.

After that, graph expansion is performed. Starting from the graph entry point records, Neo4j looks for additional records connected through shared keyword nodes. For this question, useful shared keywords can include generic concepts such as:

```text
ranking table
application
ranking position
ranking result
application identifier
```

The graph-expanded records are not the main retrieval result. They are supporting context. Their role is to add related technical information that may not have appeared in the top vector matches but is connected through the graph structure.

The evaluation script then builds one combined retrieved context from three retrieval layers:

```text
1. Direct Vector Matches
2. Neo4j Graph Entry Points
3. Graph-Expanded Neighbor Records
```

The Direct Vector Matches are the main evidence. The Graph Entry Points confirm that the vector results are also present in Neo4j. The Graph-Expanded Neighbor Records provide additional related context through graph relationships.

Then the evaluation script checks whether the expected keywords for the question appear in the retrieved context. For this example, the expected public keywords are:

```text
application
ranking table
position
result
ranking
```

If three of these five keywords appear in the retrieved context, the score for this question is:

```text
3 / 5 = 0.600
```

This score does not evaluate the final generated answer. It evaluates retrieval quality. More specifically, it checks whether the retrieval pipeline found context containing the important domain concepts expected for that question.

For this example, the evaluation output should show three important things:

```text
Vector IDs are returned.
Graph entry IDs match the vector IDs.
Graph neighbor IDs are returned through keyword-based graph expansion.
```

This means the full GraphRAG retrieval path is working:

```text
Evaluation question
→ question embedding
→ vector search over documentation records
→ top record IDs
→ Neo4j graph entry points using the same IDs
→ keyword-based graph expansion
→ combined retrieved context
→ keyword-based retrieval evaluation score
```

The most important point is that the evaluation does not only check whether the LLM gives a good final answer. It checks whether the retrieval system finds the right evidence before generation. This is useful because the local LLM is small and may sometimes produce an imperfect answer, even when the retrieval results are relevant.

Therefore, this evaluation helps separate two different questions:

```text
Did the retrieval system find relevant documentation?
Did the language model generate a good final answer from that documentation?
```

In this project, the evaluation focuses on the first question: whether the GraphRAG retrieval pipeline successfully finds relevant vector matches, maps them into Neo4j, expands to related graph neighbors, and retrieves context containing the expected domain keywords.