# Mini-RAG — Sherlock Holmes Question Answering

## Overview

This project implements a small end-to-end Retrieval-Augmented Generation (RAG) pipeline over five Sherlock Holmes stories.

The pipeline:

1. Loads the stories from `data/corpus/`
2. Splits the stories into smaller chunks
3. Generates embeddings using OpenAI `text-embedding-3-small`
4. Stores the embeddings and chunk metadata in Qdrant
5. Retrieves the most relevant chunks for a user question
6. Generates an answer using OpenAI `gpt-4o-mini`
7. Displays the source story and section used for the answer

The project is implemented directly with the `openai` and `qdrant-client` libraries. It does not use LangChain or LlamaIndex.

---

## Project Structure

```text
.
├── mp2_rag.py
├── settings.py
├── .env
├── data/
│   ├── corpus/
│   │   ├── 01_red_headed_league.txt
│   │   ├── 02_speckled_band.txt
│   │   ├── 03_blue_carbuncle.txt
│   │   ├── 04_engineers_thumb.txt
│   │   └── 05_scandal_in_bohemia.txt
│   ├── predefined_questions.jsonl
│   └── learner_questions.jsonl
└── README.md
```

## Prerequisites

- Python 3.x
- An OpenAI API key
- Qdrant running and accessible at the URL configured in `.env`
- Project dependencies installed

The project uses OpenAI `text-embedding-3-small` for embeddings, OpenAI `gpt-4o-mini` for answer generation, and Qdrant for vector storage and retrieval.

## 1. Set Up the Python Environment


## 2. Configure Environment Variables

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=your_openai_api_key
OPENAI_BASE_URL=https://openai.vocareum.com/v1
QDRANT_URL=http://localhost:6333
```

Do not commit `.env` or expose the API key in GitHub.

## 3. Start Qdrant

The project expects Qdrant at:

```text
http://localhost:6333
```

If Qdrant is running through Docker Desktop, start the Qdrant container before ingestion.

## 4. Ingest the Corpus

Run:

```bash
python mp2_rag.py ingest
```

This performs:

```text
Corpus
  ↓
Load 5 stories
  ↓
Chunk documents
  ↓
Generate embeddings
  ↓
Create Qdrant collection
  ↓
Store vectors + metadata
```

A successful run should show output similar to:

```text
→ Loading corpus…
  5 documents loaded

→ Chunking…
  01_red_headed_league.txt: 12 chunks
  02_speckled_band.txt: 12 chunks
  03_blue_carbuncle.txt: 12 chunks
  04_engineers_thumb.txt: 15 chunks
  05_scandal_in_bohemia.txt: 18 chunks

→ Total chunks: 69
→ Setting up Qdrant collection…
→ Ingesting…

✓ Done. Try: python mp2_rag.py ask
```

The exact chunk count can change if the chunking logic is modified.

**Important:** the current `ingest` implementation recreates the Qdrant collection before loading the corpus. Run it again if the corpus or chunking logic changes.

## 5. Ask Questions Interactively

Run:

```bash
python mp2_rag.py ask
```

Enter a question at the `?` prompt. For example:

```text
What happened to Helen Stoner?
```

The system embeds the question, retrieves the top 3 relevant chunks, passes those excerpts to `gpt-4o-mini`, and displays the answer together with the retrieved story/section and latency.

Example output structure:

```text
<generated answer>

  Sources:
    - The Adventure of the Speckled Band — Stoke Moran and Doctor Roylott
    - The Adventure of the Speckled Band — Stoke Moran and Doctor Roylott
    - The Adventure of the Speckled Band — The Death of Julia Stoner

  Latency: <value>ms
```

## 6. Run the Validation

The project validates two predefined questions and three learner-created questions.

Run:

```bash
python mp2_rag.py validate
```

Validation uses:

```text
data/predefined_questions.jsonl
data/learner_questions.jsonl
```

The validation reports:

- Expected-source match
- Literal expected-fact matches
- Latency

# Understanding the Validation Output

A typical result looks like:

```text
✓ q1_red_headed_league_assistant
    Q: ...
    Cited: 01_red_headed_league.txt
    Expected: 01_red_headed_league.txt
    Facts matched: 1/5
    Latency: 3463ms

Source-match: 2/2
```

### `✓` / `✗`

This indicates whether the expected source story was found in the citations.

- `✓` = expected source was cited
- `✗` = expected source was not cited

For this project, source matching is the primary validation signal.

### `Cited`

The source files associated with the retrieved chunks.

### `Expected`

The source story specified by the validation question.

### `Facts matched`

For example:

```text
Facts matched: 3/6
```

This is **not a complete semantic correctness score**. The validation code performs a literal, case-insensitive substring check to see whether each expected fact appears in the generated answer.

Consequently, a fact can be present semantically but receive no match if the model uses different wording.

### `Latency`

For example:

```text
Latency: 2672ms
```

This is the approximate time taken for the retrieval and answer-generation operation. It can vary between runs because the pipeline makes API calls.

# What Does `Source-match` Mean?

A result such as:

```text
Source-match: 5/5
```

means that all five validation questions cited the expected source story.

The project brief asks that the five questions pass the source-match check, or come close.

The completed validation for this implementation produced:

```text
Predefined questions: 2/2
Learner questions:    3/3
Overall:              5/5
```

This indicates that the retriever successfully identified the expected story for all five validation questions.

# Interpreting RAG Quality

A successful source match does not automatically mean that the final answer contains every required fact.

### Retrieval quality

Ask:

> Did the system retrieve the correct story and relevant chunks?

`Source-match` is the main signal used by the provided validation harness.

### Answer quality

Ask:

> Did the retrieved chunks contain enough information for the LLM to answer the question completely and accurately?

The LLM is instructed to answer only from the retrieved excerpts. If the required information is not present in those excerpts, it should say that the excerpts do not contain enough information rather than inventing an answer.

Therefore:

```text
Correct source + incomplete chunks
        ↓
Potentially incomplete answer
```

This is an important RAG observation from the project.

# Troubleshooting

## Qdrant connection error

Check that Qdrant is running and that `.env` contains:

```env
QDRANT_URL=http://localhost:6333
```

If Qdrant is running in Docker Desktop, verify that the container is up and the port is exposed.

## OpenAI authentication error

Check:

```env
OPENAI_API_KEY=your_openai_api_key
```

Also verify that the configured `OPENAI_BASE_URL` is correct for the environment being used.

## The answer cites the wrong story

Run ingestion again:

```bash
python mp2_rag.py ingest
```

Then retry:

```bash
python mp2_rag.py ask
```

If the problem continues, inspect the chunks produced by the chunking function. Poor chunk boundaries or insufficient context can affect retrieval quality.

## The answer is incomplete

The system currently retrieves the top 3 chunks for a question.

An incomplete answer can occur when the retrieved chunks do not contain all the information needed to answer the question.

Areas for future investigation include:

- Chunk size
- Chunk overlap
- Section boundaries
- Number of retrieved chunks (`k`)
- Retrieval strategy

## Running the program without a command

Use one of the supported commands:

```bash
python mp2_rag.py ingest
python mp2_rag.py ask
python mp2_rag.py validate
```

| Command | Purpose |
|---|---|
| `ingest` | Load, chunk, embed, and store the corpus in Qdrant |
| `ask` | Interactively ask questions |
| `validate` | Run the predefined and learner validation questions |

# Recommended Run Sequence

For a fresh setup:

```bash
# Configure .env
# Start Qdrant

python mp2_rag.py ingest
python mp2_rag.py ask
python mp2_rag.py validate
```


## Key Takeaway

The Mini-RAG loop is:

```text
Documents
   ↓
Chunking
   ↓
Embeddings
   ↓
Qdrant
   ↓
Query embedding
   ↓
Top-k retrieval
   ↓
Relevant excerpts
   ↓
gpt-4o-mini
   ↓
Answer + source citation
```

The important result is not simply whether the LLM produces an answer. The validation also checks whether the system retrieves and cites the expected source. The completed validation for this project achieved a `5/5` source match across the two predefined and three learner questions.
