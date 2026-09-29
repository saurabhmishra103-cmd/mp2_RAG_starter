from __future__ import annotations

import json
import sys
import time
import uuid
from pathlib import Path
from typing import Any
from venv import logger
from settings import my_settings, CORPUS_DIR, DATA_DIR, COLLECTION_NAME, EMBEDDING_MODEL, EMBEDDING_DIM, CHAT_MODEL, TARGET_CHUNK_SIZE, CHUNK_OVERLAP

from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams



openai = OpenAI(
     api_key=my_settings.OPENAI_API_KEY,
     base_url=my_settings.OPENAI_BASE_URL
)
qdrant = QdrantClient(
    url=my_settings.QDRANT_URL,
)


# ─── Step 1: Load the corpus ────────────────────────────────────────────

def load_corpus(corpus_dir: Path) -> list[dict[str, Any]]:
    """Read every .txt file in the corpus directory.

    Returns a list of dicts, each with: source (filename), title (first line),
    and text (full content).
  """
    
    logger.info(f"Loading text file: {corpus_dir}")
    if not corpus_dir.exists():
        logger.error(f"Corpus directory {corpus_dir} does not exist.")

    documents = []
    for file_path in corpus_dir.glob("*.txt"):
        logger.info(f"Reading file: {file_path}")
        try:
            with open(file_path, "r", encoding = "utf-8") as f:
                text = f.read()
                title = next(
                    line.strip()
                    for line in text.splitlines()
                    if line.strip()
                    )
                documents.append({
                    "source": file_path.name,
                    "title": title,
                    "text": text,
                })
        except Exception as e:
                    logger.error(f"Error reading file {file_path}: {e}")
                    continue
    return documents
        

# ─── Step 2: Chunk each document ────────────────────────────────────────

def chunk_document(doc: dict[str, Any]) -> list[dict[str, Any]]:
    """Split a document into smaller chunks.

    Each chunk should be a dict with: source, title, section, text.
    """

    try:
        logger.info(f"Chunking document: {doc['source']} with Paragraph-based approach. Chunk size: {TARGET_CHUNK_SIZE}, Overlap: {CHUNK_OVERLAP}")

        text = doc["text"]

        # Split document into paragraphs
        paragraphs = [
            p.strip()
            for p in text.split("\n\n")
            if p.strip()
        ]

        chunks = []
        current_paragraphs = []
        current_length = 0
        current_section = doc["title"]

        for paragraph in paragraphs:

            # Check if this paragraph looks like a section heading
            lines = paragraph.splitlines()

            if (
                len(lines) == 1
                and len(paragraph) < 100
                and not paragraph.endswith((".", "?", "!"))
            ):
                current_section = paragraph
                continue

            paragraph_length = len(paragraph)

            # If adding this paragraph would make the chunk too large,
            # save the current chunk first.
            if (
                current_paragraphs
                and current_length + paragraph_length > TARGET_CHUNK_SIZE
            ):
                chunk_text = "\n\n".join(current_paragraphs)

                chunks.append({
                    "source": doc["source"],
                    "title": doc["title"],
                    "section": current_section,
                    "text": chunk_text,
                })

                current_paragraphs = []
                current_length = 0

            current_paragraphs.append(paragraph)
            current_length += paragraph_length

        # Add the final chunk
        if current_paragraphs:
            chunk_text = "\n\n".join(current_paragraphs)

            chunks.append({
                "source": doc["source"],
                "title": doc["title"],
                "section": current_section,
                "text": chunk_text,
            })

        return chunks

    except Exception as e:
        logger.error(f"Error chunking document:{e}")
        raise
         

# ─── Step 3: Embed text ─────────────────────────────────────────────────

def embed_texts(texts: list[str]) -> list[list[float]]:
    """Batch-embed a list of texts using OpenAI's embedding model.

    Returns a list of 1536-dim float vectors (same order as inputs).
    """
    try:
        logger.info(f"Embedding {len(texts)} texts using model: {EMBEDDING_MODEL}")
        if not texts:
            return []

        response = openai.embeddings.create(
            model=EMBEDDING_MODEL,
            input=texts,
        )

        embeddings = [item.embedding for item in response.data]

        return embeddings
    except Exception as e:
        logger.error(f"Error generating embeddings: {e}")
        raise


# ─── Step 4: Set up the Qdrant collection ───────────────────────────────
def setup_collection() -> None:
    """Create (or recreate) the Qdrant collection."""

    try:
        qdrant.recreate_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(
                size=EMBEDDING_DIM,
                distance=Distance.COSINE,
            ),
        )

    except Exception as e:
        logger.error(f"Error setting up collection: {e}")
        raise


# ─── Step 5: Ingest chunks into Qdrant ──────────────────────────────────

def ingest_chunks(chunks: list[dict[str, Any]]) -> None:
    """Embed every chunk and upsert into Qdrant."""

    try:
        if not chunks:
            logger.warning("No chunks provided for ingestion.")
            return

        texts = [chunk["text"] for chunk in chunks]

        embeddings = embed_texts(texts)

        points = []

        for chunk, embedding in zip(chunks, embeddings):
            point = PointStruct(
                id=str(uuid.uuid4()),
                vector=embedding,
                payload=chunk,
            )

            points.append(point)

        qdrant.upsert(
            collection_name=COLLECTION_NAME,
            points=points,
        )

        logger.info(
            f"Successfully ingested {len(points)} chunks "
            f"into Qdrant collection '{COLLECTION_NAME}'."
        )

    except Exception as e:
        logger.error(f"Error ingesting chunks into Qdrant: {e}")
        raise

# ─── Step 6: Retrieve ───────────────────────────────────────────────────

def retrieve(query: str, k: int = 3) -> list[dict[str, Any]]:
    """Retrieve top-k chunks for a query.
    TODO:
      - Embed the query
      - qdrant.search with the query vector, limit=k
      - Return list of chunk dicts (include score for citations)
    """
    try:
        if not query.strip():
            logger.warning("Empty query provided for retrieval.")
            return []

        query_embedding = embed_texts([query])[0]

        search_results = qdrant.query_points(
            collection_name=COLLECTION_NAME,
            query=query_embedding,
            limit=k,
        ).points

        retrieved_chunks = []

        for result in search_results:
            chunk = result.payload.copy()
            chunk["score"] = result.score
            retrieved_chunks.append(chunk)

        return retrieved_chunks

    except Exception as e:
        logger.error(f"Error retrieving chunks from Qdrant: {e}")
        raise


# ─── Step 7: Generate the answer ────────────────────────────────────────

SYSTEM_PROMPT = """You are a helpful assistant answering questions about a small
collection of Sherlock Holmes stories. You will be given the user's question and
several relevant excerpts. Use ONLY the provided excerpts to answer. If the
excerpts don't contain the answer, say so plainly. Cite the source (story title
+ section) in your answer."""


def answer(question: str, k: int = 3) -> dict[str, Any]:
    """End-to-end: retrieve, format context, call LLM, return result.

    TODO:
      - Call retrieve(question, k=k)
      - Format the retrieved chunks into a context string
        (include "[Source: <title> — <section>]" before each)
      - Call openai.chat.completions.create with SYSTEM_PROMPT and the user message
      - Return dict with: question, answer, citations, latency_ms
    """
    try:
        start_time = time.time()

        retrieved_chunks = retrieve(question, k=k)

        if not retrieved_chunks:
            return {
                "question": question,
                "answer": "I could not find any relevant excerpts in the corpus.",
                "citations": [],
                "latency_ms": round((time.time() - start_time) * 1000),
            }

        context_parts = []

        for chunk in retrieved_chunks:
            context_parts.append(
                f"[Source: {chunk['title']} — {chunk['section']}]\n"
                f"{chunk['text']}"
            )

        context = "\n\n".join(context_parts)

        user_message = f"""Question: {question}

                        Relevant excerpts: {context}
                        """

        response = openai.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {   "role": "system", "content": SYSTEM_PROMPT},
                {   "role": "user", "content": user_message},
            ],
        )

        answer_text = response.choices[0].message.content

        citations = [
            {
                "source": chunk["source"],
                "title": chunk["title"],
                "section": chunk["section"],
                "score": chunk["score"],
            }
            for chunk in retrieved_chunks
        ]

        latency_ms = round((time.time() - start_time) * 1000)

        return {
            "question": question,
            "answer": answer_text,
            "citations": citations,
            "latency_ms": latency_ms,
        }

    except Exception as e:
        logger.error(f"Error generating answer: {e}")
        raise


# ─── Validation harness (provided — do not modify) ──────────────────────

def validate_against(jsonl_path: Path) -> None:
    questions = [json.loads(line) for line in jsonl_path.read_text().splitlines() if line.strip()]
    print(f"\n  Validating {len(questions)} questions from {jsonl_path.name}…\n")

    hits = 0
    for q in questions:
        result = answer(q["question"], k=3)
        cited_sources = {cit["source"] for cit in result["citations"]}
        source_hit = q["expected_source"] in cited_sources

        ans_lower = result["answer"].lower()
        facts_hit = sum(1 for fact in q.get("expected_facts", []) if fact.lower() in ans_lower)
        facts_total = len(q.get("expected_facts", []))

        verdict = "✓" if source_hit else "✗"
        print(f"  {verdict} {q['id']}")
        print(f"      Q: {q['question']}")
        print(f"      Cited: {', '.join(cited_sources)}")
        print(f"      Expected: {q['expected_source']}")
        print(f"      Facts matched: {facts_hit}/{facts_total}")
        print(f"      Latency: {result.get('latency_ms', '?')}ms")
        print()
        if source_hit:
            hits += 1

    print(f"  Source-match: {hits}/{len(questions)}")


# ─── CLI (provided — do not modify) ─────────────────────────────────────

def cmd_ingest() -> None:
    print("→ Loading corpus…")
    docs = load_corpus(CORPUS_DIR)
    print(f"  {len(docs)} documents loaded")

    print("→ Chunking…")
    all_chunks: list[dict[str, Any]] = []
    for doc in docs:
        chunks = chunk_document(doc)
        all_chunks.extend(chunks)
        print(f"  {doc['source']}: {len(chunks)} chunks")

    print(f"→ Total chunks: {len(all_chunks)}")
    print("→ Setting up Qdrant collection…")
    setup_collection()

    print("→ Ingesting…")
    ingest_chunks(all_chunks)
    print("\n✓ Done. Try: python mp2_rag.py ask")


def cmd_ask() -> None:
    print("Mini-RAG over the Sherlock Holmes corpus.")
    print("Type your question. Empty line or Ctrl-C to exit.\n")
    while True:
        try:
            q = input("? ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not q:
            return
        result = answer(q, k=3)
        print(f"\n{result['answer']}\n")
        print("  Sources:")
        for c in result["citations"]:
            print(f"    - {c['title']} — {c['section']}")
        print(f"  Latency: {result.get('latency_ms', '?')}ms\n")


def cmd_validate() -> None:
    validate_against(DATA_DIR / "predefined_questions.jsonl")
    learner_path = DATA_DIR / "learner_questions.jsonl"
    if learner_path.exists():
        first = json.loads(learner_path.read_text().splitlines()[0])
        if not first["question"].startswith("Replace this"):
            validate_against(learner_path)


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(0)
    cmd = sys.argv[1]
    if cmd == "ingest":   cmd_ingest()
    elif cmd == "ask":    cmd_ask()
    elif cmd == "validate": cmd_validate()
    else:
        print(f"Unknown command: {cmd}\n")
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()
