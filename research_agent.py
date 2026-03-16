"""
RAG Research Agent
==================
Uses a local vector store built from
uploaded PDF, DOCX, and Markdown files.
"""

import os
from typing import TypedDict, Optional, List, Dict

from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq

load_dotenv()

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0,
    api_key=GROQ_API_KEY,
)

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

_embedder = SentenceTransformer("all-MiniLM-L6-v2")

_doc_store: Dict[int, dict] = {}
_faiss_index: Optional[faiss.IndexFlatL2] = None
_dim = 384


def _get_index() -> faiss.IndexFlatL2:
    global _faiss_index
    if _faiss_index is None:
        _faiss_index = faiss.IndexFlatL2(_dim)
    return _faiss_index


# ─────────────────────────────────────────────────────────
# DOCUMENT INGESTION
# ─────────────────────────────────────────────────────────

def ingest_documents(chunks: List[Dict]) -> int:

    if not chunks:
        return 0

    index = _get_index()

    texts = [c["text"] for c in chunks]
    embeddings = _embedder.encode(texts, show_progress_bar=False).astype("float32")

    start_id = index.ntotal
    index.add(embeddings)

    for i, chunk in enumerate(chunks):
        _doc_store[start_id + i] = chunk

    return len(chunks)


def retrieve(query: str, top_k: int = 6) -> List[Dict]:

    index = _get_index()

    if index.ntotal == 0:
        return []

    query_vec = _embedder.encode([query], show_progress_bar=False).astype("float32")

    k = min(top_k, index.ntotal)
    distances, ids = index.search(query_vec, k)

    results = []

    for dist, doc_id in zip(distances[0], ids[0]):

        if doc_id == -1:
            continue

        chunk = _doc_store[int(doc_id)]

        results.append({
            **chunk,
            "score": float(dist)
        })

    return results


def reset_index() -> None:
    global _faiss_index, _doc_store
    _faiss_index = None
    _doc_store = {}


def index_size() -> int:
    return _get_index().ntotal


# ─────────────────────────────────────────────────────────
# TEXT CHUNKING
# ─────────────────────────────────────────────────────────

def _chunk_text(text: str, source: str, chunk_size: int = 500, overlap: int = 80):

    words = text.split()

    chunks = []
    start = 0

    while start < len(words):

        end = start + chunk_size

        chunk_body = " ".join(words[start:end])

        # INCLUDE FILENAME IN EMBEDDING TEXT
        chunk_text = f"Source Document: {source}\n\n{chunk_body}"

        chunks.append({
            "text": chunk_text,
            "source": source
        })

        start += chunk_size - overlap

    return chunks


# ─────────────────────────────────────────────────────────
# FILE PARSERS
# ─────────────────────────────────────────────────────────

def parse_pdf(file_bytes: bytes, filename: str):

    from pypdf import PdfReader
    import io

    reader = PdfReader(io.BytesIO(file_bytes))

    pages_text = [
        page.extract_text() or ""
        for page in reader.pages
    ]

    full_text = "\n".join(pages_text)

    return _chunk_text(full_text, filename)


def parse_docx(file_bytes: bytes, filename: str):

    from docx import Document
    import io

    doc = Document(io.BytesIO(file_bytes))

    full_text = "\n".join(
        p.text for p in doc.paragraphs if p.text.strip()
    )

    return _chunk_text(full_text, filename)


def parse_md(file_bytes: bytes, filename: str):

    full_text = file_bytes.decode("utf-8", errors="replace")

    return _chunk_text(full_text, filename)


def ingest_file(file_bytes: bytes, filename: str):

    ext = filename.lower().rsplit(".", 1)[-1]

    if ext == "pdf":
        chunks = parse_pdf(file_bytes, filename)

    elif ext == "docx":
        chunks = parse_docx(file_bytes, filename)

    elif ext in ("md", "markdown", "txt"):
        chunks = parse_md(file_bytes, filename)

    else:
        raise ValueError(f"Unsupported file type: .{ext}")

    return ingest_documents(chunks)


# ─────────────────────────────────────────────────────────
# STATE
# ─────────────────────────────────────────────────────────

class ResearchState(TypedDict):

    user_query: str
    rewritten_query: Optional[str]
    needs_research: Optional[bool]
    tool_result: Optional[str]
    retrieved_sources: Optional[List[str]]
    final_answer: Optional[str]
    memory: List[Dict]
    decision_log: Optional[str]


# ─────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────

def format_memory(memory: List[Dict]):

    if not isinstance(memory, list):
        return ""

    history = ""

    for msg in memory[-6:]:

        role = msg.get("role", "")
        content = msg.get("content", "")

        history += f"{role}: {content}\n"

    return history


# ─────────────────────────────────────────────────────────
# NODES
# ─────────────────────────────────────────────────────────

def decision_node(state: ResearchState):

    if index_size() == 0:

        return {
            **state,
            "needs_research": False,
            "decision_log": "⚠️ No documents uploaded — answering from general knowledge."
        }

    return {
        **state,
        "needs_research": True,
        "decision_log": "🧠 Documents detected — retrieving relevant sections."
    }


def rewrite_query_node(state: ResearchState):

    query = state["user_query"]

    history_text = format_memory(state.get("memory", []))

    prompt = f"""
Rewrite the user's question into a standalone search query.

Conversation:
{history_text}

Question:
{query}

Search query:
"""

    response = llm.invoke(prompt)

    return {
        **state,
        "rewritten_query": response.content.strip()
    }


def rag_retrieval_node(state: ResearchState):

    query = state.get("rewritten_query") or state["user_query"]

    decision_log = state.get("decision_log", "")
    decision_log += "\n🔎 Retrieving from documents..."

    chunks = retrieve(query, top_k=6)

    if not chunks:

        tool_result = "No relevant content found in uploaded documents."
        sources = []

    else:

        tool_result = "\n\n---\n\n".join(
            c["text"] for c in chunks
        )

        seen = set()
        sources = []

        for c in chunks:

            if c["source"] not in seen:
                seen.add(c["source"])
                sources.append(c["source"])

    decision_log += f"\n📄 Retrieved {len(chunks)} chunk(s) from: {', '.join(sources) if sources else 'none'}"

    return {
        **state,
        "tool_result": tool_result,
        "retrieved_sources": sources,
        "decision_log": decision_log
    }


def response_node(state: ResearchState):

    query = state["user_query"]

    tool_result = state.get("tool_result")

    memory = state.get("memory", [])

    sources = state.get("retrieved_sources", [])

    history_text = format_memory(memory)

    if tool_result:

        prompt = f"""
You are a research assistant.

Use the document excerpts to answer the question.

Conversation:
{history_text}

Documents:
{tool_result}

Question:
{query}

Answer clearly:
"""

    else:

        prompt = f"""
You are a helpful assistant.

Conversation:
{history_text}

Question:
{query}

Answer clearly:
"""

    response = llm.invoke(prompt)

    answer = response.content.strip()

    if sources:

        answer += "\n\n**Sources:**\n"

        for s in sources:
            answer += f"- {s}\n"

    updated_memory = (
        memory
        + [
            {"role": "user", "content": query},
            {"role": "assistant", "content": answer},
        ]
    )[-10:]

    return {
        **state,
        "final_answer": answer,
        "memory": updated_memory
    }


# ─────────────────────────────────────────────────────────
# GRAPH
# ─────────────────────────────────────────────────────────

def build_graph():

    graph = StateGraph(ResearchState)

    graph.add_node("decision", decision_node)
    graph.add_node("rewrite_query", rewrite_query_node)
    graph.add_node("rag_retrieval", rag_retrieval_node)
    graph.add_node("response", response_node)

    graph.set_entry_point("decision")

    graph.add_edge("decision", "rewrite_query")
    graph.add_edge("rewrite_query", "rag_retrieval")
    graph.add_edge("rag_retrieval", "response")
    graph.add_edge("response", END)

    return graph.compile()


agent = build_graph()


# ─────────────────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────────────────

def ask_agent(query: str, memory):

    if memory is None:
        memory = []

    initial_state: ResearchState = {

        "user_query": query,
        "rewritten_query": None,
        "needs_research": None,
        "tool_result": None,
        "retrieved_sources": [],
        "final_answer": None,
        "memory": memory,
        "decision_log": ""
    }

    result = agent.invoke(initial_state)

    return (
        result["final_answer"],
        result["memory"],
        result["decision_log"],
    )