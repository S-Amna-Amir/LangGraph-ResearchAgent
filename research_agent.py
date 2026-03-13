"""
RAG Research Agent
==================
Replaces web search / Scrapling with a local vector store built from
uploaded PDF, DOCX, and Markdown files.

Dependencies:
    pip install langgraph langchain-groq langchain-core langchain-community \
                python-dotenv faiss-cpu sentence-transformers \
                pypdf python-docx
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


# ══════════════════════════════════════════════════════════════════════════════
# RAG  –  Vector store (module-level singleton, shared with app.py)
# ══════════════════════════════════════════════════════════════════════════════

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Embedding model (runs locally, no API key needed)
_embedder = SentenceTransformer("all-MiniLM-L6-v2")

# In-memory document store  { id -> {"text": ..., "source": ...} }
_doc_store: Dict[int, dict] = {}
_faiss_index: Optional[faiss.IndexFlatL2] = None
_dim = 384   # dimension for all-MiniLM-L6-v2


def _get_index() -> faiss.IndexFlatL2:
    global _faiss_index
    if _faiss_index is None:
        _faiss_index = faiss.IndexFlatL2(_dim)
    return _faiss_index


def ingest_documents(chunks: List[Dict]) -> int:
    """
    Add document chunks to the vector store.
    Each chunk must be: {"text": str, "source": str}
    Returns the number of chunks added.
    """
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


def retrieve(query: str, top_k: int = 4) -> List[Dict]:
    """Return the top-k most relevant chunks for query. Returns [] if index is empty."""
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
        results.append({**chunk, "score": float(dist)})
    return results


def reset_index() -> None:
    """Clear all ingested documents (called when user clears/re-uploads files)."""
    global _faiss_index, _doc_store
    _faiss_index = None
    _doc_store = {}


def index_size() -> int:
    return _get_index().ntotal


# ── Document parsers ──────────────────────────────────────────────────────────

def _chunk_text(text: str, source: str, chunk_size: int = 500, overlap: int = 80) -> List[Dict]:
    """Split text into overlapping word-level chunks."""
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk_text = " ".join(words[start:end])
        chunks.append({"text": chunk_text, "source": source})
        start += chunk_size - overlap
    return chunks


def parse_pdf(file_bytes: bytes, filename: str) -> List[Dict]:
    from pypdf import PdfReader
    import io
    reader = PdfReader(io.BytesIO(file_bytes))
    pages_text = [page.extract_text() or "" for page in reader.pages]
    full_text = "\n".join(pages_text)
    return _chunk_text(full_text, source=filename)


def parse_docx(file_bytes: bytes, filename: str) -> List[Dict]:
    from docx import Document
    import io
    doc = Document(io.BytesIO(file_bytes))
    full_text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
    return _chunk_text(full_text, source=filename)


def parse_md(file_bytes: bytes, filename: str) -> List[Dict]:
    full_text = file_bytes.decode("utf-8", errors="replace")
    return _chunk_text(full_text, source=filename)


def ingest_file(file_bytes: bytes, filename: str) -> int:
    """Auto-detect file type, parse, chunk, and ingest. Returns chunk count."""
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


# ══════════════════════════════════════════════════════════════════════════════
# State
# ══════════════════════════════════════════════════════════════════════════════

class ResearchState(TypedDict):
    user_query: str
    rewritten_query: Optional[str]
    needs_research: Optional[bool]
    tool_result: Optional[str]            # retrieved RAG context
    retrieved_sources: Optional[List[str]]
    final_answer: Optional[str]
    memory: List[Dict]
    decision_log: Optional[str]


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def format_memory(memory: List[Dict]) -> str:
    if not isinstance(memory, list):
        return ""
    history = ""
    for msg in memory[-6:]:
        history += f"{msg.get('role', '')}: {msg.get('content', '')}\n"
    return history


# ══════════════════════════════════════════════════════════════════════════════
# Nodes
# ══════════════════════════════════════════════════════════════════════════════

def decision_node(state: ResearchState) -> ResearchState:
    """Decide whether to retrieve from uploaded documents or answer from general knowledge."""
    query = state["user_query"]
    memory = state.get("memory", [])
    history_text = format_memory(memory)

    # If no documents have been ingested, skip RAG entirely
    if index_size() == 0:
        return {
            **state,
            "needs_research": False,
            "decision_log": "⚠️ No documents uploaded — answering from general knowledge.",
        }

    prompt = f"""You are a routing assistant.

The user has uploaded documents. Decide whether the question below is best answered 
by searching those documents (YES) or from general knowledge / conversation context (NO).

Answer only YES or NO.

Conversation history:
{history_text}

Question: {query}
"""

    response = llm.invoke(prompt)
    needs_research = response.content.strip().upper().startswith("Y")

    log = "🧠 Decision: " + ("Search uploaded documents" if needs_research else "Answer from general knowledge")
    return {**state, "needs_research": needs_research, "decision_log": log}


def rewrite_query_node(state: ResearchState) -> ResearchState:
    """Rewrite the query into a standalone form suitable for semantic search."""
    query = state["user_query"]
    history_text = format_memory(state.get("memory", []))

    prompt = f"""Rewrite the user's question into a standalone, explicit search query 
suitable for semantic search over documents. Use conversation context if needed.

Conversation:
{history_text}

User question: {query}

Rewritten search query:"""

    response = llm.invoke(prompt)
    return {**state, "rewritten_query": response.content.strip()}


def rag_retrieval_node(state: ResearchState) -> ResearchState:
    """Retrieve the most relevant chunks from the vector store."""
    query = state.get("rewritten_query") or state["user_query"]
    decision_log = state.get("decision_log", "") + "\n🔎 Retrieving from documents..."

    chunks = retrieve(query, top_k=4)

    if not chunks:
        tool_result = "No relevant content found in the uploaded documents."
        sources = []
    else:
        tool_result = "\n\n---\n\n".join(c["text"] for c in chunks)
        seen, sources = set(), []
        for c in chunks:
            if c["source"] not in seen:
                seen.add(c["source"])
                sources.append(c["source"])

    decision_log += f"\n📄 Retrieved {len(chunks)} chunk(s) from: {', '.join(sources) if sources else 'none'}"

    return {
        **state,
        "tool_result": tool_result,
        "retrieved_sources": sources,
        "decision_log": decision_log,
    }


def response_node(state: ResearchState) -> ResearchState:
    """Generate the final answer from RAG context or general knowledge."""
    query = state["user_query"]
    tool_result = state.get("tool_result")
    memory = state.get("memory", [])
    sources = state.get("retrieved_sources", [])
    history_text = format_memory(memory)

    if tool_result:
        prompt = f"""You are a research assistant answering questions based on provided documents.

Use the document excerpts below to answer the question. If the excerpts don't contain 
enough information, say so and supplement with general knowledge where appropriate.

Conversation history:
{history_text}

Document excerpts:
{tool_result}

Question: {query}

Answer clearly and concisely:"""
    else:
        prompt = f"""You are a helpful assistant.

Conversation history:
{history_text}

Question: {query}

Answer clearly and concisely:"""

    response = llm.invoke(prompt)
    answer = response.content.strip()

    if sources:
        answer += "\n\n**Sources:**\n" + "\n".join(f"- {s}" for s in sources)

    updated_memory = (memory + [
        {"role": "user", "content": query},
        {"role": "assistant", "content": answer},
    ])[-10:]

    return {**state, "final_answer": answer, "memory": updated_memory}


# ══════════════════════════════════════════════════════════════════════════════
# Router + Graph
# ══════════════════════════════════════════════════════════════════════════════

def route_after_decision(state: ResearchState) -> str:
    return "rewrite_query" if state["needs_research"] else "response"


def build_graph():
    graph = StateGraph(ResearchState)

    graph.add_node("decision", decision_node)
    graph.add_node("rewrite_query", rewrite_query_node)
    graph.add_node("rag_retrieval", rag_retrieval_node)
    graph.add_node("response", response_node)

    graph.set_entry_point("decision")

    graph.add_conditional_edges(
        "decision",
        route_after_decision,
        {
            "rewrite_query": "rewrite_query",
            "response": "response",
        },
    )

    graph.add_edge("rewrite_query", "rag_retrieval")
    graph.add_edge("rag_retrieval", "response")
    graph.add_edge("response", END)

    return graph.compile()


agent = build_graph()


# ══════════════════════════════════════════════════════════════════════════════
# Public API  (called by app.py)
# ══════════════════════════════════════════════════════════════════════════════

def ask_agent(query: str, memory):
    if isinstance(memory, dict):
        if "previous_query" in memory:
            memory = [
                {"role": "user", "content": memory["previous_query"]},
                {"role": "assistant", "content": memory["previous_answer"]},
            ]
        else:
            memory = []

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
        "decision_log": "",
    }

    result = agent.invoke(initial_state)

    return (
        result["final_answer"],
        result["memory"],
        result["decision_log"],
    )