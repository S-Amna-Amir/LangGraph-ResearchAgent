import os
from typing import TypedDict, Optional, List, Dict

from scrapling import Fetcher
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from tavily import TavilyClient

load_dotenv()

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY")

tavily = TavilyClient(api_key=TAVILY_API_KEY)

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0,
    api_key=GROQ_API_KEY,
)

fetcher = Fetcher()


# ------------------------------
# State
# ------------------------------

class ResearchState(TypedDict):
    user_query: str
    rewritten_query: Optional[str]
    needs_research: Optional[bool]
    tool_result: Optional[str]
    final_answer: Optional[str]
    sources: Optional[List[str]]
    memory: List[Dict]
    decision_log: Optional[str]


# ------------------------------
# Helper: Format Memory
# ------------------------------

def format_memory(memory):

    history = ""

    for msg in memory[-6:]:
        history += f"{msg['role']}: {msg['content']}\n"

    return history


# ------------------------------
# Query Rewriter
# ------------------------------

def rewrite_query_node(state: ResearchState):

    query = state["user_query"]
    memory = state.get("memory", [])

    history_text = format_memory(memory)

    prompt = f"""
Rewrite the user's question into a fully explicit search query.

Use conversation context if needed.

Conversation:
{history_text}

User question:
{query}

Rewritten search query:
"""

    response = llm.invoke(prompt)

    rewritten = response.content.strip()

    return {
        **state,
        "rewritten_query": rewritten
    }


# ------------------------------
# Decision Node
# ------------------------------

def decision_node(state: ResearchState):

    query = state["user_query"]
    memory = state.get("memory", [])

    history_text = format_memory(memory)

    query_lower = query.lower()

    realtime_keywords = [
        "today",
        "current",
        "latest",
        "recent",
        "weather",
        "news",
        "now",
        "score"
    ]

    if any(k in query_lower for k in realtime_keywords):

        return {
            **state,
            "needs_research": True,
            "decision_log": "🧠 Decision: Research required (keyword trigger)"
        }

    prompt = f"""
Decide if the user question requires live web research.

Answer only YES or NO.

Conversation history:
{history_text}

Question:
{query}
"""

    response = llm.invoke(prompt)

    decision = response.content.strip().upper()

    needs_research = decision.startswith("Y")

    log = "🧠 Decision: "

    if needs_research:
        log += "Research required"
    else:
        log += "Answer from general knowledge"

    return {
        **state,
        "needs_research": needs_research,
        "decision_log": log
    }


# ------------------------------
# Research Tool
# ------------------------------

def research_tool_node(state: ResearchState):

    query = state.get("rewritten_query") or state["user_query"]

    decision_log = state.get("decision_log", "")
    decision_log += "\n🔎 Searching web...\n🌐 Scraping sources..."

    try:

        results = tavily.search(
            query=query,
            max_results=3,
            search_depth="basic"
        )

        urls = [r["url"] for r in results["results"]]

        collected_text = []

        for url in urls:

            try:

                page = fetcher.get(url, timeout=15)

                page_text = page.get_all_text(
                    ignore_tags=("script", "style", "nav", "footer")
                )

                collected_text.append(page_text[:1500])

            except Exception:
                continue

        combined_text = "\n\n".join(collected_text)

        if not combined_text:
            combined_text = "No useful web content retrieved."

        combined_text = combined_text[:4000]

        return {
            **state,
            "tool_result": combined_text,
            "sources": urls,
            "decision_log": decision_log
        }

    except Exception as exc:

        return {
            **state,
            "tool_result": f"Research failed: {exc}",
            "sources": [],
            "decision_log": decision_log
        }


# ------------------------------
# Response Node
# ------------------------------

def response_node(state: ResearchState):

    query = state["user_query"]
    tool_result = state.get("tool_result")
    memory = state.get("memory", [])
    sources = state.get("sources", [])

    history_text = format_memory(memory)

    if tool_result:

        prompt = f"""
You are a research assistant.

Use the research content below to answer the question.

Conversation history:
{history_text}

Research content:
{tool_result}

Question:
{query}

Answer clearly.
"""

    else:

        prompt = f"""
You are a helpful assistant.

Conversation history:
{history_text}

Question:
{query}

Answer clearly.
"""

    response = llm.invoke(prompt)

    answer = response.content.strip()

    if sources:
        answer += "\n\nSources:\n"
        for s in sources:
            answer += f"- {s}\n"

    updated_memory = memory + [
        {"role": "user", "content": query},
        {"role": "assistant", "content": answer}
    ]

    updated_memory = updated_memory[-10:]

    return {
        **state,
        "final_answer": answer,
        "memory": updated_memory
    }


# ------------------------------
# Router
# ------------------------------

def route_after_decision(state):

    if state["needs_research"]:
        return "rewrite_query"

    return "response"


# ------------------------------
# Graph Builder
# ------------------------------

def build_graph():

    graph = StateGraph(ResearchState)

    graph.add_node("decision", decision_node)
    graph.add_node("rewrite_query", rewrite_query_node)
    graph.add_node("research_tool", research_tool_node)
    graph.add_node("response", response_node)

    graph.set_entry_point("decision")

    graph.add_conditional_edges(
        "decision",
        route_after_decision,
        {
            "rewrite_query": "rewrite_query",
            "response": "response"
        }
    )

    graph.add_edge("rewrite_query", "research_tool")
    graph.add_edge("research_tool", "response")
    graph.add_edge("response", END)

    return graph.compile()


agent = build_graph()


# ------------------------------
# Public Function
# ------------------------------

def ask_agent(query: str, memory):

    if memory is None:
        memory = []

    initial_state = {
        "user_query": query,
        "rewritten_query": None,
        "needs_research": None,
        "tool_result": None,
        "final_answer": None,
        "sources": [],
        "memory": memory,
        "decision_log": ""
    }

    result = agent.invoke(initial_state)

    return (
        result["final_answer"],
        result["memory"],
        result["decision_log"]
    )