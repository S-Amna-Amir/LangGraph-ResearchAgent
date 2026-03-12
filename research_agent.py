import os
from typing import TypedDict, Optional

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


class ResearchState(TypedDict):
    user_query: str
    needs_research: Optional[bool]
    tool_result: Optional[str]
    final_answer: Optional[str]
    memory: dict


# ------------------------------
# Decision Node
# ------------------------------

def decision_node(state: ResearchState) -> ResearchState:

    query = state["user_query"]
    memory = state.get("memory", {})

    memory_context = ""
    if memory.get("previous_query"):
        memory_context = (
            f"\nPrevious turn:\n"
            f"User: {memory['previous_query']}\n"
            f"Assistant: {memory['previous_answer']}"
        )

    prompt = f"""
You are a routing assistant.

Determine if the question requires live web research.

Reply with exactly ONE word:
YES or NO.

YES = web research needed
NO = general knowledge is enough.

{memory_context}

Question: {query}
Answer:
"""

    response = llm.invoke(prompt)
    answer = response.content.strip().upper()

    needs_research = answer.startswith("Y")

    return {**state, "needs_research": needs_research}


# ------------------------------
# Research Tool Node
# ------------------------------

def research_tool_node(state: ResearchState) -> ResearchState:

    query = state["user_query"]

    try:

        search_results = tavily.search(
            query=query,
            search_depth="basic",
            max_results=3
        )

        preferred_domains = [
            ".org",
            ".edu",
            "docs.",
            "developer.",
            "official"
        ]

        urls = [
            r["url"] for r in search_results["results"]
            if any(p in r["url"] for p in preferred_domains)
        ] or [r["url"] for r in search_results["results"]]

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

        tool_result = "\n\n".join(collected_text)

        if not tool_result:
            tool_result = "No useful web content retrieved."

        tool_result = tool_result[:4000]

    except Exception as exc:
        tool_result = f"[Research failed: {exc}]"

    return {**state, "tool_result": tool_result}


# ------------------------------
# Response Node
# ------------------------------

def response_node(state: ResearchState) -> ResearchState:

    query = state["user_query"]
    tool_result = state.get("tool_result")
    memory = state.get("memory", {})

    memory_context = ""
    if memory.get("previous_query"):
        memory_context = (
            f"\nPrevious conversation:\n"
            f"User: {memory['previous_query']}\n"
            f"Assistant: {memory['previous_answer']}"
        )

    if tool_result:

        prompt = f"""
You are a helpful research assistant.

Use the research content below to answer the user's question.

{memory_context}

--- WEB RESEARCH CONTENT ---
{tool_result}
----------------------------

Question: {query}

Answer clearly and concisely.
"""

    else:

        prompt = f"""
You are a helpful assistant.

{memory_context}

Question: {query}

Answer clearly.
"""

    response = llm.invoke(prompt)

    final_answer = response.content.strip()

    updated_memory = {
        "previous_query": query,
        "previous_answer": final_answer,
    }

    return {
        **state,
        "final_answer": final_answer,
        "memory": updated_memory
    }


# ------------------------------
# Routing
# ------------------------------

def route_after_decision(state: ResearchState):

    if state.get("needs_research"):
        return "research_tool"

    return "response"


# ------------------------------
# Graph Builder
# ------------------------------

def build_graph():

    graph = StateGraph(ResearchState)

    graph.add_node("decision", decision_node)
    graph.add_node("research_tool", research_tool_node)
    graph.add_node("response", response_node)

    graph.set_entry_point("decision")

    graph.add_conditional_edges(
        "decision",
        route_after_decision,
        {
            "research_tool": "research_tool",
            "response": "response"
        }
    )

    graph.add_edge("research_tool", "response")
    graph.add_edge("response", END)

    return graph.compile()


# ------------------------------
# Public function for UI
# ------------------------------

def ask_agent(query: str, memory: dict):

    agent = build_graph()

    initial_state: ResearchState = {
        "user_query": query,
        "needs_research": None,
        "tool_result": None,
        "final_answer": None,
        "memory": memory,
    }

    result = agent.invoke(initial_state)

    return result["final_answer"], result["memory"]