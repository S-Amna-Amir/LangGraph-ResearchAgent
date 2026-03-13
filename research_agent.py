import os
from typing import TypedDict, Optional, List

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
    sources: Optional[List[str]]
    memory: dict
    decision_log: Optional[str]


# ------------------------------
# Decision Node
# ------------------------------

def decision_node(state: ResearchState):

    query = state["user_query"]
    memory = state.get("memory", {})

    memory_context = ""

    if memory.get("previous_query"):
        memory_context = f"""
Previous conversation:
User: {memory['previous_query']}
Assistant: {memory['previous_answer']}
"""

    prompt = f"""
You are a routing assistant.

Determine if the user question requires LIVE web research.

Reply with only one word:

YES
or
NO

{memory_context}

Question:
{query}
"""

    response = llm.invoke(prompt)

    decision = response.content.strip().upper()

    needs_research = decision.startswith("Y")

    decision_log = "🧠 Decision: "

    if needs_research:
        decision_log += "Research required"
    else:
        decision_log += "Answer from general knowledge"

    return {
        **state,
        "needs_research": needs_research,
        "decision_log": decision_log
    }


# ------------------------------
# Research Node
# ------------------------------

def research_tool_node(state: ResearchState):

    query = state["user_query"]

    search_log = "🔎 Searching web...\n"

    try:

        results = tavily.search(
            query=query,
            max_results=3,
            search_depth="basic"
        )

        urls = [r["url"] for r in results["results"]]

        scrape_log = "🌐 Scraping sources...\n"

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
            "decision_log": state["decision_log"] + "\n" + search_log + scrape_log
        }

    except Exception as exc:

        return {
            **state,
            "tool_result": f"Research failed: {exc}",
            "sources": [],
            "decision_log": state["decision_log"] + "\nResearch failed."
        }


# ------------------------------
# Response Node
# ------------------------------

def response_node(state: ResearchState):

    query = state["user_query"]
    tool_result = state.get("tool_result")
    memory = state.get("memory", {})
    sources = state.get("sources", [])

    memory_context = ""

    if memory.get("previous_query"):
        memory_context = f"""
Previous conversation:
User: {memory['previous_query']}
Assistant: {memory['previous_answer']}
"""

    if tool_result:

        prompt = f"""
You are a research assistant.

Answer the question using the research content below.

{memory_context}

Research content:
{tool_result}

Question:
{query}

Provide a clear answer.
"""

    else:

        prompt = f"""
You are a helpful assistant.

{memory_context}

Question:
{query}
"""

    response = llm.invoke(prompt)

    answer = response.content.strip()

    if sources:
        answer += "\n\nSources:\n"
        for s in sources:
            answer += f"- {s}\n"

    updated_memory = {
        "previous_query": query,
        "previous_answer": answer
    }

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
        return "research_tool"

    return "response"


# ------------------------------
# Build Graph
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


agent = build_graph()


# ------------------------------
# Public function
# ------------------------------

def ask_agent(query: str, memory: dict):

    initial_state = {
        "user_query": query,
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