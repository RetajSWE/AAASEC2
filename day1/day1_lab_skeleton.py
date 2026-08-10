
# Setup: `uv sync`, then create .env (or set USE_FAKE=1 — see README.md).
# ============================================================
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.embeddings import DeterministicFakeEmbedding
import os
import operator
from datetime import datetime
from typing import Annotated, List, Dict
from typing_extensions import TypedDict

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain_core.messages import HumanMessage

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
load_dotenv()


# ============================================================
# STEP 1 — THE STATE  (the "digital clipboard" from the slides)
# ============================================================

class AgentState(TypedDict):
    topic: str
    search_query: str
    collected_data: List[Dict]
    analyzed_data: List[Dict]
    quality_score: int
    iteration_count: int
    final_report: str
    execution_logs: Annotated[List[str], operator.add]


# ============================================================
# STEP 2 — MODEL, SEARCH TOOL, EMBEDDINGS
# ============================================================
llm = ChatOpenAI(
    model="nvidia/nemotron-3-nano-30b-a3b:free",
    temperature=0,
    base_url="https://openrouter.ai/api/v1",
)

search_tool = TavilySearch(max_results=5)
vector_store = InMemoryVectorStore(
    embedding=DeterministicFakeEmbedding(size=1536)
)


# ============================================================
# STEP 3 — STRUCTURED OUTPUT for the quality score
# ============================================================


class QualityScore(BaseModel):
    """Evaluation of research quality."""
    score: int = Field(ge=1, le=10)
    reasoning: str = Field(description="One-sentence justification")

# TODO: evaluator = llm.with_structured_output(QualityScore)
evaluator = llm.with_structured_output(QualityScore)


# ============================================================
# STEP 4 — NODES
# ============================================================
def collect_node(state: AgentState):
    """Search the web. On retries, CHANGE the query!"""
    iteration = state["iteration_count"] + 1

    query = (
        f"{state['topic']} enterprise agentic AI systems "
        f"architecture applications challenges iteration {iteration}"
    )

    results = search_tool.invoke({"query": query})["results"][:3]

    return {
        "search_query": query,
        "collected_data": results,
        "iteration_count": iteration,
        "execution_logs": [
            f"Collection attempt {iteration}: found {len(results)} sources"
        ],
    }



def store_memory_node(state: AgentState):
    """Save source contents into the vector store."""
    contents = [
        item.get("content", "")
        for item in state["collected_data"]
        if item.get("content")
    ]

    if contents:
        vector_store.add_texts(contents)

    return {
        "execution_logs": [
            f"Stored {len(contents)} sources in memory"
        ]
    }

def analyze_node(state: AgentState):
    """LLM-analyze each source. Bonus: retrieve related past
    research with vector_store.similarity_search(content, k=2)
    and include it in the prompt — that's what makes this RAG."""

    analyzed = []

    for item in state["collected_data"]:
        content = item.get("content", "")

        if not content:
            continue

        # Retrieve related information from memory
        related = vector_store.similarity_search(content, k=2)

        related_text = "\n".join(
            doc.page_content for doc in related
        )

        prompt = f"""
Create an enterprise research report about:

{state["topic"]}

Use the following analyzed research:

{research}

The report should include:
1. Executive Summary
2. Key Findings
3. Business Implications
4. Risks and Challenges
5. Conclusion

Write a clear, professional report.

IMPORTANT:
- Do NOT invent a date.
- Do NOT add "Prepared by", author names, organization names, contact information,
  or disclaimers.
- Do NOT add information that is not supported by the provided research.
- Do NOT invent sources or statistics.
"""

        print("DEBUG: calling LLM for analysis...", flush=True)
        response = llm.invoke(prompt)
        print("DEBUG: LLM analysis returned", flush=True)
        analyzed.append({
            "title": item.get("title", ""),
            "url": item.get("url", ""),
            "analysis": response.content,
        })

    return {
        "analyzed_data": analyzed,
        "execution_logs": [
            f"Analyzed {len(analyzed)} sources"
        ],
    }

def evaluate_node(state: AgentState):
    """Score the research with the STRUCTURED evaluator (Step 3)."""

    research = "\n\n".join(
        item["analysis"]
        for item in state["analyzed_data"]
    )

    prompt = f"""
Evaluate the quality of this enterprise research.

Research:
{research}

Give a quality score from 1 to 10.
Consider:
- relevance
- completeness
- accuracy
- usefulness
"""

    result = evaluator.invoke(prompt)

    return {
        "quality_score": result.score,
        "execution_logs": [
            f"Research quality: {result.score}/10 — {result.reasoning}"
        ],
    }

def report_node(state: AgentState):
    """Generate the enterprise report from analyzed_data."""

    research = "\n\n".join(
        f"Source: {item['title']}\n"
        f"URL: {item['url']}\n"
        f"Analysis: {item['analysis']}"
        for item in state["analyzed_data"]
    )

    prompt = f"""
Create an enterprise research report about:

{state["topic"]}

Use the following analyzed research:

{research}

The report should include:
1. Executive Summary
2. Key Findings
3. Business Implications
4. Risks and Challenges
5. Conclusion

Write a clear, professional report.
"""

    response = llm.invoke(prompt)

    return {
        "final_report": response.content,
        "execution_logs": [
            "Final enterprise research report generated"
        ],
    }

def audit_node(state: AgentState):
    """Log completion stats."""

    return {
        "execution_logs": [
            f"Audit complete: "
            f"{state['iteration_count']} collection attempt(s), "
            f"{len(state['collected_data'])} sources collected, "
            f"{len(state['analyzed_data'])} sources analyzed, "
            f"final quality score {state['quality_score']}/10"
        ]
    }
# ============================================================
# STEP 5 — THE CONDITIONAL EDGE (the heart of this lab)
# ============================================================

def quality_router(state: AgentState) -> str:
    if state["quality_score"] >= 7:
        return "report"

    if state["iteration_count"] < 3:
        return "collect"

    return "report"

# ============================================================
# STEP 6 — WIRE THE GRAPH
# ============================================================

workflow = StateGraph(AgentState)

# Add all nodes
workflow.add_node("collect", collect_node)
workflow.add_node("store_memory", store_memory_node)
workflow.add_node("analyze", analyze_node)
workflow.add_node("evaluate", evaluate_node)
workflow.add_node("report", report_node)
workflow.add_node("audit", audit_node)

# Start
workflow.add_edge(START, "collect")

# Linear flow
workflow.add_edge("collect", "store_memory")
workflow.add_edge("store_memory", "analyze")
workflow.add_edge("analyze", "evaluate")

# Conditional routing
workflow.add_conditional_edges(
    "evaluate",
    quality_router,
    {
        "collect": "collect",
        "report": "report",
    },
)

# Finish
workflow.add_edge("report", "audit")
workflow.add_edge("audit", END)

# ============================================================
# STEP 7 — COMPILE with a checkpointer, VISUALIZE, RUN
# ============================================================
# 4. BONUS — human-in-the-loop: compile with
#       interrupt_before=["report"]
#    then inspect state and resume. WHERE TO LOOK:
#       https://docs.langchain.com/oss/python/langgraph/interrupts


if __name__ == "__main__":

    initial_state = {
        "topic": "Enterprise Agentic AI Systems",
        "search_query": "",
        "collected_data": [],
        "analyzed_data": [],
        "quality_score": 0,
        "iteration_count": 0,
        "final_report": "",
        "execution_logs": [],
    }

    # Compile with checkpointer + human-in-the-loop
    app = workflow.compile(
        checkpointer=InMemorySaver(),
        interrupt_before=["report"]
    )

    # Visualize
    print("\n=== GRAPH ===")
    print(app.get_graph().draw_mermaid())

    # Config
    config = {
        "configurable": {
            "thread_id": "run-1"
        }
    }

    # Run
    print("\n=== RUNNING AGENT ===")

    final_state = None

    for state in app.stream(
        initial_state,
        config,
        stream_mode="values",
    ):
        final_state = state

        print(
            f"\nIteration: {state['iteration_count']}"
        )

        print(
            f"Quality: {state['quality_score']}/10"
        )

    # ========================================================
    # HUMAN REVIEW
    # ========================================================

    snapshot = app.get_state(config)

    print("\n=== HUMAN REVIEW ===")
    print("Quality:", snapshot.values["quality_score"])
    print("Sources:", len(snapshot.values["collected_data"]))
    print("Analyzed:", len(snapshot.values["analyzed_data"]))
    print("Next:", snapshot.next)

    input("\nPress ENTER to approve and continue...")

    # ========================================================
    # RESUME
    # ========================================================

    print("\n=== RESUMING AFTER HUMAN REVIEW ===")

    for state in app.stream(
        None,
        config,
        stream_mode="values",
    ):
        final_state = state

        print(
            f"\nIteration: {state['iteration_count']}"
        )

        print(
            f"Quality: {state['quality_score']}/10"
        )

    # ========================================================
    # FINAL REPORT
    # ========================================================

    print("\n=== FINAL REPORT ===")
    print(final_state["final_report"])

    # ========================================================
    # EXECUTION LOGS
    # ========================================================

    print("\n=== EXECUTION LOGS ===")

    for log in final_state["execution_logs"]:
        print("-", log)
# ============================================================
# SELF-CHECK before you look at the solution
# ============================================================
# [ ] My nodes return partial dicts, never the whole mutated state
# [ ] execution_logs uses a reducer, and I can explain why
# [ ] My router has BOTH a quality exit AND an iteration cap
# [ ] Retried searches use a different query than the first attempt
# [ ] I saw the Mermaid diagram and it matches the intended flow
# [ ] I know what GraphRecursionError is and how to trigger it
# [ ] The quality score comes from with_structured_output, not int()
#
# Stuck? Debugging order that works:
#   1. print() the raw return of search_tool.invoke — check its shape
#   2. run app.stream(..., stream_mode="updates") — shows exactly
#      which node produced which state update
#   3. compare your edge wiring against the diagram at the top
#   4. only THEN open day1_lab_solution.py
# ============================================================
