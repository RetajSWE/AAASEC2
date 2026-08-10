# ============================================================
# DAY 2 LAB — Build a Multi-Agent Research Team
# ============================================================

import os
import operator
from datetime import datetime
from typing import Annotated, List, Literal
from typing_extensions import TypedDict

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver

load_dotenv()

MAX_REVISIONS = 2
MAX_TURNS = 12


# ============================================================
# STEP 1 — SHARED STATE
# ============================================================

class TeamState(TypedDict):
    task: str
    research_notes: Annotated[List[str], operator.add]
    analysis: str
    draft: str
    critique: str
    revision_count: int
    turn_count: int
    next_agent: str
    execution_logs: Annotated[List[str], operator.add]


# ============================================================
# STEP 2 — STRUCTURED ROUTING DECISION
# ============================================================

class RouterDecision(BaseModel):
    """The supervisor's choice of who acts next."""

    next_agent: Literal[
        "researcher",
        "analyst",
        "writer",
        "critic",
        "FINISH"
    ]

    reason: str = Field(
        description="One sentence explaining the choice"
    )


# ============================================================
# STEP 3 — ONE LLM, FOUR PERSONAS
# ============================================================

PERSONAS = {
    "researcher": """
You are the Researcher.

Your job is to gather factual information relevant to the task.

You may use the provided web search results.

You MUST:
- focus only on collecting factual information
- identify useful evidence
- include important source information
- summarize the findings clearly

You MUST NOT:
- perform deep analysis
- criticize the draft
- write the final answer
""",

    "analyst": """
You are the Analyst.

Your job is to analyze the research notes.

You MUST:
- identify important findings
- identify benefits and risks
- identify patterns and implications
- provide useful conclusions for the writer

You MUST NOT:
- search the web
- write the final draft
- invent unsupported facts

Use only the available research notes.
""",

    "writer": """
You are the Writer.

Your job is to create or revise the final draft.

You MUST:
- use the research and analysis provided
- produce a clear, coherent answer
- address the original task directly
- follow the critic's feedback when revising

You MUST NOT:
- perform new research
- ignore the research
- evaluate your own work

Return only the draft.
""",

    "critic": """
You are the Critic.

Your job is to evaluate the draft against the research and analysis.

Check:
- factual support
- completeness
- clarity
- consistency
- whether the draft answers the original task

You MUST NOT rewrite the draft.

Return exactly one of:

APPROVED

or:

REVISE: <specific fixes>
"""
}


# ============================================================
# LLM SETUP
# ============================================================

USE_FAKE = os.getenv("USE_FAKE", "0") == "1"

if not USE_FAKE:

    from langchain_openai import ChatOpenAI
    from langchain_tavily import TavilySearch

    api_key = os.getenv("OPENROUTER_API_KEY")

    if not api_key:
        raise ValueError(
            "OPENROUTER_API_KEY is missing. "
            "Add it to your .env file or set USE_FAKE=1."
        )

    llm = ChatOpenAI(
        model=os.getenv(
            "OPENROUTER_MODEL",
            "openai/gpt-oss-20b:free"
        ),
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
        temperature=0.3,
        max_retries=2,
    )

    search_tool = TavilySearch(
        max_results=4
    )

    supervisor_llm = llm.with_structured_output(
        RouterDecision
    )

else:

    llm = None
    search_tool = None
    supervisor_llm = None


# ============================================================
# FAKE MODE
# ============================================================

def fake_persona(role, user_content):

    if role == "researcher":
        return (
            "Research notes: Multi-agent AI systems divide work "
            "among specialized agents. They can improve task "
            "decomposition, parallel work, and specialization, "
            "but introduce additional latency, cost, complexity, "
            "and coordination challenges."
        )

    if role == "analyst":
        return (
            "Analysis: Multi-agent systems can be valuable when "
            "tasks naturally decompose into independent specialized "
            "activities. However, they should only be adopted when "
            "the additional coordination cost is justified by "
            "measurable improvements in quality or efficiency."
        )

    if role == "writer":
        return (
            "Companies should consider adopting multi-agent AI "
            "systems in 2026 selectively rather than universally. "
            "They are most useful for complex workflows that benefit "
            "from specialized roles, but organizations should first "
            "evaluate cost, latency, reliability, and maintenance "
            "requirements."
        )

    if role == "critic":
        return "APPROVED"

    return "No response."


def run_persona(role, user_content):

    if USE_FAKE:
        return fake_persona(role, user_content)

    response = llm.invoke([
        SystemMessage(
            content=PERSONAS[role]
        ),
        HumanMessage(
            content=user_content
        )
    ])

    return response.content


# ============================================================
# STEP 4 — SUPERVISOR NODE
# ============================================================

def supervisor_node(state: TeamState):

    turn_count = state["turn_count"] + 1

    research_available = len(state["research_notes"]) > 0
    analysis_available = bool(state["analysis"])
    draft_available = bool(state["draft"])
    critique_available = bool(state["critique"])

    critique_preview = (
        state["critique"][:500]
        if state["critique"]
        else "None"
    )

    status_summary = f"""
TASK:
{state["task"]}

CURRENT TEAM STATUS:

Research available: {research_available}
Number of research notes: {len(state["research_notes"])}

Analysis available: {analysis_available}

Draft available: {draft_available}

Critique:
{critique_preview}

Revision count:
{state["revision_count"]}/{MAX_REVISIONS}

Turn count:
{turn_count}/{MAX_TURNS}
"""

    if USE_FAKE:

        if not research_available:
            next_agent = "researcher"
            reason = "Research is required before analysis."

        elif not analysis_available:
            next_agent = "analyst"
            reason = "Research exists but analysis is missing."

        elif not draft_available:
            next_agent = "writer"
            reason = "Analysis exists but a draft has not been created."

        elif not critique_available:
            next_agent = "critic"
            reason = "The draft needs to be reviewed."

        elif state["critique"].startswith("REVISE"):

            if state["revision_count"] < MAX_REVISIONS:
                next_agent = "writer"
                reason = "The critic requested a revision."
            else:
                next_agent = "FINISH"
                reason = "Maximum revisions have been reached."

        else:
            next_agent = "FINISH"
            reason = "The critic approved the draft."

    else:

        decision = supervisor_llm.invoke([
            SystemMessage(
                content="""
You are the supervisor of a multi-agent research team.

Your job is to decide which agent should act next.

Available agents:

researcher:
Collects factual information using web search.

analyst:
Analyzes research notes.

writer:
Creates or revises the draft.

critic:
Reviews the draft and returns APPROVED or REVISE.

FINISH:
Ends the workflow when the final draft is ready.

Workflow logic:

1. If research is missing, choose researcher.
2. If research exists but analysis is missing, choose analyst.
3. If analysis exists but draft is missing, choose writer.
4. If a draft exists but has not been reviewed, choose critic.
5. If the critic says REVISE, choose writer.
6. If the critic says APPROVED, choose FINISH.

Never invent another agent.
Use only the available state information.
"""
            ),
            HumanMessage(
                content=status_summary
            )
        ])

        next_agent = decision.next_agent
        reason = decision.reason

    # --------------------------------------------------------
    # GUARDRAIL A — MAX TURNS
    # --------------------------------------------------------

    if turn_count > MAX_TURNS:
        next_agent = "FINISH"
        reason = "Maximum turn limit reached."

    # --------------------------------------------------------
    # GUARDRAIL B — MAX REVISIONS
    # --------------------------------------------------------

    if (
        next_agent in ["writer", "critic"]
        and state["revision_count"] >= MAX_REVISIONS
        and state["draft"]
    ):
        next_agent = "FINISH"
        reason = "Maximum revision limit reached."

    log = (
        f"{datetime.now().strftime('%H:%M:%S')} | "
        f"Supervisor -> {next_agent} | {reason}"
    )

    return {
        "next_agent": next_agent,
        "turn_count": turn_count,
        "execution_logs": [log],
    }


# ============================================================
# STEP 5 — RESEARCHER NODE
# ============================================================

def researcher_node(state: TeamState):
    """Search the web and condense results into research notes."""

    if USE_FAKE:

        notes = run_persona(
            "researcher",
            f"Task: {state['task']}"
        )

    else:

        search_response = search_tool.invoke({
            "query": state["task"]
        })

        if isinstance(search_response, dict):
            results = search_response.get(
                "results",
                search_response
            )
        else:
            results = search_response

        raw_parts = []

        for result in results:

            if not isinstance(result, dict):
                raw_parts.append(str(result))
                continue

            title = result.get(
                "title",
                "Untitled"
            )

            content = result.get(
                "content",
                ""
            )

            url = result.get(
                "url",
                ""
            )

            raw_parts.append(
                f"Title: {title}\n"
                f"Content: {content}\n"
                f"URL: {url}"
            )

        raw = "\n\n".join(raw_parts)

        notes = run_persona(
            "researcher",
            f"""
Task:
{state["task"]}

Search results:
{raw}

Condense the useful factual information into
clear research notes.

Do not perform analysis.
"""
        )

    return {
        "research_notes": [notes],
        "execution_logs": [
            "Researcher completed research."
        ],
    }


# ============================================================
# ANALYST NODE
# ============================================================

def analyst_node(state: TeamState):
    """Turn raw research notes into analysis."""

    notes = "\n\n".join(
        state["research_notes"]
    )

    analysis = run_persona(
        "analyst",
        f"""
Task:
{state["task"]}

Research notes:
{notes}

Analyze the research.

Identify:
- key findings
- benefits
- risks
- implications
- important considerations
- overall conclusion

Do not write the final draft.
"""
    )

    return {
        "analysis": analysis,
        "execution_logs": [
            "Analyst completed analysis."
        ],
    }


# ============================================================
# WRITER NODE
# ============================================================

def writer_node(state: TeamState):
    """Write a new draft or revise an existing draft."""

    revising = (
        bool(state["critique"])
        and state["critique"].startswith("REVISE")
    )

    if revising:

        prompt = f"""
You are revising an existing draft.

Task:
{state["task"]}

Analysis:
{state["analysis"]}

Previous draft:
{state["draft"]}

Critique:
{state["critique"]}

Revise the draft according to the critic's feedback.

Fix every issue identified by the critic.

Return ONLY the revised draft.
"""

    else:

        notes = "\n\n".join(
            state["research_notes"]
        )

        prompt = f"""
Create the first draft for the following task.

Task:
{state["task"]}

Research:
{notes}

Analysis:
{state["analysis"]}

Write a clear, well-structured answer.

Return ONLY the draft.
"""

    draft = run_persona(
        "writer",
        prompt
    )

    revision_increment = (
        1 if revising else 0
    )

    if revising:
        log_message = (
            "Writer revised the draft."
        )
    else:
        log_message = (
            "Writer created the initial draft."
        )

    return {
        "draft": draft,

        # IMPORTANT:
        # The writer has consumed the previous critique.
        # The critic should create NEW feedback.
        "critique": "",

        "revision_count": (
            state["revision_count"]
            + revision_increment
        ),

        "execution_logs": [
            log_message
        ],
    }


# ============================================================
# CRITIC NODE
# ============================================================

def critic_node(state: TeamState):
    """Review the draft against the research."""

    notes = "\n\n".join(
        state["research_notes"]
    )

    critique = run_persona(
        "critic",
        f"""
Task:
{state["task"]}

Research:
{notes}

Analysis:
{state["analysis"]}

Draft:
{state["draft"]}

Evaluate the draft.

Check:
- factual support
- completeness
- clarity
- consistency
- whether it answers the task

Return exactly:

APPROVED

or:

REVISE: <specific fixes>
"""
    )

    return {
        "critique": critique,
        "execution_logs": [
            f"Critic result: {critique}"
        ],
    }


# ============================================================
# STEP 6 — ROUTING FUNCTION
# ============================================================

def route_from_supervisor(
    state: TeamState
) -> str:

    return state["next_agent"]


# ============================================================
# STEP 6 — BUILD THE GRAPH
# ============================================================

graph = StateGraph(TeamState)

# Add nodes
graph.add_node(
    "supervisor",
    supervisor_node
)

graph.add_node(
    "researcher",
    researcher_node
)

graph.add_node(
    "analyst",
    analyst_node
)

graph.add_node(
    "writer",
    writer_node
)

graph.add_node(
    "critic",
    critic_node
)


# START -> SUPERVISOR

graph.add_edge(
    START,
    "supervisor"
)


# SUPERVISOR -> WORKER OR FINISH

graph.add_conditional_edges(
    "supervisor",
    route_from_supervisor,
    {
        "researcher": "researcher",
        "analyst": "analyst",
        "writer": "writer",
        "critic": "critic",
        "FINISH": END,
    }
)


# EVERY WORKER -> SUPERVISOR

for worker in [
    "researcher",
    "analyst",
    "writer",
    "critic"
]:
    graph.add_edge(
        worker,
        "supervisor"
    )


# ============================================================
# STEP 7 — COMPILE
# ============================================================

checkpointer = InMemorySaver()

app = graph.compile(
    checkpointer=checkpointer
)


# ============================================================
# INITIAL STATE
# ============================================================

initial_state = {
    "task": (
        "Should our company adopt multi-agent AI "
        "systems in 2026?"
    ),

    "research_notes": [],

    "analysis": "",

    "draft": "",

    "critique": "",

    "revision_count": 0,

    "turn_count": 0,

    "next_agent": "",

    "execution_logs": [],
}


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print("DAY 2 — MULTI-AGENT RESEARCH TEAM")
    print("=" * 70)

    print("\nGRAPH:")
    print(app.get_graph().draw_mermaid())

    config = {
        "configurable": {
            "thread_id": "day2-research-team"
        }
    }

    print("\n" + "=" * 70)
    print("EXECUTION")
    print("=" * 70)

    final_state = None

    for state in app.stream(
        initial_state,
        config=config,
        stream_mode="values"
    ):

        final_state = state

        print(
            f"\nCurrent agent decision: "
            f"{state.get('next_agent', '')}"
        )

        print(
            f"Turn: "
            f"{state.get('turn_count', 0)}"
        )

        print(
            f"Revisions: "
            f"{state.get('revision_count', 0)}"
        )

    # ========================================================
    # FINAL OUTPUT
    # ========================================================

    print("\n" + "=" * 70)
    print("FINAL DRAFT")
    print("=" * 70)

    print(
        final_state["draft"]
    )

    print("\n" + "=" * 70)
    print("STATS")
    print("=" * 70)

    print(
        "Turns:",
        final_state["turn_count"]
    )

    print(
        "Revisions:",
        final_state["revision_count"]
    )

    print(
        "Research notes:",
        len(final_state["research_notes"])
    )

    print("\n" + "=" * 70)
    print("EXECUTION LOG")
    print("=" * 70)

    for log in final_state["execution_logs"]:
        print(log)