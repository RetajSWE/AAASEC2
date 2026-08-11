"""
DAY 3 — Agent implementation.

READ FIRST: ../01-deep-agents.md

Do not continue to api.py until:
    USE_FAKE=1 uv run python src/agent.py
prints a reply, AND (with real keys) the agent answers using its tools.

The contract this file must satisfy — the ONLY thing api.py will rely on:

    def build_agent() -> object with .ainvoke({"messages": [...]})
"""

import ast
import operator
from datetime import datetime
import os
from pathlib import Path
import asyncio

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend


# TODO:
# 1. Two boring tools: calculate(expression) and current_time().
#    (Boring is the point. Day 3 is about everything AROUND the agent.)


def calculate(expression: str) -> float:
    """Calculate a basic arithmetic expression."""
    tree = ast.parse(expression, mode="eval")

    operators = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
    }

    def evaluate(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value

        if isinstance(node, ast.BinOp) and type(node.op) in operators:
            left = evaluate(node.left)
            right = evaluate(node.right)
            return operators[type(node.op)](left, right)

        raise ValueError("Only basic arithmetic is allowed.")

    return evaluate(tree.body)


def current_time() -> str:
    """Return the current local date and time."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# 2. build_agent():
#    - if USE_FAKE: return a FakeAgent with the same .ainvoke shape
#    - else: create_deep_agent(model=<ChatOpenAI via OpenRouter>,
#                              tools=[...], system_prompt=...,
#                              backend=FilesystemBackend(root_dir=<day3/>,
#                                                        virtual_mode=True),
#                              skills=["/skills/"])
#
# 3. A __main__ smoke test that invokes the agent once and prints the reply.
#
# NOTE: default backends give the agent FILESYSTEM tools but NO shell.
# An execute tool requires a sandbox backend — that is Day 4, on purpose.


load_dotenv()

USE_FAKE = os.getenv("USE_FAKE", "0") == "1"

day3_root = Path(__file__).resolve().parent.parent


class FakeAgent:
    async def ainvoke(self, input_data):
        return {
            "messages": [
                {
                    "role": "assistant",
                    "content": "Fake agent reply.",
                }
            ]
        }


def build_agent():
    if USE_FAKE:
        return FakeAgent()

    llm = ChatOpenAI(
        model="nvidia/nemotron-3-super-120b-a12b:free",
        temperature=0,
        base_url="https://openrouter.ai/api/v1",
    )

    return create_deep_agent(
        model=llm,
        tools=[calculate, current_time],
        system_prompt=(
            "You are a helpful research assistant. "
            "Use the calculate tool for arithmetic and "
            "the current_time tool when asked for the current time."
        ),
        backend=FilesystemBackend(
            root_dir=str(day3_root),
            virtual_mode=True,
        ),
        skills=["/skills/"],
    )


if __name__ == "__main__":

    async def main():
        agent = build_agent()

        result = await agent.ainvoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": "Create a research brief about multi-agent AI systems.",
                    }
                ]
            }
        )

        message = result["messages"][-1]

        if isinstance(message, dict):
            print(message["content"])
        else:
            print(message.content)

    asyncio.run(main())