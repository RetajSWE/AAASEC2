"""
DAY 3 — HTTP API.

FastAPI service exposing the Day 3 agent through an OpenResponses-shaped endpoint.
"""

import os
import time
import uuid

from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel

from src.agent import build_agent


load_dotenv()


# ---------------------------------------------------------
# App + Agent
# ---------------------------------------------------------

app = FastAPI(
    title="Day 3 Agent API",
    description="FastAPI service exposing the Day 3 deep agent.",
    version="1.0.0",
)

# Build the agent ONCE when the API starts.
agent = build_agent()


# ---------------------------------------------------------
# Request model
# ---------------------------------------------------------

class ResponseRequest(BaseModel):
    input: str
    model: str | None = None


# ---------------------------------------------------------
# Health check
# ---------------------------------------------------------

@app.get("/healthz")
async def healthz():
    return {"status": "ok"}


# ---------------------------------------------------------
# OpenResponses subset
# ---------------------------------------------------------

@app.post("/v1/responses")
async def create_response(request: ResponseRequest):
    result = await agent.ainvoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": request.input,
                }
            ]
        }
    )

    message = result["messages"][-1]

    if isinstance(message, dict):
        text = message.get("content", "")
    else:
        text = message.content

    # Some LangChain messages may contain structured content.
    if not isinstance(text, str):
        text = str(text)

    model_name = request.model or "day3-agent"

    return {
        "id": f"resp_{uuid.uuid4().hex}",
        "object": "response",
        "created_at": int(time.time()),
        "status": "completed",
        "model": model_name,
        "output": [
            {
                "type": "message",
                "role": "assistant",
                "content": [
                    {
                        "type": "output_text",
                        "text": text,
                    }
                ],
            }
        ],
    }


# ---------------------------------------------------------
# A2A Agent Card — stub for now
# ---------------------------------------------------------

@app.get("/.well-known/agent-card.json")
async def agent_card():
    student_name = os.getenv("STUDENT_NAME", "student")
    public_url = os.getenv("PUBLIC_URL", "http://localhost:8000")

    return {
        "protocolVersion": "1.0",
        "name": f"{student_name}-agent",
        "description": "An AI agent that researches topics and produces structured research briefs.",
        "url": f"{public_url}/v1/responses",
        "version": "0.1.0",
        "capabilities": {
            "streaming": False
        },
        "defaultInputModes": ["text/plain"],
        "defaultOutputModes": ["text/plain"],
        "skills": [
            {
                "id": "research-brief",
                "name": "Research Brief",
                "description": "Researches a topic and produces a concise research brief.",
                "tags": ["research", "brief", "analysis"]
            }
        ]
    }

