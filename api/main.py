"""
FastAPI REST API layer for the multi-agent system — with Eval & Observability.

This file extends the original main.py with:
  - /eval/* router (evaluation, dashboard, reports, trends)
  - Shadow evaluation: every /chat response is auto-scored asynchronously
  - No changes to existing /chat, /health, /history endpoints

Drop-in replacement: rename this to main.py to activate.
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional
import uuid
import time
import logging

from agents.graph import get_graph
from agents.state import AgentState

# ── Eval platform ─────────────────────────────────────────────────────────
from eval.api.eval_api import router as eval_router
from eval.runners.eval_runner import EvalRunner
from eval.storage.result_store import ResultStore

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s"
)
logger = logging.getLogger(__name__)

# Eval singletons
_eval_store = ResultStore()
_eval_runner = EvalRunner(store=_eval_store)

# ── FastAPI app ────────────────────────────────────────────────────────────
app = FastAPI(
    title="CX Intelligence Multi-Agent API + LLMOps",
    description=(
        "Multi-agent customer support system built with LangGraph. "
        "Includes a full LLM Evaluation & Observability platform at /eval/*."
    ),
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount eval router — all endpoints under /eval
app.include_router(eval_router)

# In-memory session store (use Redis in production)
session_store: dict = {}


# ── Pydantic Models ────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000, description="Customer query")
    session_id: Optional[str] = Field(None, description="Session ID for conversation continuity")


class ChatResponse(BaseModel):
    session_id: str
    query: str
    response: str
    intent: str
    confidence: float
    needs_human: bool
    current_agent: str
    processing_time_ms: float
    # Added by eval integration:
    eval_score: Optional[float] = Field(None, description="Real-time overall eval score (shadow eval)")


class HealthResponse(BaseModel):
    status: str
    version: str
    agents: list[str]
    eval_platform: str


# ── Background shadow evaluation ───────────────────────────────────────────

def _shadow_evaluate(api_response: dict) -> None:
    """
    Run evaluation in the background so it never adds latency to /chat.
    Fires-and-forgets: if it fails, we log and move on.
    """
    try:
        _eval_runner.evaluate_from_api_response(api_response)
    except Exception as exc:
        logger.warning("Shadow evaluation failed (non-fatal): %s", exc)


# ── Endpoints ──────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint — includes eval platform status."""
    return HealthResponse(
        status="healthy",
        version="2.0.0",
        agents=["supervisor", "faq_agent", "escalation_agent", "human_handoff_agent"],
        eval_platform="active — /eval/health for details",
    )


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, background_tasks: BackgroundTasks):
    """
    Main chat endpoint.

    Processes customer query through multi-agent pipeline, then
    automatically queues a shadow evaluation in the background so
    every production response is continuously monitored.
    """
    start_time = time.time()
    session_id = request.session_id or str(uuid.uuid4())

    logger.info("[%s] Query: %s", session_id, request.query)

    try:
        initial_state: AgentState = {
            "query": request.query,
            "intent": "",
            "context": "",
            "response": "",
            "confidence": 1.0,
            "needs_human": False,
            "conversation_history": [],
            "current_agent": "",
            "escalation_reason": "",
        }

        graph = get_graph()
        final_state = graph.invoke(initial_state)

        if session_id not in session_store:
            session_store[session_id] = []
        session_store[session_id].append({
            "query": request.query,
            "response": final_state["response"],
            "intent": final_state["intent"],
            "agent": final_state["current_agent"],
        })

        processing_time = (time.time() - start_time) * 1000
        logger.info("[%s] Done in %.1fms via %s", session_id, processing_time, final_state["current_agent"])

        api_response = {
            "session_id": session_id,
            "query": request.query,
            "response": final_state["response"],
            "intent": final_state["intent"],
            "confidence": final_state["confidence"],
            "needs_human": final_state["needs_human"],
            "current_agent": final_state["current_agent"],
            "processing_time_ms": round(processing_time, 2),
            "model_name": "gemini-pro",
        }

        # Queue non-blocking shadow evaluation
        background_tasks.add_task(_shadow_evaluate, api_response)

        return ChatResponse(**api_response, eval_score=None)

    except Exception as e:
        logger.error("[%s] Error: %s", session_id, str(e))
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")


@app.get("/history/{session_id}")
async def get_history(session_id: str):
    """Get conversation history for a session."""
    if session_id not in session_store:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"session_id": session_id, "history": session_store[session_id]}
