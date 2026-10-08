"""
FastAPI REST API layer for the multi-agent system.

Exposes endpoints:
- POST /chat       — main endpoint, processes customer query
- GET  /health     — health check
- GET  /history/{session_id} — conversation history

Industry practices followed:
- Pydantic models for request/response validation
- Proper HTTP status codes
- Error handling with meaningful messages
- Request/response logging
- Session-based conversation history
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional
import uuid
import time
import logging

from agents.graph import get_graph
from agents.state import AgentState

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="CX Intelligence Multi-Agent API",
    description="Multi-agent customer support system built with LangGraph",
    version="1.0.0"
)

# CORS middleware for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# In-memory session store (use Redis in production)
session_store: dict = {}


# ── Pydantic Models ──────────────────────────────────────────────────

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

class HealthResponse(BaseModel):
    status: str
    version: str
    agents: list[str]


# ── Endpoints ────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint — always available."""
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        agents=["supervisor", "faq_agent", "escalation_agent", "human_handoff_agent"]
    )


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Main chat endpoint.
    
    Processes customer query through multi-agent pipeline:
    supervisor -> [faq | escalation | human_handoff] -> response
    """
    start_time = time.time()
    
    # Generate session ID if not provided
    session_id = request.session_id or str(uuid.uuid4())
    
    logger.info(f"[{session_id}] Query received: {request.query}")
    
    try:
        # Build initial state
        initial_state: AgentState = {
            "query": request.query,
            "intent": "",
            "context": "",
            "response": "",
            "confidence": 1.0,
            "needs_human": False,
            "conversation_history": [],
            "current_agent": "",
            "escalation_reason": ""
        }
        
        # Run through multi-agent graph
        graph = get_graph()
        final_state = graph.invoke(initial_state)
        
        # Store session history
        if session_id not in session_store:
            session_store[session_id] = []
        session_store[session_id].append({
            "query": request.query,
            "response": final_state["response"],
            "intent": final_state["intent"],
            "agent": final_state["current_agent"]
        })
        
        processing_time = (time.time() - start_time) * 1000
        logger.info(f"[{session_id}] Processed in {processing_time:.1f}ms by {final_state['current_agent']}")
        
        return ChatResponse(
            session_id=session_id,
            query=request.query,
            response=final_state["response"],
            intent=final_state["intent"],
            confidence=final_state["confidence"],
            needs_human=final_state["needs_human"],
            current_agent=final_state["current_agent"],
            processing_time_ms=round(processing_time, 2)
        )
        
    except Exception as e:
        logger.error(f"[{session_id}] Error processing query: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing query: {str(e)}"
        )


@app.get("/history/{session_id}")
async def get_history(session_id: str):
    """Get conversation history for a session."""
    if session_id not in session_store:
        raise HTTPException(status_code=404, detail="Session not found")
    return {
        "session_id": session_id,
        "history": session_store[session_id]
    }
