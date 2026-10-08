"""
State definition for the multi-agent system.

In LangGraph, State is a TypedDict that flows through all nodes (agents).
Every agent reads from and writes to this shared state object.
This is how agents share context with each other.
"""

from typing import TypedDict, Annotated, Literal
import operator


class AgentState(TypedDict):
    """
    Shared state that flows through all agents in the graph.
    
    Fields:
        query: The original customer query
        intent: Classified intent (faq / escalation / human_handoff)
        context: Retrieved FAQ context from vector store
        response: Final response to return to customer
        confidence: How confident the FAQ agent is (0.0 to 1.0)
        needs_human: Whether human handoff is needed
        conversation_history: List of messages in the conversation
        current_agent: Which agent is currently handling the query
        escalation_reason: Why escalation was triggered (if any)
    """
    query: str
    intent: str
    context: str
    response: str
    confidence: float
    needs_human: bool
    conversation_history: Annotated[list, operator.add]
    current_agent: str
    escalation_reason: str
