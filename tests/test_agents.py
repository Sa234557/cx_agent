"""
Unit tests for the multi-agent system.

Tests cover:
- State transitions
- Intent routing logic
- FAQ retrieval
- Agent output validation

Industry practice: tests prove your system works as documented.
Having tests = senior engineers trust your code.
"""

import pytest
from unittest.mock import MagicMock, patch
from agents.state import AgentState
from agents.supervisor import route_after_supervisor
from agents.faq_agent import route_after_faq
from agents.escalation_agent import route_after_escalation


# ── Helper to build test state ───────────────────────────────────────

def make_state(**kwargs) -> AgentState:
    """Build a test AgentState with defaults."""
    defaults = {
        "query": "How do I reset my password?",
        "intent": "faq",
        "context": "",
        "response": "",
        "confidence": 0.9,
        "needs_human": False,
        "conversation_history": [],
        "current_agent": "supervisor",
        "escalation_reason": ""
    }
    defaults.update(kwargs)
    return defaults


# ── Routing Tests ────────────────────────────────────────────────────

class TestSupervisorRouting:
    """Test that supervisor routes to correct agents."""

    def test_routes_faq_intent_to_faq_agent(self):
        state = make_state(intent="faq")
        assert route_after_supervisor(state) == "faq_agent"

    def test_routes_escalation_intent_to_escalation_agent(self):
        state = make_state(intent="escalation")
        assert route_after_supervisor(state) == "escalation_agent"

    def test_routes_human_handoff_intent_correctly(self):
        state = make_state(intent="human_handoff")
        assert route_after_supervisor(state) == "human_handoff_agent"

    def test_defaults_to_faq_for_unknown_intent(self):
        state = make_state(intent="unknown_intent")
        assert route_after_supervisor(state) == "faq_agent"


class TestFAQAgentRouting:
    """Test FAQ agent routing based on confidence and needs_human."""

    def test_ends_when_high_confidence_and_no_human_needed(self):
        state = make_state(confidence=0.9, needs_human=False)
        assert route_after_faq(state) == "END"

    def test_routes_to_human_when_needs_human_true(self):
        state = make_state(confidence=0.2, needs_human=True)
        assert route_after_faq(state) == "human_handoff_agent"

    def test_ends_when_needs_human_false_regardless_of_confidence(self):
        state = make_state(confidence=0.5, needs_human=False)
        assert route_after_faq(state) == "END"


class TestEscalationAgentRouting:
    """Test escalation agent routing."""

    def test_ends_when_no_human_needed(self):
        state = make_state(needs_human=False)
        assert route_after_escalation(state) == "END"

    def test_routes_to_human_when_flagged(self):
        state = make_state(needs_human=True)
        assert route_after_escalation(state) == "human_handoff_agent"


# ── State Tests ──────────────────────────────────────────────────────

class TestAgentState:
    """Test state structure and field validation."""

    def test_state_has_required_fields(self):
        state = make_state()
        required_fields = [
            "query", "intent", "context", "response",
            "confidence", "needs_human", "conversation_history",
            "current_agent", "escalation_reason"
        ]
        for field in required_fields:
            assert field in state, f"Missing field: {field}"

    def test_conversation_history_is_list(self):
        state = make_state()
        assert isinstance(state["conversation_history"], list)

    def test_confidence_default_is_valid(self):
        state = make_state()
        assert 0.0 <= state["confidence"] <= 1.0

    def test_needs_human_default_is_false(self):
        state = make_state()
        assert state["needs_human"] is False


# ── Vector Store Tests ───────────────────────────────────────────────

class TestVectorStore:
    """Test FAQ retrieval from vector store."""

    def test_retrieve_context_returns_tuple(self):
        from agents.vector_store import get_vector_store, retrieve_context
        collection = get_vector_store()
        context, confidence = retrieve_context("reset password", collection)
        assert isinstance(context, str)
        assert isinstance(confidence, float)

    def test_confidence_between_zero_and_one(self):
        from agents.vector_store import get_vector_store, retrieve_context
        collection = get_vector_store()
        _, confidence = retrieve_context("random query about nothing", collection)
        assert 0.0 <= confidence <= 1.0

    def test_relevant_query_returns_context(self):
        from agents.vector_store import get_vector_store, retrieve_context
        collection = get_vector_store()
        context, confidence = retrieve_context("how to reset password", collection)
        assert len(context) > 0

    def test_knowledge_base_loads_correctly(self):
        from agents.vector_store import get_vector_store
        from data.knowledge_base import FAQ_DATA
        collection = get_vector_store()
        assert collection.count() == len(FAQ_DATA)
