"""
Quick runner to test the multi-agent system without starting the API.
Run this first to verify everything works before starting FastAPI.

Usage:
    python run.py
"""

from agents.graph import get_graph
from agents.state import AgentState


def run_query(query: str) -> dict:
    """Run a single query through the multi-agent pipeline."""
    
    print("\n" + "="*60)
    print(f"QUERY: {query}")
    print("="*60)
    
    graph = get_graph()
    
    initial_state: AgentState = {
        "query": query,
        "intent": "",
        "context": "",
        "response": "",
        "confidence": 1.0,
        "needs_human": False,
        "conversation_history": [],
        "current_agent": "",
        "escalation_reason": ""
    }
    
    result = graph.invoke(initial_state)
    
    print("\n── RESULT ──────────────────────────────────────────────────")
    print(f"Intent:        {result['intent']}")
    print(f"Handled by:    {result['current_agent']}")
    print(f"Confidence:    {result['confidence']:.2f}")
    print(f"Needs human:   {result['needs_human']}")
    print(f"\nResponse:\n{result['response']}")
    print("="*60)
    
    return result


if __name__ == "__main__":
    
    # Test cases covering all agent paths
    test_queries = [
        # FAQ agent path
        "How do I reset my password?",
        
        # FAQ agent path — different topic
        "What payment methods do you accept?",
        
        # Escalation agent path
        "The API keeps returning 500 errors and it's been happening for 3 days",
        
        # Human handoff path
        "My account was hacked and someone stole money from me, I need immediate help",
        
        # Low confidence — should trigger human handoff from FAQ
        "I need help with something very specific about my enterprise contract SLA terms"
    ]
    
    for query in test_queries:
        run_query(query)
        print()
