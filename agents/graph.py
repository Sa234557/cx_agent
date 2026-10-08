"""
LangGraph Graph Definition — the core of the multi-agent system.

This file wires all agents together into a directed graph.
Each node is an agent. Edges define how agents connect.
Conditional edges define routing logic based on state.

Graph structure:
    START
      |
   supervisor  <-- classifies intent
      |
   [conditional routing based on intent]
      |          |              |
  faq_agent  escalation  human_handoff
      |          |              |
   [conditional: low confidence?]
      |                         |
     END              human_handoff_agent
                               |
                              END

Maps to Coinbase JD: "orchestration layer that manages state transitions"
"""

from langgraph.graph import StateGraph, END
from agents.state import AgentState
from agents.supervisor import supervisor_node, route_after_supervisor
from agents.faq_agent import faq_agent_node, route_after_faq
from agents.escalation_agent import escalation_agent_node, route_after_escalation
from agents.human_handoff_agent import human_handoff_agent_node


def build_graph():
    """
    Build and compile the multi-agent LangGraph.
    
    Returns a compiled graph ready to invoke.
    """
    
    # Initialize graph with our state schema
    graph = StateGraph(AgentState)
    
    # ── Add Nodes (each node is one agent) ──────────────────────────
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("faq_agent", faq_agent_node)
    graph.add_node("escalation_agent", escalation_agent_node)
    graph.add_node("human_handoff_agent", human_handoff_agent_node)
    
    # ── Add Edges (define flow between agents) ───────────────────────
    
    # Entry point: always start at supervisor
    graph.set_entry_point("supervisor")
    
    # After supervisor: conditional routing based on intent
    graph.add_conditional_edges(
        "supervisor",
        route_after_supervisor,
        {
            "faq_agent": "faq_agent",
            "escalation_agent": "escalation_agent",
            "human_handoff_agent": "human_handoff_agent"
        }
    )
    
    # After FAQ agent: check confidence, maybe route to human handoff
    graph.add_conditional_edges(
        "faq_agent",
        route_after_faq,
        {
            "human_handoff_agent": "human_handoff_agent",
            "END": END
        }
    )
    
    # After escalation agent: check if human needed
    graph.add_conditional_edges(
        "escalation_agent",
        route_after_escalation,
        {
            "human_handoff_agent": "human_handoff_agent",
            "END": END
        }
    )
    
    # Human handoff always ends the graph
    graph.add_edge("human_handoff_agent", END)
    
    # Compile and return
    compiled = graph.compile()
    print("[Graph] Multi-agent graph compiled successfully")
    return compiled


# Singleton graph instance — build once, reuse
_graph = None

def get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph
