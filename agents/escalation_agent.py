"""
Escalation Agent — handles complex technical issues.

Maps to Coinbase JD: "state transitions" and "context sharing"
"""

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from agents.state import AgentState
import os
from dotenv import load_dotenv

load_dotenv()

ESCALATION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a senior technical support specialist handling complex issues.
Your job is to:
1. Acknowledge the complexity of the issue
2. Provide structured troubleshooting steps if possible
3. Set clear expectations on resolution timeline
4. Determine if immediate human intervention is needed

Keep response professional, empathetic, and actionable.
End your response with either:
- [RESOLVED]: If you've provided a complete solution
- [NEEDS_HUMAN]: If the issue requires human agent intervention
"""),
    ("human", "Customer issue: {query}")
])


def escalation_agent_node(state: AgentState) -> AgentState:
    """Escalation Agent node: handles complex technical issues."""
    print(f"\n[Escalation Agent] Handling complex issue: {state['query']}")

    llm = ChatGoogleGenerativeAI(
        model="gemini-3.8-flash",
        temperature=0.2,
        google_api_key=os.getenv("GEMINI_API_KEY")
    )

    chain = ESCALATION_PROMPT | llm
    result = chain.invoke({"query": state["query"]})
    response = (result.content if isinstance(result.content, str) else result.content[0]["text"]).strip()

    needs_human = "[NEEDS_HUMAN]" in response
    response = response.replace("[RESOLVED]", "").replace("[NEEDS_HUMAN]", "").strip()

    print(f"[Escalation Agent] Needs human: {needs_human}")

    return {
        **state,
        "response": response,
        "needs_human": needs_human,
        "escalation_reason": "Complex technical issue requiring investigation",
        "current_agent": "escalation_agent",
        "conversation_history": [
            f"Escalation Agent handled issue. Human needed: {needs_human}"
        ]
    }


def route_after_escalation(state: AgentState) -> str:
    """Route to human handoff if escalation agent flagged it."""
    if state.get("needs_human", False):
        return "human_handoff_agent"
    return "END"
