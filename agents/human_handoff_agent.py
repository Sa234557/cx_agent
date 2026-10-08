"""
Human Handoff Agent — the human-in-the-loop component.

Maps directly to Coinbase JD: "human participants" and "handoff logic"
Also maps to your Oracle experience: "human-in-the-loop review"
"""

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from agents.state import AgentState
import os
from dotenv import load_dotenv

load_dotenv()

CUSTOMER_RESPONSE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a customer support agent.
The customer's issue requires attention from a human specialist.
Write a warm, empathetic response that:
1. Acknowledges their issue
2. Assures them a human specialist will help
3. Sets expectation of response within 2-4 hours
4. Thanks them for their patience
Keep it under 80 words and genuinely empathetic.
"""),
    ("human", "Customer issue: {query}\nReason for escalation: {reason}")
])

HANDOFF_SUMMARY_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are preparing a handoff summary for a human support agent.
Create a concise structured summary including:
- Issue Type
- Customer Query (verbatim)
- Priority Level (Low/Medium/High/Critical)
- Recommended Action
- Context from previous agents (if any)

Format it clearly so the human agent can act immediately.
"""),
    ("human", """
Customer query: {query}
Escalation reason: {reason}
Previous agent context: {history}
""")
])


def human_handoff_agent_node(state: AgentState) -> AgentState:
    """
    Human Handoff Agent node.
    Two outputs:
    1. Customer-facing: empathetic holding response
    2. Internal: structured handoff summary for human agent
    """
    print(f"\n[Human Handoff Agent] Preparing handoff for: {state['query']}")

    llm = ChatGoogleGenerativeAI(
        model="gemini-3.8-flash",
        temperature=0.3,
        google_api_key=os.getenv("GEMINI_API_KEY")
    )

    reason = state.get("escalation_reason", "")
    if not reason:
        if state.get("confidence", 1.0) < 0.3:
            reason = "Low confidence in automated response — query outside knowledge base"
        else:
            reason = "Sensitive issue requiring human judgment"

    # Generate customer-facing response
    customer_chain = CUSTOMER_RESPONSE_PROMPT | llm
    customer_result = customer_chain.invoke({
        "query": state["query"],
        "reason": reason
    })

    # Generate internal handoff summary
    history_text = "\n".join(state.get("conversation_history", []))
    summary_chain = HANDOFF_SUMMARY_PROMPT | llm
    summary_result = summary_chain.invoke({
        "query": state["query"],
        "reason": reason,
        "history": history_text or "No prior agent context"
    })

    print(f"[Human Handoff Agent] Handoff prepared successfully")

    return {
        **state,
        "response": (customer_result.content if isinstance(customer_result.content, str) else customer_result.content[0]["text"]).strip(),
        "needs_human": True,
        "current_agent": "human_handoff_agent",
        "conversation_history": [
            f"Human Handoff Agent: Escalated. Reason: {reason}",
            f"Handoff Summary:\n{(summary_result.content if isinstance(summary_result.content, str) else summary_result.content[0]['text']).strip()}"
        ]
    }
