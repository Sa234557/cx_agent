"""
Supervisor Agent — the orchestrator of the multi-agent system.

Responsibilities:
1. Classify the customer intent
2. Route to the appropriate specialist agent
3. Decide if human handoff is needed

Maps to Coinbase JD: "intent routing" and "orchestration layer"
"""

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from agents.state import AgentState
import os
from dotenv import load_dotenv

load_dotenv()


def get_llm():
    return ChatGoogleGenerativeAI(
        model="gemini-3.8-flash",   # Free model on Google AI Studio
        temperature=0,
        google_api_key=os.getenv("GEMINI_API_KEY")
    )


SUPERVISOR_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a customer support supervisor agent. 
Your job is to classify customer queries into one of these intents:
- faq: Common questions that can be answered from knowledge base
- escalation: Complex issues requiring deeper investigation  
- human_handoff: Sensitive issues requiring human agent

Respond with ONLY one word: faq, escalation, or human_handoff.

Examples:
"How do I reset my password?" -> faq
"I want a refund" -> faq
"My account was hacked and I lost money" -> human_handoff
"I've been waiting 3 weeks for resolution and nobody is helping" -> human_handoff
"My integration is broken and costing us $10k/day" -> escalation
"The API keeps returning 500 errors randomly" -> escalation
"""),
    ("human", "Customer query: {query}")
])


def supervisor_node(state: AgentState) -> AgentState:
    """
    Supervisor node: classifies intent and routes to correct agent.
    This is a LangGraph node — a function that takes state and returns updated state.
    """
    print(f"\n[Supervisor] Classifying query: {state['query']}")

    llm = get_llm()
    chain = SUPERVISOR_PROMPT | llm

    result = chain.invoke({"query": state["query"]})
    content_text = result.content if isinstance(result.content, str) else result.content[0]["text"]
    intent = content_text.strip().lower()

    valid_intents = ["faq", "escalation", "human_handoff"]
    if intent not in valid_intents:
        intent = "faq"

    print(f"[Supervisor] Intent classified as: {intent}")

    return {
        **state,
        "intent": intent,
        "current_agent": "supervisor",
        "conversation_history": [f"Supervisor classified intent as: {intent}"]
    }


def route_after_supervisor(state: AgentState) -> str:
    """Conditional edge — decides which agent to route to next."""
    intent = state.get("intent", "faq")
    routing_map = {
        "faq": "faq_agent",
        "escalation": "escalation_agent",
        "human_handoff": "human_handoff_agent"
    }
    next_node = routing_map.get(intent, "faq_agent")
    print(f"[Router] Routing to: {next_node}")
    return next_node
