"""
FAQ Agent — handles common customer questions using RAG.

Flow:
1. Retrieve relevant context from vector store (RAG)
2. Generate answer using Gemini + context
3. If confidence too low, escalate to human handoff

Maps to Coinbase JD: "information retrieval" and "RAG pipelines"
"""

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from agents.state import AgentState
from agents.vector_store import get_vector_store, retrieve_context
import os
from dotenv import load_dotenv

load_dotenv()

CONFIDENCE_THRESHOLD = 0.3

FAQ_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful customer support agent.
Use the provided context to answer the customer's question accurately and concisely.
If the context does not contain enough information, say so honestly.
Keep responses friendly, clear, and under 100 words.

Context from knowledge base:
{context}
"""),
    ("human", "Customer question: {query}")
])


def faq_agent_node(state: AgentState) -> AgentState:
    """
    FAQ Agent node: retrieves context and generates answer using RAG.
    RAG = Retrieval (ChromaDB) + Augmented (context in prompt) + Generation (Gemini)
    """
    print(f"\n[FAQ Agent] Processing query: {state['query']}")

    collection = get_vector_store()
    context, confidence = retrieve_context(state["query"], collection)

    print(f"[FAQ Agent] Retrieved context with confidence: {confidence:.2f}")

    if confidence < CONFIDENCE_THRESHOLD:
        print(f"[FAQ Agent] Low confidence ({confidence:.2f}), flagging for human handoff")
        return {
            **state,
            "context": context,
            "confidence": confidence,
            "needs_human": True,
            "current_agent": "faq_agent",
            "conversation_history": [f"FAQ Agent: Low confidence ({confidence:.2f}), escalating to human"]
        }

    llm = ChatGoogleGenerativeAI(
        model="gemini-3.8-flash",
        temperature=0.1,
        google_api_key=os.getenv("GEMINI_API_KEY")
    )

    chain = FAQ_PROMPT | llm
    result = chain.invoke({"context": context, "query": state["query"]})
    response = (result.content if isinstance(result.content, str) else result.content[0]["text"]).strip()

    print(f"[FAQ Agent] Generated response: {response[:100]}...")

    return {
        **state,
        "context": context,
        "confidence": confidence,
        "response": response,
        "needs_human": False,
        "current_agent": "faq_agent",
        "conversation_history": [f"FAQ Agent responded with confidence {confidence:.2f}"]
    }


def route_after_faq(state: AgentState) -> str:
    """After FAQ agent, check if human handoff is needed."""
    if state.get("needs_human", False):
        print("[Router] FAQ confidence too low, routing to human handoff")
        return "human_handoff_agent"
    print("[Router] FAQ handled successfully, ending")
    return "END"
