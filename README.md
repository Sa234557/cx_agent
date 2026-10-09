# CX Intelligence Multi-Agent System

A production-grade multi-agent customer support system built with **LangGraph**, **LangChain**, **FastAPI**, and **ChromaDB**.

Directly aligned with industry patterns used at companies like Coinbase for conversational AI platforms.

---

## Architecture

```
Customer Query
      │
      ▼
┌─────────────┐
│  Supervisor  │  ← Classifies intent (faq / escalation / human_handoff)
└──────┬──────┘
       │
  ┌────┴─────────────────┐
  │                       │                    │
  ▼                       ▼                    ▼
┌──────────┐    ┌──────────────────┐   ┌──────────────────┐
│ FAQ Agent │    │ Escalation Agent │   │ Human Handoff    │
│  (RAG)   │    │ (Complex issues) │   │ Agent            │
└────┬─────┘    └────────┬─────────┘   └──────────────────┘
     │                   │
     │ low confidence?   │ needs human?
     ▼                   ▼
┌──────────────────────────────┐
│     Human Handoff Agent      │
│  (Empathetic response +      │
│   structured handoff summary)│
└──────────────────────────────┘
```

---

## Tech Stack

| Component | Technology | Why |
|---|---|---|
| Agent Orchestration | LangGraph | State machines for multi-agent workflows |
| LLM | Groq (Llama 3.1 8B) | Free, fast inference |
| Embeddings | sentence-transformers | Free, runs locally |
| Vector Store | ChromaDB | Free, local RAG |
| API Layer | FastAPI | Production-grade REST API |
| Containerization | Docker | Portable deployment |

---

## Setup

### 1. Get free Groq API key
Go to https://console.groq.com → Sign up → Create API Key (no credit card needed)

### 2. Clone and setup
```bash
git clone <your-repo>
cd cx_agent
cp .env.example .env
# Add your GROQ_API_KEY to .env
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run quick test (no API needed)
```bash
python run.py
```

### 5. Start the API
```bash
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

### 6. Or run with Docker
```bash
docker-compose up --build
```

---

## API Endpoints

### POST /chat
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "How do I reset my password?"}'
```

Response:
```json
{
  "session_id": "uuid",
  "query": "How do I reset my password?",
  "response": "To reset your password...",
  "intent": "faq",
  "confidence": 0.87,
  "needs_human": false,
  "current_agent": "faq_agent",
  "processing_time_ms": 342.5
}
```

### GET /health
```bash
curl http://localhost:8000/health
```

### GET /history/{session_id}
```bash
curl http://localhost:8000/history/your-session-id
```

---

## Run Tests
```bash
pytest tests/ -v
```

---

## Key Concepts 

### Why LangGraph over plain LangChain?
LangGraph gives you **stateful, cyclical graphs** — agents can loop, branch, and share state. LangChain alone is linear chains. For multi-agent systems with routing and state transitions, LangGraph is the right tool.

### What is the State?
`AgentState` is a TypedDict that flows through every node. Every agent reads from it and writes to it. This is how agents share context — the FAQ agent's confidence score is read by the router to decide if human handoff is needed.

### What is a Node?
Each agent is a node — a Python function that takes state and returns updated state. Pure functions make testing straightforward.

### What is a Conditional Edge?
Instead of always going to the same next node, conditional edges run a function on the state and return which node to go to next. This is the routing logic.

### Why RAG in the FAQ agent?
Instead of hardcoding answers or fine-tuning a model, RAG retrieves relevant knowledge base entries at runtime. Cheaper, updatable without retraining, and more accurate for specific domain knowledge.

### Human-in-the-loop
When confidence is below threshold (0.3) or the supervisor classifies intent as human_handoff, the system generates both a customer-facing empathetic response AND an internal handoff summary for the human agent — ensuring the human agent has full context immediately.
