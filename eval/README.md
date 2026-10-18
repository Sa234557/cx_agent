# LLM Evaluation & Observability Platform

> **LLMOps add-on for `cx_agent`** — runs 100% locally, zero paid APIs, zero extra infrastructure.

## What it does

| Capability | Details |
|---|---|
| **Metric scoring** | 6 automated scorers: relevance, faithfulness, toxicity, response length, latency, intent accuracy |
| **Shadow evaluation** | Every `/chat` response is auto-evaluated asynchronously (no latency added) |
| **Batch testing** | Run evaluation suites over golden datasets to catch regressions |
| **Persistent storage** | SQLite stores all results — query, filter, and trend them |
| **REST API** | Full `/eval/*` router with dashboard, reports, trends, and health |
| **CLI** | Run smoke tests, suites, and one-off evals from the terminal |

## Architecture

```
cx_agent/
└── eval/
    ├── metrics/
    │   └── scorers.py          ← 6 independent metric scorers
    ├── runners/
    │   └── eval_runner.py      ← orchestrates scoring + persistence
    ├── storage/
    │   └── result_store.py     ← SQLite repository (swap for Postgres later)
    ├── api/
    │   └── eval_api.py         ← FastAPI router mounted at /eval
    ├── data/
    │   ├── golden_set.json     ← curated test dataset
    │   └── eval_results.db     ← auto-created SQLite DB
    ├── run_eval.py             ← CLI entrypoint
    └── README.md
```

## Quick Start

### 1. Activate the eval platform

Replace `api/main.py` with the integrated version:

```bash
cp api/main_with_eval.py api/main.py
```

### 2. Start the server

```bash
uvicorn api.main:app --reload
```

### 3. Use the API

```bash
# Health check
curl http://localhost:8000/eval/health

# Evaluate a single response
curl -X POST http://localhost:8000/eval/evaluate \
  -H "Content-Type: application/json" \
  -d '{
    "query": "How do I reset my password?",
    "response": "Click Forgot Password on the login page to receive a reset link.",
    "expected_intent": "faq",
    "actual_intent": "faq",
    "latency_ms": 850,
    "agent_used": "faq_agent"
  }'

# Dashboard metrics (last 24h)
curl "http://localhost:8000/eval/dashboard?since_hours=24"

# Run batch regression test
curl -X POST http://localhost:8000/eval/suite \
  -H "Content-Type: application/json" \
  -d @eval/data/golden_set.json   # wrap dataset in {"dataset": [...]}

# Built-in smoke test (great for CI)
curl -X POST http://localhost:8000/eval/smoke
```

### 4. Use the CLI

```bash
# Smoke test
python eval/run_eval.py smoke

# Golden dataset suite
python eval/run_eval.py suite --dataset eval/data/golden_set.json

# One-off single evaluation
python eval/run_eval.py single \
    --query "How do I cancel?" \
    --response "Go to Settings → Subscription → Cancel." \
    --intent faq

# Dashboard
python eval/run_eval.py dashboard --hours 48
```

### 5. Run tests

```bash
pytest tests/test_eval_platform.py -v
```

## Metric Scorers

| Metric | Method | Threshold | What it catches |
|---|---|---|---|
| **Relevance** | Jaccard token overlap (query ↔ response) | 0.05 | Off-topic responses |
| **Faithfulness** | Bi-gram overlap (context ↔ response) | 0.30 | Hallucination |
| **Toxicity** | Regex pattern matching | 0.50 | Harmful/offensive output |
| **Response Length** | Word count in [10, 300] | 0.50 | Too brief or verbose |
| **Latency** | Normalised ms score | 0.30 | Slow responses |
| **Intent Accuracy** | Exact label match | 0.50 | Routing errors |

## API Reference

| Endpoint | Method | Description |
|---|---|---|
| `/eval/health` | GET | Platform liveness + scorer list |
| `/eval/evaluate` | POST | Score one prompt-response pair |
| `/eval/suite` | POST | Batch evaluation over a dataset |
| `/eval/smoke` | POST | Built-in synthetic smoke test |
| `/eval/dashboard` | GET | Aggregated dashboard stats |
| `/eval/reports` | GET | Paginated list of past evaluations |
| `/eval/reports/{id}` | GET | Single report with metric breakdown |
| `/eval/metrics/{name}/trend` | GET | Time-series scores for one metric |

## Why This Matters for LLMOps

- **Regression prevention** — run the golden set in CI; fail the build on `pass_rate < 0.8`
- **Continuous monitoring** — shadow eval on every production request surfaces drift over time
- **Model comparison** — set `model_name` to compare Gemini vs GPT vs Groq on the same prompts
- **Debugging** — per-metric scores pinpoint *which* quality dimension degraded and *why*

## Extending the Platform

```python
# Add a custom scorer in eval/metrics/scorers.py
class MyCustomScorer:
    NAME = "my_metric"
    THRESHOLD = 0.5

    def score(self, sample: EvalSample) -> MetricResult:
        score = ...  # your logic
        return MetricResult(metric_name=self.NAME, score=score, reasoning="...", passed=score >= self.THRESHOLD)

# Register it
from eval.metrics.scorers import DEFAULT_SCORERS
DEFAULT_SCORERS.append(MyCustomScorer())
```
