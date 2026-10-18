"""
Evaluation & Observability REST API.

Mounts onto the existing cx_agent FastAPI app at the /eval prefix.

Endpoints
---------
POST  /eval/evaluate            — Evaluate a single prompt-response pair
POST  /eval/suite               — Run a batch evaluation suite
POST  /eval/smoke               — Built-in smoke test (CI-friendly)
GET   /eval/dashboard           — Aggregated metrics dashboard
GET   /eval/reports             — Paginated list of past evaluations
GET   /eval/reports/{id}        — Single report detail (with metric breakdown)
GET   /eval/metrics/{name}/trend — Time-series scores for one metric
GET   /eval/health              — Eval platform health check

All responses are JSON.  HTTP 422 on validation errors, 500 on unexpected
failures.
"""

from __future__ import annotations

import logging
import time
from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from eval.metrics.scorers import EvalSample
from eval.runners.eval_runner import EvalRunner, EvalSuite
from eval.storage.result_store import ResultStore

logger = logging.getLogger(__name__)

# Shared singletons (created once, reused across requests)
_store = ResultStore()
_runner = EvalRunner(store=_store)

router = APIRouter(prefix="/eval", tags=["Evaluation & Observability"])


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class EvaluateRequest(BaseModel):
    """Evaluate a single prompt-response pair."""
    query: str = Field(..., min_length=1, description="The user's input")
    response: str = Field(..., min_length=1, description="The LLM / agent's output")
    context: str = Field("", description="RAG / retrieval context used (if any)")
    expected_intent: str = Field("", description="Ground-truth intent label")
    actual_intent: str = Field("", description="Intent classified by the system")
    latency_ms: float = Field(0.0, ge=0, description="End-to-end latency in ms")
    model_name: str = Field("unknown", description="Model that produced the response")
    agent_used: str = Field("unknown", description="Agent that handled the query")
    session_id: str = Field("", description="Session ID for correlation")


class SuiteRequest(BaseModel):
    """Batch evaluation over an in-memory dataset."""
    dataset: list[dict] = Field(
        ...,
        description="List of test cases (query, response, expected_intent, …)",
    )
    suite_name: str = Field("api_suite", description="Human-readable suite label")


class EvaluateResponse(BaseModel):
    session_id: str
    overall_score: float
    passed: bool
    metrics: list[dict]
    evaluated_at: float


class SuiteResponse(BaseModel):
    suite_id: str
    dataset_name: str
    total_samples: int
    passed: int
    failed: int
    pass_rate: float
    avg_overall_score: float
    metric_averages: dict


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.get("/health")
def eval_health():
    """Eval platform liveness check."""
    count = _store.count()
    return {
        "status": "ok",
        "platform": "LLM Evaluation & Observability",
        "total_evaluations_stored": count,
        "scorers": [
            "relevance",
            "faithfulness",
            "toxicity",
            "response_length",
            "latency",
            "intent_accuracy",
        ],
    }


@router.post("/evaluate", response_model=EvaluateResponse)
def evaluate_single(req: EvaluateRequest):
    """
    Score a single prompt-response pair against all metrics.

    Typical usage: hook this into the cx_agent /chat endpoint so every
    production response is auto-evaluated (shadow evaluation).
    """
    sample = EvalSample(
        query=req.query,
        response=req.response,
        context=req.context,
        expected_intent=req.expected_intent,
        actual_intent=req.actual_intent,
        latency_ms=req.latency_ms,
        model_name=req.model_name,
        agent_used=req.agent_used,
        session_id=req.session_id or "",
    )
    try:
        report = _runner.evaluate_sample(sample)
    except Exception as exc:
        logger.exception("Evaluation failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))

    return EvaluateResponse(
        session_id=sample.session_id,
        overall_score=report.overall_score,
        passed=report.passed,
        metrics=[r.to_dict() for r in report.results],
        evaluated_at=time.time(),
    )


@router.post("/suite", response_model=SuiteResponse)
def run_suite(req: SuiteRequest):
    """
    Batch evaluation over a dataset provided inline.

    Useful for regression testing: send your golden test set and get
    back a pass/fail summary with per-metric averages.
    """
    try:
        suite: EvalSuite = _runner.run_suite(
            dataset=req.dataset, suite_name=req.suite_name
        )
        summary = suite.summary()
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        logger.exception("Suite failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))

    return SuiteResponse(
        suite_id=summary["suite_id"],
        dataset_name=summary["dataset_name"],
        total_samples=summary["total_samples"],
        passed=summary["passed"],
        failed=summary["failed"],
        pass_rate=summary["pass_rate"],
        avg_overall_score=summary["avg_overall_score"],
        metric_averages=summary["metric_averages"],
    )


@router.post("/smoke")
def smoke_test():
    """
    Run built-in synthetic test cases.
    Returns immediately — good for CI/CD health gates.
    """
    try:
        summary = _runner.smoke_test()
    except Exception as exc:
        logger.exception("Smoke test failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))
    return {"status": "ok" if summary["pass_rate"] >= 0.8 else "degraded", **summary}


@router.get("/dashboard")
def dashboard(since_hours: float = Query(24.0, ge=0, description="Look-back window in hours")):
    """
    Aggregated metrics for the monitoring dashboard.

    Includes: overall pass rate, avg score, per-agent breakdown,
    per-metric averages, score distribution, and hourly trend.
    """
    since = time.time() - since_hours * 3600
    try:
        stats = _store.get_dashboard_stats(since=since)
    except Exception as exc:
        logger.exception("Dashboard query failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))
    return {"window_hours": since_hours, **stats}


@router.get("/reports")
def list_reports(
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    model_name: Optional[str] = None,
    agent_used: Optional[str] = None,
    passed: Optional[bool] = None,
):
    """
    Paginated list of past evaluation reports.

    Supports filtering by model_name, agent_used, and pass/fail status.
    """
    try:
        rows = _store.list_reports(
            limit=limit,
            offset=offset,
            model_name=model_name,
            agent_used=agent_used,
            passed=passed,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    return {
        "total_returned": len(rows),
        "limit": limit,
        "offset": offset,
        "reports": rows,
    }


@router.get("/reports/{report_id}")
def get_report(report_id: int):
    """
    Detailed view of a single evaluation report, including per-metric
    scores and reasoning text.
    """
    report = _store.get_report(report_id)
    if not report:
        raise HTTPException(status_code=404, detail=f"Report {report_id} not found.")
    return report


@router.get("/metrics/{metric_name}/trend")
def metric_trend(
    metric_name: str,
    limit: int = Query(100, ge=1, le=500),
):
    """
    Time-series scores for a specific metric.

    Use this to render a line chart of how e.g. 'faithfulness' has
    trended over recent evaluations.

    Valid metric names: relevance, faithfulness, toxicity,
    response_length, latency, intent_accuracy
    """
    valid = {
        "relevance", "faithfulness", "toxicity",
        "response_length", "latency", "intent_accuracy",
    }
    if metric_name not in valid:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown metric '{metric_name}'. Valid: {sorted(valid)}",
        )
    try:
        trend = _store.get_metric_trend(metric_name=metric_name, limit=limit)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    return {"metric_name": metric_name, "data_points": trend}
