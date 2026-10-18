"""
Evaluation Runner.

Orchestrates running one or more EvalSamples through the scorer pipeline
and persisting results to the storage layer.

Usage
-----
1. Direct API call evaluation (real-time):
       runner = EvalRunner()
       report = runner.evaluate_from_api_response(api_response_dict)

2. Batch dataset evaluation:
       runner = EvalRunner()
       suite = runner.run_suite(dataset_path="eval/data/golden_set.json")
       print(suite.summary())

Dataset format (golden_set.json)
---------------------------------
[
  {
    "query": "How do I reset my password?",
    "expected_response": "...",           # optional
    "expected_intent": "faq",
    "context": "..."                      # optional RAG context
  },
  ...
]
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from eval.metrics.scorers import (
    DEFAULT_SCORERS,
    EvalReport,
    EvalSample,
    run_all_scorers,
)
from eval.storage.result_store import ResultStore

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Suite-level container
# ---------------------------------------------------------------------------

@dataclass
class EvalSuite:
    """Results for a batch evaluation run."""

    suite_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    dataset_name: str = "unnamed"
    reports: list[EvalReport] = field(default_factory=list)
    run_at: float = field(default_factory=time.time)

    def summary(self) -> dict:
        if not self.reports:
            return {"suite_id": self.suite_id, "total": 0}

        scores = [r.overall_score for r in self.reports]
        passed_count = sum(1 for r in self.reports if r.passed)

        # Per-metric averages
        metric_scores: dict[str, list[float]] = {}
        for report in self.reports:
            for result in report.results:
                metric_scores.setdefault(result.metric_name, []).append(result.score)

        metric_averages = {
            m: round(sum(v) / len(v), 4) for m, v in metric_scores.items()
        }

        return {
            "suite_id": self.suite_id,
            "dataset_name": self.dataset_name,
            "total_samples": len(self.reports),
            "passed": passed_count,
            "failed": len(self.reports) - passed_count,
            "pass_rate": round(passed_count / len(self.reports), 4),
            "avg_overall_score": round(sum(scores) / len(scores), 4),
            "min_score": round(min(scores), 4),
            "max_score": round(max(scores), 4),
            "metric_averages": metric_averages,
            "run_at": self.run_at,
        }


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

class EvalRunner:
    """
    Central class that ties together sampling, scoring, and storing.

    Parameters
    ----------
    store : ResultStore, optional
        Where to persist results. Defaults to SQLite-backed store.
    scorers : list, optional
        Which scorer instances to run. Defaults to all DEFAULT_SCORERS.
    """

    def __init__(
        self,
        store: Optional[ResultStore] = None,
        scorers=None,
    ):
        self.store = store or ResultStore()
        self.scorers = scorers or DEFAULT_SCORERS

    # ------------------------------------------------------------------
    # Single-sample evaluation
    # ------------------------------------------------------------------

    def evaluate_sample(self, sample: EvalSample) -> EvalReport:
        """Score one EvalSample and persist the report."""
        report = run_all_scorers(sample, self.scorers)
        self.store.save_report(report)
        logger.info(
            "Evaluated sample session=%s overall=%.3f passed=%s",
            sample.session_id or "?",
            report.overall_score,
            report.passed,
        )
        return report

    def evaluate_from_api_response(
        self,
        api_response: dict,
        expected_intent: str = "",
        context: str = "",
    ) -> EvalReport:
        """
        Convenience wrapper: build an EvalSample from the dict returned
        by cx_agent's POST /chat endpoint and run evaluation.
        """
        sample = EvalSample(
            query=api_response.get("query", ""),
            response=api_response.get("response", ""),
            context=context,
            expected_intent=expected_intent,
            actual_intent=api_response.get("intent", ""),
            latency_ms=api_response.get("processing_time_ms", 0.0),
            model_name=api_response.get("model_name", "gemini"),
            agent_used=api_response.get("current_agent", "unknown"),
            session_id=api_response.get("session_id", str(uuid.uuid4())),
        )
        return self.evaluate_sample(sample)

    # ------------------------------------------------------------------
    # Batch / dataset evaluation
    # ------------------------------------------------------------------

    def run_suite(
        self,
        dataset: Optional[list[dict]] = None,
        dataset_path: Optional[str] = None,
        suite_name: str = "manual",
    ) -> EvalSuite:
        """
        Run evaluation over a dataset.

        Parameters
        ----------
        dataset : list[dict], optional
            In-memory list of test cases (see module docstring for schema).
        dataset_path : str, optional
            Path to a JSON file with test cases.
        suite_name : str
            Human-readable label stored in the suite.

        Returns
        -------
        EvalSuite with all results populated.
        """
        if dataset_path:
            path = Path(dataset_path)
            if not path.exists():
                raise FileNotFoundError(f"Dataset not found: {dataset_path}")
            with open(path) as f:
                dataset = json.load(f)
            suite_name = path.stem

        if not dataset:
            raise ValueError("No dataset provided — pass `dataset` or `dataset_path`.")

        suite = EvalSuite(dataset_name=suite_name)

        for idx, case in enumerate(dataset):
            sample = EvalSample(
                query=case.get("query", ""),
                response=case.get("response", ""),        # may be empty for live eval
                context=case.get("context", ""),
                expected_intent=case.get("expected_intent", ""),
                actual_intent=case.get("actual_intent", ""),
                latency_ms=case.get("latency_ms", 0.0),
                model_name=case.get("model_name", "unknown"),
                agent_used=case.get("agent_used", "unknown"),
                session_id=case.get("session_id", str(uuid.uuid4())),
            )

            # If no response in dataset, skip scoring (live evaluation mode)
            if not sample.response.strip():
                logger.warning("Case %d has no response — skipping.", idx)
                continue

            report = self.evaluate_sample(sample)
            suite.reports.append(report)

        summary = suite.summary()
        logger.info(
            "Suite '%s' complete: %d samples, pass_rate=%.1f%%, avg=%.3f",
            suite_name,
            summary["total_samples"],
            summary["pass_rate"] * 100,
            summary["avg_overall_score"],
        )
        return suite

    # ------------------------------------------------------------------
    # Quick health check / smoke test
    # ------------------------------------------------------------------

    def smoke_test(self) -> dict:
        """
        Run a built-in smoke test with synthetic samples.
        Useful for CI pipelines and deployment health checks.
        """
        synthetic = [
            {
                "query": "How do I reset my password?",
                "response": (
                    "To reset your password, click 'Forgot Password' on the login page. "
                    "You will receive an email with a reset link valid for 24 hours. "
                    "Click the link and follow the instructions to set a new password."
                ),
                "context": "Password reset requires email verification. Link expires in 24 hours.",
                "expected_intent": "faq",
                "actual_intent": "faq",
                "latency_ms": 850,
                "model_name": "gemini-pro",
                "agent_used": "faq_agent",
            },
            {
                "query": "I am extremely angry about my order being delayed for two weeks!!",
                "response": (
                    "I sincerely apologise for the delay with your order. "
                    "This is unacceptable and I understand your frustration. "
                    "Let me escalate this to our priority support team immediately."
                ),
                "context": "Order delays beyond 5 business days qualify for priority handling.",
                "expected_intent": "escalation",
                "actual_intent": "escalation",
                "latency_ms": 1200,
                "model_name": "gemini-pro",
                "agent_used": "escalation_agent",
            },
            {
                "query": "I need to speak to a human agent right now.",
                "response": (
                    "Of course. I am connecting you to a live agent now. "
                    "Please hold — average wait time is 3 minutes."
                ),
                "context": "",
                "expected_intent": "human_handoff",
                "actual_intent": "human_handoff",
                "latency_ms": 600,
                "model_name": "gemini-pro",
                "agent_used": "human_handoff_agent",
            },
        ]

        suite = self.run_suite(dataset=synthetic, suite_name="smoke_test")
        return suite.summary()
