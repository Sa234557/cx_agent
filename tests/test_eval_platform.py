"""
Test suite for the LLM Evaluation & Observability Platform.

Tests cover:
  - Individual metric scorers (unit tests)
  - EvalRunner batch and single-sample evaluation
  - ResultStore persistence and querying
  - API endpoint contracts

Run with:
    pytest tests/test_eval_platform.py -v
"""

from __future__ import annotations

import json
import os
import tempfile
import time

import pytest

from eval.metrics.scorers import (
    EvalSample,
    FaithfulnessScorer,
    IntentAccuracyScorer,
    LatencyScorer,
    LengthScorer,
    RelevanceScorer,
    ToxicityScorer,
    run_all_scorers,
)
from eval.runners.eval_runner import EvalRunner
from eval.storage.result_store import ResultStore


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_db(tmp_path):
    """ResultStore backed by a temporary SQLite file."""
    db_path = str(tmp_path / "test_eval.db")
    return ResultStore(db_path=db_path)


@pytest.fixture
def runner(tmp_db):
    return EvalRunner(store=tmp_db)


@pytest.fixture
def good_sample():
    return EvalSample(
        query="How do I reset my password?",
        response=(
            "To reset your password, click Forgot Password on the login page. "
            "You will receive an email with a reset link valid for 24 hours."
        ),
        context="Password reset requires email verification. Link valid 24 hours.",
        expected_intent="faq",
        actual_intent="faq",
        latency_ms=900,
        model_name="gemini-pro",
        agent_used="faq_agent",
        session_id="test-session-001",
    )


@pytest.fixture
def toxic_sample():
    return EvalSample(
        query="This is terrible!!!",
        response="I hate this stupid idiot system!!! It is completely dumb and moron-designed!!!",
        context="",
        latency_ms=500,
    )


# ---------------------------------------------------------------------------
# Metric scorer — unit tests
# ---------------------------------------------------------------------------

class TestRelevanceScorer:
    def test_high_relevance(self):
        sample = EvalSample(
            query="How do I reset password?",
            response="To reset your password click the password reset link.",
        )
        result = RelevanceScorer().score(sample)
        assert result.metric_name == "relevance"
        assert result.score > 0.3, "Same-domain query should have decent relevance"

    def test_off_topic_response(self):
        sample = EvalSample(
            query="How do I reset my password?",
            response="The weather in London is sunny today with mild temperatures.",
        )
        result = RelevanceScorer().score(sample)
        assert result.score < 0.5

    def test_empty_query_tokens(self):
        sample = EvalSample(
            query="is it a",   # all stop-words
            response="yes it is a thing",
        )
        result = RelevanceScorer().score(sample)
        assert 0.0 <= result.score <= 1.0  # should not crash


class TestFaithfulnessScorer:
    def test_grounded_response(self):
        sample = EvalSample(
            query="How long is the refund process?",
            response="The refund process takes 5 to 7 business days.",
            context="Refunds are processed within 5 to 7 business days after approval.",
        )
        result = FaithfulnessScorer().score(sample)
        assert result.score >= 0.3

    def test_no_context_defaults_neutral(self):
        sample = EvalSample(query="q", response="r", context="")
        result = FaithfulnessScorer().score(sample)
        assert result.score == 0.7
        assert result.passed is True

    def test_empty_response(self):
        sample = EvalSample(query="q", response="", context="some context")
        result = FaithfulnessScorer().score(sample)
        assert result.score == 0.0
        assert result.passed is False


class TestToxicityScorer:
    def test_clean_response(self, good_sample):
        result = ToxicityScorer().score(good_sample)
        assert result.score >= 0.9
        assert result.passed is True

    def test_toxic_response(self, toxic_sample):
        result = ToxicityScorer().score(toxic_sample)
        assert result.score < 0.5
        assert result.passed is False

    def test_mild_profanity_reduces_score(self):
        sample = EvalSample(query="q", response="Damn, that is a good question.")
        result = ToxicityScorer().score(sample)
        assert result.score < 1.0


class TestLengthScorer:
    def test_optimal_length(self):
        response = " ".join(["word"] * 70)
        sample = EvalSample(query="q", response=response)
        result = LengthScorer().score(sample)
        assert result.score >= 0.7
        assert result.passed is True

    def test_too_short(self):
        sample = EvalSample(query="q", response="OK.")
        result = LengthScorer().score(sample)
        assert result.score < 0.5
        assert result.passed is False

    def test_too_long(self):
        response = " ".join(["word"] * 500)
        sample = EvalSample(query="q", response=response)
        result = LengthScorer().score(sample)
        assert result.score < 0.8


class TestLatencyScorer:
    def test_excellent_latency(self):
        sample = EvalSample(query="q", response="r", latency_ms=500)
        result = LatencyScorer().score(sample)
        assert result.score == 1.0

    def test_poor_latency(self):
        sample = EvalSample(query="q", response="r", latency_ms=9000)
        result = LatencyScorer().score(sample)
        assert result.score == 0.0
        assert result.passed is False

    def test_no_latency_defaults_to_one(self):
        sample = EvalSample(query="q", response="r", latency_ms=0)
        result = LatencyScorer().score(sample)
        assert result.score == 1.0

    def test_intermediate_latency(self):
        sample = EvalSample(query="q", response="r", latency_ms=5000)
        result = LatencyScorer().score(sample)
        assert 0.0 < result.score < 1.0


class TestIntentAccuracyScorer:
    def test_correct_intent(self):
        sample = EvalSample(query="q", response="r", expected_intent="faq", actual_intent="faq")
        result = IntentAccuracyScorer().score(sample)
        assert result.score == 1.0
        assert result.passed is True

    def test_wrong_intent(self):
        sample = EvalSample(query="q", response="r", expected_intent="faq", actual_intent="escalation")
        result = IntentAccuracyScorer().score(sample)
        assert result.score == 0.0
        assert result.passed is False

    def test_no_expected_intent_skipped(self):
        sample = EvalSample(query="q", response="r", expected_intent="", actual_intent="faq")
        result = IntentAccuracyScorer().score(sample)
        assert result.score == 0.8
        assert result.passed is True


# ---------------------------------------------------------------------------
# run_all_scorers integration
# ---------------------------------------------------------------------------

class TestRunAllScorers:
    def test_returns_all_metrics(self, good_sample):
        report = run_all_scorers(good_sample)
        metric_names = {r.metric_name for r in report.results}
        expected = {
            "relevance", "faithfulness", "toxicity",
            "response_length", "latency", "intent_accuracy",
        }
        assert expected.issubset(metric_names)

    def test_overall_score_is_average(self, good_sample):
        report = run_all_scorers(good_sample)
        computed = sum(r.score for r in report.results) / len(report.results)
        assert abs(report.overall_score - computed) < 0.001

    def test_good_sample_passes(self, good_sample):
        report = run_all_scorers(good_sample)
        assert report.overall_score > 0.5

    def test_to_dict_serialisable(self, good_sample):
        report = run_all_scorers(good_sample)
        d = report.to_dict()
        json_str = json.dumps(d)   # must not raise
        assert isinstance(json_str, str)


# ---------------------------------------------------------------------------
# ResultStore persistence
# ---------------------------------------------------------------------------

class TestResultStore:
    def test_save_and_retrieve(self, tmp_db, good_sample):
        report = run_all_scorers(good_sample)
        report_id = tmp_db.save_report(report)
        assert isinstance(report_id, int) and report_id > 0

        retrieved = tmp_db.get_report(report_id)
        assert retrieved is not None
        assert retrieved["query"] == good_sample.query
        assert len(retrieved["metrics"]) == len(report.results)

    def test_get_nonexistent_returns_none(self, tmp_db):
        assert tmp_db.get_report(999999) is None

    def test_list_reports_pagination(self, tmp_db, good_sample):
        runner = EvalRunner(store=tmp_db)
        for _ in range(5):
            runner.evaluate_sample(good_sample)

        page1 = tmp_db.list_reports(limit=3, offset=0)
        page2 = tmp_db.list_reports(limit=3, offset=3)
        assert len(page1) == 3
        assert len(page2) >= 2

    def test_dashboard_stats_structure(self, tmp_db, good_sample):
        runner = EvalRunner(store=tmp_db)
        runner.evaluate_sample(good_sample)
        stats = tmp_db.get_dashboard_stats()

        required_keys = {
            "total_evaluations", "pass_rate", "avg_overall_score",
            "by_agent", "by_metric", "score_distribution",
        }
        assert required_keys.issubset(stats.keys())
        assert stats["total_evaluations"] >= 1

    def test_count(self, tmp_db, good_sample):
        runner = EvalRunner(store=tmp_db)
        assert tmp_db.count() == 0
        runner.evaluate_sample(good_sample)
        assert tmp_db.count() == 1

    def test_metric_trend(self, tmp_db, good_sample):
        runner = EvalRunner(store=tmp_db)
        runner.evaluate_sample(good_sample)
        trend = tmp_db.get_metric_trend("relevance")
        assert len(trend) >= 1
        assert "score" in trend[0]
        assert "timestamp" in trend[0]


# ---------------------------------------------------------------------------
# EvalRunner
# ---------------------------------------------------------------------------

class TestEvalRunner:
    def test_evaluate_sample(self, runner, good_sample):
        report = runner.evaluate_sample(good_sample)
        assert report.overall_score > 0
        assert len(report.results) > 0

    def test_evaluate_from_api_response(self, runner):
        api_response = {
            "session_id": "api-test-001",
            "query": "How do I cancel?",
            "response": "Go to Settings → Subscription → Cancel.",
            "intent": "faq",
            "current_agent": "faq_agent",
            "processing_time_ms": 800,
            "model_name": "gemini-pro",
        }
        report = runner.evaluate_from_api_response(api_response, expected_intent="faq")
        assert report.overall_score >= 0

    def test_run_suite_inline(self, runner):
        dataset = [
            {
                "query": "Reset password?",
                "response": "Click Forgot Password to reset your password.",
                "context": "Password reset link sent by email.",
                "expected_intent": "faq",
                "actual_intent": "faq",
                "latency_ms": 700,
                "agent_used": "faq_agent",
                "model_name": "gemini-pro",
            }
        ]
        suite = runner.run_suite(dataset=dataset, suite_name="unit_test")
        summary = suite.summary()
        assert summary["total_samples"] == 1
        assert "avg_overall_score" in summary

    def test_suite_skips_empty_response(self, runner):
        dataset = [{"query": "q", "response": "", "expected_intent": "faq"}]
        suite = runner.run_suite(dataset=dataset)
        assert suite.summary()["total_samples"] == 0

    def test_smoke_test(self, runner):
        summary = runner.smoke_test()
        assert "total_samples" in summary
        assert summary["total_samples"] == 3
        assert summary["pass_rate"] >= 0.0

    def test_run_suite_missing_dataset_raises(self, runner):
        with pytest.raises(ValueError):
            runner.run_suite()


# ---------------------------------------------------------------------------
# FastAPI endpoints (using TestClient)
# ---------------------------------------------------------------------------

try:
    from fastapi.testclient import TestClient
    from fastapi import FastAPI
    from eval.api.eval_api import router as eval_router

    test_app = FastAPI()
    test_app.include_router(eval_router)
    client = TestClient(test_app)
    HAS_FASTAPI_TEST = True
except Exception:
    HAS_FASTAPI_TEST = False


@pytest.mark.skipif(not HAS_FASTAPI_TEST, reason="FastAPI TestClient unavailable")
class TestEvalAPI:
    def test_health(self):
        resp = client.get("/eval/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "scorers" in data

    def test_evaluate_endpoint(self):
        payload = {
            "query": "How do I reset my password?",
            "response": "Click Forgot Password on the login page to reset your password.",
            "context": "Password reset link sent by email.",
            "expected_intent": "faq",
            "actual_intent": "faq",
            "latency_ms": 800,
            "model_name": "gemini-pro",
            "agent_used": "faq_agent",
        }
        resp = client.post("/eval/evaluate", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert "overall_score" in data
        assert "metrics" in data
        assert isinstance(data["metrics"], list)

    def test_smoke_endpoint(self):
        resp = client.post("/eval/smoke")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] in ("ok", "degraded")

    def test_suite_endpoint(self):
        payload = {
            "suite_name": "api_test",
            "dataset": [
                {
                    "query": "Can I cancel my subscription?",
                    "response": "Yes, go to Settings → Subscription → Cancel to cancel anytime.",
                    "expected_intent": "faq",
                    "actual_intent": "faq",
                    "latency_ms": 750,
                    "agent_used": "faq_agent",
                    "model_name": "gemini-pro",
                }
            ],
        }
        resp = client.post("/eval/suite", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_samples"] == 1

    def test_dashboard_endpoint(self):
        resp = client.get("/eval/dashboard?since_hours=24")
        assert resp.status_code == 200
        data = resp.json()
        assert "total_evaluations" in data

    def test_reports_list_endpoint(self):
        resp = client.get("/eval/reports?limit=10")
        assert resp.status_code == 200
        data = resp.json()
        assert "reports" in data

    def test_metric_trend_endpoint(self):
        resp = client.get("/eval/metrics/relevance/trend?limit=10")
        assert resp.status_code == 200
        data = resp.json()
        assert data["metric_name"] == "relevance"

    def test_invalid_metric_name(self):
        resp = client.get("/eval/metrics/nonexistent_metric/trend")
        assert resp.status_code == 422

    def test_missing_report_404(self):
        resp = client.get("/eval/reports/999999")
        assert resp.status_code == 404
