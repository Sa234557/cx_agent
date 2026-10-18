"""
LLM Evaluation Metric Scorers.

Each scorer takes a EvalSample and returns a MetricResult with:
  - score: float [0.0 – 1.0]
  - reasoning: human-readable explanation
  - metadata: extra computed data

Design principle: every scorer is stateless, deterministic, and
independently testable.  We do NOT call paid LLM APIs here — all
scorers use local heuristics + lightweight NLP so the eval platform
is 100% free to run.

Scorers:
  - RelevanceScorer   : semantic similarity between query and response
  - FaithfulnessScorer: does the answer stay grounded in the context?
  - ToxicityScorer    : detect harmful / offensive content
  - LengthScorer      : response length within expected bounds
  - LatencyScorer     : normalised latency score
  - IntentAccuracyScorer: was the right agent used for this intent?
"""

from __future__ import annotations

import math
import re
import time
from dataclasses import dataclass, field
from typing import Optional

# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class EvalSample:
    """A single prompt-response pair ready to be scored."""
    query: str
    response: str
    context: str = ""           # Retrieved context / RAG docs
    expected_intent: str = ""   # Ground-truth intent label
    actual_intent: str = ""     # Intent returned by the system
    latency_ms: float = 0.0     # End-to-end latency
    model_name: str = "unknown"
    agent_used: str = "unknown"
    session_id: str = ""
    timestamp: float = field(default_factory=time.time)


@dataclass
class MetricResult:
    """Result of a single metric scorer."""
    metric_name: str
    score: float                # 0.0 (worst) – 1.0 (best)
    reasoning: str
    metadata: dict = field(default_factory=dict)
    passed: bool = True         # True when score >= threshold

    def to_dict(self) -> dict:
        return {
            "metric_name": self.metric_name,
            "score": round(self.score, 4),
            "reasoning": self.reasoning,
            "metadata": self.metadata,
            "passed": self.passed,
        }


@dataclass
class EvalReport:
    """Aggregated results for one EvalSample."""
    sample: EvalSample
    results: list[MetricResult] = field(default_factory=list)
    overall_score: float = 0.0
    passed: bool = True

    def compute_overall(self) -> None:
        if not self.results:
            self.overall_score = 0.0
            self.passed = False
            return
        self.overall_score = round(
            sum(r.score for r in self.results) / len(self.results), 4
        )
        self.passed = all(r.passed for r in self.results)

    def to_dict(self) -> dict:
        return {
            "session_id": self.sample.session_id,
            "query": self.sample.query,
            "response": self.sample.response,
            "model_name": self.sample.model_name,
            "agent_used": self.sample.agent_used,
            "latency_ms": self.sample.latency_ms,
            "timestamp": self.sample.timestamp,
            "overall_score": self.overall_score,
            "passed": self.passed,
            "metrics": [r.to_dict() for r in self.results],
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tokenise(text: str) -> set[str]:
    """Lowercase word tokens, punctuation stripped."""
    return set(re.findall(r"\b[a-z]{2,}\b", text.lower()))


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    union = a | b
    return len(a & b) / len(union) if union else 0.0


# ---------------------------------------------------------------------------
# Individual scorers
# ---------------------------------------------------------------------------

class RelevanceScorer:
    """
    Token-overlap (Jaccard) between the query and the response.
    A relevant answer should reuse key terms from the question.

    Threshold: 0.05  (low — LLMs naturally paraphrase)
    """

    NAME = "relevance"
    THRESHOLD = 0.05

    def score(self, sample: EvalSample) -> MetricResult:
        q_tokens = _tokenise(sample.query)
        r_tokens = _tokenise(sample.response)

        # Remove very common stop-words from denominator
        stops = {"is", "the", "a", "an", "of", "to", "in", "it", "i",
                 "me", "my", "we", "you", "do", "be", "for", "on", "are"}
        q_tokens -= stops
        r_tokens -= stops

        raw = _jaccard(q_tokens, r_tokens)

        # Scale: jaccard of 0.05+ → score of 0.5, 0.20+ → near 1.0
        scaled = min(1.0, raw / 0.12)

        passed = scaled >= self.THRESHOLD
        reasoning = (
            f"Query–response token overlap (Jaccard): {raw:.3f} "
            f"→ relevance score {scaled:.3f}. "
            + ("Sufficient overlap." if passed else "Very low overlap — possible off-topic response.")
        )
        return MetricResult(
            metric_name=self.NAME,
            score=round(scaled, 4),
            reasoning=reasoning,
            metadata={"jaccard": round(raw, 4), "query_tokens": len(q_tokens)},
            passed=passed,
        )


class FaithfulnessScorer:
    """
    Checks whether the response is grounded in the provided context.
    Uses keyword (unigram) overlap between context and response, after
    removing stop-words.  This is intentionally lenient — LLMs naturally
    rephrase source material.

    If no context is available, defaults to a neutral 0.7.

    Threshold: 0.15
    """

    NAME = "faithfulness"
    THRESHOLD = 0.15

    # Common stop-words to exclude from overlap counting
    _STOPS = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "shall", "to", "of", "in", "on", "at",
        "by", "for", "with", "about", "as", "it", "its", "this", "that",
        "and", "or", "but", "not", "if", "so", "we", "you", "i", "my",
        "your", "our", "their", "he", "she", "they", "me", "him", "her",
        "us", "them", "which", "who", "what", "how", "when", "where",
    }

    def _keywords(self, text: str) -> set[str]:
        tokens = set(re.findall(r"\b[a-z]{3,}\b", text.lower()))
        return tokens - self._STOPS

    def score(self, sample: EvalSample) -> MetricResult:
        if not sample.context.strip():
            return MetricResult(
                metric_name=self.NAME,
                score=0.7,
                reasoning="No context provided — faithfulness cannot be measured; defaulting to 0.7.",
                metadata={"skipped": True},
                passed=True,
            )

        ctx_kw = self._keywords(sample.context)
        res_kw = self._keywords(sample.response)

        if not res_kw:
            return MetricResult(
                metric_name=self.NAME,
                score=0.0,
                reasoning="Empty response — cannot evaluate faithfulness.",
                metadata={},
                passed=False,
            )

        shared = ctx_kw & res_kw
        # Precision: what fraction of response keywords also appear in context
        precision = len(shared) / len(res_kw)
        # Recall: what fraction of context keywords the response covers
        recall = len(shared) / len(ctx_kw) if ctx_kw else 0.0
        # F0.5 score — weight precision more (we care about hallucination)
        beta = 0.5
        if precision + recall > 0:
            f_score = (1 + beta**2) * precision * recall / (beta**2 * precision + recall)
        else:
            f_score = 0.0

        score = round(min(1.0, f_score * 1.4), 4)   # slight boost for practical thresholds
        passed = score >= self.THRESHOLD
        reasoning = (
            f"{len(shared)} shared keywords out of {len(res_kw)} in response / "
            f"{len(ctx_kw)} in context. "
            f"Precision={precision:.2f}, Recall={recall:.2f}, Score={score:.3f}. "
            + ("Response appears grounded." if passed else "Low keyword overlap — possible hallucination.")
        )
        return MetricResult(
            metric_name=self.NAME,
            score=score,
            reasoning=reasoning,
            metadata={
                "shared_keywords": sorted(shared)[:10],
                "precision": round(precision, 4),
                "recall": round(recall, 4),
            },
            passed=passed,
        )


class ToxicityScorer:
    """
    Rule-based toxicity detection using curated word/pattern lists.
    Returns score near 1.0 when clean, near 0.0 when toxic.

    Threshold: 0.5
    """

    NAME = "toxicity"
    THRESHOLD = 0.5

    # Curated minimal toxic-signal patterns (kept conservative)
    _PATTERNS = [
        r"\b(hate|kill|attack|destroy|idiot|stupid|dumb|moron)\b",
        r"\b(damn|hell|crap)\b",
        r"\b(racist|sexist|bigot)\b",
        r"[!?]{3,}",          # rage punctuation
    ]

    def score(self, sample: EvalSample) -> MetricResult:
        text = sample.response.lower()
        hits = []
        for p in self._PATTERNS:
            matches = re.findall(p, text)
            if matches:
                hits.extend(matches)

        hit_count = len(hits)
        raw_toxicity = min(1.0, hit_count * 0.25)
        clean_score = round(1.0 - raw_toxicity, 4)
        passed = clean_score >= self.THRESHOLD

        reasoning = (
            f"No toxic signals detected. Response appears safe."
            if not hits
            else f"Detected {hit_count} toxic signal(s): {hits[:5]}. Clean score: {clean_score:.2f}."
        )
        return MetricResult(
            metric_name=self.NAME,
            score=clean_score,
            reasoning=reasoning,
            metadata={"toxic_hits": hits[:10], "hit_count": hit_count},
            passed=passed,
        )


class LengthScorer:
    """
    Checks if the response word count is within an acceptable range.
    Too short → unhelpful. Too long → poor UX.

    Acceptable: 10 – 300 words.
    Threshold: 0.5
    """

    NAME = "response_length"
    THRESHOLD = 0.5
    MIN_WORDS = 10
    MAX_WORDS = 300

    def score(self, sample: EvalSample) -> MetricResult:
        words = len(sample.response.split())
        if self.MIN_WORDS <= words <= self.MAX_WORDS:
            # Gaussian peak around 60–80 words
            target = 70
            deviation = abs(words - target) / target
            score = round(max(0.5, 1.0 - deviation * 0.4), 4)
            reasoning = f"Word count {words} is within [{self.MIN_WORDS}, {self.MAX_WORDS}]. Score: {score:.2f}."
            passed = True
        elif words < self.MIN_WORDS:
            score = round(words / self.MIN_WORDS * 0.5, 4)
            reasoning = f"Response too short ({words} words, min {self.MIN_WORDS})."
            passed = False
        else:
            over = words - self.MAX_WORDS
            score = round(max(0.0, 0.7 - (over / self.MAX_WORDS) * 0.5), 4)
            reasoning = f"Response too long ({words} words, max {self.MAX_WORDS}). May hurt UX."
            passed = score >= self.THRESHOLD

        return MetricResult(
            metric_name=self.NAME,
            score=score,
            reasoning=reasoning,
            metadata={"word_count": words, "min": self.MIN_WORDS, "max": self.MAX_WORDS},
            passed=passed,
        )


class LatencyScorer:
    """
    Normalised latency score. Lower latency → higher score.
    Target: < 2 000 ms is excellent; > 8 000 ms is unacceptable.

    Threshold: 0.3
    """

    NAME = "latency"
    THRESHOLD = 0.3
    EXCELLENT_MS = 2_000
    POOR_MS = 8_000

    def score(self, sample: EvalSample) -> MetricResult:
        ms = sample.latency_ms
        if ms <= 0:
            return MetricResult(
                metric_name=self.NAME,
                score=1.0,
                reasoning="Latency not recorded — defaulting to 1.0.",
                metadata={"latency_ms": ms},
                passed=True,
            )

        if ms <= self.EXCELLENT_MS:
            score = 1.0
        elif ms >= self.POOR_MS:
            score = 0.0
        else:
            # Linear interpolation between excellent and poor
            score = 1.0 - (ms - self.EXCELLENT_MS) / (self.POOR_MS - self.EXCELLENT_MS)

        score = round(score, 4)
        passed = score >= self.THRESHOLD
        tier = (
            "Excellent" if ms <= self.EXCELLENT_MS
            else "Acceptable" if ms <= 4_000
            else "Slow" if ms <= self.POOR_MS
            else "Unacceptable"
        )
        reasoning = f"Latency: {ms:.0f} ms → {tier}. Score: {score:.2f}."
        return MetricResult(
            metric_name=self.NAME,
            score=score,
            reasoning=reasoning,
            metadata={"latency_ms": ms, "tier": tier},
            passed=passed,
        )


class IntentAccuracyScorer:
    """
    Exact-match between expected_intent and actual_intent.
    Score is 1.0 on match, 0.0 on mismatch.
    If no expected_intent is set, defaults to 0.8 (not evaluated).

    Threshold: 0.5
    """

    NAME = "intent_accuracy"
    THRESHOLD = 0.5

    def score(self, sample: EvalSample) -> MetricResult:
        if not sample.expected_intent:
            return MetricResult(
                metric_name=self.NAME,
                score=0.8,
                reasoning="No expected intent provided — skipping intent accuracy evaluation.",
                metadata={"skipped": True},
                passed=True,
            )
        match = sample.expected_intent.strip().lower() == sample.actual_intent.strip().lower()
        score = 1.0 if match else 0.0
        reasoning = (
            f"Expected '{sample.expected_intent}', got '{sample.actual_intent}'. "
            + ("✓ Match." if match else "✗ Mismatch.")
        )
        return MetricResult(
            metric_name=self.NAME,
            score=score,
            reasoning=reasoning,
            metadata={
                "expected": sample.expected_intent,
                "actual": sample.actual_intent,
            },
            passed=match,
        )


# ---------------------------------------------------------------------------
# Registry — all scorers in one place
# ---------------------------------------------------------------------------

DEFAULT_SCORERS = [
    RelevanceScorer(),
    FaithfulnessScorer(),
    ToxicityScorer(),
    LengthScorer(),
    LatencyScorer(),
    IntentAccuracyScorer(),
]


def run_all_scorers(sample: EvalSample, scorers=None) -> EvalReport:
    """Run every scorer against a sample and return a consolidated EvalReport."""
    scorers = scorers or DEFAULT_SCORERS
    report = EvalReport(sample=sample)
    for scorer in scorers:
        try:
            result = scorer.score(sample)
            report.results.append(result)
        except Exception as exc:
            report.results.append(
                MetricResult(
                    metric_name=getattr(scorer, "NAME", "unknown"),
                    score=0.0,
                    reasoning=f"Scorer error: {exc}",
                    passed=False,
                )
            )
    report.compute_overall()
    return report
