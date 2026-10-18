"""
ResultStore — SQLite-backed persistence layer for evaluation results.

Why SQLite?
  - Zero infra: no external DB needed, file is committed alongside code.
  - Fast enough for eval workloads (thousands of rows).
  - Easy to swap for Postgres/MySQL via the Repository pattern used here.

Schema
------
  eval_reports (one row per evaluated sample)
  eval_metrics (one row per metric per sample, FK → eval_reports)

Both tables are created on first use (idempotent).
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

from eval.metrics.scorers import EvalReport, MetricResult

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = "eval/data/eval_results.db"


class ResultStore:
    """
    Thread-safe SQLite repository for EvalReport persistence.

    Parameters
    ----------
    db_path : str
        Path to the SQLite file. Created automatically if absent.
    """

    def __init__(self, db_path: str = DEFAULT_DB_PATH):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @contextmanager
    def _conn(self):
        """Yield a connection with WAL mode for concurrent reads."""
        conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_schema(self) -> None:
        with self._conn() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS eval_reports (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id      TEXT,
                    query           TEXT NOT NULL,
                    response        TEXT NOT NULL,
                    model_name      TEXT,
                    agent_used      TEXT,
                    latency_ms      REAL,
                    overall_score   REAL,
                    passed          INTEGER,
                    timestamp       REAL,
                    created_at      REAL DEFAULT (unixepoch('now'))
                );

                CREATE TABLE IF NOT EXISTS eval_metrics (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    report_id       INTEGER NOT NULL
                                    REFERENCES eval_reports(id) ON DELETE CASCADE,
                    metric_name     TEXT NOT NULL,
                    score           REAL,
                    reasoning       TEXT,
                    metadata_json   TEXT,
                    passed          INTEGER
                );

                CREATE INDEX IF NOT EXISTS idx_reports_session
                    ON eval_reports(session_id);
                CREATE INDEX IF NOT EXISTS idx_reports_ts
                    ON eval_reports(timestamp);
                CREATE INDEX IF NOT EXISTS idx_metrics_name
                    ON eval_metrics(metric_name);
            """)
        logger.debug("ResultStore schema initialised at %s", self.db_path)

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def save_report(self, report: EvalReport) -> int:
        """Persist an EvalReport (and its metrics) and return the row id."""
        with self._conn() as conn:
            cur = conn.execute(
                """
                INSERT INTO eval_reports
                    (session_id, query, response, model_name, agent_used,
                     latency_ms, overall_score, passed, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    report.sample.session_id,
                    report.sample.query,
                    report.sample.response,
                    report.sample.model_name,
                    report.sample.agent_used,
                    report.sample.latency_ms,
                    report.overall_score,
                    int(report.passed),
                    report.sample.timestamp,
                ),
            )
            report_id = cur.lastrowid

            conn.executemany(
                """
                INSERT INTO eval_metrics
                    (report_id, metric_name, score, reasoning, metadata_json, passed)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        report_id,
                        r.metric_name,
                        r.score,
                        r.reasoning,
                        json.dumps(r.metadata),
                        int(r.passed),
                    )
                    for r in report.results
                ],
            )

        logger.debug("Saved report id=%d session=%s", report_id, report.sample.session_id)
        return report_id

    # ------------------------------------------------------------------
    # Read — single / paginated
    # ------------------------------------------------------------------

    def get_report(self, report_id: int) -> Optional[dict]:
        """Fetch a single report with its metrics."""
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM eval_reports WHERE id = ?", (report_id,)
            ).fetchone()
            if not row:
                return None
            metrics = conn.execute(
                "SELECT * FROM eval_metrics WHERE report_id = ?", (report_id,)
            ).fetchall()

        return self._hydrate_report(row, metrics)

    def list_reports(
        self,
        limit: int = 50,
        offset: int = 0,
        model_name: Optional[str] = None,
        agent_used: Optional[str] = None,
        passed: Optional[bool] = None,
    ) -> list[dict]:
        """
        Return paginated report rows (without metrics, for list views).
        Supports filtering by model, agent, and pass/fail status.
        """
        clauses, params = [], []
        if model_name:
            clauses.append("model_name = ?")
            params.append(model_name)
        if agent_used:
            clauses.append("agent_used = ?")
            params.append(agent_used)
        if passed is not None:
            clauses.append("passed = ?")
            params.append(int(passed))

        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        params += [limit, offset]

        with self._conn() as conn:
            rows = conn.execute(
                f"""
                SELECT id, session_id, query, model_name, agent_used,
                       latency_ms, overall_score, passed, timestamp
                FROM eval_reports
                {where}
                ORDER BY timestamp DESC
                LIMIT ? OFFSET ?
                """,
                params,
            ).fetchall()

        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Aggregates / analytics
    # ------------------------------------------------------------------

    def get_dashboard_stats(
        self, since: Optional[float] = None
    ) -> dict:
        """
        Return aggregated statistics suitable for the monitoring dashboard.

        Parameters
        ----------
        since : float, optional
            Unix timestamp. If provided, only includes records after this time.
        """
        since = since or 0.0
        with self._conn() as conn:
            totals = conn.execute(
                """
                SELECT
                    COUNT(*)            AS total,
                    SUM(passed)         AS total_passed,
                    AVG(overall_score)  AS avg_score,
                    AVG(latency_ms)     AS avg_latency_ms,
                    MIN(overall_score)  AS min_score,
                    MAX(overall_score)  AS max_score
                FROM eval_reports
                WHERE timestamp >= ?
                """,
                (since,),
            ).fetchone()

            # Per-agent breakdown
            agents = conn.execute(
                """
                SELECT agent_used,
                       COUNT(*)           AS count,
                       AVG(overall_score) AS avg_score,
                       SUM(passed)        AS passed_count
                FROM eval_reports
                WHERE timestamp >= ?
                GROUP BY agent_used
                ORDER BY count DESC
                """,
                (since,),
            ).fetchall()

            # Per-metric averages
            metrics = conn.execute(
                """
                SELECT m.metric_name,
                       AVG(m.score)  AS avg_score,
                       SUM(m.passed) AS pass_count,
                       COUNT(*)      AS total
                FROM eval_metrics m
                JOIN eval_reports r ON r.id = m.report_id
                WHERE r.timestamp >= ?
                GROUP BY m.metric_name
                ORDER BY avg_score
                """,
                (since,),
            ).fetchall()

            # Score distribution buckets
            buckets = conn.execute(
                """
                SELECT
                    CASE
                        WHEN overall_score >= 0.9 THEN '0.9-1.0'
                        WHEN overall_score >= 0.7 THEN '0.7-0.9'
                        WHEN overall_score >= 0.5 THEN '0.5-0.7'
                        ELSE '0.0-0.5'
                    END AS bucket,
                    COUNT(*) AS count
                FROM eval_reports
                WHERE timestamp >= ?
                GROUP BY bucket
                """,
                (since,),
            ).fetchall()

            # Last 24h trend (hourly)
            trend = conn.execute(
                """
                SELECT
                    CAST(strftime('%s', datetime(timestamp, 'unixepoch')) / 3600 AS INT) * 3600 AS hour_ts,
                    AVG(overall_score) AS avg_score,
                    COUNT(*) AS count
                FROM eval_reports
                WHERE timestamp >= ?
                GROUP BY hour_ts
                ORDER BY hour_ts
                LIMIT 48
                """,
                (since,),
            ).fetchall()

        total = totals["total"] or 0
        passed = totals["total_passed"] or 0

        return {
            "total_evaluations": total,
            "total_passed": passed,
            "total_failed": total - passed,
            "pass_rate": round(passed / total, 4) if total else 0.0,
            "avg_overall_score": round(totals["avg_score"] or 0, 4),
            "avg_latency_ms": round(totals["avg_latency_ms"] or 0, 1),
            "min_score": round(totals["min_score"] or 0, 4),
            "max_score": round(totals["max_score"] or 0, 4),
            "by_agent": [dict(a) for a in agents],
            "by_metric": [dict(m) for m in metrics],
            "score_distribution": [dict(b) for b in buckets],
            "hourly_trend": [dict(t) for t in trend],
        }

    def get_metric_trend(
        self, metric_name: str, limit: int = 100
    ) -> list[dict]:
        """Return the most recent `limit` scores for a specific metric."""
        with self._conn() as conn:
            rows = conn.execute(
                """
                SELECT r.timestamp, m.score, r.agent_used, r.session_id
                FROM eval_metrics m
                JOIN eval_reports r ON r.id = m.report_id
                WHERE m.metric_name = ?
                ORDER BY r.timestamp DESC
                LIMIT ?
                """,
                (metric_name, limit),
            ).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    @staticmethod
    def _hydrate_report(row: sqlite3.Row, metrics: list) -> dict:
        d = dict(row)
        d["metrics"] = [
            {
                "metric_name": m["metric_name"],
                "score": m["score"],
                "reasoning": m["reasoning"],
                "metadata": json.loads(m["metadata_json"] or "{}"),
                "passed": bool(m["passed"]),
            }
            for m in metrics
        ]
        d["passed"] = bool(d["passed"])
        return d

    def count(self) -> int:
        with self._conn() as conn:
            return conn.execute(
                "SELECT COUNT(*) FROM eval_reports"
            ).fetchone()[0]
