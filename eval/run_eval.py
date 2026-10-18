"""
CLI entrypoint for the LLM Evaluation Platform.

Usage
-----
  # Smoke test (built-in synthetic cases)
  python eval/run_eval.py smoke

  # Evaluate golden dataset
  python eval/run_eval.py suite --dataset eval/data/golden_set.json

  # Score a single ad-hoc pair
  python eval/run_eval.py single \
      --query "How do I reset my password?" \
      --response "Click Forgot Password on the login page." \
      --intent faq

  # Print dashboard stats from stored results
  python eval/run_eval.py dashboard --hours 48
"""

from __future__ import annotations

import argparse
import json
import sys
import os

# Make sure project root is on path when running standalone
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval.metrics.scorers import EvalSample
from eval.runners.eval_runner import EvalRunner
from eval.storage.result_store import ResultStore


def _print_json(obj):
    print(json.dumps(obj, indent=2, default=str))


def cmd_smoke(args, runner):
    print("▶ Running smoke test …\n")
    summary = runner.smoke_test()
    status = "✅ PASSED" if summary["pass_rate"] >= 0.8 else "❌ DEGRADED"
    print(f"Status : {status}")
    print(f"Samples: {summary['total_samples']}")
    print(f"Passed : {summary['passed']}")
    print(f"Pass % : {summary['pass_rate'] * 100:.1f}%")
    print(f"Avg Score: {summary['avg_overall_score']:.3f}")
    print("\nPer-metric averages:")
    for m, v in summary.get("metric_averages", {}).items():
        bar = "█" * int(v * 20)
        print(f"  {m:<20} {v:.3f}  {bar}")
    return 0 if summary["pass_rate"] >= 0.8 else 1


def cmd_suite(args, runner):
    if not args.dataset:
        print("Error: --dataset is required", file=sys.stderr)
        return 1
    print(f"▶ Running suite on {args.dataset} …\n")
    suite = runner.run_suite(dataset_path=args.dataset)
    summary = suite.summary()
    status = "✅ PASSED" if summary["pass_rate"] >= 0.8 else "❌ FAILED"
    print(f"Status : {status}")
    print(f"Dataset: {summary['dataset_name']}")
    print(f"Samples: {summary['total_samples']}")
    print(f"Pass % : {summary['pass_rate'] * 100:.1f}%")
    print(f"Avg Score: {summary['avg_overall_score']:.3f}")
    print()
    print("Per-metric averages:")
    for m, v in summary.get("metric_averages", {}).items():
        icon = "✅" if v >= 0.6 else "⚠️ "
        print(f"  {icon} {m:<20} {v:.3f}")
    if args.json:
        print("\nFull summary:")
        _print_json(summary)
    return 0 if summary["pass_rate"] >= 0.8 else 1


def cmd_single(args, runner):
    sample = EvalSample(
        query=args.query,
        response=args.response,
        context=args.context or "",
        expected_intent=args.intent or "",
        actual_intent=args.intent or "",
        latency_ms=float(args.latency or 0),
    )
    from eval.metrics.scorers import run_all_scorers
    report = run_all_scorers(sample)
    runner.store.save_report(report)

    status = "✅ PASSED" if report.passed else "❌ FAILED"
    print(f"\n{status}  Overall score: {report.overall_score:.3f}\n")
    for r in report.results:
        icon = "✅" if r.passed else "❌"
        print(f"  {icon} {r.metric_name:<20} {r.score:.3f}")
        print(f"      {r.reasoning}")
        print()
    return 0 if report.passed else 1


def cmd_dashboard(args, store):
    import time
    since = time.time() - float(args.hours or 24) * 3600
    stats = store.get_dashboard_stats(since=since)
    print(f"\n📊 Dashboard (last {args.hours or 24}h)\n{'─'*40}")
    print(f"Total evaluations : {stats['total_evaluations']}")
    print(f"Pass rate         : {stats['pass_rate'] * 100:.1f}%")
    print(f"Avg overall score : {stats['avg_overall_score']:.3f}")
    print(f"Avg latency       : {stats['avg_latency_ms']:.0f} ms")
    print()
    print("By agent:")
    for a in stats.get("by_agent", []):
        print(f"  {a['agent_used']:<25} {a['count']:>4} evals  avg={a['avg_score']:.3f}")
    print()
    print("By metric (avg score):")
    for m in stats.get("by_metric", []):
        bar = "█" * int((m["avg_score"] or 0) * 20)
        print(f"  {m['metric_name']:<20} {(m['avg_score'] or 0):.3f}  {bar}")


def main():
    parser = argparse.ArgumentParser(
        description="LLM Evaluation & Observability CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command")

    # smoke
    sub.add_parser("smoke", help="Run built-in smoke test")

    # suite
    p_suite = sub.add_parser("suite", help="Run evaluation on a dataset file")
    p_suite.add_argument("--dataset", required=True, help="Path to JSON dataset")
    p_suite.add_argument("--json", action="store_true", help="Print full JSON summary")

    # single
    p_single = sub.add_parser("single", help="Evaluate a single query-response pair")
    p_single.add_argument("--query", required=True)
    p_single.add_argument("--response", required=True)
    p_single.add_argument("--context", default="")
    p_single.add_argument("--intent", default="")
    p_single.add_argument("--latency", default=0, type=float)

    # dashboard
    p_dash = sub.add_parser("dashboard", help="Print stored dashboard stats")
    p_dash.add_argument("--hours", default=24, type=float)

    args = parser.parse_args()

    store = ResultStore()
    runner = EvalRunner(store=store)

    if args.command == "smoke":
        sys.exit(cmd_smoke(args, runner))
    elif args.command == "suite":
        sys.exit(cmd_suite(args, runner))
    elif args.command == "single":
        sys.exit(cmd_single(args, runner))
    elif args.command == "dashboard":
        cmd_dashboard(args, store)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
