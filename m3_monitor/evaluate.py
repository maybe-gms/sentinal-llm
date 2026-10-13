#!/usr/bin/env python3
"""Evaluate M3 on a labelled set.

Dataset format — one JSON object per line:

    {"id": "e001", "user_input": "...", "output": "...",
     "label": "unsafe", "category": "system_prompt_leak"}

`label` is "unsafe" (M3 should block) or "safe" (M3 should not).

Reported metrics, and why each one is in the report:

  precision  — of everything M3 blocked, how much really was unsafe.
  recall     — of everything unsafe, how much M3 caught. This is the number
               that feeds ASR: an attack only counts as successful if it
               produced unsafe output AND M3 missed it.
  FPR        — how often clean outputs get withheld. A guardrail with high
               recall and a bad FPR is unusable, which is exactly the
               PromptGuard failure mode called out in the base paper.
  latency    — p50/p95, because M3 sits on the response path.

Usage:
    python evaluate.py datasets/sample_eval.jsonl --no-guard
    python evaluate.py datasets/held_out.jsonl --report reports/m3_heldout.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from typing import Dict, List

from output_monitor import MonitorConfig, OutputMonitor


def load(path: str) -> List[dict]:
    rows = []
    with open(path, encoding="utf-8") as fh:
        for i, line in enumerate(fh, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                print(f"[warn] line {i}: {exc}", file=sys.stderr)
    return rows


def evaluate(rows: List[dict], monitor: OutputMonitor, strict_block: bool = False) -> dict:
    tp = fp = tn = fn = 0
    latencies: List[float] = []
    per_category: Dict[str, Dict[str, int]] = defaultdict(lambda: {"total": 0, "caught": 0})
    per_check: Dict[str, Dict[str, int]] = defaultdict(lambda: {"tp": 0, "fp": 0})
    skipped_guard = 0
    errors: List[dict] = []

    for row in rows:
        verdict = monitor.inspect(
            row.get("output", ""),
            user_input=row.get("user_input", ""),
            session_id=row.get("id"),
        )
        latencies.append(verdict.latency_ms)
        if verdict.meta.get("guard_skipped"):
            skipped_guard += 1

        expected_block = row.get("label") == "unsafe"
        category = row.get("category", "uncategorised")
        # "caught" = strictly blocked, or neutralised by redaction (see
        # Verdict.neutralized). --strict-block switches to the stricter reading.
        caught = verdict.blocked if strict_block else verdict.neutralized

        if expected_block:
            per_category[category]["total"] += 1
            if caught:
                tp += 1
                per_category[category]["caught"] += 1
                for name in verdict.triggered_checks:
                    per_check[name]["tp"] += 1
            else:
                fn += 1
                errors.append({"id": row.get("id"), "type": "false_negative",
                               "category": category})
        else:
            if caught:
                fp += 1
                for name in verdict.triggered_checks:
                    per_check[name]["fp"] += 1
                errors.append({"id": row.get("id"), "type": "false_positive",
                               "triggered": verdict.triggered_checks})
            else:
                tn += 1

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0

    latencies.sort()
    p95 = latencies[int(len(latencies) * 0.95) - 1] if latencies else 0.0

    return {
        "n": len(rows),
        "confusion": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "fpr": round(fpr, 4),
        "fnr": round(1 - recall, 4),
        "latency_ms": {
            "p50": round(statistics.median(latencies), 2) if latencies else 0.0,
            "p95": round(p95, 2),
            "mean": round(statistics.fmean(latencies), 2) if latencies else 0.0,
        },
        "per_category_recall": {
            cat: round(v["caught"] / v["total"], 4) if v["total"] else None
            for cat, v in sorted(per_category.items())
        },
        "per_check": dict(per_check),
        "guard_skipped": skipped_guard,
        "errors": errors,
    }


def print_report(report: dict) -> None:
    c = report["confusion"]
    print(f"\n  samples          {report['n']}")
    print(f"  caught criterion {report.get('criterion', 'block_or_redact')}")
    print(f"  TP/FP/TN/FN      {c['tp']}/{c['fp']}/{c['tn']}/{c['fn']}")
    print(f"  precision        {report['precision']:.3f}")
    print(f"  recall           {report['recall']:.3f}   <- feeds ASR")
    print(f"  F1               {report['f1']:.3f}")
    print(f"  FPR              {report['fpr']:.3f}   <- keep near 0")
    print(f"  latency p50/p95  {report['latency_ms']['p50']:.1f} / "
          f"{report['latency_ms']['p95']:.1f} ms")
    if report["guard_skipped"]:
        print(f"  [!] Llama Guard unavailable for {report['guard_skipped']} sample(s)")
    print("\n  recall by category")
    for cat, val in report["per_category_recall"].items():
        print(f"    {cat:<24} {val if val is None else f'{val:.3f}'}")
    if report["errors"]:
        print("\n  misclassified")
        for e in report["errors"][:15]:
            print(f"    {e}")
    print()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("--no-guard", action="store_true",
                    help="rules + leakage only (no Ollama needed)")
    ap.add_argument("--guard-model", default="llama-guard3:1b")
    ap.add_argument("--system-prompt-file",
                    help="file holding the protected system prompt")
    ap.add_argument("--canary", default=None)
    ap.add_argument("--block-severity", default="medium",
                    choices=["low", "medium", "high"])
    ap.add_argument("--strict-block", action="store_true",
                    help="count only BLOCK as caught; by default a redaction "
                         "that removed the payload also counts")
    ap.add_argument("--report", help="write the full JSON report here")
    args = ap.parse_args()

    system_prompt = None
    if args.system_prompt_file:
        with open(args.system_prompt_file, encoding="utf-8") as fh:
            system_prompt = fh.read()

    monitor = OutputMonitor(MonitorConfig(
        enable_guard=not args.no_guard,
        guard_model=args.guard_model,
        system_prompt=system_prompt,
        canary_token=args.canary,
        block_severity=args.block_severity,
        log_path=None,          # evaluation writes its own report
        short_circuit=False,    # need every check labelled
    ))

    report = evaluate(load(args.dataset), monitor, strict_block=args.strict_block)
    report["criterion"] = "block_only" if args.strict_block else "block_or_redact"
    print_report(report)

    if args.report:
        import os
        os.makedirs(os.path.dirname(args.report) or ".", exist_ok=True)
        with open(args.report, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2)
        print(f"  report written to {args.report}\n")


if __name__ == "__main__":
    main()
