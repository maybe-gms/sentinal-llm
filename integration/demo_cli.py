"""Terminal version of the integrated-stack demo.

Same pipeline as the web UI, no browser. Handy as a projector-proof fallback
during the review, and for a quick offline smoke test. Runs every preset
scenario through the chosen layer configuration and prints a stage-by-stage
trace plus a summary table.

    python demo_cli.py                 # offline mock, full stack
    python demo_cli.py --layers m1     # M1 only
    python demo_cli.py --real          # real Ollama
    python demo_cli.py --only steal_subtle
"""

from __future__ import annotations

import argparse
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_ROOT, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from shared import build_pipeline, Outcome  # noqa: E402
from scenarios import SCENARIOS, SCENARIOS_BY_ID, CANARY, APP_INSTRUCTION  # noqa: E402

# ANSI colours (fall back to plain if not a tty)
def _c(code, s, on):
    return f"\033[{code}m{s}\033[0m" if on else s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", default="full", choices=["none", "m1", "m3", "full"])
    ap.add_argument("--real", action="store_true", help="Drive real Ollama.")
    ap.add_argument("--only", help="Run a single scenario by id.")
    args = ap.parse_args()
    color = sys.stdout.isatty()

    pipe = build_pipeline(
        args.layers, mock=not args.real,
        system_prompt=APP_INSTRUCTION, canary_token=CANARY,
    )
    scenarios = ([SCENARIOS_BY_ID[args.only]] if args.only else SCENARIOS)

    mode = "REAL (Ollama)" if args.real else "MOCK (offline)"
    print(f"\n  Integrated defense stack — layers={args.layers} — mode={mode}\n")

    summary = []
    for s in scenarios:
        r = pipe.run(s["instruction"], s["data"])
        print(_c("1;37", f"▌ {s['label']}", color))
        print(_c("2", f"  {s['note']}", color))
        for st in r.stages:
            icon = {
                Outcome.PASS: _c("32", "✓ pass", color),
                Outcome.GENERATED: _c("34", "• generated", color),
                Outcome.FLAGGED: _c("31", "✗ BLOCKED", color),
                Outcome.REDACTED: _c("33", "✎ redacted", color),
                Outcome.SKIPPED: _c("2", "· off", color),
                Outcome.ERROR: _c("31", "! error", color),
            }.get(st.outcome, st.outcome.value)
            cats = f"  [{', '.join(st.categories)}]" if st.categories else ""
            print(f"    {st.stage.value:20} {icon}{cats}")
        verdict = _c("1;32", r.headline, color) if r.delivered and not r.defended \
            else _c("1;31", r.headline, color) if not r.delivered \
            else _c("1;33", r.headline, color)
        print(f"  → {verdict}")
        if not r.delivered and r.raw_output:
            print(_c("2", f"    (model produced: {r.raw_output[:90]})", color))
        print()
        summary.append((s["label"], r.headline, r.delivered))

    print(_c("1;37", "  Summary", color))
    for label, headline, delivered in summary:
        mark = _c("32", "DELIVERED", color) if delivered else _c("31", "STOPPED  ", color)
        print(f"    {mark}  {label}")
    print()


if __name__ == "__main__":
    main()
