#!/usr/bin/env python3
"""Inspect arbitrary text with M3 — no bot, no dataset.

For when someone says "check THIS" and hands you a paragraph. Useful in a
review: it proves the module is judging text in front of them, not replaying a
fixture.

    py inspect_text.py "Your API key is sk-abc123def456ghi789jkl"
    py inspect_text.py --no-guard "some text"
    py inspect_text.py                  # interactive, paste and press Enter twice
    type response.txt | py inspect_text.py --stdin
"""

from __future__ import annotations

import argparse
import sys

from output_monitor import MonitorConfig, OutputMonitor

DEFAULT_SYSTEM_PROMPT_FILE = "system_prompt.txt"
CANARY = "CANARY-7f3a91d2c4"


def read_block() -> str:
    print("Paste the text to inspect. Press Enter on a blank line to finish.\n")
    lines = []
    while True:
        try:
            line = input()
        except (EOFError, KeyboardInterrupt):
            break
        if line == "" and lines:
            break
        lines.append(line)
    return "\n".join(lines)


def show(verdict, text: str) -> None:
    tag = {"allow": "PASS", "redact": "REDACTED", "block": "BLOCKED"}[
        verdict.decision.value]
    print("\n" + "-" * 60)
    print(f"  verdict     {tag}")
    print(f"  severity    {verdict.max_severity.value}")
    print(f"  categories  {', '.join(verdict.categories) if verdict.categories else '-'}")
    print(f"  latency     {verdict.latency_ms:.0f} ms")
    print("-" * 60)
    for c in verdict.checks:
        state = "SKIP" if c.skipped else ("HIT " if c.triggered else "    ")
        detail = c.evidence[0] if c.evidence else (c.error or "")
        print(f"  [{state}] {c.name:<20} {detail[:60]}")
    print("-" * 60)
    if verdict.decision.value == "redact":
        print(f"  user would receive:\n    {verdict.sanitized_output}")
    elif verdict.decision.value == "block":
        print("  user would receive nothing — response withheld")
    else:
        print("  user would receive the text unchanged")
    print()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("text", nargs="*", help="text to inspect; omit for interactive")
    ap.add_argument("--stdin", action="store_true", help="read all of stdin")
    ap.add_argument("--user-input", default="",
                    help="the user turn that produced this text, for context")
    ap.add_argument("--no-guard", action="store_true")
    ap.add_argument("--system-prompt-file", default=DEFAULT_SYSTEM_PROMPT_FILE)
    ap.add_argument("--canary", default=CANARY)
    args = ap.parse_args()

    if args.stdin:
        text = sys.stdin.read()
    elif args.text:
        text = " ".join(args.text)
    else:
        text = read_block()

    if not text.strip():
        sys.exit("  nothing to inspect")

    system_prompt = None
    try:
        with open(args.system_prompt_file, encoding="utf-8") as fh:
            system_prompt = fh.read().strip()
    except OSError:
        print(f"  [warn] {args.system_prompt_file} not found — "
              f"leak detection limited to the canary")

    monitor = OutputMonitor(MonitorConfig(
        enable_guard=not args.no_guard,
        system_prompt=system_prompt,
        canary_token=args.canary,
        log_path=None,
    ))
    show(monitor.inspect(text, user_input=args.user_input), text)


if __name__ == "__main__":
    main()
