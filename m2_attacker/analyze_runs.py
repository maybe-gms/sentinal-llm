"""Summarise attack logs, broken down by attack type.

Usage (from m2_attacker/):
    py analyze_runs.py                         # every *_attacks.jsonl in runs/
    py analyze_runs.py runs/none_m3_round0_attacks.jsonl
    py analyze_runs.py --show-monitor-blocks   # also print what M3 blocked

For each attack type it shows:
  tried     how many attempts used that attack
  followed  how many times the LLM obeyed the injection (before M3)
  det       blocked by the input detector (M1 / KAD)
  M3        blocked by the output monitor
  success   got all the way through (this is what ASR counts)
"""

import glob
import json
import sys


def load(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def summarise(path, show_monitor_blocks):
    rows = load(path)
    print(f"\n=== {path}  ({len(rows)} attempts) ===")
    print(f"{'attack type':22s} {'tried':>6s} {'followed':>9s} {'det':>5s} {'M3':>5s} {'success':>8s} {'ASR':>6s}")

    by_type = {}
    for r in rows:
        by_type.setdefault(r["injected_task"], []).append(r)

    total_success = 0
    for name in sorted(by_type):
        group = by_type[name]
        tried = len(group)
        followed = sum(1 for r in group if r.get("llm_followed"))
        det = sum(1 for r in group if r.get("blocked_by") == "detector")
        mon = sum(1 for r in group if r.get("blocked_by") == "monitor")
        success = sum(1 for r in group if r["injected_success"])
        total_success += success
        print(f"{name:22s} {tried:6d} {followed:9d} {det:5d} {mon:5d} {success:8d} {success / tried:6.2f}")

    print(f"{'TOTAL':22s} {len(rows):6d} {'':9s} {'':5s} {'':5s} {total_success:8d} "
          f"{total_success / max(1, len(rows)):6.2f}")

    if show_monitor_blocks:
        # M3 blocks where the scorer says the LLM did NOT follow the injection.
        # These are either partial leaks the scorer is too strict about, or
        # false alarms. Read them and decide.
        unclear = [r for r in rows if r.get("blocked_by") == "monitor" and not r.get("llm_followed")]
        print(f"\n  M3 blocked {len(unclear)} answers where the LLM did not (fully) follow the attack:")
        for r in unclear:
            raw = r.get("raw_output", "")
            if not raw:
                print("   (this log is from before raw_output was saved - rerun to see the text)")
                break
            print(f"   [{r['target_task']} / {r['injected_task']}] {raw[:110]!r}")


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    show = "--show-monitor-blocks" in sys.argv
    paths = args or sorted(glob.glob("runs/*_attacks.jsonl"))
    if not paths:
        print("No attack logs found in runs/. Run run_loop.py first.")
        return
    for p in paths:
        summarise(p, show)


if __name__ == "__main__":
    main()
