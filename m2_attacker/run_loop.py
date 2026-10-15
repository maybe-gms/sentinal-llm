"""Closed-loop driver: the experiment that produces your headline chart.

Round 0 : attack the app with NO defense           -> baseline ASR (high)
Round 1 : attack the app behind the KAD detector   -> ASR should drop
Round N : each round, successful attacks are handed back as new 'contaminated'
          training examples (written to disk for M1 to consume), and the
          attacker re-runs against the current defense.

This version does three things the first draft didn't:

  1. TRIAL AVERAGING. Each round is run REPEAT_TRIALS times and the ASR is
     reported as mean +/- standard deviation. LLM outputs are stochastic even
     at low temperature, so a single trial is one roll of the dice; the mean
     over several is a number you can defend in a viva.

  2. DETECTOR-TAGGED OUTPUTS. Every file is prefixed with the detector name
     (e.g. runs/none_round0_attacks.jsonl, runs/kad_round0_attacks.jsonl) so a
     `none` run and a `kad` run no longer overwrite each other. You need both
     on disk at once to draw the baseline-vs-defended chart.

  3. BLOCK COUNTS. Blocked-vs-succeeded prints per round instead of hiding in
     the log, so "2/240 blocked" lands on a slide directly.

The contaminated .jsonl schema is byte-for-byte unchanged, so M1 is unaffected.

When M1 is ready, import your M1Detector and pass it in place of KAD; the loop
code does not change.

Usage:
    python run_loop.py --mock                      # offline smoke test, no Ollama
    python run_loop.py --detector none --rounds 1
    python run_loop.py --detector kad  --rounds 2
    python run_loop.py --detector kad  --trials 3  # override REPEAT_TRIALS

    # with M3 (output monitor) on the answer path:
    python run_loop.py --detector none --monitor m3   # M3 alone
    python run_loop.py --detector kad  --monitor m3   # both doors guarded
    python run_loop.py --detector kad  --monitor m3 --m3-guard   # + Llama Guard
"""

import argparse
import csv
import json
import os
import statistics

import config
from attacker import EvolutionaryAttacker
from target_app import TargetApp
from detectors import build_detector
from output_guard import build_output_guard
from tasks import TARGET_TASKS, ATTACK_SETS


def get_client(mock: bool):
    if mock:
        from mock_client import MockClient
        return MockClient()
    from llm_client import OllamaClient
    return OllamaClient()


def _run_one_round(app, client, rnd, detector_name, trials, write_artifacts=True,
                   attack_set="default"):
    """Run one round `trials` times. Return an aggregated stats dict.

    Artifacts (attack log + contaminated training file) are written from the
    FIRST trial only - they're illustrative examples for M1 and for eyeballing,
    and we don't want trial 2/3 clobbering them. The reported NUMBERS, however,
    are averaged over all trials.
    """
    asrs, block_rates, blocks, successes, attempts = [], [], [], [], []
    det_blocks, mon_blocks = [], []
    first_attacker = None

    for t in range(trials):
        attacker = EvolutionaryAttacker(app, client, injected_tasks=ATTACK_SETS[attack_set])
        best = attacker.run(verbose=(t == 0))  # only narrate the first trial

        log = attacker.attack_log
        n = len(log)
        n_block = sum(1 for r in log if r.blocked)
        n_succ = sum(1 for r in log if r.injected_success)
        det_blocks.append(sum(1 for r in log if r.blocked_by == "detector"))
        mon_blocks.append(sum(1 for r in log if r.blocked_by == "monitor"))

        asrs.append(n_succ / max(1, n))
        block_rates.append(n_block / max(1, n))
        blocks.append(n_block)
        successes.append(n_succ)
        attempts.append(n)

        if t == 0:
            first_attacker = attacker
            best_template = best[0].template if best else ""

        if trials > 1:
            print(f"    trial {t + 1}/{trials}: ASR={n_succ / max(1, n):.3f} "
                  f"blocked={n_block}/{n}")

    mean_asr = statistics.mean(asrs)
    std_asr = statistics.pstdev(asrs) if len(asrs) > 1 else 0.0
    mean_block = statistics.mean(blocks)

    if write_artifacts and first_attacker is not None:
        os.makedirs("runs", exist_ok=True)
        prefix = f"runs/{detector_name}_round{rnd}"

        ex_path = f"{prefix}_contaminated.jsonl"
        with open(ex_path, "w", encoding="utf-8") as f:
            for ex in first_attacker.successful_examples():
                f.write(json.dumps(ex) + "\n")

        first_attacker.dump_log(f"{prefix}_attacks.jsonl")
        n_ex = len(first_attacker.successful_examples())
        print(f"          -> {n_ex} contaminated examples -> {ex_path} (for M1)")
        print(f"          -> full attack log -> {prefix}_attacks.jsonl")

    return {
        "round": rnd,
        "detector": detector_name,
        "trials": trials,
        "asr_mean": round(mean_asr, 4),
        "asr_std": round(std_asr, 4),
        "attempts": attempts[0] if attempts else 0,
        "blocked_mean": round(mean_block, 1),
        "blocked_by_detector_mean": round(statistics.mean(det_blocks), 1),
        "blocked_by_monitor_mean": round(statistics.mean(mon_blocks), 1),
        "success_mean": round(statistics.mean(successes), 1),
        "best_template": best_template,
    }


def measure_clean_fpr(app):
    """Send every CLEAN target sample (no injection) through the app and count
    how many get blocked anyway. Those are false alarms. A defense that blocks
    everything would also get ASR = 0, so ASR is only meaningful next to this."""
    total = blocked = 0
    for task in TARGET_TASKS:
        for sample in task.samples:
            total += 1
            resp = app.query(task, sample["data"])
            if resp.blocked:
                blocked += 1
                print(f"    false alarm ({resp.blocked_by}) on clean "
                      f"{task.name}: {resp.raw_output[:60]!r}")
    return blocked, total


def run(detector_kind: str, rounds: int, mock: bool, trials: int,
        monitor_kind: str = "none", m3_guard: bool = False, attack_set: str = "default"):
    client = get_client(mock)
    detector = build_detector(detector_kind, client)
    guard = build_output_guard(monitor_kind, use_llama_guard=m3_guard)
    output_guard = None if guard.name == "none" else guard
    app = TargetApp(client, detector=detector, output_guard=output_guard)

    # Run name tags every output file, e.g. "kad", "kad_m3", "none_m3".
    run_name = detector.name if output_guard is None else f"{detector.name}_{guard.name}"
    if attack_set != "default":
        run_name += f"_{attack_set}"   # keeps evasion runs from overwriting normal runs

    os.makedirs("runs", exist_ok=True)
    history = []

    print(f"\n=== Closed loop: detector={detector.name}, monitor={guard.name}, "
          f"attacks={attack_set}, rounds={rounds}, trials={trials}, mock={mock} ===")

    fp, n_clean = measure_clean_fpr(app)
    clean_fpr = fp / max(1, n_clean)
    print(f"[Clean check] false alarms on clean inputs: {fp}/{n_clean} "
          f"(FPR = {clean_fpr:.2f})")

    for rnd in range(rounds):
        print(f"\n[Round {rnd}] attacking (detector={detector.name})")
        stats = _run_one_round(app, client, rnd, run_name, trials, attack_set=attack_set)
        stats["monitor"] = guard.name
        stats["clean_fpr"] = round(clean_fpr, 4)
        history.append(stats)

        pct_block = 100 * stats["blocked_mean"] / max(1, stats["attempts"])
        print(f"[Round {rnd}] ASR = {stats['asr_mean']:.3f} "
              f"+/- {stats['asr_std']:.3f}  |  "
              f"blocked {stats['blocked_mean']:.0f}/{stats['attempts']} "
              f"({pct_block:.0f}%)  |  "
              f"succeeded {stats['success_mean']:.0f}/{stats['attempts']}")
        if output_guard is not None:
            print(f"          blocked by detector: {stats['blocked_by_detector_mean']:.0f}  |  "
                  f"blocked by M3 monitor: {stats['blocked_by_monitor_mean']:.0f}")

        # In a real run, M1 retrains here on accumulated contaminated examples,
        # then you rebuild `detector`/`app` with the updated M1 and continue.
        # With the static KAD baseline the detector does not change, so the ASR
        # curve across rounds shows what a NON-adapting defense looks like -
        # exactly the static baseline your project compares against.

    # ASR history CSV is per-detector so runs don't overwrite each other.
    out_csv = config.ASR_LOG.replace(".csv", f"_{run_name}.csv")
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        w.writeheader()
        w.writerows(history)
    print(f"\nASR history -> {out_csv}")
    return history


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--detector", default="none", choices=["none", "kad"])
    ap.add_argument("--rounds", type=int, default=1)
    ap.add_argument("--trials", type=int, default=None,
                    help="How many times to repeat each round and average. "
                         "Defaults to config.REPEAT_TRIALS.")
    ap.add_argument("--mock", action="store_true",
                    help="Use the fake client - no Ollama needed. Plumbing test only.")
    ap.add_argument("--monitor", default="none", choices=["none", "m3"],
                    help="Output monitor on the answer path. 'm3' = Jerin's M3.")
    ap.add_argument("--m3-guard", action="store_true",
                    help="Also run Llama Guard inside M3 (needs `ollama pull "
                         "llama-guard3:1b`, and is much slower).")
    ap.add_argument("--attack-set", default="default", choices=sorted(ATTACK_SETS),
                    help="default = original 5 attacks; m3_evasion = plain vs "
                         "disguised prompt stealing; all = everything.")
    args = ap.parse_args()
    trials = args.trials if args.trials is not None else config.REPEAT_TRIALS
    run(args.detector, args.rounds, args.mock, trials, args.monitor, args.m3_guard,
        args.attack_set)


if __name__ == "__main__":
    main()