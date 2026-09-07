"""Closed-loop driver: the experiment that produces your headline chart.

Round 0 : attack the app with NO defense           -> baseline ASR (high)
Round 1 : attack the app behind the KAD detector   -> ASR should drop
Round N : each round, successful attacks are handed back as new 'contaminated'
          training examples (written to disk for M1 to consume), and the
          attacker re-runs against the current defense.

Because a real retrain of M1 happens in your teammate's module, this driver
supports two modes:

  --detector none   : Act 1, no guardrail.
  --detector kad    : defended by the known-answer baseline.

When M1 is ready, import your M1Detector and pass it in place of KAD; the loop
code does not change. The per-round successful_examples() output is exactly the
labelled data M1 retrains on between rounds.

Usage:
    python run_loop.py --mock                 # offline smoke test, no Ollama
    python run_loop.py --detector none
    python run_loop.py --detector kad --rounds 3
"""

import argparse
import csv
import os

import config
from attacker import EvolutionaryAttacker
from target_app import TargetApp
from detectors import build_detector


def get_client(mock: bool):
    if mock:
        from mock_client import MockClient
        return MockClient()
    from llm_client import OllamaClient
    return OllamaClient()


def run(detector_kind: str, rounds: int, mock: bool):
    client = get_client(mock)
    detector = build_detector(detector_kind, client)
    app = TargetApp(client, detector=detector)

    os.makedirs("runs", exist_ok=True)
    history = []

    print(f"\n=== Closed loop: detector={detector.name}, rounds={rounds}, "
          f"mock={mock} ===")

    for rnd in range(rounds):
        print(f"\n[Round {rnd}] attacking (detector={detector.name})")
        attacker = EvolutionaryAttacker(app, client)
        best = attacker.run(verbose=True)

        asr = attacker.current_asr()
        n_success = len(attacker.successful_examples())
        history.append({
            "round": rnd,
            "detector": detector.name,
            "asr": round(asr, 4),
            "attempts": len(attacker.attack_log),
            "successful": n_success,
            "best_template": best[0].template if best else "",
        })
        print(f"[Round {rnd}] ASR={asr:.3f} over {len(attacker.attack_log)} "
              f"attempts; {n_success} successful attacks logged")

        # Hand the winning attacks to M1 as fresh training signal.
        ex_path = f"runs/round{rnd}_contaminated.jsonl"
        with open(ex_path, "w", encoding="utf-8") as f:
            import json
            for ex in attacker.successful_examples():
                f.write(json.dumps(ex) + "\n")
        print(f"          -> {n_success} examples written to {ex_path} for M1")

        attacker.dump_log(f"runs/round{rnd}_attacks.jsonl")

        # In a real run, M1 retrains here on accumulated contaminated examples,
        # then you rebuild `detector`/`app` with the updated M1 and continue.
        # With the static KAD baseline the detector does not change, so the ASR
        # curve across rounds shows what a NON-adapting defense looks like -
        # which is exactly the static baseline your project compares against.

    # write the ASR history CSV that the dashboard/report chart reads
    with open(config.ASR_LOG, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        w.writeheader()
        w.writerows(history)
    print(f"\nASR history -> {config.ASR_LOG}")
    return history


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--detector", default="none", choices=["none", "kad"])
    ap.add_argument("--rounds", type=int, default=1)
    ap.add_argument("--mock", action="store_true",
                    help="Use the fake client - no Ollama needed. For testing plumbing only.")
    args = ap.parse_args()
    run(args.detector, args.rounds, args.mock)


if __name__ == "__main__":
    main()
