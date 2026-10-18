"""Closed-loop driver: the experiment that produces your headline chart.

Round 0 : attack the app with NO defense           -> baseline ASR (high)
Round 1 : attack the app behind a detector         -> ASR should drop
Round N : each round, successful attacks are handed back as new 'contaminated'
          training examples (written to disk for M1 to consume), and the
          attacker re-runs against the current defense.

What this driver does:

  1. TRIAL AVERAGING. Each round is run REPEAT_TRIALS times and the ASR is
     reported as mean +/- standard deviation. Every trial gets its own random
     seed (SEED, SEED+1, ...) so trials sample different task pairs and
     mutations - a single trial is one roll of the dice.

  2. ARMS. --arm baseline runs the original search (five seeds, rule + LLM
     mutation). --arm enriched adds the structure templates to generation 0
     and the structure-inspired operators + crossover. Same population size
     and same number of attempts, so the two ASRs are directly comparable.

  3. DETECTORS. --detector none | kad | m1. 'm1' calls George's
     m1_detector.predict() (the fine-tuned model served by Ollama as
     "m1-detector"). With --mock, a fake keyword-based M1 is used instead so
     the plumbing can be tested without Ollama.

  4. TAGGED OUTPUTS. Every file is prefixed with <detector>_<arm>, e.g.
     runs/m1_enriched_round0_attacks.jsonl, so runs never overwrite each other.

The contaminated .jsonl schema is byte-for-byte unchanged, so M1 is unaffected.

Usage:
    py run_loop.py --mock                                  # offline smoke test
    py run_loop.py --detector none --arm baseline
    py run_loop.py --detector none --arm enriched
    py run_loop.py --detector m1   --arm enriched --trials 3
"""

import argparse
import csv
import functools
import json
import os
import random
import statistics
import sys

import config
from attacker import EvolutionaryAttacker
from target_app import TargetApp
from detectors import build_detector, M1Detector

# Which attacker switches each arm turns on.
ARMS = {
    "baseline": {"use_templates": False, "use_new_ops": False},
    "enriched": {"use_templates": True, "use_new_ops": True},
}


def get_client(mock: bool):
    if mock:
        from mock_client import MockClient
        return MockClient()
    from llm_client import OllamaClient
    return OllamaClient()


# ---- M1 detector ---------------------------------------------------------

def _load_m1_predict():
    """Import predict() from the m1_detector package in the repo root.

    m2_attacker/ and m1_detector/ are sibling folders, so we add their parent
    (the repo root) to Python's import path once, then import normally.
    """
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    from m1_detector import predict
    return predict


def _mock_m1_predict(text: str) -> bool:
    """FAKE M1 for --mock runs only: flags text containing a few obvious cue
    phrases. It is not the real detector; its numbers mean nothing."""
    t = text.lower()
    return any(cue in t for cue in ["ignore previous", "disregard", "[system]", "<|system|>"])


def make_detector(kind: str, client, mock: bool):
    if kind != "m1":
        return build_detector(kind, client)
    predict = _mock_m1_predict if mock else _load_m1_predict()
    # M1 answers at temperature 0, so the same text always gets the same
    # verdict. Remembering verdicts avoids re-asking Ollama for repeated texts
    # (elites are re-tested every generation), which saves a lot of CPU time.
    return M1Detector(functools.lru_cache(maxsize=None)(predict))


# ---- one round -----------------------------------------------------------

def _run_one_round(app, client, rnd, detector_name, arm, trials, write_artifacts=True):
    """Run one round `trials` times. Return an aggregated stats dict.

    Files written (prefix = runs/<tag>_round<N>):
      _contaminated.jsonl  successful attacks for M1 - FIRST trial only, as before
      _attacks.jsonl       every attempt from EVERY trial (field 'trial')
      _generations.csv     best strategy of each generation, every trial
      _best.jsonl          final elite strategies of every trial (for goal 4)
    """
    asrs, blocks, successes, attempts = [], [], [], []
    best_roots = []
    gen_rows, best_rows = [], []
    first_attacker = None
    best_template = ""
    prefix = f"runs/{detector_name}_{arm}_round{rnd}"
    os.makedirs("runs", exist_ok=True)

    for t in range(trials):
        # A different random seed per trial (SEED, SEED+1, ...). With one shared
        # seed every trial drew identical pairs and mutations, hit the response
        # cache, and replayed trial 1 exactly - so std was always 0. Trial 0
        # still uses config.SEED, so it reproduces the earlier runs.
        attacker = EvolutionaryAttacker(app, client, rng=random.Random(config.SEED + t))
        best = attacker.run(verbose=(t == 0), **ARMS[arm])  # only narrate trial 1

        log = attacker.attack_log
        n = len(log)
        n_block = sum(1 for r in log if r.blocked)
        n_succ = sum(1 for r in log if r.injected_success)

        asrs.append(n_succ / max(1, n))
        blocks.append(n_block)
        successes.append(n_succ)
        attempts.append(n)
        best_roots.append(best[0].root if best else "")

        for row in attacker.gen_history:
            gen_rows.append({"trial": t, **row})
        for rank, s in enumerate(best):
            best_rows.append({"trial": t, "rank": rank, "template": s.template,
                              "origin": s.origin, "root": s.root,
                              "search_asr": round(s.fitness, 4)})

        if write_artifacts:
            attacker.dump_log(f"{prefix}_attacks.jsonl", trial=t, append=(t > 0))

        if t == 0:
            first_attacker = attacker
            best_template = best[0].template if best else ""

        if trials > 1:
            print(f"    trial {t + 1}/{trials}: ASR={n_succ / max(1, n):.3f} "
                  f"blocked={n_block}/{n} best_root={best_roots[-1]}")

    mean_asr = statistics.mean(asrs)
    std_asr = statistics.pstdev(asrs) if len(asrs) > 1 else 0.0

    if write_artifacts and first_attacker is not None:
        ex_path = f"{prefix}_contaminated.jsonl"
        with open(ex_path, "w", encoding="utf-8") as f:
            for ex in first_attacker.successful_examples():
                f.write(json.dumps(ex) + "\n")

        _write_csv(f"{prefix}_generations.csv", gen_rows)
        with open(f"{prefix}_best.jsonl", "w", encoding="utf-8") as f:
            for row in best_rows:
                f.write(json.dumps(row) + "\n")

        n_ex = len(first_attacker.successful_examples())
        print(f"          -> {n_ex} contaminated examples -> {ex_path} (for M1)")
        print(f"          -> attack log (all trials) -> {prefix}_attacks.jsonl")
        print(f"          -> per-generation best -> {prefix}_generations.csv")

    return {
        "round": rnd,
        "detector": detector_name,
        "arm": arm,
        "trials": trials,
        "asr_mean": round(mean_asr, 4),
        "asr_std": round(std_asr, 4),
        "attempts": attempts[0] if attempts else 0,
        "blocked_mean": round(statistics.mean(blocks), 1),
        "success_mean": round(statistics.mean(successes), 1),
        "best_roots": ";".join(best_roots),   # family of the final best, per trial
        "best_template": best_template,
    }


def _write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


# ---- main ----------------------------------------------------------------

def run(detector_kind: str, rounds: int, mock: bool, trials: int, arm: str = "baseline"):
    client = get_client(mock)
    detector = make_detector(detector_kind, client, mock)
    app = TargetApp(client, detector=detector)
    tag = f"{detector.name}_{arm}"

    os.makedirs("runs", exist_ok=True)
    history = []

    print(f"\n=== Closed loop: detector={detector.name}, arm={arm}, rounds={rounds}, "
          f"trials={trials}, mock={mock} ===")

    for rnd in range(rounds):
        print(f"\n[Round {rnd}] attacking (detector={detector.name}, arm={arm})")
        stats = _run_one_round(app, client, rnd, detector.name, arm, trials)
        history.append(stats)

        pct_block = 100 * stats["blocked_mean"] / max(1, stats["attempts"])
        print(f"[Round {rnd}] ASR = {stats['asr_mean']:.3f} "
              f"+/- {stats['asr_std']:.3f}  |  "
              f"blocked {stats['blocked_mean']:.0f}/{stats['attempts']} "
              f"({pct_block:.0f}%)  |  "
              f"succeeded {stats['success_mean']:.0f}/{stats['attempts']}")

        # In a real run, M1 retrains here on accumulated contaminated examples,
        # then you rebuild `detector`/`app` with the updated M1 and continue.
        # With a static detector the ASR curve across rounds shows what a
        # NON-adapting defense looks like - the static baseline.

    # ASR history CSV is per detector+arm so runs don't overwrite each other.
    out_csv = config.ASR_LOG.replace(".csv", f"_{tag}.csv")
    _write_csv(out_csv, history)
    print(f"\nASR history -> {out_csv}")
    return history


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--detector", default="none", choices=["none", "kad", "m1"])
    ap.add_argument("--arm", default="baseline", choices=list(ARMS),
                    help="baseline = original search; enriched = + structure "
                         "templates and structure-inspired operators.")
    ap.add_argument("--rounds", type=int, default=1)
    ap.add_argument("--trials", type=int, default=None,
                    help="How many times to repeat each round and average. "
                         "Defaults to config.REPEAT_TRIALS.")
    ap.add_argument("--mock", action="store_true",
                    help="Use the fake client - no Ollama needed. Plumbing test only.")
    args = ap.parse_args()
    trials = args.trials if args.trials is not None else config.REPEAT_TRIALS
    run(args.detector, args.rounds, args.mock, trials, args.arm)


if __name__ == "__main__":
    main()
