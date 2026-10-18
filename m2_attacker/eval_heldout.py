"""Goal 4 - generalisation check (no evolution).

The enriched search is allowed to seed and mutate only the TRAIN-split
structure templates. The HELD-OUT templates (one per category) are never shown
to the search. If those held-out templates - which the search never touched -
land attacks about as well as the train templates, then what works is the
STRUCTURE FAMILY (persona, fake-system, ...), not the specific templates the
search happened to use. That is the difference between a real property and
cherry-picking.

This script tests each template DIRECTLY: no mutation, no selection. Every
template is scored on the SAME fixed set of (target, injected) pairs, through
the exact same scoring path the main experiment uses, against the chosen
defender. Three groups are compared:

    seeds     - the five original SEED_SEPARATORS (reference point)
    train     - the train-split structure templates (the search could use these)
    heldout   - the held-out structure templates (the search never saw these)

Read the result as: heldout ASR ~ train ASR  =>  generalises, not memorised.

Usage:
    py eval_heldout.py --mock
    py eval_heldout.py --detector none
    py eval_heldout.py --detector m1 --pairs 16
"""

import argparse
import csv
import os
import random
import statistics

import config
from attacker import EvolutionaryAttacker, Strategy, SEED_SEPARATORS
from target_app import TargetApp
from template_bank import load_templates
from run_loop import get_client, make_detector


def _groups():
    """The three groups of templates to compare, as Strategy objects."""
    seeds = [Strategy(template=s, origin="seed", root="seed") for s in SEED_SEPARATORS]
    train = [Strategy(template=t["template"], origin=t["id"], root=f"structure:{t['category']}")
             for t in load_templates("train")]
    heldout = [Strategy(template=t["template"], origin=t["id"], root=f"structure:{t['category']}")
               for t in load_templates("heldout")]
    return {"seeds": seeds, "train": train, "heldout": heldout}


def evaluate(detector_kind, mock, pairs_n):
    client = get_client(mock)
    detector = make_detector(detector_kind, client, mock)
    app = TargetApp(client, detector=detector)

    # One attacker instance, only so we can reuse its fixed pair sampler and its
    # scoring. The SAME pairs are used for every template, so the comparison is
    # fair. No mutation or selection happens here.
    scorer = EvolutionaryAttacker(app, client, rng=random.Random(config.SEED))
    pairs = scorer._sample_pairs(pairs_n)

    groups = _groups()
    rows = []          # one row per template
    for group_name, strategies in groups.items():
        for strat in strategies:
            scorer.attack_log.clear()                 # score this template alone
            scorer._eval_strategy(strat, pairs)
            rows.append({"group": group_name, "template_id": strat.origin,
                         "family": strat.root, "asr": round(strat.fitness, 4)})
    client.flush()
    return detector.name, pairs_n, rows


def summarise(rows):
    """Group -> (mean ASR across its templates, std, n templates)."""
    out = {}
    for group in ["seeds", "train", "heldout"]:
        vals = [r["asr"] for r in rows if r["group"] == group]
        if vals:
            std = statistics.pstdev(vals) if len(vals) > 1 else 0.0
            out[group] = (statistics.mean(vals), std, len(vals))
    return out


def draw_chart(detector_name, summary, path):
    try:
        import report  # reuse the chart styling from report.py
    except Exception as e:
        print(f"  (skipping chart: {e})")
        return
    import matplotlib.pyplot as plt
    groups = list(summary)
    means = [summary[g][0] for g in groups]
    errs = [summary[g][1] for g in groups]
    fig, ax = plt.subplots(figsize=(5, 4), layout="constrained")
    colours = {"seeds": "#898781", "train": "#2a78d6", "heldout": "#eb6834"}
    bars = ax.bar(range(len(groups)), means, 0.6, yerr=errs, capsize=3,
                  color=[colours[g] for g in groups],
                  error_kw={"ecolor": report.INK_2, "elinewidth": 1})
    report._label_bars(ax, bars, [f"{m:.2f}" for m in means], offsets=errs)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([f"{g}\n(n={summary[g][2]})" for g in groups], color=report.INK_2)
    report._style(ax, f"Generalisation: ASR by template group ({detector_name})")
    report._save(fig, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--detector", default="none", choices=["none", "kad", "m1"])
    ap.add_argument("--pairs", type=int, default=12,
                    help="How many fixed (target, injected) pairs each template faces.")
    ap.add_argument("--mock", action="store_true")
    args = ap.parse_args()

    detector_name, pairs_n, rows = evaluate(args.detector, args.mock, args.pairs)
    summary = summarise(rows)

    print(f"\n=== Generalisation (detector={detector_name}, {pairs_n} fixed pairs) ===")
    print(f"{'group':<10}{'mean ASR':>10}{'std':>8}{'n':>4}")
    for group, (mean, std, n) in summary.items():
        print(f"{group:<10}{mean:>10.3f}{std:>8.3f}{n:>4}")
    if "train" in summary and "heldout" in summary:
        gap = summary["heldout"][0] - summary["train"][0]
        print(f"\nheldout - train = {gap:+.3f}  "
              f"(near zero => the structure family generalises, not memorised)")

    os.makedirs("runs", exist_ok=True)
    csv_path = f"runs/heldout_{detector_name}.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["group", "template_id", "family", "asr"])
        w.writeheader()
        w.writerows(rows)
    print(f"\nper-template ASR -> {csv_path}")
    draw_chart(detector_name, summary, f"runs/chart_generalization_{detector_name}.png")


if __name__ == "__main__":
    main()
