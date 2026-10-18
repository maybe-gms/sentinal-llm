"""Turns the run outputs in runs/ into slide-ready tables and charts.

Reads (written by run_loop.py):
    runs/asr_history_<detector>_<arm>.csv       ASR mean/std per round
    runs/<detector>_<arm>_round0_attacks.jsonl  every attempt, all trials

Writes:
    runs/chart_arms.png        (1) ASR baseline vs enriched, per defender
    runs/chart_by_task.png     (2) ASR by injected task
    runs/chart_by_origin.png   (3) which template family the attacks came from
    runs/report.md             the same numbers as markdown tables

Only round 0 is used: with a static (non-retraining) detector every round is
the same experiment, so round 0 is the clean comparison.

Usage:
    py report.py                 # reads ./runs
    py report.py --runs some_dir
"""

import argparse
import csv
import glob
import json
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")          # draw straight to files, no window needed
import matplotlib.pyplot as plt

ARMS = ["baseline", "enriched"]
DETECTOR_ORDER = ["none", "known_answer", "m1"]
DETECTOR_LABEL = {"none": "No defense", "known_answer": "KAD baseline", "m1": "M1 detector"}
TASK_ORDER = ["inj_spam", "inj_sentiment", "inj_redirect", "inj_prompt_steal", "inj_availability"]

# One colour per arm, always the same colour for the same arm. Checked for
# colour-blind safety; every bar is also labelled with its value, so colour is
# never the only way to read the chart.
ARM_COLOR = {"baseline": "#2a78d6", "enriched": "#eb6834"}
SURFACE, INK, INK_2, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7"


# ---- reading -------------------------------------------------------------

def read_history(runs):
    """{(detector, arm): round-0 row dict} from every asr_history_<det>_<arm>.csv."""
    out = {}
    for path in glob.glob(os.path.join(runs, "asr_history_*.csv")):
        name = os.path.basename(path)[len("asr_history_"):-len(".csv")]
        detector, _, arm = name.rpartition("_")
        if arm not in ARMS:
            continue                     # skip old files from before --arm existed
        rows = list(csv.DictReader(open(path, encoding="utf-8")))
        if rows:
            out[(detector, arm)] = rows[0]
    return out


def read_attacks(runs, detector, arm):
    """Every attempt (all trials) for one detector+arm, or [] if not run."""
    path = os.path.join(runs, f"{detector}_{arm}_round0_attacks.jsonl")
    if not os.path.exists(path):
        return []
    return [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]


def rate(rows):
    """Fraction of attempts that succeeded (= ASR)."""
    return sum(bool(r["injected_success"]) for r in rows) / len(rows) if rows else 0.0


def present_detectors(history):
    """Detectors that actually have data, in a sensible order."""
    found = {d for (d, _) in history}
    return [d for d in DETECTOR_ORDER if d in found]


# ---- chart chrome --------------------------------------------------------

def _style(ax, title, ylabel="ASR", ymax=1.08):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, color=INK, fontsize=10, loc="left")
    ax.set_ylabel(ylabel, color=INK_2)
    ax.set_ylim(0, ymax)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ["top", "right", "left"]:
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.tick_params(colors=INK_2, length=0)


def _label_bars(ax, bars, texts, offsets=None):
    """Write each bar's value just above it (clear of any error bar)."""
    offsets = offsets or [0.0] * len(bars)
    for bar, text, off in zip(bars, texts, offsets):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + off + 0.02,
                text, ha="center", va="bottom", fontsize=8, color=INK_2)


def _save(fig, path):
    fig.savefig(path, dpi=150, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {path}")


# ---- chart 1: baseline vs enriched, per defender -------------------------

def chart_arms(history, runs):
    """Grouped bars: for each defender, baseline ASR beside enriched ASR,
    with the +/- std error bar. This is the headline 'did the structure
    templates help?' chart."""
    detectors = present_detectors(history)
    if not detectors:
        return
    fig, ax = plt.subplots(figsize=(2.2 + 1.4 * len(detectors), 4), layout="constrained")
    width = 0.38
    for i, arm in enumerate(ARMS):
        xs, heights, errs, labels = [], [], [], []
        for j, det in enumerate(detectors):
            row = history.get((det, arm))
            if not row:
                continue
            xs.append(j + (i - 0.5) * width)
            heights.append(float(row["asr_mean"]))
            errs.append(float(row["asr_std"]))
            labels.append(f"{float(row['asr_mean']):.2f}")
        bars = ax.bar(xs, heights, width, yerr=errs, capsize=3,
                      color=ARM_COLOR[arm], label=arm,
                      error_kw={"ecolor": INK_2, "elinewidth": 1})
        _label_bars(ax, bars, labels, offsets=errs)   # clear the error-bar cap
    ax.set_xticks(range(len(detectors)))
    ax.set_xticklabels([DETECTOR_LABEL.get(d, d) for d in detectors], color=INK_2)
    _style(ax, "Attack success: baseline vs enriched")
    ax.legend(frameon=False, labelcolor=INK_2)
    _save(fig, os.path.join(runs, "chart_arms.png"))


# ---- chart 2: ASR by injected task ---------------------------------------

def chart_by_task(history, runs):
    """For the no-defense run, ASR split by injected task, baseline vs
    enriched. Shows which attack goals got easier."""
    detector = "none" if ("none", "baseline") in history else None
    if detector is None:
        return
    tasks_seen = []
    per_arm = {}
    for arm in ARMS:
        rows = read_attacks(runs, detector, arm)
        if not rows:
            continue
        by_task = defaultdict(list)
        for r in rows:
            by_task[r["injected_task"]].append(r)
        per_arm[arm] = {t: rate(by_task[t]) for t in by_task}
        for t in by_task:
            if t not in tasks_seen:
                tasks_seen.append(t)
    if not per_arm:
        return
    tasks = [t for t in TASK_ORDER if t in tasks_seen] + \
            [t for t in tasks_seen if t not in TASK_ORDER]

    fig, ax = plt.subplots(figsize=(2.5 + 1.1 * len(tasks), 4), layout="constrained")
    width = 0.38
    for i, arm in enumerate(ARMS):
        if arm not in per_arm:
            continue
        xs = [j + (i - 0.5) * width for j in range(len(tasks))]
        heights = [per_arm[arm].get(t, 0.0) for t in tasks]
        bars = ax.bar(xs, heights, width, color=ARM_COLOR[arm], label=arm)
        _label_bars(ax, bars, [f"{h:.2f}" for h in heights])
    ax.set_xticks(range(len(tasks)))
    ax.set_xticklabels([t.replace("inj_", "") for t in tasks], color=INK_2, rotation=20, ha="right")
    _style(ax, "Attack success by injected task (no defense)")
    ax.legend(frameon=False, labelcolor=INK_2)
    _save(fig, os.path.join(runs, "chart_by_task.png"))


# ---- chart 3: which template family the successes came from --------------

def chart_by_origin(history, runs):
    """For the enriched no-defense run, how many SUCCESSFUL attacks came from
    each starting family (seed vs each structure category). Answers 'which
    origin produced the best strategies'."""
    rows = read_attacks(runs, "none", "enriched")
    successes = [r for r in rows if r["injected_success"]]
    if not successes:
        return
    counts = defaultdict(int)
    for r in successes:
        counts[r.get("root", "seed")] += 1
    families = sorted(counts, key=lambda k: counts[k], reverse=True)
    heights = [counts[f] for f in families]

    fig, ax = plt.subplots(figsize=(2.5 + 1.0 * len(families), 4), layout="constrained")
    bars = ax.bar(range(len(families)), heights, 0.6, color=ARM_COLOR["enriched"])
    _label_bars(ax, bars, [str(h) for h in heights])
    ax.set_xticks(range(len(families)))
    ax.set_xticklabels([f.replace("structure:", "") for f in families],
                       color=INK_2, rotation=20, ha="right")
    # This chart counts attacks, not ASR, so give it its own y-axis height.
    _style(ax, "Successful attacks by starting family (enriched, no defense)",
           ylabel="successful attacks", ymax=max(heights) * 1.25 + 1)
    _save(fig, os.path.join(runs, "chart_by_origin.png"))


# ---- markdown report -----------------------------------------------------

def write_markdown(history, runs):
    detectors = present_detectors(history)
    lines = ["# M2 results\n", "## ASR: baseline vs enriched\n",
             "| Defender | Arm | ASR mean | ASR std | Blocked | Attempts |",
             "|---|---|---|---|---|---|"]
    for det in detectors:
        for arm in ARMS:
            row = history.get((det, arm))
            if not row:
                continue
            lines.append(f"| {DETECTOR_LABEL.get(det, det)} | {arm} | "
                         f"{row['asr_mean']} | {row['asr_std']} | "
                         f"{row['blocked_mean']} | {row['attempts']} |")
    path = os.path.join(runs, "report.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  -> {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs", help="folder with the run outputs")
    args = ap.parse_args()

    history = read_history(args.runs)
    if not history:
        print(f"No asr_history_<detector>_<arm>.csv files found in {args.runs}/. "
              f"Run run_loop.py first.")
        return
    print(f"Found results for: {sorted(history)}")
    chart_arms(history, args.runs)
    chart_by_task(history, args.runs)
    chart_by_origin(history, args.runs)
    write_markdown(history, args.runs)


if __name__ == "__main__":
    main()
