#!/usr/bin/env python3
"""Convert AEGIS2.0 into the JSONL format evaluate.py expects.

AEGIS2.0 (Ghosh et al., NAACL 2025) is your own M3 seminar paper's dataset, and
unlike HarmBench or JailbreakBench it carries *response* labels, not just prompt
labels. That is what makes it usable for an output-side monitor.

  https://huggingface.co/datasets/nvidia/Aegis-AI-Content-Safety-Dataset-2.0

This is the only script in the project that needs pip packages:

    pip install pandas pyarrow

Then either let it download:

    py prepare_aegis.py --out datasets/aegis_eval.jsonl --limit 500

or point it at a file you downloaded by hand from the HF "Files" tab:

    py prepare_aegis.py --file aegis2_test.parquet --out datasets/aegis_eval.jsonl

Column names on HF datasets do move between revisions, so this script detects
them rather than hard-coding, and prints what it found. If detection fails it
lists the columns so you can pass them explicitly with --response-col etc.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

RESPONSE_CANDIDATES = ["response", "assistant", "completion", "answer", "output"]
PROMPT_CANDIDATES = ["prompt", "user", "user_message", "instruction", "input"]
LABEL_CANDIDATES = [
    "response_label", "label_response", "response_safety",
    "labels_response", "response_violated_categories", "label",
]
CATEGORY_CANDIDATES = [
    "response_violated_categories", "violated_categories",
    "response_category", "category",
]


def pick(cols, candidates, explicit=None):
    if explicit:
        if explicit not in cols:
            sys.exit(f"  column '{explicit}' not in dataset. Columns: {list(cols)}")
        return explicit
    lower = {c.lower(): c for c in cols}
    for cand in candidates:
        if cand in lower:
            return lower[cand]
    return None


def normalise_label(value) -> str | None:
    """Map whatever the label column holds onto safe / unsafe."""
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in ("", "nan", "none", "null"):
        return None
    if "unsafe" in text or text in ("1", "true", "harmful", "needs caution"):
        return "unsafe"
    if "safe" in text or text in ("0", "false", "harmless"):
        return "safe"
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", help="local .parquet/.csv instead of downloading")
    ap.add_argument("--split", default="train")
    ap.add_argument("--out", default="datasets/aegis_eval.jsonl")
    ap.add_argument("--limit", type=int, default=500,
                    help="cap rows; keep it small until the pipeline is fast")
    ap.add_argument("--balance", action="store_true",
                    help="equal numbers of safe and unsafe rows")
    ap.add_argument("--response-col")
    ap.add_argument("--prompt-col")
    ap.add_argument("--label-col")
    args = ap.parse_args()

    try:
        import pandas as pd
    except ImportError:
        sys.exit("  needs pandas:  pip install pandas pyarrow")

    if args.file:
        df = (pd.read_csv(args.file) if args.file.endswith(".csv")
              else pd.read_parquet(args.file))
    else:
        try:
            from datasets import load_dataset
        except ImportError:
            sys.exit("  needs the datasets library:  pip install datasets\n"
                     "  (or download a file from the HF Files tab and use --file)")
        ds = load_dataset("nvidia/Aegis-AI-Content-Safety-Dataset-2.0",
                          split=args.split)
        df = ds.to_pandas()

    cols = list(df.columns)
    print(f"\n  loaded {len(df)} rows")
    print(f"  columns: {cols}\n")

    resp = pick(cols, RESPONSE_CANDIDATES, args.response_col)
    prompt = pick(cols, PROMPT_CANDIDATES, args.prompt_col)
    label = pick(cols, LABEL_CANDIDATES, args.label_col)
    category = pick(cols, CATEGORY_CANDIDATES)

    if not resp or not label:
        sys.exit("  could not auto-detect the response and label columns.\n"
                 "  Re-run with --response-col and --label-col set from the "
                 "list above.")
    print(f"  response column : {resp}")
    print(f"  prompt column   : {prompt or '(none — will be blank)'}")
    print(f"  label column    : {label}")
    print(f"  category column : {category or '(none)'}\n")

    rows, skipped = [], 0
    for i, row in df.iterrows():
        text = row.get(resp)
        if not isinstance(text, str) or not text.strip():
            skipped += 1
            continue
        lab = normalise_label(row.get(label))
        if lab is None:
            skipped += 1
            continue
        cat = row.get(category) if category else None
        cat = str(cat).strip() if cat is not None and str(cat).strip() else "aegis"
        rows.append({
            "id": f"aegis{i}",
            "user_input": str(row.get(prompt) or "") if prompt else "",
            "output": text.strip(),
            "label": lab,
            "category": f"{cat}_guard" if lab == "unsafe" else "benign",
        })

    if args.balance:
        unsafe = [r for r in rows if r["label"] == "unsafe"]
        safe = [r for r in rows if r["label"] == "safe"]
        n = min(len(unsafe), len(safe), args.limit // 2)
        rows = unsafe[:n] + safe[:n]
    rows = rows[:args.limit]

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    n_unsafe = sum(1 for r in rows if r["label"] == "unsafe")
    print(f"  wrote {len(rows)} rows to {args.out} "
          f"({n_unsafe} unsafe / {len(rows) - n_unsafe} safe), {skipped} skipped")
    print(f"\n  next:  py evaluate.py {args.out}\n")
    print("  Note: these rows exercise the Llama Guard path only. They contain")
    print("  no system-prompt leaks or credentials, so rules_pii and")
    print("  system_prompt_leak will show near-zero recall on this set — that")
    print("  is correct, not a bug. Report the two sets separately.\n")


if __name__ == "__main__":
    main()
