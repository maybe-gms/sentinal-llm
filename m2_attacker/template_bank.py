"""Loads the structure-category separator templates for M2.

The templates live in templates/structure_templates.json. They are
hand-written, benign versions of the STRUCTURES seen in jailbreak prompts
(persona, refusal suppression, fake system message, delimiter injection,
ignore-above). Each has exactly one {INJ} slot, which the attacker fills with
one of the benign injected tasks from tasks.py - exactly like SEED_SEPARATORS.

Each template has a 'split':
  train   - may be used to seed the evolutionary search
  heldout - never used during search; only for the generalisation test

Run this file directly to check the template file and print a summary:
    py template_bank.py
"""

import json
import os
from collections import Counter
from typing import Dict, List, Optional

TEMPLATE_PATH = os.path.join(os.path.dirname(__file__), "templates", "structure_templates.json")

CATEGORIES = ["persona", "refusal_suppression", "fake_system", "delimiter", "ignore_above"]
SPLITS = ["train", "heldout"]


def load_templates(split: Optional[str] = None, path: str = TEMPLATE_PATH) -> List[Dict[str, str]]:
    """Return the list of template dicts (id, category, split, template).

    Pass split="train" or split="heldout" to get only that slice.
    Raises ValueError if any template is malformed, so a typo in the JSON
    file fails loudly instead of silently producing a broken attack.
    """
    with open(path, "r", encoding="utf-8") as f:
        items = json.load(f)["templates"]

    seen_ids = set()
    for t in items:
        if t["id"] in seen_ids:
            raise ValueError(f"duplicate template id: {t['id']}")
        seen_ids.add(t["id"])
        if t["template"].count("{INJ}") != 1:
            raise ValueError(f"{t['id']}: template must contain {{INJ}} exactly once")
        if t["category"] not in CATEGORIES:
            raise ValueError(f"{t['id']}: unknown category {t['category']!r}")
        if t["split"] not in SPLITS:
            raise ValueError(f"{t['id']}: unknown split {t['split']!r}")

    if split is not None:
        items = [t for t in items if t["split"] == split]
    return items


if __name__ == "__main__":
    all_t = load_templates()
    print(f"{len(all_t)} templates loaded from {TEMPLATE_PATH}")
    counts = Counter((t["category"], t["split"]) for t in all_t)
    print(f"{'category':<22}{'train':>6}{'heldout':>9}")
    for c in CATEGORIES:
        print(f"{c:<22}{counts[(c, 'train')]:>6}{counts[(c, 'heldout')]:>9}")
