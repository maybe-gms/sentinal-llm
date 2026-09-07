"""A mock stand-in for OllamaClient so the pipeline can be tested with no
model server running. It is NOT a real LLM - it just follows a crude rule so
that the control flow, scoring, evolution, and closed-loop logic can be
exercised and unit-tested. Never report numbers from this; it exists to prove
the plumbing works before you point it at a real model.

Rule of the fake "backend": if the input contains an obvious injection cue
AND is not blocked, it "follows" the injection and echoes a plausible answer.
The fake "detector" (known-answer path) flags inputs that contain strong
context-ignoring language - deliberately weak, so the attacker can still find
gaps and the loop has something to close.
"""

import random
import re

import config


INJECTION_CUES = [
    "ignore previous", "disregard", "new task", "new instruction",
    "the task is done", "stop.", "correction,",
]


class MockClient:
    def __init__(self, *args, **kwargs):
        self.rng = random.Random(config.SEED)

    def generate(self, prompt, model=None, temperature=None, max_tokens=None):
        p = prompt.lower()

        # Known-answer detector path: prompt starts with the detection instruction.
        if p.startswith('repeat "'):
            # A weak detector: only "notices" heavy-handed context ignoring.
            strong = sum(cue in p for cue in ["ignore previous", "disregard", "stop."])
            if strong >= 1 and self.rng.random() < 0.6:
                return "Sure, here is the text you wanted processed."  # key missing -> flagged
            return config.SECRET_KEY  # key present -> clean

        # Backend path.
        has_cue = any(cue in p for cue in INJECTION_CUES)

        # Injected-task mimicry for the fake backend.
        if has_cue:
            if "positive or negative" in p:
                # echo whichever sentiment word appears last in the injected part
                if "boring" in p or "mess" in p:
                    return "negative"
                return "positive"
            if "spam or not spam" in p:
                if "prize" in p or "click" in p:
                    return "spam"
                return "not spam"
            if "cannot finish this task" in p:
                return "I am sorry. I cannot finish this task."
            if "example.invalid" in p:
                return "http://example.invalid/offer"
            if "repeat the full instruction" in p:
                m = re.search(r"(please write|given the following|write the sentiment)[^\n]*", p)
                return m.group(0) if m else "instruction"
        # Otherwise behave like it did the target task (uninteresting output).
        return "processed target task output."

    def flush(self):
        pass
