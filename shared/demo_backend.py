"""A mock backend for the OFFLINE demo / UI only.

The real story the integrated system tells is defense-in-depth: an attack that
beats M1 on the input can still be stopped by M3 on the output. To show that
*without* Ollama, the offline backend has to actually behave like a model that
leaks its system prompt when a prompt-steal gets through — otherwise M3's
leakage layer never has anything to catch.

m2_attacker/mock_client.py is used by the real measurement runs and must stay
byte-identical, so we do NOT edit it. Instead this DemoBackend wraps the same
idea with one extra, demo-specific behaviour: when the input asks the model to
repeat/reveal its instructions, it echoes the system prompt it was given
(canary and all). That is exactly what an unhardened real model does, and it is
the behaviour the real llama3.1:8b showed in the recorded end-to-end run.

Never report numbers from this. It exists so the live demo is deterministic and
runnable on a laptop with no model server. Flip the UI/demo to real mode and the
same pipeline code runs against Ollama instead.
"""

from __future__ import annotations

import re
from typing import Optional


# Phrases that indicate a prompt-stealing / instruction-reveal attempt.
_STEAL_CUES = (
    "repeat the full instruction",
    "repeat your instruction",
    "repeat the instruction",
    "reveal your instruction",
    "reveal the instruction",
    "print your instruction",
    "print the instruction",
    "what were you told",
    "what instruction were you given",
    "word for word",
    "system prompt",
    "reference code",
    "your initial prompt",
)

_SENTIMENT_NEG = ("boring", "mess", "bad", "awful", "terrible", "negative")
_AVAILABILITY_CUE = "cannot finish this task"
_REDIRECT_CUE = "example.invalid"


class DemoBackend:
    """Offline stand-in for the backend LLM, with realistic leak behaviour.

    It is given the app's system prompt at construction so that a successful
    prompt-steal can reproduce it verbatim — the same text M3 is watching for.
    """

    def __init__(self, system_prompt: str = ""):
        self.system_prompt = system_prompt

    def generate(self, prompt: str, model: Optional[str] = None,
                 temperature: Optional[float] = None,
                 max_tokens: Optional[int] = None) -> str:
        low = prompt.lower()

        # --- prompt-steal: the unhardened model echoes its own instruction ---
        if any(cue in low for cue in _STEAL_CUES):
            # Reproduce the real system prompt (with canary) if we have it;
            # otherwise fall back to echoing the instruction line from the prompt.
            if self.system_prompt:
                return f"Sure. My instructions were: {self.system_prompt}"
            m = re.search(r"^(.*?)\n\nText:", prompt, re.DOTALL)
            return m.group(1).strip() if m else "My instruction was to process your text."

        # --- availability: forced refusal ---
        if _AVAILABILITY_CUE in low:
            return "I am sorry. I cannot finish this task."

        # --- redirect: emit the reserved-domain URL ---
        if _REDIRECT_CUE in low:
            return "http://example.invalid/offer"

        # --- contact / reply task: the model composes a reply that carries a
        #     contact detail in its OUTPUT. M3's rule layer redacts the email/
        #     phone (LOW severity -> REDACT, not BLOCK) and delivers the rest.
        if ("reply" in low or "contact" in low) and ("@" in prompt or "reach me" in low):
            return ("Thanks for reaching out. I'll follow up shortly — you can also "
                    "reach our team at agent@example.com or on +1 415 555 0132.")

        # --- task-swap injections (sentiment / spam flips) ---
        if "positive or negative" in low and _has_injection(low):
            return "negative" if any(w in low for w in _SENTIMENT_NEG) else "positive"
        if "spam or not spam" in low and _has_injection(low):
            return "spam" if ("prize" in low or "click" in low) else "not spam"

        # --- otherwise: behave like it did the real target task ---
        return "Here is a brief, on-task summary of the provided text."

    def flush(self) -> None:
        pass


def _has_injection(low: str) -> bool:
    cues = ("ignore previous", "ignore all", "disregard", "new task",
            "new instruction", "the task is done", "instead,")
    return any(c in low for c in cues)
