"""The LLM-integrated application under test.

This is the thing M2 attacks. It mirrors the paper's formalisation:

    prompt = target_instruction || target_data
    output = backend_llm(prompt)

Optionally a detector (M1, or the known-answer baseline) sits in front of it.
When a detector is attached and flags the input, the app refuses and the
attack is scored as blocked. This is what makes the closed loop work: in
round N+1 the attacker is searching against the *defended* system.

An OPTIONAL output monitor (M3) can also sit AFTER the backend: once the LLM
has produced its output, the monitor inspects it for leaks / unsafe content
and reports allow / redact / block. It is injected (duck-typed: anything with
`.inspect(output, user_input=...)` returning a verdict), so this file keeps no
hard dependency on M3. It is OFF by default (monitor=None), and when it is off
the response is byte-for-byte what it was before - the measurement runs do not
see it. M3 is a SEPARATE output-side signal: it never changes `blocked` or
`output` (those stay the input-side detector's / backend's values), it only
records its verdict on the response for a caller (e.g. the end-to-end demo).
"""

from dataclasses import dataclass, field
from typing import List, Optional

from llm_client import OllamaClient
from tasks import Task


BLOCKED_RESPONSE = "[BLOCKED] This input was flagged as containing an injected instruction."


@dataclass
class AppResponse:
    output: str
    blocked: bool
    prompt: str
    # --- M3 output-monitor verdict (only set when a monitor is attached) ---
    monitor_decision: Optional[str] = None       # "allow" | "redact" | "block"
    monitor_blocked: bool = False
    sanitized_output: Optional[str] = None        # redacted/withheld text, if any
    monitor_categories: List[str] = field(default_factory=list)


class TargetApp:
    """A document-QA / text-processing bot built on a local LLM."""

    def __init__(self, client: OllamaClient, detector=None, model: Optional[str] = None,
                 monitor=None):
        self.client = client
        self.detector = detector
        self.model = model
        self.monitor = monitor         # optional M3 output monitor; None = off

    def query(self, target_task: Task, target_data: str) -> AppResponse:
        prompt = f"{target_task.instruction}\n\nText: {target_data}"

        if self.detector is not None and self.detector.is_contaminated(target_data):
            return AppResponse(output=BLOCKED_RESPONSE, blocked=True, prompt=prompt)

        output = self.client.generate(prompt, model=self.model)
        resp = AppResponse(output=output, blocked=False, prompt=prompt)

        # Optional M3 output stage. Records its verdict; does NOT alter output
        # or blocked, so a run with no monitor is completely unaffected.
        if self.monitor is not None:
            verdict = self.monitor.inspect(output, user_input=target_data)
            resp.monitor_decision = verdict.decision.value
            resp.monitor_blocked = verdict.blocked
            resp.sanitized_output = verdict.sanitized_output
            resp.monitor_categories = list(verdict.categories)

        return resp
