"""The LLM-integrated application under test.

This is the thing M2 attacks. It mirrors the paper's formalisation:

    prompt = target_instruction || target_data
    output = backend_llm(prompt)

Optionally a detector (M1, or the known-answer baseline) sits in front of it.
When a detector is attached and flags the input, the app refuses and the
attack is scored as blocked. This is what makes the closed loop work: in
round N+1 the attacker is searching against the *defended* system.
"""

from dataclasses import dataclass
from typing import Optional

from llm_client import OllamaClient
from tasks import Task


BLOCKED_RESPONSE = "[BLOCKED] This input was flagged as containing an injected instruction."


@dataclass
class AppResponse:
    output: str
    blocked: bool
    prompt: str


class TargetApp:
    """A document-QA / text-processing bot built on a local LLM."""

    def __init__(self, client: OllamaClient, detector=None, model: Optional[str] = None):
        self.client = client
        self.detector = detector
        self.model = model

    def query(self, target_task: Task, target_data: str) -> AppResponse:
        prompt = f"{target_task.instruction}\n\nText: {target_data}"

        if self.detector is not None and self.detector.is_contaminated(target_data):
            return AppResponse(output=BLOCKED_RESPONSE, blocked=True, prompt=prompt)

        output = self.client.generate(prompt, model=self.model)
        return AppResponse(output=output, blocked=False, prompt=prompt)
