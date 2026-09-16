"""The LLM-integrated application under test.

This is the thing M2 attacks. It mirrors the paper's formalisation:

    prompt = target_instruction || target_data
    output = backend_llm(prompt)

Two optional defenses can be attached:

  detector      (M1, or the KAD baseline) checks the target data BEFORE the
                LLM runs. If it flags the input, the LLM never sees it.
  output_guard  (M3) checks the LLM's answer AFTER it runs. If it flags the
                answer, the user gets a withheld/redacted response instead.

Either one stopping the attack counts as blocked. `blocked_by` records which
layer did it, so we can report how much each module contributes.
"""

from dataclasses import dataclass
from typing import Optional

from llm_client import OllamaClient
from tasks import Task


BLOCKED_RESPONSE = "[BLOCKED] This input was flagged as containing an injected instruction."


@dataclass
class AppResponse:
    output: str                      # what the user receives
    blocked: bool                    # stopped by ANY defense layer
    prompt: str
    blocked_by: Optional[str] = None  # None, "detector", or "monitor"
    raw_output: str = ""             # what the LLM said before the output guard


class TargetApp:
    """A document-QA / text-processing bot built on a local LLM."""

    def __init__(self, client: OllamaClient, detector=None, model: Optional[str] = None,
                 output_guard=None):
        self.client = client
        self.detector = detector
        self.model = model
        self.output_guard = output_guard

    def query(self, target_task: Task, target_data: str) -> AppResponse:
        prompt = f"{target_task.instruction}\n\nText: {target_data}"

        # Door 1: input detector (M1 / KAD)
        if self.detector is not None and self.detector.is_contaminated(target_data):
            return AppResponse(output=BLOCKED_RESPONSE, blocked=True, prompt=prompt,
                               blocked_by="detector")

        output = self.client.generate(prompt, model=self.model)

        # Door 2: output guard (M3)
        if self.output_guard is not None:
            check = self.output_guard.check(target_task.instruction, target_data, output)
            if check.blocked:
                return AppResponse(output=check.output, blocked=True, prompt=prompt,
                                   blocked_by="monitor", raw_output=output)

        return AppResponse(output=output, blocked=False, prompt=prompt, raw_output=output)
