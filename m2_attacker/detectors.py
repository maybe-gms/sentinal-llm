"""Detectors that can sit in front of the target app.

Three implementations:

  NullDetector          - no defense. Use for Act 1 (baseline vulnerability).
  KnownAnswerDetector   - the KAD baseline from the paper, implemented over
                          Ollama. Works immediately with no training, and is
                          exactly the baseline your M1 has to beat.
  M1Detector            - hook for your teammate's fine-tuned detector.

The KAD rule (paper Eq. 1):
    contaminated  if  secret_key NOT IN detection_llm(detection_instruction || x)
    clean         otherwise
"""

from typing import Callable, Optional

import config
from llm_client import OllamaClient


class NullDetector:
    name = "none"

    def is_contaminated(self, target_data: str) -> bool:
        return False


class KnownAnswerDetector:
    """Baseline detector. No training required."""

    name = "known_answer"

    def __init__(self, client: OllamaClient, secret_key: Optional[str] = None):
        self.client = client
        self.secret_key = secret_key or config.SECRET_KEY

    def is_contaminated(self, target_data: str) -> bool:
        instruction = config.DETECTION_INSTRUCTION.format(key=self.secret_key)
        prompt = f"{instruction}\n\nText: {target_data}"
        output = self.client.generate(
            prompt, model=config.DETECTOR_MODEL, max_tokens=64
        )
        return self.secret_key not in output


class M1Detector:
    """Wraps whatever M1 exposes.

    Give it any callable that takes the raw target data string and returns
    True for 'contaminated'. If your teammate's M1 is a fine-tuned known-answer
    detector, that callable is just KnownAnswerDetector pointed at the
    fine-tuned model, so you can also do:

        M1Detector(KnownAnswerDetector(client_with_finetuned_model).is_contaminated)
    """

    name = "m1"

    def __init__(self, predict_fn: Callable[[str], bool]):
        self.predict_fn = predict_fn

    def is_contaminated(self, target_data: str) -> bool:
        return bool(self.predict_fn(target_data))


def build_detector(kind: str, client: OllamaClient):
    kind = (kind or "none").lower()
    if kind in ("none", "null", "off"):
        return NullDetector()
    if kind in ("kad", "known_answer"):
        return KnownAnswerDetector(client)
    raise ValueError(
        f"Unknown detector '{kind}'. Use 'none' or 'kad', or construct "
        f"M1Detector yourself once M1 is trained."
    )
