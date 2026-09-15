"""System-prompt leakage detection.

Two complementary signals:

1. Canary token — a high-entropy string embedded in the protected system prompt.
   If it appears in an output, the model has reproduced privileged text verbatim.
   Zero false positives by construction, but trivially defeated by an attacker
   who asks for a paraphrase.

2. N-gram containment — what fraction of the system prompt's n-grams appear in
   the output. Catches partial and lightly-reworded leaks that the canary misses.
   Containment (not Jaccard) is the right measure here: a short leak inside a
   long output should still score high, so we normalise by the system prompt,
   not by the union.

Together these cover the PLeak-style prompt-stealing attacks the base paper
lists as an application, but on the *output* side rather than the input side.
"""

from __future__ import annotations

import re
from typing import List, Optional, Set

from ..schemas import CheckResult, Severity, Timer

_WORD_RE = re.compile(r"[a-z0-9]+")


def _normalize(text: str) -> List[str]:
    return _WORD_RE.findall(text.lower())


def _ngrams(tokens: List[str], n: int) -> Set[str]:
    if len(tokens) < n:
        return {" ".join(tokens)} if tokens else set()
    return {" ".join(tokens[i:i + n]) for i in range(len(tokens) - n + 1)}


def _canary_present(text: str, canary: str) -> bool:
    """Exact match, plus tolerance for whitespace/case mangling by the model."""
    if canary in text:
        return True
    squashed_out = re.sub(r"\s+", "", text).lower()
    squashed_canary = re.sub(r"\s+", "", canary).lower()
    return squashed_canary in squashed_out


def containment(system_prompt: str, output: str, n: int) -> float:
    """Fraction of the system prompt's n-grams that appear in the output."""
    sp = _ngrams(_normalize(system_prompt), n)
    if not sp:
        return 0.0
    out = _ngrams(_normalize(output), n)
    return len(sp & out) / len(sp)


def run(
    output: str,
    system_prompt: Optional[str] = None,
    canary_token: Optional[str] = None,
    n: int = 8,
    threshold: float = 0.12,
) -> CheckResult:
    with Timer() as t:
        if not system_prompt and not canary_token:
            return CheckResult(
                name="system_prompt_leak",
                triggered=False,
                skipped=True,
                error="no system_prompt or canary_token configured",
            )

        if canary_token and _canary_present(output, canary_token):
            return CheckResult(
                name="system_prompt_leak",
                triggered=True,
                severity=Severity.HIGH,
                score=1.0,
                categories=["canary_leak", "system_prompt_leak"],
                evidence=["canary token reproduced verbatim in output"],
                latency_ms=t.ms,
            )

        if not system_prompt:
            return CheckResult(name="system_prompt_leak", triggered=False, latency_ms=t.ms)

        score = containment(system_prompt, output, n)

    if score < threshold:
        return CheckResult(
            name="system_prompt_leak",
            triggered=False,
            score=score,
            latency_ms=t.ms,
        )

    # Scale severity with how much of the prompt came back.
    severity = Severity.HIGH if score >= threshold * 3 else Severity.MEDIUM
    return CheckResult(
        name="system_prompt_leak",
        triggered=True,
        severity=severity,
        score=score,
        categories=["system_prompt_leak"],
        evidence=[f"{score:.0%} of protected system-prompt {n}-grams echoed"],
        latency_ms=t.ms,
    )
