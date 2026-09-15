"""Shared data types for M3 — Output Safety Monitor.

Every check in the module returns a CheckResult; the monitor aggregates them
into a single Verdict. Keeping these dataclasses stable is what lets M2
(red-teaming) and the closed-loop driver treat M3 as a black box.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Dict, List, Optional


class Decision(str, Enum):
    """Final action taken on a model output."""

    ALLOW = "allow"      # nothing triggered — output passes through untouched
    REDACT = "redact"    # recoverable issue (PII) — output is sanitised, then passed
    BLOCK = "block"      # unsafe / leaked — output is withheld from the user


class Severity(str, Enum):
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


SEVERITY_ORDER = {
    Severity.NONE: 0,
    Severity.LOW: 1,
    Severity.MEDIUM: 2,
    Severity.HIGH: 3,
}


@dataclass
class CheckResult:
    """Outcome of a single detector inside M3."""

    name: str
    triggered: bool
    severity: Severity = Severity.NONE
    score: float = 0.0                       # 0.0-1.0 confidence / magnitude
    categories: List[str] = field(default_factory=list)
    evidence: List[str] = field(default_factory=list)   # redacted snippets only
    spans: List[tuple] = field(default_factory=list)    # (start, end, label)
    latency_ms: float = 0.0
    skipped: bool = False                    # e.g. Llama Guard unreachable
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["severity"] = self.severity.value
        d["spans"] = [list(s) for s in self.spans]
        return d


@dataclass
class Verdict:
    """Aggregated M3 result for one model output."""

    decision: Decision
    blocked: bool
    original_output: str
    sanitized_output: str
    categories: List[str] = field(default_factory=list)
    max_severity: Severity = Severity.NONE
    checks: List[CheckResult] = field(default_factory=list)
    latency_ms: float = 0.0
    session_id: Optional[str] = None
    round_id: Optional[int] = None
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def triggered_checks(self) -> List[str]:
        return [c.name for c in self.checks if c.triggered]

    @property
    def neutralized(self) -> bool:
        """Was the attack stopped, whether by blocking or by redaction?

        `blocked` is the strict reading: the response never reached the user.
        `neutralized` also counts a redaction that actually removed the payload
        — the attacker asked for a customer's card number and got
        [REDACTED:CREDIT_CARD], so the exfiltration failed even though the turn
        completed. Which of the two you feed into ASR is a threat-model choice;
        state it explicitly in the report rather than leaving it implicit.
        """
        if self.blocked:
            return True
        return (
            self.decision == Decision.REDACT
            and self.sanitized_output != self.original_output
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "decision": self.decision.value,
            "blocked": self.blocked,
            "categories": self.categories,
            "max_severity": self.max_severity.value,
            "triggered": self.triggered_checks,
            "latency_ms": round(self.latency_ms, 2),
            "session_id": self.session_id,
            "round_id": self.round_id,
            "checks": [c.to_dict() for c in self.checks],
            "meta": self.meta,
        }


class Timer:
    """Context manager for per-check latency.

    `ms` is readable both inside the block (live elapsed) and after it (frozen),
    so a check can return early from inside the `with` and still report latency.
    """

    def __enter__(self) -> "Timer":
        self._t0 = time.perf_counter()
        self._frozen: Optional[float] = None
        return self

    def __exit__(self, *exc) -> None:
        self._frozen = (time.perf_counter() - self._t0) * 1000.0

    @property
    def ms(self) -> float:
        if self._frozen is not None:
            return self._frozen
        return (time.perf_counter() - self._t0) * 1000.0
