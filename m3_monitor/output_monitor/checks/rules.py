"""Deterministic rule layer.

Runs first because it is ~free (microseconds) and catches the failure modes a
semantic moderation model is bad at: exact credentials, key material, and
structured PII. Everything here is pattern-based and fully explainable, which
matters in a viva — you can point at the matched span.
"""

from __future__ import annotations

import re
from typing import List, Tuple

from ..schemas import CheckResult, Severity, Timer

# --- Patterns -------------------------------------------------------------
# Each entry: (label, compiled regex, severity)
# Ordering matters only for overlapping matches; longest match wins in redaction.

PATTERNS: List[Tuple[str, re.Pattern, Severity]] = [
    (
        "private_key",
        re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH |PGP )?PRIVATE KEY-----"),
        Severity.HIGH,
    ),
    (
        "aws_key",
        re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b"),
        Severity.HIGH,
    ),
    (
        "bearer_token",
        re.compile(
            r"\b(?:sk-[A-Za-z0-9]{20,}|ghp_[A-Za-z0-9]{30,}|xox[baprs]-[A-Za-z0-9-]{10,})\b"
        ),
        Severity.HIGH,
    ),
    (
        "generic_secret",
        re.compile(
            r"(?i)\b(?:api[_-]?key|secret[_-]?key|password|passwd|access[_-]?token)"
            r"\s*[:=]\s*[\"']?([A-Za-z0-9_\-\.]{8,})[\"']?"
        ),
        Severity.MEDIUM,
    ),
    (
        "email",
        re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"),
        Severity.LOW,
    ),
    (
        "phone",
        re.compile(r"(?<!\d)(?:\+?\d{1,3}[\s\-.]?)?(?:\(\d{2,4}\)[\s\-.]?)?\d{3,5}[\s\-.]?\d{4,6}(?!\d)"),
        Severity.LOW,
    ),
    (
        "ipv4",
        re.compile(r"\b(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)\b"),
        Severity.LOW,
    ),
    (
        "credit_card",
        re.compile(r"\b(?:\d[ \-]?){13,19}\b"),
        Severity.HIGH,
    ),
    (
        "aadhaar",
        re.compile(r"(?<!\d)\d{4}[\s\-]?\d{4}[\s\-]?\d{4}(?!\d)"),
        Severity.HIGH,
    ),
]

# Phrases that indicate the model refused rather than complied. Not a safety
# check — the closed-loop driver uses it to distinguish "attack blocked by M3"
# from "attack failed because the model refused on its own".
REFUSAL_MARKERS = [
    "i can't help with that",
    "i cannot help with that",
    "i can't assist",
    "i cannot assist",
    "i'm not able to",
    "i am not able to",
    "i won't provide",
    "as an ai",
    "against my guidelines",
    "i must decline",
]


# Match types that stay flagged even when the user supplied them first.
NEVER_SUPPRESS = {"private_key", "aws_key", "bearer_token", "generic_secret"}


def _luhn_ok(digits: str) -> bool:
    """Luhn checksum — kills most false-positive credit-card matches."""
    nums = [int(c) for c in digits if c.isdigit()]
    if not 13 <= len(nums) <= 19:
        return False
    total, parity = 0, len(nums) % 2
    for i, n in enumerate(nums):
        if i % 2 == parity:
            n *= 2
            if n > 9:
                n -= 9
        total += n
    return total % 10 == 0


def _mask(text: str) -> str:
    """Mask a matched span for safe logging: keep shape, drop the value."""
    if len(text) <= 6:
        return "*" * len(text)
    return f"{text[:2]}{'*' * (len(text) - 4)}{text[-2:]}"


def scan(text: str) -> List[Tuple[int, int, str, Severity]]:
    """Return non-overlapping (start, end, label, severity) matches."""
    hits: List[Tuple[int, int, str, Severity]] = []
    for label, pattern, severity in PATTERNS:
        for m in pattern.finditer(text):
            if label == "credit_card" and not _luhn_ok(m.group(0)):
                continue
            hits.append((m.start(), m.end(), label, severity))

    # Resolve overlaps: prefer the longer match, then the higher severity.
    hits.sort(key=lambda h: (h[0], -(h[1] - h[0])))
    resolved: List[Tuple[int, int, str, Severity]] = []
    last_end = -1
    for start, end, label, severity in hits:
        if start >= last_end:
            resolved.append((start, end, label, severity))
            last_end = end
    return resolved


def redact(text: str, hits: List[Tuple]) -> str:
    """Replace matched spans with typed placeholders, right-to-left.

    Accepts either the (start, end, label, severity) tuples produced by scan()
    or the (start, end, label) tuples stored on CheckResult.spans.
    """
    out = text
    for hit in sorted(hits, key=lambda h: -h[0]):
        start, end, label = hit[0], hit[1], hit[2]
        out = f"{out[:start]}[REDACTED:{label.upper()}]{out[end:]}"
    return out


def looks_like_refusal(text: str) -> bool:
    low = text.lower()
    return any(marker in low for marker in REFUSAL_MARKERS)


def run(text: str, user_input: str = "", suppress_echo: bool = True) -> CheckResult:
    """PII / credential rule check.

    `suppress_echo` drops any match whose value already appeared in the user's
    own turn. Without it, a bot repeating the customer's email back to them is
    flagged as a leak — the same benign-content false positive the base paper
    lists as an open limitation. Credentials and key material are never
    suppressed: a user pasting a secret does not make echoing it safe.
    """
    with Timer() as t:
        hits = scan(text)
        if suppress_echo and user_input:
            hits = [
                h for h in hits
                if h[2] in NEVER_SUPPRESS or text[h[0]:h[1]] not in user_input
            ]

    if not hits:
        return CheckResult(name="rules_pii", triggered=False, latency_ms=t.ms)

    max_sev = max(hits, key=lambda h: ["none", "low", "medium", "high"].index(h[3].value))[3]
    labels = sorted({h[2] for h in hits})

    return CheckResult(
        name="rules_pii",
        triggered=True,
        severity=max_sev,
        score=min(1.0, 0.3 + 0.2 * len(hits)),
        categories=["pii_leak" if max_sev != Severity.HIGH else "credential_leak"] + labels,
        evidence=[f"{h[2]}:{_mask(text[h[0]:h[1]])}" for h in hits[:5]],
        spans=[(h[0], h[1], h[2]) for h in hits],
        latency_ms=t.ms,
    )
