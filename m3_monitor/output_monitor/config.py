"""Configuration for M3.

Every threshold that affects the precision/recall trade-off lives here, so the
evaluation script can sweep them without touching detector code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


# Llama Guard 3 hazard codes -> human-readable names.
# These are also the labels reported in the ASR breakdown, so keep them stable.
GUARD_CATEGORIES: Dict[str, str] = {
    "S1": "violent_crimes",
    "S2": "non_violent_crimes",
    "S3": "sex_crimes",
    "S4": "child_exploitation",
    "S5": "defamation",
    "S6": "specialized_advice",
    "S7": "privacy",
    "S8": "intellectual_property",
    "S9": "indiscriminate_weapons",
    "S10": "hate",
    "S11": "self_harm",
    "S12": "sexual_content",
    "S13": "elections",
    "S14": "code_interpreter_abuse",
}

# Categories produced by M3's own rule layer (not by Llama Guard).
RULE_CATEGORIES = [
    "system_prompt_leak",
    "canary_leak",
    "pii_leak",
    "credential_leak",
]


@dataclass
class MonitorConfig:
    # --- Llama Guard (moderation model) ---
    enable_guard: bool = True
    ollama_url: str = "http://localhost:11434"
    guard_model: str = "llama-guard3:1b"
    guard_timeout_s: float = 30.0
    # Categories that should NOT block in this project's threat model.
    # Empty by default; add e.g. "specialized_advice" if the target app is a
    # medical/legal QA bot and that category is expected behaviour.
    guard_ignore_categories: List[str] = field(default_factory=list)

    # --- System-prompt leakage ---
    system_prompt: Optional[str] = None      # protected text M3 watches for
    canary_token: Optional[str] = None       # unique marker embedded in the system prompt
    overlap_ngram_n: int = 4                 # n-gram size for fuzzy leak detection
    overlap_threshold: float = 0.12          # fraction of system-prompt n-grams echoed

    # --- PII / credentials ---
    redact_pii: bool = True                  # REDACT instead of BLOCK for PII-only hits
    suppress_user_echo: bool = True          # don't flag PII the user supplied themselves
    pii_block_types: List[str] = field(
        default_factory=lambda: ["private_key", "aws_key", "bearer_token"]
    )

    # --- Aggregation ---
    # Minimum severity that forces a BLOCK. "medium" is the sane default:
    # LOW findings are logged and surfaced but do not withhold the response.
    block_severity: str = "medium"
    short_circuit: bool = False              # stop at first BLOCK-worthy check (prod latency)

    # --- Logging ---
    log_path: Optional[str] = "logs/m3_monitor.jsonl"
    log_raw_output: bool = False             # keep False outside controlled experiments
