"""M3 — Output Safety Monitor.

Public surface is deliberately tiny:

    monitor = OutputMonitor(MonitorConfig(...))
    verdict = monitor.inspect(model_output, user_input="...")
    if verdict.blocked: ...

Ordering is cheap-to-expensive: rules (µs) -> leakage (ms) -> Llama Guard
(hundreds of ms). With `short_circuit=True` the monitor stops as soon as
something BLOCK-worthy fires, which is what you want in production. Leave it
False during evaluation so every check produces a label for the per-check
precision/recall breakdown.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from typing import Any, Dict, List, Optional

from .config import MonitorConfig
from .checks import leakage, moderation, rules
from .schemas import CheckResult, Decision, Severity, SEVERITY_ORDER, Verdict


class OutputMonitor:
    def __init__(self, config: Optional[MonitorConfig] = None) -> None:
        self.config = config or MonitorConfig()
        self._block_level = SEVERITY_ORDER[Severity(self.config.block_severity)]
        self._guard = (
            moderation.GuardClient(
                base_url=self.config.ollama_url,
                model=self.config.guard_model,
                timeout_s=self.config.guard_timeout_s,
            )
            if self.config.enable_guard
            else None
        )
        if self.config.log_path:
            os.makedirs(os.path.dirname(self.config.log_path) or ".", exist_ok=True)

    # --- main entry point -------------------------------------------------
    def inspect(
        self,
        model_output: str,
        user_input: str = "",
        session_id: Optional[str] = None,
        round_id: Optional[int] = None,
        meta: Optional[Dict[str, Any]] = None,
    ) -> Verdict:
        t0 = time.perf_counter()
        cfg = self.config
        checks: List[CheckResult] = []

        # 1. Rules — PII and credentials.
        rule_result = rules.run(model_output, user_input=user_input,
                                suppress_echo=cfg.suppress_user_echo)
        checks.append(rule_result)

        # 2. Leakage — canary and system-prompt containment.
        if not (cfg.short_circuit and self._is_blocking(rule_result)):
            checks.append(
                leakage.run(
                    model_output,
                    system_prompt=cfg.system_prompt,
                    canary_token=cfg.canary_token,
                    n=cfg.overlap_ngram_n,
                    threshold=cfg.overlap_threshold,
                )
            )

        # 3. Llama Guard — semantic harm classification.
        if self._guard is not None and not (
            cfg.short_circuit and any(self._is_blocking(c) for c in checks)
        ):
            checks.append(
                moderation.run(
                    self._guard,
                    user_input,
                    model_output,
                    ignore_categories=cfg.guard_ignore_categories,
                )
            )

        verdict = self._aggregate(model_output, checks)
        verdict.session_id = session_id or uuid.uuid4().hex[:12]
        verdict.round_id = round_id
        verdict.latency_ms = (time.perf_counter() - t0) * 1000.0
        verdict.meta = {
            **(meta or {}),
            "refusal": rules.looks_like_refusal(model_output),
            "guard_skipped": any(c.name == "guard_moderation" and c.skipped for c in checks),
        }

        self._log(verdict, user_input)
        return verdict

    # --- aggregation ------------------------------------------------------
    def _is_blocking(self, check: CheckResult) -> bool:
        return check.triggered and SEVERITY_ORDER[check.severity] >= self._block_level

    def _aggregate(self, output: str, checks: List[CheckResult]) -> Verdict:
        triggered = [c for c in checks if c.triggered]

        if not triggered:
            return Verdict(
                decision=Decision.ALLOW,
                blocked=False,
                original_output=output,
                sanitized_output=output,
                checks=checks,
            )

        max_sev = max(triggered, key=lambda c: SEVERITY_ORDER[c.severity]).severity
        categories = sorted({cat for c in triggered for cat in c.categories})

        # PII-only path: redact rather than withhold, if configured and if none
        # of the matched types are in the hard-block list.
        pii_only = all(c.name == "rules_pii" for c in triggered)
        hard_pii = any(
            label in self.config.pii_block_types
            for c in triggered
            for label in c.categories
        )
        if pii_only and self.config.redact_pii and not hard_pii:
            sanitized = rules.redact(output, triggered[0].spans)
            return Verdict(
                decision=Decision.REDACT,
                blocked=False,
                original_output=output,
                sanitized_output=sanitized,
                categories=categories,
                max_severity=max_sev,
                checks=checks,
            )

        if SEVERITY_ORDER[max_sev] >= self._block_level:
            return Verdict(
                decision=Decision.BLOCK,
                blocked=True,
                original_output=output,
                sanitized_output=(
                    "[response withheld by output safety monitor: "
                    + ", ".join(categories) + "]"
                ),
                categories=categories,
                max_severity=max_sev,
                checks=checks,
            )

        # Triggered but below the block bar — surface it, let it through.
        return Verdict(
            decision=Decision.ALLOW,
            blocked=False,
            original_output=output,
            sanitized_output=output,
            categories=categories,
            max_severity=max_sev,
            checks=checks,
        )

    # --- logging ----------------------------------------------------------
    def _log(self, verdict: Verdict, user_input: str) -> None:
        if not self.config.log_path:
            return
        record = verdict.to_dict()
        record["ts"] = time.time()
        if self.config.log_raw_output:
            record["user_input"] = user_input
            record["model_output"] = verdict.original_output
        try:
            with open(self.config.log_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        except OSError:
            pass  # logging must never break the pipeline
