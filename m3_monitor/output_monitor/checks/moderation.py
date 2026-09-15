"""Llama Guard moderation via Ollama.

Llama Guard is used in *response classification* mode: we pass the user turn and
the assistant turn, and it judges the assistant turn. Ollama applies the model's
own chat template, so we must not hand-build the guard prompt — passing plain
`messages` is what makes the classification correct.

Expected raw output is either `safe` or `unsafe\\nS2,S9`.

Failure policy: if Ollama is unreachable or the model is not pulled, the check is
marked `skipped` rather than raising. A dead moderation model must not take the
whole pipeline down mid-demo, and a skipped check is visible in the logs so the
evaluation script can exclude those rows instead of silently scoring them as safe.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import List, Optional, Tuple

from ..config import GUARD_CATEGORIES
from ..schemas import CheckResult, Severity, Timer


class GuardClient:
    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "llama-guard3:1b",
        timeout_s: float = 30.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_s = timeout_s

    # --- transport -------------------------------------------------------
    def _post(self, path: str, payload: dict) -> dict:
        req = urllib.request.Request(
            f"{self.base_url}{path}",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def available(self) -> bool:
        try:
            urllib.request.urlopen(f"{self.base_url}/api/tags", timeout=3.0).read()
            return True
        except Exception:
            return False

    # --- classification --------------------------------------------------
    def classify(self, user_input: str, assistant_output: str) -> Tuple[bool, List[str], str]:
        """Return (is_safe, category_names, raw_response)."""
        payload = {
            "model": self.model,
            "messages": [
                {"role": "user", "content": user_input or ""},
                {"role": "assistant", "content": assistant_output},
            ],
            "stream": False,
            "options": {"temperature": 0.0, "num_predict": 32},
        }
        data = self._post("/api/chat", payload)
        raw = (data.get("message", {}).get("content") or "").strip()
        return (*self._parse(raw), raw)

    @staticmethod
    def _parse(raw: str) -> Tuple[bool, List[str]]:
        lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
        if not lines:
            return True, []
        if lines[0].lower().startswith("safe"):
            return True, []
        codes: List[str] = []
        for line in lines[1:]:
            for token in line.replace(" ", "").split(","):
                token = token.upper()
                if token in GUARD_CATEGORIES:
                    codes.append(GUARD_CATEGORIES[token])
        return False, codes or ["unspecified_hazard"]


def run(
    client: GuardClient,
    user_input: str,
    assistant_output: str,
    ignore_categories: Optional[List[str]] = None,
) -> CheckResult:
    ignore = set(ignore_categories or [])
    with Timer() as t:
        try:
            is_safe, categories, raw = client.classify(user_input, assistant_output)
        except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
            return CheckResult(
                name="guard_moderation",
                triggered=False,
                skipped=True,
                error=f"{type(exc).__name__}: {exc}",
                latency_ms=t.ms,
            )

    reportable = [c for c in categories if c not in ignore]
    if is_safe or not reportable:
        return CheckResult(
            name="guard_moderation",
            triggered=False,
            score=0.0,
            latency_ms=t.ms,
            evidence=[raw[:80]] if raw else [],
        )

    return CheckResult(
        name="guard_moderation",
        triggered=True,
        severity=Severity.HIGH,
        score=1.0,
        categories=reportable,
        evidence=[f"llama-guard: unsafe ({', '.join(reportable)})"],
        latency_ms=t.ms,
    )
