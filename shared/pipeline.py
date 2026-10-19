"""The integration pipeline: M1 (input) -> backend LLM -> M3 (output).

This is the single place the three runtime modules are composed into one
defended application. It is deliberately small and explicit: every request
produces a `PipelineResult` that records, for each stage, what happened and
why, so the web UI and the demo can show the *whole* story of one request
rather than just a final yes/no.

Design rules (match the project's defense-in-depth claim):

  * M1 sits on the INPUT. If it flags the data as contaminated, the request is
    refused and the backend LLM is never called. This is the cheap first line.

  * The backend LLM runs only on input M1 let through.

  * M3 sits on the OUTPUT. It inspects what the model actually produced and can
    BLOCK (withhold), REDACT (sanitise), or ALLOW. It is independent of M1: an
    attack that slips past M1 on the input can still be caught by M3 on the
    output. That independence is the point — an attacker must beat both.

  * Any layer can be turned off (detector=None, monitor=None) without changing
    the behaviour of the others. With both off, this is just the raw app, which
    is what Act 1 of the demo (baseline vulnerability) needs.

Nothing here imports M1/M2/M3 internals directly. The backend, detector and
monitor are injected (duck-typed), exactly as m2_attacker/target_app.py already
does it, so this layer has no hard dependency on any module's package layout
and works the same in mock mode and against real Ollama.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Protocol, runtime_checkable


# --------------------------------------------------------------------------
# Duck-typed interfaces the pipeline expects. These are documentation; the
# pipeline never isinstance-checks them, it just calls the method.
# --------------------------------------------------------------------------
@runtime_checkable
class Backend(Protocol):
    def generate(self, prompt: str, model: Optional[str] = None,
                 temperature: Optional[float] = None,
                 max_tokens: Optional[int] = None) -> str: ...


@runtime_checkable
class Detector(Protocol):
    # M1 / KAD wrapper from m2_attacker/detectors.py
    def is_contaminated(self, target_data: str) -> bool: ...


@runtime_checkable
class Monitor(Protocol):
    # M3 OutputMonitor
    def inspect(self, model_output: str, user_input: str = "", **kw) -> Any: ...


# --------------------------------------------------------------------------
# Result types
# --------------------------------------------------------------------------
class Stage(str, Enum):
    INPUT_DETECTOR = "m1_input_detector"   # M1
    BACKEND = "backend_llm"                # the model under test
    OUTPUT_MONITOR = "m3_output_monitor"   # M3


class Outcome(str, Enum):
    """What a single stage decided about the request passing through it."""
    PASS = "pass"            # stage ran, nothing wrong, request continues
    FLAGGED = "flagged"      # stage flagged the request and stopped it (block)
    REDACTED = "redacted"    # stage altered the output but let it continue
    SKIPPED = "skipped"      # stage disabled or not reached
    GENERATED = "generated"  # backend produced output (its "pass")
    ERROR = "error"          # stage raised; recorded, not swallowed silently


@dataclass
class StageResult:
    stage: Stage
    outcome: Outcome
    detail: str = ""                         # one-line human summary
    categories: List[str] = field(default_factory=list)
    latency_ms: float = 0.0
    data: Dict[str, Any] = field(default_factory=dict)   # stage-specific extras

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["stage"] = self.stage.value
        d["outcome"] = self.outcome.value
        return d


@dataclass
class PipelineResult:
    """The full story of one request through the defended app."""
    instruction: str
    data: str
    prompt: str

    # The bottom line the end user would experience:
    delivered: bool                 # did ANY text reach the user?
    final_output: str               # what the user actually sees
    blocked_by: Optional[Stage] = None   # which stage stopped it, if blocked

    # The raw model output BEFORE M3 (what the attacker achieved at the model):
    raw_output: Optional[str] = None

    stages: List[StageResult] = field(default_factory=list)
    total_latency_ms: float = 0.0

    # --- convenience views for the UI / demo ---
    @property
    def defended(self) -> bool:
        """True if some defense layer stopped or altered the request."""
        return self.blocked_by is not None or any(
            s.outcome == Outcome.REDACTED for s in self.stages
        )

    @property
    def headline(self) -> str:
        if self.blocked_by == Stage.INPUT_DETECTOR:
            return "BLOCKED BY M1 (input) — the LLM was never called"
        if self.blocked_by == Stage.OUTPUT_MONITOR:
            return "BLOCKED BY M3 (output) — the model leaked, M3 withheld it"
        if any(s.outcome == Outcome.REDACTED for s in self.stages):
            return "REDACTED BY M3 (output) — sensitive content removed"
        return "DELIVERED — no layer objected"

    def stage(self, stage: Stage) -> Optional[StageResult]:
        for s in self.stages:
            if s.stage == stage:
                return s
        return None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "instruction": self.instruction,
            "data": self.data,
            "prompt": self.prompt,
            "delivered": self.delivered,
            "defended": self.defended,
            "final_output": self.final_output,
            "raw_output": self.raw_output,
            "blocked_by": self.blocked_by.value if self.blocked_by else None,
            "headline": self.headline,
            "total_latency_ms": round(self.total_latency_ms, 1),
            "stages": [s.to_dict() for s in self.stages],
        }


# --------------------------------------------------------------------------
# The pipeline
# --------------------------------------------------------------------------
class Pipeline:
    """Compose M1 + backend + M3 into one defended application.

    Parameters
    ----------
    backend   : object with .generate(prompt, ...) -> str   (required)
    detector  : M1 / KAD detector with .is_contaminated(text) -> bool, or None
    monitor   : M3 OutputMonitor with .inspect(output, user_input=...), or None
    model     : backend model name passed through to .generate (optional)

    `detector` and `monitor` are each optional and independent. The three
    on/off combinations the demo uses:
        none     : detector=None, monitor=None     (Act 1 — baseline)
        m1 only  : detector=M1,   monitor=None
        full     : detector=M1,   monitor=M3        (the integrated system)
    """

    REFUSAL_TEXT = "[BLOCKED] This input was flagged as containing an injected instruction."

    def __init__(
        self,
        backend: Backend,
        detector: Optional[Detector] = None,
        monitor: Optional[Monitor] = None,
        model: Optional[str] = None,
    ) -> None:
        self.backend = backend
        self.detector = detector
        self.monitor = monitor
        self.model = model

    # -- introspection for the UI banner ----------------------------------
    @property
    def layers(self) -> Dict[str, bool]:
        return {
            "m1": self.detector is not None,
            "m3": self.monitor is not None,
        }

    def build_prompt(self, instruction: str, data: str) -> str:
        """Exactly the layout the target app uses, so measurements line up."""
        return f"{instruction}\n\nText: {data}"

    def run(self, instruction: str, data: str) -> PipelineResult:
        t_start = time.perf_counter()
        prompt = self.build_prompt(instruction, data)
        stages: List[StageResult] = []

        # --- Stage 1: M1 input detector -----------------------------------
        if self.detector is not None:
            t0 = time.perf_counter()
            try:
                flagged = bool(self.detector.is_contaminated(data))
                ms = (time.perf_counter() - t0) * 1000.0
                if flagged:
                    stages.append(StageResult(
                        Stage.INPUT_DETECTOR, Outcome.FLAGGED,
                        detail="M1 flagged the input as contaminated.",
                        categories=["injected_instruction"], latency_ms=ms,
                    ))
                    return self._finish(
                        instruction, data, prompt, stages, t_start,
                        delivered=False, final_output=self.REFUSAL_TEXT,
                        raw_output=None, blocked_by=Stage.INPUT_DETECTOR,
                    )
                stages.append(StageResult(
                    Stage.INPUT_DETECTOR, Outcome.PASS,
                    detail="M1 judged the input clean.", latency_ms=ms,
                ))
            except Exception as e:  # a broken detector must be visible, not silent
                ms = (time.perf_counter() - t0) * 1000.0
                stages.append(StageResult(
                    Stage.INPUT_DETECTOR, Outcome.ERROR,
                    detail=f"M1 detector error: {e}", latency_ms=ms,
                ))
                # Fail closed on the detector? No — fail OPEN so the demo keeps
                # running and the error is shown. The monitor still defends.
        else:
            stages.append(StageResult(
                Stage.INPUT_DETECTOR, Outcome.SKIPPED,
                detail="M1 disabled for this run.",
            ))

        # --- Stage 2: backend LLM -----------------------------------------
        t0 = time.perf_counter()
        try:
            raw_output = self.backend.generate(prompt, model=self.model)
        except Exception as e:
            ms = (time.perf_counter() - t0) * 1000.0
            stages.append(StageResult(
                Stage.BACKEND, Outcome.ERROR,
                detail=f"Backend error: {e}", latency_ms=ms,
            ))
            return self._finish(
                instruction, data, prompt, stages, t_start,
                delivered=False, final_output=f"[backend error] {e}",
                raw_output=None, blocked_by=Stage.BACKEND,
            )
        ms = (time.perf_counter() - t0) * 1000.0
        stages.append(StageResult(
            Stage.BACKEND, Outcome.GENERATED,
            detail="Backend produced an output.", latency_ms=ms,
            data={"output_preview": _preview(raw_output)},
        ))

        # --- Stage 3: M3 output monitor -----------------------------------
        if self.monitor is not None:
            t0 = time.perf_counter()
            try:
                verdict = self.monitor.inspect(raw_output, user_input=data)
                ms = (time.perf_counter() - t0) * 1000.0
                decision = _verdict_decision(verdict)
                categories = list(getattr(verdict, "categories", []) or [])
                if decision == "block":
                    stages.append(StageResult(
                        Stage.OUTPUT_MONITOR, Outcome.FLAGGED,
                        detail="M3 blocked the output (unsafe / leaked content).",
                        categories=categories, latency_ms=ms,
                        data={"max_severity": _verdict_severity(verdict)},
                    ))
                    withheld = getattr(verdict, "sanitized_output", None) \
                        or "[response withheld by output safety monitor]"
                    return self._finish(
                        instruction, data, prompt, stages, t_start,
                        delivered=False, final_output=withheld,
                        raw_output=raw_output, blocked_by=Stage.OUTPUT_MONITOR,
                    )
                if decision == "redact":
                    sanitized = getattr(verdict, "sanitized_output", raw_output)
                    stages.append(StageResult(
                        Stage.OUTPUT_MONITOR, Outcome.REDACTED,
                        detail="M3 redacted sensitive content; delivered sanitised.",
                        categories=categories, latency_ms=ms,
                    ))
                    return self._finish(
                        instruction, data, prompt, stages, t_start,
                        delivered=True, final_output=sanitized,
                        raw_output=raw_output, blocked_by=None,
                    )
                # allow
                stages.append(StageResult(
                    Stage.OUTPUT_MONITOR, Outcome.PASS,
                    detail="M3 allowed the output.",
                    categories=categories, latency_ms=ms,
                ))
            except Exception as e:
                ms = (time.perf_counter() - t0) * 1000.0
                stages.append(StageResult(
                    Stage.OUTPUT_MONITOR, Outcome.ERROR,
                    detail=f"M3 monitor error: {e}", latency_ms=ms,
                ))
                # fail open: deliver the raw output but record the error
        else:
            stages.append(StageResult(
                Stage.OUTPUT_MONITOR, Outcome.SKIPPED,
                detail="M3 disabled for this run.",
            ))

        return self._finish(
            instruction, data, prompt, stages, t_start,
            delivered=True, final_output=raw_output,
            raw_output=raw_output, blocked_by=None,
        )

    # -- helpers ----------------------------------------------------------
    def _finish(self, instruction, data, prompt, stages, t_start, *,
                delivered, final_output, raw_output, blocked_by) -> PipelineResult:
        return PipelineResult(
            instruction=instruction, data=data, prompt=prompt,
            delivered=delivered, final_output=final_output,
            raw_output=raw_output, blocked_by=blocked_by, stages=stages,
            total_latency_ms=(time.perf_counter() - t_start) * 1000.0,
        )


# --------------------------------------------------------------------------
# small utilities
# --------------------------------------------------------------------------
def _preview(text: str, n: int = 160) -> str:
    text = (text or "").replace("\n", " ").strip()
    return text if len(text) <= n else text[: n - 1] + "…"


def _verdict_decision(verdict: Any) -> str:
    """Pull 'allow'|'redact'|'block' out of an M3 Verdict (enum or str)."""
    d = getattr(verdict, "decision", None)
    if d is None:
        # last resort: blocked flag
        return "block" if getattr(verdict, "blocked", False) else "allow"
    return getattr(d, "value", str(d))


def _verdict_severity(verdict: Any) -> str:
    s = getattr(verdict, "max_severity", None)
    return getattr(s, "value", str(s)) if s is not None else "unknown"


# --------------------------------------------------------------------------
# convenience builder
# --------------------------------------------------------------------------
def build_pipeline(
    layers: str = "full",
    *,
    mock: bool = False,
    backend=None,
    detector=None,
    monitor=None,
    model: Optional[str] = None,
    system_prompt: Optional[str] = None,
    canary_token: Optional[str] = None,
) -> "Pipeline":
    """Construct a Pipeline for a named layer configuration.

    `layers` is one of:
        "none"  -> no M1, no M3        (baseline vulnerable app)
        "m1"    -> M1 only
        "m3"    -> M3 only
        "full"  -> M1 + M3             (the integrated system)

    Passing `backend`/`detector`/`monitor` explicitly overrides the auto-build.
    With `mock=True` no Ollama is required (uses m2's MockClient and a mock M1).

    This function imports the modules lazily so that importing `shared` alone
    (e.g. for the dataclasses) never drags in requests / Ollama.
    """
    import os
    import sys

    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(here)
    for p in (here,                               # shared/ itself (demo_backend)
              os.path.join(root, "m2_attacker"),
              os.path.join(root, "m3_monitor"),
              root):
        if p not in sys.path:
            sys.path.insert(0, p)

    layers = (layers or "full").lower()
    want_m1 = layers in ("m1", "full")
    want_m3 = layers in ("m3", "full")

    # --- backend ---
    if backend is None:
        if mock:
            # DemoBackend reproduces system-prompt leaks on a steal, so M3 has
            # something real to catch offline. (MockClient stays for the
            # measurement harness; this is the demo/UI backend.)
            from demo_backend import DemoBackend
            backend = DemoBackend(system_prompt=system_prompt or "")
        else:
            from llm_client import OllamaClient
            backend = OllamaClient()

    # --- M1 detector ---
    if detector is None and want_m1:
        detector = _build_detector(mock, backend)

    # --- M3 monitor ---
    if monitor is None and want_m3:
        monitor = _build_monitor(mock, system_prompt, canary_token)

    return Pipeline(backend=backend, detector=detector, monitor=monitor, model=model)


def _build_detector(mock: bool, backend):
    from detectors import M1Detector
    if mock:
        # A deliberately imperfect mock M1: flags heavy context-ignoring
        # language, misses subtler steals — same spirit as mock_client, so the
        # "M1 missed it -> M3 caught it" story is reproducible offline.
        cues = ("ignore previous", "ignore all", "disregard", "new instruction")
        def mock_predict(text: str) -> bool:
            t = text.lower()
            return any(c in t for c in cues)
        return M1Detector(mock_predict)
    # Real M1: the teammate's fine-tuned detector served by Ollama.
    from m1_detector import predict
    return M1Detector(predict)


def _build_monitor(mock: bool, system_prompt: Optional[str], canary_token: Optional[str]):
    from output_monitor import OutputMonitor, MonitorConfig
    cfg = MonitorConfig(
        enable_guard=False,          # rules + leakage only; no third model needed
        system_prompt=system_prompt,
        canary_token=canary_token,
        log_path=None,               # UI/demo manage their own logging
    )
    return OutputMonitor(cfg)
