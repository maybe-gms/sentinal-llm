"""Shared integration layer for the co-evolving LLM security framework.

This package is what turns four independently-built modules (M1, M2, M3, M4)
into one system. Nothing here re-implements a module; it only *wires* them
through the interface contracts each module already exposes:

    M1  exposes  predict(text) -> bool          (input-side injection detector)
    M3  exposes  OutputMonitor.inspect(...)     (output-side safety monitor)
    the backend is any object with .generate(prompt, ...) -> str  (Ollama / mock)

The single public object is `Pipeline` (see pipeline.py). Build it once, call
`.run(instruction, data)`, and read a `PipelineResult` that records, stage by
stage, exactly what each defense layer decided. That object is what the web UI
renders and what the demo prints.
"""

from .pipeline import (
    Pipeline,
    PipelineResult,
    StageResult,
    Stage,
    Outcome,
    build_pipeline,
)

__all__ = [
    "Pipeline",
    "PipelineResult",
    "StageResult",
    "Stage",
    "Outcome",
    "build_pipeline",
]
__version__ = "1.0.0"
