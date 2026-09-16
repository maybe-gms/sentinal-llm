"""Output guards that can sit AFTER the backend LLM in the target app.

Big picture - the target app now has two doors:

    input  ->  [ detector (M1 / KAD) ]  ->  backend LLM  ->  [ output guard (M3) ]  ->  user
               checks the DATA going in                      checks the ANSWER coming out

  NullOutputGuard  - no output checking (the old behaviour).
  M3OutputGuard    - wraps Jerin's OutputMonitor from ../m3_monitor.

This file is only an adapter. It does not change anything inside M3.

How M3 is pointed at M2's app:
  M3's leakage check needs to know which text is "secret". In M2's app the
  secret is the target task's instruction (that's what the prompt-steal
  injection tries to extract), so we give M3 that instruction as its
  `system_prompt`. There are several target tasks with different
  instructions, so we keep one monitor per instruction.
"""

import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List

# M3 lives in a sibling folder, so add it to the import path.
_HERE = os.path.dirname(os.path.abspath(__file__))
_M3_DIR = os.path.abspath(os.path.join(_HERE, "..", "m3_monitor"))
if _M3_DIR not in sys.path:
    sys.path.insert(0, _M3_DIR)


@dataclass
class OutputCheck:
    blocked: bool                 # True if the guard stopped this answer
    output: str                   # what the user would actually receive
    categories: List[str] = field(default_factory=list)


class NullOutputGuard:
    name = "none"

    def check(self, instruction: str, data: str, output: str) -> OutputCheck:
        return OutputCheck(blocked=False, output=output)


class M3OutputGuard:
    name = "m3"

    def __init__(self, use_llama_guard: bool = False):
        # Imported here so M2 still runs even if m3_monitor is missing.
        from output_monitor import MonitorConfig, OutputMonitor

        self._MonitorConfig = MonitorConfig
        self._OutputMonitor = OutputMonitor
        self.use_llama_guard = use_llama_guard
        self._monitors: Dict[str, object] = {}

    def _monitor_for(self, instruction: str):
        if instruction not in self._monitors:
            cfg = self._MonitorConfig(
                enable_guard=self.use_llama_guard,
                system_prompt=instruction,   # the text M3 must not let leak
                log_path=None,               # M2 writes its own attack log
            )
            self._monitors[instruction] = self._OutputMonitor(cfg)
        return self._monitors[instruction]

    def check(self, instruction: str, data: str, output: str) -> OutputCheck:
        verdict = self._monitor_for(instruction).inspect(output, user_input=data)
        # `neutralized` = blocked, OR redacted in a way that removed the payload.
        return OutputCheck(
            blocked=verdict.neutralized,
            output=verdict.sanitized_output,
            categories=list(verdict.categories),
        )


def build_output_guard(kind: str, use_llama_guard: bool = False):
    kind = (kind or "none").lower()
    if kind in ("none", "off"):
        return NullOutputGuard()
    if kind == "m3":
        return M3OutputGuard(use_llama_guard=use_llama_guard)
    raise ValueError(f"Unknown output guard '{kind}'. Use 'none' or 'm3'.")
