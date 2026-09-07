# Adversarially Co-Evolving LLM Security Testing Framework

Final Year B.Tech Project — Cyber Security.

A closed-loop system that red-teams an LLM-integrated application, feeds the
attacks that succeed back into its own defenses, and measures whether the
defense gets stronger over successive rounds than a static, train-once baseline.

**The single claim this project proves:** a defense that keeps retraining on the
attacks that beat it stays strong, while a defense trained once and frozen
decays as attacks adapt. We measure this with **Adversarial Success Rate (ASR)**
tracked across rounds.

---

## The four modules

| Module | Owner | Job | Status |
|---|---|---|---|
| **M1** | *(teammate)* | Detect whether an input is contaminated with an injected instruction | not started |
| **M2** | *(you)* | Automatically generate + evolve prompt-injection attacks; measure ASR | **working** |
| **M3** | *(teammate)* | Post-hoc filter checking model outputs for unsafe / leaked content | not started |
| **M4** | *(teammate)* | Test security of LLM agents that call tools / APIs | not started |

Each module lives in its own folder and is developed independently. They connect
in the final phase through **agreed interfaces** (see below), not shared internals.

```
final-year-project/
├─ README.md            ← this file
├─ .gitignore
├─ m1_detector/         (teammate)
├─ m2_attacker/         ← M2, the red-teaming attacker + experiment harness
├─ m3_monitor/          (teammate)
├─ m4_agent/            (teammate)
└─ shared/              (anything used by more than one module, added later)
```

---

## Interface contracts — READ BEFORE CODING YOUR MODULE

Modules must agree on how they talk to each other. Build toward these so nothing
has to be rewritten at integration.

### M1 → M2 (the detector the loop attacks)

M1 must expose a single function:

```python
def predict(text: str) -> bool:
    """Return True if `text` is contaminated with an injected instruction."""
```

That's the whole contract. M1 can be anything inside — a fine-tuned model, a
classifier, many files — as long as it exposes `predict(text) -> bool`.
M2 wraps it with no changes:

```python
from m1_detector import predict
from m2_attacker.detectors import M1Detector
detector = M1Detector(predict)
```

### M2 → M1 (the training signal the loop produces)

After every round, M2 writes the attacks that succeeded to
`runs/roundN_contaminated.jsonl`, one JSON object per line:

```json
{"text": "...contaminated input...", "label": "contaminated",
 "target_task": "...", "injected_task": "..."}
```

M1 retrains on the accumulated files from these rounds.

### The shared metric: ASR

Adversarial Success Rate = (successful attacks) / (total attempts). Produced by
M2, written to `runs/asr_history.csv`. This is the number every graph uses.

---

## Running M2

See `m2_attacker/README.md` for full setup. Short version:

```bash
ollama pull llama3.1:8b
cd m2_attacker
py -m pip install -r requirements.txt
py run_loop.py --detector none --rounds 1     # baseline, no defense
py run_loop.py --detector kad  --rounds 2     # behind the known-answer baseline
```

`kad` is a **baseline placeholder detector**, not M1. It exists so the loop runs
before M1 is built. Once M1 lands, swap it in via the interface above.
