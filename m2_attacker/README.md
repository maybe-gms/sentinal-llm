# M2 — Automated Prompt-Injection Attacker

The red-teaming module for the Adversarially Co-Evolving LLM Security Framework.
It generates prompt-injection attacks against a local LLM-integrated app, scores
them by Adversarial Success Rate (ASR), evolves stronger attacks over
generations, and hands every successful attack back to M1 (the detector) as new
labelled training data — closing the loop.

## What this module does (and doesn't)

- **Does:** injection attacks with *benign injected tasks* (make a summariser do
  sentiment analysis, print a reserved-domain URL, refuse its job, or echo its
  own instruction). Success = the app performed the wrong task. Deterministic,
  no LLM judge needed.
- **Doesn't:** generate harmful/jailbreak content. That is out of scope for M2
  because M1 detects *injected instructions*, not unsafe output. Harmful-output
  testing belongs in M3.

## Why not GCG like the DataSentinel paper?

GCG needs token gradients. Ollama is a black-box HTTP API with no gradient
access, so GCG cannot run through it. M2 instead does **black-box evolutionary
search**: seed with the documented heuristic separators (Naive, Escape,
Context-Ignoring, Fake-Completion, Combined), mutate them (rule-based + LLM
paraphrase), score against the live app, keep the winners, repeat. This matches
the automated red-teaming literature cited in the project. It is a deliberate,
defensible deviation from the anchor paper — be ready to say so in the viva.

## Setup

```bash
# 1. Install Ollama from https://ollama.com, then pull the models:
ollama pull llama3:8b        # backend (the app's LLM)
ollama pull mistral:7b       # attacker mutator + KAD detector LLM

# 2. Make sure the server is running (usually automatic):
ollama serve                 # leave running in its own terminal

# 3. Python deps:
pip install -r requirements.txt
```

## Run it

```bash
# Offline smoke test — no Ollama needed, proves the plumbing works.
# (Numbers are meaningless; this only checks control flow.)
python run_loop.py --mock --detector kad --rounds 2

# Act 1: baseline vulnerability, no defense.
python run_loop.py --detector none --rounds 1

# Act 2/3: attack behind the known-answer detector baseline.
python run_loop.py --detector kad --rounds 3
```

Outputs land in `runs/`:
- `asr_history.csv` — per-round ASR. This is your headline chart.
- `roundN_attacks.jsonl` — every attempt, with output and block status.
- `roundN_contaminated.jsonl` — successful attacks as `{text, label}` pairs.
  **This file is the handoff to M1.**

## Wiring in M1 (your teammate's detector)

The loop doesn't change. Once M1 exposes a `predict(text) -> bool`:

```python
from detectors import M1Detector
from target_app import TargetApp

detector = M1Detector(m1.predict)      # your fine-tuned detector
app = TargetApp(client, detector=detector)
# ... hand `app` to EvolutionaryAttacker exactly as run_loop.py does
```

Between rounds, M1 retrains on the accumulated `roundN_contaminated.jsonl`
files, you rebuild `detector`/`app` with the updated M1, and continue. With the
**static KAD baseline**, the detector never changes — so that ASR curve is
exactly the static-baseline comparison the project needs. With **M1 retraining
each round**, ASR should fall faster. Plotting both curves on one chart is the
single clearest proof of the closed loop's value.

## Files

| File | Role |
|---|---|
| `config.py` | all knobs (models, population size, generations, secret key) |
| `tasks.py` | target + injected task definitions and sample data |
| `llm_client.py` | Ollama client with disk cache |
| `mock_client.py` | fake client for offline testing only |
| `target_app.py` | the LLM app under test |
| `detectors.py` | NullDetector, KnownAnswerDetector (KAD baseline), M1Detector hook |
| `scorer.py` | deterministic ASR scoring |
| `attacker.py` | the evolutionary attacker (the heart of M2) |
| `run_loop.py` | closed-loop driver, writes ASR history |

## Scaling up for the real report

- Set `REPEAT_TRIALS = 3` and average — LLM outputs are stochastic even at low
  temperature; report a rate, not a one-off.
- Replace the inline `samples` in `tasks.py` with the seven HuggingFace datasets
  the paper uses (SST2, SMS Spam, Gigaword, HSOL, RTE, MRPC, Jfleg) for a
  defensible held-out split.
- Keep a **held-out injected task** never seen during any round, and report ASR
  on it separately, to show generalisation rather than memorisation.
