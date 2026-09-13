# M2 — Automated Prompt-Injection Attacker

The red-teaming module for the Adversarially Co-Evolving LLM Security Framework.
It generates prompt-injection attacks against a local LLM-integrated app, scores
them by **Adversarial Success Rate (ASR)**, evolves stronger attacks over
generations, and hands every successful attack back to M1 (the detector) as new
labelled training data — closing the loop.

M2 is the engine that produces the attacks *and* the ASR metric the whole
project is measured by.

---

## What this module does (and doesn't)

- **Does:** injection attacks with *benign injected tasks* — make a summariser do
  sentiment analysis instead, print a reserved-domain URL, refuse its job, or
  echo back its own instruction. Success = the app performed the **wrong task**.
  Scoring is deterministic (string / label match), so no LLM judge is needed and
  numbers are reproducible.
- **Doesn't:** generate harmful or jailbreak content. That is deliberately out of
  scope for M2, because M1 detects *injected instructions*, not unsafe output.
  Feeding M1 harmful-content successes would be training signal it can't learn
  from. Harmful-output testing belongs in **M3**.

## Why evolutionary search, not GCG like the DataSentinel paper?

The anchor paper's inner-max attack uses **GCG**, which optimises a token
sequence via gradients. **Ollama is a black-box HTTP API with no gradient
access**, so GCG cannot run through it. M2 instead does **black-box evolutionary
search**: seed a population with the five documented heuristic separators
(Naive, Escape, Context-Ignoring, Fake-Completion, Combined), mutate them
(rule-based edits + LLM paraphrase), score each against the live app, keep the
winners, breed the next generation, repeat. This matches the automated
red-teaming literature the project cites. **It is a deliberate, defensible
deviation from the paper — be ready to explain it in the viva; it's the most
likely question.**

---

## Setup (CPU-only, 16 GB machine)

This module was developed on an AMD 7840U / Radeon 780M laptop with **no GPU
offload** (Ollama runs the model on CPU). It runs fine there; it just isn't fast.

```bash
# 1. Install Ollama from https://ollama.com (starts automatically on Windows).

# 2. Pull ONE model — used for all three roles (backend, mutator, detector).
ollama pull llama3.1:8b

# 3. Python deps (just `requests`):
py -m pip install -r requirements.txt   # Windows;  use python3 on Mac/Linux
```

**Why one model for everything:** on 16 GB RAM, a separate backend model and
detector model would force Ollama to unload+reload gigabytes on every
alternating call. Sharing `llama3.1:8b` across backend, mutator, and detector
(set in `config.py`) keeps one model resident and avoids the reloads. Trade-off:
the attacker and detector share weights, which slightly weakens the "attacker
isn't exploiting self-knowledge" claim — note it if you separate them for final
numbers (e.g. a smaller `llama3.2:3b` as the mutator).

---

## Run it

Everything runs from **inside the `m2_attacker` folder** (output paths are
relative to it).

```bash
# Offline smoke test — no Ollama needed, checks control flow only.
# (Numbers are meaningless here; the mock client is not a real LLM.)
py run_loop.py --mock --detector kad --rounds 2

# Act 1: baseline vulnerability, no defense.
py run_loop.py --detector none --rounds 1

# Act 2/3: attack behind the known-answer detector baseline.
py run_loop.py --detector kad --rounds 2

# Override the trial count (default comes from config.REPEAT_TRIALS):
py run_loop.py --detector kad --rounds 2 --trials 3
```

**Time budget (CPU):** ~240 model calls per round per trial. At full search size
with 3 trials, a 2-round KAD run is ~1,400+ calls — think hours, run it
overnight. The disk cache (`runs/llm_cache.json`) means a re-run reuses anything
already computed, so a crash mid-run is cheap to resume.

### What the run does

- Repeats each round `REPEAT_TRIALS` times and reports **ASR mean ± std** (LLM
  outputs are stochastic; a single trial is one roll of the dice).
- Prints **blocked-vs-succeeded** counts per round, so detector effectiveness is
  visible on screen, not buried in a log.
- Tags every output file by detector name so a `none` run and a `kad` run
  **don't overwrite each other** — you need both on disk for the chart.

### Outputs (in `runs/`)

- `asr_history_<detector>.csv` — per-round ASR mean/std + block counts. **This is
  your headline-chart data.** One file per detector (`none`, `known_answer`).
- `<detector>_roundN_attacks.jsonl` — every attempt, with model output and block
  status. Written from the first trial of each round.
- `<detector>_roundN_contaminated.jsonl` — successful attacks as
  `{text, label, target_task, injected_task}`. **This is the handoff file M1
  retrains on.** Schema is stable — don't change it without telling M1's owner.
- `llm_cache.json` — Ollama response cache. Regenerable; git-ignored.

---

## Current measured results (baseline, pre-M1)

Full search size, averaged over 3 trials, `llama3.1:8b` backend on CPU:

| Setup | ASR | Blocked |
|---|---|---|
| No defense (`none`) | **0.45** | 0 / 240 |
| Known-answer baseline (`kad`), round 0 | 0.425 | 2 / 240 |
| Known-answer baseline (`kad`), round 1 | 0.425 | 2 / 240 |

Success by injected task (no defense):

| Injected task | Success |
|---|---|
| Spam-flip | 69% |
| Sentiment-flip | 53% |
| Redirect (print URL) | 37% |
| Prompt-steal | 34% |
| Availability (force refusal) | 12% |

**How to read this:** the KAD baseline barely helps (2/240 blocked) and is flat
across rounds because it never retrains. **That is the expected, on-thesis
result** — untuned known-answer detection is weak, which is exactly why M1
fine-tunes it. The flat KAD curve *is* the static-baseline comparison the project
needs; M1's curve should slope down against it. The per-task spread is real
signal: instruction-echo attacks (prompt-steal) and forced-refusal
(availability) are much harder to land than task-swaps (spam/sentiment).

---

## Wiring in M1 (teammate's detector)

The loop code does not change. M1 must expose one function:

```python
def predict(text: str) -> bool:
    """Return True if `text` is contaminated with an injected instruction."""
```

Then:

```python
from detectors import M1Detector
from target_app import TargetApp

detector = M1Detector(predict)          # any predict(text)->bool works
app = TargetApp(client, detector=detector)
# hand `app` to EvolutionaryAttacker exactly as run_loop.py does
```

Between rounds M1 retrains on the accumulated `*_contaminated.jsonl` files, you
rebuild `detector`/`app` with the updated M1, and continue. With the **static
KAD baseline** the detector never changes (flat curve = static baseline). With
**M1 retraining each round**, ASR should fall — plotting both on one chart is the
single clearest proof of the closed loop's value.

> Note: `run_loop.py` currently records the retrain seam as a comment between
> rounds — the actual `retrain -> reload detector` call is the next integration
> step, wired once M1 (or a stub M1) exists.

---

## Files

| File | Role |
|---|---|
| `config.py` | all knobs — models, search sizes, trials, secret key |
| `tasks.py` | target + injected task definitions and sample data |
| `llm_client.py` | Ollama client with disk cache |
| `mock_client.py` | fake client for offline plumbing tests only |
| `target_app.py` | the LLM app under test |
| `detectors.py` | `NullDetector`, `KnownAnswerDetector` (baseline), `M1Detector` hook |
| `scorer.py` | deterministic ASR scoring (incl. same-task collapse guard) |
| `attacker.py` | the evolutionary attacker — **the heart of M2** |
| `run_loop.py` | closed-loop driver: trials, averaging, tagged outputs, ASR history |

If you understand one file for the viva, make it `attacker.py`.

---

## Known limitations (state these before an examiner finds them)

- **std ~ 0** in current runs: fixed seed + temperature 0.1 makes the backend
  near-deterministic, so trials barely differ. This is reproducibility, not a
  bug — raise temperature (~0.7) for one run if you want to *show* variance.
- **Evolution isn't innovating much yet:** the best template stays close to the
  seed separators, because nothing is blocking attacks, so there's no selection
  pressure to climb. M2's adaptive search becomes *measurably* valuable only once
  M1 blocks a meaningful fraction. Say this explicitly.
- **Same-task pairs are unscoreable:** when target and injected task are the same
  type (e.g. both sentiment), a correct-looking answer can't be attributed to the
  injection. The scorer refuses to credit these (see `scorer.py`); the held-out
  set should avoid such pairs.

## Later (not needed for review one)

- Swap the inline `samples` in `tasks.py` for the seven HuggingFace datasets the
  paper uses (SST2, SMS Spam, Gigaword, HSOL, RTE, MRPC, Jfleg).
- Hold out an injected task never seen during any round; report its ASR
  separately to show generalisation, not memorisation.
- Wire the retrain-and-reload hook so the loop runs adaptively against M1.
