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
py run_loop.py --mock --detector m1 --arm enriched

# Act 1: baseline vulnerability, no defense.
py run_loop.py --detector none --arm baseline
py run_loop.py --detector none --arm enriched

# Act 2/3: attack behind M1 (needs the m1-detector model in Ollama).
py run_loop.py --detector m1 --arm baseline
py run_loop.py --detector m1 --arm enriched

# Draw the charts + markdown table once the runs above exist.
py report.py
```

**`--detector`** picks the defender: `none`, `kad` (known-answer baseline), or
`m1` (George's fine-tuned detector, served by Ollama as `m1-detector`).
**`--arm`** picks the attacker: `baseline` is the original five-seed search;
`enriched` also seeds from the structure templates (`template_bank.py`) and
uses the structure-inspired operators + crossover. Running both arms against
the same detector is the "did the structure templates help?" comparison.

**Time budget (CPU):** ~240 model calls per round per trial. At full search size
with 3 trials, a 2-round KAD run is ~1,400+ calls — think hours, run it
overnight. The disk cache (`runs/llm_cache.json`) means a re-run reuses anything
already computed, so a crash mid-run is cheap to resume.

### What the run does

- Repeats each round `REPEAT_TRIALS` times and reports **ASR mean ± std** (LLM
  outputs are stochastic; a single trial is one roll of the dice).
- Prints **blocked-vs-succeeded** counts per round, so detector effectiveness is
  visible on screen, not buried in a log.
- Tags every output file by `<detector>_<arm>` so different runs
  **don't overwrite each other** — you need them all on disk for the chart.

### Outputs (in `runs/`)

- `asr_history_<detector>_<arm>.csv` — per-round ASR mean/std + block counts.
  **This is your headline-chart data.** One file per detector+arm.
- `<detector>_<arm>_roundN_attacks.jsonl` — every attempt from **every trial**
  (field `trial`), with model output, block status, and the `origin`/`root`
  family it came from.
- `<detector>_<arm>_roundN_contaminated.jsonl` — successful attacks as
  `{text, label, target_task, injected_task}`. **This is the handoff file M1
  retrains on.** Schema is stable — don't change it without telling M1's owner.
- `<detector>_<arm>_roundN_generations.csv` — the best strategy of each
  generation (how the search climbed).
- `<detector>_<arm>_roundN_best.jsonl` — the final elite strategies per trial.
- `llm_cache.json` — Ollama response cache. Regenerable; git-ignored.

`py report.py` reads these and writes `chart_arms.png` (baseline vs enriched
per defender), `chart_by_task.png`, `chart_by_origin.png`, and `report.md`.

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
| `template_bank.py` | loads + validates the structure templates (`templates/`) |
| `eval_heldout.py` | generalisation check: held-out templates vs train vs seeds |
| `report.py` | reads the run outputs and draws the charts + `report.md` |

If you understand one file for the viva, make it `attacker.py`.

---

## Known limitations (state these before an examiner finds them)

- **std ~ 0 in the *earlier* runs was a bug, now fixed.** Every trial used the
  same random seed, so all trials drew identical task pairs and mutations, hit
  the response cache, and replayed trial 1 exactly — the std was zero because
  the trials were literally the same run. `run_loop.py` now seeds trial `t`
  with `SEED + t` (trial 0 still uses `config.SEED`, so old numbers still
  reproduce), so trials genuinely differ and the mean ± std is meaningful.
- **Evolution isn't innovating much yet:** the best template stays close to the
  seed separators, because nothing is blocking attacks, so there's no selection
  pressure to climb. M2's adaptive search becomes *measurably* valuable only once
  M1 blocks a meaningful fraction. Say this explicitly.
- **Same-task pairs are unscoreable:** when target and injected task are the same
  type (e.g. both sentiment), a correct-looking answer can't be attributed to the
  injection. The scorer refuses to credit these (see `scorer.py`); the held-out
  set should avoid such pairs.

## Generalisation check (`eval_heldout.py`)

One template per structure category is held out: the search never seeds or
mutates it. `eval_heldout.py` scores the **held-out** templates, the **train**
templates, and the five **seeds** on the *same* fixed pairs, with no evolution,
and reports ASR per group (plus `runs/chart_generalization_<detector>.png`).

```bash
py eval_heldout.py --detector none
py eval_heldout.py --detector m1      # needs the m1-detector model in Ollama
```

Read it as: **held-out ASR ≈ train ASR ⇒ the structure *family* is what works,
not the specific templates the search happened to use** (so it is not
cherry-picking). Against `m1` it doubles as a probe of how M1 copes with
structures it never saw in training.

## Later (not needed for review one)

- Swap the inline `samples` in `tasks.py` for the seven HuggingFace datasets the
  paper uses (SST2, SMS Spam, Gigaword, HSOL, RTE, MRPC, Jfleg).
- Wire the retrain-and-reload hook so the loop runs adaptively against M1.
