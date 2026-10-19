# Integration + Live Demo (Review 2)

This folder is where the three runtime modules become **one system**. It does
not re-implement any module — it wires them through the interfaces they already
expose, using the small integration layer in [`../shared/`](../shared).

```
user input
   │
   ▼
 ┌──────────────┐   flagged?        ┌───────────────────────────────┐
 │ M1 detector  │ ───── yes ──────▶ │ BLOCKED on input (LLM skipped)│
 │ predict(x)   │                   └───────────────────────────────┘
 └──────┬───────┘
        │ clean
        ▼
 ┌──────────────┐
 │ backend LLM  │   llama3.1:8b
 │ generate()   │
 └──────┬───────┘
        │ output
        ▼
 ┌──────────────┐   block / redact / allow
 │ M3 monitor   │ ─────────────────────────▶ what the user finally sees
 │ inspect()    │
 └──────────────┘
```

M1 (input) and M3 (output) are **independent**: an attack that slips past M1 can
still be caught by M3, so an attacker must beat both. That is the project's
defense-in-depth claim, made concrete and clickable.

## Run it

```bash
pip install -r requirements.txt

python app.py            # OFFLINE / mock mode — no Ollama needed (rehearsal)
python app.py --real     # drive real Ollama: needs llama3.1:8b + m1-detector
```

Open <http://127.0.0.1:5000>. Pick a defense configuration (none / M1 / M3 /
M1+M3), click a preset attack or type your own input, and press **Run**. Each
stage shows what it decided, with latencies and the categories that fired. The
same pipeline code runs in both modes, so the offline rehearsal is faithful to
the live run.

## What each preset shows

| Preset | What happens (full stack) | Point |
|---|---|---|
| Clean summarisation / message | passes M1, LLM answers, M3 allows → **delivered** | no false alarms |
| Sentiment-flip / spam-flip injection | **M1 blocks on the input**, LLM never runs | cheap first line |
| Prompt-steal (subtle) | M1 misses → model leaks → **M3 blocks on the output** | the headline: layers are independent |
| Redirect / task-swap | M1 blocks the heavy version; a bare swap with no leak passes M3 | M3 is output-*safety*, not injection detection |
| PII in the output | M3 **redacts** the contact detail and delivers the rest | not every hit is a block |

## Modes

- **Mock (default).** Uses `shared/demo_backend.py`, a deterministic stand-in
  that reproduces a system-prompt leak when a prompt-steal gets through, so M3
  has a real leak to catch with no model server. Never quote numbers from mock
  mode — it exists to make the control flow demonstrable and repeatable.
- **Real (`--real`).** Uses the actual Ollama backend (`llama3.1:8b`) and the
  fine-tuned `m1-detector`. Identical pipeline, real verdicts.

## Files

- `app.py` — Flask server + JSON API (`/api/run`, `/api/scenario/<id>`).
- `scenarios.py` — the preset catalog and layer configs.
- `templates/index.html` — the single-page UI (no build step).
- `../shared/pipeline.py` — the real integration logic (framework-agnostic).
- `../shared/demo_backend.py` — offline backend for the demo only.
