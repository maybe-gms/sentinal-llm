"""Live demo web app for the integrated M1 + LLM + M3 defense stack.

Run it:

    cd integration
    python app.py                 # offline/mock mode — no Ollama needed
    python app.py --real          # drive real Ollama (llama3.1:8b + m1-detector)

Then open http://127.0.0.1:5000 in a browser. Pick a layer configuration,
click a preset attack (or type your own input), hit Run, and watch the request
flow through each defense stage with its verdict. This is the Review 2 demo:
the same pipeline code runs in both modes, so what you rehearse offline is what
runs live against the real models.
"""

from __future__ import annotations

import argparse
import os
import sys

from flask import Flask, jsonify, render_template, request

# Make the project root importable so `import shared` works from here.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_ROOT, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from shared import build_pipeline, Stage  # noqa: E402
from scenarios import (  # noqa: E402
    SCENARIOS, SCENARIOS_BY_ID, LAYER_CONFIGS, CANARY, APP_INSTRUCTION,
)

app = Flask(__name__)

# Runtime mode is set in main(); default is mock so the app is safe to import.
app.config["MOCK"] = True


def _pipeline(layers: str):
    """Build a pipeline for the requested layer config in the current mode."""
    return build_pipeline(
        layers,
        mock=app.config["MOCK"],
        system_prompt=APP_INSTRUCTION,
        canary_token=CANARY,
        model=None,
    )


@app.route("/")
def index():
    return render_template(
        "index.html",
        scenarios=SCENARIOS,
        layer_configs=LAYER_CONFIGS,
        mock=app.config["MOCK"],
    )


@app.route("/api/run", methods=["POST"])
def api_run():
    payload = request.get_json(force=True, silent=True) or {}
    layers = str(payload.get("layers", "full"))
    instruction = str(payload.get("instruction", "")).strip()
    data = str(payload.get("data", "")).strip()

    if not data:
        return jsonify({"error": "Please enter some input text."}), 400
    if not instruction:
        instruction = APP_INSTRUCTION

    try:
        pipe = _pipeline(layers)
        result = pipe.run(instruction, data)
    except Exception as e:  # surface real errors (e.g. Ollama down in --real)
        return jsonify({
            "error": f"{type(e).__name__}: {e}",
            "hint": ("If you started with --real, is `ollama serve` running with "
                     "llama3.1:8b and m1-detector pulled? Otherwise run without "
                     "--real for offline mode."),
        }), 500

    out = result.to_dict()
    out["layers_active"] = pipe.layers
    out["mode"] = "mock" if app.config["MOCK"] else "real"
    return jsonify(out)


@app.route("/api/scenario/<sid>")
def api_scenario(sid):
    s = SCENARIOS_BY_ID.get(sid)
    if not s:
        return jsonify({"error": "unknown scenario"}), 404
    return jsonify(s)


@app.route("/healthz")
def healthz():
    return jsonify({"ok": True, "mock": app.config["MOCK"]})


def main():
    ap = argparse.ArgumentParser(description="Integrated defense-stack demo UI.")
    ap.add_argument("--real", action="store_true",
                    help="Drive real Ollama (llama3.1:8b + m1-detector). "
                         "Default is offline mock mode.")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=5000)
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    app.config["MOCK"] = not args.real
    mode = "REAL (Ollama)" if args.real else "MOCK (offline)"
    print(f"\n  Integrated defense-stack demo — mode: {mode}")
    print(f"  Open http://{args.host}:{args.port}\n")
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
