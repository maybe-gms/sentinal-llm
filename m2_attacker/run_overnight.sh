#!/usr/bin/env bash
#
# run_overnight.sh — finish the experiments the review needs, unattended.
#
# WHAT THIS DOES (in order), closing the two gaps noted in RESULTS.md and
# refreshing every chart:
#
#   1. held-out templates vs M1        <- the one missing cell (was memory-killed)
#   2. M1 baseline arm at 3 trials     <- put M1 on equal footing with no-defense
#   3. M1 enriched arm at 3 trials     <- (RESULTS.md had these at 2 / 1 trials)
#   4. regenerate all charts + report.md
#
# It is SAFE to leave running overnight:
#   * each step logs to runs/overnight/<step>.log
#   * each finished step drops a .done marker and is SKIPPED on re-run, so if the
#     box sleeps or you re-launch, it resumes instead of repeating work
#   * the Ollama cache (runs/llm_cache.json) means the backend side of step 1 is
#     mostly cached from your earlier no-defense held-out run; only the M1 calls
#     are new
#
# USAGE (from inside m2_attacker/):
#     bash run_overnight.sh                 # real run (needs Ollama + models)
#     FORCE=1 bash run_overnight.sh         # ignore .done markers, redo everything
#     TRIALS=3 bash run_overnight.sh        # override trial count (default 3)
#     bash run_overnight.sh --mock          # plumbing test, no Ollama (fast)
#
# PREREQUISITES on this machine:
#     * `ollama serve` running
#     * models pulled/registered:  llama3.1:8b   and   m1-detector
#       (register M1:  cd ../m1_detector && ollama create m1-detector -f Modelfile)
#
# ---------------------------------------------------------------------------

set -u  # treat unset vars as errors; we do NOT set -e (we want to continue
        # past a failed step and report it, not abort the whole night)

# --- config ---------------------------------------------------------------
PY="${PY:-python3}"              # override if your launcher is `py` or `python`
TRIALS="${TRIALS:-3}"
MObj=""                          # mock flag passthrough
for arg in "$@"; do
  [ "$arg" = "--mock" ] && MObj="--mock"
done

OUT="runs/overnight"
mkdir -p "$OUT"

# --- Ollama memory settings (16 GB-safe): one model resident at a time ------
# Exported for any child the script spawns. If you run Ollama as a separate
# service, also set these in that service's environment and restart it.
export OLLAMA_MAX_LOADED_MODELS="${OLLAMA_MAX_LOADED_MODELS:-1}"
export OLLAMA_KEEP_ALIVE="${OLLAMA_KEEP_ALIVE:-30s}"
export OLLAMA_NUM_PARALLEL="${OLLAMA_NUM_PARALLEL:-1}"

# --- helpers --------------------------------------------------------------
ts()   { date "+%Y-%m-%d %H:%M:%S"; }
log()  { echo "[$(ts)] $*"; }

# run_step <name> <logfile> <command...>
# Skips if runs/overnight/<name>.done exists (unless FORCE=1).
run_step() {
  local name="$1"; shift
  local logf="$OUT/${name}.log"
  local done="$OUT/${name}.done"
  if [ -z "${FORCE:-}" ] && [ -f "$done" ]; then
    log "SKIP  $name (already done — delete $done or set FORCE=1 to redo)"
    return 0
  fi
  log "START $name  -> $logf"
  local t0=$(date +%s)
  # shellcheck disable=SC2068
  if $@ >"$logf" 2>&1; then
    local dt=$(( $(date +%s) - t0 ))
    echo "ok $(ts) ${dt}s" > "$done"
    log "DONE  $name  (${dt}s)"
    return 0
  else
    local dt=$(( $(date +%s) - t0 ))
    log "FAIL  $name  (${dt}s) — see $logf (tail below)"
    tail -n 15 "$logf" | sed 's/^/        | /'
    return 1
  fi
}

banner() {
  echo
  echo "=================================================================="
  echo "  Overnight experiment runner — $(ts)"
  echo "  trials=$TRIALS  mock=${MObj:-no}"
  echo "  OLLAMA_MAX_LOADED_MODELS=$OLLAMA_MAX_LOADED_MODELS  KEEP_ALIVE=$OLLAMA_KEEP_ALIVE"
  echo "=================================================================="
}

# --- preflight (skip in mock) --------------------------------------------
preflight() {
  [ -n "$MObj" ] && { log "mock mode: skipping Ollama preflight"; return 0; }
  log "preflight: checking Ollama + models"
  if ! curl -sf http://localhost:11434/api/tags >/dev/null 2>&1; then
    log "WARNING: Ollama does not seem to be up at localhost:11434."
    log "         Start it with 'ollama serve' in another terminal, then re-run."
  else
    local tags; tags=$(curl -s http://localhost:11434/api/tags)
    echo "$tags" | grep -q "llama3.1" || log "WARNING: llama3.1:8b not found in 'ollama list'."
    echo "$tags" | grep -q "m1-detector" || log "WARNING: m1-detector not registered (ollama create m1-detector -f ../m1_detector/Modelfile)."
  fi
}

# --- main -----------------------------------------------------------------
banner
preflight
START_ALL=$(date +%s)

# Step 1 — the missing cell: held-out templates evaluated against M1.
run_step "01_heldout_vs_m1" "" \
  $PY eval_heldout.py --detector m1 $MObj

# Step 2 — M1 baseline arm at full trial count.
run_step "02_m1_baseline_t${TRIALS}" "" \
  $PY run_loop.py --detector m1 --arm baseline --trials "$TRIALS" $MObj

# Step 3 — M1 enriched arm at full trial count.
run_step "03_m1_enriched_t${TRIALS}" "" \
  $PY run_loop.py --detector m1 --arm enriched --trials "$TRIALS" $MObj

# Step 4 — regenerate every chart + report.md from whatever is now in runs/.
run_step "04_report_charts" "" \
  $PY report.py

# --- summary --------------------------------------------------------------
DT=$(( $(date +%s) - START_ALL ))
echo
echo "=================================================================="
echo "  FINISHED at $(ts)  (total ${DT}s ≈ $(( DT/60 )) min)"
echo "  Step results:"
for d in 01_heldout_vs_m1 02_m1_baseline_t${TRIALS} 03_m1_enriched_t${TRIALS} 04_report_charts; do
  if [ -f "$OUT/${d}.done" ]; then
    echo "    [ok]   $d   ($(cat "$OUT/${d}.done"))"
  else
    echo "    [FAIL] $d   (see $OUT/${d}.log)"
  fi
done
echo
echo "  Key outputs to check in the morning:"
echo "    runs/heldout_m1.csv                  <- held-out vs M1 (the missing cell)"
echo "    runs/chart_generalization_m1.png     <- its chart"
echo "    runs/asr_history_m1_baseline.csv     <- M1 baseline, $TRIALS trials"
echo "    runs/asr_history_m1_enriched.csv     <- M1 enriched, $TRIALS trials"
echo "    runs/chart_arms.png, chart_by_task.png, report.md"
echo "=================================================================="
